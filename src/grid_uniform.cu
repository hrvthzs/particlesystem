#include "grid_uniform.cuh"

#include "buffer_memory.cuh"
#include "utils.cuh"
#include "grid_kernel.cu"

namespace Grid {

    ////////////////////////////////////////////////////////////////////////////

    Uniform::Uniform() {

        this->_allocated = false;

        this->_cellBufferManager = new Buffer::Manager<GridBuffers>();
        this->_particleBufferManager = new Buffer::Manager<GridBuffers>();

        Buffer::Memory<uint>* hash = new Buffer::Memory<uint>();
        Buffer::Memory<uint>* index = new Buffer::Memory<uint>();
        Buffer::Memory<uint>* cellStart = new Buffer::Memory<uint>();
        Buffer::Memory<uint>* cellStop = new Buffer::Memory<uint>();

        this->_cellBufferManager
            ->addBuffer(CellStart, (Buffer::Abstract<void>*) cellStart)
            ->addBuffer(CellStart, (Buffer::Abstract<void>*) cellStop);

        this->_particleBufferManager
            ->addBuffer(Hash, (Buffer::Abstract<void>*) hash)
            ->addBuffer(Index, (Buffer::Abstract<void>*) index);
    }

    ////////////////////////////////////////////////////////////////////////////

    Uniform::~Uniform() {
        delete this->_cellBufferManager;
        delete this->_particleBufferManager;
    }

    ////////////////////////////////////////////////////////////////////////////

    GridParams Uniform::getParams() const {
        return this->_params;
    }

    ////////////////////////////////////////////////////////////////////////////

    GridData Uniform::getData() {
        GridData data;

        data.hash = (uint*) this->_particleBufferManager->get(Hash)->get();
        data.index = (uint*) this->_particleBufferManager->get(Index)->get();
        data.cellStart = (uint*) this->_cellBufferManager->get(CellStart)->get();
        data.cellStop = (uint*) this->_cellBufferManager->get(CellStop)->get();

        return data;
    }

    ////////////////////////////////////////////////////////////////////////////

    uint Uniform::getNumCells() const {
        return this->_numCells;
    }

    ////////////////////////////////////////////////////////////////////////////

    void Uniform::allocate(uint numParticles, float cellSize, float gridSize) {
        if (this->_allocated) {
            this->free();
        }

        // calculate parameters of grid with requested sizes
        this->_calcParams(cellSize, gridSize);

        this->_numParticles = numParticles;

        // calculate number of cell in grid
        this->_numCells =
            (uint) ceil(
                this->_params.resolution.x *
                this->_params.resolution.y *
                this->_params.resolution.z
            );

        // allocate buffers
        this->_particleBufferManager->allocateBuffers(this->_numParticles);
        this->_cellBufferManager->allocateBuffers(this->_numCells);

        GridData data = this->getData();

        // create device pointers for sorting indexes
        this->_thrustHash = new thrust::device_ptr<uint>(data.hash);
        this->_thrustIndex = new thrust::device_ptr<uint>(data.index);

        this->_allocated = true;

    }

    ////////////////////////////////////////////////////////////////////////////

    void Uniform::free() {

        if (!this->_allocated) {
            return;
        }

        this->_particleBufferManager->freeBuffers();
        this->_cellBufferManager->freeBuffers();

        delete this->_thrustHash;
        delete this->_thrustIndex;

        this->_allocated = false;
    }

    ////////////////////////////////////////////////////////////////////////////

    void Uniform::hash() {
        this->_particleBufferManager->get(Hash)->memset(0);

        uint minBlockSize, numBlocks, numThreads;
        minBlockSize = 192;
        ::Utils::computeGridSize(
            this->_numParticles,
            minBlockSize,
            numBlocks,
            numThreads
        );

        GridData data = this->getData();
        Kernel::hash<<<numBlocks, numThreads>>>(
            this->_numParticles, NULL, data
        );

    }

    ////////////////////////////////////////////////////////////////////////////

    void Uniform::sort() {
        thrust::sort_by_key(
            this->_thrustHash,
            this->_thrustHash + this->_numParticles,
            this->_thrustIndex
        );
    }

    ////////////////////////////////////////////////////////////////////////////

    void Uniform::_calcParams(float cellSize, float gridSize) {
        this->_params.min = make_float3(0.0, 0.0, 0.0);
        this->_params.max =
            make_float3(
                this->_params.min.x + gridSize,
                this->_params.min.y + gridSize,
                this->_params.min.z + gridSize
            );

        this->_params.size =
            make_float3(
                this->_params.max.x - this->_params.min.x,
                this->_params.max.y - this->_params.min.y,
                this->_params.max.z - this->_params.min.z
            );

        this->_params.resolution =
            make_float3(
                ceil(this->_params.size.x / cellSize),
                ceil(this->_params.size.y / cellSize),
                ceil(this->_params.size.z / cellSize)
            );

        this->_params.size =
            make_float3(
                this->_params.resolution.x * cellSize,
                this->_params.resolution.y * cellSize,
                this->_params.resolution.z * cellSize
            );

        this->_params.delta =
            make_float3(
                this->_params.resolution.x / this->_params.size.x,
                this->_params.resolution.y / this->_params.size.y,
                this->_params.resolution.z / this->_params.size.z
            );

        // Copy parameters to GPU's constant memory
        CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(
                cudaGridParams,
                &this->_params,
                sizeof(GridParams)
            )
        );
        CUDA_SAFE_CALL(cudaThreadSynchronize());
    }

    ////////////////////////////////////////////////////////////////////////////

};