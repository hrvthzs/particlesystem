#include "grid_uniform.cuh"

#include <iostream>

#include "buffer_memory.cuh"
#include "grid_kernel.cu"
#include "sph.h"
#include "utils.cuh"


namespace Grid {

    ////////////////////////////////////////////////////////////////////////////

    Uniform::Uniform() {

        this->_allocated = false;
        this->_numParticles = 0;
        this->_numCells = 0;

        this->_cellBufferManager = new Buffer::Manager<GridBuffers>();
        this->_particleBufferManager = new Buffer::Manager<GridBuffers>();

        Buffer::Memory<uint>* hash = new Buffer::Memory<uint>();
        Buffer::Memory<uint>* index = new Buffer::Memory<uint>();
        Buffer::Memory<uint>* cellStart = new Buffer::Memory<uint>();
        Buffer::Memory<uint>* cellStop = new Buffer::Memory<uint>();

        this->_cellBufferManager
            ->addBuffer(CellStart, (Buffer::Abstract<void>*) cellStart)
            ->addBuffer(CellStop, (Buffer::Abstract<void>*) cellStop);

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

    GridData& Uniform::getData() {
        return _data;
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

        // setup pointers to buffers
        this->_data.hash =
            (uint*) this->_particleBufferManager->get(Hash)->get();
        this->_data.index =
            (uint*) this->_particleBufferManager->get(Index)->get();
        this->_data.cellStart =
            (uint*) this->_cellBufferManager->get(CellStart)->get();
        this->_data.cellStop =
            (uint*) this->_cellBufferManager->get(CellStop)->get();


        // create device pointers for sorting indexes
        this->_thrustHash = new thrust::device_ptr<uint>(this->_data.hash);
        this->_thrustIndex = new thrust::device_ptr<uint>(this->_data.index);

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
        this->_numParticles = 0;
        this->_numCells = 0;
    }

    ////////////////////////////////////////////////////////////////////////////

    void Uniform::hash(float4* positions) {
        this->_particleBufferManager->get(Hash)->memset(0);

        uint minBlockSize, numBlocks, numThreads;
        minBlockSize = 192;
        ::Utils::computeGridSize(
            this->_numParticles,
            minBlockSize,
            numBlocks,
            numThreads
        );

        Kernel::hash<<<numBlocks, numThreads>>>(
            this->_numParticles, positions, this->_data
        );

        /*Buffer::Memory<uint>* buffer =
        new Buffer::Memory<uint>(new Buffer::Allocator(), Buffer::Host);

        Buffer::Memory<float4>* posBuffer =
            new Buffer::Memory<float4>(new Buffer::Allocator(), Buffer::Host);

        posBuffer->allocate(this->_numParticles);
        buffer->allocate(this->_numParticles);

        cudaMemcpy(posBuffer->get(), positions, this->_numParticles * sizeof(float4), cudaMemcpyDeviceToHost);
        cudaMemcpy(buffer->get(), this->_data.hash, this->_numParticles * sizeof(uint), cudaMemcpyDeviceToHost);
        float4* pos = posBuffer->get();
        uint* hash = buffer->get();


        cutilSafeCall(cutilDeviceSynchronize());

        for (uint i=0;i<this->_numParticles; i++) {
            std::cout << pos[i].x << " " << pos[i].y << " " << pos[i].z;
            std::cout << " hash: " << hash[i] << std::endl;
        }*/

    }

    ////////////////////////////////////////////////////////////////////////////

    void Uniform::sort() {

        thrust::sort_by_key(
            *this->_thrustHash,
            *this->_thrustHash + this->_numParticles,
            *this->_thrustIndex
        );

    }

    ////////////////////////////////////////////////////////////////////////////

    void Uniform::emptyCells() {
        this->_cellBufferManager->get(CellStart)->memset(EMPTY_CELL_VALUE);
    }

    ////////////////////////////////////////////////////////////////////////////

    void Uniform::_calcParams(float cellSize, float gridSize) {
        this->_params.min = make_float3(-1.0f, -1.0f, -1.0f);
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
        // declaration of symbol is in grid.cuh
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

    void Uniform::printParams () {
        std::cout
            << "Min["
            << this->_params.min.x << ", "
            << this->_params.min.y << ", "
            << this->_params.min.z << "]"
            << std::endl;

        std::cout
            << "Max["
            << this->_params.max.x << ", "
            << this->_params.max.y << ", "
            << this->_params.max.z << "]"
            << std::endl;

        std::cout
            << "Size["
            << this->_params.size.x << ", "
            << this->_params.size.y << ", "
            << this->_params.size.z << "]"
            << std::endl;

        std::cout
            << "Resolution["
            << this->_params.resolution.x << ", "
            << this->_params.resolution.y << ", "
            << this->_params.resolution.z << "]"
            << std::endl;

        std::cout
            << "Delta["
            << this->_params.delta.x << ", "
            << this->_params.delta.y << ", "
            << this->_params.delta.z << "]"
            << std::endl;

        std::cout
            << "Num cells: "
            << this->_numCells
            << std::endl;
    }

    ////////////////////////////////////////////////////////////////////////////

};