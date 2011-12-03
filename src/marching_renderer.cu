#include "marching_renderer.cuh"

#include "buffer_memory.cuh"
#include "buffer_vertex.h"
#include "marching_kernel.cu"
#include "marching_tables.h"

#include <iostream>

using namespace std;

namespace Marching {

    ////////////////////////////////////////////////////////////////////////////

    Renderer::Renderer(Grid::Uniform* grid) {
        this->_grid = grid;
        this->_gridParams = grid->getParams();
        this->_numCells = grid->getNumCells();
        this->_maxVertices = 16*this->_numCells;
        this->_createBuffers();

        // TODO this is the 3rd constant memory copy code of this symbol
        // don't know why must in each cu file copy it
        // files are compiled into separate object file, but maybe there
        // another reason
        CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(
                cudaGridParams,
                &this->_gridParams,
                sizeof(GridParams)
            )
        );

    }

    ////////////////////////////////////////////////////////////////////////////

    Renderer::~Renderer() {
        this->_deleteBuffers();
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::render() {
        uint numThreads = this->_gridParams.resolution.x;
        dim3 gridDim(
            this->_gridParams.resolution.y,
            this->_gridParams.resolution.z
        );

        Marching::Kernel::classifyVoxel<<<gridDim, numThreads>>>(
            this->_voxelData,
            this->_tableData,
            this->_grid->getData(),
            this->_numCells
        );

        this->_thrustScanWrapper(
            this->_voxelData.occupied,
            this->_voxelData.occupiedScan
        );

        uint activeVoxels = 0;
        {
            uint lastElement, lastScanElement;
            cutilSafeCall(
                cudaMemcpy(
                    (void *) &lastElement,
                    (void *) (this->_voxelData.occupied + this->_numCells-1),
                    sizeof(uint),
                    cudaMemcpyDeviceToHost
                )
            );
            cutilSafeCall(
                cudaMemcpy(
                    (void *) &lastScanElement,
                    (void *) (this->_voxelData.occupiedScan + this->_numCells-1),
                    sizeof(uint),
                    cudaMemcpyDeviceToHost
                )
            );
            activeVoxels = lastElement + lastScanElement;
        }

        //cout << "Active voxels:" << activeVoxels << endl;

        if (activeVoxels == 0) {
            // return if there are no full voxels
            this->_numVertices = 0;
            return;
        }

        Marching::Kernel::compactVoxels<<<gridDim, numThreads>>>(
            this->_voxelData,
            this->_numCells
        );

        this->_thrustScanWrapper(
            this->_voxelData.vertices,
            this->_voxelData.verticesScan
        );

        // readback total number of vertices
        {
            uint lastElement, lastScanElement;
            cutilSafeCall(
                cudaMemcpy((void *) &lastElement,
                    (void *) (this->_voxelData.compact + this->_numCells-1),
                    sizeof(uint),
                    cudaMemcpyDeviceToHost
                )
            );

            cutilSafeCall(
                cudaMemcpy(
                    (void *) &lastScanElement,
                    (void *) (this->_voxelData.verticesScan + this->_numCells-1),
                    sizeof(uint),
                    cudaMemcpyDeviceToHost
                )
            );

            this->_numVertices = lastElement + lastScanElement;

        }

        //cout << "Num vertices:" << this->_numVertices << endl;

        gridDim.x = (int) ceil(activeVoxels / (float) 32);
        gridDim.y = 1;
        gridDim.z = 1;

        while(gridDim.x > 65535) {
            gridDim.x/=2;
            gridDim.y*=2;
        }

        Marching::Kernel::generateTriangles<<<gridDim, 32>>>(
            this->_vertexData,
            this->_voxelData,
            this->_tableData,
            this->_grid->getData(),
            this->_maxVertices,
            activeVoxels,
            this->_gridParams.cellSize
        );

        cutilSafeCall(cutilDeviceSynchronize());

        /*Buffer::Memory<uint>* buffer =
            new Buffer::Memory<uint>(new Buffer::Allocator(), Buffer::Host);

        buffer->allocate(this->_numCells);
        uint* e = buffer->get();

        cudaMemcpy(
            buffer->get(),
                this->_voxelData.compact,
                this->_numCells * sizeof(uint),
                cudaMemcpyDeviceToHost
            );

        for(uint i=0;i<this->_numCells; i++) {
            cout << e[i] << endl;
        }
        cout << "______" << endl;
        */

        /*Buffer::Memory<float4>* buffer =
        new Buffer::Memory<float4>(new Buffer::Allocator(), Buffer::Host);

        buffer->allocate(this->_maxVertices);
        float4* p = buffer->get();

        cudaMemcpy(
            buffer->get(),
                   this->_vertexData.normals,
                   this->_numVertices * sizeof(float4),
                   cudaMemcpyDeviceToHost
        );
        */
        /*for(uint i=0;i<this->_numVertices; i++) {
            cout << p[i].x << ", " << p[i].y << ", " << p[i].z << "," << p[i].w << endl;
        }
        cout << "______" << endl;
        */
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::bindBuffers() {
        this->_voxelBuffMan->bindBuffers();
        this->_tableBuffMan->bindBuffers();
        this->_vertexBuffMan->bindBuffers();
    }

    ////////////////////////////////////////////////////////////////////////////


    void Renderer::unbindBuffers() {
        this->_voxelBuffMan->unbindBuffers();
        this->_tableBuffMan->unbindBuffers();
        this->_vertexBuffMan->unbindBuffers();
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::freeBuffers() {
        this->_voxelBuffMan->freeBuffers();
        this->_tableBuffMan->freeBuffers();
        this->_vertexBuffMan->freeBuffers();
    }

    ////////////////////////////////////////////////////////////////////////////

    Marching::VertexData& Renderer::getData() {
        return this->_vertexData;
    }

    ////////////////////////////////////////////////////////////////////////////

    uint Renderer::getNumVertices() {
        return this->_numVertices;
    }

    ////////////////////////////////////////////////////////////////////////////

    GLuint Renderer::getPositionsVBO() {
        return this->_positionsVBO;
    }

    ////////////////////////////////////////////////////////////////////////////

    GLuint Renderer::getNormalsVBO() {
        return this->_normalsVBO;
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_createBuffers() {

        // TABLE BUFFERS
        this->_tableBuffMan = new Buffer::Manager<Marching::Buffers>();

        Buffer::Memory<uint>* edges       = new Buffer::Memory<uint>();
        Buffer::Memory<uint>* triangles   = new Buffer::Memory<uint>();
        Buffer::Memory<uint>* numVertices = new Buffer::Memory<uint>();

        triangles->allocate(256*16);
        edges->allocate(256);
        numVertices->allocate(256);

        this->_tableBuffMan
            ->addBuffer(
                Marching::EdgesTable,
                (Buffer::Abstract<void>*) edges
            )
            ->addBuffer(
                Marching::TrianglesTable,
                (Buffer::Abstract<void>*) triangles
            )
            ->addBuffer(
                Marching::NumVerticesTable,
                (Buffer::Abstract<void>*) numVertices
            );


        this->_tableBuffMan->bindBuffers();

        // copy tables to graphical memory
        edges->copyFrom(Marching::Tables::edges, Buffer::Host);
        triangles->copyFrom(Marching::Tables::triangles, Buffer::Host);
        numVertices->copyFrom(Marching::Tables::numVertices, Buffer::Host);

        cutilSafeCall(cutilDeviceSynchronize());


        this->_tableData.edges       = edges->get();
        this->_tableData.triangles   = triangles->get();
        this->_tableData.numVertices = numVertices->get();

        this->_tableBuffMan->unbindBuffers();


        // VOXEL BUFFERS
        this->_voxelBuffMan = new Buffer::Manager<Marching::Buffers>();

        Buffer::Memory<uint>* vertices     = new Buffer::Memory<uint>();
        Buffer::Memory<uint>* verticesScan = new Buffer::Memory<uint>();
        Buffer::Memory<uint>* occupied     = new Buffer::Memory<uint>();
        Buffer::Memory<uint>* occupiedScan = new Buffer::Memory<uint>();
        Buffer::Memory<uint>* compact      = new Buffer::Memory<uint>();

        this->_voxelBuffMan
            ->addBuffer(
                Marching::VoxelVertices,
                (Buffer::Abstract<void>*) vertices
            )
            ->addBuffer(
                Marching::VoxelVerticesScan,
                (Buffer::Abstract<void>*) verticesScan
            )
            ->addBuffer(
                Marching::VoxelOccupied,
                (Buffer::Abstract<void>*) occupied
            )
            ->addBuffer(
                Marching::VoxelOccupiedScan,
                (Buffer::Abstract<void>*) occupiedScan
            )
            ->addBuffer(
                Marching::VoxelCompact,
                (Buffer::Abstract<void>*) compact
            );


        this->_voxelBuffMan->allocateBuffers(this->_grid->getNumCells());

        this->_voxelBuffMan->bindBuffers();
        this->_voxelBuffMan->memsetBuffers(0);

        this->_voxelData.vertices     = vertices->get();
        this->_voxelData.verticesScan = verticesScan->get();
        this->_voxelData.occupied     = occupied->get();
        this->_voxelData.occupiedScan = occupiedScan->get();
        this->_voxelData.compact      = compact->get();

        this->_voxelBuffMan->unbindBuffers();

        // VERTEX BUFFERS
        this->_vertexBuffMan = new Buffer::Manager<Marching::Buffers>();

        Buffer::Vertex<float4>* positions = new Buffer::Vertex<float4>();
        Buffer::Vertex<float4>* normals   = new Buffer::Vertex<float4>();

        this->_vertexBuffMan
            ->addBuffer(
                Marching::VertexPosition,
                (Buffer::Abstract<void>*) positions
            )
            ->addBuffer(
                Marching::VertexNormal,
                (Buffer::Abstract<void>*) normals
            );

        this->_vertexBuffMan->allocateBuffers(this->_maxVertices);

        this->_vertexBuffMan->bindBuffers();

        this->_vertexData.positions = positions->get();
        this->_vertexData.normals = normals->get();

        this->_positionsVBO = positions->getVBO();
        this->_normalsVBO = normals->getVBO();

        this->_vertexBuffMan->unbindBuffers();

    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_deleteBuffers() {
        delete this->_tableBuffMan;
        delete this->_voxelBuffMan;
        delete this->_vertexBuffMan;
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_thrustScanWrapper(
        uint* input,
        uint* output
    ){
        thrust::exclusive_scan(
            thrust::device_ptr<unsigned int>(input),
            thrust::device_ptr<unsigned int>(input + this->_numCells),
            thrust::device_ptr<unsigned int>(output)
        );
    }

    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
};