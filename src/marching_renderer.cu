#include "marching_renderer.cuh"

#include "buffer_memory.cuh"
#include "buffer_vertex.h"
#include "marching_kernel.cu"
#include "marching_tables.h"

#include <thrust/device_vector.h>
#include <thrust/scan.h>


#include <iostream>

using namespace std;

namespace Marching {

    ////////////////////////////////////////////////////////////////////////////

    Renderer::Renderer(Grid::Uniform* grid) {
        this->_interpolation = false;
        this->_grid = grid;
        this->_gridParams = grid->getParams();

        this->_numCells =
            (this->_gridParams.resolution.x + GRID_OFFSET) *
            (this->_gridParams.resolution.y + GRID_OFFSET) *
            (this->_gridParams.resolution.z + GRID_OFFSET);

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
        uint numThreads = this->_gridParams.resolution.x + GRID_OFFSET;
        dim3 gridDim(
            this->_gridParams.resolution.y + GRID_OFFSET,
            this->_gridParams.resolution.z + GRID_OFFSET
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
                cudaMemcpy(
                    (void *) &lastElement,
                    (void *) (this->_voxelData.vertices + this->_numCells-1),
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
        //cout << "Active voxels:" << activeVoxels << endl;

        gridDim.x = (int) ceil(activeVoxels / (float) NTHREADS);
        gridDim.y = 1;
        gridDim.z = 1;

        while(gridDim.x > 65535) {
            gridDim.x/=2;
            gridDim.y*=2;
        }

        Marching::Kernel::generateTriangles<<<gridDim, NTHREADS>>>(
            this->_vertexData,
            this->_voxelData,
            this->_tableData,
            this->_grid->getData(),
            this->_maxVertices,
            activeVoxels,
            this->_gridParams.cellSize
        );

        if (this->_interpolation) {
            Marching::Kernel::interpolateNormals<<<gridDim, NTHREADS>>>(
                this->_vertexData,
                this->_voxelData,
                this->_tableData,
                this->_grid->getData(),
                this->_maxVertices,
                activeVoxels,
                this->_gridParams.cellSize
            );
        }

        cutilSafeCall(cutilDeviceSynchronize());

/*
        uint numCells =
            (this->_gridParams.resolution.x + GRID_OFFSET) *
            (this->_gridParams.resolution.x + GRID_OFFSET) *
            (this->_gridParams.resolution.x + GRID_OFFSET);


        Buffer::Memory<uint>* buffer1 =
            new Buffer::Memory<uint>(Buffer::Host);
        Buffer::Memory<uint>* buffer2 =
            new Buffer::Memory<uint>(Buffer::Host);

        buffer1->allocate(numCells);
        buffer2->allocate(numCells);

        uint* p1 = buffer1->get();
        uint* p2 = buffer2->get();

        cudaMemcpy(
            buffer1->get(),
            this->_voxelData.cubeIndex,
            numCells * sizeof(uint),
            cudaMemcpyDeviceToHost
        );

        cudaMemcpy(
            buffer2->get(),
            this->_grid->getData().cellStop,
            this->_grid->getNumCells() * sizeof(uint),
            cudaMemcpyDeviceToHost
        );

        for(uint i=0;i<numCells; i++) {
            if (p1[i] != 255) {
                cout << i  << ": ";
                for(int j=0; j<16; j++) {
                    if (Marching::Tables::triangles[p1[i]][j] != 255) {
                        cout << Marching::Tables::triangles[p1[i]][j] << " ";
                    }
                }
                cout << endl;

            }
        }
        cout << "______" << endl;


        Buffer::Memory<float4>* buffer =
        new Buffer::Memory<float4>(Buffer::Host);

        buffer->allocate(this->_maxVertices);
        float4* p = buffer->get();

        cudaMemcpy(
            buffer->get(),
                   this->_vertexData.inormals,
                   this->_numVertices * sizeof(float4),
                   cudaMemcpyDeviceToHost
        );

        for(uint i=0;i<this->_numVertices; i++) {
            cout << "[" << p[i].x << ", " << p[i].y << ", " << p[i].z << "," << p[i].w << "]";
            if (((i+1) % 3) == 0) {
                cout << endl;
            } else {
                cout << " ";
            }
        }
        cout << "______" << endl;


        Buffer::Memory<uint>* buffer3 =
            new Buffer::Memory<uint>(Buffer::Host);
        Buffer::Memory<uint>* buffer4 =
            new Buffer::Memory<uint>(Buffer::Host);

        buffer3->allocate(this->_grid->getNumCells());

        uint* p3 = buffer3->get();

        cudaMemcpy(
            buffer3->get(),
                   this->_voxelData.compact,
                   this->_grid->getNumCells() * sizeof(uint),
                   cudaMemcpyDeviceToHost
        );


        cout << "Compact:" << endl;
        for(uint i=0;i<activeVoxels; i++) {
            cout << p3[i] << endl;
        }
        cout << "______" << endl;*/

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

    void Renderer::setInterpolation(bool interpolation) {
        // set new flag
        this->_interpolation = interpolation;

        // get type of buffer
        Buffers type = (this->_interpolation) ? VertexINormal : VertexNormal;

        // get pointer to buffer from it's manager
        Buffer::Vertex<float4>* buffer =
            (Buffer::Vertex<float4>*)this->_vertexBuffMan->get(type);

        // get pointer to vertex buffer object of the selected buffer
        this->_normalsVBO = buffer->getVBO();

    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_createBuffers() {

        // TABLE BUFFERS
        this->_tableBuffMan = new Buffer::Manager<Marching::Buffers>();

        Buffer::Memory<uint>* edges            = new Buffer::Memory<uint>();
        Buffer::Memory<uint>* triangles        = new Buffer::Memory<uint>();
        Buffer::Memory<uint>* numVertices      = new Buffer::Memory<uint>();
        Buffer::Memory<uint>* adjacentEdges    = new Buffer::Memory<uint>();
        Buffer::Memory<int3>* adjacentEdgesPos = new Buffer::Memory<int3>();

        triangles->allocate(256*16);
        edges->allocate(256);
        numVertices->allocate(256);
        adjacentEdges->allocate(36);
        adjacentEdgesPos->allocate(36);

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
            )
            ->addBuffer(
                Marching::AdjacentEdgesTable,
                (Buffer::Abstract<void>*) adjacentEdges
            )
            ->addBuffer(
                Marching::AdjacentEdgesPosTable,
                (Buffer::Abstract<void>*) adjacentEdgesPos
            );


        this->_tableBuffMan->bindBuffers();

        // copy tables to graphical memory
        edges->copyFrom(Marching::Tables::edges, Buffer::Host);
        triangles->copyFrom(Marching::Tables::triangles, Buffer::Host);
        numVertices->copyFrom(Marching::Tables::numVertices, Buffer::Host);
        adjacentEdges->copyFrom(Marching::Tables::adjacentEdges, Buffer::Host);
        adjacentEdgesPos
            ->copyFrom(Marching::Tables::adjacentEdgesPos, Buffer::Host);

        cutilSafeCall(cutilDeviceSynchronize());


        this->_tableData.edges            = edges->get();
        this->_tableData.triangles        = triangles->get();
        this->_tableData.numVertices      = numVertices->get();
        this->_tableData.adjacentEdges    = adjacentEdges->get();
        this->_tableData.adjacentEdgesPos = adjacentEdgesPos->get();

        this->_tableBuffMan->unbindBuffers();


        // VOXEL BUFFERS
        this->_voxelBuffMan = new Buffer::Manager<Marching::Buffers>();

        Buffer::Memory<uint>* vertices     = new Buffer::Memory<uint>();
        Buffer::Memory<uint>* verticesScan = new Buffer::Memory<uint>();
        Buffer::Memory<uint>* occupied     = new Buffer::Memory<uint>();
        Buffer::Memory<uint>* occupiedScan = new Buffer::Memory<uint>();
        Buffer::Memory<uint>* compact      = new Buffer::Memory<uint>();
        Buffer::Memory<uint>* cubeIndex    = new Buffer::Memory<uint>();

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
            )
            ->addBuffer(
                Marching::VoxelCubeIndex,
                (Buffer::Abstract<void>*) cubeIndex
            );

        uint numCells =
            (this->_gridParams.resolution.x + GRID_OFFSET + 1) *
            (this->_gridParams.resolution.y + GRID_OFFSET + 1) *
            (this->_gridParams.resolution.z + GRID_OFFSET + 1);

        this->_voxelBuffMan->allocateBuffers(numCells);

        this->_voxelBuffMan->bindBuffers();
        this->_voxelBuffMan->memsetBuffers(0);

        this->_voxelData.vertices     = vertices->get();
        this->_voxelData.verticesScan = verticesScan->get();
        this->_voxelData.occupied     = occupied->get();
        this->_voxelData.occupiedScan = occupiedScan->get();
        this->_voxelData.compact      = compact->get();
        this->_voxelData.cubeIndex    = cubeIndex->get();

        this->_voxelBuffMan->unbindBuffers();

        // VERTEX BUFFERS
        this->_vertexBuffMan = new Buffer::Manager<Marching::Buffers>();

        Buffer::Vertex<float4>* positions = new Buffer::Vertex<float4>();
        Buffer::Vertex<float4>* normals   = new Buffer::Vertex<float4>();
        Buffer::Vertex<float4>* inormals  = new Buffer::Vertex<float4>();

        this->_vertexBuffMan
            ->addBuffer(
                Marching::VertexPosition,
                (Buffer::Abstract<void>*) positions
            )
            ->addBuffer(
                Marching::VertexNormal,
                (Buffer::Abstract<void>*) normals
            )
            ->addBuffer(
                Marching::VertexINormal,
                (Buffer::Abstract<void>*) inormals
            );

        this->_vertexBuffMan->allocateBuffers(this->_maxVertices);

        this->_vertexBuffMan->bindBuffers();

        this->_vertexData.positions = positions->get();
        this->_vertexData.normals   = normals->get();
        this->_vertexData.inormals  = inormals->get();

        this->_positionsVBO = positions->getVBO();
        this->_normalsVBO   = inormals->getVBO();

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
};