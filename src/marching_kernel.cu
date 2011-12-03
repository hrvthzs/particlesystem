#ifndef __MARCHING_KERNEL_CU__
#define __MARCHING_KERNEL_CU__

#include "grid.cuh"
#include "grid_utils.cu"
#include "marching.h"

#include <thrust/device_vector.h>
#include <thrust/scan.h>

namespace Marching {

    namespace Kernel {

        using namespace Grid;

        ////////////////////////////////////////////////////////////////////////

        __device__ float sampleVolume(
            int3 cell,
            GridData &gridData
        ) {
            if (cell.x >= cudaGridParams.resolution.x ||
                cell.y >= cudaGridParams.resolution.y ||
                cell.z >= cudaGridParams.resolution.z
            ) {
                return 0.0f;
            }

            volatile uint hash = Utils::computeCellHash(cell, cudaGridParams);
            volatile uint cellStart = gridData.cellStart[hash];

            return (cellStart != EMPTY_CELL_VALUE) ? 1.0f : 0.0f;
        }

        ////////////////////////////////////////////////////////////////////////

        __device__ int3 computeVoxelPosition(
            uint &index
        ) {
            float px = cudaGridParams.resolution.x;
            float pxy = px * cudaGridParams.resolution.y;

            int3 cell;
            cell.z = floor(index / pxy);
            int tmp = index - (cell.z * pxy);
            cell.y = floor(tmp / px);
            cell.x = floor(tmp - cell.y * px);

            return cell;
        }

        ////////////////////////////////////////////////////////////////////////

        // compute interpolated vertex along an edge
        __device__ float3 vertexInterolation(float3 p0, float3 p1) {
            return (p0 + p1) / 2.0f;
        }

        ////////////////////////////////////////////////////////////////////////

        // calculate triangle normal
        __device__ float3 calculateNormal(float3 *v0, float3 *v1, float3 *v2) {
            float3 edge0 = *v1 - *v0;
            float3 edge1 = *v2 - *v0;
            // note - it's faster to perform normalization in vertex shader
            // rather than here
            return cross(edge0, edge1);
        }

        ////////////////////////////////////////////////////////////////////////

        __global__ void classifyVoxel(
            Marching::VoxelData voxelData,
            Marching::TableData tableData,
            GridData gridData,
            uint numCells
        ) {
            int hash =
                threadIdx.x +
                __mul24(
                    gridDim.x,
                    blockIdx.x + __mul24(blockIdx.y, gridDim.y)
                );

            if (hash >= numCells) {
                return;
            }

            int3 cell = make_int3(threadIdx.x, blockIdx.x, blockIdx.y);

            float field[8];
            field[0] = sampleVolume(cell, gridData);
            field[1] = sampleVolume(cell + make_int3(1,0,0), gridData);
            field[2] = sampleVolume(cell + make_int3(1,1,0), gridData);
            field[3] = sampleVolume(cell + make_int3(0,1,0), gridData);
            field[4] = sampleVolume(cell + make_int3(0,0,1), gridData);
            field[5] = sampleVolume(cell + make_int3(1,0,1), gridData);
            field[6] = sampleVolume(cell + make_int3(1,1,1), gridData);
            field[7] = sampleVolume(cell + make_int3(0,1,1), gridData);

            float isoValue = 0.5f;

            uint cubeIndex = 0;
            cubeIndex  = uint(field[0] < isoValue);
            cubeIndex += uint(field[1] < isoValue) * 2;
            cubeIndex += uint(field[2] < isoValue) * 4;
            cubeIndex += uint(field[3] < isoValue) * 8;
            cubeIndex += uint(field[4] < isoValue) * 16;
            cubeIndex += uint(field[5] < isoValue) * 32;
            cubeIndex += uint(field[6] < isoValue) * 64;
            cubeIndex += uint(field[7] < isoValue) * 128;

            uint numVertices = tableData.numVertices[cubeIndex];

            voxelData.vertices[hash] = numVertices;
            voxelData.occupied[hash] = (numVertices > 0);

        }

        ////////////////////////////////////////////////////////////////////////

        __global__ void compactVoxels(Marching::VoxelData data, uint numCells) {
            int hash =
                threadIdx.x +
                __mul24(
                    gridDim.x,
                    blockIdx.x + __mul24(blockIdx.y, gridDim.y)
                );

            if (hash >= numCells) {
                return;
            }

            // if cell is occupied then get index to compact array
            // from sorted occupiedScan
            // and put hash of cell to compact array
            if (data.occupied[hash]) {
                data.compact[data.occupiedScan[hash]] = hash;
            }
        }

        ////////////////////////////////////////////////////////////////////////

        __global__ void generateTriangles(
            Marching::VertexData vertexData,
            Marching::VoxelData voxelData,
            Marching::TableData tableData,
            GridData gridData,
            uint maxVertices,
            uint activeVoxels,
            float3 cellSize
        ) {
            int index =
                threadIdx.x +
                __mul24(
                    gridDim.x,
                    blockIdx.x + __mul24(blockIdx.y, gridDim.y)
                );

            if (index >= activeVoxels) {
                return;
            }

            uint voxel = voxelData.compact[index];

            int3 cell = computeVoxelPosition(voxel);

            float3 p;
            p.x = cell.x + cudaGridParams.min.x;
            p.y = cell.y + cudaGridParams.min.y;
            p.z = cell.z + cudaGridParams.min.z;

            // calculate cell vertex positions
            float3 v[8];
            v[0] = p;
            v[1] = p + make_float3(cellSize.x, 0, 0);
            v[2] = p + make_float3(cellSize.x, cellSize.y, 0);
            v[3] = p + make_float3(0, cellSize.y, 0);
            v[4] = p + make_float3(0, 0, cellSize.z);
            v[5] = p + make_float3(cellSize.x, 0, cellSize.z);
            v[6] = p + make_float3(cellSize.x, cellSize.y, cellSize.z);
            v[7] = p + make_float3(0, cellSize.y, cellSize.z);

            float field[8];
            field[0] = sampleVolume(cell, gridData);
            field[1] = sampleVolume(cell + make_int3(1,0,0), gridData);
            field[2] = sampleVolume(cell + make_int3(1,1,0), gridData);
            field[3] = sampleVolume(cell + make_int3(0,1,0), gridData);
            field[4] = sampleVolume(cell + make_int3(0,0,1), gridData);
            field[5] = sampleVolume(cell + make_int3(1,0,1), gridData);
            field[6] = sampleVolume(cell + make_int3(1,1,1), gridData);
            field[7] = sampleVolume(cell + make_int3(0,1,1), gridData);

            float isoValue = 0.5f;

            uint cubeIndex = 0;
            cubeIndex  = uint(field[0] < isoValue);
            cubeIndex += uint(field[1] < isoValue) * 2;
            cubeIndex += uint(field[2] < isoValue) * 4;
            cubeIndex += uint(field[3] < isoValue) * 8;
            cubeIndex += uint(field[4] < isoValue) * 16;
            cubeIndex += uint(field[5] < isoValue) * 32;
            cubeIndex += uint(field[6] < isoValue) * 64;
            cubeIndex += uint(field[7] < isoValue) * 128;

            // use shared memory to avoid using local
            __shared__ float3 vertlist[12*32];

            vertlist[        threadIdx.x] = vertexInterolation(v[0], v[1]);
            vertlist[32     +threadIdx.x] = vertexInterolation(v[1], v[2]);
            vertlist[(32* 2)+threadIdx.x] = vertexInterolation(v[2], v[3]);
            vertlist[(32* 3)+threadIdx.x] = vertexInterolation(v[3], v[0]);
            vertlist[(32* 4)+threadIdx.x] = vertexInterolation(v[4], v[5]);
            vertlist[(32* 5)+threadIdx.x] = vertexInterolation(v[5], v[6]);
            vertlist[(32* 6)+threadIdx.x] = vertexInterolation(v[6], v[7]);
            vertlist[(32* 7)+threadIdx.x] = vertexInterolation(v[7], v[4]);
            vertlist[(32* 8)+threadIdx.x] = vertexInterolation(v[0], v[4]);
            vertlist[(32* 9)+threadIdx.x] = vertexInterolation(v[1], v[5]);
            vertlist[(32*10)+threadIdx.x] = vertexInterolation(v[2], v[6]);
            vertlist[(32*11)+threadIdx.x] = vertexInterolation(v[3], v[7]);

            __syncthreads();

            // output triangle vertices
            uint numVerts = tableData.numVertices[cubeIndex];

            for(int i=0; i<numVerts; i+=3) {
                uint index = voxelData.verticesScan[voxel] + i;

                float3 *v[3];
                uint edge;
                edge = tableData.triangles[(cubeIndex*16) + i];
                v[0] = &vertlist[(edge*32)+threadIdx.x];

                edge = tableData.triangles[(cubeIndex*16) + i + 1];
                v[1] = &vertlist[(edge*32)+threadIdx.x];

                edge = tableData.triangles[(cubeIndex*16) + i + 2];
                v[2] = &vertlist[(edge*32)+threadIdx.x];

                // calculate triangle surface normal
                float3 n = calculateNormal(v[0], v[1], v[2]);

                if (index < (maxVertices - 3)) {
                    vertexData.positions[index] = make_float4(*v[0], 1.0f);
                    vertexData.normals[index] = make_float4(n, 0.0f);

                    vertexData.positions[index+1] = make_float4(*v[1], 1.0f);
                    vertexData.normals[index+1] = make_float4(n, 0.0f);

                    vertexData.positions[index+2] = make_float4(*v[2], 1.0f);
                    vertexData.normals[index+2] = make_float4(n, 0.0f);
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////
    };
};

#endif // __MARCHING_KERNEL_CU__