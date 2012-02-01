#ifndef __MARCHING_KERNEL_CU__
#define __MARCHING_KERNEL_CU__

#include "grid.cuh"
#include "grid_utils.cu"
#include "marching.h"

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
                cell.z >= cudaGridParams.resolution.z ||
                cell.x < 0 ||
                cell.y < 0 ||
                cell.z < 0
            ) {
                return 0.0f;
            }

            volatile uint hash = Utils::computeCellHash(cell, cudaGridParams);
            volatile uint cellStart = gridData.cellStart[hash];

            //return (cellStart != EMPTY_CELL_VALUE) ? 1.0f : 0.0f;

            if (cellStart != EMPTY_CELL_VALUE) {
                return (float) gridData.cellStop[hash] - cellStart;
            } else {
                return 0.0f;
            }
        }

        ////////////////////////////////////////////////////////////////////////

        __device__ int3 computeVoxelPosition(
            uint &index
        ) {

            float px = cudaGridParams.resolution.x + GRID_OFFSET;
            float pxy = px * (cudaGridParams.resolution.y + GRID_OFFSET);


            int3 cell;
            cell.z = floor(index / pxy);
            int tmp = index - (cell.z * pxy);
            cell.y = floor(tmp / px);
            cell.x = floor(tmp - cell.y * px);

            cell -= 1;

            return cell;
        }

        ////////////////////////////////////////////////////////////////////////

        // compute interpolated vertex along an edge
        __device__ float3 vertexInterolation(float3 p0, float3 p1, float f0, float f1) {
            p0 = ((p0 + 0.5f) * cudaGridParams.cellSize + cudaGridParams.min);
            p1 = ((p1 + 0.5f) * cudaGridParams.cellSize + cudaGridParams.min);
            float t = (0.5f - f0) / (f1 - f0);
            return lerp(p0, p1, t) - 1.0f + GRID_OFFSET;
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
                    blockDim.x,
                    blockIdx.x + __mul24(blockIdx.y, gridDim.x)
                );

            if (hash >= numCells) {
                return;
            }

            int3 cell = make_int3(
                threadIdx.x - GRID_OFFSET,
                blockIdx.x  - GRID_OFFSET,
                blockIdx.y  - GRID_OFFSET
            );

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
                    blockDim.x,
                    blockIdx.x + __mul24(blockIdx.y, gridDim.x)
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
                    blockDim.x,
                    blockIdx.x + __mul24(blockIdx.y, gridDim.x)
                );


            if (index > activeVoxels - 1) {
                index = activeVoxels - 1;
            }

            uint voxel = voxelData.compact[index];

            int3 cell = computeVoxelPosition(voxel);

            float3 p;
            p.x = cell.x;// + cudaGridParams.min.x;
            p.y = cell.y;// + cudaGridParams.min.y;
            p.z = cell.z;// + cudaGridParams.min.z;

            // calculate cell vertex positions
            float3 v[8];
            v[0] = p;
            v[1] = p + make_float3(1, 0, 0);
            v[2] = p + make_float3(1, 1, 0);
            v[3] = p + make_float3(0, 1, 0);
            v[4] = p + make_float3(0, 0, 1);
            v[5] = p + make_float3(1, 0, 1);
            v[6] = p + make_float3(1, 1, 1);
            v[7] = p + make_float3(0, 1, 1);

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

            //vertexData.positions[index].x = field[1] / (field[0] + field[1]);
            //return;

            // use shared memory to avoid using local
            __shared__ float3 vertlist[12*NTHREADS];

            vertlist[threadIdx.x] =
                vertexInterolation(v[0], v[1], field[0], field[1]);
            vertlist[NTHREADS + threadIdx.x] =
                vertexInterolation(v[1], v[2], field[1], field[2]);
            vertlist[NTHREADS * 2 + threadIdx.x] =
                vertexInterolation(v[2], v[3], field[2], field[3]);
            vertlist[NTHREADS * 3 + threadIdx.x] =
                vertexInterolation(v[3], v[0], field[3], field[0]);
            vertlist[NTHREADS * 4 + threadIdx.x] =
                vertexInterolation(v[4], v[5], field[4], field[5]);
            vertlist[NTHREADS * 5 + threadIdx.x] =
                vertexInterolation(v[5], v[6], field[5], field[6]);
            vertlist[NTHREADS * 6 + threadIdx.x] =
                vertexInterolation(v[6], v[7], field[6], field[7]);
            vertlist[NTHREADS * 7 + threadIdx.x] =
                vertexInterolation(v[7], v[4], field[7], field[4]);
            vertlist[NTHREADS * 8 + threadIdx.x] =
                vertexInterolation(v[0], v[4], field[0], field[4]);
            vertlist[NTHREADS * 9 + threadIdx.x] =
                vertexInterolation(v[1], v[5], field[1], field[5]);
            vertlist[NTHREADS * 10 + threadIdx.x] =
                vertexInterolation(v[2], v[6], field[2], field[6]);
            vertlist[NTHREADS * 11 + threadIdx.x] =
                vertexInterolation(v[3], v[7], field[3], field[7]);

            __syncthreads();

            // output triangle vertices
            uint numVerts = tableData.numVertices[cubeIndex];

            for(int i=0; i<numVerts; i+=3) {
                uint index = voxelData.verticesScan[voxel] + i;

                float3 *v[3];
                uint edge;
                edge = tableData.triangles[(cubeIndex*16) + i];
                v[0] = &vertlist[(edge*NTHREADS)+threadIdx.x];

                edge = tableData.triangles[(cubeIndex*16) + i + 1];
                v[1] = &vertlist[(edge*NTHREADS)+threadIdx.x];

                edge = tableData.triangles[(cubeIndex*16) + i + 2];
                v[2] = &vertlist[(edge*NTHREADS)+threadIdx.x];

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