#include "particle_system.cuh"
#include "particle_system.h"

#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include "cutil_math.h"

__constant__ ParticleSystemParameters parameters;

struct integrate_functor
{
    float deltaTime;

    __host__ __device__
    integrate_functor(float delta_time) : deltaTime(delta_time) {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        volatile float4 posData = thrust::get<0>(t);
        volatile float4 velData = thrust::get<1>(t);
        float3 pos = make_float3(posData.x, posData.y, posData.z);
        float3 vel = make_float3(velData.x, velData.y, velData.z);
        float3 gravity = parameters.gravity;

        vel += gravity * deltaTime;
        //vel *= parameters.globalDamping;

        // new position = old position + velocity * deltaTime
        pos += vel * deltaTime;

        float particleRadius = parameters.particleRadius;
        float boundaryDamping = parameters.boundaryDamping;

        // set this to zero to disable collisions with cube sides
        /*#if 1
            if (pos.x >  1.0f - particleRadius) {
                pos.x =  1.0f - particleRadius;
                vel.x *= boundaryDamping;
            }
            if (pos.x < -1.0f + particleRadius) {
                pos.x = -1.0f + particleRadius;
                vel.x *= boundaryDamping;
            }
            if (pos.y >  1.0f - particleRadius) {
                pos.y =  1.0f - particleRadius;
                vel.y *= boundaryDamping;
            }
            if (pos.z >  1.0f - particleRadius) {
                pos.z =  1.0f - particleRadius;
                vel.z *= boundaryDamping;
            }
            if (pos.z < -1.0f + particleRadius) {
                pos.z = -1.0f + particleRadius;
                vel.z *= boundaryDamping;
            }
        #endif
        if (pos.y < -1.0f + particleRadius) {
            pos.y = -1.0f + particleRadius;
            vel.y *= boundaryDamping;
        }*/

        float len = length(pos);

        if (len >= 1.0f - particleRadius) {
            pos = normalize(pos) * 0.99;
            float velL = length(vel);
            vel -= (pos / (len / velL));
            vel *= boundaryDamping;
        }

        // store new position and velocity
        thrust::get<0>(t) = make_float4(pos, posData.w);
        thrust::get<1>(t) = make_float4(vel, velData.w);

    }
};

__device__ int3 deviceCalculateGridPosition(float3 p) {
    int3 gridPos;
    gridPos.x = floor((p.x - parameters.gridOrigin.x) / parameters.cellSize.x);
    gridPos.y = floor((p.y - parameters.gridOrigin.y) / parameters.cellSize.y);
    gridPos.z = floor((p.z - parameters.gridOrigin.z) / parameters.cellSize.z);
    return gridPos;
}

__device__ uint deviceCalculateGridHash(int3 gridPos) {
    gridPos.x = gridPos.x & (parameters.gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (parameters.gridSize.y-1);
    gridPos.z = gridPos.z & (parameters.gridSize.z-1);
    return __umul24(__umul24(gridPos.z, parameters.gridSize.y), parameters.gridSize.x) + __umul24(gridPos.y, parameters.gridSize.x) + gridPos.x;
}

__global__ void calculateHashKernel(
    uint* particlesGridHash,
    uint* particlesGridIndex,
    float4* positions,
    uint count
) {
    uint index = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    if (index >= count) return;

    volatile float4 position = positions[index];

    int3 gridPosition =
        deviceCalculateGridPosition(make_float3(position.x, position.y, position.z));
    uint hashCode = deviceCalculateGridHash(gridPosition);

    particlesGridHash[index] = hashCode;
    particlesGridIndex[index] = index;
}


uint iDivCeil(uint a, uint b) {
    return (a % b == 0) ? (a / b) :  a / b + 1;
}

void calculateGridSize(uint count, uint blockSize, uint &blocks, uint &threads) {
    threads = min(count, blockSize);
    blocks = iDivCeil(count, threads);
}

void calculateHash(
    uint* particlesGridHash,
    uint* particlesGridIndex,
    float* positions,
    uint count
) {
    uint blocks, threads;
    calculateGridSize(count, 256, blocks, threads);
    calculateHashKernel<<<blocks, threads>>>(
        particlesGridHash,
        particlesGridIndex,
        (float4*) positions,
        count
    );
}

void integrateSystem(float *positions,
                     float *velocities,
                     float deltaTime,
                     uint numParticles
) {

    thrust::device_ptr<float4> d_pos4((float4 *)positions);
    thrust::device_ptr<float4> d_vel4((float4 *)velocities);

    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4)),
        thrust::make_zip_iterator(thrust::make_tuple(d_pos4+numParticles, d_vel4+numParticles)),
        integrate_functor(deltaTime));

}

void sortParticles(uint *gridHash, uint *gridIndex, uint count)
{
    thrust::sort_by_key(thrust::device_ptr<uint>(gridHash),
                        thrust::device_ptr<uint>(gridHash + count),
                        thrust::device_ptr<uint>(gridIndex));
}

__global__ void reorderDataAndFindCellStartKernel(
    uint*  cellStart,
    uint*  cellEnd,
    float4* sortedPositions,
    float4* sortedVelocities,
    uint*  gridHash,
    uint*  gridIndex,
    float4* oldPositions,
    float4* oldVelocities,
    uint   count,
    uint   cells
) {
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    uint hash;
    if (index < count) {
        hash = gridHash[index];
        sharedHash[threadIdx.x+1] = hash;
        if (index > 0 && threadIdx.x == 0) {
            sharedHash[0] = gridHash[index-1];
        }
    }

    __syncthreads();

    if (index < count) {
        if (index == 0 || hash != sharedHash[threadIdx.x]) {
            cellStart[hash] = index;
            if (index > 0) {
                cellEnd[sharedHash[threadIdx.x]] = index;
            }
        }

        if (index == count - 1) {
            cellEnd[hash] = index + 1;
        }

        uint sortedIndex = gridIndex[index];
        float4 pos = oldPositions[sortedIndex];
        float4 vel = oldVelocities[sortedIndex];

        sortedPositions[index] = pos;
        sortedVelocities[index] = vel;
    }
}

void reorderDataAndFindCellStart(
    uint*  cellStart,
    uint*  cellEnd,
    float* sortedPositions,
    float* sortedVelocities,
    uint*  gridHash,
    uint*  gridIndex,
    float* oldPositions,
    float* oldVelocities,
    uint   count,
    uint   cells
) {
    uint threads, blocks;
    calculateGridSize(count, 256, blocks, threads);

    // set all cells to empty
    cutilSafeCall(cudaMemset(cellStart, 0xffffffff, cells*sizeof(uint)));

    uint shmSize = sizeof(uint)*(threads+1);
    reorderDataAndFindCellStartKernel<<< blocks, threads, shmSize>>>(
        cellStart,
        cellEnd,
        (float4 *) sortedPositions,
        (float4 *) sortedVelocities,
        gridHash,
        gridIndex,
        (float4 *) oldPositions,
        (float4 *) oldVelocities,
        count,
        cells
    );
    cutilCheckMsg("Kernel execution failed: reorderDataAndFindCellStartD");
}

// collide two spheres using DEM method
__device__
float3 collideSpheres(float3 posA, float3 posB,
                      float3 velA, float3 velB,
                      float radiusA, float radiusB,
                      float attraction)
{
    // calculate relative position
    float3 relPos = posB - posA;

    float dist = length(relPos);
    float collideDist = radiusA + radiusB;


    // DEM method http://http.developer.nvidia.com/GPUGems3/gpugems3_ch29.html
    float3 force = make_float3(0.0f);
    if (dist < collideDist) {
        float3 norm = relPos / dist;

        // relative velocity
        float3 relVel = velB - velA;

        // relative tangential velocity
        float3 tanVel = relVel - (dot(relVel, norm) * norm);

        // spring force
        force = -parameters.spring*(collideDist - dist) * norm;
        // dashpot (damping) force
        force += parameters.damping*relVel;
        // tangential shear force
        force += parameters.shear*tanVel;
        // attraction
        force += attraction*relPos;
    }

    return force;
}



// collide a particle against all other particles in a given cell
__device__
float3 collideCell(int3    gridPos,
                   uint    index,
                   float3  pos,
                   float3  vel,
                   float4* oldPos,
                   float4* oldVel,
                   uint*   cellStart,
                   uint*   cellEnd)
{
    uint gridHash = deviceCalculateGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = cellStart[gridHash];

    float3 force = make_float3(0.0f);
    if (startIndex != 0xffffffff) {        // cell is not empty
        // iterate over particles in this cell
        uint endIndex = cellEnd[gridHash];
        for(uint j=startIndex; j<endIndex; j++) {
            if (j != index) {              // check not colliding with self
                float3 pos2 = make_float3(oldPos[j]);
                float3 vel2 = make_float3(oldVel[j]);

                // collide two spheres
                force += collideSpheres(pos, pos2, vel, vel2, parameters.particleRadius, parameters.particleRadius, parameters.attraction);
            }
        }
    }
    return force;
}

__global__ void collideKernel(
    float4* newVelocities,
    float4* oldPositions,
    float4* oldVelocities,
    uint*   gridIndex,
    uint*   cellStart,
    uint*   cellEnd,
    uint    count
) {
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= count) return;

    // read particle data from sorted arrays
    float3 pos = make_float3(oldPositions[index]);
    float3 vel = make_float3(oldVelocities[index]);

    // get address in grid
    int3 gridPos = deviceCalculateGridPosition(pos);

    // examine neighbouring cells
    float3 force = make_float3(0.0f);
    for(int z=-1; z<=1; z++) {
        for(int y=-1; y<=1; y++) {
            for(int x=-1; x<=1; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                force += collideCell(neighbourPos, index, pos, vel, oldPositions, oldVelocities, cellStart, cellEnd);
            }
        }
    }

    // write new velocity back to original unsorted location
    uint originalIndex = gridIndex[index];
    newVelocities[originalIndex] = make_float4(vel + force, 0.0f);
}

void collide(
    float* newVelocities,
    float* sortedPositions,
    float* sortedVelocities,
    uint*  gridIndex,
    uint*  cellStart,
    uint*  cellEnd,
    uint   count,
    uint   cells
) {
    uint threads, blocks;
    calculateGridSize(count, 64, blocks, threads);
    // execute the kernel
    collideKernel<<< blocks, threads >>>(
        (float4*)newVelocities,
        (float4*)sortedPositions,
        (float4*)sortedVelocities,
        gridIndex,
        cellStart,
        cellEnd,
        count
    );

    // check if kernel invocation generated an error
    cutilCheckMsg("Kernel execution failed");
}

void setParameters(ParticleSystemParameters *particleSystemParameters) {
    // copy parameters to constant memory
    cutilSafeCall(cudaMemcpyToSymbol(parameters, particleSystemParameters, sizeof(ParticleSystemParameters)));
}