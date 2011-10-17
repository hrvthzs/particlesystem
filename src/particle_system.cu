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
        //vel *= params.globalDamping;

        // new position = old position + velocity * deltaTime
        pos += vel * deltaTime;

        float particleRadius = parameters.particleRadius;
        float boundaryDamping = parameters.boundaryDamping;

        // set this to zero to disable collisions with cube sides
        #if 1
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

void setParameters(ParticleSystemParameters *particleSystemParameters) {
    // copy parameters to constant memory
    cutilSafeCall(cudaMemcpyToSymbol(parameters, particleSystemParameters, sizeof(ParticleSystemParameters)));
}