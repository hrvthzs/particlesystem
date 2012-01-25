#ifndef __SPH_KERNEL_CU__
#define __SPH_KERNEL_CU__

__constant__ SPH::FluidParams    cudaFluidParams;
__constant__ SPH::PrecalcParams  cudaPrecalcParams;

#include "boundary_walls.cu"
#include "grid.cuh"
#include "grid_utils.cu"
#include "sph_density.cu"
#include "sph_force.cu"
#include "sph_neighbours.cu"

namespace SPH {

    namespace Kernel {

        using namespace Grid::Utils;

        ////////////////////////////////////////////////////////////////////////

        template<class D>
        __global__ void integrate(
            int numParticles,
            float deltaTime,
            D data,
            D sortedData,
            GridData gridData
        ) {
            int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
            if (index >= numParticles) {
                return;
            }

            float3 position = make_float3(sortedData.position[index]);
            float3 velocity = make_float3(sortedData.velocity[index]);
            float3 veleval = make_float3(sortedData.veleval[index]);

            float3 force = make_float3(sortedData.force[index]);
            //float pressure = sortedData.pressure[index];

            float3 externalForce = make_float3(0.0f, 0.0f, 0.0f);
            //externalForce.y -= 9.8f;

            // add no-penetration force due to "walls"
            externalForce += Boundary::Walls::calculateWallsNoPenetrationForce(
                    position, veleval,
                    cudaGridParams.min,
                    cudaGridParams.max,
                    cudaFluidParams.boundaryDistance,
                    cudaFluidParams.boundaryStiffness,
                    cudaFluidParams.boundaryDampening);

            // add no-slip force due to "walls"
            /*externalForce += Boundary::Walls::calculateWallsNoSlipForce(
                    position, veleval, force + externalForce,
                    cudaGridParams.min,
                    cudaGridParams.max,
                    cudaFluidParams.boundaryDistance,
                    cudaFluidParams.frictionKinetic/deltaTime,
                    cudaFluidParams.frictionStaticLimit);
            */
            float3 f = force + externalForce;

            float speed = length(force);

            if (speed > cudaFluidParams.velocityLimit) {
                f *= cudaFluidParams.velocityLimit / speed;
            }

            float3 vnext = velocity + f * deltaTime;
            veleval = vnext;//(velocity + vnext) * 0.5f;
            velocity = veleval;

            position += vnext * deltaTime;

            uint sortedIndex = gridData.index[index];

            /*if ((position.y - EPSILON) <= cudaGridParams.min.y) {
                position.y =  cudaGridParams.min.y + EPSILON;
            }

            if ((position.y + EPSILON) >= cudaGridParams.max.y) {
                position.y =  cudaGridParams.max.y - EPSILON;
            }*/


            data.position[sortedIndex] = make_float4(position, 1.0f);
            data.velocity[sortedIndex] = make_float4(velocity, 1.0f);
            data.veleval[sortedIndex] = make_float4(veleval, 1.0f);

            if (cudaFluidParams.dynamicColoring) {
                float3 color = (position - cudaGridParams.min) / cudaGridParams.size;
                data.color[sortedIndex] = make_float4(color, 1.0f);
            }

        }

        ////////////////////////////////////////////////////////////////////////


        // TODO this is the same for classical simulator, so place somewhere
        // where general codes are
        template<class D>
        __global__ void update (
            uint numParticles,
            D unsortedData,
            D sortedData,
            GridData gridData
        ) {
            uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
            if (index >= numParticles) {
                return;
            }

            extern __shared__ uint sharedHash[]; // blockSize + 1 elements

            uint hash = gridData.hash[index];

            sharedHash[threadIdx.x+1] = hash;

            if (index > 0 && threadIdx.x  == 0) {
                sharedHash[0] = gridData.hash[index-1];
            }

            __syncthreads();

            if (index == 0 || hash != sharedHash[threadIdx.x]) {
                gridData.cellStart[hash] = index;

                if (index > 0) {
                    gridData.cellStop[sharedHash[threadIdx.x]] = index;
                }
            }

            if (index == numParticles - 1) {
                gridData.cellStop[hash] = index + 1;
            }

            uint sortedIndex = gridData.index[index];

            sortedData.position[index] = unsortedData.position[sortedIndex];
            sortedData.velocity[index] = unsortedData.velocity[sortedIndex];

        }

        ////////////////////////////////////////////////////////////////////////

        template<class D>
        __global__ void computeDensity(
            uint numParticles,
            D sortedData,
            GridData gridData
        ) {
            uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
            if (index >= numParticles) {
                return;
            }

            float3 position = make_float3(sortedData.position[index]);

            Density::Data data;
            data.sorted = sortedData;

            iterateNeighbourCells<Neighbours<Density, Density::Data>, Density::Data>(
              index, position, data, gridData
            );
        }

        ////////////////////////////////////////////////////////////////////////

        template<class D>
        __global__ void computeForce(
            uint numParticles,
            D sortedData,
            GridData gridData
        ) {
            uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
            if (index >= numParticles) {
                return;
            }

            float3 position = make_float3(sortedData.position[index]);

            Force::Data data;
            data.sorted = sortedData;

            iterateNeighbourCells<Neighbours<Force, Force::Data>, Force::Data>(
                index, position, data, gridData
            );
        }

        ////////////////////////////////////////////////////////////////////////

    };
};

#endif // __SPH_KERNEL_CU__
