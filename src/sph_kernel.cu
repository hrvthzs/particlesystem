#ifndef __SPH_INTEGRATOR_CU__
#define __SPH_INTEGRATOR_CU__

#include "grid.cuh"

namespace SPH {

    namespace Kernel {

        ////////////////////////////////////////////////////////////////////////

        __global__ void integrate(
            int numParticles,
            float deltaTime,
            float4 *pos
        ) {
            int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
            if (index >= numParticles) return;

            pos[index].y -= 0.001f;


        }

        ////////////////////////////////////////////////////////////////////////

        template<class D>
        __global__ void update (
            uint numParticles,
            D unsortedData,
            D sortedData,
            GridData gridData
        ) {
            uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
            if (index < numParticles) {
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

            uint sortedIndex = index;//gridData.index[index];

            sortedData.position[index] = unsortedData.position[sortedIndex];
            sortedData.velocity[index] = unsortedData.velocity[sortedIndex];

        }

        ////////////////////////////////////////////////////////////////////////

    };
};

#endif // __SPH_INTEGRATOR_CU__
