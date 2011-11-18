#ifndef __GRID_KERNEL_CU__
#define __GRID_KERNEL_CU__

#include <cutil_math.h>

#include "grid.cuh"
#include "grid_utils.cu"

namespace Grid {

    namespace Kernel {

        ////////////////////////////////////////////////////////////////////////

        /**
         * Compute hash code for each particle
         *
         * @param numParticles number of particles
         * @param positions array of particles' positions
         * @param data GridData where the hash codes are stored
         */
        __global__ void hash(
            uint numParticles,
            float4* positions,
            GridData data
        ) {

            uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
            if (index >= numParticles) {
                return;
            }

            float3 position = make_float3(positions[index]);

            int3 cellPosition =
                Grid::Utils::computeCellPosition(position, cudaGridParams);
            uint hash =
                Grid::Utils::computeCellHash(cellPosition, cudaGridParams);

            data.index[index] = index;
            data.hash[index] = hash;
        }

        ////////////////////////////////////////////////////////////////////////
    }
}

#endif //__GRID_KERNEL_CU__