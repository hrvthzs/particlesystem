#ifndef __GRID_UTILS_CU__
#define __GRID_UTILS_CU__

#include <cutil_math.h>

#include "grid.cuh"

namespace Grid {

    namespace Utils {

        ////////////////////////////////////////////////////////////////////////

        /**
         * Compute grid cell position
         *
         * @param position particle position
         * @param data grid parameters
         */
        inline __device__ int3 computeCellPosition(
            float3 const &position,
            GridParams const &params
        ) {
            // subtract grid_min (cell position) and multiply by delta
            return make_int3((position - params.min) * params.delta);
        }

        ////////////////////////////////////////////////////////////////////////

        /**
         * Compute grid cell hash
         *
         * @param position cell position
         * @param data grid parameters
         */
        inline __device__ uint computeCellHash(
            int3 const &position,
            GridParams const &params
        ) {
            // hash = x + y*width + z*width+height
            return
                position.x +
                __mul24(position.y, params.resolution.x) +
                __mul24(
                    params.resolution.x,
                    __umul24(position.z, params.resolution.y)
                );
        }

        ////////////////////////////////////////////////////////////////////////

    };
};

#endif // __GRID_UTILS_CU__