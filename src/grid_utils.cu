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

            int rx = (int) floor(params.resolution.x);
            int ry = (int) floor(params.resolution.y);
            int rz = (int) floor(params.resolution.z);

            // wrap grid... but since we can not assume size is power of 2 we can't use binary AND/& :/
            int px = position.x % rx;
            int py = position.y % ry;
            int pz = position.z % rz;

            if(px < 0) px += rx;
            if(py < 0) py += ry;
            if(pz < 0) pz += rz;

            // hash = x + y*width + z*width+height
            return
                px +
                __mul24(py, params.resolution.x) +
                __mul24(
                    params.resolution.x,
                    __umul24(pz, params.resolution.y)
                );
        }

        ////////////////////////////////////////////////////////////////////////

    };
};

#endif // __GRID_UTILS_CU__