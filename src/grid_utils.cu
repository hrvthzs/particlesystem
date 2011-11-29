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
            float3 cell = (position - params.min) * params.delta;

            return make_int3(
                ceil(cell.x)-1.0f,
                ceil(cell.y)-1.0f,
                ceil(cell.z)-1.0f
            );
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

            // wrap grid... but since we can not assume size is power of 2
            // we can't use binary AND/& :/
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

        template<class C, class D>
        __device__ void iterateCell(
            D &data,
            int3 const &cell,
            uint const &index,
            float3 const &position,
            GridData const &gridData
        ) {
            volatile uint hash = computeCellHash(cell, cudaGridParams);
            volatile uint cellStart = gridData.cellStart[hash];

            if (cellStart != EMPTY_CELL_VALUE) {

                volatile uint cellStop = gridData.cellStop[hash];

                for (uint indexN = cellStart; indexN<cellStop; indexN++) {
                    C::processNeighbour(data, index, indexN, position);
                }

                // debug
                data.sorted.neighbours[index] += 1.0f;
            }
        }

        ////////////////////////////////////////////////////////////////////////

        template<class C, class D>
        __device__ void iterateNeighbourCells(
            uint const &index,
            float3 const &position,
            D &data,
            GridData const &gridData
        ) {

            C::preProcess(data, index);

            // debug
            data.sorted.neighbours[index] = 0.0f;

            volatile int3 cell = computeCellPosition(position, cudaGridParams);
            data.sorted.cellPos[index] = make_int3(cell.x, cell.y, cell.z);
            for (int z=cell.z-1; z<=cell.z+1; z++) {
                for (int y=cell.y-1; y<=cell.y+1; y++) {
                    for (int x=cell.x-1; x<=cell.x+1; x++) {
                        iterateCell<C,D>(
                            data,
                            make_int3(x,y,z),
                            index,
                            position,
                            gridData
                        );
                    }
                }
            }

            C::postProcess(data, index);
        }
    };
};

#endif // __GRID_UTILS_CU__