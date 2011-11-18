#ifndef __GRID_UNIFORM_CUH__
#define __GRID_UNIFORM_CUH__

#include "thrust/sort.h"

#include "grid.cuh"
#include "buffer_manager.cuh"

namespace Grid {

    /**
     * Uniform Grid class
     *
     * Provides a way to arrange content in a grid where all the cells
     * in the grid have the same size.
     * Grid contains buffers for storing data such as hash code, indexes,
     * cellStart and cellStop indexes.
     */
    class Uniform {

        public:
            /**
             * Constructor
             */
            Uniform();

            /**
             * Destructor
             */
            virtual ~Uniform();

            /**
             * Get parameters of grid
             */
            GridParams getParams() const;

            /**
             * Get grid data
             *
             * @return struct which contains pointers to buffers (hashcodes,
             *         indexes, ..,)
             */
            GridData getData() const;

            /**
             * Get number of grid cells
             */
            uint getNumCells() const;

            /**
             * Allocate grid for given parameters
             */
            void allocate(uint numParticles, float cellSize, float gridSize);

            /**
             * Free grid and it's buffers
             */
            void free();

            /**
             * Hash particle positions
             * @param positions particles' positions
             */
            void hash(float4* positions);

            /**
             * Sort indexes
             */
            void sort();

        private:

            /**
             * Calculate parameters of grid
             *
             * @param cellSize size of a uniform cell
             * @param gridSize size of a uniform grid
             */
            void _calcParams(float cellSize, float gridSize);

            // Buffer managers
            Buffer::Manager<GridBuffers>* _cellBufferManager;
            Buffer::Manager<GridBuffers>* _particleBufferManager;

            GridParams _params;
            GridData _data;

            bool _allocated;
            uint _numParticles;
            uint _numCells;

            // Device pointer for sorting
            thrust::device_ptr<uint>* _thrustHash;
            thrust::device_ptr<uint>* _thrustIndex;
    };

};

#endif // __GRID_UNIFORM_CUH__