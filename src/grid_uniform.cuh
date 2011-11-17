#ifndef __GRID_UNIFORM_CUH__
#define __GRID_UNIFORM_CUH__

#include "thrust/sort.h"

#include "grid.cuh"
#include "buffer_manager.cuh"

namespace Grid {

    class Uniform {

        public:
            Uniform();
            virtual ~Uniform();

            GridParams getParams() const;
            GridData getData();
            uint getNumCells() const;

            void allocate(uint numParticles, float cellSize, float gridSize);
            void free();

            void hash();
            void sort();

        private:

            void _calcParams(float cellSize, float gridSize);

            Buffer::Manager<GridBuffers>* _cellBufferManager;
            Buffer::Manager<GridBuffers>* _particleBufferManager;

            GridParams _params;
            bool _allocated;
            uint _numParticles;
            uint _numCells;

            thrust::device_ptr<uint>* _thrustHash;
            thrust::device_ptr<uint>* _thrustIndex;
    };

};

#endif // __GRID_UNIFORM_CUH__