#ifndef __SPH_SIMULATOR_CUH__
#define __SPH_SIMULATOR_CUH__

#include "cutil_math.h"
#include "sph.h"
#include "buffer_manager.cuh"

namespace SPH {


    struct ParticleData {

        // position of each particle
        float4* position;

        // color of each particle
        float4* color;

        // current velocity vector
        float4* velocity;

        // sum of SPH forces at particle position
        float4* force;

        // pressure at particle position
        float4* pressure;

        // density at particle position
        float4* density;
    };

    class Simulator {

        public:
            Simulator();
            ~Simulator();
            void integrate (int numParticles, float deltaTime, float *pos);

        protected:
            uint _iDivUp(uint a, uint b);
            void _computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads);

            Buffer::Manager<sph_buffer_t> *_bufferManager;
    };

};

#endif // __SPH_SIMULATOR_CUH__
