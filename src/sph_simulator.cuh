#ifndef __SPH_SIMULATOR_CUH__
#define __SPH_SIMULATOR_CUH__

#include "cutil_math.h"
#include "sph.h"
#include "buffer_manager.cuh"
#include "buffer_vertex.h"
#include "particles_simulator.h"

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

    class Simulator : public Particles::Simulator {

        public:
            Simulator();
            ~Simulator();
            void init();
            void update();
            float* getPositions();
            void bindBuffers();
            void unbindBuffers();
            void integrate (int numParticles, float deltaTime, float *pos);
            //virtual Buffer::Vertex<float>* getPositionsBuffer();

        protected:
            uint _iDivUp(uint a, uint b);
            void _computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads);

            Buffer::Manager<sph_buffer_t> *_bufferManager;
            Buffer::Vertex<float>* _positionsBuffer;
    };

};

#endif // __SPH_SIMULATOR_CUH__
