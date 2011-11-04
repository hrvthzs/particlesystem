#ifndef __SPH_SIMULATOR_CUH__
#define __SPH_SIMULATOR_CUH__

#include "cutil_math.h"
#include "sph.h"
#include "buffer_manager.cuh"
#include "buffer_vertex.h"
#include "particles_simulator.h"

namespace SPH {

    class Simulator : public Particles::Simulator {

        public:
            Simulator();
            ~Simulator();

            /**
             * Initialize simulator
             *
             * !!! Important !!!
             * Don't call before the GL context is created
             * else can cause segmentation fault
             * Must be called as first method
             */
            void init();
            void stop();
            /**
             *
             */
            void update();
            float* getPositions();
            void bindBuffers();
            void unbindBuffers();
            void integrate (int numParticles, float deltaTime, float4* pos);
            //virtual Buffer::Vertex<float>* getPositionsBuffer();

        protected:
            uint _iDivUp(uint a, uint b);
            void _computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads);

            Buffer::Manager<sph_buffer_t> *_bufferManager;
            Buffer::Vertex<float4>* _positionsBuffer;
    };

};

#endif // __SPH_SIMULATOR_CUH__
