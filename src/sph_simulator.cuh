#ifndef __SPH_SIMULATOR_CUH__
#define __SPH_SIMULATOR_CUH__

#include <cutil_math.h>
#include "buffer_manager.cuh"
#include "buffer_vertex.h"
#include "grid_uniform.cuh"
#include "marching_renderer.cuh"
#include "particles.h"
#include "particles_simulator.h"
#include "settings.h"
#include "settings_database.h"
#include "sph.h"

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
            void init(uint numParticles);
            void stop();
            /**
             *
             */
            void update(float x, float y, float z);
            float* getPositions();
            float* getColors();
            void bindBuffers();
            void unbindBuffers();
            //virtual Buffer::Vertex<float>* getPositionsBuffer();

            void valueChanged(Settings::RecordType type);
            void generateParticles();
            uint getNumVertices();

            void setRenderMode(int mode);
            Particles::GridMinMax getGridMinMax();

        protected:


            Buffer::Manager<Buffers> *_bufferManager;

            Buffer::Vertex<float4>* _positionsBuffer;

            Grid::Uniform* _grid;

            Data _particleData;
            Data _sortedData;

            FluidParams _fluidParams;
            PrecalcParams _precalcParams;
            GridParams _gridParams;

            Marching::Renderer* _marchingRenderer;
            int _renderMode;

        private:
            void _integrate (float deltaTime, float3 gravity);
            void _createBuffers();
            void _orderData();
            void _updateParams();

            void _step1();
            void _step2();

    };

};

#endif // __SPH_SIMULATOR_CUH__
