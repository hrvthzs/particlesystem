#ifndef __PARTICLES_SIMULATOR_H__
#define __PARTICLES_SIMULATOR_H__

#include <GL/glew.h>

#include "settings.h"
#include "settings_updatecallback.h"

namespace Particles{

    class Simulator : public Settings::UpdateCallback {

    public:
        Simulator();
        virtual ~Simulator();

        virtual void init(unsigned int numParticles) = 0;
        virtual void stop() = 0;
        virtual void update() = 0;
        virtual float* getPositions() = 0;
        virtual float* getColors() = 0;
        virtual GLuint getPositionsVBO();
        virtual GLuint getColorsVBO();
        virtual void bindBuffers() = 0;
        virtual void unbindBuffers() = 0;
        unsigned int getNumParticles();

        virtual void generateParticles() = 0;

        virtual void valueChanged(Settings::RecordType type);

    protected:
        unsigned int _numParticles;

        GLuint _positionsVBO;
        GLuint _colorsVBO;

    };
};

#endif // __PARTICLES_SIMULATOR_H__
