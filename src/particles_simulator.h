#ifndef __PARTICLES_SIMULATOR_H__
#define __PARTICLES_SIMULATOR_H__

#include <GL/glew.h>

#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <vector_types.h>

#include "particles.h"
#include "settings.h"
#include "settings_database.h"
#include "settings_updatecallback.h"

namespace Particles {

    class Simulator : public Settings::UpdateCallback {

    public:
        /**
         * Constructor
         */
        Simulator();

        /**
         * Desctructor
         */
        virtual ~Simulator();

        /**
         * Init simulator
         * @param numParticles number of particles
         */
        virtual void init(unsigned int numParticles) = 0;

        /**
         * Stop simulation
         */
        virtual void stop() = 0;

        /**
         * Update simulation
         * @param x gravity vector x
         * @param y gravity vector y
         * @param z gravity vector z
         */
        virtual void update(bool animate, float x, float y, float z) = 0;

        /**
         * Get positions of particles
         */
        virtual float* getPositions() = 0;

        /**
         * Get colors of particles
         */
        virtual float* getColors() = 0;

        /**
         * Get positions VBO of particles
         */
        virtual GLuint getPositionsVBO();

        /**
         * Get colors VBO of particles
         */
        virtual GLuint getColorsVBO();

        /**
         * Get normals VBO of particles
         */
        virtual GLuint getNormalsVBO();

        /**
         * Bind simulator buffers
         */
        virtual void bindBuffers() = 0;

        /**
         * Unbind simulator buffers
         */
        virtual void unbindBuffers() = 0;

        /**
         * Get number of particles
         */
        unsigned int getNumParticles();

        /**
         * Get number of generated vertices for rendering
         */
        virtual unsigned int getNumVertices() = 0;

        /**
         * Generate particle positions
         */
        virtual void generateParticles() = 0;

        /**
         * Switch between rendering modes of simulator
         *
         */
        virtual void setRenderMode(int mode) = 0;

        /**
         * Get rendering mode of simulator
         *
         */
        virtual int getRenderMode() = 0;

        /**
         * Get struct containing wrapper grid min and max coords
         */
        virtual GridMinMax getGridMinMax() = 0;

        /**
         * Set value for requested record
         * @param type record type
         * @param value record value
         */
        virtual void setValue(Settings::RecordType type, float value);

        /**
         * Get current value of requested record
         * @param type record type
         */
        virtual float getValue(Settings::RecordType type);

        /**
         * Abstract method of settings updatecallback
         * This method should call callback for specified record type
         * @param type record type
         */
        virtual void valueChanged(Settings::RecordType type);

        /**
         * Set animated
         *
         * @param animate - flag
         */
        virtual void setAnimated(bool animate);

        /**
         * Is animated
         */
        virtual bool isAnimated();


    protected:

        bool _animation;

        unsigned int _numParticles;
        unsigned int _numVertices;

        GLuint _positionsVBO;
        GLuint _colorsVBO;
        GLuint _normalsVBO;

        Settings::Database* _database;

    };
};

#endif // __PARTICLES_SIMULATOR_H__
