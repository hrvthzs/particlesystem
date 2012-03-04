#ifndef __PARTICLES_RENDERER_H__
#define __PARTICLES_RENDERER_H__

// OpenGL includes
#include <GL/glew.h>
#include <SDL/SDL.h>

#include <cutil_math.h>
#include "shader_program.h"
#include "particles.h"
#include "particles_simulator.h"

namespace Particles {

    // Replacement for gluErrorString
    const char * getGlErrorString(GLenum error);

    struct SDL_Exception : public std::runtime_error {
        SDL_Exception() throw() :
            std::runtime_error(
                std::string("SDL : ") + SDL_GetError()
            ) {}

        SDL_Exception(const char * text) throw() :
            std::runtime_error(
                std::string("SDL : ") + text + " : " + SDL_GetError()
            ) {}

        SDL_Exception(const std::string & text) throw() :
            std::runtime_error(
                std::string("SDL : ") + text + " : " + SDL_GetError()
            ) {}
    };

    struct GL_Exception : public std::runtime_error {
        GL_Exception(const GLenum error = glGetError()) throw() :
            std::runtime_error(
                std::string("OpenGL : ") + (const char*) getGlErrorString(error)
            ) {}

        GL_Exception(
            const char* text,
            const GLenum error = glGetError()
        ) throw() :
            std::runtime_error(
                std::string("OpenGL : ") +
                text + " : " +
                getGlErrorString(error)
            ) {}

        GL_Exception(
            const std::string & text,
            const GLenum error = glGetError()
        ) throw() :
            std::runtime_error(
                std::string("OpenGL : ") +
                text + " : " +
                getGlErrorString(error)
            ) {}
    };

    class Renderer {
        public:

            /**
             * @param simulator inherited Simulator instance
             */
            Renderer(Simulator* simulator);

            /**
             * Destructor
             */
            virtual ~Renderer();

            /**
             * Render results of simulator
             */
            bool render();

        private:

            /**
             * Run rendering loop
             */
            void _render();

            /**
             * Render periodically
             * @param period period in ms
             */
            void _render(unsigned period);

            /**
             * Init SDL
             */
            void _initSDL(unsigned depth, unsigned stencil);

            /**
             * Create surface for SDL
             */
            void _createSDLSurface();

            /**
             * Init resources callback
             */
            void _onInit();

            /**
             * Window resize callback
             */
            void _onWindowResized(int width, int height);

            /**
             * Window redraw callback
             */
            void _onWindowRedraw();

            /**
             * Key down callback
             */
            void _onKeyDown(SDLKey key, Uint16 mod);

            /**
             * Key up callback
             */
            void _onKeyUp(SDLKey key, Uint16 mod);

            /**
             * Mouse move callback
             */
            void _onMouseMove(
                unsigned x,
                unsigned y,
                int xrel,
                int yrel,
                Uint8 buttons
            );

            /**
             * Mouse down callback
             */
            void _onMouseDown(Uint8 button, unsigned x, unsigned y);

            /**
             * Mouse up callback
             */
            void _onMouseUp(Uint8 button, unsigned x, unsigned y);

            /**
             * Force redrawing
             */
            void _redraw();

            /**
             * Stop rendering
             */
            void _stop();

            /**
             * Create wrapping cube vertices
             */
            void _createCube();

            Simulator* _simulator;
            SDL_Surface* _SDLSurface;
            Shader::Program* _shaderProgram;
            Shader::Program* _marchingProgram;
            Shader::Program* _cubeProgram;
            Shader::Program* _tesselationProgram;
            Shader::Program* _normalsProgram;

            bool _animate;
            bool _dynamicColoring;

            int _windowWidth;
            int _windowHeight;

            uint _colorBits;
            uint _tessLevelInner;
            uint _tessLevelOuter;
            uint _numParticles;

            float _aspectRatio;
            float _rotationX;
            float _rotationY;
            float _translationZ;

            RenderMode _renderMode;

            GLuint _dataVBO;
            GLuint _dataEBO;
            GLuint _positionsVBO;
            GLuint _colorsVBO;
            GLuint _normalsVBO;
            GLuint _cubeVBO;

    };
};

#endif // __PARTICLES_RENDERER_H__