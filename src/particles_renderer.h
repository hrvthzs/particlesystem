#ifndef __PARTICLES_RENDERER_H__
#define __PARTICLES_RENDERER_H__

// OpenGL includes
# include <GL/glew.h>
#include <SDL/SDL.h>

#include "shader_program.h"
#include "particle_system.h"
#include "particles_simulator.h"

// Replacement for gluErrorString
const char * getGlErrorString(GLenum error);

struct SDL_Exception : public std::runtime_error
{
    SDL_Exception() throw()
        : std::runtime_error(std::string("SDL : ") + SDL_GetError()) {}
    SDL_Exception(const char * text) throw()
        : std::runtime_error(std::string("SDL : ") + text + " : " + SDL_GetError()) {}
    SDL_Exception(const std::string & text) throw()
        : std::runtime_error(std::string("SDL : ") + text + " : " + SDL_GetError()) {}
};

struct GL_Exception : public std::runtime_error
{
    GL_Exception(const GLenum error = glGetError()) throw()
        : std::runtime_error(std::string("OpenGL : ") + (const char*)getGlErrorString(error)) {}
    GL_Exception(const char * text, const GLenum error = glGetError()) throw()
        : std::runtime_error(std::string("OpenGL : ") + text + " : " + getGlErrorString(error)) {}
    GL_Exception(const std::string & text, const GLenum error = glGetError()) throw()
        : std::runtime_error(std::string("OpenGL : ") + text + " : " + getGlErrorString(error)) {}
};


namespace Particles {
    class Renderer {
        public:

            Renderer(Simulator* simulator);
            virtual ~Renderer();

            bool render();

        private:

            void _render();
            void _render(unsigned period);
            void _initSDL(unsigned depth, unsigned stencil);
            void _createSDLSurface();
            void _onInit();
            void _onWindowResized(int width, int height);
            void _onWindowRedraw();
            void _onKeyDown(SDLKey key, Uint16 mod);
            void _onKeyUp(SDLKey key, Uint16 mod);
            void _onMouseMove(
                unsigned x,
                unsigned y,
                int xrel,
                int yrel,
                Uint8 buttons
            );
            void _onMouseDown(Uint8 button, unsigned x, unsigned y);
            void _onMouseUp(Uint8 button, unsigned x, unsigned y);

            void _redraw();
            void _stop();

            Simulator* _simulator;
            ShaderProgram* _shaderProgram;
            SDL_Surface* _SDLSurface;

            GLuint _positionAttribute;
            GLuint _colorAttribute;
            GLuint _mvpUniform;
            GLuint _mvUniform;
            // window dimensions x -> width, y -> height
            GLuint _windowUniform;
            GLuint _aspectRatioUniform;
            GLuint _pointScale;
            GLuint _pointRadius;

            int _windowWidth;
            int _windowHeight;
            float _aspectRatio;

            unsigned _colorBits;

            float _rotationX;
            float _rotationY;
            float _translationZ;

            ////////////////////////////////////////////////////////////////////
            // TEMPORARY SOLUTION
            ////////////////////////////////////////////////////////////////////
            GLuint _dataVBO, _dataEBO;
            ParticleSystem* _particleSystem;

            unsigned int _numParticles;

            GLuint _vbo;
            GLuint _colorsVBO;
            bool _animate;
            bool _dynamicColoring;
            float _deltaTime;

            void _createVBO(
                GLuint* vbo,
                struct cudaGraphicsResource **vbo_res,
                unsigned int vbo_res_flags
            );

            void _deleteVBO(
                GLuint* vbo,
                struct cudaGraphicsResource *vbo_res
            );

            void _runCuda();

    };
};

#endif // __PARTICLES_RENDERER_H__