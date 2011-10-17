#ifndef MAIN_H
#define MAIN_H

#include <cassert>

#include <string>
#include <exception>
#include <stdexcept>
#include <iostream>


////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
// OpenGL includes
# include <GL/glew.h>
#include <SDL.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// CUDA utilities and system includes
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_gl_inline.h> // includes cuda_gl_interop.h// includes cuda_gl_interop.h
#include <cutil_gl_error.h>
// #include <rendercheck_gl.h>
// #include <vector_types.h>

// #include <stdlib.h>
// #include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iterator>

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Error handling
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Shaders
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Load whole file and return it as std::string
std::string loadFile(const char * const file);

// Common shader log code
#ifndef USE_GLEE
    std::string getInfoLog(GLuint id, PFNGLGETSHADERIVPROC getLen, PFNGLGETSHADERINFOLOGPROC getLog);
#else
    std::string getInfoLog(GLuint id, GLEEPFNGLGETSHADERIVPROC getLen, GLEEPFNGLGETSHADERINFOLOGPROC getLog);
#endif//USE_GLEE

// Info logs contain errors and warnings from shader compilation and linking
inline std::string getShaderInfoLog(GLuint shader)
{
    assert(glIsShader(shader));
    return getInfoLog(shader, glGetShaderiv, glGetShaderInfoLog);
}
inline std::string getProgramInfoLog(GLuint program)
{
    assert(glIsProgram(program));
    return getInfoLog(program, glGetProgramiv, glGetProgramInfoLog);
}

GLuint compileShader(const GLenum type, const char * source);

GLuint linkShader(size_t count, ...);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Event handlers
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void onInit();
void onWindowRedraw();
void onWindowResized(int width, int height);
void onKeyDown(SDLKey key, Uint16 mod);
void onKeyUp(SDLKey key, Uint16 mod);
void onMouseMove(unsigned x, unsigned y, int xrel, int yrel, Uint8 buttons);
void onMouseDown(Uint8 button, unsigned x, unsigned y);
void onMouseUp(Uint8 button, unsigned x, unsigned y);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Event functions
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Send quit event
inline void quit()
{
    SDL_Event event;
    event.type = SDL_QUIT;
    if(SDL_PushEvent(&event) < 0) throw SDL_Exception();
}

// Send redraw event
inline void redraw()
{
    SDL_Event event;
    event.type = SDL_VIDEOEXPOSE;
    if(SDL_PushEvent(&event) < 0) throw SDL_Exception();
}

SDL_Surface * initSDL(unsigned width, unsigned height, unsigned color, unsigned depth, unsigned stencil)
{
    // Set OpenGL attributes
    if(SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, color) < 0) throw SDL_Exception();
    if(SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, depth) < 0) throw SDL_Exception();
    if(SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, stencil) < 0) throw SDL_Exception();
    if(SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1) < 0) throw SDL_Exception();

    // Create window
    SDL_Surface * screen = SDL_SetVideoMode(width, height, color, SDL_OPENGL | SDL_RESIZABLE);
    if(screen == NULL) throw SDL_Exception();

#ifndef USE_GLEE
    // Init extensions
    GLenum error = glewInit();
    if(error != GLEW_OK)
        throw std::runtime_error(std::string("GLEW : Init failed : ") + (const char*)glewGetErrorString(error));
#endif//USE_GLEE

    // Call init code
    onInit();

    onWindowResized(width, height);

    return screen;
}

#endif