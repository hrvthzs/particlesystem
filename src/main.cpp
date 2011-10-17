/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 3
 * of the programming guide with some additions like error checking.
 *
 */

//define debug uncomment this if using large arrays


////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
// OpenGL includes
#include <GL/glew.h>

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

#include "main.h"
#include "particle_system.h";
#include "shader_program.h"


#define REFRESH_DELAY     10 //ms

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;

float g_fAnim = 0.0;
bool render = false;
// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

extern "C"
void launch_kernel(float4* pos, unsigned int mesh_width, unsigned int mesh_height, float time);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Geometry Data
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

const GLenum dataIndexType = GL_UNSIGNED_SHORT;


struct vertices {
        float position[3];
} pointVertices[3] = {
    {{ 0.0, 0.0, 0.0 }},
    {{ 0.5, 0.5, 0.0 }},
    {{ 0.5, 0.0, 0.0 }}
};
const unsigned short point[1][3] =
{
    { 0, 1, 2 }

};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Data
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GLuint dataVBO, dataEBO;

int width, height;// Window size
float rx = 0.0f, ry = 0.0f, pz = -2.0f;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Shaders
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
ShaderProgram *shaderProgram;

GLuint positionAttrib;
GLuint mvpUniform;

void createVBO(GLuint* vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags);
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource *vbo_res);
void cleanup();

void runCuda(struct cudaGraphicsResource **vbo_resource);
ParticleSystem *pSystem;
////////////////////////////////////////////////////////////////////////////////

// Simple main loop
void mainLoop();
void mainLoop(unsigned period);

// Host code
int main(int, char**)
{
    try {
        if(SDL_Init(SDL_INIT_VIDEO) < 0) throw SDL_Exception();

        // Shutdown SDL when program ends
        atexit(SDL_Quit);

        initSDL(800, 600, 24, 24, 0);

        mainLoop(10);

    } catch(SDL_Exception & ex) {
        cout << "ERROR : " << ex.what() << endl;
        return EXIT_FAILURE;
    }

}

void onInit()
{

    shaderProgram = new ShaderProgram("shaders/shader.vs", "shaders/shader.fs");

    positionAttrib = shaderProgram->getAttributeLocation("position");
    mvpUniform = shaderProgram->getUniformLocation("mvp");

    // Copy data to graphics card
    //glGenBuffers(1, &dataVBO);
    //glBindBuffer(GL_ARRAY_BUFFER, dataVBO);
    //glBufferData(GL_ARRAY_BUFFER, sizeof(pointVertices), pointVertices, GL_STATIC_DRAW);

    //glGenBuffers(1, &dataEBO);
    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, dataEBO);
    //glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(point), point, GL_STATIC_DRAW);


    cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
    uint3 gridSize;

    gridSize.x = gridSize.y = gridSize.z = 10;
    pSystem = new ParticleSystem(mesh_height*mesh_width, gridSize);

    vbo = pSystem->getPositionsVBO();
    cuda_vbo_resource = pSystem->getCudaPositionsVBOResource();

    //createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
    runCuda(&cuda_vbo_resource);
}

void onWindowRedraw()
{
    //runCuda(&cuda_vbo_resource);
    if (render)
        pSystem->update(0.02f);
    // run CUDA kernel to generate vertex positions


    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);

    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);

    // Calculate ModelViewProjection matrix

    glm::mat4 projection = glm::perspective(45.0f, (float)width/(float)height, 1.0f, 1000.0f);

    glm::mat4 mv = glm::rotate(
            glm::rotate(
                glm::translate(
                    glm::mat4(1.0f),
                    glm::vec3(0, 0, pz)
                    ),
                ry, glm::vec3(1, 0, 0)
                ),
            rx, glm::vec3(0, 1, 0)
            );

    glm::mat4 mvp = projection*mv;

    shaderProgram->use();

    // Set matrices
    glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, glm::value_ptr(mvp));

    glEnableVertexAttribArray(positionAttrib);

    // Draw data
    /*glBindBuffer(GL_ARRAY_BUFFER, dataVBO);
    glVertexAttribPointer(positionAttrib, 3, GL_FLOAT, GL_FALSE, sizeof(vertices), (void*)offsetof(vertices, position));

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, dataEBO);
    glDrawElements(GL_TRIANGLES, sizeof(point)/sizeof(**point), dataIndexType, NULL);*/


    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(positionAttrib, 4, GL_FLOAT, GL_FALSE, 0, (void*) 0);


    glDrawArrays(GL_POINTS, 0, mesh_height * mesh_height);


    SDL_GL_SwapBuffers();

        g_fAnim += 0.01f;
}

void onWindowResized(int w, int h)
{
    glViewport(0, 0, w, h);
    width = w; height = h;

    // TODO reinit screen, solve this in other way
    SDL_SetVideoMode(width, height, 24, SDL_OPENGL | SDL_RESIZABLE);
}

void onKeyDown(SDLKey key, Uint16 /*mod*/)
{
    switch(key) {
        case SDLK_ESCAPE : quit(); deleteVBO(&vbo, cuda_vbo_resource); break;
        case SDLK_r: render = !render; break;
        case SDLK_c: runCuda(&cuda_vbo_resource);
        default : break;//nothing-doing defaut to shut up warning
    }
}

void onKeyUp(SDLKey /*key*/, Uint16 /*mod*/)
{

}

void onMouseMove(unsigned /*x*/, unsigned /*y*/, int xrel, int yrel, Uint8 buttons)
{
    if(buttons & SDL_BUTTON_LMASK)
    {
        rx += xrel;
        ry += yrel;
        redraw();
    }
    if(buttons & SDL_BUTTON_RMASK)
    {
        pz += yrel*0.1;
        redraw();
    }
}

void onMouseDown(Uint8 /*button*/, unsigned /*x*/, unsigned /*y*/)
{

}

void onMouseUp(Uint8 /*button*/, unsigned /*x*/, unsigned /*y*/)
{

}

// Simple main loop
void mainLoop()
{
    // Window is not minimized
    bool active = true;

    for(;;)// Infinite loop
    {
        SDL_Event event;

        // Wait for event
        if(SDL_WaitEvent(&event) == 0) throw SDL_Exception();

        // Screen needs redraw
        bool redraw = false;

        // Handle all waiting events
        do
        {
            // Call proper event handlers
            switch(event.type)
            {
                case SDL_ACTIVEEVENT :// Stop redraw when minimized
                    if(event.active.state == SDL_APPACTIVE)
                        active = event.active.gain;
                    break;
                case SDL_KEYDOWN :
                    onKeyDown(event.key.keysym.sym, event.key.keysym.mod);
                    break;
                case SDL_KEYUP :
                    onKeyUp(event.key.keysym.sym, event.key.keysym.mod);
                    break;
                case SDL_MOUSEMOTION :
                    onMouseMove(event.motion.x, event.motion.y, event.motion.xrel, event.motion.yrel, event.motion.state);
                    break;
                case SDL_MOUSEBUTTONDOWN :
                    onMouseDown(event.button.button, event.button.x, event.button.y);
                    break;
                case SDL_MOUSEBUTTONUP :
                    onMouseUp(event.button.button, event.button.x, event.button.y);
                    break;
                case SDL_QUIT :
                    return;// End main loop
                case SDL_VIDEORESIZE :
                    onWindowResized(event.resize.w, event.resize.h);
                    break;
                case SDL_VIDEOEXPOSE :
                    redraw = true;
                    break;
                default :// Do nothing
                    break;
            }
        } while(SDL_PollEvent(&event) == 1);

        // Optionally redraw window
        if(active && redraw) onWindowRedraw();
    }
}


// Animation main loop
// period - maximum time between redraws in ms
void mainLoop(unsigned period)
{
    // This main loop requires timer support
    if(SDL_InitSubSystem(SDL_INIT_TIMER) < 0) throw SDL_Exception();

    // Create redraw timer
    class RedrawTimer
    {
        private :
            SDL_TimerID id;
            static Uint32 callback(Uint32 interval, void *)
            {
                redraw();
                return interval;
            }
        public :
            RedrawTimer(unsigned interval)
                : id(SDL_AddTimer(interval, callback, NULL))
            {
                if(id == NULL) throw SDL_Exception();
            }
            ~RedrawTimer()
            {
                if(id != NULL) SDL_RemoveTimer(id);
            }
    } redrawTimer(period);

    // Start simple main loop
    mainLoop();
}


////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint* vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags)
{
    if (vbo) {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
        glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // register this buffer object with CUDA
        // DEPRECATED: cutilSafeCall(cudaGLRegisterBufferObject(*vbo));
        cutilSafeCall(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

        CUT_CHECK_ERROR_GL();
    } else {
        cutilSafeCall( cudaMalloc( (void **)&d_vbo_buffer, mesh_width*mesh_height*4*sizeof(float) ) );
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource *vbo_res)
{
    if (vbo) {
        // unregister this buffer object with CUDA
        //DEPRECATED: cutilSafeCall(cudaGLUnregisterBufferObject(*pbo));
        cudaGraphicsUnregisterResource(vbo_res);

        glBindBuffer(1, *vbo);
        glDeleteBuffers(1, vbo);

        *vbo = 0;
    } else {
        cudaFree(d_vbo_buffer);
        d_vbo_buffer = NULL;
    }
}

void cleanup()
{
    deleteVBO(&vbo, cuda_vbo_resource);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    // DEPRECATED: cutilSafeCall(cudaGLMapBufferObject((void**)&dptr, vbo));
    cutilSafeCall(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
                                                       *vbo_resource));
    //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

    // execute the kernel
    //    dim3 block(8, 8, 1);
    //    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    //    kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);

    launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);

    // unmap buffer object
    // DEPRECATED: cutilSafeCall(cudaGLUnmapBufferObject(vbo));
    cutilSafeCall(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}
