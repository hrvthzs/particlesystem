#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "particles_renderer.h"
#include "kernel.cuh"

namespace Particles {

    ////////////////////////////////////////////////////////////////////////////

    Renderer::Renderer(Simulator* simulator) {
        this->_simulator = simulator;

        this->_meshWidth = 128;
        this->_meshHeight = 128;
        this->_animate = false;
        this->_deltaTime = 0.0;

        this->_rotationX = 0.0f;
        this->_rotationY = 0.0f;
        this->_translationZ = -5.0f;

        this->_windowHeight = 600;
        this->_windowWidth = 800;
        this->_colorBits = 24;
        this->_SDLSurface = NULL;

    }

    ////////////////////////////////////////////////////////////////////////////

    Renderer::~Renderer() {
        delete this->_simulator;
        delete this->_shaderProgram;
        //delete this->_particleSystem;
    }

    ////////////////////////////////////////////////////////////////////////////

    bool Renderer::render() {
        try {
            if(SDL_Init(SDL_INIT_VIDEO) < 0) throw SDL_Exception();

            // Shutdown SDL when program ends
            atexit(SDL_Quit);

            this->_initSDL(24, 0);
            this->_simulator->init();
            this->_onInit();
            this->_runCuda();
            this->_render(10);

        } catch(SDL_Exception & ex) {
            cout << "ERROR : " << ex.what() << endl;
            return false;
        }

        return true;
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_initSDL(
        unsigned depth,
        unsigned stencil
    ) {

        unsigned color = this->_colorBits;

        // Set OpenGL attributes
        if(SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE,    color) < 0 ||
           SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE,     depth) < 0 ||
           SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, stencil) < 0 ||
           SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER,       1) < 0
        ) {
            //throw SDL_Exception();
        }

        this->_createSDLSurface();

        #ifndef USE_GLEE
            // Init extensions
            GLenum error = glewInit();
            if(error != GLEW_OK) {
                throw std::runtime_error(
                    std::string("GLEW : Init failed : ") +
                    (const char*)glewGetErrorString(error)
                );
            }
        #endif //USE_GLEE

        cudaGLSetGLDevice(cutGetMaxGflopsDeviceId());
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_createSDLSurface() {

        // Create window
        this->_SDLSurface =
            SDL_SetVideoMode(
                this->_windowWidth,
                this->_windowHeight,
                this->_colorBits,
                SDL_OPENGL | SDL_RESIZABLE
            );


        if(this->_SDLSurface == NULL) {
            throw SDL_Exception();
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_render() {
        // Window is not minimized
        bool active = true;

        for(;;) {
            SDL_Event event;

            // Wait for event
            if(SDL_WaitEvent(&event) == 0) {
                throw SDL_Exception();
            }

            // Screen needs redraw
            bool redraw = false;

            // Handle all waiting events
            do {

                // Call proper event handlers
                switch(event.type) {
                    case SDL_ACTIVEEVENT: // Stop redraw when minimized
                        if(event.active.state == SDL_APPACTIVE)
                            active = event.active.gain;
                        break;
                    case SDL_KEYDOWN:
                        this->_onKeyDown(
                            event.key.keysym.sym,
                            event.key.keysym.mod
                        );
                        break;
                    case SDL_KEYUP:
                        this->_onKeyUp(
                            event.key.keysym.sym,
                            event.key.keysym.mod
                        );
                        break;
                    case SDL_MOUSEMOTION:
                        this->_onMouseMove(
                            event.motion.x,
                            event.motion.y,
                            event.motion.xrel,
                            event.motion.yrel,
                            event.motion.state
                        );
                        break;
                    case SDL_MOUSEBUTTONDOWN:
                        this->_onMouseDown(
                            event.button.button,
                            event.button.x,
                            event.button.y
                        );
                        break;
                    case SDL_MOUSEBUTTONUP:
                        this->_onMouseUp(
                            event.button.button,
                            event.button.x,
                            event.button.y
                        );
                        break;
                    case SDL_QUIT:
                        return;// End main loop
                    case SDL_VIDEORESIZE :
                        this->_onWindowResized(
                            event.resize.w,
                            event.resize.h
                        );
                        break;
                    case SDL_VIDEOEXPOSE:
                        redraw = true;
                        break;
                    default :// Do nothing
                        break;
                }
            } while(SDL_PollEvent(&event) == 1);

            // Optionally redraw window
            if(active && redraw) {
                this->_onWindowRedraw();
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_render(unsigned period) {
        // This main loop requires timer support
        if(SDL_InitSubSystem(SDL_INIT_TIMER) < 0) {
            throw SDL_Exception();
        }

        // Create redraw timer
        class RedrawTimer {
            private :
                SDL_TimerID id;
                static Uint32 callback(Uint32 interval, void *) {
                    SDL_Event event;
                    event.type = SDL_VIDEOEXPOSE;
                    if(SDL_PushEvent(&event) < 0) {
                        throw SDL_Exception();
                    }
                    return interval;
                }
            public :
                RedrawTimer(unsigned interval)
                    : id(SDL_AddTimer(interval, callback, NULL)) {
                    if(id == NULL) throw SDL_Exception();
                }
                ~RedrawTimer() {
                    if(id != NULL) SDL_RemoveTimer(id);
                }
        } redrawTimer(period);

        this->_render();
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_onInit() {
        this->_shaderProgram =
            new ShaderProgram("shaders/shader.vs", "shaders/shader.fs");

        this->_positionAttribute =
            this->_shaderProgram->getAttributeLocation("position");

        this->_mvpUniform =
            this->_shaderProgram->getUniformLocation("mvp");

        /*uint3 gridSize;
        gridSize.x = gridSize.y = gridSize.z = 10;

        this->_particleSystem =
            new ParticleSystem(this->_meshWidth * this->_meshHeight, gridSize);
        this->_vbo = this->_particleSystem->getPositionsVBO();
        */
        this->_vbo = this->_simulator->getPositionsVBO();

    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_onWindowResized(int width, int height) {
        glViewport(0, 0, width, height);
        this->_windowWidth = width;
        this->_windowHeight = height;

        this->_createSDLSurface();
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_onWindowRedraw() {


        if (this->_animate) {
            //this->_particleSystem->update(0.05f);
            this->_simulator->update();
        } else {
            //this->_runCuda();

        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);
        glCullFace(GL_BACK);
        glEnable(GL_CULL_FACE);

        // Calculate ModelViewProjection matrix
        glm::mat4 projection =
            glm::perspective(
                45.0f,
                (float) this->_windowWidth / (float) this->_windowHeight,
                1.0f,
                1000.0f
            );

        glm::mat4 mv = glm::rotate(
            glm::rotate(
                glm::translate(
                    glm::mat4(1.0f),
                    glm::vec3(0, 0, this->_translationZ)
                ),
                this->_rotationY,
                glm::vec3(1, 0, 0)
            ),
            this->_rotationX,
            glm::vec3(0, 1, 0)
        );

        glm::mat4 mvp = projection*mv;


        this->_shaderProgram->use();

        // Set matrices
        glUniformMatrix4fv(this->_mvpUniform, 1, GL_FALSE, glm::value_ptr(mvp));
        glEnableVertexAttribArray(this->_positionAttribute);

        // Draw data
        //glBindBuffer(GL_ARRAY_BUFFER, this->_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, this->_vbo);
        glVertexAttribPointer(
            this->_positionAttribute,
            4,
            GL_FLOAT,
            GL_FALSE,
            0,
            (void*) 0
        );

        glDrawArrays(GL_POINTS, 0, this->_meshWidth * this->_meshHeight);

        SDL_GL_SwapBuffers();

        this->_deltaTime += 0.01f;

    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_onKeyDown(
        SDLKey key,
        Uint16 //mod
    ) {
        switch(key) {
            case SDLK_ESCAPE:
                this->_stop();
                break;
            case SDLK_r:
                this->_animate = !this->_animate;
                break;
            case SDLK_c:
                this->_runCuda();
                break;
            default:
                break;
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_onKeyUp(
        SDLKey , //key,
        Uint16   //mod
    ) {
        // not used
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_onMouseMove(
        unsigned , //x,
        unsigned , //y,
        int xrel,
        int yrel,
        Uint8 buttons
    ) {
        if(buttons & SDL_BUTTON_LMASK)
        {
            this->_rotationX += xrel;
            this->_rotationY += yrel;
            this->_redraw();
        }
        if(buttons & SDL_BUTTON_RMASK) {
            this->_translationZ += yrel * 0.1;
            this->_redraw();
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_onMouseDown(
        Uint8,    //button,
        unsigned, //x,
        unsigned  //y
    ) {
        // not used
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_onMouseUp(
        Uint8,    //button,
        unsigned, //x,
        unsigned  //y
    ) {
        // not used
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_redraw() {
        SDL_Event event;
        event.type = SDL_VIDEOEXPOSE;

        if(SDL_PushEvent(&event) < 0) {
            throw SDL_Exception();
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_stop() {
        SDL_Event event;
        event.type = SDL_QUIT;

        if(SDL_PushEvent(&event) < 0) {
            throw SDL_Exception();

        }

        this->_simulator->stop();
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_deleteVBO(
        GLuint* vbo,
        struct cudaGraphicsResource *vbo_res
    ) {
        if (vbo) {
            // unregister this buffer object with CUDA
            //DEPRECATED: cutilSafeCall(cudaGLUnregisterBufferObject(*pbo));
            cudaGraphicsUnregisterResource(vbo_res);

            glBindBuffer(1, *vbo);
            glDeleteBuffers(1, vbo);

            *vbo = 0;
        }
}

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_runCuda() {
        /*struct cudaGraphicsResource *vbo_resource =
            this->_particleSystem->getCudaPositionsVBOResource();
        // map OpenGL buffer object for writing from CUDA
        float4 *dptr;
        // DEPRECATED: cutilSafeCall(cudaGLMapBufferObject((void**)&dptr, vbo));
        cutilSafeCall(cudaGraphicsMapResources(1, &vbo_resource, 0));
        size_t num_bytes;
        cutilSafeCall(
            cudaGraphicsResourceGetMappedPointer(
                (void **)&dptr,
                &num_bytes,
                vbo_resource
            )
        );

        // unmap buffer object
        // DEPRECATED: cutilSafeCall(cudaGLUnmapBufferObject(vbo));
        cutilSafeCall(cudaGraphicsUnmapResources(1, &vbo_resource, 0));
        */
        this->_simulator->bindBuffers();
        float* ptr = this->_simulator->getPositions();

        launch_kernel(
            (float4*) ptr,
                      this->_meshWidth,
                      this->_meshHeight,
                      this->_deltaTime
        );
        this->_simulator->unbindBuffers();
    }

    ////////////////////////////////////////////////////////////////////////////

}