#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "particles_renderer.h"

namespace Particles {

    ////////////////////////////////////////////////////////////////////////////

    Renderer::Renderer(Simulator* simulator) {
        this->_simulator = simulator;

        this->_numParticles = 25000;

        this->_animate = false;
        this->_drawNormals = false;
        this->_run = true;

        this->_rotationX = 0.0f;
        this->_rotationY = 0.0f;
        this->_translationZ = -5.0f;

        this->_windowHeight = 600;
        this->_windowWidth = 800;
        this->_aspectRatio =
            (float) this->_windowWidth / this->_windowHeight;
        this->_colorBits = 24;
        this->_SDLSurface = NULL;

        this->_renderMode = RenderPoints;

        this->_tessLevel = 4.0f;
        this->_tessAlpha = 1.0f;

    }

    ////////////////////////////////////////////////////////////////////////////

    Renderer::~Renderer() {
        delete this->_simulator;
        delete this->_shaderProgram;
        delete this->_marchingProgram;
        delete this->_tesselationProgram;
        delete this->_tesselationTrianglesProgram;
        delete this->_normalsProgram;
        delete this->_cubeProgram;
    }

    ////////////////////////////////////////////////////////////////////////////

    bool Renderer::render() {
        try {
            if(SDL_Init(SDL_INIT_VIDEO) < 0) throw SDL_Exception();

            // Shutdown SDL when program ends
            atexit(SDL_Quit);

            this->_initSDL(24, 0);
            this->_simulator->init(this->_numParticles);
            this->_simulator->setRenderMode(this->_renderMode);

            this->_dynamicColoring =
                this->_simulator->getValue(Settings::DynamicColoring);

            this->_onInit();
            this->_createCube();
            this->_simulator->generateParticles();
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

        this->_shaderProgram = new Shader::Program();
        this->_shaderProgram
            ->add(Shader::Vertex, "shaders/shader.vs")
            ->add(Shader::Fragment, "shaders/shader.fs")
            ->link();

        this->_marchingProgram = new Shader::Program();
        this->_marchingProgram
            ->add(Shader::Vertex, "shaders/marching.vs")
            ->add(Shader::Fragment, "shaders/marching.fs")
            ->link();

        this->_cubeProgram = new Shader::Program();
        this->_cubeProgram
            ->add(Shader::Vertex, "shaders/cube.vs")
            ->add(Shader::Fragment, "shaders/cube.fs")
            ->link();

        this->_tesselationProgram = new Shader::Program();
        this->_tesselationProgram
            ->add(Shader::Vertex, "shaders/marching_tess.vs")
            ->add(Shader::Control, "shaders/marching_tess.cs")
            ->add(Shader::Evaluation, "shaders/marching_tess.es")
            ->add(Shader::Fragment, "shaders/marching_tess.fs")
            ->link();

        this->_tesselationTrianglesProgram = new Shader::Program();
        this->_tesselationTrianglesProgram
            ->add(Shader::Vertex, "shaders/marching_tess.vs")
            ->add(Shader::Geometry, "shaders/marching_tess.gs")
            ->add(Shader::Control, "shaders/marching_tess.cs")
            ->add(Shader::Evaluation, "shaders/marching_tess.es")
            ->add(Shader::Fragment, "shaders/marching_tess_triangles.fs")
            ->link();

        this->_normalsProgram = new Shader::Program();
        this->_normalsProgram
            ->add(Shader::Vertex, "shaders/normals.vs")
            ->add(Shader::Geometry, "shaders/normals.gs")
            ->add(Shader::Fragment, "shaders/normals.fs")
            ->link();
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_onWindowResized(int width, int height) {
        glViewport(0, 0, width, height);
        this->_windowWidth = width;
        this->_windowHeight = height;
        this->_aspectRatio = (float) width / height;

        this->_createSDLSurface();
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_onWindowRedraw() {

        // Calculate ModelViewProjection matrix
        glm::mat4 projection =
        glm::perspective(45.0f, this->_aspectRatio, 0.0001f, 1000.0f);

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

        glm::mat3 mn = glm::mat3(
            glm::rotate(
                glm::rotate(
                    glm::mat4(1.0f),
                            this->_rotationY,
                            glm::vec3(0, 1, 0)
                ),
                this->_rotationX,
                glm::vec3(1, 0, 0)
            )
        );

        glm::mat4 mv2 = glm::rotate(
            glm::rotate(
                glm::mat4(1.0f),
                this->_rotationY,
                glm::vec3(1, 0, 0)
            ),
            this->_rotationX,
            glm::vec3(0, 1, 0)
        );

        glm::vec4 gravity = mv2* glm::vec4(0,-9.8,0,1);
        glm::mat4 mvp = projection*mv;


        GLfloat* mnPtr = glm::value_ptr(mn);
        GLfloat* mvPtr = glm::value_ptr(mv);
        GLfloat* mvpPtr = glm::value_ptr(mvp);

        if (this->_run) {
            this->_simulator->update(
                this->_intergrate,
                gravity.x, gravity.y, gravity.z
            );
        }

        glClearColor(1,1,1,0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);
        this->_cubeProgram->enable();
        this->_cubeProgram->setUniformMatrix4fv("mvp", 1, GL_FALSE, mvpPtr);


        //TODO create VAO for VBOs and attributes
        // Draw data
        glBindBuffer(GL_ARRAY_BUFFER, this->_cubeVBO);
        this->_cubeProgram->setAttribute(
            "position", 4, GL_FLOAT, GL_FALSE, 0, (void*) 0
        );

        glDrawArrays(GL_LINES, 0, 96);

        switch (this->_renderMode) {
            case RenderPoints:
                this->_renderPoints(mvPtr, mvpPtr);
                break;
            case RenderMarching:
                this->_renderMC(mnPtr, mvPtr, mvpPtr);
                break;
            case RenderNormals:
                this->_renderMCNormals(mvpPtr);
                break;
            case RenderTesselation:
                this->_renderMCTess(mnPtr, mvpPtr);
                break;
            case RenderTesselationTriangles:
                this->_renderMCTessTriangles(mnPtr, mvpPtr);
                break;
        }

        if (this->_simulator->getRenderMode() == RenderMarching &&
            this->_drawNormals
        ) {
            this->_renderMCNormals(mvpPtr);
        }

        SDL_GL_SwapBuffers();

    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_onKeyDown(
        SDLKey key,
        Uint16 modifier
    ) {
        float i, f;
        int g, s, n;

        switch(key) {
            case SDLK_ESCAPE:
                this->_stop();
                break;
            case SDLK_r:
                this->_run = !this->_run;
                break;
            case SDLK_e:
                this->_intergrate = !this->_intergrate;
            case SDLK_a:
                this->_animate = !this->_animate;
                this->_simulator->setAnimated(this->_animate);
                break;
            case SDLK_c:
                this->_simulator->generateParticles();
                break;
            case SDLK_d:
                this->_dynamicColoring = !this->_dynamicColoring;
                this->_simulator->setValue(
                    Settings::DynamicColoring,
                    this->_dynamicColoring
                );
                break;
            case SDLK_n:
                this->_drawNormals = !this->_drawNormals;
                break;
            case SDLK_i:
                i = this->_simulator->getValue(Settings::Interpolation);
                this->_simulator->setValue(Settings::Interpolation, !i);
                this->_simulator->setRenderMode(
                    this->_simulator->getRenderMode()
                );
                break;
            case SDLK_g:
                g = this->_simulator->getValue(Settings::ColorGradient) + 1;
                this->_simulator->setValue(Settings::ColorGradient, g % 5);
                break;
            case SDLK_s:
                s = this->_simulator->getValue(Settings::ColorSource) + 1;
                this->_simulator->setValue(Settings::ColorSource, s % 2);
                break;
            case SDLK_F1:
                this->_renderMode = RenderPoints;
                this->_simulator->setRenderMode(this->_renderMode);
                break;
            case SDLK_F2:
                this->_renderMode = RenderMarching;
                this->_simulator->setRenderMode(this->_renderMode);
                break;
            case SDLK_F3:
                this->_renderMode = RenderNormals;
                this->_simulator->setRenderMode(RenderMarching);
                break;
            case SDLK_F4:
                this->_renderMode = RenderTesselation;
                this->_simulator->setRenderMode(RenderMarching);
                break;
            case SDLK_F5:
                this->_renderMode = RenderTesselationTriangles;
                this->_simulator->setRenderMode(RenderMarching);
                break;
            case SDLK_w:
                if (modifier & KMOD_LCTRL) {
                    if (this->_tessLevel < 64) {
                        this->_tessLevel++;
                    }
                } else {
                    if (this->_tessAlpha < 10)
                    this->_tessAlpha += 0.1f;
                }
                break;
            case SDLK_q:
                if (modifier & KMOD_LCTRL) {
                    if (this->_tessLevel > 1) {
                        this->_tessLevel--;
                    }
                } else {
                    if(this->_tessAlpha > 0) {
                        this->_tessAlpha -= 0.1f;
                    }
                }
                break;
            case SDLK_f:
                f = this->_simulator->getValue(Settings::AnimPartForce);
                if (modifier & KMOD_LCTRL && f > 0.0f) {
                    f -= 0.25;
                } else {
                    f += 0.25;
                }
                this->_simulator->setValue(Settings::AnimPartForce, f);
                break;
            case SDLK_p:
                n = this->_simulator->getValue(Settings::AnimPartNum);
                if (modifier & KMOD_LCTRL && n > 1) {
                    n--;
                } else {
                    n++;
                }
                this->_simulator->setValue(Settings::AnimPartNum, n);
                break;
            case SDLK_x:
                i = this->_simulator->getValue(Settings::AnimChangeAxis);
                this->_simulator->setValue(Settings::AnimChangeAxis, !i);
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

    void Renderer::_createCube() {
        glGenBuffers(1, &this->_cubeVBO);
        GridMinMax grid = this->_simulator->getGridMinMax();
        float3 min = grid.min;
        float3 max = grid.max;

        float cube[96] = {
            // FRONT
            min.x, min.y, min.z, 1.0f,
            max.x, min.y, min.z, 1.0f,

            max.x, min.y, min.z, 1.0f,
            max.x, max.y, min.z, 1.0f,

            max.x, max.y, min.z, 1.0f,
            min.x, max.y, min.z, 1.0f,

            min.x, max.y, min.z, 1.0f,
            min.x, min.y, min.z, 1.0f,

            // BACK
            min.x, min.y, max.z, 1.0f,
            max.x, min.y, max.z, 1.0f,

            max.x, min.y, max.z, 1.0f,
            max.x, max.y, max.z, 1.0f,

            max.x, max.y, max.z, 1.0f,
            min.x, max.y, max.z, 1.0f,

            min.x, max.y, max.z, 1.0f,
            min.x, min.y, max.z, 1.0f,

            // LEFT
            min.x, min.y, min.z, 1.0f,
            min.x, min.y, max.z, 1.0f,

            min.x, max.y, min.z, 1.0f,
            min.x, max.y, max.z, 1.0f,

            //RIGHT
            max.x, min.y, min.z, 1.0f,
            max.x, min.y, max.z, 1.0f,

            max.x, max.y, min.z, 1.0f,
            max.x, max.y, max.z, 1.0f
        };

        glBindBuffer(GL_ARRAY_BUFFER, this->_cubeVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(cube), &cube, GL_DYNAMIC_DRAW);
    }
    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_renderPoints(const GLfloat* mv, const GLfloat* mvp) {
        glm::vec2 windowSize =
            glm::vec2(this->_windowWidth, this->_windowHeight);

        this->_shaderProgram->enable();

        this->_shaderProgram
            ->setUniformMatrix4fv("mv", 1, GL_FALSE, mv)
            ->setUniformMatrix4fv("mvp", 1, GL_FALSE, mvp)
            ->setUniform1f("pointRadius", 50.f)
            ->setUniform1f("aspectRatio", this->_aspectRatio)
            ->setUniform2fv("windowSize", 1, glm::value_ptr(windowSize))
            ->setUniform1f(
                "pointScale",
                this->_windowWidth / tanf(45.0f*0.5f*(float)M_PI/180.0f)
            );

        //TODO create VAO for VBOs and attributes
        // Draw data
        glBindBuffer(GL_ARRAY_BUFFER, this->_simulator->getPositionsVBO());
        this->_shaderProgram->setAttribute(
            "position", 4, GL_FLOAT, GL_FALSE, 0, (void*) 0
        );

        glBindBuffer(GL_ARRAY_BUFFER, this->_simulator->getColorsVBO());
        this->_shaderProgram->setAttribute(
            "color", 4, GL_FLOAT, GL_FALSE, 0, (void*) 0
        );

        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
        glDrawArrays(GL_POINTS, 0, this->_numParticles);
        glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_renderMC (
        const GLfloat* mn,
        const GLfloat* mv,
        const GLfloat* mvp
    ) {
        this->_marchingProgram->enable();

        this->_marchingProgram
            ->setUniformMatrix4fv("mv", 1, GL_FALSE, mv)
            ->setUniformMatrix4fv("mvp", 1, GL_FALSE, mvp)
            ->setUniformMatrix3fv("mn", 1, GL_FALSE, mn)
            ->setUniform3f("lightPosition", 0.0f, 2.0f, 0.0f);

        // Draw data
        glBindBuffer(GL_ARRAY_BUFFER, this->_simulator->getPositionsVBO());
        this->_marchingProgram->setAttribute(
            "position", 4, GL_FLOAT, GL_FALSE, 0, (void*) 0
        );

        glBindBuffer(GL_ARRAY_BUFFER, this->_simulator->getNormalsVBO());
        this->_marchingProgram->setAttribute(
            "normal", 4, GL_FLOAT, GL_FALSE, 0, (void*) 0
        );

        glDrawArrays(GL_TRIANGLES, 0, this->_simulator->getNumVertices());
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_renderMCNormals(const GLfloat* mvp) {
        this->_normalsProgram->enable();

        this->_normalsProgram->setUniformMatrix4fv("mvp", 1, GL_FALSE, mvp);

        // Draw data
        glBindBuffer(GL_ARRAY_BUFFER, this->_simulator->getPositionsVBO());
        this->_marchingProgram->setAttribute(
            "position", 4, GL_FLOAT, GL_FALSE, 0, (void*) 0
        );

        glBindBuffer(GL_ARRAY_BUFFER, this->_simulator->getNormalsVBO());
        this->_marchingProgram->setAttribute(
            "normal", 4, GL_FLOAT, GL_FALSE, 0, (void*) 0
        );

        glDrawArrays(GL_TRIANGLES, 0, this->_simulator->getNumVertices());
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_renderMCTess(const GLfloat* mn, const GLfloat* mvp) {
        this->_tesselationProgram->enable();

        this->_tesselationProgram
            ->setUniform1f("tessLevel", this->_tessLevel)
            ->setUniform1f("tessAlpha", this->_tessAlpha)
            ->setUniformMatrix4fv("mvp", 1, GL_FALSE, mvp)
            ->setUniformMatrix3fv("mn", 1, GL_FALSE, mn)
            ->setUniform3f("lightPosition", 0.0f, 5.0f, 5.0f)
            ->setUniform3f("diffuseMaterial", 0.09f, 0.31f, 0.98f)
            ->setUniform3f("ambientMaterial", 0.04f, 0.04f, 0.04f);

        // Draw data
        glBindBuffer(GL_ARRAY_BUFFER, this->_simulator->getPositionsVBO());
        this->_tesselationProgram->setAttribute(
            "position", 4, GL_FLOAT, GL_FALSE, 0, (void*) 0
        );

        glBindBuffer(GL_ARRAY_BUFFER, this->_simulator->getNormalsVBO());
        this->_tesselationProgram->setAttribute(
            "normal", 4, GL_FLOAT, GL_FALSE, 0, (void*) 0
        );

        glPatchParameteri(GL_PATCH_VERTICES, 3);
        glDrawArrays(GL_PATCHES, 0, this->_simulator->getNumVertices());
    }

    ////////////////////////////////////////////////////////////////////////////

    void Renderer::_renderMCTessTriangles(const GLfloat* mn, const GLfloat* mvp) {
        // Tesselation
        this->_tesselationTrianglesProgram->enable();

        this->_tesselationTrianglesProgram
            ->setUniform1f("tessLevel", this->_tessLevel)
            ->setUniform1f("tessAlpha", this->_tessAlpha)
            ->setUniformMatrix4fv("mvp", 1, GL_FALSE, mvp)
            ->setUniformMatrix3fv("mn", 1, GL_FALSE, mn)
            ->setUniform3f("lightPosition", 0.0f, 5.0f, 5.0f)
            ->setUniform3f("diffuseMaterial", 0.09f, 0.31f, 0.98f)
            ->setUniform3f("ambientMaterial", 0.04f, 0.04f, 0.04f);

        // Draw data
        glBindBuffer(GL_ARRAY_BUFFER, this->_simulator->getPositionsVBO());
        this->_tesselationTrianglesProgram->setAttribute(
            "position", 4, GL_FLOAT, GL_FALSE, 0, (void*) 0
        );

        glBindBuffer(GL_ARRAY_BUFFER, this->_simulator->getNormalsVBO());
        this->_tesselationTrianglesProgram->setAttribute(
            "normal", 4, GL_FLOAT, GL_FALSE, 0, (void*) 0
        );

        glPatchParameteri(GL_PATCH_VERTICES, 3);
        glDrawArrays(GL_PATCHES, 0, this->_simulator->getNumVertices());
    }

    ////////////////////////////////////////////////////////////////////////////
}
