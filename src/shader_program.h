/**
 * shader_program.h
 *
 * @author Zsolt Horváth
 * @date   2. 4. 2011
 *
 */

#ifndef __SHADER_PROGRAM_H__
#define __SHADER_PROGRAM_H__

#include <string>
#include <fstream>
#include <iterator>
#include <exception>
#include <stdexcept>
#include <iostream>
#include <map>

#include <SDL/SDL.h>
#include <GL/glew.h>

#include "shader.h"

using namespace std;

namespace Shader {
    /**
    * Shader exceptions
    */
    struct Exception : public runtime_error {
        Exception(const GLenum error = glGetError()) throw()
            : std::runtime_error(std::string("OpenGL : ") +
              (const char*) gluErrorString(error)) {}

        Exception(const char * text, const GLenum error = glGetError()) throw()
            : std::runtime_error(std::string("OpenGL : ") +
              text + " - " + (const char*) gluErrorString(error)) {}
    };

    /**
    * OpenGL shader program
    */
    class Program {

        public:
            /**
            * Construct
            *
            * @param string vertexSource - path to vertex shader
            * @param string fragmentSource - path to fragment shader
            */
            Program ();

            /**
            * Desctuctor
            */
            ~Program ();

            /**
             * Add shader
             *
             * @param const GLenum type - type of shader
             * @param const char *soruce - path to shader
             */
            Program* add(Types type, const char *source);

            /**
             * Link shaders into program
             */
            void link();

            /**
            * Enable shader program
            */
            void enable();

            /**
             * Disable shader program
             */
            void disable();

            /**
            * Set shader attribute
            *
            * @see glVertexAttribPointer
            */
            Program* setAttribute (
                string attribute,
                GLint size,
                GLenum type,
                GLboolean normalized,
                GLsizei stride,
                const GLvoid* pointer);

            /**
            * Set uniform values
            */
            Program* setUniform1f(string uniform, GLfloat v0);
            Program* setUniform3f(string uniform, GLfloat v0, GLfloat v1, GLfloat v2);
            Program* setUniform2fv(string uniform, GLsizei count, const GLfloat* value);
            Program* setUniform3fv(string uniform, GLsizei count, const GLfloat* value);
            Program* setUniform4fv(string uniform, GLsizei count, const GLfloat* value);

            Program* setUniformMatrix3fv(
                string uniform,
                GLsizei count,
                GLboolean transpose,
                const GLfloat* value
            );

            Program* setUniformMatrix4fv(
                string uniform,
                GLsizei count,
                GLboolean transpose,
                const GLfloat* value
            );

            /**
            * Get attribute location
            *
            * @param string attribute - attribute name
            * @return pointer to attribute
            */
            GLuint getAttributeLocation (string attribute);

            /**
            * Get uniform location
            *
            * @param string uniform - uniform name
            * @return pointer to uniform
            */
            GLuint getUniformLocation (string uniform);

            /**
             * Get reference to shader program (shader program ID)
             */
            GLuint getReference();

        protected:


            /**
            * Load shader form file
            *
            * @param const char *filename - path to shader
            * @return content of loaded file
            */
            string _load(const char *filename);

            /**
            * Get info log of shader
            *
            * @param const GLuint shader - shader pointer
            * @return info string
            */
            string _getShaderInfoLog(const GLuint shader);

            /**
            * Get info log of program
            *
            * @param const GLuint program - program pointer
            * @return info string
            */
            string _getProgramInfoLog(const GLuint program);

            /**
            * Compile shader
            *
            * @param const GLenum type - type of shader
            * @param const char * source - shader source code
            * @return pointer to shader
            */
            GLuint _compile (const GLenum type, const char * source);

            /**
            * Program pointer
            */
            GLuint _program;

            /**
             * Map of shaders
             */
            map<Types, GLuint> _shaders;

    };

};

#endif // __SHADER_PROGRAM_H__
