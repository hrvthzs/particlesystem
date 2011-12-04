/**
 * shader_program.h
 *
 * @author Zsolt Horváth
 * @date   2. 4. 2011
 *
 */

#ifndef _SHADER_PROGRAM_H_
#define _SHADER_PROGRAM_H_

#include <string>
#include <fstream>
#include <iterator>
#include <exception>
#include <stdexcept>
#include <iostream>

#include <SDL/SDL.h>

#include <GL/glew.h>

using namespace std;

/**
 * Shader exceptions
 */
struct ShaderException : public runtime_error
{
    ShaderException(const GLenum error = glGetError()) throw()
        : std::runtime_error(std::string("OpenGL : ") + (const char*) gluErrorString(error)) {}
    ShaderException(const char * text, const GLenum error = glGetError()) throw()
        : std::runtime_error(std::string("OpenGL : ") + text + " - " + (const char*) gluErrorString(error)) {}
};

/**
 * OpenGL shader program
 */
class ShaderProgram {

    public:
        /**
         * Construct
         *
         * @param string vertexSource - path to vertex shader
         * @param string fragmentSource - path to fragment shader
         */
        ShaderProgram (string vertexSource, string fragmentSource);

        /**
         * Desctuctor
         */
        ~ShaderProgram ();

        /**
         * Use shader program
         */
        void use();

        /**
         * Set shader attribute
         *
         * @see glVertexAttribPointer
         */
        ShaderProgram* setAttribute (
            string attribute,
            GLint size,
            GLenum type,
            GLboolean normalized,
            GLsizei stride,
            const GLvoid* pointer);

        /**
         * Set uniform values
         */
        ShaderProgram* setUniform1f(string uniform, GLfloat v0);
        ShaderProgram* setUniform3f(string uniform, GLfloat v0, GLfloat v1, GLfloat v2);
        ShaderProgram* setUniform2fv(string uniform, GLsizei count, const GLfloat* value);
        ShaderProgram* setUniform3fv(string uniform, GLsizei count, const GLfloat* value);
        ShaderProgram* setUniform4fv(string uniform, GLsizei count, const GLfloat* value);

        ShaderProgram* setUniformMatrix3fv(
            string uniform,
            GLsizei count,
            GLboolean transpose,
            const GLfloat* value
        );

        ShaderProgram* setUniformMatrix4fv(
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


        GLuint getReference();

    protected:
        /**
         * Add shader
         *
         * @param const GLenum type - type of shader
         * @param const char *soruce - path to shader
         */
        void add(const GLenum type, const char *source);

        /**
         * Load shader form file
         *
         * @param const char *filename - path to shader
         * @return content of loaded file
         */
        string load(const char *filename);

        /**
         * Get info log of shader
         *
         * @param const GLuint shader - shader pointer
         * @return info string
         */
        string getShaderInfoLog(const GLuint shader);

        /**
         * Get info log of program
         *
         * @param const GLuint program - program pointer
         * @return info string
         */
        string getProgramInfoLog(const GLuint program);

        /**
         * Compile shader
         *
         * @param const GLenum type - type of shader
         * @param const char * source - shader source code
         * @return pointer to shader
         */
        GLuint compile (const GLenum type, const char * source);

        /**
         * Link shader into program
         *
         * @param size_t count - count of shaders to compile
         * @param ... - shader pointers
         * @return pointer to program
         */
        GLuint link(size_t count, ...);

        /**
         * Vertex shader pointer
         */
        GLuint vertexShader;

        /**
         * Fragment shader pointer
         */
        GLuint fragmentShader;

        /**
         * Program pointer
         */
        GLuint program;

};

#endif /* _SHADER_PROGRAM_H_ */