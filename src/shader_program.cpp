#include "shader_program.h"

namespace  Shader {

    ////////////////////////////////////////////////////////////////////////////

    Program::Program () {
        // Create program object
        this->_program = glCreateProgram();

        if(this->_program == 0) {
            throw Exception("glCreateProgram failed");
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    Program::~Program () {
        map<Types,GLuint>::iterator it;

        for (it=this->_shaders.begin(); it!=this->_shaders.end(); it++) {
            GLuint shader = it->second;
            glDetachShader(this->_program, shader);
        }

        glDeleteProgram(this->_program);
    }

    ////////////////////////////////////////////////////////////////////////////

    void Program::enable () {
        glUseProgram(this->_program);
    }

    ////////////////////////////////////////////////////////////////////////////

    void Program::disable () {
        glUseProgram(0);
    }

    ////////////////////////////////////////////////////////////////////////////

    Program* Program::setAttribute (
        string attribute,
        GLint size,
        GLenum type,
        GLboolean normalized,
        GLsizei stride,
        const GLvoid * pointer
    ) {
        GLuint index = this->getAttributeLocation(attribute);
        glVertexAttribPointer(index, size, type, normalized, stride, pointer);
        glEnableVertexAttribArray(index);
        return this;

    }

    ////////////////////////////////////////////////////////////////////////////

    Program* Program::setUniform1f(string uniform, GLfloat v0) {
        GLuint location = this->getUniformLocation(uniform);
        glUniform1f(location, v0);
        return this;
    }

    ////////////////////////////////////////////////////////////////////////////

    Program* Program::setUniform3f(
        string uniform,
        GLfloat v0,
        GLfloat v1,
        GLfloat v2
    ) {
        GLuint location = this->getUniformLocation(uniform);
        glUniform3f(location, v0, v1, v2);
        return this;
    }

    ////////////////////////////////////////////////////////////////////////////

    Program* Program::setUniform2fv(
        string uniform,
        GLsizei count,
        const GLfloat* value
    ) {
        GLuint location = this->getUniformLocation(uniform);
        glUniform2fv(location, count, value);
        return this;
    }

    ////////////////////////////////////////////////////////////////////////////

    Program* Program::setUniform3fv(
        string uniform,
        GLsizei count,
        const GLfloat* value
    ) {
        GLuint location = this->getUniformLocation(uniform);
        glUniform3fv(location, count, value);
        return this;
    }

    ////////////////////////////////////////////////////////////////////////////

    Program* Program::setUniform4fv(
        string uniform,
        GLsizei count,
        const GLfloat* value
    ) {
        GLuint location = this->getUniformLocation(uniform);
        glUniform4fv(location, count, value);
        return this;
    }

    ////////////////////////////////////////////////////////////////////////////

    Program* Program::setUniformMatrix3fv(
        string uniform,
        GLsizei count,
        GLboolean transpose,
        const GLfloat* value
    ) {
        GLuint location = this->getUniformLocation(uniform);
        glUniformMatrix3fv(location, count, transpose, value);
        return this;
    }

    ////////////////////////////////////////////////////////////////////////////

    Program* Program::setUniformMatrix4fv(
        string uniform,
        GLsizei count,
        GLboolean transpose,
        const GLfloat* value
    ) {
        GLuint location = this->getUniformLocation(uniform);
        glUniformMatrix4fv(location, count, transpose, value);
        return this;
    }

    ////////////////////////////////////////////////////////////////////////////

    GLuint Program::getAttributeLocation (string attribute) {
        return glGetAttribLocation(this->_program, attribute.c_str());
    }

    ////////////////////////////////////////////////////////////////////////////

    GLuint Program::getUniformLocation (string uniform) {
        return glGetUniformLocation(this->_program, uniform.c_str());
    }

    ////////////////////////////////////////////////////////////////////////////

    GLuint Program::getReference() {
        return this->_program;
    }

    ////////////////////////////////////////////////////////////////////////////

    Program* Program::add(Types type, const char *source) {

        GLuint glType;

        switch (type) {
            case Vertex:
                glType = GL_VERTEX_SHADER;
                break;
            case Fragment:
                glType = GL_FRAGMENT_SHADER;
                break;
            case Geometry:
                glType = GL_GEOMETRY_SHADER;
                break;
            case Control:
                glType = GL_TESS_CONTROL_SHADER;
                break;
            case Evaluation:
                glType = GL_TESS_EVALUATION_SHADER;
                break;
            default:
                cout << "Invalid shader type." << endl;
                return this;
        }

        GLuint shader = this->_compile(glType, this->_load(source).c_str());

        this->_shaders[type] = shader;

        return this;
    }

    ////////////////////////////////////////////////////////////////////////////

    void Program::link() {

        map<Types,GLuint>::iterator it;

        for (it=this->_shaders.begin(); it!=this->_shaders.end(); it++) {
            GLuint shader = it->second;
            glAttachShader(this->_program, shader);
            glDeleteShader(shader);
        }

        // Link program and check for errors
        glLinkProgram(this->_program);

        cout << this->_getProgramInfoLog(this->_program);

        int linkStatus;
        glGetProgramiv(this->_program, GL_LINK_STATUS, &linkStatus);

        if(linkStatus == GL_FALSE) {
            throw runtime_error("shader linking failed");
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    string Program::_load (const char *filename) {

        ifstream stream(filename);

        if(stream.fail()) {
            throw runtime_error(string("Can't open \'") + filename + "\'");
        }

        return string(
            istream_iterator<char>(stream >> noskipws),
            istream_iterator<char>()
        );

    }

    ////////////////////////////////////////////////////////////////////////////

    GLuint Program::_compile (const GLenum type, const char * source) {
        GLuint shader = glCreateShader(type);

        if(shader == 0) throw Exception("glCreateShader failed");

        glShaderSource(shader, 1, &source, NULL);
        glCompileShader(shader);

        cout << this->_getShaderInfoLog(shader);

        int compileStatus;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compileStatus);

        if(compileStatus == GL_FALSE) {
            throw runtime_error("shader compilation failed");

        }

        return shader;
    }

    ////////////////////////////////////////////////////////////////////////////

    string Program::_getShaderInfoLog (const GLuint shader) {
        int length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        string log(length, ' ');
        glGetShaderInfoLog(shader, length, NULL, &log[0]);
        return log;
    }

    ////////////////////////////////////////////////////////////////////////////

    string Program::_getProgramInfoLog (const GLuint program) {
        int length;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        string log(length, ' ');
        glGetProgramInfoLog(program, length, NULL, &log[0]);
        return log;
    }

    ////////////////////////////////////////////////////////////////////////////

};