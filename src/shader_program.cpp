#include "shader_program.h"

using namespace std;

ShaderProgram::ShaderProgram (string vertexSource, string fragmentSource) {
    this->add(GL_VERTEX_SHADER, vertexSource.c_str());
    this->add(GL_FRAGMENT_SHADER, fragmentSource.c_str());
    this->program = this->link(2, this->vertexShader, this->fragmentShader);
}

ShaderProgram::~ShaderProgram () {
    glDetachShader(this->program, this->vertexShader);
    glDetachShader(this->program, this->fragmentShader);
    glDeleteProgram(this->program);
}

void ShaderProgram::use () {
    glUseProgram(this->program);
}

void ShaderProgram::addAttribute (
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

}

GLuint ShaderProgram::getAttributeLocation (string attribute) {
    return glGetAttribLocation(this->program, attribute.c_str());
}

GLuint ShaderProgram::getUniformLocation (string uniform) {
    return glGetUniformLocation(this->program, uniform.c_str());
}

void ShaderProgram::add(const GLenum type, const char *source) {
    GLuint shader = this->compile(type, this->load(source).c_str());

    switch (type) {
        case GL_VERTEX_SHADER:
            this->vertexShader = shader;
        case GL_FRAGMENT_SHADER:
            this->fragmentShader = shader;
    }
}

string ShaderProgram::load (const char *filename) {

    ifstream stream(filename);

    if(stream.fail()) {
        throw runtime_error(string("Can't open \'") + filename + "\'");
    }

    return string(
        istream_iterator<char>(stream >> noskipws),
        istream_iterator<char>()
    );

}

GLuint ShaderProgram::compile (const GLenum type, const char * source) {
    GLuint shader = glCreateShader(type);

    if(shader == 0) throw ShaderException("glCreateShader failed");

    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    cout << this->getShaderInfoLog(shader);

    int compileStatus;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compileStatus);

    if(compileStatus == GL_FALSE) {
        throw runtime_error("shader compilation failed");

    }

    return shader;
}

/**
 * Info log contains errors and warnings from shader compilation
 */
string ShaderProgram::getShaderInfoLog (const GLuint shader) {
    int length;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
    string log(length, ' ');
    glGetShaderInfoLog(shader, length, NULL, &log[0]);
    return log;
}

/**
 * Info log contains errors and warnings from shader linking
 */
string ShaderProgram::getProgramInfoLog (const GLuint program) {
    int length;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
    string log(length, ' ');
    glGetProgramInfoLog(program, length, NULL, &log[0]);
    return log;
}

GLuint ShaderProgram::link (size_t count, ...) {
    // Create program object
    GLuint program = glCreateProgram();
    if(program == 0) throw ShaderException("glCreateProgram failed");

    // Attach arguments
    va_list args;
    va_start(args, count);

    for(size_t i = 0; i < count; ++i) {
        GLuint shader = va_arg(args, GLuint);
        glAttachShader(program, shader);
        glDeleteShader(shader);
    }
    
    va_end(args);

    // Link program and check for errors
    glLinkProgram(program);

    cout << getProgramInfoLog(program);

    int linkStatus;
    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);

    if(linkStatus == GL_FALSE) {
        throw runtime_error("shader linking failed");
    }

    return program;
}
