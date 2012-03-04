#version 330

in vec4 position;
in vec4 normal;
uniform mat4 mvp;

out Data
{
    vec4 position;
    vec4 normal;
    vec4 color;
    mat4 mvp;
} vdata;

void main()
{
    vdata.mvp = mvp;
    vdata.position = position;
    vdata.normal = normal;
}