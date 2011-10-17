#version 150


in vec4 position;
uniform mat4 mvp;


void main()
{
    gl_Position = mvp * vec4(position);

}