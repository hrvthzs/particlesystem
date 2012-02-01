#version 150

in vec4 Position;
uniform mat4 mvp;
out vec3 vPosition;

void main()
{
    vPosition = Position.xyz;


    //gl_Position = mvp * Position;
}
