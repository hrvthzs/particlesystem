#version 130
in vec4 position;
in vec4 color;
uniform mat4 mvp;
uniform mat4 mv;
uniform float pointRadius;
uniform float pointScale;
out vec4 col;


void main()
{
    vec3 posEye = vec3(mv * position);
    float dist = length(posEye);

    gl_PointSize = pointScale / (pointRadius * dist);
    gl_Position = mvp * vec4(position);
    col = color;

}