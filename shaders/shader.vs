#version 130

in vec4 position;
in vec4 color;
uniform mat4 mvp;
uniform mat4 mv;
uniform float pointRadius;
uniform float pointScale;
out float correction;

out vec4 col;
out vec4 pos;



void main()
{
    vec3 posEye = vec3(mv * position);
    float dist = length(posEye);
    correction = dist * pointRadius;

    gl_PointSize = pointScale / correction;
    pos = mvp * vec4(position);
    col = color;
    col = vec4(1.0, 0.0, 0.0, 1.0);
    gl_Position = pos;

}