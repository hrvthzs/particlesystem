#version 130
in vec4 position;
uniform mat4 mvp;
uniform mat4 mv;
uniform float pointRadius;
uniform float pointScale;
out vec2 vTexCoord;


void main()
{
    vec3 posEye = vec3(mv * position);
    float dist = length(posEye);
    //vTexCoord = gl_MultiTexCoord0.xy;
    gl_PointSize = pointScale / (pointRadius * dist);
    gl_Position = mvp * vec4(position);

}