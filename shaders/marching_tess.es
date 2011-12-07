#version 400 core
#extension GL_EXT_gpu_shader4 : enable

layout(triangles, equal_spacing, ccw) in;
in vec3 tcPosition[];
out vec3 tePosition;
out vec3 tePatchDistance;
uniform mat4 mvp;

void main()
{
    vec3 p0 = gl_TessCoord.x * tcPosition[0];
    vec3 p1 = gl_TessCoord.y * tcPosition[1];
    vec3 p2 = gl_TessCoord.z * tcPosition[2];

    tePatchDistance = gl_TessCoord;

    vec3 normal = normalize(cross(tcPosition[1] - tcPosition[0],
                                  tcPosition[2] - tcPosition[0]));


    vec3 a = tcPosition[0] - tcPosition[1];
    vec3 b = tcPosition[1] - tcPosition[2];
    vec3 c = tcPosition[2] - tcPosition[0];

    float al = length(a);
    float bl = length(b);
    float cl = length(c);

    vec3 distVec = gl_TessCoord.xyz - vec3(0.5, 0.5, 0.5);
    tePosition = (p0 + p1 + p2);
    tePosition += normal * (al + bl + cl) / 6.0 * (1.0 -smoothstep(0.0, 0.5, length(distVec)));

    gl_Position = mvp * vec4(tePosition, 1);


    /*float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;
    float w = gl_TessCoord.z;
    tePatchDistance = gl_TessCoord;

    vec3 a = mix(tcPosition[0], tcPosition[1], u);
    vec3 b = mix(tcPosition[1], tcPosition[2], v);
    vec3 c = mix(tcPosition[2], tcPosition[0], w);
    tePosition = a + b + c / 3.0;
    gl_Position = mvp * vec4(tePosition, 1);*/


}