#version 420 core
#extension GL_EXT_geometry_shader4 : enable


layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

layout(location = 0) in vec3 oNormal[3];
layout(location = 1) in vec3 iPosition[3];
layout(location = 2) in vec3 iPatchDistance[3];

uniform mat3 mn;

layout(location = 0) out vec3 oFacetNormal;
layout(location = 1) out vec3 oPatchDistance;
layout(location = 2) out vec3 oTriDistance;

void main() {
    vec3 A = iPosition[2] - iPosition[0];
    vec3 B = iPosition[1] - iPosition[0];
    oFacetNormal = mn * normalize(cross(A, B));

    oPatchDistance = iPatchDistance[0];
    oTriDistance = vec3(1, 0, 0);
    gl_Position = gl_in[0].gl_Position; EmitVertex();

    oPatchDistance = iPatchDistance[1];
    oTriDistance = vec3(0, 1, 0);
    gl_Position = gl_in[1].gl_Position; EmitVertex();

    oPatchDistance = iPatchDistance[2];
    oTriDistance = vec3(0, 0, 1);
    gl_Position = gl_in[2].gl_Position; EmitVertex();

    EndPrimitive();
}