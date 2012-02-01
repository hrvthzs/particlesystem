#version 150
#extension GL_EXT_geometry_shader4 : enable

uniform mat3 mn;
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
in vec3 tePosition[3];
in vec3 tePatchDistance[3];
out vec3 gFacetNormal;
out vec3 gPatchDistance;
out vec3 gTriDistance;

void main()
{
    vec3 A = tePosition[2] - tePosition[0];
    vec3 B = tePosition[1] - tePosition[0];
    gFacetNormal = mn * normalize(cross(A, B));

    gPatchDistance = tePatchDistance[0];
    gTriDistance = vec3(1, 0, 0);
    gl_Position = gl_in[0].gl_Position; EmitVertex();

    gPatchDistance = tePatchDistance[1];
    gTriDistance = vec3(0, 1, 0);
    gl_Position = gl_in[1].gl_Position; EmitVertex();

    gPatchDistance = tePatchDistance[2];
    gTriDistance = vec3(0, 0, 1);
    gl_Position = gl_in[2].gl_Position; EmitVertex();

    EndPrimitive();
}

/*layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

//model-view-projection matice
uniform mat4 mvp;

//barva vertexu
out vec4 color;

void main()
{
    gl_Position = vec4(0,0,0,1) * mvp;
    color = vec4(1,0,0,1);
    EmitVertex();

    gl_Position = vec4(1,0,0,1) * mvp;
    color = vec4(0,1,0,1);
    EmitVertex();

    gl_Position = vec4(0,1,0,1) * mvp;
    color = vec4(0,0,1,1);
    EmitVertex();

    gl_Position = vec4(1,1,0,1) * mvp;
    color = vec4(1,1,0,1);
    EmitVertex();
}*/