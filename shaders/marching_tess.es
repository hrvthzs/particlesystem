#version 420 core
#extension GL_EXT_gpu_shader4 : enable

// PN patch data
struct PnPatch
{
    float b210;
    float b120;
    float b021;
    float b012;
    float b102;
    float b201;
    float b111;
    float n110;
    float n011;
    float n101;
};

uniform mat4 mvp;
uniform float tessAlpha;

layout(triangles, fractional_odd_spacing, ccw) in;

layout(location = 0) in vec3 iNormal[];
layout(location = 3) in PnPatch iPnPatch[];

layout(location = 0) out vec3 oNormal;
layout(location = 1) out vec3 oPosition;
layout(location = 2) out vec3 oPatchDistance;

#define b300    gl_in[0].gl_Position.xyz
#define b030    gl_in[1].gl_Position.xyz
#define b003    gl_in[2].gl_Position.xyz
#define n200    iNormal[0]
#define n020    iNormal[1]
#define n002    iNormal[2]
#define uvw     gl_TessCoord

void main()
{
    vec3 uvwSquared = uvw * uvw;
    vec3 uvwCubed   = uvwSquared * uvw;

    // extract control points
    vec3 b210 = vec3(iPnPatch[0].b210, iPnPatch[1].b210, iPnPatch[2].b210);
    vec3 b120 = vec3(iPnPatch[0].b120, iPnPatch[1].b120, iPnPatch[2].b120);
    vec3 b021 = vec3(iPnPatch[0].b021, iPnPatch[1].b021, iPnPatch[2].b021);
    vec3 b012 = vec3(iPnPatch[0].b012, iPnPatch[1].b012, iPnPatch[2].b012);
    vec3 b102 = vec3(iPnPatch[0].b102, iPnPatch[1].b102, iPnPatch[2].b102);
    vec3 b201 = vec3(iPnPatch[0].b201, iPnPatch[1].b201, iPnPatch[2].b201);
    vec3 b111 = vec3(iPnPatch[0].b111, iPnPatch[1].b111, iPnPatch[2].b111);

    // extract control normals
    vec3 n110 =
        normalize(vec3(iPnPatch[0].n110, iPnPatch[1].n110, iPnPatch[2].n110));
    vec3 n011 =
        normalize(vec3(iPnPatch[0].n011, iPnPatch[1].n011, iPnPatch[2].n011));
    vec3 n101 =
        normalize(vec3(iPnPatch[0].n101, iPnPatch[1].n101, iPnPatch[2].n101));

    // normal
    vec3 barNormal =
        gl_TessCoord[2] * iNormal[0] +
        gl_TessCoord[0] * iNormal[1] +
        gl_TessCoord[1] * iNormal[2];

    vec3 pnNormal =
        n200 * uvwSquared[2] +
        n020 * uvwSquared[0] +
        n002 * uvwSquared[1] +
        n110 * uvw[2] * uvw[0] +
        n011 * uvw[0] * uvw[1] +
        n101 * uvw[2] * uvw[1];

    // should we normalize ?
    oNormal = tessAlpha * pnNormal + (1.0 - tessAlpha) * barNormal;

    // compute interpolated pos
    vec3 barPos =
        gl_TessCoord[2] * b300 +
        gl_TessCoord[0] * b030 +
        gl_TessCoord[1] * b003;

    // save some computations
    uvwSquared *= 3.0;

    // compute PN position
    vec3 pnPos  =
        b300 * uvwCubed[2] +
        b030 * uvwCubed[0] +
        b003 * uvwCubed[1] +
        b210 * uvwSquared[2] * uvw[0] +
        b120 * uvwSquared[0] * uvw[2] +
        b201 * uvwSquared[2] * uvw[1] +
        b021 * uvwSquared[0] * uvw[1] +
        b102 * uvwSquared[1] * uvw[2] +
        b012 * uvwSquared[1] * uvw[0] +
        b111 * 6.0 * uvw[0] * uvw[1] * uvw[2];

    // final position and normal
    vec3 finalPos = (1.0 - tessAlpha) * barPos + tessAlpha * pnPos;

    oPosition = finalPos;
    oPatchDistance = gl_TessCoord;

    gl_Position   = mvp * vec4(finalPos,1.0);

}