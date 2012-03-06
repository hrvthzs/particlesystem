#version 420 core
#extension GL_EXT_gpu_shader4 : enable

#define ID gl_InvocationID

// PN patch data
struct PnPatch {
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

layout(vertices=3) out;

layout(location = 0) in vec3 iPosition[];
layout(location = 1) in vec3 iNormal[];

uniform float tessLevel;

layout(location = 0) out vec3 oNormal[3];
layout(location = 3) out PnPatch oPnPatch[3];
layout(location = 6) out vec3 oPosition[];

float wij(int i, int j) {
    return dot(iPosition[j] - iPosition[i], iNormal[i]);
}

float vij(int i, int j) {
    vec3 Pj_minus_Pi = iPosition[j] - iPosition[i];
    vec3 Ni_plus_Nj  = iNormal[i] + iNormal[j];
    return 2.0 * dot(Pj_minus_Pi, Ni_plus_Nj) / dot(Pj_minus_Pi, Pj_minus_Pi);
}

void main() {
    // just to shut up warnings
    oPnPatch[ID].b111 = 0.0;
    oPnPatch[ID].n110 = 0.0;
    oPnPatch[ID].n011 = 0.0;
    oPnPatch[ID].n101 = 0.0;

    // get data
    gl_out[ID].gl_Position.xyz = iPosition[ID];
    oNormal[ID] = iNormal[ID];

    // set base
    float P0 = iPosition[0][ID];
    float P1 = iPosition[1][ID];
    float P2 = iPosition[2][ID];
    float N0 = iNormal[0][ID];
    float N1 = iNormal[1][ID];
    float N2 = iNormal[2][ID];

    // compute control points (will be evaluated three times ...)
    oPnPatch[ID].b210 = (2.0 * P0 + P1 - wij(0, 1) * N0) / 3.0;
    oPnPatch[ID].b120 = (2.0 * P1 + P0 - wij(1, 0) * N1) / 3.0;
    oPnPatch[ID].b021 = (2.0 * P1 + P2 - wij(1, 2) * N1) / 3.0;
    oPnPatch[ID].b012 = (2.0 * P2 + P1 - wij(2, 1) * N2) / 3.0;
    oPnPatch[ID].b102 = (2.0 * P2 + P0 - wij(2, 0) * N2) / 3.0;
    oPnPatch[ID].b201 = (2.0 * P0 + P2 - wij(0, 2) * N0) / 3.0;

    float E =
        (
            oPnPatch[ID].b210 +
            oPnPatch[ID].b120 +
            oPnPatch[ID].b021 +
            oPnPatch[ID].b012 +
            oPnPatch[ID].b102 +
            oPnPatch[ID].b201
        ) / 6.0;

    float V = (P0 + P1 + P2) / 3.0;

    oPnPatch[ID].b111 = E + (E - V) * 0.5;
    oPnPatch[ID].n110 = N0 + N1 - vij(0, 1) * (P1 - P0);
    oPnPatch[ID].n011 = N1 + N2 - vij(1, 2) * (P2 - P1);
    oPnPatch[ID].n101 = N2 + N0 - vij(2, 0) * (P0 - P2);

    // set tess levels
    gl_TessLevelOuter[ID] = tessLevel;
    gl_TessLevelInner[0] = tessLevel;
}
