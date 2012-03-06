#version 420 core
#extension GL_EXT_gpu_shader4 : enable

layout(location = 0) in vec3 iFacetNormal;
layout(location = 1) in vec3 iPatchDistance;
layout(location = 2) in vec3 iTriDistance;

uniform vec3 lightPosition;
uniform vec3 diffuseMaterial;
uniform vec3 ambientMaterial;

layout(location=0) out vec4 oColor;

float amplify(float d, float scale, float offset) {
    d = scale * d + offset;
    d = clamp(d, 0, 1);
    d = 1 - exp2(-2.0 * d * d);
    return d;
}

void main() {
    vec3 N = normalize(iFacetNormal);
    vec3 L = normalize(vec3(1,1,1));
    float df = clamp(dot(N, L), 0, 1);
    vec3 color = ambientMaterial + df * diffuseMaterial;

    float d1 = min(min(iTriDistance.x, iTriDistance.y), iTriDistance.z);
    float d2 = min(min(iPatchDistance.x, iPatchDistance.y), iPatchDistance.z);
    color = amplify(d1, 40, -0.5) * amplify(d2, 60, -0.5) * color;

    oColor = vec4(color, 1.0);
}
