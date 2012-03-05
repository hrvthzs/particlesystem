#version 420 core
#extension GL_EXT_gpu_shader4 : enable

layout(location = 0) in vec3 normal;

uniform vec3 lightPosition;
uniform vec3 diffuseMaterial;
uniform vec3 ambientMaterial;

layout(location = 0) out vec4 oColour;

void main() {

    vec3 N = normalize(normal);
    vec3 L = lightPosition;
    float df = abs(dot(N, L));
    vec3 color = ambientMaterial + df * diffuseMaterial;

    oColour = vec4(color, 1.0);
}