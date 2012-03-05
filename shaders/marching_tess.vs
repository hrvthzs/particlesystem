#version 420 core
#extension GL_EXT_gpu_shader4 : enable

layout(location = 0)   in vec4 position;
layout(location = 1)   in vec4 normal;

layout(location = 0)   out vec3 oPosition;
layout(location = 1)   out vec3 oNormal;

void main() {
    oPosition.xyz = position.xyz;
    oNormal         = normalize(normal.xyz);
}