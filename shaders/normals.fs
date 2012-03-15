#version 330

in Data {
    vec4 color;
} gdata;

out vec4 outputColor;

void main() {
    outputColor = gdata.color;
}