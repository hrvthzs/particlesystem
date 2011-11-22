#version 150

in vec4 col;

out vec4 fragColor;

void main()
{
    vec4 c = vec4(0.09, 0.31, 0.98, 1.0);

    fragColor = abs(col);
}
