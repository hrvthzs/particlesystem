#version 150

in vec4 col;
in vec4 pos;
in float correction;

uniform vec2 windowSize;
uniform float aspectRatio;
out vec4 fragColor;

void main()
{

    vec4 fragPos = gl_FragCoord;

    fragPos.x /= windowSize.x;
    fragPos.y /= windowSize.y;

    vec4 position = pos / pos.w;

    position.xy *= vec2(0.5, 0.5);
    position.xy += vec2(0.5, 0.5);

    vec3 N;
    N.xy = position.xy - fragPos.xy;
    N.z = 0.0;
    N *= correction;
    N.y /= aspectRatio;
    float mag = dot(N.xy, N.xy);
    if (mag > 1.0) discard; // kill pixels outside circle

    N.z = sqrt(1.0 - mag);
    const vec3 lightDir = vec3(0.577, 0.577, 0.577);
    // calculate lighting
    float diffuse = max(0.0, dot(lightDir, N));

    fragColor = col * diffuse;

}

