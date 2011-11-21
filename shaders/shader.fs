#version 150

in vec4 color;
in vec2 vTextCoord;
out vec4 fragColor;

void main()
{
    vec4 c = vec4(0.09, 0.31, 0.98, 1.0);

    const vec3 lightDir = vec3(0.577, 0.577, 0.577);

    // calculate normal from texture coordinates
    //vec3 N;
    //N.xy = vTextCoord.xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    //float mag = dot(N.xy, N.xy);
    //if (mag > 1.0) discard;   // kill pixels outside circle
    //N.z = sqrt(1.0-mag);

    // calculate lighting
    //float diffuse = max(1.0, dot(lightDir, N));


    //fragColor = c * diffuse;
    fragColor = c;
}
