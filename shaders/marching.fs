#version 130

//light
uniform vec3 lightPosition;

//material

in vec3 gouraudColor;

in vec3 eyePosition, eyeNormal;

out vec4 gl_FragColor;

void main()
{


    float mode = 0.0;
    float shininess = 0.0;

    vec3 N = normalize(eyeNormal);
    vec3 L = normalize(lightPosition-eyePosition);

    //Blin-Phong model
    vec3 lightColor = vec3(0.5, 0.5, 0.5);
    float diffuse = dot(L, N);
    if(diffuse >= 0.0)
    {
        lightColor += vec3(diffuse, diffuse, diffuse);

        if(mode > 0)
        {
            //Phong model
            vec3 V = normalize(-eyePosition);
            vec3 R = reflect(-L, N);
            float specular = pow(dot(R, V), shininess);

            if(specular >= 0.0)
            {
                lightColor += vec3(specular);
            }
        }
        else
        {
            //Blinn-Phong model
            vec3 V = normalize(-eyePosition);
            vec3 H = (L + V)*0.5;
            float specular = pow(dot(N, H), shininess);

            if(specular >= 0.0)
            {
                lightColor += vec3(specular);
            }
        }
    }
    vec4 c = vec4(0.09, 0.31, 0.98, 1.0);
    if(abs(mode) < 0.5)
        gl_FragColor = vec4(gouraudColor, 1.0);
    else
        gl_FragColor = vec4(lightColor, 1.0);

    gl_FragColor = c*vec4(gouraudColor, 1.0);
}