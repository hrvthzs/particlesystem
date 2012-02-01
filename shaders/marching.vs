#version 130

in vec4 position, normal;

uniform mat4 mvp, mv;
uniform mat3 mn;

//light
uniform vec3 lightPosition;

out vec3 gouraudColor;

out vec3 eyePosition, eyeNormal;

void main()
{
    float mode = 0.0;
    float shininess = 0.0;

    vec3 norm = normalize(normal.xyz);

    gl_Position = mvp*position;

    eyePosition = (mv*position).xyz;
    eyeNormal = normalize(mn*norm);

    vec3 N = eyeNormal;
    vec3 L = normalize(lightPosition-eyePosition);

    //Lighting
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


    gouraudColor = lightColor;
}