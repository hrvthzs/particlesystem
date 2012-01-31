#ifndef __COLORS_CU__
#define __COLORS_CU__

#include "cutil_math.h"
#include "colors.cuh"

namespace Colors {

    __device__ float3 HSV2RGB(float h, float s, float v ) {
        float r = 0;
        float g = 0;
        float b = 0;
        int i;
        float f, p, q, t;

        if( s == 0 ) {
            // achromatic (grey)
            r = g = b = v;
            return make_float3(r,g,b);
        }

        // sector 0 to 5
        h /= 60;
        i = floor( h );
        // factorial part of h
        f = h - i;
        p = v * ( 1.0f - s );
        q = v * ( 1.0f - s * f );
        t = v * ( 1.0f - s * ( 1.0f - f ) );

        switch( i ) {
            case 0:
                r = v;  g = t;  b = p;
                break;
            case 1:
                r = q;  g = v;  b = p;
                break;
            case 2:
                r = p;  g = v;  b = t;
                break;
            case 3:
                r = p;  g = q;  b = v;
                break;
            case 4:
                r = t;  g = p;  b = v;
                break;
            default:        // case 5:
                r = v;  g = p;  b = q;
                break;
        }

        return make_float3(r,g,b);
    }


    __device__ float3 calculateGradient(
        Gradient gradient,
        float value
    ) {
        float3 color = make_float3(0.0f, 0.0f, 0.0f);

        switch(gradient) {
            case White:
                // completely white
                color = make_float3(1.0f, 1.0f, 1.0f);
                break;
            case Blackish:
                // acromatic gradient with V from 0 to 0.5
                {
                    float h = value * 0.5f;
                    color = make_float3(h, h, h);
                }
                break;
            case BlackToCyan:
                color = make_float3(0.0f, value, value);
                break;
            case BlueToWhite:
                // blue to white gradient
                color = make_float3(1.0f - value, 1.0f - 0.5f * value, 1.0f);
                break;
            case HSVBlueToRed:
                // hsv gradient from blue to red (0 to 245 degrees in hue)
                {
                    float h = clamp((1.0f - value) * 245.0f, 0.0f, 245.0f);
                    color = HSV2RGB(h, 0.5f, 1.0f);
                }
                break;
        }
        return color;
    }


    static __device__ float3 calculateColor(
        Gradient gradient,
        Source source,
        float3 velocity,
        float pressure,
        float3 force
    ) {
        float3 color = make_float3(0);

        switch(source) {
            case Velocity:
                // color given by velocity
                {
                    float value =
                        fabs(velocity.x)+
                        fabs(velocity.y)+
                        fabs(velocity.z) / 11000.0f;
                    value = clamp(value, 0.0f, 1.0f);
                    color =  calculateGradient(gradient, value);
                }
                break;
            case Pressure:
                // color given by pressure
                {
                    float value =
                        clamp(
                            (
                                (pressure - cudaFluidParams.restPressure)
                                / 1000.0f
                            ), 0.1f, 1.0f
                        );
                    color =  calculateGradient(gradient, value);
                }
                break;
            case Force:
                // color given by force
                {
                    if(gradient == Direct) {
                        color =
                            clamp(
                                fabs(force) / 80.0f,
                                make_float3(0),
                                make_float3(1)
                            );
                    } else {
                        force /= 2.0f;
                        float value =
                            clamp(
                                (force.x + force.y + force.z) / 3.0f,
                                0.1f,
                                1.0f
                            );
                        color =  calculateGradient(gradient, value);
                    }
                }
                break;
        }
        return color;
    }

}

#endif // __COLORS_CU__