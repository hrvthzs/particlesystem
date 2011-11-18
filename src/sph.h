#ifndef __SPH_H__
#define __SPH_H__

#include "particles.h"

namespace SPH {

    enum eBuffers {
        Color,
        Density,
        Force,
        Position,
        Pressure,
        Velocity,
        SortedColor,
        SortedDensity,
        SortedForce,
        SortedPosition,
        SortedPressure,
        SortedVelocity
    };

    struct sData : public Particles::Data {

        // density at particle position
        float4* density;

        // sum of SPH forces at particle position
        float4* force;

        // pressure at particle position
        float4* pressure;

    };

    typedef struct sData Data;
    typedef enum eBuffers Buffers;
}

#endif // __SPH_H__