#ifndef __SPH_H__
#define __SPH_H__

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

    struct sParticleData {

        // position of each particle
        float4* position;

        // color of each particle
        float4* color;

        // current velocity vector
        float4* velocity;

        // sum of SPH forces at particle position
        float4* force;

        // pressure at particle position
        float4* pressure;

        // density at particle position
        float4* density;
    };


    typedef struct sParticleData ParticleData;
    typedef enum eBuffers Buffers;
}

#endif // __SPH_H__