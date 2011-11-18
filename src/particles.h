#ifndef __PARTICLES_H__
#define __PARTICLES_H__

namespace Particles {

    enum eBuffers {
        Color,
        Position,
        Velocity,
        SortedColor,
        SortedPosition,
        SortedVelocity
    };

    struct sData {

        // color of each particle
        float4* color;

        // position of each particle
        float4* position;

        // current velocity vector
        float4* velocity;

    };


    typedef struct sData Data;
    typedef enum eBuffers Buffers;

};

#endif // __PARTICLES_H__


