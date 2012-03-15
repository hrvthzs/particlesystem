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

    enum eRenderMode {
        RenderPoints,
        RenderMarching,
        RenderTesselation,
        RenderTesselationTriangles,
        RenderNormals
    };

    struct sData {

        // color of each particle
        float4* color;

        // position of each particle
        float4* position;

        // current velocity vector
        float4* velocity;

        // vel_eval (in world space, used for leap-frog integration)
        float4* veleval;

    };

    struct sGridMinMax {
        float3 min;
        float3 max;
    };

    typedef struct sData Data;
    typedef struct sGridMinMax GridMinMax;
    typedef enum eBuffers Buffers;
    typedef enum eRenderMode RenderMode;

};

#endif // __PARTICLES_H__


