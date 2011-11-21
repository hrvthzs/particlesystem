#ifndef __SPH_H__
#define __SPH_H__

#include "particles.h"

namespace SPH {

    enum eBuffers {
        Colors,
        Densities,
        Forces,
        Positions,
        Pressures,
        Velevals,
        Velocities,
        SortedColors,
        SortedDensities,
        SortedForces,
        SortedPositions,
        SortedPressures,
        SortedVelevals,
        SortedVelocities
    };

    struct sData : public Particles::Data {

        // density at particle position
        float* density;

        // sum of SPH forces at particle position
        float4* force;

        // pressure at particle position
        float* pressure;

    };

    struct sPrecalcParams {
        // smoothing length^2
        float smoothLenSq;

        // precalculated terms for smoothing kernels
        float poly6Coeff;
        float pressurePrecalc;
        float spikyGradCoeff;
        float viscosityLapCoeff;
        float viscosityPrecalc;
    };

    struct sFluidParams
    {
        // the smoothing length of the kernels
        float smoothingLength;

        // the "ideal" rest state distance between each particle
        float particleRestDistance;

        // the scale of the simulation (ie. 0.1 of world scale)
        float scaleToSimulation;

        // the mass of each particle
        float particleMass;

        // pressure calculation parameters
        float restDensity;
        float restPressure;

        // viscosity of fluid
        float viscosity;

        // internal stiffness in fluid
        float gasStiffness;

        // external stiffness (against boundaries)
        float boundaryStiffness;

        // external dampening (against boundaries)
        float boundaryDampening;

        // the distance a particle will be "pushed" away from boundaries
        float boundaryDistance;

        // velocity limit of particle (artificial dampening of system)
        float velocityLimit;

        float frictionStaticLimit;
        float frictionKinetic;

    };

    typedef struct sData Data;
    typedef struct sFluidParams FluidParams;
    typedef struct sPrecalcParams PrecalcParams;
    typedef enum eBuffers Buffers;
}

#endif // __SPH_H__