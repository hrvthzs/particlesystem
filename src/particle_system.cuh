/**
 *
 */

#include "particle_system.h"

extern "C"
{
    void integrateSystem(
        float *pos,
        float *vel,
        float deltaTime,
        uint numParticles
    );


    void calculateHash(
        uint* particlesGridHash,
        uint* particlesGridIndex,
        float* positions,
        uint count
    );

    // sort particles based on hash
    void sortParticles(
        uint* gridParticleHash,
        uint* gridParticleIndex,
        uint count
    );

    void reorderDataAndFindCellStart(
        uint*  cellStart,
        uint*  cellEnd,
        float* sortedPositions,
        float* sortedVelocities,
        uint*  gridHash,
        uint*  gridIndex,
        float* oldPositions,
        float* oldVelocities,
        uint   count,
        uint   cells
    );

    void setParameters(ParticleSystemParameters *particleSystemParameters);
}