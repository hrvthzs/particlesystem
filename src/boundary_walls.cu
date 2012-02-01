#ifndef __BOUNDARY_WALLS_CU__
#define __BOUNDARY_WALLS_CU__

#include "boundary.cu"

namespace Boundary {

    namespace Walls {

    __device__ float3 calculateWallsNoPenetrationForce(
        float3 const& position,
        float3 const& velocity,
        float3 const& gridMin,
        float3 const& gridMax,
        float const& boundaryDistance,
        float const& boundaryStiffness,
        float const& boundaryDampening
    ) {
        float3 repulsionForce = make_float3(0,0,0);
        float difference;

        difference = boundaryDistance - (position.y - gridMin.y );
        if (difference > EPSILON) {
            float3 normal = make_float3(0,1,0);
            repulsionForce += calculateRepulsionForce(velocity, normal, difference, boundaryDampening, boundaryStiffness);
        }

        difference = boundaryDistance - (gridMax.y - position.y);
        if (difference > EPSILON) {
            float3 normal = make_float3(0,-1,0);
            repulsionForce  += calculateRepulsionForce(velocity, normal, difference, boundaryDampening, boundaryStiffness);
        }

        difference = boundaryDistance - (position.z - gridMin.z);
        if (difference > EPSILON ) {
            float3 normal = make_float3(0,0,1);
            repulsionForce += calculateRepulsionForce(velocity, normal, difference, boundaryDampening, boundaryStiffness);
        }

        difference = boundaryDistance - (gridMax.z - position.z);
        if (difference > EPSILON) {
            float3 normal = make_float3(0,0,-1);
            repulsionForce += calculateRepulsionForce(velocity, normal, difference, boundaryDampening, boundaryStiffness);;
        }

        difference = boundaryDistance - (position.x - gridMin.x);
        if (difference > EPSILON ) {
            float3 normal = make_float3(1,0,0);
            repulsionForce += calculateRepulsionForce(velocity, normal, difference, boundaryDampening, boundaryStiffness);
        }

        difference = boundaryDistance - (gridMax.x - position.x);
        if (difference > EPSILON) {
            float3 normal = make_float3(-1,0,0);
            repulsionForce += calculateRepulsionForce(velocity, normal, difference, boundaryDampening, boundaryStiffness);
        }

        return repulsionForce;
    }


    __device__ float3 calculateWallsNoSlipForce(
        float3 const& position,
        float3 const& velocity,
        float3 const& force,
        float3 const& gridMin,
        float3 const& gridMax,
        float const& boundaryDistance,
        float const& kineticFriction,
        float const& staticFrictionLimit
    ) {
        float3 frictionForce = make_float3(0,0,0);
        float difference;

        difference = boundaryDistance - (position.y - gridMin.y);
        if (difference > EPSILON) {
            float3 normal = make_float3(0,1,0);
            frictionForce += calculateFrictionForce(velocity, force, normal, kineticFriction, staticFrictionLimit);
        }

        difference = boundaryDistance - (gridMax.y - position.y);
        if (difference > EPSILON) {
            float3 normal = make_float3(0,-1,0);
            frictionForce += calculateFrictionForce(velocity, force, normal, kineticFriction, staticFrictionLimit);
        }

        difference = boundaryDistance - (position.z - gridMin.z);
        if (difference > EPSILON ) {
            float3 normal = make_float3(0,0,1);
            frictionForce += calculateFrictionForce(velocity, force, normal, kineticFriction, staticFrictionLimit);
        }

        difference = boundaryDistance - (gridMax.z - position.z);
        if (difference > EPSILON) {
            float3 normal = make_float3(0,0,-1);
            frictionForce += calculateFrictionForce(velocity, force, normal, kineticFriction, staticFrictionLimit);
        }

        difference = boundaryDistance - (position.x - gridMin.x);
        if (difference > EPSILON ) {
            float3 normal = make_float3(1,0,0);
            frictionForce += calculateFrictionForce(velocity, force, normal, kineticFriction, staticFrictionLimit);
        }

        difference = boundaryDistance - (gridMax.x - position.x);
        if (difference > EPSILON) {
            float3 normal = make_float3(-1,0,0);
            frictionForce += calculateFrictionForce(velocity, force, normal, kineticFriction, staticFrictionLimit);
        }

        return frictionForce;
    }

    };
};

#endif // __BOUNDARY_WALLS_CU__