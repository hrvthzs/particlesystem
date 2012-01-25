#ifndef __BOUNDARY_CU__
#define __BOUNDARY_CU__

namespace Boundary {

    //for collision detection
    #define EPSILON 0.00001f


    __device__ float3 calculateRepulsionForce(
        float3 const& velocity,
        float3 const& normal,
        float const& boundaryDistance,
        float const& boundaryDampening,
        float const& boundaryStiffness
    ) {
        return (boundaryStiffness * boundaryDistance -
                boundaryDampening * dot(normal, velocity)
               ) * normal;
    }

    __device__ float3 calculateFrictionForce(
        float3 const& vel,
        float3 const& force,
        float3 const& normal,
        float const& kineticFriction,
        float const& staticFrictionLimit
    )
    {
        float3 frictionForce = make_float3(0,0,0);

        float3 fNor = force * dot(normal, force);
        float3 fTan = force - fNor;

        float3 vNor = vel * dot(normal, vel);
        float3 vTan = vel - vNor;

        if((vTan.x + vTan.y + vTan.z)/3.0f > staticFrictionLimit) {
            frictionForce = -vTan;
        } else {
            frictionForce = kineticFriction * -vTan;
        }

        return frictionForce;
    }

};

#endif // __BOUNDARY_CU__