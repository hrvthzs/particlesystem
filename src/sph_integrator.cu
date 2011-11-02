#ifndef __SPH_INTEGRATOR_CU__
#define __SPH_INTEGRATOR_CU__

__global__ void integrate_kernel(
    int numParticles,
    float deltaTime,
    float4 *pos
) {
    int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;

    pos[index].y -= 0.001f;


}

#endif // __SPH_INTEGRATOR_CU__
