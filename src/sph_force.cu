#ifndef __SPH_FORCE_CU__
#define __SPH_FORCE_CU__

#include "sph_kernels.cu"

namespace SPH {

    class Force {

        public:

            ////////////////////////////////////////////////////////////////////

            struct Data {
                float3 veleval;
                float density;
                float pressure;

                float3 velevalN;
                float densityN;
                float pressureN;

                float3 viscosityForce;
                float3 pressureForce;

                SPH::Data sorted;
            };

            ////////////////////////////////////////////////////////////////////

            __device__ static void preProcess(
                Data &data,
                uint const &index
            ) {
                data.veleval  = make_float3(data.sorted.veleval[index]);
                data.density  = data.sorted.density[index];
                data.pressure = data.sorted.pressure[index];

                data.pressureForce = make_float3(0,0,0);
                data.viscosityForce = make_float3(0,0,0);
            }

            ////////////////////////////////////////////////////////////////////

            __device__ static void process(
                Data &data,
                uint const &index,
                uint const &indexN, // neighbour
                float3 const &r,
                float const &rLen,
                float const &rLenSq
            ) {
                data.velevalN  = make_float3(data.sorted.veleval[indexN]);
                data.densityN  = data.sorted.density[indexN];
                data.pressureN = data.sorted.pressure[indexN];

                data.pressureForce  +=
                    (
                        (data.pressure + data.pressureN) /
                        (data.densityN * data.density)
                    ) *
                    SPH::Kernels::Spiky::getGradientVariable(
                        cudaFluidParams.smoothingLength,
                        r,
                        rLen
                    );

                /*data.pressureForce +=
                    (
                        (
                            data.pressure/
                            (data.density*data.density)
                        ) +
                        (
                            data.pressureN/
                            (data.densityN*data.densityN))
                    ) *
                    SPH::Kernels::Spiky::getGradientVariable(
                        cudaFluidParams.smoothingLength,
                        r,
                        rLen
                    );
                */
                /*data.pressureForce +=
                    (
                        (data.pressure + data.pressureN) /
                        (2.0f*data.densityN)
                    ) *
                    SPH::Kernels::Spiky::getGradientVariable(
                        cudaFluidParams.smoothingLength,
                        r,
                        rLen
                    );
                */
                data.viscosityForce +=
                    (
                        (data.veleval  - data. velevalN ) /
                        (data.density * data.densityN)
                    ) *
                    SPH::Kernels::Viscosity::getLaplacianVariable(
                        cudaFluidParams.smoothingLength,
                        r,
                        rLen
                    );
            }

            ////////////////////////////////////////////////////////////////////

            __device__ static void postProcess(
                Data &data,
                uint const &index
            ) {

                float3 sumForce = (
                    cudaPrecalcParams.pressurePrecalc * data.pressureForce +
                    cudaPrecalcParams.viscosityPrecalc * data.viscosityForce
                );

                // Calculate the force, the particle_mass is added here because
                // it is constant and thus there is no need for it to
                // be inside the sum loop.
                data.sorted.force[index] =
                    make_float4(sumForce * cudaFluidParams.particleMass, 1.0f);
            }

            ////////////////////////////////////////////////////////////////////

    };

};

#endif // __SPH_FORCE_CU__