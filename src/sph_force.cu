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
                data.density +=
                    SPH::Kernels::Poly6::getVariable(
                        cudaPrecalcParams.smoothLenSq, r, rLenSq);
            }

            ////////////////////////////////////////////////////////////////////

            __device__ static void postProcess(
                Data &data,
                uint const &index
            ) {
                float density =
                    cudaFluidParams.particleMass *
                    cudaPrecalcParams.poly6Coeff *
                    data.density;

                density = max(1.0, density);

                data.sorted.density[index] = density;
                data.sorted.pressure[index] =
                    cudaFluidParams.restPressure +
                    cudaFluidParams.gasStiffness *
                    (density - cudaFluidParams.restDensity);
            }

            ////////////////////////////////////////////////////////////////////

    };

};

#endif // __SPH_FORCE_CU__