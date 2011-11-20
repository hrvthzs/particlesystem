#ifndef __SPH_DENSITY_CU__
#define __SPH_DENSITY_CU__

#include "sph_kernels.cu"

namespace SPH {

    class Density {

        public:

            ////////////////////////////////////////////////////////////////////

            struct Data {
                float density;
                SPH::Data sorted;
            };

            ////////////////////////////////////////////////////////////////////

            __device__ static void preProcess(
                Data &data,
                uint const &index
            ) {
                data.density = 0.0f;
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

#endif // __SPH_DENSITY_CU__