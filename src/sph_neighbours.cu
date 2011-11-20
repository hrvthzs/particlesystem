#ifndef __SPH_NEIGHBOURS_CU__
#define __SPH_NEIGHBOURS_CU__

namespace SPH {


    template<class C, class D>
    class Neighbours {

        public:

            ////////////////////////////////////////////////////////////////////

            __device__ static void preProcess(
                D &data,
                uint const &index
            ) {
                C::preProcess(data, index);
            }

            ////////////////////////////////////////////////////////////////////

            __device__ static void process(
                D &data,
                uint const &index,
                uint const &indexN, // neighbour
                float3 const &r,
                float const &rLen,
                float const &rLenSq
            ) {
                C::process(index, indexN, r, rLen, rLenSq);
            }

            ////////////////////////////////////////////////////////////////////

            __device__ static void postProcess(
                D &data,
                uint const &index
            ) {
                C::postProcess(data, index);
            }

            ////////////////////////////////////////////////////////////////////

            __device__ static void processNeighbour(
                D &data,
                uint const &index,
                uint const &indexN, // neighbour
                float3 const &position
            ) {
                if (index != indexN) {
                    float3 positionN = make_float3(data.sorted.position[indexN]);

                    float3 r =
                        (position - positionN) *
                        cudaFluidParams.scaleToSimulation;

                    float rLenSq = dot(r, r);
                    float rLen = sqrtf(rLenSq);

                    if (rLen < cudaFluidParams.smoothingLength) {
                        C::process(data, index, indexN, r, rLen, rLenSq);
                    }
                }
            }

            ////////////////////////////////////////////////////////////////////

    };

};

#endif // __SPH_NEIGHBOURS_CU__