#ifndef _SPH_KERNELS_POLY6_CU_
#define _SPH_KERNELS_POLY6_CU_

class Poly6 {

    public:

        ////////////////////////////////////////////////////////////////////////

        static __device__ __host__
        float getConstant (float smoothLen) {
            return 315.0f / (64.0f * M_PI * pow(smoothLen, 9.0f));
        }

        ////////////////////////////////////////////////////////////////////////

        static __device__ __host__
        float getVariable(float smoothLenSq, float3 r, float rLenSq) {
            float variableSquare = smoothLenSq - rLenSq;
            return variableSquare * variableSquare * variableSquare ;
        }

        ////////////////////////////////////////////////////////////////////////

        static __device__ __host__
        float getGradientConstant (float smoothLen) {
            return -945.0f / (32.0f * M_PI * pow(smoothLen, 9.0f));
        }

        ////////////////////////////////////////////////////////////////////////

        static __device__ __host__
        float3 getGradientVariable(float smoothLenSq, float3 r, float rLenSq) {
            float variableSquare = smoothLenSq - rLenSq;
            return r * (variableSquare * variableSquare);
        }

        ////////////////////////////////////////////////////////////////////////

        static __device__ __host__
        float3 getGradient(
            float smoothLen,
            float smoothLenSq,
            float3 r,
            float rLenSq
        ) {
            return
                getGradientConstant(smoothLen) *
                getGradientVariable(smoothLenSq, r, rLenSq);
        }

        ////////////////////////////////////////////////////////////////////////

        static __device__ __host__
        float getLaplacianConstant (float smoothLen) {
            return 945.0f / (8.0f * M_PI * pow(smoothLen, 9.0f));
        }

        ////////////////////////////////////////////////////////////////////////

        static __device__ __host__
        float getLaplacianVariable(float smoothLenSq, float rLenSq) {
            float variableSquare = smoothLenSq - rLenSq;
            return
            variableSquare * (rLenSq - ((3.0/4.0) * variableSquare));
        }

        ////////////////////////////////////////////////////////////////////////

};

#endif // _SPH_KERNELS_POLY6_CU_