#ifndef _SPH_KERNELS_SPIKY_CU_
#define _SPH_KERNELS_SPIKY_CU_

class Spiky {

    public:

        ////////////////////////////////////////////////////////////////////////

        static __device__ __host__
        float getConstant (float smoothLen) {
            return 15.0f / (M_PI * pow(smoothLen, 6.0f));
        }

        ////////////////////////////////////////////////////////////////////////

        static __device__ __host__
        float getVariable(float smoothLen, float3 r, float rLen) {
            float variable = smoothLen - rLen;
            return variable * variable * variable;
        }

        ////////////////////////////////////////////////////////////////////////

        static __device__ __host__
        float getGradientConstant (float smoothLen) {
            return -45.0f / (M_PI * pow(smoothLen, 6.0f));
        }

        ////////////////////////////////////////////////////////////////////////

        static __device__ __host__
        float3 getGradientVariable(float smoothLen, float3 r, float rLen) {
            float variable = smoothLen - rLen;
            return r * (1.0f / rLen) * (variable * variable);
        }

        ////////////////////////////////////////////////////////////////////////

        static __device__ __host__
        float3 getGradient(float smoothLen, float3 r, float rLen) {
            return
                getGradientConstant(smoothLen) *
                getGradientVariable(smoothLen, r, rLen);
        }

        ////////////////////////////////////////////////////////////////////////

        static __device__ __host__
        float getLaplacianConstant (float smoothLen) {
            return -90.0f / (M_PI * pow(smoothLen, 9.0f));
        }

        ////////////////////////////////////////////////////////////////////////

        static __device__ __host__
        float3 getLaplacianVariable(float smoothLen, float3 r, float rLen) {
            float variable = smoothLen - rLen;
            float variable2 = smoothLen - (2 * rLen);
            return (1.0f / r) * (variable * variable2);
        }

        ////////////////////////////////////////////////////////////////////////

};

#endif // _SPH_KERNELS_SPIKY_CU_