#ifndef _SPH_KERNELS_VISCOSITY_CU_
#define _SPH_KERNELS_VISCOSITY_CU_

class Viscosity {

    public:

        static __device__ __host__
        float getConstant (float smoothLen) {
            return 15.0f / (M_PI * pow(smoothLen, 6.0f));
        }

        static __device__ __host__
        float getVariable(float smoothLen, float3 r, float rLen) {
            float variable = smoothLen - rLen;
            return variable * variable * variable;
        }

        static __device__ __host__
        float getGradientConstant (float smoothLen) {
            return 15.0f / (2.0f * M_PI * pow(smoothLen, 3.0f));
        }

        static __device__ __host__
        float getGradientVariable(float smoothLen, float3 r, float rLen) {
            float variable1 = (-3.0f * rLen) / (2.0f * pow(smoothLen, 3.0f));
            float variable2 = 2.0f / (smoothLen * smoothLen);
            float variable3 = -smoothLen / (2.0f * pow(rLen, 3.0f));
            return variable1 + variable2 + variable3;
        }

        static __device__ __host__
        float getLaplacianConstant (float smoothLen) {
            return 45.0f / (M_PI * pow(smoothLen, 6.0f));
        }

        static __device__ __host__
        float getLaplacianVariable(float smoothLen, float3 r, float rLen) {
            return smoothLen - rLen;
        }

};

#endif // _SPH_KERNELS_VISCOSITY_CU_