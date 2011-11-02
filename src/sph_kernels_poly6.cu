#ifndef _SPH_KERNELS_POLY6_CU_
#define _SPH_KERNELS_POLY6_CU_

class Poly6 {

    static __device__ __host__ float getConstant (float length) {
        return 315.0f / (64.0f * M_PI * pow(length, 9.0f));
    }

    static __device__ __host__ float getVariable(float lengthSquare, float radiusSquare) {
        float variableSquare = lengthSquare * radiusSquare;
        return variableSquare  * variableSquare  * variableSquare ;
    }


    static __device__ __host__ float getGradientConstant (float length) {
        return -945.0f / (32.0f * M_PI * pow(length, 9.0f));
    }

    static __device__ __host__ float getGradientVariable(float lengthSquare, float radiusSquare) {
        float variableSquare = lengthSquare * radiusSquare;
        return variableSquare  * variableSquare  * variableSquare ;
    }

    static __device__ __host__ float getLaplacianConstant (float length) {
        return 945.0f / (8.0f * M_PI * pow(length, 9.0f));
    }
};

#endif // _SPH_KERNELS_POLY6_CU_