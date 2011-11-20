#ifndef _SPH_KERNELS_CU_
#define _SPH_KERNELS_CU_

#include <math.h>

namespace SPH {
    namespace Kernels {
        #include "sph_kernels_poly6.cu"
        #include "sph_kernels_spiky.cu"
        #include "sph_kernels_viscosity.cu"
    };
};

#endif // _SPH_KERNELS_CU_