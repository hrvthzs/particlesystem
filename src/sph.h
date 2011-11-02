#ifndef __SPH_H__
#define __SPH_H__

namespace SPH {

    enum Buffers {
        density,
        force,
        pressure,
        sortedDensity,
        sortedForce,
        sortedPressure
    };

    typedef enum Buffers sph_buffer_t;
}

#endif // __SPH_H__