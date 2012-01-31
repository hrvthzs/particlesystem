#ifndef __COLORS_CUH__
#define __COLORS_CUH__

namespace Colors {

    enum Gradient {
        Direct,
        White,
        Blackish,
        BlackToCyan,
        BlueToWhite,
        HSVBlueToRed,
    };

    enum Source {
        Force,
        Pressure,
        Velocity
    };

}

#endif // __COLORS_CUH__