#ifndef __SETTINGS_H__
#define __SETTINGS_H__

#include <string>

using namespace std;

namespace Settings {

    struct sRecord {
        string name;
        float minimum;
        float maximum;
        float value;
        string unit;
        bool editable;
    };

    enum RecordType {
        AnimPartForce,
        AnimPartNum,
        AnimChangeAxis,
        BoundaryDistance,
        BoundaryStiffness,
        BoundaryDampening,
        CellSize,
        ColorGradient,
        ColorSource,
        DynamicColoring,
        GasStiffness,
        GridSize,
        Interpolation,
        KineticFriction,
        ParticleMass,
        ParticleNumber,
        ParticleRestDistance,
        RestDensity,
        RestPressure,
        SimulationScale,
        SmootingLength,
        StaticFrictionLimit,
        Timestep,
        VelocityLimit,
        Viscosity
    };

    typedef struct sRecord Record;

};

#endif // __SETTING_H__