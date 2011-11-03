#ifndef __PARTICLES_SIMULATOR_H__
#define __PARTICLES_SIMULATOR_H__


namespace Particles {

    class Simulator {

    public:
        Simulator() {}
        virtual ~Simulator() {}

        virtual void update() = 0;

    };
};

#endif // __PARTICLES_SIMULATOR_H__
