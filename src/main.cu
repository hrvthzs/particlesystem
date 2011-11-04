#include <stdlib.h>
#include "particles_renderer.h"
#include "sph_simulator.cuh"

int main(int, char**) {

    Particles::Simulator *simulator = new SPH::Simulator();
    Particles::Renderer *renderer = new Particles::Renderer(simulator);
    return renderer->render() ? EXIT_SUCCESS : EXIT_FAILURE;
}