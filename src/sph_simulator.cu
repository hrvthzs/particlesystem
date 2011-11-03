#ifndef __SPH_SIMULATOR_CU__
#define __SPH_SIMULATOR_CU__

#include "sph_simulator.cuh"
#include "sph_integrator.cu"

// !!! template classes must be included definition too
#include "buffer_buffer.cu"
#include "buffer_manager.cu"


namespace SPH {

    /**
     * Constructor
     */
    Simulator::Simulator () {
        Buffer::Allocator *allocator = new Buffer::Allocator();
        //Buffer::Buffer<float> *buffer = new Buffer::Buffer<float>(allocator, Buffer::host);


        this->_bufferManager = new Buffer::Manager<sph_buffer_t>();

    }

    Simulator::~Simulator() {
        delete this->_bufferManager;
    }

    /**
     * Intergrate system
     */
    void Simulator::integrate (int numParticles, float deltaTime, float *pos) {
        uint minBlockSize, numBlocks, numThreads;
        minBlockSize = 416;
        this->_computeGridSize(numParticles, minBlockSize, numBlocks, numThreads);

        integrate_kernel<<<numBlocks, numThreads>>>(numParticles, deltaTime, (float4*) pos);

    }

    void Simulator::update() {

    }

    /**
     * Round a / b to nearest higher integer value
     */
    uint Simulator::_iDivUp(uint a, uint b) {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    /*
     * Compute grid and thread block size for a given number of elements
     *
     * @param n - number of particles
     * @param blockSize - minimal number of threads in block
     * @param numBlocks - outputs number of required block in grid
     * @param numThreads - outputs number of required threads in blocks
     */
    void Simulator::_computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads) {
        numThreads = min(blockSize, n);
        numBlocks = this->_iDivUp(n, numThreads);
    }

};

#endif // __SPH_SIMULATOR_CU__
