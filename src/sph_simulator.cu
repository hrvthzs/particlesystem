#ifndef __SPH_SIMULATOR_CU__
#define __SPH_SIMULATOR_CU__

#include "sph_simulator.cuh"
#include "sph_integrator.cu"

// !!! for template classes definitions must be included too
#include "buffer_memory.cu"
#include "buffer_vertex.cpp"
#include "buffer_manager.cu"


namespace SPH {

    /**
     * Constructor
     */
    Simulator::Simulator () {
    }

    Simulator::~Simulator() {
        delete this->_bufferManager;
    }

    void Simulator::init() {
        this->_numParticles = 8*8;

        //Buffer::Allocator *allocator = new Buffer::Allocator();

        this->_positionsBuffer = new Buffer::Vertex<float4>();

        this->_positionsBuffer->allocate(this->_numParticles);
        //this->_positionsBuffer->bind();

        std::cout << this->_numParticles * sizeof(float) << std::endl;
        std::cout << this->_positionsBuffer->getMemorySize() << std::endl;
        this->_positionsVBO = this->_positionsBuffer->getVBO();
        this->_bufferManager = new Buffer::Manager<sph_buffer_t>();


    }

     float* Simulator::getPositions() {
         return (float*)this->_positionsBuffer->get();
    }

    void Simulator::bindBuffers() {
        this->_positionsBuffer->bind();
    }

    void Simulator::unbindBuffers() {
        this->_positionsBuffer->unbind();
    }


    /**
     * Intergrate system
     */
    void Simulator::integrate (int numParticles, float deltaTime, float4 *pos) {
        uint minBlockSize, numBlocks, numThreads;
        minBlockSize = 416;
        this->_computeGridSize(numParticles, minBlockSize, numBlocks, numThreads);

        integrate_kernel<<<numBlocks, numThreads>>>(numParticles, deltaTime, pos);

    }

    void Simulator::update() {
        this->integrate(this->_numParticles, 0.05f, this->_positionsBuffer->get());
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
