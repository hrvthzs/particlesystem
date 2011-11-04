#ifndef __SPH_SIMULATOR_CU__
#define __SPH_SIMULATOR_CU__

#include "sph_simulator.cuh"
#include "sph_integrator.cu"

// !!! for template classes definitions must be included too
#include "buffer_abstract.h"
#include "buffer_vertex.cpp"
#include "buffer_memory.cu"
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
        this->_numParticles = 128*128;

        this->_bufferManager = new Buffer::Manager<sph_buffer_t>();
        Buffer::Allocator *allocator = new Buffer::Allocator();

        Buffer::Vertex<float4>* positionBuffer = new Buffer::Vertex<float4>();
        Buffer::Memory<float4>* densityBuffer  = new Buffer::Memory<float4>();
        Buffer::Memory<float4>* velocityBuffer = new Buffer::Memory<float4>();
        Buffer::Memory<float4>* forceBuffer    = new Buffer::Memory<float4>();
        Buffer::Memory<float4>* pressureBuffer = new Buffer::Memory<float4>();

        this->_positionsVBO = positionBuffer->getVBO();

        this->_bufferManager
            ->addBuffer(Position, (Buffer::Abstract<void>*) positionBuffer);
        this->_bufferManager
            ->addBuffer(Density, (Buffer::Abstract<void>*) densityBuffer);
        this->_bufferManager
            ->addBuffer(Velocity, (Buffer::Abstract<void>*) velocityBuffer);
        this->_bufferManager
            ->addBuffer(Force, (Buffer::Abstract<void>*) forceBuffer);
        this->_bufferManager
            ->addBuffer(Pressure, (Buffer::Abstract<void>*) pressureBuffer);

        this->_bufferManager->allocateBuffers(this->_numParticles);

        size_t size = 0;
        size += positionBuffer->getMemorySize();
        size += densityBuffer->getMemorySize();
        size += velocityBuffer->getMemorySize();
        size += forceBuffer->getMemorySize();
        size += pressureBuffer->getMemorySize();
        std::cout << "Memory usage: " << size / 1024.0 / 1024.0 << " MB" << std::endl;

    }

    void Simulator::stop() {
        this->_bufferManager->freeBuffers();
    }

     float* Simulator::getPositions() {
        Buffer::Abstract<void>* buffer = this->_bufferManager->get(Position);
        return (float*) buffer->get();
    }

    void Simulator::bindBuffers() {
        this->_bufferManager->bindBuffers();
    }

    void Simulator::unbindBuffers() {
        this->_bufferManager->unbindBuffers();
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
        this->integrate(this->_numParticles, 0.05f, (float4 *)this->getPositions());
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
