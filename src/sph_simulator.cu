#ifndef __SPH_SIMULATOR_CU__
#define __SPH_SIMULATOR_CU__

#include "sph_simulator.cuh"
#include "sph_integrator.cu"

#include "buffer_abstract.h"
#include "buffer_vertex.h"
#include "buffer_memory.cuh"
#include "buffer_manager.cuh"
#include "utils.cuh"

#include <iostream>

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

        this->_bufferManager = new Buffer::Manager<Buffers>();
        Buffer::Allocator *allocator = new Buffer::Allocator();

        Buffer::Vertex<float4>* positionBuffer = new Buffer::Vertex<float4>();
        Buffer::Memory<float4>* densityBuffer  = new Buffer::Memory<float4>();
        Buffer::Memory<float4>* velocityBuffer = new Buffer::Memory<float4>();
        Buffer::Memory<float4>* forceBuffer    = new Buffer::Memory<float4>();
        Buffer::Memory<float4>* pressureBuffer = new Buffer::Memory<float4>();

        this->_positionsVBO = positionBuffer->getVBO();

        this->_bufferManager
            ->addBuffer(Position, (Buffer::Abstract<void>*) positionBuffer)
            ->addBuffer(Density,  (Buffer::Abstract<void>*) densityBuffer)
            ->addBuffer(Velocity, (Buffer::Abstract<void>*) velocityBuffer)
            ->addBuffer(Force,    (Buffer::Abstract<void>*) forceBuffer)
            ->addBuffer(Pressure, (Buffer::Abstract<void>*) pressureBuffer);

        this->_bufferManager->allocateBuffers(this->_numParticles);

        size_t size = 0;
        size += positionBuffer->getMemorySize();
        size += densityBuffer->getMemorySize();
        size += velocityBuffer->getMemorySize();
        size += forceBuffer->getMemorySize();
        size += pressureBuffer->getMemorySize();
        std::cout << "Memory usage: " << size / 1024.0 / 1024.0 << " MB\n";

    }

    void Simulator::stop() {
        this->_bufferManager->freeBuffers();
    }

     float* Simulator::getPositions() {
        return (float*) this->_bufferManager->get(Position)->get();
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
        Utils::computeGridSize(numParticles, minBlockSize, numBlocks, numThreads);

        integrate_kernel<<<numBlocks, numThreads>>>(numParticles, deltaTime, pos);

    }

    void Simulator::update() {
        this->integrate(this->_numParticles, 0.05f, (float4 *)this->getPositions());
    }

};

#endif // __SPH_SIMULATOR_CU__
