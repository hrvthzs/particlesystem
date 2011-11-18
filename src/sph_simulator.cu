#ifndef __SPH_SIMULATOR_CU__
#define __SPH_SIMULATOR_CU__

#include "sph_simulator.cuh"
#include "sph_kernel.cu"

#include "buffer_abstract.h"
#include "buffer_vertex.h"
#include "buffer_memory.cuh"
#include "buffer_manager.cuh"
#include "utils.cuh"

#include <iostream>

using namespace std;

namespace SPH {

    ////////////////////////////////////////////////////////////////////////////

    Simulator::Simulator () {

    }

    ////////////////////////////////////////////////////////////////////////////

    Simulator::~Simulator() {
        delete this->_bufferManager;
        delete this->_grid;
        delete this->_database;
    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::init() {
        this->_numParticles = 128*128;

        this->_createBuffers();

        this->_grid = new Grid::Uniform();
        this->_grid->allocate(this->_numParticles, 1.0f, 128.0f);

        this->_database = new Settings::Database();
        /*this->_database->addUpdateCallback(this);
        this->_database
            ->insert(Settings::GridSize, "Grid size", 10.0f, 20.0f, 5.3f);
        this->_database->print();

        this->_database->updateValue(Settings::GridSize, 5.4f);
        this->_database->print();
        */

    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::stop() {
        this->_bufferManager->freeBuffers();
        this->_grid->free();
    }

    ////////////////////////////////////////////////////////////////////////////

     float* Simulator::getPositions() {
        return (float*) this->_bufferManager->get(Position)->get();
    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::bindBuffers() {
        this->_bufferManager->bindBuffers();
    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::unbindBuffers() {
        this->_bufferManager->unbindBuffers();
    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::integrate (int numParticles, float deltaTime, float4 *pos) {
        uint minBlockSize, numBlocks, numThreads;
        minBlockSize = 416;
        Utils::computeGridSize(numParticles, minBlockSize, numBlocks, numThreads);

        Kernel::integrate<<<numBlocks, numThreads>>>(numParticles, deltaTime, pos);

    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::update() {
        this->_grid->hash((float4*) this->getPositions());
        this->_grid->sort();
        this->_orderData();
        this->integrate(this->_numParticles, 0.05f, (float4 *)this->getPositions());
        //cutilSafeCall(cutilDeviceSynchronize());
    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::valueChanged(Settings::RecordType type) {
        cout << "Value changed: " << type << endl;
    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::_createBuffers() {
        this->_bufferManager = new Buffer::Manager<Buffers>();

        //Buffer::Allocator* allocator = new Buffer::Allocator();

        Buffer::Memory<float4>* color    = new Buffer::Memory<float4>();
        Buffer::Memory<float4>* density  = new Buffer::Memory<float4>();
        Buffer::Memory<float4>* force    = new Buffer::Memory<float4>();
        Buffer::Vertex<float4>* position = new Buffer::Vertex<float4>();
        Buffer::Memory<float4>* pressure = new Buffer::Memory<float4>();
        Buffer::Memory<float4>* velocity = new Buffer::Memory<float4>();

        Buffer::Memory<float4>* sColor    = new Buffer::Memory<float4>();
        Buffer::Memory<float4>* sDensity  = new Buffer::Memory<float4>();
        Buffer::Memory<float4>* sForce    = new Buffer::Memory<float4>();
        Buffer::Vertex<float4>* sPosition = new Buffer::Vertex<float4>();
        Buffer::Memory<float4>* sPressure = new Buffer::Memory<float4>();
        Buffer::Memory<float4>* sVelocity = new Buffer::Memory<float4>();

        this->_positionsVBO = position->getVBO();

        this->_bufferManager
            ->addBuffer(Color,          (Buffer::Abstract<void>*) color)
            ->addBuffer(Density,        (Buffer::Abstract<void>*) density)
            ->addBuffer(Force,          (Buffer::Abstract<void>*) force)
            ->addBuffer(Position,       (Buffer::Abstract<void>*) position)
            ->addBuffer(Pressure,       (Buffer::Abstract<void>*) pressure)
            ->addBuffer(Velocity,       (Buffer::Abstract<void>*) velocity)
            ->addBuffer(SortedColor,    (Buffer::Abstract<void>*) sColor)
            ->addBuffer(SortedDensity,  (Buffer::Abstract<void>*) sDensity)
            ->addBuffer(SortedForce,    (Buffer::Abstract<void>*) sForce)
            ->addBuffer(SortedPosition, (Buffer::Abstract<void>*) sPosition)
            ->addBuffer(SortedPressure, (Buffer::Abstract<void>*) sPressure)
            ->addBuffer(SortedVelocity, (Buffer::Abstract<void>*) sVelocity);

        this->_bufferManager->allocateBuffers(this->_numParticles);

        size_t size = 0;
        size += color->getMemorySize();
        size += position->getMemorySize();
        size += density->getMemorySize();
        size += velocity->getMemorySize();
        size += force->getMemorySize();
        size += pressure->getMemorySize();
        std::cout << "Memory usage: " << 2 * size / 1024.0 / 1024.0 << " MB\n";

        this->_particleData.color    = color->get();
        this->_particleData.density  = density->get();
        this->_particleData.force    = force->get();
        this->_particleData.position = position->get();
        this->_particleData.pressure = pressure->get();
        this->_particleData.velocity = velocity->get();

        this->_sortedData.color    = sColor->get();
        this->_sortedData.density  = sDensity->get();
        this->_sortedData.force    = sForce->get();
        this->_sortedData.position = sPosition->get();
        this->_sortedData.pressure = sPressure->get();
        this->_sortedData.velocity = sVelocity->get();

    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::_orderData() {
        this->_grid->emptyCells();

        uint minBlockSize, numBlocks, numThreads;
        minBlockSize = 256;
        ::Utils::computeGridSize(
            this->_numParticles,
            minBlockSize,
            numBlocks,
            numThreads
        );

        uint sharedMemory = (numThreads + 1) * sizeof(uint);

        Kernel::update<Data><<<numBlocks, numThreads, sharedMemory>>>(
            this->_numParticles,
            this->_particleData,
            this->_sortedData,
            this->_grid->getData()
         );

    }

    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////


};

#endif // __SPH_SIMULATOR_CU__
