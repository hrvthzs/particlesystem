#ifndef __SPH_SIMULATOR_CU__
#define __SPH_SIMULATOR_CU__

#include "sph_simulator.cuh"
#include "sph_kernel.cu"

#include "buffer_abstract.h"
#include "buffer_vertex.h"
#include "buffer_memory.cuh"
#include "buffer_manager.cuh"
#include "sph_kernels.cu"
#include "utils.cuh"

#include <iostream>

using namespace std;
using namespace Settings;

namespace SPH {

    ////////////////////////////////////////////////////////////////////////////

    Simulator::Simulator () {

    }

    ////////////////////////////////////////////////////////////////////////////

    Simulator::~Simulator() {
        delete this->_bufferManager;
        delete this->_grid;
    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::init(uint numParticles) {
        this->_numParticles = numParticles;

        this->_createBuffers();


        // DATABASE
        this->_database
            ->insert(ParticleNumber, "Particles", this->_numParticles)
            ->insert(GridSize, "Grid size", 2.0f)
            ->insert(Timestep, "Timestep", 0.0f, 1.0f, 0.01f)
            ->insert(RestDensity, "Rest density", 0.0f, 10000.0f, 2000.0f)
            ->insert(RestPressure, "Rest pressure", 0.0f, 10000.0f, 0.0f)
            ->insert(GasStiffness, "Gas Stiffness", 0.001f, 10.0f, 1.0f)
            ->insert(Viscosity, "Viscosity", 0.0f, 100.0f, 1.0f)
            ->insert(BoundaryDampening, "Bound. damp.", 0.0f, 10000.0f, 256.0f)
            ->insert(BoundaryStiffness, "Bound. stiff.", 0.0f, 100000.0f, 20000.0f)
            ->insert(VelocityLimit, "Veloc. limit", 0.0f, 10000.0f, 600.0f)
            ->insert(SimulationScale, "Sim. scale", 0.0f, 1.0f, 1.0f)
            ->insert(KineticFriction, "Kinet. fric.", 0.0f, 10000.0f, 0.0f)
            ->insert(StaticFrictionLimit, "Stat. Fric. Lim.", 0.0f, 10000.0f, 0.0f)
            ->insert(DynamicColoring, "DynamicColoring", 0.0f, true);

            uint n = this->_numParticles;

            this->_numParticles *= 4;

            float particleMass =
                ((128.0f * 1024.0f ) / this->_numParticles) * 0.0002f * 0.0005;
            float particleRestDist =
                0.87f *
                pow(
                    particleMass / this->_database->selectValue(RestDensity),
                    1.0f/3.0f
                );
            float boundaryDist = 1.5 * particleRestDist;
            //float smoothingLength = pow(this->_numParticles, 1.0/3.0)*1.2;//2.0 * particleRestDist;
            float cellSize = 2.0/pow(this->_numParticles, 1.0/3.0);
            // maybe 2 x cellSize is the ideal value for smoothing length
            // but not only if simulation scale is 1
            float smoothingLength = cellSize*1.9;//2.0 * particleRestDist;
                //smoothingLength * this->_database->selectValue(SimulationScale);

        this->_database
            ->insert(ParticleMass, "Particle mass", particleMass)
            ->insert(ParticleRestDistance, "Part. rest dist.", particleRestDist)
            ->insert(BoundaryDistance, "Bound. dist.", boundaryDist)
            ->insert(SmootingLength, "Smooth. len.", smoothingLength)
            ->insert(CellSize, "Cell size", cellSize);

        this->_database->print();

        this->_numParticles = n;

        this->_grid = new Grid::Uniform();
        this->_grid->allocate(
            this->_numParticles,
            cellSize,
            this->_database->selectValue(GridSize)
        );
        this->_grid->printParams();
        this->_gridParams = this->_grid->getParams();

        this->_updateParams();



    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::stop() {
        this->_bufferManager->freeBuffers();
        this->_grid->free();
    }

    ////////////////////////////////////////////////////////////////////////////

     float* Simulator::getPositions() {
        return (float*) this->_bufferManager->get(Positions)->get();
    }

    ////////////////////////////////////////////////////////////////////////////

    float* Simulator::getColors() {
        return (float*) this->_bufferManager->get(Colors)->get();
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

    void Simulator::_step1() {
        uint numBlocks, numThreads;

        Utils::computeGridSize(this->_numParticles, 128, numBlocks, numThreads);

        Kernel::computeDensity<<<numBlocks, numThreads>>>(
            this->_numParticles,
            this->_sortedData,
            this->_grid->getData()
        );
    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::_step2() {
        uint numBlocks, numThreads;
        Utils::computeGridSize(this->_numParticles, 128, numBlocks, numThreads);

        Kernel::computeForce<<<numBlocks, numThreads>>>(
            this->_numParticles,
            this->_sortedData,
            this->_grid->getData()
        );
    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::integrate(int numParticles, float deltaTime) {
        uint minBlockSize, numBlocks, numThreads;
        minBlockSize = 416;
        Utils::computeGridSize(numParticles, minBlockSize, numBlocks, numThreads);

        Kernel::integrate<Data><<<numBlocks, numThreads>>>(
            numParticles,
            deltaTime,
            this->_particleData,
            this->_sortedData,
            this->_grid->getData()
        );

    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::update() {
        this->_grid->hash((float4*) this->getPositions());
        this->_grid->sort();
        this->_orderData();

        Buffer::Memory<float>* buffer =
            new Buffer::Memory<float>(new Buffer::Allocator(), Buffer::Host);

        buffer->allocate(this->_numParticles);

        GridData gridData = this->_grid->getData();

        cudaMemcpy(buffer->get(), this->_sortedData.neighbours, this->_numParticles * sizeof(float), cudaMemcpyDeviceToHost);

        float* e = buffer->get();

        /*Buffer::Memory<float4>* posBuffer =
            new Buffer::Memory<float4>(new Buffer::Allocator(), Buffer::Host);

        posBuffer->allocate(this->_numParticles);

        cudaMemcpy(posBuffer->get(), this->_sortedData.force, this->_numParticles * sizeof(float4), cudaMemcpyDeviceToHost);
        float4* pos = posBuffer->get();
        */
        /*Buffer::Memory<int3>* cellBuffer =
        new Buffer::Memory<int3>(new Buffer::Allocator(), Buffer::Host);

        cellBuffer->allocate(this->_numParticles);

        cudaMemcpy(cellBuffer->get(), this->_sortedData.cellPos, this->_numParticles * sizeof(int3), cudaMemcpyDeviceToHost);
        int3* cell = cellBuffer->get();
        */
        //cutilSafeCall(cutilDeviceSynchronize());
        /*int c = 0;
        for(uint i=0;i< this->_numParticles; i++) {
            //cout << e[i] << " " << pos[i].x << " " << pos[i].y << " " << pos[i].z << endl;
            //cout << pos[i].x << " " << pos[i].y << " " << pos[i].z << endl;
            //cout << e[i] << " " << cell[i].x << " " << cell[i].y << " " << cell[i].z << endl;
            //cout << e[i] << endl;
            if (e[i] < 18) c++;
            //cout << "-----" << endl;
        }*/

        //std::cout << "____________________" << std::endl;
        //std::cout << c << std::endl;



        this->_step1();
        this->_step2();
        this->integrate(this->_numParticles, this->_database->selectValue(Timestep));

        //cutilSafeCall(cutilDeviceSynchronize());
    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::valueChanged(Settings::RecordType type) {
        cout << "Value changed: " << type << endl;
        this->_updateParams();
    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::generateParticles() {
        this->_bufferManager->bindBuffers();

        this->_bufferManager->memsetBuffers(0);

        Buffer::Memory<float4>* positionBuffer =
            new Buffer::Memory<float4>(new Buffer::Allocator(), Buffer::Host);
        Buffer::Memory<float4>* colorBuffer =
            new Buffer::Memory<float4>(new Buffer::Allocator(), Buffer::Host);

        positionBuffer->allocate(this->_numParticles);
        colorBuffer->allocate(this->_numParticles);

        float4* positions = positionBuffer->get();
        float4* colors = colorBuffer->get();

        cout << "Generating particles" << endl;

        uint resolution = ceil(pow(this->_numParticles, 1.0/3.0));

        /*for (uint x=0; x<resolution; x++) {
            for (uint y=0; y<resolution; y++) {
                for (uint z=0; z<resolution; z++) {
                    uint index = x + y*resolution + z*resolution*resolution;
                    if (index < this->_numParticles) {
                        positions[index].x = 1.0 / resolution * (x+1) - 0.5;
                        positions[index].y = 1.0 / resolution * (y+1) - 0.5;
                        positions[index].z = 1.0 / resolution * (z+1) - 0.5;
                        positions[index].w = 1.0;
                    }
                }
            }
        }*/

        GridParams params = this->_grid->getParams();

        for (uint x=0; x<resolution; x++) {
            for (uint y=0; y<resolution; y++) {
                for (uint z=0; z<resolution; z++) {
                    uint index = x + y*resolution + z*resolution*resolution;
                    if (index < this->_numParticles) {
                        positions[index].x = 2.0 / resolution * (x+1) - 1.0;
                        positions[index].y = 2.0 / resolution * (y+1) - 1.0;
                        positions[index].z = 1.0 / resolution * (z+1) - 1.0;
                        positions[index].w = 1.0;

                        if (this->_fluidParams.dynamicColoring)
                            continue;

                        colors[index].x =
                            (positions[index].x - params.min.x) / params.size.x;
                        colors[index].y =
                            (positions[index].y - params.min.y) / params.size.y;
                        colors[index].z =
                            (positions[index].z - params.min.z) / params.size.z;
                        colors[index].z = 1.0f;
                    }
                }
            }
        }

        cudaMemcpy(
            this->_particleData.position,
            positions,
                this->_numParticles * sizeof(float4),
                cudaMemcpyHostToDevice
        );

        cudaMemcpy(
            this->_particleData.color,
            colors,
            this->_numParticles * sizeof(float4),
                   cudaMemcpyHostToDevice
        );

        cutilSafeCall(cutilDeviceSynchronize());

        this->_bufferManager->unbindBuffers();

    };


    ////////////////////////////////////////////////////////////////////////////

    void Simulator::_createBuffers() {
        this->_bufferManager = new Buffer::Manager<Buffers>();

        //Buffer::Allocator* allocator = new Buffer::Allocator();

        Buffer::Vertex<float4>* color    = new Buffer::Vertex<float4>();
        Buffer::Memory<float>*  density  = new Buffer::Memory<float>();
        Buffer::Memory<float4>* force    = new Buffer::Memory<float4>();
        Buffer::Vertex<float4>* position = new Buffer::Vertex<float4>();
        Buffer::Memory<float>*  pressure = new Buffer::Memory<float>();
        Buffer::Memory<float4>* veleval  = new Buffer::Memory<float4>();
        Buffer::Memory<float4>* velocity = new Buffer::Memory<float4>();
        Buffer::Memory<float>* neighb    = new Buffer::Memory<float>();
        Buffer::Memory<int3>* cellPos   = new Buffer::Memory<int3>();

        Buffer::Vertex<float4>* sColor    = new Buffer::Vertex<float4>();
        Buffer::Memory<float>*  sDensity  = new Buffer::Memory<float>();
        Buffer::Memory<float4>* sForce    = new Buffer::Memory<float4>();
        Buffer::Vertex<float4>* sPosition = new Buffer::Vertex<float4>();
        Buffer::Memory<float>*  sPressure = new Buffer::Memory<float>();
        Buffer::Memory<float4>* sVeleval  = new Buffer::Memory<float4>();
        Buffer::Memory<float4>* sVelocity = new Buffer::Memory<float4>();
        Buffer::Memory<float>* sNeighb    = new Buffer::Memory<float>();
        Buffer::Memory<int3>* sCellPos   = new Buffer::Memory<int3>();

        this->_positionsVBO = position->getVBO();
        this->_colorsVBO = color->getVBO();

        this->_bufferManager
            ->addBuffer(Colors,           (Buffer::Abstract<void>*) color)
            ->addBuffer(Densities,        (Buffer::Abstract<void>*) density)
            ->addBuffer(Forces,           (Buffer::Abstract<void>*) force)
            ->addBuffer(Positions,        (Buffer::Abstract<void>*) position)
            ->addBuffer(Pressures,        (Buffer::Abstract<void>*) pressure)
            ->addBuffer(Velevals,         (Buffer::Abstract<void>*) veleval)
            ->addBuffer(Velocities,       (Buffer::Abstract<void>*) velocity)
            ->addBuffer(SortedColors,     (Buffer::Abstract<void>*) sColor)
            ->addBuffer(SortedDensities,  (Buffer::Abstract<void>*) sDensity)
            ->addBuffer(SortedForces,     (Buffer::Abstract<void>*) sForce)
            ->addBuffer(SortedPositions,  (Buffer::Abstract<void>*) sPosition)
            ->addBuffer(SortedPressures,  (Buffer::Abstract<void>*) sPressure)
            ->addBuffer(SortedVelevals,   (Buffer::Abstract<void>*) sVeleval)
            ->addBuffer(SortedVelocities, (Buffer::Abstract<void>*) sVelocity)
            ->addBuffer(Neighb,           (Buffer::Abstract<void>*) neighb)
            ->addBuffer(SortedNeighb,     (Buffer::Abstract<void>*) sNeighb)
            ->addBuffer(CellPos,          (Buffer::Abstract<void>*) cellPos)
            ->addBuffer(SortedCellPos,    (Buffer::Abstract<void>*) sCellPos);

        this->_bufferManager->allocateBuffers(this->_numParticles);

        size_t size = 0;
        size += color->getMemorySize();
        size += position->getMemorySize();
        size += density->getMemorySize();
        size += velocity->getMemorySize();
        size += force->getMemorySize();
        size += pressure->getMemorySize();
        std::cout << "Memory usage: " << 2 * size / 1024.0 / 1024.0 << " MB\n";

        // !!! WARNING !!!
        // without binding vertex buffers can't return valid pointer to memory
        this->_bufferManager->bindBuffers();

        this->_bufferManager->memsetBuffers(0);

        this->_particleData.color    = color->get();
        this->_particleData.density  = density->get();
        this->_particleData.force    = force->get();
        this->_particleData.position = position->get();
        this->_particleData.pressure = pressure->get();
        this->_particleData.veleval  = veleval->get();
        this->_particleData.velocity = velocity->get();
        this->_particleData.neighbours = neighb->get();
        this->_particleData.cellPos = cellPos->get();

        this->_sortedData.color    = sColor->get();
        this->_sortedData.density  = sDensity->get();
        this->_sortedData.force    = sForce->get();
        this->_sortedData.position = sPosition->get();
        this->_sortedData.pressure = sPressure->get();
        this->_sortedData.veleval  = sVeleval->get();
        this->_sortedData.velocity = sVelocity->get();
        this->_sortedData.neighbours = sNeighb->get();
        this->_sortedData.cellPos = sCellPos->get();

        this->_bufferManager->unbindBuffers();
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

    void Simulator::_updateParams() {

        // FLUID PARAMETERS
        this->_fluidParams.restDensity =
            this->_database->selectValue(RestDensity);

        this->_fluidParams.restPressure =
            this->_database->selectValue(RestPressure);

        this->_fluidParams.gasStiffness =
            this->_database->selectValue(GasStiffness);

        this->_fluidParams.viscosity =
            this->_database->selectValue(Viscosity);

        this->_fluidParams.particleMass =
            this->_database->selectValue(ParticleMass);
        this->_fluidParams.particleRestDistance =
            this->_database->selectValue(ParticleRestDistance);

        this->_fluidParams.boundaryDistance =
            this->_database->selectValue(BoundaryDistance);
        this->_fluidParams.boundaryStiffness =
            this->_database->selectValue(BoundaryStiffness);
        this->_fluidParams.boundaryDampening =
            this->_database->selectValue(BoundaryDampening);

        this->_fluidParams.velocityLimit =
            this->_database->selectValue(VelocityLimit);

        this->_fluidParams.scaleToSimulation =
            this->_database->selectValue(SimulationScale);

        this->_fluidParams.smoothingLength =
            this->_database->selectValue(SmootingLength);

        this->_fluidParams.frictionKinetic =
            this->_database->selectValue(KineticFriction);

        this->_fluidParams.frictionStaticLimit =
            this->_database->selectValue(StaticFrictionLimit);

        this->_fluidParams.dynamicColoring =
            this->_database->selectValue(DynamicColoring);

        cout << "DynamicColoring: " << this->_fluidParams.dynamicColoring << endl;

        // PRECALCULATED PARAMETERS
        float smoothLen = this->_fluidParams.smoothingLength;

        this->_precalcParams.smoothLenSq = pow(smoothLen, 2);

        this->_precalcParams.poly6Coeff =
            Kernels::Poly6::getConstant(smoothLen);

        this->_precalcParams.spikyGradCoeff =
            Kernels::Spiky::getGradientConstant(smoothLen);

        this->_precalcParams.viscosityLapCoeff =
            Kernels::Viscosity::getLaplacianConstant(smoothLen);

        this->_precalcParams.pressurePrecalc =
            this->_precalcParams.spikyGradCoeff;

        this->_precalcParams.viscosityPrecalc =
            this->_fluidParams.viscosity *
            this->_precalcParams.viscosityLapCoeff;

        // DEBUG

        cout
            << "SmoothLen: "
            << this->_precalcParams.smoothLenSq
            << endl
            << "Poly6Coeff: "
            << this->_precalcParams.poly6Coeff
            << endl
            << "SpikyGradCoeff: "
            << this->_precalcParams.spikyGradCoeff
            << endl
            << "ViscosityLapCoeff: "
            << this->_precalcParams.viscosityLapCoeff
            << endl
            << "PressurePrecalc: "
            << this->_precalcParams.pressurePrecalc
            << endl
            << "ViscosityPrecalc: "
            << this->_precalcParams.viscosityPrecalc
            << endl;


        // Copy parameters to GPU's constant memory
        // declarations of symbols are in sph_kernel.cu
        CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(
                cudaFluidParams,
                &this->_fluidParams,
                sizeof(FluidParams)
            )
        );

        CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(
                cudaPrecalcParams,
                &this->_precalcParams,
                sizeof(PrecalcParams)
            )
        );

        CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(
                cudaGridParams,
                &this->_gridParams,
                sizeof(GridParams)
            )
        );

        CUDA_SAFE_CALL(cudaThreadSynchronize());
    }

    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////


};

#endif // __SPH_SIMULATOR_CU__