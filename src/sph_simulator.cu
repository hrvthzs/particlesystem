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
        cudaEventDestroy(this->_startFPS);
        cudaEventDestroy(this->_stopFPS);

        delete this->_bufferManager;
        delete this->_grid;
        delete this->_marchingRenderer;
    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::init(uint numParticles) {
        this->_numParticles = numParticles;
        this->_colorGradient = Colors::HSVBlueToRed;
        this->_colorSource = Colors::Velocity;
        this->_lastAnimatedParticle = 0;
        this->_numAnimatedParticles = 5;
        this->_animationForce = 5;
        this->_animation = true;
        this->_animChangeAxis = true;

        // DATABASE
        this->_database
            ->insert(ParticleNumber, "Particles", this->_numParticles)
            ->insert(GridSize, "Grid size", 3.0f)
            ->insert(Timestep, "Timestep", 0.0f, 1.0f, 0.01f)
            ->insert(RestDensity, "Rest density", 0.0f, 10000.0f, 6000.0f) // step 1000
            ->insert(RestPressure, "Rest pressure", 0.0f, 10000.0f, 1000.0f)
            ->insert(GasStiffness, "Gas Stiffness", 0.001f, 10.0f, 1.0f)
            ->insert(Viscosity, "Viscosity", 0.0f, 100.0f, 100.0f) // step 0.1
            ->insert(BoundaryDampening, "Bound. damp.", 0.0f, 10000.0f, 0.0f) // 256
            ->insert(BoundaryStiffness, "Bound. stiff.", 0.0f, 100000.0f, 10000.0f)
            ->insert(VelocityLimit, "Veloc. limit", 0.0f, 10000.0f, 500.0f)
            ->insert(SimulationScale, "Sim. scale", 0.0f, 1.0f, 1.0f)
            ->insert(KineticFriction, "Kinet. fric.", 0.0f, 10000.0f, 0.0f)
            ->insert(StaticFrictionLimit, "Stat. Fric. Lim.", 0.0f, 10000.0f, 0.0f)
            ->insert(DynamicColoring, "DynamicColoring", 1.0f, true);

        float particleMass = 1.0 * pow(10, -7);
        float particleRestDist =
            0.01f *
            pow(
                particleMass / this->_database->selectValue(RestDensity),
                1.0f/3.0f
            );

        float cellSize = 0.1;
        float boundaryDist = cellSize;
        float smoothingLength = cellSize;

        this->_database
            ->insert(ParticleMass, "Particle mass", particleMass)
            ->insert(ParticleRestDistance, "Part. rest dist.", particleRestDist)
            ->insert(BoundaryDistance, "Bound. dist.", boundaryDist)
            ->insert(SmootingLength, "Smooth. len.", smoothingLength)
            ->insert(CellSize, "Cell size", cellSize)
            ->insert(Interpolation, "Interpolation", 1)
            ->insert(ColorSource, "Color source", this->_colorSource)
            ->insert(ColorGradient, "Color gradient", this->_colorGradient)
            ->insert(AnimPartNum, "Anim. p. num.", this->_numAnimatedParticles)
            ->insert(AnimPartForce, "Anim. part. force", this->_animationForce)
            ->insert(AnimChangeAxis, "Anim. change axis", this->_animChangeAxis);

        this->_database->print();

        this->_grid = new Grid::Uniform();
        this->_grid->allocate(
            this->_numParticles,
            cellSize,
            this->_database->selectValue(GridSize)
        );
        this->_grid->printParams();
        this->_gridParams = this->_grid->getParams();

        this->_updateParams();
        this->_createBuffers();

        this->_marchingRenderer = new Marching::Renderer(this->_grid);
        this->_marchingRenderer
            ->setInterpolation(this->_database->selectValue(Interpolation));

        cudaEventCreate(&this->_startFPS);
        cudaEventCreate(&this->_stopFPS);

        cudaEventRecord(this->_startFPS, 0);
        cudaEventRecord(this->_stopFPS, 0);

        this->_iterFPS = 0;
        this->_sumTimeFPS = 0;
    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::stop() {
        this->_bufferManager->freeBuffers();
        this->_marchingRenderer->freeBuffers();
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
        this->_marchingRenderer->bindBuffers();
    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::unbindBuffers() {
        this->_bufferManager->unbindBuffers();
        this->_marchingRenderer->unbindBuffers();
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

    void Simulator::_integrate(float deltaTime, float3 gravity) {
        uint minBlockSize, numBlocks, numThreads;

        minBlockSize = 416;
        Utils::computeGridSize(
            this->_numParticles, minBlockSize, numBlocks, numThreads
        );

        Kernel::integrate<Data><<<numBlocks, numThreads>>>(
            this->_numParticles,
            deltaTime,
            gravity,
            this->_particleData,
            this->_sortedData,
            this->_grid->getData(),
            this->_colorGradient,
            this->_colorSource
        );
    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::_animate() {

        uint num = pow(this->_numAnimatedParticles, 2);

        uint minBlockSize, numBlocks, numThreads;

        minBlockSize = this->_numAnimatedParticles;
        Utils::computeGridSize(num, minBlockSize, numBlocks, numThreads);

        Kernel::animate<<<numBlocks, numThreads>>>(
            this->_numParticles,
            this->_lastAnimatedParticle,
            this->_animationForce,
            this->_animChangeAxis,
            this->_particleData,
            this->_grid->getData()
        );

        this->_lastAnimatedParticle += num;

        if (this->_lastAnimatedParticle >= this->_numParticles) {
            this->_lastAnimatedParticle = 0;
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::update(bool intergrate, float x, float y, float z) {
        float3 gravity = make_float3(x, y, z);

        this->_grid->hash((float4*) this->getPositions());
        this->_grid->sort();
        this->_orderData();

        if (this->_renderMode == Particles::RenderMarching) {
            this->_marchingRenderer->render();
        }

        if (intergrate) {
            this->_step1();
            this->_step2();

            this->_integrate(
                this->_database->selectValue(Timestep),
                gravity
            );

            if (this->_animation) {
                this->_animate();
            }
        }

        cutilSafeCall(cutilDeviceSynchronize());

        float time;

        cudaEventRecord(this->_stopFPS, 0);
        cudaEventSynchronize(this->_stopFPS);
        cudaEventElapsedTime(&time, this->_startFPS, this->_stopFPS);

        this->_iterFPS++;
        this->_sumTimeFPS += 1000 / time;

        //cout << "Time for the kernel: " << this->_sumTimeFPS / this->_iterFPS << " fps" << endl;
        cudaEventRecord(this->_startFPS, 0);

    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::valueChanged(Settings::RecordType type) {
        switch (type) {
            case AnimChangeAxis:
                this->_animChangeAxis =
                    this->_database->selectValue(AnimChangeAxis);
                break;
            case AnimPartForce:
                this->_animationForce =
                    this->_database->selectValue(AnimPartForce);
                break;
            case AnimPartNum:
                this->_numAnimatedParticles =
                    this->_database->selectValue(AnimPartNum);
                break;
            case ColorGradient:
                this->_colorGradient =
                    (Colors::Gradient) this->_database->selectValue(ColorGradient);
                break;
            case ColorSource:
                this->_colorSource =
                    (Colors::Source) this->_database->selectValue(ColorSource);
                break;
            case Interpolation:
                this->_marchingRenderer
                ->setInterpolation(
                    this->_database->selectValue(Interpolation)
                );
                break;
            default:
                this->_updateParams();
                break;
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::generateParticles() {
        this->_bufferManager->bindBuffers();

        this->_bufferManager->memsetBuffers(0);

        Buffer::Memory<float4>* positionBuffer =
            new Buffer::Memory<float4>(Buffer::Host);
        Buffer::Memory<float4>* colorBuffer =
            new Buffer::Memory<float4>(Buffer::Host);

        positionBuffer->allocate(this->_numParticles);
        colorBuffer->allocate(this->_numParticles);

        float4* positions = positionBuffer->get();
        float4* colors = colorBuffer->get();

        cout << "Generating particles" << endl;

        uint resolution = ceil(pow(this->_numParticles, 1.0/3.0));

        GridParams params = this->_grid->getParams();
        float centering = this->_database->selectValue(CellSize) / 2.0f;

        for (uint x=0; x<resolution; x++) {
            for (uint y=0; y<resolution; y++) {
                for (uint z=0; z<resolution; z++) {
                    uint index = x + y*resolution + z*resolution*resolution;
                    if (index < this->_numParticles) {
                        positions[index].x = 1.0 / resolution * (x+1) - 1.0;
                        positions[index].y = 2.0 / resolution * (y+1) - 1.5;
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

    uint Simulator::getNumVertices() {
        if (this->_renderMode == Particles::RenderPoints) {
            return this->_numParticles;
        } else {
            return this->_marchingRenderer->getNumVertices();
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    void Simulator::setRenderMode(int mode) {
        this->_renderMode = mode;

        if(mode == Particles::RenderPoints) {
            Buffer::Vertex<float4>* positions;
            Buffer::Vertex<float4>* colors;

            positions =
                (Buffer::Vertex<float4>*) this->_bufferManager->get(Positions);

            colors =
                (Buffer::Vertex<float4>*) this->_bufferManager->get(Colors);

            this->_positionsVBO = positions->getVBO();
            this->_colorsVBO = colors->getVBO();
            this->_normalsVBO = 0;
        } else {
            this->_positionsVBO = this->_marchingRenderer->getPositionsVBO();
            this->_normalsVBO = this->_marchingRenderer->getNormalsVBO();
            this->_colorsVBO = 0;
        }

    }

    ////////////////////////////////////////////////////////////////////////////

    int Simulator::getRenderMode() {
        return this->_renderMode;
    }

    ////////////////////////////////////////////////////////////////////////////

    Particles::GridMinMax Simulator::getGridMinMax() {
        Particles::GridMinMax grid;
        grid.min = this->_gridParams.min;
        grid.max = this->_gridParams.max;
        return grid;
    }

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
        // debug
        Buffer::Memory<float>*  neighb    = new Buffer::Memory<float>();
        Buffer::Memory<int3>*   cellPos   = new Buffer::Memory<int3>();
        Buffer::Memory<float4>* viscosity = new Buffer::Memory<float4>();


        Buffer::Vertex<float4>* sColor    = new Buffer::Vertex<float4>();
        Buffer::Memory<float>*  sDensity  = new Buffer::Memory<float>();
        Buffer::Memory<float4>* sForce    = new Buffer::Memory<float4>();
        Buffer::Vertex<float4>* sPosition = new Buffer::Vertex<float4>();
        Buffer::Memory<float>*  sPressure = new Buffer::Memory<float>();
        Buffer::Memory<float4>* sVeleval  = new Buffer::Memory<float4>();
        Buffer::Memory<float4>* sVelocity = new Buffer::Memory<float4>();
        //debug
        Buffer::Memory<float>* sNeighb    = new Buffer::Memory<float>();
        Buffer::Memory<int3>* sCellPos   = new Buffer::Memory<int3>();
        Buffer::Memory<float4>* sViscosity = new Buffer::Memory<float4>();

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
            ->addBuffer(SortedCellPos,    (Buffer::Abstract<void>*) sCellPos)
            ->addBuffer(Viscosities,       (Buffer::Abstract<void>*) viscosity)
            ->addBuffer(SortedViscosities, (Buffer::Abstract<void>*) sViscosity);

        this->_bufferManager->allocateBuffers(this->_numParticles);

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
        this->_particleData.viscosity = viscosity->get();

        this->_sortedData.color    = sColor->get();
        this->_sortedData.density  = sDensity->get();
        this->_sortedData.force    = sForce->get();
        this->_sortedData.position = sPosition->get();
        this->_sortedData.pressure = sPressure->get();
        this->_sortedData.veleval  = sVeleval->get();
        this->_sortedData.velocity = sVelocity->get();
        this->_sortedData.neighbours = sNeighb->get();
        this->_sortedData.cellPos = sCellPos->get();
        this->_sortedData.viscosity = sViscosity->get();

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

        this->_precalcParams.smoothLenSq = smoothLen * smoothLen;

        this->_precalcParams.poly6Coeff =
            Kernels::Poly6::getConstant(smoothLen);

        this->_precalcParams.spikyGradCoeff =
            Kernels::Spiky::getGradientConstant(smoothLen);

        this->_precalcParams.viscosityLapCoeff =
            Kernels::Viscosity::getLaplacianConstant(smoothLen);

        this->_precalcParams.pressurePrecalc =
            this->_precalcParams.spikyGradCoeff;

        this->_precalcParams.viscosityPrecalc =
            -this->_fluidParams.viscosity *
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