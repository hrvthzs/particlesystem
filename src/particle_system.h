/**
 * Particle system
 *
 * @author Zsotl Horváth
 * @date   13.10.2011
 */


#ifndef _PARTICLE_SYSTEM_H_
#define _PARTICLE_SYSTEM_H_

#include <GL/glew.h>

#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <vector_types.h>
#include <memory.h>

typedef unsigned int uint;

struct ParticleSystemParameters {
    float particleRadius;
    float boundaryDamping;
    float globalDamping;
    float3 gravity;
    float3 gridOrigin;
    float3 cellSize;
    uint3 gridSize;
};

class ParticleSystem {
public:
    ParticleSystem(uint particles, uint3 gridSize);
    virtual ~ParticleSystem();

    uint getCount() const;
    float getRadius() const;
    float getGravity() const;

    void setRadius(float r);
    void setGravity(float g);

    void update(float deltaTime);

    void * getCudaPositionsVBO() const;
    void * getCudaColorsVBO() const;

    void setPositionsVBO(GLuint vbo);
    void setCudaPositionsVBOResource(struct cudaGraphicsResource * resource);

    GLuint getPositionsVBO() const;
    struct cudaGraphicsResource* getCudaPositionsVBOResource() const;

protected:

    uint _count;
    uint3 _gridSize;

    float* _hPositions;
    float* _hVelocities;

    // GPU data
    float* _cudaVelocities;

    uint* _cudaGridHash;
    uint* _cudaGridIndex;
    uint* _cudaCellStart;
    uint* _cudaCellEnd;

    float* _hSortedPositions;
    float* _hSortedVelocities;

    float* _cudaPositionsVBO;        // these are the CUDA deviceMem Pos
    float* _cudaColorsVBO;      // these are the CUDA deviceMem Color

    ParticleSystemParameters _paramters;

    GLuint _positionsVBO;

    struct cudaGraphicsResource* _cudaPositionsVBOResource; // handles OpenGL-CUDA exchange
    //struct cudaGraphicsResource* _cudaColorsVBOResource; // handles OpenGL-CUDA exchange

    void _initialize();
    void _allocateCudaArray(void **pointer, size_t size);
    void _freeCudaArray(void *pointer);
    void _createVBO(GLuint *vbo);
    void _deleteVBO(GLuint *vbo);
    void _cudaMapVBO(GLuint vbo, struct cudaGraphicsResource **resource, unsigned int flags);
    void _cudaUnmapVBO(struct cudaGraphicsResource *resource);
    void * _mapGLBufferObject(struct cudaGraphicsResource **resource);
    void _unmapGLBufferObject(struct cudaGraphicsResource *resource);

};

#endif  /* _PARTICLE_SYSTEM_H_ */