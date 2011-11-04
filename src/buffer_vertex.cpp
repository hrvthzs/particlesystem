#ifndef __BUFFER_VERTEX_CPP__
#define __BUFFER_VERTEX_CPP__

#include "buffer_vertex.h"
#include "buffer_abstract_buffer.cpp"
#include "kernel.cuh"

#include <iostream>

namespace Buffer {

    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    Vertex<T>::Vertex() {
        this->_bound = false;
        this->_size = 0;
        this->_memoryPtr = NULL;
    }

    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    Vertex<T>::~Vertex() {
        this->free();
    }

    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    error_t Vertex<T>::bind() {

        error_t error = success;
        size_t bytes;

        // TODO error handling
        cutilSafeCall(cudaGraphicsMapResources(1, &this->_cudaVBOResource, 0));

        cutilSafeCall(
            cudaGraphicsResourceGetMappedPointer(
                (void **)&this->_memoryPtr,
                &bytes,
                this->_cudaVBOResource
            )
        );

        this->_bound = true;

        return error;
    }

    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    void Vertex<T>::unbind() {
        if (this->_bound) {
            cutilSafeCall(
                cudaGraphicsUnmapResources(1, &this->_cudaVBOResource, 0)
            );
            this->_bound = false;
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    error_t Vertex<T>::memset(int) {
        return success;;
    }

    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    void Vertex<T>::allocate(size_t size) {

        if (this->_size > 0 ) {
            if (size == this->_size) {
                return;
            } else {
                this->free();
            }
        }

        this->_size = size;

        // calculate size of required memory
        size_t allocationSize = size * sizeof(T);

        // create buffer object
        glGenBuffers(1, &this->_VBO);
        glBindBuffer(GL_ARRAY_BUFFER, this->_VBO);
        glBufferData(GL_ARRAY_BUFFER, allocationSize, 0, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // TODO error handling
        cutilSafeCall(
            cudaGraphicsGLRegisterBuffer(
                &this->_cudaVBOResource,
                this->_VBO,
                cudaGraphicsMapFlagsNone
            )
        );

    }

    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    void Vertex<T>::free() {
        this->unbind();

        cudaGraphicsUnregisterResource(this->_cudaVBOResource);

        glBindBuffer(1, this->_VBO);
        glDeleteBuffers(1, &this->_VBO);
        this->_VBO = 0;
    }

    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    GLuint Vertex<T>::getVBO() {
        return this->_VBO;
    }

    ////////////////////////////////////////////////////////////////////////////
}

#endif // __VERTEX_BUFFER_CPP__