#ifndef __BUFFER_MEMORY_CU__
#define __BUFFER_MEMORY_CU__

#include "buffer_memory.cuh"
#include "buffer_abstract_buffer.cpp"

#include <iostream>

namespace Buffer {

    ///////////////////////////////////////////////////////////////////////////

    template<class T>
    Memory<T>::Memory(Allocator* allocator, memory_t memory) {
        this->_allocator = allocator;
        this->_memory = memory;
        this->_bound = false;
        this->_size = 0;
        this->_memoryPtr = NULL;
    }

    ///////////////////////////////////////////////////////////////////////////

    template<class T>
    Memory<T>::Memory() : Memory(new Allocator(), device){

    }

    ///////////////////////////////////////////////////////////////////////////

    template<class T>
    Memory<T>::~Memory() {
        this->free();
    }

    ///////////////////////////////////////////////////////////////////////////

    template<class T>
    error_t Memory<T>::bind() {

        error_t error = success;

        if (this->_memory == device) {
            cudaError_t cudaError = cudaBindTexture(
                0,
                this->_textureRef,
                this->_memoryPtr,
                this->_size*sizeof(T)
            );

            error = parseCudaError(cudaError);

        }

        if (error == success) {
            this->_bound = true;
        }

        return error;
    }

    ///////////////////////////////////////////////////////////////////////////

    template<class T>
    void Memory<T>::unbind() {
        if (this->_bound) {
            if (this->_memory == device) {
                cudaUnbindTexture(this->_textureRef);
            }
            this->_bound = false;
        }
    }

    ///////////////////////////////////////////////////////////////////////////

    template<class T>
    error_t Memory<T>::memset(int value) {
        size_t size = this->getMemorySize();
        error_t error = success;
        cudaError_t cudaError;

        if (size > 0) {
            switch (this->_memory) {
                case host:
                case hostPinned:
                    ::memset(this->_memoryPtr, value, size);
                    error = success;
                    break;
                case device:
                    cudaError = cudaMemset(this->_memoryPtr, value, size);
                    error = parseCudaError(cudaError);
                    break;
                default:
                    error = unknownMemoryTypeError;
            }
        }

        return error;
    }

    ///////////////////////////////////////////////////////////////////////////

    template<class T>
    void Memory<T>::allocate(size_t size) {

        if (this->_size > 0 ) {
            if (size == this->_size) {
                return;
            } else {
                this->free();
            }
        }

        // calculate size of required memory
        size_t allocationSize = size * sizeof(T);

        error_t error =
            this->_allocator->allocate(
                (void **) &this->_memoryPtr,
                allocationSize,
                this->_memory
            );


        // TODO handle error
        if (error == success) {
            this->_size = size;
        }
    }

    ///////////////////////////////////////////////////////////////////////////

    template<class T>
    void Memory<T>::free() {
        if (this->_size > 0 ) {
            error_t error =
                this->_allocator->free(
                    (void**) &this->_memoryPtr,
                    this->_memory
                );

            // TODO handle error
            if (error == success) {
                this->_size = 0;
                this->_memoryPtr = NULL;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
}

#endif // __BUFFER_MEMORY_BUFFER_ĆU__