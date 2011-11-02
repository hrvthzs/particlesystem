#include "buffer_buffer.cuh"

#include <iostream>

namespace Buffer {

    template<class T> Buffer<T>::Buffer(Allocator* allocator, memory_t memory) {
        this->_allocator = allocator;
        this->_memory = memory;
        this->_bound = false;
        this->_size = 0;
        this->_memoryPtr = NULL;
    }

    template<class T> Buffer<T>::~Buffer() {
        this->free();
    }

    template<class T> error_t Buffer<T>::bind() {
        cudaError_t cudaError = cudaBindTexture(
            0,
            this->_textureRef,
            this->_memoryPtr,
            this->_size*sizeof(T)
        );

        error_t error;

        switch (cudaError) {
            case cudaSuccess:
                error = success;
                this->_bound = true;
                break;
            case cudaErrorInvalidValue:
                error = invalidValue;
                break;
            case cudaErrorInvalidDevicePointer:
                error = invalidPointerError;
                break;
            case cudaErrorInvalidTexture:
                error = invalidTexture;
                break;
            default:
                error = unknownError;
                break;
        }

        return error;
    }

    template<class T> void Buffer<T>::unbind() {
        if (this->_bound) {
            cudaUnbindTexture(this->_textureRef);
        }

    }

    template<class T> void Buffer<T>::memset(T value) {

    }

    template<class T> void Buffer<T>::allocate(size_t size) {

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
                this->_memoryPtr,
                allocationSize,
                this->_memory
            );


        // TODO handle error
        if (error == success) {
            this->_size = size;
        }
    }

    template<class T> void Buffer<T>::free() {
        if (this->_size > 0 ) {
            error_t error =
                this->_allocator->free(this->_memoryPtr, this->_memory);

            // TODO handle error
            if (error == success) {
                this->_size = 0;
                this->_memoryPtr = NULL;
            }
        }
    }

    template<class T> T* Buffer<T>::get() {
        return this->_memoryPtr;
    }

    template<class T> size_t Buffer<T>::getSize() {
        return this->_size;
    }

    template<class T> size_t Buffer<T>::getMemorySize() {
        return this->_size * sizeof(T);
    }
}