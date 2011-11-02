#include "buffer_allocator.h"

#include <stdlib.h>

namespace Buffer {

    ///////////////////////////////////////////////////////////////////////////

    Allocator::Allocator() {
        this->_hAllocatedMemory = 0;
        this->_dAllocatedMemory = 0;
    }

    ///////////////////////////////////////////////////////////////////////////

    Allocator::~Allocator() {

    }

    ///////////////////////////////////////////////////////////////////////////

    error_t Allocator::allocate(void** ptr, size_t size, memory_t memory) {
        error_t error;

        switch (memory) {
            case host:
                error = this->_allocateHost(ptr, size);
                break;
            case device:
                error = this->_allocateDevice(ptr, size);
                break;
            default:
                error = unknownMemoryTypeError;
                break;
        }

        return error;
    }

    ///////////////////////////////////////////////////////////////////////////

    error_t Allocator::free(void **ptr, memory_t memory) {
        error_t error;

        switch (memory) {
            case host:
                error = this->_freeHost(ptr);
                break;
            case device:
                error = this->_freeDevice(ptr);
                break;
            default:
                error = unknownMemoryTypeError;
        }

        return error;
    }

    ///////////////////////////////////////////////////////////////////////////

    error_t Allocator::_allocateHost(void **ptr, size_t size) {
        error_t error;

        *ptr = malloc(size);

        if (*ptr == NULL) {
            error = memoryAllocationError;
        } else {
            this->_hAllocatedMemory += size;
            this->_hMemoryMap[*ptr] = size;
            error = success;
        }

        return error;
    }

    ///////////////////////////////////////////////////////////////////////////

    error_t Allocator::_allocateDevice(void **ptr, size_t size) {
        error_t error;

        cudaError_t cudaError = cudaMalloc(ptr, size);

        if (cudaError == cudaErrorMemoryAllocation) {
            error = memoryAllocationError;
        } else {
            this->_dAllocatedMemory += size;
            this->_dMemoryMap[*ptr] = size;
            error = success;
        }

        return error;
    }

    ///////////////////////////////////////////////////////////////////////////

    error_t Allocator::_freeHost (void **ptr) {
        error_t error;

        if (*ptr != NULL) {
            // C function from stdlib, not class method
            ::free(*ptr);
            this->_hAllocatedMemory -= this->_hMemoryMap[*ptr];
            this->_hMemoryMap[*ptr] = 0;
            *ptr = NULL;
            error = success;
        } else {
            error = invalidPointerError;
        }

        return error;
    }

    ///////////////////////////////////////////////////////////////////////////

    error_t Allocator::_freeDevice (void **ptr) {
        error_t error;

        cudaError_t cudaError = cudaFree(*ptr);

        error = parseCudaError(cudaError);

        if (error == success) {
            this->_dAllocatedMemory -= this->_dMemoryMap[*ptr];
            this->_dMemoryMap[*ptr] = 0;
            *ptr = NULL;
        }

        return error;
    }

    ///////////////////////////////////////////////////////////////////////////

}