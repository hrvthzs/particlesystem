#ifndef __BUFFER_MEMORY_CU__
#define __BUFFER_MEMORY_CU__

#include <iostream>

namespace Buffer {

    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    Memory<T>::Memory(Allocator* allocator, memory_t memory) {
        this->_init(allocator, memory);
    }

    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    Memory<T>::Memory() {
        this->_init(new Allocator(), Device);
    }

    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    Memory<T>::~Memory() {
        this->free();
    }

    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    error_t Memory<T>::bind() {

        error_t error = Success;

        // TODO texture memory and fetch in kernels
        /*if (this->_memory == Device) {
            cudaError_t cudaError = cudaBindTexture(
                0,
                this->_textureRef,
                this->_memoryPtr,
                this->_size*sizeof(T)
            );

            error = parseCudaError(cudaError);
        }

        if (error == Success) {
            this->_bound = true;
        }*/

        return error;
    }

    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    void Memory<T>::unbind() {
        /*if (this->_bound) {
            if (this->_memory == Device) {
                cudaUnbindTexture(this->_textureRef);
            }
            this->_bound = false;
        }*/
    }

    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    error_t Memory<T>::memset(int value) {
        size_t size = this->getMemorySize();
        error_t error = Success;
        cudaError_t cudaError;

        if (size > 0) {
            switch (this->_memory) {
                case Host:
                case HostPinned:
                    ::memset(this->_memoryPtr, value, size);
                    error = Success;
                    break;
                case Device:
                    cudaError = cudaMemset(this->_memoryPtr, value, size);
                    error = parseCudaError(cudaError);
                    break;
                default:
                    error = UnknownMemoryTypeError;
            }
        }

        return error;
    }

    ////////////////////////////////////////////////////////////////////////////

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
        if (error == Success) {
            this->_size = size;
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    void Memory<T>::copyTo(void* dst, memory_t dstMem) {
        this->_copy(this->get(), dst, this->_memory, dstMem);
    }

    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    void Memory<T>::copyFrom(void* src, memory_t srcMem) {
        this->_copy(src, this->get(), srcMem, this->_memory);
    }

    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    void Memory<T>::free() {
        if (this->_size > 0 ) {
            error_t error =
                this->_allocator->free(
                    (void**) &this->_memoryPtr,
                    this->_memory
                );

            // TODO handle error
            if (error == Success) {
                this->_size = 0;
                this->_memoryPtr = NULL;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    void Memory<T>::_init(Allocator* allocator, memory_t memory) {
        this->_allocator = allocator;
        this->_memory = memory;
        this->_bound = false;
        this->_size = 0;
        this->_memoryPtr = NULL;
    }

    ////////////////////////////////////////////////////////////////////////////

    template<class T>
    void Memory<T>::_copy(
        void* src,
        void* dst,
        memory_t srcMem,
        memory_t dstMem
    ) {

        enum cudaMemcpyKind kind;

        if (srcMem == HostPinned) {
            srcMem = Host;
        }

        if (dstMem == HostPinned) {
            dstMem = Host;
        }

        if (srcMem == Device && dstMem == Device) {
            kind = cudaMemcpyDeviceToDevice;
        } else if (srcMem == Device && dstMem == Host) {
            kind = cudaMemcpyDeviceToHost;
        } else if (srcMem == Host && dstMem == Device) {
            kind = cudaMemcpyHostToDevice;
        } else if (srcMem == Host && dstMem == Host) {
            kind = cudaMemcpyHostToHost;
        } else {
            return;
        }

        cutilSafeCall(
            cudaMemcpy(dst, src, this->_size * sizeof(T), kind)
        );
    }

    ////////////////////////////////////////////////////////////////////////////

}

#endif // __BUFFER_MEMORY_BUFFER_ĆU__