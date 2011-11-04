#ifndef __BUFFER_H__
#define __BUFFER_H__

#include <cutil_inline.h>

namespace Buffer {

    /**
     * Error codes
     */
    enum Errors {
        Success,
        MemoryAllocationError,
        UnknownMemoryTypeError,
        InvalidPointerError,
        InvalidTexture,
        InvalidValue,
        InitializationError,
        UnknownError
    };

    /**
     * Buffer memory locations
     */
    enum MemoryLocation {
        Host,
        Device,
        HostPinned,
        UnknownMemory,
    };

    typedef enum Errors error_t;
    typedef enum MemoryLocation memory_t;

    /**
     * Parse cuda error codes
     * @param cudaError cuda error code
     * @return code from enum Errors
     */
    error_t parseCudaError(cudaError_t cudaError);

};

#endif // __BUFFER_H__