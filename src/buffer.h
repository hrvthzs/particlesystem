#ifndef __BUFFER_H__
#define __BUFFER_H__

#include <cutil_inline.h>

namespace Buffer {

    /**
     * Error codes
     */
    enum Errors {
        success,
        memoryAllocationError,
        unknownMemoryTypeError,
        invalidPointerError,
        invalidTexture,
        invalidValue,
        initializationError,
        unknownError
    };

    /**
     * Buffer memory locations
     */
    enum MemoryLocation {
        host,
        device,
        hostPinned,
        unknownMemory,
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