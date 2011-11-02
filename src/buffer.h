#ifndef __BUFFER_H__
#define __BUFFER_H__

namespace Buffer {

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

    enum MemoryLocation {
        host,
        device,
        hostPinned,
    };

    typedef enum Errors error_t;
    typedef enum MemoryLocation memory_t;

};

#endif // __BUFFER_H__