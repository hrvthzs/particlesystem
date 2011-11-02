#include "buffer.h"

namespace Buffer {

    error_t parseCudaError(cudaError_t cudaError) {
        error_t error;

        switch (cudaError) {
            case cudaSuccess:
                error = success;
                break;
            case cudaErrorInvalidValue:
                error = invalidValue;
                break;
            case cudaErrorInitializationError:
                error = initializationError;
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

}