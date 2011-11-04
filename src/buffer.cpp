#include "buffer.h"

namespace Buffer {

    error_t parseCudaError(cudaError_t cudaError) {
        error_t error;

        switch (cudaError) {
            case cudaSuccess:
                error = Success;
                break;
            case cudaErrorInvalidValue:
                error = InvalidValue;
                break;
            case cudaErrorInitializationError:
                error = InitializationError;
            case cudaErrorInvalidDevicePointer:
                error = InvalidPointerError;
                break;
            case cudaErrorInvalidTexture:
                error = InvalidTexture;
                break;
            default:
                error = UnknownError;
                break;
        }

        return error;
    }

}