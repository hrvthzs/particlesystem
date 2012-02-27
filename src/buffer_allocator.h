#ifndef __BUFFER_ALLOCATOR_H__
#define __BUFFER_ALLOCATOR_H__

#include <cutil.h>
#include <cutil_math.h>
#include <map>

#include "buffer.h"

namespace Buffer {

    typedef std::map<void*, size_t> MemoryMap;

    class Allocator {

        public:
            /**
             * Get an instance of class
             */
            static Allocator* getInstance();

            /**
             * Destructor
             */
            virtual ~Allocator();

            /**
             * Allocate memory with specified size and location
             * @param ptr pointer to memory
             * @param size memory size
             * @param memory memory location
             * @return error type
             */
            error_t allocate(void **ptr, size_t size, memory_t memory);

            /**
             * Free allocated memory
             * @param ptr pointer to memory
             * @param memory memory location
             * @return error type
             */
            error_t free(void **ptr, memory_t memory);

            /**
             * Get memory usage of requested memory type in bytes
             * @param memory type of memory
             */
            size_t getUsage(memory_t memory);

        private:
            /**
             * Constructor
             */
            Allocator();

            // static instance of class
            static Allocator* _instance;

            // allocated host memory size
            size_t _hAllocatedMemory;
            // allocated device memory size
            size_t _dAllocatedMemory;

            // map with key host memory pointer and value memory size
            MemoryMap _hMemoryMap;

            // map with key device memory pointer and value memory size
            MemoryMap _dMemoryMap;

            /**
             * Allocate memory with specified size at host
             * @param ptr pointer to memory
             * @param size memory size
             * @return error type
             */
            error_t _allocateHost(void **ptr, size_t size);

            /**
             * Allocate memory with specified size at device
             * @param ptr pointer to memory
             * @param size memory size
             * @return error type
             */
            error_t _allocateDevice(void **ptr, size_t size);

            /**
             * Free allocated memory at host
             * @param ptr pointer to memory
             * @return error type
             */
            error_t _freeHost(void **ptr);

            /**
             * Free allocated memory at device
             * @param ptr pointer to memory
             * @return error type
             */
            error_t _freeDevice(void **ptr);
    };
};

#endif // __BUFFER_ALLOCATOR_H__