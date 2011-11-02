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
            Allocator();
            virtual ~Allocator();

            error_t allocate(void **ptr, size_t size, memory_t memory);
            error_t free(void **ptr, memory_t memory);

        private:
            size_t _hAllocatedMemory;
            size_t _dAllocatedMemory;


            MemoryMap _hMemoryMap;
            MemoryMap _dMemoryMap;

            error_t _allocateHost(void **ptr, size_t size);
            error_t _allocateDevice(void **ptr, size_t size);

            error_t _freeHost(void **ptr);
            error_t _freeDevice(void **ptr);
    };
};

#endif // __BUFFER_ALLOCATOR_H__