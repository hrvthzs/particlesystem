#ifndef __BUFFER_MEMORY_CUH__
#define __BUFFER_MEMORY_CUH__

#include "buffer.h"
#include "buffer_allocator.h"
#include "buffer_abstract.h"

namespace Buffer {

    /**
     * Memory Buffer
     *
     * !!! Important !!!
     * For template classes definition must be included too
     */
    template <class T>
    class Memory : public Abstract<T> {

        public:

            /**
             * Constructor
             *
             * @param allocator memory allocator instance
             * @param memory memory location @see Buffer::MemoryLocation
             */
            Memory(Allocator* allocator, memory_t memory);

            /**
             * Constructor
             */
            Memory();

            /**
             * Destructor
             */
            virtual ~Memory();

            /**
             * Bind buffer to texture
             * @return type of error
             */
            virtual error_t bind();

            /**
             * Unbind memory buffer from texture
             */
            virtual void unbind();

            /**
             * Initialize memory with specified value
             * @param value init value
             */
            virtual error_t memset(int value);

            /**
             * Allocate memory for required number of elements
             * @param size number of elements
             */
            virtual void allocate(size_t size);

            /**
             * Copies from the memory area pointed
             * to by buffer to the memory area pointed to by dst
             * @see cudaMemcpy
             *
             * @param dst destination memory address
             * @param memory target memory type
             */
            virtual void copyTo(void* dst, memory_t memory);

            /**
             * Copies from the memory area pointed
             * to by buffer to the memory area pointed to by dst
             * @see cudaMemcpy
             *
             * @param dst source memory address
             * @param memory source memory type
             */
            virtual void copyFrom(void* src, memory_t memory);

            /**
             * Free buffer from memory
             */
            virtual void free();

        private:

            /**
             * Initialize class variables
             */
            void _init(Allocator* allocator, memory_t memory);

            /**
             * Copies from one memory to another
             *
             * @param src source memory
             * @param dst destination memory
             * @param srcMem source memory location
             * @param dstMem  destination memory location
             */
            void _copy(void* src, void* dst, memory_t srcMem, memory_t dstMem);

            // allocator instance
            Allocator* _allocator;

            // location of memory
            memory_t _memory;

            // texture reference
            texture<T, cudaTextureType1D, cudaReadModeElementType> _textureRef;

    };
};

// template class must include definition too
#include "buffer_memory.cu"

#endif // __BUFFER_MEMORY_CUH__