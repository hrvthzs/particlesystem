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
             * Free buffer from memory
             */
            virtual void free();

        private:

            /**
             * Initialize class variables
             */
            void _init(Allocator* allocator, memory_t memory);

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