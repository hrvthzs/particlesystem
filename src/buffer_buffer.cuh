#ifndef __BUFFER_BUFFER_CUH__
#define __BUFFER_BUFFER_CUH__

#include "buffer.h"
#include "buffer_allocator.h"

namespace Buffer {

    typedef enum Errors error_t;
    typedef enum MemoryLocation memory_t;

    /**
     * Memory Buffer
     *
     * !!! Important !!!
     * Template classes must be included definition too
     */
    template <class T> class Buffer {

        public:

            /** Constructor
             *
             * @param allocator memory allocator instance
             * @param memory memory location @see Buffer::MemoryLocation
             */
            Buffer(Allocator* allocator, memory_t memory);

            /**
             * Destructor
             */
            virtual ~Buffer();

            /**
             * Bind buffer to texture
             * @return type of error
             */
            error_t bind();

            /**
             * Unbind memory buffer from texture
             */
            void unbind();

            /**
             * Initialize memory with specified value
             * @param value init value
             */
            error_t memset(int value);

            /**
             * Allocate memory for required number of elements
             * @param size number of elements
             */
            void allocate(size_t size);

            /**
             * Free buffer from memory
             */
            void free();

            /**
             * Returns pointer to memory
             */
            T* get();

            /**
             * Returns the number of elements in buffer
             */
            size_t getSize();

            /**
             * Returns allocated memory size
             */
            size_t getMemorySize();

            /**
             * Returns true if buffer is bound to texture,
             * false otherwise
             */
            bool isBound();

        private:
            // allocator instance
            Allocator* _allocator;

            // location of memory
            memory_t _memory;

            // number of elements
            size_t _size;

            // flag whether is buffer bound to texture
            bool _bound;

            // pointer to buffer memory
            T* _memoryPtr;

            // texture reference
            texture<T, cudaTextureType1D, cudaReadModeElementType> _textureRef;
    };
};

#endif // __BUFFER_BUFFER_CUH__