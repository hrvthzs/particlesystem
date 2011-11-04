#ifndef __BUFFER_ABSTRACT_BUFFER_H__
#define __BUFFER_ABSTRACT_BUFFER_H__

#include "buffer.h"

namespace Buffer {

    /**
     * Abstract Buffer
     *
     * !!! Important !!!
     * For template classes definition must be included too
     */
    template <class T>
    class AbstractBuffer {

        public:

            /**
            * Constructor
            */
            AbstractBuffer();

            /**
            * Destructor
            */
            virtual ~AbstractBuffer();

            /**
            * Bind buffer to texture
            * @return type of error
            */
            virtual error_t bind() = 0;

            /**
            * Unbind memory buffer from texture
            */
            virtual void unbind() = 0;

            /**
            * Initialize memory with specified value
            * @param value init value
            */
            virtual error_t memset(int value) = 0;

            /**
            * Allocate memory for required number of elements
            * @param size number of elements
            */
            virtual void allocate(size_t size) = 0;

            /**
            * Free buffer from memory
            */
            virtual void free() = 0;

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

        protected:

            // number of elements
            size_t _size;

            // flag whether is buffer bound to texture
            bool _bound;

            // pointer to buffer memory
            T* _memoryPtr;
    };
};

#endif // __BUFFER_ABSTRACT_BUFFER_H__