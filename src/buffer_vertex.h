#ifndef __BUFFER_VERTEX_H__
#define __BUFFER_VERTEX_H__

#include <GL/glew.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include "buffer_abstract_buffer.h"

namespace Buffer {

    template <class T>
    class Vertex : public AbstractBuffer<T> {
        public:

            /**
             * Constructor
             */
            Vertex();

            /**
             * Destructor
             */
            virtual ~Vertex();

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

            /**
             * Get VBO
             */
            GLuint getVBO();

        private:

            struct cudaGraphicsResource* _cudaVBOResource;
            GLuint _VBO;
    };
};

#endif // __BUFFER_VERTEX_H__