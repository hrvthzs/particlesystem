#ifndef __BUFFER_VERTEX_H__
#define __BUFFER_VERTEX_H__

#include <GL/glew.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include "buffer_abstract.h"

namespace Buffer {

    template <class T>
    class Vertex : public Abstract<T> {
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
             * Must called before the GL/SDL context is destroyed,
             * it not program ends with segmentation fault error
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

// template class must include definition too
#include "buffer_vertex.cpp"

#endif // __BUFFER_VERTEX_H__