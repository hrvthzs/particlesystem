#ifndef __BUFFER_MANAGER_CUH__
#define __BUFFER_MANAGER_CUH__

#include "buffer_abstract.h"

#include <map>

namespace Buffer {

    /**
     * Manager handles allocation, freeing of buffers.
     * Class allows setting memory for all containing buffer to
     * a specified value. Was is important, number of elements for
     * all buffers is set to same value via allocation method.
     * Most important is when an instance of Manager class is
     * deleted from memory, it's destructor will free and delete all
     * containing buffers too.
     *
     * !!! Important !!!
     * Template classes must be included definition too
     */
    template <class T>
    class Manager {

        public:
            /**
             * Constructor
             */
            Manager();

            /**
             * Destructor
             * Free and delete buffers
             */
            virtual ~Manager();

            /**
             * Add buffer
             * @param id unique id of buffer
             * @param buffer buffer instance
             */
            void addBuffer(T id, Abstract<void>* buffer);

            /**
             * Remove buffer, free it and delete from memory
             * @param id unique id of buffer
             */
            void removeBuffer(T id);

            /**
             * Allocate buffer with specified size
             * @param size number of elements
             */
            void allocateBuffers(size_t size);

            /**
             * Free buffers
             */
            void freeBuffers();

            /**
             * Initialize buffers with requested value
             */
            void memsetBuffers(int value);

            /**
             * Bind buffers (texture / VBO mapping)
             */
            void bindBuffers();

            /**
             * Unbind buffers
             */
            void unbindBuffers();

            /**
             * Get buffer via unique id
             * @param id unique id of buffer
             * @return pointer to instance of buffer
             */
            Abstract<void>* get(T id);

            /**
             * Get buffer via unique id
             * @param id unique id of buffer
             * @return pointer to instance of buffer
             */
            Abstract<void>* operator[] (T id);

        private:

            typedef std::map<T, Abstract<void> *> BuffersMap;
            typedef typename BuffersMap::const_iterator BufferIterator;

            // buffers map
            BuffersMap _buffers;

            // buffers map iterator
            BufferIterator _iterator;

    };

};

#endif // __BUFFER_MANAGER_CUH__