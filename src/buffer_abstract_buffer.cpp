#ifndef __BUFFER_ABSTRACT_BUFFER_CPP__
#define __BUFFER_ABSTRACT_BUFFER_CPP__

#include "buffer_abstract_buffer.h"

namespace Buffer {

    ///////////////////////////////////////////////////////////////////////////

    template<class T>
    AbstractBuffer<T>::AbstractBuffer() {
        this->_bound = false;
        this->_size = 0;
        this->_memoryPtr = NULL;
    }

    ///////////////////////////////////////////////////////////////////////////

    template<class T>
    AbstractBuffer<T>::~AbstractBuffer() {}

    ///////////////////////////////////////////////////////////////////////////

    template<class T>
    T* AbstractBuffer<T>::get() {
        return this->_memoryPtr;
    }

    ///////////////////////////////////////////////////////////////////////////

    template<class T>
    size_t AbstractBuffer<T>::getSize() {
        return this->_size;
    }

    ///////////////////////////////////////////////////////////////////////////

    template<class T>
    size_t AbstractBuffer<T>::getMemorySize() {
        return this->_size * sizeof(T);
    }

    ///////////////////////////////////////////////////////////////////////////

    template<class T>
    bool AbstractBuffer<T>::isBound() {
        return this->_bound;
    }

    ///////////////////////////////////////////////////////////////////////////
}

#endif // __BUFFER_ABSTRACT_BUFFER_CPP__
