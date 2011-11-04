#ifndef __BUFFER_ABSTRACT_CPP__
#define __BUFFER_ABSTRACT_CPP__

#include "buffer_abstract.h"

namespace Buffer {

    ///////////////////////////////////////////////////////////////////////////

    template<class T>
    Abstract<T>::Abstract() {
        this->_bound = false;
        this->_size = 0;
        this->_memoryPtr = NULL;
    }

    ///////////////////////////////////////////////////////////////////////////

    template<class T>
    Abstract<T>::~Abstract() {}

    ///////////////////////////////////////////////////////////////////////////

    template<class T>
    T* Abstract<T>::get() {
        return this->_memoryPtr;
    }

    ///////////////////////////////////////////////////////////////////////////

    template<class T>
    size_t Abstract<T>::getSize() {
        return this->_size;
    }

    ///////////////////////////////////////////////////////////////////////////

    template<class T>
    size_t Abstract<T>::getMemorySize() {
        return this->_size * sizeof(T);
    }

    ///////////////////////////////////////////////////////////////////////////

    template<class T>
    bool Abstract<T>::isBound() {
        return this->_bound;
    }

    ///////////////////////////////////////////////////////////////////////////
}

#endif // __BUFFER_ABSTRACT_CPP__