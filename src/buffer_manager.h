#ifndef __BUFFER_MANAGER_H__
#define __BUFFER_MANAGER_H__

namespace Buffer {

    class Manager {

        Manager();
        ~Manager();


        addBuffer(Buffer* buffer);
        removeBuffer();
        memsetBuffers();
        allocateBuffers(size_t size);
        freeBuffers();


    }

}

#endif // __BUFFER_MANAGER_H__