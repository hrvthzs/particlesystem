#include "utils.cuh"

namespace Utils {

    ////////////////////////////////////////////////////////////////////////////

    inline uint iDivUp(uint a, uint b) {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    ////////////////////////////////////////////////////////////////////////////

    void computeGridSize(
        uint n,
        uint blockSize,
        uint &numBlocks,
        uint &numThreads
    ) {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    ////////////////////////////////////////////////////////////////////////////

};