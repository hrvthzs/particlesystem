namespace Utils {

    /**
     * Round a / b to nearest higher integer value
     */
    inline uint iDivUp(uint a, uint b);

    /**
     * Compute grid and thread block size for a given number of elements
     *
     * @param n number of elements
     * @param blockSize minimal number of threads in block
     * @param numBlocks outputs number of required block in grid
     * @param numThreads outputs number of required threads in blocks
     */
    void computeGridSize(
        uint n,
        uint minBlockSize,
        uint &numBlocks,
        uint &numThreads
    );

};