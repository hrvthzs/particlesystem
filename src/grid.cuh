#ifndef __GRID_CUH__
#define __GRID_CUH__

    enum eGridBuffers {
        Hash,
        Index,
        CellStart,
        CellStop,
    };

    struct sGridData {
        uint* hash;
        uint* index;
        uint* cellStart;
        uint* cellStop;
    };

    struct sGridParams {
        float3 size;
        float3 min;
        float3 max;
        // number of cells in each dimension
        float3 resolution;
        float3 delta;
        float3 cellSize;
    };

    typedef enum eGridBuffers GridBuffers;
    typedef struct sGridData GridData;
    typedef struct sGridParams GridParams;

    /**
     * If cell is empty in grid this value idicates it
     */
    #define EMPTY_CELL_VALUE 0xffffffff

    __constant__ GridParams  cudaGridParams;

#endif // __GRID_CUH__