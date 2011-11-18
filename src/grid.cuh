#ifndef __GRID_H__
#define __GRID_H__

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
    };

    typedef enum eGridBuffers GridBuffers;
    typedef struct sGridData GridData;
    typedef struct sGridParams GridParams;

    __constant__ GridParams cudaGridParams;


#endif // __GRID_H__