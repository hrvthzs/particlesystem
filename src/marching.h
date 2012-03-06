#ifndef __MARCHING_H__
#define __MARCHING_H__

namespace Marching {

    enum eBuffers {
        EdgesTable,
        TrianglesTable,
        NumVerticesTable,
        AdjacentEdgesTable,
        AdjacentEdgesPosTable,

        VoxelVertices,
        VoxelVerticesScan,
        VoxelOccupied,
        VoxelOccupiedScan,
        VoxelCompact,
        VoxelCubeIndex,

        VertexPosition,
        VertexNormal,
        VertexINormal
    };

    struct sVoxelData {
        uint* vertices;
        uint* verticesScan;
        uint* occupied;
        uint* occupiedScan;
        uint* compact;
        uint* cubeIndex;
    };

    struct sTableData {
        uint* edges;
        uint* triangles;
        uint* numVertices;
        uint* adjacentEdges;
        int3* adjacentEdgesPos;
    };

    struct sVertexData {
        float4* positions;
        float4* normals;
        float4* inormals;
    };

    typedef enum eBuffers Buffers;
    typedef struct sVoxelData VoxelData;
    typedef struct sTableData TableData;
    typedef struct sVertexData VertexData;

    #define GRID_OFFSET 1.0f
    #define NTHREADS 32

}

#endif // __MARCHING_H__