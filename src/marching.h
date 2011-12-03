#ifndef __MARCHING_H__
#define __MARCHING_H__

namespace Marching {

    enum eBuffers {
        EdgesTable,
        TrianglesTable,
        NumVerticesTable,

        VoxelVertices,
        VoxelVerticesScan,
        VoxelOccupied,
        VoxelOccupiedScan,
        VoxelCompact,

        VertexPosition,
        VertexNormal
    };

    struct sVoxelData {
        uint* vertices;
        uint* verticesScan;
        uint* occupied;
        uint* occupiedScan;
        uint* compact;
    };

    struct sTableData {
        uint* edges;
        uint* triangles;
        uint* numVertices;
    };

    struct sVertexData {
        float4* positions;
        float4* normals;
    };

    typedef enum eBuffers Buffers;
    typedef struct sVoxelData VoxelData;
    typedef struct sTableData TableData;
    typedef struct sVertexData VertexData;

}

#endif // __MARCHING_H__