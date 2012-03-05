#ifndef __MARCHING_RENDERER_CUH__
#define __MARCHING_RENDERER_CUH__

#include <GL/glew.h>
#include "grid_uniform.cuh"
#include "marching.h"

namespace Marching {

    class Renderer {
        public:
            Renderer(Grid::Uniform* grid);
            virtual ~Renderer();

            /**
             * !!! Important !!!
             * Call bindBuffers before rendering
             * Call unbindBuffers after rendering
             *
             */
            void render();

            /**
             * Bind buffers
             */
            void bindBuffers();

            /**
             * Unbind buffers
             */
            void unbindBuffers();

            /**
             * Get rendered data
             */
            Marching::VertexData& getData();

            /**
             * Get number of vertices
             */
            uint getNumVertices();

            /**
             * Get positions vertex buffer
             */
            GLuint getPositionsVBO();

            /**
             * Get normals vertex buffer
             */
            GLuint getNormalsVBO();

            /**
             * Free buffers
             *
             */
            void freeBuffers();

            /**
             * Set interpolation of vertex normals
             *
             * @param interpolation flag of interpolation
             */
            void setInterpolation(bool interpolation);

        private:
            void _createBuffers();
            void _deleteBuffers();
            void _thrustScanWrapper(uint* input, uint* output);

            Buffer::Manager<Marching::Buffers> *_tableBuffMan;
            Buffer::Manager<Marching::Buffers> *_voxelBuffMan;
            Buffer::Manager<Marching::Buffers> *_vertexBuffMan;

            Marching::VoxelData _voxelData;
            Marching::TableData _tableData;
            Marching::VertexData _vertexData;

            GridParams _gridParams;
            Grid::Uniform* _grid;

            uint _numCells;
            uint _maxVertices;
            uint _numVertices;

            bool _interpolation;

            GLuint _positionsVBO;
            GLuint _normalsVBO;
    };
};


#endif // __MARCHING_RENDERER_CUH__