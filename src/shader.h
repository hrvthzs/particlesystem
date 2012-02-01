#ifndef __SHADER_H__
#define __SHADER_H__

namespace Shader {

    enum sTypes {
        Vertex,
        Geometry,
        Control,
        Evaluation,
        Fragment
    };

    typedef enum sTypes Types;

};

#endif // __SHADER_H__