#version 450

// Вершина в NDC: xy = [-1,1], z = нормализованная глубина [0,1]
layout(location = 0) in vec3 inPos;

layout(push_constant) uniform MVP {
    layout(offset = 16) mat4 VP;
    layout(offset = 80) mat4 transform;
} matrices;

void main() {
    gl_Position = matrices.VP * matrices.transform * vec4(inPos.x, inPos.y, inPos.z, 1.0);
}
