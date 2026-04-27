#version 450

// Цвет задаётся через push constant (vec4 rgba)
layout(push_constant) uniform PC {
    vec4 color;
} pc;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(pc.color.z, pc.color.y, pc.color.x, pc.color.w);
}
