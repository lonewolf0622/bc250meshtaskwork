#version 460
#extension GL_EXT_nonuniform_qualifier : require
layout(push_constant) uniform PC { uint heap; uint tex; uint smp; uint base; } pc;
layout(set = 0, binding = 0, std430) readonly buffer SB { vec4 v[]; } sbufs[];
layout(location = 0) out vec4 col;
void main() {
  vec4 d = sbufs[pc.heap].v[gl_VertexIndex & 15];
  gl_Position = vec4(float(gl_VertexIndex & 1), float(gl_VertexIndex >> 1), 0.5, 1.0) + d * 0.001;
  col = d;
}
