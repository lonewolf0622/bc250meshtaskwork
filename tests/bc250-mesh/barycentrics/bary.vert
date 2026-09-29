#version 460
/* Last pre-rasterization stage for bary.c (or the vertex stage in front of bary.geom / bary.tes*). */
layout(push_constant) uniform PC { uint base; } pc;
layout(location = 0) out vec4 c0;
layout(location = 1) flat out uint c1;
void main()
{
   uint v = uint(gl_VertexIndex) + pc.base;
   gl_Position = vec4(float(v & 1u) - 0.5, float((v >> 1) & 1u) - 0.5, 0.25 * float(v % 3u), 1.0 + 0.125 * float(v));
   c0 = vec4(float(v), float(v * 2u), float(v * 3u), 1.0);
   c1 = v;
}
