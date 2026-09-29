#version 460
/* Vertex stage for vrs.c. WRITE_RATE: also writes gl_PrimitiveShadingRateEXT (a pure function of the
 * vertex index, dead once RADV_BC250_VRS_NOOP strips the output: the code must equal the plain one). */
#if WRITE_RATE
#extension GL_EXT_fragment_shading_rate : require
#endif
layout(push_constant) uniform PC { uint base; } pc;
layout(location = 0) out vec4 c0;
layout(location = 1) flat out uint c1;
void main()
{
   uint v = uint(gl_VertexIndex) + pc.base;
   gl_Position = vec4(float(v & 1u) - 0.5, float((v >> 1) & 1u) - 0.5, 0.25 * float(v % 3u), 1.0 + 0.125 * float(v));
   c0 = vec4(float(v), float(v * 2u), float(v * 3u), 1.0);
   c1 = v;
#if WRITE_RATE
   gl_PrimitiveShadingRateEXT = int((v * 5u) & 15u);
#endif
}
