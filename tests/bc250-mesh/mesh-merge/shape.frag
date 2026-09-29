#version 460
#extension GL_EXT_mesh_shader : require
/* -DPERPRIM=<n>: reads the per-primitive outputs of shape.mesh (flat, as Hellblade 2's FS reads its
 * perprimitiveEXT uvec4 into an imageAtomicMax visibility-buffer write). */
layout(location = 0) in vec4 col;
#if PERPRIM == 1
layout(location = 1) perprimitiveEXT flat in vec4 pcol;
#elif PERPRIM == 2
layout(location = 1) flat in uvec3 vtc1;
layout(location = 2) flat in ivec4 vtc2;
layout(location = 4) flat perprimitiveEXT in uvec4 ptc7;
#elif PERPRIM == 3
layout(location = 1) perprimitiveEXT flat in vec4 pcol;
layout(location = 2) perprimitiveEXT flat in ivec2 pid2;
layout(location = 3) perprimitiveEXT flat in uint pid1;
#elif PERPRIM == 4
#endif
layout(location = 0) out vec4 o;
void main() {
#if PERPRIM == 1
  o = col + pcol;
#elif PERPRIM == 2
  o = col + vec4(uvec4(vtc1, ptc7.x ^ ptc7.y)) + vec4(vtc2) + vec4(uintBitsToFloat(ptc7.z), float(ptc7.w), 0.0, 0.0);
#elif PERPRIM == 3
  o = col + pcol + vec4(vec2(pid2), float(pid1), float(gl_PrimitiveID));
#elif PERPRIM == 4
  o = col + vec4(float(gl_Layer));
#else
  o = col;
#endif
}
