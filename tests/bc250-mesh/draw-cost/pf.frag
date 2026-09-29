#version 460
/* Fragment shaders for the draw-cost suite (nanite shape, per-primitive input):
 *   KIND 0 (vis)   no output, one 32-bit atomicMax whose result is unused (order-independent, like
 *                  Hellblade 2's Nanite visibility-buffer shaders, which use a 64-bit image atomic max)
 *   KIND 1 (used)  the same atomic, but its result feeds a second atomic (order-dependent)
 *   KIND 2 (exch)  atomicExchange (order-dependent)
 *   KIND 3 (store) a plain storage write (order-dependent) */
#extension GL_EXT_mesh_shader : require
layout(location = 0) in vec4 col;
layout(location = 1) perprimitiveEXT flat in vec4 pcol;
layout(set = 0, binding = 0) buffer V { uint v[]; } vis;
void main() {
  uint i = uint(gl_FragCoord.x) & 255u;
  uint key = floatBitsToUint(col.x + pcol.y);
#if KIND == 0
  atomicMax(vis.v[i], key);
#elif KIND == 1
  uint old = atomicMax(vis.v[i], key);
  atomicOr(vis.v[256u + i], old);
#elif KIND == 2
  atomicExchange(vis.v[i], key);
#else
  vis.v[i] = key;
#endif
}
