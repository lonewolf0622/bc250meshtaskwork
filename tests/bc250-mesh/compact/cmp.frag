#version 460
#extension GL_EXT_mesh_shader : require
#if BARY
/* BC250 barycentrics: the Mesh shader also exports its position references. */
#extension GL_EXT_fragment_shader_barycentric : require
#endif
/* Reads every output of dr.mesh for the same -D options (so none is dropped as unread). */
layout(location = 0) in vec4 v0;
layout(location = 1) flat in uint v1;
#if UNIFORM
layout(location = 2) flat in uint u0;
layout(location = 3) in vec4 u1;
layout(location = 4) flat in uint u2;
layout(location = 5) flat in uint u3;
layout(location = 6) perprimitiveEXT flat in uvec4 u4;
#endif
#if PARTIAL
layout(location = 2) in vec4 p0;
#endif
#if ARRAYED
layout(location = 2) in float ar[3];
#endif
#if PERPRIM
layout(location = 2) perprimitiveEXT flat in vec4 pp0;
#endif
#if MULTISTORE
layout(location = 2) flat in uint ms0;
#endif
layout(location = 0) out vec4 o0;
void main()
{
   vec4 r = v0 + vec4(float(v1));
#if UNIFORM
   r += vec4(float(u0), float(u2), float(u3), 0.0) + u1 + vec4(u4);
#endif
#if PARTIAL
   r += p0;
#endif
#if ARRAYED
   r.xyz += vec3(ar[0], ar[1], ar[2]);
#endif
#if PERPRIM
   r += pp0;
#if PRIMID
   r.w += float(gl_PrimitiveID);
#endif
#endif
#if MULTISTORE
   r.x += float(ms0);
#endif
#if BARY
   r.xyz += gl_BaryCoordEXT;
#endif
   o0 = r;
}
