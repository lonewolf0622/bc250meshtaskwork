#version 460
/* Reads every generic output of cc.mesh for the same -D options; FSCLIP also reads the interpolated
 * gl_ClipDistance / gl_CullDistance inputs, BARY the barycentric coordinates. */
#if BARY
#extension GL_EXT_fragment_shader_barycentric : require
#endif
#if PERPRIM
#extension GL_EXT_mesh_shader : require
#endif
#ifndef NCLIP
#define NCLIP 0
#endif
#ifndef NCULL
#define NCULL 0
#endif
#if FSCLIP && NCLIP
in float gl_ClipDistance[NCLIP];
#endif
#if FSCLIP && NCULL
in float gl_CullDistance[NCULL];
#endif
layout(location = 0) in vec4 v0;
layout(location = 1) flat in uint v1;
#if PERPRIM
layout(location = 2) perprimitiveEXT flat in vec4 pp0;
#endif
layout(location = 0) out vec4 o0;
void main()
{
   vec4 r = v0 + vec4(float(v1));
#if PERPRIM
   r += pp0;
#endif
#if FSCLIP && NCLIP
   r.x += gl_ClipDistance[0];
#endif
#if FSCLIP && NCULL
   r.y += gl_CullDistance[NCULL - 1];
#endif
#if BARY
   r.xyz += gl_BaryCoordEXT;
#endif
   o0 = r;
}
