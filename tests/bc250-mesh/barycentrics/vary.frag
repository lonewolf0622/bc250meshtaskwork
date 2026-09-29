#version 460
#extension GL_EXT_fragment_shader_barycentric : require
/* Fragment stage for bary.c (outputs of bary.vert / bary.geom / bary.tese):
 *   (none)    : plain interpolation, no barycentrics (must compile exactly as without the extension).
 *   BARY      : also reads gl_BaryCoordEXT and gl_BaryCoordNoPerspEXT.
 *   PERVERTEX : reads c0 per vertex (pervertexEXT, strict vertex order) and weights it itself. */
#if PERVERTEX
layout(location = 0) pervertexEXT in vec4 c0[3];
#else
layout(location = 0) in vec4 c0;
#endif
layout(location = 1) flat in uint c1;
layout(location = 0) out vec4 o0;
void main()
{
#if PERVERTEX
   vec4 r = c0[0] * gl_BaryCoordEXT.x + c0[1] * gl_BaryCoordEXT.y + c0[2] * gl_BaryCoordEXT.z;
#else
   vec4 r = c0;
#endif
   r.w += float(c1);
#if BARY
   r.xyz += gl_BaryCoordEXT * 3.0 + gl_BaryCoordNoPerspEXT;
#endif
   o0 = r;
}
