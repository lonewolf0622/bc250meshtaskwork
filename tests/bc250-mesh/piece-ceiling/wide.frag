#version 460
/* Reads every output of wide.mesh (5 render targets, like Control's G-buffer pass). */
layout(location = 0) flat in uint a0;
layout(location = 1) in vec3 a1;
layout(location = 2) in vec3 a2;
layout(location = 3) in vec4 a3;
layout(location = 4) in vec2 a4;
#if FIT
layout(location = 5) in vec4 a5;
layout(location = 6) in vec4 a6;
#else
layout(location = 6) in vec4 a6;
layout(location = 7) in vec4 a7;
layout(location = 8) in vec3 a8;
#endif
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
layout(location = 3) out vec4 o3;
layout(location = 4) out vec4 o4;
void main()
{
   o0 = vec4(a1, float(a0));
   o1 = vec4(a2, a4.x);
   o2 = a3;
   o3 = a6 + vec4(a4.y);
#if FIT
   o4 = a5;
#else
   o4 = a7 + vec4(a8, 0.0);
#endif
#if PRIMID
   o4.w += float(gl_PrimitiveID);
#endif
}
