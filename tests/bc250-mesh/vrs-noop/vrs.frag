#version 460
/* Fragment stage for vrs.c. READ_RATE: scales the result by (1 + gl_ShadingRateEXT); with
 * RADV_BC250_VRS_NOOP the rate reads 0 (1x1), so the code must equal the plain one. */
#if READ_RATE
#extension GL_EXT_fragment_shading_rate : require
#endif
layout(location = 0) in vec4 c0;
layout(location = 1) flat in uint c1;
layout(location = 0) out vec4 o0;
void main()
{
   vec4 r = c0;
   r.w += float(c1);
#if READ_RATE
   r *= float(1 + gl_ShadingRateEXT);
#endif
   o0 = r;
}
