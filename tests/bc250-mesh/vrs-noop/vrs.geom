#version 460
/* Pass-through geometry stage for vrs.c. WRITE_RATE: also writes gl_PrimitiveShadingRateEXT per primitive. */
#if WRITE_RATE
#extension GL_EXT_fragment_shading_rate : require
#endif
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
layout(location = 0) in vec4 i0[];
layout(location = 1) flat in uint i1[];
layout(location = 0) out vec4 c0;
layout(location = 1) flat out uint c1;
void main()
{
   for (int i = 0; i < 3; i++) {
      gl_Position = gl_in[i].gl_Position;
      c0 = i0[i];
      c1 = i1[i];
#if WRITE_RATE
      gl_PrimitiveShadingRateEXT = int(uint(gl_PrimitiveIDIn) & 15u);
#endif
      EmitVertex();
   }
   EndPrimitive();
}
