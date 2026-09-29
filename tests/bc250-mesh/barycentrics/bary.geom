#version 460
/* Pass-through geometry stage: triangles in, a triangle strip out, outputs of bary.vert forwarded. */
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
layout(location = 0) in vec4 i0[];
layout(location = 1) flat in uint i1[];
layout(location = 0) out vec4 c0;
layout(location = 1) flat out uint c1;
void main()
{
   for (int i = 0; i < 3; i++) {
      gl_Position = gl_in[i].gl_Position + vec4(0.0, 0.0, 0.0, 0.0625 * float(i));
      c0 = i0[i];
      c1 = i1[i];
      EmitVertex();
   }
   EndPrimitive();
}
