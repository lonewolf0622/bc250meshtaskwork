#version 460
layout(triangles, equal_spacing, ccw) in;
layout(location = 0) in vec4 i0[];
layout(location = 1) flat in uint i1[];
layout(location = 0) out vec4 c0;
layout(location = 1) flat out uint c1;
void main()
{
   vec3 t = gl_TessCoord;
   gl_Position = t.x * gl_in[0].gl_Position + t.y * gl_in[1].gl_Position + t.z * gl_in[2].gl_Position;
   c0 = t.x * i0[0] + t.y * i0[1] + t.z * i0[2];
   c1 = i1[0];
}
