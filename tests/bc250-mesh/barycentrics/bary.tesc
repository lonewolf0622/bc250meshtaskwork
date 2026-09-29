#version 460
layout(vertices = 3) out;
layout(location = 0) in vec4 i0[];
layout(location = 1) flat in uint i1[];
layout(location = 0) out vec4 o0[];
layout(location = 1) flat out uint o1[];
void main()
{
   gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
   o0[gl_InvocationID] = i0[gl_InvocationID];
   o1[gl_InvocationID] = i1[gl_InvocationID];
   gl_TessLevelOuter[0] = 2.0; gl_TessLevelOuter[1] = 2.0; gl_TessLevelOuter[2] = 2.0;
   gl_TessLevelInner[0] = 2.0;
}
