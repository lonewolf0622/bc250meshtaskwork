#version 460
#extension GL_EXT_mesh_shader : require
// Declares TEXCOORD_1 like Hellblade 2's Nanite fragment shaders, never reads it.
layout(location = 0) in vec4 tc0;
layout(location = 1) flat in uvec3 tc1;
layout(location = 2) flat in ivec4 tc2;
layout(location = 4) perprimitiveEXT flat in uvec4 tc7;
layout(location = 0) out vec4 o;
void main() { o = tc0 + vec4(tc2) + vec4(tc7); }
