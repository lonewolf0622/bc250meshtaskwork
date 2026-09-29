#version 460
#extension GL_EXT_mesh_shader : require
layout(location = 0) in vec4 v0;
layout(location = 1) in vec4 v1;
layout(location = 2) in vec4 v2;
layout(location = 3) in vec4 v3;
layout(location = 4) in vec4 v4;
layout(location = 5) in vec4 v5;
layout(location = 6) perprimitiveEXT flat in uvec4 pp;
layout(location = 0) out vec4 o;
void main() { o = vec4(pp) + v0 + v1 + v2 + v3 + v4 + v5; }
