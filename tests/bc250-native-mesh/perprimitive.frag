#version 460
#extension GL_EXT_mesh_shader : require
layout(location=0) perprimitiveEXT in vec3 color;
layout(location=0) out vec4 outColor;
void main() { outColor=vec4(color,1); }
