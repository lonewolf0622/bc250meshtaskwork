#version 460
#extension GL_EXT_mesh_shader : require
layout(location=0) perprimitiveEXT flat in vec4 color;
layout(location=1) in vec4 vtx;
layout(location=0) out vec4 o;
void main(){ o=vec4(color.rgb, 1.0) * 0.875 + vtx * 0.125; }
