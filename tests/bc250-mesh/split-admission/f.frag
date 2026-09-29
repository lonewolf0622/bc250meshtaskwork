#version 460
#extension GL_EXT_mesh_shader : require
layout(location=0) perprimitiveEXT flat in vec4 pcol;
layout(location=0) out vec4 o;
void main(){ o=pcol; }
