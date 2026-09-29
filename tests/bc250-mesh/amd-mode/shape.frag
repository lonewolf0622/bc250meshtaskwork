#version 460
#extension GL_EXT_mesh_shader : require
layout(location = 0) in vec4 col;
#if PERPRIM
layout(location = 1) perprimitiveEXT flat in vec4 pcol;
#endif
layout(location = 0) out vec4 o;
void main() {
#if PERPRIM
  o = col + pcol;
#else
  o = col;
#endif
}
