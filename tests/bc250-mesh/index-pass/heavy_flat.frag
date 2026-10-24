#version 460
#extension GL_EXT_mesh_shader : require
#ifndef PER_PRIMITIVE
#define PER_PRIMITIVE 0
#endif
#ifndef REFERENCE
#define REFERENCE 0
#endif
#if PER_PRIMITIVE
#if REFERENCE
layout(location=0) flat in vec4 color;
#else
layout(location=0) perprimitiveEXT flat in vec4 color;
#endif
#endif
layout(location=1) flat in vec4 vtx;
layout(location=0) out vec4 o;
void main() {
 // Recover discrete output bytes without interpolation or weighted float mixing.
 uvec3 bytes=uvec3(round(vtx.rgb*255.0));
#if PER_PRIMITIVE
 uvec3 primitive=uvec3(round(color.rgb*vec3(255.0,255.0,7.0)));
 bytes=(bytes ^ (primitive*uvec3(17u,29u,37u))) & uvec3(255u);
#endif
 o=vec4(vec3(bytes)/255.0,1.0);
}
