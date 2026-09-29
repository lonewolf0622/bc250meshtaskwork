#version 460
#extension GL_EXT_mesh_shader : require
/* Mirrors Hellblade 2's Nanite fragment shaders: reads tc0, tc2 and tc7.xy, declares tc1 without reading it. */
layout(location = 0) in vec4 tc0;
layout(location = 1) flat in uvec3 tc1;
layout(location = 2) flat in ivec4 tc2;
#if EXTRA
layout(location = 3) in vec4 tc3;
#endif
layout(location = 4) perprimitiveEXT flat in uvec4 tc7;
layout(location = 0) out vec4 o;
void main()
{
   o = tc0 + vec4(tc2) + vec4(float(tc7.x), float(tc7.y), 0.0, 0.0);
#if EXTRA
   o += tc3;
#endif
}
