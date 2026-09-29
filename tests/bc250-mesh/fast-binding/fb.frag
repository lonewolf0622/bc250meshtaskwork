#version 460
#extension GL_EXT_nonuniform_qualifier : require
/* fast-binding fragment shader: heap texture with a static (immutable / embedded) sampler and a heap sampler. */
#extension GL_EXT_mesh_shader : require
layout(location = 0) in vec4 col;
#if PERPRIM
layout(location = 1) perprimitiveEXT flat in vec4 pcol;
#endif
layout(location = 0) out vec4 o;
layout(push_constant) uniform PC { uint heap; uint tex; uint smp; uint base; } pc;
layout(set = 0, binding = 0) uniform texture2D texs[];
layout(set = 1, binding = 0) uniform sampler samps[];
layout(set = 2, binding = 0) uniform sampler ssamp;
void main() {
  vec2 uv = col.xy;
  o = texture(sampler2D(texs[pc.tex], ssamp), uv) + texture(sampler2D(texs[pc.tex], samps[pc.smp]), uv.yx) + col;
#if PERPRIM
  o += pcol;
#endif
}
