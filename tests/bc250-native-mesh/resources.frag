#version 460
layout(push_constant) uniform Constants { layout(offset=240) uint challenge; uint side; uint magic; uint reserved; } pc;
layout(location=0) in vec3 color;
layout(location=0) out vec4 outputColor;
void main() { outputColor=pc.magic==0xabcdef01u?vec4(color,1):vec4(0,0,1,1); }
