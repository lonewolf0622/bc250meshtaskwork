#version 460
layout(location=0) out vec4 color;
#if DB
layout(set=0,binding=0,std430) readonly buffer Resource { uint words[]; } heap_data;
layout(set=1,binding=0,std140) uniform Root { uint value; } root;
layout(push_constant) uniform Constants { uint value; } constants;
#endif
void main() {
 color=vec4(0.25,0.5,0.75,1);
#if DB
 if(heap_data.words[0]+root.value+constants.value!=31u)color=vec4(0);
#endif
}
