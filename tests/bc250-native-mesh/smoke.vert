#version 460
layout(location=0) out vec3 color;
void main() {
    const vec2 pos[3]=vec2[3](vec2(-0.7,-0.7),vec2(0.7,-0.7),vec2(0,0.7));
    gl_Position=vec4(pos[gl_VertexIndex],0,1);
    color=vec3(1,0,0);
}
