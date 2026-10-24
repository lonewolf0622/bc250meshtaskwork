#version 460
layout(location=1) out vec4 vtx;
layout(push_constant) uniform Dispatch { uint tag; uint indirect; } dispatch;
void main() {
 uint k=uint(gl_VertexIndex), w=k/192u, i=(k/3u)%64u, corner=k%3u;
 // Record two has a 16 x 4 grid; geometry uses its logical X coordinate.
 if(dispatch.indirect!=0u && gl_DrawID==2) w=w%16u;
 uint a,b,c; uint M=63u; a=((i&3u)==3u)?i%2u:(i*5u+w)%M; uint ob=1u+(i*7u)%(M-1u); uint oc=1u+((ob-1u)+1u+(i*3u)%(M-2u))%(M-1u); b=(a+ob)%M;c=(a+oc)%M;
 uint j=corner==0u?a:(corner==1u?b:c);
 vec2 tile=vec2(float(w&7u),float(w>>3))/4.0-1.0;
 vec2 g=vec2(float(j&7u),float(j>>3))/8.0;
 gl_Position=vec4(tile+g*0.25+vec2(0.01),0.5,1.0);
 vtx=vec4(float(j+(j+1u)%64u)/64.0,float(w)/64.0,0.75,1.0);
 
}
