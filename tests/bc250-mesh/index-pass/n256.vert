#version 460
layout(location=1) out vec4 vtx;
void main() {
 uint k=uint(gl_VertexIndex), w=k/384u, i=(k/3u)%128u, corner=k%3u;
 uint a,b,c; uint M=255u; a=((i&3u)==3u)?i%2u:(i*5u+w)%M; uint ob=1u+(i*7u)%(M-1u); uint oc=1u+((ob-1u)+1u+(i*3u)%(M-2u))%(M-1u); b=(a+ob)%M;c=(a+oc)%M;
 uint j=corner==0u?a:(corner==1u?b:c);
 vec2 tile=vec2(float(w&7u),float(w>>3))/4.0-1.0;
 vec2 g=vec2(float(j%16u),float(j/16u))/vec2(16.0,16.0);
 gl_Position=vec4(tile+g*0.25+vec2(0.01),0.5,1.0);
 vtx=vec4(float(j)/256.0,float(w)/64.0,0.75,1.0);
 
}
