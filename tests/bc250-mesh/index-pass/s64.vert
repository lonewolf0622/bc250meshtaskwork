#version 460
layout(location=1) out vec4 vtx;
void main() {
 uint k=uint(gl_VertexIndex), w=k/192u, i=(k/3u)%64u, corner=k%3u;
 uint a,b,c; uint s=min(i,61u); a=s; b=s+1u+((i+w)%2u); c=(i>=24u && (i%4u)==0u) ? s-20u : s+2u; if(i>=61u){a=61u+(i-61u)%3u; b=60u; c=63u;}
 uint j=corner==0u?a:(corner==1u?b:c);
 vec2 tile=vec2(float(w&7u),float(w>>3))/4.0-1.0;
 vec2 g=vec2(float(j&7u),float(j>>3))/8.0;
 gl_Position=vec4(tile+g*0.25+vec2(0.01),0.5,1.0);
 vtx=vec4(float(j)/64.0,float(w)/64.0,0.75,1.0);
 
}
