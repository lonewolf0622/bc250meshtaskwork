#version 460
layout(location=1) out vec4 vtx;
void main() {
 uint k=uint(gl_VertexIndex), w=k/192u, i=(k/3u)%64u, corner=k%3u;
 uint a,b,c; uint s=(i*50u)/64u; uint q=s+s/4u; a=q; b=min(q+1u+(i%2u),63u); c=(i>=24u && (i%4u)==0u) ? q-20u : min(q+2u,63u); if((a%5u)==4u)a--; if((b%5u)==4u)b--; if((c%5u)==4u)c--;
 uint j=corner==0u?a:(corner==1u?b:c);
 vec2 tile=vec2(float(w&7u),float(w>>3))/4.0-1.0;
 vec2 g=vec2(float(j&7u),float(j>>3))/8.0;
 gl_Position=vec4(tile+g*0.25+vec2(0.01),0.5,1.0);
 vtx=vec4(float(j)/64.0,float(w)/64.0,0.75,1.0);
 
}
