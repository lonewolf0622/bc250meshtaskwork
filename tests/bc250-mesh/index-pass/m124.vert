#version 460
layout(location=1) out vec4 vtx;
void main() {
 uint k=uint(gl_VertexIndex), w=k/372u, i=(k/3u)%124u, corner=k%3u;
 uint a,b,c; uint kind=w%3u; if(kind==0u){uint s=(i*62u)/124u; a=s;b=s+1u;c=(s>=10u && i%3u==0u)?s-10u:s+2u;}else if(kind==1u){a=(i*5u+w)%63u;b=(a+1u+(i*7u)%61u)%63u;c=(a+2u+(i*3u)%59u)%63u;}else{uint s=(i*61u)/123u;a=s;b=s+1u;c=s+2u;}
 uint j=corner==0u?a:(corner==1u?b:c);
 vec2 tile=vec2(float(w&7u),float(w>>3))/4.0-1.0;
 vec2 g=vec2(float(j&7u),float(j>>3))/8.0;
 gl_Position=vec4(tile+g*0.25+vec2(0.01),0.5,1.0);
 vtx=vec4(float(j)/64.0,float(w)/64.0,0.75,1.0);
 
}
