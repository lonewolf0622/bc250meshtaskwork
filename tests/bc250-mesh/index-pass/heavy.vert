#version 460
#ifndef VERTICES
#define VERTICES 64
#endif
#ifndef PER_PRIMITIVE
#define PER_PRIMITIVE 0
#endif
#ifndef USE_BALLOT
#define USE_BALLOT 0
#endif
#ifndef FLAT_ORACLE
#define FLAT_ORACLE 0
#endif
#if FLAT_ORACLE
layout(location=1) flat out vec4 vtx;
#else
layout(location=1) out vec4 vtx;
#endif
#if PER_PRIMITIVE
layout(location=0) flat out vec4 color;
#endif
layout(push_constant) uniform Dispatch {
 uint tag; uint indirect; uint steps;
#if FLAT_ORACLE
 uint oracle_corner;
#endif
} dispatch;
void main() {
 uint k=uint(gl_VertexIndex), w=k/(VERTICES*3u), primitive=(k/3u)%VERTICES, corner=k%3u;
 if(dispatch.indirect!=0u && gl_DrawID==2) w=w%16u;
#if FLAT_ORACLE
 corner=(corner+dispatch.oracle_corner)%3u;
#endif
 // Independent VS constructs each triangle corner without mesh output storage.
 uint M=VERTICES-1u;
 uint a=((primitive&3u)==3u) ? primitive%2u : (primitive*5u+w)%M;
 uint ob=1u+(primitive*7u)%(M-1u);
 uint oc=1u+((ob-1u)+1u+(primitive*3u)%(M-2u))%(M-1u);
 uint i=corner==0u?a:(corner==1u?(a+ob)%M:(a+oc)%M);
 uint h=(i+1u)*747796405u ^ (w+dispatch.tag)*2891336453u ^ (USE_BALLOT!=0 ? 64u*277803737u : 0u);
 for(uint step=0u;step<dispatch.steps;step++) {
  h ^= h>>16u; h *= 2246822519u;
  h ^= h>>13u; h *= 3266489917u;
  h ^= h>>16u; h += step*668265263u+374761393u;
 }
 vec2 tile=vec2(float(w&7u),float(w>>3))/4.0-1.0;
 vec2 g=VERTICES==64 ? vec2(float(i&7u),float(i>>3))/8.0 :
                     vec2(float(i%16u),float(i/16u))/vec2(16.0,8.0);
 gl_Position=vec4(tile+g*0.25+vec2(0.01),0.5,1.0);
 vtx=vec4(float(h&255u),float((h>>8u)&255u),float((h>>16u)&255u),255.0)/255.0;
#if PER_PRIMITIVE
 color=vec4(float((primitive*37u+w*11u)&255u)/255.0,float((primitive*13u+w*7u)&255u)/255.0,float(primitive&7u)/7.0,1.0);
#endif
}
