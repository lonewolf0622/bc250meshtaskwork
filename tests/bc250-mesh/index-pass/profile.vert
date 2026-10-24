#version 460
#ifndef VERTICES
#define VERTICES 128
#endif
#ifndef PRIMITIVES
#define PRIMITIVES 128
#endif
#ifndef PER_PRIMITIVE
#define PER_PRIMITIVE 0
#endif
#ifndef CULL_DISTANCE
#define CULL_DISTANCE 0
#endif
#ifndef FLAT_ORACLE
#define FLAT_ORACLE 0
#endif
#if CULL_DISTANCE
out gl_PerVertex { vec4 gl_Position; float gl_CullDistance[1]; };
#endif
#if FLAT_ORACLE
layout(location=1) flat out vec4 vtx;
#else
layout(location=1) out vec4 vtx;
#endif
#if PER_PRIMITIVE
layout(location=0) flat out vec4 color;
#endif
layout(push_constant) uniform Dispatch { uint tag; uint indirect; uint steps; uint oracle_corner; } dispatch;
void main() {
 // Construct each triangle independently, without reading producer outputs.
 uint k=uint(gl_VertexIndex), w=k/(PRIMITIVES*3u), p=(k/3u)%PRIMITIVES, corner=k%3u;
 if(dispatch.indirect!=0u && gl_DrawID==2) w=w%16u;
#if FLAT_ORACLE
 corner=(corner+dispatch.oracle_corner)%3u;
#endif
 uint nv=VERTICES-(dispatch.tag&3u), np=PRIMITIVES-(dispatch.tag&3u), m=nv-1u;
 uint a=(p*5u+w)%m;
 uint ob=1u+(p*7u)%(m-1u), oc=1u+((ob-1u)+1u+(p*3u)%(m-2u))%(m-1u);
 uint i=p==0u?(corner==0u?0u:(corner==1u?7u:21u)):(corner==0u?a:(corner==1u?(a+ob)%m:(a+oc)%m));
 // Inactive primitive slots collapse to one point, matching degenerate fill.
 if(p>=np) i=0u;
 uint h=(i+1u)*747796405u ^ (w+dispatch.tag)*2891336453u;
 for(uint step=0u;step<dispatch.steps;step++) {
  h^=h>>16u; h*=2246822519u; h^=h>>13u; h*=3266489917u;
  h^=h>>16u; h+=step*668265263u+374761393u;
 }
 vec2 tile=vec2(float(w&7u),float(w>>3))/4.0-1.0;
 vec2 g=vec2(float(i%16u),float(i/16u))/vec2(16.0,float(VERTICES/16));
 gl_Position=vec4(tile+g*0.25+vec2(0.01),0.5,1.0);
#if CULL_DISTANCE
 gl_CullDistance[0]=i%7u==0u?-1.0:1.0;
#endif
 vtx=vec4(float(h&255u),float((h>>8u)&255u),float((h>>16u)&255u),255.0)/255.0;
#if PER_PRIMITIVE
 color=vec4(float((p*37u+w*11u)&255u)/255.0,float((p*13u+w*7u)&255u)/255.0,float(p&7u)/7.0,1.0);
#endif
}
