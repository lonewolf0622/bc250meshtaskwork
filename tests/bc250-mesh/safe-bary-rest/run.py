"""Private bary corners against the existing split NIR; noop shim only."""
import os,re,json,subprocess
from pathlib import Path
S=Path(__file__).resolve().parents[3];B=Path(os.environ['BUILD']);O=Path(os.environ['KEEP']);O.mkdir(exist_ok=True,parents=True)
assert not Path('/dev/dri').exists() and os.getenv('AMDGPU_GPU_ID')=='gfx1013'
base=dict(os.environ)
base.update(x.split('=',1) for x in re.search(r"policy='([^']+)'",(S/'tests/bc250-mesh/safe-direct/run.py').read_text())[1].split())
base.update(VK_DRIVER_FILES=os.environ['ICD'],VK_ICD_FILENAMES=os.environ['ICD'],NIR_DEBUG='validate',ACO_DEBUG='validateir,validatera',BC250_TRACE_COMPILE='1',BC250_CAPTURE_POLICY_SHADERS='1',RADV_DEBUG='dumpibs',RADV_BC250_PERF_PIECE_PRIMS='64',RADV_BC250_MESH_NO_SPLIT='1',RADV_BC250_MESH_SAFE_STATS='0',RADV_BC250_MESH_AUTOCULL='1',RADV_BC250_MESH_AUTOCULL_WIDE='1')
def run(args,log,env=base):
 with log.open('w') as f:p=subprocess.run(list(map(str,args)),env=env,stdout=f,stderr=subprocess.STDOUT)
 t=log.read_text();assert p.returncode==0,(log,p.returncode,t[-2500:]);return t
run(['cc','-O1','-Wall','-o',O/'pipe',S/'tests/bc250-mesh/safe-direct/pipe.c','-lvulkan'],O/'pipe-build.log')
run(['bash',S/'tests/bc250-mesh/direct-read/build_oracle.sh',O/'oracle'],O/'oracle-build.log',dict(base,SRC=str(S),BUILD=str(B)))
rows=[]
for v,p,lanes in ((6,6,32),):
 for prov in ('first','last','dynamic','dynamic-first'):
  for direct_read in ('full','0'):
   name=f'v{v}p{p}-{prov}-dr{direct_read}'
   if os.getenv('ONLY') and not re.search(os.environ['ONLY'],name):continue
   d=O/name;d.mkdir(exist_ok=True)
   defs=[f'-DVERTS={v}',f'-DPRIMS={p}',f'-DLANES={lanes}','-DPERPRIM=1','-DNOCULL=1','-DBARY=1','-DPERVERTEX=1']
   for st,src in [('mesh','compact/cmp.mesh'),('frag','barycentrics/bary.frag')]:
    source=S/'tests/bc250-mesh'/src
    if v in (6,32) and st=='mesh':
     source=d/'pointsize.mesh';source.write_text((S/'tests/bc250-mesh'/src).read_text().replace('gl_MeshVerticesEXT[v].gl_Position =', 'gl_MeshVerticesEXT[v].gl_PointSize = 1.0; gl_MeshVerticesEXT[v].gl_Position ='))
    run(['glslangValidator','--target-env','vulkan1.3','-S',st,*defs,'-o',d/(st+'.spv'),source],d/(st+'-build.log'))
   for mode in ('split-noref','split','on'):
    dump=d/mode;dump.mkdir(exist_ok=True)
    e=dict(base,PIPE_NOOPT='1',PIPE_PROVOKING=prov,PIPE_CULL='back',RADV_BC250_MESH_DIRECT_READ=direct_read,BC250_MESH_NIR_DUMP=str(dump),RADV_BC250_DIAG_BARY_NO_REF=str(int(mode=='split-noref')))
    for flag in ('FAST','PIECES','OWNED','LOCAL','CHECK','CORNERS','AUTOCULL','PARALLEL','BARY','BARY_TINY','BARY_AFFINE'):e['RADV_BC250_MESH_SAFE_'+flag]=str(int(mode=='on'))
    t=run([O/'pipe',d/'mesh.spv',d/'frag.spv','-','1'],d/(mode+'.log'),e)
    assert 'SUBMIT_OK' in t and 'PIPELINE_RESULT=0' in t
    assert not re.search(r'validation failed|Validation Error|VUID-|NIR_VALIDATE|Assertion .*failed',t)
    if mode=='on':assert 'fallback_needed=0' in t and 'SAFE BARY TINY:' in t and 'rotation from reference' in t,(name,t[-3000:])
   a=next((d/'split').glob('*.nir'));b=next((d/'on').glob('*.nir'));z=next((d/'split-noref').glob('*.nir'))
   t=run([O/'oracle','--bary-ref','--seeds',os.getenv('SEEDS','16'),z,a],d/'ref-alias.log');assert 'ORACLE PASS' in t,t[-2000:]
   t=run([O/'oracle','--compact','--all-corners','--safe-owned','--bary-rot','--w','2','--seeds',os.getenv('SEEDS','16'),a,b],d/'oracle.log');assert 'ORACLE PASS' in t and 'candidate_ub=0' in t and 'both_ub=0' in t,t[-2000:]
   if p>64:
    t=run([O/'oracle','--compact','--all-corners','--safe-owned','--w','2','--geometry','--geometry-params','--grid','--pieces-a','2','--pieces-b','2','--seeds','8',a,b],d/'order.log');assert 'ORACLE GEOMETRY PASS' in t and 'candidate_ub=0' in t,t[-2000:]
   rows.append(dict(case=name,status='PASS'));print(name,'PASS',flush=True)
   (O/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')


# The affine proof must reject alternate but valid connectivity, partial stores
# and counts derived from input. No CPU-generated replacement shader is used.
mesh = """#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x=32) in;
layout(lines,max_vertices=256,max_primitives=256) out;
layout(location=0) out vec4 tag[];
layout(push_constant) uniform PC {uint count;} pc;
void main(){uint i=gl_LocalInvocationIndex;SetMeshOutputsEXT(256,128);
for(uint v=i;v<256;v+=32){gl_MeshVerticesEXT[v].gl_Position=vec4(float(v),float(v%3),0,1);tag[v]=vec4(float(v),2,3,4);}
for(uint p=i;p<128;p+=32){gl_PrimitiveLineIndicesEXT[p]=uvec2(2*p,2*p+1);}}
"""
frag = """#version 460
#extension GL_EXT_fragment_shader_barycentric : require
layout(location=0) pervertexEXT in vec4 tag[2];
layout(location=0) out vec4 color;
void main(){color=tag[0]*gl_BaryCoordEXT.x+tag[1]*gl_BaryCoordEXT.y;}
"""
variants={
 "affine":mesh,
 "reverse":mesh.replace("uvec2(2*p,2*p+1)","uvec2(2*p+1,2*p)"),
 "dynamic-count":mesh.replace("SetMeshOutputsEXT(256,128)","SetMeshOutputsEXT(2*pc.count,pc.count)"),
 "extra-vertices":mesh.replace("SetMeshOutputsEXT(256,128)","SetMeshOutputsEXT(256,127)"),
 "partial-stores":mesh.replace("gl_PrimitiveLineIndicesEXT[p]=uvec2(2*p,2*p+1);","gl_PrimitiveLineIndicesEXT[p].x=2*p;gl_PrimitiveLineIndicesEXT[p].y=2*p+1;"),
}
for name,source in variants.items():
 d=O/name;d.mkdir(exist_ok=True)
 for st,src in (("mesh",source),("frag",frag)):
  (d/("shader."+st)).write_text(src)
  run(["glslangValidator","--target-env","vulkan1.3","-S",st,"-o",d/(st+".spv"),d/("shader."+st)],d/(st+"-build.log"))
 for noopt in (0,1):
  for mode in ("split","on"):
   dump=d/(mode+str(noopt));dump.mkdir(exist_ok=True)
   e=dict(base,PIPE_NOOPT=str(noopt),PIPE_PROVOKING="last",BC250_MESH_NIR_DUMP=str(dump),RADV_BC250_MESH_DIRECT_READ="full")
   for flag in ("FAST","PIECES","OWNED","LOCAL","CHECK","CORNERS","AUTOCULL","PARALLEL","BARY","BARY_AFFINE"):
    e["RADV_BC250_MESH_SAFE_"+flag]=str(int(mode=="on"))
   e["RADV_BC250_MESH_SAFE_BARY_TINY"]="0"
   if not noopt:e.pop("PIPE_NOOPT",None)
   text=run([O/"pipe",d/"mesh.spv",d/"frag.spv","-","1"],d/(mode+str(noopt)+".log"),e)
   if mode=="on":
    admitted="SAFE BARY AFFINE:" in text
    # Optimization is allowed to combine the two partial stores into a full
    # tuple; both outcomes are safe. Changed topology/count must never pass.
    if name!="partial-stores":assert admitted==(name=="affine"),(name,text[-2000:])
    if admitted:
     a=next((d/("split"+str(noopt))).glob("*.nir"));b=next(dump.glob("*.nir"))
     proof=run([O/"oracle","--compact","--all-corners","--safe-direct","--w","1","--seeds","16",a,b],d/("oracle"+str(noopt)+".log"))
     assert "ORACLE PASS" in proof and re.search(r"compared=[1-9]",proof),proof[-2000:]
  rows.append(dict(case=name,noopt=noopt,status="PASS"))
  (O/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
print("safe-bary-rest PASS",flush=True)
