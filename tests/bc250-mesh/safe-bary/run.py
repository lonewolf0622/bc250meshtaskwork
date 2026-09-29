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
for v,p,lanes in ((6,6,32),(32,32,32),(192,64,64),(256,128,128)):
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
    for flag in ('FAST','PIECES','OWNED','LOCAL','CHECK','CORNERS','AUTOCULL','PARALLEL','BARY'):e['RADV_BC250_MESH_SAFE_'+flag]=str(int(mode=='on'))
    t=run([O/'pipe',d/'mesh.spv',d/'frag.spv','-','1'],d/(mode+'.log'),e)
    assert 'SUBMIT_OK' in t and 'PIPELINE_RESULT=0' in t
    assert not re.search(r'validation failed|Validation Error|VUID-|NIR_VALIDATE|Assertion .*failed',t)
    if mode=='on' and v==6:assert 'reason=bary_small_class_cost_not_better_than_split' in t,t[-3000:]
    elif mode=='on':assert 'fallback_needed=0' in t and 'MESH SAFE CORNERS:' in t and 'rotation from reference' in t and 'provoking='+('dynamic' if prov.startswith('dynamic') else prov) in t,(name,t[-3000:])
   a=next((d/'split').glob('*.nir'));b=next((d/'on').glob('*.nir'));z=next((d/'split-noref').glob('*.nir'))
   t=run([O/'oracle','--bary-ref','--seeds',os.getenv('SEEDS','16'),z,a],d/'ref-alias.log');assert 'ORACLE PASS' in t,t[-2000:]
   if v==6:
    t=run([O/'oracle','--seeds',os.getenv('SEEDS','16'),a,b],d/'oracle.log');assert 'ORACLE PASS' in t,t[-2000:]
    rows.append(dict(case=name,status='PASS',route='split_cost'));print(name,'PASS fallback',flush=True);(O/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');continue
   t=run([O/'oracle','--compact','--all-corners','--safe-owned','--bary-rot','--w','2','--seeds',os.getenv('SEEDS','16'),a,b],d/'oracle.log');assert 'ORACLE PASS' in t and 'candidate_ub=0' in t and 'both_ub=0' in t,t[-2000:]
   if p>64:
    t=run([O/'oracle','--compact','--all-corners','--safe-owned','--w','2','--geometry','--geometry-params','--grid','--pieces-a','2','--pieces-b','2','--seeds','8',a,b],d/'order.log');assert 'ORACLE GEOMETRY PASS' in t and 'candidate_ub=0' in t,t[-2000:]
   rows.append(dict(case=name,status='PASS'));print(name,'PASS',flush=True)
   (O/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')

# These exclusions must remain transactional: shared corners and a missing
# reference convention never reach the direct exporter.
d=O/'v32p32-first-drfull'
if not os.getenv('ONLY'):
 for name,extra,reason in [('shared',{'RADV_BC250_MESH_SAFE_CORNERS':'0'},'bary_private_corner_shape_or_budget_unproven'),('no-ref',{'RADV_BC250_DIAG_BARY_NO_REF':'1'},'bary_two_reference_parameters_unavailable')]:
  e=dict(base)
  for flag in ('FAST','PIECES','OWNED','LOCAL','CHECK','CORNERS','AUTOCULL','PARALLEL','BARY'):e['RADV_BC250_MESH_SAFE_'+flag]='1'
  e.update(extra)
  t=run([O/'pipe',d/'mesh.spv',d/'frag.spv','-','1'],O/(name+'.log'),e)
  assert 'fallback_needed=1' in t and 'reason='+reason in t,t[-2000:]
  rows.append(dict(case=name,status='PASS'))
 (O/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')

if not os.getenv('ONLY'):
 d=O/'full-params';d.mkdir(exist_ok=True)
 mesh='#version 460\n#extension GL_EXT_mesh_shader : require\nlayout(local_size_x=32) in;layout(triangles,max_vertices=3,max_primitives=1) out;\n'
 frag='#version 460\n#extension GL_EXT_fragment_shader_barycentric : require\n'
 for i in range(31):
  mesh+=f'layout(location={i}) out vec4 v{i}[];\n'
  frag+=f'layout(location={i}) '+('pervertexEXT in vec4 v0[3];\n' if i==0 else f'in vec4 v{i};\n')
 mesh+='void main(){SetMeshOutputsEXT(3,1);uint v=gl_LocalInvocationIndex;if(v<3){gl_MeshVerticesEXT[v].gl_Position=vec4(float(v),float(v*v),0,1);'
 for i in range(31):mesh+=f'v{i}[v]=vec4(float(v+{i}u));'
 mesh+='}if(v==0)gl_PrimitiveTriangleIndicesEXT[0]=uvec3(0,1,2);}'
 frag+='layout(location=0) out vec4 color;void main(){color=v0[0]+v0[1]+v0[2]'+''.join(f'+v{i}' for i in range(1,31))+';}'
 for stage,source in [('mesh',mesh),('frag',frag)]:
  (d/('src.'+stage)).write_text(source)
  run(['glslangValidator','--target-env','vulkan1.3','-S',stage,'-o',d/(stage+'.spv'),d/('src.'+stage)],d/(stage+'-build.log'))
 e=dict(base)
 for flag in ('FAST','PIECES','OWNED','LOCAL','CHECK','CORNERS','AUTOCULL','PARALLEL','BARY'):e['RADV_BC250_MESH_SAFE_'+flag]='1'
 t=run([O/'pipe',d/'mesh.spv',d/'frag.spv','-','1'],d/'on.log',e)
 assert 'fallback_needed=1' in t and 'reason=bary_two_reference_parameters_unavailable' in t,t[-3000:]
 rows.append(dict(case='full-params',status='PASS'))
 (O/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')

if not os.getenv('ONLY'):
 # Strict per-vertex input without any barycentric-weight intrinsic.
 d=O/'pervertex-only';d.mkdir(exist_ok=True)
 mesh=(mesh[:mesh.index('layout(location=1)')]+mesh[mesh.index('void main()'):]).replace('max_primitives=1','max_primitives=32')
 mesh=re.sub(r'v(?:[1-9]|[12][0-9]|30)\[v\]=vec4\(float\(v\+\d+u\)\);','',mesh)
 frag='#version 460\n#extension GL_EXT_fragment_shader_barycentric : require\nlayout(location=0) pervertexEXT in vec4 v0[3];layout(location=0) out vec4 color;void main(){color=v0[0]+2.0*v0[1]+4.0*v0[2];}'
 for stage,source in [('mesh',mesh),('frag',frag)]:
  (d/('src.'+stage)).write_text(source)
  run(['glslangValidator','--target-env','vulkan1.3','-S',stage,'-o',d/(stage+'.spv'),d/('src.'+stage)],d/(stage+'-build.log'))
 for corners in (0,1):
  t=run([O/'pipe',d/'mesh.spv',d/'frag.spv','-','1'],d/(str(corners)+'.log'),dict(e,RADV_BC250_MESH_SAFE_CORNERS=str(corners)))
  assert ('fallback_needed='+str(1-corners)) in t,t[-3000:]
  if corners:assert 'rotation from reference' in t and 'SAFE CORNERS:' in t,t[-3000:]
 rows.append(dict(case='pervertex-only',status='PASS'))
 (O/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
