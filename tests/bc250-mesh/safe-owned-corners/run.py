"""Compare owned private corners to the actual split/expansion lowering on the CPU."""
import os,re,json,subprocess
from pathlib import Path
S=Path(__file__).resolve().parents[3];B=Path(os.environ['BUILD']);O=Path(os.environ['KEEP']);O.mkdir(exist_ok=True,parents=True)
assert not Path('/dev/dri').exists() and os.getenv('AMDGPU_GPU_ID')=='gfx1013'
base=dict(os.environ)
base.update(x.split('=',1) for x in re.search(r"policy='([^']+)'",(S/'tests/bc250-mesh/safe-direct/run.py').read_text())[1].split())
icd=os.environ['ICD']
base.update(VK_DRIVER_FILES=icd,VK_ICD_FILENAMES=icd,NIR_DEBUG='validate',ACO_DEBUG='validateir,validatera',BC250_TRACE_COMPILE='1',RADV_BC250_MESH_SAFE_CORNERS='1',RADV_BC250_MESH_SAFE_PARALLEL='1',RADV_BC250_MESH_SAFE_AUTOCULL='1',RADV_BC250_MESH_AUTOCULL='1',RADV_BC250_MESH_AUTOCULL_WIDE='1',RADV_BC250_MESH_DIRECT_READ='full',BC250_CAPTURE_POLICY_SHADERS='1',RADV_DEBUG='dumpibs',RADV_BC250_PERF_PIECE_PRIMS='64',RADV_BC250_MESH_NO_SPLIT='1')
def run(args,log,env=base):
 with log.open('w') as f:p=subprocess.run(list(map(str,args)),env=env,stdout=f,stderr=subprocess.STDOUT)
 t=log.read_text();assert p.returncode==0,(log,p.returncode,t[-2500:]);return t
run(['cc','-O1','-Wall','-o',O/'pipe',S/'tests/bc250-mesh/safe-direct/pipe.c','-lvulkan'],O/'pipe-build.log')
run(['bash',S/'tests/bc250-mesh/direct-read/build_oracle.sh',O/'oracle'],O/'oracle-build.log',dict(base,SRC=str(S),BUILD=str(B)))
cases=[
 ('owned32',['-DVERTS=32','-DPRIMS=32','-DLANES=32','-DPERPRIM=1','-DNOCULL=1'],False,'owned'),
 ('owned_id',['-DVERTS=64','-DPRIMS=64','-DLANES=64','-DPERPRIM=1','-DPRIMID=1','-DNOCULL=1'],False,'owned'),
 ('owned_task',['-DVERTS=32','-DPRIMS=32','-DLANES=32','-DPERPRIM=1','-DNOCULL=1','-DTASK=1'],True,'owned'),
 ('owned_counts',['-DVERTS=32','-DPRIMS=32','-DLANES=64','-DPERPRIM=1','-DNOCULL=1','-DPARTIAL=1'],False,'owned'),
 ('owned_pieces',['-DVERTS=64','-DPRIMS=128','-DLANES=128','-DPERPRIM=1','-DNOCULL=1'],False,'pieces')]

rows=json.loads((O/'summary.json').read_text()) if os.getenv('ONLY') and (O/'summary.json').exists() else []
for name,defs,task,kind in cases:
 if os.getenv('ONLY') and not re.search(os.environ['ONLY'],name):continue
 d=O/name;d.mkdir(exist_ok=True)
 source=S/'tests/bc250-mesh/compact'
 if name=='owned_counts':
  for st in ('mesh','frag'):
   (d/('cmp.'+st)).write_text((source/('cmp.'+st)).read_text().replace('location = 2) out vec4 p0','location = 3) out vec4 p0').replace('location = 2) in vec4 p0','location = 3) in vec4 p0'))
  source=d
 for st in ('mesh','frag'):
  run(['glslangValidator','--target-env','vulkan1.3','-S',st,*defs,'-o',d/(st+'.spv'),source/('cmp.'+st)],d/(st+'-build.log'))
 taskspv='-'
 if task:
  taskspv=d/'task.spv';run(['glslangValidator','--target-env','vulkan1.3','-S','task','-o',taskspv,source/'cmp.task'],d/'task-build.log')
 for mode in ('ref','on'):
  dump=d/mode;dump.mkdir(exist_ok=True)
  for p in dump.glob('*.nir'):p.unlink()
  env=dict(base,PIPE_NOOPT='1',RADV_BC250_MESH_SAFE_FAST=str(int(mode=='on' and kind=='fast')),RADV_BC250_MESH_SAFE_DIRECT='0',RADV_BC250_MESH_SAFE_PIECES=str(int(mode=='on' and kind=='pieces')),RADV_BC250_MESH_SAFE_OWNED=str(int(mode=='on')),BC250_MESH_NIR_DUMP=str(dump))
  t=run([O/'pipe',d/'mesh.spv',d/'frag.spv',taskspv,'1'],d/(mode+'.log'),env)
  assert 'PIPELINE_RESULT=0' in t and 'SUBMIT_OK' in t
  assert not re.search(r'validation failed|Validation Error|VUID-|NIR_VALIDATE|Assertion .*failed',t)
  if mode=='on':assert ('MESH SAFE '+kind.upper()+':') in t,(name,t[-2000:])
 ref=sorted((d/'ref').glob('*.nir'))[-1];on=sorted((d/'on').glob('*.nir'))[-1]
 t=run([O/'oracle','--compact','--safe-owned','--w','31','--seeds',os.getenv('SEEDS','16'),ref,on],d/'oracle.log')
 assert 'ORACLE PASS' in t and 'candidate_ub=0' in t and 'both_ub=0' in t,(name,t[-2500:])
 if kind=='pieces':
  pieces=2
  t=run([O/'oracle','--compact','--safe-owned','--w','31','--geometry','--geometry-params','--grid','--pieces-a',str(pieces),'--pieces-b',str(pieces),'--seeds','16',ref,on],d/'geometry-order.log')
  assert 'ORACLE GEOMETRY PASS' in t and 'candidate_ub=0' in t and re.search(r'compared=[1-9]',t),t[-2000:]
 rows=[r for r in rows if r['case']!=name]+[dict(case=name,status='PASS')];print(name,'PASS',flush=True)
 (O/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
