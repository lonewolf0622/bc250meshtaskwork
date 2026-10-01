"""Offline Mesh/Task DGC, generated-PM4 interpretation and lifetime checks."""
import json, os, re, subprocess
from pathlib import Path
assert not Path('/dev/dri').exists()
assert os.getenv('AMDGPU_GPU_ID') == 'gfx1013'
assert 'libamdgpu_noop_drm_shim.so' in os.getenv('LD_PRELOAD','')
src=Path(__file__).resolve().parents[3];out=Path.cwd();rows=[]
base=dict(os.environ,VK_DRIVER_FILES=os.environ['ICD'],RADV_DIRECTMESH='1',RADV_BC250_EXPOSE_DGC='1',
          BC250_TRACE_COMPILE='1',NIR_DEBUG='validate',ACO_DEBUG='validateir,validatera',BC250_CAPTURE_POLICY_SHADERS='1')
base.pop('VK_INSTANCE_LAYERS',None)
def run(args,log,env=base):
 p=subprocess.run(list(map(str,args)),env=env,capture_output=True,text=True)
 text=p.stdout+p.stderr;(out/log).write_text(text)
 assert p.returncode==0,(log,p.returncode,text[-3000:])
 assert not re.search(r'validation failed|NIR_VALIDATE|Assertion .*failed|raw_unproven',text),log
 return text
run(['cc','-O1','-Wall','-o',out/'pipe',src/'tests/bc250-mesh/dgc/pipe.c','-lvulkan'],'build.log')
run(['cc','-O1','-Wall','-o',out/'exposure',src/'tests/bc250-mesh/dgc/exposure.c','-lvulkan'],'exposure-build.log')
for switch, expected in ((None,0),('0',0),('1',1)):
 e=dict(base)
 if switch is None:e.pop('RADV_BC250_EXPOSE_DGC')
 else:e['RADV_BC250_EXPOSE_DGC']=switch
 run([out/'exposure',expected],'exposure-'+str(switch)+'.log',e)
run([out/'exposure',0],'exposure-native-task.log',dict(base,RADV_BC250_NATIVE_TASK='1'))
run(['python3',src/'tests/bc250-mesh/dgc/build_oracle.py',out/'oracle'],'oracle-build.log')
fixtures=[('plain',[],False),('pieces',['-DVERTS=128','-DPRIMS=128','-DLANES=128'],False),
          ('share',['-DVERTS=64','-DPRIMS=64','-DLANES=64','-DPERPRIM=1','-DNOCULL=1'],False),
          ('task',['-DTASK=1'],True),('task-pieces',['-DTASK=1','-DVERTS=128','-DPRIMS=128','-DLANES=128'],True),
          ('fold',['-DTASK=1','-DVERTS=128','-DPRIMS=128','-DLANES=128'],True)]
for name,defs,task in fixtures:
 case=out/name;case.mkdir(exist_ok=True)
 defs=list({x.split('=')[0]:x for x in ['-DVERTS=32','-DPRIMS=32','-DLANES=32']+defs}.values())
 for stage in ['mesh','frag']+(['task'] if task else []):
  source=src/('tests/bc250-mesh/pipeline-plan/fold.task' if name=='fold' and stage=='task' else 'tests/bc250-mesh/compact/cmp.'+stage)
  run(['glslangValidator','--target-env','vulkan1.3','-S',stage,*defs,'-o',case/(stage+'.spv'),source],name+'-'+stage+'.log')
 for variant in ('private','shared','adaptive'):
  # The private-corner exporter oracles remain separate inherited gaps.
  dump=case/variant;dump.mkdir(exist_ok=True)
  e=dict(base,RADV_BC250_MESH_PP_SHARE='0' if variant=='private' else '1',
         RADV_BC250_MESH_SAFE_ADAPTIVE='1' if variant=='adaptive' else '0',
         BC250_DGC_DUMP=str(dump))
  # Explicit preprocess and dynamic constants cover all three transport ABIs.
  pcs=name in ('plain','pieces','task','task-pieces','fold')
  if pcs:e['DGC_PUSH_CONSTANTS']='1'
  if variant=='adaptive':e['DGC_PREPROCESS']='1'
  text=run([out/'pipe',case/'mesh.spv',case/'frag.spv',case/'task.spv' if task else '-',1],name+'-'+variant+'.log',e)
  assert text.count('DGC_RECORDED')==2 and 'SUBMIT_OK' in text,text[-2500:]
  if name=='fold':assert 'records=(pieces,x,y)' in text,text[-2500:]
  for count in (0,1):
   run([out/'oracle',dump,count,int(task),int(pcs)],name+'-'+variant+'-cpu-'+str(count)+'.log',e)
  rows.append(dict(fixture=name,policy=variant,task=task,pc=pcs,explicit=variant=='adaptive',cpu_cases=32,pm4_equal=True))
  (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
  print(name,variant,'PASS',flush=True)
print('DGC suite PASS',len(rows),'routes',sum(r['cpu_cases'] for r in rows),'CPU cases',flush=True)

# The capture scratch ABI is not admitted for concurrent reuse. Recording must
# return FEATURE_NOT_PRESENT rather than passing through to generic native PM4.
e=dict(base,DGC_REJECT_SIMULT='1')
p=subprocess.run([str(out/'pipe'),str(out/'plain/mesh.spv'),str(out/'plain/frag.spv'),'-','1'],env=e,capture_output=True,text=True)
text=p.stdout+p.stderr;(out/'refuse-simultaneous.log').write_text(text)
assert p.returncode==1 and 'FAIL vkEndCommandBuffer(cb) = -8' in text and 'SUBMIT_OK' not in text,text[-3000:]
assert 'raw_unproven' not in text and 'Assertion' not in text
print('DGC concurrent reuse fail-closed PASS',flush=True)
