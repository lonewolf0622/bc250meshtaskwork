"""Compile and interpret W31 latest-copy/closure fixtures. ONLY run via scripts/offline.sh."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

SRC=Path(__file__).resolve().parents[3]
BUILD=Path(os.environ['BUILD'])
OUT=Path(os.environ['KEEP']); OUT.mkdir(exist_ok=True,parents=True)
assert not Path('/dev/dri').exists()
assert os.getenv('AMDGPU_GPU_ID')=='gfx1013'
assert 'libamdgpu_noop_drm_shim.so' in os.getenv('LD_PRELOAD','')
ICD=os.environ['ICD']
OLD=os.environ.get('OLDICD',ICD)
variant=os.getenv('VARIANT','serial')
assert variant in ('serial','parallel','autocull','local','local-cull','check','check-cull','corners','corners-cull')
cull_args=['--autocull'] if variant in ('autocull','local-cull','check-cull','corners-cull') else []
base=dict(os.environ)
base.update(dict(VK_DRIVER_FILES=ICD,VK_ICD_FILENAMES=ICD,NIR_DEBUG='validate',
 ACO_DEBUG='validateir,validatera',BC250_TRACE_COMPILE='1',MESA_SHADER_CACHE_DISABLE='true'))
policy='RADV_BC250_NATIVE_TASK=0 RADV_BC250_HYBRID_TASK=1 BC250_EXPERIMENTAL_COMPUTE_CU_MODE=false BC250_BALANCED_SLICES=true BC250_SINGLE_PIECE=true BC250_CACHE_PLAN=true BC250_TRANSIENT_ARENA=true BC250_PARALLEL_CULL=true BC250_COMPACT_VERTICES=false BC250_OUTPUT_REGIONS=true BC250_EXPERIMENTAL_PRIVATE_GTT=true RADV_BC250_MESH_ONCE=false BC250_EXPERIMENTAL_ZEROED_PRIVATE_GTT=false RADV_PERFTEST=mesh,nircache BC250_EXPERIMENTAL_DIRECT_SPLIT=true BC250_EXPERIMENTAL_SKIP_INACTIVE_CHUNKS=true RADV_BC250_SPLIT_MESH=true RADV_BC250_EXPAND_PRIMITIVES=true BC250_EXPERIMENTAL_CULL_COMPACT=true BC250_EXPERIMENTAL_PACK_TRIANGLE_VERTICES=true BC250_EXPERIMENTAL_POST_MESH_VGT_FLUSH=0'
base.update(item.split('=',1) for item in policy.split())
base['VK_INSTANCE_LAYERS']='VK_LAYER_KHRONOS_validation'

def run(args,log,env=None):
 with open(log,'w') as f:
    p=subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,env=env or base)
 text=Path(log).read_text(errors='replace')
 if p.returncode: raise RuntimeError(f'{log}: exit {p.returncode}\n{text[-2500:]}')
 return text

run(['cc','-O1','-Wall','-o',OUT/'pipe',SRC/'tests/bc250-mesh/safe-direct/pipe.c','-lvulkan'],OUT/'pipe-build.log')
run(['bash',SRC/'tests/bc250-mesh/direct-read/build_oracle.sh',OUT/'mesh_oracle'],OUT/'oracle-build.log',dict(base,BUILD=str(BUILD),SRC=str(SRC)))
common=['-DLANES=32','-DVERTS=32','-DPRIMS=32']
# Every case is checked unset/0 against the unmodified frozen driver, and with
# the feature on against the expanded reference using the CPU NIR interpreter.
cases=[('d0',[],True),('plain',[],True),('v33',['-DVERTS=33'],True),
 ('v64p64',['-DVERTS=64','-DPRIMS=64','-DLANES=64'],True),
 ('v128p64',['-DVERTS=128','-DPRIMS=64','-DLANES=64'],True),
 ('v128p64t128',['-DVERTS=128','-DPRIMS=64','-DLANES=128'],True),
 ('v128p85',['-DVERTS=128','-DPRIMS=85','-DLANES=128'],True),
 ('v128p86',['-DVERTS=128','-DPRIMS=86','-DLANES=128'],False),
 ('v128p128',['-DVERTS=128','-DPRIMS=128','-DLANES=128'],False),
 ('counts',['-DCOUNTS=1'],True),('early',['-DCOUNTS=1','-DEARLY=1'],True),
 ('shared',['-DVERTS=128','-DPRIMS=64','-DLANES=64','-DSHARED=1'],True),
 ('query128',['-DVERTS=128','-DPRIMS=64','-DLANES=128','-DCOUNTS=1','-DSHARED=1'],True),
 ('shared96',['-DVERTS=128','-DPRIMS=64','-DLANES=96','-DSHARED=1'],True),
 ('shared128',['-DVERTS=128','-DPRIMS=64','-DLANES=128','-DSHARED=1'],True),
 ('effects',['-DVERTS=128','-DPRIMS=64','-DLANES=64','-DEFFECTS=1'],True),
 ('counts128',['-DVERTS=128','-DPRIMS=64','-DLANES=64','-DCOUNTS=1'],True),
 ('early128',['-DVERTS=128','-DPRIMS=64','-DLANES=64','-DCOUNTS=1','-DEARLY=1'],True),
 ('smallv256p',['-DVERTS=32','-DPRIMS=256','-DLANES=128'],False),
 ('smallv6p2',['-DVERTS=6','-DPRIMS=2'],True),
 ('t2',['-DVERTS=6','-DPRIMS=2','-DLANES=2'],True),
 ('v256p64',['-DVERTS=256','-DPRIMS=64','-DLANES=128'],True),
 ('arrayed',['-DARRAYED=1'],True),('partial',['-DPARTIAL=1'],True),
 ('payload',['-DPERPRIM=1','-DNOCULL=1'],False),('cull',['-DPERPRIM=1'],False),
 ('bary',['-DBARY=1'],False),('clip',['-DCLIP=1'],False),
 ('points',['-DPOINTS=1'],False),('lines',['-DLINES=1'],False),
 ('task',['-DTASK=1'],False)]
if variant.startswith('corners'):
 cases += [
  ('v64p16',['-DVERTS=64','-DPRIMS=16','-DLANES=64'],True),
  ('v64p16t32',['-DVERTS=64','-DPRIMS=16','-DLANES=32'],False),
  ('v64p64t128',['-DVERTS=64','-DPRIMS=64','-DLANES=128'],True),
  ('counts64',['-DVERTS=64','-DPRIMS=64','-DLANES=64','-DCOUNTS=1'],True),
  ('early64',['-DVERTS=64','-DPRIMS=64','-DLANES=64','-DCOUNTS=1','-DEARLY=1'],True),
  ('shared64',['-DVERTS=64','-DPRIMS=64','-DLANES=128','-DSHARED=1'],True),
  ('effects64',['-DVERTS=64','-DPRIMS=64','-DLANES=64','-DEFFECTS=1'],True)]
if os.getenv('ONLY'):
 cases=[c for c in cases if re.search(os.environ['ONLY'],c[0])]
summary=json.loads((OUT/'summary.json').read_text()) if os.getenv('ONLY') and (OUT/'summary.json').exists() else []
for name,extra,admit in cases:
 if variant.startswith(('local','check','corners')):
    admit = admit and name not in ('v33', 'v128p85')
 d=OUT/name; d.mkdir(exist_ok=True)
 defs=list({d.split('=')[0]:d for d in common+extra}.values())
 mesh=SRC/'tests/bc250-mesh/safe-direct/d0.mesh' if name=='d0' else SRC/'tests/bc250-mesh/safe-direct/plain.mesh'
 if name in ('arrayed','partial','payload','cull','bary','clip','points','lines','task'): mesh=SRC/'tests/bc250-mesh/compact/cmp.mesh'
 frag=SRC/'tests/bc250-mesh/compact/cmp.frag'
 for stage,source in [('mesh',mesh),('frag',frag)]:
    run(['glslangValidator','--target-env','vulkan1.3','-S',stage,*defs,'-o',d/(stage+'.spv'),source],d/(stage+'-build.log'))
 task='-'
 if name=='task':
    task=d/'task.spv'
    run(['glslangValidator','--target-env','vulkan1.3','-S','task','-o',task,SRC/'tests/bc250-mesh/compact/cmp.task'],d/'task-build.log')
 logs={}
 for stage in ('mesh','frag'):
    run(['spirv-val','--target-env','vulkan1.3',d/(stage+'.spv')],d/(stage+'-spirv-val.log'))
 for mode in ['old','ref','zero','on']+(['noref','noon'] if admit else []):
    env=dict(base,BC250_CAPTURE_POLICY_SHADERS='1',RADV_DEBUG='dumpibs')
    if name=='clip': env['RADV_BC250_MESH_ALLOW_POS1']='1'
    if name=='query128': env.update(PIPE_MESH_QUERIES='1',RADV_BC250_SPLIT_MESH='false')
    if mode=='old': env.update(VK_DRIVER_FILES=OLD,VK_ICD_FILENAMES=OLD)
    if mode in ('zero','on','noon'): env['RADV_BC250_MESH_SAFE_DIRECT']='0' if mode=='zero' else '1'
    if mode in ('on','noon') and variant in ('parallel','autocull'): env['RADV_BC250_MESH_SAFE_PARALLEL']='1'
    if mode in ('on','noon') and variant=='autocull': env['RADV_BC250_MESH_SAFE_AUTOCULL']='1'
    if mode in ('on','noon','zero') and variant.startswith(('local','check','corners')):
       env['RADV_BC250_MESH_SAFE_DIRECT']='0'
       env['RADV_BC250_MESH_SAFE_FAST']='0' if mode=='zero' else '1'
       env['RADV_BC250_MESH_SAFE_LOCAL']='0' if mode=='zero' else '1'
       if variant.startswith(('check','corners')): env['RADV_BC250_MESH_SAFE_CHECK']='0' if mode=='zero' else '1'
       if variant.startswith('corners'): env['RADV_BC250_MESH_SAFE_CORNERS']='0' if mode=='zero' else '1'
       if mode!='zero' and variant.endswith('-cull'): env['RADV_BC250_MESH_SAFE_AUTOCULL']='1'
    if mode.startswith('no'): env['PIPE_NOOPT']='1'
    dump=d/mode; dump.mkdir(exist_ok=True)
    for prior in dump.glob('*.nir'): prior.unlink()
    env['BC250_MESH_NIR_DUMP']=str(dump)
    logs[mode]=run([OUT/'pipe',d/'mesh.spv',d/'frag.spv',task,'1'],d/(mode+'.log'),env)
    assert 'PIPELINE_RESULT=0' in logs[mode] and 'SUBMIT_OK' in logs[mode],(name,mode)
    assert not re.search(r'validation failed|Validation Error|VUID-|NIR_VALIDATE|assert|error:',logs[mode],re.I),(name,mode)
 def signature(t):
    return sorted(re.sub(r' va=[0-9a-f]+','',line) for line in t.splitlines() if re.match(r'^BC250POLICY(CODE)? ',line))
 assert signature(logs['old']) and signature(logs['old'])==signature(logs['ref'])==signature(logs['zero']),name+' off identity'
 def ib_bytes(t):
    words=re.findall(r'^([0-9a-f]{8}) ',t,re.M)
    assert words, name+' missing IB words'
    return b''.join(int(x,16).to_bytes(4,'little') for x in words)
 ibs={mode:ib_bytes(t) for mode,t in logs.items()}
 for mode,raw in ibs.items(): (d/(mode+'.ib.bin')).write_bytes(raw)
 assert ibs['old']==ibs['ref']==ibs['zero'],name+' off IB byte identity'

 applied=({'serial':'MESH SAFE DIRECT: applied','parallel':'MESH SAFE PARALLEL:','autocull':'MESH SAFE AUTOCULL:',
           'local':'MESH SAFE LOCAL:', 'local-cull':'MESH SAFE LOCAL:',
           'check':'MESH SAFE CHECK:', 'check-cull':'MESH SAFE CHECK:', 'corners':'MESH SAFE CHECK:', 'corners-cull':'MESH SAFE CHECK:'}[variant]) in logs['on']
 assert applied==admit,(name,'admission',applied)
 if variant.startswith('corners'):
    private = name in ('v64p64','v64p16','v64p64t128','counts64','early64','shared64','effects64')
    assert ('MESH SAFE CORNERS:' in logs['on']) == private, (name,'corner admission')
 if not admit:
    assert signature(logs['ref'])==signature(logs['on']),name+' fallback shader identity'
    assert ibs['ref']==ibs['on'],name+' fallback IB byte identity'
 pairs=[]
 if admit:
    optrefs=sorted((d/'ref').glob('*.nir')); optcands=sorted((d/'on').glob('*.nir'))
    text=run([OUT/'mesh_oracle','--compact','--safe-direct',*cull_args,'--safety-only','--w','31','--seeds',os.getenv('SEEDS','200'),optrefs[-1],optcands[-1]],d/'safety.log')
    assert text.splitlines()[-1].startswith('ORACLE PASS') and 'both_ub=0' in text
    pairs.append('optimized safety: '+text.splitlines()[-1])
    refs=sorted(((OUT/'smallv6p2' if name=='t2' else d)/'noref').glob('*.nir')); candidates=sorted((d/'noon').glob('*.nir'))
    assert refs and candidates
    # Source shader hashes can change when the compiler key changes. Every
    # stage-1 case is unsplit; compare the final admitted compilation.
    text=run([OUT/'mesh_oracle','--compact','--safe-direct',*cull_args,'--w','31','--seeds',os.getenv('SEEDS','200'),refs[-1],candidates[-1]],d/'oracle.log')
    assert text.splitlines()[-1].startswith('ORACLE PASS') and re.search(r'compared=[1-9]',text),(name,text[-2500:])
    pairs.append(text.splitlines()[-1])
    if name in ('d0','plain','v33','v64p64','v128p64','v128p64t128','v128p85','v256p64','shared','shared96','shared128','query128','t2','effects','counts','early','counts128','early128','smallv256p','smallv6p2','counts64','early64','effects64','v64p16','v64p64t128','shared64'):
       ar,br=(optrefs[-1],optcands[-1]) if name=='d0' and variant!='autocull' else (refs[-1],candidates[-1])
       text=run([OUT/'mesh_oracle','--compact','--safe-direct',*cull_args,'--w','31','--gen','0','--seeds','1' if name=='d0' else os.getenv('SEEDS','200'),ar,br],d/'application-oracle.log')
       assert text.splitlines()[-1].startswith('ORACLE PASS')
       pairs.append(text.splitlines()[-1])
 if name=='query128':
    text=run([OUT/'mesh_oracle','--compact','--safe-direct',*cull_args,'--w','31','--query','1','--gen','0','--seeds',os.getenv('SEEDS','64'),refs[-1],candidates[-1]],d/'query.log')
    assert 'ORACLE PASS' in text and 'skipped_reference_ub=0' in text and 'candidate_ub=0' in text
    pairs.append('queries enabled: '+text.strip().splitlines()[-1])
 row={'case':name,'applied':applied,'off_shader_identity':True,'off_ib_sha256':hashlib.sha256(ibs['ref']).hexdigest(),'off_ib_bytes':len(ibs['ref']),'oracle':pairs}
 if name=='t2': row['equality_reference']='Equivalent T32 smallv6p2 geometry; API invocation statistics are not compared across these two fixtures.'
 if name=='query128': row['policy']='SPLIT_MESH=false exposes meshShaderQueries; same policy and API size in both builds.'
 summary=[r for r in summary if r['case']!=name]+[row]
 print(name,'PASS','applied' if applied else 'fallback',flush=True)
 Path(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('PASS cases=',len(summary),flush=True)
