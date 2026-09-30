"""vkd3d-style descriptor-buffer heaps, embedded samplers, root CBV and constants."""
import json,os,re,struct,subprocess
from pathlib import Path
src=Path(__file__).resolve().parents[3];out=Path(os.environ.get('KEEP','.')).resolve();out.mkdir(parents=True,exist_ok=True);os.chdir(out)
assert not Path('/dev/dri').exists() and os.getenv('AMDGPU_GPU_ID')=='gfx1013'
base=dict(os.environ,VK_DRIVER_FILES=os.environ['ICD'],RADV_DIRECTMESH='1',RADV_BC250_EXPOSE_DGC='1',
 NIR_DEBUG='validate',ACO_DEBUG='validateir,validatera',BC250_TRACE_COMPILE='1',DGC_PUSH_CONSTANTS='1')
base.pop('VK_INSTANCE_LAYERS',None)
def run(args,log,env=base):
 p=subprocess.run(list(map(str,args)),env=env,capture_output=True,text=True);t=p.stdout+p.stderr
 (out/log).write_text(t)
 assert p.returncode==0 and not re.search(r'validation failed|NIR_VALIDATE|Assertion .*failed|raw_unproven',t),(log,p.returncode,t[-3000:])
 return t
run(['cc','-O1','-Wall','-DDGC_BINDING_FIXTURE','-o',out/'binding-pipe',src/'tests/bc250-mesh/dgc/pipe.c','-lvulkan'],'binding-build.log')
run(['python3',src/'tests/bc250-mesh/dgc/build_oracle.py',out/'oracle'],'binding-oracle-build.log')
fb=src/'tests/bc250-mesh/fast-binding'
mesh_source=out/'binding.mesh'
mesh_source.write_text((fb/'fb.mesh').read_text().replace('uint g = gl_WorkGroupID.x + pc.base + gl_DrawID;',
 '#if TASK\n  uint g = gl_WorkGroupID.x + pc.base;\n#else\n  uint g = gl_WorkGroupID.x + pc.base + gl_DrawID;\n#endif'))
for name,defs in [('small',{'WS':64,'NV':64,'NP':64}),('nanite',{'WS':128,'NV':256,'NP':128,'PERPRIM':1}),('tsmall',{'WS':64,'NV':64,'NP':64,'TASK':1}),('tnanite',{'WS':128,'NV':256,'NP':128,'PERPRIM':1,'TASK':1})]:
 run(['glslangValidator','--target-env','vulkan1.3','-S','mesh',*[f'-D{k}={v}' for k,v in defs.items()],'-o',out/(name+'.mesh.spv'),mesh_source],name+'-build.log')
for stage in ('frag','task','comp','vert'):
 run(['glslangValidator','--target-env','vulkan1.3','-S',stage,'-o',out/('fb.'+stage+'.spv'),fb/('fb.'+stage)],stage+'-build.log')
run(['glslangValidator','--target-env','vulkan1.3','-S','frag','-DPERPRIM=1','-o',out/'fbpp.frag.spv',fb/'fb.frag'],'frag-perprim-build.log')
rows=[]
for mode in ('db','legacydb'):
 for shape,script,task in [('mesh','B m sc E',False),('task','B t tc E',True),('task-pieces','B T tc E',True)]:
  dump=out/(mode+'-'+shape);dump.mkdir(exist_ok=True)
  env=dict(base,BC250_DGC_DUMP=str(dump),RADV_BC250_MESH_PP_SHARE='1',RADV_BC250_MESH_SAFE_ADAPTIVE='1',DGC_PREPROCESS='1')
  text=run([out/'binding-pipe',mode,script],mode+'-'+shape+'.log',env)
  assert 'DONE' in text and text.count('DGC_RECORDED')==4 and 'SUBMIT_OK' in text,text[-3000:]
  for count in (0,1):
   run([out/'oracle',dump,count,int(task),1],mode+'-'+shape+'-cpu-'+str(count)+'.log',env)
   capture=(dump/f'capture-{count}.bin').read_bytes();params=struct.unpack_from('<QQQQIIII',capture);seqs,stride,code,records=params[4:]
   for seq in range(seqs):
    bindings=json.loads((dump/f'bindings-{count}-{seq}.json').read_text())
    assert any(b['set']==0 and b['stage']==0 for b in bindings),bindings
    assert any(b['set']==3 and b['stage']==0 for b in bindings),bindings
    if task:assert {b['set'] for b in bindings if b['stage']==2} >= {0,3},bindings
    root=[b['va'] for b in bindings if b['set']==3]
    assert root and all(params[1]+seq*stride+code<=v<params[1]+(seq+1)*stride for v in root),root
    words=struct.unpack_from('<'+'I'*(code//4),capture,48+seq*stride);i=0;regs={};consumers=producers=0
    while i<len(words):
     h=words[i];n=1 if h==0xffff1000 or h>>30==2 else ((h>>16)&0x3fff)+2;op=h>>8&255
     assert i+n<=len(words)
     if op==0x76:
      reg=0xb000+(words[i+1]&0xffff)*4
      for w in words[i+2:i+n]:regs[reg]=w;reg+=4
     if op==0x4c:
      consumers+=1
      for b in bindings:
       if b['stage'] in (0,1):assert regs.get(b['reg'])==b['va']&0xffffffff,(b,regs.get(b['reg']))
     if op==0x16 and task:
      producers+=1
      for b in bindings:
       if b['stage']==2:assert regs.get(b['reg'])==b['va']&0xffffffff,(b,regs.get(b['reg']))
     i+=n
    assert consumers and (not task or producers==consumers),(consumers,producers)
  rows.append(dict(binding=mode,shape=shape,task=task,cpu_cases=32,descriptor_sgprs=True,root_cbv_preprocess_owned=True,explicit_lifetime=True))
  (out/'binding-summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(mode,shape,'PASS',flush=True)
print('Binding proof PASS',len(rows),'routes',sum(r['cpu_cases'] for r in rows),'CPU cases')
