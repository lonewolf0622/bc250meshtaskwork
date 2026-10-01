"""Merged Mesh indirect helper keeps application root constants in DGC memory."""
import os,re,json,struct,subprocess
from pathlib import Path
src=Path(__file__).resolve().parents[3];o=Path(os.environ['KEEP']);o.mkdir(parents=True,exist_ok=True);os.chdir(o)
assert not Path('/dev/dri').exists() and os.getenv('AMDGPU_GPU_ID')=='gfx1013'
e=dict(os.environ,VK_DRIVER_FILES=os.environ['ICD'],RADV_DIRECTMESH='1',RADV_BC250_EXPOSE_DGC='1',RADV_BC250_MESH_MERGE='1',RADV_BC250_MESH_SAFE_FAST='0',RADV_BC250_MESH_SAFE_OWNED='0',RADV_BC250_MESH_SAFE_PIECES='0',BC250_TRACE_COMPILE='1',NIR_DEBUG='validate',ACO_DEBUG='validateir,validatera',DGC_PUSH_CONSTANTS='1',DGC_PREPROCESS='1')
def run(args,name,env=e):
 p=subprocess.run(list(map(str,args)),env=env,capture_output=True,text=True);t=p.stdout+p.stderr;(o/(name+'.log')).write_text(t)
 assert not p.returncode and not re.search(r'Assertion .*failed|validation failed|NIR_VALIDATE|raw_unproven',t),(name,p.returncode,t[-3000:])
 return t
run(['cc','-O1','-o',o/'pipe',src/'tests/bc250-mesh/dgc/pipe.c','-lvulkan'],'build')
run(['python3',src/'tests/bc250-mesh/dgc/build_oracle.py',o/'oracle'],'oracle-build')
for stage in ('mesh','frag'):
 run(['glslangValidator','--target-env','vulkan1.3','-S',stage,'-DWS=32','-DNV=32','-DNP=32','-o',o/(stage+'.spv'),src/('tests/bc250-mesh/mesh-merge/shape.'+stage)],stage+'-build')
for option in ('a','b'):
 dump=o/option;dump.mkdir(exist_ok=True)
 t=run([o/'pipe',o/'mesh.spv',o/'frag.spv','-',1],option,dict(e,RADV_BC250_MESH_MERGE_INDIRECT=option,BC250_DGC_DUMP=str(dump)))
 assert 'SUBMIT_OK' in t and re.search(r'merge_k=[2-9]',t),t[-3000:]
 for count in (0,1):
  run([o/'oracle',dump,count,0,1],option+'-cpu-'+str(count))
  data=(dump/f'capture-{count}.bin').read_bytes();p=struct.unpack_from('<QQQQIIII',data)
  for seq in range(p[4]):
   words=struct.unpack_from('<'+'I'*(p[6]//4),data,48+seq*p[5]);i=0;loads=[]
   while i<len(words):
    h=words[i];n=1 if h==0xffff1000 or h>>30==2 else ((h>>16)&0x3fff)+2
    if h>>8&255==0x63:loads.append(words[i+1]|words[i+2]<<32)
    i+=n
   app=p[1]+seq*p[5]+p[6]+((4+p[7]*12+15)&~15)
   assert app in loads,(app,loads)
(o/'summary.json').write_text(json.dumps({'merged_root_constants':True,'options':['a','b'],'cpu_cases':64},indent=2)+'\n');print('Merged Mesh DGC PASS')
