"""Execution query scopes, inherited graphics state and both predicate senses."""
import os,json,re,struct,subprocess
from pathlib import Path
src=Path(__file__).resolve().parents[3];out=Path(os.environ['KEEP']);out.mkdir(parents=True,exist_ok=True);os.chdir(out)
assert not Path('/dev/dri').exists() and os.getenv('AMDGPU_GPU_ID')=='gfx1013'
e=dict(os.environ,VK_DRIVER_FILES=os.environ['ICD'],RADV_DIRECTMESH='1',RADV_BC250_EXPOSE_DGC='1',NIR_DEBUG='validate',ACO_DEBUG='validateir,validatera',BC250_TRACE_COMPILE='1',DGC_PUSH_CONSTANTS='1')
def run(args,name,env=e):
 p=subprocess.run(list(map(str,args)),env=env,capture_output=True,text=True);t=p.stdout+p.stderr;(out/(name+'.log')).write_text(t)
 assert not p.returncode and not re.search(r'Assertion .*failed|validation failed|NIR_VALIDATE|raw_unproven',t),(name,p.returncode,t[-3000:])
 return t
run(['cc','-O1','-Wall','-o',out/'pipe',src/'tests/bc250-mesh/dgc/pipe.c','-lvulkan'],'build')
run(['cc','-O1','-Wall','-o',out/'ordinary',src/'tests/bc250-mesh/safe-direct/pipe.c','-lvulkan'],'ordinary-build')
run(['python3',src/'tests/bc250-mesh/dgc/build_oracle.py',out/'oracle'],'oracle-build')
fixtures=Path(os.environ['FIXTURES']);rows=[]
def check_patches(dump,count,native):
 data=(dump/f'capture-{count}.bin').read_bytes();p=struct.unpack_from('<QQQQIIII',data)
 owner=(dump/f'query-owner-{count}.bin').read_bytes();w=struct.unpack('<'+'I'*(len(owner)//4),owner)
 writes={};i=0
 while i<len(w):
  h=w[i];n=1 if h==0xffff1000 or h>>30==2 else ((h>>16)&0x3fff)+2
  assert i+n<=len(w)
  if h>>8&255==0x37 and n==6:writes[w[i+2]|w[i+3]<<32]=w[i+4:i+6]
  i+=n
 sites=0;dispatches=0
 for seq in range(p[4]):
  begin=seq*p[5];words=struct.unpack_from('<'+'I'*(p[6]//4),data,48+begin);i=0
  while i<len(words):
   h=words[i];n=1 if h==0xffff1000 or h>>30==2 else ((h>>16)&0x3fff)+2
   op=h>>8&255
   if op in (0x15,0x16):dispatches+=1
   assert op!=0x69,'generated program overwrites execution graphics context'
   if op==0x46 and n>1 and words[i+1]&63 in (0x19,0x1a):
    assert n==2
    assert writes[p[1]+begin+i*4]==(h if native else 0xc0001000,words[i+1])
    sites+=1
   i+=n
 assert sites>0 or not dispatches
variants=[(q,'1',False,explicit) for q in ('pipeline','occlusion','mesh-pipeline','mesh-primitives') for explicit in (False,True)]
variants += [(None,value,inverted,True) for value in ('0','1') for inverted in (False,True)]
for shape in ('plain','task','fold'):
 args=[out/'pipe',fixtures/shape/'mesh.spv',fixtures/shape/'frag.spv',fixtures/shape/'task.spv' if shape!='plain' else '-',1]
 for query,predicate,inverted,explicit in variants:
  name=f'{shape}-{query}-{predicate}-{inverted}-{explicit}';dump=out/name;dump.mkdir(exist_ok=True)
  env=dict(e,BC250_DGC_DUMP=str(dump),DGC_CONDITIONAL=predicate)
  if query:env['DGC_QUERY']=query
  if query and query.startswith('mesh-'):env['RADV_BC250_SPLIT_MESH']='false'
  if inverted:env['DGC_INVERTED']='1'
  if explicit:env['DGC_PREPROCESS']='1'
  if shape=='fold' and query and query.startswith('mesh-'):
   # This large Mesh declaration loses its ordinary protected route when
   # replay splitting is disabled to expose Mesh queries. Preserve that gap.
   for executable in (out/'pipe',out/'ordinary'):
    p=subprocess.run([str(executable),*map(str,args[1:])],env=dict(env,PIPE_MESH_QUERIES='1'),capture_output=True,text=True)
    t=p.stdout+p.stderr;(out/(name+'-'+executable.name+'.log')).write_text(t)
    assert p.returncode==1 and 'PIPELINE_RESULT=-8' in t and 'SUBMIT_OK' not in t and 'Assertion' not in t,t[-2000:]
   rows.append(dict(shape=shape,query=query,explicit=explicit,cpu_cases=0,ordinary_protected_route_refused=True))
   (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(name,'ordinary route refusal PASS',flush=True);continue
  t=run(args,name,env);assert 'SUBMIT_OK' in t and t.count('DGC_RECORDED')==2
  for count in (0,1):
   check_patches(dump,count,query in ('pipeline','mesh-pipeline'))
   run([out/'oracle',dump,count,int(shape!='plain'),1],name+'-cpu-'+str(count),env)
   data=(dump/f'capture-{count}.bin').read_bytes();p=struct.unpack_from('<QQQQIIII',data)
   for seq in range(p[4]):
    offset=48+seq*p[5]+(p[6]-1048576 if count and shape!='plain' else 0)
    code=1048576 if shape!='plain' else p[6];words=struct.unpack_from('<'+'I'*(code//4),data,offset);i=0;draws=producers=0
    while i<len(words):
     h=words[i];n=1 if h==0xffff1000 or h>>30==2 else ((h>>16)&0x3fff)+2;op=h>>8&255
     if op in (0x4c,0x15,0x16):
      assert h&1,'ordinary draw/producer lost predicate'
      if op==0x4c:draws+=1
      else:producers+=1
     i+=n
    assert draws and (shape=='plain' or producers>=draws)
  rows.append(dict(shape=shape,query=query,predicate=int(predicate),inverted=inverted,explicit=explicit,cpu_cases=32,ordinary_pm4=True))
  (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(name,'PASS',flush=True)
 for after in (False,True):
  name=shape+('-query-after-outside-preprocess' if after else '-changed-query')
  dump=out/name;dump.mkdir(exist_ok=True)
  env=dict(e,DGC_QUERY='pipeline',DGC_PREPROCESS='1',BC250_DGC_DUMP=str(dump))
  if after:env.update(DGC_QUERY_AFTER_PREPROCESS='1',DGC_PREPROCESS_OUTSIDE='1')
  else:env['DGC_QUERY_CHANGE']='1'
  t=run(args,name,env);assert 'SUBMIT_OK' in t
  for count in (0,1):
   check_patches(dump,count,after)
   run([out/'oracle',dump,count,int(shape!='plain'),1],name+'-cpu-'+str(count),env)
  rows.append(dict(shape=shape,query_transition=True,outside_preprocess=after,cpu_cases=32))
 (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print('State proof PASS',len(rows),'routes',sum(r['cpu_cases'] for r in rows),'CPU cases; execution query transitions admitted',flush=True)
