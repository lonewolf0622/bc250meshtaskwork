"""Chained hybrid Task count programs, memory boundary and final DrawID proof."""
import os,json,re,subprocess,struct
from pathlib import Path
src=Path(__file__).resolve().parents[3];out=Path(os.environ['KEEP']);out.mkdir(parents=True,exist_ok=True);os.chdir(out)
assert not Path('/dev/dri').exists() and os.getenv('AMDGPU_GPU_ID')=='gfx1013'
e=dict(os.environ,VK_DRIVER_FILES=os.environ['ICD'],RADV_DIRECTMESH='1',RADV_BC250_EXPOSE_DGC='1',NIR_DEBUG='validate',ACO_DEBUG='validateir,validatera',BC250_TRACE_COMPILE='1')
def run(args,name,env=e):
 p=subprocess.run(list(map(str,args)),env=env,capture_output=True,text=True);t=p.stdout+p.stderr;(out/(name+'.log')).write_text(t)
 assert not p.returncode and not re.search(r'Assertion .*failed|validation failed|NIR_VALIDATE|raw_unproven',t),(name,p.returncode,t[-3000:])
 return t
run(['cc','-O1','-Wall','-o',out/'pipe',src/'tests/bc250-mesh/dgc/pipe.c','-lvulkan'],'build')
run(['python3',src/'tests/bc250-mesh/dgc/build_oracle.py',out/'oracle'],'oracle-build')
fixture=Path(os.environ['FIXTURES']);rows=[]
for name in ('task','task-pieces','fold'):
 case=fixture/name;args=[out/'pipe',case/'mesh.spv',case/'frag.spv',case/'task.spv','1']
 t=run(args,name+'-bounds',dict(e,DGC_REQUIREMENTS_ONLY='1'))
 bounds={int(n):int(b) for c,n,b in re.findall(r'DGC_BOUND count=(\d) draws=(\d+) bytes=(\d+)',t) if c=='1'}
 assert all(bounds[n]>0 for n in (1,3,4,8,64,128,256,511,512,4096)) and bounds[4097]==0,bounds
 for draws in (4,8,64):
  dump=out/(name+'-'+str(draws));dump.mkdir(exist_ok=True)
  env=dict(e,DGC_MAX_DRAWS=str(draws),DGC_PREPROCESS='1',DGC_PUSH_CONSTANTS='1',BC250_DGC_DUMP=str(dump),RADV_BC250_MESH_PP_SHARE='1',RADV_BC250_MESH_SAFE_ADAPTIVE='1')
  t=run(args,name+'-'+str(draws),env);assert 'SUBMIT_OK' in t and t.count('DGC_RECORDED')==2
  for count in (0,1):run([out/'oracle',dump,count,1,1],name+'-'+str(draws)+'-cpu-'+str(count),env)
  capture=(dump/'capture-1.bin').read_bytes();p=struct.unpack_from('<QQQQIIII',capture)
  assert p[6]==1048576+131072*draws and p[7]==draws,p
  rows.append(dict(fixture=name,draws=draws,cpu_cases=32,ordinary_pm4_equal=True,local_conditionals=True,bounds=bounds))
  (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(name,draws,'PASS',flush=True)
print('Many Task PASS',len(rows),'routes',sum(r['cpu_cases'] for r in rows),'CPU cases',flush=True)
