"""Create fresh named hardware gates. Never overwrite a freeze."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent
original=here.parent/'gates'
freeze,candidate,policy,prerequisite=map(Path,sys.argv[1:])
freeze.mkdir()
shutil.copyfile(candidate,freeze/'candidate.so')
(freeze/'candidate.json').write_text(json.dumps({'file_format_version':'1.0.0','ICD':{'library_path':str((freeze/'candidate.so').resolve()),'api_version':'1.4.305'}})+'\n')
shutil.copyfile(policy,freeze/'policy.json')
(freeze/'prerequisites').mkdir()
assert 'PASS images byte-identical' in prerequisite.read_text()
shutil.copyfile(prerequisite,freeze/'prerequisites/D2-hardware.last')
for name in ('run_once.py','audit.py'):
 shutil.copyfile(here/name,freeze/name)
shutil.copyfile(here.parents[1]/'descriptor-buffer-pointer-v2/check.py',freeze/'pointer_check.py')
text=(original/'gate.c').read_text()
needle='      puts("GATE_DESCRIPTOR_BUFFERS root_cbv=13 root_constant=7 resource=11 producer_and_consumer=1");'
assert text.count(needle)==1
text=text.replace(needle,needle+'\n      void *maps[]={resource_map,table_map}; VkBuffer buffers[]={resource,table};\n      for(unsigned b=0;b<2;b++) {\n         fprintf(stderr,"BC250MEM va=%llx bytes=256 data=",(unsigned long long)dgc_address(buffers[b]));\n         for(unsigned i=0;i<256;i++)fprintf(stderr,"%02x",((unsigned char *)maps[b])[i]);\n         fputc(\'\\n\',stderr);\n      }')
(freeze/'gate.c').write_text(text)
subprocess.run(['cc','-O1','-Wall','-o',str(freeze/'gate'),str(freeze/'gate.c'),'-lvulkan'],check=True)
for name in ('descriptor-buffers-v2','many-task-v2'):
 out=freeze/name;out.mkdir()
 defs=['-DTASK=1','-DDB='+str(int(name.startswith('descriptor'))),'-DMANY='+str(int(name.startswith('many')))]
 for stage in ('mesh','frag','task'):
  shutil.copyfile(original/('gate.'+stage),out/('gate.'+stage))
  subprocess.run(['glslangValidator','--target-env','vulkan1.3','-S',stage,*defs,'-o',str(out/(stage+'.spv')),str(out/('gate.'+stage))],check=True)
pins={str(p.relative_to(freeze)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(freeze.rglob('*')) if p.is_file()}
(freeze/'gate-manifest.json').write_text(json.dumps({'sha256':pins,'hardware_runs':0,'gates':['descriptor-buffers-v2','many-task-v2']},indent=2)+'\n')
