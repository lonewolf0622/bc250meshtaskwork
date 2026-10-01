"""Descriptor-buffer-pointer-v2: MONO, CPU/noop only, no hardware entry point."""
import json, os, re, subprocess
from pathlib import Path
from check import check, mutations
here = Path(__file__).resolve().parent
source = here.parent / 'dgc/gates'
out = Path.cwd()
assert not Path('/dev/dri').exists()
assert os.environ['AMDGPU_GPU_ID'] == 'gfx1013'
assert 'libamdgpu_noop_drm_shim.so' in os.environ['LD_PRELOAD']
# Reuse the exact draw/descriptor fixture, but request only MONO features.
# No DGC extension or feature is needed for this ordinary application draw.
text = (source / 'gate.c').read_text()
def replace(old, new):
 global text
 assert text.count(old) == 1, old
 text = text.replace(old, new)
replace('.pNext=&dgc_enable,.descriptorBuffer=VK_TRUE', '.pNext=&maintenance5,.descriptorBuffer=VK_TRUE')
replace('const char *extensions[]={VK_EXT_MESH_SHADER_EXTENSION_NAME,VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME,VK_KHR_MAINTENANCE_5_EXTENSION_NAME,VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};', 'const char *extensions[]={VK_EXT_MESH_SHADER_EXTENSION_NAME,VK_KHR_MAINTENANCE_5_EXTENSION_NAME,VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};')
replace('.enabledExtensionCount=gate_db?5:3', '.enabledExtensionCount=4')
replace('   puts("GATE_DESCRIPTOR_BUFFERS root_cbv=13 root_constant=7 resource=11 producer_and_consumer=1");', '   puts("GATE_DESCRIPTOR_BUFFERS root_cbv=13 root_constant=7 resource=11 producer_and_consumer=1");\n      void *maps[]={resource_map,table_map}; VkBuffer buffers[]={resource,table};\n      for(unsigned b=0;b<2;b++) {\n         fprintf(stderr,"BC250MEM va=%llx bytes=256 data=",(unsigned long long)dgc_address(buffers[b]));\n         for(unsigned i=0;i<256;i++)fprintf(stderr,"%02x",((unsigned char *)maps[b])[i]);\n         fputc(\'\\n\',stderr);\n      }')
(out/'mono.c').write_text(text)
subprocess.run(['cc','-O1','-Wall','-o',str(out/'mono'),str(out/'mono.c'),'-lvulkan'],check=True)
for stage in ('mesh','frag','task'):
 subprocess.run(['glslangValidator','--target-env','vulkan1.3','-S',stage,'-DTASK=1','-DDB=1','-DMANY=0','-o',str(out/(stage+'.spv')),str(source/('gate.'+stage))],check=True)
env=dict(os.environ, VK_DRIVER_FILES=os.environ['ICD'], MESA_SHADER_CACHE_DISABLE='1', RADV_DIRECTMESH='1', RADV_BC250_EXPOSE_DGC='1', GATE_KIND='descriptor-buffers', GATE_MODE='mono', NIR_DEBUG='validate', ACO_DEBUG='validateir,validatera', BC250_TRACE_COMPILE='1', BC250_CAPTURE_POLICY_SHADERS='1', BC250_CAPTURE_POINTERS='1', RADV_DEBUG='dumpibs,shaders')
env.update(json.loads(Path(os.environ['POLICY']).read_text()))
env.update({k:'1' for k in ('RADV_BC250_MESH_SAFE_FAST','RADV_BC250_MESH_SAFE_PIECES','RADV_BC250_MESH_SAFE_OWNED','RADV_BC250_MESH_SAFE_CORNERS','RADV_BC250_MESH_SAFE_LOCAL','RADV_BC250_MESH_SAFE_AUTOCULL','RADV_BC250_MESH_SAFE_PARALLEL','RADV_BC250_MESH_SAFE_CHECK','RADV_BC250_MESH_FAIL_CLOSED')})
env.update({k:'0' for k in ('RADV_BC250_TASK_GRID_FOLD','RADV_BC250_MESH_SAFE_PIECES_EXT','RADV_BC250_MESH_PP_SHARE','RADV_BC250_MESH_LEAN_CHECK','RADV_BC250_MESH_SAFE_COMPACT','RADV_BC250_MESH_DEAD_PAYLOAD','RADV_BC250_MESH_PIECE_PRIMID','RADV_BC250_MESH_SAFE_ADAPTIVE')})
with (out/'mono.log').open('w') as f:
 p=subprocess.run([str(out/'mono'),'--offline',*[str(out/(s+'.spv')) for s in ('mesh','frag','task')],str(out/'never-rendered.rgba')],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
log=(out/'mono.log').read_text()
assert p.returncode==0 and 'NOOP_COMPLETE pixels_not_rendered' in log
assert not re.search(r'API_VALIDATION_ERROR|VUID-|Assertion .*failed|validation failed',log)
assert not (out/'never-rendered.rgba').exists()
proof=check(log)
proof['negative_cases']=mutations(log)
(out/'pointer-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print('descriptor-buffer-pointer-v2 PASS',proof,flush=True)
