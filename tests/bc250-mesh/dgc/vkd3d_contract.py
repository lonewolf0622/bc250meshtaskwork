"""Audit the supplied local vkd3d-proton Mesh ExecuteIndirect contract offline."""
import hashlib,json,os,re
from pathlib import Path
r=Path(os.environ['VKD3D_SRC']);command=(r/'libs/vkd3d/command.c').read_text();device=(r/'libs/vkd3d/device.c').read_text()
a=command.index('static void d3d12_command_list_execute_indirect_state_template_dgc(');b=command.index('\nstatic ',a+20);execute=command[a:b]
assert 'VkGeneratedCommandsPipelineInfoEXT pipeline_info' in execute and 'generated_ext.pNext = &pipeline_info' in execute
assert 'memset(&generated_ext, 0, sizeof(generated_ext))' in execute and 'indirectExecutionSet' not in execute
assert 'vkCmdPreprocessGeneratedCommandsEXT(list->cmd.vk_post_indirect_barrier_commands,' in execute
assert '&generated_ext, list->cmd.vk_command_buffer)' in execute
assert 'VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_EXT' in command
assert 'VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_COUNT_EXT' not in command
assert 'VK_INDIRECT_COMMANDS_TOKEN_TYPE_EXECUTION_SET_EXT' not in command
assert 'VK_COMMAND_BUFFER_LEVEL_SECONDARY' not in command and 'vkCmdExecuteCommands' not in command
assert 'dynamicOffsetCount' not in command
for call in re.findall(r'VK_CALL\(vkCmdBindDescriptorSets\((.*?)\)\);',command,re.S):
 assert re.search(r',\s*0,\s*NULL\s*$',call),call
sim=command.index('VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT');assert 'object->barrier_command_buffer' in command[sim:sim+250]
assert 'vkCmdBindDescriptorBuffersEXT' in command and 'vkCmdSetDescriptorBufferOffsetsEXT' in command
assert 'vkCmdBindDescriptorBufferEmbeddedSamplersEXT' in command and 'vkCmdPushDescriptorSetKHR' in command
assert 'dynamicGeneratedPipelineLayout = VK_FALSE' in device
row={'source_sha256':{name:hashlib.sha256((r/name).read_bytes()).hexdigest() for name in ('libs/vkd3d/command.c','libs/vkd3d/device.c')},
 'fixed_pipeline':True,'primary_state_and_preprocess_buffers':True,'descriptor_buffers':True,'embedded_samplers':True,'push_root_descriptors':True,
 'dynamic_offsets':False,'execution_sets':False,'secondary_buffers':False,'dynamic_generated_layouts':False,'mesh_count_tokens':False,
 'simultaneous_use_only_reusable_queue_barrier':True,'root_va_tokens':'64-bit push constants',
 'mesh_tokens':'DRAW_MESH_TASKS plus root push constants; sequenceCountAddress for count/predication'}
Path(os.environ['CONTRACT_OUT']).write_text(json.dumps(row,indent=2)+'\n');print('vkd3d Mesh ExecuteIndirect contract PASS')
