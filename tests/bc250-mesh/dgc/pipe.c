/* SPDX-License-Identifier: MIT
 * Reuse the ordinary offline graphics fixture and replace its three draw calls
 * with DGC calls. Device addresses are enabled only in this test fixture. */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static VkDevice dgc_device;
static VkPhysicalDevice dgc_physical;
static VkPipeline dgc_pipeline;
static int dgc_task;
static struct {VkPipeline pipeline;VkPipelineLayout layout;int task;} dgc_pipelines[32];
static unsigned dgc_pipeline_count;
static VkQueryPool dgc_query_pool;
static VkBuffer dgc_predicate;
static VkDeviceMemory dgc_predicate_memory;
static int dgc_query_ended;
static void dgc_reset_query(VkCommandBuffer cb);

static int dgc_failed;
static VkPipelineLayout dgc_pipeline_layout;
static void dgc_bind_pipeline(VkCommandBuffer cb,VkPipelineBindPoint bp,VkPipeline pipeline)
{
   vkCmdBindPipeline(cb,bp,pipeline);
   if(bp==VK_PIPELINE_BIND_POINT_GRAPHICS)
      for(unsigned i=0;i<dgc_pipeline_count;i++)if(dgc_pipelines[i].pipeline==pipeline) {
         dgc_pipeline=pipeline;dgc_pipeline_layout=dgc_pipelines[i].layout;dgc_task=dgc_pipelines[i].task;
      }
}
static VkCommandBuffer dgc_preprocess[4];
static VkCommandPool dgc_preprocess_pool[4];
static unsigned dgc_preprocess_count;
static VkIndirectCommandsLayoutEXT dgc_layouts[32];
static unsigned dgc_layout_count;
static VkPipeline dgc_compute[8];
static unsigned dgc_compute_count;
static VkResult dgc_create_compute(VkDevice device,VkPipelineCache cache,uint32_t n,
   const VkComputePipelineCreateInfo *ci,const VkAllocationCallbacks *alloc,VkPipeline *pipelines)
{
   VkResult r=vkCreateComputePipelines(device,cache,n,ci,alloc,pipelines);
   if(!r)for(unsigned i=0;i<n;i++){if(dgc_compute_count==8)return VK_ERROR_OUT_OF_HOST_MEMORY;dgc_compute[dgc_compute_count++]=pipelines[i];}
   return r;
}
static void dgc_destroy_device(VkDevice device,const VkAllocationCallbacks *alloc)
{
   PFN_vkDestroyIndirectCommandsLayoutEXT destroy=(void *)vkGetDeviceProcAddr(device,"vkDestroyIndirectCommandsLayoutEXT");
   for(unsigned i=0;i<dgc_layout_count;i++)destroy(device,dgc_layouts[i],NULL);
   for(unsigned i=0;i<dgc_pipeline_count;i++)vkDestroyPipeline(device,dgc_pipelines[i].pipeline,NULL);
   for(unsigned i=0;i<dgc_compute_count;i++)vkDestroyPipeline(device,dgc_compute[i],NULL);
   if(dgc_query_pool)vkDestroyQueryPool(device,dgc_query_pool,NULL);
   if(dgc_predicate){vkDestroyBuffer(device,dgc_predicate,NULL);vkFreeMemory(device,dgc_predicate_memory,NULL);}
   vkDestroyDevice(device,alloc);
}
static VkResult dgc_create_device(VkPhysicalDevice pd, const VkDeviceCreateInfo *info,
                                 const VkAllocationCallbacks *alloc, VkDevice *device)
{
   VkDeviceCreateInfo ci = *info;
   VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT feature = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_EXT,
      .pNext = (void *)ci.pNext, .deviceGeneratedCommands = VK_TRUE};
   VkPhysicalDeviceFeatures core=ci.pEnabledFeatures?*ci.pEnabledFeatures:(VkPhysicalDeviceFeatures){0};
   if(getenv("DGC_QUERY") && ci.pEnabledFeatures){core.pipelineStatisticsQuery=1;core.occlusionQueryPrecise=1;ci.pEnabledFeatures=&core;}
   for (VkBaseOutStructure *s = (void *)ci.pNext; s; s = s->pNext) {
      if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
         ((VkPhysicalDeviceVulkan12Features *)s)->bufferDeviceAddress = VK_TRUE;
      if(getenv("DGC_QUERY") && s->sType==VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
         ((VkPhysicalDeviceFeatures2 *)s)->features.pipelineStatisticsQuery=1;
         ((VkPhysicalDeviceFeatures2 *)s)->features.occlusionQueryPrecise=1;
      }
      if(getenv("DGC_QUERY") && !strncmp(getenv("DGC_QUERY"),"mesh-",5) && s->sType==VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT)
         ((VkPhysicalDeviceMeshShaderFeaturesEXT *)s)->meshShaderQueries=1;
   }
   const char *extensions[16];
   for (uint32_t i = 0; i < ci.enabledExtensionCount; i++) extensions[i] = ci.ppEnabledExtensionNames[i];
   extensions[ci.enabledExtensionCount++] = VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME;
   int has_maintenance5=0;
   for(uint32_t i=0;i<ci.enabledExtensionCount;i++)has_maintenance5|=!strcmp(extensions[i],VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
   if(!has_maintenance5)extensions[ci.enabledExtensionCount++]=VK_KHR_MAINTENANCE_5_EXTENSION_NAME;
   if(getenv("DGC_CONDITIONAL"))extensions[ci.enabledExtensionCount++]=VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME;
   ci.ppEnabledExtensionNames = extensions;
   VkPhysicalDeviceMaintenance5FeaturesKHR maintenance5={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,
      .pNext=feature.pNext,.maintenance5=1};feature.pNext=&maintenance5;
   ci.pNext = &feature;
   VkPhysicalDeviceConditionalRenderingFeaturesEXT conditional={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT,
      .pNext=&feature,.conditionalRendering=1};
   if(getenv("DGC_CONDITIONAL"))ci.pNext=&conditional;
   VkResult r = vkCreateDevice(pd, &ci, alloc, device);
   dgc_device = *device; dgc_physical = pd;
   return r;
}
static VkResult dgc_begin(VkCommandBuffer cb,const VkCommandBufferBeginInfo *info)
{
   VkCommandBufferBeginInfo ci=*info;
   if(getenv("DGC_REJECT_SIMULT"))ci.flags|=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
   VkResult r=vkBeginCommandBuffer(cb,&ci);
   if(!r)dgc_reset_query(cb);
   return r;
}
static VkResult dgc_create_pipeline(VkDevice d, VkPipelineCache c, uint32_t n,
                                    const VkGraphicsPipelineCreateInfo *i, const VkAllocationCallbacks *a, VkPipeline *p)
{
   VkResult r = vkCreateGraphicsPipelines(d,c,n,i,a,p);
   if (!r) for(unsigned index=0;index<n;index++) {
      if(dgc_pipeline_count==32)return VK_ERROR_OUT_OF_HOST_MEMORY;
      unsigned entry=dgc_pipeline_count++;dgc_pipelines[entry].pipeline=p[index];
      dgc_pipelines[entry].layout=i[index].layout;
      for (unsigned stage=0;stage<i[index].stageCount;stage++)
         dgc_pipelines[entry].task |= i[index].pStages[stage].stage == VK_SHADER_STAGE_TASK_BIT_EXT;
   }
   return r;
}
static VkResult dgc_create_buffer(VkDevice d, const VkBufferCreateInfo *i, const VkAllocationCallbacks *a, VkBuffer *p)
{
   VkBufferCreateInfo ci = *i;
   ci.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
   return vkCreateBuffer(d,&ci,a,p);
}
static VkResult dgc_alloc(VkDevice d, const VkMemoryAllocateInfo *i, const VkAllocationCallbacks *a, VkDeviceMemory *p)
{
   VkMemoryAllocateInfo ci = *i;
   VkMemoryAllocateFlagsInfo flags = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .pNext = ci.pNext, .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
   ci.pNext = &flags;
   return vkAllocateMemory(d,&ci,a,p);
}
#define CHECK(x) do { VkResult r = (x); if (r) { fprintf(stderr,"DGC_ERROR %s %d\n",#x,r); dgc_failed=1; return; } } while(0)
static void dgc_buffer(VkDeviceSize bytes, uint32_t bits, VkBuffer *buffer, VkDeviceMemory *memory, void **map)
{
   VkBufferCreateInfo ci = {.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=bytes,
      .usage=VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT|
             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
   if(getenv("DGC_CONDITIONAL"))ci.usage|=VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT;
   VkBufferUsageFlags2CreateInfo usage = {.sType=VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO,
      .usage=ci.usage | (bits != ~0u ? VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT : 0)};
   ci.pNext=&usage;
   CHECK(vkCreateBuffer(dgc_device,&ci,NULL,buffer));
   VkMemoryRequirements req; vkGetBufferMemoryRequirements(dgc_device,*buffer,&req);
   VkPhysicalDeviceMemoryProperties props; vkGetPhysicalDeviceMemoryProperties(dgc_physical,&props);
   uint32_t type;
   for(type=0;type<props.memoryTypeCount;type++)
      if((req.memoryTypeBits & bits & (1u<<type)) &&
         (props.memoryTypes[type].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
          (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) break;
   if(type==props.memoryTypeCount) {dgc_failed=1;return;}
   VkMemoryAllocateFlagsInfo flags={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,.flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
   VkMemoryAllocateInfo ai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.pNext=&flags,.allocationSize=req.size,.memoryTypeIndex=type};
   CHECK(vkAllocateMemory(dgc_device,&ai,NULL,memory));
   CHECK(vkBindBufferMemory(dgc_device,*buffer,*memory,0));
   CHECK(vkMapMemory(dgc_device,*memory,0,bytes,0,map));
}
static VkDeviceAddress dgc_address(VkBuffer buffer)
{
   VkBufferDeviceAddressInfo i={.sType=VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,.buffer=buffer};
   return vkGetBufferDeviceAddress(dgc_device,&i);
}
static void dgc_reset_query(VkCommandBuffer cb)
{
   dgc_query_ended=0;
   if(!getenv("DGC_QUERY"))return;
   if(!dgc_query_pool) {
      const char *kind=getenv("DGC_QUERY");
      VkQueryPoolCreateInfo ci={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,.queryCount=1,
         .queryType=!strcmp(kind,"occlusion")?VK_QUERY_TYPE_OCCLUSION:!strcmp(kind,"mesh-primitives")?VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT:VK_QUERY_TYPE_PIPELINE_STATISTICS,
         .pipelineStatistics=VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT|
           VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT};
      if(!strcmp(kind,"mesh-pipeline"))ci.pipelineStatistics|=VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT|VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT;
      CHECK(vkCreateQueryPool(dgc_device,&ci,NULL,&dgc_query_pool));
   }
   vkCmdResetQueryPool(cb,dgc_query_pool,0,1);
}
static void dgc_scope_begin(VkCommandBuffer cb)
{
   if(dgc_query_pool)vkCmdBeginQuery(cb,dgc_query_pool,0,!strcmp(getenv("DGC_QUERY"),"occlusion")?VK_QUERY_CONTROL_PRECISE_BIT:0);
   if(getenv("DGC_CONDITIONAL")) {
      if(!dgc_predicate) {
         void *map;dgc_buffer(256,~0u,&dgc_predicate,&dgc_predicate_memory,&map);
         if(dgc_failed)return;
         *(uint32_t *)map=strtoul(getenv("DGC_CONDITIONAL"),NULL,0);
      }
      VkConditionalRenderingBeginInfoEXT ci={.sType=VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT,.buffer=dgc_predicate,
         .flags=getenv("DGC_INVERTED")?VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT:0};
      PFN_vkCmdBeginConditionalRenderingEXT begin=(void *)vkGetDeviceProcAddr(dgc_device,"vkCmdBeginConditionalRenderingEXT");begin(cb,&ci);
   }
}
static void dgc_scope_end(VkCommandBuffer cb)
{
   if(getenv("DGC_CONDITIONAL")){PFN_vkCmdEndConditionalRenderingEXT end=(void *)vkGetDeviceProcAddr(dgc_device,"vkCmdEndConditionalRenderingEXT");end(cb);}
   if(dgc_query_pool&&!dgc_query_ended)vkCmdEndQuery(cb,dgc_query_pool,0);
}
static void dgc_render_begin(VkCommandBuffer cb,const VkRenderPassBeginInfo *i,VkSubpassContents c){vkCmdBeginRenderPass(cb,i,c);dgc_scope_begin(cb);}
static void dgc_render_end(VkCommandBuffer cb){dgc_scope_end(cb);vkCmdEndRenderPass(cb);}
static void dgc_rendering_begin(VkCommandBuffer cb,const VkRenderingInfo *i){vkCmdBeginRendering(cb,i);dgc_scope_begin(cb);}
static void dgc_rendering_end(VkCommandBuffer cb){dgc_scope_end(cb);vkCmdEndRendering(cb);}
static void dgc_draw(VkCommandBuffer cb, int count)
{
   PFN_vkCreateIndirectCommandsLayoutEXT create=(void *)vkGetDeviceProcAddr(dgc_device,"vkCreateIndirectCommandsLayoutEXT");
   PFN_vkGetGeneratedCommandsMemoryRequirementsEXT requirements=(void *)vkGetDeviceProcAddr(dgc_device,"vkGetGeneratedCommandsMemoryRequirementsEXT");
   PFN_vkCmdExecuteGeneratedCommandsEXT execute=(void *)vkGetDeviceProcAddr(dgc_device,"vkCmdExecuteGeneratedCommandsEXT");
   if(!create||!requirements||!execute) {dgc_failed=1;return;}
   VkIndirectCommandsLayoutTokenEXT token={.sType=VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT,
      .type=count ? VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_COUNT_EXT : VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_EXT,
      .offset=getenv("DGC_PUSH_CONSTANTS")?8:0};
   VkIndirectCommandsPushConstantTokenEXT pc={.updateRange={VK_SHADER_STAGE_ALL,0,4}};
   VkIndirectCommandsPushConstantTokenEXT sequence={.updateRange={VK_SHADER_STAGE_ALL,4,4}};
   VkIndirectCommandsLayoutTokenEXT tokens[3]={
      {.sType=VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT,.type=VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_EXT,
       .offset=0,.data.pPushConstant=&pc},
      {.sType=VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT,.type=VK_INDIRECT_COMMANDS_TOKEN_TYPE_SEQUENCE_INDEX_EXT,
       .offset=4,.data.pPushConstant=&sequence}, token};
   int pcs=getenv("DGC_PUSH_CONSTANTS")!=NULL;
   VkIndirectCommandsLayoutCreateInfoEXT ci={.sType=VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT,
      .shaderStages=VK_SHADER_STAGE_MESH_BIT_EXT|VK_SHADER_STAGE_FRAGMENT_BIT|
                    (dgc_task?VK_SHADER_STAGE_TASK_BIT_EXT:0),.indirectStride=32,
      .pipelineLayout=dgc_pipeline_layout,.tokenCount=pcs?3:1,.pTokens=pcs?tokens:&token,
      .flags=getenv("DGC_PREPROCESS") ? VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_EXT : 0};
   VkIndirectCommandsLayoutEXT layout; CHECK(create(dgc_device,&ci,NULL,&layout));
   if(dgc_layout_count==32){dgc_failed=1;return;}
   dgc_layouts[dgc_layout_count++]=layout;
   VkGeneratedCommandsPipelineInfoEXT pi={.sType=VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT,.pipeline=dgc_pipeline};
   unsigned draws=getenv("DGC_MAX_DRAWS")?strtoul(getenv("DGC_MAX_DRAWS"),NULL,0):2;
   unsigned sequences=getenv("DGC_MAX_SEQUENCES")?strtoul(getenv("DGC_MAX_SEQUENCES"),NULL,0):2;
   VkGeneratedCommandsMemoryRequirementsInfoEXT mi={.sType=VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_EXT,
      .pNext=&pi,.indirectCommandsLayout=layout,.maxSequenceCount=sequences,.maxDrawCount=draws};
   VkMemoryRequirements2 mr={.sType=VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2}; requirements(dgc_device,&mi,&mr);
   if(getenv("DGC_SEQUENCE_REQUIREMENTS")) {
      const unsigned bounds[]={0,1,4096,5000,1048576,1048577};
      for(unsigned b=0;b<sizeof(bounds)/sizeof(bounds[0]);b++) {
         mi.maxSequenceCount=bounds[b];requirements(dgc_device,&mi,&mr);
         printf("DGC_SEQUENCE_BOUND count=%d sequences=%u bytes=%llu\n",count,bounds[b],(unsigned long long)mr.memoryRequirements.size);
      }
      return;
   }
   if(getenv("DGC_REQUIREMENTS_ONLY")) {
      const unsigned bounds[]={1,3,4,8,64,128,256,511,512,4096,4097};
      mi.maxSequenceCount=1;
      for(unsigned b=0;b<sizeof(bounds)/sizeof(bounds[0]);b++) {
         mi.maxDrawCount=bounds[b];requirements(dgc_device,&mi,&mr);
         printf("DGC_BOUND count=%d draws=%u bytes=%llu\n",count,bounds[b],(unsigned long long)mr.memoryRequirements.size);
      }
      return;
   }
   printf("DGC_REQUIREMENTS count=%d bytes=%llu\n",count,(unsigned long long)mr.memoryRequirements.size);fflush(stdout);
   if(!mr.memoryRequirements.size) {dgc_failed=1;return;}
   VkBuffer stream,output; VkDeviceMemory sm,om; void *map,*out;
   unsigned stream_bytes=(64+draws*16+31)&~15u;
   if(stream_bytes<256)stream_bytes=256;
   unsigned count_offset=stream_bytes-16;
   dgc_buffer(stream_bytes,~0u,&stream,&sm,&map); if(dgc_failed)return;
   dgc_buffer(mr.memoryRequirements.size,mr.memoryRequirements.memoryTypeBits,&output,&om,&out); if(dgc_failed)return;
   memset(map,0,stream_bytes);
   uint32_t *words=map;
   unsigned draw_offset=pcs?2:0;
   if(count) {
      for(unsigned seq=0;seq<2;seq++) {
         uint64_t va=dgc_address(stream)+64;
         memcpy(words+seq*8+draw_offset,&va,8);words[seq*8+draw_offset+2]=16;words[seq*8+draw_offset+3]=draws;
      }
      for(unsigned draw=0;draw<draws;draw++){words[16+draw*4]=3+draw;words[17+draw*4]=1;words[18+draw*4]=1;}
   } else {
      words[draw_offset]=7;words[draw_offset+1]=1;words[draw_offset+2]=1;words[draw_offset+8]=2;words[draw_offset+9]=2;words[draw_offset+10]=1;
   }
   if(pcs) {
#ifdef DGC_BINDING_FIXTURE
      words[0]=0;words[8]=1;
#else
      words[0]=11;words[8]=23;
#endif
   }
   words[count_offset/4]=sequences;
   VkGeneratedCommandsInfoEXT info={.sType=VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT,.pNext=&pi,
      .shaderStages=ci.shaderStages,.indirectCommandsLayout=layout,.indirectAddress=dgc_address(stream),
      .indirectAddressSize=64,.preprocessAddress=dgc_address(output),.preprocessSize=mr.memoryRequirements.size,
      .maxSequenceCount=sequences,.sequenceCountAddress=dgc_address(stream)+count_offset,.maxDrawCount=draws};
   const char *dump=getenv("BC250_DGC_DUMP");
   if(dump) {
      char path[4096];snprintf(path,sizeof(path),"%s/token-%u.bin",dump,count);
      FILE *f=fopen(path,"wb");if(f){fwrite(map,1,stream_bytes,f);fclose(f);}
   }
   if(getenv("DGC_PREPROCESS")) {
      unsigned n=dgc_preprocess_count++;
      VkCommandPoolCreateInfo pool_info={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
      CHECK(vkCreateCommandPool(dgc_device,&pool_info,NULL,&dgc_preprocess_pool[n]));
      VkCommandBufferAllocateInfo alloc={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
         .commandPool=dgc_preprocess_pool[n],.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
      CHECK(vkAllocateCommandBuffers(dgc_device,&alloc,&dgc_preprocess[n]));
      VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      CHECK(vkBeginCommandBuffer(dgc_preprocess[n],&begin));
      PFN_vkCmdPreprocessGeneratedCommandsEXT prepare=(void *)vkGetDeviceProcAddr(dgc_device,"vkCmdPreprocessGeneratedCommandsEXT");
      prepare(dgc_preprocess[n],&info,cb);
      CHECK(vkEndCommandBuffer(dgc_preprocess[n]));
      if(getenv("DGC_QUERY_CHANGE")&&!dgc_query_ended){vkCmdEndQuery(cb,dgc_query_pool,0);dgc_query_ended=1;}
      execute(cb,VK_TRUE,&info);
   } else execute(cb,VK_FALSE,&info);
   puts("DGC_RECORDED");fflush(stdout);
}
static void dgc_direct(VkCommandBuffer cb,uint32_t x,uint32_t y,uint32_t z) { (void)x;(void)y;(void)z;dgc_draw(cb,0); }
static void dgc_indirect(VkCommandBuffer cb,VkBuffer b,VkDeviceSize o,uint32_t n,uint32_t s) { (void)cb;(void)b;(void)o;(void)n;(void)s; }
static void dgc_count(VkCommandBuffer cb,VkBuffer b,VkDeviceSize o,VkBuffer c,VkDeviceSize co,uint32_t n,uint32_t s) { (void)b;(void)o;(void)c;(void)co;(void)n;(void)s;dgc_draw(cb,1); }
static PFN_vkVoidFunction dgc_get_proc(VkDevice d,const char *name)
{
   if(!strcmp(name,"vkCmdDrawMeshTasksEXT"))return (PFN_vkVoidFunction)dgc_direct;
   if(!strcmp(name,"vkCmdDrawMeshTasksIndirectEXT"))return (PFN_vkVoidFunction)dgc_indirect;
   if(!strcmp(name,"vkCmdDrawMeshTasksIndirectCountEXT"))return (PFN_vkVoidFunction)dgc_count;
   return vkGetDeviceProcAddr(d,name);
}
static VkResult dgc_submit(VkQueue q,uint32_t n,const VkSubmitInfo *infos,VkFence f)
{
   if(dgc_failed)return VK_ERROR_FEATURE_NOT_PRESENT;
   if(dgc_preprocess_count) {
      VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=dgc_preprocess_count,.pCommandBuffers=dgc_preprocess};
      VkResult r=vkQueueSubmit(q,1,&submit,VK_NULL_HANDLE);if(r)return r;
      r=vkQueueWaitIdle(q);if(r)return r;
      for(unsigned i=0;i<dgc_preprocess_count;i++)vkDestroyCommandPool(dgc_device,dgc_preprocess_pool[i],NULL);
      dgc_preprocess_count=0;
   }
   return vkQueueSubmit(q,n,infos,f);
}
#define vkCreateComputePipelines dgc_create_compute
#define vkCmdBeginRenderPass dgc_render_begin
#define vkCmdEndRenderPass dgc_render_end
#define vkCmdBeginRendering dgc_rendering_begin
#define vkCmdEndRendering dgc_rendering_end
#define vkDestroyDevice dgc_destroy_device
#define vkCmdBindPipeline dgc_bind_pipeline
#define vkBeginCommandBuffer dgc_begin
#define vkQueueSubmit dgc_submit
#define vkCreateDevice dgc_create_device
#define vkCreateGraphicsPipelines dgc_create_pipeline
#define vkCreateBuffer dgc_create_buffer
#define vkAllocateMemory dgc_alloc
#define vkGetDeviceProcAddr dgc_get_proc
#define main ordinary_fixture
#ifdef DGC_BINDING_FIXTURE
#include "../fast-binding/fb.c"
#else
#include "../safe-direct/pipe.c"
#endif
#undef main
int main(int argc,char **argv) { int r=ordinary_fixture(argc,argv);return r ? r : dgc_failed; }
