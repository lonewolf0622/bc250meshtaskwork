/* SPDX-License-Identifier: MIT
 * Experimental BC250 query regression harness, corrected in v14.
 * Counts measure API invocations, including local workgroup size. Default
 * shaders use one local invocation; the focused runner also tests larger
 * multidimensional workgroups and exact payload identities across chunks.
 * Required features are enabled and every query pool is reset before use.
 * Host reset checks availability, not undefined numerical result contents.
 * Exit 0 means PASS, 2 failure, 3 unsupported feature. These custom cases
 * are not a claim of full Vulkan conformance. The old probe remains quarantined.
 */
#define COMBINED_VERTEX_EXPECT 0u

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifndef TEST_BOUNDARY_AXIS
#define TEST_BOUNDARY_AXIS -1
#endif
#ifndef TEST_BOUNDARY_INDIRECT
#define TEST_BOUNDARY_INDIRECT 0
#endif
#define VK(call) do { VkResult r=(call); if(r!=VK_SUCCESS) { fprintf(stderr,"%s: VkResult %d at line %d\n",#call,r,__LINE__); exit(2); } } while(0)
static VkDevice dev;
static VkPhysicalDeviceMemoryProperties mp;
static uint32_t memtype(uint32_t bits,VkMemoryPropertyFlags flags) {
   for(uint32_t i=0;i<mp.memoryTypeCount;i++) if((bits&(1u<<i)) && (mp.memoryTypes[i].propertyFlags&flags)==flags) return i;
   fprintf(stderr,"No memory type\n"); exit(2);
}
static VkDeviceMemory alloc_mem(VkMemoryRequirements req,VkMemoryPropertyFlags flags) {
   VkMemoryAllocateInfo a={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=req.size,.memoryTypeIndex=memtype(req.memoryTypeBits,flags)};
   VkDeviceMemory m; VK(vkAllocateMemory(dev,&a,NULL,&m)); return m;
}
static VkShaderModule shader(const char *name) {
   FILE *f=fopen(name,"rb"); if(!f){perror(name);exit(2);} fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);
   uint32_t *p=malloc(n); if(fread(p,1,n,f)!=(size_t)n)exit(2);fclose(f);
   VkShaderModuleCreateInfo ci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=n,.pCode=p};
   VkShaderModule m;VK(vkCreateShaderModule(dev,&ci,NULL,&m));free(p);return m;
}
static void check(uint64_t got, uint64_t want, const char *what) {
   if (got != want) { printf("FAIL %s: got %llu want %llu\n", what, (unsigned long long)got, (unsigned long long)want); exit(2); }
   printf("ok %s = %llu\n", what, (unsigned long long)got);
}
static uint64_t read_query(VkQueryPool pool, uint32_t idx) {
   uint64_t v = 0;
   VkResult r = vkGetQueryPoolResults(dev, pool, idx, 1, sizeof(v), &v, sizeof(v), VK_QUERY_RESULT_64_BIT);
   if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) { fprintf(stderr,"vkGetQueryPoolResults: %d\n", r); exit(2); }
   return v;
}

int main(int argc,char **argv) {
   const char *mode=argc>1?argv[1]:"hw";
   /* Pools per mode: mesh_pool for mesh/both/zero/seq/indirect, task_pool
    * for task/both/chunked/indirect/zero/seq. combined uses a dedicated
    * HW+GDS pool but still needs the task/mesh pipeline. */
   int indirect_large=!strcmp(mode,"indirect-large"), cond_mode=!strcmp(mode,"indirect-cond");
   int mesh_mode=!strcmp(mode,"mesh")||!strcmp(mode,"both")||!strcmp(mode,"zero")||!strcmp(mode,"seq")||!strcmp(mode,"indirect")||indirect_large||cond_mode;
   int task_mode=!strcmp(mode,"task")||!strcmp(mode,"both")||!strcmp(mode,"chunked")||!strcmp(mode,"indirect")||!strcmp(mode,"zero")||!strcmp(mode,"seq")||indirect_large||cond_mode;
   int combined_mode=!strcmp(mode,"combined");
   /* probe: HW statistics only around a mesh draw; no meshShaderQueries
    * needed, so it must run (and print) on the unmodified v7 driver. */
   int probe_mode=!strcmp(mode,"probe");
   int primgen_mode=!strcmp(mode,"primgen");
   int reset_mode=!strcmp(mode,"reset");
   int primgen_large=!strcmp(mode,"primgen-large");
   int needs_mesh_ext=mesh_mode||task_mode||combined_mode||probe_mode||primgen_mode||reset_mode||primgen_large;

   /* apiVersion matters: without pApplicationInfo the instance defaults to
    * Vulkan 1.0 and vkGetPhysicalDeviceFeatures2 reports meshShader/task
    * features as 0 (mesh_smoke.c uses 1.3 and sees them). The original -8
    * device-creation failure in probe mode is believed related to this. */
   VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.pApplicationName="bc250-query-test",.apiVersion=VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
   VkInstance inst; VK(vkCreateInstance(&ici,NULL,&inst));
   uint32_t n=1; VK(vkEnumeratePhysicalDevices(inst,&n,NULL));
   VkPhysicalDevice pd; VK(vkEnumeratePhysicalDevices(inst,&n,&pd));
   vkGetPhysicalDeviceMemoryProperties(pd,&mp);

   VkPhysicalDeviceMeshShaderFeaturesEXT msf={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
   VkPhysicalDeviceFeatures2 f2={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,.pNext=&msf};
   vkGetPhysicalDeviceFeatures2(pd,&f2);
   printf("features2: meshShader=%u taskShader=%u meshShaderQueries=%u\n",(unsigned)msf.meshShader,(unsigned)msf.taskShader,(unsigned)msf.meshShaderQueries);
   if((mesh_mode||task_mode||combined_mode||primgen_mode||reset_mode||primgen_large) && !(msf.meshShader&&msf.taskShader)) { printf("SKIP_FEATURE mesh/task not exposed\n"); return 3; }
   if((mesh_mode||task_mode||combined_mode||primgen_mode||reset_mode||primgen_large) && !msf.meshShaderQueries) { printf("SKIP_FEATURE meshShaderQueries not exposed\n"); return 3; }

   uint32_t qfam=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&n,NULL); /* void in this generation */
   VkQueueFamilyProperties *qfs=malloc(n*sizeof(*qfs)); vkGetPhysicalDeviceQueueFamilyProperties(pd,&n,qfs);
   for(uint32_t i=0;i<n;i++) if(qfs[i].queueFlags&VK_QUEUE_GRAPHICS_BIT){qfam=i;break;}
   float prio=1.0f;
   VkDeviceQueueCreateInfo dqci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=qfam,.queueCount=1,.pQueuePriorities=&prio};
   const char *exts[]={"VK_EXT_mesh_shader"};
   /* Explicitly enable extension and core features used by this harness. */
   static VkPhysicalDeviceMeshShaderFeaturesEXT want={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
      .meshShader=VK_TRUE,.taskShader=VK_TRUE};
   /* meshShaderQueries is only requested where actually needed: on the
    * unmodified v7 driver it is not exposed and device creation would fail
    * with VK_ERROR_FEATURE_NOT_PRESENT (probe mode must still run). */
   want.meshShaderQueries=(mesh_mode||task_mode||combined_mode||primgen_mode||reset_mode||primgen_large)?VK_TRUE:VK_FALSE;
   static VkPhysicalDeviceVulkan12Features f12={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,.pNext=NULL};
   VkPhysicalDeviceVulkan11Features f11={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
      .shaderDrawParameters=VK_TRUE,.pNext=needs_mesh_ext?&want:NULL};
   f12.pNext=&f11;
   f12.drawIndirectCount=(TEST_BOUNDARY_INDIRECT==2);
   if (!f2.features.pipelineStatisticsQuery) { puts("SKIP_FEATURE pipelineStatisticsQuery"); return 3; }
   VkPhysicalDeviceVulkan12Features supported12={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
   VkPhysicalDeviceFeatures2 supported={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,.pNext=&supported12};
   vkGetPhysicalDeviceFeatures2(pd,&supported);
   if(!supported12.hostQueryReset) { puts("SKIP_FEATURE hostQueryReset"); return 3; }
   f12.hostQueryReset=VK_TRUE;
   static VkPhysicalDeviceFeatures basefeat={.multiDrawIndirect=VK_TRUE,.pipelineStatisticsQuery=VK_TRUE,
      .vertexPipelineStoresAndAtomics=(TEST_BOUNDARY_AXIS>=0)};
   VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.queueCreateInfoCount=1,.pQueueCreateInfos=&dqci,
      .enabledExtensionCount=needs_mesh_ext?1:0,.ppEnabledExtensionNames=needs_mesh_ext?exts:NULL,
      .pNext=&f12,.pEnabledFeatures=&basefeat};
   VkResult dcr=vkCreateDevice(pd,&dci,NULL,&dev);
   if(dcr!=VK_SUCCESS){ fprintf(stderr,"vkCreateDevice failed: %d (mode=%s meshShaderQueries=%d)\n",(int)dcr,mode,(int)want.meshShaderQueries); exit(2); }
   VkQueue queue; vkGetDeviceQueue(dev,qfam,0,&queue); /* void in this generation */

   /* 64x64 render target + render pass (shared by all modes). */
   VkMemoryRequirements req;
   VkImageCreateInfo imci={.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,.imageType=VK_IMAGE_TYPE_2D,.format=VK_FORMAT_R8G8B8A8_UNORM,
      .extent={64,64,1},.mipLevels=1,.arrayLayers=1,.samples=VK_SAMPLE_COUNT_1_BIT,.tiling=VK_IMAGE_TILING_OPTIMAL,
      .usage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED};
   VkImage im;VK(vkCreateImage(dev,&imci,NULL,&im));vkGetImageMemoryRequirements(dev,im,&req);
   VkDeviceMemory immem=alloc_mem(req,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);VK(vkBindImageMemory(dev,im,immem,0));
   VkImageViewCreateInfo ivci={.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,.image=im,.viewType=VK_IMAGE_VIEW_TYPE_2D,
      .format=imci.format,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
   VkImageView view;VK(vkCreateImageView(dev,&ivci,NULL,&view));
   VkAttachmentDescription att={.format=imci.format,.samples=VK_SAMPLE_COUNT_1_BIT,.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp=VK_ATTACHMENT_STORE_OP_STORE,.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,.finalLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL};
   VkAttachmentReference ref={0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   VkSubpassDescription sub={.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,.colorAttachmentCount=1,.pColorAttachments=&ref};
   VkRenderPassCreateInfo rpci={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,.attachmentCount=1,.pAttachments=&att,
      .subpassCount=1,.pSubpasses=&sub};
   VkRenderPass rp;VK(vkCreateRenderPass(dev,&rpci,NULL,&rp));
   VkFramebufferCreateInfo fbci={.sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,.renderPass=rp,.attachmentCount=1,
      .pAttachments=&view,.width=64,.height=64,.layers=1};
   VkFramebuffer fb;VK(vkCreateFramebuffer(dev,&fbci,NULL,&fb));

   /* Host-visible scratch: indirect records. */
   VkBufferCreateInfo bci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=8192,
      .usage=VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT};
   VkBuffer buf;VK(vkCreateBuffer(dev,&bci,NULL,&buf));vkGetBufferMemoryRequirements(dev,buf,&req);
   VkDeviceMemory bm=alloc_mem(req,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   VK(vkBindBufferMemory(dev,buf,bm,0));
   uint8_t *map;VK(vkMapMemory(dev,bm,0,VK_WHOLE_SIZE,0,(void**)&map));memset(map,0,8192);

   uint32_t *visited=NULL;
   VkDescriptorSet visit_set=VK_NULL_HANDLE;
   VkDescriptorSetLayout visit_layout=VK_NULL_HANDLE;
   if(TEST_BOUNDARY_AXIS>=0) {
      VkBufferCreateInfo vci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=8196*4,
         .usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
      VkBuffer vb; VK(vkCreateBuffer(dev,&vci,NULL,&vb));
      vkGetBufferMemoryRequirements(dev,vb,&req);
      VkDeviceMemory vm=alloc_mem(req,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      VK(vkBindBufferMemory(dev,vb,vm,0));
      VK(vkMapMemory(dev,vm,0,VK_WHOLE_SIZE,0,(void**)&visited));
      memset(visited,0,8196*4);
      visited[8195]=0xa5a5a5a5;
      VkDescriptorSetLayoutBinding binding={.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount=1,.stageFlags=VK_SHADER_STAGE_MESH_BIT_EXT};
      VkDescriptorSetLayoutCreateInfo dlci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
         .bindingCount=1,.pBindings=&binding};
      VK(vkCreateDescriptorSetLayout(dev,&dlci,NULL,&visit_layout));
      VkDescriptorPoolSize ps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1};
      VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,
         .poolSizeCount=1,.pPoolSizes=&ps};
      VkDescriptorPool dp; VK(vkCreateDescriptorPool(dev,&dpci,NULL,&dp));
      VkDescriptorSetAllocateInfo dai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool=dp,.descriptorSetCount=1,.pSetLayouts=&visit_layout};
      VK(vkAllocateDescriptorSets(dev,&dai,&visit_set));
      VkDescriptorBufferInfo bi={vb,0,8196*4};
      VkWriteDescriptorSet wr={.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=visit_set,
         .dstBinding=0,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi};
      vkUpdateDescriptorSets(dev,1,&wr,0,NULL);
   }
   VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
   if(visit_layout) { plci.setLayoutCount=1; plci.pSetLayouts=&visit_layout; }
   VkPipelineLayout layout;VK(vkCreatePipelineLayout(dev,&plci,NULL,&layout));

   /* Pipelines: hw mode uses VS/FS fullscreen triangle; mesh modes use task+mesh+FS. */
   VkPipeline pipeline=VK_NULL_HANDLE;
   {
      VkPipelineVertexInputStateCreateInfo vi={.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
      VkPipelineInputAssemblyStateCreateInfo ia={.sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
         .topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
      VkPipelineViewportStateCreateInfo vp={.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,.viewportCount=1,.scissorCount=1};
      VkViewport vpt={0,0,64,64,0,1}; VkRect2D sc={{0,0},{64,64}};
      vp.pViewports=&vpt; vp.pScissors=&sc;
      VkPipelineRasterizationStateCreateInfo rs={.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
         .polygonMode=VK_POLYGON_MODE_FILL,.cullMode=VK_CULL_MODE_NONE,.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,
         .lineWidth=1.0f};
      VkPipelineMultisampleStateCreateInfo mss={.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
         .rasterizationSamples=VK_SAMPLE_COUNT_1_BIT};
      /* This generation's struct has an explicit colorWriteMask field. */
      VkPipelineColorBlendAttachmentState cb[1]={{VK_FALSE,VK_BLEND_FACTOR_ONE,VK_BLEND_FACTOR_ZERO,
         VK_BLEND_OP_ADD,VK_BLEND_FACTOR_ONE,VK_BLEND_FACTOR_ZERO,VK_BLEND_OP_ADD,
         VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT}};
      VkPipelineColorBlendStateCreateInfo cbs={.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,.attachmentCount=1,.pAttachments=cb};
      if(!needs_mesh_ext) {
         VkShaderModule vs=shader("qvert.spv"), fs=shader("qfrag.spv");
         VkPipelineShaderStageCreateInfo stages[2]={
            {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_VERTEX_BIT,.module=vs,.pName="main"},
            {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_FRAGMENT_BIT,.module=fs,.pName="main"}};
         VkGraphicsPipelineCreateInfo gpci={.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,.layout=layout,
            .stageCount=2,.pStages=stages,.pVertexInputState=&vi,.pInputAssemblyState=&ia,.pViewportState=&vp,
            .pRasterizationState=&rs,.pMultisampleState=&mss,.pColorBlendState=&cbs,.renderPass=rp};
         VK(vkCreateGraphicsPipelines(dev,VK_NULL_HANDLE,1,&gpci,NULL,&pipeline));
      } else {
         /* indirect-cond needs a task shader that emits conditionally. */
         const char *ts_name=cond_mode?"qtask_cond.spv":"qtask.spv";
         VkShaderModule ts=shader(ts_name), msh=shader("qmesh.spv"), fs=shader("qfrag.spv");
         VkPipelineShaderStageCreateInfo stages[3]={
            {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_TASK_BIT_EXT,.module=ts,.pName="main"},
            {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_MESH_BIT_EXT,.module=msh,.pName="main"},
            {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_FRAGMENT_BIT,.module=fs,.pName="main"}};
         VkGraphicsPipelineCreateInfo gpci={.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,.layout=layout,
            .stageCount=3,.pStages=stages,.pVertexInputState=&vi,.pInputAssemblyState=&ia,.pViewportState=&vp,
            .pRasterizationState=&rs,.pMultisampleState=&mss,.pColorBlendState=&cbs,.renderPass=rp};
         VK(vkCreateGraphicsPipelines(dev,VK_NULL_HANDLE,1,&gpci,NULL,&pipeline));
      }
   }

   /* Query pools: one pool per statistic under test (slot order irrelevant). */
   VkQueryPool mesh_pool=VK_NULL_HANDLE, task_pool=VK_NULL_HANDLE, hw_pool=VK_NULL_HANDLE;
   if(mesh_mode) {
      VkQueryPoolCreateInfo qpci={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
         .queryType=VK_QUERY_TYPE_PIPELINE_STATISTICS,.queryCount=2,
         .pipelineStatistics=VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT};
      VK(vkCreateQueryPool(dev,&qpci,NULL,&mesh_pool));
      vkResetQueryPool(dev,mesh_pool,0,qpci.queryCount);
   }
   if(task_mode) {
      VkQueryPoolCreateInfo qpci={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
         .queryType=VK_QUERY_TYPE_PIPELINE_STATISTICS,.queryCount=2,
         .pipelineStatistics=VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT};
      VK(vkCreateQueryPool(dev,&qpci,NULL,&task_pool));
      vkResetQueryPool(dev,task_pool,0,qpci.queryCount);
   }

   VkQueryPool pg_pool=VK_NULL_HANDLE;
   if(primgen_mode||primgen_large) {
      VkQueryPoolCreateInfo qpci={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
         .queryType=VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT,.queryCount=2};
      VK(vkCreateQueryPool(dev,&qpci,NULL,&pg_pool));
      vkResetQueryPool(dev,pg_pool,0,qpci.queryCount);
   }

   /* v11: reset-semantics pools. Note: this Vulkan generation has no
    * standalone TASK/MESH invocation query types (verified against the spec:
    * only MESH_PRIMITIVES_GENERATED_EXT exists); invocation counts are only
    * queryable through PIPELINE_STATISTICS pools with the EXT bits, which is
    * what mixed_pool exercises. */
   VkQueryPool mixed_pool=VK_NULL_HANDLE, rst_pg_pool=VK_NULL_HANDLE;
   if(reset_mode) {
      VkQueryPoolCreateInfo qpci={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
         .queryType=VK_QUERY_TYPE_PIPELINE_STATISTICS,.queryCount=1,
         .pipelineStatistics=VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT|
                             VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT|
                             VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT};
      VK(vkCreateQueryPool(dev,&qpci,NULL,&mixed_pool));
      vkResetQueryPool(dev,mixed_pool,0,qpci.queryCount);
      qpci.queryType=VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT;
      qpci.pipelineStatistics=0;
      VK(vkCreateQueryPool(dev,&qpci,NULL,&rst_pg_pool));
      vkResetQueryPool(dev,rst_pg_pool,0,qpci.queryCount);
   }

   /* Indirect task records. Default (indirect mode): two (1,1,1) dispatches.
    * indirect-large: one record spanning two chunks and one with a dimension
    * above the old 4096 clamp. indirect-cond: two small records whose
    * conditional emission alternates via DrawID. */
   uint32_t *ind=(uint32_t*)(map+4096);
   if(indirect_large||primgen_large){ ind[0]=8192;ind[1]=1;ind[2]=1; ind[3]=5000;ind[4]=1;ind[5]=1; }
   else if(cond_mode){ ind[0]=4;ind[1]=1;ind[2]=1; ind[3]=4;ind[4]=1;ind[5]=1; }
   else { ind[0]=1;ind[1]=1;ind[2]=1; ind[3]=1;ind[4]=1;ind[5]=1; }

   VkFenceCreateInfo fci={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;VK(vkCreateFence(dev,&fci,NULL,&fence));

   uint32_t task_groups = !strcmp(mode,"chunked") ? 4097u : (!strcmp(mode,"seq") ? 1u : 3u);
   uint32_t boundary_dims[3]={2,1,1};
   if(TEST_BOUNDARY_AXIS>=0) {
      boundary_dims[TEST_BOUNDARY_AXIS]=4097;
      boundary_dims[(TEST_BOUNDARY_AXIS+1)%3]=2;
      boundary_dims[(TEST_BOUNDARY_AXIS+2)%3]=1;
      task_groups=8194;
      memcpy(ind,boundary_dims,sizeof(boundary_dims));
   }
   const uint64_t mesh_per_task = 2; /* qtask emits one (2,1,1) record per invocation */

   VkCommandPoolCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=qfam};
   VkCommandPool pool;VK(vkCreateCommandPool(dev,&cpci,NULL,&pool));
   VkCommandBufferAllocateInfo cai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=pool,
      .level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
   VkCommandBuffer cmd;VK(vkAllocateCommandBuffers(dev,&cai,&cmd));

   PFN_vkCmdDrawMeshTasksEXT dmt=(PFN_vkCmdDrawMeshTasksEXT)(void*)vkGetDeviceProcAddr(dev,"vkCmdDrawMeshTasksEXT");
   PFN_vkCmdDrawMeshTasksIndirectEXT dmi=(PFN_vkCmdDrawMeshTasksIndirectEXT)(void*)vkGetDeviceProcAddr(dev,"vkCmdDrawMeshTasksIndirectEXT");

   VkCommandBufferBeginInfo cbi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   VK(vkBeginCommandBuffer(cmd,&cbi));
   VkClearValue cv[1]={{.color={.float32={0,0,0,1}}}};
   VkRenderPassBeginInfo rpi={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,.renderPass=rp,.framebuffer=fb,
      .renderArea={{0,0},{64,64}},.clearValueCount=1,.pClearValues=cv};
   vkCmdBeginRenderPass(cmd,&rpi,VK_SUBPASS_CONTENTS_INLINE); /* void in this generation */
   vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline);
   if(visit_set) vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,layout,0,1,&visit_set,0,NULL);

   if(!mesh_mode && !task_mode && !combined_mode && !primgen_mode &&
     !reset_mode && !primgen_large && !probe_mode) {
      /* Hardware baseline: five fullscreen-triangle draws. */
      VkQueryPoolCreateInfo qpci={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
         .queryType=VK_QUERY_TYPE_PIPELINE_STATISTICS,.queryCount=1,
         .pipelineStatistics=VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT|
                             VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT};
      VK(vkCreateQueryPool(dev,&qpci,NULL,&hw_pool));
      vkResetQueryPool(dev,hw_pool,0,qpci.queryCount);
      vkCmdBeginQuery(cmd,hw_pool,0,0);
      for(int i=0;i<5;i++) vkCmdDraw(cmd,3,1,0,0);
      vkCmdEndQuery(cmd,hw_pool,0);
   } else if(combined_mode) {
      /* One pool mixing an HW statistic (vertex invocations, written by
       * the PIPELINESTAT event) with MESH_SHADER_INVOCATIONS (written from
       * GDS snapshots). Validates that the extended result block does not
       * clobber HW-written slots. dmt(3,1,1): 3 task invocations x 2 mesh
       * groups = 6 mesh invocations, each emitting exactly one triangle of
       * 3 vertices; COMBINED_VERTEX_EXPECT is set from the probe run on v7.
       */
      VkQueryPoolCreateInfo qpci={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
         .queryType=VK_QUERY_TYPE_PIPELINE_STATISTICS,.queryCount=1,
         .pipelineStatistics=VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT|
                             VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT};
      VK(vkCreateQueryPool(dev,&qpci,NULL,&hw_pool));
      vkResetQueryPool(dev,hw_pool,0,qpci.queryCount); /* reuse hw_pool var */
      vkCmdBeginQuery(cmd,hw_pool,0,0);
      dmt(cmd,3,1,1);
      vkCmdEndQuery(cmd,hw_pool,0);
   } else if(probe_mode) {
      /* HW statistics only (no meshShaderQueries): what do the PIPELINESTAT
       * counters report for a task/mesh draw on this chip? */
      VkQueryPoolCreateInfo qpci={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
         .queryType=VK_QUERY_TYPE_PIPELINE_STATISTICS,.queryCount=1,
         .pipelineStatistics=VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT|
                             VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT};
      VK(vkCreateQueryPool(dev,&qpci,NULL,&hw_pool));
      vkResetQueryPool(dev,hw_pool,0,qpci.queryCount);
      vkCmdBeginQuery(cmd,hw_pool,0,0);
      dmt(cmd,3,1,1);
      vkCmdEndQuery(cmd,hw_pool,0);
   } else if(primgen_mode) {
      /* Region 1: dmt(3,1,1): 6 mesh invocations x exactly one triangle each.
       * Region 2: empty region must stay 0 (isolation). */
      vkCmdBeginQuery(cmd,pg_pool,0,0);
      dmt(cmd,3,1,1);
      vkCmdEndQuery(cmd,pg_pool,0);
      vkCmdBeginQuery(cmd,pg_pool,1,0);
      vkCmdEndQuery(cmd,pg_pool,1);
   } else if(!strcmp(mode,"zero")) {
      vkCmdBeginQuery(cmd,mesh_pool,0,0);
      vkCmdEndQuery(cmd,mesh_pool,0);
      vkCmdBeginQuery(cmd,task_pool,0,0);
      vkCmdEndQuery(cmd,task_pool,0);
   } else if(!strcmp(mode,"seq")) {
      /* Region 1: 2 task groups; region 2: 5 task groups (per pool). */
      vkCmdBeginQuery(cmd,mesh_pool,0,0); dmt(cmd,2,1,1); vkCmdEndQuery(cmd,mesh_pool,0);
      vkCmdBeginQuery(cmd,mesh_pool,1,0); dmt(cmd,5,1,1); vkCmdEndQuery(cmd,mesh_pool,1);
      vkCmdBeginQuery(cmd,task_pool,0,0); dmt(cmd,2,1,1); vkCmdEndQuery(cmd,task_pool,0);
      vkCmdBeginQuery(cmd,task_pool,1,0); dmt(cmd,5,1,1); vkCmdEndQuery(cmd,task_pool,1);
   } else if(!strcmp(mode,"indirect")||indirect_large||cond_mode) {
      /* One indirect draw of two records per region. Record contents depend
       * on the mode (see above). */
      vkCmdBeginQuery(cmd,mesh_pool,0,0);
      dmi(cmd,buf,4096,2,12);
      vkCmdEndQuery(cmd,mesh_pool,0);
      vkCmdBeginQuery(cmd,task_pool,0,0);
      dmi(cmd,buf,4096,2,12);
      vkCmdEndQuery(cmd,task_pool,0);
   } else if(reset_mode) {
      /* Two draws: one bracketed by the mixed pool, one by the standalone
       * primgen pool. The host resets both pools and resubmits this buffer. */
      vkCmdBeginQuery(cmd,mixed_pool,0,0);
      dmt(cmd,3,1,1);
      vkCmdEndQuery(cmd,mixed_pool,0);
      vkCmdBeginQuery(cmd,rst_pg_pool,0,0);
      dmt(cmd,3,1,1);
      vkCmdEndQuery(cmd,rst_pg_pool,0);
   } else if(primgen_large) {
      /* Prim-gen accounting across v10 sub-record expansion + chunking. */
      vkCmdBeginQuery(cmd,pg_pool,0,0);
      dmi(cmd,buf,4096,2,12);
      vkCmdEndQuery(cmd,pg_pool,0);
   } else {
      /* mesh / task / both / chunked: one region around the direct dispatch. */
      if(mesh_mode) vkCmdBeginQuery(cmd,mesh_pool,0,0);
      if(task_mode) vkCmdBeginQuery(cmd,task_pool,0,0);
      if(TEST_BOUNDARY_AXIS<0) dmt(cmd,task_groups,1,1);
      else if(TEST_BOUNDARY_INDIRECT==1) dmi(cmd,buf,4096,1,12);
      else if(TEST_BOUNDARY_INDIRECT==2) {
         *(uint32_t*)map=1;
         PFN_vkCmdDrawMeshTasksIndirectCountEXT dic=(void*)vkGetDeviceProcAddr(dev,"vkCmdDrawMeshTasksIndirectCountEXT");
         dic(cmd,buf,4096,buf,0,1,12);
      } else dmt(cmd,boundary_dims[0],boundary_dims[1],boundary_dims[2]);
      if(mesh_mode) vkCmdEndQuery(cmd,mesh_pool,0);
      if(task_mode) vkCmdEndQuery(cmd,task_pool,0);
   }

   vkCmdEndRenderPass(cmd);
   if(visited) {
      VkMemoryBarrier mb={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};
      vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&mb,0,NULL,0,NULL);
   }
   VK(vkEndCommandBuffer(cmd));

   VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cmd};
   VK(vkQueueSubmit(queue,1,&si,fence));
   VK(vkWaitForFences(dev,1,&fence,VK_TRUE,30000000000ull));

   if(!mesh_mode && !task_mode && !combined_mode && !primgen_mode &&
     !reset_mode && !primgen_large && !probe_mode) {
      /* hw baseline: one 64-bit word per statistic, in bit order (vertex
       * invocations first, then fragment). */
      uint64_t vals[2]={0,0};
      VkResult r=vkGetQueryPoolResults(dev,hw_pool,0,1,sizeof(vals),vals,sizeof(uint64_t)*2,VK_QUERY_RESULT_64_BIT);
      if(r!=VK_SUCCESS&&r!=VK_SUBOPTIMAL_KHR){fprintf(stderr,"get results: %d\n",r);exit(2);}
      check(vals[0], 5*3, "vertex_shader_invocations");
      check(vals[1], 5*64*64, "fragment_shader_invocations");
   } else if(combined_mode) {
      /* Bit order: VERTEX_SHADER_INVOCATIONS (0x4) before MESH_SHADER_INVOCATIONS_EXT. */
      uint64_t vals[2]={0,0};
      VkResult r=vkGetQueryPoolResults(dev,hw_pool,0,1,sizeof(vals),vals,sizeof(uint64_t)*2,VK_QUERY_RESULT_64_BIT);
      if(r!=VK_SUCCESS&&r!=VK_SUBOPTIMAL_KHR){fprintf(stderr,"get results: %d\n",r);exit(2);}
      check(vals[0], COMBINED_VERTEX_EXPECT, "vertex invocations (HW stat)");
      check(vals[1], 3*mesh_per_task, "mesh invocations (GDS stat, same pool)");
   } else if(probe_mode) {
      uint64_t vals[2]={0,0};
      VkResult r=vkGetQueryPoolResults(dev,hw_pool,0,1,sizeof(vals),vals,sizeof(uint64_t)*2,VK_QUERY_RESULT_64_BIT);
      if(r!=VK_SUCCESS&&r!=VK_SUBOPTIMAL_KHR){fprintf(stderr,"get results: %d\n",r);exit(2);}
      printf("probe vertex_shader_invocations = %llu (mesh draw: 3 task x 2 mesh groups, 1 triangle of 3 verts each)\n",
             (unsigned long long)vals[0]);
      printf("probe fragment_shader_invocations = %llu\n", (unsigned long long)vals[1]);
   } else if(!strcmp(mode,"zero")) {
      check(read_query(mesh_pool,0), 0, "mesh invocations (empty region)");
      check(read_query(task_pool,0), 0, "task invocations (empty region)");
   } else if(!strcmp(mode,"seq")) {
      check(read_query(mesh_pool,0), 2*mesh_per_task, "mesh region 1");
      check(read_query(mesh_pool,1), 5*mesh_per_task, "mesh region 2");
      check(read_query(task_pool,0), 2, "task region 1");
      check(read_query(task_pool,1), 5, "task region 2");
   } else if(primgen_mode) {
      check(read_query(pg_pool,0), 3*mesh_per_task, "mesh primitives generated");
      check(read_query(pg_pool,1), 0, "mesh primitives generated (empty region)");
   } else if(!strcmp(mode,"indirect")) {
      /* two (1,1,1) records per draw: 2 task invocations, 4 mesh. */
      check(read_query(mesh_pool,0), 2*mesh_per_task, "mesh invocations (indirect)");
      check(read_query(task_pool,0), 2, "task invocations (indirect)");
   } else if(indirect_large) {
      /* (8192,1,1) + (5000,1,1): both exceed one chunk; every workgroup must
       * still run exactly once. */
      check(read_query(task_pool,0), 8192+5000, "task invocations (indirect large)");
      check(read_query(mesh_pool,0), (8192+5000)*mesh_per_task, "mesh invocations (indirect large)");
   } else if(cond_mode) {
      /* Two (4,1,1) records; emission alternates per record via DrawID, so
       * each record emits from 2 of its 4 workgroups. Stale slots from the
       * first record must be zeroed by the second's producer or the mesh
       * count would double for record 2. */
      check(read_query(task_pool,0), 8, "task invocations (indirect cond)");
      check(read_query(mesh_pool,0), 4*mesh_per_task, "mesh invocations (indirect cond)");
   } else if(reset_mode) {
      /* Mixed pool bit order in the result array: VERTEX (bit 0), TASK
       * (bit 11), MESH (bit 12). */
      uint64_t vals[3]={~0ull,~0ull,~0ull};
      VkResult r=vkGetQueryPoolResults(dev,mixed_pool,0,1,sizeof(vals),vals,sizeof(uint64_t)*3,VK_QUERY_RESULT_64_BIT);
      if(r!=VK_SUCCESS&&r!=VK_SUBOPTIMAL_KHR){fprintf(stderr,"get results mixed: %d\n",(int)r);exit(2);}
      check(vals[0], COMBINED_VERTEX_EXPECT, "vertex invocations (mixed pool)");
      check(vals[1], 3, "task invocations (mixed pool)");
      check(vals[2], 3*mesh_per_task, "mesh invocations (mixed pool)");
      check(read_query(rst_pg_pool,0), 3*mesh_per_task, "primitives generated (pre-reset)");

      /* Reset makes numerical values undefined; only availability is specified. */
      vkResetQueryPool(dev,mixed_pool,0,1);
      vkResetQueryPool(dev,rst_pg_pool,0,1);
      uint64_t unavailable[4]={~0ull,~0ull,~0ull,~0ull};
      r=vkGetQueryPoolResults(dev,mixed_pool,0,1,sizeof(unavailable),unavailable,sizeof(unavailable),
                              VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
      if(r!=VK_NOT_READY){fprintf(stderr,"mixed reset status: %d\n",(int)r);exit(2);}
      check(unavailable[3],0,"mixed availability after reset");
      uint64_t pg_unavailable[2]={~0ull,~0ull};
      r=vkGetQueryPoolResults(dev,rst_pg_pool,0,1,sizeof(pg_unavailable),pg_unavailable,sizeof(pg_unavailable),
                              VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
      if(r!=VK_NOT_READY){fprintf(stderr,"primgen reset status: %d\n",(int)r);exit(2);}
      check(pg_unavailable[1],0,"primitive availability after reset");

      /* Resubmit the same command buffer: GDS snapshots must re-baseline and
       * reproduce the original counts. */
      VK(vkResetFences(dev,1,&fence));
      VK(vkQueueSubmit(queue,1,&si,fence));
      VK(vkWaitForFences(dev,1,&fence,VK_TRUE,30000000000ull));
      vals[0]=vals[1]=vals[2]=~0ull;
      r=vkGetQueryPoolResults(dev,mixed_pool,0,1,sizeof(vals),vals,sizeof(uint64_t)*3,VK_QUERY_RESULT_64_BIT);
      if(r!=VK_SUCCESS&&r!=VK_SUBOPTIMAL_KHR){fprintf(stderr,"get results mixed resubmit: %d\n",(int)r);exit(2);}
      check(vals[0], COMBINED_VERTEX_EXPECT, "vertex invocations (resubmit)");
      check(vals[1], 3, "task invocations (resubmit)");
      check(vals[2], 3*mesh_per_task, "mesh invocations (resubmit)");
      check(read_query(rst_pg_pool,0), 3*mesh_per_task, "primitives generated (resubmit)");
   } else if(primgen_large) {
      /* (8192+5000) task invocations x 2 mesh groups x exactly one triangle
       * each; the total must stay exact across sub-record expansion and
       * GPU-side chunks. */
      check(read_query(pg_pool,0), (8192+5000)*mesh_per_task, "primitives generated (indirect large)");
   } else {
      #ifndef TEST_TASK_INVOCATIONS
#define TEST_TASK_INVOCATIONS 1
#define TEST_MESH_INVOCATIONS 1
#endif
      uint64_t want_task = task_groups;
      if(mesh_mode && task_mode) {
         check(read_query(mesh_pool,0), want_task*mesh_per_task*TEST_MESH_INVOCATIONS, "mesh invocations");
         check(read_query(task_pool,0), want_task*TEST_TASK_INVOCATIONS, "task invocations");
      } else if(mesh_mode) {
         check(read_query(mesh_pool,0), (uint64_t)task_groups*mesh_per_task, "mesh invocations");
      } else {
         check(read_query(task_pool,0), task_groups, "task invocations");
      }
   }

   if(visited) {
      for(unsigned i=0;i<8194;i++) if(visited[i]!=2) {
         fprintf(stderr,"FAIL visited[%u]=%u expected 2\n",i,visited[i]); exit(2);
      }
      check(visited[8194],0,"payload/builtin errors");
      check(visited[8195],0xa5a5a5a5,"storage guard");
      puts("ok all 8194 task payload identities visited exactly twice");
   }
   printf("PASS %s\n", mode);
   return 0;
}
