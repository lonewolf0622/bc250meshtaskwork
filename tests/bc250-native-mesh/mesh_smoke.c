#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

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
int main(int argc,char **argv) {
   const char *mode=argc>1?argv[1]:"direct";
   int indirect_two=!strcmp(mode,"task-indirect-two"), count_replay=!strcmp(mode,"task-count-replay");
   int indirect_zero=!strcmp(mode,"task-indirect-zero"), count_zero=!strcmp(mode,"task-count-zero");
   int gpu_indirect=!strcmp(mode,"task-resources-indirect");
   int resources=!strncmp(mode,"task-resources",14);
   int native_colors=!strncmp(mode,"native-colors",13);
   int task=!strncmp(mode,"task",4), materialize=strstr(mode,"materialize")!=NULL;
   int vertex=!strncmp(mode,"vertex",6), nodraw=!strcmp(mode,"mesh-nodraw");
   /* v12: expansion-coverage modes (layer routing, non-triangle topologies). */
   int layer_mode=!strcmp(mode,"layer-two"), lines_mode=!strcmp(mode,"lines-two");
   int points_mode=!strcmp(mode,"points-four");
   /* v13: gl_CullPrimitiveEXT coverage (culled primitives must not rasterize). */
   int cull_two=!strcmp(mode,"cull-two");
   VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.pApplicationName="bc250-smoke",.apiVersion=VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
   VkInstance inst;VK(vkCreateInstance(&ici,NULL,&inst));
   uint32_t n=1;VkPhysicalDevice pd;VK(vkEnumeratePhysicalDevices(inst,&n,&pd));
   VkPhysicalDeviceProperties prop;vkGetPhysicalDeviceProperties(pd,&prop);printf("GPU=%s mode=%s\n",prop.deviceName,mode);fflush(stdout);
   VkPhysicalDeviceMeshShaderFeaturesEXT mf={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
   VkPhysicalDeviceFeatures2 f2={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,.pNext=&mf};vkGetPhysicalDeviceFeatures2(pd,&f2);
   printf("meshShader=%u taskShader=%u\n",mf.meshShader,mf.taskShader);fflush(stdout);
   if((!vertex&&!mf.meshShader) || (task&&!mf.taskShader)){fprintf(stderr,"Required feature unavailable\n");return 3;}
   uint32_t nq=0;vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,NULL);VkQueueFamilyProperties *qp=calloc(nq,sizeof(*qp));vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,qp);
   uint32_t qi=0;while(qi<nq&&!(qp[qi].queueFlags&VK_QUEUE_GRAPHICS_BIT))qi++;if(qi==nq)return 2;free(qp);
   float priority=1;VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=qi,.queueCount=1,.pQueuePriorities=&priority};
   const char *ext="VK_EXT_mesh_shader";
   mf=(VkPhysicalDeviceMeshShaderFeaturesEXT){.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,.meshShader=VK_TRUE,.taskShader=task};
   VkPhysicalDeviceFeatures features={.multiDrawIndirect=VK_TRUE};
   VkPhysicalDeviceVulkan12Features f12={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,.pNext=&mf,.drawIndirectCount=VK_TRUE};
   VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=&f12,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci,.enabledExtensionCount=1,.ppEnabledExtensionNames=&ext,.pEnabledFeatures=&features};
   if(vertex){f12.pNext=NULL;dci.enabledExtensionCount=0;}
   VK(vkCreateDevice(pd,&dci,NULL,&dev));vkGetPhysicalDeviceMemoryProperties(pd,&mp);VkQueue queue;vkGetDeviceQueue(dev,qi,0,&queue);
   VkImageCreateInfo imci={.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,.imageType=VK_IMAGE_TYPE_2D,.format=VK_FORMAT_R8G8B8A8_UNORM,.extent={64,64,1},.mipLevels=1,.arrayLayers=layer_mode?4:1,.samples=VK_SAMPLE_COUNT_1_BIT,.tiling=VK_IMAGE_TILING_OPTIMAL,.usage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
   VkImage im;VK(vkCreateImage(dev,&imci,NULL,&im));VkMemoryRequirements req;vkGetImageMemoryRequirements(dev,im,&req);VkDeviceMemory immem=alloc_mem(req,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);VK(vkBindImageMemory(dev,im,immem,0));
   VkImageViewCreateInfo ivci={.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,.image=im,.viewType=layer_mode?VK_IMAGE_VIEW_TYPE_2D_ARRAY:VK_IMAGE_VIEW_TYPE_2D,.format=imci.format,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,layer_mode?4:1}};
   VkImageView view;VK(vkCreateImageView(dev,&ivci,NULL,&view));
   VkAttachmentDescription att={.format=imci.format,.samples=VK_SAMPLE_COUNT_1_BIT,.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,.storeOp=VK_ATTACHMENT_STORE_OP_STORE,.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,.finalLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL};
   VkAttachmentReference ref={0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};VkSubpassDescription sub={.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,.colorAttachmentCount=1,.pColorAttachments=&ref};
   VkSubpassDependency dep={.srcSubpass=0,.dstSubpass=VK_SUBPASS_EXTERNAL,.srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,.dstStageMask=VK_PIPELINE_STAGE_TRANSFER_BIT,.srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT};
   VkRenderPassCreateInfo rpci={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,.attachmentCount=1,.pAttachments=&att,.subpassCount=1,.pSubpasses=&sub,.dependencyCount=1,.pDependencies=&dep};VkRenderPass rp;VK(vkCreateRenderPass(dev,&rpci,NULL,&rp));
   VkFramebufferCreateInfo fbci={.sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,.renderPass=rp,.attachmentCount=1,.pAttachments=&view,.width=64,.height=64,.layers=layer_mode?4:1};VkFramebuffer fb;VK(vkCreateFramebuffer(dev,&fbci,NULL,&fb));
   unsigned img_bytes=64*64*4,layers_n=layer_mode?4u:1u;
   VkBufferCreateInfo bci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=layers_n*img_bytes+2048,.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT|VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};VkBuffer buf;VK(vkCreateBuffer(dev,&bci,NULL,&buf));vkGetBufferMemoryRequirements(dev,buf,&req);VkDeviceMemory bm=alloc_mem(req,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);VK(vkBindBufferMemory(dev,buf,bm,0));
   uint8_t *map;VK(vkMapMemory(dev,bm,0,VK_WHOLE_SIZE,0,(void**)&map));memset(map,0,layers_n*img_bytes+2048);uint32_t *ind=(uint32_t*)(map+64*64*4);ind[0]=1;ind[1]=1;ind[2]=1;ind[3]=1;ind[4]=1;ind[5]=1;ind[6]=1;
   if(count_replay)ind[6]=2;
   if(indirect_zero)ind[0]=0;
   if(count_zero)ind[6]=0;
   VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};VkDescriptorSetLayout app_set=VK_NULL_HANDLE, app_set1=VK_NULL_HANDLE;
   VkDescriptorSetLayout app_layouts[2];
   VkDescriptorPool app_pool=VK_NULL_HANDLE;
   VkDescriptorSet app_sets[2];
   VkPushConstantRange app_push={.stageFlags=VK_SHADER_STAGE_TASK_BIT_EXT,.size=4};
   if(!strcmp(mode,"task-push")){plci.pushConstantRangeCount=1;plci.pPushConstantRanges=&app_push;}
   if(!strcmp(mode,"task-descriptor")||resources){
      VkDescriptorSetLayoutBinding binding={.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_TASK_BIT_EXT};
      VkDescriptorSetLayoutCreateInfo sl={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=1,.pBindings=&binding};
      if(resources){binding.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;binding.stageFlags=VK_SHADER_STAGE_TASK_BIT_EXT|VK_SHADER_STAGE_MESH_BIT_EXT|VK_SHADER_STAGE_COMPUTE_BIT;}
      VK(vkCreateDescriptorSetLayout(dev,&sl,NULL,&app_set));plci.setLayoutCount=1;plci.pSetLayouts=&app_set;
   }
   if(resources){
      VkDescriptorSetLayoutBinding binding={.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_TASK_BIT_EXT|VK_SHADER_STAGE_MESH_BIT_EXT|VK_SHADER_STAGE_COMPUTE_BIT};
      VkDescriptorSetLayoutCreateInfo sl={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=1,.pBindings=&binding};
      VK(vkCreateDescriptorSetLayout(dev,&sl,NULL,&app_set1));
      app_layouts[0]=app_set;app_layouts[1]=app_set1;plci.setLayoutCount=2;plci.pSetLayouts=app_layouts;
      app_push=(VkPushConstantRange){.stageFlags=VK_SHADER_STAGE_TASK_BIT_EXT|VK_SHADER_STAGE_MESH_BIT_EXT|VK_SHADER_STAGE_COMPUTE_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,.offset=240,.size=16};
      plci.pushConstantRangeCount=1;plci.pPushConstantRanges=&app_push;
   }
   if(app_set){
      VkDescriptorPoolSize sizes[2]={{resources?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1}};
      VkDescriptorPoolCreateInfo pci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=2,.poolSizeCount=resources?2:1,.pPoolSizes=sizes};
      VK(vkCreateDescriptorPool(dev,&pci,NULL,&app_pool));
      VkDescriptorSetAllocateInfo sai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=app_pool,.descriptorSetCount=plci.setLayoutCount,.pSetLayouts=plci.pSetLayouts};
      VK(vkAllocateDescriptorSets(dev,&sai,app_sets));
      *(uint32_t*)(map+64*64*4+256)=resources?0xdeadbeefu:0x12345678u;
      *(uint32_t*)(map+64*64*4+512)=77;
      *(uint32_t*)(map+64*64*4+768)=99;
      VkDescriptorBufferInfo bi[2]={{buf,64*64*4+256,4},{buf,64*64*4+768,128}};
      VkWriteDescriptorSet writes[2];
      for(unsigned i=0;i<plci.setLayoutCount;i++)writes[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=app_sets[i],.dstBinding=0,.descriptorCount=1,.descriptorType=sizes[i].type,.pBufferInfo=&bi[i]};
      vkUpdateDescriptorSets(dev,plci.setLayoutCount,writes,0,NULL);
   }
   VkPipelineLayout layout;VK(vkCreatePipelineLayout(dev,&plci,NULL,&layout));
   VkShaderModule ms=shader(vertex?"vert.spv":task?"task-mesh.spv":"mesh.spv"),fs=shader("frag.spv"),ts=task?shader("task.spv"):VK_NULL_HANDLE;
   VkPipelineShaderStageCreateInfo stages[3]={{.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_MESH_BIT_EXT,.module=ms,.pName="main"},{.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_FRAGMENT_BIT,.module=fs,.pName="main"},{.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_TASK_BIT_EXT,.module=ts,.pName="main"}};
   VkViewport vp={0,0,64,64,0,1};VkRect2D sc={{0,0},{64,64}};
   VkPipelineViewportStateCreateInfo vps={.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,.viewportCount=1,.pViewports=&vp,.scissorCount=1,.pScissors=&sc};
   VkPipelineRasterizationStateCreateInfo rs={.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,.polygonMode=VK_POLYGON_MODE_FILL,.cullMode=VK_CULL_MODE_NONE,.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,.lineWidth=1};
   VkPipelineMultisampleStateCreateInfo mss={.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT};
   VkPipelineColorBlendAttachmentState ba={.colorWriteMask=15};VkPipelineColorBlendStateCreateInfo bs={.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,.attachmentCount=1,.pAttachments=&ba};
   VkGraphicsPipelineCreateInfo gp={.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,.stageCount=task?3:2,.pStages=stages,.pViewportState=&vps,.pRasterizationState=&rs,.pMultisampleState=&mss,.pColorBlendState=&bs,.layout=layout,.renderPass=rp};
   VkPipelineVertexInputStateCreateInfo vis={.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia={.sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,.topology=lines_mode?VK_PRIMITIVE_TOPOLOGY_LINE_LIST:points_mode?VK_PRIMITIVE_TOPOLOGY_POINT_LIST:VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   if(vertex){stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT;gp.pVertexInputState=&vis;gp.pInputAssemblyState=&ia;}
   else if(lines_mode||points_mode) gp.pInputAssemblyState=&ia;
   VkPipeline pipeline;VK(vkCreateGraphicsPipelines(dev,VK_NULL_HANDLE,1,&gp,NULL,&pipeline));printf("PIPELINE_CREATED\n");fflush(stdout);
   if(materialize) {printf("PASS materialization\n");return 0;}
   VkPipeline generator_pipeline=VK_NULL_HANDLE;VkShaderModule generator_shader=VK_NULL_HANDLE;
   VkPipeline restore_pipeline=VK_NULL_HANDLE;VkShaderModule restore_shader=VK_NULL_HANDLE;
   if(resources){
      restore_shader=shader("restore.spv");
      VkComputePipelineCreateInfo rci={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=restore_shader,.pName="main"},.layout=layout};
      VK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&rci,NULL,&restore_pipeline));
      if(gpu_indirect){
         generator_shader=shader("generate-indirect.spv");rci.stage.module=generator_shader;
         VK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&rci,NULL,&generator_pipeline));
      }
   }
   VkCommandPoolCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=qi};VkCommandPool pool;VK(vkCreateCommandPool(dev,&cpci,NULL,&pool));VkCommandBufferAllocateInfo cai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};VkCommandBuffer cmd;VK(vkAllocateCommandBuffers(dev,&cai,&cmd));
   VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};VK(vkBeginCommandBuffer(cmd,&begin));VkClearValue clear={.color={{0,0,0,1}}};VkRenderPassBeginInfo rpb={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,.renderPass=rp,.framebuffer=fb,.renderArea=sc,.clearValueCount=1,.pClearValues=&clear};if(resources){
      uint32_t compute_offset=0;
      vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,restore_pipeline);
      vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,2,app_sets,1,&compute_offset);
   }
   if(gpu_indirect){
      vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,generator_pipeline);
      vkCmdDispatch(cmd,1,1,1);
      VkMemoryBarrier generated={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_INDIRECT_COMMAND_READ_BIT|VK_ACCESS_SHADER_READ_BIT};
      vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT|VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT,0,1,&generated,0,NULL,0,NULL);
      vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,restore_pipeline);
   }
   vkCmdBeginRenderPass(cmd,&rpb,VK_SUBPASS_CONTENTS_INLINE);vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline);
   if(app_set){
      uint32_t dynamic_offset=256;
      vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,layout,0,plci.setLayoutCount,app_sets,resources?1:0,resources?&dynamic_offset:NULL);
   }
   if(!strcmp(mode,"task-push")){
      uint32_t value=0x12345678u;
      vkCmdPushConstants(cmd,layout,VK_SHADER_STAGE_TASK_BIT_EXT,0,4,&value);
   }
   if(resources){
      PFN_vkCmdDrawMeshTasksEXT draw=(void*)vkGetDeviceProcAddr(dev,"vkCmdDrawMeshTasksEXT");
      uint32_t values[4]={0x11111111u,0,0xabcdef01u,0};
      vkCmdPushConstants(cmd,layout,app_push.stageFlags,240,16,values);
      if(gpu_indirect){
         PFN_vkCmdDrawMeshTasksIndirectCountEXT indirect=(void*)vkGetDeviceProcAddr(dev,"vkCmdDrawMeshTasksIndirectCountEXT");
         indirect(cmd,buf,64*64*4+768+16,buf,64*64*4+768+40,2,12);
      } else {
         draw(cmd,1,1,1);
         values[0]=0x22222222u;values[1]=1;
         vkCmdPushConstants(cmd,layout,app_push.stageFlags,240,16,values);
         draw(cmd,1,1,1);
      }
   } else if(nodraw) {
      /* Control: render-pass clear and readback, without launching the mesh shader. */
   } else if(vertex) {
      vkCmdDraw(cmd,3,1,0,0);
   } else if(!strcmp(mode,"indirect")||!strcmp(mode,"indirect-two")||!strcmp(mode,"task-indirect")||indirect_two||indirect_zero||!strcmp(mode,"task-draw-limit")) {
      PFN_vkCmdDrawMeshTasksIndirectEXT draw=(void*)vkGetDeviceProcAddr(dev,"vkCmdDrawMeshTasksIndirectEXT");draw(cmd,buf,64*64*4,!strcmp(mode,"task-draw-limit")?4097:(!strcmp(mode,"indirect-two")||indirect_two)?2:1,12);
   } else if(!strcmp(mode,"indirect-count")||!strcmp(mode,"task-indirect-count")||count_replay||count_zero) {
      PFN_vkCmdDrawMeshTasksIndirectCountEXT draw=(void*)vkGetDeviceProcAddr(dev,"vkCmdDrawMeshTasksIndirectCountEXT");draw(cmd,buf,64*64*4,buf,64*64*4+24,2,12);
   } else {
      PFN_vkCmdDrawMeshTasksEXT draw=(void*)vkGetDeviceProcAddr(dev,"vkCmdDrawMeshTasksEXT");draw(cmd,!strcmp(mode,"task-limit")?4097:(!strcmp(mode,"task-two")||!strcmp(mode,"task-skip")||!strcmp(mode,"task-repeat")||!strcmp(mode,"native-colors-two"))?2:1,!strcmp(mode,"task-y")?2:1,!strcmp(mode,"task-z")?2:1);
   }
   vkCmdEndRenderPass(cmd);
   if(resources){
      VkMemoryBarrier ready={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_WRITE_BIT};
      vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&ready,0,NULL,0,NULL);
      vkCmdDispatch(cmd,1,1,1);
   }
   if(layer_mode){for(unsigned l=0;l<4;l++){VkBufferImageCopy lc={.bufferOffset=l*img_bytes,.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,l,1},.imageExtent={64,64,1}};vkCmdCopyImageToBuffer(cmd,im,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,buf,1,&lc);}}
   else {VkBufferImageCopy copy={.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.imageExtent={64,64,1}};vkCmdCopyImageToBuffer(cmd,im,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,buf,1,&copy);}
   VkMemoryBarrier barrier={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,NULL,0,NULL);VK(vkEndCommandBuffer(cmd));
   if(strstr(mode,"record-only")){printf("PASS command recording (no GPU submission)\n");return 0;}
   VkFenceCreateInfo fci={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};VkFence fence;VK(vkCreateFence(dev,&fci,NULL,&fence));VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cmd};printf("SUBMIT\n");fflush(stdout);for(unsigned replay=0;replay<(!strcmp(mode,"task-repeat")?10u:1u);replay++){VK(vkResetFences(dev,1,&fence));VK(vkQueueSubmit(queue,1,&si,fence));VK(vkWaitForFences(dev,1,&fence,VK_TRUE,10000000000ull));}
   unsigned pixels=0,red=0,green=0;for(unsigned i=0;i<4096;i++){pixels+=!!(map[i*4]|map[i*4+1]|map[i*4+2]);red+=map[i*4]>200&&map[i*4+1]<20;green+=map[i*4+1]>200&&map[i*4]<20;}
   unsigned c=(32*64+32)*4;printf("center_rgba=%u,%u,%u,%u NON_BLACK_PIXELS=%u red=%u green=%u\n",map[c],map[c+1],map[c+2],map[c+3],pixels,red,green);
   unsigned tl=map[(9*64+16)*4],tr=map[(9*64+48)*4],bl=map[(54*64+16)*4],br=map[(54*64+48)*4];
   if(cull_two)printf("cull_samples TL=%u TR=%u BL=%u BR=%u\n",tl,tr,bl,br);
   FILE *out=fopen("last-frame.ppm","wb");fprintf(out,"P6\n64 64\n255\n");for(unsigned i=0;i<4096;i++)fwrite(map+4*i,1,3,out);fclose(out);
   unsigned lp[4]={0},lr[4]={0},lg[4]={0};
   if(layer_mode){for(unsigned l=0;l<4;l++)for(unsigned i=0;i<4096;i++){unsigned o=(l*4096+i)*4;lp[l]+=!!(map[o]|map[o+1]|map[o+2]);lr[l]+=map[o]>200&&map[o+1]<20;lg[l]+=map[o+1]>200&&map[o]<20;}
      printf("layer_pixels=%u,%u,%u,%u layer_red=%u,%u,%u,%u layer_green=%u,%u,%u,%u\n",lp[0],lp[1],lp[2],lp[3],lr[0],lr[1],lr[2],lr[3],lg[0],lg[1],lg[2],lg[3]);}
   int pass=layer_mode?(lp[0]==lr[0]&&lr[0]>400&&lp[3]==lg[3]&&lg[3]>400&&lp[1]==0&&lp[2]==0):lines_mode?(red>20&&green>20&&pixels==red+green):points_mode?(red>100&&green>100&&pixels==red+green):native_colors?(red>400&&green>400&&pixels==red+green):(count_zero||indirect_zero)?pixels==0:(indirect_two||count_replay)?(red>400&&green>400&&pixels==red+green):resources?(red>400&&green>400&&pixels==red+green):nodraw?pixels==0:!strcmp(mode,"task-skip")?(red==0&&green>400&&pixels==green):(!strcmp(mode,"task-two")||!strcmp(mode,"task-y")||!strcmp(mode,"task-z")||!strcmp(mode,"task-fanout")||!strcmp(mode,"task-repeat"))?(red>400&&green>400&&pixels==red+green):cull_two?(tl>200&&tr>200&&bl==0&&br==0):(red>400&&map[c]>200&&map[c+1]<20);
   if(resources){
      uint32_t *observed=(uint32_t*)(map+64*64*4+768);
      printf("restored_compute: uniform=%08x challenge=%08x magic=%08x\n",observed[1],observed[2],observed[3]);
      pass=pass&&observed[1]==0xdeadbeefu&&observed[2]==(gpu_indirect?0x11111111u:0x22222222u)&&observed[3]==0xabcdef01u;
   }
   if(count_replay){
      ind[6]=0;
      VK(vkResetFences(dev,1,&fence));VK(vkQueueSubmit(queue,1,&si,fence));VK(vkWaitForFences(dev,1,&fence,VK_TRUE,10000000000ull));
      unsigned replay_pixels=0;for(unsigned i=0;i<4096;i++)replay_pixels+=!!(map[4*i]|map[4*i+1]|map[4*i+2]);
      printf("zero-count replay pixels=%u\n",replay_pixels);pass=pass&&replay_pixels==0;
   }
   printf("%s\n",pass?"PASS":"FAIL");
   vkDestroyFence(dev,fence,NULL);vkDestroyCommandPool(dev,pool,NULL);vkDestroyPipeline(dev,pipeline,NULL);if(generator_pipeline)vkDestroyPipeline(dev,generator_pipeline,NULL);if(generator_shader)vkDestroyShaderModule(dev,generator_shader,NULL);if(restore_pipeline)vkDestroyPipeline(dev,restore_pipeline,NULL);if(restore_shader)vkDestroyShaderModule(dev,restore_shader,NULL);vkDestroyShaderModule(dev,ms,NULL);vkDestroyShaderModule(dev,fs,NULL);if(ts)vkDestroyShaderModule(dev,ts,NULL);vkDestroyPipelineLayout(dev,layout,NULL);if(app_pool)vkDestroyDescriptorPool(dev,app_pool,NULL);if(app_set1)vkDestroyDescriptorSetLayout(dev,app_set1,NULL);if(app_set)vkDestroyDescriptorSetLayout(dev,app_set,NULL);vkUnmapMemory(dev,bm);vkDestroyBuffer(dev,buf,NULL);vkFreeMemory(dev,bm,NULL);vkDestroyFramebuffer(dev,fb,NULL);vkDestroyRenderPass(dev,rp,NULL);vkDestroyImageView(dev,view,NULL);vkDestroyImage(dev,im,NULL);vkFreeMemory(dev,immem,NULL);vkDestroyDevice(dev,NULL);vkDestroyInstance(inst,NULL);return !pass;
}
