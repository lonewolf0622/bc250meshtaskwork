/* DGC: one application Mesh/Task draw of one workgroup, then RGBA8 readback.
 * Build/audit offline first. Real execution requires a separate explicit owner OK.
 * Driver-internal attachment clear/copy work is additional to the one Mesh draw.
 */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#define SIDE 256
#define BYTES (SIDE * SIDE * 4)
#define CK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { \
   fprintf(stderr, "FAIL %s: %d\n", #x, r); return 1; } } while (0)
static unsigned validation_errors;
static VkDevice gate_device;
static int gate_db, gate_many;
static VkBufferUsageFlags dgc_extra_usage;
static VkBuffer owned_buffers[8];
static VkDeviceMemory owned_memory[8];
static unsigned owned_count;
static VkIndirectCommandsLayoutEXT owned_layout;
static VKAPI_ATTR VkBool32 VKAPI_CALL
validation_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT type,
                    const VkDebugUtilsMessengerCallbackDataEXT *data, void *user)
{
   (void)type; (void)user;
   if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
      validation_errors++;
      fprintf(stderr, "API_VALIDATION_ERROR: %s\n", data->pMessage);
   }
   return VK_FALSE;
}
static uint32_t memtype(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags flags)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags) return i;
   fputs("Required memory type unavailable\n", stderr); exit(2);
}
static VkShaderModule gate_load(VkDevice dev, const char *name)
{
   FILE *f = fopen(name, "rb");
   if (!f) { perror(name); exit(2); }
   if (fseek(f, 0, SEEK_END)) exit(2);
   long n = ftell(f);
   if (n <= 0 || n % 4 || n > 1048576) exit(2);
   rewind(f);
   uint32_t *bytes = malloc((size_t)n);
   if (!bytes || fread(bytes, 1, (size_t)n, f) != (size_t)n || fclose(f)) exit(2);
   VkShaderModuleCreateInfo ci = {.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize=(size_t)n, .pCode=bytes};
   VkShaderModule module;
   if (vkCreateShaderModule(dev, &ci, NULL, &module)) exit(2);
   free(bytes); return module;
}
static VkDevice dgc_device;
static VkPhysicalDevice dgc_physical;
static VkPipeline dgc_pipeline;
static VkPipelineLayout dgc_pipeline_layout;
static int dgc_failed;
#define CHECK(x) do { VkResult r = (x); if (r) { fprintf(stderr,"DGC_ERROR %s %d\n",#x,r); dgc_failed=1; return; } } while(0)
static void dgc_buffer(VkDeviceSize bytes, uint32_t bits, VkBuffer *buffer, VkDeviceMemory *memory, void **map)
{
   VkBufferCreateInfo ci = {.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=bytes,
      .usage=VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT|
             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
   VkBufferUsageFlags2CreateInfo usage = {.sType=VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO,
      .usage=ci.usage | (bits != ~0u ? VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT : 0)};
   ci.usage |= dgc_extra_usage;
   usage.usage |= dgc_extra_usage;
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
   if(owned_count==8){dgc_failed=1;return;}
   owned_buffers[owned_count]=*buffer;owned_memory[owned_count++]=*memory;
}
static VkDeviceAddress dgc_address(VkBuffer buffer)
{
   VkBufferDeviceAddressInfo i={.sType=VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,.buffer=buffer};
   return vkGetBufferDeviceAddress(dgc_device,&i);
}
static void dgc_draw(VkCommandBuffer cb,int task)
{
   PFN_vkCreateIndirectCommandsLayoutEXT create=(void *)vkGetDeviceProcAddr(dgc_device,"vkCreateIndirectCommandsLayoutEXT");
   PFN_vkGetGeneratedCommandsMemoryRequirementsEXT requirements=(void *)vkGetDeviceProcAddr(dgc_device,"vkGetGeneratedCommandsMemoryRequirementsEXT");
   PFN_vkCmdExecuteGeneratedCommandsEXT execute=(void *)vkGetDeviceProcAddr(dgc_device,"vkCmdExecuteGeneratedCommandsEXT");
   if(!create||!requirements||!execute){dgc_failed=1;return;}
   unsigned draws=gate_many?64:1;
   VkIndirectCommandsLayoutTokenEXT token={.sType=VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT,
      .type=gate_many?VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_COUNT_EXT:VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_EXT,.offset=gate_db?4:0};
   VkIndirectCommandsPushConstantTokenEXT pc={.updateRange={VK_SHADER_STAGE_MESH_BIT_EXT|VK_SHADER_STAGE_TASK_BIT_EXT,0,4}};
   VkIndirectCommandsLayoutTokenEXT tokens[]={
      {.sType=VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT,.type=VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_EXT,
       .offset=0,.data.pPushConstant=&pc},token};
   unsigned stream_stride=gate_many?16:12;
   if(gate_db)stream_stride+=4;
   VkIndirectCommandsLayoutCreateInfoEXT ci={.sType=VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT,
      .shaderStages=VK_SHADER_STAGE_MESH_BIT_EXT|VK_SHADER_STAGE_FRAGMENT_BIT|(task?VK_SHADER_STAGE_TASK_BIT_EXT:0),
      .pipelineLayout=dgc_pipeline_layout,.indirectStride=stream_stride,.tokenCount=gate_db?2:1,.pTokens=gate_db?tokens:&token};
   CHECK(create(dgc_device,&ci,NULL,&owned_layout));
   VkGeneratedCommandsPipelineInfoEXT pi={.sType=VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT,.pipeline=dgc_pipeline};
   VkGeneratedCommandsMemoryRequirementsInfoEXT mi={.sType=VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_EXT,
      .pNext=&pi,.indirectCommandsLayout=owned_layout,.maxSequenceCount=1,.maxDrawCount=draws};
   VkMemoryRequirements2 mr={.sType=VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};requirements(dgc_device,&mi,&mr);
   if(!mr.memoryRequirements.size){dgc_failed=1;return;}
   VkBuffer stream,output;VkDeviceMemory sm,om;void *map,*out;
   dgc_buffer(gate_many?32+12*draws:16,~0u,&stream,&sm,&map);if(dgc_failed)return;
   dgc_buffer(mr.memoryRequirements.size,mr.memoryRequirements.memoryTypeBits,&output,&om,&out);if(dgc_failed)return;
   uint32_t *words=map;
   unsigned data_offset=gate_db?1:0;
   if(gate_many) {
      uint64_t address=dgc_address(stream)+32;memcpy(words+data_offset,&address,8);words[data_offset+2]=12;words[data_offset+3]=draws;
      for(unsigned draw=0;draw<draws;draw++){words[8+draw*3]=1;words[9+draw*3]=1;words[10+draw*3]=1;}
   } else {words[data_offset]=1;words[data_offset+1]=1;words[data_offset+2]=1;}
   if(gate_db)words[0]=7;
   VkGeneratedCommandsInfoEXT info={.sType=VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT,.pNext=&pi,
      .shaderStages=ci.shaderStages,.indirectCommandsLayout=owned_layout,.indirectAddress=dgc_address(stream),
      .indirectAddressSize=stream_stride,.preprocessAddress=dgc_address(output),.preprocessSize=mr.memoryRequirements.size,
      .maxSequenceCount=1,.maxDrawCount=draws};
   execute(cb,VK_FALSE,&info);
}

int main(int argc, char **argv)
{
   setvbuf(stdout, NULL, _IONBF, 0);
   if (argc != 6 || (strcmp(argv[1], "--offline") && strcmp(argv[1], "--authorized-gpu-DGC"))) {
      fputs("usage: fast_gate --offline|--authorized-gpu-DGC MESH.spv FRAG.spv TASK.spv|- OUT.rgba\n", stderr); return 2;
   }
   const int task = strcmp(argv[4], "-") != 0;
   gate_db=getenv("GATE_KIND")&&!strcmp(getenv("GATE_KIND"),"descriptor-buffers");
   gate_many=getenv("GATE_KIND")&&!strcmp(getenv("GATE_KIND"),"many-task");
   if((gate_db||gate_many)&&!task)return 2;
   const int offline = !strcmp(argv[1], "--offline");
   const char *gpu=getenv("AMDGPU_GPU_ID"), *preload=getenv("LD_PRELOAD"), *safe=getenv("RADV_BC250_MESH_SAFE_FAST");
   if (!offline && (!safe || strcmp(safe, "1"))) { fputs("REFUSE: SAFE_FAST must be enabled\n", stderr); return 3; }
   if (offline) {
      if (access("/dev/dri", F_OK)==0 || !gpu || strcmp(gpu,"gfx1013") || !preload ||
          !strstr(preload,"libamdgpu_noop_drm_shim.so")) {
         fputs("REFUSE: offline mode requires bwrap/noop with no /dev/dri\n", stderr); return 3;
      }
   } else if (access("/dev/dri", F_OK)!=0 || gpu || preload ||
              !getenv("BC250_DGC_OWNER_AUTHORIZED") || strcmp(getenv("BC250_DGC_OWNER_AUTHORIZED"),"DGC-once")) {
      fputs("REFUSE: hardware mode requires the authorized one-shot launcher\n", stderr); return 3;
   }
   VkApplicationInfo app = {.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName="BC250 DGC single draw",
      .apiVersion=VK_API_VERSION_1_3};
   VkDebugUtilsMessengerCreateInfoEXT debug={.sType=VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
      .messageSeverity=VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
      .messageType=VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
      .pfnUserCallback=validation_callback};
   const char *instance_extensions[]={VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
   VkInstanceCreateInfo ici = {.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo=&app,
      .pNext=&debug,.enabledExtensionCount=1,.ppEnabledExtensionNames=instance_extensions};
   VkInstance instance; CK(vkCreateInstance(&ici,NULL,&instance));
   VkDebugUtilsMessengerEXT messenger=VK_NULL_HANDLE;
   PFN_vkCreateDebugUtilsMessengerEXT create_messenger=(void *)vkGetInstanceProcAddr(instance,"vkCreateDebugUtilsMessengerEXT");
   CK(create_messenger(instance,&debug,NULL,&messenger));
   uint32_t np=0; CK(vkEnumeratePhysicalDevices(instance,&np,NULL));
   if (np != 1) { fprintf(stderr,"REFUSE: expected one physical device, got %u\n",np); return 3; }
   VkPhysicalDevice pd; CK(vkEnumeratePhysicalDevices(instance,&np,&pd));
   VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd,&props);
   if (props.vendorID != 0x1002 || (!strstr(props.deviceName,"GFX1013") && !strstr(props.deviceName,"gfx1013"))) {
      fputs("REFUSE: not GFX1013\n",stderr); return 3;
   }
   VkPhysicalDeviceMeshShaderFeaturesEXT mesh_support={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
   VkPhysicalDeviceVulkan13Features support13={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,.pNext=&mesh_support};
   VkPhysicalDeviceFeatures2 features={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,.pNext=&support13};
   vkGetPhysicalDeviceFeatures2(pd,&features);
   if (!mesh_support.meshShader || !support13.maintenance4) { fputs("REFUSE: Mesh unavailable\n",stderr); return 3; }
   uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,NULL);
   VkQueueFamilyProperties *qp=calloc(nq,sizeof(*qp)); if (!qp) return 2;
   vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,qp);
   uint32_t family=0; while (family<nq && !(qp[family].queueFlags&VK_QUEUE_GRAPHICS_BIT)) family++;
   free(qp); if (family==nq) return 2;
   float priority=1;
   VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=family,
      .queueCount=1,.pQueuePriorities=&priority};
   VkPhysicalDeviceMeshShaderFeaturesEXT mesh_enable={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,.meshShader=VK_TRUE,.taskShader=mesh_support.taskShader};
   VkPhysicalDeviceVulkan13Features enable13={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .pNext=&mesh_enable,.maintenance4=VK_TRUE};
   VkPhysicalDeviceVulkan12Features enable12={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext=&enable13,.bufferDeviceAddress=VK_TRUE};
   VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT dgc_enable={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_EXT,
      .deviceGeneratedCommands=VK_TRUE};
   VkPhysicalDeviceVulkan11Features enable11={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
      .pNext=&enable12,.shaderDrawParameters=VK_TRUE};
   VkPhysicalDeviceMaintenance5FeaturesKHR maintenance5={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,
      .pNext=&enable11,.maintenance5=VK_TRUE};
   dgc_enable.pNext=&maintenance5;
   VkPhysicalDeviceDescriptorBufferFeaturesEXT db_enable={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT,
      .pNext=&dgc_enable,.descriptorBuffer=VK_TRUE,.descriptorBufferPushDescriptors=VK_TRUE};
   const char *extensions[]={VK_EXT_MESH_SHADER_EXTENSION_NAME,VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME,VK_KHR_MAINTENANCE_5_EXTENSION_NAME,VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};
   VkPhysicalDeviceFeatures core={.multiDrawIndirect=VK_TRUE};
   VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=gate_db?(void *)&db_enable:(void *)&dgc_enable,
      .queueCreateInfoCount=1,.pQueueCreateInfos=&qci,.enabledExtensionCount=gate_db?5:3,.ppEnabledExtensionNames=extensions,.pEnabledFeatures=&core};
   VkDevice dev; CK(vkCreateDevice(pd,&dci,NULL,&dev));
   gate_device = dev;
   VkQueue queue; vkGetDeviceQueue(dev,family,0,&queue);
   dgc_device=dev;dgc_physical=pd;
   VkDescriptorSetLayout descriptor_layouts[2]={0};
   VkDescriptorSetLayoutBinding binding={.binding=0,.descriptorCount=1,
      .stageFlags=VK_SHADER_STAGE_MESH_BIT_EXT|VK_SHADER_STAGE_TASK_BIT_EXT,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
   VkPushConstantRange range={VK_SHADER_STAGE_MESH_BIT_EXT|VK_SHADER_STAGE_TASK_BIT_EXT,0,4};
   if(gate_db) {
      VkDescriptorSetLayoutCreateInfo set={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
         .flags=VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT,.bindingCount=1,.pBindings=&binding};
      CK(vkCreateDescriptorSetLayout(dev,&set,NULL,&descriptor_layouts[0]));
      binding.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      set.flags|=VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
      CK(vkCreateDescriptorSetLayout(dev,&set,NULL,&descriptor_layouts[1]));
   }
   VkPipelineLayoutCreateInfo lci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount=gate_db?2:0,.pSetLayouts=descriptor_layouts,.pushConstantRangeCount=gate_db?1:0,.pPushConstantRanges=&range};
   VkPipelineLayout layout; CK(vkCreatePipelineLayout(dev,&lci,NULL,&layout));
   VkAttachmentDescription attachment={.format=VK_FORMAT_R8G8B8A8_UNORM,.samples=VK_SAMPLE_COUNT_1_BIT,
      .loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,.storeOp=VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,.finalLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL};
   VkAttachmentReference color={0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   VkSubpassDescription subpass={.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,.colorAttachmentCount=1,.pColorAttachments=&color};
   VkSubpassDependency dependencies[2]={
      {.srcSubpass=VK_SUBPASS_EXTERNAL,.dstSubpass=0,.srcStageMask=VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
       .dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,.dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT},
      {.srcSubpass=0,.dstSubpass=VK_SUBPASS_EXTERNAL,.srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
       .dstStageMask=VK_PIPELINE_STAGE_TRANSFER_BIT,.srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
       .dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT}};
   VkRenderPassCreateInfo rpci={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,.attachmentCount=1,.pAttachments=&attachment,
      .subpassCount=1,.pSubpasses=&subpass,.dependencyCount=2,.pDependencies=dependencies};
   VkRenderPass rp; CK(vkCreateRenderPass(dev,&rpci,NULL,&rp));
   VkShaderModule modules[3]={gate_load(dev,argv[2]),gate_load(dev,argv[3]),task?gate_load(dev,argv[4]):VK_NULL_HANDLE};
   VkPipelineShaderStageCreateInfo stages[3]={
      {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_MESH_BIT_EXT,.module=modules[0],.pName="main"},
      {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_FRAGMENT_BIT,.module=modules[1],.pName="main"},
      {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_TASK_BIT_EXT,.module=modules[2],.pName="main"}};
   VkViewport viewport={0,0,SIDE,SIDE,0,1}; VkRect2D rect={{0,0},{SIDE,SIDE}};
   VkPipelineViewportStateCreateInfo vs={.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount=1,.pViewports=&viewport,.scissorCount=1,.pScissors=&rect};
   VkPipelineRasterizationStateCreateInfo rs={.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode=VK_POLYGON_MODE_FILL,.cullMode=VK_CULL_MODE_NONE,.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,.lineWidth=1};
   VkPipelineMultisampleStateCreateInfo ms={.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT};
   VkPipelineColorBlendAttachmentState blend={.colorWriteMask=15};
   VkPipelineColorBlendStateCreateInfo bs={.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,.attachmentCount=1,.pAttachments=&blend};
   VkGraphicsPipelineCreateInfo pci={.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,.stageCount=task?3:2,.pStages=stages,
      .pViewportState=&vs,.pRasterizationState=&rs,.pMultisampleState=&ms,.pColorBlendState=&bs,.layout=layout,.renderPass=rp};
   VkPipelineCreateFlags2CreateInfoKHR flags2={.sType=VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR,
      .flags=VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT};
   if(gate_db)pci.pNext=&flags2;
   VkPipeline pipeline; CK(vkCreateGraphicsPipelines(dev,VK_NULL_HANDLE,1,&pci,NULL,&pipeline));
   VkImageCreateInfo image_ci={.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,.imageType=VK_IMAGE_TYPE_2D,
      .format=VK_FORMAT_R8G8B8A8_UNORM,.extent={SIDE,SIDE,1},.mipLevels=1,.arrayLayers=1,.samples=VK_SAMPLE_COUNT_1_BIT,
      .tiling=VK_IMAGE_TILING_OPTIMAL,.usage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
   VkImage image; CK(vkCreateImage(dev,&image_ci,NULL,&image));
   VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev,image,&mr);
   VkMemoryAllocateInfo mai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=mr.size,.memoryTypeIndex=memtype(pd,mr.memoryTypeBits,0)};
   VkDeviceMemory image_mem; CK(vkAllocateMemory(dev,&mai,NULL,&image_mem)); CK(vkBindImageMemory(dev,image,image_mem,0));
   VkImageViewCreateInfo ivci={.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,.image=image,.viewType=VK_IMAGE_VIEW_TYPE_2D,
      .format=VK_FORMAT_R8G8B8A8_UNORM,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
   VkImageView view; CK(vkCreateImageView(dev,&ivci,NULL,&view));
   VkFramebufferCreateInfo fbci={.sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,.renderPass=rp,.attachmentCount=1,
      .pAttachments=&view,.width=SIDE,.height=SIDE,.layers=1};
   VkFramebuffer fb; CK(vkCreateFramebuffer(dev,&fbci,NULL,&fb));
   VkBufferCreateInfo bci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=BYTES,.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT};
   VkBuffer buffer; CK(vkCreateBuffer(dev,&bci,NULL,&buffer)); vkGetBufferMemoryRequirements(dev,buffer,&mr);
   mai.allocationSize=mr.size; mai.memoryTypeIndex=memtype(pd,mr.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   VkDeviceMemory buffer_mem; CK(vkAllocateMemory(dev,&mai,NULL,&buffer_mem)); CK(vkBindBufferMemory(dev,buffer,buffer_mem,0));
   void *mapped; CK(vkMapMemory(dev,buffer_mem,0,VK_WHOLE_SIZE,0,&mapped)); memset(mapped,0xa5,BYTES);
   VkCommandPoolCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=family};
   VkCommandPool pool; CK(vkCreateCommandPool(dev,&cpci,NULL,&pool));
   VkCommandBufferAllocateInfo cbai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=pool,
      .level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
   VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev,&cbai,&cb));
   VkCommandBufferBeginInfo cbi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
   CK(vkBeginCommandBuffer(cb,&cbi));
   VkClearValue clear={.color={{0,0,0,0}}};
   VkRenderPassBeginInfo rbi={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,.renderPass=rp,.framebuffer=fb,
      .renderArea=rect,.clearValueCount=1,.pClearValues=&clear};
   VkBuffer indirect=VK_NULL_HANDLE;
   if(gate_many) {
      VkDeviceMemory memory;void *map;dgc_buffer(64*12,~0u,&indirect,&memory,&map);if(dgc_failed)return 1;
      for(unsigned i=0;i<64;i++) {uint32_t xyz[]={1,1,1};memcpy((char *)map+i*12,xyz,12);}
   }
   vkCmdBeginRenderPass(cb,&rbi,VK_SUBPASS_CONTENTS_INLINE); vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline);
   if(gate_db) {
      VkPhysicalDeviceDescriptorBufferPropertiesEXT properties={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT};
      VkPhysicalDeviceProperties2 p2={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,.pNext=&properties};vkGetPhysicalDeviceProperties2(pd,&p2);
      if(!properties.bufferlessPushDescriptors){fputs("REFUSE: unproven push descriptor buffer handle\n",stderr);return 3;}
      VkBuffer resource,table;VkDeviceMemory rm,tm;void *resource_map,*table_map;
      dgc_extra_usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
      dgc_buffer(256,~0u,&resource,&rm,&resource_map);if(dgc_failed)return 1;
      ((uint32_t *)resource_map)[0]=11;((uint32_t *)resource_map)[4]=13;
      dgc_extra_usage=VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT;
      dgc_buffer(256,~0u,&table,&tm,&table_map);if(dgc_failed)return 1;
      dgc_extra_usage=0;
      PFN_vkGetDescriptorEXT get=(void *)vkGetDeviceProcAddr(dev,"vkGetDescriptorEXT");
      PFN_vkGetDescriptorSetLayoutBindingOffsetEXT binding_offset=(void *)vkGetDeviceProcAddr(dev,"vkGetDescriptorSetLayoutBindingOffsetEXT");
      VkDeviceSize offset;binding_offset(dev,descriptor_layouts[0],0,&offset);
      if(offset+properties.storageBufferDescriptorSize>256)return 2;
      VkDescriptorAddressInfoEXT address={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,.address=dgc_address(resource),.range=16};
      VkDescriptorGetInfoEXT get_info={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.data.pStorageBuffer=&address};
      get(dev,&get_info,properties.storageBufferDescriptorSize,(char *)table_map+offset);
      VkDescriptorBufferBindingInfoEXT buffer_binding={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
         .address=dgc_address(table),.usage=VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT};
      PFN_vkCmdBindDescriptorBuffersEXT bind=(void *)vkGetDeviceProcAddr(dev,"vkCmdBindDescriptorBuffersEXT");bind(cb,1,&buffer_binding);
      PFN_vkCmdSetDescriptorBufferOffsetsEXT offsets=(void *)vkGetDeviceProcAddr(dev,"vkCmdSetDescriptorBufferOffsetsEXT");
      unsigned index=0;VkDeviceSize set_offset=0;offsets(cb,VK_PIPELINE_BIND_POINT_GRAPHICS,layout,0,1,&index,&set_offset);
      VkDescriptorBufferInfo root={.buffer=resource,.offset=16,.range=16};
      VkWriteDescriptorSet write={.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstBinding=0,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,.pBufferInfo=&root};
      PFN_vkCmdPushDescriptorSetKHR push=(void *)vkGetDeviceProcAddr(dev,"vkCmdPushDescriptorSetKHR");push(cb,VK_PIPELINE_BIND_POINT_GRAPHICS,layout,1,1,&write);
      uint32_t value=7;vkCmdPushConstants(cb,layout,range.stageFlags,0,4,&value);
      puts("GATE_DESCRIPTOR_BUFFERS root_cbv=13 root_constant=7 resource=11 producer_and_consumer=1");
   }
   PFN_vkCmdDrawMeshTasksEXT draw=(PFN_vkCmdDrawMeshTasksEXT)vkGetDeviceProcAddr(dev,"vkCmdDrawMeshTasksEXT");
   if (!draw) return 2;
   if (!getenv("GATE_MODE") || strcmp(getenv("GATE_MODE"),"feature")) {
      if(gate_many) {PFN_vkCmdDrawMeshTasksIndirectEXT draw_many=(void *)vkGetDeviceProcAddr(dev,"vkCmdDrawMeshTasksIndirectEXT");draw_many(cb,indirect,0,64,12);}
      else draw(cb,1,1,1);
   }
   else { dgc_device=dev;dgc_physical=pd;dgc_pipeline=pipeline;dgc_pipeline_layout=layout;dgc_draw(cb,task);if(dgc_failed)return 1; }
   vkCmdEndRenderPass(cb);
   VkBufferImageCopy copy={.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.imageExtent={SIDE,SIDE,1}};
   vkCmdCopyImageToBuffer(cb,image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,buffer,1,&copy);
   VkBufferMemoryBarrier host={.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask=VK_ACCESS_HOST_READ_BIT,.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
      .buffer=buffer,.size=VK_WHOLE_SIZE};
   vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,NULL,1,&host,0,NULL);
   CK(vkEndCommandBuffer(cb));
   if (validation_errors) { fputs("REFUSE: validation errors before submit\n",stderr); return 1; }
   VkFenceCreateInfo fci={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; CK(vkCreateFence(dev,&fci,NULL,&fence));
   VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb};
   printf("DGC_READY mode=%s mesh_draws=%u workgroups=1,1,1 submits=1\n",offline?"noop":"hardware",gate_many?64:1);
   /* The one-shot launcher verifies the actual compilation before releasing
    * this single submission. EOF, a bad token, or launcher failure stops here. */
   if (!offline) {
      char token=0;
      if (read(STDIN_FILENO,&token,1)!=1 || token!='S') {
         fputs("REFUSE: submission not released by audited launcher\n",stderr); return 3;
      }
   }
   CK(vkQueueSubmit(queue,1,&submit,fence));
   VkResult wait=vkWaitForFences(dev,1,&fence,VK_TRUE,30000000000ull);
   if (wait!=VK_SUCCESS) { fprintf(stderr,"DGC_STOP fence=%d; retain .last; do not retry\n",wait); fflush(NULL); _Exit(124); }
   if (offline) {
      const unsigned char *p=mapped;
      for (unsigned i=0;i<BYTES;i++) if (p[i]!=0xa5) { fputs("NOOP_SENTINEL_CHANGED\n",stderr); return 1; }
      puts("NOOP_COMPLETE pixels_not_rendered");
   } else {
      FILE *out=fopen(argv[5],"wbx"); if (!out) { perror(argv[5]); return 2; }
      if (fwrite(mapped,1,BYTES,out)!=BYTES || fflush(out) || fsync(fileno(out)) || fclose(out)) return 2;
      puts("DGC_FENCE_COMPLETE readback_saved");
   }
   vkDestroyFence(dev,fence,NULL); vkDestroyCommandPool(dev,pool,NULL);
   vkUnmapMemory(dev,buffer_mem); vkDestroyBuffer(dev,buffer,NULL); vkFreeMemory(dev,buffer_mem,NULL);
   vkDestroyFramebuffer(dev,fb,NULL); vkDestroyImageView(dev,view,NULL); vkDestroyImage(dev,image,NULL); vkFreeMemory(dev,image_mem,NULL);
   vkDestroyPipeline(dev,pipeline,NULL); for (unsigned i=0;i<(task?3:2);i++) vkDestroyShaderModule(dev,modules[i],NULL);
   vkDestroyRenderPass(dev,rp,NULL); vkDestroyPipelineLayout(dev,layout,NULL);
   if(owned_layout){PFN_vkDestroyIndirectCommandsLayoutEXT destroy=(void *)vkGetDeviceProcAddr(dev,"vkDestroyIndirectCommandsLayoutEXT");destroy(dev,owned_layout,NULL);}
   for(unsigned i=0;i<owned_count;i++){vkUnmapMemory(dev,owned_memory[i]);vkDestroyBuffer(dev,owned_buffers[i],NULL);vkFreeMemory(dev,owned_memory[i],NULL);}
   for(unsigned i=0;i<2;i++)if(descriptor_layouts[i])vkDestroyDescriptorSetLayout(dev,descriptor_layouts[i],NULL);
   vkDestroyDevice(dev,NULL);
   if (messenger) {
      PFN_vkDestroyDebugUtilsMessengerEXT destroy=(PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance,"vkDestroyDebugUtilsMessengerEXT");
      destroy(instance,messenger,NULL);
   }
   vkDestroyInstance(instance,NULL);
   printf("VALIDATION_ERRORS=%u\n",validation_errors);
   return validation_errors?1:0;
}
