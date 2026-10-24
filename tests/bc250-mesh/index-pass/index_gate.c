/* One parameterized Mesh or Task+Mesh draw, followed by RGBA8 readback.
 * Hardware execution is released only by the audited one-shot launcher.
 * Attachment clear/copy commands are additional to the application draw.
 * Optional cases (environment):
 *   INDEX_CLEAR_BETWEEN=1  draw, vkCmdClearAttachments over part of the image, draw again (meta operation between
 *                          two draws of the same pipeline; the second draw restores the cleared area)
 *   INDEX_QUERY=1          pipeline-statistics query active around the draw
 *   INDEX_DIRECT_AROUND=1  direct draws before and between indirect calls
 *   INDEX_REPEAT_INDIRECT=2 repeat each pass's indirect calls for reset/lifetime checks
 *   INDEX_MULTIVIEW=1      render pass view mask 0b11 (two layers; layer 0 is read back)
 *   INDEX_CYCLE=free|reset submit and wait every recording, then free its command buffer or reset it and leave
 *                          it idle (each recording uses a new command buffer)
 *   INDEX_BENCH_DRAWS=1..4096 enable GPU timestamps around repeated draws in one render pass
 *   INDEX_BENCH_SAMPLES=1..31 measured submissions (default 9), after 2 warmup submissions
 *   INDEX_WARMUP_MS=0..2000 optionally resubmit for this wall time before the normal benchmark loop
 *   INDEX_GROUPS=1|8|64 set groups per live indirect record or direct draw (default 64)
 *   INDEX_BENCH_EMPTY=1 measure only attachment clear/render-pass cost, without draws
 */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#define SIDE 256
#define BYTES (SIDE * SIDE * 4)
#define CK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { \
   fprintf(stderr, "FAIL %s: %d\n", #x, r); return 1; } } while (0)
static unsigned validation_errors;
static uint64_t monotonic_ns(void)
{
   struct timespec t;
   if (clock_gettime(CLOCK_MONOTONIC, &t)) { perror("clock_gettime"); exit(2); }
   return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_cb(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
   VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT *data, void *user)
{
   (void)type; (void)user;
   if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) validation_errors++;
   fprintf(stderr, "VALIDATION: %s\n", data->pMessage);
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
static VkShaderModule load(VkDevice dev, const char *name)
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
int main(int argc, char **argv)
{
   setvbuf(stdout, NULL, _IONBF, 0);
   if ((argc != 5 && argc != 6) || (strcmp(argv[1], "--offline") && strcmp(argv[1], "--authorized-gpu-INDEX"))) {
      fputs("usage: index_gate --offline|--authorized-gpu-INDEX MESH.spv FRAG.spv OUT.rgba [TASK.spv]\n", stderr); return 2;
   }
   const int task = argc == 6;
   const int reference = getenv("INDEX_REFERENCE") != NULL;
   const char *last_env = getenv("INDEX_LAST_VERTEX");
   if (last_env && strcmp(last_env,"0") && strcmp(last_env,"1")) return 2;
   const int last = last_env && !strcmp(last_env,"1");
   const int offline = !strcmp(argv[1], "--offline");
   const unsigned bench_draws = getenv("INDEX_BENCH_DRAWS") ? atoi(getenv("INDEX_BENCH_DRAWS")) : 0;
   const unsigned bench_samples = getenv("INDEX_BENCH_SAMPLES") ? atoi(getenv("INDEX_BENCH_SAMPLES")) : 9;
   const unsigned warmup_ms = getenv("INDEX_WARMUP_MS") ? atoi(getenv("INDEX_WARMUP_MS")) : 0;
   if (warmup_ms > 2000 || (warmup_ms && !bench_draws)) return 2;
   const unsigned vertex_steps = getenv("INDEX_VERTEX_STEPS") ? atoi(getenv("INDEX_VERTEX_STEPS")) : 0;
   const unsigned groups = getenv("INDEX_GROUPS") ? atoi(getenv("INDEX_GROUPS")) : 64;
   if (groups != 1 && groups != 8 && groups != 64) return 2;
   const int bench_empty = getenv("INDEX_BENCH_EMPTY") != NULL;
   if (bench_empty && !bench_draws) return 2;
   if (vertex_steps > 1024) return 2;
   const unsigned oracle_corner = getenv("INDEX_ORACLE_CORNER") ? atoi(getenv("INDEX_ORACLE_CORNER")) : 0;
   if (oracle_corner > 2) return 2;
   if ((getenv("INDEX_BENCH_DRAWS") && (!bench_draws || bench_draws > 4096)) ||
       !bench_samples || bench_samples > 31) return 2;
   const int clear_between = getenv("INDEX_CLEAR_BETWEEN") != NULL;
   const int direct_around = getenv("INDEX_DIRECT_AROUND") != NULL;
   const unsigned repeat_indirect = getenv("INDEX_REPEAT_INDIRECT") ? atoi(getenv("INDEX_REPEAT_INDIRECT")) : 1;
   if (!repeat_indirect || repeat_indirect > 32) return 2;
   const int query = getenv("INDEX_QUERY") != NULL;
   const int multiview = getenv("INDEX_MULTIVIEW") != NULL;
   const char *cycle_env = getenv("INDEX_CYCLE");
   const int cycle = !cycle_env ? 0 : !strcmp(cycle_env, "free") ? 1 : !strcmp(cycle_env, "reset") ? 2 : -1;
   const int discard_recording = getenv("INDEX_DISCARD_RECORDING") != NULL;
   if (cycle < 0) { fputs("INDEX_CYCLE must be free or reset\n", stderr); return 2; }
   if (bench_draws && (task || reference || clear_between || query || multiview || cycle ||
                       getenv("INDEX_RECORDINGS") || getenv("INDEX_GRID") || discard_recording)) return 2;
   if (discard_recording && (cycle || task || reference || query || multiview || clear_between ||
                            getenv("INDEX_RECORDINGS"))) return 2;
   const char *gpu=getenv("AMDGPU_GPU_ID"), *preload=getenv("LD_PRELOAD"), *safe=getenv("RADV_BC250_MESH_SAFE_FAST");
   if (!offline && (!safe || strcmp(safe, "1"))) { fputs("REFUSE: SAFE_FAST must be enabled\n", stderr); return 3; }
   if (offline) {
      if (access("/dev/dri", F_OK)==0 || !gpu || (strcmp(gpu,"gfx1013") && strcmp(gpu,"navi10") && strcmp(gpu,"navi12") && strcmp(gpu,"navi14")) || !preload ||
          !strstr(preload,"libamdgpu_noop_drm_shim.so")) {
         fputs("REFUSE: offline mode requires bwrap/noop with no /dev/dri\n", stderr); return 3;
      }
   } else if (access("/dev/dri", F_OK)!=0 || gpu || preload ||
              !getenv("BC250_INDEX_OWNER_AUTHORIZED") || strcmp(getenv("BC250_INDEX_OWNER_AUTHORIZED"),"INDEX-once")) {
      fputs("REFUSE: hardware mode requires the authorized one-shot launcher\n", stderr); return 3;
   }
   VkDebugUtilsMessengerCreateInfoEXT dbg = {.sType=VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
      .messageSeverity=VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
      .messageType=VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT|
                   VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT, .pfnUserCallback=debug_cb};
   VkValidationFeatureEnableEXT sync_validation=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
   VkValidationFeaturesEXT validation={.sType=VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,.pNext=&dbg,
      .enabledValidationFeatureCount=1,.pEnabledValidationFeatures=&sync_validation};
   const char *layer="VK_LAYER_KHRONOS_validation";
   const char *debug_exts[]={"VK_EXT_debug_utils","VK_EXT_validation_features"};
   VkApplicationInfo app = {.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName="BC250 parity single draw",
      .apiVersion=VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo=&app,
      .pNext=offline?&validation:NULL, .enabledLayerCount=offline?1:0, .ppEnabledLayerNames=offline?&layer:NULL,
      .enabledExtensionCount=offline?2:0, .ppEnabledExtensionNames=offline?debug_exts:NULL};
   VkInstance instance; CK(vkCreateInstance(&ici,NULL,&instance));
   VkDebugUtilsMessengerEXT messenger=VK_NULL_HANDLE;
   if (offline) {
      PFN_vkCreateDebugUtilsMessengerEXT create=(PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance,"vkCreateDebugUtilsMessengerEXT");
      if (!create) return 2;
      CK(create(instance,&dbg,NULL,&messenger));
   }
   uint32_t np=0; CK(vkEnumeratePhysicalDevices(instance,&np,NULL));
   if (np != 1) { fprintf(stderr,"REFUSE: expected one physical device, got %u\n",np); return 3; }
   VkPhysicalDevice pd; CK(vkEnumeratePhysicalDevices(instance,&np,&pd));
   VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd,&props);
   if (props.vendorID != 0x1002 || (!offline && !strstr(props.deviceName,"GFX1013") && !strstr(props.deviceName,"gfx1013"))) {
      fputs("REFUSE: not GFX1013\n",stderr); return 3;
   }
   VkPhysicalDeviceMeshShaderFeaturesEXT mesh_support={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
   VkPhysicalDeviceVulkan13Features support13={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,.pNext=&mesh_support};
   VkPhysicalDeviceFeatures2 features={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,.pNext=&support13};
   vkGetPhysicalDeviceFeatures2(pd,&features);
   if (!features.features.geometryShader) { fputs("REFUSE: geometryShader (gl_PrimitiveID) unavailable\n",stderr); return 3; }
   if (!mesh_support.meshShader || !support13.maintenance4) { fputs("REFUSE: Mesh unavailable\n",stderr); return 3; }
   if (query && !features.features.pipelineStatisticsQuery) { fputs("REFUSE: pipeline statistics unavailable\n",stderr); return 3; }
   if (multiview && (reference ? 0 : !mesh_support.multiviewMeshShader)) { fputs("REFUSE: multiview Mesh unavailable\n",stderr); return 3; }
   uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,NULL);
   VkQueueFamilyProperties *qp=calloc(nq,sizeof(*qp)); if (!qp) return 2;
   vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,qp);
   uint32_t family=0; while (family<nq && !(qp[family].queueFlags&VK_QUEUE_GRAPHICS_BIT)) family++;
   unsigned timestamp_bits=family<nq?qp[family].timestampValidBits:0;
   free(qp); if (family==nq) return 2;
   if (bench_draws && (!timestamp_bits || timestamp_bits > 64 || props.limits.timestampPeriod <= 0)) return 2;
   float priority=1;
   VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=family,
      .queueCount=1,.pQueuePriorities=&priority};
   VkPhysicalDeviceProvokingVertexFeaturesEXT provoking_enable={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT,.provokingVertexLast=VK_TRUE};
   VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR bary_enable={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR,.pNext=&provoking_enable,.fragmentShaderBarycentric=VK_TRUE};
   VkPhysicalDeviceMeshShaderFeaturesEXT mesh_enable={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,.pNext=&bary_enable,.meshShader=VK_TRUE,.taskShader=task,
      .multiviewMeshShader=multiview&&mesh_support.multiviewMeshShader};
   VkPhysicalDeviceVulkan11Features enable11={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,.pNext=&mesh_enable,.multiview=multiview,.shaderDrawParameters=VK_TRUE};
   VkPhysicalDeviceVulkan12Features enable12={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext=&enable11,.drawIndirectCount=VK_TRUE};
   VkPhysicalDeviceVulkan13Features enable13={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .pNext=&enable12,.maintenance4=VK_TRUE};
   const char *mesh_exts[3]={"VK_EXT_mesh_shader","VK_KHR_fragment_shader_barycentric","VK_EXT_provoking_vertex"};
   VkPhysicalDeviceFeatures base_enable={.geometryShader=VK_TRUE, /* SPIR-V Geometry capability for gl_PrimitiveID */
      .shaderCullDistance=features.features.shaderCullDistance,
      .pipelineStatisticsQuery=query,.multiDrawIndirect=VK_TRUE};
   VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=&enable13,.pEnabledFeatures=&base_enable,
      .queueCreateInfoCount=1,.pQueueCreateInfos=&qci,.enabledExtensionCount=3,.ppEnabledExtensionNames=mesh_exts};
   VkDevice dev; CK(vkCreateDevice(pd,&dci,NULL,&dev));
   VkQueue queue; vkGetDeviceQueue(dev,family,0,&queue);
   VkPushConstantRange range={.stageFlags=VK_SHADER_STAGE_MESH_BIT_EXT|VK_SHADER_STAGE_VERTEX_BIT,.size=16};
   VkPipelineLayoutCreateInfo lci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.pushConstantRangeCount=1,.pPushConstantRanges=&range};
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
   const uint32_t view_mask=3;
   VkRenderPassMultiviewCreateInfo multiview_ci={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO,
      .subpassCount=1,.pViewMasks=&view_mask};
   VkRenderPassCreateInfo rpci={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,.pNext=multiview?&multiview_ci:NULL,
      .attachmentCount=1,.pAttachments=&attachment,
      .subpassCount=1,.pSubpasses=&subpass,.dependencyCount=2,.pDependencies=dependencies};
   VkRenderPass rp; CK(vkCreateRenderPass(dev,&rpci,NULL,&rp));
   VkShaderModule modules[3]={load(dev,argv[2]),load(dev,argv[3]),task?load(dev,argv[5]):VK_NULL_HANDLE};
   VkPipelineShaderStageCreateInfo stages[3]={
      {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=reference?VK_SHADER_STAGE_VERTEX_BIT:VK_SHADER_STAGE_MESH_BIT_EXT,.module=modules[0],.pName="main"},
      {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_FRAGMENT_BIT,.module=modules[1],.pName="main"},
      {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_TASK_BIT_EXT,.module=modules[2],.pName="main"}};
   VkViewport viewport={0,0,SIDE,SIDE,0,1}; VkRect2D rect={{0,0},{SIDE,SIDE}};
   VkPipelineViewportStateCreateInfo vs={.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount=1,.pViewports=&viewport,.scissorCount=1,.pScissors=&rect};
   VkPipelineRasterizationProvokingVertexStateCreateInfoEXT provoking={.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,.provokingVertexMode=last?VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT:VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT};
   VkPipelineVertexInputStateCreateInfo vertex={.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo assembly={.sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkPipelineRasterizationStateCreateInfo rs={.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .pNext=&provoking,.polygonMode=VK_POLYGON_MODE_FILL,.cullMode=VK_CULL_MODE_NONE,.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,.lineWidth=1};
   VkPipelineMultisampleStateCreateInfo ms={.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT};
   VkPipelineColorBlendAttachmentState blend={.colorWriteMask=15};
   VkPipelineColorBlendStateCreateInfo bs={.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,.attachmentCount=1,.pAttachments=&blend};
   VkGraphicsPipelineCreateInfo pci={.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,.stageCount=task?3:2,.pStages=stages,
      .pVertexInputState=reference?&vertex:NULL,.pInputAssemblyState=reference?&assembly:NULL,.pViewportState=&vs,.pRasterizationState=&rs,.pMultisampleState=&ms,.pColorBlendState=&bs,.layout=layout,.renderPass=rp};
   VkPipeline pipeline; CK(vkCreateGraphicsPipelines(dev,VK_NULL_HANDLE,1,&pci,NULL,&pipeline));
   VkImageCreateInfo image_ci={.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,.imageType=VK_IMAGE_TYPE_2D,
      .format=VK_FORMAT_R8G8B8A8_UNORM,.extent={SIDE,SIDE,1},.mipLevels=1,.arrayLayers=multiview?2:1,.samples=VK_SAMPLE_COUNT_1_BIT,
      .tiling=VK_IMAGE_TILING_OPTIMAL,.usage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
   VkImage image; CK(vkCreateImage(dev,&image_ci,NULL,&image));
   VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev,image,&mr);
   VkMemoryAllocateInfo mai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=mr.size,.memoryTypeIndex=memtype(pd,mr.memoryTypeBits,0)};
   VkDeviceMemory image_mem; CK(vkAllocateMemory(dev,&mai,NULL,&image_mem)); CK(vkBindImageMemory(dev,image,image_mem,0));
   VkImageViewCreateInfo ivci={.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,.image=image,
      .viewType=multiview?VK_IMAGE_VIEW_TYPE_2D_ARRAY:VK_IMAGE_VIEW_TYPE_2D,
      .format=VK_FORMAT_R8G8B8A8_UNORM,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,multiview?2:1}};
   VkImageView view; CK(vkCreateImageView(dev,&ivci,NULL,&view));
   VkFramebufferCreateInfo fbci={.sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,.renderPass=rp,.attachmentCount=1,
      .pAttachments=&view,.width=SIDE,.height=SIDE,.layers=1};
   VkFramebuffer fb; CK(vkCreateFramebuffer(dev,&fbci,NULL,&fb));
   VkBufferCreateInfo bci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=BYTES,.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT};
   VkBuffer buffer; CK(vkCreateBuffer(dev,&bci,NULL,&buffer)); vkGetBufferMemoryRequirements(dev,buffer,&mr);
   mai.allocationSize=mr.size; mai.memoryTypeIndex=memtype(pd,mr.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   VkDeviceMemory buffer_mem; CK(vkAllocateMemory(dev,&mai,NULL,&buffer_mem)); CK(vkBindBufferMemory(dev,buffer,buffer_mem,0));
   void *mapped; CK(vkMapMemory(dev,buffer_mem,0,VK_WHOLE_SIZE,0,&mapped)); memset(mapped,0xa5,BYTES);
   const char *indirect=getenv("INDEX_INDIRECT");
   VkBuffer records=VK_NULL_HANDLE;
   VkDeviceMemory records_mem=VK_NULL_HANDLE;
   unsigned grid[3]={groups,1,1};
   if (getenv("INDEX_GRID") && sscanf(getenv("INDEX_GRID"),"%u,%u,%u",&grid[0],&grid[1],&grid[2])!=3) return 2;
   if (indirect) {
      /* Three counted records followed by a fourth beyond the GPU count. Zero
       * middle record exercises repeated prefixes; the final record has a 2D grid. */
      VkBufferCreateInfo ri={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=80,.usage=VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT};
      CK(vkCreateBuffer(dev,&ri,NULL,&records)); vkGetBufferMemoryRequirements(dev,records,&mr);
      mai.allocationSize=mr.size; mai.memoryTypeIndex=memtype(pd,mr.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      CK(vkAllocateMemory(dev,&mai,NULL,&records_mem)); CK(vkBindBufferMemory(dev,records,records_mem,0));
      uint32_t *record_map; CK(vkMapMemory(dev,records_mem,0,VK_WHOLE_SIZE,0,(void**)&record_map));
      const unsigned final_x=groups<16?groups:16;
      const uint32_t data[16]={groups,1,1,0,0,0,final_x,groups/final_x,1,1,1,1,3,0,0,0};
      memset(record_map,0,80);
      if (reference) {
         unsigned vertices=groups*3*atoi(getenv("INDEX_PRIMITIVES"));
         const uint32_t ref_data[20]={vertices,1,0,0,0,1,0,0,vertices,1,0,0,vertices/groups,1,0,0,3,0,0,0};
         memcpy(record_map,ref_data,sizeof(ref_data));
      } else memcpy(record_map,data,sizeof(data));
      vkUnmapMemory(dev,records_mem);
   }
   VkCommandPoolCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=family,
      .flags=cycle==2||discard_recording?VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT:0};
   VkQueryPool queries=VK_NULL_HANDLE;
   VkQueryPool timestamps=VK_NULL_HANDLE;
   if (bench_draws) {
      VkQueryPoolCreateInfo qi={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
         .queryType=VK_QUERY_TYPE_TIMESTAMP,.queryCount=2};
      CK(vkCreateQueryPool(dev,&qi,NULL,&timestamps));
   }
   if (query) {
      VkQueryPoolCreateInfo qpci={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,.queryType=VK_QUERY_TYPE_PIPELINE_STATISTICS,
         .queryCount=1,.pipelineStatistics=VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT};
      CK(vkCreateQueryPool(dev,&qpci,NULL,&queries));
   }
   VkFenceCreateInfo fci={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; CK(vkCreateFence(dev,&fci,NULL,&fence));
   int released=0;
   VkCommandPool pool; CK(vkCreateCommandPool(dev,&cpci,NULL,&pool));
   VkCommandBufferAllocateInfo cbai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=pool,
      .level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
   unsigned recordings=getenv("INDEX_RECORDINGS")?atoi(getenv("INDEX_RECORDINGS")):discard_recording?2:1;
   if (!recordings || recordings>192) return 2;
   VkCommandBuffer cb=VK_NULL_HANDLE;
   for (unsigned recording=0; recording<recordings; recording++) {
   if (!discard_recording || !recording) CK(vkAllocateCommandBuffers(dev,&cbai,&cb));
   struct timespec record_start, record_end;
   if (bench_draws && clock_gettime(CLOCK_MONOTONIC, &record_start)) return 2;
   VkCommandBufferBeginInfo cbi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags=bench_draws?0:VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
   CK(vkBeginCommandBuffer(cb,&cbi));
   if (bench_draws) {
      vkCmdResetQueryPool(cb,timestamps,0,2);
      vkCmdWriteTimestamp(cb,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,timestamps,0);
   }
   if (query) vkCmdResetQueryPool(cb,queries,0,1);
   VkClearValue clear={.color={{0,0,0,0}}};
   VkRenderPassBeginInfo rbi={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,.renderPass=rp,.framebuffer=fb,
      .renderArea=rect,.clearValueCount=1,.pClearValues=&clear};
   vkCmdBeginRenderPass(cb,&rbi,VK_SUBPASS_CONTENTS_INLINE); vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline);
   const uint32_t constants[4]={7,indirect?1:0,vertex_steps,oracle_corner}; vkCmdPushConstants(cb,layout,VK_SHADER_STAGE_MESH_BIT_EXT|VK_SHADER_STAGE_VERTEX_BIT,0,16,constants);
   PFN_vkCmdDrawMeshTasksEXT draw=(PFN_vkCmdDrawMeshTasksEXT)vkGetDeviceProcAddr(dev,"vkCmdDrawMeshTasksEXT");
   if (!draw) return 2;
   if (query) vkCmdBeginQuery(cb,queries,0,0);
   if (direct_around) {
      if (reference) vkCmdDraw(cb,groups*3*atoi(getenv("INDEX_PRIMITIVES")),1,0,0);
      else draw(cb,grid[0],grid[1],grid[2]);
   }
   for (unsigned pass=0; pass<(bench_empty?0u:bench_draws?bench_draws:(clear_between?2u:1u)); pass++) {
      if (getenv("INDEX_ALTERNATE_STEPS")) {
         uint32_t varying[4]={7,indirect?1:0,(pass&1)?vertex_steps:0,oracle_corner};
         vkCmdPushConstants(cb,layout,VK_SHADER_STAGE_MESH_BIT_EXT|VK_SHADER_STAGE_VERTEX_BIT,0,16,varying);
      }
      if (pass && clear_between) {
         VkClearAttachment ca={.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,.colorAttachment=0,.clearValue={.color={{0,0,0,0}}}};
         VkClearRect cr={.rect={{0,0},{SIDE/2,SIDE/2}},.baseArrayLayer=0,.layerCount=1};
         vkCmdClearAttachments(cb,1,&ca,1,&cr);
      }
      for (unsigned repeat=0; repeat<repeat_indirect; repeat++) {
      if (reference) {
         if (indirect && !strcmp(indirect,"single")) vkCmdDrawIndirect(cb,records,0,1,16);
         else if (indirect && !strcmp(indirect,"count")) vkCmdDrawIndirectCount(cb,records,0,records,64,4,16);
         else if (indirect) vkCmdDrawIndirect(cb,records,0,3,16);
         else vkCmdDraw(cb,groups*3*atoi(getenv("INDEX_PRIMITIVES")),1,0,0);
      }
      else if (indirect) {
         if (!strcmp(indirect,"single"))
            ((PFN_vkCmdDrawMeshTasksIndirectEXT)vkGetDeviceProcAddr(dev,"vkCmdDrawMeshTasksIndirectEXT"))(cb,records,0,1,12);
         else if (!strcmp(indirect,"count"))
            ((PFN_vkCmdDrawMeshTasksIndirectCountEXT)vkGetDeviceProcAddr(dev,"vkCmdDrawMeshTasksIndirectCountEXT"))(cb,records,0,records,48,4,12);
         else if (!strcmp(indirect,"multi"))
            ((PFN_vkCmdDrawMeshTasksIndirectEXT)vkGetDeviceProcAddr(dev,"vkCmdDrawMeshTasksIndirectEXT"))(cb,records,0,3,12);
         else return 2;
      } else draw(cb,grid[0],grid[1],grid[2]);
      }
      if (direct_around) {
         if (reference) vkCmdDraw(cb,groups*3*atoi(getenv("INDEX_PRIMITIVES")),1,0,0);
         else draw(cb,grid[0],grid[1],grid[2]);
      }
   }
   if (discard_recording && !recording) {
      /* Drop a recording while its IDX side batch is still open, then record the same command buffer again. */
      CK(vkResetCommandBuffer(cb,0));
      puts("INDEX_DISCARDED_OPEN_RECORDING");
      continue;
   }
   if (query) vkCmdEndQuery(cb,queries,0);
   vkCmdEndRenderPass(cb);
   if (bench_draws) vkCmdWriteTimestamp(cb,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,timestamps,1);
   VkBufferImageCopy copy={.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.imageExtent={SIDE,SIDE,1}};
   vkCmdCopyImageToBuffer(cb,image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,buffer,1,&copy);
   VkBufferMemoryBarrier host={.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask=VK_ACCESS_HOST_READ_BIT,.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
      .buffer=buffer,.size=VK_WHOLE_SIZE};
   vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,NULL,1,&host,0,NULL);
   CK(vkEndCommandBuffer(cb));
   if (bench_draws) {
      if (clock_gettime(CLOCK_MONOTONIC, &record_end)) return 2;
      double us=(record_end.tv_sec-record_start.tv_sec)*1000000.0+(record_end.tv_nsec-record_start.tv_nsec)/1000.0;
      printf("INDEX_RECORD_CPU_US recording=%u us=%.6f\n",recording,us);
   }
   if (!cycle && recording+1<recordings) continue;
   if (validation_errors) { fputs("REFUSE: validation errors before submit\n",stderr); return 1; }
   VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb};
   if (!released) {
      if (bench_draws) printf("INDEX_BENCH_CONFIG draws=%u samples=%u warmup=2 provoking=%s timestamp_bits=%u timestamp_period_ns=%.9g\n",
         bench_draws,bench_samples,last?"last":"first",timestamp_bits,props.limits.timestampPeriod);
      printf("INDEX_WORKLOAD groups=%u live_records=%u empty=%u\n",groups,indirect&&strcmp(indirect,"single")?2u:1u,bench_empty);
      printf("INDEX_READY mode=%s mesh_draws=%d workgroups=%u,%u,%u submits=%u\n",offline?"noop":"hardware",
             bench_empty?0:bench_draws?(int)bench_draws:(clear_between?2:1),grid[0],grid[1],grid[2],bench_draws?bench_samples+2:(cycle?recordings:1));
      /* The one-shot launcher verifies the actual compilation before releasing
       * the submissions. EOF, a bad token, or launcher failure stops here. */
      if (!offline) {
         char token=0;
         if (read(STDIN_FILENO,&token,1)!=1 || token!='S') {
            fputs("REFUSE: submission not released by audited launcher\n",stderr); return 3;
         }
      }
      released=1;
   }
   if (warmup_ms && !offline) {
      uint64_t begin=monotonic_ns(), end;
      unsigned warmup_submissions=0;
      printf("INDEX_WARMUP_BEGIN requested_ms=%u monotonic_ns=%llu\n",warmup_ms,(unsigned long long)begin);
      do {
         CK(vkQueueSubmit(queue,1,&submit,fence));
         VkResult wait=vkWaitForFences(dev,1,&fence,VK_TRUE,30000000000ull);
         if (wait!=VK_SUCCESS) { fprintf(stderr,"INDEX_STOP warmup fence=%d; preserve logs\n",wait); fflush(NULL); _Exit(124); }
         CK(vkResetFences(dev,1,&fence));
         warmup_submissions++;
         end=monotonic_ns();
      } while (end-begin < (uint64_t)warmup_ms*1000000ull);
      printf("INDEX_WARMUP_COMPLETE submissions=%u wall_ms=%.6f monotonic_ns=%llu\n",
             warmup_submissions,(double)(end-begin)/1000000.0,(unsigned long long)end);
      printf("INDEX_MEASURE_BEGIN monotonic_ns=%llu\n",(unsigned long long)monotonic_ns());
   } else if (warmup_ms) {
      printf("INDEX_WARMUP_SKIPPED mode=noop requested_ms=%u\n",warmup_ms);
   }
   for (unsigned sample=0;sample<(bench_draws?bench_samples+2:1);sample++) {
      printf("INDEX_SUBMIT sample=%u\n",sample);
      CK(vkQueueSubmit(queue,1,&submit,fence));
      VkResult wait=vkWaitForFences(dev,1,&fence,VK_TRUE,30000000000ull);
      if (wait!=VK_SUCCESS) { fprintf(stderr,"INDEX_STOP fence=%d; retain .last; do not retry\n",wait); fflush(NULL); _Exit(124); }
      CK(vkResetFences(dev,1,&fence));
      if (bench_draws && !offline) {
         uint64_t ticks[2];
         CK(vkGetQueryPoolResults(dev,timestamps,0,2,sizeof(ticks),ticks,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT));
         uint64_t mask=timestamp_bits==64?UINT64_MAX:((1ull<<timestamp_bits)-1);
         uint64_t elapsed=(ticks[1]-ticks[0])&mask;
         if (!elapsed) { fputs("INDEX_STOP invalid zero timestamp\n",stderr); return 1; }
         double us=(double)elapsed*props.limits.timestampPeriod/1000.0;
         printf("INDEX_BENCH_SAMPLE sample=%u warmup=%u gpu_us=%.6f draws=%u\n",sample,sample<2,us,bench_draws);
      }
   }
   if (warmup_ms && !offline)
      printf("INDEX_MEASURE_COMPLETE monotonic_ns=%llu\n",(unsigned long long)monotonic_ns());
   if (cycle==1) vkFreeCommandBuffers(dev,pool,1,&cb);
   else if (cycle==2) CK(vkResetCommandBuffer(cb,0)); /* kept allocated but idle */
   }
   if (query && !offline) { /* the noop device never writes query results */
      uint64_t stats=0;
      CK(vkGetQueryPoolResults(dev,queries,0,1,sizeof(stats),&stats,sizeof(stats),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT));
      printf("INDEX_QUERY fragment_invocations=%llu\n",(unsigned long long)stats);
   }
   if (queries) vkDestroyQueryPool(dev,queries,NULL);
   if (timestamps) vkDestroyQueryPool(dev,timestamps,NULL);
   if (offline) {
      const unsigned char *p=mapped;
      for (unsigned i=0;i<BYTES;i++) if (p[i]!=0xa5) { fputs("NOOP_SENTINEL_CHANGED\n",stderr); return 1; }
      puts("NOOP_COMPLETE pixels_not_rendered");
   } else {
      FILE *out=fopen(argv[4],"wbx"); if (!out) { perror(argv[4]); return 2; }
      if (fwrite(mapped,1,BYTES,out)!=BYTES || fflush(out) || fsync(fileno(out)) || fclose(out)) return 2;
      puts("INDEX_FENCE_COMPLETE readback_saved");
   }
   vkDestroyFence(dev,fence,NULL); vkDestroyCommandPool(dev,pool,NULL);
   if (records) { vkDestroyBuffer(dev,records,NULL); vkFreeMemory(dev,records_mem,NULL); }
   vkUnmapMemory(dev,buffer_mem); vkDestroyBuffer(dev,buffer,NULL); vkFreeMemory(dev,buffer_mem,NULL);
   vkDestroyFramebuffer(dev,fb,NULL); vkDestroyImageView(dev,view,NULL); vkDestroyImage(dev,image,NULL); vkFreeMemory(dev,image_mem,NULL);
   vkDestroyPipeline(dev,pipeline,NULL); for (unsigned i=0;i<(task?3:2);i++) vkDestroyShaderModule(dev,modules[i],NULL);
   vkDestroyRenderPass(dev,rp,NULL); vkDestroyPipelineLayout(dev,layout,NULL); vkDestroyDevice(dev,NULL);
   if (messenger) {
      PFN_vkDestroyDebugUtilsMessengerEXT destroy=(PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance,"vkDestroyDebugUtilsMessengerEXT");
      destroy(instance,messenger,NULL);
   }
   vkDestroyInstance(instance,NULL);
   printf("VALIDATION_ERRORS=%u\n",validation_errors);
   return validation_errors?1:0;
}
