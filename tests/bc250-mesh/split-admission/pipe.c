/* G1 shim-only pipeline admission probe: pipe MESH.spv FRAG.spv [tri|line|point]
 * Creates one Mesh+FS graphics pipeline, records (never submits) one direct
 * and one indirect Mesh draw, prints VkResults. No queue submission exists. */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static VkDevice dev;
static VkShaderModule load(const char *p){FILE*f=fopen(p,"rb");if(!f){perror(p);exit(2);}fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);
 void*b=malloc(n);if(fread(b,1,n,f)!=(size_t)n)exit(2);fclose(f);VkShaderModuleCreateInfo c={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,0,0,n,b};VkShaderModule m;
 if(vkCreateShaderModule(dev,&c,0,&m))exit(2);return m;}
int main(int argc,char**argv){
 VkApplicationInfo app={VK_STRUCTURE_TYPE_APPLICATION_INFO,.apiVersion=VK_API_VERSION_1_3};
 VkInstanceCreateInfo ici={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};VkInstance in;
 if(vkCreateInstance(&ici,0,&in)){puts("NO_INSTANCE");return 3;}
 uint32_t n=1;VkPhysicalDevice pd;if(vkEnumeratePhysicalDevices(in,&n,&pd)<0||!n){puts("NO_DEVICE");return 3;}
 VkPhysicalDeviceProperties pr;vkGetPhysicalDeviceProperties(pd,&pr);
 VkPhysicalDeviceMeshShaderPropertiesEXT mp={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
 VkPhysicalDeviceProperties2 p2={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,&mp};vkGetPhysicalDeviceProperties2(pd,&p2);
 VkPhysicalDeviceMeshShaderFeaturesEXT mf={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
 VkPhysicalDeviceFeatures2 f2={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,&mf};vkGetPhysicalDeviceFeatures2(pd,&f2);
 printf("DEVICE=%s DRIVER=%u.%u.%u meshShader=%u maxOutV=%u maxOutP=%u maxInv=%u maxShared=%u\n",pr.deviceName,
  VK_VERSION_MAJOR(pr.driverVersion),VK_VERSION_MINOR(pr.driverVersion),VK_VERSION_PATCH(pr.driverVersion),mf.meshShader,
  mp.maxMeshOutputVertices,mp.maxMeshOutputPrimitives,mp.maxMeshWorkGroupInvocations,mp.maxMeshSharedMemorySize);
 if(!strstr(pr.deviceName,"GFX1013")){puts("NOT_SHIM_GFX1013");return 3;}
 if(!mf.meshShader){puts("NO_MESH");return 3;}
 float q=1;VkDeviceQueueCreateInfo qc={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueCount=1,.pQueuePriorities=&q};
 VkPhysicalDeviceMeshShaderFeaturesEXT en={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,.meshShader=1};
 VkPhysicalDeviceVulkan11Features f11={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,&en,.shaderDrawParameters=1};
 VkPhysicalDeviceFeatures fe={.vertexPipelineStoresAndAtomics=1,.fragmentStoresAndAtomics=1,.multiDrawIndirect=1};
 const char*ext="VK_EXT_mesh_shader";
 VkDeviceCreateInfo dc={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,&f11,0,1,&qc,0,0,1,&ext,&fe};
 VkResult r=vkCreateDevice(pd,&dc,0,&dev);if(r){printf("DEVICE_FAIL %d\n",r);return 3;}
 VkDescriptorSetLayoutBinding b[2]={{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_ALL},{1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_ALL}};
 VkDescriptorSetLayoutCreateInfo dl={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,0,0,2,b};VkDescriptorSetLayout sl;vkCreateDescriptorSetLayout(dev,&dl,0,&sl);
 VkPushConstantRange pc={VK_SHADER_STAGE_ALL,0,128};
 VkPipelineLayoutCreateInfo lc={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,0,0,1,&sl,1,&pc};VkPipelineLayout lay;vkCreatePipelineLayout(dev,&lc,0,&lay);
 VkAttachmentDescription at={0,VK_FORMAT_R8G8B8A8_UNORM,1,VK_ATTACHMENT_LOAD_OP_CLEAR,VK_ATTACHMENT_STORE_OP_STORE,2,2,0,VK_IMAGE_LAYOUT_GENERAL};
 VkAttachmentReference ar={0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};VkSubpassDescription sp={0,0,0,0,1,&ar};
 VkRenderPassCreateInfo rc={VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,0,0,1,&at,1,&sp};VkRenderPass rp;vkCreateRenderPass(dev,&rc,0,&rp);
 VkPipelineShaderStageCreateInfo st[2]={{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_MESH_BIT_EXT,load(argv[1]),"main"},
  {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_FRAGMENT_BIT,load(argv[2]),"main"}};
 VkViewport vp={0,0,64,64,0,1};VkRect2D sc={{0,0},{64,64}};
 VkPipelineViewportStateCreateInfo vs={VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,0,0,1,&vp,1,&sc};
 VkPipelineRasterizationStateCreateInfo rs={VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,.lineWidth=1};
 VkPipelineMultisampleStateCreateInfo ms={VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,.rasterizationSamples=1};
 VkPipelineColorBlendAttachmentState ba={.colorWriteMask=15};VkPipelineColorBlendStateCreateInfo bs={VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,.attachmentCount=1,.pAttachments=&ba};
 VkGraphicsPipelineCreateInfo gp={VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,.stageCount=2,.pStages=st,.pViewportState=&vs,.pRasterizationState=&rs,.pMultisampleState=&ms,.pColorBlendState=&bs,.layout=lay,.renderPass=rp};
 VkPipeline p;r=vkCreateGraphicsPipelines(dev,0,1,&gp,0,&p);printf("PIPELINE_RESULT=%d\n",r);fflush(stdout);if(r)return 1;
 VkCommandPoolCreateInfo cp={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};VkCommandPool pool;vkCreateCommandPool(dev,&cp,0,&pool);
 VkCommandBufferAllocateInfo ca={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,0,pool,0,1};VkCommandBuffer cb;vkAllocateCommandBuffers(dev,&ca,&cb);
 VkCommandBufferBeginInfo bi={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};vkBeginCommandBuffer(cb,&bi);
 VkImageCreateInfo im={VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,0,0,VK_IMAGE_TYPE_2D,VK_FORMAT_R8G8B8A8_UNORM,{64,64,1},1,1,1,0,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};VkImage img;vkCreateImage(dev,&im,0,&img);
 VkMemoryRequirements mr;vkGetImageMemoryRequirements(dev,img,&mr);VkMemoryAllocateInfo ma={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,0,mr.size,__builtin_ctz(mr.memoryTypeBits)};VkDeviceMemory mem;vkAllocateMemory(dev,&ma,0,&mem);vkBindImageMemory(dev,img,mem,0);
 VkImageViewCreateInfo iv={VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,0,0,img,VK_IMAGE_VIEW_TYPE_2D,VK_FORMAT_R8G8B8A8_UNORM,{0},{1,0,1,0,1}};VkImageView view;vkCreateImageView(dev,&iv,0,&view);
 VkFramebufferCreateInfo fc={VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,0,0,rp,1,&view,64,64,1};VkFramebuffer fb;vkCreateFramebuffer(dev,&fc,0,&fb);
 VkClearValue cv={0};VkRenderPassBeginInfo rb={VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,0,rp,fb,sc,1,&cv};vkCmdBeginRenderPass(cb,&rb,0);
 vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_GRAPHICS,p);
 PFN_vkCmdDrawMeshTasksEXT d=(void*)vkGetDeviceProcAddr(dev,"vkCmdDrawMeshTasksEXT");d(cb,2,1,1);
 vkCmdEndRenderPass(cb);r=vkEndCommandBuffer(cb);printf("RECORD_DIRECT_RESULT=%d (no submission)\n",r);
 return r?1:0;}
