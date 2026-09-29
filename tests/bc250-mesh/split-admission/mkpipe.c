/* create one graphics pipeline from mesh.spv [+task.spv] + frag.spv; print VkResult. */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static VkDevice dev;
static VkShaderModule sm(const char *n){FILE*f=fopen(n,"rb");if(!f){perror(n);exit(2);}fseek(f,0,SEEK_END);long s=ftell(f);rewind(f);void*p=malloc(s);fread(p,1,s,f);fclose(f);
 VkShaderModuleCreateInfo ci={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,0,0,s,p};VkShaderModule m;if(vkCreateShaderModule(dev,&ci,0,&m))exit(2);return m;}
int main(int argc,char**argv){
 const char*mesh=argv[1],*frag=argv[2],*task=argc>3&&strcmp(argv[3],"-")?argv[3]:NULL; int topo=argc>4?atoi(argv[4]):3;
 VkApplicationInfo app={VK_STRUCTURE_TYPE_APPLICATION_INFO,.apiVersion=VK_API_VERSION_1_3};
 VkInstanceCreateInfo ici={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};VkInstance in;if(vkCreateInstance(&ici,0,&in))return 2;
 uint32_t n=1;VkPhysicalDevice pd;if(vkEnumeratePhysicalDevices(in,&n,&pd)<0||!n)return 2;
 VkPhysicalDeviceProperties pp;vkGetPhysicalDeviceProperties(pd,&pp); if(pp.deviceID!=0x13fe){fprintf(stderr,"not gfx1013 shim\n");return 2;}
 VkPhysicalDeviceMeshShaderFeaturesEXT mf={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,.meshShader=1,.taskShader=task!=NULL};
 VkPhysicalDeviceVulkan13Features f13={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,&mf,.dynamicRendering=1,.maintenance4=1};
 VkPhysicalDeviceFeatures feat={.multiViewport=1,.shaderClipDistance=1,.shaderCullDistance=1,.largePoints=1};
 float pr=1;VkDeviceQueueCreateInfo q={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=0,.queueCount=1,.pQueuePriorities=&pr};
 const char*ext="VK_EXT_mesh_shader";
 VkDeviceCreateInfo dci={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,&f13,0,1,&q,0,0,1,&ext,&feat};
 VkResult r=vkCreateDevice(pd,&dci,0,&dev);if(r){printf("DEVICE %d\n",r);return 2;}
 VkPipelineLayoutCreateInfo plci={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};VkPipelineLayout lay;vkCreatePipelineLayout(dev,&plci,0,&lay);
 VkPipelineShaderStageCreateInfo st[3]={{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_MESH_BIT_EXT,sm(mesh),"main"},
  {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_FRAGMENT_BIT,sm(frag),"main"}};
 if(task)st[2]=(VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_TASK_BIT_EXT,sm(task),"main"};
 VkViewport vp[2]={{0,0,64,64,0,1},{0,0,64,64,0,1}};VkRect2D sc[2]={{{0,0},{64,64}},{{0,0},{64,64}}};
 VkPipelineViewportStateCreateInfo vs={VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,0,0,2,vp,2,sc};
 VkPipelineRasterizationStateCreateInfo rs={VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,.polygonMode=VK_POLYGON_MODE_FILL,.lineWidth=1};
 VkPipelineMultisampleStateCreateInfo ms={VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,.rasterizationSamples=1};
 VkPipelineColorBlendAttachmentState ba={.colorWriteMask=15};VkPipelineColorBlendStateCreateInfo bs={VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,.attachmentCount=1,.pAttachments=&ba};
 VkFormat fmt=VK_FORMAT_R8G8B8A8_UNORM;VkPipelineRenderingCreateInfo ri={VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,.colorAttachmentCount=1,.pColorAttachmentFormats=&fmt};
 VkGraphicsPipelineCreateInfo gp={VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,&ri,0,task?3:2,st,0,0,0,&vs,&rs,&ms,0,&bs,0,lay};
 VkPipeline p;r=vkCreateGraphicsPipelines(dev,0,1,&gp,0,&p);printf("RESULT %d\n",r);return r?1:0;}
