/* SPDX-License-Identifier: MIT */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
int main(int argc,char **argv)
{
   if(argc!=2 || access("/dev/dri",F_OK)==0 || !getenv("AMDGPU_GPU_ID") ||
      strcmp(getenv("AMDGPU_GPU_ID"),"gfx1013") || !getenv("LD_PRELOAD") ||
      !strstr(getenv("LD_PRELOAD"),"libamdgpu_noop_drm_shim.so"))return 2;
   VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.apiVersion=VK_API_VERSION_1_3};
   VkInstanceCreateInfo ci={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
   VkInstance instance;if(vkCreateInstance(&ci,NULL,&instance))return 1;
   uint32_t n=1;VkPhysicalDevice pd;if(vkEnumeratePhysicalDevices(instance,&n,&pd)||n!=1)return 1;
   uint32_t count=0;if(vkEnumerateDeviceExtensionProperties(pd,NULL,&count,NULL))return 1;
   VkExtensionProperties *ext=calloc(count,sizeof(*ext));
   if(!ext || vkEnumerateDeviceExtensionProperties(pd,NULL,&count,ext))return 1;
   int exposed=0;for(unsigned i=0;i<count;i++)exposed |= !strcmp(ext[i].extensionName,VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME);
   VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT dgc={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_EXT};
   VkPhysicalDeviceFeatures2 features={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,.pNext=&dgc};
   vkGetPhysicalDeviceFeatures2(pd,&features);
   printf("DGC_EXPOSURE extension=%d feature=%u expected=%d\n",exposed,dgc.deviceGeneratedCommands,atoi(argv[1]));
   free(ext);vkDestroyInstance(instance,NULL);
   return exposed!=atoi(argv[1]) || dgc.deviceGeneratedCommands!=(unsigned)atoi(argv[1]);
}
