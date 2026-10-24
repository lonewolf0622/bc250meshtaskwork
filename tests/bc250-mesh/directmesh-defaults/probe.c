/* SPDX-License-Identifier: MIT
 * Enumeration only. The runner requires a fresh /dev and the noop DRM shim. */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
int main(int argc, char **argv)
{
   if (access("/dev/dri", F_OK) == 0 || !getenv("LD_PRELOAD") ||
       !strstr(getenv("LD_PRELOAD"), "libamdgpu_noop_drm_shim.so")) return 3;
   for (unsigned iteration = 0; iteration < 2; iteration++) {
      VkApplicationInfo app = {.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion=VK_API_VERSION_1_3,
                              .pEngineName=argc > 1 ? argv[1] : "DirectMesh probe"};
      VkInstanceCreateInfo ci = {.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo=&app};
      VkInstance instance;
      if (vkCreateInstance(&ci, NULL, &instance)) return 4;
      unsigned n = 1;
      VkPhysicalDevice pd;
      if (vkEnumeratePhysicalDevices(instance, &n, &pd) || n != 1) return 5;
      VkPhysicalDeviceProperties prop;
      vkGetPhysicalDeviceProperties(pd, &prop);
      VkPhysicalDeviceMeshShaderFeaturesEXT mesh = {.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
      VkPhysicalDeviceFeatures2 f = {.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext=&mesh};
      vkGetPhysicalDeviceFeatures2(pd, &f);
      unsigned count = 0;
      vkEnumerateDeviceExtensionProperties(pd, NULL, &count, NULL);
      VkExtensionProperties *ext = calloc(count, sizeof(*ext));
      if (vkEnumerateDeviceExtensionProperties(pd, NULL, &count, ext)) return 6;
      int gpl = 0, rt = 0;
      for (unsigned i = 0; i < count; i++) {
         gpl |= !strcmp(ext[i].extensionName, "VK_EXT_graphics_pipeline_library");
         rt |= !strcmp(ext[i].extensionName, "VK_KHR_ray_tracing_pipeline");
      }
      printf("PROBE iteration=%u device=%04x mesh=%u task=%u gpl=%d rt=%d preset=%s store=%s hybrid=%s\n",
             iteration, prop.deviceID, mesh.meshShader, mesh.taskShader, gpl, rt,
             getenv("RADV_BC250_MESH_SAFE_FAST") ?: "unset",
             getenv("RADV_BC250_MESH_IDXPASS_STORE") ?: "unset",
             getenv("RADV_BC250_HYBRID_TASK") ?: "unset");
      free(ext);
      vkDestroyInstance(instance, NULL);
   }
   return 0;
}
