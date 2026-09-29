/* BC250 RADV_BC250_VRS_NOOP probe (drm-shim only, never the real GPU).
 *
 *   vrs features
 *     Prints one line: EXT (VK_KHR_fragment_shading_rate listed), the three rate features, the
 *     properties vkd3d-proton checks for tier 2 (non-trivial combiners, attachment texel size), the
 *     fragment sizes of vkGetPhysicalDeviceFragmentShadingRatesKHR, primitiveFragmentShadingRateMeshShader,
 *     the R8_UINT shading rate attachment format feature, vkGetPhysicalDeviceImageFormatProperties2 with
 *     the shading rate attachment usage, vkCreateDevice with every exposed rate feature enabled, and the
 *     shader object binary UUID (SBUUID).
 *
 *   vrs <api> <fsr> <shaders...>
 *     api: mono (one pipeline), gpl (vkd3d-proton style library: vertex input + pre-rasterization +
 *          fragment shader, linked with a fragment output library), eso (linked shader objects), mesh
 *          (Mesh (+Task) pipeline).
 *     fsr: none, static (pipeline rate 2x2, combiners KEEP/REPLACE), dynamic (dynamic rate:
 *          vkCmdSetFragmentShadingRateKHR 2x2 REPLACE/KEEP, then 1x2 MAX/REPLACE), attachment (dynamic
 *          rendering with an R8_UINT shading rate attachment, 8x8 texels, filled with 2x2 by a clear, the
 *          pipeline flag, a depth attachment and the dynamic rates), attachment-nodepth (the same without
 *          depth), renderpass (VkRenderPass2 with a shading rate attachment, depth and a static rate);
 *          attachment-unbound, attachment-nodepth-unbound, renderpass-unbound: the same rate image, clear,
 *          barriers, depth and rates, but no shading rate attachment in the rendering / subpass and no
 *          pipeline flag (the command streams must be identical to the bound ones).
 *     shaders: VERT FRAG | VERT GEOM FRAG (mono, gpl, eso) ; MESH FRAG [TASK|-] (mesh).
 *   With VRS_ALLOW_MISSING set, a driver without the extension runs "none" (identity checks). Records the
 *   draws twice (mesh: direct, indirect, indirect count; otherwise one draw; with dynamic rates each pass
 *   sets a different rate first), submits them to the noop
 *   drm-shim queue and waits. Prints PIPELINE_RESULT and SUBMIT_OK. Refuses any device that is not the
 *   shim's GFX1013 (VRS_CONTROL_DEVICE=<device name part>: a control device of the noop shim instead, e.g.
 *   NAVI21 for a GFX10.3 control of the IB checks).
 */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do { VkResult r_ = (x); if (r_) { printf("FAIL %s = %d\n", #x, r_); return 1; } } while (0)

static VkDevice dev;

static void *
readfile(const char *p, size_t *size)
{
   FILE *f = fopen(p, "rb");
   if (!f) { perror(p); exit(2); }
   fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
   void *b = malloc(n);
   if (fread(b, 1, n, f) != (size_t)n) exit(2);
   fclose(f);
   *size = n;
   return b;
}

static VkShaderModule
load(const char *p)
{
   size_t n;
   void *b = readfile(p, &n);
   VkShaderModuleCreateInfo c = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, 0, 0, n, b};
   VkShaderModule m;
   if (vkCreateShaderModule(dev, &c, 0, &m)) exit(2);
   return m;
}

static uint32_t
memtype(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
   return __builtin_ctz(bits);
}

static int
has_ext(VkPhysicalDevice pd, const char *name)
{
   uint32_t n = 0;
   vkEnumerateDeviceExtensionProperties(pd, NULL, &n, NULL);
   VkExtensionProperties *e = calloc(n, sizeof(*e));
   vkEnumerateDeviceExtensionProperties(pd, NULL, &n, e);
   int found = 0;
   for (uint32_t i = 0; i < n; i++)
      found |= !strcmp(e[i].extensionName, name);
   free(e);
   return found;
}

static VkResult
make_image(VkPhysicalDevice pd, VkFormat fmt, uint32_t w, uint32_t h, VkImageUsageFlags usage, VkImageAspectFlags aspect,
           VkImage *img, VkImageView *view)
{
   VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, 0, 0, VK_IMAGE_TYPE_2D, fmt, {w, h, 1}, 1, 1, 1, 0, usage};
   VkResult r = vkCreateImage(dev, &ic, 0, img);
   if (r) return r;
   VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, *img, &mr);
   VkMemoryAllocateInfo ma = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, mr.size, memtype(pd, mr.memoryTypeBits, 0)};
   VkDeviceMemory mem;
   if ((r = vkAllocateMemory(dev, &ma, 0, &mem))) return r;
   if ((r = vkBindImageMemory(dev, *img, mem, 0))) return r;
   VkImageViewCreateInfo iv = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, 0, 0, *img, VK_IMAGE_VIEW_TYPE_2D, fmt, {0}, {aspect, 0, 1, 0, 1}};
   return vkCreateImageView(dev, &iv, 0, view);
}

int
main(int argc, char **argv)
{
   if (argc < 2) { fprintf(stderr, "usage: see vrs.c\n"); return 2; }
   VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VkInstance in;
   if (vkCreateInstance(&ici, 0, &in)) { puts("NO_INSTANCE"); return 3; }
   uint32_t n = 1; VkPhysicalDevice pd;
   if (vkEnumeratePhysicalDevices(in, &n, &pd) < 0 || !n) { puts("NO_DEVICE"); return 3; }
   VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(pd, &pr);
   /* VRS_CONTROL_DEVICE=<name part> (control runs of the IB checks only, e.g. the shim's NAVI21, GFX10.3):
    * accept that device instead. */
   const char *control = getenv("VRS_CONTROL_DEVICE");
   if (!strstr(pr.deviceName, control && *control ? control : "GFX1013")) { puts("NOT_SHIM_GFX1013"); return 3; }

   VkPhysicalDeviceShaderObjectFeaturesEXT qeso = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
   VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT qgpl = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT, &qeso};
   VkPhysicalDeviceMeshShaderFeaturesEXT qmesh = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT, &qgpl};
   VkPhysicalDeviceFragmentShadingRateFeaturesKHR qfsr = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR, &qmesh};
   VkPhysicalDeviceFeatures2 qf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &qfsr};
   vkGetPhysicalDeviceFeatures2(pd, &qf);
   const int ext_fsr = has_ext(pd, "VK_KHR_fragment_shading_rate");
   const int gpl_ok = has_ext(pd, "VK_EXT_graphics_pipeline_library") && qgpl.graphicsPipelineLibrary;
   const int eso_ok = has_ext(pd, "VK_EXT_shader_object") && qeso.shaderObject;

   float q = 1;
   VkDeviceQueueCreateInfo qc = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1, .pQueuePriorities = &q};

   if (!strcmp(argv[1], "features")) {
      VkPhysicalDeviceFragmentShadingRatePropertiesKHR pf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES_KHR};
      VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, ext_fsr ? &pf : NULL};
      vkGetPhysicalDeviceProperties2(pd, &p2);
      char rates[256] = "-";
      int dev_result = -1, img_fsr = -1;
      if (ext_fsr) {
         PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR getRates =
            (void *)vkGetInstanceProcAddr(in, "vkGetPhysicalDeviceFragmentShadingRatesKHR");
         uint32_t rn = 0;
         getRates(pd, &rn, NULL);
         VkPhysicalDeviceFragmentShadingRateKHR rr[16];
         for (unsigned i = 0; i < 16; i++) rr[i] = (VkPhysicalDeviceFragmentShadingRateKHR){VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_KHR};
         if (rn > 16) rn = 16;
         getRates(pd, &rn, rr);
         rates[0] = 0;
         for (unsigned i = 0; i < rn; i++)
            snprintf(rates + strlen(rates), sizeof(rates) - strlen(rates), "%s%ux%u:%x", i ? "," : "", rr[i].fragmentSize.width,
                     rr[i].fragmentSize.height, rr[i].sampleCounts & 0xff);

         VkPhysicalDeviceImageFormatInfo2 fi = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, 0, VK_FORMAT_R8_UINT,
            VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
         VkImageFormatProperties2 fp = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
         img_fsr = vkGetPhysicalDeviceImageFormatProperties2(pd, &fi, &fp);

         VkPhysicalDeviceFragmentShadingRateFeaturesKHR ffsr = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR, 0,
            qfsr.pipelineFragmentShadingRate, qfsr.primitiveFragmentShadingRate, qfsr.attachmentFragmentShadingRate};
         const char *e = "VK_KHR_fragment_shading_rate";
         VkDeviceCreateInfo dc = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &ffsr, 0, 1, &qc, 0, 0, 1, &e, 0};
         VkDevice d;
         dev_result = vkCreateDevice(pd, &dc, 0, &d);
         if (!dev_result) vkDestroyDevice(d, 0);
      }
      VkFormatProperties3 f3 = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
      VkFormatProperties2 f2 = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &f3};
      vkGetPhysicalDeviceFormatProperties2(pd, VK_FORMAT_R8_UINT, &f2);
      VkPhysicalDeviceShaderObjectPropertiesEXT pso = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_PROPERTIES_EXT};
      VkPhysicalDeviceProperties2 p2so = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &pso};
      vkGetPhysicalDeviceProperties2(pd, &p2so);
      char sbuuid[2 * VK_UUID_SIZE + 1];
      for (unsigned i = 0; i < VK_UUID_SIZE; i++)
         snprintf(sbuuid + 2 * i, 3, "%02x", pso.shaderBinaryUUID[i]);
      printf("EXT=%d PIPELINE=%d PRIMITIVE=%d ATTACHMENT=%d NONTRIVIAL=%d TEXEL=%ux%u-%ux%u MAXSIZE=%ux%u RATES=%s MESH_FSR=%d "
             "R8_FSR=%d IMG_FSR=%d DEVICE=%d MSAA2=%d MESH=%d TASK=%d GPL=%d ESO=%d SBUUID=%s\n",
             ext_fsr, ext_fsr && qfsr.pipelineFragmentShadingRate, ext_fsr && qfsr.primitiveFragmentShadingRate,
             ext_fsr && qfsr.attachmentFragmentShadingRate, ext_fsr && pf.fragmentShadingRateNonTrivialCombinerOps,
             pf.minFragmentShadingRateAttachmentTexelSize.width, pf.minFragmentShadingRateAttachmentTexelSize.height,
             pf.maxFragmentShadingRateAttachmentTexelSize.width, pf.maxFragmentShadingRateAttachmentTexelSize.height,
             pf.maxFragmentSize.width, pf.maxFragmentSize.height, rates, qmesh.primitiveFragmentShadingRateMeshShader,
             !!(f3.optimalTilingFeatures & VK_FORMAT_FEATURE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR), img_fsr, dev_result,
             !!(pr.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_2_BIT), qmesh.meshShader, qmesh.taskShader, gpl_ok, eso_ok,
             sbuuid);
      return 0;
   }

   if (argc < 5) { fprintf(stderr, "usage: see vrs.c\n"); return 2; }
   const char *api = argv[1], *fsr = argv[2];
   const int mesh = !strcmp(api, "mesh"), gpl = !strcmp(api, "gpl"), eso = !strcmp(api, "eso");
   const int f_static = !strcmp(fsr, "static"), f_rp = !strncmp(fsr, "renderpass", 10);
   const int f_att = !strncmp(fsr, "attachment", 10) || f_rp;
   /* <mode>-unbound: the rate image, its clear and barriers, but no shading rate attachment in the
    * rendering / subpass and no pipeline flag (to compare the command streams). */
   const int f_bind = f_att && !strstr(fsr, "-unbound");
   const int f_depth = (f_att && !strstr(fsr, "nodepth")) || f_rp;
   const int f_dyn = !strcmp(fsr, "dynamic") || !strncmp(fsr, "attachment", 10);
   const int f_any = strcmp(fsr, "none") != 0;
   const int fsr_on = ext_fsr && qfsr.pipelineFragmentShadingRate;
   if (!fsr_on && (f_any || !getenv("VRS_ALLOW_MISSING"))) { puts("NO_FSR"); return 4; }
   if (gpl && !gpl_ok) { puts("NO_GPL"); return 4; }
   if (eso && !eso_ok) { puts("NO_ESO"); return 4; }
   if (eso && (f_static || f_rp)) { fprintf(stderr, "shader objects: none, dynamic or attachment*\n"); return 2; }
   if (f_att && !qfsr.attachmentFragmentShadingRate) { puts("NO_FSR_ATTACHMENT"); return 4; }

   VkPhysicalDeviceFragmentShadingRateFeaturesKHR ffsr = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR, 0,
      1, qfsr.primitiveFragmentShadingRate, qfsr.attachmentFragmentShadingRate};
   VkPhysicalDeviceShaderObjectFeaturesEXT feso = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT, fsr_on ? (void *)&ffsr : NULL,
      .shaderObject = eso};
   VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT fgpl = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT, &feso,
      .graphicsPipelineLibrary = gpl};
   VkPhysicalDeviceMeshShaderFeaturesEXT fmesh = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT, &fgpl, .meshShader = mesh,
      .taskShader = mesh && qmesh.taskShader};
   VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &fmesh, .dynamicRendering = 1};
   VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13, .drawIndirectCount = 1};
   VkPhysicalDeviceFeatures fe = {.geometryShader = 1, .multiDrawIndirect = 1};
   const char *exts[8]; unsigned next = 0;
   if (fsr_on) exts[next++] = "VK_KHR_fragment_shading_rate";
   if (mesh) exts[next++] = "VK_EXT_mesh_shader";
   if (gpl) { exts[next++] = "VK_KHR_pipeline_library"; exts[next++] = "VK_EXT_graphics_pipeline_library"; }
   if (eso) exts[next++] = "VK_EXT_shader_object";
   VkDeviceCreateInfo dc = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f12, 0, 1, &qc, 0, 0, next, exts, &fe};
   CK(vkCreateDevice(pd, &dc, 0, &dev));
   VkQueue queue; vkGetDeviceQueue(dev, 0, 0, &queue);

   const unsigned nfiles = argc - 3;
   const char **files = (const char **)argv + 3;
   VkShaderStageFlagBits stage_bits[3];
   unsigned nstages = nfiles;
   if (mesh) {
      stage_bits[0] = VK_SHADER_STAGE_MESH_BIT_EXT; stage_bits[1] = VK_SHADER_STAGE_FRAGMENT_BIT;
      nstages = 2;
      if (nfiles > 2 && strcmp(files[2], "-")) { stage_bits[2] = VK_SHADER_STAGE_TASK_BIT_EXT; nstages = 3; }
   } else if (nfiles == 2) {
      stage_bits[0] = VK_SHADER_STAGE_VERTEX_BIT; stage_bits[1] = VK_SHADER_STAGE_FRAGMENT_BIT;
   } else if (nfiles == 3) {
      stage_bits[0] = VK_SHADER_STAGE_VERTEX_BIT; stage_bits[1] = VK_SHADER_STAGE_GEOMETRY_BIT; stage_bits[2] = VK_SHADER_STAGE_FRAGMENT_BIT;
   } else {
      fprintf(stderr, "2 or 3 shaders\n"); return 2;
   }

   VkPushConstantRange pc = {VK_SHADER_STAGE_ALL, 0, 16};
   VkPipelineLayoutCreateInfo lc = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 0, 0, 1, &pc};
   VkPipelineLayout lay; CK(vkCreatePipelineLayout(dev, &lc, 0, &lay));
   const VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM, dfmt = VK_FORMAT_D32_SFLOAT, rfmt = VK_FORMAT_R8_UINT;
   const VkExtent2D texel = {8, 8};

   VkImage img, dimg = VK_NULL_HANDLE, rimg = VK_NULL_HANDLE;
   VkImageView view, dview = VK_NULL_HANDLE, rview = VK_NULL_HANDLE;
   CK(make_image(pd, fmt, 64, 64, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT, &img, &view));
   if (f_depth)
      CK(make_image(pd, dfmt, 64, 64, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, &dimg, &dview));
   if (f_att)
      CK(make_image(pd, rfmt, 64 / texel.width, 64 / texel.height,
                    VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT, &rimg, &rview));

   /* Legacy render pass with a shading rate attachment (attachment 2). */
   VkRenderPass rp = VK_NULL_HANDLE;
   VkFramebuffer fb = VK_NULL_HANDLE;
   if (f_rp) {
      VkAttachmentDescription2 at[3] = {
         {VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2, 0, 0, fmt, 1, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
          VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL},
         {VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2, 0, 0, dfmt, 1, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
          VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_UNDEFINED,
          VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL},
         {VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2, 0, 0, rfmt, 1, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_NONE,
          VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
          VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR, VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR}};
      VkAttachmentReference2 cref = {VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, 0, 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     VK_IMAGE_ASPECT_COLOR_BIT};
      VkAttachmentReference2 dref = {VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, 0, 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                     VK_IMAGE_ASPECT_DEPTH_BIT};
      VkAttachmentReference2 rref = {VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, 0, 2,
                                     VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR, VK_IMAGE_ASPECT_COLOR_BIT};
      VkFragmentShadingRateAttachmentInfoKHR fsa = {VK_STRUCTURE_TYPE_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR, 0, &rref, texel};
      VkSubpassDescription2 sp = {VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2, f_bind ? &fsa : NULL, 0, VK_PIPELINE_BIND_POINT_GRAPHICS, 0, 0, 0, 1, &cref,
                                  0, &dref};
      VkRenderPassCreateInfo2 rc = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2, 0, 0, 3, at, 1, &sp};
      CK(vkCreateRenderPass2(dev, &rc, 0, &rp));
      VkImageView fv[3] = {view, dview, rview};
      VkFramebufferCreateInfo fc = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, 0, 0, rp, 3, fv, 64, 64, 1};
      CK(vkCreateFramebuffer(dev, &fc, 0, &fb));
   }

   VkPipeline p = VK_NULL_HANDLE;
   VkShaderEXT shaders[3] = {0};
   VkResult r = VK_SUCCESS;
   const VkPipelineCreateFlags att_flag = f_bind && !f_rp ? VK_PIPELINE_CREATE_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR : 0;

   if (!eso) {
      VkPipelineShaderStageCreateInfo st[3];
      for (unsigned i = 0; i < nstages; i++)
         st[i] = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, stage_bits[i],
                                                   load(files[i]), "main"};
      VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
      VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                   .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
      VkViewport vp = {0, 0, 64, 64, 0, 1}; VkRect2D sc = {{0, 0}, {64, 64}};
      VkPipelineViewportStateCreateInfo vs = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, 0, 0, 1, &vp, 1, &sc};
      VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .lineWidth = 1};
      VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = 1};
      VkPipelineDepthStencilStateCreateInfo dss = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, .depthTestEnable = 1,
                                                   .depthWriteEnable = 1, .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL};
      VkPipelineColorBlendAttachmentState ba = {.colorWriteMask = 15};
      VkPipelineColorBlendStateCreateInfo bs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1, .pAttachments = &ba};
      VkDynamicState dyn[1] = {VK_DYNAMIC_STATE_FRAGMENT_SHADING_RATE_KHR};
      VkPipelineDynamicStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, 0, 0, f_dyn, dyn};
      VkPipelineFragmentShadingRateStateCreateInfoKHR fsrs = {VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_STATE_CREATE_INFO_KHR, 0,
         f_rp ? (VkExtent2D){1, 1} : (VkExtent2D){2, 2},
         {VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR, VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR}};
      VkPipelineRenderingCreateInfo ri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, (f_static || f_rp) ? &fsrs : NULL, 0, 1, &fmt,
                                          f_depth ? dfmt : VK_FORMAT_UNDEFINED};
      VkGraphicsPipelineCreateInfo gp = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, f_rp ? (void *)&fsrs : (void *)&ri,
         .flags = att_flag, .stageCount = nstages, .pStages = st,
         .pVertexInputState = mesh ? NULL : &vi, .pInputAssemblyState = mesh ? NULL : &ia,
         .pViewportState = &vs, .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = f_depth ? &dss : NULL,
         .pColorBlendState = &bs, .pDynamicState = &ds, .layout = lay, .renderPass = rp};
      if (!gpl) {
         r = vkCreateGraphicsPipelines(dev, 0, 1, &gp, 0, &p);
      } else {
         /* vkd3d-proton style: vertex input + pre-rasterization + fragment shader, then fragment output. */
         const VkGraphicsPipelineLibraryFlagsEXT groups[2] = {
            VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT | VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT |
               VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT,
            VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT};
         VkPipeline lib[2];
         for (unsigned g = 0; g < 2; g++) {
            VkGraphicsPipelineLibraryCreateInfoEXT li = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT, gp.pNext, groups[g]};
            VkGraphicsPipelineCreateInfo lp = gp;
            lp.pNext = &li;
            lp.flags |= VK_PIPELINE_CREATE_LIBRARY_BIT_KHR;
            if (g) { lp.stageCount = 0; lp.pStages = NULL; lp.layout = VK_NULL_HANDLE; }
            r = vkCreateGraphicsPipelines(dev, 0, 1, &lp, 0, &lib[g]);
            printf("LIBRARY_RESULT=%d\n", r);
            if (r) break;
         }
         if (!r) {
            VkPipelineLibraryCreateInfoKHR link = {VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR, 0, 2, lib};
            VkGraphicsPipelineCreateInfo lp = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &link, .flags = att_flag, .layout = lay};
            r = vkCreateGraphicsPipelines(dev, 0, 1, &lp, 0, &p);
         }
      }
   } else {
      PFN_vkCreateShadersEXT createShaders = (void *)vkGetDeviceProcAddr(dev, "vkCreateShadersEXT");
      VkShaderCreateInfoEXT sci[3];
      for (unsigned i = 0; i < nstages; i++) {
         size_t size;
         void *code = readfile(files[i], &size);
         sci[i] = (VkShaderCreateInfoEXT){VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT, 0, VK_SHADER_CREATE_LINK_STAGE_BIT_EXT, stage_bits[i],
            i + 1 < nstages ? stage_bits[i + 1] : 0, VK_SHADER_CODE_TYPE_SPIRV_EXT, size, code, "main", 0, 0, 1, &pc};
         if (stage_bits[i] == VK_SHADER_STAGE_FRAGMENT_BIT && f_bind)
            sci[i].flags |= VK_SHADER_CREATE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_EXT;
      }
      r = createShaders(dev, nstages, sci, 0, shaders);
   }
   printf("PIPELINE_RESULT=%d\n", r); fflush(stdout);
   if (r) return 1;

   /* Mesh indirect records {x,y,z} at 0 and 16 (stride 16), count (=2) at 64. */
   VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, 0, 0, 256, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT};
   VkBuffer ib; CK(vkCreateBuffer(dev, &bci, 0, &ib));
   VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, ib, &mr);
   VkMemoryAllocateInfo ma = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, mr.size,
      memtype(pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
   VkDeviceMemory im; CK(vkAllocateMemory(dev, &ma, 0, &im)); CK(vkBindBufferMemory(dev, ib, im, 0));
   uint32_t *rec; CK(vkMapMemory(dev, im, 0, 256, 0, (void **)&rec));
   memset(rec, 0, 256);
   rec[0] = 3; rec[1] = 1; rec[2] = 1; rec[4] = 2; rec[5] = 2; rec[6] = 1; rec[16] = 2;

   VkCommandPoolCreateInfo cp = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   VkCommandPool pool; CK(vkCreateCommandPool(dev, &cp, 0, &pool));
   VkCommandBufferAllocateInfo ca = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, pool, 0, 1};
   VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &ca, &cb));
   VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CK(vkBeginCommandBuffer(cb, &bi));

   if (f_att) {
      /* Fill the rate image with 2x2 ((1 << 2) | 1), then make it a shading rate attachment. */
      VkImageMemoryBarrier2 b0 = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, 0, VK_PIPELINE_STAGE_2_NONE, 0,
         VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, 0, rimg, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
      VkDependencyInfo d0 = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b0};
      vkCmdPipelineBarrier2(cb, &d0);
      VkClearColorValue rate = {.uint32 = {5, 0, 0, 0}};
      VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      vkCmdClearColorImage(cb, rimg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &rate, 1, &rr);
      VkImageMemoryBarrier2 b1 = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, 0, VK_PIPELINE_STAGE_2_CLEAR_BIT,
         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
         VK_ACCESS_2_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR, 0, 0, rimg, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
      VkDependencyInfo d1 = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b1};
      vkCmdPipelineBarrier2(cb, &d1);
   }

   if (f_rp) {
      VkClearValue cv[3] = {{.color = {{0}}}, {.depthStencil = {1.0f, 0}}, {{{0}}}};
      VkRenderPassBeginInfo rb = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, 0, rp, fb, {{0, 0}, {64, 64}}, 3, cv};
      vkCmdBeginRenderPass(cb, &rb, 0);
   } else {
      VkRenderingAttachmentInfo att = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, 0, view, VK_IMAGE_LAYOUT_GENERAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
      VkRenderingAttachmentInfo datt = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, 0, dview, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .clearValue = {.depthStencil = {1.0f, 0}}};
      VkRenderingFragmentShadingRateAttachmentInfoKHR ratt = {VK_STRUCTURE_TYPE_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR, 0,
         rview, VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR, texel};
      VkRenderingInfo rinfo = {VK_STRUCTURE_TYPE_RENDERING_INFO, f_bind ? &ratt : NULL, 0, {{0, 0}, {64, 64}}, 1, 0, 1, &att,
                               f_depth ? &datt : NULL};
      vkCmdBeginRendering(cb, &rinfo);
   }

   if (!eso) {
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
   } else {
      PFN_vkCmdBindShadersEXT bindShaders = (void *)vkGetDeviceProcAddr(dev, "vkCmdBindShadersEXT");
      VkShaderStageFlagBits all[5] = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                                      VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, VK_SHADER_STAGE_GEOMETRY_BIT,
                                      VK_SHADER_STAGE_FRAGMENT_BIT};
      VkShaderEXT bound[5] = {0};
      for (unsigned s = 0; s < 5; s++)
         for (unsigned i = 0; i < nstages; i++)
            if (stage_bits[i] == all[s]) bound[s] = shaders[i];
      bindShaders(cb, 5, all, bound);
      /* Every state of a shader object draw is dynamic. */
      PFN_vkCmdSetVertexInputEXT setVI = (void *)vkGetDeviceProcAddr(dev, "vkCmdSetVertexInputEXT");
      PFN_vkCmdSetPolygonModeEXT setPoly = (void *)vkGetDeviceProcAddr(dev, "vkCmdSetPolygonModeEXT");
      PFN_vkCmdSetRasterizationSamplesEXT setSamples = (void *)vkGetDeviceProcAddr(dev, "vkCmdSetRasterizationSamplesEXT");
      PFN_vkCmdSetSampleMaskEXT setMask = (void *)vkGetDeviceProcAddr(dev, "vkCmdSetSampleMaskEXT");
      PFN_vkCmdSetAlphaToCoverageEnableEXT setA2C = (void *)vkGetDeviceProcAddr(dev, "vkCmdSetAlphaToCoverageEnableEXT");
      PFN_vkCmdSetColorBlendEnableEXT setBlend = (void *)vkGetDeviceProcAddr(dev, "vkCmdSetColorBlendEnableEXT");
      PFN_vkCmdSetColorWriteMaskEXT setWrite = (void *)vkGetDeviceProcAddr(dev, "vkCmdSetColorWriteMaskEXT");
      setVI(cb, 0, NULL, 0, NULL);
      vkCmdSetPrimitiveTopology(cb, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
      vkCmdSetPrimitiveRestartEnable(cb, VK_FALSE);
      VkViewport vp = {0, 0, 64, 64, 0, 1}; VkRect2D sc = {{0, 0}, {64, 64}};
      vkCmdSetViewportWithCount(cb, 1, &vp);
      vkCmdSetScissorWithCount(cb, 1, &sc);
      vkCmdSetRasterizerDiscardEnable(cb, VK_FALSE);
      setPoly(cb, VK_POLYGON_MODE_FILL);
      vkCmdSetCullMode(cb, VK_CULL_MODE_NONE);
      vkCmdSetFrontFace(cb, VK_FRONT_FACE_COUNTER_CLOCKWISE);
      vkCmdSetDepthTestEnable(cb, f_depth);
      vkCmdSetDepthWriteEnable(cb, f_depth);
      vkCmdSetDepthCompareOp(cb, VK_COMPARE_OP_LESS_OR_EQUAL);
      vkCmdSetDepthBiasEnable(cb, VK_FALSE);
      vkCmdSetStencilTestEnable(cb, VK_FALSE);
      vkCmdSetDepthBoundsTestEnable(cb, VK_FALSE);
      vkCmdSetLineWidth(cb, 1.0f);
      setSamples(cb, VK_SAMPLE_COUNT_1_BIT);
      VkSampleMask sm = ~0u;
      setMask(cb, VK_SAMPLE_COUNT_1_BIT, &sm);
      setA2C(cb, VK_FALSE);
      VkBool32 off = VK_FALSE;
      setBlend(cb, 0, 1, &off);
      VkColorComponentFlags wm = 15;
      setWrite(cb, 0, 1, &wm);
   }
   uint32_t base = 5;
   vkCmdPushConstants(cb, lay, VK_SHADER_STAGE_ALL, 0, 4, &base);

   PFN_vkCmdSetFragmentShadingRateKHR setRate = (void *)vkGetDeviceProcAddr(dev, "vkCmdSetFragmentShadingRateKHR");
   PFN_vkCmdDrawMeshTasksEXT drawMesh = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksEXT");
   PFN_vkCmdDrawMeshTasksIndirectEXT drawInd = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksIndirectEXT");
   PFN_vkCmdDrawMeshTasksIndirectCountEXT drawCnt = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksIndirectCountEXT");
   const VkExtent2D sizes[2] = {{2, 2}, {1, 2}};
   const VkFragmentShadingRateCombinerOpKHR ops[2][2] = {
      {VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR, VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR},
      {VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MAX_KHR, VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR}};
   unsigned rate_calls = 0;
   /* Two passes in every mode, so that the command streams of the modes can be compared. */
   for (unsigned pass = 0; pass < 2; pass++) {
      if (f_dyn) {
         setRate(cb, &sizes[pass], ops[pass]);
         rate_calls++;
      }
      if (mesh) {
         drawMesh(cb, 7, 1, 1);
         drawInd(cb, ib, 0, 2, 16);
         drawCnt(cb, ib, 0, ib, 64, 2, 16);
      } else {
         vkCmdDraw(cb, 6, 1, 0, 0);
      }
   }
   if (f_rp)
      vkCmdEndRenderPass(cb);
   else
      vkCmdEndRendering(cb);
   CK(vkEndCommandBuffer(cb));
   VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &cb};
   CK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CK(vkQueueWaitIdle(queue));
   printf("RATE_CALLS=%u\nSUBMIT_OK\n", rate_calls);
   return 0;
}
