/* BC250 barycentrics probe (drm-shim only, never the real GPU).
 *
 *   bary features
 *     Prints EXT=<VK_KHR_fragment_shader_barycentric listed> FEATURE=<fragmentShaderBarycentric>
 *     TRISTRIP_INDEPENDENT=<property> MESH=<meshShader> GPL=<graphicsPipelineLibrary> ESO=<shaderObject>.
 *
 *   bary <mode> <topology> <provoking> VERT.spv [TESC.spv TESE.spv | GEOM.spv] FRAG.spv
 *     mode:      mono (one pipeline), gpl-vkd3d (vertex input + pre-rasterization + fragment shader in
 *                one library, like vkd3d-proton, linked with a fragment output library), gpl-split (one
 *                library per part, fast-linked), gpl-lto (one library per part with retained NIR,
 *                link-time optimized), eso-linked / eso-unlinked (VK_EXT_shader_object).
 *     topology:  tri | strip | fan | line | point | patch (tessellation) ; "gs" as the first shader
 *                argument count decides the stages: 2 files VS+FS, 3 files VS+GS+FS, 4 files
 *                VS+TCS+TES+FS.
 *     provoking: first | last | dynamic (VK_EXT_provoking_vertex / extended dynamic state 3).
 *   With BARY_ALLOW_MISSING set, a driver without the extension runs too (plain fragment shaders only).
 *   Records a draw (and for dynamic, a second draw after switching to the last-vertex mode), submits it
 *   to the noop drm-shim queue and waits. Prints PIPELINE_RESULT and SUBMIT_OK. Refuses any device
 *   that is not the shim's GFX1013.
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

int
main(int argc, char **argv)
{
   if (argc < 2) { fprintf(stderr, "usage: see bary.c\n"); return 2; }
   VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VkInstance in;
   if (vkCreateInstance(&ici, 0, &in)) { puts("NO_INSTANCE"); return 3; }
   uint32_t n = 1; VkPhysicalDevice pd;
   if (vkEnumeratePhysicalDevices(in, &n, &pd) < 0 || !n) { puts("NO_DEVICE"); return 3; }
   VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(pd, &pr);
   if (!strstr(pr.deviceName, "GFX1013")) { puts("NOT_SHIM_GFX1013"); return 3; }

   VkPhysicalDeviceShaderObjectFeaturesEXT qeso = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
   VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT qgpl = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT, &qeso};
   VkPhysicalDeviceMeshShaderFeaturesEXT qmesh = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT, &qgpl};
   VkPhysicalDeviceExtendedDynamicState3FeaturesEXT qeds3 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT, &qmesh};
   VkPhysicalDeviceProvokingVertexFeaturesEXT qpv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT, &qeds3};
   VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR qbary = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR, &qpv};
   VkPhysicalDeviceFeatures2 qf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &qbary};
   vkGetPhysicalDeviceFeatures2(pd, &qf);
   const int ext_bary = has_ext(pd, "VK_KHR_fragment_shader_barycentric");

   if (!strcmp(argv[1], "features")) {
      VkPhysicalDeviceFragmentShaderBarycentricPropertiesKHR pb = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_PROPERTIES_KHR};
      VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &pb};
      vkGetPhysicalDeviceProperties2(pd, &p2);
      printf("EXT=%d FEATURE=%d TRISTRIP_INDEPENDENT=%d MESH=%d TASK=%d GPL=%d ESO=%d PROVOKING_LAST=%d\n", ext_bary,
             ext_bary && qbary.fragmentShaderBarycentric, pb.triStripVertexOrderIndependentOfProvokingVertex,
             qmesh.meshShader, qmesh.taskShader, has_ext(pd, "VK_EXT_graphics_pipeline_library") && qgpl.graphicsPipelineLibrary,
             has_ext(pd, "VK_EXT_shader_object") && qeso.shaderObject, qpv.provokingVertexLast);
      return 0;
   }

   if (argc < 6) { fprintf(stderr, "usage: see bary.c\n"); return 2; }
   const char *mode = argv[1], *topo = argv[2], *prov = argv[3];
   const unsigned nfiles = argc - 4;
   const int gpl = !strncmp(mode, "gpl", 3), eso = !strncmp(mode, "eso", 3);
   const int dyn_prov = !strcmp(prov, "dynamic"), last = !strcmp(prov, "last");
   /* A base build without the extension still runs the plain fragment shader (identity checks). */
   const int bary_on = ext_bary && qbary.fragmentShaderBarycentric;
   if (!bary_on && !getenv("BARY_ALLOW_MISSING")) { puts("NO_BARYCENTRICS"); return 4; }
   if (gpl && !qgpl.graphicsPipelineLibrary) { puts("NO_GPL"); return 4; }
   if (eso && !qeso.shaderObject) { puts("NO_ESO"); return 4; }

   float q = 1;
   VkDeviceQueueCreateInfo qc = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1, .pQueuePriorities = &q};
   VkPhysicalDeviceShaderObjectFeaturesEXT feso = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT, .shaderObject = eso};
   VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT fgpl = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT, &feso, .graphicsPipelineLibrary = gpl};
   VkPhysicalDeviceExtendedDynamicState3FeaturesEXT feds3 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT, &fgpl,
      .extendedDynamicState3ProvokingVertexMode = dyn_prov};
   VkPhysicalDeviceProvokingVertexFeaturesEXT fpv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT, &feds3, .provokingVertexLast = 1};
   VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR fbary = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR, &fpv, .fragmentShaderBarycentric = 1};
   VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, bary_on ? (void *)&fbary : (void *)&fpv,
                                           .dynamicRendering = 1};
   VkPhysicalDeviceFeatures fe = {.geometryShader = 1, .tessellationShader = 1};
   const char *exts[8]; unsigned next = 0;
   if (bary_on) exts[next++] = "VK_KHR_fragment_shader_barycentric";
   exts[next++] = "VK_EXT_provoking_vertex";
   if (dyn_prov) exts[next++] = "VK_EXT_extended_dynamic_state3";
   if (gpl) { exts[next++] = "VK_KHR_pipeline_library"; exts[next++] = "VK_EXT_graphics_pipeline_library"; }
   if (eso) exts[next++] = "VK_EXT_shader_object";
   VkDeviceCreateInfo dc = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f13, 0, 1, &qc, 0, 0, next, exts, &fe};
   CK(vkCreateDevice(pd, &dc, 0, &dev));
   VkQueue queue; vkGetDeviceQueue(dev, 0, 0, &queue);

   VkShaderStageFlagBits stage_bits[4];
   const char *files[4];
   for (unsigned i = 0; i < nfiles && i < 4; i++) files[i] = argv[4 + i];
   if (nfiles == 2) {
      stage_bits[0] = VK_SHADER_STAGE_VERTEX_BIT; stage_bits[1] = VK_SHADER_STAGE_FRAGMENT_BIT;
   } else if (nfiles == 3) {
      stage_bits[0] = VK_SHADER_STAGE_VERTEX_BIT; stage_bits[1] = VK_SHADER_STAGE_GEOMETRY_BIT;
      stage_bits[2] = VK_SHADER_STAGE_FRAGMENT_BIT;
   } else if (nfiles == 4) {
      stage_bits[0] = VK_SHADER_STAGE_VERTEX_BIT; stage_bits[1] = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
      stage_bits[2] = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT; stage_bits[3] = VK_SHADER_STAGE_FRAGMENT_BIT;
   } else {
      fprintf(stderr, "2, 3 or 4 shaders\n"); return 2;
   }
   const int tess = nfiles == 4;

   VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
   if (!strcmp(topo, "strip")) topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
   else if (!strcmp(topo, "fan")) topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
   else if (!strcmp(topo, "line")) topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
   else if (!strcmp(topo, "point")) topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
   if (tess) topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;

   VkPushConstantRange pc = {VK_SHADER_STAGE_ALL, 0, 16};
   VkPipelineLayoutCreateInfo lc = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 0, 0, 1, &pc};
   VkPipelineLayout lay; CK(vkCreatePipelineLayout(dev, &lc, 0, &lay));
   const VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;

   /* Color target. */
   VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, 0, 0, VK_IMAGE_TYPE_2D, fmt, {64, 64, 1}, 1, 1, 1, 0,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
   VkImage img; CK(vkCreateImage(dev, &ic, 0, &img));
   VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, img, &mr);
   VkMemoryAllocateInfo ma = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, mr.size, memtype(pd, mr.memoryTypeBits, 0)};
   VkDeviceMemory mem; CK(vkAllocateMemory(dev, &ma, 0, &mem)); CK(vkBindImageMemory(dev, img, mem, 0));
   VkImageViewCreateInfo iv = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, 0, 0, img, VK_IMAGE_VIEW_TYPE_2D, fmt, {0}, {1, 0, 1, 0, 1}};
   VkImageView view; CK(vkCreateImageView(dev, &iv, 0, &view));

   VkPipeline p = VK_NULL_HANDLE;
   VkShaderEXT shaders[4] = {0};
   VkResult r = VK_SUCCESS;

   if (!eso) {
      VkPipelineShaderStageCreateInfo st[4];
      for (unsigned i = 0; i < nfiles; i++)
         st[i] = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, stage_bits[i],
                                                   load(files[i]), "main"};
      VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
      VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = topology};
      VkPipelineTessellationStateCreateInfo ts = {VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO, .patchControlPoints = 3};
      VkViewport vp = {0, 0, 64, 64, 0, 1}; VkRect2D sc = {{0, 0}, {64, 64}};
      VkPipelineViewportStateCreateInfo vs = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, 0, 0, 1, &vp, 1, &sc};
      VkPipelineRasterizationProvokingVertexStateCreateInfoEXT rpv = {
         VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT, 0,
         last ? VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT : VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT};
      VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, &rpv, .lineWidth = 1};
      VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = 1};
      VkPipelineColorBlendAttachmentState ba = {.colorWriteMask = 15};
      VkPipelineColorBlendStateCreateInfo bs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1, .pAttachments = &ba};
      VkDynamicState dyn[1] = {VK_DYNAMIC_STATE_PROVOKING_VERTEX_MODE_EXT};
      VkPipelineDynamicStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, 0, 0, dyn_prov, dyn};
      VkPipelineRenderingCreateInfo ri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, 0, 0, 1, &fmt};
      VkGraphicsPipelineCreateInfo gp = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &ri, .stageCount = nfiles, .pStages = st,
         .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pTessellationState = tess ? &ts : NULL,
         .pViewportState = &vs, .pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &bs,
         .pDynamicState = &ds, .layout = lay};
      if (!gpl) {
         r = vkCreateGraphicsPipelines(dev, 0, 1, &gp, 0, &p);
      } else {
         /* Parts: 0 vertex input, 1 pre-rasterization, 2 fragment shader, 3 fragment output. */
         const int split = strcmp(mode, "gpl-vkd3d") != 0, lto = !strcmp(mode, "gpl-lto");
         VkPipeline lib[4]; unsigned nlib = 0;
         const VkGraphicsPipelineLibraryFlagsEXT parts[4] = {
            VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT, VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT,
            VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT};
         VkGraphicsPipelineLibraryFlagsEXT groups[4];
         unsigned ngroups = 0;
         if (split) {
            for (unsigned i = 0; i < 4; i++) groups[ngroups++] = parts[i];
         } else {
            groups[ngroups++] = parts[0] | parts[1] | parts[2];
            groups[ngroups++] = parts[3];
         }
         for (unsigned g = 0; g < ngroups; g++) {
            VkGraphicsPipelineLibraryCreateInfoEXT li = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT, &ri, groups[g]};
            VkGraphicsPipelineCreateInfo lp = gp;
            lp.pNext = &li;
            lp.flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR |
                       (lto ? VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT : 0);
            VkPipelineShaderStageCreateInfo lst[4]; unsigned nst = 0;
            for (unsigned i = 0; i < nfiles; i++) {
               const int is_fs = stage_bits[i] == VK_SHADER_STAGE_FRAGMENT_BIT;
               if ((is_fs && (groups[g] & parts[2])) || (!is_fs && (groups[g] & parts[1])))
                  lst[nst++] = st[i];
            }
            lp.stageCount = nst; lp.pStages = nst ? lst : NULL;
            if (!(groups[g] & (parts[1] | parts[2]))) lp.layout = VK_NULL_HANDLE;
            r = vkCreateGraphicsPipelines(dev, 0, 1, &lp, 0, &lib[nlib]);
            printf("LIBRARY_RESULT=%d\n", r);
            if (r) break;
            nlib++;
         }
         if (!r) {
            VkPipelineLibraryCreateInfoKHR link = {VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR, 0, nlib, lib};
            VkGraphicsPipelineCreateInfo lp = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &link,
               .flags = lto ? VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT : 0, .layout = lay};
            r = vkCreateGraphicsPipelines(dev, 0, 1, &lp, 0, &p);
         }
      }
   } else {
      PFN_vkCreateShadersEXT createShaders = (void *)vkGetDeviceProcAddr(dev, "vkCreateShadersEXT");
      const int linked = !strcmp(mode, "eso-linked");
      VkShaderCreateInfoEXT sci[4];
      for (unsigned i = 0; i < nfiles; i++) {
         size_t size;
         void *code = readfile(files[i], &size);
         sci[i] = (VkShaderCreateInfoEXT){VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT, 0,
            linked ? VK_SHADER_CREATE_LINK_STAGE_BIT_EXT : 0, stage_bits[i],
            i + 1 < nfiles ? stage_bits[i + 1] : 0, VK_SHADER_CODE_TYPE_SPIRV_EXT, size, code, "main", 0, 0, 1, &pc};
         if (stage_bits[i] == VK_SHADER_STAGE_VERTEX_BIT && nfiles == 2)
            sci[i].nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
      }
      r = createShaders(dev, nfiles, sci, 0, shaders);
   }
   printf("PIPELINE_RESULT=%d\n", r); fflush(stdout);
   if (r) return 1;

   VkCommandPoolCreateInfo cp = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   VkCommandPool pool; CK(vkCreateCommandPool(dev, &cp, 0, &pool));
   VkCommandBufferAllocateInfo ca = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, pool, 0, 1};
   VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &ca, &cb));
   VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CK(vkBeginCommandBuffer(cb, &bi));
   VkRenderingAttachmentInfo att = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, 0, view, VK_IMAGE_LAYOUT_GENERAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
   VkRenderingInfo rinfo = {VK_STRUCTURE_TYPE_RENDERING_INFO, 0, 0, {{0, 0}, {64, 64}}, 1, 0, 1, &att};
   vkCmdBeginRendering(cb, &rinfo);
   PFN_vkCmdSetProvokingVertexModeEXT setProv = (void *)vkGetDeviceProcAddr(dev, "vkCmdSetProvokingVertexModeEXT");
   if (!eso) {
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
   } else {
      PFN_vkCmdBindShadersEXT bindShaders = (void *)vkGetDeviceProcAddr(dev, "vkCmdBindShadersEXT");
      VkShaderStageFlagBits all[5] = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                                      VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, VK_SHADER_STAGE_GEOMETRY_BIT,
                                      VK_SHADER_STAGE_FRAGMENT_BIT};
      VkShaderEXT bound[5] = {0};
      for (unsigned s = 0; s < 5; s++)
         for (unsigned i = 0; i < nfiles; i++)
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
      PFN_vkCmdSetPatchControlPointsEXT setPatch = (void *)vkGetDeviceProcAddr(dev, "vkCmdSetPatchControlPointsEXT");
      setVI(cb, 0, NULL, 0, NULL);
      vkCmdSetPrimitiveTopology(cb, topology);
      vkCmdSetPrimitiveRestartEnable(cb, VK_FALSE);
      VkViewport vp = {0, 0, 64, 64, 0, 1}; VkRect2D sc = {{0, 0}, {64, 64}};
      vkCmdSetViewportWithCount(cb, 1, &vp);
      vkCmdSetScissorWithCount(cb, 1, &sc);
      vkCmdSetRasterizerDiscardEnable(cb, VK_FALSE);
      setPoly(cb, VK_POLYGON_MODE_FILL);
      vkCmdSetCullMode(cb, VK_CULL_MODE_NONE);
      vkCmdSetFrontFace(cb, VK_FRONT_FACE_COUNTER_CLOCKWISE);
      vkCmdSetDepthTestEnable(cb, VK_FALSE);
      vkCmdSetDepthWriteEnable(cb, VK_FALSE);
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
      if (tess) setPatch(cb, 3);
      setProv(cb, last ? VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT : VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT);
   }
   uint32_t base = 5;
   vkCmdPushConstants(cb, lay, VK_SHADER_STAGE_ALL, 0, 4, &base);
   if (dyn_prov && !eso)
      setProv(cb, VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT);
   vkCmdDraw(cb, 6, 1, 0, 0);
   if (dyn_prov) {
      setProv(cb, VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT);
      vkCmdDraw(cb, 6, 1, 0, 0);
   }
   vkCmdEndRendering(cb);
   CK(vkEndCommandBuffer(cb));
   VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &cb};
   CK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CK(vkQueueWaitIdle(queue));
   printf("SUBMIT_OK\n");
   return 0;
}
