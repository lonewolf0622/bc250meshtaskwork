/* RADV_BC250_EXPOSE_FAST_BINDING offline harness: a vkd3d-proton-like binding model on BC250 Mesh/Task
 * pipelines. drm-shim ONLY: refuses to run when the noop drm-shim is not preloaded or when the device is
 * not the shim's GFX1013 (run.sh also hides /dev/dri).
 *
 * usage: fb <mode> <script> [submits]
 *   mode legacy    descriptor sets (update-after-bind heap sets, immutable static sampler set, push
 *                  descriptor root CBV): what vkd3d-proton records when VK_EXT_descriptor_buffer is hidden
 *   mode legacydb  the same bindings, but VK_EXT_descriptor_buffer is enabled and the two descriptor
 *                  buffers are allocated exactly as in "db" (unused): same memory/VA layout as "db"
 *   mode db        what vkd3d-proton records with VK_EXT_descriptor_buffer: descriptor-buffer set layouts
 *                  (heap sets, EMBEDDED_IMMUTABLE_SAMPLERS static sampler set, push descriptor set),
 *                  pipelines with VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT,
 *                  vkCmdBindDescriptorBuffersEXT + vkCmdSetDescriptorBufferOffsetsEXT +
 *                  vkCmdBindDescriptorBufferEmbeddedSamplersEXT + vkCmdPushDescriptorSetKHR
 * Set layout (vkd3d-like): set 0 = resource heap (MUTABLE {STORAGE_BUFFER, SAMPLED_IMAGE}, 64, variable
 * count), set 1 = sampler heap (SAMPLER, 16), set 2 = static sampler (1 immutable sampler), set 3 = root
 * CBV (push descriptor UNIFORM_BUFFER); 16 bytes of root constants (push constants, all stages).
 * The graphics bind point uses the heap at FB_GFX_OFFSET; the compute bind point uses its own copy of the heap at
 * FB_COMPUTE_OFFSET (descriptor buffer) / its own descriptor sets (legacy), so stale graphics bindings on
 * the compute bind point are visible in the IB.
 * Shaders (work directory): small.mesh.spv (32V/32P), nanite.mesh.spv (256V/128P per-primitive: direct
 * split), tsmall.mesh.spv + fb.task.spv (hybrid Task), tnanite.mesh.spv + fb.task.spv (hybrid Task with a
 * split Mesh), fb.frag.spv / fbpp.frag.spv, fb.vert.spv, fb.comp.spv.
 * Script tokens ("tok*N" repeats):
 *   B E     vkCmdBeginRendering / vkCmdEndRendering
 *   m mi    small Mesh direct (3,1,1) / indirect (2 records, stride 12)
 *   sd      nanite direct (2,1,1) (direct split)
 *   si sc   nanite indirect (2 records, stride 12) / indirect count (max 4, stride 16, count k % 5)
 *   t ti tc Task + small Mesh direct (2,1,1) / indirect (2 records) / indirect count (max 3, count 2)
 *   T       Task + nanite Mesh direct (2,1,1)
 *   v       VS draw
 *   cd      application compute dispatch (7,3,1) on the compute bind point (outside a render pass)
 *   rb      rebind all descriptors (both bind points)
 * Prints FB_* lines (stdout): features, the descriptor addresses the IB must carry, one FB_DRAW per
 * draw/dispatch. */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("FAIL %s = %d @%d\n", #x, r_, __LINE__); fflush(stdout); exit(1); } } while (0)

enum { MODE_LEGACY, MODE_LEGACYDB, MODE_DB };
static int mode;
static VkDevice dev;
static VkPhysicalDevice pd;
/* Set offsets inside the descriptor buffers: odd values so that the set pointers are distinct from
 * every other address the IB carries (indirect argument buffers, driver scratch). */
#define FB_GFX_OFFSET 0x440u
#define FB_COMPUTE_OFFSET 0x10880u
#define FB_HEAP_COUNT 64u
#define FB_SAMPLER_COUNT 16u

static PFN_vkCmdDrawMeshTasksEXT drawMesh;
static PFN_vkCmdDrawMeshTasksIndirectEXT drawMeshInd;
static PFN_vkCmdDrawMeshTasksIndirectCountEXT drawMeshCnt;
static PFN_vkCmdPushDescriptorSetKHR pushDesc;
static PFN_vkCmdBindDescriptorBuffersEXT bindDB;
static PFN_vkCmdSetDescriptorBufferOffsetsEXT setDBOffsets;
static PFN_vkCmdBindDescriptorBufferEmbeddedSamplersEXT bindEmbedded;
static PFN_vkGetDescriptorSetLayoutSizeEXT getLayoutSize;
static PFN_vkGetDescriptorSetLayoutBindingOffsetEXT getBindingOffset;
static PFN_vkGetDescriptorEXT getDescriptor;

static VkShaderModule load(const char *path)
{
   FILE *f = fopen(path, "rb");
   if (!f) { perror(path); exit(2); }
   fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
   uint32_t *code = malloc(n); if (fread(code, 1, n, f) != (size_t)n) exit(2); fclose(f);
   VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, n, code};
   VkShaderModule m; CK(vkCreateShaderModule(dev, &ci, NULL, &m)); free(code); return m;
}

static uint32_t memtype(uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
   exit(2);
}

struct buf { VkBuffer b; VkDeviceMemory m; void *map; VkDeviceAddress va; VkDeviceSize size; };

static void mkbuf(struct buf *out, VkDeviceSize size, VkBufferUsageFlags usage)
{
   VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size,
                            usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_SHARING_MODE_EXCLUSIVE};
   CK(vkCreateBuffer(dev, &bi, NULL, &out->b));
   VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, out->b, &mr);
   VkMemoryAllocateFlagsInfo fi = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
   VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fi, mr.size,
                              memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)};
   CK(vkAllocateMemory(dev, &ai, NULL, &out->m)); CK(vkBindBufferMemory(dev, out->b, out->m, 0));
   CK(vkMapMemory(dev, out->m, 0, size, 0, &out->map));
   memset(out->map, 0, size);
   VkBufferDeviceAddressInfo bai = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, out->b};
   out->va = vkGetBufferDeviceAddress(dev, &bai);
   out->size = size;
}

enum { SH_SMALL, SH_NANITE, SH_TSMALL, SH_TNANITE, SH_VS, SH_COUNT };
static const char *shape_names[SH_COUNT] = {"small", "nanite", "tsmall", "tnanite", "vs"};
static VkPipeline pipes[SH_COUNT];
static VkPipeline cpipe;
static VkPipelineLayout pl;
static VkFormat cf = VK_FORMAT_R8G8B8A8_UNORM;

static VkPipeline pipeline(int shape)
{
   if (pipes[shape])
      return pipes[shape];
   char m[64];
   VkPipelineRenderingCreateInfo prci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, 1, &cf};
   VkPipelineCreateFlags2CreateInfo flags2 = {VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO, &prci,
                                              VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT};
   VkPipelineViewportStateCreateInfo vps = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL};
   VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
   rs.lineWidth = 1; rs.cullMode = VK_CULL_MODE_BACK_BIT; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
   rs.polygonMode = VK_POLYGON_MODE_FILL;
   VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
   ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
   VkPipelineColorBlendAttachmentState cba = {0}; cba.colorWriteMask = 0xf;
   VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, 0, 0, 1, &cba};
   VkDynamicState dyns[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 2, dyns};
   VkPipelineShaderStageCreateInfo st[3] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_MESH_BIT_EXT, VK_NULL_HANDLE, "main"},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, VK_NULL_HANDLE, "main"},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_TASK_BIT_EXT, VK_NULL_HANDLE, "main"}};
   VkPipelineVertexInputStateCreateInfo vis = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ias = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0,
                                                 VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkGraphicsPipelineCreateInfo gpi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, mode == MODE_DB ? (void *)&flags2 : (void *)&prci,
                                       0, 2, st, NULL, NULL, NULL, &vps, &rs, &ms, NULL, &cbs, &ds, pl};
   const int perprim = shape == SH_NANITE || shape == SH_TNANITE;
   if (shape == SH_VS) {
      st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
      st[0].module = load("fb.vert.spv");
      gpi.pVertexInputState = &vis; gpi.pInputAssemblyState = &ias;
   } else {
      snprintf(m, sizeof(m), "%s.mesh.spv", shape_names[shape]);
      st[0].module = load(m);
   }
   st[1].module = load(perprim ? "fbpp.frag.spv" : "fb.frag.spv");
   if (shape == SH_TSMALL || shape == SH_TNANITE) {
      st[2].module = load("fb.task.spv");
      gpi.stageCount = 3;
   }
   VkResult pr = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, NULL, &pipes[shape]);
   printf("FB_PIPELINE %s RESULT=%d\n", shape_names[shape], pr); fflush(stdout);
   if (pr) exit(1);
   return pipes[shape];
}

static struct buf args, sdata, ubo, dbres, dbsmp;
static uint32_t *argmap;
static unsigned draw_k;
static VkDescriptorSet gsets[3], csets[3];
static VkDescriptorBufferInfo cbv;
static VkDeviceSize heap_stride, smp_stride, heap_off0, smp_off0;

static uint32_t rnd_state = 12345;
static uint32_t rnd(void) { rnd_state = rnd_state * 1103515245u + 12345u; return (rnd_state >> 16) & 0x7fff; }

static void fill(unsigned k, unsigned records, unsigned stride, int count_value)
{
   for (unsigned r = 0; r < records; r++) {
      uint32_t *p = argmap + (256 * k + r * stride) / 4;
      p[0] = 1 + rnd() % 3; p[1] = 1 + (k + r) % 2; p[2] = 1;
   }
   if (count_value >= 0)
      argmap[(256 * k + 192) / 4] = count_value;
}

static void bind_all(VkCommandBuffer cb)
{
   const VkPipelineBindPoint bps[2] = {VK_PIPELINE_BIND_POINT_GRAPHICS, VK_PIPELINE_BIND_POINT_COMPUTE};
   if (mode == MODE_DB) {
      VkDescriptorBufferBindingPushDescriptorBufferHandleEXT push_handle = {
         VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_PUSH_DESCRIPTOR_BUFFER_HANDLE_EXT, NULL, dbres.b};
      VkPhysicalDeviceDescriptorBufferPropertiesEXT dbp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT};
      VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &dbp};
      vkGetPhysicalDeviceProperties2(pd, &p2);
      VkDescriptorBufferBindingInfoEXT bbi[2] = {
         {VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT, dbp.bufferlessPushDescriptors ? NULL : &push_handle, dbres.va,
          VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
             (dbp.bufferlessPushDescriptors ? 0 : VK_BUFFER_USAGE_PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER_BIT_EXT)},
         {VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT, NULL, dbsmp.va, VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT}};
      bindDB(cb, 2, bbi);
      for (int i = 0; i < 2; i++) {
         const uint32_t idx[2] = {0, 1};
         const VkDeviceSize off[2] = {i ? FB_COMPUTE_OFFSET : FB_GFX_OFFSET, i ? FB_COMPUTE_OFFSET : FB_GFX_OFFSET};
         setDBOffsets(cb, bps[i], pl, 0, 2, idx, off);
         bindEmbedded(cb, bps[i], pl, 2);
      }
   } else {
      vkCmdBindDescriptorSets(cb, bps[0], pl, 0, 3, gsets, 0, NULL);
      vkCmdBindDescriptorSets(cb, bps[1], pl, 0, 3, csets, 0, NULL);
   }
   VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 0, 0, 1,
                             VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, NULL, &cbv};
   for (int i = 0; i < 2; i++)
      pushDesc(cb, bps[i], pl, 3, 1, &w);
   const uint32_t rc[4] = {0, 2, 0, 0};
   vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_ALL, 0, 16, rc);
}

static void indirect(VkCommandBuffer cb, const char *tok, int shape, unsigned records, unsigned stride, int count_value)
{
   unsigned k = draw_k++;
   if (256 * (k + 1) > 64 * 1024) { printf("FAIL too many draws\n"); exit(1); }
   fill(k, records, stride, count_value);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline(shape));
   printf("FB_DRAW k=%u tok=%s shape=%s input=0x%llx records=%u stride=%u count=0x%llx\n", k, tok, shape_names[shape],
          (unsigned long long)(args.va + 256 * k), records, stride,
          count_value >= 0 ? (unsigned long long)(args.va + 256 * k + 192) : 0ull);
   if (count_value >= 0)
      drawMeshCnt(cb, args.b, 256 * k, args.b, 256 * k + 192, records, stride);
   else
      drawMeshInd(cb, args.b, 256 * k, records, stride);
}

static void write_heap_descriptors(VkImageView view, VkSampler sampler)
{
   /* Heap: [0] storage buffer (sdata 0..4K), [1] storage buffer (sdata 4K..8K), [2] sampled image.
    * Sampler heap: [0] sampler. Graphics copy at FB_GFX_OFFSET, compute copy at FB_COMPUTE_OFFSET (db),
    * gsets / csets (legacy). */
   VkDescriptorBufferInfo sb[2] = {{sdata.b, 0, 4096}, {sdata.b, 4096, 4096}};
   VkDescriptorImageInfo ii = {VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
   VkDescriptorImageInfo si = {sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
   if (mode == MODE_DB) {
      VkDescriptorAddressInfoEXT ai[2] = {{VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, NULL, sdata.va, 4096},
                                          {VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, NULL, sdata.va + 4096, 4096}};
      VkPhysicalDeviceDescriptorBufferPropertiesEXT dbp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT};
      VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &dbp};
      vkGetPhysicalDeviceProperties2(pd, &p2);
      for (int copy = 0; copy < 2; copy++) {
         char *res = (char *)dbres.map + heap_off0 + (copy ? FB_COMPUTE_OFFSET : FB_GFX_OFFSET);
         char *smp = (char *)dbsmp.map + smp_off0 + (copy ? FB_COMPUTE_OFFSET : FB_GFX_OFFSET);
         for (int i = 0; i < 2; i++) {
            VkDescriptorGetInfoEXT gi = {VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT, NULL, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
            gi.data.pStorageBuffer = &ai[i];
            getDescriptor(dev, &gi, dbp.storageBufferDescriptorSize, res + i * heap_stride);
         }
         VkDescriptorGetInfoEXT gi = {VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT, NULL, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE};
         gi.data.pSampledImage = &ii;
         getDescriptor(dev, &gi, dbp.sampledImageDescriptorSize, res + 2 * heap_stride);
         VkDescriptorGetInfoEXT gs = {VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT, NULL, VK_DESCRIPTOR_TYPE_SAMPLER};
         gs.data.pSampler = &sampler;
         getDescriptor(dev, &gs, dbp.samplerDescriptorSize, smp);
      }
   } else {
      for (int copy = 0; copy < 2; copy++) {
         VkDescriptorSet *s = copy ? csets : gsets;
         VkWriteDescriptorSet w[3] = {
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, s[0], 0, 0, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, sb},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, s[0], 0, 2, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ii},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, s[1], 0, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER, &si}};
         vkUpdateDescriptorSets(dev, 3, w, 0, NULL);
      }
   }
}

int main(int argc, char **argv)
{
   if (argc < 3) { fprintf(stderr, "usage: %s legacy|legacydb|db script [submits]\n", argv[0]); return 2; }
   mode = !strcmp(argv[1], "db") ? MODE_DB : !strcmp(argv[1], "legacydb") ? MODE_LEGACYDB : MODE_LEGACY;
   const int submits = argc > 3 ? atoi(argv[3]) : 2;
   const char *pre = getenv("LD_PRELOAD");
   if (!pre || !strstr(pre, "drm_shim")) {
      printf("REFUSED: drm-shim only (hide /dev/dri and preload libamdgpu_noop_drm_shim.so)\n");
      return 9;
   }
   VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "fastbinding", 1, NULL, 0, VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
   VkInstance inst; CK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1; if (vkEnumeratePhysicalDevices(inst, &n, &pd) < 0 || !n) return 3;
   VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pd, &pp);
   if (!strstr(pp.deviceName, "GFX1013")) { printf("REFUSED: device %s is not the shim GFX1013\n", pp.deviceName); return 9; }

   /* Exposure report: the six features the BC250 hybrid Task mode used to hide, and the limit. */
   {
      uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, NULL);
      VkExtensionProperties *ep = calloc(ne, sizeof(*ep)); vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, ep);
      const char *names[] = {"VK_EXT_descriptor_buffer", "VK_EXT_descriptor_heap", "VK_EXT_device_generated_commands",
                             "VK_EXT_graphics_pipeline_library", "VK_EXT_shader_object", "VK_KHR_pipeline_binary"};
      printf("FB_EXTENSIONS");
      for (unsigned i = 0; i < 6; i++) {
         int found = 0;
         for (uint32_t j = 0; j < ne; j++) found |= !strcmp(ep[j].extensionName, names[i]);
         printf(" %s=%d", names[i] + 3, found);
      }
      free(ep);
      VkPhysicalDeviceDescriptorBufferFeaturesEXT dbf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT};
      VkPhysicalDeviceFeatures2 f2q = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &dbf};
      vkGetPhysicalDeviceFeatures2(pd, &f2q);
      printf(" descriptorBuffer=%d descriptorBufferPushDescriptors=%d descriptorBufferCaptureReplay=%d "
             "descriptorBufferImageLayoutIgnored=%d maxDrawIndirectCount=%u\n", dbf.descriptorBuffer,
             dbf.descriptorBufferPushDescriptors, dbf.descriptorBufferCaptureReplay,
             dbf.descriptorBufferImageLayoutIgnored, pp.limits.maxDrawIndirectCount);
      if (mode != MODE_LEGACY && !(dbf.descriptorBuffer && dbf.descriptorBufferPushDescriptors)) {
         printf("FB_DB_HIDDEN\nDONE\n");
         return 0;
      }
   }

   VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT mutf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT};
   mutf.mutableDescriptorType = VK_TRUE;
   VkPhysicalDeviceDescriptorBufferFeaturesEXT dbf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT, &mutf};
   dbf.descriptorBuffer = VK_TRUE; dbf.descriptorBufferPushDescriptors = VK_TRUE;
   VkPhysicalDeviceMeshShaderFeaturesEXT meshf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
                                                  mode != MODE_LEGACY ? (void *)&dbf : (void *)&mutf};
   {
      /* Task shaders only with hybrid Task (RADV_BC250_HYBRID_TASK=1). */
      VkPhysicalDeviceMeshShaderFeaturesEXT q = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
      VkPhysicalDeviceFeatures2 f2q = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &q};
      vkGetPhysicalDeviceFeatures2(pd, &f2q);
      meshf.meshShader = VK_TRUE; meshf.taskShader = q.taskShader;
   }
   VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &meshf};
   f12.drawIndirectCount = VK_TRUE; f12.bufferDeviceAddress = VK_TRUE; f12.runtimeDescriptorArray = VK_TRUE;
   f12.descriptorBindingVariableDescriptorCount = VK_TRUE; f12.descriptorBindingPartiallyBound = VK_TRUE;
   f12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE; f12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
   f12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE; f12.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
   f12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
   VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
   f13.dynamicRendering = VK_TRUE; f13.synchronization2 = VK_TRUE; f13.maintenance4 = VK_TRUE;
   VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
   const char *exts[] = {"VK_EXT_mesh_shader", "VK_KHR_push_descriptor", "VK_EXT_mutable_descriptor_type",
                         "VK_KHR_maintenance5", "VK_EXT_descriptor_buffer"};
   float prio = 1;
   VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
   VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, 1, &qci, 0, NULL, mode != MODE_LEGACY ? 5 : 4, exts, NULL};
   CK(vkCreateDevice(pd, &dci, NULL, &dev));
   drawMesh = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksEXT");
   drawMeshInd = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksIndirectEXT");
   drawMeshCnt = (void *)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksIndirectCountEXT");
   pushDesc = (void *)vkGetDeviceProcAddr(dev, "vkCmdPushDescriptorSetKHR");
   if (mode != MODE_LEGACY) {
      bindDB = (void *)vkGetDeviceProcAddr(dev, "vkCmdBindDescriptorBuffersEXT");
      setDBOffsets = (void *)vkGetDeviceProcAddr(dev, "vkCmdSetDescriptorBufferOffsetsEXT");
      bindEmbedded = (void *)vkGetDeviceProcAddr(dev, "vkCmdBindDescriptorBufferEmbeddedSamplersEXT");
      getLayoutSize = (void *)vkGetDeviceProcAddr(dev, "vkGetDescriptorSetLayoutSizeEXT");
      getBindingOffset = (void *)vkGetDeviceProcAddr(dev, "vkGetDescriptorSetLayoutBindingOffsetEXT");
      getDescriptor = (void *)vkGetDeviceProcAddr(dev, "vkGetDescriptorEXT");
   }
   VkQueue q; vkGetDeviceQueue(dev, 0, 0, &q);

   mkbuf(&args, 64 * 1024, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
   argmap = args.map;
   mkbuf(&sdata, 8192, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
   mkbuf(&ubo, 256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
   cbv = (VkDescriptorBufferInfo){ubo.b, 0, 256};
   printf("FB_ARGS_VA 0x%llx\n", (unsigned long long)args.va);

   VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, cf,
                           {256, 256, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL,
                           VK_IMAGE_LAYOUT_UNDEFINED};
   VkImage img[2]; VkImageView views[2];
   for (int i = 0; i < 2; i++) {
      CK(vkCreateImage(dev, &ii, NULL, &img[i]));
      VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, img[i], &mr);
      VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, memtype(mr.memoryTypeBits, 0)};
      VkDeviceMemory im; CK(vkAllocateMemory(dev, &ai, NULL, &im)); CK(vkBindImageMemory(dev, img[i], im, 0));
      VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, img[i], VK_IMAGE_VIEW_TYPE_2D,
                                  cf, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
      CK(vkCreateImageView(dev, &vi, NULL, &views[i]));
   }
   VkSamplerCreateInfo sci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
   sci.magFilter = sci.minFilter = VK_FILTER_LINEAR; sci.maxLod = 1;
   VkSampler heap_sampler, static_sampler;
   CK(vkCreateSampler(dev, &sci, NULL, &heap_sampler));
   sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
   CK(vkCreateSampler(dev, &sci, NULL, &static_sampler));

   /* Set layouts (vkd3d-proton root signature shape). */
   const int db = mode == MODE_DB;
   VkDescriptorType mut_types[2] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE};
   VkMutableDescriptorTypeListEXT mut_list = {2, mut_types};
   VkMutableDescriptorTypeCreateInfoEXT mut_ci = {VK_STRUCTURE_TYPE_MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT, NULL, 1, &mut_list};
   VkDescriptorBindingFlags heap_flags = db ? VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT :
      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
      VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
   VkDescriptorSetLayoutBindingFlagsCreateInfo heap_bf = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO, &mut_ci, 1, &heap_flags};
   VkDescriptorSetLayoutBindingFlagsCreateInfo smp_bf = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO, NULL, 1, &heap_flags};
   const VkDescriptorSetLayoutCreateFlags heap_lflags = db ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT :
      VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
   VkDescriptorSetLayoutBinding b_heap = {0, VK_DESCRIPTOR_TYPE_MUTABLE_EXT, FB_HEAP_COUNT, VK_SHADER_STAGE_ALL};
   VkDescriptorSetLayoutBinding b_smp = {0, VK_DESCRIPTOR_TYPE_SAMPLER, FB_SAMPLER_COUNT, VK_SHADER_STAGE_ALL};
   VkDescriptorSetLayoutBinding b_static = {0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_ALL, &static_sampler};
   VkDescriptorSetLayoutBinding b_cbv = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL};
   VkDescriptorSetLayoutCreateInfo lci[4] = {
      {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, &heap_bf, heap_lflags, 1, &b_heap},
      {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, &smp_bf, heap_lflags, 1, &b_smp},
      {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL,
       db ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT | VK_DESCRIPTOR_SET_LAYOUT_CREATE_EMBEDDED_IMMUTABLE_SAMPLERS_BIT_EXT : 0,
       1, &b_static},
      {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL,
       VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR | (db ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT : 0),
       1, &b_cbv}};
   VkDescriptorSetLayout sl[4];
   for (int i = 0; i < 4; i++)
      CK(vkCreateDescriptorSetLayout(dev, &lci[i], NULL, &sl[i]));
   VkPushConstantRange pcr = {VK_SHADER_STAGE_ALL, 0, 16};
   VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 4, sl, 1, &pcr};
   CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));

   /* Legacy descriptor pool and sets: allocated in every mode (same memory layout). The legacy sets
    * need legacy-compatible layouts, so db mode creates those separately. */
   {
      VkDescriptorSetLayout lsl[3] = {sl[0], sl[1], sl[2]};
      if (db) {
         VkDescriptorSetLayoutCreateInfo l2[3] = {lci[0], lci[1], lci[2]};
         static VkDescriptorBindingFlags lflags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
            VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
            VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
         static VkDescriptorSetLayoutBindingFlagsCreateInfo hb2, sb2;
         hb2 = heap_bf; hb2.pBindingFlags = &lflags; sb2 = smp_bf; sb2.pBindingFlags = &lflags;
         l2[0].pNext = &hb2; l2[1].pNext = &sb2;
         l2[0].flags = l2[1].flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
         l2[2].flags = 0;
         for (int i = 0; i < 3; i++)
            CK(vkCreateDescriptorSetLayout(dev, &l2[i], NULL, &lsl[i]));
      }
      VkDescriptorPoolSize ps[3] = {{VK_DESCRIPTOR_TYPE_MUTABLE_EXT, 2 * FB_HEAP_COUNT},
                                    {VK_DESCRIPTOR_TYPE_SAMPLER, 2 * FB_SAMPLER_COUNT + 2}};
      VkMutableDescriptorTypeCreateInfoEXT pmut = {VK_STRUCTURE_TYPE_MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT, NULL, 2, NULL};
      VkMutableDescriptorTypeListEXT plist[2] = {mut_list, {0, NULL}};
      pmut.pMutableDescriptorTypeLists = plist;
      VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, &pmut,
                                         VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT, 6, 2, ps};
      VkDescriptorPool pool; CK(vkCreateDescriptorPool(dev, &dpci, NULL, &pool));
      uint32_t counts[3] = {FB_HEAP_COUNT, FB_SAMPLER_COUNT, 1};
      VkDescriptorSetVariableDescriptorCountAllocateInfo vc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO, NULL, 3, counts};
      VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, &vc, pool, 3, lsl};
      CK(vkAllocateDescriptorSets(dev, &dsai, gsets));
      CK(vkAllocateDescriptorSets(dev, &dsai, csets));
   }
   /* Descriptor buffers: allocated in legacydb and db (same memory layout). */
   if (mode != MODE_LEGACY) {
      VkPhysicalDeviceDescriptorBufferPropertiesEXT dbp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT};
      VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &dbp};
      vkGetPhysicalDeviceProperties2(pd, &p2);
      mkbuf(&dbres, 2 * FB_COMPUTE_OFFSET, VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
            (dbp.bufferlessPushDescriptors ? 0 : VK_BUFFER_USAGE_PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER_BIT_EXT));
      mkbuf(&dbsmp, 2 * FB_COMPUTE_OFFSET, VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT);
      printf("FB_DB bufferless_push=%d offset_alignment=%llu\n", dbp.bufferlessPushDescriptors,
             (unsigned long long)dbp.descriptorBufferOffsetAlignment);
   }
   if (db) {
      /* Mutable stride: difference between two array elements' offsets is not queryable, use the
       * variable-count layout size for 1 and 2 elements (vkd3d uses the largest listed type). */
      VkDeviceSize s0, s1;
      getLayoutSize(dev, sl[0], &s0);
      getBindingOffset(dev, sl[0], 0, &heap_off0);
      getBindingOffset(dev, sl[1], 0, &smp_off0);
      getLayoutSize(dev, sl[1], &s1);
      heap_stride = (s0 - heap_off0) / FB_HEAP_COUNT;
      smp_stride = (s1 - smp_off0) / FB_SAMPLER_COUNT;
      printf("FB_DB_LAYOUT heap_size=%llu heap_stride=%llu sampler_size=%llu sampler_stride=%llu\n",
             (unsigned long long)s0, (unsigned long long)heap_stride, (unsigned long long)s1, (unsigned long long)smp_stride);
   }
   /* The addresses the set pointer user SGPRs must carry (low 32 bits). */
   if (db)
      printf("FB_SETVA gfx0=0x%llx gfx1=0x%llx cs0=0x%llx cs1=0x%llx\n", (unsigned long long)(dbres.va + FB_GFX_OFFSET),
             (unsigned long long)(dbsmp.va + FB_GFX_OFFSET), (unsigned long long)(dbres.va + FB_COMPUTE_OFFSET),
             (unsigned long long)(dbsmp.va + FB_COMPUTE_OFFSET));
   write_heap_descriptors(views[1], heap_sampler);

   {
      VkPipelineCreateFlags2CreateInfo flags2 = {VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO, NULL,
                                                 VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT};
      VkComputePipelineCreateInfo cpi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, db ? &flags2 : NULL, 0,
         {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, load("fb.comp.spv"), "main"}, pl};
      CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &cpipe));
   }

   VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, 0};
   VkCommandPool cpool; CK(vkCreateCommandPool(dev, &cpci, NULL, &cpool));
   VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cpool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
   VkCommandBuffer cmd; CK(vkAllocateCommandBuffers(dev, &cai, &cmd));
   VkRenderingAttachmentInfo att = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, views[0], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
   VkViewport vp = {0, 0, 256, 256, 0, 1};
   VkRect2D sc = {{0, 0}, {256, 256}};

   char *script = strdup(argv[2]);
   char *toks[4096]; unsigned ntok = 0;
   for (char *t = strtok(script, " "); t; t = strtok(NULL, " ")) {
      char *star = strchr(t, '*');
      unsigned rep = 1;
      if (star) { *star = 0; rep = atoi(star + 1); }
      for (unsigned r = 0; r < rep && ntok < 4096; r++) toks[ntok++] = t;
   }

   for (int s = 0; s < submits; s++) {
      CK(vkResetCommandPool(dev, cpool, 0));
      draw_k = 0; rnd_state = 12345;
      VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
      CK(vkBeginCommandBuffer(cmd, &bi));
      vkCmdSetViewport(cmd, 0, 1, &vp); vkCmdSetScissor(cmd, 0, 1, &sc);
      bind_all(cmd);
      for (unsigned i = 0; i < ntok; i++) {
         const char *t = toks[i];
         VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {256, 256}}, 1, 0, 1, &att};
         if (!strcmp(t, "B")) {
            vkCmdBeginRendering(cmd, &ri);
         } else if (!strcmp(t, "E")) {
            vkCmdEndRendering(cmd);
         } else if (!strcmp(t, "rb")) {
            bind_all(cmd);
         } else if (!strcmp(t, "m") || !strcmp(t, "sd") || !strcmp(t, "t") || !strcmp(t, "T")) {
            const int shape = t[0] == 'm' ? SH_SMALL : t[0] == 's' ? SH_NANITE : t[0] == 't' ? SH_TSMALL : SH_TNANITE;
            const uint32_t x = t[0] == 'm' ? 3 : 2;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline(shape));
            printf("FB_DRAW k=- tok=%s shape=%s direct=%u,1,1\n", t, shape_names[shape], x);
            drawMesh(cmd, x, 1, 1);
         } else if (!strcmp(t, "mi")) {
            indirect(cmd, t, SH_SMALL, 2, 12, -1);
         } else if (!strcmp(t, "si")) {
            indirect(cmd, t, SH_NANITE, 2, 12, -1);
         } else if (!strcmp(t, "sc")) {
            indirect(cmd, t, SH_NANITE, 4, 16, (int)(draw_k % 5));
         } else if (!strcmp(t, "ti")) {
            indirect(cmd, t, SH_TSMALL, 2, 12, -1);
         } else if (!strcmp(t, "tc")) {
            indirect(cmd, t, SH_TSMALL, 3, 12, 2);
         } else if (!strcmp(t, "v")) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline(SH_VS));
            printf("FB_DRAW k=- tok=v shape=vs\n");
            vkCmdDraw(cmd, 3, 1, 0, 0);
         } else if (!strcmp(t, "cd")) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cpipe);
            printf("FB_DRAW k=- tok=cd dispatch=7,3,1\n");
            vkCmdDispatch(cmd, 7, 3, 1);
         } else {
            printf("FAIL unknown token %s\n", t);
            return 1;
         }
      }
      CK(vkEndCommandBuffer(cmd));
      VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cmd};
      CK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
      CK(vkQueueWaitIdle(q));
      printf("SUBMIT_OK %d\n", s); fflush(stdout);
   }
   vkDestroyDevice(dev, NULL);
   printf("DONE\n");
   return 0;
}
