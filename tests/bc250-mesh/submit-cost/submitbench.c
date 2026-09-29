/* submitbench: CPU cost of vkQueueSubmit2 with vkd3d-proton-like submissions (drm-shim ONLY).
 *
 * usage: submitbench <bench.comp.spv> [key=value ...]
 *   frames=300 warmup=30 gfx=48 ace=10 cbs=4 acecbs=4 dispatches=8 pipelines=16 querypools=4
 *   mem=500 exportable=0 threads=0 onetime=0 xq=1 cbless=1 bigevery=0 bigdispatches=3000 json=<file>
 *
 * Per frame (vkd3d-proton's pattern on RADV, which has no barrier command buffer):
 *   graphics queue: `gfx` submissions of `cbs` command buffers; each waits on the queue's binary
 *     "serializing" semaphore (signalled by the previous submission of the queue) and signals it
 *     again plus the queue's timeline (value + 1);
 *   compute queue (family 1 queue 0): `ace` submissions of `acecbs` command buffers, same pattern;
 *   xq=1: each compute submission also waits on the graphics timeline value submitted just before it,
 *     and the next graphics submission waits on the compute timeline (cross-queue D3D12 fences);
 *   cbless=1: one command-buffer-less graphics submission per frame that only signals the timeline
 *     (vkd3d's fence-only submissions);
 *   the host waits for the graphics timeline value of two frames ago, then resets and re-records that
 *     frame slot's command pools (no ONE_TIME_SUBMIT unless onetime=1, like vkd3d).
 * Each command buffer: `dispatches` compute dispatches cycling through `pipelines` distinct pipelines
 * (buffer device address in push constants), a timestamp into each of `querypools` query pools, a
 * global memory barrier (bigevery=k: every k-th graphics command buffer records `bigdispatches`
 * dispatches instead, so its command stream grows past the initial IB, like large D3D12 command lists). `mem` memory objects with device address are allocated and touched through
 * the dispatches (vkd3d's heaps); `exportable` more are exportable (not VM-local: RADV keeps them in its
 * global BO list, like WSI images).
 * threads=1: the graphics and compute submissions come from two threads (vkd3d's queue workers) with
 *   the cross-queue order kept by a host-side handshake.
 * Only the vkQueueSubmit2 calls are timed. With ioctlspy preloaded, ioctls per submission are counted
 * (the shim's time inside ioctl is reported separately and is ~0). */
#define _GNU_SOURCE
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("FAIL %s = %d @%d\n", #x, r_, __LINE__); fflush(stdout); exit(1); } } while (0)
#define SLOTS 3
#define MAXCB 4096

struct ioctlspy_counts {
   uint64_t total, drm, cs, syncobj_wait, syncobj_timeline_wait, syncobj_reset, syncobj_signal,
      syncobj_timeline_signal, syncobj_transfer, syncobj_query, syncobj_create, syncobj_destroy,
      syncobj_fd, bo_list, gem_create, gem_va, gem_other, other_drm;
   uint64_t cs_ib_chunks, cs_bo_handles, cs_wait_entries, cs_signal_entries, cs_with_bo_list_handle;
   uint64_t ns_in_next;
};
static void (*spy_snapshot)(struct ioctlspy_counts *);

static int opt(int argc, char **argv, const char *k, int def)
{
   size_t n = strlen(k);
   for (int i = 2; i < argc; i++)
      if (!strncmp(argv[i], k, n) && argv[i][n] == '=')
         return atoi(argv[i] + n + 1);
   return def;
}
static const char *opts(int argc, char **argv, const char *k)
{
   size_t n = strlen(k);
   for (int i = 2; i < argc; i++)
      if (!strncmp(argv[i], k, n) && argv[i][n] == '=')
         return argv[i] + n + 1;
   return NULL;
}

static uint64_t now_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static VkDevice dev;
static VkPhysicalDevice pd;
static uint32_t memtype(uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
         return i;
   exit(2);
}

/* per-queue submission statistics */
struct qstat {
   const char *name;
   uint64_t *ns;
   size_t n, cap;
   struct ioctlspy_counts io;
};
static void rec(struct qstat *q, uint64_t ns)
{
   if (q->n == q->cap) {
      q->cap = q->cap ? q->cap * 2 : 4096;
      q->ns = realloc(q->ns, q->cap * sizeof(*q->ns));
   }
   q->ns[q->n++] = ns;
}
static int cmpu64(const void *a, const void *b)
{
   uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
   return x < y ? -1 : x > y;
}

struct queue_ctx {
   VkQueue q;
   uint32_t family;
   VkCommandPool pool[SLOTS];
   VkCommandBuffer *cb[SLOTS];
   uint32_t ncb;
   VkSemaphore timeline, serial;
   uint64_t value;
   int serial_signaled;
   struct qstat st;
};

static int frames, warmup, n_gfx, n_ace, cbs, acecbs, dispatches, npipe, nqp, xq, cbless, onetime, threads;
static int bigevery, bigdispatches;
static VkPipeline *pipes;
static VkPipelineLayout layout;
static VkQueryPool *qpools;
static uint64_t record_ns; /* graphics reset + recording (IB growth / BO churn shows here) */
static VkDeviceAddress *addrs;
static int nmem;
static struct queue_ctx G, A;
static VkSemaphore frame_tl; /* == G.timeline */

static void record(struct queue_ctx *qc, int slot, int base)
{
   VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL,
                                  onetime ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT : 0};
   for (uint32_t c = 0; c < qc->ncb; c++) {
      VkCommandBuffer cb = qc->cb[slot][c];
      CK(vkBeginCommandBuffer(cb, &bi));
      for (int q = 0; q < nqp; q++)
         vkCmdResetQueryPool(cb, qpools[q], (c % 32) * 2, 2);
      const int nd = (bigevery && qc == &G && c % bigevery == 0) ? bigdispatches : dispatches;
      for (int d = 0; d < nd; d++) {
         const int k = base + c * dispatches + d;
         vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[k % npipe]);
         struct { VkDeviceAddress a; uint32_t v; } pc = {addrs[k % nmem], (uint32_t)k};
         vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &pc);
         vkCmdDispatch(cb, 1, 1, 1);
      }
      for (int q = 0; q < nqp; q++)
         vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, qpools[q], (c % 32) * 2);
      VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                             VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                             VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
      VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 1, &mb};
      vkCmdPipelineBarrier2(cb, &di);
      CK(vkEndCommandBuffer(cb));
   }
}

/* one vkd3d-like submission of command buffers [first, first+count) of the slot */
static void submit(struct queue_ctx *qc, int slot, uint32_t first, uint32_t count, VkSemaphore xsem, uint64_t xval,
                   int measure)
{
   VkCommandBufferSubmitInfo cbi[64];
   for (uint32_t i = 0; i < count; i++)
      cbi[i] = (VkCommandBufferSubmitInfo){VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, NULL, qc->cb[slot][first + i]};
   VkSemaphoreSubmitInfo waits[2], sigs[2];
   uint32_t nw = 0;
   if (qc->serial_signaled)
      waits[nw++] = (VkSemaphoreSubmitInfo){VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, NULL, qc->serial, 0,
                                            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
   if (xsem && xval)
      waits[nw++] = (VkSemaphoreSubmitInfo){VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, NULL, xsem, xval,
                                            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
   sigs[0] = (VkSemaphoreSubmitInfo){VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, NULL, qc->timeline, ++qc->value,
                                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
   sigs[1] = (VkSemaphoreSubmitInfo){VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, NULL, qc->serial, 0,
                                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
   VkSubmitInfo2 si = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2, NULL, 0, nw, waits, count, cbi, 2, sigs};
   struct ioctlspy_counts a, b;
   if (measure && spy_snapshot)
      spy_snapshot(&a);
   const uint64_t t0 = now_ns();
   CK(vkQueueSubmit2(qc->q, 1, &si, VK_NULL_HANDLE));
   const uint64_t t1 = now_ns();
   qc->serial_signaled = 1;
   if (measure) {
      if (spy_snapshot) {
         spy_snapshot(&b);
         uint64_t *pa = (uint64_t *)&a, *pb = (uint64_t *)&b, *po = (uint64_t *)&qc->st.io;
         for (size_t i = 0; i < sizeof(a) / 8; i++)
            po[i] += pb[i] - pa[i];
      }
      rec(&qc->st, t1 - t0);
   }
}

static void submit_cbless(struct queue_ctx *qc, int measure)
{
   VkSemaphoreSubmitInfo sig = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, NULL, qc->timeline, ++qc->value,
                                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
   VkSubmitInfo2 si = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2, NULL, 0, 0, NULL, 0, NULL, 1, &sig};
   (void)measure;
   CK(vkQueueSubmit2(qc->q, 1, &si, VK_NULL_HANDLE));
}

/* threads=1 handshake: the compute thread submits its k-th submission after the graphics thread
 * published the graphics timeline value it must wait for; the graphics thread waits (host) for the
 * compute submission before submitting work that waits on it (vkd3d's ordering). */
static pthread_mutex_t hs_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t hs_cond = PTHREAD_COND_INITIALIZER;
static uint64_t hs_gfx_value[1 << 16], hs_ace_value[1 << 16];
static int hs_gfx_posted, hs_ace_done;

static int gfx_interval(void) { return n_ace ? n_gfx / n_ace : n_gfx + 1; }

static void *ace_thread(void *arg)
{
   (void)arg;
   int k = 0;
   for (int f = 0; f < frames + warmup; f++) {
      const int slot = f % SLOTS;
      for (int j = 0; j < n_ace; j++, k++) {
         pthread_mutex_lock(&hs_mtx);
         while (hs_gfx_posted <= k)
            pthread_cond_wait(&hs_cond, &hs_mtx);
         const uint64_t xv = hs_gfx_value[k & 0xffff];
         pthread_mutex_unlock(&hs_mtx);
         submit(&A, slot, j * acecbs, acecbs, xq ? G.timeline : VK_NULL_HANDLE, xv, f >= warmup);
         pthread_mutex_lock(&hs_mtx);
         hs_ace_value[k & 0xffff] = A.value;
         hs_ace_done = k + 1;
         pthread_cond_broadcast(&hs_cond);
         pthread_mutex_unlock(&hs_mtx);
      }
   }
   return NULL;
}

static void print_stats(struct qstat *q, int nframes, FILE *json)
{
   if (!q->n)
      return;
   qsort(q->ns, q->n, sizeof(uint64_t), cmpu64);
   double sum = 0;
   for (size_t i = 0; i < q->n; i++)
      sum += q->ns[i];
   const double n = q->n;
   const struct ioctlspy_counts *io = &q->io;
   printf("QUEUE %s submits=%zu per_frame=%.1f mean_us=%.2f median_us=%.2f p10_us=%.2f p90_us=%.2f"
          " shim_ioctl_us=%.2f ioctls=%.2f cs=%.2f probe_or_wait=%.2f reset=%.2f other_syncobj=%.2f"
          " bo_handles_per_cs=%.1f ib_chunks_per_cs=%.2f wait_entries_per_cs=%.2f signal_entries_per_cs=%.2f\n",
          q->name, q->n, n / nframes, sum / n / 1e3, q->ns[q->n / 2] / 1e3, q->ns[q->n / 10] / 1e3,
          q->ns[q->n * 9 / 10] / 1e3, io->ns_in_next / n / 1e3, io->drm / n, io->cs / n,
          (io->syncobj_timeline_wait + io->syncobj_wait) / n, io->syncobj_reset / n,
          (io->syncobj_signal + io->syncobj_timeline_signal + io->syncobj_transfer + io->syncobj_query + io->syncobj_fd) / n,
          io->cs ? (double)io->cs_bo_handles / io->cs : 0, io->cs ? (double)io->cs_ib_chunks / io->cs : 0,
          io->cs ? (double)io->cs_wait_entries / io->cs : 0, io->cs ? (double)io->cs_signal_entries / io->cs : 0);
   if (json)
      fprintf(json, "\"%s\": {\"submits\": %zu, \"mean_us\": %.3f, \"median_us\": %.3f, \"p90_us\": %.3f, "
              "\"ioctls\": %.3f, \"cs\": %.3f, \"probe_or_wait\": %.3f, \"reset\": %.3f, \"bo_handles_per_cs\": %.2f, "
              "\"shim_ioctl_us\": %.3f},\n",
              q->name, q->n, sum / n / 1e3, q->ns[q->n / 2] / 1e3, q->ns[q->n * 9 / 10] / 1e3, io->drm / n, io->cs / n,
              (io->syncobj_timeline_wait + io->syncobj_wait) / n, io->syncobj_reset / n,
              io->cs ? (double)io->cs_bo_handles / io->cs : 0, io->ns_in_next / n / 1e3);
}

int main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "usage: %s bench.comp.spv [key=value...]\n", argv[0]);
      return 2;
   }
   const char *pre = getenv("LD_PRELOAD");
   if (!pre || !strstr(pre, "drm_shim")) {
      printf("REFUSED: drm-shim only (hide /dev/dri and preload libamdgpu_noop_drm_shim.so)\n");
      return 9;
   }
   spy_snapshot = (void (*)(struct ioctlspy_counts *))dlsym(RTLD_DEFAULT, "ioctlspy_snapshot");
   frames = opt(argc, argv, "frames", 300);
   warmup = opt(argc, argv, "warmup", 30);
   n_gfx = opt(argc, argv, "gfx", 48);
   n_ace = opt(argc, argv, "ace", 10);
   cbs = opt(argc, argv, "cbs", 4);
   acecbs = opt(argc, argv, "acecbs", 4);
   dispatches = opt(argc, argv, "dispatches", 8);
   npipe = opt(argc, argv, "pipelines", 16);
   nqp = opt(argc, argv, "querypools", 4);
   nmem = opt(argc, argv, "mem", 500);
   const int nexp = opt(argc, argv, "exportable", 0);
   threads = opt(argc, argv, "threads", 0);
   onetime = opt(argc, argv, "onetime", 0);
   xq = opt(argc, argv, "xq", 1);
   cbless = opt(argc, argv, "cbless", 1);
   bigevery = opt(argc, argv, "bigevery", 0);
   bigdispatches = opt(argc, argv, "bigdispatches", 3000);
   if (nmem < 1)
      nmem = 1;
   if (npipe < 1)
      npipe = 1;

   VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "submitbench", 1, NULL, 0, VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
   VkInstance inst;
   CK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1;
   if (vkEnumeratePhysicalDevices(inst, &n, &pd) < 0 || !n)
      return 3;
   VkPhysicalDeviceProperties pp;
   vkGetPhysicalDeviceProperties(pd, &pp);
   if (!strstr(pp.deviceName, "GFX1013")) {
      printf("REFUSED: device %s is not the shim GFX1013\n", pp.deviceName);
      return 9;
   }
   uint32_t nf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nf, NULL);
   VkQueueFamilyProperties qfp[8];
   if (nf > 8)
      nf = 8;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nf, qfp);
   uint32_t gfam = UINT32_MAX, cfam = UINT32_MAX;
   for (uint32_t i = 0; i < nf; i++) {
      if ((qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && gfam == UINT32_MAX)
         gfam = i;
      if ((qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && cfam == UINT32_MAX)
         cfam = i;
   }
   if (gfam == UINT32_MAX || cfam == UINT32_MAX) {
      printf("FAIL no compute family\n");
      return 1;
   }

   VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
   f12.timelineSemaphore = VK_TRUE;
   f12.bufferDeviceAddress = VK_TRUE;
   f12.hostQueryReset = VK_TRUE;
   VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
   f13.synchronization2 = VK_TRUE;
   VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
   float prio[1] = {1};
   VkDeviceQueueCreateInfo qci[2] = {{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, gfam, 1, prio},
                                     {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, cfam, 1, prio}};
   const char *dext[] = {"VK_KHR_external_memory_fd"};
   VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, 2, qci, 0, NULL, nexp ? 1 : 0, dext, NULL};
   CK(vkCreateDevice(pd, &dci, NULL, &dev));
   vkGetDeviceQueue(dev, gfam, 0, &G.q);
   vkGetDeviceQueue(dev, cfam, 0, &A.q);
   G.family = gfam;
   A.family = cfam;
   G.st.name = "gfx";
   A.st.name = "ace";

   /* memory objects with device addresses (vkd3d heaps) */
   addrs = calloc(nmem, sizeof(*addrs));
   for (int i = 0; i < nmem + nexp; i++) {
      VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 65536,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                VK_SHARING_MODE_EXCLUSIVE};
      VkBuffer buf;
      CK(vkCreateBuffer(dev, &bci, NULL, &buf));
      VkMemoryRequirements mr;
      vkGetBufferMemoryRequirements(dev, buf, &mr);
      VkExportMemoryAllocateInfo emi = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, NULL,
                                        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
      VkMemoryAllocateFlagsInfo mafi = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, i >= nmem ? &emi : NULL,
                                        VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
      VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &mafi, mr.size, memtype(mr.memoryTypeBits, 0)};
      VkDeviceMemory mem;
      CK(vkAllocateMemory(dev, &mai, NULL, &mem));
      CK(vkBindBufferMemory(dev, buf, mem, 0));
      if (i < nmem) {
         VkBufferDeviceAddressInfo bdai = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, buf};
         addrs[i] = vkGetBufferDeviceAddress(dev, &bdai);
      }
   }

   /* pipelines */
   FILE *f = fopen(argv[1], "rb");
   if (!f) {
      perror(argv[1]);
      return 2;
   }
   fseek(f, 0, SEEK_END);
   long sz = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint32_t *code = malloc(sz);
   if (fread(code, 1, sz, f) != (size_t)sz)
      return 2;
   fclose(f);
   VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sz, code};
   VkShaderModule sm;
   CK(vkCreateShaderModule(dev, &smci, NULL, &sm));
   VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
   VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 0, NULL, 1, &pcr};
   CK(vkCreatePipelineLayout(dev, &plci, NULL, &layout));
   pipes = calloc(npipe, sizeof(*pipes));
   for (int i = 0; i < npipe; i++) {
      uint32_t val = i + 1;
      VkSpecializationMapEntry me = {0, 0, 4};
      VkSpecializationInfo spi = {1, &me, 4, &val};
      VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
                                          {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                                           VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", &spi}, layout};
      CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipes[i]));
   }
   qpools = calloc(nqp ? nqp : 1, sizeof(*qpools));
   for (int i = 0; i < nqp; i++) {
      VkQueryPoolCreateInfo qpci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 64};
      CK(vkCreateQueryPool(dev, &qpci, NULL, &qpools[i]));
   }

   /* semaphores */
   VkSemaphoreTypeCreateInfo stci = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, NULL, VK_SEMAPHORE_TYPE_TIMELINE, 0};
   VkSemaphoreCreateInfo sci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &stci};
   VkSemaphoreCreateInfo bsci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
   CK(vkCreateSemaphore(dev, &sci, NULL, &G.timeline));
   CK(vkCreateSemaphore(dev, &sci, NULL, &A.timeline));
   CK(vkCreateSemaphore(dev, &bsci, NULL, &G.serial));
   CK(vkCreateSemaphore(dev, &bsci, NULL, &A.serial));
   frame_tl = G.timeline;

   /* command buffers per frame slot */
   struct queue_ctx *qs[2] = {&G, &A};
   G.ncb = n_gfx * cbs;
   A.ncb = n_ace * acecbs;
   if (G.ncb > MAXCB || A.ncb > MAXCB || cbs > 64 || acecbs > 64) {
      printf("FAIL too many command buffers\n");
      return 2;
   }
   for (int k = 0; k < 2; k++) {
      struct queue_ctx *qc = qs[k];
      for (int s = 0; s < SLOTS; s++) {
         VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, qc->family};
         CK(vkCreateCommandPool(dev, &cpci, NULL, &qc->pool[s]));
         qc->cb[s] = calloc(qc->ncb ? qc->ncb : 1, sizeof(VkCommandBuffer));
         if (qc->ncb) {
            VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, qc->pool[s],
                                               VK_COMMAND_BUFFER_LEVEL_PRIMARY, qc->ncb};
            CK(vkAllocateCommandBuffers(dev, &cai, qc->cb[s]));
         }
      }
   }

   pthread_t th;
   if (threads)
      pthread_create(&th, NULL, ace_thread, NULL);
   uint64_t frame_end_value[SLOTS] = {0};
   const uint64_t t_start = now_ns();
   int k_ace = 0;
   for (int fr = 0; fr < frames + warmup; fr++) {
      const int slot = fr % SLOTS;
      const int measure = fr >= warmup;
      /* host wait for the frame that used this slot (two frames ago) */
      if (frame_end_value[slot]) {
         VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, NULL, 0, 1, &frame_tl, &frame_end_value[slot]};
         CK(vkWaitSemaphores(dev, &wi, UINT64_MAX));
      }
      if (threads) {
         /* the compute thread may still use the slot of three frames ago: it finished them (handshake) */
      }
      const uint64_t rec_t0 = now_ns();
      CK(vkResetCommandPool(dev, G.pool[slot], 0));
      record(&G, slot, fr * 7);
      if (measure)
         record_ns += now_ns() - rec_t0;
      if (!threads) {
         CK(vkResetCommandPool(dev, A.pool[slot], 0));
         record(&A, slot, fr * 5);
      } else {
         /* compute: wait until the compute thread finished the submissions of three frames ago */
         pthread_mutex_lock(&hs_mtx);
         while (hs_ace_done < (fr - SLOTS + 1) * n_ace)
            pthread_cond_wait(&hs_cond, &hs_mtx);
         pthread_mutex_unlock(&hs_mtx);
         CK(vkResetCommandPool(dev, A.pool[slot], 0));
         record(&A, slot, fr * 5);
      }
      const int gi = gfx_interval();
      int j_ace = 0;
      uint64_t pending_ace_wait = 0;
      for (int i = 0; i < n_gfx; i++) {
         submit(&G, slot, i * cbs, cbs, xq ? A.timeline : VK_NULL_HANDLE, pending_ace_wait, measure);
         pending_ace_wait = 0;
         if (n_ace && j_ace < n_ace && (i + 1) % gi == 0) {
            if (!threads) {
               submit(&A, slot, j_ace * acecbs, acecbs, xq ? G.timeline : VK_NULL_HANDLE, G.value, measure);
               pending_ace_wait = A.value;
            } else {
               pthread_mutex_lock(&hs_mtx);
               hs_gfx_value[k_ace & 0xffff] = G.value;
               hs_gfx_posted = k_ace + 1;
               pthread_cond_broadcast(&hs_cond);
               while (hs_ace_done <= k_ace)
                  pthread_cond_wait(&hs_cond, &hs_mtx);
               pending_ace_wait = hs_ace_value[k_ace & 0xffff];
               pthread_mutex_unlock(&hs_mtx);
               k_ace++;
            }
            j_ace++;
         }
      }
      while (j_ace < n_ace) { /* n_gfx not a multiple: finish the frame's compute submissions */
         if (!threads) {
            submit(&A, slot, j_ace * acecbs, acecbs, xq ? G.timeline : VK_NULL_HANDLE, G.value, measure);
         } else {
            pthread_mutex_lock(&hs_mtx);
            hs_gfx_value[k_ace & 0xffff] = G.value;
            hs_gfx_posted = k_ace + 1;
            pthread_cond_broadcast(&hs_cond);
            while (hs_ace_done <= k_ace)
               pthread_cond_wait(&hs_cond, &hs_mtx);
            pthread_mutex_unlock(&hs_mtx);
            k_ace++;
         }
         j_ace++;
      }
      if (cbless)
         submit_cbless(&G, measure);
      frame_end_value[slot] = G.value;
   }
   if (threads)
      pthread_join(th, NULL);
   CK(vkDeviceWaitIdle(dev));
   const uint64_t t_end = now_ns();

   const char *jpath = opts(argc, argv, "json");
   FILE *json = jpath ? fopen(jpath, "w") : NULL;
   if (json)
      fputs("{\n", json);
   printf("CONFIG frames=%d gfx=%d ace=%d cbs=%d acecbs=%d dispatches=%d pipelines=%d querypools=%d mem=%d exportable=%d"
          " threads=%d onetime=%d xq=%d cbless=%d bigevery=%d bigdispatches=%d spy=%d wall_ms=%.1f record_ms_per_frame=%.3f\n",
          frames, n_gfx, n_ace, cbs, acecbs, dispatches, npipe, nqp, nmem, nexp, threads, onetime, xq, cbless, bigevery,
          bigdispatches, spy_snapshot != NULL, (t_end - t_start) / 1e6, record_ns / 1e6 / frames);
   print_stats(&G.st, frames, json);
   print_stats(&A.st, frames, json);
   if (json) {
      fprintf(json, "\"frames\": %d\n}\n", frames);
      fclose(json);
   }
   /* tear down (the driver prints its final profile / timer lines at device destruction) */
   for (int k = 0; k < 2; k++)
      for (int s = 0; s < SLOTS; s++)
         vkDestroyCommandPool(dev, qs[k]->pool[s], NULL);
   vkDestroySemaphore(dev, G.timeline, NULL);
   vkDestroySemaphore(dev, A.timeline, NULL);
   vkDestroySemaphore(dev, G.serial, NULL);
   vkDestroySemaphore(dev, A.serial, NULL);
   for (int i = 0; i < nqp; i++)
      vkDestroyQueryPool(dev, qpools[i], NULL);
   for (int i = 0; i < npipe; i++)
      vkDestroyPipeline(dev, pipes[i], NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyShaderModule(dev, sm, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   printf("DONE\n");
   return 0;
}
