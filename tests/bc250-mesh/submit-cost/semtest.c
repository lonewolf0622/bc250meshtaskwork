/* semtest: semaphore semantics of vkQueueSubmit2 for the submit-cost switches (drm-shim ONLY).
 *
 * Runs with ioctlspy preloaded before the shim and IOCTLSPY_SYNCOBJ=1 (syncobj model: points
 * become available when a CS / host signal attaches them; the kernel's WAIT_FOR_SUBMIT is modelled),
 * IOCTLSPY_LOG=<file>. Every scenario is delimited by "MARK begin <name>" / "MARK end <name>" in the
 * log; every semaphore creation is preceded by "MARK sem <name>" (the next SYNCOBJ_CREATE is its
 * handle). checksem.py checks the kernel requests (CS wait / signal entries, probe ioctls, resets) and
 * the host-side results printed here (RESULT lines). Scenarios:
 *   xq         cross-queue timeline chain gfx -> compute 0 -> gfx (+ binary), waits honoured in order
 *   serial     vkd3d's serializing binary semaphore on one queue (wait + re-signal), 6 submissions
 *   consume    binary signalled on graphics, consumed on compute (reset after the wait)
 *   hostsig    vkSignalSemaphore then a queue wait on that value
 *   empty      zero-submit calls, fence-only call, command-buffer-less signal then a wait on it
 *   secondary  a primary that executes a secondary command buffer
 *   multi      one vkQueueSubmit2 with three VkSubmitInfo2 (signal / wait chains inside)
 *   sparse     (if a sparse-binding queue exists) sparse bind with timeline wait + signal, then a wait
 *   wbs_host   wait-before-signal on compute queue 1, the host signals 100 ms later
 *   wbs_queue  wait-before-signal on graphics, compute queue 0 signals 100 ms later from a thread
 * usage: semtest (no arguments) */
#define _GNU_SOURCE
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("FAIL %s = %d @%d\n", #x, r_, __LINE__); fflush(stdout); exit(1); } } while (0)

static void (*mark)(const char *);
static VkDevice dev;
static VkPhysicalDevice pd;
static VkQueue gq, cq[2], sq;
static uint32_t gfam, cfam, sfam = UINT32_MAX;
static VkBuffer buf;
static VkCommandPool gpool, cpool;

static void M(const char *fmt, const char *a)
{
   char b[256];
   snprintf(b, sizeof(b), fmt, a);
   if (mark)
      mark(b);
}

static uint64_t now_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static uint32_t memtype(uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
         return i;
   exit(2);
}

static VkSemaphore sem(const char *name, int timeline, uint64_t initial)
{
   M("sem %s", name);
   VkSemaphoreTypeCreateInfo stci = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, NULL,
                                     timeline ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY, initial};
   VkSemaphoreCreateInfo sci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &stci};
   VkSemaphore s;
   CK(vkCreateSemaphore(dev, &sci, NULL, &s));
   return s;
}

static VkCommandBuffer cmd(VkCommandPool pool, uint32_t value)
{
   VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool,
                                     VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
   VkCommandBuffer cb;
   CK(vkAllocateCommandBuffers(dev, &ai, &cb));
   VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CK(vkBeginCommandBuffer(cb, &bi));
   vkCmdFillBuffer(cb, buf, 0, 256, value);
   CK(vkEndCommandBuffer(cb));
   return cb;
}

#define SI(s, v) ((VkSemaphoreSubmitInfo){VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, NULL, (s), (v), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT})

static void submit(VkQueue q, VkCommandBuffer cb, uint32_t nw, const VkSemaphoreSubmitInfo *w, uint32_t ns,
                   const VkSemaphoreSubmitInfo *s)
{
   VkCommandBufferSubmitInfo ci = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, NULL, cb};
   VkSubmitInfo2 si = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2, NULL, 0, nw, w, cb ? 1 : 0, cb ? &ci : NULL, ns, s};
   CK(vkQueueSubmit2(q, 1, &si, VK_NULL_HANDLE));
}

static int host_value(VkSemaphore s, uint64_t v, uint64_t timeout_ns)
{
   VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, NULL, 0, 1, &s, &v};
   return vkWaitSemaphores(dev, &wi, timeout_ns) == VK_SUCCESS;
}

struct delayed {
   VkQueue q;
   VkSemaphore s;
   uint64_t v;
   int host;
   VkCommandBuffer cb;
};
static void *delayed_signal(void *arg)
{
   struct delayed *d = arg;
   usleep(100000);
   M("%s", d->host ? "delayed host signal" : "delayed queue signal");
   if (d->host) {
      VkSemaphoreSignalInfo si = {VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, NULL, d->s, d->v};
      CK(vkSignalSemaphore(dev, &si));
   } else {
      VkSemaphoreSubmitInfo s = SI(d->s, d->v);
      submit(d->q, d->cb, 0, NULL, 1, &s);
   }
   return NULL;
}

int main(void)
{
   const char *pre = getenv("LD_PRELOAD");
   if (!pre || !strstr(pre, "drm_shim")) {
      printf("REFUSED: drm-shim only (hide /dev/dri and preload libamdgpu_noop_drm_shim.so)\n");
      return 9;
   }
   mark = (void (*)(const char *))dlsym(RTLD_DEFAULT, "ioctlspy_mark");
   if (!mark) {
      printf("REFUSED: needs ioctlspy.so preloaded\n");
      return 9;
   }
   VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "semtest", 1, NULL, 0, VK_API_VERSION_1_3};
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
   gfam = cfam = UINT32_MAX;
   for (uint32_t i = 0; i < nf; i++) {
      if ((qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && gfam == UINT32_MAX)
         gfam = i;
      else if ((qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && cfam == UINT32_MAX)
         cfam = i;
      else if (qfp[i].queueFlags == VK_QUEUE_SPARSE_BINDING_BIT && sfam == UINT32_MAX)
         sfam = i;
   }
   if (gfam == UINT32_MAX || cfam == UINT32_MAX || qfp[cfam].queueCount < 2) {
      printf("FAIL queue families\n");
      return 1;
   }
   VkPhysicalDeviceFeatures2 f2q = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
   vkGetPhysicalDeviceFeatures2(pd, &f2q);
   const int sparse = sfam != UINT32_MAX && f2q.features.sparseBinding;
   printf("FAMILIES gfx=%u compute=%u sparse=%d\n", gfam, cfam, sparse ? (int)sfam : -1);

   VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
   f12.timelineSemaphore = VK_TRUE;
   VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
   f13.synchronization2 = VK_TRUE;
   VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
   f2.features.sparseBinding = sparse ? VK_TRUE : VK_FALSE;
   float prio[2] = {1, 1};
   VkDeviceQueueCreateInfo qci[3] = {{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, gfam, 1, prio},
                                     {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, cfam, 2, prio},
                                     {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, sfam, 1, prio}};
   VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, sparse ? 3 : 2, qci, 0, NULL, 0, NULL, NULL};
   CK(vkCreateDevice(pd, &dci, NULL, &dev));
   vkGetDeviceQueue(dev, gfam, 0, &gq);
   vkGetDeviceQueue(dev, cfam, 0, &cq[0]);
   vkGetDeviceQueue(dev, cfam, 1, &cq[1]);
   if (sparse)
      vkGetDeviceQueue(dev, sfam, 0, &sq);

   uint32_t fams[2] = {gfam, cfam};
   VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 65536, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             VK_SHARING_MODE_CONCURRENT, 2, fams};
   CK(vkCreateBuffer(dev, &bci, NULL, &buf));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, buf, &mr);
   VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, memtype(mr.memoryTypeBits, 0)};
   VkDeviceMemory mem;
   CK(vkAllocateMemory(dev, &mai, NULL, &mem));
   CK(vkBindBufferMemory(dev, buf, mem, 0));
   VkCommandPoolCreateInfo gpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, gfam};
   VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, cfam};
   CK(vkCreateCommandPool(dev, &gpci, NULL, &gpool));
   CK(vkCreateCommandPool(dev, &cpci, NULL, &cpool));

   /* ---- xq */
   {
      VkSemaphore T = sem("xq_T", 1, 0), U = sem("xq_U", 1, 0), B = sem("xq_B", 0, 0);
      M("begin %s", "xq");
      VkSemaphoreSubmitInfo s1[2] = {SI(T, 1), SI(B, 0)};
      submit(gq, cmd(gpool, 1), 0, NULL, 2, s1);
      VkSemaphoreSubmitInfo w2 = SI(T, 1), s2 = SI(U, 1);
      submit(cq[0], cmd(cpool, 2), 1, &w2, 1, &s2);
      VkSemaphoreSubmitInfo w3[2] = {SI(U, 1), SI(B, 0)}, s3 = SI(T, 2);
      submit(gq, cmd(gpool, 3), 2, w3, 1, &s3);
      VkSemaphoreSubmitInfo w4 = SI(T, 2), s4 = SI(U, 2);
      submit(cq[0], cmd(cpool, 4), 1, &w4, 1, &s4);
      printf("RESULT xq host_T2=%d host_U2=%d\n", host_value(T, 2, 1000000000ull), host_value(U, 2, 1000000000ull));
      M("end %s", "xq");
   }
   /* ---- serial */
   {
      VkSemaphore T = sem("serial_T", 1, 0), B = sem("serial_B", 0, 0);
      M("begin %s", "serial");
      for (int i = 0; i < 6; i++) {
         VkSemaphoreSubmitInfo w = SI(B, 0), s[2] = {SI(T, i + 1), SI(B, 0)};
         submit(gq, cmd(gpool, 10 + i), i ? 1 : 0, &w, 2, s);
      }
      printf("RESULT serial host_T6=%d\n", host_value(T, 6, 1000000000ull));
      M("end %s", "serial");
   }
   /* ---- consume */
   {
      VkSemaphore B = sem("consume_B", 0, 0), T = sem("consume_T", 1, 0);
      M("begin %s", "consume");
      for (int i = 0; i < 2; i++) {
         VkSemaphoreSubmitInfo s = SI(B, 0), w = SI(B, 0), t = SI(T, i + 1);
         submit(gq, cmd(gpool, 20 + i), 0, NULL, 1, &s);
         submit(cq[0], cmd(cpool, 30 + i), 1, &w, 1, &t);
      }
      printf("RESULT consume host_T2=%d\n", host_value(T, 2, 1000000000ull));
      M("end %s", "consume");
   }
   /* ---- hostsig */
   {
      VkSemaphore T = sem("hostsig_T", 1, 0), U = sem("hostsig_U", 1, 7);
      M("begin %s", "hostsig");
      VkSemaphoreSignalInfo si = {VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, NULL, T, 3};
      CK(vkSignalSemaphore(dev, &si));
      VkSemaphoreSubmitInfo w[2] = {SI(T, 3), SI(U, 7)}, s = SI(T, 4);
      submit(gq, cmd(gpool, 40), 2, w, 1, &s);
      printf("RESULT hostsig host_T4=%d\n", host_value(T, 4, 1000000000ull));
      M("end %s", "hostsig");
   }
   /* ---- empty */
   {
      VkSemaphore T = sem("empty_T", 1, 0);
      M("begin %s", "empty");
      CK(vkQueueSubmit2(gq, 0, NULL, VK_NULL_HANDLE));
      VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      VkFence f;
      CK(vkCreateFence(dev, &fci, NULL, &f));
      CK(vkQueueSubmit2(gq, 0, NULL, f));
      const int fence_ok = vkWaitForFences(dev, 1, &f, VK_TRUE, 1000000000ull) == VK_SUCCESS;
      VkSemaphoreSubmitInfo s = SI(T, 1);
      submit(cq[0], VK_NULL_HANDLE, 0, NULL, 1, &s); /* command-buffer-less signal */
      VkSemaphoreSubmitInfo w = SI(T, 1), s2 = SI(T, 2);
      submit(gq, cmd(gpool, 50), 1, &w, 1, &s2);
      printf("RESULT empty fence=%d host_T2=%d\n", fence_ok, host_value(T, 2, 1000000000ull));
      vkDestroyFence(dev, f, NULL);
      M("end %s", "empty");
   }
   /* ---- secondary */
   {
      VkSemaphore T = sem("secondary_T", 1, 0);
      M("begin %s", "secondary");
      VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, gpool,
                                        VK_COMMAND_BUFFER_LEVEL_SECONDARY, 1};
      VkCommandBuffer sec, pri;
      CK(vkAllocateCommandBuffers(dev, &ai, &sec));
      VkCommandBufferInheritanceInfo inh = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
      VkCommandBufferBeginInfo sbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, 0, &inh};
      CK(vkBeginCommandBuffer(sec, &sbi));
      vkCmdFillBuffer(sec, buf, 256, 256, 60);
      CK(vkEndCommandBuffer(sec));
      ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      CK(vkAllocateCommandBuffers(dev, &ai, &pri));
      VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      CK(vkBeginCommandBuffer(pri, &bi));
      vkCmdExecuteCommands(pri, 1, &sec);
      CK(vkEndCommandBuffer(pri));
      VkSemaphoreSubmitInfo s = SI(T, 1);
      submit(gq, pri, 0, NULL, 1, &s);
      printf("RESULT secondary host_T1=%d\n", host_value(T, 1, 1000000000ull));
      M("end %s", "secondary");
   }
   /* ---- multi */
   {
      VkSemaphore T = sem("multi_T", 1, 0);
      M("begin %s", "multi");
      VkCommandBufferSubmitInfo c[3] = {{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, NULL, cmd(gpool, 70)},
                                        {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, NULL, cmd(gpool, 71)},
                                        {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, NULL, cmd(gpool, 72)}};
      VkSemaphoreSubmitInfo s1 = SI(T, 1), w2 = SI(T, 1), s2 = SI(T, 2), s3 = SI(T, 3);
      VkSubmitInfo2 si[3] = {{VK_STRUCTURE_TYPE_SUBMIT_INFO_2, NULL, 0, 0, NULL, 1, &c[0], 1, &s1},
                             {VK_STRUCTURE_TYPE_SUBMIT_INFO_2, NULL, 0, 1, &w2, 1, &c[1], 1, &s2},
                             {VK_STRUCTURE_TYPE_SUBMIT_INFO_2, NULL, 0, 0, NULL, 1, &c[2], 1, &s3}};
      CK(vkQueueSubmit2(gq, 3, si, VK_NULL_HANDLE));
      printf("RESULT multi host_T3=%d\n", host_value(T, 3, 1000000000ull));
      M("end %s", "multi");
   }
   /* ---- sparse */
   if (sparse) {
      VkSemaphore T = sem("sparse_T", 1, 0);
      M("begin %s", "sparse");
      VkBufferCreateInfo sbci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, VK_BUFFER_CREATE_SPARSE_BINDING_BIT, 65536,
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
      VkBuffer sb;
      CK(vkCreateBuffer(dev, &sbci, NULL, &sb));
      VkMemoryRequirements smr;
      vkGetBufferMemoryRequirements(dev, sb, &smr);
      VkMemoryAllocateInfo smai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, smr.size, memtype(smr.memoryTypeBits, 0)};
      VkDeviceMemory smem;
      CK(vkAllocateMemory(dev, &smai, NULL, &smem));
      VkSparseMemoryBind bind = {0, smr.size, smem, 0, 0};
      VkSparseBufferMemoryBindInfo bbi = {sb, 1, &bind};
      VkSemaphoreSubmitInfo s0 = SI(T, 1);
      submit(gq, cmd(gpool, 80), 0, NULL, 1, &s0);
      uint64_t wv = 1, sv = 2;
      VkTimelineSemaphoreSubmitInfo tsi = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, NULL, 1, &wv, 1, &sv};
      VkBindSparseInfo bsi = {VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, &tsi, 1, &T, 1, &bbi, 0, NULL, 0, NULL, 1, &T};
      CK(vkQueueBindSparse(sq, 1, &bsi, VK_NULL_HANDLE));
      CK(vkQueueWaitIdle(sq));
      VkSemaphoreSubmitInfo w = SI(T, 2), s = SI(T, 3);
      submit(gq, cmd(gpool, 81), 1, &w, 1, &s);
      printf("RESULT sparse host_T3=%d\n", host_value(T, 3, 1000000000ull));
      M("end %s", "sparse");
   } else {
      printf("RESULT sparse skipped=1\n");
   }
   /* ---- wbs_host: compute queue 1 waits on a value the host signals 100 ms later */
   {
      VkSemaphore T = sem("wbs_host_T", 1, 0), U = sem("wbs_host_U", 1, 0);
      M("begin %s", "wbs_host");
      struct delayed d = {.s = T, .v = 5, .host = 1};
      pthread_t th;
      pthread_create(&th, NULL, delayed_signal, &d);
      const uint64_t t0 = now_ns();
      VkSemaphoreSubmitInfo w = SI(T, 5), s = SI(U, 1);
      submit(cq[1], cmd(cpool, 90), 1, &w, 1, &s);
      const uint64_t t1 = now_ns();
      pthread_join(th, NULL);
      const int ok = host_value(U, 1, 2000000000ull);
      printf("RESULT wbs_host submit_ms=%.1f host_U1=%d\n", (t1 - t0) / 1e6, ok);
      M("end %s", "wbs_host");
   }
   /* ---- wbs_queue: graphics waits on a value compute queue 0 signals 100 ms later */
   {
      VkSemaphore T = sem("wbs_queue_T", 1, 0), U = sem("wbs_queue_U", 1, 0);
      M("begin %s", "wbs_queue");
      struct delayed d = {.q = cq[0], .s = T, .v = 1, .host = 0, .cb = cmd(cpool, 100)};
      pthread_t th;
      pthread_create(&th, NULL, delayed_signal, &d);
      const uint64_t t0 = now_ns();
      VkSemaphoreSubmitInfo w = SI(T, 1), s = SI(U, 1);
      submit(gq, cmd(gpool, 101), 1, &w, 1, &s);
      const uint64_t t1 = now_ns();
      pthread_join(th, NULL);
      const int ok = host_value(U, 1, 2000000000ull);
      /* after the switch to the submit thread: more submissions keep their order */
      VkSemaphoreSubmitInfo w2 = SI(U, 1), s2 = SI(U, 2);
      submit(gq, cmd(gpool, 102), 1, &w2, 1, &s2);
      const int ok2 = host_value(U, 2, 2000000000ull);
      printf("RESULT wbs_queue submit_ms=%.1f host_U1=%d host_U2=%d\n", (t1 - t0) / 1e6, ok, ok2);
      M("end %s", "wbs_queue");
   }
   CK(vkDeviceWaitIdle(dev));
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   printf("DONE\n");
   return 0;
}
