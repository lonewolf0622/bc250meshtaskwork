/* ioctlspy: LD_PRELOAD interposer for the submit-cost suite (drm-shim only).
 *
 * Preload it BEFORE the noop drm-shim (LD_PRELOAD="ioctlspy.so libamdgpu_noop_drm_shim.so"): every
 * ioctl() of the process passes through here, is counted per DRM command, timed, and forwarded to
 * the next definition (the shim, which answers DRM ioctls on its fake render node).
 *
 * - ioctlspy_snapshot(): counters for the microbenchmark (dlsym'd, optional).
 * - IOCTLSPY_LOG=<file>: one line per AMDGPU_CS (IB count, BO handles, syncobj waits / signals with
 *   handle, point and flags) and per syncobj ioctl, plus SYNCOBJ_CREATE results, so a checker can
 *   verify what the driver asked the kernel to wait for and signal.
 * - ioctlspy_mark(const char *): writes a marker line into the log (dlsym'd by the tests).
 * - The shim gives every syncobj handle 1; ioctlspy hands out unique handles (1000, 1001, ...).
 * - IOCTLSPY_SYNCOBJ=1: a small syncobj model on top of the shim's stubs, so the Vulkan runtime's
 *   wait-before-signal logic sees real answers: a point is "available" once a CS / signal / transfer
 *   has attached it (binary: point 0, cleared by RESET), and complete at the same moment (the noop GPU
 *   finishes instantly). SYNCOBJ_TIMELINE_WAIT / SYNCOBJ_WAIT block (condition variable) until the
 *   points are available / the absolute timeout passes (-ETIME), and a CS whose wait chunk names a
 *   point that is not available yet blocks like the kernel's WAIT_FOR_SUBMIT (5 s, then -ETIME). */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>

#include "drm-uapi/drm.h"
#include "drm-uapi/amdgpu_drm.h"

struct ioctlspy_counts {
   uint64_t total, drm, cs, syncobj_wait, syncobj_timeline_wait, syncobj_reset, syncobj_signal,
      syncobj_timeline_signal, syncobj_transfer, syncobj_query, syncobj_create, syncobj_destroy,
      syncobj_fd, bo_list, gem_create, gem_va, gem_other, other_drm;
   uint64_t cs_ib_chunks, cs_bo_handles, cs_wait_entries, cs_signal_entries, cs_with_bo_list_handle;
   uint64_t ns_in_next; /* time spent in the next ioctl (the shim) */
};

static struct ioctlspy_counts C;
static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
static FILE *logf;
static int log_init;
static int (*next_ioctl)(int, unsigned long, ...);

/* syncobj model (IOCTLSPY_SYNCOBJ=1) */
#define MAX_SYNCOBJ 65536
static int model = -1;
static uint32_t next_handle = 1000;
static struct { uint8_t has_fence; uint64_t max_point; } S[MAX_SYNCOBJ];
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

static int model_on(void)
{
   if (model < 0) {
      const char *m = getenv("IOCTLSPY_SYNCOBJ");
      model = m && m[0] == '1';
   }
   return model;
}

/* mtx held */
static int avail(uint32_t h, uint64_t point)
{
   if (h >= MAX_SYNCOBJ)
      return 1;
   return point ? S[h].max_point >= point : S[h].has_fence;
}

static void attach(uint32_t h, uint64_t point)
{
   if (h >= MAX_SYNCOBJ)
      return;
   S[h].has_fence = 1;
   if (point > S[h].max_point)
      S[h].max_point = point;
   pthread_cond_broadcast(&cond);
}

/* mtx held; waits until all (or any) are available or the absolute CLOCK_MONOTONIC timeout passes */
static int wait_avail(const uint32_t *h, const uint64_t *points, unsigned n, int64_t abs_timeout, int any)
{
   for (;;) {
      unsigned ok = 0;
      for (unsigned i = 0; i < n; i++)
         ok += avail(h[i], points ? points[i] : 0);
      if (any ? ok > 0 : ok == n)
         return 0;
      struct timespec now;
      clock_gettime(CLOCK_MONOTONIC, &now);
      const int64_t now_ns = now.tv_sec * 1000000000ll + now.tv_nsec;
      if (abs_timeout <= now_ns)
         return -ETIME;
      /* pthread condvars use CLOCK_REALTIME by default: wait in short slices */
      struct timespec rt;
      clock_gettime(CLOCK_REALTIME, &rt);
      int64_t slice = abs_timeout - now_ns;
      if (slice > 10000000)
         slice = 10000000;
      rt.tv_nsec += slice;
      rt.tv_sec += rt.tv_nsec / 1000000000;
      rt.tv_nsec %= 1000000000;
      pthread_cond_timedwait(&cond, &mtx, &rt);
   }
}

void ioctlspy_snapshot(struct ioctlspy_counts *out)
{
   pthread_mutex_lock(&mtx);
   *out = C;
   pthread_mutex_unlock(&mtx);
}

static void open_log(void)
{
   if (log_init)
      return;
   log_init = 1;
   const char *p = getenv("IOCTLSPY_LOG");
   if (p && p[0])
      logf = fopen(p, "w");
}

void ioctlspy_mark(const char *text)
{
   pthread_mutex_lock(&mtx);
   open_log();
   if (logf) {
      fprintf(logf, "MARK %s\n", text);
      fflush(logf);
   }
   pthread_mutex_unlock(&mtx);
}

static uint64_t now_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

/* mtx held: 0 or -ETIME (the CS would wait for submission like the kernel's WAIT_FOR_SUBMIT) */
static int model_cs_waits(const union drm_amdgpu_cs *cs)
{
   const uint64_t *chunk_ptrs = (const uint64_t *)(uintptr_t)cs->in.chunks;
   for (unsigned i = 0; i < cs->in.num_chunks; i++) {
      const struct drm_amdgpu_cs_chunk *ch = (const struct drm_amdgpu_cs_chunk *)(uintptr_t)chunk_ptrs[i];
      if (ch->chunk_id != AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_WAIT)
         continue;
      const unsigned n = ch->length_dw * 4 / sizeof(struct drm_amdgpu_cs_chunk_syncobj);
      const struct drm_amdgpu_cs_chunk_syncobj *s = (const void *)(uintptr_t)ch->chunk_data;
      for (unsigned k = 0; k < n; k++) {
         if (avail(s[k].handle, s[k].point))
            continue;
         if (logf)
            fprintf(logf, "CS_BLOCKS_FOR_SUBMIT %u:%llu\n", s[k].handle, (unsigned long long)s[k].point);
         struct timespec now;
         clock_gettime(CLOCK_MONOTONIC, &now);
         const uint64_t pt = s[k].point;
         if (wait_avail(&s[k].handle, &pt, 1, now.tv_sec * 1000000000ll + now.tv_nsec + 5000000000ll, 0))
            return -ETIME;
      }
   }
   return 0;
}

static void model_cs_signals(const union drm_amdgpu_cs *cs)
{
   const uint64_t *chunk_ptrs = (const uint64_t *)(uintptr_t)cs->in.chunks;
   for (unsigned i = 0; i < cs->in.num_chunks; i++) {
      const struct drm_amdgpu_cs_chunk *ch = (const struct drm_amdgpu_cs_chunk *)(uintptr_t)chunk_ptrs[i];
      if (ch->chunk_id == AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_SIGNAL) {
         const unsigned n = ch->length_dw * 4 / sizeof(struct drm_amdgpu_cs_chunk_syncobj);
         const struct drm_amdgpu_cs_chunk_syncobj *s = (const void *)(uintptr_t)ch->chunk_data;
         for (unsigned k = 0; k < n; k++)
            attach(s[k].handle, s[k].point);
      } else if (ch->chunk_id == AMDGPU_CHUNK_ID_SYNCOBJ_OUT) {
         const unsigned n = ch->length_dw * 4 / sizeof(struct drm_amdgpu_cs_chunk_sem);
         const struct drm_amdgpu_cs_chunk_sem *s = (const void *)(uintptr_t)ch->chunk_data;
         for (unsigned k = 0; k < n; k++)
            attach(s[k].handle, 0);
      }
   }
}

static void log_cs(const union drm_amdgpu_cs *cs)
{
   const uint64_t *chunk_ptrs = (const uint64_t *)(uintptr_t)cs->in.chunks;
   unsigned ibs = 0, bos = 0, waits = 0, sigs = 0;
   char wbuf[1024] = "", sbuf[1024] = "", ibbuf[512] = "";
   size_t wo = 0, so = 0, io = 0;
   for (unsigned i = 0; i < cs->in.num_chunks; i++) {
      const struct drm_amdgpu_cs_chunk *ch = (const struct drm_amdgpu_cs_chunk *)(uintptr_t)chunk_ptrs[i];
      switch (ch->chunk_id) {
      case AMDGPU_CHUNK_ID_IB: {
         const struct drm_amdgpu_cs_chunk_ib *ib = (const void *)(uintptr_t)ch->chunk_data;
         ibs++;
         if (io < sizeof(ibbuf) - 32)
            io += snprintf(ibbuf + io, sizeof(ibbuf) - io, "%s%u:%u", io ? "," : "", ib->ip_type, ib->ib_bytes / 4);
         break;
      }
      case AMDGPU_CHUNK_ID_BO_HANDLES: {
         const struct drm_amdgpu_bo_list_in *bl = (const void *)(uintptr_t)ch->chunk_data;
         bos += bl->bo_number;
         break;
      }
      case AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_WAIT:
      case AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_SIGNAL: {
         const unsigned n = ch->length_dw * 4 / sizeof(struct drm_amdgpu_cs_chunk_syncobj);
         const struct drm_amdgpu_cs_chunk_syncobj *s = (const void *)(uintptr_t)ch->chunk_data;
         const int w = ch->chunk_id == AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_WAIT;
         for (unsigned k = 0; k < n; k++) {
            char *b = w ? wbuf : sbuf;
            size_t *o = w ? &wo : &so;
            if (*o < 1000)
               *o += snprintf(b + *o, 1024 - *o, "%s%u:%llu:%u", *o ? "," : "", s[k].handle,
                              (unsigned long long)s[k].point, s[k].flags);
         }
         if (w)
            waits += n;
         else
            sigs += n;
         break;
      }
      case AMDGPU_CHUNK_ID_SYNCOBJ_IN:
      case AMDGPU_CHUNK_ID_SYNCOBJ_OUT: {
         const unsigned n = ch->length_dw * 4 / sizeof(struct drm_amdgpu_cs_chunk_sem);
         const struct drm_amdgpu_cs_chunk_sem *s = (const void *)(uintptr_t)ch->chunk_data;
         const int w = ch->chunk_id == AMDGPU_CHUNK_ID_SYNCOBJ_IN;
         for (unsigned k = 0; k < n; k++) {
            char *b = w ? wbuf : sbuf;
            size_t *o = w ? &wo : &so;
            if (*o < 1000)
               *o += snprintf(b + *o, 1024 - *o, "%s%u:0:0", *o ? "," : "", s[k].handle);
         }
         if (w)
            waits += n;
         else
            sigs += n;
         break;
      }
      default:
         break;
      }
   }
   C.cs_ib_chunks += ibs;
   C.cs_bo_handles += bos;
   C.cs_wait_entries += waits;
   C.cs_signal_entries += sigs;
   if (cs->in.bo_list_handle)
      C.cs_with_bo_list_handle++;
   if (logf)
      fprintf(logf, "CS ctx=%u bo_list=%u ibs=%u ib=%s bos=%u waits=%s signals=%s\n", cs->in.ctx_id,
              cs->in.bo_list_handle, ibs, ibbuf, bos, wo ? wbuf : "-", so ? sbuf : "-");
}

static void log_handles(const char *what, const uint32_t *h, const uint64_t *points, unsigned n, unsigned flags)
{
   if (!logf)
      return;
   fprintf(logf, "%s flags=%u", what, flags);
   for (unsigned i = 0; i < n; i++)
      fprintf(logf, " %u:%llu", h[i], points ? (unsigned long long)points[i] : 0ull);
   fputc('\n', logf);
}

int ioctl(int fd, unsigned long request, ...)
{
   va_list ap;
   va_start(ap, request);
   void *arg = va_arg(ap, void *);
   va_end(ap);

   if (!next_ioctl)
      next_ioctl = (int (*)(int, unsigned long, ...))dlsym(RTLD_NEXT, "ioctl");

   if (_IOC_TYPE(request) != DRM_IOCTL_BASE)
      return next_ioctl(fd, request, arg);

   const unsigned nr = _IOC_NR(request);
   pthread_mutex_lock(&mtx);
   open_log();
   C.total++;
   C.drm++;
   if (nr == DRM_COMMAND_BASE + DRM_AMDGPU_CS) {
      C.cs++;
      log_cs(arg);
   } else if (request == DRM_IOCTL_SYNCOBJ_WAIT) {
      const struct drm_syncobj_wait *w = arg;
      C.syncobj_wait++;
      log_handles("SYNCOBJ_WAIT", (const uint32_t *)(uintptr_t)w->handles, NULL, w->count_handles, w->flags);
   } else if (request == DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT) {
      const struct drm_syncobj_timeline_wait *w = arg;
      C.syncobj_timeline_wait++;
      log_handles(w->timeout_nsec == 0 ? "SYNCOBJ_TIMELINE_WAIT_PROBE" : "SYNCOBJ_TIMELINE_WAIT",
                  (const uint32_t *)(uintptr_t)w->handles, (const uint64_t *)(uintptr_t)w->points,
                  w->count_handles, w->flags);
   } else if (request == DRM_IOCTL_SYNCOBJ_RESET) {
      const struct drm_syncobj_array *a = arg;
      C.syncobj_reset++;
      log_handles("SYNCOBJ_RESET", (const uint32_t *)(uintptr_t)a->handles, NULL, a->count_handles, 0);
   } else if (request == DRM_IOCTL_SYNCOBJ_SIGNAL) {
      const struct drm_syncobj_array *a = arg;
      C.syncobj_signal++;
      log_handles("SYNCOBJ_SIGNAL", (const uint32_t *)(uintptr_t)a->handles, NULL, a->count_handles, 0);
   } else if (request == DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL) {
      const struct drm_syncobj_timeline_array *a = arg;
      C.syncobj_timeline_signal++;
      log_handles("SYNCOBJ_TIMELINE_SIGNAL", (const uint32_t *)(uintptr_t)a->handles,
                  (const uint64_t *)(uintptr_t)a->points, a->count_handles, a->flags);
   } else if (request == DRM_IOCTL_SYNCOBJ_TRANSFER) {
      const struct drm_syncobj_transfer *t = arg;
      C.syncobj_transfer++;
      if (logf)
         fprintf(logf, "SYNCOBJ_TRANSFER dst=%u:%llu src=%u:%llu\n", t->dst_handle, (unsigned long long)t->dst_point,
                 t->src_handle, (unsigned long long)t->src_point);
   } else if (request == DRM_IOCTL_SYNCOBJ_QUERY) {
      C.syncobj_query++;
   } else if (request == DRM_IOCTL_SYNCOBJ_CREATE) {
      C.syncobj_create++;
   } else if (request == DRM_IOCTL_SYNCOBJ_DESTROY) {
      C.syncobj_destroy++;
   } else if (request == DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD || request == DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE) {
      C.syncobj_fd++;
   } else if (nr == DRM_COMMAND_BASE + DRM_AMDGPU_BO_LIST) {
      C.bo_list++;
   } else if (nr == DRM_COMMAND_BASE + DRM_AMDGPU_GEM_CREATE) {
      C.gem_create++;
   } else if (nr == DRM_COMMAND_BASE + DRM_AMDGPU_GEM_VA) {
      C.gem_va++;
   } else if (nr >= DRM_COMMAND_BASE) {
      C.other_drm++;
   } else {
      C.gem_other++;
   }
   int model_r = 0, modeled = 0;
   if (model_on()) {
      if (nr == DRM_COMMAND_BASE + DRM_AMDGPU_CS) {
         model_r = model_cs_waits(arg);
      } else if (request == DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT) {
         const struct drm_syncobj_timeline_wait *w = arg;
         model_r = wait_avail((const uint32_t *)(uintptr_t)w->handles, (const uint64_t *)(uintptr_t)w->points,
                              w->count_handles, w->timeout_nsec, !(w->flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL));
         modeled = 1;
      } else if (request == DRM_IOCTL_SYNCOBJ_WAIT) {
         const struct drm_syncobj_wait *w = arg;
         model_r = wait_avail((const uint32_t *)(uintptr_t)w->handles, NULL, w->count_handles, w->timeout_nsec,
                              !(w->flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL));
         modeled = 1;
      } else if (request == DRM_IOCTL_SYNCOBJ_RESET) {
         const struct drm_syncobj_array *a = arg;
         for (unsigned i = 0; i < a->count_handles; i++) {
            const uint32_t h = ((const uint32_t *)(uintptr_t)a->handles)[i];
            if (h < MAX_SYNCOBJ)
               S[h].has_fence = 0;
         }
      } else if (request == DRM_IOCTL_SYNCOBJ_SIGNAL) {
         /* the shim has no SYNCOBJ_SIGNAL (host signal of a binary syncobj, e.g. the sparse queue) */
         const struct drm_syncobj_array *a = arg;
         for (unsigned i = 0; i < a->count_handles; i++)
            attach(((const uint32_t *)(uintptr_t)a->handles)[i], 0);
         modeled = 1;
      } else if (request == DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL) {
         const struct drm_syncobj_timeline_array *a = arg;
         for (unsigned i = 0; i < a->count_handles; i++)
            attach(((const uint32_t *)(uintptr_t)a->handles)[i], ((const uint64_t *)(uintptr_t)a->points)[i]);
      } else if (request == DRM_IOCTL_SYNCOBJ_TRANSFER) {
         const struct drm_syncobj_transfer *t = arg;
         if (t->src_handle < MAX_SYNCOBJ && avail(t->src_handle, t->src_point))
            attach(t->dst_handle, t->dst_point);
      } else if (request == DRM_IOCTL_SYNCOBJ_QUERY) {
         const struct drm_syncobj_timeline_array *a = arg;
         for (unsigned i = 0; i < a->count_handles; i++) {
            const uint32_t h = ((const uint32_t *)(uintptr_t)a->handles)[i];
            ((uint64_t *)(uintptr_t)a->points)[i] = h < MAX_SYNCOBJ ? S[h].max_point : 0;
         }
         modeled = 1;
      }
   }
   pthread_mutex_unlock(&mtx);

   const uint64_t t0 = now_ns();
   int r = modeled ? 0 : next_ioctl(fd, request, arg);
   int e = errno;
   const uint64_t t1 = now_ns();
   if (model_r) {
      r = -1;
      e = -model_r;
   }

   pthread_mutex_lock(&mtx);
   C.ns_in_next += t1 - t0;
   if (request == DRM_IOCTL_SYNCOBJ_CREATE && r == 0) {
      struct drm_syncobj_create *c = arg;
      c->handle = next_handle++;
      if (c->handle < MAX_SYNCOBJ && (c->flags & DRM_SYNCOBJ_CREATE_SIGNALED))
         S[c->handle].has_fence = 1;
      if (logf)
         fprintf(logf, "SYNCOBJ_CREATE handle=%u flags=%u\n", c->handle, c->flags);
   }
   if (model_on() && r == 0 && nr == DRM_COMMAND_BASE + DRM_AMDGPU_CS)
      model_cs_signals(arg);
   if (logf && r != 0)
      fprintf(logf, "IOCTL_FAIL nr=0x%x r=%d errno=%d\n", nr, r, e);
   pthread_mutex_unlock(&mtx);
   errno = e;
   return r;
}
