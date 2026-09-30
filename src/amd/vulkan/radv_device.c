/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__
#include <sys/inotify.h>
#endif

#include "layers/radv_app_workarounds.h"
#include "meta/radv_meta.h"
#include "tools/radv_debug_hang.h"
#include "tools/radv_rmv.h"
#include "tools/radv_spm.h"
#include "tools/radv_sqtt.h"
#include "util/u_debug.h"
#include "radv_bc250.h"
#include "radv_cs.h"
#include "radv_entrypoints.h"
#include "radv_formats.h"
#include "radv_physical_device.h"
#include "radv_shader.h"
#include "vk_common_entrypoints.h"
#include "vk_pipeline_cache.h"
#include "vk_util.h"
#ifndef _WIN32
#include "winsys/amdgpu/radv_amdgpu_winsys_public.h"
#endif
#include "util/mesa-blake3.h"
#include "util/u_atomic.h"
#include "util/u_process.h"
#include "git_sha1.h"
#include "vk_sync.h"

#include "aco_interface.h"

static bool
radv_trap_handler_enabled()
{
   return !!os_get_option("RADV_TRAP_HANDLER");
}

bool
radv_device_should_clear_vram(const struct radv_device *device)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);

   /* Ignore drirc radv_zero_vram=true if the feature is enabled to let applications take control. */
   return instance->drirc.debug.zero_vram && !device->vk.enabled_features.zeroInitializeDeviceMemory;
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_GetMemoryHostPointerPropertiesEXT(VkDevice _device, VkExternalMemoryHandleTypeFlagBits handleType,
                                       const void *pHostPointer,
                                       VkMemoryHostPointerPropertiesEXT *pMemoryHostPointerProperties)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   const struct radv_physical_device *pdev = radv_device_physical(device);

   switch (handleType) {
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT: {
      uint32_t memoryTypeBits = 0;
      for (int i = 0; i < pdev->memory_properties.memoryTypeCount; i++) {
         if (pdev->memory_domains[i] == RADEON_DOMAIN_GTT && !(pdev->memory_flags[i] & RADEON_FLAG_GTT_WC)) {
            memoryTypeBits = (1 << i);
            break;
         }
      }
      pMemoryHostPointerProperties->memoryTypeBits = memoryTypeBits;
      return VK_SUCCESS;
   }
   default:
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   }
}

static VkResult
radv_device_init_border_color(struct radv_device *device)
{
   VkResult result;

   result = radv_bo_create(device, NULL, RADV_BORDER_COLOR_BUFFER_SIZE, 4096, RADEON_DOMAIN_VRAM,
                           RADEON_FLAG_CPU_ACCESS | RADEON_FLAG_READ_ONLY | RADEON_FLAG_NO_INTERPROCESS_SHARING,
                           RADV_BO_PRIORITY_SHADER, 0, true, &device->border_color_data.bo);

   if (result != VK_SUCCESS)
      return vk_error(device, result);

   radv_rmv_log_border_color_palette_create(device, device->border_color_data.bo);

   result = device->ws->buffer_make_resident(device->ws, device->border_color_data.bo, true);
   if (result != VK_SUCCESS)
      return vk_error(device, result);

   device->border_color_data.colors_gpu_ptr = radv_buffer_map(device->ws, device->border_color_data.bo);
   if (!device->border_color_data.colors_gpu_ptr)
      return vk_error(device, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   mtx_init(&device->border_color_data.mutex, mtx_plain);

   return VK_SUCCESS;
}

static void
radv_device_finish_border_color(struct radv_device *device)
{
   if (device->border_color_data.bo) {
      radv_rmv_log_border_color_palette_destroy(device, device->border_color_data.bo);
      device->ws->buffer_make_resident(device->ws, device->border_color_data.bo, false);
      radv_bo_destroy(device, NULL, device->border_color_data.bo);

      mtx_destroy(&device->border_color_data.mutex);
   }
}

static struct radv_shader_part *
_radv_create_vs_prolog(struct radv_device *device, const void *_key)
{
   struct radv_vs_prolog_key *key = (struct radv_vs_prolog_key *)_key;
   return radv_create_vs_prolog(device, key);
}

static uint32_t
radv_hash_vs_prolog(const void *key_)
{
   const struct radv_vs_prolog_key *key = key_;
   return _mesa_hash_data(key, sizeof(*key));
}

static bool
radv_cmp_vs_prolog(const void *a_, const void *b_)
{
   const struct radv_vs_prolog_key *a = a_;
   const struct radv_vs_prolog_key *b = b_;

   return memcmp(a, b, sizeof(*a)) == 0;
}

static struct radv_shader_part_cache_ops vs_prolog_ops = {
   .create = _radv_create_vs_prolog,
   .hash = radv_hash_vs_prolog,
   .equals = radv_cmp_vs_prolog,
};

static VkResult
radv_device_init_vs_prologs(struct radv_device *device)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);

   radv_shader_part_cache_init(&device->vs_prologs, &vs_prolog_ops);

   /* don't pre-compile prologs if we want to print them */
   if (instance->debug_flags & RADV_DEBUG_DUMP_PROLOGS)
      return VK_SUCCESS;

   struct radv_vs_prolog_key key;
   memset(&key, 0, sizeof(key));
   key.as_ls = false;
   key.is_ngg = pdev->use_ngg;
   key.next_stage = MESA_SHADER_VERTEX;
   key.wave32 = pdev->ge_wave_size == 32;

   for (unsigned i = 1; i <= MAX_VERTEX_ATTRIBS; i++) {
      key.instance_rate_inputs = 0;
      key.num_attributes = i;

      device->simple_vs_prologs[i - 1] = radv_create_vs_prolog(device, &key);
      if (!device->simple_vs_prologs[i - 1])
         return vk_error(instance, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   unsigned idx = 0;
   for (unsigned num_attributes = 1; num_attributes <= 16; num_attributes++) {
      for (unsigned count = 1; count <= num_attributes; count++) {
         for (unsigned start = 0; start <= (num_attributes - count); start++) {
            key.instance_rate_inputs = BITFIELD_RANGE(start, count);
            key.num_attributes = num_attributes;

            struct radv_shader_part *prolog = radv_create_vs_prolog(device, &key);
            if (!prolog)
               return vk_error(instance, VK_ERROR_OUT_OF_DEVICE_MEMORY);

            assert(idx == radv_instance_rate_prolog_index(num_attributes, key.instance_rate_inputs));
            device->instance_rate_vs_prologs[idx++] = prolog;
         }
      }
   }
   assert(idx == ARRAY_SIZE(device->instance_rate_vs_prologs));

   return VK_SUCCESS;
}

static void
radv_device_finish_vs_prologs(struct radv_device *device)
{
   if (device->vs_prologs.ops)
      radv_shader_part_cache_finish(device, &device->vs_prologs);

   for (unsigned i = 0; i < ARRAY_SIZE(device->simple_vs_prologs); i++) {
      if (!device->simple_vs_prologs[i])
         continue;

      radv_shader_part_unref(device, device->simple_vs_prologs[i]);
   }

   for (unsigned i = 0; i < ARRAY_SIZE(device->instance_rate_vs_prologs); i++) {
      if (!device->instance_rate_vs_prologs[i])
         continue;

      radv_shader_part_unref(device, device->instance_rate_vs_prologs[i]);
   }
}

static struct radv_shader_part *
_radv_create_ps_epilog(struct radv_device *device, const void *_key)
{
   struct radv_ps_epilog_key *key = (struct radv_ps_epilog_key *)_key;
   return radv_create_ps_epilog(device, key, NULL);
}

static uint32_t
radv_hash_ps_epilog(const void *key_)
{
   const struct radv_ps_epilog_key *key = key_;
   return _mesa_hash_data(key, sizeof(*key));
}

static bool
radv_cmp_ps_epilog(const void *a_, const void *b_)
{
   const struct radv_ps_epilog_key *a = a_;
   const struct radv_ps_epilog_key *b = b_;

   return memcmp(a, b, sizeof(*a)) == 0;
}

static struct radv_shader_part_cache_ops ps_epilog_ops = {
   .create = _radv_create_ps_epilog,
   .hash = radv_hash_ps_epilog,
   .equals = radv_cmp_ps_epilog,
};

VkResult
radv_device_init_vrs_state(struct radv_device *device)
{
   VkDeviceMemory mem;
   VkBuffer buffer;
   VkResult result;
   VkImage image;

   VkImageCreateInfo image_create_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_D16_UNORM,
      .extent = {MAX_FRAMEBUFFER_WIDTH, MAX_FRAMEBUFFER_HEIGHT, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_2_DEPTH_STENCIL_ATTACHMENT_BIT_KHR,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = NULL,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };

   result =
      radv_image_create(radv_device_to_handle(device), &(struct radv_image_create_info){.vk_info = &image_create_info},
                        &device->meta_state.alloc, &image, true);
   if (result != VK_SUCCESS)
      return result;

   VkBufferCreateInfo buffer_create_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext =
         &(VkBufferUsageFlags2CreateInfo){
            .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO,
            .usage = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
         },
      .size = radv_image_from_handle(image)->planes[0].surface.meta_size,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };

   result = radv_create_buffer(device, &buffer_create_info, &device->meta_state.alloc, &buffer, true);
   if (result != VK_SUCCESS)
      goto fail_create;

   VkDeviceBufferMemoryRequirements buffer_mem_req_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS,
      .pCreateInfo = &buffer_create_info,
   };
   VkMemoryRequirements2 mem_req = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
   };

   radv_GetDeviceBufferMemoryRequirements(radv_device_to_handle(device), &buffer_mem_req_info, &mem_req);

   VkMemoryAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mem_req.memoryRequirements.size,
   };

   result = radv_alloc_memory(device, &alloc_info, &device->meta_state.alloc, &mem, true);
   if (result != VK_SUCCESS)
      goto fail_alloc;

   VkBindBufferMemoryInfo bind_info = {.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
                                       .buffer = buffer,
                                       .memory = mem,
                                       .memoryOffset = 0};

   result = radv_BindBufferMemory2(radv_device_to_handle(device), 1, &bind_info);
   if (result != VK_SUCCESS)
      goto fail_bind;

   device->vrs.image = radv_image_from_handle(image);
   device->vrs.buffer = radv_buffer_from_handle(buffer);
   device->vrs.mem = radv_device_memory_from_handle(mem);

   return VK_SUCCESS;

fail_bind:
   radv_FreeMemory(radv_device_to_handle(device), mem, &device->meta_state.alloc);
fail_alloc:
   radv_DestroyBuffer(radv_device_to_handle(device), buffer, &device->meta_state.alloc);
fail_create:
   radv_DestroyImage(radv_device_to_handle(device), image, &device->meta_state.alloc);

   return result;
}

static void
radv_device_finish_vrs_image(struct radv_device *device)
{
   if (!device->vrs.image)
      return;

   radv_FreeMemory(radv_device_to_handle(device), radv_device_memory_to_handle(device->vrs.mem),
                   &device->meta_state.alloc);
   radv_DestroyBuffer(radv_device_to_handle(device), radv_buffer_to_handle(device->vrs.buffer),
                      &device->meta_state.alloc);
   radv_DestroyImage(radv_device_to_handle(device), radv_image_to_handle(device->vrs.image), &device->meta_state.alloc);
}

static enum radv_force_vrs
radv_parse_vrs_rates(const char *str)
{
   if (!strcmp(str, "2x2")) {
      return RADV_FORCE_VRS_2x2;
   } else if (!strcmp(str, "2x1")) {
      return RADV_FORCE_VRS_2x1;
   } else if (!strcmp(str, "1x2")) {
      return RADV_FORCE_VRS_1x2;
   } else if (!strcmp(str, "1x1")) {
      return RADV_FORCE_VRS_1x1;
   }

   fprintf(stderr, "radv: Invalid VRS rates specified (valid values are 2x2, 2x1, 1x2 and 1x1)\n");
   return RADV_FORCE_VRS_1x1;
}

static const char *
radv_get_force_vrs_config_file(void)
{
   return os_get_option("RADV_FORCE_VRS_CONFIG_FILE");
}

static enum radv_force_vrs
radv_parse_force_vrs_config_file(const char *config_file)
{
   enum radv_force_vrs force_vrs = RADV_FORCE_VRS_1x1;
   char buf[4];
   FILE *f;

   f = fopen(config_file, "r");
   if (!f) {
      fprintf(stderr, "radv: Can't open file: '%s'.\n", config_file);
      return force_vrs;
   }

   if (fread(buf, sizeof(buf), 1, f) == 1) {
      buf[3] = '\0';
      force_vrs = radv_parse_vrs_rates(buf);
   }

   fclose(f);
   return force_vrs;
}

#ifdef __linux__

#define BUF_LEN ((10 * (sizeof(struct inotify_event) + NAME_MAX + 1)))

static int
radv_notifier_thread_run(void *data)
{
   struct radv_device *device = data;
   struct radv_notifier *notifier = &device->notifier;
   char buf[BUF_LEN];

   while (!notifier->quit) {
      const char *file = radv_get_force_vrs_config_file();
      struct timespec tm = {.tv_nsec = 100000000}; /* 1OOms */
      int length, i = 0;

      length = read(notifier->fd, buf, BUF_LEN);
      while (i < length) {
         struct inotify_event *event = (struct inotify_event *)&buf[i];

         i += sizeof(struct inotify_event) + event->len;
         if (event->mask & IN_MODIFY || event->mask & IN_DELETE_SELF) {
            /* Sleep 100ms for editors that use a temporary file and delete the original. */
            thrd_sleep(&tm, NULL);
            device->force_vrs = radv_parse_force_vrs_config_file(file);

            fprintf(stderr, "radv: Updated the per-vertex VRS rate to '%d'.\n", device->force_vrs);

            if (event->mask & IN_DELETE_SELF) {
               inotify_rm_watch(notifier->fd, notifier->watch);
               notifier->watch = inotify_add_watch(notifier->fd, file, IN_MODIFY | IN_DELETE_SELF);
            }
         }
      }

      thrd_sleep(&tm, NULL);
   }

   return 0;
}

#endif

static int
radv_device_init_notifier(struct radv_device *device)
{
#ifndef __linux__
   return true;
#else
   struct radv_notifier *notifier = &device->notifier;
   const char *file = radv_get_force_vrs_config_file();
   int ret;

   notifier->fd = inotify_init1(IN_NONBLOCK);
   if (notifier->fd < 0)
      return false;

   notifier->watch = inotify_add_watch(notifier->fd, file, IN_MODIFY | IN_DELETE_SELF);
   if (notifier->watch < 0)
      goto fail_watch;

   ret = thrd_create(&notifier->thread, radv_notifier_thread_run, device);
   if (ret)
      goto fail_thread;

   return true;

fail_thread:
   inotify_rm_watch(notifier->fd, notifier->watch);
fail_watch:
   close(notifier->fd);

   return false;
#endif
}

static void
radv_device_finish_notifier(struct radv_device *device)
{
#ifdef __linux__
   struct radv_notifier *notifier = &device->notifier;

   if (!notifier->thread)
      return;

   notifier->quit = true;
   thrd_join(notifier->thread, NULL);
   inotify_rm_watch(notifier->fd, notifier->watch);
   close(notifier->fd);
#endif
}

static VkResult
radv_device_init_perf_counter(struct radv_device *device)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const size_t bo_size = PERF_CTR_BO_PASS_OFFSET + sizeof(uint64_t) * PERF_CTR_MAX_PASSES;
   VkResult result;

   result = radv_bo_create(device, NULL, bo_size, 4096, RADEON_DOMAIN_GTT,
                           RADEON_FLAG_CPU_ACCESS | RADEON_FLAG_NO_INTERPROCESS_SHARING, RADV_BO_PRIORITY_UPLOAD_BUFFER,
                           0, true, &device->perf_counter_bo);
   if (result != VK_SUCCESS)
      return result;

   device->perf_counter_lock_cs = calloc(sizeof(struct radv_cmd_stream *), 2 * PERF_CTR_MAX_PASSES);
   if (!device->perf_counter_lock_cs)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   if (!pdev->ac_perfcounters.blocks)
      return VK_ERROR_INITIALIZATION_FAILED;

   return VK_SUCCESS;
}

static void
radv_device_finish_perf_counter(struct radv_device *device)
{
   if (device->perf_counter_bo)
      radv_bo_destroy(device, NULL, device->perf_counter_bo);

   if (!device->perf_counter_lock_cs)
      return;

   for (unsigned i = 0; i < 2 * PERF_CTR_MAX_PASSES; ++i) {
      if (device->perf_counter_lock_cs[i])
         radv_destroy_cmd_stream(device, device->perf_counter_lock_cs[i]);
   }

   free(device->perf_counter_lock_cs);
}

static VkResult
radv_device_init_memory_cache(struct radv_device *device)
{
   struct vk_pipeline_cache_create_info info = {.weak_ref = true};

   device->mem_cache = vk_pipeline_cache_create(&device->vk, &info, NULL);
   if (!device->mem_cache)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   return VK_SUCCESS;
}

static void
radv_device_finish_memory_cache(struct radv_device *device)
{
   if (device->mem_cache)
      vk_pipeline_cache_destroy(device->mem_cache, NULL);
}

static VkResult
radv_device_init_rgp(struct radv_device *device)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);

   if (!(instance->vk.trace_mode & RADV_TRACE_MODE_RGP))
      return VK_SUCCESS;

   if (pdev->info.gfx_level < GFX8 || pdev->info.gfx_level > GFX12) {
      fprintf(stderr, "GPU hardware not supported: refer to "
                      "the RGP documentation for the list of "
                      "supported GPUs!\n");
      abort();
   }

   if (!radv_sqtt_init(device))
      return VK_ERROR_INITIALIZATION_FAILED;

   fprintf(stderr,
           "radv: Thread trace support is enabled (initial buffer size: %u MiB, "
           "instruction timing: %s, cache counters: %s, queue events: %s).\n",
           device->sqtt.buffer_size / (1024 * 1024), radv_is_instruction_timing_enabled() ? "enabled" : "disabled",
           radv_spm_trace_enabled(pdev) ? "enabled" : "disabled",
           radv_sqtt_queue_events_enabled() ? "enabled" : "disabled");

   if (radv_spm_trace_enabled(pdev)) {
      if (pdev->info.gfx_level >= GFX10 && pdev->info.gfx_level <= GFX12) {
         if (!radv_spm_init(device))
            return VK_ERROR_INITIALIZATION_FAILED;
      } else {
         fprintf(stderr, "radv: SPM isn't supported for this GPU (%s)!\n", pdev->name);
      }
   }

   return VK_SUCCESS;
}

static void
radv_device_finish_rgp(struct radv_device *device)
{
   radv_sqtt_finish(device);
   radv_spm_finish(device);
}

static void
radv_device_init_rmv(struct radv_device *device)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);

   if (!(instance->vk.trace_mode & VK_TRACE_MODE_RMV))
      return;

   struct vk_rmv_device_info info;
   memset(&info, 0, sizeof(struct vk_rmv_device_info));
   radv_rmv_fill_device_info(pdev, &info);
   vk_memory_trace_init(&device->vk, &info);
   radv_memory_trace_init(device);
}

static VkResult
radv_device_init_trap_handler(struct radv_device *device)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);

   if (!pdev->info.has_trap_handler_support)
      return VK_SUCCESS;

   if (!radv_trap_handler_enabled())
      return VK_SUCCESS;

   fprintf(stderr, "**********************************************************************\n");
   fprintf(stderr, "* WARNING: RADV_TRAP_HANDLER is experimental and only for debugging! *\n");
   fprintf(stderr, "**********************************************************************\n");

   if (!radv_trap_handler_init(device))
      return VK_ERROR_INITIALIZATION_FAILED;

   return VK_SUCCESS;
}

static VkResult
radv_device_init_device_fault_detection(struct radv_device *device)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   struct radv_instance *instance = radv_physical_device_instance(pdev);

   if (!radv_device_fault_detection_enabled(device))
      return VK_SUCCESS;

   if (!radv_init_trace(device))
      return VK_ERROR_INITIALIZATION_FAILED;

   fprintf(stderr, "*****************************************************************************\n");
   fprintf(stderr, "* WARNING: RADV_DEBUG=hang is costly and should only be used for debugging! *\n");
   fprintf(stderr, "*****************************************************************************\n");

   /* Wait for idle after every draw/dispatch to identify the
    * first bad call.
    */
   instance->debug_flags |= RADV_DEBUG_SYNC_SHADERS;

   radv_dump_enabled_options(device, stderr);

   return VK_SUCCESS;
}

static void
radv_device_finish_device_fault_detection(struct radv_device *device)
{
   radv_finish_trace(device);
   ralloc_free(device->gpu_hang_report);
}

static VkResult
radv_shader_abort_data_init(struct radv_device *device)
{
   struct radv_shader_abort_data *shader_abort = &device->shader_abort;

   shader_abort->buffer_size = sizeof(uint32_t) + sizeof(uint64_t) + RADV_MAX_SHADER_ABORT_MESSAGE_SIZE;

   VkResult result =
      radv_backed_buffer_init(device, &shader_abort->buffer, shader_abort->buffer_size, radv_memory_type_visible_vram,
                              VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT, true);
   if (result != VK_SUCCESS)
      return result;

   shader_abort->buffer_addr = radv_backed_buffer_get_va(device, &shader_abort->buffer);

   /* Initialize the offset to write in the buffer header. */
   uint32_t *data = shader_abort->buffer.map;
   data[0] = sizeof(uint32_t);

   return VK_SUCCESS;
}

static void
radv_shader_abort_data_finish(struct radv_device *device)
{
   struct radv_shader_abort_data *shader_abort = &device->shader_abort;

   radv_backed_buffer_finish(device, &shader_abort->buffer);
}

static VkResult
radv_device_init_tools(struct radv_device *device)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   struct radv_instance *instance = radv_physical_device_instance(pdev);
   VkResult result;

   result = radv_device_init_device_fault_detection(device);
   if (result != VK_SUCCESS)
      return result;

   if (instance->debug_flags & RADV_DEBUG_VALIDATE_VAS) {
      result = radv_init_va_validation(device);
      if (result != VK_SUCCESS)
         return result;
   }

   result = radv_device_init_rgp(device);
   if (result != VK_SUCCESS)
      return result;

   radv_device_init_rmv(device);

   result = radv_device_init_trap_handler(device);
   if (result != VK_SUCCESS)
      return result;

   if (radv_bvh_dumping_enabled(instance) && radv_enable_rt(pdev)) {
      result = radv_rra_trace_init(device);
      if (result != VK_SUCCESS)
         return result;
   }

   result = radv_printf_data_init(device);
   if (result != VK_SUCCESS)
      return result;

   if (device->vk.enabled_features.shaderAbort) {
      result = radv_shader_abort_data_init(device);
      if (result != VK_SUCCESS)
         return result;
   }

   return VK_SUCCESS;
}

static void
radv_device_finish_tools(struct radv_device *device)
{
   radv_shader_abort_data_finish(device);
   radv_printf_data_finish(device);
   radv_rra_trace_finish(radv_device_to_handle(device), &device->rra_trace);
   radv_trap_handler_finish(device);
   radv_memory_trace_finish(device);
   radv_device_finish_rgp(device);
   radv_finish_va_validation(device);
   radv_device_finish_device_fault_detection(device);
}

struct dispatch_table_builder {
   struct vk_device_dispatch_table *tables[RADV_DISPATCH_TABLE_COUNT];
   bool used[RADV_DISPATCH_TABLE_COUNT];
   bool initialized[RADV_DISPATCH_TABLE_COUNT];
};

static void
add_entrypoints(struct dispatch_table_builder *b, const struct vk_device_entrypoint_table *entrypoints,
                enum radv_dispatch_table table)
{
   for (int32_t i = table - 1; i >= RADV_DEVICE_DISPATCH_TABLE; i--) {
      if (i == RADV_DEVICE_DISPATCH_TABLE || b->used[i]) {
         vk_device_dispatch_table_from_entrypoints(b->tables[i], entrypoints, !b->initialized[i]);
         b->initialized[i] = true;
      }
   }

   if (table < RADV_DISPATCH_TABLE_COUNT)
      b->used[table] = true;
}

static void
init_app_workarounds_entrypoints(struct radv_device *device, struct dispatch_table_builder *b)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);
   struct vk_device_entrypoint_table table = {0};

#define SET_ENTRYPOINT(app_layer, entrypoint) table.entrypoint = app_layer##_##entrypoint;
   if (!strcmp(instance->drirc.debug.app_layer, "metroexodus")) {
      SET_ENTRYPOINT(metro_exodus, GetSemaphoreCounterValue);
   } else if (!strcmp(instance->drirc.debug.app_layer, "rage2")) {
      SET_ENTRYPOINT(rage2, CmdBeginRenderPass);
   } else if (!strcmp(instance->drirc.debug.app_layer, "quanticdream")) {
      SET_ENTRYPOINT(quantic_dream, UnmapMemory2);
   } else if (!strcmp(instance->drirc.debug.app_layer, "no_mans_sky")) {
      SET_ENTRYPOINT(no_mans_sky, CreateImageView);
   } else if (!strcmp(instance->drirc.debug.app_layer, "strange_brigade")) {
      SET_ENTRYPOINT(strange_brigade, CmdPipelineBarrier2);
   } else if (!strcmp(instance->drirc.debug.app_layer, "gfxbench5")) {
      SET_ENTRYPOINT(gfxbench5, CmdPipelineBarrier2);
   } else if (!strcmp(instance->drirc.debug.app_layer, "ue5")) {
      SET_ENTRYPOINT(ue5, CmdSetViewport);
      SET_ENTRYPOINT(ue5, CmdSetScissor);
   }
#undef SET_ENTRYPOINT

   add_entrypoints(b, &table, RADV_APP_DISPATCH_TABLE);
}

static void
init_dispatch_tables(struct radv_device *device, struct radv_physical_device *pdev)
{
   const struct radv_instance *instance = radv_physical_device_instance(pdev);
   struct dispatch_table_builder b = {0};
   b.tables[RADV_DEVICE_DISPATCH_TABLE] = &device->vk.dispatch_table;
   b.tables[RADV_ANNOTATE_DISPATCH_TABLE] = &device->layer_dispatch.annotate;
   b.tables[RADV_APP_DISPATCH_TABLE] = &device->layer_dispatch.app;
   b.tables[RADV_RGP_DISPATCH_TABLE] = &device->layer_dispatch.rgp;
   b.tables[RADV_RRA_DISPATCH_TABLE] = &device->layer_dispatch.rra;
   b.tables[RADV_RMV_DISPATCH_TABLE] = &device->layer_dispatch.rmv;
   b.tables[RADV_UTRACE_DISPATCH_TABLE] = &device->layer_dispatch.utrace;
   b.tables[RADV_CTX_ROLL_DISPATCH_TABLE] = &device->layer_dispatch.ctx_roll;

   bool gather_ctx_rolls = instance->vk.trace_mode & RADV_TRACE_MODE_CTX_ROLLS;
   if (radv_device_fault_detection_enabled(device) || gather_ctx_rolls)
      add_entrypoints(&b, &annotate_device_entrypoints, RADV_ANNOTATE_DISPATCH_TABLE);

   init_app_workarounds_entrypoints(device, &b);

   if (instance->vk.trace_mode & RADV_TRACE_MODE_RGP)
      add_entrypoints(&b, &sqtt_device_entrypoints, RADV_RGP_DISPATCH_TABLE);

   if (radv_bvh_dumping_enabled(instance) && radv_enable_rt(pdev))
      add_entrypoints(&b, &rra_device_entrypoints, RADV_RRA_DISPATCH_TABLE);

#ifndef _WIN32
   if (instance->vk.trace_mode & VK_TRACE_MODE_RMV)
      add_entrypoints(&b, &rmv_device_entrypoints, RADV_RMV_DISPATCH_TABLE);
#endif

   if (device->utrace.context)
      add_entrypoints(&b, &utrace_device_entrypoints, RADV_UTRACE_DISPATCH_TABLE);

   if (gather_ctx_rolls)
      add_entrypoints(&b, &ctx_roll_device_entrypoints, RADV_CTX_ROLL_DISPATCH_TABLE);

   add_entrypoints(&b, &radv_device_entrypoints, RADV_DISPATCH_TABLE_COUNT);
   add_entrypoints(&b, &wsi_device_entrypoints, RADV_DISPATCH_TABLE_COUNT);
   add_entrypoints(&b, &vk_common_device_entrypoints, RADV_DISPATCH_TABLE_COUNT);
}

static VkResult
get_timestamp(struct vk_device *_device, uint64_t *timestamp)
{
   struct radv_device *device = container_of(_device, struct radv_device, vk);
   *timestamp = device->ws->query_value(device->ws, RADEON_TIMESTAMP);
   return VK_SUCCESS;
}

static VkResult
capture_trace(VkQueue _queue)
{
   VK_FROM_HANDLE(radv_queue, queue, _queue);
   struct radv_device *device = radv_queue_device(queue);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);

   VkResult result = VK_SUCCESS;

   if (instance->vk.trace_mode & (RADV_TRACE_MODE_RRA | RADV_TRACE_MODE_GAMMA))
      device->rra_trace.triggered = true;

   if (device->vk.memory_trace_data.is_enabled) {
      simple_mtx_lock(&device->vk.memory_trace_data.token_mtx);
      radv_rmv_collect_trace_events(device);
      vk_dump_rmv_capture(&device->vk.memory_trace_data);
      simple_mtx_unlock(&device->vk.memory_trace_data.token_mtx);
   }

   if (instance->vk.trace_mode & RADV_TRACE_MODE_RGP)
      device->sqtt_triggered = true;

   if (instance->vk.trace_mode & RADV_TRACE_MODE_CTX_ROLLS) {
      char filename[2048];
      time_t t = time(NULL);
      struct tm now = *localtime(&t);
      snprintf(filename, sizeof(filename), "/tmp/%s_%04d.%02d.%02d_%02d.%02d.%02d.ctxroll", util_get_process_name(),
               1900 + now.tm_year, now.tm_mon + 1, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec);

      simple_mtx_lock(&device->ctx_roll_mtx);

      device->ctx_roll_file = fopen(filename, "w");
      if (device->ctx_roll_file)
         fprintf(stderr, "radv: Writing context rolls to '%s'...\n", filename);

      simple_mtx_unlock(&device->ctx_roll_mtx);
   }

   return result;
}

static void
radv_device_init_cache_key(struct radv_device *device)
{
   STATIC_ASSERT(sizeof(device->compiler_info.hw) == 8);
   STATIC_ASSERT(sizeof(device->compiler_info.key) == 24);

   uint32_t ptr_size = sizeof(void *);

   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   _mesa_blake3_update(&ctx, &ptr_size, sizeof(ptr_size));
   _mesa_blake3_update(&ctx, device->compiler_info.ac, sizeof(struct ac_compiler_info));
   _mesa_blake3_update(&ctx, &device->compiler_info.hw, sizeof(device->compiler_info.hw));
   _mesa_blake3_update(&ctx, &device->compiler_info.key, sizeof(device->compiler_info.key));
   STATIC_ASSERT(sizeof(device->compiler_info.bc250x) == 4);
   uint32_t bc250x;
   memcpy(&bc250x, &device->compiler_info.bc250x, sizeof(bc250x));
   if (bc250x)
      _mesa_blake3_update(&ctx, &bc250x, sizeof(bc250x));
   if (device->bc250_env.pipeline_plan) {
      static const char tag[] = "bc250-pipeline-plan-v4";
      _mesa_blake3_update(&ctx, tag, sizeof(tag));
   }
   _mesa_blake3_final(&ctx, device->cache_hash);
   if (debug_get_bool_option("BC250_CAPTURE_POLICY_SHADERS", false)) {
      fprintf(stderr, "BC250POLICYCACHE cu=%u hash=", device->compiler_info.key.bc250_compute_cu_mode);
      for (unsigned i = 0; i < sizeof(device->cache_hash); ++i)
         fprintf(stderr, "%02x", device->cache_hash[i]);
      fprintf(stderr, "\n");
   }

}

static void
radv_create_gfx_preamble(struct radv_device *device)
{
   struct radv_cmd_stream *cs;
   VkResult result;

   result = radv_create_cmd_stream(device, AMD_IP_GFX, false, &cs);
   if (result != VK_SUCCESS)
      return;

   radeon_check_space(device->ws, cs->b, 512);

   radv_emit_graphics(device, cs);

   device->ws->cs_pad(cs->b, 0);

   const uint32_t gfx_init_bo_flags = RADEON_FLAG_CPU_ACCESS | RADEON_FLAG_NO_INTERPROCESS_SHARING |
                                      RADEON_FLAG_READ_ONLY | RADEON_FLAG_GTT_WC | RADEON_FLAG_GL2_BYPASS;

   result = radv_bo_create(device, NULL, cs->b->cdw * 4, 4096, device->ws->cs_domain(device->ws), gfx_init_bo_flags,
                           RADV_BO_PRIORITY_CS, 0, true, &device->gfx_init);
   if (result != VK_SUCCESS)
      goto fail;

   void *map = radv_buffer_map(device->ws, device->gfx_init);
   if (!map) {
      radv_bo_destroy(device, NULL, device->gfx_init);
      device->gfx_init = NULL;
      goto fail;
   }
   memcpy(map, cs->b->buf, cs->b->cdw * 4);

   device->ws->buffer_unmap(device->ws, device->gfx_init, false);
   device->gfx_init_size_dw = cs->b->cdw;
fail:
   radv_destroy_cmd_stream(device, cs);
}

/* For MSAA sample positions. */
#define FILL_SREG(s0x, s0y, s1x, s1y, s2x, s2y, s3x, s3y)                                                              \
   ((((unsigned)(s0x) & 0xf) << 0) | (((unsigned)(s0y) & 0xf) << 4) | (((unsigned)(s1x) & 0xf) << 8) |                 \
    (((unsigned)(s1y) & 0xf) << 12) | (((unsigned)(s2x) & 0xf) << 16) | (((unsigned)(s2y) & 0xf) << 20) |              \
    (((unsigned)(s3x) & 0xf) << 24) | (((unsigned)(s3y) & 0xf) << 28))

/* For obtaining location coordinates from registers */
#define SEXT4(x)               ((int)((x) | ((x) & 0x8 ? 0xfffffff0 : 0)))
#define GET_SFIELD(reg, index) SEXT4(((reg) >> ((index) * 4)) & 0xf)
#define GET_SX(reg, index)     GET_SFIELD((reg)[(index) / 4], ((index) % 4) * 2)
#define GET_SY(reg, index)     GET_SFIELD((reg)[(index) / 4], ((index) % 4) * 2 + 1)

/* 1x MSAA */
static const uint32_t sample_locs_1x = FILL_SREG(0, 0, 0, 0, 0, 0, 0, 0);
static const unsigned max_dist_1x = 0;
static const uint64_t centroid_priority_1x = 0x0000000000000000ull;

/* 2xMSAA */
static const uint32_t sample_locs_2x = FILL_SREG(4, 4, -4, -4, 0, 0, 0, 0);
static const unsigned max_dist_2x = 4;
static const uint64_t centroid_priority_2x = 0x1010101010101010ull;

/* 4xMSAA */
static const uint32_t sample_locs_4x = FILL_SREG(-2, -6, 6, -2, -6, 2, 2, 6);
static const unsigned max_dist_4x = 6;
static const uint64_t centroid_priority_4x = 0x3210321032103210ull;

/* 8xMSAA */
static const uint32_t sample_locs_8x[] = {
   FILL_SREG(1, -3, -1, 3, 5, 1, -3, -5),
   FILL_SREG(-5, 5, -7, -1, 3, 7, 7, -7),
   /* The following are unused by hardware, but we emit them to IBs
    * instead of multiple SET_CONTEXT_REG packets. */
   0,
   0,
};
static const unsigned max_dist_8x = 7;
static const uint64_t centroid_priority_8x = 0x7654321076543210ull;

unsigned
radv_get_default_max_sample_dist(int log_samples)
{
   unsigned max_dist[] = {
      max_dist_1x,
      max_dist_2x,
      max_dist_4x,
      max_dist_8x,
   };
   return max_dist[log_samples];
}

void
radv_emit_default_sample_locations(const struct radv_physical_device *pdev, struct radv_cmd_stream *cs, int nr_samples)
{
   uint64_t centroid_priority;

   radeon_begin(cs);

   switch (nr_samples) {
   default:
   case 1:
      centroid_priority = centroid_priority_1x;

      radeon_set_context_reg(R_028BF8_PA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_0, sample_locs_1x);
      radeon_set_context_reg(R_028C08_PA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y0_0, sample_locs_1x);
      radeon_set_context_reg(R_028C18_PA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y1_0, sample_locs_1x);
      radeon_set_context_reg(R_028C28_PA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y1_0, sample_locs_1x);
      break;
   case 2:
      centroid_priority = centroid_priority_2x;

      radeon_set_context_reg(R_028BF8_PA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_0, sample_locs_2x);
      radeon_set_context_reg(R_028C08_PA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y0_0, sample_locs_2x);
      radeon_set_context_reg(R_028C18_PA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y1_0, sample_locs_2x);
      radeon_set_context_reg(R_028C28_PA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y1_0, sample_locs_2x);
      break;
   case 4:
      centroid_priority = centroid_priority_4x;

      radeon_set_context_reg(R_028BF8_PA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_0, sample_locs_4x);
      radeon_set_context_reg(R_028C08_PA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y0_0, sample_locs_4x);
      radeon_set_context_reg(R_028C18_PA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y1_0, sample_locs_4x);
      radeon_set_context_reg(R_028C28_PA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y1_0, sample_locs_4x);
      break;
   case 8:
      centroid_priority = centroid_priority_8x;

      radeon_set_context_reg_seq(R_028BF8_PA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_0, 14);
      radeon_emit_array(sample_locs_8x, 4);
      radeon_emit_array(sample_locs_8x, 4);
      radeon_emit_array(sample_locs_8x, 4);
      radeon_emit_array(sample_locs_8x, 2);
      break;
   }

   /* The exclusion bits can be set to improve rasterization efficiency if no sample lies on the
    * pixel boundary (-8 sample offset). It's currently always TRUE because the driver doesn't
    * support 16 samples.
    */
   if (pdev->info.gfx_level >= GFX7 && pdev->info.gfx_level < GFX12) {
      radeon_set_context_reg(R_02882C_PA_SU_PRIM_FILTER_CNTL,
                             S_02882C_XMAX_RIGHT_EXCLUSION(1) | S_02882C_YMAX_BOTTOM_EXCLUSION(1));
   }

   if (pdev->info.gfx_level >= GFX12) {
      radeon_set_context_reg_seq(R_028BF0_PA_SC_CENTROID_PRIORITY_0, 2);
   } else {
      radeon_set_context_reg_seq(R_028BD4_PA_SC_CENTROID_PRIORITY_0, 2);
   }
   radeon_emit(centroid_priority);
   radeon_emit(centroid_priority >> 32);

   radeon_end();
}

static void
radv_get_sample_position(struct radv_device *device, unsigned sample_count, unsigned sample_index, float *out_value)
{
   const uint32_t *sample_locs;

   switch (sample_count) {
   case 1:
   default:
      sample_locs = &sample_locs_1x;
      break;
   case 2:
      sample_locs = &sample_locs_2x;
      break;
   case 4:
      sample_locs = &sample_locs_4x;
      break;
   case 8:
      sample_locs = sample_locs_8x;
      break;
   }

   out_value[0] = (GET_SX(sample_locs, sample_index) + 8) / 16.0f;
   out_value[1] = (GET_SY(sample_locs, sample_index) + 8) / 16.0f;
}

static void
radv_device_init_msaa(struct radv_device *device)
{
   int i;

   radv_get_sample_position(device, 1, 0, device->sample_locations_1x[0]);

   for (i = 0; i < 2; i++)
      radv_get_sample_position(device, 2, i, device->sample_locations_2x[i]);
   for (i = 0; i < 4; i++)
      radv_get_sample_position(device, 4, i, device->sample_locations_4x[i]);
   for (i = 0; i < 8; i++)
      radv_get_sample_position(device, 8, i, device->sample_locations_8x[i]);
}

static bool
radv_device_is_cache_disabled(const struct radv_device *device)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);

   /* The buffer address used for debug printf is hardcoded. */
   if (device->debug_nir.printf.buffer_addr)
      return true;

   /* The buffer address used for validating VAs is hardcoded. */
   if (device->debug_nir.valid_va.buffer_addr)
      return true;

   /* The buffer address used for shader abort is hardcoded. */
   if (device->shader_abort.buffer_addr)
      return true;

   /* Pipeline caches can be disabled with RADV_DEBUG=nocache, with MESA_GLSL_CACHE_DISABLE=1 and
    * when ACO_DEBUG is used. MESA_GLSL_CACHE_DISABLE is done elsewhere.
    */
   if ((instance->debug_flags & RADV_DEBUG_NO_CACHE) || (pdev->use_llvm ? 0 : aco_get_codegen_flags()))
      return true;

   return false;
}

/* RADV_BC250_MESH_REFERENCE=1 (GFX1013 BC250 Mesh only, default off): see radv_compiler_info::hw.
 * RADV_BC250_MESH_REFERENCE_CULLDIST=keep|strip (default keep) chooses what the reference mode does
 * with clip/cull distances of raw-route Mesh shaders. Returns 0 (off), 1 (keep) or 2 (strip). */
static unsigned
radv_bc250_parse_mesh_reference(const struct radv_physical_device *pdev)
{
   if (!debug_get_bool_option("RADV_BC250_MESH_REFERENCE", false))
      return 0;
   if (pdev->info.family != CHIP_GFX1013 || !pdev->bc250_native_mesh) {
      fprintf(stderr, "radv/bc250: RADV_BC250_MESH_REFERENCE ignored (GFX1013 Mesh only)\n");
      return 0;
   }
   const char *cd = debug_get_option("RADV_BC250_MESH_REFERENCE_CULLDIST", "keep");
   unsigned mode = 1;
   /* The reference driver has none of these experiments; the reference mode switches them off. */
   static const char *const experiments[] = {"RADV_BC250_MESH_AMD", "RADV_BC250_MESH_AMD_SIZE", "RADV_BC250_MESH_AMD_REUSE_OFF",
                                             "RADV_BC250_MESH_MERGE", "RADV_BC250_MESH_MIN2WAVES",
                                             "RADV_BC250_MESH_ALLOC_BARRIER"};
   for (unsigned i = 0; i < ARRAY_SIZE(experiments); i++) {
      if (debug_get_bool_option(experiments[i], false))
         fprintf(stderr, "radv/bc250: %s ignored with RADV_BC250_MESH_REFERENCE=1\n", experiments[i]);
   }
   if (!strcmp(cd, "strip"))
      mode = 2;
   else if (strcmp(cd, "keep"))
      fprintf(stderr, "radv/bc250: RADV_BC250_MESH_REFERENCE_CULLDIST=%s ignored (keep or strip)\n", cd);
   fprintf(stderr, "radv/bc250: raw-route Mesh pipelines use the reference driver's form (RADV_BC250_MESH_REFERENCE, "
           "clip/cull distances %s)\n", mode == 1 ? "kept" : "stripped");
   return mode;
}

static void
radv_device_init_compiler_info(struct radv_device *device)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   struct radv_instance *instance = radv_physical_device_instance(pdev);
   VkShaderStageFlags dump_shaders = 0;
   uint32_t nggc_max_ps_params = 0;

   if (instance->debug_flags & RADV_DEBUG_DUMP_VS)
      dump_shaders |= VK_SHADER_STAGE_VERTEX_BIT;
   if (instance->debug_flags & RADV_DEBUG_DUMP_TCS)
      dump_shaders |= VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
   if (instance->debug_flags & RADV_DEBUG_DUMP_TES)
      dump_shaders |= VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
   if (instance->debug_flags & RADV_DEBUG_DUMP_GS)
      dump_shaders |= VK_SHADER_STAGE_GEOMETRY_BIT;
   if (instance->debug_flags & RADV_DEBUG_DUMP_PS)
      dump_shaders |= VK_SHADER_STAGE_FRAGMENT_BIT;
   if (instance->debug_flags & RADV_DEBUG_DUMP_TASK)
      dump_shaders |= VK_SHADER_STAGE_TASK_BIT_EXT;
   if (instance->debug_flags & RADV_DEBUG_DUMP_MESH)
      dump_shaders |= VK_SHADER_STAGE_MESH_BIT_EXT;
   if (instance->debug_flags & RADV_DEBUG_DUMP_CS)
      dump_shaders |= VK_SHADER_STAGE_COMPUTE_BIT | RADV_RT_STAGE_BITS;

   if (pdev->use_ngg_culling) {
      /* Shader based culling efficiency can depend on PS throughput.
       * Estimate an upper limit for PS input param count based on GPU info.
       */
      nggc_max_ps_params = pdev->info.has_dedicated_vram ? 12 : 8;
   }

   bool image_2d_view_of_3d = device->vk.enabled_features.image2DViewOf3D && pdev->info.gfx_level == GFX9;
   bool mesh_shader_queries = device->vk.enabled_features.meshShaderQueries &&
                              (pdev->emulate_mesh_shader_queries || pdev->bc250_native_mesh);
   bool primitives_generated_query = radv_uses_primitives_generated_query(device);

   /* The Vulkan spec says:
    *  "Binary shaders retrieved from a physical device with a certain shaderBinaryUUID are
    *   guaranteed to be compatible with all other physical devices reporting the same
    *   shaderBinaryUUID and the same or higher shaderBinaryVersion."
    *
    * That means the driver should compile shaders for the "worst" case of all features being
    * enabled, regardless of what features are actually enabled on the logical device.
    */
   if (device->vk.enabled_features.shaderObject) {
      image_2d_view_of_3d = pdev->info.gfx_level == GFX9;
      primitives_generated_query = true;
   }

   const unsigned bc250_reference = radv_bc250_parse_mesh_reference(pdev);

   struct radv_compiler_info info = {
      /* Hardware info */
      .ac = &pdev->info.compiler_info,
      .hw =
         {
            .address32_hi = pdev->info.address32_hi,
            .instr_prefetch_distance = pdev->info.instr_prefetch_distance,
            .address_prt_wa_control_bit = pdev->info.address_prt_wa_control_bit,
            .rbplus_allowed = pdev->info.rbplus_allowed,
            .bc250_barycentrics = pdev->bc250_barycentrics,
            .bc250_bary_no_ref = pdev->bc250_barycentrics &&
               debug_get_bool_option("RADV_BC250_DIAG_BARY_NO_REF", false),
            .bc250_vrs_noop = pdev->bc250_vrs_noop,
            .bc250_mesh_cc_const = pdev->bc250_native_mesh &&
               debug_get_bool_option("RADV_BC250_MESH_CLIPCULL_CONST", true),
            .bc250_mesh_culldist_cull = pdev->bc250_native_mesh &&
               debug_get_bool_option("RADV_BC250_MESH_CULLDIST_CULL", true),
            .bc250_mesh_allow_pos1 = pdev->bc250_native_mesh &&
               debug_get_bool_option("RADV_BC250_MESH_ALLOW_POS1", false),
            .bc250_mesh_reference = bc250_reference != 0,
            .bc250_mesh_reference_keep_cd = bc250_reference == 1,
            /* On by default: exact barrier total for waves without API invocations (radv_shader.h). */
            .bc250_mesh_wave_sync = pdev->bc250_native_mesh &&
               debug_get_bool_option("RADV_BC250_MESH_WAVE_SYNC", true),
            .bc250_mesh_safe_compact = pdev->bc250_native_mesh && pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("RADV_BC250_MESH_SAFE_COMPACT", false),
            .bc250_mesh_dead_payload = pdev->bc250_native_mesh && pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("RADV_BC250_MESH_DEAD_PAYLOAD", false),
            .bc250_mesh_piece_primid = pdev->bc250_native_mesh && pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("RADV_BC250_MESH_PIECE_PRIMID", false),
            .bc250_mesh_fail_closed = pdev->bc250_native_mesh && pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("RADV_BC250_MESH_FAIL_CLOSED", false),
         },
      /* Misc values included as part of the cache key */
      .key =
         {
            /* Shader features */
            .use_llvm = pdev->use_llvm,
            .use_ngg = pdev->use_ngg,
            .use_ngg_culling = pdev->use_ngg_culling,
            .nggc_max_ps_params = nggc_max_ps_params,
            .no_ngg_gs = instance->drirc.performance.disable_ngg_gs,
            .load_grid_size_from_user_sgpr = pdev->load_grid_size_from_user_sgpr,
            .bc250_native_mesh = pdev->bc250_native_mesh,
            .bc250_compute_cu_mode = pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("BC250_EXPERIMENTAL_COMPUTE_CU_MODE", false),
            .bc250_output_regions = pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("BC250_OUTPUT_REGIONS", false),
            .bc250_cache_plan = pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("BC250_CACHE_PLAN", false),
            .bc250_parallel_cull = pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("BC250_PARALLEL_CULL", false),
            .bc250_compact_vertices = pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("BC250_COMPACT_VERTICES", false),
            .bc250_single_piece = pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("BC250_SINGLE_PIECE", false),
            .bc250_balanced_slices = pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("BC250_BALANCED_SLICES", false),
            .bc250_cull_compact = pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("BC250_EXPERIMENTAL_CULL_COMPACT", false),
            .bc250_pack_triangle_vertices = pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("BC250_EXPERIMENTAL_PACK_TRIANGLE_VERTICES", false),
            .bc250_native_task = pdev->bc250_native_task,
            .bc250_split_mesh = pdev->bc250_native_mesh &&
               debug_get_bool_option("RADV_BC250_SPLIT_MESH", false),
            .bc250_direct_split = pdev->bc250_native_mesh &&
               debug_get_bool_option("BC250_EXPERIMENTAL_DIRECT_SPLIT", false),
            .bc250_expand_primitives = pdev->bc250_native_mesh &&
               debug_get_bool_option("RADV_BC250_EXPAND_PRIMITIVES", false),
            /* RADV_BC250_MESH_AMD=1 enables all three parts; the
             * RADV_BC250_MESH_AMD_{ROUTE,SIZE,REUSE_OFF} switches enable one
             * part each, to bisect the parts on hardware. */
            .bc250_mesh_amd = pdev->bc250_native_mesh &&
               (debug_get_bool_option("RADV_BC250_MESH_AMD", false) ||
                debug_get_bool_option("RADV_BC250_MESH_AMD_ROUTE", false) || bc250_reference),
            .bc250_mesh_safe_autocull = pdev->bc250_native_mesh && pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("RADV_BC250_MESH_SAFE_AUTOCULL", false),
            .bc250_mesh_safe_pieces = pdev->bc250_native_mesh && pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("RADV_BC250_MESH_SAFE_PIECES", false),
            .bc250_mesh_safe_owned = pdev->bc250_native_mesh && pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("RADV_BC250_MESH_SAFE_OWNED", false),
            .bc250_mesh_safe_fast = pdev->bc250_native_mesh && pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("RADV_BC250_MESH_SAFE_FAST", false),
            .bc250_mesh_safe_ordered = pdev->bc250_native_mesh && pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("RADV_BC250_MESH_SAFE_ORDERED", false),
            .bc250_mesh_safe_parallel = pdev->bc250_native_mesh && pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("RADV_BC250_MESH_SAFE_PARALLEL", false),
            .bc250_mesh_safe_direct = pdev->bc250_native_mesh && pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("RADV_BC250_MESH_SAFE_DIRECT", false),
            .bc250_mesh_amd_size = pdev->bc250_native_mesh && !bc250_reference &&
               (debug_get_bool_option("RADV_BC250_MESH_AMD", false) ||
                debug_get_bool_option("RADV_BC250_MESH_AMD_SIZE", false)),
            /* GS_FAST_LAUNCH=0 is the only Mesh launch mode on GFX1013: every Mesh
             * pipeline the base driver produces (the base driver expansion, split pieces, hybrid-Task replay,
             * multiview, raw and merged shapes) launches with it, direct and
             * indirect. A clean in-game A/B matched the base driver's fast launch 1 (Hellblade 2:
             * 62 FPS both), as did every meshbench A/B (dir/ind, 1000 calls,
             * Mesh/VS mixing, heavy PS with 16x overdraw). An earlier "52 FPS"
             * result was an invalid A/B. */
            .bc250_mesh_fl0 = pdev->bc250_native_mesh,
            /* RADV_BC250_MESH_ALLOC_BARRIER=1: every Mesh shader gets a workgroup
             * s_barrier directly before GS_ALLOC_REQ (AMD's GFX10.1 NGG workaround
             * placement). ACO otherwise narrows the finale barrier of single-wave
             * workgroups and schedules GS_ALLOC_REQ ahead of the output writes. */
            /* RADV_BC250_MESH_MIN2WAVES=1: a raw (RADV_BC250_MESH_AMD_ROUTE) Mesh
             * workgroup that fits in one wave is launched as two waves. No
             * hardware run on any tree ever passed a one-wave subgroup with more
             * than 20 triangles, and 64V/64P/64T hung 5 times, while the same
             * index pattern with 98 lanes (2 waves) passes
             * (MESH_PERF/raw_boundary/REPORT.md). */
            /* RADV_BC250_MESH_FAST=1: mesh-only triangle Mesh shaders declaring more
             * than 64 primitives run raw (shared vertices, no split, no expansion)
             * on fast launch 0. Every such raw hardware run passed (64V with
             * 65/80/98 triangles, at VS parity); every 64-triangle raw run hung.
             * Raw-route shaders always launch with fast launch 0 (mixed mode). */
            .bc250_mesh_fast = pdev->bc250_native_mesh &&
               debug_get_bool_option("RADV_BC250_MESH_FAST", false),
            /* RADV_BC250_MESH_MERGE=1: pack K consecutive small triangle Mesh
             * workgroups (P <= 64, mesh-only) into one raw fast-launch-0
             * subgroup of K*P >= 65 triangles (radv_bc250_merge_mesh,
             * MESH_PERF/merge/DESIGN.md). Off by default; not hardware-run. */
            .bc250_mesh_merge = pdev->bc250_native_mesh && !bc250_reference &&
               debug_get_bool_option("RADV_BC250_MESH_MERGE", false),
            /* RADV_BC250_MESH_MERGE_INDIRECT (default a): with a, indirect
             * draws of merged shaders run a prep dispatch and launch
             * ceil(N/K) groups (option A, dims user SGPR); b keeps the stage 1/2
             * shaders and launches N groups, the surplus exiting empty. */
            .bc250_mesh_merge_prep = pdev->bc250_native_mesh && !bc250_reference &&
               debug_get_bool_option("RADV_BC250_MESH_MERGE", false) &&
               strcmp(debug_get_option("RADV_BC250_MESH_MERGE_INDIRECT", "a"), "b") != 0,
            /* RADV_BC250_MESH_AUTOCULL=1: the base driver's expanded (private-vertex) triangle Mesh
             * shaders cull their triangles in the epilogue (backface, frustum, small
             * primitive, from the runtime NGG culling settings like VS NGG culling)
             * and export only the survivors (ac_nir_lower_ngg_mesh.c,
             * ms_autocull_compact; MESH_PERF/autocull). Without face culling at draw time the
             * shader branches to its switch-off epilogue; by default only the measured-win
             * size class (wave32, 24..32 triangles) gets it. Off by default. */
            .bc250_mesh_autocull = pdev->bc250_native_mesh &&
               debug_get_bool_option("RADV_BC250_MESH_AUTOCULL", false),
            /* RADV_BC250_MESH_AUTOCULL_ALL=1 (experiments, with RADV_BC250_MESH_AUTOCULL=1):
             * autocull every candidate instead of only the measured-win size class
             * (radv_shader_info.c, calc_mesh_workgroup_size). */
            .bc250_mesh_autocull_all = pdev->bc250_native_mesh &&
               debug_get_bool_option("RADV_BC250_MESH_AUTOCULL", false) &&
               debug_get_bool_option("RADV_BC250_MESH_AUTOCULL_ALL", false),
            /* RADV_BC250_MESH_COMPACT_LDS=1: every expanded Mesh shader gets a
             * smaller LDS layout with the same exports in the same order: dead
             * shared copies dropped (radv_bc250_expand_primitive_attributes),
             * per-component primitive stores turned into slice stores
             * (radv_bc250_split_mesh), packed per-vertex output records
             * (ac_nir_lower_ngg_mesh). Off by default (hardware A/B). */
            .bc250_mesh_compact_lds = pdev->bc250_native_mesh &&
               debug_get_bool_option("RADV_BC250_MESH_COMPACT_LDS", false),
            /* RADV_BC250_SPLIT_PREP_FREE=1: direct mesh-only split shaders also take
             * their piece from bc250_constants.split_piece, so order-independent split
             * indirect draws can skip the argument setup (radv_bc250.c,
             * bc250_draw_split_prep_free). Off by default (hardware A/B). */
            .bc250_split_prep_free = pdev->bc250_native_mesh &&
               debug_get_bool_option("RADV_BC250_SPLIT_PREP_FREE", false),
            /* RADV_BC250_MESH_DIRECT_READ=1|full|<parts>: expanded Mesh shaders
             * export their outputs straight from the LDS staging instead of an
             * expanded copy (radv_bc250_direct_read_parts). Off by default. */
            .bc250_mesh_direct_read = pdev->bc250_native_mesh ?
               (radv_bc250_direct_read_parts(debug_get_option("RADV_BC250_MESH_DIRECT_READ", NULL)) |
                ((pdev->info.family == CHIP_GFX1013 &&
                  debug_get_bool_option("RADV_BC250_MESH_SAFE_CHECK", false)) ? RADV_BC250_MESH_SAFE_CHECK_KEY : 0) |
                ((pdev->info.family == CHIP_GFX1013 &&
                  debug_get_bool_option("RADV_BC250_MESH_SAFE_LOCAL", false)) ? RADV_BC250_MESH_SAFE_LOCAL_KEY : 0) |
                ((pdev->info.family == CHIP_GFX1013 &&
                  debug_get_bool_option("RADV_BC250_MESH_SAFE_STATS", false) &&
                  debug_get_bool_option("BC250_MESH_TIMER", false)) ? RADV_BC250_MESH_SAFE_STATS_KEY : 0)) : 0,
            /* RADV_BC250_MESH_AUTOCULL_WIDE=1|2 (with RADV_BC250_MESH_AUTOCULL=1): autocull
             * the wider expanded shapes too, e.g. 64 triangles in wave64 (radv_shader_info.c,
             * radv_bc250_autocull_wide_level). Off by default. */
            .bc250_mesh_autocull_wide = pdev->bc250_native_mesh &&
                  debug_get_bool_option("RADV_BC250_MESH_AUTOCULL", false) ?
               radv_bc250_autocull_wide_level(debug_get_option("RADV_BC250_MESH_AUTOCULL_WIDE", NULL)) : 0,
            .bc250_mesh_safe_corners = pdev->bc250_native_mesh && pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("RADV_BC250_MESH_SAFE_CORNERS", false),
            .bc250_mesh_safe_adaptive = pdev->bc250_native_mesh && pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("RADV_BC250_MESH_SAFE_ADAPTIVE", false),
            .bc250_mesh_safe_bary = pdev->bc250_native_mesh && pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("RADV_BC250_MESH_SAFE_BARY", false),
            .bc250_mesh_safe_bary_tiny = pdev->bc250_native_mesh && pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("RADV_BC250_MESH_SAFE_BARY_TINY", false),
            .bc250_mesh_safe_bary_affine = pdev->bc250_native_mesh && pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("RADV_BC250_MESH_SAFE_BARY_AFFINE", false),
            .bc250_bary_io16 = pdev->bc250_native_mesh && pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("RADV_BC250_BARY_IO16", false),
            .bc250_mesh_safe_bary_last = pdev->bc250_native_mesh && pdev->info.family == CHIP_GFX1013 &&
               debug_get_bool_option("RADV_BC250_MESH_SAFE_BARY_LAST", false),
            /* RADV_BC250_MESH_IMPLICIT_TRIS=1: expanded Mesh shaders without the compact
             * vertex map derive their primitive connectivity (3p, 3p+1, 3p+2) instead of
             * storing it to LDS and reloading it (ac_nir_lower_ngg_mesh.c,
             * ms_implicit_index). Same exports. Off by default. */
            .bc250_mesh_implicit_tris = pdev->bc250_native_mesh &&
               debug_get_bool_option("RADV_BC250_MESH_IMPLICIT_TRIS", false),
            /* RADV_BC250_MESH_COMPACT=1: the base driver's expanded Mesh shaders export
             * shared vertices, renumbered in the epilogue so that every exported vertex
             * is referenced and no backjump exceeds the hardware-proven window
             * (ac_nir_lower_ngg_mesh.c, ms_compact_vertices; MESH_PERF/compact). The
             * launch shape is unchanged. Off by default (hardware gates first). */
            .bc250_mesh_compact = pdev->bc250_native_mesh &&
               debug_get_bool_option("RADV_BC250_MESH_COMPACT", false),
            .bc250_mesh_min2waves = pdev->bc250_native_mesh && !bc250_reference &&
               debug_get_bool_option("RADV_BC250_MESH_MIN2WAVES", false),
            .bc250_mesh_alloc_barrier = pdev->bc250_native_mesh && !bc250_reference &&
               debug_get_bool_option("RADV_BC250_MESH_ALLOC_BARRIER", false),
            .bc250_mesh_amd_reuse = pdev->bc250_native_mesh && !bc250_reference &&
               (debug_get_bool_option("RADV_BC250_MESH_AMD", false) ||
                debug_get_bool_option("RADV_BC250_MESH_AMD_REUSE_OFF", false)),
            .emulate_ngg_gs_query_pipeline_stat = pdev->emulate_ngg_gs_query_pipeline_stat,
            .primitives_generated_query = primitives_generated_query,
            .mesh_shader_queries = mesh_shader_queries,
            .image_2d_view_of_3d = image_2d_view_of_3d,
            .use_fmask = pdev->use_fmask,
            .force_64_byte_sampled_image = pdev->force_64_byte_sampled_image,
            .robust_buffer_access = pdev->use_llvm && (device->vk.enabled_features.robustBufferAccess2 ||
                                                       device->vk.enabled_features.robustBufferAccess),
            .coop_matrix_robust_buffer_access = false,
            .mitigate_smem_oob = pdev->info.compiler_info.has_smem_oob_access_bug &&
                                 !(instance->debug_flags & RADV_DEBUG_NO_SMEM_MITIGATION),
            .mitigate_smem_with_null_prt =
               pdev->info.compiler_info.has_smem_with_null_prt_bug && radv_sparse_enabled(pdev),
            .bvh8 = radv_use_bvh8(pdev),
            .no_rt = !!(instance->debug_flags & RADV_DEBUG_NO_RT),
            .rt_cps = !!(instance->perftest_flags & RADV_PERFTEST_RT_CPS),
            .clear_lds = instance->drirc.misc.clear_lds,
            .disable_aniso_single_level = instance->drirc.debug.disable_aniso_single_level,
            .disable_shrink_image_store = instance->drirc.debug.disable_shrink_image_store,
            .disable_sinking_load_input_fs = instance->drirc.debug.disable_sinking_load_input_fs,
            .disable_trunc_coord = instance->drirc.debug.disable_trunc_coord,
            .enable_mrt_output_nan_fixup = instance->drirc.debug.enable_mrt_output_nan_fixup,
            .emulate_rt = radv_emulate_rt(pdev),
            .split_fma = instance->drirc.debug.split_fma,
            .ssbo_non_uniform = instance->drirc.debug.ssbo_non_uniform,
            .tex_non_uniform = instance->drirc.debug.tex_non_uniform,
            .lower_terminate_to_discard = instance->drirc.debug.lower_terminate_to_discard,
            .no_implicit_varying_subgroup_size = instance->drirc.debug.no_implicit_varying_subgroup_size,
            .force_nan_preserve_min_max = instance->drirc.debug.force_nan_preserve_min_max,
            .nir_debug_info = !!(instance->debug_flags & RADV_DEBUG_NIR_DEBUG_INFO),
            .force_aniso = device->force_aniso,
            /* Use CHIP_UNKNOWN for increased compatiblity between caches. */
            .family = pdev->use_llvm ? pdev->info.family : CHIP_UNKNOWN,

            /* Wave/subgroup sizes */
            .ge_wave_size = pdev->ge_wave_size,
            .ps_wave_size = pdev->ps_wave_size,
            .cs_wave_size = pdev->cs_wave_size,
            .rt_wave_size = pdev->rt_wave_size,

            /* RADV_BC250_PERF_PIECE_PRIMS (1..85, default 63): per-piece
             * primitive ceiling of the Mesh split (radv_bc250_split_mesh). It
             * changes the compiled pieces, so it is part of the cache key. */
            .bc250_piece_prims = pdev->bc250_native_mesh ?
               CLAMP(debug_get_num_option("RADV_BC250_PERF_PIECE_PRIMS", RADV_BC250_DEFAULT_PIECE_PRIMS), 1, 85) : 0,
         },
      /* Debug/tracing */
      .debug =
         {
            .dump_spirv = !!(instance->debug_flags & RADV_DEBUG_DUMP_SPIRV),
            .dump_backend_ir = !!(instance->debug_flags & RADV_DEBUG_DUMP_BACKEND_IR),
            .dump_preopt_ir = !!(instance->debug_flags & RADV_DEBUG_DUMP_PREOPT_IR),
            .dump_nir = !!(instance->debug_flags & RADV_DEBUG_DUMP_NIR),
            .dump_asm = !!(instance->debug_flags & RADV_DEBUG_DUMP_ASM),
            .dump_meta_shaders = !!(instance->debug_flags & RADV_DEBUG_DUMP_META_SHADERS),
            .dump_shader_stats = !!(instance->debug_flags & RADV_DEBUG_DUMP_SHADER_STATS),
            .dump_shaders = dump_shaders,
            .check_ir = !!(instance->debug_flags & RADV_DEBUG_CHECKIR),
            .printf_enabled = !!device->debug_nir.printf.buffer_addr,
            .trap_enabled = !!device->trap_handler_shader,
            .trap_excp_flags = instance->trap_excp_flags,
            .debug_report = &instance->vk.debug_report,
            .debug_nir = &device->debug_nir,
            .shader_abort = &device->shader_abort,
            .shader_dump_mtx = &instance->shader_dump_mtx,
            .keep_shader_info = device->keep_shader_info,
            .capture_shaders = (instance->debug_flags & RADV_DEBUG_DUMP_SHADERS) || device->keep_shader_info,
            /* Capture shader statistics when RGP is enabled to correlate shader hashes with Fossilize. */
            .capture_shader_stats = (instance->debug_flags & (RADV_DEBUG_DUMP_SHADER_STATS | RADV_DEBUG_PSO_HISTORY)) ||
                                    device->keep_shader_info || (instance->vk.trace_mode & RADV_TRACE_MODE_RGP),
            .family = pdev->info.family,
         },
      .rra_trace = &device->rra_trace,
      /* Cache */
      .cache_disabled = radv_device_is_cache_disabled(device),
      .enable_nir_cache = !!(instance->perftest_flags & RADV_PERFTEST_NIR_CACHE),
      .mem_cache = device->mem_cache,
      .override_graphics_shader_version = instance->drirc.misc.override_graphics_shader_version,
      .override_ray_tracing_shader_version = instance->drirc.misc.override_ray_tracing_shader_version,
      .override_compute_shader_version = instance->drirc.misc.override_compute_shader_version,
      /* Descriptors */
      .sampled_image_desc_size = radv_get_sampled_image_desc_size(pdev),
      .combined_image_sampler_desc_size = radv_get_combined_image_sampler_desc_size(pdev),
      .combined_image_sampler_offset = radv_get_combined_image_sampler_offset(pdev),
      .sampler_descriptor_size = pdev->vk.properties.samplerDescriptorSize,
      .sampler_descriptor_alignment = pdev->vk.properties.samplerDescriptorAlignment,
      .image_descriptor_size = pdev->vk.properties.imageDescriptorSize,
      .image_descriptor_alignment = pdev->vk.properties.imageDescriptorAlignment,
      .buffer_descriptor_size = pdev->vk.properties.bufferDescriptorSize,
      .buffer_descriptor_alignment = pdev->vk.properties.bufferDescriptorAlignment,
      /* Shader features, included as part of the pipeline key */
      .device_robustness_state = &device->vk.robustness_state,
      .smooth_lines = device->vk.enabled_features.smoothLines,
      .force_vrs_enabled = device->force_vrs_enabled,
      /* Wave/subgroup sizes */
      .subgroup_size = device->vk.physical->properties.subgroupSize,
      .min_subgroup_size = device->vk.physical->properties.minSubgroupSize,
      .max_subgroup_size = device->vk.physical->properties.maxSubgroupSize,
      /* NIR/SPIR-V */
      .spirv_caps = vk_physical_device_get_spirv_capabilities(device->vk.physical),
   };

   radv_get_nir_options(&info);

   const bool bc250_mesh = pdev->bc250_native_mesh && pdev->info.family == CHIP_GFX1013;
   info.bc250x.task_grid_fold = bc250_mesh && debug_get_bool_option("RADV_BC250_TASK_GRID_FOLD", false);
   info.bc250x.safe_pieces_ext = bc250_mesh && debug_get_bool_option("RADV_BC250_MESH_SAFE_PIECES_EXT", false);
   info.bc250x.pp_share = bc250_mesh && debug_get_bool_option("RADV_BC250_MESH_PP_SHARE", false);
   info.bc250x.lean_check = bc250_mesh && debug_get_bool_option("RADV_BC250_MESH_LEAN_CHECK", false);

   device->compiler_info = info;
}

static VkResult
radv_create_winsys(struct radv_device *device)
{
#ifdef _WIN32
   return VK_ERROR_INCOMPATIBLE_DRIVER;
#else
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);
   drmDevicePtr drm_device;
   VkResult result;
   int fd = -1;
   int r;

   if (pdev->drm_device_type == RADV_DRM_DEVICE_AMDGPU || pdev->drm_device_type == RADV_DRM_DEVICE_VIRTIO) {
      r = drmGetDeviceFromDevId(pdev->render_devid, 0, &drm_device);
      if (r)
         return VK_ERROR_INITIALIZATION_FAILED;

      const char *path = drm_device->nodes[DRM_NODE_RENDER];

      fd = open(path, O_RDWR | O_CLOEXEC);
      drmFreeDevice(&drm_device);
      if (fd < 0)
         return VK_ERROR_INITIALIZATION_FAILED;
   }

   const bool is_virtio =
      pdev->drm_device_type == RADV_DRM_DEVICE_AMDGPU_VPIPE || pdev->drm_device_type == RADV_DRM_DEVICE_VIRTIO;

   /* RADV_BC250_LOCAL_BOS: the upstream RADV_PERFTEST=localbos behaviour for this device's winsys. */
   const uint64_t perftest_flags =
      instance->perftest_flags | (device->bc250_env.local_bos ? RADV_PERFTEST_LOCAL_BOS : 0);
   result = radv_amdgpu_winsys_create(fd, &pdev->info, instance->debug_flags, perftest_flags, is_virtio,
                                      &device->ws);

   if (fd != -1)
      close(fd);

   return result;
#endif
}

static void
radv_destroy_device(struct radv_device *device, const VkAllocationCallbacks *pAllocator)
{
   if (device->bc250_env.submit_profile) {
      radv_bc250_submit_profile_print(device, true);
      if (device->bc250_prof_file)
         fclose(device->bc250_prof_file);
      device->bc250_prof_file = NULL;
   }
   radv_bc250_timer_finish(device);
   radv_device_finish_utrace(device);
   radv_device_finish_perf_counter(device);

   if (device->zero_bo) {
      device->ws->buffer_make_resident(device->ws, device->zero_bo, false);
      radv_bo_destroy(device, NULL, device->zero_bo);
   }

   if (device->gfx_init)
      radv_bo_destroy(device, NULL, device->gfx_init);

   radv_device_finish_notifier(device);
   radv_device_finish_vs_prologs(device);
   if (device->ps_epilogs.ops)
      radv_shader_part_cache_finish(device, &device->ps_epilogs);
   radv_device_finish_border_color(device);
   radv_device_finish_vrs_image(device);

   for (unsigned i = 0; i < RADV_MAX_QUEUE_FAMILIES; i++) {
      for (unsigned q = 0; q < device->queue_count[i]; q++)
         radv_queue_finish(&device->queues[i][q]);
      for (unsigned q = 0; q < device->queue_count_protected[i]; q++)
         radv_queue_finish(&device->queues_protected[i][q]);
      if (device->queue_count[i])
         vk_free(&device->vk.alloc, device->queues[i]);
      if (device->queue_count_protected[i])
         vk_free(&device->vk.alloc, device->queues_protected[i]);
   }
   if (device->private_sdma_queue != VK_NULL_HANDLE) {
      radv_queue_finish(device->private_sdma_queue);
      vk_free(&device->vk.alloc, device->private_sdma_queue);
   }

   _mesa_hash_table_destroy(device->rt_handles, NULL);

   radv_device_finish_meta(device);
   radv_device_finish_tools(device);
   radv_device_finish_memory_cache(device);

   radv_destroy_shader_upload_queue(device);

   for (unsigned i = 0; i < RADV_NUM_HW_CTX; i++) {
      if (device->hw_ctx[i])
         device->ws->ctx_destroy(device->hw_ctx[i]);
   }
   if (device->hw_vcn_enc_ctx)
      device->ws->ctx_destroy(device->hw_vcn_enc_ctx);

   mtx_destroy(&device->overallocation_mutex);
   simple_mtx_destroy(&device->ctx_roll_mtx);
   simple_mtx_destroy(&device->pstate_mtx);
   simple_mtx_destroy(&device->trace_mtx);
   simple_mtx_destroy(&device->rt_handles_mtx);
   simple_mtx_destroy(&device->pso_cache_stats_mtx);
   simple_mtx_destroy(&device->blit_queue_mtx);

   radv_destroy_shader_arenas(device);
   if (device->capture_replay_arena_vas)
      _mesa_hash_table_u64_destroy(device->capture_replay_arena_vas);

   if (device->ws)
      device->ws->destroy(device->ws);

   vk_device_finish(&device->vk);
   vk_free(&device->vk.alloc, device);
}

/* RADV_BC250_MESH_DEALLOC_DIST=N: see radv_device.h. N must fit
 * VGT_OUT_DEALLOC_CNTL.DEALLOC_DIST (7 bits); 0, garbage and larger values are ignored. */
static uint32_t
radv_bc250_parse_mesh_dealloc_dist(const struct radv_physical_device *pdev)
{
   const char *v = getenv("RADV_BC250_MESH_DEALLOC_DIST");
   if (!v || !*v)
      return 0;

   const uint32_t max = G_028C5C_DEALLOC_DIST(~0u);
   char *end = NULL;
   errno = 0;
   const long n = strtol(v, &end, 10);
   if (errno || end == v || *end || n < 1 || n > max) {
      fprintf(stderr, "radv/bc250: RADV_BC250_MESH_DEALLOC_DIST=%s ignored (1..%u)\n", v, max);
      return 0;
   }
   if (pdev->info.family != CHIP_GFX1013 || !pdev->bc250_native_mesh) {
      fprintf(stderr, "radv/bc250: RADV_BC250_MESH_DEALLOC_DIST ignored (GFX1013 Mesh only)\n");
      return 0;
   }
   fprintf(stderr, "radv/bc250: Mesh draws use VGT_OUT_DEALLOC_CNTL.DEALLOC_DIST=%ld (RADV_BC250_MESH_DEALLOC_DIST)\n",
           n);
   return n;
}

/* RADV_BC250_MESH_VERT_GRP=clamp|off: see radv_device.h. Anything else is ignored. */
static uint32_t
radv_bc250_parse_mesh_vert_grp(const struct radv_physical_device *pdev)
{
   const char *v = getenv("RADV_BC250_MESH_VERT_GRP");
   if (!v || !*v)
      return 0;

   uint32_t mode;
   if (!strcmp(v, "clamp"))
      mode = RADV_BC250_MESH_VERT_GRP_CLAMP;
   else if (!strcmp(v, "off"))
      mode = RADV_BC250_MESH_VERT_GRP_OFF;
   else {
      fprintf(stderr, "radv/bc250: RADV_BC250_MESH_VERT_GRP=%s ignored (clamp or off)\n", v);
      return 0;
   }
   if (pdev->info.family != CHIP_GFX1013 || !pdev->bc250_native_mesh) {
      fprintf(stderr, "radv/bc250: RADV_BC250_MESH_VERT_GRP ignored (GFX1013 Mesh only)\n");
      return 0;
   }
   fprintf(stderr, "radv/bc250: Mesh draws use GE_CNTL.VERT_GRP_SIZE mode %s (RADV_BC250_MESH_VERT_GRP)\n", v);
   return mode;
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_CreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *pCreateInfo,
                  const VkAllocationCallbacks *pAllocator, VkDevice *pDevice)
{
   VK_FROM_HANDLE(radv_physical_device, pdev, physicalDevice);
   struct radv_instance *instance = radv_physical_device_instance(pdev);
   VkResult result;
   struct radv_device *device;

   bool overallocation_disallowed = false;

   vk_foreach_struct_const (ext, pCreateInfo->pNext) {
      switch (ext->sType) {
      case VK_STRUCTURE_TYPE_DEVICE_MEMORY_OVERALLOCATION_CREATE_INFO_AMD: {
         const VkDeviceMemoryOverallocationCreateInfoAMD *overallocation = (const void *)ext;
         if (overallocation->overallocationBehavior == VK_MEMORY_OVERALLOCATION_BEHAVIOR_DISALLOWED_AMD)
            overallocation_disallowed = true;
         break;
      }
      default:
         break;
      }
   }

   device = vk_zalloc2(&instance->vk.alloc, pAllocator, sizeof(*device), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!device)
      return vk_error(instance, VK_ERROR_OUT_OF_HOST_MEMORY);

   result = vk_device_init(&device->vk, &pdev->vk, NULL, pCreateInfo, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free(&device->vk.alloc, device);
      return result;
   }

   /* Before anything that records, compiles or submits (radv_bc250_chain_enabled). */
   radv_bc250_device_env_init(device, pdev);

   result = radv_create_winsys(device);
   if (result != VK_SUCCESS)
      goto fail;

   device->vk.get_timestamp = get_timestamp;
   device->vk.capture_trace = capture_trace;

   device->vk.command_buffer_ops = &radv_cmd_buffer_ops;

   result = radv_device_init_utrace(device);
   if (result != VK_SUCCESS)
      goto fail;

   init_dispatch_tables(device, pdev);

   /* Initialize everything required for compilation, first. */

   simple_mtx_init(&device->ctx_roll_mtx, mtx_plain);
   simple_mtx_init(&device->trace_mtx, mtx_plain);
   simple_mtx_init(&device->pstate_mtx, mtx_plain);
   simple_mtx_init(&device->rt_handles_mtx, mtx_plain);
   simple_mtx_init(&device->pso_cache_stats_mtx, mtx_plain);
   simple_mtx_init(&device->blit_queue_mtx, mtx_plain);

   device->rt_handles = _mesa_hash_table_create(NULL, _mesa_hash_u32, _mesa_key_u32_equal);

   radv_init_shader_arenas(device);

   if (!device->vk.disable_internal_cache) {
      result = radv_device_init_memory_cache(device);
      if (result != VK_SUCCESS)
         goto fail;
   }

   if (pdev->info.gfx_level == GFX10_3) {
      if (os_get_option("RADV_FORCE_VRS_CONFIG_FILE")) {
         const char *file = radv_get_force_vrs_config_file();

         device->force_vrs = radv_parse_force_vrs_config_file(file);

         if (radv_device_init_notifier(device)) {
            device->force_vrs_enabled = true;
         } else {
            fprintf(stderr, "radv: Failed to initialize the notifier for RADV_FORCE_VRS_CONFIG_FILE!\n");
         }
      } else if (os_get_option("RADV_FORCE_VRS")) {
         const char *vrs_rates = os_get_option("RADV_FORCE_VRS");

         device->force_vrs = radv_parse_vrs_rates(vrs_rates);
         device->force_vrs_enabled = device->force_vrs != RADV_FORCE_VRS_1x1;
      }
   }

   device->force_aniso = MIN2(16, (int)debug_get_num_option("RADV_TEX_ANISO", -1));
   if (device->force_aniso >= 0) {
      fprintf(stderr, "radv: Forcing anisotropy filter to %ix\n", 1 << util_logbase2(device->force_aniso));
   }

   device->vk.sync = device->ws->get_sync_provider(device->ws);

   /* Disable unordered submits when SQTT queue events are enabled because queue present events
    * might be missing otherwise.
    */
   device->vk.copy_sync_payloads = ((instance->vk.trace_mode & RADV_TRACE_MODE_RGP) && radv_sqtt_queue_events_enabled())
                                      ? NULL
                                      : device->ws->copy_sync_payloads;

   /* VM_ALWAYS_VALID must be supported. */
   if (!pdev->info.has_vm_always_valid) {
      result = VK_ERROR_INITIALIZATION_FAILED;
      goto fail;
   }

   device->overallocation_disallowed = overallocation_disallowed;
   mtx_init(&device->overallocation_mutex, mtx_plain);

   if (pdev->info.has_kernelq_reg_shadowing || instance->debug_flags & RADV_DEBUG_SHADOW_REGS)
      device->uses_shadow_regs = true;

   bool video_dec_queue = false;
   bool video_enc_queue = false;

   /* Create one context per queue priority. */
   device->bc250_compute_priority = radv_bc250_compute_priority_option(pdev);
   for (unsigned i = 0; i < pCreateInfo->queueCreateInfoCount; i++) {
      const VkDeviceQueueCreateInfo *queue_create = &pCreateInfo->pQueueCreateInfos[i];
      const VkDeviceQueueGlobalPriorityCreateInfo *global_priority =
         vk_find_struct_const(queue_create->pNext, DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO);
      enum radeon_ctx_priority priority =
         radv_bc250_queue_priority(device, queue_create->queueFamilyIndex, global_priority);
      enum radv_queue_family qf = vk_queue_to_radv(pdev, queue_create->queueFamilyIndex);

      if (qf == RADV_QUEUE_VIDEO_DEC)
         video_dec_queue = true;
      else if (qf == RADV_QUEUE_VIDEO_ENC)
         video_enc_queue = true;

      if (device->hw_ctx[priority])
         continue;

      result = device->ws->ctx_create(device->ws, priority, &device->hw_ctx[priority]);
      if (result != VK_SUCCESS && !global_priority && qf == RADV_QUEUE_COMPUTE &&
          device->bc250_compute_priority > RADEON_CTX_PRIORITY_MEDIUM) {
         /* RADV_BC250_COMPUTE_QUEUE_PRIORITY=high|realtime without the privilege: keep the default. */
         fprintf(stderr, "radv/bc250: compute queue priority %d refused by the kernel (%d), using medium\n",
                 device->bc250_compute_priority, result);
         device->bc250_compute_priority = RADEON_CTX_PRIORITY_MEDIUM;
         priority = RADEON_CTX_PRIORITY_MEDIUM;
         if (device->hw_ctx[priority])
            continue;
         result = device->ws->ctx_create(device->ws, priority, &device->hw_ctx[priority]);
      }
      if (result != VK_SUCCESS)
         goto fail;
      if (!global_priority && qf == RADV_QUEUE_COMPUTE && device->bc250_compute_priority != RADEON_CTX_PRIORITY_INVALID)
         fprintf(stderr, "radv/bc250: compute queues use kernel context priority %d (RADV_BC250_COMPUTE_QUEUE_PRIORITY)\n",
                 priority);
   }

   /* Use extra context to allow use of both VCN instances for transcoding. */
   if (video_dec_queue && video_enc_queue && pdev->info.ip[AMD_IP_VCN_ENC].num_instances > 1) {
      result = device->ws->ctx_create(device->ws, RADEON_CTX_PRIORITY_MEDIUM, &device->hw_vcn_enc_ctx);
      if (result != VK_SUCCESS)
         return result;
   }

   for (unsigned i = 0; i < pCreateInfo->queueCreateInfoCount; i++) {
      const VkDeviceQueueCreateInfo *queue_create = &pCreateInfo->pQueueCreateInfos[i];
      uint32_t qfi = queue_create->queueFamilyIndex;
      const VkDeviceQueueGlobalPriorityCreateInfo *global_priority =
         vk_find_struct_const(queue_create->pNext, DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO);

      struct radv_queue **queues = queue_create->flags & VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT
                                      ? &device->queues_protected[qfi]
                                      : &device->queues[qfi];
      *queues = vk_zalloc(&device->vk.alloc, queue_create->queueCount * sizeof(struct radv_queue), 8,
                          VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
      if (!*queues) {
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto fail;
      }

      if (queue_create->flags & VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT)
         device->queue_count_protected[qfi] = queue_create->queueCount;
      else
         device->queue_count[qfi] = queue_create->queueCount;

      for (unsigned q = 0; q < queue_create->queueCount; q++) {
         result = radv_queue_init(device, &(*queues)[q], q, queue_create, global_priority);
         if (result != VK_SUCCESS)
            goto fail;
      }
   }
   device->private_sdma_queue = VK_NULL_HANDLE;

   device->shader_use_invisible_vram = (instance->perftest_flags & RADV_PERFTEST_DMA_SHADERS) &&
                                       /* SDMA buffer copy is only implemented for GFX7+. */
                                       pdev->info.gfx_level >= GFX7;
   result = radv_init_shader_upload_queue(device);
   if (result != VK_SUCCESS)
      goto fail;

   device->pbb_allowed = pdev->info.gfx_level >= GFX9 && !(instance->debug_flags & RADV_DEBUG_NOBINNING);

   /* The maximum number of scratch waves. Scratch space isn't divided
    * evenly between CUs. The number is only a function of the number of CUs.
    * We can decrease the constant to decrease the scratch buffer size.
    *
    * sctx->scratch_waves must be >= the maximum possible size of
    * 1 threadgroup, so that the hw doesn't hang from being unable
    * to start any.
    *
    * The recommended value is 4 per CU at most. Higher numbers don't
    * bring much benefit, but they still occupy chip resources (think
    * async compute). I've seen ~2% performance difference between 4 and 32.
    */
   uint32_t max_threads_per_block = 2048;
   device->scratch_waves = MAX2(32 * pdev->info.num_cu, max_threads_per_block / 64);

   device->dispatch_initiator = S_00B800_COMPUTE_SHADER_EN(1);

   if (pdev->info.gfx_level >= GFX7 && (pdev->info.family < CHIP_GFX940 || pdev->info.has_graphics)) {
      /* If the KMD allows it (there is a KMD hw register for it),
       * allow launching waves out-of-order.
       */
      device->dispatch_initiator |= S_00B800_ORDER_MODE(1);
   }
   if (pdev->info.gfx_level >= GFX10) {
      /* Enable asynchronous compute tunneling. The KMD restricts this feature
       * to high-priority compute queues, so setting the bit on any other queue
       * is a no-op. PAL always sets this bit as well.
       */
      device->dispatch_initiator |= S_00B800_TUNNEL_ENABLE(1);
   }

   /* Disable partial preemption for task shaders.
    * The kernel may not support preemption, but PAL always sets this bit,
    * so let's also set it here for consistency.
    */
   device->dispatch_initiator_task = device->dispatch_initiator | S_00B800_DISABLE_DISP_PREMPT_EN(1);

   /* Keep shader info for GPU hangs debugging. */
   device->keep_shader_info = radv_device_fault_detection_enabled(device) || radv_trap_handler_enabled();

   result = radv_device_init_tools(device);
   if (result != VK_SUCCESS)
      goto fail;

   result = radv_device_init_meta(device);
   if (result != VK_SUCCESS)
      goto fail;

   radv_device_init_msaa(device);

   /* If the border color extension is enabled, let's create the buffer we need. */
   if (device->vk.enabled_features.customBorderColors) {
      result = radv_device_init_border_color(device);
      if (result != VK_SUCCESS)
         goto fail;
   }

   radv_device_init_compiler_info(device);

   radv_device_init_cache_key(device);

   if (device->vk.enabled_features.vertexInputDynamicState || device->vk.enabled_features.graphicsPipelineLibrary ||
       device->vk.enabled_features.shaderObject) {
      result = radv_device_init_vs_prologs(device);
      if (result != VK_SUCCESS)
         goto fail;
   }

   if (device->vk.enabled_features.graphicsPipelineLibrary || device->vk.enabled_features.shaderObject ||
       device->vk.enabled_features.extendedDynamicState3ColorBlendEnable ||
       device->vk.enabled_features.extendedDynamicState3ColorWriteMask ||
       device->vk.enabled_features.extendedDynamicState3AlphaToCoverageEnable ||
       device->vk.enabled_features.extendedDynamicState3ColorBlendEquation)
      radv_shader_part_cache_init(&device->ps_epilogs, &ps_epilog_ops);

   if (pdev->info.has_zero_index_buffer_bug || device->compiler_info.key.mitigate_smem_oob) {
      result = radv_bo_create(device, NULL, 4096, 4096, RADEON_DOMAIN_VRAM,
                              RADEON_FLAG_NO_CPU_ACCESS | RADEON_FLAG_NO_INTERPROCESS_SHARING | RADEON_FLAG_READ_ONLY |
                                 RADEON_FLAG_ZERO_VRAM | RADEON_FLAG_32BIT,
                              RADV_BO_PRIORITY_VIRTUAL, 0, true, &device->zero_bo);
      if (result != VK_SUCCESS)
         goto fail;

      result = device->ws->buffer_make_resident(device->ws, device->zero_bo, true);
      if (result != VK_SUCCESS)
         goto fail;
   }

   if (pdev->info.has_graphics && !(instance->debug_flags & RADV_DEBUG_NO_IB_CHAINING))
      radv_create_gfx_preamble(device);

   if (device->vk.enabled_features.performanceCounterQueryPools) {
      result = radv_device_init_perf_counter(device);
      if (result != VK_SUCCESS)
         goto fail;
   }

   if (device->vk.enabled_features.rayTracingPipelineShaderGroupHandleCaptureReplay) {
      device->capture_replay_arena_vas = _mesa_hash_table_u64_create(NULL);
   }

   radv_bc250_timer_init(device); /* BC250_MESH_TIMER (off unless set) */
   device->bc250_split_batch_prep = pdev->info.family == CHIP_GFX1013 &&
                                    debug_get_bool_option("RADV_BC250_SPLIT_BATCH_PREP", false);
   device->bc250_mesh_dealloc_dist = radv_bc250_parse_mesh_dealloc_dist(pdev);
   device->bc250_mesh_vert_grp = radv_bc250_parse_mesh_vert_grp(pdev);

   *pDevice = radv_device_to_handle(device);
   return VK_SUCCESS;

fail:
   radv_destroy_device(device, pAllocator);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
radv_DestroyDevice(VkDevice _device, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(radv_device, device, _device);

   if (!device)
      return;

   radv_destroy_device(device, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL
radv_GetImageMemoryRequirements2(VkDevice _device, const VkImageMemoryRequirementsInfo2 *pInfo,
                                 VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   VK_FROM_HANDLE(radv_image, image, pInfo->image);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   uint32_t alignment;
   uint64_t size;

   const VkImagePlaneMemoryRequirementsInfo *plane_info =
      vk_find_struct_const(pInfo->pNext, IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO);

   if (plane_info) {
      const uint32_t plane = radv_plane_from_aspect(plane_info->planeAspect);

      size = image->planes[plane].surface.total_size;
      alignment = 1 << image->planes[plane].surface.alignment_log2;
   } else {
      size = image->size;
      alignment = image->alignment;
   }

   pMemoryRequirements->memoryRequirements.memoryTypeBits =
      ((1u << pdev->memory_properties.memoryTypeCount) - 1u) & ~pdev->memory_types_32bit;

   if (image->vk.create_flags & VK_IMAGE_CREATE_2_PROTECTED_BIT_KHR)
      pMemoryRequirements->memoryRequirements.memoryTypeBits &= pdev->memory_types_protected;

   if (image->vk.usage & VK_IMAGE_USAGE_2_HOST_TRANSFER_BIT_KHR) {
      /* Only expose host visible memory types for images that need to be mapped on the CPU. */
      pMemoryRequirements->memoryRequirements.memoryTypeBits &= pdev->memory_types_host_visible;
   }

   pMemoryRequirements->memoryRequirements.size = size;
   pMemoryRequirements->memoryRequirements.alignment = alignment;

   vk_foreach_struct (ext, pMemoryRequirements->pNext) {
      switch (ext->sType) {
      case VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS: {
         VkMemoryDedicatedRequirements *req = (VkMemoryDedicatedRequirements *)ext;
         req->requiresDedicatedAllocation =
            image->vk.external_handle_types && image->vk.tiling != VK_IMAGE_TILING_LINEAR;
         req->prefersDedicatedAllocation = req->requiresDedicatedAllocation;
         break;
      }
      default:
         break;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
radv_GetDeviceImageMemoryRequirements(VkDevice device, const VkDeviceImageMemoryRequirements *pInfo,
                                      VkMemoryRequirements2 *pMemoryRequirements)
{
   UNUSED VkResult result;
   VkImage image;

   /* Determining the image size/alignment require to create a surface, which isn't really possible
    * without creating an image.
    */
   result =
      radv_image_create(device, &(struct radv_image_create_info){.vk_info = pInfo->pCreateInfo}, NULL, &image, true);
   assert(result == VK_SUCCESS);

   VkImageMemoryRequirementsInfo2 info2 = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
      .image = image,
   };

   radv_GetImageMemoryRequirements2(device, &info2, pMemoryRequirements);

   radv_DestroyImage(device, image, NULL);
}

void
radv_gfx11_set_db_render_control(const struct radv_device *device, unsigned num_samples, unsigned *db_render_control)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   unsigned max_allowed_tiles_in_wave = 0;

   if (pdev->info.has_dedicated_vram) {
      if (num_samples == 8)
         max_allowed_tiles_in_wave = 6;
      else if (num_samples == 4)
         max_allowed_tiles_in_wave = 13;
      else
         max_allowed_tiles_in_wave = 0;
   } else {
      if (num_samples == 8)
         max_allowed_tiles_in_wave = 7;
      else if (num_samples == 4)
         max_allowed_tiles_in_wave = 15;
      else
         max_allowed_tiles_in_wave = 0;
   }

   *db_render_control |= S_028000_MAX_ALLOWED_TILES_IN_WAVE(max_allowed_tiles_in_wave);
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_GetMemoryFdKHR(VkDevice _device, const VkMemoryGetFdInfoKHR *pGetFdInfo, int *pFD)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   VK_FROM_HANDLE(radv_device_memory, memory, pGetFdInfo->memory);

   assert(pGetFdInfo->sType == VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR);

   /* At the moment, we support only the below handle types. */
   assert(pGetFdInfo->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT ||
          pGetFdInfo->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);

   /* Set BO metadata for dedicated image allocations.  We don't need it for import when the image
    * tiling is VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT, but we set it anyway for foreign consumers.
    */
   if (memory->image) {
      struct radv_image *image = memory->image;

      radv_image_bo_set_metadata(device, image, memory->bo);
   }

   bool ret = device->ws->buffer_get_fd(device->ws, memory->bo, pFD);
   if (ret == false)
      return vk_error(device, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return VK_SUCCESS;
}

static uint32_t
radv_compute_valid_memory_types_attempt(struct radv_physical_device *pdev, enum radeon_bo_domain domains,
                                        enum radeon_bo_flag flags, enum radeon_bo_flag ignore_flags)
{
   /* Don't count GTT/CPU as relevant:
    *
    * - We're not fully consistent between the two.
    * - Sometimes VRAM gets VRAM|GTT.
    */
   const enum radeon_bo_domain relevant_domains = RADEON_DOMAIN_VRAM | RADEON_DOMAIN_GDS | RADEON_DOMAIN_OA;
   uint32_t bits = 0;
   for (unsigned i = 0; i < pdev->memory_properties.memoryTypeCount; ++i) {
      if ((domains & relevant_domains) != (pdev->memory_domains[i] & relevant_domains))
         continue;

      if ((flags & ~ignore_flags) != (pdev->memory_flags[i] & ~ignore_flags))
         continue;

      bits |= 1u << i;
   }

   return bits;
}

static uint32_t
radv_compute_valid_memory_types(struct radv_physical_device *pdev, enum radeon_bo_domain domains,
                                enum radeon_bo_flag flags)
{
   enum radeon_bo_flag ignore_flags = ~(RADEON_FLAG_NO_CPU_ACCESS | RADEON_FLAG_GTT_WC);
   uint32_t bits = radv_compute_valid_memory_types_attempt(pdev, domains, flags, ignore_flags);

   if (!bits) {
      ignore_flags |= RADEON_FLAG_GTT_WC;
      bits = radv_compute_valid_memory_types_attempt(pdev, domains, flags, ignore_flags);
   }

   if (!bits) {
      ignore_flags |= RADEON_FLAG_NO_CPU_ACCESS;
      bits = radv_compute_valid_memory_types_attempt(pdev, domains, flags, ignore_flags);
   }

   /* Avoid 32-bit memory types for shared memory. */
   bits &= ~pdev->memory_types_32bit;

   return bits;
}
VKAPI_ATTR VkResult VKAPI_CALL
radv_GetMemoryFdPropertiesKHR(VkDevice _device, VkExternalMemoryHandleTypeFlagBits handleType, int fd,
                              VkMemoryFdPropertiesKHR *pMemoryFdProperties)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   struct radv_physical_device *pdev = radv_device_physical(device);

   switch (handleType) {
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT: {
      enum radeon_bo_domain domains;
      enum radeon_bo_flag flags;
      if (!device->ws->buffer_get_flags_from_fd(device->ws, fd, &domains, &flags))
         return vk_error(device, VK_ERROR_INVALID_EXTERNAL_HANDLE);

      pMemoryFdProperties->memoryTypeBits = radv_compute_valid_memory_types(pdev, domains, flags);
      return VK_SUCCESS;
   }
   default:
      /* The valid usage section for this function says:
       *
       *    "handleType must not be one of the handle types defined as
       *    opaque."
       *
       * So opaque handle types fall into the default "unsupported" case.
       */
      return vk_error(device, VK_ERROR_INVALID_EXTERNAL_HANDLE);
   }
}

bool
radv_device_set_pstate(struct radv_device *device, bool enable)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);
   struct radeon_winsys *ws = device->ws;
   enum radeon_ctx_pstate pstate = enable ? instance->profile_pstate : RADEON_CTX_PSTATE_NONE;

   /* pstate is per-device; setting it for one ctx is sufficient. We pick the first initialized one
    * below. */
   for (unsigned i = 0; i < RADV_NUM_HW_CTX; i++)
      if (device->hw_ctx[i])
         return ws->ctx_set_pstate(device->hw_ctx[i], pstate) >= 0;

   return true;
}

bool
radv_device_acquire_performance_counters(struct radv_device *device)
{
   bool result = true;
   simple_mtx_lock(&device->pstate_mtx);

   if (device->pstate_cnt == 0) {
      result = radv_device_set_pstate(device, true);
      if (result)
         ++device->pstate_cnt;
   }

   simple_mtx_unlock(&device->pstate_mtx);
   return result;
}

void
radv_device_release_performance_counters(struct radv_device *device)
{
   simple_mtx_lock(&device->pstate_mtx);

   if (--device->pstate_cnt == 0)
      radv_device_set_pstate(device, false);

   simple_mtx_unlock(&device->pstate_mtx);
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_AcquireProfilingLockKHR(VkDevice _device, const VkAcquireProfilingLockInfoKHR *pInfo)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   bool result = radv_device_acquire_performance_counters(device);
   return result ? VK_SUCCESS : VK_ERROR_UNKNOWN;
}

VKAPI_ATTR void VKAPI_CALL
radv_ReleaseProfilingLockKHR(VkDevice _device)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   radv_device_release_performance_counters(device);
}

VKAPI_ATTR void VKAPI_CALL
radv_GetDeviceImageSubresourceLayout(VkDevice device, const VkDeviceImageSubresourceInfo *pInfo,
                                     VkSubresourceLayout2 *pLayout)
{
   UNUSED VkResult result;
   VkImage image;

   result =
      radv_image_create(device, &(struct radv_image_create_info){.vk_info = pInfo->pCreateInfo}, NULL, &image, true);
   assert(result == VK_SUCCESS);

   radv_GetImageSubresourceLayout2(device, image, pInfo->pSubresource, pLayout);

   radv_DestroyImage(device, image, NULL);
}

static VkDeviceFaultAddressInfoKHR
radv_get_device_fault_addr_info(struct radv_device *device, bool *vm_fault_occurred)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   struct radv_winsys_gpuvm_fault_info fault_info = {0};
   VkDeviceFaultAddressInfoKHR addr_fault_info = {0};

   *vm_fault_occurred = radv_vm_fault_occurred(device, &fault_info);

   if (*vm_fault_occurred) {
      addr_fault_info.reportedAddress = ((int64_t)fault_info.addr << 16) >> 16;
      addr_fault_info.addressPrecision = 4096; /* 4K page granularity */

      if (pdev->info.gfx_level >= GFX10) {
         addr_fault_info.addressType = G_00A130_RW(fault_info.status) ? VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_KHR
                                                                      : VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_KHR;
      } else {
         /* Not sure how to get the access status on GFX6-9. */
         addr_fault_info.addressType = VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_KHR;
      }
   }

   return addr_fault_info;
}

static VkDeviceFaultVendorBinaryHeaderVersionOneKHR
radv_get_device_fault_vendor_binary_header(struct radv_device *device)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);
   VkDeviceFaultVendorBinaryHeaderVersionOneKHR hdr;

   hdr.headerSize = sizeof(VkDeviceFaultVendorBinaryHeaderVersionOneKHR);
   hdr.headerVersion = VK_DEVICE_FAULT_VENDOR_BINARY_HEADER_VERSION_ONE_KHR;
   hdr.vendorID = pdev->vk.properties.vendorID;
   hdr.deviceID = pdev->vk.properties.deviceID;
   hdr.driverVersion = pdev->vk.properties.driverVersion;
   memcpy(hdr.pipelineCacheUUID, pdev->cache_uuid, VK_UUID_SIZE);
   hdr.applicationNameOffset = 0;
   hdr.applicationVersion = instance->vk.app_info.app_version;
   hdr.engineNameOffset = 0;
   hdr.engineVersion = instance->vk.app_info.engine_version;
   hdr.apiVersion = instance->vk.app_info.api_version;

   return hdr;
}

/* VK_EXT_device_fault */
VKAPI_ATTR VkResult VKAPI_CALL
radv_GetDeviceFaultInfoEXT(VkDevice _device, VkDeviceFaultCountsEXT *pFaultCounts, VkDeviceFaultInfoEXT *pFaultInfo)
{
   VK_OUTARRAY_MAKE_TYPED(VkDeviceFaultAddressInfoKHR, out, pFaultInfo ? pFaultInfo->pAddressInfos : NULL,
                          &pFaultCounts->addressInfoCount);
   VK_FROM_HANDLE(radv_device, device, _device);
   bool vm_fault_occurred = false;

   pFaultCounts->vendorInfoCount = 0;
   pFaultCounts->vendorBinarySize = 0;

   if (device->gpu_hang_report) {
      VkDeviceFaultVendorBinaryHeaderVersionOneKHR hdr = radv_get_device_fault_vendor_binary_header(device);

      pFaultCounts->vendorBinarySize = sizeof(hdr) + strlen(device->gpu_hang_report);
      if (pFaultInfo) {
         memcpy(pFaultInfo->pVendorBinaryData, &hdr, sizeof(hdr));
         memcpy((char *)pFaultInfo->pVendorBinaryData + sizeof(hdr), device->gpu_hang_report,
                strlen(device->gpu_hang_report));
      }
   }

   VkDeviceFaultAddressInfoKHR addr_fault_info = radv_get_device_fault_addr_info(device, &vm_fault_occurred);

   if (vm_fault_occurred) {
      if (pFaultInfo)
         strncpy(pFaultInfo->description, "A GPUVM fault has been detected", sizeof(pFaultInfo->description));
      vk_outarray_append_typed(VkDeviceFaultAddressInfoKHR, &out, elem) *elem = addr_fault_info;
   }

   return vk_outarray_status(&out);
}

/* VK_KHR_device_fault */
static void
radv_shader_abort_get_data(struct radv_device *device, uint64_t *msg_data_size, uint8_t **msg_data)
{
   struct radv_shader_abort_data *shader_abort = &device->shader_abort;

   *msg_data_size = 0;
   *msg_data = NULL;

   if (!shader_abort->buffer.map)
      return;

   device->vk.dispatch_table.DeviceWaitIdle(radv_device_to_handle(device));

   uint32_t *data = shader_abort->buffer.map;

   *msg_data_size = data[0] - sizeof(uint32_t); /* substract original offset */
   *msg_data = (uint8_t *)shader_abort->buffer.map + sizeof(uint32_t);
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_GetDeviceFaultDebugInfoKHR(VkDevice _device, VkDeviceFaultDebugInfoKHR *pDebugInfo)
{
   VK_FROM_HANDLE(radv_device, device, _device);

   pDebugInfo->vendorBinarySize = 0;

   VkDeviceFaultShaderAbortMessageInfoKHR *abort_msg_info =
      vk_find_struct(pDebugInfo->pNext, DEVICE_FAULT_SHADER_ABORT_MESSAGE_INFO_KHR);
   if (abort_msg_info) {
      uint64_t msg_data_size;
      uint8_t *msg_data;

      radv_shader_abort_get_data(device, &msg_data_size, &msg_data);

      abort_msg_info->messageDataSize = msg_data_size;
      if (abort_msg_info->pMessageData && msg_data && msg_data_size > 0)
         memcpy((uint8_t *)abort_msg_info->pMessageData, msg_data, msg_data_size);
   }

   if (device->gpu_hang_report) {
      VkDeviceFaultVendorBinaryHeaderVersionOneKHR hdr = radv_get_device_fault_vendor_binary_header(device);

      pDebugInfo->vendorBinarySize = sizeof(hdr) + strlen(device->gpu_hang_report);
      if (pDebugInfo->pVendorBinaryData) {
         memcpy(pDebugInfo->pVendorBinaryData, &hdr, sizeof(hdr));
         memcpy((char *)pDebugInfo->pVendorBinaryData + sizeof(hdr), device->gpu_hang_report,
                strlen(device->gpu_hang_report));
      }
   }

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_GetDeviceFaultReportsKHR(VkDevice _device, uint64_t timeout, uint32_t *pFaultCounts,
                              VkDeviceFaultInfoKHR *pFaultInfo)
{
   VK_OUTARRAY_MAKE_TYPED(VkDeviceFaultInfoKHR, out, pFaultInfo, pFaultCounts);
   VK_FROM_HANDLE(radv_device, device, _device);
   VkDeviceFaultAddressInfoKHR addr_fault_info;
   bool vm_fault_occurred = false;
   bool timed_out = false;

   uint64_t abs_timeout = os_time_get_absolute_timeout(timeout);
   do {
      addr_fault_info = radv_get_device_fault_addr_info(device, &vm_fault_occurred);
   } while (timeout > 0 && !vm_fault_occurred && !(timed_out = (abs_timeout < os_time_get_nano())));

   if (!vm_fault_occurred)
      return VK_TIMEOUT;

   VkDeviceFaultInfoKHR fault_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_KHR,
      .flags = VK_DEVICE_FAULT_FLAG_MEMORY_ADDRESS_KHR,
      .faultAddressInfo = addr_fault_info,
   };
   strncpy(fault_info.description, "A GPUVM fault has been detected", sizeof(fault_info.description));

   vk_outarray_append_typed(VkDeviceFaultInfoKHR, &out, elem) *elem = fault_info;

   return vk_outarray_status(&out);
}
