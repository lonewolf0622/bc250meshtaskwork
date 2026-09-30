/* SPDX-License-Identifier: MIT */
#ifndef RADV_BC250_DGC_H
#define RADV_BC250_DGC_H
#include "radv_dgc.h"

bool radv_bc250_dgc_before(struct radv_cmd_buffer *cmd);
bool radv_bc250_dgc_layout(const struct radv_device *device,
                           const struct radv_indirect_command_layout *layout);
struct nir_shader *radv_bc250_dgc_shader(struct radv_device *device,
                                        const struct radv_indirect_command_layout *layout);
void radv_bc250_dgc_requirements(struct radv_device *device,
                                 const VkGeneratedCommandsMemoryRequirementsInfoEXT *info,
                                 VkMemoryRequirements2 *requirements);
void radv_bc250_dgc_prepare(struct radv_cmd_buffer *cmd,
                            const VkGeneratedCommandsInfoEXT *info, struct radv_cmd_buffer *state);
void radv_bc250_dgc_execute(struct radv_cmd_buffer *cmd, VkBool32 preprocessed,
                            const VkGeneratedCommandsInfoEXT *info);
#endif
