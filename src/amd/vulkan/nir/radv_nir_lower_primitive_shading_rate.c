/*
 * Copyright © 2023 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "nir.h"
#include "nir_builder.h"
#include "radv_nir.h"

bool
radv_nir_lower_primitive_shading_rate(nir_shader *nir, enum amd_gfx_level gfx_level)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   bool progress = false;

   nir_builder b = nir_builder_create(impl);

   /* Iterate in reverse order since there should be only one deref store to PRIMITIVE_SHADING_RATE
    * after lower_io_to_temporaries for vertex shaders.
    */
   nir_foreach_block_reverse (block, impl) {
      nir_foreach_instr_reverse (instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_deref)
            continue;

         nir_variable *var = nir_intrinsic_get_var(intr, 0);
         if (var->data.mode != nir_var_shader_out || var->data.location != VARYING_SLOT_PRIMITIVE_SHADING_RATE)
            continue;

         b.cursor = nir_before_instr(instr);

         nir_def *val = intr->src[1].ssa;

         /* x_rate = (shadingRate & (Horizontal2Pixels | Horizontal4Pixels)) ? 0x1 : 0x0; */
         nir_def *x_rate = nir_iand_imm(&b, val, 12);
         x_rate = nir_b2i32(&b, nir_ine_imm(&b, x_rate, 0));

         /* y_rate = (shadingRate & (Vertical2Pixels | Vertical4Pixels)) ? 0x1 : 0x0; */
         nir_def *y_rate = nir_iand_imm(&b, val, 3);
         y_rate = nir_b2i32(&b, nir_ine_imm(&b, y_rate, 0));

         nir_def *out = NULL;

         /* MS:
          * Primitive shading rate is a per-primitive output, it is
          * part of the second channel of the primitive export.
          * Bits [28:31] = VRS rate
          * This will be added to the other bits of that channel in the backend.
          *
          * VS, TES, GS:
          * Primitive shading rate is a per-vertex output pos export.
          * Bits [2:5] = VRS rate
          * HW shading rate = (xRate << 2) | (yRate << 4)
          *
          * GFX11: 4-bit VRS_SHADING_RATE enum
          * GFX10: X = low 2 bits, Y = high 2 bits
          */
         unsigned x_rate_shift = 2;
         unsigned y_rate_shift = 4;

         if (gfx_level >= GFX11) {
            x_rate_shift = 4;
            y_rate_shift = 2;
         }
         if (nir->info.stage == MESA_SHADER_MESH) {
            x_rate_shift += 26;
            y_rate_shift += 26;
         }

         out = nir_ior(&b, nir_ishl_imm(&b, x_rate, x_rate_shift), nir_ishl_imm(&b, y_rate, y_rate_shift));

         nir_src_rewrite(&intr->src[1], out);

         progress = true;
         if (nir->info.stage == MESA_SHADER_VERTEX)
            break;
      }
      if (nir->info.stage == MESA_SHADER_VERTEX && progress)
         break;
   }

   return nir_progress(progress, impl, nir_metadata_control_flow);
}

/* GFX1013 no-op VRS (on by default, RADV_BC250_VRS_NOOP=0 hides it): GFX10.1 has no VRS
 * hardware, yet VK_KHR_fragment_shading_rate is exposed and every fragment is shaded at 1x1.
 *
 * Pre-rasterization stages: the PrimitiveShadingRateKHR output becomes a shader
 * temporary. Its stores are dead and removed (a Mesh shader that reads its own
 * output back still sees the value it wrote); the output never reaches the
 * position or primitive export. Fragment shaders: ShadingRateKHR reads 0 (1x1).
 *
 * This runs on deref-level NIR right after spirv_to_nir (radv_shader_spirv_to_nir),
 * so every consumer of the NIR sees a shader without the output: VS/TES/GS, every
 * BC250 Mesh route (expanded, split pieces, direct read, autocull, compact LDS,
 * implicit triangles, merge, AMD route, hybrid Task replay), graphics pipeline
 * libraries and shader objects. A shader that neither writes nor reads the shading
 * rate is not changed.
 */
bool
radv_nir_bc250_vrs_noop(nir_shader *nir, unsigned *stripped_outputs, unsigned *zeroed_reads)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   bool progress = false;

   *stripped_outputs = 0;
   *zeroed_reads = 0;

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      nir_builder b = nir_builder_create(impl);

      nir_foreach_block (block, impl) {
         nir_foreach_instr_safe (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_frag_shading_rate)
               continue;

            b.cursor = nir_before_instr(instr);
            nir_def_replace(&intr->def, nir_imm_zero(&b, 1, intr->def.bit_size));
            (*zeroed_reads)++;
            progress = true;
         }
      }

      nir_foreach_variable_with_modes_safe (var, nir, nir_var_system_value) {
         if (var->data.location == SYSTEM_VALUE_FRAG_SHADING_RATE) {
            exec_node_remove(&var->node);
            progress = true;
         }
      }
      BITSET_CLEAR(nir->info.system_values_read, SYSTEM_VALUE_FRAG_SHADING_RATE);

      return nir_progress(progress, impl, nir_metadata_control_flow);
   }

   nir_foreach_variable_with_modes (var, nir, nir_var_shader_out) {
      if (var->data.location != VARYING_SLOT_PRIMITIVE_SHADING_RATE)
         continue;

      var->data.mode = nir_var_shader_temp;
      var->data.per_primitive = false;
      (*stripped_outputs)++;
   }

   if (*stripped_outputs) {
      nir_fixup_deref_modes(nir);
      /* Like the other temporaries at this point: a function temporary, SSA values once lowered. */
      nir_lower_global_vars_to_local(nir);
      nir_remove_dead_variables(nir, nir_var_function_temp | nir_var_shader_temp, NULL);
      nir_opt_dce(nir);
      progress = true;
   }

   nir->info.outputs_written &= ~VARYING_BIT_PRIMITIVE_SHADING_RATE;
   nir->info.outputs_read &= ~VARYING_BIT_PRIMITIVE_SHADING_RATE;
   nir->info.per_primitive_outputs &= ~VARYING_BIT_PRIMITIVE_SHADING_RATE;

   return nir_progress(progress, impl, nir_metadata_none);
}
