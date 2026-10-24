/* SPDX-License-Identifier: MIT */
#ifndef RADV_DIRECTMESH_H
#define RADV_DIRECTMESH_H

#include "amd_family.h"
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

static inline bool
radv_directmesh_family(enum radeon_family family)
{
   return family == CHIP_NAVI10 || family == CHIP_NAVI12 || family == CHIP_NAVI14 ||
          family == CHIP_GFX1013;
}

/* Exact display-device IDs from Linux's amdgpu PCI table. Do not match a whole
 * PCI range: adjacent IDs include unrelated devices and audio functions. */
static inline bool
radv_directmesh_pci(uint16_t vendor, uint16_t device)
{
   if (vendor != 0x1002)
      return false;
   switch (device) {
   case 0x13fe:
   case 0x143f:
   case 0x7310: case 0x7312: case 0x7318: case 0x7319:
   case 0x731a: case 0x731b: case 0x731e: case 0x731f:
   case 0x7340: case 0x7341: case 0x7347: case 0x734f:
   case 0x7360: case 0x7362:
      return true;
   default:
      return false;
   }
}

static inline bool
radv_directmesh_preset(const char *option, bool detected)
{
   return option ? !strcmp(option, "1") : detected;
}

#endif
