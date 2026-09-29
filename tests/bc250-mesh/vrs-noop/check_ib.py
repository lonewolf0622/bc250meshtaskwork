#!/usr/bin/env python3
"""check_ib.py <RADV_DEBUG=dumpibs stderr> [expected DISPATCH count in the GFX IBs]

GFX10.1 has no VRS hardware: no IB may write a GFX10.3+ VRS register (by name or, as the GFX10 register
tables do not know them, by offset) or set a GFX10.3 VRS field in a register GFX10.1 has. Prints one
key=value line; exits 1 on any violation.
  VRS registers:  PA_CL_VRS_CNTL 0x28848, GE_VRS_RATE 0x3098c, DB_VRS_OVERRIDE_CNTL 0x28064,
                  PA_SC_VRS_OVERRIDE_CNTL 0x283d0, PA_SC_VRS_INFO 0x283e0, PA_SC_VRS_RATE_BASE 0x283f0,
                  PA_SC_VRS_RATE_BASE_EXT 0x283f4, PA_SC_VRS_RATE_SIZE_XY 0x283f8 (any name containing VRS).
  VRS fields:     PA_CL_VS_OUT_CNTL.USE_VTX_VRS_RATE (bit 28), DB_HTILE_SURFACE.VRS_HTILE_ENCODING (bits 19-20),
                  VGT_DRAW_PAYLOAD_CNTL.EN_VRS_RATE (bit 6).
The DISPATCH count (compute packets in the GFX IBs: meta clears, and the GFX10.3 VRS-rate-to-HTILE copy that
must never run here) is checked when given."""
import re, sys

ansi = re.compile(r'\x1b\[[0-9;]*m')
reg_re = re.compile(r'^[0-9a-f]{8}\s+([A-Z][A-Z0-9_]*) <- (.*)$')
unk_re = re.compile(r'^\s*(?:[0-9a-f]{8}\s+)?0x([0-9a-f]{5}) <- 0x([0-9a-f]{8})$')
pkt_re = re.compile(r'^[0-9a-f]{8} ([A-Z0-9_]+)(\(.*\))?:')
VRS_OFFSETS = {0x28848, 0x3098c, 0x28064, 0x283d0, 0x283e0, 0x283f0, 0x283f4, 0x283f8}
FIELDS = {'PA_CL_VS_OUT_CNTL': (1 << 28, 'USE_VTX_VRS_RATE'), 'DB_HTILE_SURFACE': (3 << 19, 'VRS_HTILE_ENCODING'),
          'VGT_DRAW_PAYLOAD_CNTL': (1 << 6, 'EN_VRS_RATE')}


def value(s):
    m = re.search(r'\((0x[0-9a-f]+)\)', s)
    if m:
        return int(m.group(1), 16)
    s = s.strip()
    try:
        return int(s, 0)
    except ValueError:
        return None


errs = []
ibs = writes = unknown = dispatches = draws = 0
seen = {k: 0 for k in FIELDS}
in_ib = False
gfx_ib = False
for line in open(sys.argv[1], errors='replace'):
    line = ansi.sub('', line.rstrip('\n'))
    if 'IB begin' in line:
        in_ib = True
        gfx_ib = 'GFX' in line
        ibs += 1
        continue
    if 'IB end' in line:
        in_ib = False
        continue
    if not in_ib:
        continue
    m = pkt_re.match(line)
    if m:
        if gfx_ib and m.group(1).startswith('DISPATCH'):
            dispatches += 1
        if m.group(1).startswith('DRAW') or m.group(1).startswith('DISPATCH_MESH'):
            draws += 1
        continue
    m = reg_re.match(line)
    if m:
        writes += 1
        name, v = m.group(1), value(m.group(2))
        if 'VRS' in name:
            errs.append('reg=%s' % name)
        if name in FIELDS:
            seen[name] += 1
            mask, field = FIELDS[name]
            if v is None or v & mask:
                errs.append('%s.%s(0x%s)' % (name, field, '%08x' % v if v is not None else '?'))
        continue
    m = unk_re.match(line)
    if m:
        writes += 1
        unknown += 1
        if int(m.group(1), 16) in VRS_OFFSETS:
            errs.append('offset=0x%s' % m.group(1))

if ibs == 0:
    errs.append('no-ib')
if len(sys.argv) > 2 and dispatches != int(sys.argv[2]):
    errs.append('dispatches=%d' % dispatches)
print('ibs=%d writes=%d unknown_offsets=%d draws=%d gfx_dispatches=%d vs_out_cntl=%d htile_surface=%d draw_payload=%d vrs_writes=%d %s' % (
    ibs, writes, unknown, draws, dispatches, seen['PA_CL_VS_OUT_CNTL'], seen['DB_HTILE_SURFACE'],
    seen['VGT_DRAW_PAYLOAD_CNTL'], len(errs), 'OK' if not errs else 'FAIL ' + ' '.join(sorted(set(errs)))))
sys.exit(1 if errs else 0)
