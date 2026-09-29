#!/usr/bin/env python3
"""check.py <ib-stderr> <route amd|base> <subgrp> <maxvert>: checks one RADV_BC250_MESH_AMD dump.
Prints one key=value line; exits 1 on any mismatch."""
import re, sys
ansi = re.compile(r'\x1b\[[0-9;]*m')
reg_re = re.compile(r'^[0-9a-f]{8}\s+([A-Z0-9_]+) <- (.*)$')
pkt_re = re.compile(r'^[0-9a-f]{8} ([A-Z0-9_]+)(\(.*\))?:')
DRAWS = ('DRAW_INDEX_AUTO', 'DISPATCH_MESH_INDIRECT_MULTI', 'DISPATCH_DIRECT', 'DISPATCH_INDIRECT')
path, route, want_subgrp, want_maxvert = sys.argv[1:5]
def val(s):
    m = re.search(r'\((0x[0-9a-f]+)\)', s)
    return m.group(1) if m else s.strip()
state = {}; ev = []; draws = []; cur = None; in_main = False; last_reuse = None
for line in open(path, errors='replace'):
    line = ansi.sub('', line.rstrip('\n'))
    if 'Main IB begin' in line: in_main = True
    if not in_main: continue
    m = pkt_re.match(line)
    if m:
        cur = m.group(1)
        if cur in DRAWS:
            draws.append((cur, dict(state), ev)); ev = []
        elif not cur.startswith('SET_'):
            ev.append(cur)
        continue
    if 'EVENT_TYPE =' in line: ev.append(line.split('=')[1].strip())
    mm = reg_re.match(line)
    if mm and cur and cur.startswith('SET_'):
        state[mm.group(1)] = val(mm.group(2))
        if mm.group(1) == 'VGT_REUSE_OFF': last_reuse = val(mm.group(2))
def num(v):
    try: return int(v, 0)
    except Exception: return None
mesh = [(k, s, e) for k, s, e in draws if s.get('VGT_SHADER_STAGES_EN') in ('0x00092020', '0x00c92020', '0x00012020', '0x00c12020')]
other = [(k, s, e) for k, s, e in draws if k == 'DRAW_INDEX_AUTO' and (k, s, e) not in mesh]
disp = [d for d in draws if d[0].startswith('DISPATCH_DIRECT') or d[0] == 'DISPATCH_INDIRECT']
errs = []
kinds = ','.join(k for k, _, _ in mesh)
if kinds != 'DRAW_INDEX_AUTO,DISPATCH_MESH_INDIRECT_MULTI,DISPATCH_MESH_INDIRECT_MULTI':
    errs.append('mesh_draws=' + kinds)
if disp: errs.append('setup_dispatches=%d' % len(disp))
subgrps = sorted({s.get('GE_NGG_SUBGRP_CNTL') for _, s, _ in mesh})
maxverts = sorted({s.get('VGT_GS_MAX_VERT_OUT') for _, s, _ in mesh})
if [num(x) for x in subgrps] != [int(want_subgrp, 0)]: errs.append('subgrp=' + ','.join(map(str, subgrps)))
if [num(x) for x in maxverts] != [int(want_maxvert, 0)]: errs.append('maxvert=' + ','.join(map(str, maxverts)))
reuse = sorted({str(s.get('VGT_REUSE_OFF')) for _, s, _ in mesh})
vs_reuse = sorted({str(s.get('VGT_REUSE_OFF')) for _, s, _ in other})
# events recorded between consecutive draws: a post-Mesh VGT_FLUSH shows up before the next draw
flushes = sum(1 for i, d in enumerate(draws[1:], 1) if 'VGT_FLUSH' in d[2] and
              draws[i - 1][1].get('VGT_SHADER_STAGES_EN') in ('0x00092020', '0x00c92020', '0x00012020', '0x00c12020'))
if route == 'amd':
    if reuse != ['1']: errs.append('mesh_reuse_off=' + ','.join(reuse))
    if vs_reuse != ['0']: errs.append('vs_reuse_off=' + ','.join(vs_reuse))
    if flushes: errs.append('post_mesh_vgt_flush=%d' % flushes)
    if last_reuse != '0': errs.append('end_reuse_off=%s' % last_reuse)
else:
    if reuse != ['None'] or vs_reuse != ['None']: errs.append('reuse_off_written')
print('mesh_draws=%d setup_dispatches=%d GE_NGG_SUBGRP_CNTL=%s VGT_GS_MAX_VERT_OUT=%s VGT_REUSE_OFF(mesh/vs/end)=%s/%s/%s post_mesh_vgt_flush=%d %s' % (
    len(mesh), len(disp), ','.join(map(str, subgrps)), ','.join(map(str, maxverts)), ','.join(reuse), ','.join(vs_reuse),
    last_reuse, flushes, 'OK' if not errs else 'FAIL ' + ' '.join(errs)))
sys.exit(1 if errs else 0)
