#!/usr/bin/env python3
"""check.py <on-stderr> <off-stderr> <expect> <settings>: checks one RADV_BC250_MESH_AUTOCULL case.

<on-stderr>/<off-stderr> are one run each with the switch on/off, RADV_DEBUG=dumpibs,shaders,
BC250_TRACE_COMPILE=1 and BC250_CAPTURE_MESH_CODE=1.
<expect>   yes = autocull candidate and applied (yes:app_cull: also combined with application CullPrimitive),
           and the non-culling side of the runtime branch is the switch-off epilogue (skipcmp.py);
           policy = a candidate refused by the size policy; no:<reason> = not a candidate for <reason>.
           For policy and no, the on run must be byte-identical to the off run (the
           "BC250 MESH AUTOCULL" trace lines excepted).
<settings> comma list of the NGG culling settings (hex) the Mesh draws must see, in draw order
           (only for yes cases).
Prints one key=value line and exits 1 on any mismatch."""
import re, sys

ansi = re.compile(r'\x1b\[[0-9;]*m')
reg_re = re.compile(r'^[0-9a-f]{8}\s+([A-Z0-9_]+) <- (.*)$')
pkt_re = re.compile(r'^([0-9a-f]{8}) ([A-Z0-9_]+)(\(.*\))?:')
MESH_STAGES = ('0x00092020', '0x00c92020', '0x00012020', '0x00c12020')
DRAWS = ('DRAW_INDEX_AUTO', 'DISPATCH_MESH_INDIRECT_MULTI', 'DISPATCH_DIRECT', 'DISPATCH_INDIRECT')


def val(s):
    m = re.search(r'\((0x[0-9a-f]+)\)', s)
    if m:
        return m.group(1)
    s = s.strip()
    if re.fullmatch(r'0x[0-9a-f]+', s):
        return s
    if re.fullmatch(r'\d+', s):
        return '0x%x' % int(s)
    return s


def lines(path):
    return [ansi.sub('', l.rstrip('\n')) for l in open(path, errors='replace')]


def parse_ib(ls):
    """Draw packets of the main IB: (kind, register state)."""
    state = {}; draws = []; cur = None; in_main = False
    for line in ls:
        if 'Main IB begin' in line:
            in_main = True
        if not in_main:
            continue
        m = pkt_re.match(line)
        if m:
            cur = m.group(2)
            if cur in DRAWS:
                draws.append([cur, dict(state)])
            continue
        mm = reg_re.match(line)
        if mm and cur and cur.startswith('SET_'):
            state[mm.group(1)] = val(mm.group(2))
    return draws


def mesh_isa(ls):
    out = []; on = False
    for line in ls:
        if 'Mesh Shader as NGG' in line:
            on = True
        elif on and line.startswith('shader: MESA_SHADER_FRAGMENT'):
            break
        if on:
            out.append(line)
    return out


def user_data(state):
    return {k: v for k, v in state.items() if k.startswith('SPI_SHADER_USER_DATA_GS_')}


on_path, off_path, expect, settings = sys.argv[1:5]
on, off = lines(on_path), lines(off_path)
errs = []
for tag, ls in (('on', on), ('off', off)):
    if any(re.search(r'validation failed|NIR_VALIDATE|assertion|error:|^FAIL |Segmentation', l, re.I) for l in ls):
        errs.append('validation_' + tag)

trace = [l for l in on if l.startswith('BC250 MESH AUTOCULL')]
cand = [l for l in trace if l.startswith('BC250 MESH AUTOCULL candidate:')]
applied = [l for l in trace if l.startswith('BC250 MESH AUTOCULL: applied')]
if any(l.startswith('BC250 MESH AUTOCULL') for l in off):
    errs.append('trace_in_off_run')

if expect == 'policy':
    # a candidate refused by the size policy: compiled exactly as with the switch off
    strip = lambda ls: [l for l in ls if not l.startswith('BC250 MESH AUTOCULL')]
    refused = [l for l in trace if l.startswith('BC250 MESH AUTOCULL policy: refused')]
    if not cand or not all(' yes ' in l for l in cand):
        errs.append('not_candidate')
    if not refused:
        errs.append('not_refused')
    if applied or any(l.startswith('BC250 MESH AUTOCULL: ') for l in trace):
        errs.append('lowered_with_autocull')
    if strip(on) != strip(off):
        errs.append('DIFFERS_FROM_OFF')
    shape = re.search(r'P=(\d+) wave=(\d+) lanes=(\d+)', refused[0]) if refused else None
    print('candidate=yes policy=refused %s identical_to_off=%s %s' % (
        'P=%s wave=%s lanes=%s' % shape.groups() if shape else '?', 'DIFFERS_FROM_OFF' not in errs,
        'OK' if not errs else 'FAIL ' + ' '.join(errs)))
    sys.exit(1 if errs else 0)

if expect.startswith('no'):
    want_reason = expect.split(':', 1)[1] if ':' in expect else ''
    strip = lambda ls: [l for l in ls if not l.startswith('BC250 MESH AUTOCULL')]
    if applied:
        errs.append('applied_unexpectedly')
    reasons = sorted({l.split('reason=')[1] for l in cand if 'reason=' in l})
    if want_reason and not any(want_reason in r for r in reasons):
        errs.append('reason=%s' % '|'.join(reasons))
    if strip(on) != strip(off):
        errs.append('DIFFERS_FROM_OFF')
    print('candidate=no reason="%s" identical_to_off=%s %s' % (
        '|'.join(reasons) or 'no candidate trace', 'DIFFERS_FROM_OFF' not in errs,
        'OK' if not errs else 'FAIL ' + ' '.join(errs)))
    sys.exit(1 if errs else 0)

if expect == 'yes:app_cull' and not re.search(r'app_cull=1', ' '.join(applied)):
    errs.append('no_app_cull')
if not cand or not all(' yes ' in l for l in cand):
    errs.append('not_candidate')
if not applied:
    errs.append('not_applied')
m = re.search(r'V=(\d+) P=(\d+) wave=(\d+) hw_workgroup=(\d+) lds=(\d+) app_cull=(\d)', applied[0]) if applied else None
V, P, wave, hwwg, lds_nir, app_cull = map(int, m.groups()) if m else (0, 0, 0, 0, 0, 0)
pieces = len(applied)

# ISA: the culling test, the compaction (ballot/mbcnt, bit count), 2 GS_ALLOC_REQ per piece (fully-culled + live)
isa = mesh_isa(on)
text = '\n'.join(isa)
isa_off = '\n'.join(mesh_isa(off))
allocs = text.count('sendmsg(gs_alloc_req)')
barriers = len(re.findall(r'\ts_barrier\b', text))
barriers_off = len(re.findall(r'\ts_barrier\b', isa_off))
for need, pat in (('mbcnt', r'v_mbcnt_lo'), ('bcnt', r's_bcnt1'), ('rcp', r'v_rcp_f32'),
                  ('prim_export', r'done prim'), ('pos_export', r'pos0')):
    if not re.search(pat, text):
        errs.append('isa_no_' + need)
if allocs < 2:
    errs.append('gs_alloc_req=%d' % allocs)

# Runtime skip: the non-culling side of the settings branch is the switch-off epilogue (skipcmp.py)
import os, subprocess
skip = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skipcmp.py'),
                       on_path, off_path], capture_output=True, text=True)
skip_line = skip.stdout.strip().splitlines()[-1] if skip.stdout.strip() else 'skipcmp=no output'
if skip.returncode != 0:
    errs.append('runtime_skip')
m_skip = re.search(r'skip_overhead=(\d+)', skip_line)
m_eq = re.search(r'skip_insns=(\d+) off_epilogue_insns=(\d+) skip_equals_off=(\w+)', skip_line)

# IB: mesh draws on the same launch registers as with the switch off; the NGG culling settings
# and viewport user SGPRs are written (and absent with the switch off)
d_on = [d for d in parse_ib(on) if d[1].get('VGT_SHADER_STAGES_EN') in MESH_STAGES]
d_off = [d for d in parse_ib(off) if d[1].get('VGT_SHADER_STAGES_EN') in MESH_STAGES]
if len(d_on) != len(d_off) or not d_on:
    errs.append('mesh_draws=%d/%d' % (len(d_on), len(d_off)))
for key in ('VGT_SHADER_STAGES_EN', 'GE_NGG_SUBGRP_CNTL', 'VGT_GS_MAX_VERT_OUT', 'GE_CNTL'):
    a = sorted({d[1].get(key) for d in d_on}); b = sorted({d[1].get(key) for d in d_off})
    if a != b:
        errs.append('%s=%s/%s' % (key, a, b))
stages = sorted({d[1].get('VGT_SHADER_STAGES_EN') for d in d_on})
if not stages or any(s not in ('0x00012020', '0x00c12020') for s in stages):
    errs.append('fast_launch_stages=%s' % stages)
wg = 0
amd = [l for l in on if l.startswith('BC250 MESH AMD:')]
a = re.search(r'WG=(\d+) WAVE=(\d+)', amd[0]) if amd else None
if a:
    wg = int(a.group(1))
subgrps = sorted({int(d[1].get('GE_NGG_SUBGRP_CNTL', '0'), 0) for d in d_on})
if not subgrps or min(s & 0x1ff for s in subgrps) < wg:
    errs.append('amp<wg')
want = ['0x%x' % int(s, 0) for s in settings.split(',') if s]
got = []
for i, d in enumerate(d_on):
    ud = user_data(d[1])
    vals = set(ud.values())
    w = want[i] if i < len(want) else (want[-1] if want else None)
    got.append(next((v for v in ud.values() if w and v == w), '?'))
    if w and w not in vals:
        errs.append('draw%d_settings!=%s' % (i, w))
    if '0x43000000' not in vals:
        errs.append('draw%d_no_viewport_sgpr' % i)
off_vals = set(v for d in d_off for v in user_data(d[1]).values())
for w in set(want):
    if w not in ('0x0', '0x1', '0x3') and w in off_vals:
        errs.append('settings_%s_also_off' % w)

# LDS / scratch of the uploaded Mesh binaries
code = [re.search(r'lds=(\d+) scratch=(\d+)', l) for l in on if l.startswith('BC250SHADER ') and 'lds=' in l]
lds = max((int(c.group(1)) for c in code), default=None)
scratch = max((int(c.group(2)) for c in code), default=None)
if lds is None or lds >= 32 * 1024 or scratch != 0:
    errs.append('lds=%s scratch=%s' % (lds, scratch))

print('candidate=yes applied=%d V=%d P=%d wave=%d hw_wg=%d app_cull=%d draws=%d settings=%s '
      'GE_NGG_SUBGRP_CNTL=%s fast_launch=0 gs_alloc_req=%d s_barrier=%d(off %d) lds=%s scratch=%s '
      'skip_overhead=%s skip_equals_off=%s(%s insns) %s' % (
          pieces, V, P, wave, hwwg, app_cull, len(d_on), ','.join(dict.fromkeys(got)), ','.join('0x%08x' % x for x in subgrps),
          allocs, barriers, barriers_off, lds, scratch, m_skip.group(1) if m_skip else '?',
          m_eq.group(3) if m_eq else '?', m_eq.group(1) if m_eq else '?',
          'OK' if not errs else 'FAIL ' + ' '.join(errs) + ' [' + skip_line + ']'))
sys.exit(1 if errs else 0)
