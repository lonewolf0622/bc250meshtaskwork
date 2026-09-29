#!/usr/bin/env python3
"""check.py <on-stderr> <off-stderr> <K> <S> <gx> <gy> <gz> [wave [merged_V]]: checks one
RADV_BC250_MESH_MERGE case (wave defaults to 64; merged_V 0 = not checked).

The on run uses option A indirect draws (RADV_BC250_MESH_MERGE_INDIRECT=a, the default): one prep
dispatch per indirect call, DISPATCH_MESH_INDIRECT_MULTI on the driver records (stride 32) and the
dims user SGPR (DRAW_INDEX_LOC + 1) = low 32 bits of the records (0 for direct draws). With
OPTB=<stderr of a RADV_BC250_MESH_MERGE_INDIRECT=b run> the option B run is checked too: no setup
dispatch and indirect packets byte-identical to the switch off.

<on-stderr>/<off-stderr> are one run each with the switch on/off, RADV_DEBUG=dumpibs,shaders,
BC250_TRACE_COMPILE=1, BC250_TRACE_MERGE_NIR=1 and BC250_CAPTURE_MESH_CODE=1. K=0 means "not a merge
candidate": the on run must then be byte-identical to the off run (the "BC250 MESH MERGE:" planner
trace line excepted). Prints one key=value line and exits 1 on any mismatch."""
import re, sys

ansi = re.compile(r'\x1b\[[0-9;]*m')
reg_re = re.compile(r'^[0-9a-f]{8}\s+([A-Z0-9_]+) <- (.*)$')
pkt_re = re.compile(r'^([0-9a-f]{8}) ([A-Z0-9_]+)(\(.*\))?:')
dword_re = re.compile(r'^([0-9a-f]{8})\s')
MESH_STAGES = ('0x00092020', '0x00c92020', '0x00012020', '0x00c12020')
DRAWS = ('DRAW_INDEX_AUTO', 'DISPATCH_MESH_INDIRECT_MULTI', 'DISPATCH_DIRECT', 'DISPATCH_INDIRECT')


def val(s):
    m = re.search(r'\((0x[0-9a-f]+)\)', s)
    return m.group(1) if m else s.strip()


def lines(path):
    return [ansi.sub('', l.rstrip('\n')) for l in open(path, errors='replace')]


def parse_ib(ls):
    """Draw packets of the main IB: (kind, register state, raw dwords, INDEX_COUNT)."""
    state = {}; draws = []; cur = None; raw = None; in_main = False
    for line in ls:
        if 'Main IB begin' in line:
            in_main = True
        if not in_main:
            continue
        m = pkt_re.match(line)
        if m:
            cur = m.group(2)
            raw = [m.group(1)]
            if cur in DRAWS:
                draws.append([cur, dict(state), raw, None])
            continue
        d = dword_re.match(line)
        if d and raw is not None:
            raw.append(d.group(1))
        mm = reg_re.match(line)
        if mm and cur:
            if cur.startswith('SET_'):
                state[mm.group(1)] = val(mm.group(2))
            elif cur == 'DRAW_INDEX_AUTO' and mm.group(1) == 'INDEX_COUNT':
                draws[-1][3] = int(val(mm.group(2)), 0)
    return draws


field_re = re.compile(r'^(?:[0-9a-f]{8})?\s+(?:\[[AB]\])?([A-Z0-9_]+) (?:<-|=) (.*)$')


def packets(ls):
    """All packets of the main IB: [name, {field: value}, raw dwords]."""
    out = []; in_main = False
    for line in ls:
        if 'Main IB begin' in line:
            in_main = True
        if not in_main:
            continue
        m = pkt_re.match(line)
        if m:
            out.append([m.group(2), {}, [m.group(1)]])
            continue
        if not out:
            continue
        d = dword_re.match(line)
        if d:
            out[-1][2].append(d.group(1))
        f = field_re.match(line)
        if f:
            out[-1][1].setdefault(f.group(1), f.group(2).strip())
    return out


def num(v):
    m = re.search(r'\((0x[0-9a-f]+)\)', v)
    return int(m.group(1) if m else v.split()[0], 0)


def option_a(on_ls, off_ls, K, errs):
    """Option A indirect: per indirect call one prep DISPATCH_DIRECT, CS_PARTIAL_FLUSH + K$ invalidation
    (no L2 invalidation) before the draw, SET_BASE on the driver records, dims SGPR = their low 32
    bits, stride 32, the application's count/count buffer. Returns a summary string."""
    pk = packets(on_ls)
    pk_off = packets(off_ls)
    ind = [i for i, p in enumerate(pk) if p[0] == 'DISPATCH_MESH_INDIRECT_MULTI']
    ind_off = [p for p in pk_off if p[0] == 'DISPATCH_MESH_INDIRECT_MULTI']
    preps = [i for i, p in enumerate(pk) if p[0] == 'DISPATCH_DIRECT']
    if len(ind) != 2 or len(ind_off) != 2 or len(preps) != 2:
        errs.append('optA_packets ind=%d prep=%d' % (len(ind), len(preps)))
        return ''
    gs = {}
    dims_reg = None; prev = 0; info = []
    for n, (i, off) in enumerate(zip(ind, ind_off)):
        p = pk[i]
        prep = [j for j in preps if prev <= j < i]
        if len(prep) != 1:
            errs.append('optA_prep_per_call')
            continue
        between = pk[prep[0] + 1:i]
        if not any(q[0] == 'EVENT_WRITE' and 'CS_PARTIAL_FLUSH' in q[1].get('EVENT_TYPE', '') for q in between):
            errs.append('optA_no_cs_partial_flush')
        acq = [q for q in between if q[0] == 'ACQUIRE_MEM']
        if not acq or any(q[1].get('GL2_INV', '0') != '0' or q[1].get('GL2_WB', '0') != '0' for q in acq) or \
           not any(q[1].get('GLK_INV') == '1' for q in acq):
            errs.append('optA_cache_ops')
        # K in the prep push constants (records, stride, K after three 64-bit addresses)
        ud = pk[prep[0] - 1][1] if prep[0] else {}
        if ud.get('COMPUTE_USER_DATA_10') is None or num(ud['COMPUTE_USER_DATA_10']) != K:
            errs.append('optA_prep_k')
        # one prep invocation per record (64 per workgroup), the application's count buffer
        cnt = num(p[1].get('COUNT', '0'))
        if num(pk[prep[0]][1].get('DIM_X', '0')) != -(-cnt // 64) or \
           num(ud.get('COMPUTE_USER_DATA_8', '0')) != cnt:
            errs.append('optA_prep_size')
        want_cva = (num(p[1]['COUNT_ADDR_LO']), num(p[1]['COUNT_ADDR_HI'])) if p[1].get('COUNT_INDIRECT_ENABLE') == '1' else (0, 0)
        if (num(ud.get('COMPUTE_USER_DATA_6', '0')), num(ud.get('COMPUTE_USER_DATA_7', '0'))) != want_cva:
            errs.append('optA_prep_count_addr')
        base = [q for q in between if q[0] == 'SET_BASE']
        lo = num(base[-1][1]['ADDRESS_LO']) if base else None
        dil = num(p[1]['DRAW_INDEX_LOC']) if 'DRAW_INDEX_LOC' in p[1] else None
        if dil is None or p[1].get('DRAW_INDEX_ENABLE') != '1':
            errs.append('optA_no_draw_index')
        else:
            dims_reg = 'SPI_SHADER_USER_DATA_GS_%d' % (dil + 1 - 0x8c)
            w = [q for q in between if q[0] == 'SET_SH_REG' and dims_reg in q[1]]
            if not w or num(w[-1][1][dims_reg]) != lo or not lo:
                errs.append('optA_dims_sgpr')
        base_off = [q for q in pk_off[:pk_off.index(off)] if q[0] == 'SET_BASE']
        lo_off = num(base_off[-1][1]['ADDRESS_LO']) if base_off else None
        if lo == lo_off:
            errs.append('optA_reads_app_buffer')
        if num(p[1]['STRIDE']) != 32:
            errs.append('optA_stride')
        for f in ('COUNT', 'COUNT_ADDR_LO', 'COUNT_ADDR_HI', 'COUNT_INDIRECT_ENABLE', 'XYZ_DIM_LOC'):
            if p[1].get(f) != off[1].get(f):
                errs.append('optA_%s' % f)
        info.append('%s:%s' % (p[1].get('COUNT', '?').split()[0], 'cnt' if p[1].get('COUNT_INDIRECT_ENABLE') == '1' else 'ind'))
        prev = i
    # the direct draw zeroes the dims SGPR (the grid SGPRs hold the application's grid)
    direct = [i for i, p in enumerate(pk) if p[0] == 'DRAW_INDEX_AUTO']
    if dims_reg and direct:
        w = [q for q in pk[:direct[0]] if q[0] == 'SET_SH_REG' and dims_reg in q[1]]
        if not w or num(w[-1][1][dims_reg]) != 0:
            errs.append('optA_direct_dims_not_zero')
    return 'optA=%s dims_reg=%s' % (','.join(info), dims_reg)


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


def merged_nir(ls):
    out = []; on = False
    for line in ls:
        if line.startswith('BC250 MERGED NIR BEGIN'):
            on = True
        elif line.startswith('BC250 MERGED NIR END'):
            break
        elif on:
            out.append(line)
    return out


on_path, off_path, K, S, gx, gy, gz = sys.argv[1:8]
K, S, n = int(K), int(S), int(gx) * int(gy) * int(gz)
WAVE = int(sys.argv[8]) if len(sys.argv) > 8 else 64
VM = int(sys.argv[9]) if len(sys.argv) > 9 else 0
on, off = lines(on_path), lines(off_path)
errs = []
for tag, ls in (('on', on), ('off', off)):
    if any(re.search(r'validation failed|NIR_VALIDATE|assertion|error:|^FAIL ', l, re.I) for l in ls):
        errs.append('validation_' + tag)

if K == 0:
    strip = lambda ls: [l for l in ls if not l.startswith('BC250 MESH MERGE:')]
    planner = [l for l in on if l.startswith('BC250 MESH MERGE:')]
    if any(l.startswith('BC250 MESH MERGE: merged ') for l in planner):
        errs.append('merged_unexpectedly')
    if strip(on) != strip(off):
        errs.append('DIFFERS_FROM_OFF')
    why = planner[0].split('reason=')[1] if planner and 'reason=' in planner[0] else 'no planner call'
    print('candidate=no identical_to_off=%s planner="%s" %s' % (
        'DIFFERS_FROM_OFF' not in errs, why, 'OK' if not errs else 'FAIL ' + ' '.join(errs)))
    sys.exit(1 if errs else 0)

plan = [l for l in on if l.startswith('BC250 MESH MERGE: merged')]
m = re.search(r'K=(\d+) S=(\d+) L=\d+ V=(\d+) P=(\d+) merged_V=(\d+) merged_P=(\d+) lanes=(\d+)', plan[0]) if plan else None
if not m:
    print('FAIL not merged: ' + ' '.join(l for l in on if l.startswith('BC250 MESH MERGE')))
    sys.exit(1)
k, s, v, p, vm, kp, lanes = map(int, m.groups())
if (k, s) != (K, S):
    errs.append('plan=K%dS%d' % (k, s))
if VM and vm != VM:
    errs.append('merged_V=%d(want %d)' % (vm, VM))
stage2 = ' stage2=1 ' in plan[0]
prov = re.search(r'provoking=(\w+)', plan[0]).group(1) if stage2 else 'none'
if vm > 256 or lanes > 256:
    errs.append('physical_vertices=%d lanes=%d' % (vm, lanes))
amd = [l for l in on if l.startswith('BC250 MESH AMD:')]
a = re.search(r'fl0=(\d) route=(\d) size=(\d) .* WG=(\d+) WAVE=(\d+) .*GE_NGG_SUBGRP_CNTL=([0-9a-f]+) VGT_GS_MAX_VERT_OUT=(\d+) '
              r'merge_k=(\d+) merge_s=(\d+) scratch_ring=(\d)', amd[0]) if amd else None
if not a:
    errs.append('no_amd_trace'); wg = wave = 0
else:
    fl0, route, size, wg, wave, subgrp, maxvert, mk, ms, ring = a.groups()
    wg, wave, maxvert, subgrp = int(wg), int(wave), int(maxvert), int(subgrp, 16)
    if (fl0, route, size) != ('1', '1', '0'): errs.append('fl0/route/size=%s/%s/%s' % (fl0, route, size))
    if (int(mk), int(ms)) != (K, S): errs.append('info_merge=%s/%s' % (mk, ms))
    if ring != '0': errs.append('scratch_ring')
    if wave != WAVE: errs.append('wave=%d' % wave)
    if wg != lanes or wg > (256 if stage2 else 128): errs.append('wg=%d' % wg)

# IB: direct DRAW_INDEX_AUTO ceil(N/K), indirect packets as with the switch off, fast launch 0
draws_on, draws_off = parse_ib(on), parse_ib(off)
mesh_on = [d for d in draws_on if d[1].get('VGT_SHADER_STAGES_EN') in MESH_STAGES]
mesh_off = [d for d in draws_off if d[1].get('VGT_SHADER_STAGES_EN') in MESH_STAGES]
kinds = ','.join(d[0] for d in mesh_on if d[0] != 'DISPATCH_DIRECT')
if kinds != 'DRAW_INDEX_AUTO,DISPATCH_MESH_INDIRECT_MULTI,DISPATCH_MESH_INDIRECT_MULTI':
    errs.append('mesh_draws=' + kinds)
if any(d[0] == 'DISPATCH_INDIRECT' for d in draws_on):
    errs.append('setup_dispatch_indirect')
direct = mesh_on[0][3] if mesh_on else None
want = -(-n // K)
if direct != want:
    errs.append('direct_count=%s(want %d)' % (direct, want))
opta = option_a(on, off, K, errs)
optb = ''
import os
if os.environ.get('OPTB'):
    # option B (RADV_BC250_MESH_MERGE_INDIRECT=b): N groups, indirect packets as with the switch off
    ob = lines(os.environ['OPTB'])
    draws_b = parse_ib(ob)
    if any(d[0] in ('DISPATCH_DIRECT', 'DISPATCH_INDIRECT') for d in draws_b):
        errs.append('optB_setup_dispatch')
    ind_b = [d[2] for d in draws_b if d[0] == 'DISPATCH_MESH_INDIRECT_MULTI' and d[1].get('VGT_SHADER_STAGES_EN') in MESH_STAGES]
    ind_off = [d[2] for d in mesh_off if d[0] == 'DISPATCH_MESH_INDIRECT_MULTI']
    if not ind_b or ind_b != ind_off:
        errs.append('optB_indirect_packets_differ')
    if any('@load_user_data_amd' in l for l in merged_nir(ob)):
        errs.append('optB_dims')
    optb = ' optB=ok' if not any(e.startswith('optB') for e in errs) else ' optB=FAIL'
stages = sorted({d[1].get('VGT_SHADER_STAGES_EN') for d in mesh_on})
# GS_FAST_LAUNCH=0; wave32 merged shaders also set the VS/GS wave32 enables (0x00c00000)
if stages != ['0x00012020' if WAVE == 64 else '0x00c12020']:
    errs.append('stages_en=' + ','.join(map(str, stages)))
subgrps = sorted({int(d[1].get('GE_NGG_SUBGRP_CNTL', '0'), 0) for d in mesh_on})
amp = subgrps[0] & 0x1ff if len(subgrps) == 1 else -1
thds = (subgrps[0] >> 9) & 0x1ff if len(subgrps) == 1 else -1
if amp < wg or thds != 0:
    errs.append('subgrp=' + ','.join('0x%08x' % x for x in subgrps))
maxverts = sorted({int(d[1].get('VGT_GS_MAX_VERT_OUT', '0'), 0) for d in mesh_on})
if maxverts != [wg]:
    errs.append('maxvert=' + ','.join(map(str, maxverts)))

# ISA (ACO): one GS_ALLOC_REQ per path (fully-culled workaround + live), live counts K*V+3 / K*P
isa = mesh_isa(on)
allocs = sum('sendmsg(gs_alloc_req)' in l for l in isa)
barriers = sum(re.search(r'\ts_barrier\b', l) is not None for l in isa)
text = '\n'.join(isa)
if allocs != 2: errs.append('gs_alloc_req=%d' % allocs)
for lit in (kp, vm):
    if not re.search(r'(0x%x|\b%d)\b' % (lit, lit), text):
        errs.append('no_literal_%d' % lit)

# merged NIR: one SetMeshOutputs, no leftover workgroup id / local id, the hardware workgroup index
nir = merged_nir(on)
nt = '\n'.join(nir)
svpc = nt.count('@set_vertex_and_primitive_count')
if svpc != 1: errs.append('set_vertex_and_primitive_count=%d' % svpc)
for bad in ('@load_workgroup_id', '@load_local_invocation_id', '@load_subgroup_id', '@load_num_subgroups'):
    if bad in nt: errs.append('leftover_' + bad[1:])
if '@load_workgroup_index' not in nt: errs.append('no_workgroup_index')
# option A: the grid comes from the driver record (scalar load at dims + 32*DrawID + 16) when the dims
# SGPR is nonzero, else from the grid SGPRs
if '@load_user_data_amd' not in nt or '@load_draw_id' not in nt or '@load_global' not in nt:
    errs.append('optA_nir')
# every application read of gl_NumWorkGroups goes through the selected grid: one grid SGPR read left
if nt.count('@load_num_workgroups') != 1:
    errs.append('optA_num_workgroups=%d' % nt.count('@load_num_workgroups'))
if not re.search(r's_load_dwordx(2|4)', text):
    errs.append('optA_no_scalar_load')
if 'max_vertices_out: %d' % vm not in nt or 'max_primitives_out: %d' % kp not in nt:
    errs.append('nir_counts')

# stage 2: every output is re-emitted by its own lane (output derefs indexed by the local invocation
# index only), only the primitive indices stay per-primitive, and each former per-primitive output is
# stored only under "not a representative slot" (t >= V-1): the provoking-vertex record.
pp_stores = 0
if stage2:
    defs = {}
    for l in nir:
        d = re.match(r'\s*(?:\d+(?:x\d+)?)?\s+(%\d+) = (.*)$', l)
        if d:
            defs[d.group(1)] = d.group(2)
    lane = [x for x, e in defs.items() if e.startswith('@load_local_invocation_index')]
    outs = [l for l in nir if 'deref_array' in l and '(shader_out' in l]
    if len(lane) != 1 or any(not re.search(r'\[%s\]' % re.escape(lane[0]), l) for l in outs):
        errs.append('cross_lane_output')
    decl = [l for l in nir if l.startswith('decl_var') and ' shader_out ' in l]
    if any('per_primitive' in l and 'PRIMITIVE_INDICES' not in l for l in decl):
        errs.append('per_primitive_output_left')
    flat_pp = []
    for l in nir:
        mm = re.match(r'decl_var per_primitive shared \S+ \S+ \S+ (\S+)#\d+', l)
        if mm and 'gl_PrimitiveTriangleIndicesEXT' not in mm.group(1) and 'CullPrimitive' not in mm.group(1):
            flat_pp.append(mm.group(1))
    for name in flat_pp:
        if not any(re.match(r'decl_var shader_out INTERP_MODE_FLAT .* %s \(' % re.escape(name), l) for l in decl):
            errs.append('not_flat_' + name)
        for i, l in enumerate(nir):
            if not re.search(r'deref_var &%s \(shader_out' % re.escape(name), l):
                continue
            pp_stores += 1
            ind = len(l) - len(l.lstrip())
            cond = None
            for j in range(i - 1, -1, -1):
                lj = nir[j]
                if lj.strip().startswith('if ') and len(lj) - len(lj.lstrip()) < ind:
                    cond = lj.split()[1]
                    break
            # the enclosing condition, through inot pairs, must be "has a provoking candidate and
            # slot(prov j) == slot": iand of an ieq (bc250_merge_pp_epilogue's zipper inverse)
            e, neg = defs.get(cond, '') if cond else '', 0
            while e.startswith('inot '):
                neg ^= 1
                e = defs.get(e.split()[1], '')
            ok = neg == 0 and e.startswith('iand ') and any(
                defs.get(o.rstrip(','), '').startswith('ieq ') for o in e.split()[1:3])
            if not ok:
                errs.append('pp_store_not_provoking_' + name)
    if prov != 'none' and not flat_pp:
        errs.append('no_per_primitive_data')
    # the fragment shader reads the former per-primitive inputs as flat attributes, exactly like
    # the base driver's expansion (switch off): same SPI_PS_INPUT_CNTL_n, no per-primitive FS input left
    spi = lambda ls: sorted({l.split()[1] + '=' + l.split('<-')[1].strip() for l in ls if 'SPI_PS_INPUT_CNTL_' in l})
    if spi(on) != spi(off):
        errs.append('spi_ps_input_cntl_differs')
    fs_hdr = False
    for l in on:
        if l.startswith('shader: MESA_SHADER_FRAGMENT'):
            fs_hdr = True
        elif l.startswith('shader: '):
            fs_hdr = False
        elif fs_hdr and l.startswith('per_primitive_inputs'):
            errs.append('fs_per_primitive_input')
            break

# LDS / scratch of the uploaded Mesh binary
code = [l for l in on if l.startswith('BC250SHADER ')]
lds = scratch = None
if code:
    c = re.search(r'lds=(\d+) scratch=(\d+)', code[0])
    lds, scratch = int(c.group(1)), int(c.group(2))
if lds is None or lds >= 32 * 1024 or scratch != 0:
    errs.append('lds=%s scratch=%s' % (lds, scratch))

print('candidate=yes K=%d S=%d V=%d P=%d merged_V=%d merged_P=%d lanes=%d wg=%d wave=%d N=%d direct_count=%s '
      'GE_NGG_SUBGRP_CNTL=%s VGT_GS_MAX_VERT_OUT=%s gs_alloc_req=%d s_barrier=%d lds=%s scratch=%s%s %s' % (
          k, s, v, p, vm, kp, lanes, wg, wave, n, direct, ','.join('0x%08x' % x for x in subgrps),
          ','.join(map(str, maxverts)), allocs, barriers, lds, scratch,
          ((' stage2=1 provoking=%s pp_stores=%d' % (prov, pp_stores)) if stage2 else '') + ' ' + opta + optb,
          'OK' if not errs else 'FAIL ' + ' '.join(errs)))
sys.exit(1 if errs else 0)
