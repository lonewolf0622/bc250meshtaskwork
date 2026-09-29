#!/usr/bin/env python3
"""check.py <on-stderr> <off-stderr> <brackets> <route> <timer-file> [skip|sample|secondary]

Checks one BC250_MESH_TIMER case (RADV_DEBUG=dumpibs on both runs, timer on in <on-stderr> with
BC250_MESH_TIMER_FAKE=1, BC250_MESH_TIMER_INTERVAL=0 and BC250_MESH_TIMER_FILE=<timer-file>).

<brackets>   timestamp pairs expected around Mesh draws per submitted command buffer (4 = every draw of
             timer.c, 2 = BC250_MESH_TIMER_SAMPLE=2, 0 = the class is skipped)
<route>      expected route= of the class line (expanded, split=3+expanded, task-replay+expanded, raw, ...)
<timer-file> the file the on run appended to (I/C lines with the fake timestamps)
[mode]       skip: the case skips the Mesh draws (no Mesh packets at all, K line, "skipping" once);
             sample: BC250_MESH_TIMER_SAMPLE, untimed Mesh draws lie outside the brackets;
             secondary: the Mesh draws are in a secondary command buffer (its IB2 is not in the dump):
             the primary shows the begin/end pair around INDIRECT_BUFFER and no Mesh draw; the file
             still counts and times the draws (entries absorbed by vkCmdExecuteCommands)

Per command buffer of the on run: the first timer packet is the command buffer's begin timestamp
(ring slot t0, before any draw), the last its end timestamp (the same slot's t1, after every draw);
every Mesh draw (DRAW_INDEX_AUTO with a Mesh VGT_SHADER_STAGES_EN, DISPATCH_MESH_INDIRECT_MULTI, and
The base driver's DISPATCH_DIRECT/INDIRECT prep dispatches) lies inside exactly one draw bracket [t0 .. t1] of its
own slot; brackets do not nest; the VS draw is outside every bracket; the number of brackets is
<brackets>. With the timer packets removed, the on run's packet stream equals the off run's, except
for address-valued fields (the timer's ring buffer shifts the shim's virtual addresses) and the IB
chaining packets (NOP padding + INDIRECT_BUFFER, which move with the extra packets). The timer
file has one I line per submit with the fake sums and a C line whose draws/timed/sum match.
Prints one key=value line and exits 1 on any mismatch."""
import re, sys

ansi = re.compile(r'\x1b\[[0-9;]*m')
pkt_re = re.compile(r'^([0-9a-f]{8}) ([A-Z0-9_]+)(\(.*\))?:')
reg_re = re.compile(r'^(?:[0-9a-f]{8})?\s+(?:\[[A-C]\])?([A-Z0-9_]+) <- (.*)$')
MESH_STAGES = {'0x00092020', '0x00c92020', '0x00012020', '0x00c12020'}
MESH_DRAWS = ('DISPATCH_MESH_INDIRECT_MULTI', 'DISPATCH_DIRECT', 'DISPATCH_INDIRECT')
ADDRESS_WORDS = ('ADDR', 'ADDRESS', 'BASE', '_LO', '_HI', 'USER_DATA', 'DATA_LO', 'DATA_HI', 'PGM_LO', 'PGM_HI',
                 'IB_BASE', 'INDIRECT_BUFFER', 'SRC_', 'DST_', 'CP_DMA', 'CMP_DATA', 'CTXID', 'MEM_', 'CMASK', 'FMASK', 'DCC')


def lines(path):
    return [ansi.sub('', l.rstrip('\n')) for l in open(path, errors='replace')]


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


def parse(ls, bo_va, slots):
    """Per submitted command buffer ("Main IB begin" .. next): a list of events
    ('T', slot_va) | ('MESH', name) | ('VS', name) | ('OTHER', name), and the raw packet lines."""
    cbs = []
    cur = None
    state = {}
    pkt = None
    rel = {}
    for line in ls:
        if 'Main IB begin' in line:
            if cur is not None and pkt == 'RELEASE_MEM':
                finish_release(cur, rel, bo_va, slots)
            cur = {'events': [], 'packets': []}
            cbs.append(cur)
            state = {}
            pkt = None
            continue
        if cur is None:
            continue
        m = pkt_re.match(line)
        if m:
            if pkt == 'RELEASE_MEM':
                finish_release(cur, rel, bo_va, slots)
            pkt = m.group(2)
            rel = {}
            cur['packets'].append([pkt, []])
            if pkt in MESH_DRAWS:
                cur['events'].append(('MESH', pkt))
            elif pkt == 'DRAW_INDEX_AUTO' or pkt.startswith('DRAW_'):
                stages = state.get('VGT_SHADER_STAGES_EN', '')
                cur['events'].append(('MESH' if stages in MESH_STAGES else 'VS', pkt))
            continue
        mm = reg_re.match(line)
        if mm and pkt:
            name, value = mm.group(1), mm.group(2)
            cur['packets'][-1][1].append((name, val(value)))
            if pkt.startswith('SET_'):
                state[name] = val(value)
            if pkt == 'RELEASE_MEM':
                rel[name] = val(value)
    if cur is not None and pkt == 'RELEASE_MEM':
        finish_release(cur, rel, bo_va, slots)
    return cbs


def finish_release(cur, rel, bo_va, slots):
    if rel.get('DATA_SEL') != 'SEND_GPU_CLOCK_COUNTER':
        return
    lo = int(rel.get('ADDRESS_LO_32B', '0x0'), 16)
    hi = int(rel.get('ADDRESS_HI', '0x0'), 16)
    va = (hi << 32) | lo
    if bo_va <= va < bo_va + slots * 16:
        cur['events'].append(('T', va))
        cur['packets'][-1][0] = 'TIMER'


def check_cb(cb, brackets, errs, tag, sample=False):
    ev = cb['events']
    ts = [e for e in ev if e[0] == 'T']
    if not ts:
        errs.append(tag + '_no_timestamps')
        return
    # command buffer begin/end
    first, last = ts[0][1], ts[-1][1]
    if first % 16 != 0 or last != first + 8:
        errs.append(tag + '_cb_slot')
    if ev[0][0] != 'T' or ev[0][1] != first:
        errs.append(tag + '_begin_not_first')
    if ev[-1][0] != 'T' or ev[-1][1] != last:
        errs.append(tag + '_end_not_last')
    # draw brackets
    open_slot = None
    n = 0
    mesh_inside = 0
    for e in ev[1:-1]:
        if e[0] == 'T':
            if open_slot is None:
                if e[1] % 16 != 0:
                    errs.append(tag + '_odd_t0')
                open_slot = e[1]
                mesh_inside = 0
            else:
                if e[1] != open_slot + 8:
                    errs.append(tag + '_t1_slot')
                if not mesh_inside:
                    errs.append(tag + '_empty_bracket')
                open_slot = None
                n += 1
        elif e[0] == 'MESH':
            if open_slot is None and not sample:
                errs.append(tag + '_mesh_outside')
            mesh_inside += 1
        elif e[0] == 'VS':
            if open_slot is not None:
                errs.append(tag + '_vs_inside')
    if open_slot is not None:
        errs.append(tag + '_unclosed')
    if brackets is not None and n != brackets:
        errs.append('%s_brackets=%d' % (tag, n))
    return n


def masked(packets, drop_timer):
    out = []
    for name, regs in packets:
        if name == 'TIMER' and drop_timer:
            continue
        if name in ('NOP', 'INDIRECT_BUFFER'):
            continue  # IB chaining (padding + chain packet) moves with the extra timer packets
        out.append(name)
        for rn, rv in regs:
            if any(w in rn for w in ADDRESS_WORDS) or name in ('DMA_DATA', 'WRITE_DATA', 'COPY_DATA', 'RELEASE_MEM',
                                                               'ACQUIRE_MEM', 'EVENT_WRITE_EOP', 'WAIT_REG_MEM',
                                                               'COND_EXEC', 'SET_BASE', 'INDIRECT_BUFFER',
                                                               'DISPATCH_MESH_INDIRECT_MULTI', 'DISPATCH_INDIRECT',
                                                               'PFP_SYNC_ME', 'NOP', 'SET_SH_REG'):
                out.append(rn + '=*')
            else:
                out.append(rn + '=' + rv)
    return out


on_path, off_path, brackets, route, timer_file = sys.argv[1:6]
mode = sys.argv[6] if len(sys.argv) > 6 else ''
skip = mode == 'skip'
sample = mode == 'sample'
secondary = mode == 'secondary'
brackets = int(brackets)
on, off = lines(on_path), lines(off_path)
errs = []
for tag, ls in (('on', on), ('off', off)):
    if any(re.search(r'validation failed|NIR_VALIDATE|assertion|error:|^FAIL |Segmentation', l, re.I) for l in ls):
        errs.append('validation_' + tag)

hdr = [l for l in on if re.match(r'# BC250_MESH_TIMER v[12] ', l)]
if not hdr:
    print('error=no_timer_header'); sys.exit(1)
bo_va = int(re.search(r'bo_va=0x([0-9a-f]+)', hdr[0]).group(1), 16)
slots = int(re.search(r'slots=(\d+)', hdr[0]).group(1))

cbs_on = parse(on, bo_va, slots)
cbs_off = parse(off, bo_va, slots)
if len(cbs_on) != len(cbs_off) or not cbs_on:
    errs.append('cb_count_on=%d_off=%d' % (len(cbs_on), len(cbs_off)))
if any(e[0] == 'T' for cb in cbs_off for e in cb['events']):
    errs.append('off_has_timer_packets')

nb = []
for i, cb in enumerate(cbs_on):
    if secondary:
        nb.append(check_cb(cb, 0, errs, 'cb%d' % i))
        if any(e[0] == 'MESH' or e[0] == 'VS' for e in cb['events']):
            errs.append('cb%d_secondary_draw_in_primary' % i)
        if not any(n == 'INDIRECT_BUFFER' for n, _ in cb['packets']):
            errs.append('cb%d_no_ib2' % i)
        continue
    nb.append(check_cb(cb, brackets, errs, 'cb%d' % i, sample))
    if skip:
        if any(e[0] == 'MESH' for e in cb['events']):
            errs.append('cb%d_skip_mesh_present' % i)
        if not any(e[0] == 'VS' for e in cb['events']):
            errs.append('cb%d_skip_vs_missing' % i)
    else:
        # the same draws as off (the skip case removes draws by design)
        mo = [e for e in cb['events'] if e[0] != 'T']
        mf = [e for e in cbs_off[i]['events'] if e[0] != 'T'] if i < len(cbs_off) else None
        if mo != mf:
            errs.append('cb%d_draw_sequence_differs' % i)
        if i < len(cbs_off) and masked(cb['packets'], True) != masked(cbs_off[i]['packets'], False):
            errs.append('cb%d_packets_differ' % i)

# timer output
skipping = [l for l in on if l.startswith('BC250_MESH_TIMER skipping')]
if skip and len(skipping) != 1:
    errs.append('skipping_lines=%d' % len(skipping))
if not skip and skipping:
    errs.append('unexpected_skipping')
tf = lines(timer_file)
I = [l for l in tf if l.startswith('I interval')]
C = [l for l in tf if l.startswith('C ')]
K = [l for l in tf if l.startswith('K ')]
F = [l for l in tf if l.startswith('I final')]
if len(I) != len(cbs_on):
    errs.append('I_lines=%d' % len(I))
if len(F) != 1:
    errs.append('final_lines=%d' % len(F))
if skip and len(K) != 1:
    errs.append('K_lines=%d' % len(K))
routes = set(re.search(r'route=(\S+)', l).group(1) for l in C)
if routes != {route}:
    errs.append('route=%s' % ','.join(sorted(routes)) if routes else 'route=none')


def kv(line):
    return dict(m.split('=', 1) for m in line.split()[1:] if '=' in m)


for i, l in enumerate(I):
    d = kv(l)
    if int(d['submits']) != 1 or int(d['mesh_draws']) != 4 or int(d['vs_draws']) != 1:
        errs.append('I%d_counts' % i)
    if int(d['timed']) != brackets:
        errs.append('I%d_timed=%s' % (i, d['timed']))
    if int(d['dropped_slots']) or int(d['dropped_submits']) or int(d['invalid']):
        errs.append('I%d_drops' % i)
    if skip and int(d['skipped']) != 4:
        errs.append('I%d_skipped=%s' % (i, d['skipped']))
# C lines of the interval prints (one class): draws 4 per submit (2 direct, 1 indirect, 1 count)
Ci = [kv(x) for x in C[:len(I)]]
for i, d in enumerate(Ci):
    if (int(d['draws']), int(d['direct']), int(d['indirect']), int(d['count'])) != (4, 2, 1, 1):
        errs.append('C%d_draws' % i)
    if int(d['timed']) != brackets:
        errs.append('C%d_timed=%s' % (i, d['timed']))
    V, P = int(d['V']), int(d['P'])
    want = brackets * (100 * V + P) * 10  # fake: 100V+P ticks of 10 ns
    if brackets and abs(float(d['sum_ns']) - want) > 1:
        errs.append('C%d_sum=%s_want=%d' % (i, d['sum_ns'], want))
    if skip and int(d['skipped']) != 4:
        errs.append('C%d_skipped' % i)

print('brackets=%s route=%s classes=%d I=%d C=%d skip=%s errors=%s' %
      (','.join(str(x) for x in nb), ','.join(sorted(routes)), len(Ci), len(I), len(C), skip, ','.join(errs) or 'none'))
sys.exit(1 if errs else 0)
