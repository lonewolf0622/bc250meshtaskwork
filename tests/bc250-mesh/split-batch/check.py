#!/usr/bin/env python3
"""check.py <script> <on.out> <on.err> <off.out> <off.err>

Checks one RADV_BC250_SPLIT_BATCH_PREP case. Both runs record the same sb.c script under
RADV_DEBUG=dumpibs; the on run has RADV_BC250_SPLIT_BATCH_PREP=1 RADV_BC250_SPLIT_BATCH_TRACE=1.

From the script, check.py models where batches open: a split indirect draw (si s1 sc ai ac) of the
primary command buffer inside a render pass instance and outside conditional rendering joins the open
batch or opens one (none open, or the open one holds 128 entries); B*/E/R/N/RE/P/C/c/X close the batch.
Every other split indirect draw (switch off, secondary, conditional rendering) has its own setup.

Per submitted command buffer, the packets between two draws (DISPATCH_MESH_INDIRECT_MULTI or DRAW_*)
must be, for a split indirect draw:
  own setup     exactly one DISPATCH_DIRECT (per-draw setup), then a CS_PARTIAL_FLUSH, no DISPATCH_INDIRECT;
  batch opens   exactly one DISPATCH_INDIRECT whose SET_BASE is the traced list address (the group
                count is read from the list header), then a CS_PARTIAL_FLUSH, no DISPATCH_DIRECT;
  batch joins   no dispatch and no CS_PARTIAL_FLUSH at all;
and no setup dispatch before any other draw. For batched draws the traced list entry must be the draw's
(slot = position in the batch, first chunk = the chunks of the batch's earlier draws, ceil(records/64)
chunks, input / count address / records / stride as sb.c recorded them, split pieces 3 for the nanite
shape and 2 for the a_ls32 shape) and the draw's DISPATCH_MESH_INDIRECT_MULTI
must read that entry's output (SET_BASE = output, DATA_OFFSET 0, COUNT = records, STRIDE 16,
COUNT_ADDR = the application's count buffer or 0). In the off run every split indirect draw has its own
setup. Draw by draw, the on and off runs must match: same draw packets and fields (except the address of
the driver records), same graphics register state at the draw (all SET_*_REG values except COMPUTE_*
and shader program addresses, which move with the extra setup shader).
Prints one key=value line and exits 1 on any mismatch."""
import re, sys

ansi = re.compile(r'\x1b\[[0-9;]*m')
pkt_re = re.compile(r'^([0-9a-f]{8}) ([A-Z0-9_]+)(\(.*\))?:')
reg_re = re.compile(r'^(?:[0-9a-f]{8})?\s+(?:\[[A-C]\])?([A-Z0-9_]+) <- (.*)$')
full_re = re.compile(r'^\s+\(FULL ADDRESS\) <- (0x[0-9a-f]+)')
SPLIT = {'si': ('nanite', 3), 's1': ('nanite', 3), 'sc': ('nanite', 3), 'sL': ('nanite', 3), 'ai': ('als32', 2),
         'ac': ('als32', 2)}
RECORDS = {'si': 2, 's1': 1, 'sc': 4, 'sL': 1000, 'ai': 3, 'ac': 3}
CLOSE = {'B', 'Bs', 'Br', 'Brs', 'Bx', 'E', 'R', 'N', 'RE', 'P', 'C', 'c'}
ENTRIES = 128
CHUNKS = 1024


def val(s):
    m = re.search(r'\((0x[0-9a-f]+)\)', s)
    if m:
        return int(m.group(1), 16)
    s = s.strip().split(' ')[0]
    try:
        return int(s, 0)
    except ValueError:
        return s


def tokens(script):
    out = []
    for t in script.split():
        if '*' in t:
            t, n = t.split('*')
            out += [t] * int(n)
        else:
            out.append(t)
    return out


def expect(script, on):
    """-> list of (token, kind) per primary draw; kind: own | open | join | none"""
    res = []
    open_, used, chunks, in_rp, cond = False, 0, 0, False, False
    for t in tokens(script):
        if t in CLOSE or t.startswith('X'):
            open_ = False
            if t.startswith('B') or t in ('R', 'N'):
                in_rp = True
            if t in ('E', 'RE'):
                in_rp = False
            if t == 'C':
                cond = True
            if t == 'c':
                cond = False
            continue
        if t in SPLIT:
            if on and in_rp and not cond:
                c = (RECORDS[t] + 63) // 64
                if not open_ or used == ENTRIES or chunks + c > CHUNKS:
                    open_, used, chunks = True, 0, 0
                    res.append((t, 'open'))
                else:
                    res.append((t, 'join'))
                used += 1
                chunks += c
            else:
                res.append((t, 'own'))
        else:
            res.append((t, 'none'))
    return res


def parse(path):
    """-> per submit: {'groups': [(draw_event, [between events])], 'trace': [trace dicts]}"""
    subs = []
    cur = None
    pending_trace = []
    pkt = None
    regs = {}
    gfx_base = cmp_base = None
    between = []
    fields = {}

    def finish():
        nonlocal pkt, fields, between
        if cur is None or pkt is None:
            return
        name, hdr = pkt
        if name == 'SET_BASE':
            if 'compute' in hdr:
                nonlocal_cmp[0] = fields.get('FULL')
            else:
                nonlocal_gfx[0] = fields.get('FULL')
        elif name == 'DISPATCH_INDIRECT':
            between.append(('PREP_BATCH', nonlocal_cmp[0]))
        elif name == 'DISPATCH_DIRECT':
            between.append(('PREP_DIRECT', None))
        elif name == 'EVENT_WRITE' and fields.get('_cspf'):
            between.append(('CSPF', None))
        elif name == 'DISPATCH_MESH_INDIRECT_MULTI' or name.startswith('DRAW_'):
            snap = {k: v for k, v in regs.items() if not k.startswith('COMPUTE_') and 'PGM_LO' not in k and 'PGM_HI' not in k}
            f = dict(fields)
            f.pop('_cspf', None)
            ev = {'name': name, 'fields': f, 'regs': snap}
            if name == 'DISPATCH_MESH_INDIRECT_MULTI':
                ev['base'] = nonlocal_gfx[0]
            cur['groups'].append((ev, between))
            between = []
        pkt = None
        fields = {}

    nonlocal_cmp = [None]
    nonlocal_gfx = [None]
    for line in open(path, errors='replace'):
        line = ansi.sub('', line.rstrip('\n'))
        if line.startswith('BC250_SPLIT_BATCH '):
            d = dict(kv.split('=', 1) for kv in line.split()[2:])
            d['kind'] = line.split()[1]
            pending_trace.append(d)
            continue
        if 'Main IB begin' in line:
            finish()
            cur = {'groups': [], 'trace': pending_trace}
            pending_trace = []
            subs.append(cur)
            regs = {}
            between = []
            nonlocal_cmp[0] = nonlocal_gfx[0] = None
            continue
        if cur is None:
            continue
        m = pkt_re.match(line)
        if m:
            finish()
            pkt = (m.group(2), m.group(3) or '')
            continue
        if pkt is None:
            continue
        mf = full_re.match(line)
        if mf:
            key = 'FULL' if pkt[0] == 'SET_BASE' else 'COUNT_ADDR'
            fields[key] = int(mf.group(1), 16)
            continue
        if 'EVENT_TYPE = CS_PARTIAL_FLUSH' in line:
            fields['_cspf'] = True
        mm = reg_re.match(line)
        if mm:
            name, value = mm.group(1), val(mm.group(2))
            if pkt[0].startswith('SET_') and pkt[0] != 'SET_BASE':
                regs[name] = value
            else:
                fields[name] = value
    finish()
    return subs


def draws_out(path):
    subs, cur = [], []
    for l in open(path):
        if l.startswith('SB_DRAW '):
            d = dict(kv.split('=', 1) for kv in l.split()[1:])
            if '@secondary' not in d['tok']:
                cur.append(d)
        elif l.startswith('SUBMIT_OK'):
            subs.append(cur)
            cur = []
    return subs


def check_run(script, subs, sbdraws, on, errs, tag):
    exp = expect(script, on)
    stats = {'batches': 0, 'own': 0, 'joined': 0, 'cspf': 0}
    if len(subs) != len(sbdraws) or not subs:
        errs.append(tag + '_submits_%d_%d' % (len(subs), len(sbdraws)))
        return stats
    for si, (sub, sbd) in enumerate(zip(subs, sbdraws)):
        t = tag + '_s%d' % si
        groups = sub['groups']
        if len(groups) != len(exp):
            errs.append(t + '_draws_%d_expected_%d' % (len(groups), len(exp)))
            return stats
        trace = list(sub['trace'])
        opens = [d for d in trace if d['kind'] == 'open']
        entries = [d for d in trace if d['kind'] == 'entry']
        if not on and trace:
            errs.append(t + '_trace_in_off_run')
        ind = [d for d in sbd]
        ind_i = 0
        list_va = None
        slot = 0
        chunk = 0
        for gi, ((ev, between), (tok, kind)) in enumerate(zip(groups, exp)):
            g = t + '_d%d_%s' % (gi, tok)
            preps = [e for e in between if e[0].startswith('PREP')]
            cspf_after = [i for i, e in enumerate(between) if e[0] == 'CSPF']
            stats['cspf'] += len(cspf_after)
            is_ind = tok in SPLIT or tok == 'mi'
            if is_ind != (ev['name'] == 'DISPATCH_MESH_INDIRECT_MULTI'):
                errs.append(g + '_draw_packet_' + ev['name'])
                continue
            sb = None
            if is_ind:
                sb = ind[ind_i]
                ind_i += 1
                if sb['tok'] != tok:
                    errs.append(g + '_sb_draw_order')
            if kind == 'none':
                if preps:
                    errs.append(g + '_unexpected_setup')
                continue
            f = ev['fields']
            records = int(sb['records'])
            if f.get('COUNT') != records or f.get('STRIDE') != 16 or f.get('DATA_OFFSET') != 0 or \
               f.get('COUNT_ADDR', 0) != int(sb['count'], 16):
                errs.append(g + '_draw_fields_%s' % f)
            if kind == 'own':
                stats['own'] += 1
                if [p[0] for p in preps] != ['PREP_DIRECT']:
                    errs.append(g + '_own_setup_%s' % [p[0] for p in preps])
                else:
                    pi = between.index(preps[0])
                    if not any(i > pi for i in cspf_after):
                        errs.append(g + '_own_no_cspf')
                continue
            # batched
            if kind == 'open':
                stats['batches'] += 1
                if [p[0] for p in preps] != ['PREP_BATCH']:
                    errs.append(g + '_open_setup_%s' % [p[0] for p in preps])
                    continue
                if not opens:
                    errs.append(g + '_no_open_trace')
                    continue
                o = opens.pop(0)
                list_va = int(o['list'], 16)
                if preps[0][1] != list_va:
                    errs.append(g + '_dispatch_base_%s_list_%x' % (preps[0][1], list_va))
                pi = between.index(preps[0])
                if not any(i > pi for i in cspf_after):
                    errs.append(g + '_open_no_cspf')
                slot = 0
                chunk = 0
            else:
                stats['joined'] += 1
                if preps or cspf_after:
                    errs.append(g + '_join_has_setup_or_cspf_%s' % [e[0] for e in between])
            if not entries:
                errs.append(g + '_no_entry_trace')
                continue
            e = entries.pop(0)
            shape, pieces = SPLIT[tok]
            want = {'list': list_va, 'slot': slot, 'input': int(sb['input'], 16), 'count': int(sb['count'], 16),
                    'records': records, 'stride': int(sb['stride']), 'pieces': pieces, 'first_chunk': chunk,
                    'chunks': (records + 63) // 64}
            got = {'list': int(e['list'], 16), 'slot': int(e['slot']), 'input': int(e['input'], 16),
                   'count': int(e['count'], 16), 'records': int(e['records']), 'stride': int(e['stride']),
                   'pieces': int(e['pieces']), 'first_chunk': int(e['first_chunk']), 'chunks': int(e['chunks'])}
            chunk += (records + 63) // 64
            if want != got:
                errs.append(g + '_entry_%s_want_%s' % (got, want))
            if ev.get('base') != int(e['output'], 16):
                errs.append(g + '_reads_%s_not_slot_output_%s' % (ev.get('base'), e['output']))
            slot += 1
        if opens or entries:
            errs.append(t + '_unconsumed_trace_%d_%d' % (len(opens), len(entries)))
    return stats


def main():
    script, on_out, on_err, off_out, off_err = sys.argv[1:6]
    errs = []
    on = parse(on_err)
    off = parse(off_err)
    s_on = check_run(script, on, draws_out(on_out), True, errs, 'on')
    s_off = check_run(script, off, draws_out(off_out), False, errs, 'off')
    # draw by draw equality of on and off
    for si, (a, b) in enumerate(zip(on, off)):
        for gi, ((ea, _), (eb, _)) in enumerate(zip(a['groups'], b['groups'])):
            if ea['name'] != eb['name']:
                errs.append('eq_s%d_d%d_name' % (si, gi))
                continue
            if ea['fields'] != eb['fields']:
                errs.append('eq_s%d_d%d_fields' % (si, gi))
            if ea['regs'] != eb['regs']:
                diff = sorted(k for k in set(ea['regs']) | set(eb['regs']) if ea['regs'].get(k) != eb['regs'].get(k))
                errs.append('eq_s%d_d%d_regs_%s' % (si, gi, ','.join(diff[:6])))
    nsub = max(1, len(on))
    print('batches=%d joined=%d own_on=%d own_off=%d cspf_on=%d cspf_off=%d submits=%d errors=%s' % (
        s_on['batches'] // nsub, s_on['joined'] // nsub, s_on['own'] // nsub, s_off['own'] // nsub,
        s_on['cspf'] // nsub, s_off['cspf'] // nsub, len(on), ','.join(errs[:8]) or 'none'))
    sys.exit(1 if errs else 0)


main()
