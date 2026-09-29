#!/usr/bin/env python3
"""check.py <script> <dir> <case> [batch]

Checks one draw-cost case. run.sh recorded the pf.c script under RADV_DEBUG=dumpibs as
<dir>/<case>.<run>.{out,err} for the runs
  off    no switch
  reuse  RADV_BC250_SCRATCH_REUSE=1
  pf     RADV_BC250_SPLIT_PREP_FREE=1
  all    RADV_BC250_SPLIT_PREP_FREE=1 RADV_BC250_SCRATCH_REUSE=1 RADV_BC250_SPLIT_LEAN_SETUP=1
(lean and old are byte-compared by run.sh).

reuse vs off: the same packets draw by draw (draw packet fields except the address of the driver records,
the setup dispatches and CS_PARTIAL_FLUSHes between draws, every SET_*_REG value except user data SGPRs,
which carry upload addresses), and every split setup of a submit writes its own record range.

pf and all vs off: from the script, check.py models which split indirect draws are prep-free: fragment
shader "vis" (the only order-independent one), and at the draw no depth write (depth test and depth write
enabled with the depth/stencil attachment, token BD) and no stencil test with that attachment. A prep-free
draw must be exactly `pieces` DISPATCH_MESH_INDIRECT_MULTI packets reading the application's records
(SET_BASE = the application's buffer, DATA_OFFSET 0, COUNT = records, STRIDE = the application's stride,
COUNT_ADDR = the application's count buffer or 0), with no setup dispatch and no CS_PARTIAL_FLUSH in front
of them, and consecutive pieces may differ only in user data SGPRs (the piece select), of which at least one
must differ. Every other split indirect draw keeps its own setup (DISPATCH_DIRECT + CS_PARTIAL_FLUSH, STRIDE
16, SET_BASE = driver records; with batch: the split-batch pattern). All other draws are unchanged. At each
draw, the graphics register state must equal the off run's except user data SGPRs and the Mesh shader's
program registers (SPI_SHADER_PGM_RSRC*_GS, the piece select code can need other SGPR/VGPR counts); the
differing register names are printed.
Prints one key=value line and exits 1 on any mismatch."""
import re, sys

ansi = re.compile(r'\x1b\[[0-9;]*m')
pkt_re = re.compile(r'^([0-9a-f]{8}) ([A-Z0-9_]+)(\(.*\))?:')
reg_re = re.compile(r'^(?:[0-9a-f]{8})?\s+(?:\[[A-C]\])?([A-Z0-9_]+) <- (.*)$')
full_re = re.compile(r'^\s+\(FULL ADDRESS\) <- (0x[0-9a-f]+)')
SPLIT_FS = {'si': 'vis', 's1': 'vis', 'sc': 'vis', 'ci': 'col', 'cc': 'col', 'ui': 'used', 'xi': 'exch', 'wi': 'store'}
ALLOWED_REG_DIFF = re.compile(r'USER_DATA|PGM_RSRC[0-9]_GS')


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


def parse(path):
    """-> list per submit of [(draw_event, [between events])]"""
    subs, cur, pkt, regs, between, fields = [], None, None, {}, [], {}
    base = {'gfx': None}

    def finish():
        nonlocal pkt, fields, between
        if cur is None or pkt is None:
            return
        name, hdr = pkt
        if name == 'SET_BASE':
            if 'compute' not in hdr:
                base['gfx'] = fields.get('FULL')
        elif name == 'DISPATCH_INDIRECT':
            between.append('PREP_BATCH')
        elif name == 'DISPATCH_DIRECT':
            between.append('PREP_DIRECT')
        elif name == 'EVENT_WRITE' and fields.get('_cspf'):
            between.append('CSPF')
        elif name == 'DISPATCH_MESH_INDIRECT_MULTI' or name.startswith('DRAW_'):
            snap = {k: v for k, v in regs.items() if not k.startswith('COMPUTE_') and 'PGM_LO' not in k and 'PGM_HI' not in k}
            f = dict(fields)
            f.pop('_cspf', None)
            ev = {'name': name, 'fields': f, 'regs': snap}
            if name == 'DISPATCH_MESH_INDIRECT_MULTI':
                ev['base'] = base['gfx']
            cur.append((ev, between))
            between = []
        pkt = None
        fields = {}

    for line in open(path, errors='replace'):
        line = ansi.sub('', line.rstrip('\n'))
        if 'Main IB begin' in line:
            finish()
            cur = []
            subs.append(cur)
            regs, between = {}, []
            base['gfx'] = None
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
            fields['FULL' if pkt[0] == 'SET_BASE' else 'COUNT_ADDR'] = int(mf.group(1), 16)
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
        if l.startswith('PF_DRAW '):
            cur.append(dict(kv.split('=', 1) for kv in l.split()[1:]))
        elif l.startswith('SUBMIT_OK'):
            subs.append(cur)
            cur = []
    return subs


def model(script, prep_free, batch):
    """-> list of (token, kind) per primary draw; kind: free | own | open | join | none"""
    res = []
    ctx = dt = dw = st = 0
    batch_open = False
    for t in tokens(script):
        if t in ('B', 'BD', 'Bx'):
            ctx = 1 if t == 'BD' else 0
            batch_open = False
        elif t == 'E' or t.startswith('X'):
            batch_open = False
        elif t[:2] in ('dt', 'dw', 'st') and len(t) == 3:
            v = int(t[2])
            if t[:2] == 'dt': dt = v
            elif t[:2] == 'dw': dw = v
            else: st = v
        elif t in SPLIT_FS:
            free = prep_free and SPLIT_FS[t] == 'vis' and not (ctx and dt and dw) and not (ctx and st)
            if free:
                res.append((t, 'free'))
            elif batch:
                res.append((t, 'join' if batch_open else 'open'))
                batch_open = True
            else:
                res.append((t, 'own'))
        elif t in ('sd', 'm', 'mi', 'v'):
            res.append((t, 'none'))
    return res


SETUP_EVENTS = ('PREP_DIRECT', 'PREP_BATCH', 'CSPF')


def without_setup(between):
    """The off run's setup/flush events in front of a draw minus its split setup: the last setup dispatch
    and the CS_PARTIAL_FLUSH after it (application dispatches and flushes stay)."""
    ev = [e for e in between if e in SETUP_EVENTS]
    preps = [i for i, e in enumerate(ev) if e.startswith('PREP')]
    if not preps:
        return ev
    i = preps[-1]
    rest = ev[i + 1:]
    if 'CSPF' in rest:
        rest.remove('CSPF')
    return ev[:i] + rest


def regdiff(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def check_pf(script, off, run, pfd, pieces, batch, errs, tag, regnames):
    exp = model(script, True, batch)
    offexp = model(script, False, batch)
    counts = {'free': 0, 'own': 0, 'open': 0, 'join': 0}
    if len(off) != len(run) or len(run) != len(pfd):
        errs.append(tag + '_submits')
        return counts
    for si, (osub, rsub, drw) in enumerate(zip(off, run, pfd)):
        # the off run: one group per draw
        if len(osub) != len(offexp):
            errs.append('%s_s%d_off_draws_%d_%d' % (tag, si, len(osub), len(offexp)))
            return counts
        ri, di = 0, 0
        for gi, ((tok, kind), (oev, obetween)) in enumerate(zip(exp, osub)):
            g = '%s_s%d_d%d_%s' % (tag, si, gi, tok)
            n = pieces if kind == 'free' else 1
            grp = rsub[ri:ri + n]
            ri += n
            if len(grp) != n:
                errs.append(g + '_missing')
                return counts
            sb = None
            if tok in SPLIT_FS or tok == 'mi':
                sb = drw[di]
                di += 1
                if sb['tok'] != tok:
                    errs.append(g + '_order')
            counts[kind] = counts.get(kind, 0) + 1
            for pi, (ev, between) in enumerate(grp):
                d = regdiff(ev['regs'], oev['regs'])
                for k in d:
                    if not ALLOWED_REG_DIFF.search(k):
                        errs.append(g + '_reg_' + k)
                    regnames.add(k)
                if ev['name'] != oev['name']:
                    errs.append(g + '_packet_%s_%s' % (ev['name'], oev['name']))
            if kind == 'free':
                for pi, (ev, between) in enumerate(grp):
                    f = ev['fields']
                    want = {'COUNT': int(sb['records']), 'STRIDE': int(sb['stride']), 'DATA_OFFSET': 0,
                            'COUNT_ADDR': int(sb['count'], 16)}
                    got = {k: f.get(k, 0) for k in want}
                    if got != want:
                        errs.append(g + '_p%d_fields_%s_want_%s' % (pi, got, want))
                    if ev.get('base') != int(sb['input'], 16):
                        errs.append(g + '_p%d_base_%s' % (pi, ev.get('base')))
                    got_between = [e for e in between if e in SETUP_EVENTS]
                    if pi == 0 and not batch and got_between != without_setup(obetween):
                        errs.append(g + '_p%d_setup_%s_off_%s' % (pi, got_between, obetween))
                    if (pi or batch) and got_between:
                        errs.append(g + '_p%d_setup_%s' % (pi, got_between))
                    if pi:
                        d = regdiff(ev['regs'], grp[pi - 1][0]['regs'])
                        if not d or any('USER_DATA' not in k for k in d):
                            errs.append(g + '_p%d_piece_regs_%s' % (pi, d))
                continue
            ev, between = grp[0]
            if ev['fields'] != oev['fields']:
                errs.append(g + '_fields_%s_%s' % (ev['fields'], oev['fields']))
            want_between = [e for e in obetween if e in SETUP_EVENTS]
            got_between = [e for e in between if e in SETUP_EVENTS]
            if batch and kind in ('open', 'join'):
                want = ['PREP_BATCH', 'CSPF'] if kind == 'open' else []
                if got_between != want:
                    errs.append(g + '_batch_%s_%s' % (kind, got_between))
            elif got_between != want_between:
                errs.append(g + '_between_%s_%s' % (got_between, want_between))
            if kind == 'own' and 'PREP_DIRECT' not in want_between:
                errs.append(g + '_own_without_setup')
        if ri != len(rsub):
            errs.append('%s_s%d_extra_draws_%d' % (tag, si, len(rsub) - ri))
    return counts


def check_reuse(off, run, errs, tag):
    if len(off) != len(run):
        errs.append(tag + '_submits')
        return 0
    moved = 0
    for si, (a, b) in enumerate(zip(off, run)):
        if len(a) != len(b):
            errs.append('%s_s%d_draws' % (tag, si))
            continue
        bases = []
        for gi, ((ea, ba), (eb, bb)) in enumerate(zip(a, b)):
            g = '%s_s%d_d%d' % (tag, si, gi)
            if ea['name'] != eb['name'] or ea['fields'] != eb['fields'] or ba != bb:
                errs.append(g + '_packets')
            d = [k for k in regdiff(ea['regs'], eb['regs']) if 'USER_DATA' not in k]
            if d:
                errs.append(g + '_regs_' + ','.join(d[:4]))
            if 'PREP_DIRECT' in bb:
                bases.append(eb.get('base'))
                moved += ea.get('base') != eb.get('base')
        if len(set(bases)) != len(bases):
            errs.append('%s_s%d_shared_records' % (tag, si))
    return moved


def main():
    script, d, case = sys.argv[1:4]
    batch = len(sys.argv) > 4 and sys.argv[4] == 'batch'
    errs = []
    P = lambda run, ext: '%s/%s.%s.%s' % (d, case, run, ext)
    off = parse(P('off', 'err'))
    m = re.findall(r'direct mesh-only split enabled: pieces=(\d+)', open(P('off', 'err')).read())
    pieces = int(m[0]) if m else 0
    regnames = set()
    moved = check_reuse(off, parse(P('reuse', 'err')), errs, 'reuse')
    c_pf = check_pf(script, off, parse(P('pf', 'err')), draws_out(P('pf', 'out')), pieces, batch, errs, 'pf', regnames)
    # all: the same model (reuse only moves driver records, lean records the same packets)
    c_all = check_pf(script, off, parse(P('all', 'err')), draws_out(P('all', 'out')), pieces, batch, errs, 'all', set())
    nsub = max(1, len(off))
    print('pieces=%d free=%d own=%d open=%d join=%d reuse_moved_records=%d pf_reg_diffs=%s errors=%s' % (
        pieces, c_pf['free'] // nsub, c_pf['own'] // nsub, c_pf['open'] // nsub, c_pf['join'] // nsub, moved // nsub,
        ','.join(sorted(regnames)) or 'none', ','.join(errs[:6]) or 'none'))
    sys.exit(1 if errs else 0)


main()
