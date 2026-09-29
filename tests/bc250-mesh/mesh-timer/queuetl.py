#!/usr/bin/env python3
"""queuetl.py <BC250_MESH_TIMER log> [--from S] [--to S] [--segment N] [--hist] [--vkd3d-trace F] [--json OUT]

Per-queue GPU/CPU timelines of a BC250_MESH_TIMER log written with BC250_MESH_TIMER_SUBMITS=1.
Answers, per frame: how long is the graphics queue idle and why (the CPU handed the work over late /
the work waited for another queue / the other queue was done but the graphics work had not started yet),
how much compute-queue work overlaps graphics work, how late the compute submissions were relative to
the graphics work they wait for, and the GPU-side cross-queue wait latency.

Log formats
  v1 (timer builds up to 78c882d, "# BC250_MESH_TIMER v1"): S lines only for graphics command buffers.
     The seq counter advances on every vkQueueSubmit of the device (any queue), so a seq without an S
     line is a submission on another queue (compute / copy), a semaphore-only submission or a dropped
     record ("foreign"). Graphics idle is split into cpu_late (the submission returned from the kernel
     after the GPU had gone idle) and gpu_wait (it was in the kernel before: waits / scheduling).
  v2 (the async-compute timer, "# BC250_MESH_TIMER v2"): a W line per vkQueueSubmit of every queue
     (queue family.index, kind, driver entry time call_ns, return time wall_ns, command buffer / wait /
     signal counts, wait and signal semaphores with values, vk_queue submit-thread flag), S lines for
     graphics AND compute command buffers (GPU begin / end), P lines (presents), T lines with
     CLOCK_MONOTONIC_RAW (for vkd3d-proton traces).

Times are CPU CLOCK_MONOTONIC ns; GPU ticks are mapped through the T calibration lines (piecewise
linear: the GPU reference clock and CLOCK_MONOTONIC drift apart by ~0.2 %). The window is in seconds
since the timer start of the segment (the "t=" of the I lines). A log can hold several device segments
(each starts with a "# BC250_MESH_TIMER" header); the one with the most S lines is used unless --segment.

--vkd3d-trace F: a VKD3D_QUEUE_PROFILE=F trace written with VKD3D_QUEUE_PROFILE_ABSOLUTE=1 (Chrome JSON,
     QueryPerformanceCounter = CLOCK_MONOTONIC_RAW time under Wine). Needs a v2 log (T lines with raw_ns).
     Sums the vkd3d submission-thread CPU blocking regions ("WAIT BEFORE SIGNAL", "CPU WAIT", "WAIT
     (shared)") per thread and per frame, maps each vkd3d queue thread to the Vulkan queue it submits to,
     and reports how much graphics-queue idle time coincides with each blocking kind.
--json OUT: also write the numbers as JSON.
"""
import bisect
import json
import re
import statistics
import sys
from collections import defaultdict

KV = re.compile(r'(\w+)=("[^"]*"|\S+)')
WALL = re.compile(rb' wall_ns=(\d+)')


def parse_args(argv):
    if not argv or argv[0] in ('-h', '--help'):
        sys.exit(__doc__)
    o = {'path': argv[0], 'from': None, 'to': None, 'segment': None, 'hist': '--hist' in argv,
         'trace': None, 'json': None}
    for k in ('from', 'to'):
        if '--' + k in argv:
            o[k] = float(argv[argv.index('--' + k) + 1])
    if '--segment' in argv:
        o['segment'] = int(argv[argv.index('--segment') + 1])
    if '--vkd3d-trace' in argv:
        o['trace'] = argv[argv.index('--vkd3d-trace') + 1]
    if '--json' in argv:
        o['json'] = argv[argv.index('--json') + 1]
    return o


# ---- reading ---------------------------------------------------------------------------------------

def read_segments(path):
    """Pass 1: segment headers, T / I / P lines, S / W counts and line ranges."""
    segs = []
    cur = None
    with open(path, 'rb') as f:
        for n, raw in enumerate(f):
            c = raw[:2]
            if c in (b'S ', b'W '):
                if cur is not None:
                    cur['n' + chr(c[0])] += 1
                    cur['last'] = n
                continue
            if raw.startswith(b'# BC250_MESH_TIMER'):
                line = raw.decode(errors='replace')
                m = re.search(r'ns_per_tick=([\d.]+)', line)
                v = re.search(r'BC250_MESH_TIMER v(\d+)', line)
                cur = {'ns_per_tick': float(m.group(1)) if m else 10.0, 'version': int(v.group(1)) if v else 1,
                       'fake': ' fake=1 ' in line, 'nS': 0, 'nW': 0, 'T': [], 'I': [], 'P': [],
                       'header': line.strip(), 'first': n, 'last': n}
                segs.append(cur)
                continue
            if cur is None:
                continue
            cur['last'] = n
            if c not in (b'T ', b'I ', b'P '):
                continue
            line = raw.decode(errors='replace')
            d = dict(KV.findall(line))
            if c == b'T ':
                cur['T'].append({k: int(v) for k, v in d.items()})
            elif c == b'I ':
                if line.startswith('I interval'):
                    cur['I'].append((float(d['t']), float(d['dt']), int(d.get('frames', 0))))
            else:
                cur['P'].append(int(d['wall_ns']))
    return segs


def read_window(path, seg, lo, hi):
    """Pass 2: S and W lines of one segment with wall_ns in [lo, hi]."""
    S, W = [], {}
    with open(path, 'rb') as f:
        for n, raw in enumerate(f):
            if n < seg['first']:
                continue
            if n > seg['last']:
                break
            c = raw[:2]
            if c not in (b'S ', b'W '):
                continue
            m = WALL.search(raw)
            if not m:
                continue
            w = int(m.group(1))
            if w < lo or w > hi:
                continue
            d = dict(KV.findall(raw.decode(errors='replace')))
            if c == b'S ':
                S.append(d)
            else:
                W[int(d['seq'])] = d
    return S, W


def parse_sem_list(s):
    out = []
    if not s or s == '-':
        return out
    for item in s.split(','):
        if ':' in item and not item.startswith('+'):
            a, b = item.split(':', 1)
            try:
                out.append((a, int(b)))
            except ValueError:
                pass
    return out


# ---- helpers ---------------------------------------------------------------------------------------

def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]


def union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def total(u):
    return sum(b - a for a, b in u)


def overlap_len(u1, u2):
    i = j = 0
    tot = 0.0
    while i < len(u1) and j < len(u2):
        a = max(u1[i][0], u2[j][0])
        b = min(u1[i][1], u2[j][1])
        if b > a:
            tot += b - a
        if u1[i][1] < u2[j][1]:
            i += 1
        else:
            j += 1
    return tot


HIST_EDGES = [10e3, 25e3, 50e3, 100e3, 200e3, 500e3, 1e6, 2e6, 1e30]
HIST_NAMES = ['<10us', '<25us', '<50us', '<100us', '<200us', '<500us', '<1ms', '<2ms', '>=2ms']


class Scale:
    def __init__(self, n_frames, span_s):
        self.n = n_frames
        self.span = max(span_s, 1e-9)
        self.unit = 'ms/frame' if n_frames else 'ms/s'
        self.nunit = '/frame' if n_frames else '/s'

    def ms(self, ns):
        return ns / self.n / 1e6 if self.n else ns / self.span / 1e6

    def cnt(self, c):
        return c / self.n if self.n else c / self.span


def hist_line(vals, sc):
    cnt = [0] * len(HIST_EDGES)
    tot = [0.0] * len(HIST_EDGES)
    for v in vals:
        for i, e in enumerate(HIST_EDGES):
            if v < e:
                cnt[i] += 1
                tot[i] += v
                break
    return '  '.join(f'{n}:{sc.cnt(c):.1f}{sc.nunit}/{sc.ms(tot[i]):.2f}ms'
                     for i, (n, c) in enumerate(zip(HIST_NAMES, cnt)) if c)


def gaps_summary(name, vals, sc, out):
    out[name] = {'count': sc.cnt(len(vals)), 'ms': sc.ms(sum(vals)), 'median_us': pct(vals, .5) / 1e3,
                 'p90_us': pct(vals, .9) / 1e3}
    return (f'{name} {sc.cnt(len(vals)):.1f}{sc.nunit} {sc.ms(sum(vals)):.2f} {sc.unit} '
            f'(median {pct(vals, .5) / 1e3:.0f} us, p90 {pct(vals, .9) / 1e3:.0f} us)')


# ---- main ------------------------------------------------------------------------------------------

def main():
    o = parse_args(sys.argv[1:])
    segs = read_segments(o['path'])
    if not segs:
        sys.exit('no "# BC250_MESH_TIMER" header in the log')
    seg = segs[o['segment']] if o['segment'] is not None else max(segs, key=lambda s: s['nS'] + s['nW'])
    if not seg['T']:
        sys.exit('segment has no T calibration lines (BC250_MESH_TIMER_SUBMITS=1 and one print interval needed)')
    npt = seg['ns_per_tick']
    cal = sorted({(t['cpu_ns'], t['gpu_tick']) for t in seg['T']}, key=lambda x: x[1])
    offs = [c - g * npt for c, g in cal]
    spread_us = (max(offs) - min(offs)) / 1e3
    cal_g = [g for c, g in cal]
    rates = [(c1 - c0) / ((g1 - g0) * npt) - 1.0 for (c0, g0), (c1, g1) in zip(cal, cal[1:]) if g1 > g0]

    def to_cpu(g):
        if len(cal) < 2:
            return g * npt + offs[0]
        i = min(max(bisect.bisect_left(cal_g, g), 1), len(cal) - 1)
        (c0, g0), (c1, g1) = cal[i - 1], cal[i]
        return c0 + (g - g0) * (c1 - c0) / (g1 - g0) if g1 != g0 else c0 + (g - g0) * npt

    # timer start of the segment: the first print writes T, then "I interval t=<s since start>"
    if seg['I']:
        start_ns = seg['T'][0]['cpu_ns'] - seg['I'][0][0] * 1e9
    else:
        start_ns = seg['T'][0]['cpu_ns']
    lo = start_ns + (o['from'] * 1e9 if o['from'] is not None else -1e30)
    hi = start_ns + (o['to'] * 1e9 if o['to'] is not None else 1e30)
    S, W = read_window(o['path'], seg, lo - 2e9, hi)  # 2 s margin: dependencies signalled before the window
    v2 = seg['version'] >= 2 or bool(W)

    # submissions by seq
    subs = {}
    for seq, d in W.items():
        subs[seq] = {'seq': seq, 'q': d['q'], 'qf': d['qf'], 'call': int(d['call_ns']), 'wall': int(d['wall_ns']),
                     'ncb': int(d['ncb']), 'w': parse_sem_list(d.get('w')), 's': parse_sem_list(d.get('s')),
                     'thr': int(d.get('thr', 0)), 'cbs': [], 'md': 0}
    for d in S:
        seq = int(d['seq'])
        sub = subs.get(seq)
        if sub is None:  # v1, or W line outside the margin
            sub = subs[seq] = {'seq': seq, 'q': d.get('q', '0.0'), 'qf': d.get('qf', 'gfx'),
                               'call': None, 'wall': int(d['wall_ns']), 'ncb': None, 'w': [], 's': [], 'thr': 0,
                               'cbs': [], 'md': 0}
        t0, t1 = int(d.get('gpu_t0', 0)), int(d.get('gpu_t1', 0))
        if t0 and t1 and t1 >= t0:
            sub['cbs'].append((to_cpu(t0), to_cpu(t1)))
        sub['md'] += int(d.get('mesh_draws', 0))
    for sub in subs.values():
        sub['start'] = min(a for a, b in sub['cbs']) if sub['cbs'] else None
        sub['end'] = max(b for a, b in sub['cbs']) if sub['cbs'] else None

    in_win = [s for s in subs.values() if lo <= s['wall'] <= hi]
    if not in_win:
        sys.exit('no submissions in the window')
    wall_span = (max(s['wall'] for s in in_win) - min(s['wall'] for s in in_win)) / 1e9
    if seg['P']:
        n_frames = sum(1 for p in seg['P'] if lo <= p <= hi)
        frame_src = 'P lines'
    else:
        n_frames = 0.0
        for t, dt, fr in seg['I']:
            a, b = start_ns + (t - dt) * 1e9, start_ns + t * 1e9
            ov = max(0.0, min(b, hi) - max(a, lo))
            if b > a:
                n_frames += fr * ov / (b - a)
        frame_src = 'I lines (prorated)'
    sc = Scale(n_frames, wall_span)
    J = {'log': o['path'], 'window': [o['from'], o['to']], 'frames': n_frames, 'format': 'v2' if v2 else 'v1'}

    print(f'log: {o["path"]}')
    print(f'segment {segs.index(seg)} of {len(segs)}: S={seg["nS"]} W={seg["nW"]}; '
          f'{len(cal)} T lines, clock offset spread {spread_us:.0f} us, rate error median '
          f'{statistics.median(rates or [0]) * 1e6:.0f} ppm (piecewise-linear calibration)')
    print(f'window {o["from"]}..{o["to"]} s: {len(in_win)} submissions, {wall_span:.1f} s, {n_frames:.0f} frames '
          f'({frame_src}; {n_frames / wall_span if wall_span else 0:.1f} fps), format {"v2" if v2 else "v1"}')

    # ---- per queue ---------------------------------------------------------------------------------
    queues = defaultdict(list)
    for s in in_win:
        queues[(s['qf'], s['q'])].append(s)
    unions = {}
    print('queues:')
    J['queues'] = {}
    for key, ss in sorted(queues.items()):
        u = union([cb for s in ss for cb in s['cbs']])
        unions[key] = u
        ioctl = [s['wall'] - s['call'] for s in ss if s['call'] is not None]
        nw = sum(len(s['w']) for s in ss)
        ns = sum(len(s['s']) for s in ss)
        empty = sum(1 for s in ss if s['ncb'] == 0)
        thr = sum(1 for s in ss if s['thr'])
        ncb = sum(len(s['cbs']) for s in ss)
        J['queues'][f'{key[0]}:{key[1]}'] = {'submits': sc.cnt(len(ss)), 'timed_cbs': sc.cnt(ncb), 'busy_ms': sc.ms(total(u)),
                                             'waits': sc.cnt(nw), 'signals': sc.cnt(ns), 'cb_less': sc.cnt(empty),
                                             'submit_us_mean': statistics.mean(ioctl) / 1e3 if ioctl else None,
                                             'threaded': thr}
        print(f'  {key[0]:5s} q={key[1]:4s} submits {sc.cnt(len(ss)):6.1f}{sc.nunit} (cb-less {sc.cnt(empty):.1f}) '
              f'timed cbs {sc.cnt(ncb):6.1f} busy {sc.ms(total(u)):6.2f} {sc.unit} waits {sc.cnt(nw):5.1f} '
              f'signals {sc.cnt(ns):5.1f}'
              + (f' driver+kernel submit mean {statistics.mean(ioctl) / 1e3:.0f} us p90 {pct(ioctl, .9) / 1e3:.0f} us'
                 if ioctl else '')
              + (f' THREADED-SUBMIT {thr}' if thr else ''))

    if not v2:
        v1_analysis(subs, in_win, sc, o, J)
    else:
        v2_analysis(subs, in_win, queues, unions, sc, o, J, seg)
    if o['json']:
        with open(o['json'], 'w') as f:
            json.dump(J, f, indent=1)


def v1_analysis(subs, in_win, sc, o, J):
    seqs = set(subs)
    smin = min(s['seq'] for s in in_win)
    smax = max(s['seq'] for s in in_win)
    foreign = sum(1 for s in range(smin, smax + 1) if s not in seqs)
    print(f'foreign submissions (seq without a graphics S line: compute / copy / semaphore-only / dropped): '
          f'{sc.cnt(foreign):.1f}{sc.nunit}')
    ivs = sorted((a, b, s['seq'], s['wall']) for s in in_win for a, b in s['cbs'])
    busy = 0.0
    late_g, wait_f, wait_n = [], [], []
    late_after_foreign = []
    end_max, end_seq = ivs[0][1], ivs[0][2]
    busy += ivs[0][1] - ivs[0][0]
    for a, b, seq, wall in ivs[1:]:
        if a > end_max:
            gap = a - end_max
            late = min(gap, max(0.0, wall - end_max))
            if late > 0:
                late_g.append(late)
                if (seq - 1) not in seqs:
                    late_after_foreign.append(late)
            wait = gap - late
            if wait > 0:
                has_foreign = any(s not in seqs for s in range(min(end_seq, seq) + 1, seq))
                (wait_f if has_foreign else wait_n).append(wait)
        busy += max(0.0, b - max(a, end_max))
        if b >= end_max:
            end_max, end_seq = b, seq
    idle = sum(late_g) + sum(wait_f) + sum(wait_n)
    J['graphics'] = {'busy_ms': sc.ms(busy), 'idle_ms': sc.ms(idle)}
    print(f'graphics queue: busy {sc.ms(busy):.2f} {sc.unit} | idle {sc.ms(idle):.2f} = cpu_late {sc.ms(sum(late_g)):.2f} '
          f'+ gpu_wait {sc.ms(sum(wait_f) + sum(wait_n)):.2f}')
    print('  ' + gaps_summary('cpu_late', late_g, sc, J['graphics']))
    print('  ' + gaps_summary('cpu_late right after a foreign vkQueueSubmit', late_after_foreign, sc, J['graphics']))
    print('  ' + gaps_summary('gpu_wait after foreign submissions', wait_f, sc, J['graphics']))
    print('  ' + gaps_summary('gpu_wait otherwise', wait_n, sc, J['graphics']))
    if o['hist']:
        print('  hist cpu_late:', hist_line(late_g, sc))
        print('  hist gpu_wait after foreign:', hist_line(wait_f, sc))
        print('  hist gpu_wait otherwise:', hist_line(wait_n, sc))


def v2_analysis(subs, in_win, queues, unions, sc, o, J, seg):
    order = sorted(subs)
    # GPU end of every submission (in-order queues): a submission without timed command buffers ends when the
    # earlier work of its queue and its dependencies have ended.
    last_end = {}
    tl_sigs = defaultdict(list)   # timeline sem -> [(value, seq)] (values ascend with seq)
    bin_sigs = defaultdict(list)  # binary sem -> [seq]

    def dep_of(sem, val, seq):
        """The submission (seq) that signals what (sem, val) waits for, or None."""
        if val:
            lst = tl_sigs.get(sem)
            if not lst:
                return None
            i = bisect.bisect_left(lst, (val, -1))
            return lst[i][1] if i < len(lst) and lst[i][1] < seq else None
        lst = bin_sigs.get(sem)
        if not lst:
            return None
        i = bisect.bisect_left(lst, seq) - 1
        return lst[i] if i >= 0 else None

    for seq in order:
        s = subs[seq]
        key = (s['qf'], s['q'])
        s['deps'] = [d for d in (dep_of(sem, val, seq) for sem, val in s['w']) if d is not None]
        if s['end'] is None:
            cand = [last_end.get(key)] + [subs[d]['end_est'] for d in s['deps']]
            cand = [c for c in cand if c is not None]
            s['end_est'] = max(cand) if cand else None
        else:
            s['end_est'] = s['end']
        if s['end_est'] is not None:
            last_end[key] = max(s['end_est'], last_end.get(key, s['end_est']))
        for sem, val in s['s']:
            if val:
                tl_sigs[sem].append((val, seq))
            else:
                bin_sigs[sem].append(seq)

    gkeys = [k for k in queues if k[0] == 'gfx']
    if not gkeys:
        print('no graphics submissions in the window')
        return
    gkey = max(gkeys, key=lambda k: len(queues[k]))

    def foreign_dep_end(s):
        ends = [(subs[d]['end_est'], d) for d in s['deps'] if (subs[d]['qf'], subs[d]['q']) != (s['qf'], s['q'])
                and subs[d]['end_est'] is not None]
        return max(ends) if ends else (None, None)

    # ---- graphics idle ---------------------------------------------------------------------------------
    ivs = sorted((a, b, s['seq']) for s in queues[gkey] for a, b in s['cbs'])
    if not ivs:
        print('no timed graphics command buffers in the window')
        return
    busy = 0.0
    parts = defaultdict(list)
    end_max = ivs[0][1]
    busy += ivs[0][1] - ivs[0][0]
    for a, b, seq in ivs[1:]:
        s = subs[seq]
        if a > end_max:
            gap = a - end_max
            t = end_max
            call = s['call'] if s['call'] is not None else s['wall']
            app = min(a, max(t, call)) - t          # vkQueueSubmit not called yet
            t += app
            drv = min(a, max(t, s['wall'])) - t     # inside the driver / kernel submit
            t += drv
            dep_end, dep_seq = foreign_dep_end(s)
            dep = rel = other = 0.0
            if dep_end is not None and dep_end > t:
                dep = min(a, dep_end) - t
                t += dep
                rel = a - t
            elif dep_end is not None:
                rel = a - t
            else:
                other = a - t
            if app > 0:
                parts['app_late'].append(app)
                # blocked behind another queue's submission (vkd3d wait-before-signal): the foreign signal
                # it waits for was submitted after the graphics queue went idle
                if dep_seq is not None and subs[dep_seq]['wall'] > end_max and subs[dep_seq]['wall'] <= call:
                    parts['app_late_until_signal_submitted'].append(min(app, subs[dep_seq]['wall'] - end_max))
            if drv > 0:
                parts['driver_submit'].append(drv)
            if dep > 0:
                parts['xq_dep'].append(dep)
            if rel > 0:
                parts['xq_release'].append(rel)
            if other > 0:
                parts['other_wait'].append(other)
            parts['idle'].append(gap)
        busy += max(0.0, b - max(a, end_max))
        end_max = max(end_max, b)
    G = J['graphics'] = {'busy_ms': sc.ms(busy), 'idle_ms': sc.ms(sum(parts['idle']))}
    print(f'graphics queue {gkey[1]}: busy {sc.ms(busy):.2f} {sc.unit} | idle {sc.ms(sum(parts["idle"])):.2f} {sc.unit} =')
    for name, what in (('app_late', 'vkQueueSubmit called after the GPU went idle (app / vkd3d thread late)'),
                       ('app_late_until_signal_submitted', '  of which: until the other queue submitted the awaited signal'),
                       ('driver_submit', 'inside the driver + kernel submit (entry to return)'),
                       ('xq_dep', 'waiting for work of another queue to finish (GPU)'),
                       ('xq_release', 'after that work finished, before the graphics work started (kernel release + preamble)'),
                       ('other_wait', 'in the kernel, no cross-queue dependency (same-queue waits, scheduling)')):
        print(f'  {gaps_summary(name, parts[name], sc, G)}  -- {what}')
    if o['hist']:
        for name in ('app_late', 'driver_submit', 'xq_dep', 'xq_release', 'other_wait'):
            print(f'  hist {name}:', hist_line(parts[name], sc))

    # ---- other queues: overlap, lateness vs their graphics dependencies -----------------------------------
    gu = unions[gkey]
    J['other'] = {}
    for key, ss in sorted(queues.items()):
        if key == gkey:
            continue
        cu = unions[key]
        cb = total(cu)
        ov = overlap_len(cu, gu)
        late = []
        for s in ss:
            dep_end, dep_seq = foreign_dep_end(s)
            if dep_end is not None and s['call'] is not None and s['call'] > dep_end:
                late.append(s['call'] - dep_end)
        J['other'][f'{key[0]}:{key[1]}'] = {'busy_ms': sc.ms(cb), 'overlap_ms': sc.ms(ov),
                                            'submitted_after_dep_done_ms': sc.ms(sum(late)),
                                            'submitted_after_dep_done_count': sc.cnt(len(late))}
        print(f'{key[0]} q={key[1]}: busy {sc.ms(cb):.2f} {sc.unit}, overlapping graphics {sc.ms(ov):.2f} '
              f'({100.0 * ov / cb if cb else 0:.0f}%), alone {sc.ms(cb - ov):.2f}; submissions called after their '
              f'other-queue dependency had finished: {sc.cnt(len(late)):.1f}{sc.nunit}, {sc.ms(sum(late)):.2f} {sc.unit} '
              f'(median {pct(late, .5) / 1e3:.0f} us)')

    # ---- GPU-side cross-queue latency ---------------------------------------------------------------------
    lat = defaultdict(list)
    for s in in_win:
        if s['start'] is None:
            continue
        for d in s['deps']:
            dd = subs[d]
            if (dd['qf'], dd['q']) == (s['qf'], s['q']) or dd['end_est'] is None:
                continue
            if s['wall'] > dd['end_est']:
                continue  # submitted after the dependency finished: no GPU-side wait
            lat[(dd['qf'], s['qf'])].append(s['start'] - dd['end_est'])
    J['xq_latency'] = {}
    for (a, b), xs in sorted(lat.items()):
        J['xq_latency'][f'{a}->{b}'] = {'count': sc.cnt(len(xs)), 'median_us': pct(xs, .5) / 1e3, 'p90_us': pct(xs, .9) / 1e3}
        print(f'cross-queue GPU wait {a} -> {b}: {sc.cnt(len(xs)):.1f}{sc.nunit}, waiter start - signaller end: '
              f'median {pct(xs, .5) / 1e3:.0f} us, p90 {pct(xs, .9) / 1e3:.0f} us, min {min(xs) / 1e3:.0f} us')

    if o['trace']:
        vkd3d_trace(o['trace'], seg, in_win, subs, gkey, queues, unions, sc, J)


def vkd3d_trace(path, seg, in_win, subs, gkey, queues, unions, sc, J):
    """VKD3D_QUEUE_PROFILE_ABSOLUTE=1 trace: blocking regions of the vkd3d submission threads."""
    raws = [(t['cpu_raw_at_ns'], t['raw_ns']) for t in seg['T'] if 'raw_ns' in t and 'cpu_raw_at_ns' in t]
    if not raws:
        print('vkd3d trace: the log has no raw_ns T lines (needs a v2 timer)')
        return
    mono_minus_raw = statistics.median(c - r for c, r in raws)
    lo = min(s['wall'] for s in in_win)
    hi = max(s['wall'] for s in in_win)
    ev = re.compile(r'"name": "([^"]*)", "ph": "X", "tid": "([^"]*)", "pid": "([^"]*)", "ts": ([\d.]+), "dur": ([\d.]+)')
    regions = defaultdict(list)   # (pid, name) -> [(a, b)] CPU MONOTONIC ns
    submits = defaultdict(list)   # pid -> [(a, b)] "submit" regions
    with open(path, errors='replace') as f:
        for line in f:
            m = ev.search(line)
            if not m:
                continue
            name, tid, pid, ts, dur = m.group(1), m.group(2), m.group(3), float(m.group(4)), float(m.group(5))
            a = ts * 1e3 + mono_minus_raw
            b = a + dur * 1e3
            if b < lo or a > hi:
                continue
            if tid == 'submit':
                submits[pid].append((a, b))
            if name in ('WAIT BEFORE SIGNAL', 'CPU WAIT', 'WAIT (shared)', 'STOP'):
                regions[(pid, name)].append((a, b))
    # vkd3d thread -> Vulkan queue: whose W call_ns falls into the thread's "submit" regions
    qmap = {}
    for pid, iv in submits.items():
        iv.sort()
        starts = [a for a, b in iv]
        votes = defaultdict(int)
        for s in in_win:
            if s['call'] is None:
                continue
            i = bisect.bisect_right(starts, s['call']) - 1
            if i >= 0 and iv[i][0] <= s['call'] <= iv[i][1] + 50e3:
                votes[(s['qf'], s['q'])] += 1
        if votes:
            qmap[pid] = max(votes, key=votes.get)
    idle = []
    gu = unions[gkey]
    for (a0, b0), (a1, b1) in zip(gu, gu[1:]):
        idle.append([b0, a1])
    J['vkd3d'] = {}
    print(f'vkd3d-proton submission-thread blocking (VKD3D_QUEUE_PROFILE; MONOTONIC-RAW offset {mono_minus_raw / 1e6:.3f} ms):')
    for pid in sorted(qmap):
        print(f'  thread {pid} submits to {qmap[pid][0]} {qmap[pid][1]}')
    for (pid, name), iv in sorted(regions.items()):
        u = union(iv)
        ov = overlap_len(u, idle)
        q = qmap.get(pid, ('?', '?'))
        J['vkd3d'][f'{pid}:{name}'] = {'queue': f'{q[0]}:{q[1]}', 'count': sc.cnt(len(iv)), 'ms': sc.ms(total(u)),
                                        'during_graphics_idle_ms': sc.ms(ov)}
        print(f'  thread {pid} ({q[0]} {q[1]}) {name}: {sc.cnt(len(iv)):.1f}{sc.nunit} {sc.ms(total(u)):.2f} {sc.unit}, '
              f'of it while the graphics queue was idle {sc.ms(ov):.2f}')


if __name__ == '__main__':
    main()
