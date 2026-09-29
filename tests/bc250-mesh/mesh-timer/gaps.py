#!/usr/bin/env python3
"""gaps.py <timer log> [--from S] [--to S]: GPU idle-gap analysis of a BC250_MESH_TIMER log written with
BC250_MESH_TIMER_SUBMITS=1 (S lines with gpu_t0/gpu_t1) and T calibration lines.

Every submission is placed on the CPU CLOCK_MONOTONIC timeline (GPU ticks converted with the T lines).
The GPU is idle between the end of all earlier work and the start of the next submission. Each gap is split
into:
  cpu_late : the next submission reached the driver after the GPU had already gone idle
             (the GPU starved: CPU / game / driver recording or submission was late)
  gpu_wait : the next submission had been handed to the driver before the GPU went idle, but still started
             later (waits on semaphores/fences, queue scheduling, kernel)
Gaps are also attributed to whether the submission before and after the gap contains Mesh draws.
Numbers are per frame (frames from the I lines) over the selected window of the log (seconds since start).
"""
import re, sys, statistics

args = sys.argv[1:]
if not args:
    sys.exit(__doc__)
path = args[0]
t_from = float(args[args.index('--from') + 1]) if '--from' in args else None
t_to = float(args[args.index('--to') + 1]) if '--to' in args else None

ns_per_tick = 10.0
cal, subs, frames_total, first_wall = [], [], 0, None
kv = re.compile(r'(\w+)=("[^"]*"|\S+)')
for line in open(path, errors='replace'):
    if line.startswith('#'):
        m = re.search(r'ns_per_tick=([\d.]+)', line)
        if m:
            ns_per_tick = float(m.group(1))
        continue
    tag = line[:2]
    if tag not in ('S ', 'T ', 'I '):
        continue
    d = {k: v for k, v in kv.findall(line)}
    if tag == 'T ':
        cal.append((int(d['cpu_ns']), int(d['gpu_tick'])))
    elif tag == 'S ':
        if d.get('qf', 'gfx') != 'gfx':
            continue  # v2 logs: compute-queue command buffers (queuetl.py analyses those)
        if 'gpu_t0' not in d:
            sys.exit('log has no gpu_t0/gpu_t1: needs a timer build with the gap fields')
        t0, t1 = int(d['gpu_t0']), int(d['gpu_t1'])
        if t0 and t1:
            subs.append((int(d['wall_ns']), t0, t1, int(d['mesh_draws']), int(d['vs_draws'])))
    elif tag == 'I ' and line.startswith('I interval'):
        frames_total += int(d.get('frames', 0))

if not cal:
    sys.exit('no T calibration lines (the log must come from a run of at least one print interval)')
# offset such that cpu_ns = gpu_tick * ns_per_tick + off (median over calibrations; drift is reported)
offs = [c - g * ns_per_tick for c, g in cal]
off = statistics.median(offs)
drift_us = (max(offs) - min(offs)) / 1e3

subs.sort(key=lambda s: s[1])
start_wall = min(s[0] for s in subs)
sel = []
for wall, t0, t1, md, vd in subs:
    ts = (wall - start_wall) / 1e9
    if (t_from is not None and ts < t_from) or (t_to is not None and ts > t_to):
        continue
    sel.append((wall, t0 * ns_per_tick + off, t1 * ns_per_tick + off, md, vd))
if len(sel) < 2:
    sys.exit('too few submissions in the window')

busy = idle = cpu_late = gpu_wait = 0.0
by_kind = {}
end_max = sel[0][2]
busy += sel[0][2] - sel[0][1]
prev_mesh = sel[0][3] > 0
for wall, s0, s1, md, vd in sel[1:]:
    if s0 > end_max:
        gap = s0 - end_max
        idle += gap
        late = min(gap, max(0.0, wall - end_max))
        cpu_late += late
        gpu_wait += gap - late
        key = ('mesh' if prev_mesh else 'plain') + '->' + ('mesh' if md else 'plain')
        k = by_kind.setdefault(key, [0, 0.0, 0.0])
        k[0] += 1; k[1] += late; k[2] += gap - late
    busy += max(0.0, s1 - max(s0, end_max))
    end_max = max(end_max, s1)
    prev_mesh = md > 0

span = end_max - sel[0][1]
span_s = span / 1e9
# frames in the window: scale the log's frame count by the window's share of the submissions
n_frames = frames_total * len(sel) / max(len(subs), 1) if frames_total else 0
fpf = (lambda x: x / n_frames / 1e6) if n_frames else (lambda x: x / span_s / 1e6)
unit = 'ms/frame' if n_frames else 'ms/s'
print(f'window: {len(sel)} submissions, {span_s:.1f} s GPU timeline, ~{n_frames:.0f} frames '
      f'({n_frames / span_s if span_s else 0:.1f} fps); calibration drift {drift_us:.0f} us over {len(cal)} T lines')
print(f'GPU busy {fpf(busy):.2f} {unit} | idle {fpf(idle):.2f} {unit} '
      f'= cpu_late {fpf(cpu_late):.2f} + gpu_wait {fpf(gpu_wait):.2f}')
print('idle by neighbours (count, cpu_late, gpu_wait):')
for key, (n, late, wait) in sorted(by_kind.items(), key=lambda kv: -(kv[1][1] + kv[1][2])):
    print(f'  {key:12s} gaps={n:7d}  cpu_late={fpf(late):.3f} {unit}  gpu_wait={fpf(wait):.3f} {unit}')
