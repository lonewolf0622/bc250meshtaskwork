#!/usr/bin/env python3
"""summarize.py <BC250_MESH_TIMER file> [--top N] [--window MS]

Reads a file written by BC250_MESH_TIMER_FILE (lines: "# BC250_MESH_TIMER v1 ..." headers, "I" interval
totals, "C" per-class rows, "K" skipped classes, optional "S" per-submit rows with BC250_MESH_TIMER_SUBMITS=1)
and prints the classes by GPU time (whole run, all interval rows summed), the Mesh share of the submitted
GPU time and a per-frame estimate: per present when the driver counted presents (frames > 0), otherwise
per submit, and, with S rows, per wall-clock window (--window, default 16.7 ms) as a second estimate.
Per-draw intervals are bottom-of-pipe timestamps around the whole draw, so they include pipeline overlap
with neighbouring work: use the sums as a relative measure between classes."""
import sys
from collections import defaultdict


def kv(line):
    d = {}
    for tok in line.split()[1:]:
        if '=' in tok:
            k, v = tok.split('=', 1)
            d[k] = v
    return d


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(2)
    top = 25
    window_ms = 16.7
    path = None
    i = 0
    while i < len(args):
        if args[i] == '--top':
            top = int(args[i + 1]); i += 2
        elif args[i] == '--window':
            window_ms = float(args[i + 1]); i += 2
        else:
            path = args[i]; i += 1
    runs = 0
    tot = defaultdict(float)
    classes = {}
    skipped = []
    submits = []
    intervals = []
    for line in open(path, errors='replace'):
        line = line.rstrip('\n')
        if line.startswith('# BC250_MESH_TIMER'):
            runs += 1
            continue
        if line.startswith('I interval'):
            d = kv(line)
            intervals.append(d)
            for k in ('dt', 'frames', 'submits', 'submit_ns', 'mesh_ns', 'mesh_draws', 'timed', 'vs_draws', 'skipped',
                      'dropped_slots', 'dropped_submits', 'invalid'):
                tot[k] += float(d.get(k, 0))
        elif line.startswith('K '):
            skipped.append(line[2:])
        elif line.startswith('S '):
            d = kv(line)
            if d.get('qf', 'gfx') == 'gfx':  # v2 logs also have compute-queue S lines
                submits.append(d)
    # C rows: only those of interval prints (the rows after "I final" repeat the run).
    classes_i = {}
    state = None
    for line in open(path, errors='replace'):
        if line.startswith('I interval'):
            state = 'interval'
        elif line.startswith('I final') or line.startswith('#'):
            state = 'final'
        elif line.startswith('C ') and state == 'interval':
            d = kv(line)
            key = (d['hash'], d['route'], d['V'], d['P'], d['LS'], d['W'], d['WG'])
            c = classes_i.setdefault(key, defaultdict(float))
            if float(d['timed']):
                c['min'] = min(c.get('min', float('inf')), float(d['min_ns']))
            c['max'] = max(c.get('max', 0.0), float(d['max_ns']))
            for k in ('draws', 'direct', 'indirect', 'count', 'timed', 'sum_ns', 'skipped'):
                c[k] += float(d.get(k, 0))
    classes = classes_i

    dt = tot['dt'] or 1e-9
    frames = tot['frames']
    mesh_ms = tot['mesh_ns'] / 1e6
    submit_ms = tot['submit_ns'] / 1e6
    print('runs=%d intervals=%d wall_s=%.1f submits=%d frames=%d mesh_draws=%d timed=%d vs_draws=%d skipped=%d '
          'dropped_slots=%d dropped_submits=%d invalid=%d' %
          (runs, len(intervals), dt, tot['submits'], frames, tot['mesh_draws'], tot['timed'], tot['vs_draws'],
           tot['skipped'], tot['dropped_slots'], tot['dropped_submits'], tot['invalid']))
    print('gpu: submit_ms/s=%.1f mesh_ms/s=%.1f mesh_share_of_submit=%.1f%%' %
          (submit_ms / dt, mesh_ms / dt, 100.0 * mesh_ms / submit_ms if submit_ms else 0.0))
    if tot['timed'] and tot['mesh_draws'] and tot['timed'] < tot['mesh_draws']:
        scale = tot['mesh_draws'] / tot['timed']
        print('note: %.0f of %.0f Mesh draws timed (BC250_MESH_TIMER_SAMPLE): Mesh sums scaled by %.2f below' %
              (tot['timed'], tot['mesh_draws'], scale))
    else:
        scale = 1.0
    if frames:
        print('per frame (presents): submit_ms=%.3f mesh_ms=%.3f mesh_draws=%.1f fps_wall=%.1f' %
              (submit_ms / frames, mesh_ms * scale / frames, tot['mesh_draws'] / frames, frames / dt))
    elif tot['submits']:
        print('per submit (no presents counted): submit_ms=%.3f mesh_ms=%.3f mesh_draws=%.1f' %
              (submit_ms / tot['submits'], mesh_ms * scale / tot['submits'], tot['mesh_draws'] / tot['submits']))
    if submits:
        # group consecutive submits into wall-clock windows
        submits.sort(key=lambda d: float(d['wall_ns']))
        wins = []
        start = None
        for d in submits:
            w = float(d['wall_ns'])
            if start is None or w - start > window_ms * 1e6:
                start = w
                wins.append([0.0, 0.0, 0])
            wins[-1][0] += float(d['submit_ns'])
            wins[-1][1] += float(d['mesh_ns'])
            wins[-1][2] += int(d['mesh_draws'])
        n = len(wins)
        print('per %.1f ms window (%d windows from %d S rows): submit_ms=%.3f mesh_ms=%.3f mesh_draws=%.1f' %
              (window_ms, n, len(submits), sum(w[0] for w in wins) / 1e6 / n, sum(w[1] for w in wins) / 1e6 / n * scale,
               sum(w[2] for w in wins) / n))
    if skipped:
        print('skipped classes (BC250_MESH_TIMER_SKIP):')
        for s in skipped:
            print('  ' + s)
    rows = sorted(classes.items(), key=lambda kv_: -kv_[1]['sum_ns'])
    total_sum = sum(c['sum_ns'] for _, c in rows) or 1.0
    print('%-16s %-26s %-24s %9s %9s %9s %6s %7s %8s %8s %8s %s' %
          ('hash', 'class', 'route', 'draws/s', 'timed/s', 'gpu_ms/s', 'mesh%', 'submit%', 'avg_us', 'min_us', 'max_us',
           'per_frame_ms' if frames else 'per_submit_ms'))
    div = frames or tot['submits'] or 1.0
    for (h, route, V, P, LS, W, WG), c in rows[:top]:
        cls = 'V=%s P=%s LS=%s W%s WG%s' % (V, P, LS, W, WG)
        print('%-16s %-26s %-24s %9.0f %9.0f %9.2f %6.1f %7.1f %8.1f %8.1f %8.1f %8.3f%s' %
              (h, cls, route, c['draws'] / dt, c['timed'] / dt, c['sum_ns'] / 1e6 / dt, 100.0 * c['sum_ns'] / total_sum,
               100.0 * c['sum_ns'] / tot['submit_ns'] if tot['submit_ns'] else 0.0,
               c['sum_ns'] / 1e3 / c['timed'] if c['timed'] else 0.0,
               c['min'] / 1e3 if c['timed'] else 0.0, c['max'] / 1e3,
               c['sum_ns'] * scale / 1e6 / div, ' (SKIPPED %d)' % c['skipped'] if c['skipped'] else ''))
    if len(rows) > top:
        print('... %d more classes' % (len(rows) - top))


if __name__ == '__main__':
    main()
