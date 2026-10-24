#!/usr/bin/env python3
"""Guide a manual Steam game comparison and audit per-frame MangoHud logs."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time

SETUP_PHASES = ('A1', 'B1', 'A2')
ROUTE_PHASES = ('M1', 'R1', 'S1', 'R2', 'M2')
PHASES = SETUP_PHASES
FLAGS = dict(RADV_DIRECTMESH='1', RADV_BC250_MESH_IDXPASS='1',
             RADV_BC250_MESH_IDXPASS_STORE='0', RADV_BC250_MESH_IDXPASS_BATCH_TRIM='1')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def phase_env(root, phase):
    if phase in ROUTE_PHASES:
        route_flags = dict(RADV_BC250_MESH_IDXPASS=str(int(phase[0] != 'M')),
                           RADV_BC250_MESH_IDXPASS_STORE=str(int(phase[0] == 'S')),
                           RADV_BC250_MESH_IDXPASS_SETUP_BATCH=str(int(phase[0] != 'M')))
    elif phase in SETUP_PHASES:
        route_flags = dict(RADV_BC250_MESH_IDXPASS_SETUP_BATCH=str(int(phase == 'B1')))
    else:
        raise ValueError('unknown comparison phase')
    return dict(FLAGS, **route_flags,
                VK_DRIVER_FILES=str(root / 'radeon_icd.json'), VK_ICD_FILENAMES=str(root / 'radeon_icd.json'),
                MANGOHUD='1', MANGOHUD_CONFIGFILE=str(root / 'configs' / (phase + '.conf')),
                MESA_SHADER_CACHE_DIR=str(root / 'shader-cache'), BC250_TRACE_USAGE='1')


def launch_options(root, phase):
    env = phase_env(root, phase)
    # Steam expands its own command placeholder; values and redirect paths are shell quoted.
    return ' '.join(k + '=' + shlex.quote(v) for k, v in env.items()) + ' %command% > ' + \
        shlex.quote(str(root / 'results' / phase / 'driver.log')) + ' 2>&1'


def prepare(root):
    if (root / 'SHA256SUMS').exists():
        raise ValueError('this kit is already sealed; preserve its configuration')
    (root / 'configs').mkdir(exist_ok=True)
    (root / 'shader-cache').mkdir(exist_ok=True)
    for phase in PHASES:
        folder = root / 'results' / phase
        folder.mkdir(parents=True, exist_ok=True)
        config = '\n'.join(('fps', 'frametime', 'frame_timing', 'gpu_stats', 'gpu_core_clock',
                            'gpu_mem_clock', 'gpu_temp', 'cpu_stats', 'cpu_temp', 'vram', 'ram',
                            'fps_limit=0', 'autostart_log=0', 'log_duration=90', 'log_interval=0',
                            'toggle_logging=Shift_L+F2', 'permit_upload=0', 'output_folder=' + str(folder))) + '\n'
        (root / 'configs' / (phase + '.conf')).write_text(config)
        (root / ('launch-' + phase + '.txt')).write_text(launch_options(root, phase) + '\n')


def frame_metrics(path):
    lines = path.read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('fps,frametime,')), None)
    if start is None:
        raise ValueError('not a raw per-frame MangoHud CSV: ' + path.name)
    rows = list(csv.DictReader(lines[start:]))
    if len(rows) < 300:
        raise ValueError('too few frames: ' + path.name)
    times = [float(row['frametime']) for row in rows]
    elapsed = [int(row['elapsed']) for row in rows]
    if not all(math.isfinite(t) and t > 0 for t in times):
        raise ValueError('invalid frame times: ' + path.name)
    if not all(b > a for a, b in zip(elapsed, elapsed[1:])):
        raise ValueError('non-monotonic timestamps: ' + path.name)
    duration = (elapsed[-1] - elapsed[0]) / 1e9
    if not 88 <= duration <= 92:
        raise ValueError(f'{path.name}: expected 90-second capture, found {duration:.2f}s')
    coverage = sum(times) / (duration * 1000)
    if not .85 <= coverage <= 1.15:
        raise ValueError('sampled or incomplete log; per-frame coverage failed: ' + path.name)
    ordered = sorted(times)
    percentile = lambda p: ordered[math.ceil(p * len(ordered)) - 1]
    result = dict(frames=len(rows), duration_s=duration, frame_time_coverage_ratio=coverage,
                  mean_frame_ms=statistics.mean(times), median_frame_ms=statistics.median(times),
                  fps=1000 / statistics.mean(times), p95_frame_ms=percentile(.95),
                  p99_frame_ms=percentile(.99), max_frame_ms=max(times),
                  frames_over_33_ms=sum(t > 33.333 for t in times),
                  frames_over_50_ms=sum(t > 50 for t in times),
                  frames_over_100_ms=sum(t > 100 for t in times), image_correctness='owner observation',
                  csv=path.name, csv_sha256=digest(path))
    for name in ('gpu_core_clock', 'gpu_mem_clock', 'gpu_load', 'gpu_temp', 'cpu_load'):
        values = [float(row[name]) for row in rows if row.get(name, '').strip()]
        if values and all(math.isfinite(v) for v in values):
            result[name + '_mean'] = statistics.mean(values)
    return result


def comparison(rows):
    if set(rows) == set(ROUTE_PHASES):
        return route_comparison(rows)
    a, b, c = (rows[p]['metrics'] for p in PHASES)
    baseline = (a['mean_frame_ms'] + c['mean_frame_ms']) / 2
    drift = (max(a['mean_frame_ms'], c['mean_frame_ms']) / min(a['mean_frame_ms'], c['mean_frame_ms']) - 1) * 100
    speedup = baseline / b['mean_frame_ms']
    threshold = 1 + max(.05, drift / 100)
    valid_images = all(rows[p]['visual_check'] == 'pass' for p in PHASES)
    tail_baseline = (a['p99_frame_ms'] + c['p99_frame_ms']) / 2
    tail_ratio = b['p99_frame_ms'] / tail_baseline
    exercised = all(rows[p]['metrics'].get('IDX_indirect_calls_during_capture', 0) > 0 for p in PHASES)
    return dict(baseline_frame_ms=baseline, batched_frame_ms=b['mean_frame_ms'],
                baseline_fps=1000 / baseline, batched_fps=b['fps'], speedup=speedup,
                frame_time_change_percent=(1 / speedup - 1) * 100,
                baseline_drift_percent=drift, baseline_stable=drift <= 5,
                p99_batched_over_baseline=tail_ratio, visual_checks_passed=valid_images,
                indirect_IDX_work_observed=exercised,
                candidate_gain=exercised and valid_images and drift <= 5 and speedup > threshold and tail_ratio <= 1.1,
                candidate_regression=valid_images and drift <= 5 and 1 / speedup > threshold,
                scope='One scene, same driver, replay IDX, setup off/on/off; exploratory game comparison')


def route_comparison(rows):
    """Keep Mesh and replay brackets separate and require observed IDX work."""
    def score(candidate, baseline_phases):
        a, b, c = (rows[p]['metrics'] for p in (baseline_phases[0], candidate, baseline_phases[1]))
        baseline = (a['mean_frame_ms'] + c['mean_frame_ms']) / 2
        drift = 100 * (max(a['mean_frame_ms'], c['mean_frame_ms']) / min(a['mean_frame_ms'], c['mean_frame_ms']) - 1)
        speedup = baseline / b['mean_frame_ms']
        threshold = 1 + max(.05, drift / 100)
        valid = all(rows[p]['visual_check'] == 'pass' for p in ROUTE_PHASES)
        tail = b['p99_frame_ms'] / ((a['p99_frame_ms'] + c['p99_frame_ms']) / 2)
        used = lambda p: sum(rows[p]['metrics'].get(k, 0) for k in
                             ('IDX_direct_calls_during_capture', 'IDX_indirect_calls_during_capture'))
        exercised = used(candidate) > 0 and all(used(p) > 0 if p[0] != 'M' else used(p) == 0
                                                for p in baseline_phases)
        return dict(candidate=candidate, baseline_phases=list(baseline_phases), baseline_frame_ms=baseline,
                    candidate_frame_ms=b['mean_frame_ms'], baseline_fps=1000 / baseline,
                    candidate_fps=b['fps'], speedup=speedup, baseline_drift_percent=drift,
                    baseline_stable=drift <= 5, p99_candidate_over_baseline=tail,
                    visual_checks_passed=valid, expected_route_work_observed=exercised,
                    candidate_gain=exercised and valid and drift <= 5 and speedup > threshold and tail <= 1.1,
                    candidate_regression=exercised and valid and drift <= 5 and 1 / speedup > threshold,
                    p99_regression=valid and drift <= 5 and tail > 1.1)

    return dict(comparisons=[score(p, ('M1', 'M2')) for p in ('R1', 'S1', 'R2')] +
                            [score('S1', ('R1', 'R2'))],
                scope='One scene, same driver, Mesh/replay/stored/replay/Mesh; exploratory game comparison',
                shared_setup_spans='UNPROVEN; usage counters count IDX calls, not shared spans',
                automatic_route_policy_changed=False)


def game_pids():
    found = []
    for p in Path('/proc').glob('[0-9]*'):
        try:
            name = (p / 'comm').read_text().strip()
            # Linux comm is limited to 15 bytes; shipping executable names may be truncated.
            if re.match(r'^ff7', name, re.I):
                found.append(p)
        except OSError:
            pass
    return found


def check_environment(raw, root, phase):
    env = dict(item.split('=', 1) for item in raw.split('\0') if '=' in item)
    expected = phase_env(root, phase)
    keys = [*FLAGS, 'RADV_BC250_MESH_IDXPASS_SETUP_BATCH', 'MANGOHUD_CONFIGFILE', 'BC250_TRACE_USAGE']
    for key in keys:
        if env.get(key) != expected[key]:
            raise ValueError(key + ': game has not loaded this phase\'s launch options')
    if any(env.get(k) for k in ('AMDGPU_GPU_ID', 'BC250_IDX_TRACE', 'BC250_TRACE_COMPILE', 'MESA_VK_TRACE')):
        raise ValueError('noop or detailed tracing enabled; do not score this run')
    # Record a small allowlist only. Never save the full process environment.
    return {k: env[k] for k in expected if k in env}


def verify_game(root, phase, manifest):
    matches = []
    for p in game_pids():
        try:
            maps = (p / 'maps').read_text()
        except OSError:
            continue
        paths = {line.split(maxsplit=5)[5] for line in maps.splitlines()
                 if 'libvulkan_radeon.so' in line and len(line.split(maxsplit=5)) == 6}
        if not paths:
            continue  # Bootstrap process may not render.
        env = check_environment((p / 'environ').read_bytes().decode(errors='replace'), root, phase)
        identities = []
        for name in paths:
            if name.endswith(' (deleted)'):
                raise ValueError('mapped driver was replaced during the game run')
            mapped = p / 'root' / name.lstrip('/')
            sha = digest(mapped)
            if sha != manifest['driver_sha256']:
                raise ValueError('FF7 loaded a different RADV driver; preserve driver.log')
            identities.append(dict(mapped_path=name, sha256=sha))
        if 'mangohud' not in maps.lower():
            raise ValueError('MangoHud is not loaded in the rendering process')
        matches.append(dict(pid=int(p.name), environment=env, drivers=identities))
    if len(matches) != 1:
        raise ValueError('expected one rendering FF7 process, found ' + str(len(matches)))
    return matches[0]


def check_background(allow_game=False):
    if Path('/sys/module/amdgpu/parameters/gpu_recovery').read_text().strip() != '0':
        raise ValueError('gpu_recovery must be 0; this runner does not change it')
    for p in Path('/proc').glob('[0-9]*'):
        try:
            name = (p / 'comm').read_text().strip()
            if re.match(r'^(meshbench|index_gate|deqp-vk|Control.*|Hellblade.*)$', name, re.I):
                raise ValueError('other game or GPU workload running: ' + name)
            if p.name != str(os.getpid()) and '/gpuq/runner.sh' in (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'):
                raise ValueError('another GPU runner is active')
        except OSError:
            pass
    if not allow_game:
        if game_pids():
            raise ValueError('close FF7 before selecting the next phase')
        busy = [p.read_text().strip() for p in Path('/sys/class/drm').glob('card[0-9]*/device/gpu_busy_percent')]
        if any(int(value) > 5 for value in busy):
            raise ValueError('GPU busy; close other GPU work before starting')


def capture(root, phase, manifest):
    folder = root / 'results' / phase
    started, finished = folder / 'capture.started.json', folder / 'capture.finished.json'
    if started.exists():
        raise ValueError('phase already started; preserve logs and do not overwrite it')
    if list(folder.glob('*.csv')):
        raise ValueError('existing CSV before capture; preserve it and use a fresh kit')
    check_background(allow_game=True)
    identity = verify_game(root, phase, manifest)
    log = folder / 'driver.log'
    usage_before = usage_counts(log)
    save(started, dict(phase=phase, UTC=datetime.now(timezone.utc).isoformat(), game=identity,
                       IDX_usage_before=usage_before))
    print('Return to FF7 and press LEFT SHIFT + F2 once. Keep the same camera/scene for 90 seconds.', flush=True)
    print('Logging stops automatically. Do not alt-tab or enter menus until it stops.', flush=True)
    deadline = time.monotonic() + 300
    selected = None
    while time.monotonic() < deadline:
        raw = [p for p in folder.glob('*.csv') if not p.name.endswith('_summary.csv')]
        if len(raw) > 1:
            raise ValueError('multiple captures found; preserve logs for inspection')
        if raw:
            selected = raw[0]
            summary = selected.with_name(selected.stem + '_summary.csv')
            if summary.exists() and summary.stat().st_size and time.time() - selected.stat().st_mtime > 2:
                break
        time.sleep(1)
    else:
        raise ValueError('capture did not complete; preserve logs and capture.started.json')
    metrics = frame_metrics(selected)
    after = verify_game(root, phase, manifest)
    if after['pid'] != identity['pid']:
        raise ValueError('game restarted during capture')
    if not log.exists():
        raise ValueError('driver.log missing; use the complete generated launch option')
    usage_after = usage_counts(log)
    metrics['IDX_usage_before'] = usage_before
    metrics['IDX_usage_after'] = usage_after
    if any(a < b for a, b in zip(usage_after, usage_before)):
        raise ValueError('IDX usage counters went backwards; preserve logs')
    metrics['IDX_direct_calls_during_capture'] = usage_after[0] - usage_before[0]
    metrics['IDX_indirect_calls_during_capture'] = max(0, usage_after[1] - usage_before[1])
    snapshot = folder / 'driver.capture.log'
    with snapshot.open('xb') as f:
        f.write(log.read_bytes())
    visual = input('Rendering looked correct in this scene? [y/n]: ').strip().lower()
    if visual not in ('y', 'yes', 'n', 'no'):
        raise ValueError('visual check needs y or n; preserve the captured CSV')
    row = dict(phase=phase, metrics=metrics, game_before=identity, game_after=after,
               driver_log_sha256=digest(snapshot), visual_check='pass' if visual in ('y', 'yes') else 'fail')
    save(finished, row)
    print(f'Saved {phase}: {metrics["fps"]:.2f} FPS, mean {metrics["mean_frame_ms"]:.3f} ms, p99 {metrics["p99_frame_ms"]:.3f} ms')
    if row['visual_check'] != 'pass':
        raise ValueError('visual difference reported; stop the comparison and retain evidence')


def usage_counts(path):
    if not path.exists():
        return [0, 0]
    found = re.findall(r'BC250_USAGE .*idx_direct=(\d+) idx_indirect_calls=(\d+)', path.read_text(errors='replace'))
    return list(map(int, found[-1])) if found else [0, 0]


def main():
    global PHASES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--routes', action='store_true', help='compare Mesh, replay IDX and stored IDX')
    parser.add_argument('--launch-options', choices=SETUP_PHASES + ROUTE_PHASES)
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text()) if (root / 'manifest.json').exists() else {}
    routes = args.routes or manifest.get('comparison_kind') == 'mesh_replay_stored'
    PHASES = ROUTE_PHASES if routes else SETUP_PHASES
    if args.launch_options and args.launch_options not in PHASES:
        raise ValueError('launch option phase does not belong to this kit')
    if args.prepare:
        prepare(root)
        return
    subprocess.run(['sha256sum', '-c', '--quiet', 'SHA256SUMS'], cwd=root, check=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    if args.launch_options:
        print(launch_options(root, args.launch_options))
        return
    if not args.compare:
        session = root / 'results/session.json'
        if not session.exists():
            check_background()
            print('FF7 test: Mesh, replay IDX, stored IDX, replay IDX, Mesh. Same driver; shared setup on for IDX.'
                  if routes else 'FF7 test: A1 batching off, B1 batching on, A2 batching off. Same driver; stored IDX off.')
            print('Use one save and camera position. Keep resolution/upscaler/settings fixed; disable frame generation, dynamic resolution, VSync and FPS caps for this comparison.')
            scene = input('Scene/save and graphics settings (briefly): ').strip()
            if not scene:
                raise ValueError('record the scene and settings before measuring')
            save(session, dict(UTC=datetime.now(timezone.utc).isoformat(), scene_and_settings=scene,
                               manifest=manifest, MangoHud_log_interval_ms=0))
        for phase in PHASES:
            folder = root / 'results' / phase
            if (folder / 'capture.finished.json').exists():
                saved = json.loads((folder / 'capture.finished.json').read_text())
                if saved['visual_check'] != 'pass':
                    raise ValueError('a previous phase reported a visual difference; do not continue')
                continue
            if (folder / 'capture.started.json').exists() or list(folder.glob('*.csv')):
                raise ValueError('incomplete or existing capture for ' + phase + '; preserve it and do not restart')
            check_background()
            print('\n' + phase + ': paste this complete line into FF7 Steam launch options:\n')
            print(launch_options(root, phase), flush=True)
            print('\nLaunch FF7 yourself. Load the same save/camera and warm up there for at least two minutes.')
            input('When ready to measure, return here and press Enter: ')
            capture(root, phase, manifest)
            input('Exit FF7 completely, then press Enter for the next phase: ')
    rows = {p: json.loads((root / 'results' / p / 'capture.finished.json').read_text()) for p in PHASES}
    for phase, row in rows.items():
        folder = root / 'results' / phase
        if Path(row['metrics']['csv']).name != row['metrics']['csv']:
            raise ValueError('invalid CSV filename in saved phase')
        metrics = frame_metrics(folder / row['metrics']['csv'])
        if any(row['metrics'].get(k) != v for k, v in metrics.items()):
            raise ValueError('CSV differs from the saved measurement: ' + phase)
        if row['driver_log_sha256'] != digest(folder / 'driver.capture.log'):
            raise ValueError('driver evidence changed: ' + phase)
        if routes:
            start = json.loads((folder / 'capture.started.json').read_text())
            before = start['IDX_usage_before']
            after = usage_counts(folder / 'driver.capture.log')
            if (start['phase'] != phase or start['game'] != row['game_before'] or
                    before != row['metrics']['IDX_usage_before'] or after != row['metrics']['IDX_usage_after'] or
                    any(a < b for a, b in zip(after, before)) or
                    row['metrics']['IDX_direct_calls_during_capture'] != after[0] - before[0] or
                    row['metrics']['IDX_indirect_calls_during_capture'] != after[1] - before[1]):
                raise ValueError('saved usage evidence differs: ' + phase)
        if row['phase'] != phase:
            raise ValueError('saved phase label differs from its folder')
        for game in ('game_before', 'game_after'):
            check_environment('\0'.join(k + '=' + v for k, v in row[game]['environment'].items()), root, phase)
            if not row[game]['drivers'] or any(d['sha256'] != manifest['driver_sha256'] for d in row[game]['drivers']):
                raise ValueError('saved driver identity differs: ' + phase)
    result = dict(manifest=manifest, runs=rows, comparison=comparison(rows),
                  game='FF7 Rebirth', GPU_runs_by_agent=0)
    text = json.dumps(result, indent=2) + '\n'
    summary = root / 'results/summary.json'
    if summary.exists():
        if summary.read_text() != text:
            raise ValueError('saved summary differs; preserve both inputs for inspection')
    else:
        summary.write_text(text)
    c = result['comparison']
    if routes:
        for row in c['comparisons']:
            print(f'{row["candidate"]} vs {"/".join(row["baseline_phases"])}: {row["speedup"]:.3f}x, '
                  f'drift {row["baseline_drift_percent"]:.2f}%, candidate gain={row["candidate_gain"]}, '
                  f'regression={row["candidate_regression"]}, p99 regression={row["p99_regression"]}')
    else:
        print(f'Baseline {c["baseline_fps"]:.2f} FPS; batched {c["batched_fps"]:.2f} FPS; '
              f'{c["speedup"]:.3f}x; baseline drift {c["baseline_drift_percent"]:.2f}%')
        print('Candidate gain:', c['candidate_gain'], '| Candidate regression:', c['candidate_regression'])
    print('Saved results/summary.json. Restore your previous Steam launch options after the comparison.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError, KeyboardInterrupt, EOFError) as error:
        print('STOP:', error, '\nPreserve this kit and its logs.', file=sys.stderr)
        sys.exit(1)
