#!/usr/bin/env python3
"""Offline validation or one-shot hardware timing of the gated Mesh fixtures."""
import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

HERE = Path(__file__).resolve().parent
BAD = r'(?i)validation failed|ACO ERROR|Assertion|Segmentation|REFUSE:|INDEX_STOP|FAIL .*:|VALIDATION:.*Error'
MODES = ('mesh-before', 'existing-before', 'stored', 'existing-after', 'mesh-after')
CASES = [dict(name=name, draw=draw, draws=draws,
              tag=f'{name}-{draw}-draws{draws}', primitives=128 if name == 'pp128' else 64)
         for name in ('j64', 'pp64', 'pp128', 'ballot_vertex64', 'shared_vertex64')
         for draw in (('direct',) if name == 'pp128' else ('direct', 'multi', 'count'))
         for draws in (1, 16)]
HEAVY_CASES = [dict(name=name, draw=draw, draws=draws, steps=steps,
                    tag=f'{name}-{draw}-steps{steps}-draws{draws}',
                    primitives=128 if name == 'heavy_pp128' else 64,
                    fragment='pp' if name == 'heavy_pp128' else 'plain',
                    existing_fallback=name == 'heavy_ballot64', require_mesh_reference=True,
                    reference_key=f'{name}-{draw}-steps{steps}')
               for name in ('heavy64', 'heavy_pp128', 'heavy_ballot64')
               for draw in ('direct', 'count') for steps in (0, 16, 64, 256, 1024) for draws in (1, 16)]


def indexed(case, mode):
    fallback = case.get('existing_fallback', case['name'] in ('ballot_vertex64', 'shared_vertex64'))
    return mode == 'stored' or (mode.startswith('existing') and not fallback)


def route(case, mode):
    return 'stored IDX' if mode == 'stored' else ('existing IDX' if indexed(case, mode) else 'Mesh')


def compile_audit(case, mode, log):
    assert not re.search(BAD, log), 'compilation error'
    candidate = (case.get('auto_candidate') or case.get('auto_replay', False)) if case.get('automatic') else indexed(case, mode)
    assert ('MESH IDXPASS: applied' in log) == candidate, 'unexpected compiled route'
    assert ('stored outputs bytes/group=' in log) == (bool(case.get('auto_candidate')) if case.get('automatic') else mode == 'stored'), 'unexpected storage mode'
    if case.get('automatic') and case.get('auto_candidate'):
        direction = 'direct' if case['draw'] == 'direct' else 'indirect'
        chosen = len(re.findall('BC250 MESH AUTO: ' + direction + ' stored', log))
        expected = case.get('expected_idx_calls', case['draws']) if indexed(case, mode) else 0
        assert chosen == expected, 'wrong automatic route decision'
    if 'corner_alias' in case:
        assert ('MESH CORNER ALIAS: applied' in log) == bool(case['corner_alias'] and
               case.get('expect_corner_alias')), 'unexpected corner alias admission'
    if candidate and 'geometry' in case:
        v, p, threads = case['geometry']
        assert f'MESH IDXPASS: applied V={v} P={p} threads={threads}' in log, 'unexpected geometry'


def compare_image(pixels, reference, allow_rounding=False):
    """Bound UNORM8 differences only when the caller permits rounding."""
    assert len(pixels) == len(reference) == 256 * 256 * 4, 'wrong readback size'
    different = sum(pixels[i:i + 4] != reference[i:i + 4] for i in range(0, len(pixels), 4))
    maximum = max(abs(a - b) for a, b in zip(pixels, reference))
    alpha_exact = pixels[3::4] == reference[3::4]
    background_exact = all(pixels[i:i + 4] == reference[i:i + 4]
                           for i in range(0, len(pixels), 4) if reference[i + 3] == 0)
    exact = different == 0
    return dict(accepted=exact or (allow_rounding and alpha_exact and background_exact and maximum <= 1),
                exact=exact, different_pixels=different, max_channel_error=maximum,
                alpha_exact=alpha_exact, background_exact=background_exact)


def image_acceptance(case, mode, pixels, reference):
    # Flat corner probes must always be exact. Scored interpolated images may
    # use the bounded check only after all three exact corner probes pass.
    permitted = not case.get('flat_oracle') and case.get('verified_flat_oracle', False)
    return compare_image(pixels, reference, allow_rounding=permitted)


def flat_cases(case):
    for corner in range(3):
        fragment = 'heavy_flat_pp' if case['fragment'] == 'pp' else 'heavy_flat_plain'
        tag = case['tag'] + '-flat-corner' + str(corner)
        yield dict(case, tag=tag, reference_key=tag, flat_oracle=True, oracle_corner=corner,
                   shader_name=case['name'] + '_flat', fragment=fragment,
                   reference_fragment=fragment + ('_ref' if case['fragment'] == 'pp' else ''))


def sample_count(case, offline=False, preflight=False):
    requested = case.get('benchmark_samples', 9)
    assert type(requested) is int and 1 <= requested <= 31, 'invalid timestamp sample count'
    return 1 if offline or preflight or case.get('flat_oracle') else requested


def warmup_duration(case, preflight=False):
    requested = case.get('warmup_ms', 0)
    assert type(requested) is int and 0 <= requested <= 2000, 'invalid warmup duration'
    return 0 if preflight or case.get('flat_oracle') else requested


def audit_usage(log, case, mode):
    """Usage counters are cumulative accepted submissions of the same recording."""
    values = [tuple(map(int, match)) for match in re.findall(
        r'BC250_USAGE[^\n]*submits=(\d+)[^\n]*idx_direct=(\d+) idx_indirect_calls=(\d+)', log)]
    direct = indirect = 0
    if indexed(case, mode):
        calls = case.get('expected_idx_calls', case['draws'])
        direct = calls if case['draw'] == 'direct' else (calls + 1 if case.get('direct_around') else 0)
        indirect = 0 if case['draw'] == 'direct' else calls
    assert values and values[0][0] == 1, 'initial accepted usage missing'
    assert all(n > 0 and (d, i) == (n * direct, n * indirect) for n, d, i in values), 'unexpected cumulative IDX usage'
    return values


def audit_warmup(log, requested_ms, telemetry_path):
    from telemetry import summarize
    started = re.search(r'INDEX_WARMUP_BEGIN requested_ms=(?P<ms>\d+) monotonic_ns=(?P<time>\d+)', log)
    complete = re.search(r'INDEX_WARMUP_COMPLETE submissions=(?P<count>\d+) wall_ms=(?P<ms>[\d.]+) '
                         r'monotonic_ns=(?P<time>\d+)', log)
    begin = re.search(r'INDEX_MEASURE_BEGIN monotonic_ns=(?P<time>\d+)', log)
    end = re.search(r'INDEX_MEASURE_COMPLETE monotonic_ns=(?P<time>\d+)', log)
    assert started and complete and begin and end, 'missing sustained warmup evidence'
    assert int(started['ms']) == requested_ms and int(complete['count']) > 0
    assert float(complete['ms']) >= requested_ms, 'warmup shorter than requested'
    warm_start, warm_end, measure_start, measure_end = (int(match['time']) for match in (started, complete, begin, end))
    assert warm_start <= warm_end <= measure_start <= measure_end, 'invalid warmup/measurement ordering'
    return dict(warmup_requested_ms=requested_ms, warmup_extra_submissions=int(complete['count']),
                warmup_wall_ms=float(complete['ms']),
                warmup_telemetry=summarize(telemetry_path, warm_start, warm_end),
                measured_telemetry=summarize(telemetry_path, measure_start, measure_end),
                telemetry_sha256=hashlib.sha256(telemetry_path.read_bytes()).hexdigest())


def run(case, mode, out, icd, shim=None, preflight=False, references=None):
    offline = shim is not None
    samples = sample_count(case, offline, preflight)
    warmup_ms = warmup_duration(case, preflight)
    tag = case['tag'] + '-' + mode + ('-preflight' if preflight else '')
    log_path, image_path = out / (tag + '.log'), out / (tag + '.rgba')
    env = dict(PATH='/usr/bin', HOME='/tmp' if offline else os.environ['HOME'],
               VK_DRIVER_FILES=str(icd), VK_ICD_FILENAMES=str(icd), MESA_SHADER_CACHE_DISABLE='1',
               RADV_DIRECTMESH='1', RADV_BC250_MESH_SAFE_FAST='1',
               RADV_BC250_MESH_IDXPASS='0' if mode.startswith('mesh') else '1',
               RADV_BC250_MESH_IDXPASS_STORE='1' if mode == 'stored' else '0',
               BC250_TRACE_COMPILE='1', BC250_TRACE_USAGE='1',
               INDEX_LAST_VERTEX=str(case.get('last', 0)),
               INDEX_PRIMITIVES=str(case['primitives']), INDEX_BENCH_DRAWS=str(case['draws']),
               INDEX_BENCH_SAMPLES=str(samples))
    if case.get('automatic'):
        env.update(RADV_BC250_MESH_IDXPASS='1', RADV_BC250_MESH_IDXPASS_STORE='auto', BC250_IDX_TRACE='1')
    if case.get('alternate_steps'):
        env['INDEX_ALTERNATE_STEPS'] = '1'
    if warmup_ms:
        env['INDEX_WARMUP_MS'] = str(warmup_ms)
    if 'steps' in case:
        env['INDEX_VERTEX_STEPS'] = str(case['steps'])
    if 'corner_alias' in case:
        env['RADV_BC250_MESH_CORNER_ALIAS'] = str(case['corner_alias'])
    if 'oracle_corner' in case:
        env['INDEX_ORACLE_CORNER'] = str(case['oracle_corner'])
    if 'prune' in case:
        env['RADV_BC250_MESH_IDXPASS_PRUNE'] = str(case['prune'])
    expected_trim = case.get('batch_trim', case.get('expected_batch_trim'))
    if 'batch_trim' in case:
        env['RADV_BC250_MESH_IDXPASS_BATCH_TRIM'] = str(case['batch_trim'])
    if expected_trim is not None:
        env['BC250_IDX_TRACE'] = '1'
    if 'setup_batch' in case:
        env['RADV_BC250_MESH_IDXPASS_SETUP_BATCH'] = str(case['setup_batch'])
        env['BC250_IDX_TRACE'] = '1'
    if 'pool_bytes' in case:
        env['RADV_BC250_IDX_POOL_BYTES'] = str(case['pool_bytes'])
    if 'groups' in case:
        env['INDEX_GROUPS'] = str(case['groups'])
    if case.get('direct_around'):
        env['INDEX_DIRECT_AROUND'] = '1'
    if case.get('empty'):
        assert mode.startswith('mesh'), 'empty baseline must use Mesh mode'
        env['INDEX_BENCH_EMPTY'] = '1'
    if case['draw'] != 'direct':
        env['INDEX_INDIRECT'] = case['draw']
    if preflight:
        env['RADV_BC250_IDX_DEBUG'] = '1'
    command = [str(HERE / 'index_gate'), '--offline' if offline else '--authorized-gpu-INDEX',
               str(HERE / (case.get('shader_name', case['name']) + '.mesh.spv')),
               str(HERE / (case.get('fragment', 'pp' if case['name'].startswith('pp') else 'plain') + '.frag.spv')),
               str(image_path)]
    if offline:
        env.update(LD_PRELOAD=str(shim), AMDGPU_GPU_ID='gfx1013',
                   NIR_DEBUG='validate', ACO_DEBUG='validateir,validatera')
        command = ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc',
                   '--tmpfs', '/tmp', '--bind', str(out), str(out), '--unshare-pid',
                   '--die-with-parent', '--', 'env', '-i',
                   *(f'{k}={v}' for k, v in env.items()), *command]
    else:
        env['BC250_INDEX_OWNER_AUTHORIZED'] = 'INDEX-once'
    with log_path.open('xb') as log:
        process = subprocess.Popen(command, cwd=HERE, env=env, stdin=subprocess.PIPE,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            ready = f'INDEX_READY mode={"noop" if offline else "hardware"} mesh_draws={0 if case.get("empty") else case["draws"]}'
            while ready not in (text := log_path.read_text(errors='replace')):
                if process.poll() is not None or time.monotonic() >= deadline or re.search(BAD, text):
                    raise RuntimeError('did not reach audited readiness: ' + tag)
                time.sleep(0.05)
            compile_audit(case, mode, text)
            if expected_trim is not None:
                # Validate the reserved IB2 and its synchronization before releasing any GPU submission.
                assert bool(audit_batches(text, expected_trim)) == indexed(case, mode), 'unexpected IDX batch: ' + tag
            if 'setup_batch' in case:
                audit_setups(text, case, mode)
            if 'groups' in case:
                records = 1 if case['draw'] in ('direct', 'single') else 2
                assert f'INDEX_WORKLOAD groups={case["groups"]} live_records={records} empty={int(bool(case.get("empty")))}' in text, tag
            output_bytes = int(re.search(r'stored outputs bytes/group=(\d+)', text).group(1)) if mode == 'stored' else 0
            if 'expected_output_bytes' in case:
                assert output_bytes == case['expected_output_bytes'], 'wrong stored layout: ' + tag
            monitor = nullcontext()
            if not offline and warmup_ms:
                from telemetry import Recorder
                monitor = Recorder(out / (tag + '-telemetry.jsonl'))
            with monitor:
                if not offline:
                    process.stdin.write(b'S')
                    process.stdin.flush()
                process.stdin.close()
                code = process.wait(timeout=60)
            text = log_path.read_text(errors='replace')
            assert code == 0 and not re.search(BAD, text) and 'VALIDATION_ERRORS=0' in text, tag
            usage = audit_usage(text, case, mode)
            assert len(re.findall(r'^INDEX_SUBMIT sample=', text, re.M)) == samples + 2, tag
            provoking = 'last' if case.get('last', 0) else 'first'
            assert f'samples={samples} warmup=2 provoking={provoking} timestamp_bits=' in text, tag
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
    row = dict(case=case['tag'], mode=mode, actual_route=route(case, mode), draws=case['draws'],
               output_bytes=output_bytes, output_pruning=case.get('prune'),
               preflight=preflight, flat_oracle=case.get('flat_oracle', False),
               verified_flat_oracle=case.get('verified_flat_oracle', False))
    if expected_trim is not None:
        batches = audit_batches(text, expected_trim)
        assert bool(batches) == indexed(case, mode), 'unexpected IDX batch: ' + tag
        row.update(batch_trim=expected_trim if batches else None,
                   batch_trim_explicit='batch_trim' in case, batches=batches)
    if 'setup_batch' in case:
        row.update(setup_batch=case['setup_batch'], setups=audit_setups(text, case, mode))
    if 'groups' in case:
        row['groups_per_record'] = case['groups']
        row['empty'] = bool(case.get('empty'))
        row['record_cpu_us'] = float(re.search(r'INDEX_RECORD_CPU_US recording=0 us=([\d.]+)', text).group(1))
    if offline:
        assert 'NOOP_COMPLETE pixels_not_rendered' in text, tag
        if warmup_ms:
            assert f'INDEX_WARMUP_SKIPPED mode=noop requested_ms={warmup_ms}' in text, tag
            assert 'INDEX_WARMUP_BEGIN' not in text and 'INDEX_MEASURE_BEGIN' not in text, tag
            row['warmup_requested_ms'] = warmup_ms
            row['warmup_skipped_noop'] = True
        return row
    if warmup_ms:
        telemetry_path = out / (tag + '-telemetry.jsonl')
        row.update(audit_warmup(text, warmup_ms, telemetry_path), cumulative_usage_samples=usage)
    row['timestamp_config'] = re.search(r'^INDEX_BENCH_CONFIG .+$', text, re.M).group(0)
    assert 'INDEX_FENCE_COMPLETE readback_saved' in text, tag
    found = re.findall(r'INDEX_BENCH_SAMPLE sample=(\d+) warmup=(\d+) gpu_us=([\d.]+) draws=(\d+)', text)
    assert len(found) == samples + 2, tag
    times = []
    for n, (sample, warm, us, draws) in enumerate(found):
        assert int(sample) == n and int(warm) == int(n < 2) and int(draws) == case['draws'], tag
        us = float(us)
        assert us > 0 and math.isfinite(us), tag
        if n >= 2:
            times.append(us)
    pixels = image_path.read_bytes()
    assert len(pixels) == 256 * 256 * 4 and (case.get('empty') or any(pixels[3::4])), 'empty image: ' + tag
    if case.get('empty'):
        assert not any(pixels), 'empty baseline must retain the zero clear: ' + tag
        row.update(samples_us=times, median_us=statistics.median(times), min_us=min(times),
                   max_us=max(times), image_sha256=hashlib.sha256(pixels).hexdigest(), empty_clear_exact=True)
        return row
    references = references if references is not None else json.loads((HERE / 'reference-hashes.json').read_text())
    expected = references[case.get('reference_key', case['name'] + '-' + case['draw'])]
    digest = hashlib.sha256(pixels).hexdigest()
    # Mesh primitive order may vary. It remains diagnostic, with its own saved
    # image hash; both IDX routes must match the independent VS image exactly.
    exact = digest == expected['reference' if indexed(case, mode) else 'mesh']
    if case.get('require_mesh_reference'):
        reference = (out / (case['reference_key'] + '-reference.rgba')).read_bytes()
        assert hashlib.sha256(reference).hexdigest() == expected['reference'], 'reference changed: ' + tag
        check = image_acceptance(case, mode, pixels, reference)
        row['reference_comparison'] = check
        assert check['accepted'], 'route image differs from independent VS: ' + tag + ' ' + str(check)
    elif indexed(case, mode):
        assert exact, 'route image differs from independent VS: ' + tag
    if preflight:
        tables = re.findall(r'BC250 IDX DEBUG: call=(\d+) n=(\d+) groups=(\d+).*fallback_count=(\d+) '
                            r'drawn=(\d+) indices=(\d+) fallback_records=(\d+).* pool=(\d+)', text)
        output_bytes = int(re.search(r'stored outputs bytes/group=(\d+)', text).group(1)) if mode == 'stored' else 0
        expected_tables, pool_words = expected_pool_tables(case, output_bytes)
        assert len(tables) == min(case['draws'], 8), 'debug tables absent: ' + tag
        for n, table in enumerate(tables):
            assert tuple(map(int, table)) == expected_tables[n], \
                'GPU pool overflow or resubmission reset failed: ' + tag
        row['verified_pool_words_after_resubmission'] = pool_words
        row['verified_pool_tables'] = expected_tables[:8]
    row.update(samples_us=times, median_us=statistics.median(times),
               min_us=min(times), max_us=max(times), image_sha256=digest,
               image_matches_saved_reference=exact)
    return row


def expected_pool_tables(case, output_bytes):
    capacity = (case.get('pool_bytes', 16 << 20) - 256) // 4
    words = case['primitives'] * 3 + output_bytes // 4
    used, tables = 0, []
    for call in range(case.get('expected_idx_calls', case['draws'])):
        fitting_groups = drawn = fallback_records = 0
        suffix = False
        for groups in (case.get('groups', 64), 0, case.get('groups', 64)):
            if not suffix and used + groups * words <= capacity:
                used += groups * words
                fitting_groups += groups
                drawn += groups != 0
            else:
                suffix = True
                fallback_records += groups != 0
        if suffix:
            used = capacity
        tables.append((call, 3, fitting_groups, 3 if suffix else 0, drawn,
                       fitting_groups * case['primitives'] * 3, fallback_records))
    return [values + (used,) for values in tables], used


def audit_setups(log, case, mode):
    calls = [dict(shared=bool(int(shared)), dispatch=bool(int(dispatch)),
                  wait=bool(int(wait)), reset=bool(int(reset)), synchronization_dwords=int(sync),
                  pfp_sync_header=int(tail, 16))
             for shared, dispatch, wait, reset, sync, tail in re.findall(
                 r'BC250 IDX SETUP: call shared=(\d+) dispatch=(\d+) wait=(\d+) reset=(\d+) '
                 r'sync_dw=(\d+) tail=0x([0-9a-f]+)', log)]
    spans = [dict(calls=int(n), dispatches=int(dispatches), waits=int(waits), bytes=int(size), shared=bool(int(shared)))
             for n, dispatches, waits, size, shared in re.findall(
                 r'BC250 IDX SETUP: close calls=(\d+) dispatches=(\d+) waits=(\d+) bytes=(\d+) shared=(\d+)', log)]
    promotions = [dict(begin=int(begin), end=int(end), replacement_end=int(replacement_end),
                       preserved_dwords=int(preserved), synchronization_dwords=int(sync), pfp_sync_header=int(tail, 16))
                  for begin, end, replacement_end, preserved, sync, tail in re.findall(
                      r'BC250 IDX SETUP: promote begin=(\d+) end=(\d+) replacement_end=(\d+) preserved_dw=(\d+) '
                      r'sync_dw=(\d+) tail=0x([0-9a-f]+)', log)]
    indirect = indexed(case, mode) and case['draw'] in ('multi', 'count')
    expected = case.get('expected_idx_calls', case['draws']) if indirect else 0
    assert len(calls) == expected, 'wrong recorded setup count'
    assert all(c['dispatch'] == c['wait'] for c in calls), 'setup wait missing'
    assert all((c['synchronization_dwords'] > 2 and c['pfp_sync_header'] == 0xc0004200) if c['wait']
               else (c['synchronization_dwords'] == c['pfp_sync_header'] == 0) for c in calls), 'invalid setup synchronization'
    if calls:
        assert calls[0]['reset'] and not any(c['reset'] for c in calls[1:]), 'wrong pool reset lifetime'
    if case['setup_batch'] and indirect:
        assert all(c['shared'] for c in calls), 'setup batching unexpectedly fell back'
        assert sum(s['calls'] for s in spans) == expected, 'setup list omitted calls'
        assert all(0 <= s['calls'] <= 32 and s['bytes'] == 2832 for s in spans), 'setup list overflow'
        assert all(s['dispatches'] == s['waits'] == int(s['calls'] != 0) for s in spans)
        assert all(s['shared'] == (s['calls'] > 1) for s in spans), 'single-call span used shared scan'
        assert len(promotions) == sum(s['shared'] for s in spans), 'promotion missing or duplicated'
        assert all(0 <= p['begin'] < p['end'] and p['replacement_end'] > p['begin'] and
                   p['preserved_dwords'] > 0 and p['synchronization_dwords'] > 2 and
                   p['pfp_sync_header'] == 0xc0004200 for p in promotions), 'invalid replacement or setup wait'
        assert sum(c['dispatch'] for c in calls) == sum(s['dispatches'] for s in spans), 'wrong setup dispatch count'
        assert calls[0]['dispatch'], 'first setup dispatch missing'
    else:
        assert not any(c['shared'] for c in calls) and all(c['dispatch'] for c in calls)
        assert not any(s['calls'] for s in spans), 'unexpected shared setup entries'
        assert not promotions, 'unexpected setup replacement'
    return dict(calls=calls, spans=spans, dispatches=sum(c['dispatch'] for c in calls),
                waits=sum(c['wait'] for c in calls), promotions=promotions)


def audit_batches(log, trim):
    batches = []
    for values in re.findall(r'BC250 IDX BATCH: close dw=(\d+) of (\d+) unchained=(\d+) '
                             r'exec_dw=(\d+) flags=0x([0-9a-f]+) trim=(\d+) '
                             r'sync_dw=(\d+) tail=0x([0-9a-f]+) main_grew=(\d+) pad_dw=(\d+)', log):
        used, capacity, unchained, executed, flags, enabled, sync, tail, grew, pad = [
            int(v, 16 if n in (4, 7) else 10) for n, v in enumerate(values)]
        assert unchained == 1 and 0 < used <= capacity == 8192, 'invalid side batch'
        assert pad > 0 and not pad & (pad - 1), 'invalid GFX IB alignment'
        minimum = (used + pad - 1) & ~(pad - 1)
        assert enabled == trim and executed == (minimum if trim else capacity), 'wrong IB2 execution length'
        assert flags == 0, 'unexpected IB2 control flags'
        assert sync > 2 and sync <= used and tail == 0xc0004200, 'wait/PFP sync outside executed body'
        batches.append(dict(used_dwords=used, capacity_dwords=capacity, executed_dwords=executed,
                            control_flags=flags, synchronization_dwords=sync,
                            pfp_sync_header=tail, main_stream_grew=bool(grew), alignment_dwords=pad))
    return batches


def run_reference(case, out, icd, shim=None):
    offline = shim is not None
    tag = case['reference_key'] + '-reference'
    log_path, image_path = out / (tag + '.log'), out / (tag + '.rgba')
    env = dict(PATH='/usr/bin', HOME='/tmp' if offline else os.environ['HOME'],
               VK_DRIVER_FILES=str(icd), VK_ICD_FILENAMES=str(icd), MESA_SHADER_CACHE_DISABLE='1',
               RADV_DIRECTMESH='1', RADV_BC250_MESH_SAFE_FAST='1', RADV_BC250_MESH_IDXPASS='0',
               RADV_BC250_MESH_IDXPASS_STORE='0', BC250_TRACE_COMPILE='1', BC250_TRACE_USAGE='1',
               INDEX_REFERENCE='1', INDEX_LAST_VERTEX=str(case.get('last', 0)), INDEX_VERTEX_STEPS=str(case['steps']),
               INDEX_PRIMITIVES=str(case['primitives']))
    if 'oracle_corner' in case:
        env['INDEX_ORACLE_CORNER'] = str(case['oracle_corner'])
    if 'groups' in case:
        env['INDEX_GROUPS'] = str(case['groups'])
    if case.get('direct_around'):
        env['INDEX_DIRECT_AROUND'] = '1'
    if case['draw'] != 'direct':
        env['INDEX_INDIRECT'] = case['draw']
    command = [str(HERE / 'index_gate'), '--offline' if offline else '--authorized-gpu-INDEX',
               str(HERE / (case.get('shader_name', case['name']) + '.vert.spv')),
               str(HERE / (case.get('reference_fragment', 'pp_ref' if case['fragment'] == 'pp' else 'plain') + '.frag.spv')),
               str(image_path)]
    if offline:
        env.update(LD_PRELOAD=str(shim), AMDGPU_GPU_ID='gfx1013', NIR_DEBUG='validate', ACO_DEBUG='validateir,validatera')
        command = ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp',
                   '--bind', str(out), str(out), '--unshare-pid', '--die-with-parent', '--', 'env', '-i',
                   *(f'{k}={v}' for k, v in env.items()), *command]
    else:
        env['BC250_INDEX_OWNER_AUTHORIZED'] = 'INDEX-once'
    with log_path.open('xb') as log:
        process = subprocess.Popen(command, cwd=HERE, env=env, stdin=subprocess.PIPE,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            ready = f'INDEX_READY mode={"noop" if offline else "hardware"} mesh_draws=1'
            while ready not in (text := log_path.read_text(errors='replace')):
                if process.poll() is not None or time.monotonic() >= deadline or re.search(BAD, text):
                    raise RuntimeError('reference did not reach readiness: ' + tag)
                time.sleep(0.05)
            assert not re.search(BAD, text) and 'MESH IDXPASS: applied' not in text, tag
            if 'groups' in case:
                records = 1 if case['draw'] in ('direct', 'single') else 2
                assert f'INDEX_WORKLOAD groups={case["groups"]} live_records={records} empty=0' in text, tag
            if not offline:
                process.stdin.write(b'S')
                process.stdin.flush()
            process.stdin.close()
            code = process.wait(timeout=60)
            text = log_path.read_text(errors='replace')
            assert code == 0 and not re.search(BAD, text) and 'VALIDATION_ERRORS=0' in text, tag
            usage = re.findall(r'idx_direct=(\d+) idx_indirect_calls=(\d+)', text)
            assert usage and sum(map(int, usage[-1])) == 0, tag
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
    row = dict(reference_key=case['reference_key'], steps=case['steps'])
    if offline:
        assert 'NOOP_COMPLETE pixels_not_rendered' in text, tag
    else:
        assert 'INDEX_FENCE_COMPLETE readback_saved' in text, tag
        pixels = image_path.read_bytes()
        assert len(pixels) == 256 * 256 * 4 and any(pixels[3::4]), tag
        row['image_sha256'] = hashlib.sha256(pixels).hexdigest()
    return row


def idle_check():
    assert Path('/sys/module/amdgpu/parameters/gpu_recovery').read_text().strip() == '0', 'gpu_recovery must be 0'
    ancestors, pid = set(), os.getpid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        try:
            pid = int((Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError):
            break
    busy = []
    for p in Path('/proc').glob('[0-9]*'):
        if int(p.name) in ancestors:
            continue
        try:
            comm = (p / 'comm').read_text().strip()
            args = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        if re.search(r'^(meshbench|index_gate|deqp-vk|wine|wine64|wineserver|Control.*|Hellblade.*)$', comm, re.I) or '/gpuq/runner.sh' in args:
            busy.append((p.name, comm))
    for f in Path('/sys/class/drm').glob('card[0-9]*/device/gpu_busy_percent'):
        if int(f.read_text().strip()) > 5:
            busy.append(('GPU', f.read_text().strip() + '% busy'))
    assert not busy, 'close games and other GPU work: ' + str(busy)


def progress(out, text):
    with (out / 'hw.progress').open('a') as f:
        f.write(time.strftime('%H:%M:%S ') + text + '\n')
        f.flush()
        os.fsync(f.fileno())


def summarize_case(case, rows):
    data = {r['mode']: r for r in rows if r['case'] == case['tag']}
    stored = data['stored']['median_us']
    entry = dict(case=case['tag'], name=case['name'], draw=case['draw'], draws=case['draws'],
                 stored_us=stored, stored_us_per_draw=stored / case['draws'])
    if 'steps' in case:
        entry['steps'] = case['steps']
    for baseline in ('mesh', 'existing'):
        a, b = (data[baseline + '-' + phase]['median_us'] for phase in ('before', 'after'))
        drift = max(a, b) / min(a, b) - 1
        entry[baseline + '_baseline_us'] = (a + b) / 2
        entry['speedup_vs_' + baseline] = (a + b) / 2 / stored
        entry[baseline + '_drift_percent'] = drift * 100
    entry['existing_route'] = data['existing-before']['actual_route']
    if 'steps' in case:
        checks = [r['reference_comparison'] for r in data.values() if r['actual_route'] == 'Mesh']
        entry['mesh_reference_exact'] = all(check['exact'] for check in checks)
        entry['mesh_max_channel_error'] = max(check['max_channel_error'] for check in checks)
        idx_checks = [r['reference_comparison'] for r in data.values() if r['actual_route'] != 'Mesh']
        entry['idx_reference_exact'] = all(check['exact'] for check in idx_checks)
        entry['max_channel_error'] = max(r['reference_comparison']['max_channel_error'] for r in data.values())
        entry['flat_corner_oracles_exact'] = all(r['verified_flat_oracle'] for r in data.values())
    entry['stable'] = max(entry['mesh_drift_percent'], entry['existing_drift_percent']) <= 10
    return entry


def break_even(comparisons):
    groups = {}
    for row in comparisons:
        key = (row['name'], row['draw'], row['draws'])
        groups.setdefault(key, []).append(row)
    results = []
    for (name, draw, draws), rows in groups.items():
        rows.sort(key=lambda row: row['steps'])
        winners = [row['steps'] for row in rows if row['stable'] and
                   all(row['speedup_vs_' + baseline] > 1 + max(0.05, row[baseline + '_drift_percent'] / 100)
                       for baseline in ('mesh', 'existing'))]
        results.append(dict(name=name, draw=draw, draws=draws,
                            first_sampled_steps_beating_both=winners[0] if winners else None,
                            measured_steps=[row['steps'] for row in rows]))
    return dict(criterion='Stable timings; speedup over both baselines exceeds 5% and measured baseline drift',
                groups=results,limitation='First sampled point, not an interpolated threshold or game policy')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--heavy', action='store_true', help='sweep output-dependent vertex arithmetic')
    parser.add_argument('--start-at', help='start a new frozen kit at this workload tag')
    args = parser.parse_args()
    cases = HEAVY_CASES if args.heavy else CASES
    if args.start_at:
        tags = [case['tag'] for case in cases]
        assert args.start_at in tags, 'unknown starting workload'
        cases = cases[tags.index(args.start_at):]
    out = Path(os.environ['OUT']).resolve() if args.offline else HERE
    out.mkdir(parents=True, exist_ok=True)
    icd = Path(os.environ['ICD']).resolve() if args.offline else HERE / 'radeon_icd.json'
    shim = Path(os.environ['SHIM']).resolve() if args.offline else None
    if not args.offline:
        assert not (out / 'hw.started').exists(), 'already started; preserve results and do not retry'
        subprocess.run(['sha256sum', '--check', '--quiet', 'SHA256SUMS'], cwd=HERE, check=True)
        idle_check()
        with (out / 'hw.started').open('x') as f:
            f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + '\n')
            f.flush()
            os.fsync(f.fileno())
    rows = []
    preflights = []
    reference_rows = []
    oracle_rows = []
    oracle_references = []
    references = {} if args.heavy else None
    try:
        for case in cases:
            if args.heavy:
                for witness in flat_cases(case):
                    if not args.offline:
                        progress(out, witness['tag'] + ' exact corner oracle')
                    ref = run_reference(witness, out, icd, shim)
                    oracle_references.append(ref)
                    hashes = {witness['reference_key']: dict(reference=ref.get('image_sha256'), mesh=ref.get('image_sha256'))}
                    for mode in ('mesh-before', 'existing-before', 'stored'):
                        oracle_rows.append(run(witness, mode, out, icd, shim, references=hashes))
                        (out / 'oracle-results.partial.json').write_text(json.dumps(oracle_rows, indent=2) + '\n')
                case = dict(case, verified_flat_oracle=not args.offline)
                print('PASS ' + case['tag'] + ' ' + ('offline corner compilation' if args.offline else 'three exact corner oracles'), flush=True)
            if args.heavy and case['reference_key'] not in references:
                if not args.offline:
                    progress(out, case['reference_key'] + ' independent VS reference')
                ref = run_reference(case, out, icd, shim)
                if not args.offline:
                    prefix = case['name'] + '-' + case['draw'] + '-steps'
                    assert ref['image_sha256'] not in [value['reference'] for key, value in references.items()
                                                      if key.startswith(prefix)], \
                        'reference image did not change with vertex iteration count'
                reference_rows.append(ref)
                references[case['reference_key']] = dict(reference=ref.get('image_sha256'), mesh=ref.get('image_sha256'))
                (out / 'reference-hashes.generated.json').write_text(json.dumps(references, indent=2) + '\n')
            # Audit reused indirect reservations in a mapped debug pool first.
            # Those times are discarded; scored trials use the normal VRAM pool.
            if case['draw'] != 'direct':
                for mode in ('existing-before', 'stored'):
                    if indexed(case, mode):
                        if not args.offline:
                            progress(out, case['tag'] + ' ' + mode + ' pool preflight')
                        preflights.append(run(case, mode, out, icd, shim, preflight=True, references=references))
            for mode in (('mesh-before', 'existing-before', 'stored') if args.offline else MODES):
                if not args.offline:
                    progress(out, case['tag'] + ' ' + mode)
                row = run(case, mode, out, icd, shim, references=references)
                rows.append(row)
                (out / 'results.partial.json').write_text(json.dumps(rows, indent=2) + '\n')
                if args.offline:
                    print('PASS ' + case['tag'] + ' ' + mode + ' -> ' + row['actual_route'], flush=True)
            if not args.offline:
                entry = summarize_case(case, rows)
                print(f'{case["tag"]}: stored={entry["stored_us"]:.2f} us, '
                      f'vs Mesh={entry["speedup_vs_mesh"]:.2f}x, '
                      f'vs {entry["existing_route"]}={entry["speedup_vs_existing"]:.2f}x'
                      + (' [RGB within 1 UNORM8 step; flat corner oracles exact]'
                         if args.heavy and (not entry['mesh_reference_exact'] or not entry['idx_reference_exact']) else '')
                      + ('' if entry['stable'] else ' [clock/workload drift >10%]'), flush=True)
        result = dict(status='PASS', offline=args.offline, heavy=args.heavy, trials=rows,
                      pool_preflights=preflights, references=reference_rows,
                      corner_oracle_trials=oracle_rows, corner_oracle_references=oracle_references)
        if not args.offline:
            result['comparisons'] = [summarize_case(case, rows) for case in cases]
            with (out / 'results.csv').open('x', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(result['comparisons'][0]))
                writer.writeheader()
                writer.writerows(result['comparisons'])
            if args.heavy:
                result['break_even'] = break_even(result['comparisons'])
                (out / 'break-even.json').write_text(json.dumps(result['break_even'], indent=2) + '\n')
            progress(out, 'PASS timing and IDX image checks')
        (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        print('PASS benchmark ' + ('offline recording checks' if args.offline else 'complete; results.json saved'), flush=True)
    except BaseException:
        if not args.offline:
            progress(out, 'STOP: retain logs; do not retry this kit')
        raise


if __name__ == '__main__':
    main()
