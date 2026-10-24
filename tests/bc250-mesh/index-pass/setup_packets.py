#!/usr/bin/env python3
"""Decode recorded IDX packets in a DRM-free sandbox; never submit GPU work."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import benchmark as bench

out = Path(os.environ['OUT']).resolve()
out.mkdir(parents=True, exist_ok=True)
icd, shim = (Path(os.environ[k]).resolve() for k in ('ICD', 'SHIM'))
rows = []
for stored in (0, 1):
    for draws, mixed in [(n, False) for n in (1, 2, 31, 32, 33, 34, 65, 512)] + [(n, True) for n in (1, 2, 33)]:
        snapshots = []
        for enabled in (0, 1):
            tag = f'stored{stored}-draws{draws}-mixed{int(mixed)}-setup{enabled}'
            env = dict(PATH='/usr/bin', HOME='/tmp', LD_PRELOAD=str(shim), AMDGPU_GPU_ID='gfx1013',
                       VK_DRIVER_FILES=str(icd), MESA_SHADER_CACHE_DISABLE='1', RADV_DIRECTMESH='1',
                       RADV_BC250_MESH_IDXPASS='1', RADV_BC250_MESH_IDXPASS_STORE=str(stored),
                       RADV_BC250_MESH_IDXPASS_SETUP_BATCH=str(enabled), BC250_IDX_TRACE='1',
                       BC250_IDX_CAPTURE_BATCH='1', BC250_CAPTURE_POLICY_SHADERS='1', BC250_TRACE_COMPILE='1',
                       NIR_DEBUG='validate', ACO_DEBUG='validateir,validatera', INDEX_INDIRECT='count',
                       INDEX_PRIMITIVES='64', INDEX_GROUPS='1', INDEX_BENCH_DRAWS=str(draws))
            if mixed:
                env['INDEX_DIRECT_AROUND'] = '1'
            command = ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp',
                       '--bind', str(out), str(out), '--unshare-pid', '--die-with-parent', '--',
                       'env', '-i', *(f'{k}={v}' for k, v in env.items()), str(bench.HERE / 'index_gate'), '--offline',
                       str(bench.HERE / 'profile64.mesh.spv'), str(bench.HERE / 'plain.frag.spv'), str(out / (tag + '.rgba'))]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            log = result.stdout.decode()
            (out / (tag + '.log')).write_text(log)
            assert result.returncode == 0 and 'NOOP_COMPLETE pixels_not_rendered' in log, tag
            assert 'VALIDATION_ERRORS=0' in log and not re.search(bench.BAD, log), tag
            headers = re.findall(r'^BC250POLICY (.*)$', log, re.M)
            code = re.findall(r'^BC250POLICYCODE ([0-9a-f]+)$', log, re.M)
            assert len(headers) == len(code)
            compute = [(dict(re.findall(r'(\w+)=(\w+)', h)), hashlib.sha256(bytes.fromhex(c)).hexdigest())
                       for h, c in zip(headers, code) if h.startswith('stage=5 ')]
            # Attachment/readback meta pipelines may upload additional shaders.
            assert len(compute) >= 3 + enabled, tag
            compute = compute[:3 + enabled]
            names = ['direct', 'indirect', 'ordinary'] + (['shared'] if enabled else [])
            shaders = {int(h['va'], 16) >> 8 & 0xffffffff: (name, h, digest)
                       for name, (h, digest) in zip(names, compute)}
            decoded, batch_sequences, layouts = [], [], []
            for block in log.split('BC250 IDX BATCH: close ')[1:]:
                chunks = re.findall(r'BC250 IDX WORDS: offset=(\d+) words=([0-9a-f,]+)', block)
                words = []
                for offset, data in chunks:
                    assert int(offset) == len(words), tag
                    words += [int(w, 16) for w in data.rstrip(',').split(',')]
                used = int(block.split()[0].split('=')[1])
                assert len(words) == used, tag
                registers, sequence, layout = {}, [], []
                setup_ready = False
                i = 0
                while i < len(words):
                    header = words[i]
                    if header >> 30 == 2:
                        i += 1
                        continue
                    assert header >> 30 == 3, (tag, i, hex(header))
                    length = ((header >> 16) & 0x3fff) + 2
                    assert i + length <= len(words), tag
                    op, payload = header >> 8 & 0xff, words[i + 1:i + length]
                    layout.append((op, length))
                    if op == 0x76:  # SET_SH_REG, unconditionally emitted on gfx1013.
                        first = payload[0] & 0xffff
                        registers.update((first + j, value) for j, value in enumerate(payload[1:]))
                    if op == 0x42:  # PFP_SYNC_ME after setup and at the end of the body.
                        setup_ready = True
                    if op in (0x15, 0x16):  # DISPATCH_DIRECT / DISPATCH_INDIRECT
                        name, h, digest = shaders[registers[0x20c]]  # COMPUTE_PGM_LO
                        assert registers[0x212] == int(h['rsrc1'], 16), tag
                        assert registers[0x213] == int(h['rsrc2'], 16), tag
                        assert (op == 0x16) == (name == 'indirect'), (tag, name, op)
                        if name in ('ordinary', 'shared'):
                            setup_ready = False
                            assert payload[:3] == [1, 1, 1], tag
                        elif name == 'indirect':
                            assert setup_ready, 'indirect producer before synchronized setup: ' + tag
                        sequence.append(name)
                        decoded.append((name, digest))
                    i += length
                batch_sequences.append(sequence)
                layouts.append(layout)
            case = dict(name='profile64', draw='count', draws=draws, setup_batch=enabled)
            setup = bench.audit_setups(log, case, 'stored' if stored else 'existing-before')
            assert len(batch_sequences) == len(bench.audit_batches(log, 1)), tag
            assert sum(n == 'indirect' for n, _ in decoded) == draws, tag
            assert sum(n == 'direct' for n, _ in decoded) == (draws + 1 if mixed else 0), tag
            if enabled:
                for sequence, span in zip(batch_sequences, setup['spans']):
                    scans = [n for n in sequence if n in ('ordinary', 'shared')]
                    assert scans == (['shared' if span['calls'] > 1 else 'ordinary'] if span['calls'] else []), tag
                    assert sequence.count('indirect') == span['calls'], tag
            else:
                assert sum(n == 'ordinary' for n, _ in decoded) == draws, tag
            snapshots.append(dict(decoded=decoded, layouts=layouts, setup=setup))
        # Promotion may replace scans but must preserve the producer order and actual code.
        producers = [[entry for entry in snap['decoded'] if entry[0] in ('direct', 'indirect')] for snap in snapshots]
        assert producers[0] == producers[1], tag
        if draws == 1:
            assert snapshots[0]['decoded'] == snapshots[1]['decoded'], tag
            assert snapshots[0]['layouts'] == snapshots[1]['layouts'], tag
        rows.append(dict(stored=stored, draws=draws, mixed_direct=mixed,
                         producer_order_and_code_identical=True, single_call_packet_layout_identical=draws == 1,
                         selected_spans=snapshots[1]['setup']['spans']))
        print('PASS', tag, 'decoded setup/producer order, shader configuration and synchronization', flush=True)
(out / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
print('PASS', len(rows), 'paired packet recordings', flush=True)
