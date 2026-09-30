#!/usr/bin/env python3
"""Create new pins; never modify an existing freeze or imported gate pins."""
import hashlib, json, shutil, subprocess, sys
from pathlib import Path

source = Path(__file__).resolve().parent
freeze, candidate, imports, policy = map(Path, sys.argv[1:])
freeze.mkdir()
def run(args):
    subprocess.run(list(map(str, args)), check=True)
shutil.copyfile(candidate, freeze / 'candidate.so')
(freeze / 'candidate.json').write_text(json.dumps({'file_format_version': '1.0.0', 'ICD': {
    'library_path': str((freeze / 'candidate.so').resolve()), 'api_version': '1.4.305'}}))
shutil.copyfile(policy, freeze / 'policy.json')
for name in ('run_once.py', 'gate.c', 'gate.mesh', 'gate.task', 'gate.frag'):
    shutil.copyfile(source / name, freeze / name)
run(['cc', '-O1', '-Wall', '-o', freeze / 'gate', source / 'gate.c', '-lvulkan'])
for name in ('D1', 'D2', 'descriptor-buffers', 'many-task'):
    out = freeze / name
    out.mkdir()
    definitions = ['-DTASK=' + str(int(name != 'D1')),
                   '-DDB=' + str(int(name == 'descriptor-buffers')),
                   '-DMANY=' + str(int(name == 'many-task'))]
    for stage in ('mesh', 'frag') + (() if name == 'D1' else ('task',)):
        run(['glslangValidator', '--target-env', 'vulkan1.3', '-S', stage, *definitions,
             '-o', out / (stage + '.spv'), source / ('gate.' + stage)])
for name in ('gpl-gate.c', 'objects-gate.c', 'gate-body.h', 'pipe.c', 'binary-pipe.c', 'icd-fixture.h'):
    shutil.copyfile(imports / name, freeze / name)
for name in ('gpl', 'objects'):
    run(['cc', '-O1', '-Wall', '-o', freeze / (name + '-gate'), freeze / (name + '-gate.c'), '-ldl'])
for name in ('mesh.spv', 'frag.spv'):
    shutil.copyfile(imports / name, freeze / name)
pins = {str(p.relative_to(freeze)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(freeze.rglob('*')) if p.is_file()}
(freeze / 'gate-manifest.json').write_text(json.dumps({'sha256': pins, 'hardware_runs': 0,
    'gates': ['D1', 'D2', 'gpl', 'objects', 'descriptor-buffers', 'many-task']}, indent=2) + '\n')
