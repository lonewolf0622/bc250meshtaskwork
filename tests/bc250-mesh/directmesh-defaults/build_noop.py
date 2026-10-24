#!/usr/bin/env python3
"""Build a private noop shim; extra targets are synthetic compiler fixtures only."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
build, out = a.build.resolve(), a.out.resolve()
out.mkdir(parents=True, exist_ok=True)
commands = json.loads((build / 'compile_commands.json').read_text())
entry = next(c for c in commands if c['file'].endswith('/common/amdgpu_devices.c') and 'drm-shim' in c['output'])
source = (Path(entry['directory']) / entry['file']).resolve()
text = source.read_text()
start = text.rfind('\n   {', 0, text.index('.name = "navi10"'))
end = text.index('\n   },', start) + len('\n   },')
baseline = text[start:end]
extra = ''
for name, chip, device, revision in [('gfx1013', 'GFX1013', '13fe', '82'),
                                      ('gfx1013_143f', 'GFX1013', '143f', '82'),
                                      ('navi12', 'NAVI12', '7360', '0a'),
                                      ('navi14', 'NAVI14', '7340', '14')]:
    item = baseline.replace('"navi10"', '"' + name + '"').replace('CHIP_NAVI10', 'CHIP_' + chip)
    item = re.sub(r'\.device_id = 0x[0-9a-f]+', '.device_id = 0x' + device, item)
    item = re.sub(r'\.external_rev = 0x[0-9a-f]+', '.external_rev = 0x' + revision, item)
    extra += '\n   /* SYNTHETIC: Navi10 topology, target ASIC identity; not a hardware capture. */' + item
text = text[:end] + extra + text[end:]
fixture = out / 'amdgpu_devices.c'
fixture.write_text(text)
cmd = shlex.split(entry['command'])
cmd[cmd.index('-o') + 1] = str(out / 'amdgpu_devices.o')
cmd[cmd.index('-c') + 1] = str(fixture)
cmd += ['-I' + str(source.parent)]
# Keep dependency files private as well.
for opt in ('-MF', '-MQ'):
    if opt in cmd:
        cmd[cmd.index(opt) + 1] = str(out / ('devices.d' if opt == '-MF' else 'amdgpu_devices.o'))
subprocess.run(cmd, cwd=build, check=True)
link = subprocess.check_output(['ninja', '-t', 'commands', 'src/amd/drm-shim/libamdgpu_noop_drm_shim.so'], cwd=build, text=True).splitlines()[-1]
cmd = shlex.split(link)
cmd[cmd.index('-o') + 1] = str(out / 'libamdgpu_noop_drm_shim.so')
cmd = [str(out / 'amdgpu_devices.o') if s.endswith('/.._common_amdgpu_devices.c.o') else s for s in cmd]
subprocess.run(cmd, cwd=build, check=True)
print('Built private shim: real Navi10 capture; synthetic GFX1013/Navi12/Navi14 identities')
