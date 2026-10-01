"""Build the offline CPU oracle using this Mesa build's NIR ABI."""
import json, os, shlex, subprocess, sys
from pathlib import Path
src=Path(__file__).resolve().parents[3];build=Path(os.environ['BUILD'])
entry=next(e for e in json.loads((build/'compile_commands.json').read_text()) if e['file'].endswith('nir_clone.c'))
defs=[x for x in shlex.split(entry['command']) if x.startswith('-D')]
includes=[src/'include',src/'src',src/'src/compiler',src/'src/compiler/nir',src/'src/util',src/'src/amd/common',build/'src',build/'src/compiler',build/'src/compiler/nir',build/'src/util',build/'src/amd/common']
libs=['src/compiler/nir/libnir.a','src/compiler/libcompiler.a','src/util/libmesa_util.a','src/util/libmesa_util_simd.a','src/util/blake3/libblake3.a']
subprocess.run(['c++','-O2','-g','-std=c++17','-Wall',*defs,*['-I'+str(p) for p in includes],'-o',sys.argv[1],str(src/'tests/bc250-mesh/dgc/oracle.cpp'),*[str(build/p) for p in libs],'-lm','-lpthread','-lzstd','-lz','-ldl'],check=True)
