#!/bin/bash
# Builds mesh_oracle against a configured Mesa build tree (its static NIR/util libraries, with the
# same preprocessor definitions as the NIR sources, taken from compile_commands.json: the NIR
# structure layout depends on them).
# usage: [SRC=<mesa source>] BUILD=<mesa build dir> ./build_oracle.sh [output binary] [extra cc flags]
set -eu
here=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-$(cd "$here/../../.." && pwd)}
BUILD=${BUILD:?BUILD=<mesa build dir>}
out=${1:-$here/mesh_oracle}
shift || true
defs=$(python3 -c "
import json, shlex, sys
for e in json.load(open(sys.argv[1] + '/compile_commands.json')):
    if e['file'].endswith('nir_clone.c'):
        print(' '.join(a for a in shlex.split(e['command']) if a.startswith('-D')))
        break
" "$BUILD")
eval "cc -O2 -g -std=gnu11 -Wall -Wno-unused-function -Wno-format-truncation $defs \
  -I'$SRC/include' -I'$SRC/src' -I'$SRC/src/compiler' -I'$SRC/src/compiler/nir' -I'$SRC/src/util' \
  -I'$BUILD/src' -I'$BUILD/src/compiler' -I'$BUILD/src/compiler/nir' -I'$BUILD/src/util' $* \
  -o '$out' '$here/mesh_oracle.c' \
  '$BUILD/src/compiler/nir/libnir.a' '$BUILD/src/compiler/libcompiler.a' '$BUILD/src/util/libmesa_util.a' \
  '$BUILD/src/util/libmesa_util_simd.a' '$BUILD/src/util/blake3/libblake3.a' \
  -lm -lpthread -lzstd -lz -ldl"
