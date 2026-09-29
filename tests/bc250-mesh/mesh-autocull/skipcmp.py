#!/usr/bin/env python3
"""skipcmp.py <on-stderr> <off-stderr>: RADV_BC250_MESH_AUTOCULL runtime-skip check.

Works on the final ACO program of the Mesh shader (RADV_DEBUG=shaders, "After lowering to hw
instructions"; this build has no disassembler). The switch-on shader branches on the NGG culling
settings SGPR (`s_and_b32 <settings>, 3` + `s_cbranch_scc0`, 3 = cull front | cull back): the
branch target is the side that does not cull. Checks:
  - skip side == the switch-off program's epilogue (the same number of final instructions of the
    switch-off program), instruction for instruction: opcodes, registers, immediates, LDS offsets
    (temporary ids and block numbers normalized);
  - everything before the branch == the switch-off program before its epilogue, apart from inserted
    exec restores (`exec = s_mov -1`), the settings test and the branch: the whole cost of the skip
    path.
With --sgpr-modulo (after the two paths), the skip side is also compared modulo SGPR numbers
(a scalar value the settings SGPR displaced into another register; reported as skip_equals_off_modulo_sgpr),
the prefixes are compared as multisets (scalar instructions moved around the settings test), an exec save
counts like an exec restore, and switch-off SGPR-to-SGPR copies missing from the switch-on prefix are not
counted as differences.
Prints one key=value line; exit 1 on a mismatch."""
import difflib, re, sys

ansi = re.compile(r'\x1b\[[0-9;]*m')
TEST = re.compile(r's1: %:s\[\d*\],\s+s1: %:scc = s_and_b32 %:s\[\d*\], 3$')
EXEC = re.compile(r's[12]: %:exec(_lo)? = s_mov_b(32|64) -1$')


def program(path):
    ls = [ansi.sub('', l.rstrip('\n')) for l in open(path, errors='replace')]
    out = []; on = False
    for l in ls:
        if 'Mesh Shader as NGG' in l:
            on = True
        elif on and l.startswith('shader: MESA_SHADER_FRAGMENT'):
            break
        if on:
            out.append(l)
    for i, l in enumerate(out):
        if l.startswith('After lowering to hw instructions'):
            out = out[i:]
            break
    blocks = []; cur = None
    for l in out:
        if re.fullmatch(r'BB\d+', l):
            cur = [l, []]; blocks.append(cur)
        elif cur is not None and l.strip() and not l.startswith('/*'):
            cur[1].append(re.sub(r'block:BB\d+', 'block:B', re.sub(r'%\d+:', '%:', l)).strip())
    return blocks


def flat(blocks):
    return [i for _, ins in blocks for i in ins]


sgpr_modulo = '--sgpr-modulo' in sys.argv[3:]
on_b, off_b = program(sys.argv[1]), program(sys.argv[2])
branch = skip = None
for bi, (name, ins) in enumerate(on_b):
    for i, l in enumerate(ins):
        if TEST.match(l) and i + 1 < len(ins) and ins[i + 1].startswith('s_cbranch_scc0'):
            branch = bi
if branch is None:
    print('skip_branch=missing FAIL'); sys.exit(1)
# the branch target: the block the raw dump names (block numbers were normalized above)
raw = [ansi.sub('', l) for l in open(sys.argv[1], errors='replace')]
names = [b[0] for b in on_b]
target = None
for i, l in enumerate(raw):
    if re.search(r's_and_b32 %\d+:s\[\d+\], 3\s*$', l) and 's_cbranch_scc0' in raw[i + 1]:
        target = re.search(r'block:(BB\d+)', raw[i + 1]).group(1)
skip = names.index(target)
end = next(i for i in range(skip, len(on_b)) if any(x == 's_endpgm' for x in on_b[i][1]))
on_skip = flat(on_b[skip:end]) + flat(on_b[end:])
off_all = flat(off_b)
off_epi = off_all[-len(on_skip):]
epi_ok = on_skip == off_epi
# before the branch the settings SGPR is live through the API body, so the scalar temporaries
# there may get other register numbers: compare that part modulo SGPR numbers
sren = lambda xs: [re.sub(r's\[\d+(-\d+)?\]', 's[]', x) for x in xs]
epi_modulo_ok = sren(on_skip) == sren(off_epi)
on_pre = sren(flat(on_b[:branch + 1]))
off_pre = sren(off_all[:-len(on_skip)])
extra = []; missing = []
for op, a1, a2, b1, b2 in difflib.SequenceMatcher(None, off_pre, on_pre, autojunk=False).get_opcodes():
    if op in ('replace', 'delete'):
        missing += off_pre[a1:a2]
    if op in ('replace', 'insert'):
        extra += on_pre[b1:b2]
bad = [l for l in extra if not (EXEC.match(l) or TEST.match(l) or l.startswith('s_cbranch_scc0'))]
# --sgpr-modulo: the prefixes are compared as multisets (the scheduler may move scalar instructions
# around the settings test), a switch-off SGPR-to-SGPR copy that the switch-on prefix does not need is
# not a cost, and an exec save before the branch counts like the exec restores.
if sgpr_modulo:
    import collections
    ce = collections.Counter(extra); cm = collections.Counter(missing)
    extra = list((ce - cm).elements()); missing = list((cm - ce).elements())
    missing = [l for l in missing if not re.fullmatch(r's[12]: %:s\[\] = s_mov_b(32|64) %:s\[\]', l)]
    bad = [l for l in extra if not (EXEC.match(l) or TEST.match(l) or l.startswith('s_cbranch_scc0') or
                                     re.fullmatch(r's2: %:s\[\] = s_mov_b64 %:exec', l))]
ok = (epi_ok or (sgpr_modulo and epi_modulo_ok)) and not bad and not missing
detail = ''
if not epi_ok:
    for i, (x, y) in enumerate(zip(on_skip, off_epi)):
        if x != y:
            detail = ' first_diff=%d on="%s" off="%s"' % (i, x, y); break
if bad or missing:
    detail += ' prefix_extra=%s prefix_missing=%s' % (bad[:3], missing[:3])
short = lambda l: re.sub(r'\s+', ' ', re.sub(r'%:|s[12]: ', '', l))
print('skip_overhead=%d(%s) pre_insns=%d/%d skip_insns=%d off_epilogue_insns=%d skip_equals_off=%s%s %s' % (
    len(extra), ';'.join(short(l) for l in extra), len(on_pre), len(off_pre), len(on_skip), len(off_all) - len(off_pre), epi_ok,
    detail + (' skip_equals_off_modulo_sgpr=%s' % epi_modulo_ok if sgpr_modulo else ''), 'OK' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
