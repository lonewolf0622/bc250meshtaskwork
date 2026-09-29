#!/usr/bin/env python3
"""check.py <ib-stderr> <N|off> [<off-ib-stderr>]: checks one RADV_BC250_MESH_DEALLOC_DIST dump.

Events in IB order (nested IB2s, i.e. executed secondaries, inline): VGT_OUT_DEALLOC_CNTL writes,
Mesh draws (VGT_SHADER_STAGES_EN with GS_EN and no ES_EN, or DISPATCH_MESH_INDIRECT_MULTI) and
other draws. The register holds 32 (CLEAR_STATE) when an IB starts.
  N:   every Mesh draw runs with DEALLOC_DIST = N, every other draw with 32, every IB (main and
       nested) ends with 32, and no write repeats the value the same IB already set.
  off: no VGT_OUT_DEALLOC_CNTL write at all.
With <off-ib-stderr> (same case, switch unset): after removing the VGT_OUT_DEALLOC_CNTL packets and
the NOP packets (IB padding, stale payload) and masking INDIRECT_BUFFER sizes, both IBs must be identical.
Prints one key=value line; exits 1 on any mismatch."""
import re, sys

ansi = re.compile(r'\x1b\[[0-9;]*m')
CLEAR = 32


def ib_lines(path):
    out, on = [], False
    for line in open(path, errors='replace'):
        line = ansi.sub('', line.rstrip('\n'))
        if 'IB begin' in line and 'nested' not in line:
            on = True
        if on:
            out.append(line)
        if 'IB end' in line and 'nested' not in line:
            on = False
    return out


PKT = re.compile(r'^[0-9a-f]{8} [A-Z0-9_]+(\(.*\))?:$')


def strip(lines):
    """Removes the VGT_OUT_DEALLOC_CNTL SET_CONTEXT_REG packets (4 lines) and the NOP packets (IB padding:
    their payload is stale memory and their size follows the IB size), and masks the size of
    INDIRECT_BUFFER packets (an executed secondary grows by its end-of-IB restore)."""
    out, i, in_nop = [], 0, False
    while i < len(lines):
        s = lines[i].strip()
        if s.endswith('SET_CONTEXT_REG:') and i + 2 < len(lines) and 'VGT_OUT_DEALLOC_CNTL <-' in lines[i + 2]:
            i += 4
            continue
        if PKT.match(s) or 'IB begin' in s or 'IB end' in s or 'nested' in s:
            in_nop = s.endswith(' NOP:')
        if not in_nop:
            if re.search(r'IB_CONTROL <-|IB_SIZE =', s):
                s = re.sub(r'(IB_CONTROL <-|IB_SIZE =).*', r'\1 <masked>', re.sub(r'^[0-9a-f]{8}', 'xxxxxxxx', s))
            out.append(s)
        i += 1
    return out


def events(lines):
    ev, stages = [], None
    for line in lines:
        s = line.strip()
        if 'Main IB begin' in s:
            ev.append(('begin', 'main'))
        elif 'nested begin' in s:
            ev.append(('begin', 'nested'))
        elif 'nested end' in s:
            ev.append(('end', 'nested'))
        elif 'Main IB end' in s:
            ev.append(('end', 'main'))
        m = re.match(r'^([0-9a-f]{8})\s+VGT_OUT_DEALLOC_CNTL <- ', s)
        if m:
            ev.append(('write', int(m.group(1), 16)))
        m = re.match(r'^([0-9a-f]{8})\s+VGT_SHADER_STAGES_EN <- ', s)
        if m:
            stages = int(m.group(1), 16)
        m = re.match(r'^[0-9a-f]{8} (DRAW_INDEX_AUTO|DRAW_INDEX_2|DRAW_INDEX_OFFSET_2|DRAW_INDIRECT\w*|DRAW_INDEX_INDIRECT\w*|DISPATCH_MESH_INDIRECT_MULTI)(\(.*\))?:', s)
        if m:
            mesh = m.group(1) == 'DISPATCH_MESH_INDIRECT_MULTI' or (stages is not None and (stages & 0x38) == 0x20)
            ev.append(('draw', 'M' if mesh else 'V'))
    return ev


path, want = sys.argv[1], sys.argv[2]
lines = ib_lines(path)
ev = events(lines)
errs = []
seq = []
stack = []           # (value at IB start, tracked value inside this IB)
cur = CLEAR
tracked = None
writes = mesh = other = 0
for kind, v in ev:
    if kind == 'begin':
        stack.append(tracked)
        tracked = None
        if v == 'main':
            cur = CLEAR
        seq.append('[' if v == 'nested' else '|')
    elif kind == 'end':
        if want != 'off' and cur != CLEAR:
            errs.append('%s_ib_ends_with_%d' % (v, cur))
        tracked = stack.pop() if stack else None
        if v == 'nested':
            seq.append(']')
    elif kind == 'write':
        writes += 1
        if tracked == v:
            errs.append('redundant_write_%d' % v)
        cur = tracked = v
        seq.append('D%d' % v)
    elif kind == 'draw':
        seq.append(v)
        if v == 'M':
            mesh += 1
            if want != 'off' and cur != int(want):
                errs.append('mesh_draw_with_%d' % cur)
        else:
            other += 1
            if want != 'off' and cur != CLEAR:
                errs.append('other_draw_with_%d' % cur)
if want == 'off' and writes:
    errs.append('writes_with_switch_unset=%d' % writes)
if not mesh and not other:
    errs.append('no_draws')
rest = '-'
if len(sys.argv) > 3:
    rest = 'identical' if strip(ib_lines(sys.argv[3])) == strip(lines) else 'DIFFERENT'
    if rest != 'identical':
        errs.append('rest_differs_from_off')
print('seq=%s writes=%d mesh_draws=%d other_draws=%d rest_vs_off=%s %s' % (
    ' '.join(seq).replace('[ ', '[').replace(' ]', ']').replace('| ', '|'), writes, mesh, other, rest,
    'OK' if not errs else 'FAIL ' + ' '.join(sorted(set(errs)))))
sys.exit(1 if errs else 0)
