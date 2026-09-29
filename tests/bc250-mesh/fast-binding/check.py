#!/usr/bin/env python3
"""check.py <script> <legacydb.out> <legacydb.err> <db.out> <db.err>

Checks one RADV_BC250_EXPOSE_FAST_BINDING case. Both runs record the same fb.c script with the switch on,
under RADV_DEBUG=dumpibs,shaders; "legacydb" binds descriptor sets (the path vkd3d-proton records while
VK_EXT_descriptor_buffer is hidden), "db" binds descriptor buffers (the path vkd3d-proton records when it
is exposed); both allocate the same memory, so every other address is equal.

1. The two dumps (IBs and shader NIR/ACO/disassembly) have the same lines except values written to
   SPI_SHADER_USER_DATA_* / COMPUTE_USER_DATA_* registers (descriptor set pointers).
2. Those differing values map one-to-one (legacy set address <-> descriptor buffer address, in both
   directions, everywhere in the dump), and every descriptor buffer value is the low 32 bits of one of the
   descriptor set addresses fb.c bound (FB_SETVA: graphics heap/sampler sets, compute heap/sampler sets).
3. db dump: at every application compute dispatch (DISPATCH_DIRECT 7,3,1) the last set pointer written to
   a COMPUTE_USER_DATA register is the compute heap set (cs0), never a graphics set: the Task producer's
   borrowed graphics bindings are restored.
   At every other dispatch (Task producer, setup helpers) the set pointers written since the previous
   dispatch are graphics ones (gfx0/gfx1) only.
4. db dump: every hybrid Task draw (t ti tc T) is fed by COMPUTE_USER_DATA writes of the graphics heap
   set (gfx0): the producer reads the application's descriptor buffer binding. (The internal mesh-only
   amplifier's producer reads no descriptors.)
Prints one key=value line and exits 1 on any mismatch."""
import re, sys

ansi = re.compile(r'\x1b\[[0-9;]*m')
reg_re = re.compile(r'^(?:[0-9a-f]{8})?\s+(?:\[[A-C]\])?((?:SPI_SHADER_USER_DATA|COMPUTE_USER_DATA)_[A-Z0-9_]+) <- (.*)$')
pkt_re = re.compile(r'^([0-9a-f]{8}) ([A-Z0-9_]+)(\(.*\))?:')
field_re = re.compile(r'^(?:[0-9a-f]{8})?\s+([A-Z_]+) <- (\d+)')


def val(s):
    m = re.search(r'\((0x[0-9a-f]+)\)', s)
    if m:
        return int(m.group(1), 16)
    return int(s.strip().split(' ')[0], 0)


def lines(path):
    """The dump without BC250_MESH_TIMER's exit table (draw rates depend on the wall clock)."""
    out = []
    with open(path, errors='replace') as f:
        for l in f:
            l = ansi.sub('', l.rstrip('\n'))
            if l.startswith('BC250_MESH_TIMER FINAL'):
                break
            out.append(l)
    return out


def tokens(script):
    out = []
    for t in script.split():
        if '*' in t:
            t, n = t.split('*')
            out += [t] * int(n)
        else:
            out.append(t)
    return out


def main():
    script, lo, le, do, de = sys.argv[1:6]
    res = {}
    ok = True
    L, D = lines(le), lines(de)
    dout = open(do).read()
    lout = open(lo).read()
    for name, out in (('legacydb', lout), ('db', dout)):
        if 'DONE' not in out or 'FAIL' in out:
            print(f'status=FAIL reason={name}_run_failed')
            return 1
    m = re.search(r'FB_SETVA gfx0=(0x[0-9a-f]+) gfx1=(0x[0-9a-f]+) cs0=(0x[0-9a-f]+) cs1=(0x[0-9a-f]+)', dout)
    setva = {k: int(v, 16) & 0xffffffff for k, v in zip(('gfx0', 'gfx1', 'cs0', 'cs1'), m.groups())}
    inv = {v: k for k, v in setva.items()}
    res['lines'] = len(D)
    if len(L) != len(D):
        print(f'status=FAIL reason=line_count legacydb={len(L)} db={len(D)}')
        return 1
    fwd, back = {}, {}
    ptr_regs = set()
    ndiff = 0
    bad = []
    for i, (a, b) in enumerate(zip(L, D)):
        if a == b:
            continue
        ndiff += 1
        ma, mb = reg_re.match(a), reg_re.match(b)
        if not ma or not mb or ma.group(1) != mb.group(1):
            bad.append(f'line{i + 1}')
            continue
        va, vb = val(ma.group(2)), val(mb.group(2))
        ptr_regs.add(ma.group(1))
        if fwd.setdefault(va, vb) != vb or back.setdefault(vb, va) != va or vb not in inv:
            bad.append(f'line{i + 1}:{va:#x}->{vb:#x}')
    res['differing_lines'] = ndiff
    res['non_set_pointer_diffs'] = len(bad)
    res['mapping'] = ','.join(f'{inv[v]}' for v in sorted(back)) or 'none'
    if bad:
        ok = False
        res['first_bad'] = bad[0]
    # A set pointer register (one that differs somewhere) holding a descriptor buffer address on a line
    # identical in both dumps would be a pointer the legacy run did not have (other USER_DATA registers,
    # e.g. the Task chunk base, can hold equal numbers by coincidence).
    res['pointer_registers'] = ','.join(sorted(ptr_regs))
    stray = [i for i, l in enumerate(D) if (mm := reg_re.match(l)) and mm.group(1) in ptr_regs and
             val(mm.group(2)) in inv and L[i] == l]
    res['stray_setva_lines'] = len(stray)
    if stray:
        ok = False

    # 3/4: COMPUTE set pointers at dispatches (db run).
    last = None
    since = []
    other_bad = 0
    cd_ok = cd_n = 0
    gfx0_compute_writes = 0
    pkt = None
    fields = {}

    def finish():
        nonlocal cd_ok, cd_n, other_bad
        if pkt == 'DISPATCH_DIRECT' and (fields.get('DIM_X'), fields.get('DIM_Y'), fields.get('DIM_Z')) == (7, 3, 1):
            cd_n += 1
            cd_ok += last == setva['cs0'] and all(v in (setva['cs0'], setva['cs1']) for v in since)
        elif pkt and pkt.startswith('DISPATCH'):
            # Driver dispatches (Task producer, setup helpers): pointers written for them are graphics ones.
            other_bad += any(v not in (setva['gfx0'], setva['gfx1']) for v in since)
        if pkt and pkt.startswith('DISPATCH'):
            since.clear()
    for l in D:
        mp = pkt_re.match(l)
        if mp:
            finish()
            pkt, fields = mp.group(2), {}
            continue
        mf = field_re.match(l)
        if mf and pkt:
            fields[mf.group(1)] = int(mf.group(2))
            continue
        mr = reg_re.match(l)
        if mr and mr.group(1).startswith('COMPUTE_USER_DATA') and mr.group(1) in ptr_regs:
            v = val(mr.group(2))
            if v in inv:
                last = v
                since.append(v)
                gfx0_compute_writes += v == setva['gfx0']
    finish()
    toks = tokens(script)
    submits = dout.count('SUBMIT_OK')
    want_cd = toks.count('cd') * submits
    hybrid = sum(toks.count(t) for t in ('t', 'ti', 'tc', 'T')) * submits
    res['cd_dispatches'] = f'{cd_ok}/{cd_n}'
    if cd_n != want_cd or cd_ok != cd_n:
        ok = False
    res['driver_dispatch_foreign_pointers'] = other_bad
    if other_bad:
        ok = False
    res['gfx0_compute_writes'] = gfx0_compute_writes
    # The internal mesh-only amplifier's producer is a driver shader without descriptors: only
    # application Task shaders read the heap.
    amplified = 'internal mesh-only amplification enabled' in open(de, errors='replace').read()
    res['hybrid_draws'] = hybrid
    res['amplifier'] = amplified
    if gfx0_compute_writes < hybrid:
        ok = False
    print(f'status={"ok" if ok else "FAIL"} ' + ' '.join(f'{k}={v}' for k, v in res.items()))
    return 0 if ok else 1


sys.exit(main())
