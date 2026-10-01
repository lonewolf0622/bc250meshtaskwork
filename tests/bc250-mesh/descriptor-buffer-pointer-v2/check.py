"""Resolve the tiny hybrid fixture's actual PM4 SGPRs against CPU upload captures.

This is a fixture-specific pointer oracle, not a GPU/ISA emulator. Unknown masks,
missing captures and unsupported SH loads fail closed. GPU-written draw records
are not interpreted as if noop had executed the producer.
"""
import hashlib, json, re, struct, sys
from pathlib import Path

MASK48 = (1 << 48) - 1

def fields(line):
    return dict(re.findall(r'(\w+)=(\S+)', line))

def parse(log):
    log = re.sub(r'\x1b\[[0-9;]*m', '', log)
    memory = []
    for line in log.splitlines():
        if line.startswith('BC250MEM '):
            f = fields(line)
            data = bytes.fromhex(f['data'])
            assert len(data) == int(f['bytes'])
            memory.append((int(f['va'], 16) & MASK48, data))
    def read(va, size):
        va &= MASK48
        assert va, 'null memory base'
        for base, data in memory:
            if base <= va and size <= len(data) - (va - base):
                return data[va-base:va-base+size]
        raise AssertionError(f'unresolved pointer {va:x}+{size}')
    pre = re.findall(r'BC250PREIB_BEGIN[^\n]*\n(.*?)BC250PREIB_END', log, re.S)
    raw_text = '\n'.join(pre) if pre else log
    raw = [int(x, 16) for x in re.findall(r'^([0-9a-f]{8})\s', raw_text, re.M)]
    regs, snapshots = {}, []
    i = 0
    while i < len(raw):
        h = raw[i]
        assert h >> 30 in (2, 3), f'bad PM4 at {i}'
        n = 1 if h >> 30 == 2 else ((h >> 16) & 0x3fff) + 2
        body = raw[i+1:i+n]
        assert i+n <= len(raw)
        op = (h >> 8) & 255
        assert op not in (0x4d, 0xaa, 0xad), 'native Task packet'
        if op in (0x76, 0x9b):
            base = 0xb000 + (body[0] & 0xffff) * 4
            for j, value in enumerate(body[1:]):
                regs[base+j*4] = value
        elif op == 0x63:
            assert len(body) == 4 and not body[0] & 3 and not body[2] >> 16
            va = body[0] | body[1] << 32
            base = 0xb000 + body[2] * 4
            for j in range(body[3]):
                regs[base+j*4] = struct.unpack('<I', read(va+j*4, 4))[0]
        if op == 0x15:
            snapshots.append((5, regs.copy()))
        elif op == 0x4c:
            snapshots += [(7, regs.copy()), (4, regs.copy())]
        i += n
    ptrs = [fields(x) for x in log.splitlines() if x.startswith('BC250PTR ')]
    descs = [fields(x) for x in log.splitlines() if x.startswith('BC250DESC ')]
    return read, snapshots, ptrs, descs

def check(log, damage=None):
    read, snapshots, ptrs, descs = parse(log)
    checks, indirect = [], 0
    # These masks come from the captured uploaded shaders for the frozen tiny
    # fixture: Task XYZ/payload/application, Mesh payload/application, FS application.
    masks = {5: 0xff, 7: 0xdc, 4: 0xc0}
    pairs = {5: [(0, 'xyz'), (2, 'payload'), (6, 'application')],
             7: [(2, 'payload'), (6, 'application')], 4: [(6, 'application')]}
    pgm = {5: 0xb830, 7: 0xb320, 4: 0xb020}
    for stage in (4, 5, 7):
        candidates = [f for f in ptrs if int(f['stage']) == stage and int(f['mask'],16) == masks[stage]]
        assert len(candidates) == 1, f'unknown/missing stage {stage} constant ABI'
        f = candidates[0]
        shader = int(f['shader'],16)
        # Refuse changed machine code until its complete pointer contract is audited.
        pins = {5: '01a6b11df061d17aea04051091c3b10d16f372058a0cf27b49fcf8c714adb134',
                7: 'c6a56c42458d5bf61001dcf946e032776bcfeab5e83e968f7e540628ea5b6a45',
                4: 'ed3a9ee051dbd550dde6f405292e199c0b763a99aed902ef020bea5c6ec0641c'}
        code = re.findall(r'BC250POLICY stage=' + str(stage) + r' va=' + f['shader'] +
                          r' [^\n]*\nBC250POLICYCODE ([0-9a-f]+)', log)
        assert len(code) == 1 and hashlib.sha256(bytes.fromhex(code[0])).hexdigest() == pins[stage], 'unaudited shader code'
        matching = [r for s,r in snapshots if s == stage and r.get(pgm[stage]) == ((shader >> 8) & 0xffffffff)]
        assert len(matching) == 1, f'missing/ambiguous dispatch for stage {stage}'
        r = matching[0]
        words = [int(f['words'][i:i+8],16) for i in range(0,len(f['words']),8)]
        source = int(f['inline_source'],16)
        inline_reg = int(f['inline_reg'])
        actual, slot = {}, 0
        for word in range(64):
            if masks[stage] & (1 << word):
                expected = struct.unpack('<I',read(source+word*4,4))[0] if source else words[word]
                actual[word] = r[inline_reg+slot*4]
                assert actual[word] == expected, f'SGPR source mismatch stage={stage} word={word}'
                slot += 1
        for word, name in pairs[stage]:
            va = actual[word] | actual[word+1] << 32
            if damage == (stage, name):
                va = 0
            assert va & MASK48, f'null {name} base stage={stage}'
            if name == 'application':
                assert struct.unpack('<I',read(va,4))[0] == 7, f'wrong application constants stage={stage}'
            else:
                read(va, 12 if name == 'xyz' else 16)
            checks.append({'stage':stage,'pointer':name,'va':f'{va:x}'})
        if int(f['pc_reg']):
            pc = int(f['pc'],16)
            assert r[int(f['pc_reg'])] == (pc & 0xffffffff)
            read(pc,32)
        dc = [d for d in descs if d['shader'] == f['shader'] and int(d['enabled'],16) == 3]
        assert len(dc) == 1
        d = dc[0]
        table = int(d['table'],16)
        indirect_reg = int(d['indirect_reg'])
        if indirect_reg:
            assert table & MASK48 and r[indirect_reg] == (table & 0xffffffff)
            read(table,8)
            indirect += 1
        for index, expected_value in ((0,11),(1,13)):
            va = int(d[f'set{index}'],16)
            if damage == (stage,f'set{index}'):
                va = 0
            assert va & MASK48, f'null descriptor set {index} stage={stage}'
            if indirect_reg:
                assert struct.unpack('<I',read(table+index*4,4))[0] == va & 0xffffffff
            else:
                assert f'set{index}_reg' in d, 'missing descriptor register capture'
                assert r[int(d[f'set{index}_reg'])] == va & 0xffffffff
            descriptor = struct.unpack('<IIII',read(va,16))
            resource = descriptor[0] | (descriptor[1] & 0xffff) << 32
            assert descriptor[2] >= 4, 'empty descriptor range'
            assert struct.unpack('<I',read(resource,4))[0] == expected_value
            checks.append({'stage':stage,'pointer':f'set{index}','va':f'{va:x}','resource':f'{resource:x}'})
    return {'stages':[5,7,4], 'pointers':checks, 'indirect_tables_consumed':indirect,
            'native_task_packets':0, 'gpu_executed':False}

def mutations(log):
    # Inject faults at the resolved pointer boundary, after actual PM4/source
    # consistency checks. This proves that each mandatory pointer is checked.
    cases = [(5,'xyz'),(5,'payload'),(5,'application'),(7,'payload'),(7,'application'),(4,'application')]
    cases += [(s,f'set{i}') for s in (5,7,4) for i in (0,1)]
    for case in cases:
        try:
            check(log,case)
        except AssertionError:
            continue
        raise AssertionError(f'undetected null injection {case}')
    return len(cases)

if __name__ == '__main__':
    print(json.dumps(check(Path(sys.argv[1]).read_text()),indent=2))
