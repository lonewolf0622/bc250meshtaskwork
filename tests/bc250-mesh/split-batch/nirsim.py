#!/usr/bin/env python3
"""nirsim.py <stderr with RADV_BC250_SPLIT_BATCH_PRINT_NIR=1> [trials]

CPU model check of RADV_BC250_SPLIT_BATCH_PREP. Interprets the builder NIR the driver printed for the
per-draw split-argument setup shader (bc250_split_arguments) and for the batched one (bc250_split_batch)
-- a small interpreter for the NIR subset they use -- and runs both on random inputs: random split
indirect draws (records 1..1000, strides 12..40, with and without a count buffer, count values below,
equal to and above the record count, dimensions including 0 and values whose product with the split
factor wraps 32 bits, split factors 2..4). Per draw, the per-draw shader runs with the driver's push
constants (input, output, count, records, stride, pieces) and ceil(records/64) workgroups; the batched
shader runs once for all draws with the driver's list (16-byte header {chunks, 1, 1, 0}, 128 48-byte
entries: input, output, count, records, stride, pieces, first chunk; then one uint32 entry index per
chunk), one workgroup per chunk (ceil(records/64) chunks per draw, as many as the per-draw dispatches). Both write into sentinel-filled memory;
the check is that every output byte is equal (records, zero records at and beyond the count, the
unwritten fourth dword) and equal to a direct Python statement of the transform.
Prints one key=value line and exits 1 on any mismatch."""
import random, re, struct, sys

ansi = re.compile(r'\x1b\[[0-9;]*m')


def extract(path, name):
    ls = [ansi.sub('', l.rstrip('\n')) for l in open(path, errors='replace')]
    for i, l in enumerate(ls):
        if l.startswith('name: ' + name):
            j = i
            while not ls[j].startswith('impl main {'):
                j += 1
            body = []
            depth = 0
            for l2 in ls[j:]:
                body.append(l2)
                depth += l2.count('{') - l2.count('}')
                if depth == 0:
                    break
            return body
    raise SystemExit('no NIR for ' + name)


def parse(body):
    """-> nested list of ('ins', text) | ('if', cond, then, else) | ('loop', body) | ('break',) | ('block', label)"""
    pos = 1  # skip "impl main {"

    def seq():
        nonlocal pos
        out = []
        while pos < len(body):
            l = body[pos].strip()
            l = re.sub(r'//.*$', '', l).strip()
            pos += 1
            if not l or l.startswith('decl_var'):
                continue
            if l == '}':
                return out, 'end'
            if l == '} else {':
                return out, 'else'
            m = re.match(r'block (b\d+):', l)
            if m:
                out.append(('block', m.group(1)))
                continue
            m = re.match(r'if (%\d+) \{$', l)
            if m:
                then, how = seq()
                els = []
                if how == 'else':
                    els, how = seq()
                out.append(('if', m.group(1), then, els))
                continue
            if l == 'loop {':
                b, _ = seq()
                out.append(('loop', b))
                continue
            if l == 'break':
                out.append(('break',))
                continue
            out.append(('ins', l))
        return out, 'eof'
    prog, _ = seq()
    return prog


class Break(Exception):
    pass


def mask(bits):
    return (1 << bits) - 1 if bits > 1 else 1


class Inv:
    def __init__(self, mem, push, wg, lid):
        self.mem, self.push, self.wg, self.lid = mem, push, wg, lid
        self.v = {}      # ssa -> (bits, [components])
        self.vars = {}
        self.block = None
        self.pred = None

    def src(self, tok):
        m = re.match(r'(%\d+)(?:\.([xyzw]+))?', tok)
        bits, comps = self.v[m.group(1)]
        if m.group(2):
            comps = [comps['xyzw'.index(c)] for c in m.group(2)]
        return bits, comps

    def load(self, addr, n, size):
        out = []
        for c in range(n):
            a = addr + c * size
            b = bytes(self.mem.get(a + k, 0xEE) for k in range(size))
            out.append(int.from_bytes(b, 'little'))
        return out

    def store(self, addr, comps, size, wrmask):
        for c, val in enumerate(comps):
            if 'xyzw'[c] not in wrmask:
                continue
            for k in range(size):
                self.mem[addr + c * size + k] = (val >> (8 * k)) & 0xff

    def ins(self, t):
        m = re.match(r'(\d+)(?:x(\d+))?\s+(%\d+) = (.*)$', t)
        if not m:
            # stores
            m = re.match(r'@store_global \((%\d+(?:\.[xyzw]+)?), (%\d+)\) \(wrmask=([xyzw]+),', t)
            if m:
                bits, comps = self.src(m.group(1))
                _, (addr,) = self.src(m.group(2))
                self.store(addr, comps, bits // 8, m.group(3))
                return
            m = re.match(r'@store_deref \((%\d+), (%\d+)\)', t)
            if m:
                self.vars[self.v[m.group(1)]] = self.v[m.group(2)]
                return
            raise SystemExit('unsupported: ' + t)
        bits, n, dst, rhs = int(m.group(1)), int(m.group(2) or 1), m.group(3), m.group(4)
        op = rhs.split(' ')[0].split('(')[0]
        args = re.findall(r'%\d+(?:\.[xyzw]+)?', rhs.split(')')[0] if rhs.startswith('@') else rhs)
        M = mask(bits)
        def a(i):
            return self.src(args[i])[1]
        if op == 'load_const':
            vals = [int(x, 16) for x in re.findall(r'0x[0-9a-f]+', re.search(r'\((.*)\)', rhs).group(1).split(' = ')[0])]
            res = vals[:n]
        elif op == 'mov':
            res = a(0)
        elif op == 'deref_var':
            self.v[dst] = re.search(r'&(\w+)', rhs).group(1)
            return
        elif op == '@load_deref':
            self.v[dst] = self.vars[self.v[args[0]]]
            return
        elif op == '@load_push_constant':
            base = a(0)[0]
            res = [int.from_bytes(self.push[base + c * bits // 8: base + (c + 1) * bits // 8], 'little') for c in range(n)]
        elif op == '@load_local_invocation_index':
            res = [self.lid]
        elif op == '@load_workgroup_id':
            res = [self.wg, 0, 0]
        elif op == '@load_global':
            res = self.load(a(0)[0], n, bits // 8)
        elif op in ('iadd', 'isub', 'imul', 'ishl', 'iand', 'ior'):
            x, y = a(0), a(1)
            f = {'iadd': lambda p, q: p + q, 'isub': lambda p, q: p - q, 'imul': lambda p, q: p * q, 'ishl': lambda p, q: p << (q % bits),
                 'iand': lambda p, q: p & q, 'ior': lambda p, q: p | q}[op]
            res = [f(p, q) & M for p, q in zip(x, y)]
        elif op == 'inot':
            res = [(~p) & M for p in a(0)]
        elif op in ('ult', 'uge', 'ine', 'ieq'):
            f = {'ult': lambda p, q: p < q, 'uge': lambda p, q: p >= q, 'ine': lambda p, q: p != q,
                 'ieq': lambda p, q: p == q}[op]
            res = [int(f(p, q)) for p, q in zip(a(0), a(1))]
        elif op == 'bcsel':
            res = [q if c else r for c, q, r in zip(a(0), a(1), a(2))]
        elif op == 'u2u64':
            res = a(0)
        elif op == 'pack_64_2x32_split':
            res = [lo | (hi << 32) for lo, hi in zip(a(0), a(1))]
        elif op in ('vec2', 'vec3', 'vec4'):
            res = [a(i)[0] for i in range(len(args))]
        elif op == 'phi':
            for blk, s in re.findall(r'(b\d+): (%\d+)', rhs):
                if blk == self.pred:
                    res = self.src(s)[1]
                    break
            else:
                raise SystemExit('phi without executed predecessor: %s (pred %s)' % (t, self.pred))
        else:
            raise SystemExit('unsupported op ' + op)
        self.v[dst] = (bits, [r & M for r in res])

    def run(self, prog):
        for st in prog:
            k = st[0]
            if k == 'block':
                self.pred, self.block = self.block, st[1]
            elif k == 'ins':
                self.ins(st[1])
            elif k == 'if':
                c = self.v[st[1]][1][0]
                self.run(st[2] if c else st[3])
            elif k == 'loop':
                try:
                    while True:
                        self.run(st[1])
                except Break:
                    pass
            elif k == 'break':
                raise Break()


def dispatch(prog, mem, push, groups):
    for wg in range(groups):
        for lid in range(64):
            Inv(mem, push, wg, lid).run(prog)


def reference(mem_app, d):
    out = {}
    count = d['records'] if not d['count'] else min(struct.unpack_from('<I', bytes(mem_app[d['count'] + k] for k in range(4)))[0], 2**32)
    for i in range(d['records']):
        if i < count:
            x, y, z = struct.unpack('<3I', bytes(mem_app[d['input'] + i * d['stride'] + k] for k in range(12)))
            f = d['pieces']
            if y >= x and z >= x:
                x = (x * f) & 0xffffffff
            elif z >= y:
                y = (y * f) & 0xffffffff
            else:
                z = (z * f) & 0xffffffff
            rec = (x, y, z)
        else:
            rec = (0, 0, 0)
        for c in range(3):
            for k in range(4):
                out[d['output'] + 16 * i + 4 * c + k] = (rec[c] >> (8 * k)) & 0xff
    return out


def main():
    path = sys.argv[1]
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    per_draw = parse(extract(path, 'bc250_split_arguments'))
    batch = parse(extract(path, 'bc250_split_batch'))
    rng = random.Random(250)
    errs = []
    ndraws = nrecords = 0
    for t in range(trials):
        app = {}
        draws = []
        in_base, cnt_base, out_base = 0x100000000, 0x200000000, 0x7fff00000000
        for k in range(rng.randint(1, 9)):
            records = rng.choice([1, 1, 2, 3, 63, 64, 65, 130, 200, 1000])
            stride = rng.choice([12, 16, 20, 40])
            d = {'input': in_base, 'output': out_base, 'records': records, 'stride': stride,
                 'pieces': rng.choice([2, 3, 4]), 'count': 0}
            for i in range(records):
                dims = [rng.choice([0, 1, 2, 3, 7, 100, 65535, 0x7fffffff, 0xffffffff]) for _ in range(3)]
                for c, v in enumerate(dims):
                    for b in range(4):
                        app[in_base + i * stride + 4 * c + b] = (v >> (8 * b)) & 0xff
            if rng.random() < 0.5:
                d['count'] = cnt_base
                cv = rng.choice([0, 1, records // 2, records, records + 3, 0xffffffff])
                for b in range(4):
                    app[cnt_base + b] = (cv >> (8 * b)) & 0xff
                cnt_base += 0x1000
            in_base += (records * stride + 0xfff) & ~0xfff
            out_base += (records * 16 + 0xfff) & ~0xfff
            draws.append(d)
        # per-draw path
        mem_a = dict(app)
        for d in draws:
            push = struct.pack('<QQQIII', d['input'], d['output'], d['count'], d['records'], d['stride'], d['pieces'])
            dispatch(per_draw, mem_a, push, (d['records'] + 63) // 64)
        # batched path (list in driver memory, CPU-written)
        mem_b = dict(app)
        lst = 0x7ffe00000000
        entries, table = b'', []
        for e, d in enumerate(draws):
            entries += struct.pack('<QQQIIII8x', d['input'], d['output'], d['count'], d['records'], d['stride'],
                                   d['pieces'], len(table))
            table += [e] * ((d['records'] + 63) // 64)
        blob = struct.pack('<4I', len(table), 1, 1, 0) + entries + bytes(48 * (128 - len(draws))) + \
            struct.pack('<%dI' % len(table), *table)
        for i, b in enumerate(blob):
            mem_b[lst + i] = b
        dispatch(batch, mem_b, struct.pack('<Q', lst), struct.unpack_from('<I', blob)[0])
        for d in draws:
            ndraws += 1
            nrecords += d['records']
            ref = reference(app, d)
            for i in range(d['records'] * 16):
                addr = d['output'] + i
                va, vb = mem_a.get(addr), mem_b.get(addr)
                if va != vb:
                    errs.append('trial%d_draw_output_%x_perdraw_%s_batch_%s' % (t, addr, va, vb))
                if i % 16 < 12 and va != ref[addr]:
                    errs.append('trial%d_reference_%x' % (t, addr))
                if i % 16 >= 12 and va is not None:
                    errs.append('trial%d_fourth_dword_written_%x' % (t, addr))
        # nothing written outside the outputs
        outs = set()
        for d in draws:
            outs.update(range(d['output'], d['output'] + 16 * d['records']))
        for mem, name in ((mem_a, 'perdraw'), (mem_b, 'batch')):
            for addr in mem:
                if addr in outs or addr in app or (name == 'batch' and lst <= addr < lst + len(blob)):
                    continue
                errs.append('%s_stray_write_%x' % (name, addr))
    print('nirsim trials=%d draws=%d records=%d errors=%s' % (trials, ndraws, nrecords, ','.join(errs[:10]) or 'none'))
    sys.exit(1 if errs else 0)


main()
