#!/usr/bin/env python3
"""CPU model of the stage-2 vertex map (bc250_merge_pp_epilogue in src/amd/vulkan/radv_bc250.c),
written with the same integer formulas as the NIR. For random instances (runtime vertex/primitive
counts, invalid instances, indices >= the runtime vertex count, provoking vertex first and last) it checks that
every kept primitive's physical corners hold the right logical vertices, that the provoking corner
is a private vertex carrying the primitive's data, that holes use only sink vertices and that all
indices are < K*C+3. It also checks the zipper (slot(rep t), slot(prov j)) is a permutation with a
correct 3-candidate inverse for every V <= 64, P <= 64, and prints the exported index pattern of
lattice meshlets (backward references: references below the largest index seen so far)."""
import random, sys

U = 0xffffffff


def prov_slot(j, R, P):
    return (j + min((((j * R) & U) // P) + 1, R)) & U


def rep_slot(t, R, P):
    return t + min((t * P + R - 1) // R, P)


def merge(K, V, P, inst, pc):
    R, C = V - 1, V - 1 + P
    vm = K * C + 3
    empty = (0, 0, [(0, 0, 0)] * P, [None] * P)
    phys = []
    for h in range(vm):
        if h >= K * C:
            phys.append('sink'); continue
        kk, t = h // C, h % C
        vk, pk, tris, pd = inst[kk] or empty
        u0 = V if pk == 0 else min(tris[0][pc], V - 1)
        jg, best = (t * P) // C, U
        for d in (-1, 0, 1):
            j = (jg + d) & U
            if j < P and t >= prov_slot(j, R, P):
                best = j
        has = best != U
        if not (has and prov_slot(best, R, P) == t):
            rt = t - ((best + 1) if has else 0)
            u = rt + (1 if rt >= u0 else 0)
            phys.append((kk, u, None) if u < vk else 'sink')
        else:
            phys.append((kk, min(tris[best][pc], V - 1), pd[best]) if best < pk else 'sink')
    idx = []
    for h in range(K * P):
        kk, j = h // P, h % P
        vk, pk, tris, pd = inst[kk] or empty
        if j >= pk:
            idx.append((K * C, K * C + 1, K * C + 2)); continue
        u0, base, out = min(tris[0][pc], V - 1), kk * C, []
        for c in range(3):
            uc = min(tris[j][c], V - 1)
            if c == pc:
                out.append(base + prov_slot(j, R, P))
            elif uc == u0:
                out.append(base + (1 if R else 0))
            else:
                out.append(base + rep_slot(uc - (1 if u0 < uc else 0), R, P))
        idx.append(tuple(out))
    return phys, idx, vm


def check(K, V, P, inst, pc):
    phys, idx, vm = merge(K, V, P, inst, pc)
    for h, t in enumerate(idx):
        assert all(x < vm for x in t)
        kk, j = h // P, h % P
        if not inst[kk] or j >= inst[kk][1]:
            assert all(phys[x] == 'sink' for x in t); continue
        vk, pk, tris, pd = inst[kk]
        for c in range(3):
            p, u = phys[t[c]], min(tris[j][c], V - 1)
            if u >= vk and c != pc:
                continue  # index >= the runtime vertex count (undefined): any in-range vertex
            assert p != 'sink' and p[0] == kk and p[1] == u, (h, c, p, tris[j])
        assert phys[t[pc]][2] == pd[j]
    return idx


def backward(idx):
    mx, back, maxback, refs = -1, 0, 0, 0
    for t in idx:
        for x in t:
            refs += 1
            if x < mx:
                back += 1; maxback = max(maxback, mx - x)
            mx = max(mx, x)
    return back / refs, maxback


def lattice(cols, V, P):
    L = []
    for y in range(V // cols - 1):
        for x in range(cols - 1):
            a = y * cols + x; b = a + 1; c = a + cols; d = c + 1
            L += [(a, b, c), (b, d, c)]
    return [L[t * len(L) // P] if len(L) >= P else L[t % len(L)] for t in range(P)]


random.seed(1)
bad = 0
for V in range(2, 65):
    for P in range(1, 65):
        R, C = V - 1, V - 1 + P
        slots = {rep_slot(t, R, P): ('r', t) for t in range(R)}
        slots.update({prov_slot(j, R, P): ('p', j) for j in range(P)})
        if sorted(slots) != list(range(C)):
            bad += 1
if bad:
    print('zipper: %d (V,P) pairs are not a permutation' % bad); sys.exit(1)
n = 0
for (K, V, P) in [(3, 32, 32), (5, 16, 16), (3, 24, 24), (2, 48, 48), (4, 20, 20), (3, 24, 32), (3, 40, 22),
                  (2, 60, 33), (3, 2, 25), (2, 64, 40)]:
    for pc in (0, 2):
        for trial in range(200):
            inst = []
            for k in range(K):
                if random.random() < 0.1:
                    inst.append(None); continue
                vk, pk = random.randint(1, V), random.randint(0, P)
                tris = [tuple(random.randrange(vk + (trial & 1)) for _ in range(3)) for _ in range(P)]
                inst.append((vk, pk, tris, [('data', k, j) for j in range(P)]))
            check(K, V, P, inst, pc)
            n += 1
print('vertex map: zipper permutation ok for V 2..64 x P 1..64; %d random merged subgroups ok' % n)
for name, cols, V, P, K in (('32/32 strip 16x2', 16, 32, 32, 3), ('32/32 grid 8x4', 8, 32, 32, 3),
                            ('16/16 grid 4x4', 4, 16, 16, 5), ('48/48 grid 8x6', 8, 48, 48, 2)):
    tris = lattice(cols, V, P)
    idx = check(K, V, P, [(V, P, tris, list(range(P)))] * K, 0)
    raw = [tuple(k * V + x for x in t) for k in range(K) for t in tris]
    print('pattern %-17s stage2 backward=%.2f max_jump=%-3d raw backward=%.2f max_jump=%-3d first=%s' % (
        name, *backward(idx), *backward(raw), idx[:3]))
