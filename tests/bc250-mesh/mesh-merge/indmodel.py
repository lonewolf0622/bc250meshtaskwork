#!/usr/bin/env python3
"""CPU model of RADV_BC250_MESH_MERGE option A indirect draws (radv_bc250.c bc250_merge_prep_pipeline,
radv_bc250_draw_merge_indirect and the grid selection in radv_bc250_merge_mesh), with the same u32
integer formulas:
  prep, per active record i (i < min(count, maxDrawCount)):
      n = (x*y)*z; out[32i..] = {n/K + (n%K != 0), 1, 1, 0, x, y, z, 0}
  CP (DISPATCH_MESH_INDIRECT_MULTI, stride 32): record i launches r0*r1*r2 groups, DrawID = i,
      grid SGPRs = (r0, r1, r2)
  shader: dims SGPR != 0 -> (X,Y,Z) = load(dims + 32*DrawID + 16), else the grid SGPRs;
      N = (X*Y)*Z; g = hardware group index; lane instance k < K: linear = g*K + k,
      valid = linear < N, WorkGroupID = (linear % X, (linear / X) % Y, linear / (X*Y)),
      live (allocates outputs) = g*K < N; gl_NumWorkGroups = (X,Y,Z)
Checks, per record: every API workgroup id of the X*Y*Z grid is run exactly once, by the record's
DrawID, gl_NumWorkGroups is the application's grid, every launched group is live (no empty
groups: option A launches ceil(N/K)), and records beyond the count are never read. Option B (the CP
reads the application's record, dims SGPR 0) is modeled too: exactly-once coverage with N groups,
of which N - ceil(N/K) are empty."""
import itertools, random

M32 = 0xffffffff


def prep(records, count, K):
    out = {}
    for i in range(min(count, len(records))):
        x, y, z = records[i]
        n = (x * y & M32) * z & M32
        out[i] = (n // K + (n % K != 0), 1, 1, 0, x, y, z, 0)
    return out


def run(records, count, K, option_a=True):
    """Returns ({(draw_id, wid): times}, launched, empty) for one indirect call."""
    table = prep(records, count, K) if option_a else None
    seen = {}; launched = empty = 0
    for i in range(min(count, len(records))):
        rec = table[i][:3] if option_a else records[i]
        groups = (rec[0] * rec[1] & M32) * rec[2] & M32
        dims = table[i][4:7] if option_a else rec  # shader: dims SGPR nonzero -> driver record
        X, Y, Z = dims
        N = (X * Y & M32) * Z & M32
        for g in range(groups):
            launched += 1
            live = (g * K & M32) < N
            if not live:
                empty += 1
                continue
            for k in range(K):
                linear = (g * K + k) & M32
                if linear >= N:
                    continue
                wid = (linear % X, (linear // X) % Y, linear // ((X * Y) & M32))
                assert (X, Y, Z) == tuple(records[i]), 'gl_NumWorkGroups'
                seen[(i, wid)] = seen.get((i, wid), 0) + 1
    return seen, launched, empty


def check(records, count, K):
    active = min(count, len(records))
    want = {(i, (a, b, c)) for i in range(active) for (a, b, c) in
            itertools.product(range(records[i][0]), range(records[i][1]), range(records[i][2]))}
    for opt in (True, False):
        seen, launched, empty = run(records, count, K, opt)
        assert set(seen) == want and all(v == 1 for v in seen.values()), ('coverage', records, count, K, opt)
        n_total = sum(records[i][0] * records[i][1] * records[i][2] for i in range(active))
        wanted = sum(-(-(records[i][0] * records[i][1] * records[i][2]) // K) for i in range(active))
        if opt:
            assert launched == wanted and empty == 0, ('option A launch', records, K, launched, wanted, empty)
        else:
            assert launched == n_total and empty == n_total - wanted, ('option B launch', records, K)
    return len(want)


random.seed(250)
cases = ids = 0
# exhaustive small 3D grids (including zero dimensions), K = 2..5, one record
for K in (2, 3, 4, 5):
    for x, y, z in itertools.product(range(0, 7), range(0, 5), range(0, 4)):
        ids += check([(x, y, z)], 1, K); cases += 1
# multi-record calls: DrawID addressing, count < max, count > max, count 0
for K in (2, 3, 5):
    for trial in range(300):
        R = random.randint(1, 40)
        recs = [(random.randint(0, 9), random.randint(0, 4), random.randint(0, 3)) for _ in range(R)]
        count = random.choice([R, random.randint(0, R), R + 5, 0])
        ids += check(recs, count, K); cases += 1
# 1000 records (the suite's ind1000 layout), large 1D grids and tails
recs = [(100, 1, 1), (64, 1, 1)] + [((7 * i) % 13 + 1, i % 3 + 1, i % 2 + 1) for i in range(2, 1000)]
for K in (3, 5):
    for count in (1000, 700, 999, 3):
        ids += check(recs, count, K); cases += 1
for K in (2, 3, 4, 5):
    for n in (1, 2, 3, 64, 65, 97, 98, 99, 100, 15625, 65535, 65536):
        ids += check([(n, 1, 1)], 1, K); cases += 1
        ids += check([(1, n, 1)], 1, K); cases += 1
print('indirect model: %d calls, %d API workgroups each covered exactly once (option A: ceil(N/K) '
      'groups, none empty; option B: N groups)' % (cases, ids))
