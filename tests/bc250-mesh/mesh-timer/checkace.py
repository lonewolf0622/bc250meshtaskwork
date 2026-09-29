#!/usr/bin/env python3
"""checkace.py <mode> <on-stderr> <off-stderr> <timer-file> <ace-stdout> <queuetl.py> [<queues>]

Checks one run of ace.c (the async-compute case of the mesh-timer suite, drm-shim only).
<mode>    compute: BC250_MESH_TIMER=1 with SUBMITS=1 (default BC250_MESH_TIMER_COMPUTE=1)
          nocompute: the same with BC250_MESH_TIMER_COMPUTE=0
<queues>  expected number of compute queues the harness got (default 2; RADV_BC250_COMPUTE_QUEUE_COUNT=1: 1)

IB dump (RADV_DEBUG=dumpibs), per submitted command buffer ("Main IB begin - GFX|COMPUTE"):
  on:  compute mode: every GFX and COMPUTE main IB starts with the begin timestamp and ends with the end
       timestamp of one ring slot (RELEASE_MEM BOTTOM_OF_PIPE_TS SEND_GPU_CLOCK_COUNTER, t0 then t0 + 8)
       around all its work (dispatches, CP DMA) and has no other timer packet; nocompute mode: GFX IBs as above, COMPUTE IBs without any timer packet;
  off: no timer packets; the on IBs with the timer packets removed equal the off IBs (the same ring order,
       packet names and non-address fields; IB chaining packets ignored).
Timer file: header v2; W lines with consecutive seq, per iteration the harness's pattern (queue, kind,
  command buffer / wait / signal counts), the waits naming the semaphore values the earlier submissions
  signalled; one S line per timed command buffer (same seq, q, qf, cb=i/n) for GFX and (compute mode only)
  COMPUTE command buffers; T lines with raw_ns; I lines with ace_submits / ace_ns.
queuetl.py on the file: v2 format, the graphics and compute queues found, the graphics idle classified
  (the fake GPU runs at submission time, so graphics waits for the compute queue's signal submission).
Prints one key=value line; exits 1 on a mismatch."""
import json
import os
import re
import subprocess
import sys
import tempfile

ansi = re.compile(r'\x1b\[[0-9;]*m')
pkt_re = re.compile(r'^([0-9a-f]{8}) ([A-Z0-9_]+)(\(.*\))?:')
reg_re = re.compile(r'^(?:[0-9a-f]{8})?\s+(?:\[[A-C]\])?([A-Z0-9_]+) <- (.*)$')
ADDRESS_WORDS = ('ADDR', 'ADDRESS', 'BASE', '_LO', '_HI', 'USER_DATA', 'DATA_LO', 'DATA_HI', 'PGM_LO', 'PGM_HI',
                 'IB_BASE', 'INDIRECT_BUFFER', 'SRC_', 'DST_', 'CP_DMA', 'CMP_DATA', 'CTXID', 'MEM_')


def lines(path):
    return [ansi.sub('', l.rstrip('\n')) for l in open(path, errors='replace')]


def val(s):
    m = re.search(r'\((0x[0-9a-f]+)\)', s)
    if m:
        return m.group(1)
    s = s.strip()
    if re.fullmatch(r'\d+', s):
        return '0x%x' % int(s)
    return s


def parse(ls, bo_va, slots):
    """Main IBs: [{'ring': 'GFX'|'COMPUTE', 'packets': [[name, [(reg, val)]]], 'timer': [va...]}]"""
    ibs, cur, pkt, rel = [], None, None, {}

    def finish():
        if cur is None or pkt != 'RELEASE_MEM':
            return
        if rel.get('DATA_SEL') == 'SEND_GPU_CLOCK_COUNTER' and rel.get('EVENT_TYPE', 'BOTTOM_OF_PIPE_TS') == 'BOTTOM_OF_PIPE_TS':
            va = (int(rel.get('ADDRESS_HI', '0x0'), 16) << 32) | int(rel.get('ADDRESS_LO_32B', '0x0'), 16)
            if bo_va <= va < bo_va + slots * 16:
                cur['timer'].append((len(cur['packets']) - 1, va))
                cur['packets'][-1][0] = 'TIMER'

    for line in ls:
        m = re.search(r'(Main|Preamble|Postamble) IB (begin|end) - (\w+)', line)
        if m:
            finish()
            pkt = None
            cur = {'ring': m.group(3), 'packets': [], 'timer': []} if (m.group(1), m.group(2)) == ('Main', 'begin') else None
            if cur is not None:
                ibs.append(cur)
            continue
        if cur is None:
            continue
        mm = pkt_re.match(line)
        if mm:
            finish()
            pkt = mm.group(2)
            rel = {}
            cur['packets'].append([pkt, []])
            continue
        if 'EVENT_TYPE = ' in line and pkt == 'RELEASE_MEM':
            rel['EVENT_TYPE'] = line.split('=', 1)[1].strip()
        mr = reg_re.match(line)
        if mr and pkt:
            cur['packets'][-1][1].append((mr.group(1), val(mr.group(2))))
            if pkt == 'RELEASE_MEM':
                rel[mr.group(1)] = val(mr.group(2))
    finish()
    return ibs


def masked(ib, drop_timer):
    out = [ib['ring']]
    for name, regs in ib['packets']:
        if name == 'TIMER' and drop_timer:
            continue
        if name in ('NOP', 'INDIRECT_BUFFER'):
            continue
        out.append(name)
        for rn, rv in regs:
            if any(w in rn for w in ADDRESS_WORDS) or name in ('DMA_DATA', 'WRITE_DATA', 'COPY_DATA', 'RELEASE_MEM',
                                                               'ACQUIRE_MEM', 'EVENT_WRITE_EOP', 'WAIT_REG_MEM',
                                                               'COND_EXEC', 'SET_BASE', 'SET_SH_REG', 'PFP_SYNC_ME'):
                out.append(rn + '=*')
            else:
                out.append(rn + '=' + rv)
    return out


def main():
    mode, on_path, off_path, tfile, out_path, queuetl = sys.argv[1:7]
    nq = int(sys.argv[7]) if len(sys.argv) > 7 else 2
    compute = mode == 'compute'
    errs = []
    on, off = lines(on_path), lines(off_path)
    for tag, ls in (('on', on), ('off', off)):
        if any(re.search(r'validation failed|NIR_VALIDATE|assertion|error:|^FAIL |Segmentation', l, re.I) for l in ls):
            errs.append('validation_' + tag)
    out = lines(out_path)
    if not any(l.startswith('DONE') for l in out):
        errs.append('harness_not_done')
    fam = [l for l in out if l.startswith('FAMILIES')]
    hdr = [l for l in on if l.startswith('# BC250_MESH_TIMER v2 ')]
    if not hdr:
        print('error=no_v2_header')
        sys.exit(1)
    bo_va = int(re.search(r'bo_va=0x([0-9a-f]+)', hdr[0]).group(1), 16)
    slots = int(re.search(r'slots=(\d+)', hdr[0]).group(1))
    if ('compute=1' in hdr[0]) != compute:
        errs.append('header_compute_flag')

    ibs_on, ibs_off = parse(on, bo_va, slots), parse(off, bo_va, slots)
    rings = [ib['ring'] for ib in ibs_on]
    if [ib['ring'] for ib in ibs_off] != rings:
        errs.append('ring_order_differs')
    n_gfx = rings.count('GFX')
    n_ace = rings.count('COMPUTE')
    if not n_gfx or not n_ace:
        errs.append('rings_gfx=%d_compute=%d' % (n_gfx, n_ace))
    for i, ib in enumerate(ibs_on):
        ts = ib['timer']
        want = 2 if (ib['ring'] == 'GFX' or compute) else 0
        if len(ts) != want:
            errs.append('ib%d_%s_timer_packets=%d' % (i, ib['ring'], len(ts)))
            continue
        if want:
            # the pair brackets the command buffer's work (dispatches / CP DMA fills)
            (p0, va0), (p1, va1) = ts
            work = [k for k, (n, _) in enumerate(ib['packets']) if n.startswith('DISPATCH') or n in ('DMA_DATA', 'CP_DMA')]
            if va0 % 16 or va1 != va0 + 8 or not work or not (p0 < work[0] and p1 > work[-1]):
                errs.append('ib%d_%s_pair_position' % (i, ib['ring']))
    if any(ib['timer'] for ib in ibs_off):
        errs.append('off_has_timer_packets')
    if len(ibs_on) == len(ibs_off):
        for i, (a, b) in enumerate(zip(ibs_on, ibs_off)):
            if masked(a, True) != masked(b, False):
                errs.append('ib%d_stream_differs' % i)
                break

    # timer file
    tl = lines(tfile)
    W = [dict(re.findall(r'(\w+)=(\S+)', l)) for l in tl if l.startswith('W ')]
    S = [dict(re.findall(r'(\w+)=(\S+)', l)) for l in tl if l.startswith('S ')]
    T = [l for l in tl if l.startswith('T ')]
    I = [dict(re.findall(r'(\w+)=(\S+)', l)) for l in tl if l.startswith('I interval')]
    seqs = [int(w['seq']) for w in W]
    if seqs != list(range(1, len(seqs) + 1)):
        errs.append('w_seq_not_consecutive')
    # the harness pattern: ace q0 (1 cb, signal), gfx (1 cb, wait+signal), ace q1 (1 cb, wait, 2 signals),
    # gfx (1 cb, wait binary), gfx (no cb, signal) then vkDeviceWaitIdle's semaphore-only submissions
    pat = [('ace', '1', '0', '1'), ('gfx', '1', '1', '1'), ('ace', '1', '1', '2'), ('gfx', '1', '1', '0'),
           ('gfx', '0', '0', '1')]
    it = 0
    i = 0
    sig = {}
    while i < len(W):
        w = W[i]
        got = [(x['qf'], x['ncb'], x['nw'], x['nsig']) for x in W[i:i + 5]]
        if got == pat:
            q0, q1 = W[i]['q'], W[i + 2]['q']
            if not q0.startswith('1.') or not q1.startswith('1.') or W[i + 1]['q'] != '0.0':
                errs.append('it%d_queues' % it)
            if nq == 2 and q0 == q1:
                errs.append('it%d_same_compute_queue' % it)
            if nq == 1 and q0 != q1:
                errs.append('it%d_two_compute_queues' % it)
            s1 = W[i]['s'].split(',')[0]
            if W[i + 1]['w'] != s1:
                errs.append('it%d_gfx_wait_not_ace_signal' % it)
            if W[i + 2]['w'] != W[i + 1]['s']:
                errs.append('it%d_ace_wait_not_gfx_signal' % it)
            if W[i + 3]['w'] != W[i + 2]['s'].split(',')[1] or not W[i + 3]['w'].endswith(':0'):
                errs.append('it%d_binary_wait' % it)
            if any(x['thr'] != '0' for x in W[i:i + 5]):
                errs.append('it%d_threaded' % it)
            if any(int(x['wall_ns']) < int(x['call_ns']) for x in W[i:i + 5]):
                errs.append('it%d_wall_before_call' % it)
            for x in W[i:i + 5]:
                sig[int(x['seq'])] = x
            it += 1
            i += 5
        else:
            i += 1
    if it != 2:
        errs.append('iterations=%d' % it)
    by_seq = {}
    for s in S:
        by_seq.setdefault(int(s['seq']), []).append(s)
    for seq, w in sig.items():
        ss = by_seq.get(seq, [])
        want = int(w['ncb']) if (w['qf'] == 'gfx' or compute) else 0
        if len(ss) != want:
            errs.append('seq%d_s_lines=%d' % (seq, len(ss)))
        for k, s in enumerate(ss):
            if s['q'] != w['q'] or s['qf'] != w['qf'] or s['cb'] != '%d/%s' % (k, w['ncb']):
                errs.append('seq%d_s_fields' % seq)
            if not (int(s['gpu_t1']) > int(s['gpu_t0']) > 0):
                errs.append('seq%d_s_times' % seq)
    if any(s['qf'] == 'ace' for s in S) != compute:
        errs.append('ace_s_lines_presence')
    if not T or not all('raw_ns=' in l and 'cpu_raw_at_ns=' in l for l in T):
        errs.append('t_lines')
    ace_sub = sum(int(x.get('ace_submits', -1)) for x in I)
    if ace_sub != (2 * 2 if compute else 0):
        errs.append('i_ace_submits=%d' % ace_sub)

    # the analyzer
    with tempfile.TemporaryDirectory() as td:
        jpath = os.path.join(td, 'q.json')
        r = subprocess.run([sys.executable, queuetl, tfile, '--json', jpath], capture_output=True, text=True)
        if r.returncode:
            errs.append('queuetl_rc=%d' % r.returncode)
        else:
            J = json.load(open(jpath))
            qs = set(J['queues'])
            want_q = {'gfx:0.0', 'ace:1.0'} | ({'ace:1.1'} if nq == 2 else set())
            if J['format'] != 'v2' or not want_q <= qs:
                errs.append('queuetl_queues=%s' % ','.join(sorted(qs)))
            if compute and not all(J['queues'][q]['timed_cbs'] > 0 for q in want_q):
                errs.append('queuetl_timed_cbs')
            if not compute and any(J['queues'][q]['timed_cbs'] for q in want_q if q.startswith('ace')):
                errs.append('queuetl_ace_timed_without_compute')
            g = J.get('graphics', {})
            if 'app_late_until_signal_submitted' not in g:
                errs.append('queuetl_no_graphics_split')
    print('mode=%s ibs=%d gfx_ibs=%d compute_ibs=%d iterations=%d w_lines=%d s_lines=%d %s ok=%s%s' % (
        mode, len(ibs_on), n_gfx, n_ace, it, len(W), len(S), fam[0].replace(' ', '_') if fam else 'FAMILIES=?',
        not errs, (' errors=' + ','.join(errs)) if errs else ''))
    sys.exit(1 if errs else 0)


if __name__ == '__main__':
    main()
