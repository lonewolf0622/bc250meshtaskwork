#!/usr/bin/env python3
"""checksem.py <ioctlspy log> <semtest stdout> <mode> [--bos-below N]

Checks one semtest run (see semtest.c). mode: "probe" (the runtime probes every submission with
waits, the default behaviour) or "known" (RADV_BC250_SUBMIT_KNOWN_SIGNALS=1: no probe for waits whose
signal was already accepted, a probe for the wait-before-signal ones).
Per scenario: the kernel submissions (CS) in order with their wait entries (semaphore, point, flags
WAIT_FOR_SUBMIT) and signal entries (named semaphores; RADV's own queue syncobjs are ignored),
binary resets after a consuming wait, probe ioctls, host signals, and the host-side RESULT values.
--bos-below N: every CS passes fewer than N BO handles (RADV_BC250_LOCAL_BOS).
Prints one line per scenario and "checksem ok" / exits 1."""
import re
import sys

log, out, mode = sys.argv[1], sys.argv[2], sys.argv[3]
bos_below = int(sys.argv[sys.argv.index('--bos-below') + 1]) if '--bos-below' in sys.argv else None

names = {}      # handle -> semaphore name
pending = None
scen = {}
cur = None
for line in open(log):
    line = line.rstrip('\n')
    if line.startswith('MARK sem '):
        pending = line.split()[2]
        continue
    if line.startswith('SYNCOBJ_CREATE handle='):
        h = int(line.split('handle=')[1].split()[0])
        if pending:
            names[h] = pending
            pending = None
        continue
    if line.startswith('MARK begin '):
        cur = line.split()[2]
        scen[cur] = []
        continue
    if line.startswith('MARK end '):
        cur = None
        continue
    if cur is not None:
        scen[cur].append(line)


def ents(field):
    if field == '-':
        return []
    r = []
    for e in field.split(','):
        h, p, f = e.split(':')
        r.append((int(h), int(p), int(f)))
    return r


def named(lst):
    return sorted((names[h], p) for h, p, f in lst if h in names)


def parse(lines):
    ev = []
    for l in lines:
        if l.startswith('CS '):
            d = dict(re.findall(r'(\w+)=(\S+)', l))
            w, s = ents(d['waits']), ents(d['signals'])
            ev.append(('CS', named(w), named(s), [f for h, p, f in w], int(d['bos']), int(d['ibs'])))
        elif l.startswith('SYNCOBJ_TIMELINE_WAIT_PROBE'):
            ev.append(('PROBE', named((int(x.split(':')[0]), int(x.split(':')[1]), 0) for x in l.split()[2:])))
        elif l.startswith('SYNCOBJ_RESET'):
            ev.append(('RESET', named((int(x.split(':')[0]), 0, 0) for x in l.split()[2:])))
        elif l.startswith('SYNCOBJ_TIMELINE_SIGNAL'):
            ev.append(('HOSTSIG', named((int(x.split(':')[0]), int(x.split(':')[1]), 0) for x in l.split()[2:])))
        elif l.startswith('SYNCOBJ_TRANSFER'):
            d = dict(re.findall(r'(\w+)=(\S+)', l))
            h, p = d['dst'].split(':')
            ev.append(('TRANSFER', named([(int(h), int(p), 0)])))
        elif l.startswith('MARK delayed'):
            ev.append(('DELAYED',))
        elif l.startswith('IOCTL_FAIL'):
            ev.append(('FAIL', l))
    return ev


results = {}
for l in open(out):
    if l.startswith('RESULT '):
        p = l.split()
        results[p[1]] = dict(kv.split('=') for kv in p[2:])
done = any(l.startswith('DONE') for l in open(out))

known = mode == 'known'
# expected CS sequence per scenario: (waits, signals) with named semaphores; expected probes (probe mode)
EXP = {
    'xq': ([([], [('xq_B', 0), ('xq_T', 1)]), ([('xq_T', 1)], [('xq_U', 1)]),
            ([('xq_B', 0), ('xq_U', 1)], [('xq_T', 2)]), ([('xq_T', 2)], [('xq_U', 2)])], 3),
    'serial': ([([], [('serial_B', 0), ('serial_T', 1)])] +
               [([('serial_B', 0)], [('serial_B', 0), ('serial_T', i + 1)]) for i in range(1, 6)], 5),
    'consume': ([([], [('consume_B', 0)]), ([('consume_B', 0)], [('consume_T', 1)]),
                 ([], [('consume_B', 0)]), ([('consume_B', 0)], [('consume_T', 2)])], 2),
    'hostsig': ([([('hostsig_T', 3), ('hostsig_U', 7)], [('hostsig_T', 4)])], 1),
    'empty': ([([('empty_T', 1)], [('empty_T', 2)])], 1),
    'secondary': ([([], [('secondary_T', 1)])], 0),
    'multi': ([([], [('multi_T', 1)]), ([('multi_T', 1)], [('multi_T', 2)]), ([], [('multi_T', 3)])], 1),
    'sparse': ([([], [('sparse_T', 1)]), ([('sparse_T', 2)], [('sparse_T', 3)])], 1),
    'wbs_host': ([([('wbs_host_T', 5)], [('wbs_host_U', 1)])], 1),
    'wbs_queue': ([([], [('wbs_queue_T', 1)]), ([('wbs_queue_T', 1)], [('wbs_queue_U', 1)]),
                   ([('wbs_queue_U', 1)], [('wbs_queue_U', 2)])], 1),
}
WBS = {'wbs_host', 'wbs_queue'}

ok = done
if not done:
    print('semtest did not finish')
for name, (cs_exp, probes_default) in EXP.items():
    if name == 'sparse' and results.get('sparse', {}).get('skipped') == '1':
        print(f'{name:10s} status=skipped (no sparse queue)')
        continue
    if name not in scen:
        print(f'{name:10s} status=FAIL missing')
        ok = False
        continue
    ev = parse(scen[name])
    cs = [e for e in ev if e[0] == 'CS']
    probes = [e for e in ev if e[0] == 'PROBE']
    fails = [e for e in ev if e[0] == 'FAIL']
    errs = []
    got = [(w, s) for _, w, s, _, _, _ in cs]
    if got != cs_exp:
        errs.append(f'cs={got} expected={cs_exp}')
    if any(f != 2 for e in cs for f in e[3]):
        errs.append('wait flags != WAIT_FOR_SUBMIT')
    if bos_below is not None and any(e[4] >= bos_below for e in cs):
        errs.append(f'bos={[e[4] for e in cs]} not below {bos_below}')
    want_probes = probes_default if (not known or name in WBS) else 0
    if len(probes) != want_probes:
        errs.append(f'probes={len(probes)} expected={want_probes}')
    # expected ioctl failures: only the wait-before-signal probe (-ETIME)
    exp_fails = 1 if name in WBS else 0
    if len(fails) != exp_fails or any('errno=62' not in f[1] for f in fails):
        errs.append(f'ioctl failures {fails}')
    if name == 'consume':
        resets = [e for e in ev if e[0] == 'RESET']
        if len(resets) != 2:
            errs.append(f'resets={len(resets)} expected=2')
    if name == 'serial' and any(e[0] == 'RESET' for e in ev):
        errs.append('serial: reset of a re-signalled binary')
    if name in WBS:
        # the waiting submission reaches the kernel only after the signal (delayed host / queue signal)
        order = [e[0] for e in ev]
        d = order.index('DELAYED') if 'DELAYED' in order else -1
        wait_cs = [i for i, e in enumerate(ev) if e[0] == 'CS' and e[1] == cs_exp[1 if name == 'wbs_queue' else 0][0]]
        if d < 0 or not wait_cs or wait_cs[0] < d:
            errs.append('wait-before-signal: waiting CS before the signal')
        if float(results.get(name, {}).get('submit_ms', '1e9')) > 50:
            errs.append(f'vkQueueSubmit blocked {results.get(name)}')
    r = results.get(name, {})
    if not r or any(v == '0' for k, v in r.items() if k.startswith('host_') or k == 'fence'):
        errs.append(f'host result {r}')
    status = 'ok' if not errs else 'FAIL'
    ok &= not errs
    print(f'{name:10s} status={status} cs={len(cs)} probes={len(probes)} bos={[e[4] for e in cs]} {"; ".join(errs)}')
print('checksem ok' if ok else 'checksem FAIL')
sys.exit(0 if ok else 1)
