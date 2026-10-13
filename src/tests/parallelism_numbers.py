"""Numbers test for the parallelism lab: every number the lab shows, recomputed here in plain Python from the lab's stated
assumptions (specs/parallelism.md), then compared with the page's engine through its read-only hook window.__zr.

- parameter counts of Llama-3.1-8B / 70B from their shapes (sku_list.py, the FeedForward rule): 8,030,261,248 and 70,553,706,496;
- model states, 18 bytes per parameter (Megatron-LM dist_optimizer.md, HF model_memory_anatomy.md): the integrator's numbers
  8B on 8 GPUs 144.5 / 60.2 / 32.1 / 18.1 GB; 70B on 64 GPUs ZeRO-2 158.7 GB, ZeRO-3 19.8 GB; Megatron's 6 + 12/d;
- activations: the recomputed layer counted tensor by tensor (167,936 bytes per token for 8B = 41h); logits in FP32, tokens × V × 4,
  as training-memory counts them (integrator decision 10, review S5);
- bytes sent per GPU per step (ring factors from nccl-tests PERFORMANCE.md; 1.5 × for ZeRO-3, ml-engineering);
- transfer time by integrator decision 8: inside a machine bytes ÷ 450 GB/s; a group across machines moves the whole array over each
  machine's network (400, 50 or 12.5 GB/s per machine), an all-gather or reduce-scatter once, an all-reduce twice;
- compute by decisions 1 and 9: 8N FLOPs per token (full recomputation) at 50% of 989.5 TFLOP/s by default; forward 1/4, backward 3/4;
  MFU = 6N × tokens ÷ (step × peak), at most 37.5% with no waiting;
- the two-stream schedule, re-implemented from the spec, for every guided part, scenario and challenge option;
- the challenge (8 machines with one 400 Gb/s NIC each): which of the 24 options pass, and the minimal one (ZeRO-3, 8,192 tokens);
  the one change that helps most (the owner's review of 2026-10-07, R1), recomputed here for every one of the 24 starting points;
- the minimal implementation printed on the page, run here, prints what its last comment says;
- the home tile (13B, 18 bytes, ZeRO-3) and the interview drill.
Run: python3 tests/parallelism_numbers.py"""
import json, pathlib, subprocess, sys, re, io, contextlib
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/parallelism.html', '--only', 'parallelism'], check=True, cwd=P, stdout=subprocess.DEVNULL)
OUT = P / 'out' / 'parallelism'; OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'wrapped-num.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>' + (P / 'out' / 'parallelism.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-num.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok: fails.append(name); print('FAIL', name, '::', info)


close = lambda a, b, r=1e-9: abs(a - b) <= r * max(1.0, abs(a), abs(b))
g1 = lambda b: round(b / 1e9 + 1e-12, 1)

# ---- shapes -> parameters
def params(h, L, nh, nkv, i, V=128256, hd=128):
    attn = h * nh * hd + 2 * h * nkv * hd + nh * hd * h
    return V * h * 2 + L * (attn + 3 * h * i + 2 * h) + h
M = {'8b': dict(P=params(4096, 32, 32, 8, 14336), L=32, h=4096, i=14336, kv=1024, V=128256), '70b': dict(P=params(8192, 80, 64, 8, 28672), L=80, h=8192, i=28672, kv=1024, V=128256)}
check('8B parameters from shapes', M['8b']['P'] == 8030261248, M['8b']['P'])
check('70B parameters from shapes', M['70b']['P'] == 70553706496, M['70b']['P'])
PEAK, CAP, GPN, BWIN = 989.5e12, 80e9, 8, 450e9
NET = {'ib': 8 * 400e9 / 8, 'one': 400e9 / 8, 'eth': 100e9 / 8}      # bytes/s per machine: 8 NICs of 400 Gb/s, one, 100 Gb/s Ethernet
ring = lambda k: (k - 1) / k


def mem(m, N, st, T, ov=1, **_):
    x = M[m]; Pp = x['P']; g = min(N, GPN)
    sw, sg, so = {'dp': (1, 1, 1), 'z1': (1, 1, N), 'z2': (1, N, N), 'z3': (N, N, N), 'hsdp': (g, g, g)}[st]
    states = 2 * Pp / sw + 4 * Pp / sg + 12 * Pp / so
    inf = (2 if ov else 1) * 2 * Pp / x['L'] if st in ('z3', 'hsdp') else 0
    aL = 2 * (6 * x['h'] + 2 * x['kv']) + 8 * x['i']
    act = T * (2 * x['h'] * x['L'] + aL)
    lg = T * x['V'] * 4
    return dict(states=states, inf=inf, act=act, lg=lg, aL=aL, total=states + inf + act + lg)


def comm(m, N, st, net='ib', **_):
    x = M[m]; lay = 2 * x['P'] / x['L']; one = N <= GPN; g = min(N, GPN); R = max(1, N // GPN)
    ag = agb = rs = ar = arx = 0.0
    if st == 'dp': ar = 2 * lay * ring(N)
    elif st in ('z1', 'z2'): ag = rs = lay * ring(N)
    elif st == 'z3': ag = agb = rs = lay * ring(N)
    else: ag = agb = rs = lay * ring(g); arx = 2 * (lay / g) * ring(R) if R > 1 else 0
    if one or st == 'hsdp':      # inside one machine: each GPU's own bytes over NVLink
        t = dict(ag=ag / BWIN, agb=agb / BWIN, rs=rs / BWIN, ar=ar / BWIN, arx=0.0)
    else:                        # a group across machines: the whole layer over each machine's network, all-reduce twice
        W = NET[net]; t = dict(ag=lay / W if ag else 0, agb=lay / W if agb else 0, rs=lay / W if rs else 0, ar=2 * lay / W if ar else 0, arx=0.0)
    if arx: t['arx'] = 2 * lay / NET[net]
    return dict(t=t, bytes=x['L'] * (ag + agb + rs + ar + arx), dp=2 * 2 * x['P'] * ring(N), time=x['L'] * sum(t.values()))


def sim(m, N, st, T, ov=1, net='ib', u=50):
    """one compute stream and one transfer stream, per layer, as specs/parallelism.md states it"""
    x = M[m]; Lr = x['L']; C = 8 * x['P'] * T / (PEAK * u / 100); tF, tB = C / 4 / Lr, 3 * C / 4 / Lr; t = comm(m, N, st, net)['t']
    fwdAG, bwdAG = st != 'dp', st in ('z3', 'hsdp')
    red = lambda s: s + (t['ar'] if st == 'dp' else t['rs'] + t['arx'])
    tc = tn = 0.0; doneB = {}
    if not fwdAG: tc = Lr * tF
    elif ov:
        tn = t['ag']; done = {0: tn}
        for i in range(Lr):
            s = max(tc, done[i])
            if i + 1 < Lr: tn = max(tn, s) + t['ag']; done[i + 1] = tn
            elif bwdAG: tn = max(tn, s) + t['agb']; doneB[Lr - 1] = tn
            tc = s + tF
    else:
        for i in range(Lr):
            s = max(tc, tn) + t['ag']; tc = s + tF; tn = tc
    if ov:
        for i in range(Lr - 1, -1, -1):
            s = max(tc, doneB[i]) if bwdAG else tc
            if bwdAG and i > 0: tn = max(tn, s) + t['agb']; doneB[i - 1] = tn
            tc = s + tB; tn = red(max(tn, tc))
    else:
        for i in range(Lr - 1, -1, -1):
            s = max(tc, tn)
            if bwdAG: s += t['agb']
            tc = s + tB; tc = tn = red(tc)
    step = max(tc, tn)
    return dict(C=C, step=step, share=(step - C) / step, mfu=6 * x['P'] * T / (step * PEAK))


# ---- the integrator's numbers (18 bytes per parameter)
for st, v in (('dp', 144.5), ('z1', 60.2), ('z2', 32.1), ('z3', 18.1)):
    check(f'8B on 8 GPUs, {st}: {v} GB of model states', g1(mem('8b', 8, st, 4096)['states']) == v, g1(mem('8b', 8, st, 4096)['states']))
check('70B on 64, ZeRO-2: 158.7 GB', g1(mem('70b', 64, 'z2', 4096)['states']) == 158.7)
check('70B on 64, ZeRO-3: 19.8 GB', g1(mem('70b', 64, 'z3', 4096)['states']) == 19.8)
check('70B on one GPU: 1.27 TB', round(18 * M['70b']['P'] / 1e12, 2) == 1.27)
check('Megatron 6 + 12/d at d = 8', close(mem('8b', 8, 'z1', 0)['states'], (6 + 12 / 8) * M['8b']['P']))
check('8B recomputed layer = 167,936 bytes per token = 41h', mem('8b', 8, 'dp', 1)['aL'] == 167936 == 41 * 4096)
check('70B recomputed layer = 40.5h', mem('70b', 8, 'dp', 1)['aL'] == 40.5 * 8192)
check('ZeRO-3 moves 1.5 times DP', close(comm('8b', 8, 'z3')['bytes'], 1.5 * comm('8b', 8, 'dp')['bytes']))
check('ZeRO-1 moves the same as DP', close(comm('8b', 8, 'z1')['bytes'], comm('8b', 8, 'dp')['bytes']))

CASES = [dict(m='8b', N=8, st=s, T=4096) for s in ('dp', 'z1', 'z2', 'z3')] + [dict(m='8b', N=8, st='z3', T=4096, ov=0),
         dict(m='8b', N=64, st='z3', T=4096), dict(m='8b', N=64, st='z3', T=16384), dict(m='8b', N=64, st='hsdp', T=4096),
         dict(m='8b', N=8, st='z1', T=4096), dict(m='8b', N=8, st='z3', T=131072), dict(m='8b', N=8, st='z2', T=131072),
         dict(m='8b', N=32, st='z1', T=4096, net='eth'), dict(m='8b', N=32, st='z1', T=4096), dict(m='8b', N=16, st='z2', T=4096),
         dict(m='8b', N=64, st='dp', T=4096), dict(m='8b', N=16, st='hsdp', T=2048, ov=0), dict(m='70b', N=8, st='z1', T=1024, u=35),
         dict(m='8b', N=64, st='z3', T=4096, net='one'), dict(m='8b', N=64, st='z3', T=16384, net='one'), dict(m='8b', N=64, st='hsdp', T=4096, net='one'),
         dict(m='8b', N=64, st='z1', T=4096, net='one', ov=0), dict(m='70b', N=64, st='z3', T=8192, net='one'), dict(m='70b', N=16, st='dp', T=2048, net='eth', u=80)]
CH_ST, CH_T = ['z1', 'z2', 'z3', 'hsdp'], [2048, 4096, 8192, 16384, 32768, 65536]
EXPECT = {('z3', 8192), ('z3', 16384)}

with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(); errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300); pg.evaluate("()=>{location.hash='parallelism'}"); pg.wait_for_timeout(400)
    K = pg.evaluate("()=>window.__zr.consts")
    check('engine parameters match the shapes', K['MOD']['8b']['P'] == M['8b']['P'] and K['MOD']['70b']['P'] == M['70b']['P'])
    check('constants', K['PEAK'] == PEAK and K['CAP'] == CAP and K['BW_IN'] == BWIN and all(close(K['NET'][k], NET[k]) for k in NET) and K['BASE']['u'] == 50 and K['CH']['u'] == 50 and K['CH']['net'] == 'one' and K['P6']['net'] == 'one', (K['NET'], K['BASE'], K['CH']))
    check('every scenario assumes 50% while computing', all(p['u'] == 50 for p in K['PRESETS']), K['PRESETS'])
    for c in CASES:
        o = {'ov': 1, 'net': 'ib', 'u': 50, **c}
        me, js = mem(**o), pg.evaluate("o=>window.__zr.mem(o)", o)
        check(f'memory {o}', close(me['total'], js['total']) and close(me['states'], js['states']) and close(me['act'], js['act']) and close(me['inf'], js['inf']) and close(me['lg'], js['lg']), (me, js))
        cm, jc = comm(**o), pg.evaluate("o=>window.__zr.comm(o)", o)
        check(f'bytes {o}', close(cm['bytes'], jc['bytes']) and close(cm['dp'], jc['dpBytes']) and close(cm['time'], jc['time']), (cm['bytes'], jc['bytes'], cm['time'], jc['time']))
        sm, js2 = sim(o['m'], o['N'], o['st'], o['T'], o['ov'], o['net'], o['u']), pg.evaluate("o=>window.__zr.sim(o)", o)
        check(f'schedule {o}', close(sm['step'], js2['step'], 1e-9) and close(sm['share'], js2['share'], 1e-7) and close(sm['mfu'], js2['mfu'], 1e-9), (sm, js2))
    # the guided parts' stated results
    sh = lambda **o: sim(o['m'], o['N'], o['st'], o['T'], o.get('ov', 1), o.get('net', 'ib'))['share']
    gb = lambda **o: g1(mem(**{'ov': 1, **o})['total'])
    check('part 2: DP 148.4 GB with logits', gb(m='8b', N=8, st='dp', T=4096) == 148.4, gb(m='8b', N=8, st='dp', T=4096))
    check('part 3: ZeRO-1 64.1 GB', gb(m='8b', N=8, st='z1', T=4096) == 64.1, gb(m='8b', N=8, st='z1', T=4096))
    check('part 4: ZeRO-2 36.0 GB', gb(m='8b', N=8, st='z2', T=4096) == 36.0, gb(m='8b', N=8, st='z2', T=4096))
    check('part 5: ZeRO-3 22.4 GB without prefetch, 22.9 GB with it', gb(m='8b', N=8, st='z3', T=4096, ov=0) == 22.4 and gb(m='8b', N=8, st='z3', T=4096) == 22.9)
    check('logits at 4,096 tokens: 2.1 GB', g1(mem('8b', 8, 'dp', 4096)['lg']) == 2.1)
    check('part 2: DP waits under 1%', sh(m='8b', N=8, st='dp', T=4096) < 0.01)
    check('part 5: fetch when needed waits 15%', round(100 * sh(m='8b', N=8, st='z3', T=4096, ov=0)) == 15, sh(m='8b', N=8, st='z3', T=4096, ov=0))
    check('part 5: prefetch waits under 1%', sh(m='8b', N=8, st='z3', T=4096) < 0.01)
    check('part 6: ZeRO-3 over 64 GPUs, one NIC per machine, waits 45%', round(100 * sh(m='8b', N=64, st='z3', T=4096, net='one')) == 45, sh(m='8b', N=64, st='z3', T=4096, net='one'))
    check('part 6: 16,384 tokens wait under 1%', sh(m='8b', N=64, st='z3', T=16384, net='one') < 0.01, sh(m='8b', N=64, st='z3', T=16384, net='one'))
    check('part 6: a NIC per GPU (400 GB/s per machine) waits under 1%', sh(m='8b', N=64, st='z3', T=4096, net='ib') < 0.01)
    check('scenario 3: HSDP waits under 1%', sh(m='8b', N=64, st='hsdp', T=4096) < 0.01)
    check('scenario 3: HSDP moves a third less over the network than ZeRO-3', close(comm('8b', 64, 'hsdp', 'ib')['time'] - 32 * 3 * (2 * M['8b']['P'] / 32) * ring(8) / BWIN, 2 / 3 * comm('8b', 64, 'z3', 'ib')['time']))
    check('scenario 4: Ethernet ZeRO-1 waits 79%', round(100 * sh(m='8b', N=32, st='z1', T=4096, net='eth')) == 79, sh(m='8b', N=32, st='z1', T=4096, net='eth'))
    check('scenario 4: the same run at 400 GB/s per machine hardly waits (under 1%)', sh(m='8b', N=32, st='z1', T=4096, net='ib') < 0.01, sh(m='8b', N=32, st='z1', T=4096, net='ib'))
    check('scenario 2: 128K tokens, ZeRO-3, 142.7 GB, does not fit', gb(m='8b', N=8, st='z3', T=131072) == 142.7 and mem('8b', 8, 'z3', 131072)['total'] > CAP)
    check('scenario 2: logits at 128K are 67.2 GB', g1(mem('8b', 8, 'z3', 131072)['lg']) == 67.2)
    s0 = sim('8b', 8, 'dp', 4096)
    check('MFU = 3/4 of 50% (37.5%) times the share of the step spent computing; 37.4% for data parallel on 8 GPUs', close(0.375 * s0['C'] / s0['step'], s0['mfu']) and round(100 * s0['mfu'], 1) == 37.4 and s0['mfu'] <= 0.375, s0)
    # the challenge
    rows = pg.evaluate("()=>window.__zr.chAll()")
    check('challenge has 24 options', len(rows) == 24)
    for r in rows:
        me = mem('70b', 64, r['st'], r['T']); s = sim('70b', 64, r['st'], r['T'], 1, 'one')
        ok = me['total'] <= CAP and s['share'] <= 0.10 + 1e-12
        check(f"challenge {r['st']} {r['T']}", ok == r['ok'] and close(me['total'], r['total']) and close(s['step'], r['step']), (ok, r))
    passing = {(r['st'], r['T']) for r in rows if r['ok']}
    check('challenge: the passing options', passing == EXPECT, passing)
    best = pg.evaluate("()=>window.__zr.chBest()")
    check('challenge: minimal is ZeRO-3 at 8,192 tokens', (best['st'], best['T']) == ('z3', 8192), best)
    # the one change that helps most (R1): of the options one setting away, the passing one with the shortest step; if none passes
    # and yours does not fit, the one-setting change that fits with the least waiting; otherwise none
    ev = {(st, T): (mem('70b', 64, st, T)['total'] <= CAP, sim('70b', 64, st, T, 1, 'one')) for st in CH_ST for T in CH_T}
    okk = lambda k: ev[k][0] and ev[k][1]['share'] <= 0.10 + 1e-12
    def one(st, T):
        nb = [k for k in ev if (k[0] == st) != (k[1] == T)]
        ps = sorted([k for k in nb if okk(k)], key=lambda k: ev[k][1]['step'])
        if ps: return {'st': ps[0][0], 'T': ps[0][1], 'pass': True}
        fs = sorted([k for k in nb if ev[k][0]], key=lambda k: ev[k][1]['share'])
        return {'st': fs[0][0], 'T': fs[0][1], 'pass': False} if not ev[(st, T)][0] and fs else None
    for k in ev:
        js = pg.evaluate("a=>window.__zr.chOne(a[0],a[1])", list(k))
        check(f'challenge: the one change from {k}', js == one(*k), (js, one(*k)))
    check('challenge: from the start (ZeRO-2, 4,096) no single change passes; ZeRO-3 alone fits but waits', one('z2', 4096) == {'st': 'z3', 'T': 4096, 'pass': False})
    check('challenge: from ZeRO-3 at 4,096, raising the tokens to 8,192 passes', one('z3', 4096) == {'st': 'z3', 'T': 8192, 'pass': True})
    check('challenge: from HSDP at 65,536, no single change even fits', one('hsdp', 65536) is None)
    sc = lambda st, T: sim('70b', 64, st, T, 1, 'one')
    check('challenge start does not pass: ZeRO-2 at 4,096, 167.6 GB', not mem('70b', 64, 'z2', 4096)['total'] <= CAP and g1(mem('70b', 64, 'z2', 4096)['total']) == 167.6)
    check('challenge minimal: 41.0 GB, 9.90 s, 5.6% waiting', g1(mem('70b', 64, 'z3', 8192)['total']) == 41.0 and round(sc('z3', 8192)['step'], 2) == 9.90 and round(100 * sc('z3', 8192)['share'], 1) == 5.6, (mem('70b', 64, 'z3', 8192)['total'], sc('z3', 8192)))
    check('challenge reveal: ZeRO-3 at 4,096 waits 45%', round(100 * sc('z3', 4096)['share']) == 45, sc('z3', 4096)['share'])
    ct = comm('70b', 64, 'z3', 'one')['time']
    check('challenge reveal: from 8,192 tokens the computation is longer than the transfers (9.3 s > 8.5 s); at 4,096 it is shorter', sc('z3', 8192)['C'] > ct > sc('z3', 4096)['C'] and round(sc('z3', 8192)['C'], 1) == 9.3 and round(ct, 1) == 8.5, (sc('z3', 4096)['C'], ct, sc('z3', 8192)['C']))
    check('challenge reveal: ZeRO-3 at 32,768 does not fit (94.0 GB)', g1(mem('70b', 64, 'z3', 32768)['total']) == 94.0)
    check('challenge reveal: ZeRO-2 at 8,192 176.4 GB, HSDP at 8,192 179.9 GB', g1(mem('70b', 64, 'z2', 8192)['total']) == 176.4 and g1(mem('70b', 64, 'hsdp', 8192)['total']) == 179.9)
    # the minimal implementation on the page prints what its comment says
    code = pg.evaluate("()=>document.querySelector('#zr-code').textContent")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): exec(code, {})
    printed = ' / '.join(l.strip() for l in buf.getvalue().strip().splitlines())
    check('minimal implementation prints its comment', printed == code.strip().splitlines()[-1].lstrip('# ').strip(), printed)
    # the transfer question
    ex = pg.evaluate("()=>window.__zr.exits()")
    check('transfer Q1 answer: about 24 GB', ex['transferA'][0] == 1 and round(mem('8b', 16, 'z2', 0)['states'] / 1e9) == 24 and round(mem('8b', 8, 'z2', 0)['states'] / 1e9) == 32)
    check('transfer Q2: DP bytes 8 vs 64 GPUs', round(comm('8b', 8, 'dp')['bytes'] / 1e9, 1) == 28.1 and round(comm('8b', 64, 'dp')['bytes'] / 1e9, 1) == 31.6)
    # the drill
    for seed in range(1, 13):
        d = pg.evaluate("s=>window.__zr.drill(s)", seed); Pp = d['Pb'] * 1e9; N = d['N']
        want = {'z1': 6 * Pp + 12 * Pp / N, 'z2': 2 * Pp + 16 * Pp / N, 'z3': 18 * Pp / N}[d['st']]
        check(f'drill {seed}', close(d['states'], want) and close(d['bytes'], (3 if d['st'] == 'z3' else 2) * 2 * Pp * (N - 1) / N) and close(d['dp'], 18 * Pp))
    # the home tile
    for k in (1, 2, 4, 8, 16, 32, 64):
        v = pg.evaluate("k=>window.__tzZr.per(k)", k)
        check(f'tile {k} GPUs', close(v, 18 * 13e9 / k))
    check('tile 4 GPUs reads 58.5 GB', '58.5 GB' in pg.evaluate("()=>window.__tzZr.read(4)"))
    # ---- G7 (the owner's message of 2026-10-08, from the model-parallel finding): walk every pick of the challenge, in both
    # languages, and check that a recommended single change makes the value the result shows strictly better at the precision shown
    def g7_walk(js):
        res = {}
        for lang in ('en', 'zh'):
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(150)
            res[lang] = pg.evaluate(js)
        pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(150)
        return res
    g7 = g7_walk("""()=>{const Z=window.__zr, C=Z.consts.CH; const R=/^The one change that helps most is|^只改一项的话/; const f=s=>/under|不到/.test(s)?0.5:parseFloat(String(s).replace(/,/g,'')); const out=[];
      C.sts.forEach(st=>C.Ts.forEach(T=>{const t=Z.chNextText(st,T), s=Z.chShown(st,T), o=Z.chOne(st,T), rec=R.test(t); let ok=true, why='';
        if(rec){if(!o||!o.pass){ok=false; why='no passing engine change';} else {const r=Z.chShown(o.st,o.T);
          if(!s.fits){ok=r.fits&&f(r.gb)<f(s.gb); why='GB '+s.gb+' -> '+r.gb;}
          else if(!s.ok){ok=r.ok&&f(r.wait)<f(s.wait); why='waiting '+s.wait+' -> '+r.wait;}
          else {ok=f(r.step)<f(s.step); why='step '+s.step+' -> '+r.step;}}}
        if(/about 0 s|约 0 秒/.test(t)){ok=false; why+=' (a zero gain)';}
        out.push({pick:[st,T],rec,ok,why,t});})); return out;}""")
    for lang in ('en', 'zh'):
        rows = g7[lang]; bad = [r for r in rows if not r['ok']]
        check(f'G7 {lang}: all 24 settings walked; every recommended single change shows a strictly better value (memory, waiting or step time)', len(rows) == 24 and not bad, bad[:3])
    check('G7: when no single change meets the goal, the sentence names what the best setting changes together',
          all(any(r['t'].startswith(Z_) and ('ZeRO-3' in r['t']) and ('8,192' in r['t']) for r in g7[l]) for l, Z_ in (('en', 'No single change meets the goal. The best setting changes both at once'), ('zh', '只改一项还达不到目标。最好的设置同时改了两项'))), [r['t'] for r in g7['zh'] if not r['rec']][:3])
    check('no page errors', not errs, errs)
    b.close()
print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
