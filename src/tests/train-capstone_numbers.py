"""Numbers test for the train-capstone lab (CONTRACT §6): every number the page shows is recomputed here in plain Python from the
lab's stated assumptions, then compared with the page's engine (window.__tcap) and the teaser (window.__tcapTz).

Assumptions (the lab's Fidelity popover and specs/train-capstone.md):
- work 6·N·D; a step does k·N FLOPs per token, k = 6 (8 with full recomputation), at 50% of 989.5 TFLOP/s while computing;
- memory per GPU: 18 bytes per parameter ÷ (t·p), by ZeRO stage over d copies; ZeRO-3 also holds one gathered layer (4N/(L·t));
  activations s·b·h·(18 + 4I/h) ÷ t per layer (Megatron-LM, sequence parallel), min(m, p) micro-batches in flight (1F1B);
  recomputation keeps 2·s·b·h ÷ t per layer plus one full layer; GB = 10⁹ bytes; 80 GB;
- step = (m + p − 1)·(compute + n·L/p all-reduces of s·b·h·2 bytes, ring 2(t − 1)/t over 450 GB/s) + exposed data-parallel time;
  data parallel in BF16, counted in passes (an all-gather or a reduce-scatter; an all-reduce is two): ZeRO-0/1 2 passes hidden by
  one backward (2/3 of a slot, 3/4 with recomputation), ZeRO-2 m + 1 passes hidden by m backwards, ZeRO-3 3m passes hidden by m
  slots. One pass (integrator decision 8, the scaling book's cross-node model): when the group spans machines (t·d > 8), the
  stage's whole BF16 array 2N/p through one machine's network, 400 GB/s (8 × 400 Gb/s) or 12.5 GB/s on Ethernet (100 Gb/s);
  inside one machine (d − 1)/d·2N/(t·p) over NVLink;
- failures: one per machine every 64 days; M = 64·24 h ÷ (G/8); checkpoint 14 bytes per parameter written at 10 GB/s; an
  asynchronous save stops training for 10 s; restart 10 min (integrator decision 11); waste = δ/τ + (τ/2 + R)/M;
  days = steps·step ÷ (1 − waste).
Run: python3 tests/train-capstone_numbers.py"""
import itertools, math, pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/train-capstone.html', '--only', 'train-capstone'], check=True, cwd=P, stdout=subprocess.DEVNULL)
OUT = P / 'out' / 'train-capstone'; OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'wrapped-num.html').write_text('<!doctype html><html><head><meta charset="utf-8"></head><body>' + (P / 'out' / 'train-capstone.html').read_text() + '</body></html>')

PEAK, MEM, NV, NODE, ETH, U0, SEQ, VOCAB = 989.5e12, 80e9, 450e9, 400e9, 100e9 / 8, 0.5, 4096, 128256
CK_BPP, CK_W, ASYNC_S, R_H = 14, 10e9, 10, 10 / 60
MODELS = {'8b': (8030261248, 32, 4096, 14336), '70b': (70553706496, 80, 8192, 28672), '405b': (405853388800, 126, 16384, 53248)}
BASE = dict(model='70b', D=15e12, G=1024, t=8, p=8, z=1, rc=0, b=1, B=1024, net='ib', ivl=3, asy=0, node=64)
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', info)


def close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def plan(**kw):
    c = {**BASE, **kw}
    N, L, h, I = MODELS[c['model']]
    t, p, G, b, B, z = c['t'], c['p'], c['G'], c['b'], c['B'], c['z']
    if G % (t * p): return None
    d = G // (t * p)
    if B % (d * b): return None
    m = B // (d * b)
    per_gpu = N / (t * p)
    weights = 2 * per_gpu / (d if z == 3 else 1)
    grads = 4 * per_gpu / (d if z >= 2 else 1)
    optim = 12 * per_gpu / (d if z >= 1 else 1)
    gathered = 4 * N / (L * t) if z == 3 else 0.0
    tokens = b * SEQ
    layer_full = tokens * h * (18 + 4 * I / h) / t
    layer_kept = 2 * tokens * h / t
    layers = L / p
    in_flight = min(m, p)
    act = in_flight * layers * layer_kept + layer_full if c['rc'] else in_flight * layers * layer_full
    mem = weights + grads + optim + gathered + act
    k = 8 if c['rc'] else 6
    compute = k * (N / p) * tokens / t / (PEAK * U0)
    one_ar = 2 * (t - 1) / t * (tokens * h * 2) / NV if t > 1 else 0.0
    tp_time = (6 if c['rc'] else 4) * layers * one_ar
    slot = compute + tp_time
    backward = slot * (3 / 4 if c['rc'] else 2 / 3)
    if t * d > 8: one_pass = 2 * N / p / (ETH if c['net'] == 'eth' else NODE)   # the stage's whole array through one machine's network
    else: one_pass = (d - 1) / d * 2 * per_gpu / NV                                # each GPU's slice over NVLink
    if z <= 1: passes, cover = 2, backward
    elif z == 2: passes, cover = m + 1, m * backward
    else: passes, cover = 3 * m, m * slot
    dp_wait = max(0.0, (passes * one_pass if d > 1 else 0.0) - cover)
    step = (m + p - 1) * slot + dp_wait
    btok = B * SEQ
    mfu = 6 * N * btok / (G * PEAK * step)
    train_s = c['D'] / btok * step
    delta_sync = CK_BPP * N / CK_W
    delta = ASYNC_S if c['asy'] else delta_sync
    M = c['node'] * 24 / (G / 8)
    w_save, w_redo, w_restart = delta / 3600 / c['ivl'], c['ivl'] / 2 / M, R_H / M
    waste = w_save + w_redo + w_restart
    wall = train_s / (1 - waste) if waste < 0.5 else float('nan')
    return dict(d=d, m=m, w=weights, g=grads, o=optim, zx=gathered, act=act, mem=mem, fits=mem <= MEM, step=step, mfu=mfu,
                useful=m * compute * 6 / k, recomp=m * compute * (k - 6) / k, tp=m * tp_time, bubble=(p - 1) * slot, dp=dp_wait,
                dSync=delta_sync, dS=delta, M=M, wSave=w_save, wRedo=w_redo, wRestart=w_restart, waste=waste, valid=waste < 0.5,
                days=wall / 86400, gpuh=G * wall / 3600, good=1 - waste, train=train_s, logits=SEQ * b * VOCAB * 4 / t)


def simple_days(u, G=1024, D=15e12, model='70b'):
    return 6 * MODELS[model][0] * D / (G * PEAK * u) / 86400


with sync_playwright() as pw:
    br = pw.chromium.launch(); pg = br.new_page()
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto('file://' + str(OUT / 'wrapped-num.html')); pg.wait_for_timeout(300)
    js = lambda expr, arg=None: pg.evaluate(expr, arg)
    K = js("()=>window.__tcap.consts")
    # ---- constants agree with the stated assumptions
    check('constants: peak, memory, NVLink 450 GB/s, 400 GB/s per machine, Ethernet 12.5 GB/s per machine, U0, sequence',
          (K['GPU']['peak'], K['GPU']['mem'], K['GPU']['nv'], K['GPU']['node'], K['GPU']['eth'], K['U0'], K['SEQ']) == (PEAK, MEM, NV, NODE, ETH, U0, SEQ), K['GPU'])
    check('constants: checkpoint 14 B/param at 10 GB/s, async stops 10 s, restart 10 min (decision 11)', (K['CK']['bpp'], K['CK']['write'], K['CK']['asyncS']) == (14, 10e9, 10) and close(K['CK']['restartH'], 1 / 6), K['CK'])
    check('constants: model shapes (Llama 3.1 config.json)', all((K['MD'][k]['N'], K['MD'][k]['L'], K['MD'][k]['h'], K['MD'][k]['I']) == v for k, v in MODELS.items()), K['MD'])
    check('constants: seven parts', K['STEP_IDS'] == ['mission', 'total', 'memory', 'time', 'failures', 'real', 'remember'], K['STEP_IDS'])
    # ---- the engine against this file over a grid of plans
    keys = ['w', 'g', 'o', 'zx', 'act', 'mem', 'step', 'mfu', 'dSync', 'dS', 'M', 'wSave', 'wRedo', 'wRestart', 'waste', 'days', 'gpuh', 'logits']
    grid = []
    for model, G, t, p, z, rc, b, net, ivl, asy, node in itertools.product(['8b', '70b', '405b'], [64, 1024, 16384], [1, 8], [1, 8, 16], [0, 1, 2, 3], [0, 1], [1, 4], ['ib', 'eth'], [0.25, 3], [0, 1], [32, 64]):
        grid.append(dict(model=model, G=G, t=t, p=p, z=z, rc=rc, b=b, net=net, ivl=ivl, asy=asy, node=node))
    got = js("cs=>cs.map(c=>window.__tcap.plan(c))", grid)
    bad = []
    for c, g in zip(grid, got):
        r = plan(**c)
        if (r is None) != (g is None): bad.append((c, 'validity')); continue
        if r is None: continue
        for k in keys:
            a, b2 = r[k], g[k] if g[k] is not None else float('nan')
            if not ((math.isnan(a) and math.isnan(b2)) or close(a, b2, 1e-9)): bad.append((c, k, a, b2)); break
        for k in ('useful', 'recomp', 'tp', 'bubble', 'dp'):
            if not close(r[k], g['seg'][k], 1e-9): bad.append((c, k, r[k], g['seg'][k])); break
    check(f'engine = this file on {len(grid)} plans (memory, step, MFU, failures, days)', not bad, bad[:3])
    # ---- part 1–2: the mission and the first estimate
    r = plan()
    check('mission: d = 16, m = 64', (r['d'], r['m']) == (16, 64), (r['d'], r['m']))
    check('mission memory: 2.20 + 4.41 + 0.83 + 10.74 = 18.18 GB', [round(r[k] / 1e9, 2) for k in ('w', 'g', 'o', 'act', 'mem')] == [2.2, 4.41, 0.83, 10.74, 18.18], [r[k] / 1e9 for k in ('w', 'g', 'o', 'act', 'mem')])
    check('mission step 4.674 s: computing 74.98%, all-reduces 14.29%, bubble 9.76%, DP wait 0.96%; MFU 37.49%',
          round(r['step'], 3) == 4.674 and [round(r[k] / r['step'] * 100, 2) for k in ('useful', 'tp', 'bubble', 'dp')] == [74.98, 14.29, 9.76, 0.96] and round(r['mfu'] * 100, 2) == 37.49,
          (r['step'], [r[k] / r['step'] for k in ('useful', 'tp', 'bubble', 'dp')], r['mfu']))
    check('mission: one data-parallel pass 2N/p ÷ 400 GB/s = 0.0441 s; Show the math slot 0.06520 s', round(2 * MODELS['70b'][0] / 8 / NODE, 4) == 0.0441 and round((r['step'] - r['dp']) / 71, 5) == 0.0652, r['step'])
    check('failures: M = 12 h, a synchronous save 98.78 s (0.99 TB at 10 GB/s)', r['M'] == 12 and round(r['dSync'], 2) == 98.78 and round(14 * MODELS['70b'][0] / 1e12, 2) == 0.99, (r['M'], r['dSync']))
    check('every 3 h sync: 14.80% wasted, 227.1 days, 5.58M GPU hours', (round(r['waste'] * 100, 2), round(r['days'], 1), round(r['gpuh'] / 1e6, 2)) == (14.80, 227.1, 5.58), (r['waste'], r['days'], r['gpuh']))
    a = plan(ivl=0.25, asy=1)
    check('every 15 min async: 3.54% wasted, 200.6 days, 4.93M GPU hours', (round(a['waste'] * 100, 2), round(a['days'], 1), round(a['gpuh'] / 1e6, 2)) == (3.54, 200.6, 4.93), (a['waste'], a['days'], a['gpuh']))
    check('fidelity card: logits 0.26 GB per micro-batch at tensor parallel 8', round(r['logits'] / 1e9, 2) == 0.26, r['logits'])
    s4 = simple_days(0.4)
    check('part 2: 6ND at 40%: 181.33 days, 4.46M GPU hours; 30% 241.8; 50% 145.1', (round(s4, 2), round(s4 * 24 * 1024 / 1e6, 2), round(simple_days(0.3), 1), round(simple_days(0.5), 1)) == (181.33, 4.46, 241.8, 145.1), s4)
    check('part 2: the page agrees', close(js("()=>window.__tcap.simpleDays(0.4,{})"), s4) and close(js("()=>window.__tcap.simpleDays(0.3,{})"), simple_days(0.3)))
    check('part 1: the right answer is “3 to 12 months” (181 days)', K['PRED_RIGHT'] == '3to12' and 90 < s4 < 365)
    check('part 2: C = 6.35 × 10²⁴ FLOP (the scaling book gives 6.3 × 10²⁴ for LLaMA-3 70B)', round(6 * MODELS['70b'][0] * 15e12 / 1e24, 2) == 6.35)
    # ---- part 3: memory
    lv3 = {o['v']: o for o in K['LV3']}
    m3 = {v: plan(t=o['t'], p=o['p'], z=o['z']) for v, o in lv3.items()}
    check('part 3: 1,355.9 / 90.7 / 12.4 / 18.2 GB', [round(m3[v]['mem'] / 1e9, 1) for v in ('none', 'z3', 'tp8z3', 'tp8pp8z1')] == [1355.9, 90.7, 12.4, 18.2], [m3[v]['mem'] / 1e9 for v in m3])
    check('part 3: ZeRO-3 alone: activations 85.9 GB, does not fit; the other two fit', round(m3['z3']['act'] / 1e9, 1) == 85.9 and not m3['z3']['fits'] and m3['tp8z3']['fits'] and m3['tp8pp8z1']['fits'])
    check('part 3: 1.27 TB of model states', round(18 * MODELS['70b'][0] / 1e12, 2) == 1.27)
    # ---- part 4: where the time goes
    lv4 = {o['v']: o for o in K['LV4']}
    m4 = {v: plan(t=o['t'], p=o['p'], z=o['z'], rc=o['rc']) for v, o in lv4.items()}
    check('part 4: MFU 37.49 / 27.74 / 37.41 / 20.70 / 38.68%', [round(m4[v]['mfu'] * 100, 2) for v in ('start', 'rc', 'tp4pp16', 'tp8z3', 'tp8z1')] == [37.49, 27.74, 37.41, 20.70, 38.68], [m4[v]['mfu'] for v in m4])
    check('part 4: the best is about 1.9 times the worst', round(max(m4[v]['mfu'] for v in m4) / min(m4[v]['mfu'] for v in m4), 1) == 1.9)
    check('part 4: TP 4 × PP 16 bubble 19.0%; TP 8 + ZeRO-3 waits 50.7% (“51%”); TP 8 + ZeRO-1 64.5 GB',
          round(m4['tp4pp16']['bubble'] / m4['tp4pp16']['step'] * 100, 1) == 19.0 and round(m4['tp8z3']['dp'] / m4['tp8z3']['step'] * 100, 1) == 50.7 and round(m4['tp8z1']['mem'] / 1e9, 1) == 64.5,
          (m4['tp4pp16']['bubble'] / m4['tp4pp16']['step'], m4['tp8z3']['dp'] / m4['tp8z3']['step'], m4['tp8z1']['mem']))
    check('part 4: every option fits', all(m4[v]['fits'] for v in m4))
    # ---- part 5: failures
    lv5 = {o['v']: o for o in K['LV5']}
    m5 = {v: plan(ivl=o['ivl'], asy=o['asy']) for v, o in lv5.items()}
    check('part 5: wasted 14.80 / 8.30 / 13.41 / 3.54%', [round(m5[v]['waste'] * 100, 2) for v in ('3s', '1s', '15s', '15a')] == [14.80, 8.30, 13.41, 3.54], [m5[v]['waste'] for v in m5])
    check('part 5: hourly sync saving 2.7%; every 15 min sync saving 11.0%, more wasted than hourly; async: a save stops 10 s, restarts 1.39%',
          round(m5['1s']['wSave'] * 100, 1) == 2.7 and round(m5['15s']['wSave'] * 100, 1) == 11.0 and m5['15s']['waste'] > m5['1s']['waste'] and m5['15a']['dS'] == 10 and round(m5['15a']['wRestart'] * 100, 2) == 1.39)
    check('part 5: async every 15 min, the three parts about equal: saving 1.11%, redone 1.04%, restarts 1.39%',
          [round(m5['15a'][k] * 100, 2) for k in ('wSave', 'wRedo', 'wRestart')] == [1.11, 1.04, 1.39], [m5['15a'][k] for k in ('wSave', 'wRedo', 'wRestart')])
    # ---- the capability card's transfer questions
    t2 = plan(G=2048)
    check('transfer 1: 2,048 GPUs: m = 32, bubble 17.64%, M = 6 h, 28.69% wasted, 150.20 days (“about 150”, a third less than 227)',
          (t2['m'], round(t2['bubble'] / t2['step'] * 100, 2), t2['M'], round(t2['waste'] * 100, 2), round(t2['days'], 2)) == (32, 17.64, 6, 28.69, 150.20)
          and round(1 - t2['days'] / r['days'], 2) == 0.34, t2)
    f2 = plan(node=32)
    check('transfer 2: twice the failures: 28.69% wasted (“about 29%”; the options 15% / 29% / 59%)', round(f2['waste'] * 100, 2) == 28.69 and [round(x * 100) for x in (r['waste'], f2['waste'], 4 * r['waste'])] == [15, 29, 59], f2['waste'])
    ex = js("()=>window.__tcap.exits()")
    check('the capability card marks “about 150” and “about 29%” as right', ex['transferA'] == [1, 1], ex['transferA'])
    # ---- part 6: real runs
    card = {'8b': 1.46e6, '70b': 7.0e6, '405b': 30.84e6}
    share = {k: 6 * MODELS[k][0] * 15e12 / (hrs * 3600 * PEAK) for k, hrs in card.items()}
    real = js("()=>window.__tcap.real()")
    check('part 6: Llama 3.1 implied share of the peak 13.9 / 25.5 / 33.2%', [round(share[k] * 100, 1) for k in ('8b', '70b', '405b')] == [13.9, 25.5, 33.2] and all(close(x['u'], share[x['id']]) for x in real), real)
    check('part 6: this lab’s plan reaches MFU × time training = 36.2%', round(a['mfu'] * a['good'] * 100, 1) == 36.2, a['mfu'] * a['good'])
    check('connect: the card’s GPU hours add up to 39.3M', round(sum(card.values()) / 1e6, 1) == 39.3)
    # ---- scenarios
    pre = {x['id']: {k: v for k, v in x.items() if k != 'id'} for x in K['PRESETS']}
    s8, s405, seth, sf = plan(**pre['8b']), plan(**pre['405b']), plan(**pre['eth']), plan(**pre['fail'])
    check('scenario 8B: one sequence per micro-batch, 20.4 GB, MFU 50.0%, 17.7 days, 27.2K GPU hours', (pre['8b']['b'], round(s8['mem'] / 1e9, 1), round(s8['mfu'] * 100, 1), round(s8['days'], 1), round(s8['gpuh'] / 1e3, 1)) == (1, 20.4, 50.0, 17.7, 27.2), (s8['mem'], s8['mfu'], s8['days'], s8['gpuh']))
    check('scenario 8B: the FSDP traffic hides (a pass 2N ÷ 400 GB/s = 0.04 s, 3 passes a micro-batch, under its compute)', s8['dp'] == 0 and round(2 * MODELS['8b'][0] / NODE, 2) == 0.04, s8['dp'])
    check('scenario 405B at BLOOM’s failure rate (224 days a machine): bubble 31.4%, MFU 30.3%, a failure every 2.6 h, 12.2% wasted, 98.0 days, 38.5M GPU hours',
          (pre['405b']['node'], round(s405['bubble'] / s405['step'] * 100, 1), round(s405['mfu'] * 100, 1), round(s405['M'], 1), round(s405['waste'] * 100, 1), round(s405['days'], 1), round(s405['gpuh'] / 1e6, 1)) == (224, 31.4, 30.3, 2.6, 12.2, 98.0, 38.5), s405)
    s405o = plan(**{**pre['405b'], 'node': 64})
    check('scenario 405B: the model card’s 30.84M is about 20% fewer; at OPT’s rate the estimate is about twice the card (1.8×)',
          round((1 - 30.84e6 / s405['gpuh']) * 100) == 20 and round(s405o['gpuh'] / 30.84e6, 1) == 1.8, (s405['gpuh'], s405o['gpuh']))
    check('scenario Ethernet: MFU 37.5% → 23.7%, 318 days instead of 201, the GPU waits for data parallel 38% of a step; 100 Gb/s is 1/32 of 400 GB/s',
          (round(seth['mfu'] * 100, 1), round(seth['days']), round(a['days']), round(seth['dp'] / seth['step'] * 100)) == (23.7, 318, 201, 38) and NODE / ETH == 32, (seth['mfu'], seth['days'], seth['dp'] / seth['step']))
    sfa = plan(**{**pre['fail'], 'ivl': 0.25, 'asy': 1})
    check('scenario frequent failures: 13.9% wasted, 5.97% with 15 min async', (round(sf['waste'] * 100, 1), round(sfa['waste'] * 100, 2)) == (13.9, 5.97), (sf['waste'], sfa['waste']))
    s70 = plan(**pre['70b'])
    check('scenario 70B: the model card’s 7.0M is about 1.4 times this plan', round(7e6 / s70['gpuh'], 1) == 1.4, s70['gpuh'])
    check('scenario 70B = the guided plan with async saving every 15 minutes', pre['70b'] == {**BASE, 'ivl': 0.25, 'asy': 1}, pre['70b'])
    # ---- the challenge
    CH = K['CH']; base = CH['base']
    combos = []
    for t, p, z, rc, ivl, asy in itertools.product(CH['Ts'], CH['Ps'], CH['Zs'], CH['RCs'], CH['IVs'], CH['ASs']):
        r = plan(**{**base, 't': t, 'p': p, 'z': z, 'rc': rc, 'ivl': ivl, 'asy': asy})
        if r: combos.append(dict(t=t, p=p, z=z, rc=rc, ivl=ivl, asy=asy, r=r, ok=r['fits'] and r['valid'] and r['days'] <= CH['days']))
    fit = [e for e in combos if e['r']['fits']]; ok = sorted([e for e in combos if e['ok']], key=lambda e: e['r']['gpuh'])
    best = ok[0]
    check('challenge: 768 plans, 462 fit, 196 meet the goal', (len(combos), len(fit), len(ok)) == (768, 462, 196), (len(combos), len(fit), len(ok)))
    check('challenge best: TP 2, no pipeline, ZeRO-3, no recomputation, every 15 min async: 40.68 days, 500K GPU hours, 47.2 GB, MFU 48.7%',
          (best['t'], best['p'], best['z'], best['rc'], best['ivl'], best['asy']) == (2, 1, 3, 0, 0.25, 1) and (round(best['r']['days'], 2), round(best['r']['gpuh'] / 1e3), round(best['r']['mem'] / 1e9, 1), round(best['r']['mfu'] * 100, 1)) == (40.68, 500, 47.2, 48.7),
          (best, best['r']['days'], best['r']['gpuh'], best['r']['mem']))
    check('challenge tier text: all-reduces 2.7% of the step (TP 8, the best no-recompute layout, 15.4%), saving wastes 2.3%',
          round(best['r']['tp'] / best['r']['step'] * 100, 1) == 2.7 and round(best['r']['waste'] * 100, 1) == 2.3)
    t8 = sorted([e for e in combos if e['t'] == 8 and not e['rc'] and e['r']['fits'] and e['r']['valid']], key=lambda e: e['r']['gpuh'])[0]
    check('challenge tier text: the cheapest TP 8 layout without recomputation spends 15.4% on all-reduces', round(t8['r']['tp'] / t8['r']['step'] * 100, 1) == 15.4, (t8['t'], t8['p'], t8['z'], t8['r']['tp'] / t8['r']['step']))
    within = sorted([e for e in ok if e['r']['gpuh'] <= best['r']['gpuh'] * 1.01], key=lambda e: e['r']['gpuh'])
    check('challenge: within 1% only the same layout saving every hour asynchronously (40.99 days)',
          [(e['t'], e['p'], e['z'], e['ivl'], round(e['r']['days'], 2)) for e in within] == [(2, 1, 3, 0.25, 40.68), (2, 1, 3, 1, 40.99)],
          [(e['t'], e['p'], e['z'], e['ivl'], e['r']['days']) for e in within])
    nxt = [e for e in ok if e['r']['gpuh'] > best['r']['gpuh'] * 1.01][0]
    check('challenge: the next plan uses 3.38% more', round((nxt['r']['gpuh'] / best['r']['gpuh'] - 1) * 100, 2) == 3.38, nxt['r']['gpuh'] / best['r']['gpuh'])
    old = [e for e in ok if (e['t'], e['p'], e['z'], e['rc'], e['ivl'], e['asy']) == (4, 2, 1, 0, 0.25, 1)][0]
    check('challenge: the earlier answer (TP 4 × PP 2, ZeRO-1) now uses 12.4% more', round((old['r']['gpuh'] / best['r']['gpuh'] - 1) * 100, 1) == 12.4, old['r']['gpuh'] / best['r']['gpuh'])
    st = plan(**{**base, **CH['start']})
    check('challenge: the initial setup (no split) does not fit', not st['fits'], st['mem'])
    page_all = js("()=>window.__tcap.chAll()")
    pm = {(e['t'], e['p'], e['z'], e['rc'], e['ivl'], e['asy']): e for e in page_all}
    check('challenge: the page’s 768 plans agree (fit, goal, days, GPU hours, the 1% band)', len(page_all) == 768 and all(
        pm[(e['t'], e['p'], e['z'], e['rc'], e['ivl'], e['asy'])]['ok'] == e['ok'] and close(pm[(e['t'], e['p'], e['z'], e['rc'], e['ivl'], e['asy'])]['gpuh'], e['r']['gpuh'])
        and pm[(e['t'], e['p'], e['z'], e['rc'], e['ivl'], e['asy'])]['min'] == (e in within) for e in combos))
    # ---- the result after Submit (Codex review b9v3, 应修 5): the summary's counterexample, best single change, main gap and rounding
    key = lambda e: (e['t'], e['p'], e['z'], e['rc'], e['ivl'], e['asy'])
    by = {key(e): e for e in combos}
    def parts(r):   # where a plan's days go: the step's parts scaled to the training time, then saving and failures
        td, W = r['train'] / 86400, r['days']
        return dict(u=td * r['useful'] / r['step'], rc=td * r['recomp'] / r['step'], tp=td * r['tp'] / r['step'], bb=td * r['bubble'] / r['step'], dp=td * r['dp'] / r['step'], fx=W * r['waste'])
    def one_change(o):
        me = by[key(o)]; cands = []
        for k, vals in (('t', CH['Ts']), ('p', CH['Ps']), ('z', CH['Zs']), ('rc', CH['RCs']), ('ivl', CH['IVs']), ('asy', CH['ASs'])):
            for v in vals:
                if v == o[k]: continue
                e = by.get(key({**o, k: v}))
                if e and e['ok'] and (not me['ok'] or e['r']['gpuh'] < me['r']['gpuh']): cands.append((e['r']['gpuh'], k, v))
        return min(cands) if cands else None
    def rough(x):   # 2 significant digits with K or M
        e = math.floor(math.log10(x)); r = round(x / 10 ** (e - 1)) * 10 ** (e - 1)
        return f"{r / 1e6:.{0 if r >= 1e7 else 1}f}M" if r >= 1e6 else f"{round(r / 1e3):,}K"
    rev = dict(t=8, p=4, z=1, rc=0, ivl=0.25, asy=1); rv = by[key(rev)]
    ctr = by[(8, best['p'], best['z'], best['rc'], best['ivl'], best['asy'])]
    check('应修 5: the review’s plan (TP 8 × PP 4, ZeRO-1, every 15 min async) meets the goal: 49.87 days, 612,819 GPU hours, “about 610K”; the recommended plan saves 18%',
          rv['ok'] and round(rv['r']['days'], 2) == 49.87 and round(rv['r']['gpuh']) == 612819 and rough(rv['r']['gpuh']) == '610K' and rough(best['r']['gpuh']) == '500K'
          and round((1 - best['r']['gpuh'] / rv['r']['gpuh']) * 100) == 18, (rv['r']['days'], rv['r']['gpuh']))
    check('应修 5: the counterexample, the recommended plan with TP 8: fits, 95.68 days (misses 60), “about 1.2M” GPU hours, 51% of each step waiting for data-parallel sync',
          ctr['r']['fits'] and not ctr['ok'] and round(ctr['r']['days'], 2) == 95.68 and rough(ctr['r']['gpuh']) == '1.2M' and round(ctr['r']['dp'] / ctr['r']['step'] * 100) == 51,
          (ctr['r']['days'], ctr['r']['gpuh'], ctr['r']['dp'] / ctr['r']['step']))
    alt = by[(1, best['p'], best['z'], best['rc'], best['ivl'], best['asy'])]
    check('应修 5: the other counterexample (when your layout is TP 8 + ZeRO-3), TP 1: 91.9 GB, does not fit, mostly activations',
          not alt['r']['fits'] and round(alt['r']['mem'] / 1e9, 1) == 91.9 and alt['r']['act'] > alt['r']['mem'] / 2, alt['r']['mem'])
    pr, pb = parts(rv['r']), parts(best['r'])
    gap = {k: pr[k] - pb[k] for k in ('rc', 'tp', 'bb', 'dp', 'fx')}; gk = max(gap, key=gap.get); ext = rv['r']['days'] - best['r']['days']
    check('应修 5: every plan computes the same 38.7 days; the review’s plan takes 9.2 more days, most of them (69%) waiting for the tensor-parallel all-reduces',
          round(pr['u'], 1) == round(pb['u'], 1) == 38.7 and abs(sum(pr.values()) - rv['r']['days']) < 1e-9 and round(ext, 1) == 9.2 and gk == 'tp' and round(gap['tp'] / ext * 100) == 69, (pr, pb, gap))
    oc = one_change(rev); oc4 = one_change({**rev, 't': 4}); ocs = one_change(dict(CH['start']))
    check('应修 5: the best single change: TP 4 instead of 8 (6% fewer GPU hours); from TP 4, 2 stages instead of 4 (3%); from the start, none meets the goal',
          oc[1:] == ('t', 4) and round((1 - oc[0] / rv['r']['gpuh']) * 100) == 6 and oc4[1:] == ('p', 2) and round((1 - oc4[0] / by[key({**rev, 't': 4})]['r']['gpuh']) * 100) == 3 and ocs is None, (oc, oc4, ocs))
    lay = {}
    for e in combos:
        k = (e['t'], e['p'], e['z'], e['rc'])
        if e['r']['fits'] and e['r']['valid'] and (k not in lay or e['r']['gpuh'] < lay[k]['r']['gpuh']): lay[k] = e
    check('应修 5: “See all plans”: 77 of 128 layouts fit; each one’s cheapest saving is every 15 min async (2.33% wasted)',
          len(lay) == 77 and len({(e['t'], e['p'], e['z'], e['rc']) for e in combos}) == 128 and all((e['ivl'], e['asy']) == (0.25, 1) for e in lay.values()) and round(best['r']['waste'] * 100, 2) == 2.33, len(lay))
    ps = js("o=>window.__tcap.chSummary(o)", rev); pso = js("o=>window.__tcap.chSummary(o)", {**rev, 't': 4})
    check('应修 5: the page’s summary agrees (counterexample, best single change, where the days go, the rounding)',
          ps['counter'] and (ps['counter']['t'], ps['counter']['p'], ps['counter']['z']) == (8, 1, 3) and close(ps['counter']['days'], ctr['r']['days']) and not ps['counter']['ok']
          and (ps['one']['k'], ps['one']['v']) == ('t', 4) and close(ps['one']['gpuh'], oc[0]) and (pso['one']['k'], pso['one']['v']) == ('p', 2)
          and all(close(ps['parts'][k], pr[k]) for k in pr) and all(close(ps['bestParts'][k], pb[k]) for k in pb)
          and [js("x=>window.__tcap.rough(x)", x) for x in (rv['r']['gpuh'], best['r']['gpuh'], ctr['r']['gpuh'])] == ['610K', '500K', '1.2M'], (ps, pso))
    # ---- the drill and the teaser
    for seed in (1, 2, 3, 7):
        dr = js("s=>window.__tcap.drill(s)", seed)
        dd = 6 * MODELS['70b'][0] * dr['D'] / (dr['G'] * PEAK * dr['u']) / 86400; MM = 64 * 24 / (dr['G'] / 8)
        dsync = CK_BPP * MODELS['70b'][0] / CK_W
        check(f'drill {seed}: days, M and τ = √(2δM) with δ = 98.78 s', close(dr['days'], dd) and close(dr['M'], MM) and close(dr['dS'], dsync) and close(dr['tau'], math.sqrt(2 * dsync / 3600 * MM)), dr)
    tz = js("()=>({gs:window.__tcapTz.GS, d:window.__tcapTz.GS.map(g=>window.__tcapTz.days(g)), h:window.__tcapTz.GS.map(g=>window.__tcapTz.gpuh(g))})")
    tp = [plan(D=2e12, G=g, t=8, p=8, z=1, rc=0, b=1, B=1024, net='ib', ivl=0.25, asy=1, node=64) for g in tz['gs']]
    check('teaser: 256 to 16,384 GPUs, doubling', tz['gs'] == [256, 512, 1024, 2048, 4096, 8192, 16384], tz['gs'])
    check('teaser = the planner for TP 8 × PP 8 × ZeRO-1, 2T tokens, async every 15 min', all(close(a, x['days']) and close(b, x['gpuh']) for a, b, x in zip(tz['d'], tz['h'], tp)), (tz['d'], [x['days'] for x in tp]))
    check('teaser: 96.5 / 50.0 / 26.7 / 15.2 / 9.56 / 7.10 / 7.01 days; 593K → 2.76M GPU hours',
          [round(x['days'], 1) if x['days'] >= 10 else round(x['days'], 2) for x in tp] == [96.5, 50.0, 26.7, 15.2, 9.56, 7.10, 7.01] and round(tp[0]['gpuh'] / 1e3) == 593 and round(tp[-1]['gpuh'] / 1e6, 2) == 2.76,
          [(x['days'], x['gpuh']) for x in tp])
    check('teaser default (2,048): 43% fewer days than 1,024, 14% more GPU hours; doubling never halves the days',
          (round((1 - tp[3]['days'] / tp[2]['days']) * 100), round((tp[3]['gpuh'] / tp[2]['gpuh'] - 1) * 100)) == (43, 14) and all(tp[i + 1]['days'] > tp[i]['days'] / 2 for i in range(6)),
          (tp[3]['days'] / tp[2]['days'], tp[3]['gpuh'] / tp[2]['gpuh']))
    # ---- G7 (the owner's message of 2026-10-08, from the model-parallel finding): walk every pick of the challenge, in both
    # languages, and check that a recommended single change makes the value the result shows strictly better at the precision shown
    def g7_walk(js):
        res = {}
        for lang in ('en', 'zh'):
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(150)
            res[lang] = pg.evaluate(js)
        pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(150)
        return res
    g7 = g7_walk("""()=>{const Z=window.__tcap; const num=s=>{const m=String(s).match(/([\\d.,]+)\\s*([KM])/); return m?parseFloat(m[1].replace(/,/g,''))*(m[2]==='M'?1e6:1e3):NaN;};
      const R=/^The best single change is|^只改一项的话/; const out=[];
      Z.chAll().forEach(e=>{const o={t:e.t,p:e.p,z:e.z,rc:e.rc,ivl:e.ivl,asy:e.asy}; if(!e.ok||e.min) return; const t=Z.chNextText(o), s=Z.chSummary(o), rec=R.test(t); let ok=true, why='';
        if(rec){if(!s.one){ok=false; why='no engine change';} else {const a=Z.rough(e.gpuh), b=Z.rough(s.one.gpuh); ok=num(b)<num(a); why='GPU hours '+a+' -> '+b;}}
        if(/less than 1%|不到 1%/.test(t)){ok=false; why+=' (an under-1% gain)';}
        out.push({pick:o,rec,ok,why,t});}); return out;}""")
    for lang in ('en', 'zh'):
        rows = g7[lang]; bad = [r for r in rows if not r['ok']]
        check(f'G7 {lang}: every passing plan that is not the recommended one walked ({len(rows)}); a recommended single change always shows fewer GPU hours',
              len(rows) > 150 and not bad and sum(r['rec'] for r in rows) > 150, bad[:3])
    nz = [r['t'] for r in g7['zh'] if not r['rec']]
    check('G7: where no single change shows fewer GPU hours, the sentence says so and names what the recommended plan changes together (TP 4, ZeRO-2, async every 15 min: 2 changes)',
          '只改一项省不下来。推荐方案同时改了 2 项：把张量并行从 4 份改成 2 份、把 ZeRO‑2 换成 ZeRO‑3。' in nz and all(t.startswith('只改一项省不') and '推荐方案同时改了' in t for t in nz), nz[:4])
    tiny = [r['t'] for r in g7['zh'] if r['pick'] == {'t': 4, 'p': 2, 'z': 1, 'rc': 0, 'ivl': 1, 'asy': 1}]
    check('G7: a saving under 1% is written with the two values as shown (TP 4 × PP 2, ZeRO-1, async hourly: 570K → 560K)', tiny == ['只改一项的话，把存档间隔从 1 小时改成 15 分钟最划算，GPU 小时从约 570K 降到约 560K。'], tiny)
    check('no page errors', not errs, errs[:3])
    br.close()
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails if fails else 'none')
sys.exit(1 if fails else 0)
