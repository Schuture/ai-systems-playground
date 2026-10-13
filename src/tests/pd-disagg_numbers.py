"""Numbers test for pd-disagg: an independent Python twin of the lab's simulator (same seeds, same requests), compared with the page.
Checks: constants (KV bytes per token, weight bytes, the pool of 450,907 tokens), the step formula on the 32,768-token prefill, KV
transfer times, every guided run the page quotes (parts 1-7), the goodput sweep of part 7, every challenge option, the scenarios, the
minimal-implementation output, and the interview drill. Run: python3 tests/pd-disagg_numbers.py"""
import math, pathlib, re, subprocess, sys, json
from playwright.sync_api import sync_playwright

L, D, KVH, HD, FFN, VOCAB = 32, 4096, 8, 128, 14336, 128256
P_LAYER = 2*D*D + 2*D*KVH*HD + 3*D*FFN + 2*D
P_LAYERS = L*P_LAYER + D
P_HEAD = VOCAB*D
P_TOTAL = P_LAYERS + 2*P_HEAD
KV_TOK = 2*L*KVH*HD*2
W_READ = 2*(P_LAYERS + P_HEAD)
W_MEM = 2*P_TOTAL
FS = 1979e12/2*0.5
BS = 3.35e12*0.6
GiB = 2**30
BLOCK = 16
POOL_T = math.floor((80*GiB*0.9 - W_MEM - 2*GiB)/KV_TOK)
POOL_B = POOL_T // BLOCK
EPS = 1e-9
LINK = {'nvl': 450e9, 'ib': 50e9}
PRE_BUDGET = 8192
GOALS = {'chat': (1.0, 0.15), 'doc': (5.0, None)}
INF = float('inf')


def attn(a, b): return 2*L*D*(b*(b+1) - a*(a+1))
def blocks(n): return (n + BLOCK - 1)//BLOCK


def mulberry(seed):
    h = [seed & 0xffffffff]
    def imul(a, b): return ((a & 0xffffffff)*(b & 0xffffffff)) & 0xffffffff
    def f():
        h[0] = (h[0] + 0x6D2B79F5) & 0xffffffff
        x = imul(h[0] ^ (h[0] >> 15), 1 | h[0])
        x = ((x + imul(x ^ (x >> 7), 61 | x)) & 0xffffffff) ^ x
        return ((x ^ (x >> 14)) & 0xffffffff)/4294967296
    return f


def ms(x): return math.floor(x*1000 + 0.5)/1000
def uni(rnd, lo, hi): return lo + math.floor(rnd()*(hi - lo + 1))



def simulate(spec, cfg):
    mode, n = cfg['mode'], cfg['n']
    p = cfg.get('p', 0) if mode == 'split' else 0
    budget = cfg.get('budget', 0)
    bw = LINK[cfg.get('link', 'nvl')]
    R = [{'id': i, 'cls': s['cls'], 'arr': s['t'], 'P': s['own'], 'out': s['out'], 'doneP': 0, 'gen': 0, 'toks': [], 'first': None,
          'fin': None, 'g': None, 'dg': None, 'x0': None, 'x1': None, 'need': blocks(s['own'] + s['out'])} for i, s in enumerate(spec)]
    G = []
    for i in range(n):
        kind = 'co' if mode == 'co' else ('pre' if i < p else 'dec')
        G.append({'i': i, 'kind': kind, 'busy': False, 't1': 0.0, 'work': None, 'run': [], 'wait': [], 'resv': 0, 'link': 0.0, 'steps': [], 'peak': 0})
    pend, xf = [], []
    nxt, t = 0, 0.0
    guard = 0

    def pending_pf(g): return sum(r['P'] - r['doneP'] for r in g['run'] if r['first'] is None) + sum(r['P'] for r in g['wait'])

    def route(r):
        pool = [g for g in G if g['kind'] in ('co', 'pre')]
        best = min(pool, key=lambda g: (pending_pf(g), len(g['run']) + len(g['wait']), g['i']))
        r['g'] = best['i']; best['wait'].append(r)

    def try_pend(now):
        while pend:
            r = pend[0]
            dec = [g for g in G if g['kind'] == 'dec']
            best = max(dec, key=lambda g: (POOL_B - g['resv'], -g['i']))
            if POOL_B - best['resv'] < r['need']: return
            pend.pop(0); best['resv'] += r['need']; best['peak'] = max(best['peak'], best['resv']); r['dg'] = best['i']
            src = G[r['g']]; s0 = max(now, src['link']); s1 = s0 + r['P']*KV_TOK/bw; src['link'] = s1
            r['x0'], r['x1'] = s0, s1; xf.append(r)

    def start(g, now):
        bud = budget if (g['kind'] == 'co' and budget) else (PRE_BUDGET if g['kind'] == 'pre' else 1e12)
        work = []
        if g['kind'] == 'dec':
            for r in g['wait']: g['run'].append(r)
            g['wait'] = []
            for r in g['run']: work.append((r, 'd', 0, 0))
        else:
            if g['kind'] == 'co':
                for r in g['run']:
                    if r['first'] is not None and r['gen'] < r['out']: work.append((r, 'd', 0, 0)); bud -= 1
            for r in g['run']:
                if r['first'] is None and r['doneP'] < r['P'] and bud > 0:
                    k = min(r['P'] - r['doneP'], bud); work.append((r, 'p', r['doneP'], r['doneP'] + k)); bud -= k
            while g['wait'] and bud > 0:
                r = g['wait'][0]
                if g['kind'] == 'co' and POOL_B - g['resv'] < r['need']: break
                g['wait'].pop(0); g['run'].append(r)
                if g['kind'] == 'co': g['resv'] += r['need']; g['peak'] = max(g['peak'], g['resv'])
                k = min(r['P'] - r['doneP'], bud); work.append((r, 'p', r['doneP'], r['doneP'] + k)); bud -= k
        if not work: return
        T = lg = fa = kv = 0
        for (r, k, a, b) in work:
            if k == 'd':
                c = r['P'] + r['gen'] - 1; T += 1; lg += 1; fa += attn(c, c + 1); kv += c
            else:
                T += b - a; lg += 1 if b == r['P'] else 0; fa += attn(a, b); kv += b
        fl = 2*P_LAYERS*T + 2*P_HEAD*lg + fa; by = W_READ + KV_TOK*kv
        dt = max(fl/FS, by/BS)
        g['busy'] = True; g['t1'] = now + dt; g['work'] = work
        g['steps'].append((now, now + dt, sum(b - a for (r, k, a, b) in work if k == 'p'), sum(1 for w in work if w[1] == 'd'),
                           any(w[0]['cls'] == 'doc' for w in work if w[1] == 'p')))

    def finish(g):
        t1 = g['t1']; g['busy'] = False
        for (r, k, a, b) in g['work']:
            if k == 'p':
                r['doneP'] = b
                if b == r['P']:
                    r['first'] = t1; r['toks'].append(t1); r['gen'] = 1
                    if g['kind'] == 'pre': g['run'].remove(r); pend.append(r)
            else:
                r['gen'] += 1; r['toks'].append(t1)
        for r in list(g['run']):
            if r['first'] is not None and r['gen'] >= r['out']:
                r['fin'] = t1; g['run'].remove(r); g['resv'] -= r['need']
        g['work'] = None

    order = sorted(range(len(R)), key=lambda i: (R[i]['arr'], i))
    while True:
        guard += 1
        if guard > 2000000: raise RuntimeError('runaway')
        tA = R[order[nxt]]['arr'] if nxt < len(order) else INF
        tG = min((g['t1'] for g in G if g['busy']), default=INF)
        tX = min((r['x1'] for r in xf), default=INF)
        t = min(tA, tG, tX)
        if t == INF: break
        for g in G:
            if g['busy'] and g['t1'] <= t + EPS: finish(g)
        for r in [r for r in xf if r['x1'] <= t + EPS]:
            xf.remove(r); G[r['dg']]['wait'].append(r)
        while nxt < len(order) and R[order[nxt]]['arr'] <= t + EPS:
            route(R[order[nxt]]); nxt += 1
        try_pend(t)
        for g in G:
            if not g['busy']: start(g, t)
    for r in R:
        if r['fin'] is None or len(r['toks']) != r['out']: raise RuntimeError('unfinished %d' % r['id'])
    return {'R': R, 'G': G}


def gap(r): return max((r['toks'][i] - r['toks'][i-1] for i in range(1, len(r['toks']))), default=0)


def metrics(run):
    R = run['R']; ok = 0; out = {}
    for cls in ('chat', 'doc'):
        rs = [r for r in R if r['cls'] == cls]
        if not rs: continue
        g = GOALS[cls]
        oks = [(r['first'] - r['arr'] <= g[0] + EPS) and (g[1] is None or gap(r) <= g[1] + EPS) for r in rs]
        out[cls] = {'n': len(rs), 'ok': sum(oks), 'ttft_max': max(r['first'] - r['arr'] for r in rs), 'gap_max': max(gap(r) for r in rs),
                    'ttft_p90': sorted(r['first'] - r['arr'] for r in rs)[max(1, math.ceil(0.9*len(rs) - 1e-9)) - 1]}
        ok += sum(oks)
    out['ok_frac'] = ok/len(R)
    return out


MIX = {'A': (0.1, 7), 'B': (0.3, 9), 'C': (0.0, 7)}
PER_GPU = 60
RATES = [i*0.5 for i in range(1, 17)]


def load_mix(kind, n, rate):
    doc, seed = MIX[kind]; rnd = mulberry(seed); t = 0.0; out = []
    for i in range(n):
        t += -math.log(1 - rnd())/rate
        u = rnd()
        if u < doc: out.append({'t': ms(t), 'cls': 'doc', 'own': uni(rnd, 16384, 32768), 'out': uni(rnd, 100, 300)})
        else: out.append({'t': ms(t), 'cls': 'chat', 'own': uni(rnd, 400, 1600), 'out': uni(rnd, 120, 320)})
    return out


def load_one(doc):
    rnd = mulberry(3); t = 0.0; out = []
    for i in range(8):
        t += -math.log(1 - rnd())/6.0; out.append({'t': ms(t), 'cls': 'chat', 'own': uni(rnd, 400, 1600), 'out': uni(rnd, 120, 320)})
    if doc: out.append({'t': 1.0, 'cls': 'doc', 'own': 32768, 'out': 200})
    return [x for _, x in sorted(enumerate(out), key=lambda p: (p[1]['t'], p[0]))]


_RUNS = {}
def run_of(kind, cfg, rate):
    key = (kind, json.dumps(cfg, sort_keys=True), rate)
    if key not in _RUNS:
        spec = load_one(True) if kind == 'one' else load_one(False) if kind == 'one0' else load_mix(kind, PER_GPU*cfg['n'], rate*cfg['n'])
        r = simulate(spec, cfg); r['m'] = metrics(r); _RUNS[key] = r
    return _RUNS[key]


def goodput(kind, cfg):
    lo, hi = -1, len(RATES)
    while hi - lo > 1:
        mid = (lo + hi)//2
        if run_of(kind, cfg, RATES[mid])['m']['ok_frac'] >= 0.9 - EPS: lo = mid
        else: hi = mid
    return 0 if lo < 0 else RATES[lo]


# ---------------------------------------------------------------- checks against the page
P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/pd-disagg-num.html', '--only', 'pd-disagg'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-pd-disagg-num.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'pd-disagg-num.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-pd-disagg-num.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


def close(a, b, tol=1e-9): return a is not None and b is not None and abs(a - b) <= tol*max(1, abs(b))
def fT(x, zh=False): return (f'{round(x*1000)} 毫秒' if zh else f'{round(x*1000)} ms') if x < 1 else (f'{x:.2f} 秒' if zh else f'{x:.2f} s')
def fP(x): return f'{100 if x >= 1 else 0 if x <= 0 else min(99, max(1, round(x*100)))}%'
def end_of(run): return max(r['fin'] for r in run['R'])
def stall_step(run): return max((s for g in run['G'] for s in g['steps'] if s[2] > 0 and s[3] > 0), key=lambda s: s[1] - s[0])


# the runs the page shows: parts 1-7, the scenarios, every challenge option and the home tile
CO1 = lambda b: {'mode': 'co', 'n': 1, 'budget': b}
SP = lambda n_, p, l='nvl': {'mode': 'split', 'n': n_, 'p': p, 'link': l}
RUNS = [('one0', CO1(0), 0)] + [('one', CO1(b), 0) for b in (0, 2048, 512, 128, 64)] + [('A', {'mode': 'co', 'n': 4, 'budget': 0}, 1.5)] + \
       [('A', SP(4, p), 1.5) for p in (1, 2, 3)] + [('A', SP(4, 2, 'ib'), 1.5)] + [('A', {'mode': 'co', 'n': 8, 'budget': b}, 3) for b in (0, 512)] + \
       [('A', SP(8, p), 3) for p in range(1, 8)] + [('A', SP(8, 6, 'ib'), 3), ('C', {'mode': 'co', 'n': 8, 'budget': 0}, 3)] + \
       [('B', {'mode': 'co', 'n': 8, 'budget': b}, 1.5) for b in (0, 2048, 512, 128)] + [('B', SP(8, p, l), 1.5) for p in range(1, 8) for l in ('nvl', 'ib')] + \
       [('one', SP(2, 1), 0)]
GP_A = [{'mode': 'co', 'n': 8, 'budget': 0}, {'mode': 'co', 'n': 8, 'budget': 512}] + [SP(8, p) for p in range(1, 8)]

with sync_playwright() as pw:
    b = pw.chromium.launch()
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(80)
    pg.evaluate("()=>{location.hash='pd-disagg'}"); pg.wait_for_timeout(600)
    J = lambda expr, arg=None: pg.evaluate(expr, arg)

    # constants and the step formula (S2)
    c = J('()=>window.__pdd.engine.C')
    for k, v in [('P_LAYERS', P_LAYERS), ('P_HEAD', P_HEAD), ('P_TOTAL', P_TOTAL), ('KV_TOK', KV_TOK), ('W_READ', W_READ), ('W_MEM', W_MEM), ('POOL_T', POOL_T), ('POOL_B', POOL_B), ('PRE_BUDGET', 8192)]:
        check(f'constant {k} equals the derivation ({v})', c[k] == v, (c[k], v))
    check('Llama-3.1-8B has 8.03 B parameters; KV cache 128 KiB per token; the pool holds 450,907 tokens', round(P_TOTAL/1e9, 2) == 8.03 and KV_TOK == 128*1024 and POOL_T == 450907, (P_TOTAL, KV_TOK, POOL_T))
    check('links: NVLink 450 GB/s one way, InfiniBand 400 Gb/s = 50 GB/s', c['LINK'] == {'nvl': 450e9, 'ib': 50e9}, c['LINK'])
    pf = (2*P_LAYERS*32768 + 2*P_HEAD + attn(0, 32768))/FS
    sc = J("()=>{const E=window.__pdd.engine; return E.stepCost([{k:'p',a:0,b:32768,r:{P:32768,gen:0}}]);}")
    check('a 32,768-token prefill alone takes 1.49 s and is set by arithmetic', close(sc['dt'], pf) and round(pf, 2) == 1.49 and sc['tc'] > sc['tm'], (sc, pf))
    for nt, nv, ib in [(1000, 0.29e-3, 2.62e-3), (32768, 9.54e-3, 85.9e-3)]:
        check(f'KV transfer of {nt} tokens: {nv*1e3:.2f} ms over NVLink, {ib*1e3:.1f} ms over InfiniBand', abs(nt*KV_TOK/450e9 - nv) < 0.01e-3 and abs(nt*KV_TOK/50e9 - ib) < 0.1e-3, (nt*KV_TOK/450e9, nt*KV_TOK/50e9))

    # every run, request by request
    for kind, cfg, rate in RUNS:
        tw = run_of(kind, cfg, rate)
        pj = J('([k,c,r])=>{const x=window.__pdd.engine.runOf(k,c,r); return {m:x.m,end:x.end,R:x.R.map(q=>[q.arr,q.first,q.fin,q.toks.length,q.g,q.dg]),G:x.G.map(g=>[g.kind,g.steps.length,g.steps.reduce((s,t)=>s+t.t1-t.t0,0)])};}', [kind, cfg, rate])
        same_r = len(pj['R']) == len(tw['R']) and all(close(a[0], r['arr']) and close(a[1], r['first']) and close(a[2], r['fin']) and a[3] == len(r['toks']) and a[4] == r['g'] and a[5] == r['dg'] for a, r in zip(pj['R'], tw['R']))
        same_g = all(a[0] == g['kind'] and a[1] == len(g['steps']) and close(a[2], sum(s[1] - s[0] for s in g['steps'])) for a, g in zip(pj['G'], tw['G']))
        tm, pm = tw['m'], pj['m']
        same_m = close(pm['all']['ok_frac'], tm['ok_frac']) and all(close(pm[k][f], tm[k][f]) for k in ('chat', 'doc') if k in tm for f in ('ttft_p90', 'ttft_max', 'gap_max'))
        check(f'run {kind} {json.dumps(cfg)} at {rate}/s per GPU: every request, every GPU step and the metrics match the twin', same_r and same_g and same_m and close(pj['end'], end_of(tw)), (same_r, same_g, same_m))

    # goodput per GPU (part 7 and the challenge): binary search here, the page's search there, and the full sweep is a prefix
    for cfg in GP_A + [{'mode': 'co', 'n': 8, 'budget': b} for b in (2048, 512)] + [SP(8, p, l) for p in (5, 6) for l in ('nvl', 'ib')]:
        kind = 'A' if cfg in GP_A else 'B'
        g = goodput(kind, cfg); pgp = J('([k,c])=>window.__pdd.engine.goodput(k,c).g', [kind, cfg])
        sweep = J('([k,c])=>{const E=window.__pdd.engine; return E.RATES.map(r=>E.runOf(k,c,r).m.all.ok_frac>=E.TARGET-1e-9);}', [kind, cfg])
        k_ = sweep.index(False) if False in sweep else len(sweep)
        check(f'goodput {kind} {json.dumps(cfg)} = {g} per GPU, page agrees, and the rates that pass are exactly those up to it', pgp == g and all(not x for x in sweep[k_:]) and (g == 0 if k_ == 0 else RATES[k_ - 1] == g), (g, pgp, sweep))
    gA = {json.dumps(c_): goodput('A', c_) for c_ in GP_A}
    check('part 7: best split 6 prefill + 2 decode at 4.0; colocated with chunks of 512 at 6.5; colocated without chunking at 0.5', gA[json.dumps(SP(8, 6))] == 4.0 and max(goodput('A', SP(8, p)) for p in range(1, 8)) == 4.0 and gA[json.dumps(GP_A[1])] == 6.5 and gA[json.dumps(GP_A[0])] == 0.5, gA)

    # the verified numbers the guided parts quote
    r1, r2 = run_of('one0', CO1(0), 0), run_of('one', CO1(0), 0)
    s = stall_step(r2)
    check('part 2: the stall step takes 1.49 s; the chats\' longest gap goes from 38 ms to 1.49 s; 1 of 9 meets its goals', fT(s[1] - s[0]) == '1.49 s' and fT(r1['m']['chat']['gap_max']) == '38 ms' and fT(r2['m']['chat']['gap_max']) == '1.49 s' and round(r2['m']['ok_frac']*9) == 1, (s, r1['m']['chat'], r2['m']))
    gaps = {b_: fT(run_of('one', CO1(b_), 0)['m']['chat']['gap_max']) for b_ in (2048, 512, 128, 64)}
    check('part 3: longest gaps 126 / 32 / 10 / 10 ms for budgets 2,048 / 512 / 128 / 64; the document\'s first token 1.55 / 1.51 / 2.28 / 4.51 s', list(gaps.values()) == ['126 ms', '32 ms', '10 ms', '10 ms']
          and [fT(run_of('one', CO1(b_), 0)['m']['doc']['ttft_max']) for b_ in (2048, 512, 128, 64)] == ['1.55 s', '1.51 s', '2.28 s', '4.51 s'], gaps)
    m4 = run_of('A', {'mode': 'co', 'n': 4, 'budget': 0}, 1.5)['m']; m5 = {p: run_of('A', SP(4, p), 1.5)['m'] for p in (1, 2, 3)}; m6 = run_of('A', SP(4, 2, 'ib'), 1.5)['m']
    check('part 4: colocated 4 GPUs: 81% meet; the chats\' longest gap 1.44 s; documents\' first token 1.41 s (p90)', fP(m4['ok_frac']) == '81%' and fT(m4['chat']['gap_max']) == '1.44 s' and fT(m4['doc']['ttft_p90']) == '1.41 s', m4)
    check('part 5: 1:3 meets 61%, 2:2 98%, 3:1 100%; gaps stay under 33 ms', [fP(m5[p]['ok_frac']) for p in (1, 2, 3)] == ['61%', '98%', '100%'] and all(m5[p]['chat']['gap_max'] < 0.033 for p in (1, 2, 3)), {p: m5[p]['ok_frac'] for p in m5})
    check('part 4 answer: the gap gets much shorter and the documents\' first token later (p90 1.41 s -> 2.85 s with 1 prefill GPU)', m5[1]['chat']['gap_max'] < m4['chat']['gap_max']/10 and m5[1]['doc']['ttft_p90'] > m4['doc']['ttft_p90'] + 1, (m5[1], m4))
    check('part 6: across nodes the longest gap goes from 27 ms to 104 ms; the share meeting goals does not move (98%)', fT(m5[2]['chat']['gap_max']) == '27 ms' and fT(m6['chat']['gap_max']) == '104 ms' and m6['ok_frac'] == m5[2]['ok_frac'], (m5[2]['chat'], m6['chat']))
    chs = {('co', b_): run_of('B', {'mode': 'co', 'n': 8, 'budget': b_}, 1.5)['m']['ok_frac'] for b_ in (0, 2048, 512, 128)}
    chs.update({(p, l): run_of('B', SP(8, p, l), 1.5)['m']['ok_frac'] for p in range(1, 8) for l in ('nvl', 'ib')})
    feas = [k for k, v in chs.items() if v >= 0.9 - EPS]
    check('challenge: the start (2 prefill) meets 7%; passing setups are chunks of 2,048 or 512 and splits 5:3 and 6:2 (either link)', fP(chs[(2, 'nvl')]) == '7%' and set(feas) == {('co', 2048), ('co', 512), (5, 'nvl'), (5, 'ib'), (6, 'nvl'), (6, 'ib')}, chs)
    check('challenge: most room is colocated with chunks (3.0 per GPU) against 1.5 for the splits', goodput('B', {'mode': 'co', 'n': 8, 'budget': 2048}) == 3.0 and goodput('B', {'mode': 'co', 'n': 8, 'budget': 512}) == 3.0 and goodput('B', SP(8, 5)) == 1.5 and goodput('B', SP(8, 6)) == 1.5)
    po = J('()=>window.__pdd.chOpts()')
    check('challenge options on the page: 18, with the same shares and pass marks', len(po) == 18 and all(close(o['share'], chs[('co', o['budget'])] if o['mode'] == 'co' else chs[(o['p'], o['link'])]) and o['ok'] == ((chs[('co', o['budget'])] if o['mode'] == 'co' else chs[(o['p'], o['link'])]) >= 0.9 - EPS) for o in po), po[:2])
    pre = {k: run_of(*a)['m'] for k, a in {'co0': ('A', {'mode': 'co', 'n': 8, 'budget': 0}, 3), 'co512': ('A', {'mode': 'co', 'n': 8, 'budget': 512}, 3), 'split': ('A', SP(8, 6), 3), 'ib': ('A', SP(8, 6, 'ib'), 3), 'chat': ('C', {'mode': 'co', 'n': 8, 'budget': 0}, 3)}.items()}
    check('scenarios: 48% / 100% / 100% / 100% / 100% meet; longest gaps 1.47 s / 32 ms / 50 ms / 126 ms / 46 ms', [fP(pre[k]['ok_frac']) for k in pre] == ['48%', '100%', '100%', '100%', '100%'] and [fT(pre[k]['chat']['gap_max']) for k in pre] == ['1.47 s', '32 ms', '50 ms', '126 ms', '46 ms'], {k: (pre[k]['ok_frac'], pre[k]['chat']['gap_max']) for k in pre})
    check('scenario 3: nine in ten chats get their first token within 438 ms, against 50 ms colocated with a budget of 512', fT(pre['split']['chat']['ttft_p90']) == '438 ms' and fT(pre['co512']['chat']['ttft_p90']) == '50 ms', (pre['split']['chat'], pre['co512']['chat']))
    q4 = run_of('C', {'mode': 'co', 'n': 4, 'budget': 0}, 4)['m']
    g4 = [goodput('C', c) for c in ({'mode': 'co', 'n': 4, 'budget': 0}, SP(4, 1), SP(4, 2), SP(4, 3))]
    check('capability card, new situation 2: 4 GPUs of chats at 4 per GPU all meet, longest gap 63 ms; colocated and every split reach 8 per GPU', fP(q4['ok_frac']) == '100%' and fT(q4['chat']['gap_max']) == '63 ms' and g4 == [8.0, 8.0, 8.0, 8.0], (q4['chat'], g4))
    check('page presets are the five scenarios', [p['id'] for p in J('()=>window.__pdd.PRESETS')] == ['co0', 'co512', 'split', 'ib', 'chat'])
    tz = run_of('one', SP(2, 1), 0)['m']
    check('home tile: one GPU stalls the chats 1.49 s; split onto 2 GPUs the longest gap is 14 ms', fT(r2['m']['chat']['gap_max']) == '1.49 s' and fT(tz['chat']['gap_max']) == '14 ms', tz['chat'])

    # what the page shows: step through the guided parts and read the numbers
    def obs_text():
        pg.wait_for_function('()=>!window.__pdd.state().playing', timeout=60000); pg.wait_for_timeout(150)
        return J("()=>{const s=document.querySelector('#pdd-rail .step.current'); return s?s.innerText:''}")
    def nxt():
        J("()=>document.querySelector('#pdd-rail .step.current [data-nav=next]').click()"); pg.wait_for_timeout(250)
    nxt(); t2 = obs_text()
    check('page part 2 quotes the 1.49 s step', 'That step takes 1.49 s' in t2 and 'dozens of times longer' in t2, t2[:300])
    nxt(); pg.click('#pdd-rail [data-lever="512"]'); pg.wait_for_timeout(250); t3 = obs_text()
    check('page part 3 at 512 quotes 32 ms, the document at 1.51 s and the late chats\' shortest wait 1.21 s', 'falls to 32 ms' in t3 and 'comes after 1.51 s' in t3 and 'The 3 chats that arrive' in t3 and 'shortest of their waits is 1.21 s' in t3, t3[-700:])
    nxt(); pg.click('#pdd-rail [data-pred="gapT"]'); pg.wait_for_timeout(200); t4 = obs_text()
    check('page part 4 quotes the gap 1.44 s and the 1.49 s prefill hint', 'up to 1.44 s' in t4 and 'takes 1.49 s of prefill on one GPU' in t4, t4[:600])
    nxt(); pg.click('#pdd-rail [data-lever="2"]'); pg.wait_for_timeout(250); t5 = obs_text()
    check('page part 5 at 2:2 quotes 98%; the verdict on 1 + 3 quotes 28 ms and 2.85 s', '98%' in t5 and 'fell to 28 ms' in t5 and 'within 2.85 s, later than before' in t5, t5[-600:])
    nxt(); pg.click('#pdd-rail .seg-ctl[data-key="p6l"] button[data-v="ib"]'); pg.wait_for_timeout(250); t6 = obs_text()
    check('page part 6 quotes 27 ms, 104 ms and 98%', 'from 27 ms to 104 ms' in t6 and 'With either link, 98% of requests' in t6, t6[-400:])
    nxt(); pg.wait_for_function('()=>window.__pdd.state().gpQueue===0', timeout=60000); pg.wait_for_timeout(300)
    pg.click('#pdd-rail .seg-ctl[data-key="p7"] button[data-v="6"]'); pg.wait_for_timeout(300); t7 = obs_text()
    check('page part 7 quotes the best split 6 + 2 at 4, colocated 0, chunks of 512 at 6.5', 'The best split gives 6 GPUs to prefill and the rest to decode: about 4 requests' in t7 and 'Colocated, the 8 GPUs take only 0.5 requests per second per GPU without chunking, and 6.5 with a budget of 512 tokens per step.' in t7, t7[-600:])
    nxt(); t8 = obs_text()
    check('page part 8 quotes 1.49 s, 61%, 4 GiB, 10 ms and 86 ms', all(x in t8 for x in ('1.49 s', '61%', '4 GiB', '10 ms', '86 ms')), t8[:600])

    # the Connect card's code runs here and prints what its comments say
    code = J("()=>document.querySelector('#pdd-code').textContent")
    src, out = code.split('\n# 1000')[0], '# 1000' + code.split('\n# 1000')[1]
    lines = src.split('\n')
    check('the code is at most 40 lines of at most 74 characters', len(lines) <= 40 and max(map(len, lines)) <= 74, (len(lines), max(map(len, lines))))
    res = subprocess.run([sys.executable, '-c', src], capture_output=True, text=True)
    check('the code runs and prints its comment lines', res.returncode == 0 and ['# ' + x for x in res.stdout.strip().split('\n')] == out.strip().split('\n'), (res.stdout, res.stderr[-200:], out))

    # the interview drill: bytes and times from the same constants
    J("()=>document.querySelector('[data-act=drill-reveal]').click()"); pg.wait_for_timeout(150)
    rnd = mulberry(1*7919 + 5); nd = 4096*(2 + math.floor(rnd()*14)); da = J("()=>document.querySelector('#pdd-drill-a').innerText")
    gb = nd*KV_TOK/1e9
    check(f'drill 1: a {nd}-token prompt moves {gb:.2f} GB: {fT(nd*KV_TOK/450e9)} over NVLink, {fT(nd*KV_TOK/50e9)} across nodes', f'{fT(nd*KV_TOK/450e9)} over NVLink' in da and f'{fT(nd*KV_TOK/50e9)} across nodes' in da, da[:200])
    check('no page errors', not errs, errs[:3])
    b.close()

print(f'{n - len(fails)}/{n} checks passed')
print('FAILED:', fails if fails else 'none')
sys.exit(1 if fails else 0)
