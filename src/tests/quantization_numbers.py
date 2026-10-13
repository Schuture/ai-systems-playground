"""Numbers test for the quantization lab: a Python twin of labs/quantization.js, same seeds, recomputes every number the lab shows.

- the generator (mulberry32, z = sum of 12 uniforms − 6, Student-t with 6 degrees of freedom, planted outliers, outlier input
  channels) must match the page bit for bit (first 64 weights, channels, hit count, sums);
- every scheme's rounding (BF16, FP8 E4M3 per tensor + per-vector activations, INT8 / INT4 per row / 128 / 32 / 16 with FP16 scales,
  MXFP4, NVFP4) and the measures (output error, weight error, same pick as BF16) on seeds 1 (the lab) and 2 (the challenge);
- the two blocks of the stage (clean and the one with a single outlier about 50 times the others), the weights that become 0 in each
  scheme, the block scales;
- the real scale: GiB of weights, H100s needed, KV tokens left on one H100, decode and prefill step times; the challenge (every
  weight × KV format), the drill questions and the transfer question's numbers.
Run: python3 tests/quantization_numbers.py"""
import json, pathlib, subprocess, sys
import numpy as np
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
(P / 'out' / 'quantization').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/quantization.html', '--only', 'quantization'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'quantization' / 'numbers-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"></head><body>' + (P / 'out' / 'quantization.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'quantization' / 'numbers-wrapped.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


close = lambda a, b, rel=1e-9: abs(a - b) <= rel * max(1.0, abs(a), abs(b))

# ---------------------------------------------------------------- the twin
O, D, NX, NU, POUT, OLO, OHI, NCH, CHF, SIG = 128, 256, 1000, 6, 1 / 256, 20, 60, 4, 20, 0.02
M32 = 0xFFFFFFFF
FMT = {'bf16': (7, -126, (2 - 2 ** -7) * 2 ** 127), 'fp16': (10, -14, 65504.0), 'e4m3': (3, -6, 448.0), 'e2m1': (1, 0, 6.0)}


def mulberry(seed, count):
    """mulberry32 as in the lab: a_k = seed + k·0x6D2B79F5 (mod 2^32), so the whole stream is computed at once."""
    k = np.arange(1, count + 1, dtype=np.uint64)
    t = ((seed & M32) + k * 0x6D2B79F5) & M32
    t = ((t ^ (t >> 15)) * (t | 1)) & M32
    t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32)) & M32
    return ((t ^ (t >> 14)) & M32).astype(np.float64) / 4294967296


def fl2(a):
    """floor(log2 a), exact (frexp gives a = m·2^e with m in [0.5, 1))."""
    return np.frexp(a)[1] - 1


def qf(x, f):
    """round to the nearest value of the float format (ties to even), saturating."""
    m, emin, mx = FMT[f]
    x = np.asarray(x, dtype=np.float64); a = np.abs(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        ex = np.maximum(fl2(np.where(a > 0, a, 1.0)), emin)
        q = np.ldexp(1.0, ex - m)
        v = np.minimum(np.rint(a / q) * q, mx)
    v = np.where(a == 0, 0.0, v)
    return np.where(x < 0, -v, v)


def zsum(U):
    s = np.zeros(U.shape[0])
    for c in range(12):
        s = s + U[:, c]
    return s - 6


GEN = {}


def gen(seed):
    if seed in GEN:
        return GEN[seed]
    nW, per = O * D, 12 * (1 + NU) + 2
    U = mulberry(seed * 7919 + 13, nW * per + 64 + NX * D * 12)
    Uw = U[:nW * per].reshape(nW, per)
    z0 = zsum(Uw[:, 0:12]); c = np.zeros(nW)
    for i in range(NU):
        t = zsum(Uw[:, 12 * (i + 1):12 * (i + 2)]); c = c + t * t
    w = SIG * z0 / np.sqrt(c / NU)
    u, f = Uw[:, 12 * (1 + NU)], OLO + (OHI - OLO) * Uw[:, 12 * (1 + NU) + 1]
    hit = u < POUT
    W = qf(np.where(hit, w * f, w), 'bf16')
    ch = []
    for v in U[nW * per:nW * per + 64]:
        if len(ch) < NCH:
            cc = int(np.floor(v * D))
            if cc not in ch:
                ch.append(cc)
    big = np.ones(D); big[ch] = CHF
    X = zsum(U[nW * per + 64:].reshape(NX * D, 12)).reshape(NX, D) * big[None, :]
    G = {'W': W.reshape(O, D), 'hit': hit.reshape(O, D), 'X': X, 'ch': ch, 'Q': {}, 'E': {}}
    GEN[seed] = G
    return G


SCH = {'bf16': (16, None, None), 'fp8': (8, None, None), 'int8': (8, D, 127), 'int4-row': (4, D, 7), 'int4-128': (4.125, 128, 7),
       'int4-32': (4.5, 32, 7), 'int4-16': (5, 16, 7), 'mx': (4.25, 32, None), 'nv': (4.5, 16, None)}
ALLK = list(SCH)
CHK = ['bf16', 'fp8', 'int8', 'int4-row', 'int4-128', 'mx', 'nv']


def quantW(seed, k):
    G = gen(seed)
    if k in G['Q']:
        return G['Q'][k]
    W = G['W'].reshape(-1)
    if k == 'bf16':
        out = W.copy()
    elif k == 'fp8':
        s = np.abs(W).max() / 448; out = qf(W / s, 'e4m3') * s
    elif k == 'nv':
        g = np.abs(W).max() / (448 * 6); B = W.reshape(-1, 16)
        s = qf(np.abs(B).max(1) / 6 / g, 'e4m3') * g
        out = (qf(B / s[:, None], 'e2m1') * s[:, None]).reshape(-1)
    elif k == 'mx':
        B = W.reshape(-1, 32); s = np.ldexp(1.0, fl2(np.abs(B).max(1)) - 2)
        out = (qf(B / s[:, None], 'e2m1') * s[:, None]).reshape(-1)
    else:
        _, bs, lv = SCH[k]; B = W.reshape(-1, bs)
        s = qf(np.abs(B).max(1) / lv, 'fp16')
        out = (np.clip(np.rint(B / s[:, None]), -lv, lv) * s[:, None]).reshape(-1)
    G['Q'][k] = out.reshape(O, D)
    return G['Q'][k]


def outputs(X, W):
    """the same summation order as the page (j = 0 … 255, one product at a time), so argmax ties break the same way."""
    Y = np.zeros((X.shape[0], W.shape[0]))
    for j in range(D):
        Y = Y + X[:, j:j + 1] * W[None, :, j]
    return Y


def actFP8(X):
    s = np.abs(X).max(1, keepdims=True) / 448
    return qf(X / s, 'e4m3') * s


def evalK(seed, k):
    G = gen(seed)
    if k in G['E']:
        return G['E'][k]
    if 'Y' not in G:
        G['Y'] = outputs(G['X'], G['W'])
    Wq = quantW(seed, k)
    Y = G['Y'] if k == 'bf16' else outputs(actFP8(G['X']) if k == 'fp8' else G['X'], Wq)
    oerr = float(np.sqrt(((Y - G['Y']) ** 2).sum() / (G['Y'] ** 2).sum()))
    werr = float(np.sqrt(((Wq - G['W']) ** 2).sum() / (G['W'] ** 2).sum()))
    same = int((Y.argmax(1) == G['Y'].argmax(1)).sum())
    G['E'][k] = {'k': k, 'oerr': oerr, 'werr': werr, 'same': same, 'agree': same / NX, 'bits': SCH[k][0]}
    return G['E'][k]


def blocks():
    G = gen(1); W, hit = G['W'], G['hit']; clean = best = None
    for r in range(O):
        for b in range(D // 32):
            h = hit[r, b * 32:b * 32 + 32]; nh = int(h.sum())
            if nh == 0 and clean is None:
                clean = {'r': r, 'b': b}
            if nh == 1:
                hi = int(np.nonzero(h)[0][0])
                if hi < 16:
                    oth = sorted(abs(W[r, b * 32 + j]) for j in range(32) if j != hi)
                    ratio = abs(W[r, b * 32 + hi]) / oth[15]
                    if best is None or abs(ratio - 50) < abs(best['ratio'] - 50):
                        best = {'r': r, 'b': b, 'ratio': ratio, 'i': hi}
    return {'clean': clean, 'outlier': best}


BSZ = {'int4-row': D, 'int4-128': 128, 'int4-32': 32, 'int4-16': 16, 'mx': 32, 'nv': 16, 'int8': D}


def view(bsel, k):
    G = gen(1); B = blocks()[bsel]; r, st = B['r'], B['b'] * 32
    w, q = G['W'][r, st:st + 32], quantW(1, k)[r, st:st + 32]
    oi = B['i'] if bsel == 'outlier' else -1
    zeros = sum(1 for j in range(32) if j != oi and w[j] != 0 and q[j] == 0)
    segs = []
    bs = BSZ.get(k)
    if bs:
        a = (st // bs) * bs
        while a < st + 32:
            am = float(np.abs(G['W'][r, a:a + bs]).max())
            if k == 'mx':
                s = float(np.ldexp(1.0, fl2(am) - 2))
            elif k == 'nv':
                g = np.abs(G['W']).max() / (448 * 6); s = float(qf(am / 6 / g, 'e4m3') * g)
            else:
                s = float(qf(am / SCH[k][2], 'fp16'))
            segs.append({'s': s, 'amax': am, 'size': bs}); a += bs
    berr = float(np.sqrt(((q - w) ** 2).sum() / (w ** 2).sum()))
    return {'zeros': zeros, 'oi': oi, 'segs': segs, 'berr': berr, 'w': w, 'q': q}


GiB, KiB, UGPU = 2 ** 30, 1024, 80 * 0.9 - 2
MOD = {'8b': (8030261248, 128 * KiB), '70b': (70553706496, 320 * KiB), '405b': (405853388800, 504 * KiB)}
RBITS = {k: v[0] for k, v in SCH.items()}
RBITS['int4-row'] = 4
KVB = {'bf16': 1, 'fp8': 0.5}
wGiB = lambda m, k: MOD[m][0] * RBITS[k] / 8 / GiB


def gpus(m, k):
    w = wGiB(m, k)
    for nn in (1, 2, 4, 8):
        if w / nn <= UGPU:
            return {'n': nn, 'w': w, 'room': UGPU - w / nn}
    return {'n': 0, 'w': w, 'room': 0}


kvTok8 = lambda k, kv: int((UGPU * GiB - MOD['8b'][0] * RBITS[k] / 8) // (MOD['8b'][1] * KVB[kv]))
LY, DM, KVH, HD, FFN, VOC = 32, 4096, 8, 128, 14336, 128256
P_LAYER = 2 * DM * DM + 2 * DM * KVH * HD + 3 * DM * FFN + 2 * DM
P_LAYERS, P_HEAD = LY * P_LAYER + DM, VOC * DM
FS, BS, CTX, BIG, PRE = 989.5e12 * 0.5, 3.35e12 * 0.6, 2048, 64, 2048
attn = lambda a, b: 2 * LY * DM * (b * (b + 1) - a * (a + 1))


def step(k, kv, kind):
    wb, fl, kvb = RBITS[k] / 8, FS * 2 if k == 'fp8' else FS, MOD['8b'][1] * KVB[kv]
    if kind in ('d1', 'd64'):
        B = 1 if kind == 'd1' else BIG; lin = B * (2 * P_LAYERS + 2 * P_HEAD); att = B * attn(CTX, CTX + 1); kvt = B * CTX
    else:
        lin = PRE * 2 * P_LAYERS + 2 * P_HEAD; att = attn(0, PRE); kvt = PRE
    tc, tw, tkv = lin / fl + att / FS, wb * (P_LAYERS + P_HEAD) / BS, kvb * kvt / BS
    return {'tc': tc, 'tw': tw, 'tkv': tkv, 'tm': tw + tkv, 't': max(tc, tw + tkv), 'mem': tw + tkv >= tc}


def chEval(k, kv):
    w = wGiB('70b', k); kvg = 16 * 8192 * MOD['70b'][1] * KVB[kv] / GiB; e = evalK(2, k)
    return {'need': w + kvg, 'fits': w + kvg <= UGPU, 'agree': e['agree'], 'ok': w + kvg <= UGPU and e['agree'] >= 0.9 - 1e-12}


def drill(seed):
    r = mulberry(seed * 104729 + 7, 6)
    return {'m': ['8b', '70b', '405b'][int(r[0] * 3)], 'k': ['fp8', 'int4-128', 'mx', 'nv'][int(r[1] * 4)],
            'kv': 'bf16' if r[2] < 0.5 else 'fp8', 'users': [8, 16, 32][int(r[3] * 3)]}


# ---------------------------------------------------------------- compare with the page
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.evaluate("()=>{location.hash='quantization'}"); pg.wait_for_timeout(400)
    J = lambda expr: pg.evaluate('()=>' + expr)
    js = {
        'consts': J("window.__qz.consts"), 'gen1': J("window.__qz.gen(1)"), 'gen2': J("window.__qz.gen(2)"),
        'ev1': J("Object.fromEntries(window.__qz.consts.ALLK.map(k=>[k,window.__qz.evalK(1,k)]))"),
        'ev2': J("Object.fromEntries(window.__qz.consts.ALLK.map(k=>[k,window.__qz.evalK(2,k)]))"),
        'qs': J("Object.fromEntries(window.__qz.consts.ALLK.map(k=>[k,window.__qz.qsum(1,k)]))"),
        'blocks': J("window.__qz.blocks()"),
        'views': J("Object.fromEntries(['clean','outlier'].flatMap(b=>window.__qz.consts.ALLK.map(k=>[b+'|'+k,window.__qz.view(b,k)])))"),
        'gib': J("Object.fromEntries(['8b','70b','405b'].flatMap(m=>window.__qz.consts.ALLK.map(k=>[m+'|'+k,window.__qz.gpus(m,k)])))"),
        'kvt': J("Object.fromEntries(window.__qz.consts.ALLK.flatMap(k=>['bf16','fp8'].map(v=>[k+'|'+v,window.__qz.kvTok8(k,v)])))"),
        'step': J("Object.fromEntries(window.__qz.consts.ALLK.flatMap(k=>['bf16','fp8'].flatMap(v=>['d1','d64','pre'].map(x=>[k+'|'+v+'|'+x,window.__qz.step(k,v,x)]))))"),
        'ch': J("window.__qz.chAll()"),
        'drill': J("Array.from({length:40},(_, i)=>window.__qz.drill(i+1))"),
        'exits': J("window.__qz.exits()"),
    }
    b.close()
check('no page errors', not errs, errs)

C = js['consts']
check('constants', (C['O'], C['D'], C['NX'], C['NU'], C['NCH'], C['CHF'], C['OLO'], C['OHI']) == (O, D, NX, NU, NCH, CHF, OLO, OHI), C)
check('usable GiB per H100 = 70', close(C['UGPU'], 70.0), C['UGPU'])
check('steps ids', C['STEP_IDS'] == ['mission', 'round', 'predict', 'blocks', 'compute', 'scale', 'remember'], C['STEP_IDS'])
check('prediction answer: most become 0', C['PRED_RIGHT'] == 'zero', C['PRED_RIGHT'])
for seed in (1, 2):
    G, g = gen(seed), js[f'gen{seed}']
    check(f'seed {seed}: first 64 weights bit for bit', np.array_equal(G['W'].reshape(-1)[:64], np.array(g['W'])), (G['W'].reshape(-1)[:4], g['W'][:4]))
    check(f'seed {seed}: outlier channels', G['ch'] == g['ch'], (G['ch'], g['ch']))
    check(f'seed {seed}: planted outliers', int(G['hit'].sum()) == g['hits'], (int(G['hit'].sum()), g['hits']))
    check(f'seed {seed}: sum of weights', close(float(G['W'].sum()), g['sumW'], 1e-9), (float(G['W'].sum()), g['sumW']))
    check(f'seed {seed}: sum of inputs', close(float(G['X'].sum()), g['sumX'], 1e-9), (float(G['X'].sum()), g['sumX']))
for k in ALLK:
    q = quantW(1, k)
    check(f'{k}: rounded weights (sum, sum |q|)', close(float(q.sum()), js['qs'][k][0], 1e-9) and close(float(np.abs(q).sum()), js['qs'][k][1], 1e-9), (float(q.sum()), js['qs'][k]))
    for seed in (1, 2):
        e, j = evalK(seed, k), js[f'ev{seed}'][k]
        check(f'seed {seed} {k}: same pick count', e['same'] == j['same'], (e['same'], j['same']))
        check(f'seed {seed} {k}: output error', close(e['oerr'], j['oerr'], 1e-9), (e['oerr'], j['oerr']))
        check(f'seed {seed} {k}: weight error', close(e['werr'], j['werr'], 1e-9), (e['werr'], j['werr']))
        check(f'seed {seed} {k}: bits', e['bits'] == j['bits'])
BL = blocks()
check('the two blocks', BL['clean'] == js['blocks']['clean'] and {x: BL['outlier'][x] for x in 'rbi'} == {x: js['blocks']['outlier'][x] for x in 'rbi'} and close(BL['outlier']['ratio'], js['blocks']['outlier']['ratio']), (BL, js['blocks']))
for bsel in ('clean', 'outlier'):
    for k in ALLK:
        v, j = view(bsel, k), js['views'][f'{bsel}|{k}']
        check(f'view {bsel} {k}: weights', np.array_equal(v['w'], np.array([e[0] for e in j['ent']])) and np.allclose(v['q'], np.array([e[1] for e in j['ent']]), rtol=1e-12, atol=0))
        check(f'view {bsel} {k}: zeros', v['zeros'] == j['zeros'], (v['zeros'], j['zeros']))
        check(f'view {bsel} {k}: scales', len(v['segs']) == len(j['segs']) and all(close(a['s'], c['s']) and close(a['amax'], c['amax']) for a, c in zip(v['segs'], j['segs'])), (v['segs'], j['segs']))
        check(f'view {bsel} {k}: block error', close(v['berr'], j['berr'], 1e-9))
for m in MOD:
    for k in ALLK:
        g, j = gpus(m, k), js['gib'][f'{m}|{k}']
        check(f'{m} {k}: GiB and GPUs', g['n'] == j['n'] and close(g['w'], j['w']), (g, j))
    for k in ALLK:
        for kv in KVB:
            check(f'kv tokens {k} {kv}', kvTok8(k, kv) == js['kvt'][f'{k}|{kv}'], (kvTok8(k, kv), js['kvt'][f'{k}|{kv}']))
for k in ALLK:
    for kv in KVB:
        for kind in ('d1', 'd64', 'pre'):
            s, j = step(k, kv, kind), js['step'][f'{k}|{kv}|{kind}']
            check(f'step {k} {kv} {kind}', all(close(s[x], j[x]) for x in ('tc', 'tw', 'tkv', 'tm', 't')) and s['mem'] == j['mem'], (s, j))
chj = {(x['k'], x['kv']): x for x in js['ch']}
for k in CHK:
    for kv in KVB:
        c, j = chEval(k, kv), chj[(k, kv)]
        check(f'challenge {k} {kv}', close(c['need'], j['need']) and c['fits'] == j['fits'] and c['ok'] == j['ok'] and close(c['agree'], j['agree']), (c, j))
passing = [k for k in chj if chj[k]['ok']]
check('challenge: exactly one setup passes (NVFP4 weights, FP8 KV)', passing == [('nv', 'fp8')], passing)
for i, d in enumerate(js['drill']):
    t = drill(i + 1)
    check(f'drill {i + 1}', all(t[x] == d[x] for x in t), (t, d))

# the numbers the copy states (seed 1 unless named)
e = {k: evalK(1, k) for k in ALLK}
vo = {k: view('outlier', k) for k in ALLK}
check('outlier block: about 50 times the median of the others', 45 <= BL['outlier']['ratio'] <= 55, BL['outlier']['ratio'])
check('INT4 per 32: 31 of the 31 small weights become 0', vo['int4-32']['zeros'] == 31, vo['int4-32']['zeros'])
check('INT4 per 16: the half without the outlier keeps its weights (18 become 0)', vo['int4-16']['zeros'] == 18, vo['int4-16']['zeros'])
check('clean block in INT4 per 32: few zeros', view('clean', 'int4-32')['zeros'] <= 8, view('clean', 'int4-32')['zeros'])
check('error falls with smaller INT4 blocks (row > 128 > 32 > 16)', e['int4-row']['oerr'] > e['int4-128']['oerr'] > e['int4-32']['oerr'] > e['int4-16']['oerr'], [round(e[k]['oerr'], 4) for k in ('int4-row', 'int4-128', 'int4-32', 'int4-16')])
check('NVFP4 beats MXFP4 at the same 4-bit values', e['nv']['oerr'] < e['mx']['oerr'])
check('INT4 per row changes the pick for 14.1% of the inputs, NVFP4 for 4.5%', round(1 - e['int4-row']['agree'], 3) == 0.141 and round(1 - e['nv']['agree'], 3) == 0.045, (1 - e['int4-row']['agree'], 1 - e['nv']['agree']))
check('S7 bits per weight: INT4/128 4.125, MXFP4 4.25, NVFP4 4.5', (RBITS['int4-128'], RBITS['mx'], RBITS['nv']) == (4.125, 4.25, 4.5))
check('70B in NVFP4 = 37.0 GiB, not 32.9', round(wGiB('70b', 'nv'), 1) == 37.0 and round(wGiB('70b', 'int4-row'), 1) == 32.9, (wGiB('70b', 'nv'), wGiB('70b', 'int4-row')))
check('8B BF16 decode step 7.60 ms, INT4/128 2.06 ms, prefill 60.0 ms either way', round(step('bf16', 'bf16', 'd1')['t'] * 1e3, 2) == 7.60 and round(step('int4-128', 'bf16', 'd1')['t'] * 1e3, 2) == 2.06 and round(step('bf16', 'bf16', 'pre')['t'] * 1e3, 1) == round(step('int4-128', 'bf16', 'pre')['t'] * 1e3, 1) == 60.0, (step('bf16', 'bf16', 'd1')['t'], step('int4-128', 'bf16', 'd1')['t'], step('int4-128', 'bf16', 'pre')['t']))
check('FP8 W+A halves the prefill (compute bound)', close(step('fp8', 'bf16', 'pre')['t'] * 2, step('bf16', 'bf16', 'pre')['t'], 0.15))
check('KV tokens on one H100 for the 8B: 450,907 (BF16) and 512,173 (FP8 weights)', kvTok8('bf16', 'bf16') == 450907 and kvTok8('fp8', 'bf16') == 512173, (kvTok8('bf16', 'bf16'), kvTok8('fp8', 'bf16')))
tr = js['exits']['transfer'][0]
check('transfer: 405B / 70B is about 5.8', round(MOD['405b'][0] / MOD['70b'][0], 1) == 5.8, MOD['405b'][0] / MOD['70b'][0])
check('transfer: 405B in MXFP4 is about 200 GiB (answer 1); 4 bits only would be 189.0', 180 <= wGiB('405b', 'mx') <= 220 and tr['a'] == 1 and f"{wGiB('405b', 'mx'):.1f} GiB" in tr['why'] and f"{wGiB('405b', 'int4-row'):.1f} GiB" in tr['why'] and f"{MOD['405b'][0] * 4.25 / 8 / 1e9:.1f}" in tr['why'] and '405.9B' in tr['q'], (wGiB('405b', 'mx'), tr))
L405, D405, F405 = 126, 16384, 53248
mlp_q = (L405 - 2) * 3 * D405 * F405
check("Meta's FP8 405B: 81.3B parameters stay BF16, 56.7 GiB per GPU on 8", round((MOD['405b'][0] - mlp_q) / 1e9, 1) == 81.3 and round((mlp_q + 2 * (MOD['405b'][0] - mlp_q)) / 8 / GiB, 1) == 56.7, ((MOD['405b'][0] - mlp_q) / 1e9, (mlp_q + 2 * (MOD['405b'][0] - mlp_q)) / 8 / GiB))
g128 = gen(1)['hit'].reshape(-1, 128).any(1).mean()
check('about 2 in 5 groups of 128 hold a planted outlier', 0.35 <= g128 <= 0.45, g128)
room8, per8 = round(UGPU - wGiB('8b', 'int4-128'), 1), round(8192 * MOD['8b'][1] * 0.5 / GiB, 2)
users = int(room8 / per8); t2 = js['exits']['transfer'][1]
check('transfer 2 (the 8B, not the challenge 70B): 66.1 GiB left, 0.50 GiB per user, about 132 users (answer 1)', (room8, per8, users) == (66.1, 0.5, 132) and t2['a'] == 1 and '132' in t2['o'][1] and f"{wGiB('8b', 'int4-128'):.1f} GiB" in t2['q'] and '0.50 GiB' in t2['q'] and '<mn>66.1</mn>' in t2['why'] and '<mn>132.2</mn>' in t2['why'], (room8, per8, users, t2))
# Show the math, step block: every row recomputes from the numbers shown on the row above (R6.6), for every format and KV cache
P_STEP = P_LAYERS + P_HEAD; pst = round(P_STEP / 1e9, 3); attG = attn(CTX, CTX + 1) / 1e9
msf = lambda v: f"{v * 1e3:.{2 if v * 1e3 < 10 else 1}f}"
bad = []
for k in ALLK:
    for kv in ('bf16', 'fp8'):
        st = step(k, kv, 'd1'); kvg = round(MOD['8b'][1] * KVB[kv] * CTX / 1e9, 3)
        byt = (RBITS[k] / 8 * P_STEP + MOD['8b'][1] * KVB[kv] * CTX) / 1e9
        if abs(pst * RBITS[k] / 8 + kvg - round(byt, 3)) > 0.0011: bad.append((k, kv, 'bytes', pst * RBITS[k] / 8 + kvg, round(byt, 3)))
        if msf(round(byt, 3) * 1e9 / BS) != msf(st['tm']): bad.append((k, kv, 't_m', msf(round(byt, 3) * 1e9 / BS), msf(st['tm'])))
        lin = 2 * pst; fl = round(lin + round(attG, 3), 3)
        if abs(fl - round((2 * P_STEP + attn(CTX, CTX + 1)) / 1e9, 3)) > 0.0011: bad.append((k, kv, 'FLOPs'))
        tc = (lin / 989.5 + round(attG, 3) / 494.75) if k == 'fp8' else fl / 494.75
        if f"{tc:.2f}" != msf(st['tc']): bad.append((k, kv, 't_c', tc, msf(st['tc'])))
check('Show the math: bytes, t_m, FLOPs and t_c recompute from the shown numbers for all 18 settings', not bad, bad)
check('405B BF16 needs more than 8 H100s; FP8 needs 8', gpus('405b', 'bf16')['n'] == 0 and gpus('405b', 'fp8')['n'] == 8, (gpus('405b', 'bf16'), gpus('405b', 'fp8')))

print('seed 1:', ', '.join(f"{k} {e[k]['bits']}b err {e[k]['oerr'] * 100:.1f}% same {e[k]['agree'] * 100:.1f}%" for k in ALLK))
print('seed 2 (challenge):', ', '.join(f"{k} {evalK(2, k)['agree'] * 100:.1f}%" for k in CHK))
print('70B GiB:', ', '.join(f"{k} {wGiB('70b', k):.1f}" for k in ALLK), '| challenge need:', ', '.join(f"{k}/{kv} {chj[(k, kv)]['need']:.1f}" for k, kv in chj))
print(f'{n - len(fails)}/{n} checks passed' + ('' if not fails else ' — FAILED: ' + ', '.join(fails[:12])))
sys.exit(1 if fails else 0)
