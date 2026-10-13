"""Numbers test for the checkpointing lab: every number on the page, recomputed independently in Python.

1. The toy run. A NumPy port of the model the lab states (fidelity popover, specs/checkpointing.md): mulberry32 seeds, three spiral
   arms, an MLP 2 → 64 → 3 with dropout 0.1, AdamW with bias correction, warmup + cosine, a shuffled order per epoch, the validation
   loss every 5 steps, a checkpoint every 100 steps; resuming with parts missing as the lab says. Its curves must match the page's
   (relative 1e-6: libm and summation order differ by an ulp), and from them: the losses at step 600, “up to +103%”, 960 / 960, the
   spike's numbers, the prediction's right answer (worse: above the gray line from the resume to step 600, far below the start), and
   the exact resume with all four parts (bit for bit, in Python).
2. The intervals: 14 bytes per parameter (BF16 weights 2 + FP32 copy of the weights 4 + Adam's m 4 + v 4), the sizes of Llama 3.1 8B / 70B / 405B (parameter
   counts computed from the configs' shapes, as in specs/training-memory.md), δ, T* = √(2δM) and the wasted share δ/T + (T/2 + R)/M
   for every interval and case the page shows; the page's own texts for part 4, the scenarios and the math are read back and checked.
   Part 4's three terms (owner b8): δ/T, T/(2M) and R/M for every interval and both modes, in the rail's table and under the picture.
3. The challenge: every plan (mode × interval × kept), the passing set, the best plan, the start, the result texts; the result as the
   train-capstone lab's (R1): the gap to the best plan, the best single change, and the passing plans only in the closed list.
4. The capability layer: the transfer answers (√ scaling; the data-loader count), the drill for seeds 1–6, the teaser at every stop.
Run: python3 tests/checkpointing_numbers.py"""
import math, pathlib, re, subprocess, sys
import numpy as np
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/checkpointing.html', '--only', 'checkpointing'], check=True, cwd=P, stdout=subprocess.DEVNULL)
OUT = P / 'out' / 'checkpointing'; OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'wrapped-num.html').write_text('<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'checkpointing.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-num.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


# ---------------------------------------------------------------- 1. the toy run
NTR, NEV, B, H, C, STEPS, WARM, PEAK, MINR = 1440, 300, 24, 64, 3, 600, 30, 0.05, 0.1
B1, B2, EPS, WD, PD, NOISE = 0.9, 0.999, 1e-8, 0.01, 0.1, 0.2
SD, SE, SI, SR, SS, EVERY, SAVE, CRASH, CK, BAD0, BADN, BADX = 11, 23, 5, 99, 7000, 5, 100, 450, 400, 470, 5, 8
EPOCH = NTR // B; NPAR = 2 * H + H + C * H + C
M32 = 0xFFFFFFFF


class Mul:
    """mulberry32, with 32-bit integer semantics."""
    def __init__(self, seed): self.a = seed & M32
    def __call__(self):
        self.a = (self.a + 0x6D2B79F5) & M32; a = self.a
        t = ((a ^ (a >> 15)) * (1 | a)) & M32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & M32)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296.0


def gen(nn, seed):
    r = Mul(seed); X = np.zeros((nn, 2)); Y = np.zeros(nn, dtype=int)
    for i in range(nn):
        y = i % C; t = r(); rad = 0.15 + 1.35 * t; ang = 2 * math.pi * y / C + t * 3.4
        nx = (r() + r() + r() - 1.5) * NOISE; ny = (r() + r() + r() - 1.5) * NOISE
        X[i] = (rad * math.cos(ang) + nx, rad * math.sin(ang) + ny); Y[i] = y
    return X, Y


XT, YT = gen(NTR, SD); XE, YE = gen(NEV, SE)
PERM = {}


def perm(e):
    if e not in PERM:
        r = Mul(SS + e); p = list(range(NTR))
        for i in range(NTR - 1, 0, -1):
            j = int(r() * (i + 1)); p[i], p[j] = p[j], p[i]
        PERM[e] = p
    return PERM[e]


def init():
    r = Mul(SI); f = np.zeros(NPAR); a1 = 1 / math.sqrt(2); a2 = 1 / math.sqrt(H)
    for i in range(H):
        for j in range(2): f[i * 2 + j] = (2 * r() - 1) * a1 * 1.7
    o = 3 * H
    for c in range(C):
        for i in range(H): f[o + c * H + i] = (2 * r() - 1) * a2 * 1.7
    return f


def lr_at(s):
    return PEAK * (s + 1) / WARM if s < WARM else PEAK * (MINR + (1 - MINR) * 0.5 * (1 + math.cos(math.pi * ((s - WARM) / (STEPS - WARM)))))


def unpack(f):
    W1 = f[:2 * H].reshape(H, 2); b1 = f[2 * H:3 * H]; W2 = f[3 * H:3 * H + C * H].reshape(C, H); b2 = f[3 * H + C * H:]
    return W1, b1, W2, b2


def loss_grad(f, xs, ys, mask=None, grad=False):
    W1, b1, W2, b2 = unpack(f); nn = len(ys)
    z = xs @ W1.T + b1; a = np.where(z > 0, z, 0.0)
    if mask is not None: a = a * mask
    lo = a @ W2.T + b2; mx = lo.max(1, keepdims=True); ex = np.exp(lo - mx); se = ex.sum(1, keepdims=True)
    L = float(np.mean(-(lo[np.arange(nn), ys] - mx[:, 0] - np.log(se[:, 0]))))
    if not grad: return L, None
    dl = ex / se; dl[np.arange(nn), ys] -= 1
    gW2 = dl.T @ a / nn; gb2 = dl.sum(0) / nn
    dh = dl @ W2
    if mask is not None: dh = dh * mask
    dh = np.where(z > 0, dh, 0.0)
    gW1 = dh.T @ xs / nn; gb1 = dh.sum(0) / nn
    return L, np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])


ev_loss = lambda f: loss_grad(f, XE, YE)[0]
fresh = lambda: dict(f=init(), m=np.zeros(NPAR), v=np.zeros(NPAR), t=0, sched=0, epoch=0, pos=0, rng=SR, step=0)
cp = lambda s: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()}


def train(st0, until, bad=(), skip=()):
    st = cp(st0); r = Mul(st['rng']); ev, snaps, seen = {}, {}, {}
    keep = 1 / (1 - PD)
    while st['step'] < until:
        s = st['step']
        if s % EVERY == 0: ev[s] = ev_loss(st['f'])
        if s % SAVE == 0: c = cp(st); c['rng'] = r.a; snaps[s] = c
        p = perm(st['epoch']); ids = p[st['pos'] * B:(st['pos'] + 1) * B]
        for i in ids: k = st['epoch'] * NTR + i; seen[k] = seen.get(k, 0) + 1
        xs = XT[ids].copy(); ys = YT[ids].copy()
        st['pos'] += 1
        if st['pos'] == EPOCH: st['pos'] = 0; st['epoch'] += 1
        if s in bad: xs *= BADX; ys = (ys + 1) % C
        mask = np.array([[keep if r() >= PD else 0.0 for _ in range(H)] for _ in range(B)])
        if s in skip: st['step'] += 1; st['sched'] += 1; continue
        _, g = loss_grad(st['f'], xs, ys, mask, True)
        lr = lr_at(st['sched']); st['t'] += 1; t = st['t']
        st['m'] = B1 * st['m'] + (1 - B1) * g; st['v'] = B2 * st['v'] + (1 - B2) * g * g
        mh = st['m'] / (1 - B1 ** t); vh = st['v'] / (1 - B2 ** t)
        st['f'] = st['f'] - lr * WD * st['f'] - lr * mh / (np.sqrt(vh) + EPS)
        st['sched'] += 1; st['step'] += 1
    ev[st['step']] = ev_loss(st['f']); c = cp(st); c['rng'] = r.a; snaps[st['step']] = c
    return dict(ev=ev, snaps=snaps, seen=seen)


def restore(ck, keep):
    st = cp(ck)
    if not keep.get('o'): st['m'] = np.zeros(NPAR); st['v'] = np.zeros(NPAR); st['t'] = 0
    if not keep.get('s'): st['sched'] = 0
    if not keep.get('d'): st['pos'] = 0
    if not keep.get('r'): st['rng'] = SR
    return st


base = train(fresh(), STEPS)
before = train(fresh(), CK)['seen']


def resumed(keep):
    run = train(restore(base['snaps'][CK], keep), STEPS)
    gaps = {s: run['ev'][s] - base['ev'][s] for s in run['ev'] if s >= CK}
    allc = dict(before)
    for k, v in run['seen'].items(): allc[k] = allc.get(k, 0) + v
    twice = sum(1 for v in allc.values() if v >= 2)
    never = sum(1 for e in range(STEPS // EPOCH) for i in range(NTR) if not allc.get(e * NTR + i))
    return dict(ev=run['ev'], same=all(g == 0 for g in gaps.values()), maxRel=max(0, max(gaps[s] / base['ev'][s] for s in gaps)),
                final=run['ev'][STEPS], final0=base['ev'][STEPS], twice=twice, never=never, gaps=gaps)


MASKS = {'0000': {}, '1000': {'o': 1}, '1101': {'o': 1, 's': 1, 'r': 1}, '1111': {'o': 1, 's': 1, 'd': 1, 'r': 1}, '0111': {'s': 1, 'd': 1, 'r': 1}}
RES = {k: resumed(v) for k, v in MASKS.items()}
ALL16 = {f'{o}{s_}{d}{r_}': resumed({'o': o, 's': s_, 'd': d, 'r': r_}) for o in (0, 1) for s_ in (0, 1) for d in (0, 1) for r_ in (0, 1)}
for k, r in ALL16.items():
    o_, s_, d_, r_ = (int(c) for c in k)
    if not s_:
        check(f'parts {k}: scheduler missing: the loss at step 600 is above the gray line (the text says so)', r['final'] > r['final0'], (r['final'], r['final0']))
    elif not o_:
        check(f'parts {k}: optimizer missing: jumps (above +10%), then back within 15% at step 600', r['maxRel'] > 0.1 and abs(r['final'] / r['final0'] - 1) < 0.15, (r['maxRel'], r['final'], r['final0']))
    elif not d_:
        check(f'parts {k}: data position missing: the loss looks healthy (within 15% at step 600) but 960 / 960', abs(r['final'] / r['final0'] - 1) < 0.15 and (r['twice'], r['never']) == (960, 960), (r['final'], r['final0'], r['twice']))
    elif not r_:
        check(f'parts {k}: RNG missing: healthy (within 15% at step 600) but not the same run', abs(r['final'] / r['final0'] - 1) < 0.15 and not r['same'], (r['final'], r['final0']))
    else:
        check(f'parts {k}: all four: exact', r['same'])
BAD = set(range(BAD0, BAD0 + BADN))
spike = train(fresh(), STEPS, BAD)
REM = {'latest': dict(ev={s: v for s, v in spike['ev'].items() if s >= 500}, final=spike['ev'][STEPS])}
for o in ('back', 'skip'):
    r = train(restore(spike['snaps'][CK], {'o': 1, 's': 1, 'd': 1, 'r': 1}), STEPS, BAD, BAD if o == 'skip' else ())
    REM[o] = dict(ev=r['ev'], final=r['ev'][STEPS])
peak = max(v for s, v in spike['ev'].items() if s >= BAD0)
lossF = lambda v: f'{v:.2f}' if v >= 1 else f'{v:.3f}'

# the claims that do not need the page
w0 = RES['0000']
check('engine: the run learns (loss at step 0 above 0.9, at 600 below 0.05)', base['ev'][0] > 0.9 and base['ev'][STEPS] < 0.05, (base['ev'][0], base['ev'][STEPS]))
above = [s for s in w0['gaps'] if s > CK and w0['gaps'][s] > 0]
check('prediction: weights only stays above the gray line at 90% of the eval points after the resume, and at every point from 590 (the right answer, “worse”: mostly above, still behind)', len(above) >= 36 and all(w0['gaps'][s] > 0 for s in (590, 595, 600)), [s for s in w0['gaps'] if s > CK and w0['gaps'][s] <= 0])
check('prediction: at step 600 still clearly behind (not “bump”): more than 1.5 times the uninterrupted loss', w0['final'] > 1.5 * w0['final0'], (w0['final'], w0['final0']))
check('prediction: not “broken”: the resumed run stays far below the starting loss', max(v for s, v in w0['ev'].items() if s > CK) < 0.3 * base['ev'][0], max(w0['ev'].values()))
check('all four parts: the resumed run equals the uninterrupted one bit for bit', RES['1111']['same'])
check('data position lost: 960 read twice, 960 never read (40 batches × 24)', (RES['1101']['twice'], RES['1101']['never']) == (960, 960), (RES['1101']['twice'], RES['1101']['never']))
check('all parts kept: nothing read twice or missed', (RES['1111']['twice'], RES['1111']['never']) == (0, 0))
check('spike: rolling back alone repeats the spike exactly', REM['back']['final'] == spike['ev'][STEPS], (REM['back']['final'], spike['ev'][STEPS]))
check('spike: rolling back and skipping ends well below the latest-checkpoint run', REM['skip']['final'] < 0.6 * REM['latest']['final'], (REM['skip']['final'], REM['latest']['final']))
first_pass = next(k for k in range(STEPS) if lr_at(k) > lr_at(CK + k))
check('part 2 text: a new scheduler passes the uninterrupted learning rate about ten steps after the resume', 5 <= first_pass <= 15, first_pass)
check('part 2 text: at step 600 the learning rate is still 8 times the uninterrupted one', round(lr_at(STEPS - 1 - CK) / lr_at(STEPS - 1)) == 8, lr_at(STEPS - 1 - CK) / lr_at(STEPS - 1))
for k, r in ALL16.items():
    if k[1] == '0':
        k2 = k[0] + '1' + k[2:]
        check(f'part 2 text: with the scheduler step added back ({k} → {k2}) the largest gap clearly shrinks (by a third or more)', ALL16[k2]['maxRel'] < (2 / 3) * r['maxRel'], (r['maxRel'], ALL16[k2]['maxRel']))
check('part 2 text: every set with the scheduler step has a smaller largest gap than every set without it', max(r['maxRel'] for k, r in ALL16.items() if k[1] == '1') < min(r['maxRel'] for k, r in ALL16.items() if k[1] == '0'),
      sorted((k, round(r['maxRel'] * 100)) for k, r in ALL16.items()))
check('scheduler numbers in the math: η(400), η(0), η(199), η(599)', [round(lr_at(s), 4) for s in (400, 0, 199, 599)] == [0.0173, 0.0017, 0.0409, 0.005], [lr_at(s) for s in (400, 0, 199, 599)])

# ---------------------------------------------------------------- 2. intervals
P8, P70, P405, BPC = 8030261248, 70553706496, 405853388800, 14
check('14 bytes per parameter = BF16 weights 2 + FP32 copy of the weights 4 + m 4 + v 4 (the optimizer state is the last three, 12 bytes)', 2 + 4 + 4 + 4 == BPC and 4 + 4 + 4 == 12)
S8, S70, S405 = P8 * BPC, P70 * BPC, P405 * BPC
check('70B checkpoint 0.99 TB, 8B 112.4 GB, 405B 5.68 TB', (f'{S70 / 1e12:.2f}', f'{S8 / 1e9:.1f}', f'{S405 / 1e12:.2f}') == ('0.99', '112.4', '5.68'), (S70, S8, S405))
waste = lambda d, T, M, R: d / T + T / (2 * M) + R / M
tOpt = lambda d, M: math.sqrt(2 * d * M)
M70, R, BW, DA = 12 * 3600, 600, 10e9, 10
M8 = 32 * 7 * 24 * 3600  # BLOOM: 1.5 GPU failures a week on 384 GPUs, for 8 GPUs: 1 every 384 / (8 × 1.5) = 32 weeks
check('M for 8 GPUs from BLOOM’s 1–2 a week on 384 (1.5 assumed): 32 weeks', abs(384 / (8 * 1.5) - 32) < 1e-12)
d70 = S70 / BW
check('δ sync 70B ≈ 98.8 s, T* ≈ 2,921 s ≈ 49 min; async T* ≈ 930 s ≈ 15 min', (f'{d70:.1f}', round(tOpt(d70, M70)), round(tOpt(d70, M70) / 60), round(tOpt(DA, M70) / 60)) == ('98.8', 2921, 49, 15), (d70, tOpt(d70, M70), tOpt(DA, M70)))
INTS = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480]
from decimal import Decimal, ROUND_HALF_UP
# the page rounds with Intl (half away from zero on the exact value: 6.25 → 6.3); Python's format rounds half to even (6.2)
pct = lambda v, d=1: f'{Decimal(v * 100).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)}%'


def mins(s):
    m = s / 60
    return (f'{m / 60:.1f} h' if m >= 90 else (f'{m:.1f} min' if m < 10 else f'{m:.0f} min'))


# ---------------------------------------------------------------- 3. the challenge
CM, CR, CBW, CST, CBACK, GOAL = 3 * 3600, 600, 50e9, 30e12, 60, 0.12
CI, CK_ = [5, 10, 15, 20, 30, 60], [2, 3, 4, 5, 6]


def ch(mode, T, k):
    d = DA if mode == 'async' else S405 / CBW
    w = waste(d, T * 60, CM, CR); st = k * S405; back = (k - 1) * T
    return dict(mode=mode, T=T, k=k, w=w, store=st, back=back, ok=st <= CST and back >= CBACK and w <= GOAL + 1e-12, fits=st <= CST, rb=back >= CBACK)


ALLP = [ch(m, T, k) for m in ('sync', 'async') for T in CI for k in CK_]
best = sorted([e for e in ALLP if e['fits'] and e['rb']], key=lambda e: (e['w'], e['k']))[0]
check('challenge: the best plan is asynchronous, every 15 min, keep 5, 10.8%', (best['mode'], best['T'], best['k'], pct(best['w'])) == ('async', 15, 5, '10.8%'), best)
check('challenge: the start (sync, 1 h, keep 2) wastes 25.4% and fails', pct(ch('sync', 60, 2)['w']) == '25.4%' and not ch('sync', 60, 2)['ok'])
check('challenge: async, 20 min, keep 4 passes with 11.9%', ch('async', 20, 4)['ok'] and pct(ch('async', 20, 4)['w']) == '11.9%', ch('async', 20, 4))
check('challenge: synchronous never passes', not any(e['ok'] for e in ALLP if e['mode'] == 'sync'))
check('challenge: 5 checkpoints of 5.68 TB fit in 30 TB, 6 do not', 5 * S405 <= CST < 6 * S405)
PASS = sorted([e for e in ALLP if e['ok']], key=lambda e: (e['w'], e['k'], e['T']))
check('challenge: three plans pass: async 15 min keep 5, async 20 min keep 4 and keep 5', [(e['mode'], e['T'], e['k']) for e in PASS] == [('async', 15, 5), ('async', 20, 4), ('async', 20, 5)], PASS)


def one(e):
    c = [ch(m, e['T'], e['k']) for m in ('sync', 'async') if m != e['mode']] + [ch(e['mode'], T, e['k']) for T in CI if T != e['T']] + [ch(e['mode'], e['T'], k) for k in CK_ if k != e['k']]
    return sorted([x for x in c if x['ok']], key=lambda x: (x['w'], x['k']))


START = ch('sync', 60, 2)
check('challenge: from the start no single change meets the goal (the result says so)', one(START) == [])
check('challenge: async 15 min keep 4 misses only the 1 hour back; one change (keep 5) meets it', [(x['mode'], x['T'], x['k']) for x in one(ch('async', 15, 4))][:1] == [('async', 15, 5)])
check('challenge: async 20 min keep 4 passes, and no single change wastes less', not [x for x in one(ch('async', 20, 4)) if x['w'] < ch('async', 20, 4)['w'] - 1e-12])

# ---------------------------------------------------------------- 4. capability layer (Python side)
to4 = tOpt(d70, M70 / 4) / 60
check('transfer 1: a quarter of M halves T*: about 24 min, nearest option “about 25 minutes”', round(to4) == 24 and min([12, 25, 50], key=lambda x: abs(x - to4)) == 25, to4)
check('transfer 2: step 2,300 with 1,000-step epochs is 300 steps into the third epoch', 2300 % 1000 == 300 and 2300 // 1000 + 1 == 3)


def drill(seed):
    a = float(seed * 7919 + 3)  # the page computes in doubles, so the products round as in JavaScript
    def r():
        nonlocal a
        a = math.fmod(a * 1103515245.0 + 12345.0, 2147483648.0); return a / 2147483648.0
    pk = lambda x: x[int(r() * len(x))]
    Pm = pk([P8, P70, P405]); bw = pk([10, 20, 50]) * 1e9; Mh = pk([3, 6, 12, 24])
    size = Pm * BPC; d = size / bw; M = Mh * 3600; to = tOpt(d, M)
    return dict(P=Pm, bw=bw, Mh=Mh, size=size, d=d, to=to, w=waste(d, to, M, 600))


TD, TM, TR, TI = 60, 8 * 3600, 600, [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240]
tw = lambda Tm: waste(TD, Tm * 60, TM, TR)
tto = tOpt(TD, TM) / 60
check('teaser: least at about every 31 min, about 9%; every 2 h about 15%', (round(tto), round(tw(tto) * 100), round(tw(120) * 100)) == (31, 9, 15), (tto, tw(tto), tw(120)))

# ---------------------------------------------------------------- the page
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(80)
    pg.evaluate("()=>{location.hash='checkpointing'}"); pg.wait_for_timeout(500)
    J = lambda e: pg.evaluate(f"()=>{{const h=window.__ckp; return {e}}}")
    pb = J("h.base().ev")
    check('page: the uninterrupted run matches the Python run at every eval point (rel 1e-6)', all(abs(pb[str(s)] - v) <= 1e-6 * v for s, v in base['ev'].items()), [(s, pb[str(s)], v) for s, v in base['ev'].items() if abs(pb[str(s)] - v) > 1e-6 * v][:3])
    for k, keep in MASKS.items():
        pr = J(f"h.resumed({{o:{keep.get('o', 0)},s:{keep.get('s', 0)},d:{keep.get('d', 0)},r:{keep.get('r', 0)}}})")
        py = RES[k]
        ok = all(abs(pr['ev'][str(s)] - v) <= 1e-6 * v for s, v in py['ev'].items())
        check(f'page: resumed with parts {k} matches Python (curve, +{py["maxRel"] * 100:.0f}%, {py["twice"]} / {py["never"]}, exact = {py["same"]})',
              ok and f"{pr['maxRel'] * 100:.0f}" == f"{py['maxRel'] * 100:.0f}" and (pr['twice'], pr['never'], pr['same']) == (py['twice'], py['never'], py['same']), (pr['maxRel'], py['maxRel'], pr['twice'], pr['same']))
    for o in ('latest', 'back', 'skip'):
        pr = J(f"h.remedy('{o}')")
        check(f'page: the spike remedy {o} matches Python ({lossF(REM[o]["final"])}, peak {lossF(peak)})', lossF(pr['final']) == lossF(REM[o]['final']) and lossF(pr['peak']) == lossF(peak), (pr['final'], REM[o]['final'], pr['peak'], peak))
    check('page: weights only: up to +103% (the metric card)', f"{w0['maxRel'] * 100:.0f}" == '103', w0['maxRel'])
    # the guided texts, read back
    pg.click('#ckp-rail [data-pred="worse"]'); pg.wait_for_timeout(100)
    pg.click('#ckp-rail [data-nav="next"]'); pg.wait_for_timeout(200)
    pg.evaluate("()=>{const i=document.querySelector('#ckp-player input[data-key=\"scrub\"]'); i.value=1000; i.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(300)
    o = pg.inner_text('#ckp-rail [data-obs]')
    check('part 2 text: the two losses at step 600', f"loss is {lossF(w0['final'])}, against {lossF(w0['final0'])}" in o, o[:300])
    check('part 2 text: the learning rate at step 600 is 8 times the uninterrupted one', f"still {round(lr_at(STEPS - 1 - CK) / lr_at(STEPS - 1))} times" in o, o[:600])
    mt = pg.inner_text('#ckp-metrics')
    check('part 2 cards: weights only: up to +103%, 960 / 960 (the data position is lost too), 50 steps redone', 'up to +103%' in mt and '960 / 960' in mt and '\n50\n' in mt, mt)
    for go, k in ((3, 'interval'),):
        pg.click(f'#ckp-rail [data-go="{go}"]'); pg.wait_for_timeout(250)
    for mode in ('sync', 'async'):
        for i, T in enumerate(INTS):
            pg.evaluate(f"()=>{{const i=document.querySelector('#ckp-ic-T-in'); i.value={i}; i.dispatchEvent(new Event('input',{{bubbles:true}}))}}"); pg.wait_for_timeout(30)
            pg.evaluate(f"()=>document.querySelector('#ckp-rail .seg-ctl[data-key=\"mode\"] button[data-v=\"{mode}\"]').click()"); pg.wait_for_timeout(30)
            d = DA if mode == 'async' else d70
            o = pg.inner_text('#ckp-rail [data-obs]').replace('\u00a0', ' ')
            mt = pg.inner_text('#ckp-metrics').replace('\u00a0', ' ')
            want = [pct(waste(d, T * 60, M70, R)), pct(waste(d, tOpt(d, M70), M70, R)), mins(tOpt(d, M70))]
            check(f'part 4 text and cards: {mode} every {T} min: {want}', all(x in o for x in want) and want[0] in mt and f'{d:.0f} s' in mt, (want, o[:200], mt))
            tp = lambda v: 'under 0.1%' if v < 0.0005 else pct(v)
            terms = [tp(d / (T * 60)), tp(T * 60 / (2 * M70)), tp(R / M70)]
            tbl = pg.evaluate("()=>[...document.querySelectorAll('#ckp-rail .ckp-tt tr')].map(r=>r.querySelector('.ckp-tt-v').textContent)")
            stg = pg.evaluate("()=>[...document.querySelectorAll('#ckp-terms [data-term] b')].map(b=>b.textContent)")
            tot = pg.evaluate("()=>{const e=document.querySelector('#ckp-terms .ckp-tsum'); return e?e.textContent:''}").replace('\u00a0', ' ')
            u = lambda t: round(float(t.rstrip('%')) * 10) if t[0].isdigit() else 0
            sym = '=' if sum(u(t) for t in terms) == u(want[0]) else '≈'
            check(f'part 4 (owner b8): the three terms δ/T, T/2M, R/M = {terms}, in the rail’s table and under the picture, {sym} {want[0]} (“≈” only when the shown shares do not add up)', tbl == terms and stg == terms and tot.strip() == f'{sym} {want[0]}', (tbl, stg, tot))
    mb = pg.evaluate("()=>document.getElementById('ckp-mathbody').textContent").replace('\u00a0', ' ')
    sx70 = round(S70 / 1e8) * 1e8; dx70 = sx70 / BW; tx70 = round(math.sqrt(2 * dx70 * M70))
    check('math: each line recomputes from what it shows: 70,553,706,496 × 14 bytes = 0.9878 TB; δ = 0.9878 TB ÷ 10 GB/s = 98.78 s; T* = √(2 × 98.78 × 43,200) = 2,921 s (typeset)',
          f'70,553,706,496 × 14 bytes = {sx70 / 1e12:.4f} TB' in mb and f'δ = {sx70 / 1e12:.4f} TB ÷ 10 GB/s = {dx70:.2f} s' in mb and f'T*=2×{dx70:.2f}×43,200={tx70:,}s' in ''.join(mb.split())
          and pg.evaluate("()=>!!document.querySelector('#ckp-mathbody math msqrt')")   # H1 (2026-10-09): T* is typeset, the root a real MathML root
          and f'{S70 / 1e12:.4f}' == '0.9878' and tx70 == round(tOpt(d70, M70)) == 2921, mb[-500:])
    tips = pg.evaluate("()=>[...document.querySelectorAll('#ckp-metrics .metric')].map(m=>m.dataset.tipx)")
    # H1 (2026-10-09): the best interval's hover is typeset (A.ui.math, kept by the kit's escMath): read its text and its root
    tipsT = pg.evaluate("()=>[...document.querySelectorAll('#ckp-metrics .metric')].map(m=>{const t=document.createElement('template'); t.innerHTML=m.dataset.tipx||''; return (t.content.querySelector('msqrt')?'√':'')+t.content.textContent.replace(/\\s+/g,'')})")
    check('part 4 cards: the hovers of “training stops per save” and “best interval” recompute (async: 10 s, 930 s), the root typeset', any('√T*=2×10s×43,200s=930s' == t for t in tipsT), tipsT)
    # scenarios
    pg.evaluate("()=>document.querySelector('#ckp-app .modes button[data-mode=\"sandbox\"]:not([data-flow=free])').click()"); pg.wait_for_timeout(300)
    t0 = pg.inner_text('#ckp-insight')
    check('scenario 1: 70B async every 15 min wastes 3.5%', pct(waste(DA, 900, M70, R)) in t0 and pct(waste(DA, 900, M70, R)) == '3.5%', t0[:200])
    parts1 = [DA / 900, 900 / (2 * M70), R / M70]; sh1 = [x / sum(parts1) for x in parts1]
    check('scenario 1: saves, redone work and restarts each about a third, restarts a little more (B2)', all(0.25 < x < 0.45 for x in sh1) and max(sh1) == sh1[2]
          and 'split about evenly between saves, redone work and restarts, restarts a little more' in t0 and 'most of it restarts' not in t0, (sh1, t0[:300]))
    pg.click('#ckp-rail [data-preset="1"]'); pg.wait_for_timeout(200)
    t1 = pg.inner_text('#ckp-insight')
    w8 = waste(S8 / BW, 360 * 60, M8, R)
    check(f'scenario 2: 8B sync every 6 h wastes {pct(w8, 2)}; about 110 GB, about ten seconds; best about 6 hours', pct(w8, 2) in t1 and f'about {round(S8 / 1e10) * 10} GB' in t1 and 9.5 < S8 / BW < 15
          and f'about {round(tOpt(S8 / BW, M8) / 3600)} hours' in t1 and round(tOpt(S8 / BW, M8) / 3600) == 6, t1[:300])
    tips = pg.evaluate("()=>[...document.querySelectorAll('#ckp-metrics .metric')].map(m=>m.dataset.tipx)")
    # H1 (2026-10-09): the best interval's hover is typeset (A.ui.math, kept by the kit's escMath): read its text and its root
    tipsT = pg.evaluate("()=>[...document.querySelectorAll('#ckp-metrics .metric')].map(m=>{const t=document.createElement('template'); t.innerHTML=m.dataset.tipx||''; return (t.content.querySelector('msqrt')?'√':'')+t.content.textContent.replace(/\\s+/g,'')})")
    dx8 = round(S8 / 1e8) * 1e8 / BW
    check('scenario 2 cards: 112.4 GB ÷ 10 GB/s = 11.24 s and √(2 × 11.24 s × 19,353,600 s) recompute (the root typeset)', f'112.4 GB ÷ 10 GB/s = {dx8:.2f} s.' in tips and f'√T*=2×{dx8:.2f}s×{M8:,}s={round(math.sqrt(2 * dx8 * M8)):,}s' in tipsT, (tips, tipsT))
    sv = pg.evaluate("()=>[...document.querySelectorAll('#ckp-chart svg text')].map(t=>t.textContent)")
    check('scenario 2: the strip shows 24 hours without a failure (S5)', '24 hours, no failure (illustration)' in sv and not pg.evaluate("()=>!!document.querySelector('#ckp-chart .ckp-fail')"), sv)
    d1 = pg.evaluate("()=>document.querySelector('#ckp-metrics [data-k=\"w\"] .d').textContent")
    check('scenario 2 after scenario 1: no ▲▼ on the wasted share (another run, R6.6)', d1 == '', d1)
    pg.click('#ckp-rail [data-preset="2"]'); pg.wait_for_timeout(200)
    pg.evaluate("()=>{const i=document.querySelector('#ckp-player input[data-key=\"scrub\"]'); i.value=1000; i.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(200)
    t2 = pg.inner_text('#ckp-insight')
    check('scenario 3: 960 read twice in epoch 7, 960 of the last epoch never read, 40 steps behind', f'960 samples were read twice in epoch {CK // EPOCH + 1}' in t2 and '960 samples of the last epoch were never read' in t2 and f'{CK - (CK // EPOCH) * EPOCH} steps behind' in t2 and CK // EPOCH + 1 == 7, t2[:300])
    pg.click('#ckp-rail [data-preset="3"]'); pg.wait_for_timeout(200)
    t3 = pg.inner_text('#ckp-insight')
    check('scenario 4: skip vs latest losses', f"is {lossF(REM['skip']['final'])}, against {lossF(REM['latest']['final'])}" in t3, t3[:300])
    # the challenge, from the page's own model
    pe = J("h.chAll()")
    check('challenge: the page evaluates all 60 plans as Python does', len(pe) == 60 and all(abs(a['w'] - e['w']) < 1e-12 and a['ok'] == e['ok'] for a, e in zip(pe, ALLP)))
    pbst = J("h.chBest()")
    check('challenge: the page picks the same best plan', (pbst['mode'], pbst['T'], pbst['k']) == ('async', 15, 5), pbst)
    pg.evaluate("()=>document.querySelector('#ckp-app .modes button[data-mode=\"challenge\"]').click()"); pg.wait_for_timeout(300)
    stt = pg.inner_text('#ckp-rail').replace('\u00a0', ' ')
    check('challenge statement: 5.68 TB, about 114 s synchronous', '5.68 TB' in stt and f'{S405 / CBW:.0f} s' in stt and f'{S405 / CBW:.0f}' == '114', stt[:400])
    pg.click('#ckp-rail [data-act="submit"]'); pg.wait_for_timeout(200)
    pg.click('#ckp-ch-res [data-act="reveal"]'); pg.wait_for_timeout(200)
    res = pg.inner_text('#ckp-ch-res').replace('\u00a0', ' ')
    ext = START['w'] - best['w']
    gp = f"{float(pct(START['w'])[:-1]) - float(pct(best['w'])[:-1]):.1f}"  # the difference of the two shares as shown
    check(f'challenge result (R1): the start ({pct(START["w"])}) against the best ({pct(best["w"])}): wastes {gp} points more, most of it redone work (half of 1 h lost per failure: 30 min)',
          f'Your plan wastes {gp} points more, most of it on redone work: saving every 1 h, a failure loses 30 min of work on average.' in res and gp == '14.6'
          and (START['w'] - best['w']) > 0 and (START['T'] * 60 / (2 * CM) - best['T'] * 60 / (2 * CM)) > 0.5 * ext, res[:700])
    check('challenge result (R1, G7): no single change meets the goal; the sentence names the 3 settings the best plan changes together', 'No single change meets the goal. The least-waste plan changes 3 settings at once: saving asynchronously, saving every 15 min and keeping 5.' in res, res[:900])
    kfit = int(CST // S405); tmin = math.ceil(60 / (kfit - 1))
    check('challenge result: 30 TB holds only 5 checkpoints, so the interval cannot be shorter than 15 minutes',
          f'30 TB holds only {kfit} checkpoints' in res and f'cannot be shorter than {tmin} minutes' in res and (kfit, tmin) == (5, 15), res[-900:])
    check('challenge result (R1): the list of passing plans is closed by default and counts them', f'See all settings that pass ({len(PASS)})' in res and pg.evaluate("()=>!document.querySelector('#ckp-ch-res details.ckp-all').open"), res[-400:])
    pg.evaluate("()=>{document.querySelector('#ckp-ch-res details.ckp-all').open=true}"); pg.wait_for_timeout(150)
    res = pg.inner_text('#ckp-ch-res').replace('\u00a0', ' ')
    rows = pg.evaluate("()=>[...document.querySelectorAll('#ckp-ch-res details.ckp-all li')].map(l=>l.textContent)")
    check('challenge list (R1): only the passing plans, least waste first, each with its share as Python computes it',
          len(rows) == len(PASS) and all(f"every {e['T']} min, keep {e['k']}: wasted {pct(e['w'])}" in r for r, e in zip(rows, PASS)), rows)
    check('challenge list: the async write takes about 2 minutes, far less than 15; restarts alone cost 5.6%',
          f'takes about {S405 / CBW / 60:.0f} minutes, far less than {best["T"]} minutes' in res and pct(CR / CM) == '5.6%' and 'restarts alone take about 5.6%' in res, res[-900:])
    pp = J("h.chPass()")
    check('challenge: the page’s passing set equals Python’s', [(e['mode'], e['T'], e['k']) for e in pp] == [(e['mode'], e['T'], e['k']) for e in PASS], pp)
    for m_, T_, k_ in (('sync', 60, 2), ('async', 15, 4), ('async', 20, 4), ('async', 10, 2)):
        po = J(f"h.chOne('{m_}',{T_},{k_})")
        check(f'challenge: single changes from {m_} {T_} min keep {k_} that pass, as Python finds them', [(x['e']['mode'], x['e']['T'], x['e']['k']) for x in po] == [(x['mode'], x['T'], x['k']) for x in one(ch(m_, T_, k_))], po)
    ex = J("(()=>{const e=h.exits(); return {t:e.transfer.map(x=>({a:x.a,o:x.o,why:x.why})), q:e.transfer.map(x=>x.q)}})()")
    check('transfer 1: the answer is the 25-minute option; the explanation says about 24 min', ex['t'][0]['a'] == 1 and '25' in ex['t'][0]['o'][1] and '24 min' in ex['t'][0]['why'] and '49 min' in ex['q'][0], ex['t'][0])
    check('transfer 2: the answer is about 300 steps', ex['t'][1]['a'] == 0 and '300' in ex['t'][1]['o'][0], ex['t'][1])
    check('transfer 2: the question starts from this lab: 60 steps per epoch, epoch 7 reread, 40 steps (R3.19)', f'an epoch is {EPOCH} steps' in ex['q'][1] and f'rereads epoch {CK // EPOCH + 1}' in ex['q'][1] and f'{CK - (CK // EPOCH) * EPOCH} steps of data' in ex['q'][1], ex['q'][1])
    pg.evaluate("()=>document.querySelector('#ckp-app [data-act=\"drill-reveal\"]').click()"); pg.wait_for_timeout(100)
    for seed in range(1, 7):
        q = J(f"h.drill({seed})"); e = drill(seed)
        check(f'drill seed {seed}: size, δ, T*, waste', abs(q['size'] - e['size']) < 1 and abs(q['to'] - e['to']) < 1e-6 and abs(q['w'] - e['w']) < 1e-12, (q, e))
        da = pg.inner_text('#ckp-drill-a').replace('\u00a0', ' ')
        sv = round(e['size'] / 1e8) * 1e8; dx = sv / e['bw']; tx_ = round(math.sqrt(2 * dx * e['Mh'] * 3600))
        st = f"{sv / 1e12:.4f} TB" if e['size'] >= 5e11 else f"{sv / 1e9:.1f} GB"
        dxs = f'{dx:.3f}'.rstrip('0').rstrip('.')
        check(f'drill seed {seed}: the answer’s lines recompute from what they show ({st} ÷ {e["bw"] / 1e9:.0f} GB/s = {dxs} s; T* = {tx_:,} s), FP32 copy of the weights',
              f'= {st} (BF16 weights 2, FP32 copy of the weights 4' in da and f'δ = {st} ÷ {e["bw"] / 1e9:.0f} GB/s = {dxs} s, and M = ' in da
              # H1 (the owner's message of 2026-10-09): the T* line is typeset (T* = √(2δM) = √(2 × δ × M) = … s), the root a real MathML root
              # (read through textContent: innerText gives MathML's single letters as math italics, 𝑇 and 𝛿)
              and f'T*=2δM=2×{dxs}×{round(e["Mh"] * 3600):,}={tx_:,}s' in ''.join(pg.evaluate("()=>document.getElementById('ckp-drill-a').textContent").split())
              and pg.evaluate("()=>!!document.querySelector('#ckp-drill-a math msqrt')")
              and abs(tx_ - e['to']) <= 3 and 'master' not in da, da)
        pg.evaluate("()=>document.querySelector('#ckp-app [data-act=\"drill-new\"]').click()"); pg.wait_for_timeout(60)
        pg.evaluate("()=>document.querySelector('#ckp-app [data-act=\"drill-reveal\"]').click()"); pg.wait_for_timeout(60)
    # the teaser
    pg.evaluate("()=>{location.hash=''}"); pg.wait_for_timeout(300)
    tz = pg.evaluate("()=>window.__tzCkp?window.__tzCkp.INTS.map((_,i)=>({...window.__tzCkp.model(i), read:window.__tzCkp.read(i)})):null")
    check('teaser: present', tz is not None)
    if tz:
        for i, m in enumerate(tz):
            T = TI[i]; w = tw(T)
            half = T / 2; hs = (f'{half:g} min' if half < 60 else f'{half / 60:g} h')
            ok = abs(m['w'] - w) < 1e-12 and f'{round(w * 100)}%' in m['read'] and f'{round(tto)} min' in m['read'] and f'{round(tw(tto) * 100)}%' in m['read'] and f'you redo {hs} on average' in m['read'] and 'latest checkpoint' in m['read']
            check(f'teaser at {T} min: resume from the latest checkpoint, redo {hs} on average, {round(w * 100)}%, the least about {round(tw(tto) * 100)}% at about every {round(tto)} min (R11.1)', ok, m)
    # ---- G7 (the owner's message of 2026-10-08, from the model-parallel finding): walk every pick of the challenge, in both
    # languages, and check that a recommended single change makes the value the result shows strictly better at the precision shown
    def g7_walk(js):
        res = {}
        for lang in ('en', 'zh'):
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(150)
            res[lang] = pg.evaluate(js)
        pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(150)
        return res
    g7 = g7_walk("""()=>{const Z=window.__ckp, C=Z.consts.CH; const R=/^The best single change is|^只改一项的话|^One change meets the goal|^只改一项就能达标/; const out=[];
      ['sync','async'].forEach(m=>C.ints.forEach(T=>C.keeps.forEach(k=>{const t=Z.chNextText(m,T,k), s=Z.chShown(m,T,k), rec=R.test(t); let ok=true, why='';
        const nums=(t.match(/([\\d.]+)%/g)||[]).map(x=>parseFloat(x));
        if(rec){if(!nums.length){ok=false; why='no waste in the sentence';}
          else if(s.ok){ok=nums[0]<parseFloat(s.w); why='waste '+s.w+' -> '+nums[0]+'%';} else {ok=nums[0]<=C.goal*100; why='meets the goal at '+nums[0]+'%';}}
        out.push({pick:[m,T,k],rec,ok,why,t});}))); return out;}""")
    for lang in ('en', 'zh'):
        rows = g7[lang]; bad = [r for r in rows if not r['ok']]
        check(f'G7 {lang}: all 60 plans walked; every recommended single change shows less waste (or meets the goal where the plan missed)', len(rows) == 60 and not bad, bad[:3])
    b.close()
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails if fails else 'none')
sys.exit(1 if fails else 0)
