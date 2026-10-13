"""Numbers test for the optimizers lab (CONTRACT §6).

An independent Python version of the lab's engine (PyTorch's update rules: SGD, heavy-ball momentum, Adam with bias correction that
can be switched off, AdamW with decoupled weight decay, warmup then cosine to 10%) runs every setting the page shows and is compared
with the page's own runs (window.__op):
- the valley ½(x² + 25 y²) from (−9, 1.5): the mission's 104 / 78 / 74 steps; part 3's sweep with SGD's limit 2 ÷ 25 = 0.08 (bounces
  forever at 0.08, diverges above it); part 4's β sweep (fewest steps at β 0.65, β 0.9 overshoots); part 5's Adam sweep with its
  first step of lr in x and in y (at SGD's 0.05: Adam 0.05 in x and y, SGD 0.45 in x and 1.88 in y; Adam first beats SGD's 104 steps
  at 0.5, ties at 0.3); part 6's first step without bias correction, (1 − β1) ÷ √(1 − β2) = 3.16 × lr;
- part 7 (AdamW): the share of a weight of 1.0 that weight decay removes per step once m and v have settled, from a direct Adam
  simulation (gradients flipping ±s, 20,000 steps): Adam + L2 0.010% (s = 10) and 1.0% (s = 0.01), about 100 times apart; AdamW 0.10%
  for both (lr × weight decay = 0.01 × 0.1); and the page's wdPull agrees;
- the four scenarios (Llama 2's AdamW with β2 0.95, ε 10⁻⁵, weight decay 0.1 and warmup; ResNet's momentum; Adam's defaults; the
  curved valley) and the warmup-cosine schedule;
- the challenge: all 18 pairs on ½(x² + 100 y²) from (−5, 0.8), the smallest passing change (momentum 0.9 at 0.01, 91 steps), and a
  margin of at least 5 steps from the goal of 100 for every pair;
- the capability layer: the explain answers (−1.5 per step; m ÷ √v about 3 after one step) and the transfer answers (2 ÷ 50 = 0.04;
  Adam's first step unchanged when the loss is 10 times larger);
- the interview drill's limit and step count against a direct simulation, for 40 seeds;
- the Connect code, run as printed, against its own comments (# 104 diverges, # 30 74);
- the home tile (window.__tzOp): SGD and Adam with the lab's part 1 settings, their paths point by point (104 and 74 steps), inside the
  view; the sentence's claims (SGD zig-zags and crawls, Adam does neither), the notes' facts (69 steps at 0.075: Adam is not always
  faster), short sentences in EN and ZH, and nothing of part 2's answer or of the challenge (R3.8).
Run: python3 tests/optimizers_numbers.py"""
import math, pathlib, re, subprocess, sys, io, contextlib
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/optimizers-numbers.html', '--only', 'optimizers'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'optimizers-numbers-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"></head><body>' + (P / 'out' / 'optimizers-numbers.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'optimizers-numbers-wrapped.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


# ---------------- the reference engine ----------------
N, GOAL, BLOW = 200, 1e-3, 1e8
SURF = {
    'v25': (lambda p: 0.5 * (p[0] ** 2 + 25 * p[1] ** 2), lambda p: [p[0], 25 * p[1]], [-9.0, 1.5]),
    'v100': (lambda p: 0.5 * (p[0] ** 2 + 100 * p[1] ** 2), lambda p: [p[0], 100 * p[1]], [-5.0, 0.8]),
    'v50': (lambda p: 0.5 * (p[0] ** 2 + 50 * p[1] ** 2), lambda p: [p[0], 50 * p[1]], [-9.0, 1.5]),
    'curve': (lambda p: (1 - p[0]) ** 2 + 10 * (p[1] - p[0] ** 2) ** 2, lambda p: [-2 * (1 - p[0]) - 40 * p[0] * (p[1] - p[0] ** 2), 20 * (p[1] - p[0] ** 2)], [-1.2, 1.0]),
}


def lr_at(c, t):
    if not c.get('warm'): return c['lr']
    if t <= c['warm']: return c['lr'] * t / c['warm']
    q = (t - c['warm']) / (N - c['warm'])
    return c['lr'] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * q)))


def run(c):
    f, g, start = SURF[c.get('surf', 'v25')]
    b1, b2, eps, bc, sc = c.get('b1', 0.9), c.get('b2', 0.999), c.get('eps', 1e-8), c.get('bc', True), c.get('scale', 1)
    x = list(start); v = [0, 0]; m = [0, 0]; s = [0, 0]; hit = div = None; d = [[0, 0]]; loss = [f(x) * sc]
    for t in range(1, N + 1):
        gr = [q * sc for q in g(x)]; lr = lr_at(c, t)
        if c['opt'] == 'sgd': u = [lr * q for q in gr]
        elif c['opt'] == 'mom':
            v = [c['beta'] * a + q for a, q in zip(v, gr)]; u = [lr * a for a in v]
        else:
            m = [b1 * a + (1 - b1) * q for a, q in zip(m, gr)]; s = [b2 * a + (1 - b2) * q * q for a, q in zip(s, gr)]
            mh = [a / (1 - b1 ** t) if bc else a for a in m]; sh = [a / (1 - b2 ** t) if bc else a for a in s]
            u = [lr * a / (math.sqrt(b) + eps) for a, b in zip(mh, sh)]
            if c['opt'] == 'adamw' and c.get('wd'): u = [a + lr * c['wd'] * xi for a, xi in zip(u, x)]
        x = [xi - a for xi, a in zip(x, u)]; fx = f(x) * sc; d.append(u); loss.append(fx)
        if not math.isfinite(fx) or fx > BLOW: div = t; break
        if hit is None and fx < GOAL * sc: hit = t
    return {'hit': hit, 'div': div, 'd': d, 'loss': loss, 'x': x}


with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(); errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    C = pg.evaluate('()=>window.__op.consts')
    J = lambda c: pg.evaluate('c=>window.__op.run(c)', c)

    def same(c, tag):
        a, r = J(c), run(c)
        ok = a['hit'] == r['hit'] and a['div'] == r['div']
        if ok and len(a['d']) > 1 and len(r['d']) > 1:
            ok = all(abs(x - y) <= 1e-9 * max(1, abs(y)) for x, y in zip(a['d'][1], r['d'][1]))
        check(f'{tag}: the page and this engine agree ({r["hit"]} / diverges {r["div"]})', ok, (a['hit'], a['div'], r['hit'], r['div']))
        return r

    check('constants: 200 steps, goal 0.001', C['N'] == 200 and C['GOAL'] == 1e-3, C)
    # ---------- the mission
    s05, m09, a05 = run({'opt': 'sgd', 'lr': 0.05}), run({'opt': 'mom', 'lr': 0.05, 'beta': 0.9}), run({'opt': 'adam', 'lr': 0.5})
    check('mission: SGD 104, momentum 0.9 78, Adam 74 steps', (s05['hit'], m09['hit'], a05['hit']) == (104, 78, 74), (s05['hit'], m09['hit'], a05['hit']))
    for c in ({'opt': 'sgd', 'lr': 0.05}, {'opt': 'mom', 'lr': 0.05, 'beta': 0.9}, {'opt': 'adam', 'lr': 0.5}): same(c, f'mission {c}')
    # ---------- part 3: SGD's sweep and its limit 2 ÷ 25
    want = {0.005: None, 0.01: None, 0.02: None, 0.03: 175, 0.04: 130, 0.05: 104, 0.06: 86, 0.07: 74, 0.075: 69}
    for lr in C['LR_PART']:
        r = same({'opt': 'sgd', 'lr': lr}, f'part 3 SGD lr {lr}')
        if lr in want: check(f'part 3: SGD at {lr} takes {want[lr]}', r['hit'] == want[lr] and r['div'] is None, r['hit'])
        if lr > 0.08 + 1e-12: check(f'part 3: SGD at {lr} diverges (above 2 ÷ 25 = 0.08)', r['div'] is not None and r['hit'] is None, r['div'])
        if lr < 0.08 - 1e-12: check(f'part 3: SGD at {lr} does not diverge', r['div'] is None)
    r08 = run({'opt': 'sgd', 'lr': 0.08})
    check('part 3: at exactly 0.08 y is multiplied by −1: no bottom and no divergence in 200 steps', r08['hit'] is None and r08['div'] is None and abs(1 - 25 * 0.08 + 1) < 1e-12)
    check('part 3: 1 − 25 × 0.10 = −1.5 (the text and the explain question)', abs((1 - 25 * 0.1) + 1.5) < 1e-12)
    r10 = run({'opt': 'sgd', 'lr': 0.1})
    check('part 2’s answer: SGD at 0.10 never reaches the bottom (the prediction “never”)', r10['hit'] is None and r10['div'] is not None and C['PRED_RIGHT'] == 'never', r10['div'])
    # ---------- part 4: momentum's β sweep at lr 0.05
    hits = {}
    for be in C['BETAS']:
        r = same({'opt': 'mom', 'lr': 0.05, 'beta': be}, f'part 4 β {be}'); hits[be] = r['hit']
    check('part 4: β 0 is plain SGD (104 steps)', hits[0] == 104)
    best = min(h for h in hits.values() if h is not None)
    check('part 4: the fewest steps come at β 0.65 (20); β 0.6 and 0.7 take 30; β 0.9 overshoots (78)', best == 20 and hits[0.65] == 20 and hits[0.6] == 30 and hits[0.7] == 30 and hits[0.9] == 78 and all(hits[be] > 30 for be in hits if be < 0.6 or be > 0.75), hits)
    check('part 4: every β from 0.05 to 0.95 beats plain SGD or at least reaches the bottom', all(h is not None for h in hits.values()), hits)
    # ---------- part 5: Adam's sweep; the first step is lr in x and in y
    for lr in C['LR_ADAM']:
        r = same({'opt': 'adam', 'lr': lr}, f'part 5 Adam lr {lr}')
        check(f'part 5: Adam at {lr}: the first step moves x and y by the learning rate', abs(abs(r['d'][1][0]) - lr) < 1e-6 * lr and abs(abs(r['d'][1][1]) - lr) < 1e-6 * lr, r['d'][1])
    check('part 5: SGD’s first step moves y 4.2 times as far as x (25 × 1.5 ÷ 9)', abs(abs(s05['d'][1][1]) / abs(s05['d'][1][0]) - 25 * 1.5 / 9) < 1e-12)
    check('part 5: Adam at 0.2 / 0.3 / 1.0 takes 112 / 104 / 76; momentum β 0.95 takes 147', (run({'opt': 'adam', 'lr': 0.2})['hit'], run({'opt': 'adam', 'lr': 0.3})['hit'], run({'opt': 'adam', 'lr': 1.0})['hit'], hits[0.95]) == (112, 104, 76, 147))
    check('part 5: Adam at 0.05 and 0.1 does not reach the bottom in 200 steps, at 0.5 it takes 74', run({'opt': 'adam', 'lr': 0.05})['hit'] is None and run({'opt': 'adam', 'lr': 0.1})['hit'] is None and a05['hit'] == 74)
    # ---------- part 6: bias correction
    off = run({'opt': 'adam', 'lr': 0.5, 'bc': False}); same({'opt': 'adam', 'lr': 0.5, 'bc': False}, 'part 6 no bias correction')
    k1 = abs(off['d'][1][0]) / 0.5
    check('part 6: without bias correction the first step is (1 − 0.9) ÷ √(1 − 0.999) = 3.16 × lr', abs(k1 - 0.1 / math.sqrt(0.001)) < 1e-6 and f'{k1:.2f}' == '3.16', k1)
    check('part 6: with it, 1.00 × lr', abs(abs(a05['d'][1][0]) / 0.5 - 1) < 1e-6)
    big = max(abs(u[0]) / 0.5 for u in off['d'][1:11])
    check('part 6: the first 10 steps reach about 4 times lr without it (largest step bigger than the first)', big > k1 and 3.5 < big < 5, big)
    check('part 6: without bias correction Adam at 0.5 still reaches the bottom (88 steps)', off['hit'] == 88, off['hit'])
    check('part 8: SGD at 0.01 is still above 0.001 after 200 steps (the misconception)', run({'opt': 'sgd', 'lr': 0.01})['hit'] is None)
    # ---------- part 5 at SGD's learning rate (owner check 2026-10-03, point 2): the first steps, and where Adam first beats SGD
    a005 = run({'opt': 'adam', 'lr': 0.05})
    check('part 5 at 0.05: Adam’s first step is 0.05 in x and y; SGD’s is 0.45 in x (0.05 × 9) and 1.88 in y (0.05 × 25 × 1.5)',
          abs(abs(a005['d'][1][0]) - 0.05) < 1e-9 and abs(abs(a005['d'][1][1]) - 0.05) < 1e-9 and f"{abs(s05['d'][1][0]):.2f}" == '0.45' and f"{abs(s05['d'][1][1]):.2f}" == '1.88', (a005['d'][1], s05['d'][1]))
    first = min(lr for lr in C['LR_ADAM'] if (run({'opt': 'adam', 'lr': lr})['hit'] or 999) < s05['hit'])
    check('part 5: on the slider, Adam first beats SGD’s 104 steps at 0.5 (“about 0.5”); 0.3 ties at 104', first == 0.5 and run({'opt': 'adam', 'lr': 0.3})['hit'] == 104, first)
    # ---------- part 7: AdamW against Adam + L2, by a direct simulation of Adam on one weight held at 1.0
    def wd_sim(kind, sgrad, lr=0.01, wd=0.1, th=1.0, T=20000, b1=0.9, b2=0.999, eps=1e-8):
        m = v = 0.0; us = []
        for t in range(1, T + 1):
            g = sgrad * (1 if t % 2 else -1) + (wd * th if kind == 'l2' else 0.0)
            m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
            u = lr * (m / (1 - b1 ** t)) / (math.sqrt(v / (1 - b2 ** t)) + eps) + (lr * wd * th if kind == 'adamw' else 0.0)
            us.append(u)
        return (us[-1] + us[-2]) / 2 / th   # the share removed per step: the mean over one flip
    W = C['WD']
    check('part 7: lr 0.01, weight decay 0.1, a weight of 1.0, gradients ±10 and ±0.01', (W['lr'], W['wd'], W['th'], W['big'], W['small']) == (0.01, 0.1, 1, 10, 0.01), W)
    pct = lambda v: f'{float(f"{v * 100:.2g}"):g}'
    sims = {(k, sv): wd_sim(k, sv) for k in ('l2', 'adamw') for sv in (10, 0.01)}
    page = {(k, sv): pg.evaluate('([k,s])=>window.__op.wdPull(k,s)', [k, sv]) for k in ('l2', 'adamw') for sv in (10, 0.01)}
    check('part 7: the page’s shares agree with the simulation (within 1%)', all(abs(page[q] - sims[q]) <= 0.01 * sims[q] for q in sims), (page, sims))
    check('part 7: Adam + L2 removes 0.010% (gradients ±10) and 1.0% (±0.01) per step; AdamW 0.10% for both',
          [f'{page[q] * 100:.2g}' for q in (('l2', 10), ('l2', 0.01), ('adamw', 10), ('adamw', 0.01))] == ['0.01', '1', '0.1', '0.1'] and f"{page[('l2', 10)] * 100:.2}" == '0.01', [page[q] * 100 for q in page])
    check('part 7: about 100 times apart under Adam + L2 (99.5), the same under AdamW (lr × weight decay = 0.001)', round(page[('l2', 0.01)] / page[('l2', 10)] / 10) * 10 == 100 and abs(page[('adamw', 10)] - 0.001) < 1e-15 and abs(page[('adamw', 0.01)] - 0.001) < 1e-15, page)
    # ---------- round-1 fixes: what momentum's gain is, on this valley (review A1), checked against numpy's roots and a simulation
    import numpy as np
    def roots_rate(lam, lr, be):   # heavy ball on one direction: r² − (1 + β − lr·λ) r + β = 0; β = 0 is SGD's 1 − lr·λ
        if be == 0: return abs(1 - lr * lam), False
        r = np.roots([1, -(1 + be - lr * lam), be]); return max(abs(r)), bool(np.iscomplexobj(r) and abs(r[0].imag) > 1e-12)
    def sim_rate(lam, lr, be, t0=150, t1=160):   # the x-only recursion, no goal: the per-step factor late in the run
        th, v, xs = -9.0, 0.0, []
        for t in range(t1 + 1):
            v = be * v + lam * th; th -= lr * v; xs.append(th)
        return (abs(xs[t1]) / abs(xs[t0])) ** (1 / (t1 - t0))
    sx = roots_rate(1, 0.05, 0)[0]; sy = roots_rate(25, 0.05, 0)[0]
    check('part 4: plain SGD at 0.05 multiplies x by 0.95 and y by −0.25 per step (its zig-zag is gone after a few steps)', abs(sx - 0.95) < 1e-12 and abs(sy - 0.25) < 1e-12)
    bad = []
    for be in C['BETAS']:
        if be == 0: continue
        px, py = J({'opt': 'sgd', 'lr': 0.05}), None
        jx = pg.evaluate('([a,b,c])=>window.__op.rate(a,b,c)', [1, 0.05, be]); jy = pg.evaluate('([a,b,c])=>window.__op.rate(a,b,c)', [25, 0.05, be])
        rx, ox = roots_rate(1, 0.05, be); ry, oy = roots_rate(25, 0.05, be)
        if abs(jx - rx) > 1e-9 or abs(jy - ry) > 1e-9: bad.append(('page rate', be, jx, rx, jy, ry))
        if be < 0.85 and not rx < sx - 1e-9: bad.append(('x not faster', be, rx))
        if be <= 0.6 and abs(sim_rate(1, 0.05, be) - rx) > 2e-3: bad.append(('simulation', be, sim_rate(1, 0.05, be), rx))
        if (ry > sy + 1e-9) != (be >= 0.1): bad.append(('y swing slower exactly from β 0.1', be, ry))
        if be >= 0.85 and not ox: bad.append(('x does not swing at β ≥ 0.85', be))
    check('part 4: every β below 0.85 shrinks x faster than SGD; the swing across dies more slowly from β 0.1 on (not at 0.05); β ≥ 0.85 swings along x too', not bad, bad[:3])
    bb = pg.evaluate('()=>window.__op.bestBeta()')
    check('part 4: the fewest steps at β 0.65, where x and y shrink at the same rate (√0.65)', bb == 0.65 and abs(roots_rate(1, 0.05, 0.65)[0] - roots_rate(25, 0.05, 0.65)[0]) < 1e-12 and abs(roots_rate(1, 0.05, 0.65)[0] - math.sqrt(0.65)) < 1e-12, bb)
    check('part 4: at β 0.6 x shrinks by × 0.80 per step instead of × 0.95', f"{roots_rate(1, 0.05, 0.6)[0]:.2f}" == '0.80')
    check('part 4: at β 0.05 the factors differ only in the third decimal (0.947 against 0.950)', f"{roots_rate(1, 0.05, 0.05)[0]:.2f}" == '0.95' and f"{roots_rate(1, 0.05, 0.05)[0]:.3f}" == '0.947')
    bg = pg.evaluate('()=>window.__op.biggest(0.5,false)')
    mine = max(((t, i, abs(off['d'][t][i]) / 0.5) for t in range(1, 11) for i in (0, 1)), key=lambda q: q[2])
    check('part 6: the largest of the first 10 steps without bias correction is step 4, in x, 4.9 × lr', (bg['t'], bg['i']) == (mine[0], mine[1]) == (4, 0) and f"{mine[2]:.1f}" == '4.9' and abs(bg['k'] - mine[2]) < 1e-9, (bg, mine))
    check('part 6: after one step m ÷ √v = 0.1 ÷ √0.001 = 3.16 (√0.001 = 0.0316)', f"{0.1 / math.sqrt(0.001):.2f}" == '3.16' and f"{math.sqrt(0.001):.4f}" == '0.0316')
    worst = max(max(abs(u[0]), abs(u[1])) / lr for lr in C['LR_ADAM'] for u in run({'opt': 'adam', 'lr': lr})['d'][1:])
    check('Adam (B7): with bias correction no step on the valley is longer than lr', worst <= 1 + 1e-9, worst)
    check('part 3 and 5: SGD’s first step is 25 × 1.5 ÷ 9 = 4.2 times longer in y than in x; y starts 6 times closer to the bottom', f"{abs(s05['d'][1][1]) / abs(s05['d'][1][0]):.1f}" == '4.2' and 9 / 1.5 == 6)
    cx = roots_rate(1, 0.01, 0.9)[0]
    check('challenge why: momentum 0.9 at 0.01 shrinks x by × 0.95 per step; SGD below its limit 0.02 by at most 2% per step', cx < 0.98 and abs(1 - 0.02 - 0.98) < 1e-12 and roots_rate(100, 0.01, 0.9)[1], cx)
    check('part 8: Adam keeps 2 extra numbers per parameter = 8 bytes in FP32', 2 * 4 == 8)
    check('the parts: eight, AdamW seventh', C['STEP_IDS'] == ['mission', 'predict', 'sgd', 'mom', 'adam', 'bias', 'adamw', 'remember'], C['STEP_IDS'])
    # ---------- scenarios
    pre = {x['id']: x for x in C['PRESETS']}
    check('scenarios: four, in order Llama 2, ResNet, Adam’s defaults, the curved valley', [x['id'] for x in C['PRESETS']] == ['llama', 'resnet', 'adam', 'curve'])
    ll = pre['llama']
    check('Llama 2: AdamW, β2 0.95, weight decay 0.1, warmup 20 steps on the valley', (ll['opt'], ll['b2'], ll['wd'], ll['warm'], ll['bc']) == ('adamw', 0.95, 0.1, 20, True), ll)
    rl = same({'opt': 'adamw', 'lr': 0.5, 'b2': 0.95, 'eps': 1e-5, 'wd': 0.1, 'warm': 20, 'surf': 'v25'}, 'scenario Llama 2')
    check('Llama 2 scenario reaches the bottom', rl['hit'] is not None, rl['hit'])
    check('the schedule: lr rises linearly over 20 steps, peaks at step 20, ends at 10% of the peak at step 200',
          abs(lr_at({'lr': 0.5, 'warm': 20}, 1) - 0.025) < 1e-12 and abs(lr_at({'lr': 0.5, 'warm': 20}, 20) - 0.5) < 1e-12 and abs(lr_at({'lr': 0.5, 'warm': 20}, 200) - 0.05) < 1e-12)
    pl = J({'opt': 'adamw', 'lr': 0.5, 'b2': 0.95, 'eps': 1e-5, 'wd': 0.1, 'warm': 20, 'surf': 'v25'})
    check('the page uses the same schedule', abs(pl['lrs'][1] - 0.025) < 1e-12 and abs(pl['lrs'][20] - 0.5) < 1e-12 and (pl['n'] < 200 or abs(pl['lrs'][200] - 0.05) < 1e-12), pl['lrs'][:3])
    rr = same({'opt': 'mom', 'lr': 0.02, 'beta': 0.9, 'surf': 'v25'}, 'scenario ResNet')
    check('ResNet scenario: momentum 0.9 at 0.02 reaches the bottom', rr['hit'] is not None and pre['resnet']['opt'] == 'mom' and pre['resnet']['beta'] == 0.9, rr['hit'])
    same({'opt': 'adam', 'lr': 0.5, 'b2': 0.999, 'surf': 'v25'}, 'scenario Adam’s defaults')
    rc = same({'opt': 'mom', 'lr': 0.01, 'beta': 0.9, 'surf': 'curve'}, 'scenario curved valley (momentum)')
    ra = same({'opt': 'adam', 'lr': 0.2, 'surf': 'curve'}, 'scenario curved valley (Adam 0.2)')
    check('curved valley: momentum 0.9 at 0.01 takes 82 steps, Adam at 0.2 takes 65', (rc['hit'], ra['hit']) == (82, 65), (rc['hit'], ra['hit']))
    check('curved valley: bottom at (1, 1), loss 0 there', SURF['curve'][0]([1, 1]) == 0 and C['SURF']['curve']['min'] == [1, 1])
    # ---------- the challenge
    CH = C['CH']
    check('challenge: valley ½(x² + 100 y²) from (−5, 0.8), goal 100 steps, initial setup SGD 0.01', C['SURF']['v100']['start'] == [-5, 0.8] and CH['max'] == 100 and CH['start'] == {'opt': 'sgd', 'lr': 0.01})
    grid = {}
    for o in CH['opts']:
        for lr in CH['lrs']:
            c = {'opt': o, 'lr': lr, 'beta': CH['beta'], 'surf': 'v100'}
            r = same(c, f'challenge {o} {lr}'); e = pg.evaluate('([o,l])=>window.__op.chEval(o,l)', [o, lr])
            grid[(o, lr)] = r['hit'] if r['div'] is None else 'div'
            ok = r['hit'] is not None and r['hit'] <= 100
            check(f'challenge {o} {lr}: pass/fail agrees', e['ok'] == ok and e['steps'] == r['hit'], (e, r['hit']))
            if r['hit'] is not None: check(f'challenge {o} {lr}: at least 5 steps from the goal ({r["hit"]})', abs(r['hit'] - 100) >= 5, r['hit'])
    print('challenge grid:', grid)
    check('challenge: SGD never passes (it diverges above 2 ÷ 100 = 0.02)', all(not (isinstance(grid[('sgd', lr)], int) and grid[('sgd', lr)] <= 100) for lr in CH['lrs']) and all(grid[('sgd', lr)] == 'div' for lr in CH['lrs'] if lr > 0.02))
    check('challenge: Adam passes from 0.1 up', all(isinstance(grid[('adam', lr)], int) and grid[('adam', lr)] <= 100 for lr in CH['lrs'] if lr >= 0.1) and not any(isinstance(grid[('adam', lr)], int) and grid[('adam', lr)] <= 100 for lr in CH['lrs'] if lr < 0.1), grid)
    best = pg.evaluate('()=>window.__op.chBest()')
    check('challenge: the smallest passing change is momentum 0.9 at 0.01 (one change, 91 steps)', best['o'] == 'mom' and best['lr'] == 0.01 and best['steps'] == 91 and best['k'][0] == 1, best)
    check('challenge: the initial setup fails (not in 200 steps)', grid[('sgd', 0.01)] is None)
    # ---------- capability layer
    ex = pg.evaluate('()=>window.__op.exits()')
    check('transfer 1: answer “about 0.04” (index 1)', ex['transferA'][0] == 1)
    r50a, r50b = run({'opt': 'sgd', 'lr': 0.039, 'surf': 'v50'}), run({'opt': 'sgd', 'lr': 0.041, 'surf': 'v50'})
    check('transfer 1: on ½(x² + 50 y²) SGD is stable at 0.039 and diverges at 0.041', r50a['div'] is None and r50b['div'] is not None, (r50a['div'], r50b['div']))
    check('transfer 2: answer “the same” (index 0)', ex['transferA'][1] == 0)
    a10 = J({'opt': 'adam', 'lr': 0.5, 'scale': 10}); a1 = J({'opt': 'adam', 'lr': 0.5})
    check('transfer 2: with the loss 10 times larger Adam’s first step is still 0.50', abs(abs(a10['d'][1][0]) - 0.5) < 1e-6 and abs(abs(a1['d'][1][0]) - 0.5) < 1e-6, (a10['d'][1], a1['d'][1]))
    check('transfer 2: SGD on the 10 times larger loss diverges at 0.05', run({'opt': 'sgd', 'lr': 0.05, 'scale': 10})['div'] is not None)
    g = 3.0; m1 = 0.1 * g; v1 = 0.001 * g * g
    check('explain 2: after one step m ÷ √v is about 3 (the corrected value is 1)', abs(m1 / math.sqrt(v1) - 3.162) < 0.001 and abs((m1 / 0.1) / math.sqrt(v1 / 0.001) - 1) < 1e-12)
    # ---------- drill
    bad = []
    for sd in range(1, 41):
        d = pg.evaluate('s=>window.__op.drill(s)', sd)
        x, t = float(d['x0']), 0
        while 0.5 * x * x >= 1e-3 and t < 100000: x *= (1 - d['lr']); t += 1
        ok = abs(d['max'] - 2 / d['b']) < 1e-12 and d['steps'] == t and d['lr'] < 2 / d['b']
        y, yd = 1.0, False
        for _ in range(400): y *= (1 - d['b'] * d['lr'])
        if abs(y) > 1: yd = True
        if not ok or yd: bad.append((sd, d, t))
    check('drill: 40 seeds, the limit 2 ÷ b, the step count of x alone, a stable learning rate', not bad, bad[:2])
    # ---------- the Connect code, run as printed
    code = C['CODE']
    out = io.StringIO()
    with contextlib.redirect_stdout(out): exec(compile(code, 'connect', 'exec'), {})
    want_lines = [ln[2:] for ln in code.splitlines() if ln.startswith('# ')]
    check('Connect code: its output matches its comments (104 diverges / 30 74)', out.getvalue().split('\n')[:2] == want_lines == ['104 diverges', '30 74'], (out.getvalue(), want_lines))
    check('Connect code: lines at most 74 characters', max(len(ln) for ln in code.splitlines()) <= 74, max(len(ln) for ln in code.splitlines()))
    # ---------- the home tile (window.__tzOp): “What does Adam do that SGD does not?” answered with the lab's part 1 runs of SGD and
    # Adam on the same valley (review home-v4-A #1). Every number and every claim of its sentence, notes and picture, against this engine.
    tz = pg.evaluate('()=>window.__tzOp?{opts:window.__tzOp.opts, view:window.__tzOp.view, H:window.__tzOp.H}:null')
    check('tile: the hook exists', tz is not None)
    if tz:
        def path(c):   # this engine's points from the start to the step that reaches the bottom (as the lab and the tile draw them)
            f, g, start = SURF['v25']; x = list(start); pts = [x[:]]; m = [0, 0]; s = [0, 0]
            for t in range(1, N + 1):
                gr = g(x)
                if c['opt'] == 'sgd': u = [c['lr'] * q for q in gr]
                else:
                    m = [0.9 * a + 0.1 * q for a, q in zip(m, gr)]; s = [0.999 * a + 0.001 * q * q for a, q in zip(s, gr)]
                    u = [c['lr'] * (a / (1 - 0.9 ** t)) / (math.sqrt(b / (1 - 0.999 ** t)) + 1e-8) for a, b in zip(m, s)]
                x = [xi - a for xi, a in zip(x, u)]; pts.append(x[:])
                if f(x) < GOAL: return pts
            return pts
        plain = lambda h: pg.evaluate("h=>{const d=document.createElement('div'); d.innerHTML=h; return d.textContent.replace(/\\u00a0/g,' ')}", h)
        check('tile: the control is SGD / Adam, the comparison in its title', tz['opts'] == ['sgd', 'adam'], tz['opts'])
        want = {'sgd': ({'opt': 'sgd', 'lr': 0.05}, 104), 'adam': ({'opt': 'adam', 'lr': 0.5}, 74)}
        P_ = {}
        for o in tz['opts']:
            cfg = pg.evaluate('o=>window.__tzOp.cfg(o)', o); t = pg.evaluate('o=>window.__tzOp.run(o)', o); lab = J(cfg); mine = path(want[o][0]); P_[o] = mine
            check(f'tile {o}: the lab’s part 1 setting {want[o][0]} (SGD05 / ADAM05)', cfg == want[o][0], cfg)
            check(f'tile {o}: {want[o][1]} steps, as the lab’s part 1, the lab’s run() and this engine say', t['hit'] == lab['hit'] == run(cfg)['hit'] == len(mine) - 1 == want[o][1] and t['n'] == t['hit'], (t['hit'], lab['hit'], len(mine) - 1))
            check(f'tile {o}: the drawn path is this engine’s, point by point', len(t['pts']) == len(mine) and all(abs(a - b) <= 1e-9 * max(1, abs(b)) for p_, q_ in zip(t['pts'], mine) for a, b in zip(p_, q_)), (t['pts'][:2], mine[:2]))
            check(f'tile {o}: no diverging run (R3.8)', t['div'] is None and lab['div'] is None)
        check('tile: the part 1 numbers are the mission’s (SGD 104, Adam 74)', (s05['hit'], a05['hit']) == (104, 74))
        check('tile: SGD’s learning rate is below 2 ÷ 25 = 0.08, so the tile never shows part 2’s answer (R3.8)', want['sgd'][0]['lr'] < 2 / 25)
        # the picture: every point and every contour inside the view (the product owner flagged ellipses cut at the frame in v3)
        x0, x1, y0, y1 = tz['view']
        check('tile: the view is the v4 one, [−10, 10] × [−2, 2]', tz['view'] == [-10, 10, -2, 2] and tz['H'] <= 130, tz)
        check('tile: both paths stay inside the view', all(x0 < p_[0] < x1 and y0 < p_[1] < y1 for o in P_ for p_ in P_[o]), [(min(p_[0] for p_ in P_[o]), max(p_[0] for p_ in P_[o]), min(p_[1] for p_ in P_[o]), max(p_[1] for p_ in P_[o])) for o in P_])
        check('tile: the largest contour, loss 32, is 8 by 1.6 around the bottom, inside the view', math.sqrt(64) < x1 and math.sqrt(64 / 25) < y1)
        # the sentence's claims: SGD zig-zags (y flips sign on every step) and crawls (its x step shrinks by 5% each step); Adam does neither
        ys, ya = [p_[1] for p_ in P_['sgd']], [p_[1] for p_ in P_['adam']]
        fs = [t for t in range(1, len(ys)) if ys[t] * ys[t - 1] < 0]; fa = [t for t in range(1, len(ya)) if ya[t] * ya[t - 1] < 0]
        check('tile SGD: y flips sign on every step, 1 − 25 × 0.05 = −0.25 (the zig-zag), and the first swings are visible (41, 10, 2.6 px of 110)', fs == list(range(1, 105)) and abs(1 - 25 * 0.05 + 0.25) < 1e-12 and abs(ys[1]) / 4 * 110 > 10, fs[:5])
        check('tile Adam: y never flips sign on two steps in a row (it crosses the floor 9 times, at least 8 steps apart): no zig-zag', fa and min(b - a for a, b in zip(fa, fa[1:])) >= 8 and len(fa) == 9, fa)
        dxs = [abs(P_['sgd'][t][0] - P_['sgd'][t - 1][0]) for t in range(1, len(P_['sgd']))]; dxa = [abs(P_['adam'][t][0] - P_['adam'][t - 1][0]) for t in range(1, len(P_['adam']))]
        k90 = lambda P: next(t for t, p_ in enumerate(P) if abs(p_[0]) <= 0.9)
        check('tile SGD crawls: each x step is 0.95 times the last (0.45 down to 0.002); Adam’s x step stays at least 0.4 for 14 steps and it covers 90% of the valley’s length in 19 steps, SGD in 45',
              all(abs(dxs[i + 1] / dxs[i] - 0.95) < 1e-9 for i in range(len(dxs) - 1)) and min(dxa[:14]) >= 0.4 and (k90(P_['adam']), k90(P_['sgd'])) == (19, 45), (dxs[0], dxs[-1], min(dxa[:14]), k90(P_['adam']), k90(P_['sgd'])))
        q075 = run({'opt': 'sgd', 'lr': 0.075})['hit']
        check('tile note: Adam is not always faster, SGD at 0.075 reaches the bottom in 69 steps (the lab’s part 3 agrees)', q075 == 69 < 74 and J({'opt': 'sgd', 'lr': 0.075})['hit'] == 69, q075)
        check('tile note: Adam’s learning rate is about the largest step a parameter takes (no step longer than 0.5 on this run)', max(max(abs(u[0]), abs(u[1])) for u in a05['d'][1:]) <= 0.5 + 1e-9)
        for lang in ('en', 'zh'):
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(250)
            rs, ra = plain(pg.evaluate("()=>window.__tzOp.read('sgd')")), plain(pg.evaluate("()=>window.__tzOp.read('adam')"))
            notes = pg.evaluate('()=>window.__tzOp.notes()')
            if lang == 'en':
                check('tile EN: each sentence at most 35 words, one sentence (R11.5)', all(len(x.split()) <= 35 and x.endswith('.') and '. ' not in x for x in (rs, ra)), [(len(x.split()), x) for x in (rs, ra)])
                check('tile EN SGD: learning rate 0.05, zig-zags, crawls, 104 steps', all(k in rs for k in ('learning rate 0.05', 'zig-zags', 'crawls', '104 steps')), rs)
                check('tile EN Adam: its own step size, learning rate 0.5, neither zig-zags nor crawls, 74 steps against SGD at 0.05 in 104, never “faster”', all(k in ra for k in ('its own step size', 'learning rate 0.5', 'neither zig-zags', 'nor crawls', '74 steps', 'SGD at 0.05 in 104')) and 'faster' not in ra, ra)
                check('tile EN notes: the valley and start, the bottom at a loss below 0.001, Adam’s learning rate, not always faster (69 steps at 0.075)', len(notes) == 4 and all(k in ' '.join(notes) for k in ('½(x² + 25 y²)', '(−9, 1.5)', '0.001', 'largest step', 'not always faster', '0.075', '69 steps')), notes)
            else:
                units = lambda t: len(re.findall(r'[一-鿿]', t)) + len(re.findall(r'[A-Za-z0-9.%]+', t))
                check('tile ZH: each sentence one sentence and about as short as the English (≤ 63 units)', all(units(x) <= 63 and x.count('。') == 1 and x.endswith('。') for x in (rs, ra)), [(units(x), x) for x in (rs, ra)])
                check('tile ZH: the same numbers', all(k in rs for k in ('0.05', '104 步')) and all(k in ra for k in ('0.5', '74 步', '0.05', '104 步', '各自的步长')), (rs, ra))
                check('tile ZH notes: the same facts', len(notes) == 4 and all(k in ' '.join(notes) for k in ('½(x² + 25 y²)', '(−9, 1.5)', '0.001', '最多走的距离', '并不总是更快', '0.075', '69 步')), notes)
            alltxt = ' '.join([rs, ra] + notes + [plain(pg.evaluate('o=>window.__tzOp.legend(o)', o)) + pg.evaluate('o=>window.__tzOp.help(o)', o) for o in tz['opts']])
            leak = re.findall(r'0\.08|2 ÷ 25|0\.10?\b|−1\.5|diverg|发散|never|永远|100 y²|\(−5, 0\.8\)|β', alltxt)
            check(f'tile {lang}: no SGD at 0.1, no divergence, no limit, nothing of the challenge or of momentum’s β (R3.8)', not leak, leak)
        pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(200)
    check('no page errors', not errs, errs)
    b.close()

print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
