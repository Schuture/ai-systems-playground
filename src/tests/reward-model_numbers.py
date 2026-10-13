"""Numbers test for the reward-model lab: an independent Python model of the toy (only the random number generator is ported bit for
bit; the data, the training and best-of-n are written again here, the training with numpy) checked against the page's engine
(window.__rm) and its home tile (window.__tzRm), then every number the copy relies on.

- mulberry32 and the seeded data (training pairs 13, held-out 113, balanced 13 + 7919, the challenge's 45 / 46, the policy bank 31);
- the live update: full-batch gradient descent on mean −log σ(r_chosen − r_rejected), lr 0.15, from w = 0; weights, loss and held-out
  accuracy (a tie, only at step 0, counts half) at steps 0, 1, 50, 100, 200, 400 for every data set the page uses;
- best-of-n over the 2,000-response bank, exactly (P(pick rank j) = (j/M)^n − ((j−1)/M)^n, equal scores share their group's mass);
- the calibration the parts rely on (R3.18): σ(2) in the right option; length weighs about as much as content; best-of-n rises then
  falls (peak at n = 16) in the default data; 50 balanced pairs make it rise all the way; 200 push the length weight below 0 and cost
  accuracy; the scenarios' claims; the challenge (one goal, best-of-16 true quality ≥ 7.6; 20 options, 8 pass, the minimal fix 100 pairs
  at 200 steps, the initial setup fails); the new cases (2% padded peaks later; 25 against 100 balanced pairs); the drill; the tile;
- the claims the review (b9v1) found seed-dependent, re-simulated over many seeds with numpy's generator (not the page's): 200 balanced
  pairs usually beat 50 at n = 256 (the page calls this run's opposite chance) and always cost accuracy; 50 pairs cut the padded
  picks a lot; the challenge's minimal fix is 100 pairs in most seeds; with no content-reading error the length weight is near 0;
- the fixed numbers in the copy (σ values, 1.7 per doubling, 42% / 82% longer, 525M and 4,096 parameters);
- the Connect code runs and prints what the page says; the quoted source lines exist in the pinned local copies, part 1's facts about
  real reward models (2026-10-08) included.
Run: python3 tests/reward-model_numbers.py"""
import json, math, pathlib, re, subprocess, sys
import numpy as np
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'reward-model'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/reward-model.html', '--only', 'reward-model'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(OUT / 'num-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"></head><body>' + (P / 'out' / 'reward-model.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'num-wrapped.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


close = lambda a, b, tol=1e-9: abs(a - b) <= tol * max(1.0, abs(a), abs(b))

# ---------------- the independent model ----------------
M32 = 0xffffffff


def mulberry(a):
    st = [a & M32]
    def r():
        st[0] = (st[0] + 0x6D2B79F5) & M32
        t = st[0]
        t = ((t ^ (t >> 15)) * (1 | t)) & M32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & M32)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296
    return r


def normal(r):
    u, v = r(), r()
    return math.sqrt(-2 * math.log(1 - u)) * math.cos(2 * math.pi * v)


sigma = lambda x: 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
MEAN_Q, SD_Q, READ_ERR, PAD_DROP, PAD_X, LEN0, LEN_SCALE, LR = 6.0, 1.5, 1.0, 2.5, 3.0, 150.0, 0.4, 0.15
TIE = {'tied': 0.35, 'none': 0.0, 'strong': 0.6}
NS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
clip = lambda q: min(9.5, max(0.5, q))


def response(r, a, pad=False):
    q = clip(MEAN_Q + SD_Q * normal(r)); e = normal(r); z = normal(r); lst = 1 if r() < 0.5 else 0
    ln_len = math.log(LEN0) + a * (q - MEAN_Q) / SD_Q + 0.25 * z
    content = q + READ_ERR * e
    if pad:
        ln_len += math.log(PAD_X); q = clip(q - PAD_DROP)
    return (q, content, math.exp(ln_len), lst, pad)


def features(x):
    return np.array([(x[1] - MEAN_Q) / SD_Q, math.log(x[2] / LEN0) / LEN_SCALE, x[3] - 0.5])


def pairs(seed, n_, a, tau, bal):
    r, out = mulberry(seed), []
    for _ in range(n_):
        x, y = response(r, a), response(r, a); u = r()
        out.append((x, y, 0 if u < sigma((x[0] - y[0]) / tau) else 1, 0))
    r2 = mulberry(seed + 7919)
    for _ in range(bal):
        x0 = response(r2, a); y0 = (clip(x0[0] - PAD_DROP), x0[1], x0[2] * PAD_X, x0[3], True); u = r2(); swap = r2() < 0.5
        x, y = (x0, y0) if swap else (y0, x0)
        out.append((x, y, 0 if u < sigma((x[0] - y[0]) / tau) else 1, 1))
    return out


def fit(data, steps):
    D = np.array([features(x if c == 0 else y) - features(y if c == 0 else x) for x, y, c, _ in data])
    w = np.zeros(3); W, LOSS = [w.copy()], []
    for t in range(steps + 1):
        m = D @ w
        LOSS.append(float(np.mean(np.where(m > -35, np.log1p(np.exp(-np.maximum(m, -35))), -m))))
        if t == steps: break
        s = 1 / (1 + np.exp(m))                     # σ(−m)
        w = w + LR * (s[:, None] * D).mean(axis=0); W.append(w.copy())
    return W, LOSS


def acc(data, w):
    ok = 0.0
    for x, y, c, _ in data:
        a, b = float(features(x) @ w), float(features(y) @ w)
        ok += 0.5 if a == b else float((a > b) if c == 0 else (b > a))
    return ok / len(data)


def ceiling(data):
    return sum(1 for x, y, c, _ in data if (x[0] > y[0]) == (c == 0)) / len(data)


def bank(ppad=0.08):
    r = mulberry(31); return [response(r, TIE['tied'], r() < ppad) for _ in range(2000)]


BANK = bank()


def best_of_n(w, B=None):
    B = B or BANK; Mb = len(B)
    rows = sorted(((float(features(y) @ w), y[0], y[2], 1.0 if y[4] else 0.0) for y in B), key=lambda z: z[0])
    groups, j = [], 0
    while j < Mb:
        k = j
        while k + 1 < Mb and rows[k + 1][0] == rows[j][0]: k += 1
        groups.append((j, k + 1)); j = k + 1
    out = []
    for nn in NS:
        e = [0.0, 0.0, 0.0, 0.0]
        for j0, j1 in groups:
            p = (j1 / Mb) ** nn - (j0 / Mb) ** nn; m = j1 - j0
            for i in range(4): e[i] += p * sum(z[i] for z in rows[j0:j1]) / m
        out.append({'n': nn, 's': e[0], 'q': e[1], 'L': e[2], 'pad': e[3]})
    return out


def peak(b):
    k = 0
    for i, x in enumerate(b):
        if x['q'] > b[k]['q'] + 1e-12: k = i
    return k


def setting(rel='tied', tau=1, N=1000, bal=0, ch=False):
    return {'ch': True, 'bal': bal} if ch else {'rel': rel, 'tau': tau, 'N': N, 'bal': bal}


RUNS = {}


def run(s):
    key = json.dumps(s, sort_keys=True)
    if key in RUNS: return RUNS[key]
    a, tau, N, seed, hseed = (TIE['strong'], 1, 1000, 45, 46) if s.get('ch') else (TIE[s['rel']], s['tau'], s['N'], 13, 113)
    data, held = pairs(seed, N, a, tau, s['bal']), pairs(hseed, 500, a, tau, 0)
    W, LOSS = fit(data, 400)
    RUNS[key] = {'data': data, 'held': held, 'W': W, 'LOSS': LOSS, 'ceil': ceiling(held)}
    return RUNS[key]


TS = [0, 1, 50, 100, 200, 400]
SETS = [setting(), setting(bal=25), setting(bal=50), setting(bal=100), setting(bal=200), setting(rel='none'), setting(tau=3), setting(N=50)] + \
       [setting(ch=True, bal=b) for b in (0, 25, 50, 100, 200)]

with sync_playwright() as pw:
    br = pw.chromium.launch(); pg = br.new_page(); errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    js = lambda expr, *a: pg.evaluate(expr, *a)
    check('the page loads without errors', not errs, errs[:3])
    C = js('()=>window.__rm.consts')
    check('constants as in the model', (C['QM'], C['QS'], C['SE'], C['PADQ'], C['PADL'], C['L0'], C['LS'], C['ETA'], C['T'], C['NH'], C['M'], C['PPAD']) == (6, 1.5, 1, 2.5, 3, 150, 0.4, 0.15, 200, 500, 2000, 0.08) and C['NS'] == NS, C)
    rj = js('()=>window.__rm.rand(13,6)'); rp = mulberry(13); rp = [rp() for _ in range(6)]
    check('mulberry32 matches bit for bit', rj == rp, (rj, rp))
    # ---- the live update, every data set the page uses
    for s in SETS:
        pr = run(s); jr = js('([s,ts])=>window.__rm.run(s,ts)', [s, TS])
        tag = json.dumps(s)
        check(f'{tag}: pair counts', jr['np'] == len(pr['data']) and jr['nb'] == s['bal'] and jr['ch0'] == sum(1 for d in pr['data'] if d[2] == 0), (jr['np'], jr['nb'], jr['ch0']))
        okW = all(close(jr['W'][i][k], float(pr['W'][t][k]), 1e-9) for i, t in enumerate(TS) for k in range(3))
        check(f'{tag}: weights at steps {TS}', okW, (jr['W'][4], pr['W'][200]))
        check(f'{tag}: loss at steps {TS}', all(close(jr['LOSS'][i], pr['LOSS'][t], 1e-9) for i, t in enumerate(TS)), (jr['LOSS'], [pr['LOSS'][t] for t in TS]))
        pa = [acc(pr['held'], pr['W'][t]) for t in TS]
        check(f'{tag}: held-out accuracy at steps {TS}', all(abs(jr['acc'][i] - pa[i]) < 1e-12 for i in range(len(TS))), (jr['acc'], pa))
        check(f'{tag}: the ceiling (labels predicted from the true quality)', abs(jr['ceil'] - pr['ceil']) < 1e-12, (jr['ceil'], pr['ceil']))
        for t in ([0, 200] if not s.get('ch') else [50, 100, 200, 400]):
            jb, pb = js('([s,t])=>window.__rm.bon(s,t)', [s, t]), best_of_n(pr['W'][t])
            check(f'{tag}: best-of-n at step {t}', all(close(jb[i][k], pb[i][k], 1e-9) for i in range(9) for k in ('s', 'q', 'L', 'pad')), (jb[4], pb[4]))
    # ---- calibration the parts rely on
    d = run(setting()); w = d['W'][200]; b = best_of_n(w); k = peak(b)
    check('part 2: σ(2) falls in the option “80% to 95%”, the page’s right answer', 0.80 <= sigma(2) < 0.95 and C['PRED_RIGHT'] == 'c', sigma(2))
    check('part 2: σ(1) ≈ 73%, +23 points from 0 to 1, +15 from 1 to 2', round(sigma(1) * 100) == 73 and round((sigma(1) - 0.5) * 100) == 23 and round((sigma(2) - sigma(1)) * 100) == 15)
    check('part 3: loss 0.69 → 0.52, held-out accuracy 75%, the ceiling 81%', round(d['LOSS'][0], 2) == 0.69 and round(d['LOSS'][200], 2) == 0.52 and round(acc(d['held'], w) * 100) == 75 and round(d['ceil'] * 100) == 81, (d['LOSS'][200], acc(d['held'], w), d['ceil']))
    check('part 3: 500 pairs move the accuracy by about ±4 points', round(196 * math.sqrt(0.75 * 0.25 / 500)) == 4)
    check('part 5: length weighs as much as content (within 0.05)', abs(w[1] - w[0]) < 0.05 and w[0] > 0.4, w)
    wn = run(setting(rel='none'))['W'][200]
    check('part 5: data with length unrelated to quality gives length almost nothing (|w| < 0.1)', abs(wn[1]) < 0.1 and wn[0] > w[0], wn)
    check('part 6: best-of-n rises then falls; peak at n = 16; the score keeps rising', NS[k] == 16 and b[-1]['q'] < b[k]['q'] - 0.4 and all(b[i + 1]['s'] > b[i]['s'] for i in range(8)), [round(x['q'], 2) for x in b])
    check('part 6: more padded picks as n grows (most at n = 256)', b[-1]['pad'] > 0.5 and b[0]['pad'] < 0.1, (b[0]['pad'], b[-1]['pad']))
    b50 = best_of_n(run(setting(bal=50))['W'][200]); w50 = run(setting(bal=50))['W'][200]
    check('part 7: 50 balanced pairs: length weight falls, best-of-n rises all the way', w50[1] < w[1] / 2 and all(b50[i + 1]['q'] > b50[i]['q'] for i in range(8)), (w50, [round(x['q'], 2) for x in b50]))
    r200 = run(setting(bal=200)); w200 = r200['W'][200]; b200 = best_of_n(w200)
    a200 = acc(r200['held'], w200)
    check('part 7: 200 pairs: length weight below 0, accuracy 75% → 69%, n = 256 a little below 50 pairs', w200[1] < 0 and round(a200 * 100) == 69 and b200[-1]['q'] < b50[-1]['q'], (w200, a200))
    # scenarios
    rz = run(setting(tau=3)); wz = rz['W'][200]; bz = best_of_n(wz)
    check('scenario noisy: every weight smaller, the length preference still there, accuracy lower', abs(wz[0]) < abs(w[0]) and abs(wz[1]) < abs(w[1]) and wz[1] > 0.1 and acc(rz['held'], wz) < 0.7, wz)
    rs = run(setting(N=50)); ws = rs['W'][200]; bs = best_of_n(ws)
    check('scenario small: length takes most of the weight; best-of-256 below a random pick', ws[1] > 3 * abs(ws[0]) and bs[-1]['q'] < bs[0]['q'], (ws, bs[-1]['q'], bs[0]['q']))
    rf = run(setting(rel='none')); bf = best_of_n(rf['W'][200])
    check('scenario flat: best-of-n keeps rising, accuracy below the default', peak(bf) == 8 and acc(rf['held'], rf['W'][200]) < acc(d['held'], w))
    # the challenge
    rows = {}
    for bal in C['CH_BAL']:
        for st in C['CH_STEPS']:
            rr = run(setting(ch=True, bal=bal)); q = best_of_n(rr['W'][st])[4]['q']; a = acc(rr['held'], rr['W'][st])
            rows[(bal, st)] = (a, q, q >= 7.6 - 1e-12)
            je = js('([b,s])=>window.__rm.chEval(b,s)', [bal, st])
            check(f'challenge {bal} pairs, {st} steps: accuracy, quality, verdict', abs(je['acc'] - a) < 1e-12 and close(je['q'], q, 1e-9) and je['ok'] == rows[(bal, st)][2], (je, a, q))
    best = js('()=>window.__rm.chBest()')
    check('challenge: the minimal fix is 100 balanced pairs at 200 steps', (best['bal'], best['steps']) == (100, 200), best)
    check('challenge: the initial setup fails on quality; no step count helps with 0, 25 or 50 pairs', not rows[(0, 200)][2] and rows[(0, 200)][0] >= 0.7 and not any(rows[(b_, s_)][2] for b_ in (0, 25, 50) for s_ in C['CH_STEPS']))
    check('challenge: one goal, true quality at least 7.6 (accuracy shown, not graded)', C['CH_Q'] == 7.6 and 'CH_ACC' not in C, C['CH_Q'])
    _src = (pathlib.Path(__file__).resolve().parent.parent / 'labs' / 'reward-model.js').read_text(encoding='utf-8')
    check('challenge: no goal text is typed by hand (every goal label reads CH_Q)', not re.search(r'(goal: |目标：|至少 )7\.\d', _src), re.findall(r'.{20}(?:goal: |目标：|至少 )7\.\d', _src)[:3])
    check('challenge: 8 of 20 options pass, 100 and 200 pairs at every step count; steps never change the verdict', sorted(k_ for k_, v_ in rows.items() if v_[2]) == sorted((b_, s_) for b_ in (100, 200) for s_ in C['CH_STEPS']))
    check('challenge: 50 pairs miss the goal by more than 0.2 at every step count; 100 pairs at 200 steps clear it by more than 0.15', max(rows[(50, s_)][1] for s_ in C['CH_STEPS']) < 7.6 - 0.2 and rows[(100, 200)][1] > 7.6 + 0.15, (rows[(50, 400)], rows[(100, 200)]))
    check('challenge: the accuracy drifts down with more pairs (0 → 200 pairs at 200 steps: 74.8% → 69.4%)', round(rows[(0, 200)][0] * 1000) == 748 and round(rows[(200, 200)][0] * 1000) == 694, (rows[(0, 200)], rows[(200, 200)]))
    # the new case and the exits
    b2 = best_of_n(w, bank(0.02)); jb2 = js('()=>window.__rm.bon2()')
    check('new case: 2% padded peaks later (n = 64)', NS[peak(b2)] == 64 and jb2['k'] == peak(b2) and close(jb2['b'][peak(b2)]['q'], b2[peak(b2)]['q'], 1e-9), (peak(b2), jb2['k']))
    ex = js('()=>window.__rm.exits()')
    rA, rB = run(setting(bal=25)), run(setting(bal=100)); aA, aB = acc(rA['held'], rA['W'][200]), acc(rB['held'], rB['W'][200]); qA, qB = best_of_n(rA['W'][200])[-1]['q'], best_of_n(rB['W'][200])[-1]['q']
    check('exits: the answers are the ones the numbers give (new case 2: 25 pairs 74% / 8.06 against 100 pairs 72% / 8.73)', ex['transferA'] == [2, 1] and peak(b2) > k and round(aA * 100) == 74 and round(aB * 100) == 72 and round(qA, 2) == 8.06 and round(qB, 2) == 8.73 and qB > qA, (ex['transferA'], aA, aB, qA, qB))
    check('exits: new case 2 names the numbers the engine gives', all(x in ex['transfer'][1]['why'] for x in ('74%', '72%', '8.06', '8.73')), ex['transfer'][1]['why'][:200])
    # the examples, the drill, the tile
    exs = js('()=>window.__rm.ex()')
    sc = lambda x: float(features((x['q'], x['c'], x['L'], x['b'], False)) @ w)
    g1, g2 = sc(exs[0]['A']) - sc(exs[0]['B']), sc(exs[1]['A']) - sc(exs[1]['B'])
    check('example 1: A ends well above B (about 90%); example 2: the padded B wins (about 20%)', 0.85 < sigma(g1) < 0.95 and 0.1 < sigma(g2) < 0.3, (sigma(g1), sigma(g2)))
    for sd in range(3, 9):
        dj = js('s=>window.__rm.drill(s)', sd); gap = round(dj['rc'] - dj['rr'], 1)
        check(f'drill {sd}: gap, loss, gradient', close(dj['d'], gap, 1e-12) and close(dj['loss'], math.log1p(math.exp(-gap)), 1e-12) and close(dj['g'], sigma(-gap), 1e-12), dj)
    tz = js('()=>window.__tzRm.steps.map(t=>({t,...window.__tzRm.at(t),w:window.__tzRm.w(t)}))')
    for x in tz:
        wt = d['W'][x['t']]; fa = np.array([(7.0 - 6) / 1.5, 0.0, -0.5]); fb = np.array([0.0, math.log(120 / 150) / 0.4, -0.5])
        check(f'tile step {x["t"]}: the lab’s weights, the pair’s scores and σ', all(close(x['w'][i], float(wt[i]), 1e-9) for i in range(3)) and close(x['ra'], float(fa @ wt), 1e-9) and close(x['rb'], float(fb @ wt), 1e-9) and close(x['p'], sigma(float((fa - fb) @ wt)), 1e-9), x)
    check('tile: its gap at 200 steps is not the prediction’s 2 (R3.8)', abs(tz[-1]['ra'] - tz[-1]['rb'] - 2) > 1, tz[-1])
    words = len(re.sub('<[^>]+>', '', js('()=>window.__tzRm.read()')).split())
    check('tile: the .tz-read sentence has at most 25 words (EN)', words <= 25, words)
    br.close()

# ---------------- fixed numbers in the copy ----------------
check('1.7 per doubling of the length', round(math.log(2) / 0.4, 1) == 1.7)
check('42% / 82% longer per standard deviation of quality', round((math.exp(0.35) - 1) * 100) == 42 and round((math.exp(0.6) - 1) * 100) == 82)
check('τ = 3: a 1-point better answer is picked 58% of the time', round(sigma(1 / 3) * 100) == 58)
check('the LM head of Llama-3.1-8B: 128,256 × 4,096 = 525M parameters; the score layer 4,096', 128256 * 4096 == 525336576)
src = P / 'labs' / 'reward-model.js'; page = src.read_text()
for s_ in ('525M', '4,096', '65.7%', '64%', 'best-of-16', '0.01'):
    check(f'the copy has {s_}', s_ in page)
# ---------------- claims that must hold beyond this run's seeds (review b9v1: B2, S5, S13) ----------------
# numpy's generator, not mulberry32: the same model of data, people, training and best-of-n, drawn afresh for every seed
def sim_resp(g, n_, a, se=READ_ERR):
    q = np.clip(MEAN_Q + SD_Q * g.normal(size=n_), 0.5, 9.5); c = q + se * g.normal(size=n_); z = g.normal(size=n_); lst = (g.random(n_) < 0.5).astype(float)
    return q, c, math.log(LEN0) + a * (q - MEAN_Q) / SD_Q + 0.25 * z, lst


sim_f = lambda c, ln_len, lst: np.stack([(c - MEAN_Q) / SD_Q, (ln_len - math.log(LEN0)) / LEN_SCALE, lst - 0.5], -1)


def sim_data(g, n_, a, bal, se=READ_ERR):
    qa, ca, la, ba = sim_resp(g, n_, a, se); qb, cb, lb, bb = sim_resp(g, n_, a, se)
    fa, fb = sim_f(ca, la, ba), sim_f(cb, lb, bb); aw = g.random(n_) < 1 / (1 + np.exp(-(qa - qb)))
    D = np.where(aw[:, None], fa - fb, fb - fa)
    if bal:
        q, c, l, lst = sim_resp(g, bal, a, se); f0, f1 = sim_f(c, l, lst), sim_f(c, l + math.log(PAD_X), lst)
        w0 = g.random(bal) < 1 / (1 + np.exp(-(q - np.clip(q - PAD_DROP, 0.5, 9.5)))); D = np.concatenate([D, np.where(w0[:, None], f0 - f1, f1 - f0)])
    return D, (fa, fb, aw)


def sim_fit(D, steps=200):
    w = np.zeros(3)
    for _ in range(steps): w = w + LR * ((1 / (1 + np.exp(D @ w)))[:, None] * D).mean(0)
    return w


def sim_bank(g):
    q, c, l, lst = sim_resp(g, 2000, TIE['tied']); pad = g.random(2000) < 0.08
    return sim_f(c, np.where(pad, l + math.log(PAD_X), l), lst), np.where(pad, np.clip(q - PAD_DROP, 0.5, 9.5), q), pad.astype(float)


def sim_bon(w, B, nn):
    F, q, pad = B; o = np.argsort(F @ w); j = np.arange(1, 2001); pj = (j / 2000) ** nn - ((j - 1) / 2000) ** nn
    return float(pj @ q[o]), float(pj @ pad[o])


sim_acc = lambda H, w: float(np.mean(np.where(H[2], H[0] @ w > H[1] @ w, H[1] @ w > H[0] @ w)))
SEEDS = 40; win200, drop, pad0, pad50, minfix = 0, [], [], [], []
for sd in range(SEEDS):
    g = np.random.default_rng(20261005 + sd); B = sim_bank(g); res = {}
    hd_ = sim_data(np.random.default_rng(sd + 1000), 500, TIE['tied'], 0)[1]
    for bal in (0, 50, 200):
        w_ = sim_fit(sim_data(np.random.default_rng(sd), 1000, TIE['tied'], bal)[0]); res[bal] = (w_, sim_acc(hd_, w_), sim_bon(w_, B, 256))
    win200 += res[200][2][0] > res[50][2][0]; drop.append(res[0][1] - res[200][1]); pad0.append(res[0][2][1]); pad50.append(res[50][2][1])
    hc = sim_data(np.random.default_rng(sd + 3000), 500, TIE['strong'], 0)[1]
    qs = {bal: sim_bon(sim_fit(sim_data(np.random.default_rng(sd + 2000), 1000, TIE['strong'], bal)[0]), B, 16)[0] for bal in (0, 25, 50, 100, 200)}
    minfix.append(next((bal for bal in (0, 25, 50, 100, 200) if qs[bal] >= 7.6), None))
check(f'part 7 (B2): over {SEEDS} seeds, 200 balanced pairs usually beat 50 at n = 256 (“多半反而略高”)', win200 / SEEDS >= 0.75, win200 / SEEDS)
check(f'part 7 (B2): over {SEEDS} seeds, 200 balanced pairs always cost held-out accuracy (mean drop about 4 points)', min(drop) > 0 and 0.025 < np.mean(drop) < 0.06, (min(drop), np.mean(drop)))
check(f'part 7 (B2): over {SEEDS} seeds, 50 balanced pairs cut the padded share of best-of-256 picks a lot (“少了很多”)', np.mean(pad50) < 0.5 * np.mean(pad0) and all(a_ < b_ for a_, b_ in zip(pad50, pad0)), (np.mean(pad0), np.mean(pad50)))
check(f'challenge (S13): over {SEEDS} seeds, the fewest pairs that reach 7.6 is 100 in most seeds, never 0 or 25', minfix.count(100) / SEEDS >= 0.6 and minfix.count(0) == 0 and minfix.count(25) <= 1, minfix)
READ_ERR_SAVE = READ_ERR; READ_ERR = 0.0; RUNS.clear()
w0e = run(setting())['W'][200]; READ_ERR = READ_ERR_SAVE; RUNS.clear()
check('part 5 (S5): with no content-reading error, the same data give length almost no weight (the noisy reading is why length counts)', abs(w0e[1]) < 0.1 and w0e[0] > 2 * w[0], w0e)
# ---------------- the Connect code ----------------
code = page[page.index('const CODE=`') + 12: page.index('`;', page.index('const CODE=`'))]
outp = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60).stdout.strip()
check('the Connect code prints what the page says', outp == '[0.63 0.52] 0.497' and '[0.63 0.52] 0.497' in page, outp)
# ---------------- quoted sources (pinned local copies) ----------------
SRC = P / 'sources' / 'b9src' / 'reward-model' / 'pinned'   # the pinned copies the quotes are checked against
quotes = [('reward_trainer.py', 'loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()'),
          ('reward_trainer.py', 'the only output of the model is the reward score'),
          ('reward_trainer.py', 'AutoModelForSequenceClassification'),
          ('reward_config.py', 'center_rewards_coefficient'),
          ('modeling_layers.py', 'self.score = nn.Linear(config.get_text_config().hidden_size, self.num_labels, bias=False)'),
          ('hh-README.md', 'best-of-16'),
          ('run_rm.py', '1 if chosen > rejected else 0'),
          ('ae-README.md', '65.7')]
for f, q in quotes:
    hits = list(SRC.glob('*' + f)) if SRC.exists() else []
    if not hits:
        print('SKIP (no local copy):', f); continue
    check(f'source {f} has “{q[:50]}”', any(q in h.read_text(errors='ignore') for h in hits), f)
# ---------------- part 1's introduction of the reward model (the owner's review of 2026-10-08, item 2, G3) ----------------
# what it is (an LLM, usually the SFT model, one output), why (people label a limited set), how it is used (best-of-n, later training):
# each fact is quoted from a local copy of a pinned source, and the page cites the facts it states
B9 = P / 'sources' / 'b9src'
quotes2 = [('reward-model/hf_blog__rlhf.md', 'OpenAI 175B LM, 6B reward model'),
           ('reward-model/hf_blog__rlhf.md', 'OpenAI used a smaller version of GPT-3 for its first popular RLHF model, [InstructGPT]'),
           ('reward-model/hf_blog__rlhf.md', '~50k labeled preference samples'),
           ('reward-model/hf_blog__rlhf.md', 'These LMs for reward modeling can be both another fine-tuned LM or a LM trained from scratch on the preference data'),
           ('reward-model/rlhf_book__book_chapters_05-reward-models.md', 'converting a language model into a model that outputs a scalar score'),
           ('reward-model/rlhf_book__book_chapters_05-reward-models.md', 'rerank $k$ completions, pick top-1 (best-of-N sampling); or provide terminal reward for RLHF'),
           ('ppo/openrlhf__README.md', '--actor.model_name_or_path OpenRLHF/Llama-3-8b-sft-mixture'),
           ('ppo/openrlhf__README.md', 'num_labels=1,')]
for f, q in quotes2:
    h = B9 / f
    if not h.exists():
        print('SKIP (no local copy):', f); continue
    check(f'source {f} has “{q[:50]}”', q in h.read_text(errors='ignore'), f)
for s_ in ('Llama-3-8b-sft-mixture', 'num_labels=1', '175B', '6B', '50,000', 'dc2a7ad326f619fc5a47737ea47d1920ce2c0b53', '998ee73e0594a21b42df14632263266a2831dafe', 'eecc49e1e1daf1be670d7242eb090240b5bb0f04'):
    check(f'the page cites {s_}', s_ in page)
check('50,000 pairs: the source’s “~50k”, written in full (R6.18: five digits)', 50 * 1000 == 50000 and '~50k' in (B9 / 'reward-model/hf_blog__rlhf.md').read_text(errors='ignore'))
print(f'{n - len(fails)}/{n} checks passed')
sys.exit(1 if fails else 0)
