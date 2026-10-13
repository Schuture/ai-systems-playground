"""Golden numbers for the sampling lab (guide R9.2, CONTRACT §6).

Everything is derived here independently of the JS engine:
- one step: softmax of logits / T (T = 0: greedy, all on the largest logit), then top-k (the k largest, renormalized), then top-p found
  by brute force (the smallest n such that the n likeliest words, after top-k, hold at least p), renormalized;
- the draw: the word whose stretch of the cumulative final chances a uniform number lands in; the random numbers are mulberry32
  (the only thing ported, since it defines the seeded runs), seeded with seed × 7919 + 17, three per run, 20 runs;
- the tree: a sentence's chance is the product of its three words' chances, each from its own list.
Checks:
- the first step against the home tile (window.__tzB.samp from teasers/set-b.js) at every tile temperature with top-p 0.9;
- the lab's engine (window.__sp) against this derivation: every list of the tree at many settings, the 20 runs, the tree's best
  sentence and the number of possible sentences, part 4's two lists, part 5's run, the scenarios, the challenge grid, its minimal
  answer and tiers, the capability layer's new cases, the interview drill, and the Connect card's code (run here with NumPy);
- the numbers the page shows in the mission, at temperature 0, in part 4, part 5, the scenarios and the challenge.
Run: python3 tests/sampling_numbers.py"""
import math, pathlib, re, subprocess, sys, io, contextlib
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/sampling-num.html', '--only', 'sampling,set-b'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'sampling-num-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'sampling-num.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'sampling-num-wrapped.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


# ---------- the derivation
def step(z, T, K=0, Pp=1.0):
    n = len(z)
    if T == 0:
        top = max(range(n), key=lambda i: (z[i], -i))
        raw = [1.0 if i == top else 0.0 for i in range(n)]
    else:
        m = max(z); e = [math.exp((v - m) / T) for v in z]; s = sum(e); raw = [x / s for x in e]
    order = sorted(range(n), key=lambda i: (-raw[i], i))
    kept_k = set(order[:K]) if 0 < K < n else set(range(n))
    sk = sum(raw[i] for i in kept_k)
    q = [raw[i] / sk if i in kept_k else 0.0 for i in range(n)]
    if T == 0:
        kept = {top}                                  # greedy: only the top word can be picked
    elif Pp >= 1:
        kept = set(kept_k)
    else:
        ok = [i for i in order if i in kept_k]
        size = next(m for m in range(1, len(ok) + 1) if sum(q[i] for i in ok[:m]) >= Pp or m == len(ok))
        kept = set(ok[:size])
    s2 = sum(q[i] for i in kept)
    fin = [q[i] / s2 if i in kept else 0.0 for i in range(n)]
    return {'raw': raw, 'fin': fin, 'kept': kept, 'nk': len(kept), 'cut': sum(raw[i] for i in range(n) if i not in kept)}


def draw(fin, u):
    c = 0.0; last = None
    for i, x in enumerate(fin):
        if x <= 0: continue
        last = i; c += x
        if u < c: return i
    return last


M32 = 0xFFFFFFFF
def imul(a, b): return ((a & M32) * (b & M32)) & M32
def mulberry(a):
    st = [a & M32]
    def f():
        st[0] = (st[0] + 0x6D2B79F5) & M32; x = st[0]
        t = imul(x ^ (x >> 15), 1 | x)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296
    return f
def runs(seed):
    r = mulberry(seed * 7919 + 17); return [(r(), r(), r()) for _ in range(20)]


Z1 = [2.0, 1.5, 1.0, 0.3, -0.5, -1.0]
W1 = ['noodles', 'rice', 'dumplings', 'hot pot', 'salad', 'a burger']
CHZ = [2.8, 2.0, 1.3, 0.8, 0.2, -0.3, -0.8]; CHBAD = [False] * 4 + [True] * 3
pct1 = lambda x: f'{x * 100:.1f}%'


def fP(x):
    """the page's percent: 1 decimal, 2 below 1%, '<0.01%' below 0.01%, '0%' and '100%' exact"""
    if x <= 0: return '0%'
    if x >= 1: return '100%'
    v = x * 100
    if v >= 99.95: return '>99.9%'
    if v < 0.01: return '<0.01%'
    return f'{v:.2f}%' if v < 1 else f'{v:.1f}%'


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    C = pg.evaluate('()=>window.__sp.consts')
    TREE = C['TREE']
    check('the first step is the tile’s six logits', C['Z1'] == Z1 and [a['w'] for a in TREE] == W1, C['Z1'])
    check('the tree has 38 three-word endings', C['N_LEAVES'] == sum(len(y['c']) for x in TREE for y in x['c']) == 38, C['N_LEAVES'])
    check('every list in the table is stored likeliest first (list order = sorted order)', all(
        [x['z'] for x in L] == sorted([x['z'] for x in L], reverse=True) for L in [TREE] + [a['c'] for a in TREE] + [b2['c'] for a in TREE for b2 in a['c']]))

    # ---------- the home tile: the same first step at every tile temperature with top-p 0.9
    tz = pg.evaluate('()=>window.__tzB&&window.__tzB.samp?{L:window.__tzB.samp.LOGITS,P:window.__tzB.samp.TOP_P}:null')
    check('the tile’s logits and top-p', tz and tz['L'] == Z1 and tz['P'] == 0.9, tz)
    worst = 0
    for i in range(1, 21):
        T = i / 10
        tile = pg.evaluate(f'()=>window.__tzB.samp.compute({T},0.9)')
        lab = pg.evaluate(f'()=>window.__sp.dist({Z1},{T},0,0.9)')
        mine = step(Z1, T, 0, 0.9)
        worst = max(worst, max(abs(a - c) for a, c in zip(tile['fin'], mine['fin'])), max(abs(a - c) for a, c in zip(lab['fin'], mine['fin'])))
        check(f'T {T}: the lab, the tile and this derivation keep the same words', [bool(x) for x in tile['keep']] == [bool(x) for x in lab['keep']] == [i2 in mine['kept'] for i2 in range(6)])
    check('the lab and the tile give the same chances as this derivation at every tile temperature (to 1e-12)', worst < 1e-12, worst)
    check('spec: at T 1 the chances are 43.7, 26.5, 16.1, 8.0, 3.6, 2.2 %', [pct1(x) for x in step(Z1, 1)['raw']] == ['43.7%', '26.5%', '16.1%', '8.0%', '3.6%', '2.2%'])
    check('spec: top-p 0.9 at T 1 keeps 4 (cumulative 86.3% after three, 94.2% after four)', step(Z1, 1, 0, 0.9)['nk'] == 4)

    # ---------- the lab's engine against the derivation: every list, many settings
    SETS = [(T / 10, K, Pp / 20) for T in (0, 1, 2, 5, 7, 10, 12, 15, 20) for K in (0, 1, 3, 5) for Pp in (4, 10, 18, 19, 20)]
    bad = []
    lists = [Z1] + [[x['z'] for x in a['c']] for a in TREE] + [[x['z'] for x in b2['c']] for a in TREE for b2 in a['c']] + [CHZ]
    for (T, K, Pp) in SETS:
        for z in lists:
            lab = pg.evaluate(f'()=>window.__sp.dist({z},{T},{K},{Pp})'); mine = step(z, T, K, Pp)
            if [bool(x) for x in lab['keep']] != [i in mine['kept'] for i in range(len(z))] or max(abs(a - c) for a, c in zip(lab['fin'], mine['fin'])) > 1e-12:
                bad.append((T, K, Pp, z))
    check(f'the engine equals the derivation on all {len(lists)} lists at {len(SETS)} settings', not bad, bad[:3])
    # the settings the page presents: the scenarios, part 4, and the challenge's grid; no cumulative sum within 0.005 of top-p there
    shown = [(z, T, Pp) for (T, Pp) in [(0.2, 0.95), (0.6, 0.9), (1.2, 0.95), (1.0, 0.9)] for z in lists[:-1]] + [(CHZ, T, Pp) for T in (0.5, 0.7, 1.0, 1.3) for Pp in (0.8, 0.9, 0.95)]
    near = [(z, T, Pp) for (z, T, Pp) in shown for m in range(1, len(z)) if abs(sum(sorted(step(z, T)['raw'], reverse=True)[:m]) - Pp) <= 0.005]
    check('no cumulative sum sits within 0.005 of top-p at any setting the page presents (no tie at the cut)', not near, near[:3])

    # ---------- the 20 runs: the random numbers, the picks, the distinct counts
    lr = pg.evaluate('()=>window.__sp.runs(1)')
    check('the seeded random numbers (mulberry32, seed 1) match', all(close(a, c, 0) for x, y in zip(lr, runs(1)) for a, c in zip(x, y)))
    def firsts(T, K=0, Pp=1.0, seed=1):
        f = step(Z1, T, K, Pp)['fin']; return [draw(f, u[0]) for u in runs(seed)]
    for (T, want) in [(1.0, 5), (0.0, 1), (0.5, 3), (1.5, 5)]:
        c = pg.evaluate(f'()=>window.__sp.compute({{T:{T},P:1,K:0,seed:1}})')
        mine = firsts(T)
        check(f'20 runs at T {T}: the same first words, {want} different', c['first'] == mine and len(set(mine)) == want, (c['first'], mine))
    check('spec: seed 1 at T 1 gives 8, 5, 5, 1, 1, 0 of the six words', [firsts(1.0).count(i) for i in range(6)] == [8, 5, 5, 1, 1, 0])
    check('top-p 0.9 at T 1: 4 different words (salad’s run moves to hot pot)', len(set(firsts(1.0, 0, 0.9))) == 4)

    # ---------- the tree
    def tree_paths(T, K=0, Pp=1.0):
        out = {}
        d1 = step(Z1, T, K, Pp)['fin']
        for i, a in enumerate(TREE):
            d2 = step([x['z'] for x in a['c']], T, K, Pp)['fin']
            for j, b2 in enumerate(a['c']):
                d3 = step([x['z'] for x in b2['c']], T, K, Pp)['fin']
                for k, c3 in enumerate(b2['c']):
                    out[(i, j, k)] = d1[i] * d2[j] * d3[k]
        return out
    for (T, K, Pp) in [(1, 0, 1.0), (0, 0, 1.0), (1, 0, 0.9), (0.6, 0, 0.9), (1.2, 3, 0.95)]:
        tp = tree_paths(T, K, Pp); c = pg.evaluate(f'()=>window.__sp.compute({{T:{T},P:{Pp},K:{K},seed:1}})')
        lab = {(a['i'], b2['j'], c3['k']): c3['p'] for a in c['tree'] for b2 in a['kids'] for c3 in b2['leaves']}
        best = max(tp, key=lambda k: tp[k]); poss = sum(1 for v in tp.values() if v > 0)
        check(f'tree at T {T}, top-k {K}, top-p {Pp}: every sentence’s chance, the chances add up to 1, the best sentence and the count',
              all(close(lab[k], tp[k], 1e-12) for k in tp) and close(sum(tp.values()), 1, 1e-9) and (c['best']['i'], c['best']['j'], c['best']['k']) == best and c['possible'] == poss,
              (best, c['best'], poss, c['possible']))
    t1 = tree_paths(1)
    check('spec: the likeliest sentence at T 1 is “noodles with beef”, 16.3%', max(t1, key=lambda k: t1[k]) == (0, 0, 0) and pct1(t1[(0, 0, 0)]) == '16.3%', t1[(0, 0, 0)])
    check('spec: at T 0 one sentence is possible, and it is the greedy one', sum(1 for v in tree_paths(0).values() if v > 0) == 1 and tree_paths(0)[(0, 0, 0)] == 1)
    # the 20 runs through the tree
    for (T, Pp) in [(1, 1.0), (1, 0.9)]:
        d1 = step(Z1, T, 0, Pp)['fin']; sents = []
        for u in runs(1):
            i = draw(d1, u[0]); a = TREE[i]; d2 = step([x['z'] for x in a['c']], T, 0, Pp)['fin']; j = draw(d2, u[1])
            d3 = step([x['z'] for x in a['c'][j]['c']], T, 0, Pp)['fin']; k = draw(d3, u[2]); sents.append((i, j, k))
        c = pg.evaluate(f'()=>window.__sp.compute({{T:{T},P:{Pp},K:0,seed:1}})')
        check(f'20 runs through the tree at T {T}, top-p {Pp}: the same sentences and the same count of different ones',
              [(r['i'], r['j'], r['k']) for r in c['runs']] == sents and c['distinctS'] == len(set(sents)), (c['distinctS'], len(set(sents))))

    # ---------- part 4: the flat list and the peaked one
    peak = [x['z'] for x in TREE[C['PEAK']]['c']]
    check('part 4’s peaked list is the word after “a burger”: and, with, without, or', [x['w'] for x in TREE[C['PEAK']]['c']] == ['and', 'with', 'without', 'or'])
    for lid, K, Pp, fk, pk in [('none', 0, 1.0, 6, 4), ('k3', 3, 1.0, 3, 3), ('p9', 0, 0.9, 4, 1)]:
        f = pg.evaluate(f"()=>window.__sp.flat('{lid}')"); q = pg.evaluate(f"()=>window.__sp.peak('{lid}')")
        check(f'part 4, {lid}: keeps {fk} of 6 and {pk} of 4', f['nk'] == step(Z1, 1, K, Pp)['nk'] == fk and q['nk'] == step(peak, 1, K, Pp)['nk'] == pk, (f['nk'], q['nk']))
    check('spec: after “a burger”, “and” alone has 94% (more than 0.9)', pct1(step(peak, 1)['raw'][0]) == '94.2%')
    check('part 4, top-k 3 cuts hot pot, which had 8.0% on the flat list', step(Z1, 1, 3)['kept'] == {0, 1, 2} and pct1(step(Z1, 1)['raw'][3]) == '8.0%')

    # ---------- part 5: your run; the first random number moves, the second and third stay
    U23 = C['U23']
    def walk(u1):
        d1 = step(Z1, 1)['fin']; i = draw(d1, u1); a = TREE[i]; d2 = step([x['z'] for x in a['c']], 1)['fin']; j = draw(d2, U23[0])
        d3 = step([x['z'] for x in a['c'][j]['c']], 1)['fin']; k = draw(d3, U23[1]); return (i, j, k), d1[i] * d2[j] * d3[k]
    for u1, want in [(C['U1_0'], ('noodles', 'with', 'beef')), (0.5, ('rice', 'with', 'chicken')), (C['U1_SHOW'], ('salad', 'with', 'chicken')), (0.99, ('a burger', 'and', 'fries'))]:
        (i, j, k), pr = walk(u1)
        words = (TREE[i]['w'], TREE[i]['c'][j]['w'], TREE[i]['c'][j]['c'][k]['w'])
        lab = pg.evaluate(f'()=>window.__sp.walk({{T:1,P:1,K:0,seed:1}},[{u1},{U23[0]},{U23[1]}])')
        check(f'part 5, u1 {u1}: “{" ".join(want)}”', words == want and (lab['i'], lab['j'], lab['k']) == (i, j, k) and close(lab['p'], pr, 1e-12), (words, lab))
    check('spec: “salad” has 3.6% and “salad with chicken” 1.5%', pct1(step(Z1, 1)['raw'][4]) == '3.6%' and pct1(walk(0.97)[1]) == '1.5%', walk(0.97))

    # ---------- the scenarios (typical settings)
    for pid, T, Pp, nk, dist_, top in [('extract', 0, 1.0, 1, 1, '100%'), ('code', 0.2, 0.95, 2, None, '92%'), ('chat', 0.6, 0.9, 3, None, '62%'), ('brain', 1.2, 0.95, 5, None, '41%')]:
        s = step(Z1, T, 0, Pp); d = len(set(firsts(T, 0, Pp)))
        c = pg.evaluate(f'()=>window.__sp.compute({{T:{T},P:{Pp},K:0,seed:1}})')
        check(f'scenario {pid}: {nk} words kept, top word {top}, {d} different first words in 20 runs', s['nk'] == nk == c['d1']['nk'] and c['distinct'] == d and (dist_ is None or d == dist_)
              and (f"{round(s['fin'][0] * 100)}%" == top), (s['nk'], c['distinct'], s['fin'][0]))
    presets = pg.evaluate('()=>window.__sp.consts.PRESETS')
    check('the scenarios’ settings: T 0; T 0.2 top-p 0.95; T 0.6 top-p 0.9; T 1.2 top-p 0.95', [(x['T'], x['P'], x['K']) for x in presets] == [(0, 1, 0), (0.2, 0.95, 0), (0.6, 0.9, 0), (1.2, 0.95, 0)], presets)

    # ---------- the challenge
    def ch(T, Pp):
        s = step(CHZ, T, 0, Pp); w = sum(s['fin'][i] for i in range(7) if CHBAD[i]); n5 = sum(1 for i, x in enumerate(s['fin']) if x >= 0.05 and not CHBAD[i])   # good words only (review A7)
        return w, n5, w < 0.01 and n5 >= 3
    grid = {(T, Pp): ch(T, Pp) for T in (0.5, 0.7, 1.0, 1.3) for Pp in (0.8, 0.9, 0.95, 1.0)}
    passing = sorted(k for k, v in grid.items() if v[2])
    check('challenge: the settings that meet both targets', passing == [(0.7, 0.9), (0.7, 0.95), (1.0, 0.8), (1.0, 0.9), (1.3, 0.8)], passing)
    check('challenge: a lower temperature alone never meets the goal', not any(grid[(T, 1.0)][2] for T in (0.5, 0.7, 1.0, 1.3)))
    check('challenge: the initial setup (T 1.0, no top-p) fails with 7.5% on the wrong words', not grid[(1.0, 1.0)][2] and pct1(grid[(1.0, 1.0)][0]) == '7.5%', grid[(1.0, 1.0)])
    check('challenge: no wrong-word chance within 0.3 points of 1%, no kept share within 0.4 points of 5%', all(abs(v[0] - 0.01) > 0.003 for v in grid.values())
          and all(abs(x - 0.05) > 0.004 for (T, Pp) in grid for x in step(CHZ, T, 0, Pp)['fin']))
    for (T, Pp), (w, n5, ok) in grid.items():
        lab = pg.evaluate(f'()=>window.__sp.chEval({T},{Pp})')
        check(f'challenge T {T}, top-p {Pp}: wrong {w:.4%}, {n5} good words at 5%+, ok {ok}', close(lab['wrong'], w, 1e-12) and lab['n5'] == n5 and lab['ok'] == ok)
    check('challenge: at T 1.3 with top-p 0.90, “sad” is at 5% or more but the count of good words is 4 (review A7)', grid[(1.3, 0.9)][1] == 4 and step(CHZ, 1.3, 0, 0.9)['fin'][4] >= 0.05)
    # the fewest changes: one setting moved beats two; then the gentler top-p (R4.5)
    ones = [k for k in passing if (k[0] != 1.0) + (k[1] != 1.0) == 1]
    best = max(ones, key=lambda k: k[1])
    lb = pg.evaluate('()=>window.__sp.chBest()')
    check('challenge: the minimal answer is temperature 1.0 with top-p 0.90 (one change, the gentlest cut)', best == (1.0, 0.9) and (lb['T'], lb['P'], lb['c']) == (1.0, 0.9, 1), (best, lb))
    check('challenge feedback: at T 1.3 top-p 0.9 lets “sad” in with 6.1%', pct1(ch(1.3, 0.9)[0]) == '6.1%' and step(CHZ, 1.3, 0, 0.9)['kept'] == {0, 1, 2, 3, 4})

    # ---------- the capability layer's new cases, from the lab's own function
    XA, XP = C['XA'], C['XP']
    xs = step([math.log(v) for v in XA], 1, 0, XP)
    check('new case 1: top-p 0.75 on 45/25/15/10/5 keeps A, B and C (70% is below 75%)', xs['kept'] == {0, 1, 2} and pg.evaluate('()=>window.__sp.exits().transferA')[0] == 2)
    check('new case 1: rescaled to 52.9%, 29.4%, 17.6%', [f'{x * 100:.1f}%' for x in xs['fin'][:3]] == ['52.9%', '29.4%', '17.6%'], xs['fin'])
    t2 = math.exp(1) / (math.exp(1) + 1); t1_ = math.exp(2) / (math.exp(2) + 1)
    check('new case 2: logits 2 and 0: 88% at T 1, 73.1% at T 2 (option 2 of 4)', f'{t1_ * 100:.0f}%' == '88%' and f'{t2 * 100:.1f}%' == '73.1%' and pg.evaluate('()=>window.__sp.exits().transferA')[1] == 1)

    # ---------- the interview drill
    for seed in range(1, 9):
        d = pg.evaluate(f'()=>window.__sp.drill({seed})')
        mine = step(d['z'], d['T'], 0, d['P'])
        check(f'drill {seed}: logits {d["z"]} at T {d["T"]}, top-p {d["P"]}: kept {sorted(mine["kept"])}', [bool(x) for x in d['d']['keep']] == [i in mine['kept'] for i in range(4)]
              and max(abs(a - c) for a, c in zip(d['d']['fin'], mine['fin'])) < 1e-12)

    # ---------- the Connect card's code, run here
    pg.evaluate("()=>{location.hash='sampling'}"); pg.wait_for_timeout(600)
    code = pg.evaluate("()=>document.getElementById('sp-code').textContent")
    check('the Connect code: at most 40 lines, each at most 74 characters (R7.6)', len(code.splitlines()) <= 40 and max(len(x) for x in code.splitlines()) <= 74, (len(code.splitlines()), max(len(x) for x in code.splitlines())))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): exec(compile(code, 'connect', 'exec'), {})
    out = buf.getvalue().splitlines()
    want = [x[2:] for x in code.splitlines() if x.startswith('# ')]
    check('the Connect code prints what its comments say, all four lines, the NumPy draws included (review B S3)', len(want) == 4 and out == want, (out, want))
    check('the Connect code says its draws are NumPy’s, not the page’s runs (review B S3)', "NumPy's draws, not the page's runs" in code)
    import numpy as np
    ns = {}; exec(compile(code.split('logits = [')[0], 'connect', 'exec'), ns)
    worst = max(abs(a - c) for (T, K, Pp) in [(0, 0, 1.0), (0.5, 0, 0.9), (1, 3, 1.0), (1, 0, 0.9), (1.2, 2, 0.95), (2, 0, 0.5)] for z in (Z1, CHZ)
                for a, c in zip(ns['probs'](z, T, K, Pp), step(z, T, K, Pp)['fin']))
    check('the Connect code’s probs() equals this derivation (temperature, top-k, top-p)', worst < 1e-12, worst)

    # ---------- review B S1: products in the tree's tooltips redo from their shown factors (R6.6), at T 1 and in Scenario 4's tree
    pat2 = re.compile(r'A run starts “.+” ([\d.]+)% of the time: “.+” is picked ([\d.]+)% of the time, and after it “.+” ([\d.]+)% of the time\.')
    pat3 = re.compile(r'A run writes “.+” ([\d.]+)% of the time: “.+” is picked ([\d.]+)% of the time, then “.+” ([\d.]+)%, then “.+” ([\d.]+)%\.')
    def redo(prod, *f):
        d = len(prod.split('.')[1]) if '.' in prod else 0
        x = 1.0
        for v in f: x *= float(v) / 100
        return f'{x * 100:.{d}f}' == prod
    leafpct = {}
    for tok in ('10_20_0_1_b', '12_19_0_1_b'):
        pg.evaluate(f"()=>{{location.hash='sampling~{tok}'}}"); pg.wait_for_timeout(700)
        tips = pg.evaluate("()=>[...document.querySelectorAll('#sp-tree .sp-blk')].map(b=>b.dataset.tipx)")
        if tok.startswith('10_'):
            leafpct = {m.group(1): m.group(2) for m in (re.match(r'A run writes “(.+?)” ([\d.]+)% of the time', t) for t in tips) if m}
        m2 = [pat2.match(t) for t in tips if t.startswith('A run starts “')]; m3 = [pat3.match(t) for t in tips if t.startswith('A run writes')]
        bad = [m.group(0) for m in m2 + m3 if not m or not redo(*m.groups())]
        check(f'tree {tok}: every second- and third-word tooltip’s chance is the product of its shown factors ({len(m2)} + {len(m3)} tooltips)', m2 and m3 and all(m2) and all(m3) and not bad, bad[:3])
    pg.evaluate("()=>{location.hash='sampling'}"); pg.wait_for_timeout(700)
    pg.evaluate("()=>document.querySelector('#sp-app .modes button[data-mode=\"guided\"]').click()"); pg.wait_for_timeout(300)
    pg.evaluate("()=>document.querySelector('#sp-rail [data-go=\"4\"]').click()"); pg.wait_for_timeout(400)
    for u in (97, 99, 20):
        pg.evaluate(f"()=>{{const i=document.getElementById('sp-ic-u1-in'); i.value={u}; i.dispatchEvent(new Event('input',{{bubbles:true}}))}}"); pg.wait_for_timeout(300)
        mt = pg.evaluate("()=>{document.getElementById('sp-math').open=true; return document.getElementById('sp-mathbody').innerText}")
        # H2 (the owner's decision of 2026-10-09): the sentence's chance is a typeset “=” chain (P = p₁ × p₂ × p₃ = …); MathML has no
        # spaces between its tokens, so the chain is read without whitespace
        eq = pg.evaluate("()=>{const e=document.querySelector('#sp-mathbody .sp-eq'); return e?e.textContent.replace(/\\s+/g,''):''}")
        m = re.search(r'=([\d.]+)×([\d.]+)×([\d.]+)=([\d.]+)', eq)
        ok = bool(m)
        if m:
            f = [float(x) for x in m.groups()[:3]]; prod = f[0] * f[1] * f[2]; shown = m.group(4)
            ok = float(f'{prod:.4g}') == float(shown) and len(re.sub(r'^0\.0*', '', shown)) == 4
        check(f'part 5 math at u1 0.{u:02d}: the product of the shown factors, to 4 significant digits (review B S1)', ok, m.group(0) if m else eq[:200])
        sm = re.search(r'For “([a-z][a-z ]+?)”:', mt)
        tv = leafpct.get(sm.group(1).strip()) if sm else None
        check(f'part 5 math at u1 0.{u:02d}: the tree’s tooltip gives the same chance as the math, 4 significant digits (re-review N3)', bool(m) and tv is not None and abs(float(tv) - float(m.group(4)) * 100) < 1e-9, (sm.group(1) if sm else None, tv, m.group(4) if m else None))
        tabs = pg.evaluate("()=>[...document.querySelectorAll('#sp-mathbody table.sp-mtab')].map(t=>t.rows[0].textContent.replace(/\\s+/g,''))")
        check(f'part 5 math at u1 0.{u:02d}: the lists after the first and second word, logits and chances (review B S6; a table per list since H2, 2026-10-09)', mt.count('e') > 10 and ('After “' in mt) and len(tabs) == 3 and all('logit' in h and 'p' in h for h in tabs), (tabs, mt[:300]))
    zl = pg.evaluate("()=>document.querySelector('#sp-controls .sp-zlist').textContent")
    allz = [f'{x["z"]:.1f}' for a in TREE for x in [a] + a['c'] + [c3 for b2 in a['c'] for c3 in b2['c']]]
    check('Settings list every logit of the tree (review B S6)', all(z.replace('-', '') in zl for z in allz) and zl.count('→') == 6, zl[:200])
    pg.evaluate("()=>document.querySelector('#sp-rail [data-go=\"0\"]').click()"); pg.wait_for_timeout(400)

    # ---------- the numbers the page shows
    def txt(sel): return pg.evaluate(f"()=>{{const e=document.querySelector('{sel}'); return e?e.innerText.replace(/\\u00a0/g,' '):''}}")
    rows = pg.evaluate("()=>[...document.querySelectorAll('#sp-rows .sp-row')].map(r=>r.querySelector('.sp-v').textContent)")
    check('mission: the bars show 43.7, 26.5, 16.1, 8.0, 3.6, 2.2 %', rows == [fP(x) for x in step(Z1, 1)['fin']] == ['43.7%', '26.5%', '16.1%', '8.0%', '3.6%', '2.2%'], rows)
    chips = pg.evaluate("()=>[...document.querySelectorAll('#sp-chips .sp-chip')].map(c=>c.textContent)")
    check('mission: the 20 chips are the 20 runs’ first words', chips == [W1[i] for i in firsts(1.0)], chips)
    check('mission: the card and the cards say 5 different first words', '5 different words' in txt('#sp-rail .step.current') and txt('#sp-metrics [data-k="runs"] .v') == '5')
    ticks = pg.evaluate("()=>[...document.querySelectorAll('#sp-drawbar .sp-tick')].map(t=>parseFloat(t.style.left))")
    check('mission: 20 ticks at the runs’ first random numbers', len(ticks) == 20 and all(abs(a - u[0] * 100) < 1e-3 for a, u in zip(ticks, runs(1))))
    b.close()
check('no page errors', not errs, errs[:3])
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:40] if fails else 'none')
sys.exit(1 if fails else 0)
