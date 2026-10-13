"""Numbers test for the train-debug lab (CONTRACT §6).

An independent NumPy re-implementation of the lab's engine (mulberry32, Box–Muller, the data, the 2 → 16 → C network with the
script's naive softmax, full-batch gradient descent, the three bugs, the three checks, the curve categories, the curve shapes and
the minimal check sets), compared with window.__td in the built page:
- every training loss, validation loss, accuracy and gradient norm of every run the page shows (3 and 4 groups), within 1e-9;
- where each blow-up turns inf and NaN; ln C; the checks' results and categories; the curve shapes the challenge reads for free;
  the challenge's minimal check sets (none: the five shapes differ), the checks that confirm a reading and the hint's check;
- the decision regions of the healthy and the shuffled run; the teaser's sentences carry the same numbers;
- the capability layer's computed answer (learning rate 3 trains), the drill's ln C, the minimal implementation's size (≤ 40 lines,
  ≤ 74 characters a line).
Run: python3 tests/train-debug_numbers.py"""
import json, math, pathlib, subprocess, sys
import numpy as np
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/train-debug-numbers.html', '--only', 'train-debug'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'train-debug-numbers-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'train-debug-numbers.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'train-debug-numbers-wrapped.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


# ---------------------------------------------------------------- the engine, again
M32 = 0xFFFFFFFF


def mulberry(a):
    st = [a & M32]
    def r():
        st[0] = (st[0] + 0x6D2B79F5) & M32
        a_ = st[0]
        t = ((a_ ^ (a_ >> 15)) * (a_ | 1)) & M32
        t = ((t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296
    return r


def gauss(r):
    u1 = 1 - r(); u2 = r()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


HID, STEPS, NTR, NVA, OF_K, OF_STEPS, OF_PASS, INIT_SEED, SHUF_SEED = 16, 300, 100, 50, 3, 200, 0.1, 7, 11
DSPEC = {3: (1.0, 1), 4: (0.8, 2)}
KINDS = ['high', 'low', 'nozero', 'twice', 'shuffle']
_data = {}


def data_of(C):
    if C in _data: return _data[C]
    S, seed = DSPEC[C]; r = mulberry(seed)
    ctr = [(2 * math.cos(math.pi / 2 + 2 * math.pi * c / C), 2 * math.sin(math.pi / 2 + 2 * math.pi * c / C)) for c in range(C)]
    def blob(n_):
        X, y = [], []
        for c in range(C):
            for _ in range(n_):
                x0 = ctr[c][0] + S * gauss(r); x1 = ctr[c][1] + S * gauss(r); X.append([x0, x1]); y.append(c)
        return np.array(X), np.array(y)
    Xtr, ytr = blob(NTR); Xva, yva = blob(NVA)
    mu = Xtr.mean(0); sd = np.sqrt(((Xtr - mu) ** 2).mean(0))
    _data[C] = ((Xtr - mu) / sd, ytr, (Xva - mu) / sd, yva)
    return _data[C]


def init_p(C):
    r = mulberry(INIT_SEED)
    W1 = np.array([[gauss(r) for _ in range(HID)] for _ in range(2)])
    W2 = np.array([[0.01 * gauss(r) for _ in range(C)] for _ in range(HID)])
    return [W1, np.zeros(HID), W2, np.zeros(C)]


def argmax_js(z):
    am = np.zeros(len(z), dtype=int)
    for c in range(1, z.shape[1]):
        am = np.where(z[:, c] > z[np.arange(len(z)), am], c, am)
    return am


def loss_grad(p, X, y, twice, need):
    W1, b1, W2, b2 = p; N = len(X); C = len(b2)
    a = X @ W1 + b1; h = np.where(a > 0, a, 0.0); z = h @ W2 + b2
    with np.errstate(all='ignore'):
        e = np.exp(z); q = e / e.sum(1, keepdims=True)
        acc = float((argmax_js(z) == y).mean())
        oh = np.eye(C)[y]
        if not twice:
            L = float(np.sum(-np.log(q[np.arange(N), y]))) / N
            dz = (q - oh) / N
        else:
            e2 = np.exp(q); r2 = e2 / e2.sum(1, keepdims=True)
            L = float(np.sum(-np.log(r2[np.arange(N), y]))) / N
            dq = (r2 - oh) / N; dot = (dq * q).sum(1, keepdims=True); dz = q * (dq - dot)
        if not need: return L, acc, None
        gW2 = h.T @ dz; gb2 = dz.sum(0); dh = (dz @ W2.T) * (h > 0); gW1 = X.T @ dh; gb1 = dh.sum(0)
    return L, acc, [gW1, gb1, gW2, gb2]


_runs = {}


def run(C=3, lr=0.3, bug='none', k=0, steps=STEPS):
    key = (C, lr, bug, k, steps)
    if key in _runs: return _runs[key]
    Xtr, ytr, Xva, yva = data_of(C)
    if k:
        keep = np.array([(i % NTR) < k for i in range(len(Xtr))]); X, Y = Xtr[keep], ytr[keep]
    else:
        X, Y = Xtr, ytr
    rs = mulberry(SHUF_SEED); p = init_p(C); G = None
    tr, va, acc, gn = [], [], [], []; dead = inf = -1
    for t in range(steps + 1):
        y = Y
        if bug == 'shuffle':
            pm = list(range(len(Y)))
            for i in range(len(pm) - 1, 0, -1):
                j = math.floor(rs() * (i + 1)); pm[i], pm[j] = pm[j], pm[i]
            y = Y[pm]
        L, _, g = loss_grad(p, X, y, bug == 'twice', t < steps)
        Lv, av, _ = loss_grad(p, Xva, yva, bug == 'twice', False)
        tr.append(L); va.append(Lv); acc.append(av)
        if math.isnan(L): dead = t; break
        if inf < 0 and not math.isfinite(L): inf = t
        if t == steps: break
        G = [a + b for a, b in zip(G, g)] if (bug == 'nozero' and G is not None) else g
        with np.errstate(all='ignore'):
            gn.append(math.sqrt(sum(float((a ** 2).sum()) for a in G)))
            p = [a - lr * b for a, b in zip(p, G)]
    out = dict(tr=tr, va=va, acc=acc, gn=gn, dead=dead, inf=inf, p=p, C=C)
    _runs[key] = out
    return out


def check1(r):
    l = math.log(r['C']); fin = [v for v in r['tr'] if math.isfinite(v)]; e = fin[-1] if fin else float('nan')
    stays = r['dead'] < 0 and r['inf'] < 0 and abs(e - l) < 0.05 and all(abs(v - l) < 0.1 for v in r['tr'])
    return dict(start=r['tr'][0], startOk=abs(r['tr'][0] - l) < 0.1, stays=stays, cat='stays' if stays else 'leaves')


def check2(C, lr, bug):
    r = run(C, lr, 'none' if bug == 'shuffle' else bug, OF_K, OF_STEPS); e = r['tr'][-1]
    return dict(end=e, pass_=math.isfinite(e) and e < OF_PASS, dead=r['dead'], cat='pass' if (math.isfinite(e) and e < OF_PASS) else 'fail')


def check3(r):
    g0 = r['gn'][0]; fin = [v for v in r['gn'] if math.isfinite(v)]; tail = fin[-50:]; tm = sum(tail) / max(1, len(tail))
    bad = r['inf'] if r['inf'] >= 0 else r['dead']
    cat = 'nan' if 0 <= bad <= 20 else ('grows' if tm > 1.5 * g0 else ('stays' if tm / g0 > 0.2 else 'fades'))
    return dict(g0=g0, tail=tm, max=max(fin), cat=cat)


def curve_cat(r):
    l = math.log(r['C'])
    if r['dead'] >= 0 or r['inf'] >= 0 or max(r['tr']) > 1.5 * l: return 'blows'
    return 'flat' if check1(r)['stays'] else 'falls'


def curve_shape(r):
    # now / late: blows up (off the chart above 1.5 ln C, inf or NaN) within 20 steps / later; flat: never leaves ln C;
    # slow: the last 100 steps give more than a tenth of the drop; stall: levels off above 0.4 ln C; falls: levels off lower
    l = math.log(r['C']); tr = r['tr']
    off = next((t for t, v in enumerate(tr) if math.isfinite(v) and v > 1.5 * l), -1)
    bad = [x for x in (r['inf'], r['dead'], off) if x >= 0]
    if bad: return 'now' if min(bad) <= 20 else 'late'
    if check1(r)['stays']: return 'flat'
    n = len(tr) - 1; drop = tr[0] - tr[n]
    if tr[n - 100] - tr[n] > 0.1 * drop: return 'slow'
    return 'stall' if tr[n] > 0.4 * l else 'falls'


def kind_cfg(k, C):
    return dict(C=C, lr=10 if k == 'high' else (0.003 if k == 'low' else 0.3), bug=k if k in ('nozero', 'twice', 'shuffle') else 'none')


def sig(k, C):
    cf = kind_cfg(k, C); r = run(**cf)
    return dict(kind=k, curve=curve_shape(r), c1=check1(r)['cat'], c2=check2(C, cf['lr'], cf['bug'])['cat'], c3=check3(r)['cat'])


SETS = [[], ['c3'], ['c1'], ['c2'], ['c1', 'c3'], ['c2', 'c3'], ['c1', 'c2'], ['c1', 'c2', 'c3']]


def confirm_of(k, C):
    al = [sig(x, C) for x in KINDS]; me = next(s for s in al if s['kind'] == k)
    for s in SETS[1:]:
        if not [o for o in al if o['kind'] != k and all(o[c] == me[c] for c in s)]: return s
    return ['c1', 'c2', 'c3']


def pair_check(a, b, C):
    sa, sb = sig(a, C), sig(b, C)
    return next((c for c in ('c3', 'c1', 'c2') if sa[c] != sb[c]), None)


def min_checks(k, C):
    al = [sig(x, C) for x in KINDS]; me = next(s for s in al if s['kind'] == k)
    for s in SETS:
        if not [o for o in al if o['kind'] != k and o['curve'] == me['curve'] and all(o[c] == me[c] for c in s)]: return s
    return ['c1', 'c2', 'c3']


def regions(p, G=40, E=3):
    W1, b1, W2, b2 = p
    xs = np.array([[-E + (j + 0.5) * 2 * E / G, E - (i + 0.5) * 2 * E / G] for i in range(G) for j in range(G)])
    a = xs @ W1 + b1; h = np.where(a > 0, a, 0.0); z = h @ W2 + b2
    return argmax_js(z).reshape(G, G).tolist()


def close(a, b, rtol=1e-9):
    if isinstance(b, str) or isinstance(a, str):
        sa = a if isinstance(a, str) else ('NaN' if math.isnan(a) else ('inf' if a > 0 else '-inf') if not math.isfinite(a) else repr(a))
        return sa == b if isinstance(b, str) else False
    if not (math.isfinite(a) and math.isfinite(b)): return (math.isnan(a) and math.isnan(b)) or a == b
    return abs(a - b) <= rtol * max(1.0, abs(a), abs(b))


def cmp_series(name, py, js, rtol=1e-9, loose_from=None):
    # loose_from: in the last steps before a blow-up the run is chaotic and the summation order (NumPy's against the page's loops)
    # moves the 7th digit; those steps are compared within 1e-4
    check(f'{name}: length', len(py) == len(js), (len(py), len(js)))
    bad = [(i, a, b) for i, (a, b) in enumerate(zip(py, js)) if not close(a, b, 1e-4 if (loose_from is not None and i >= loose_from) else rtol)]
    check(f'{name}: values', not bad, bad[:3])


# ---------------------------------------------------------------- compare with the page
CFGS = [dict(C=C, **c) for C in (3, 4) for c in (dict(), dict(lr=10), dict(lr=0.003), dict(bug='nozero'), dict(bug='twice'), dict(bug='shuffle'), dict(k=5))]
CFGS += [dict(C=3, lr=3)]
CFGS += [dict(C=C, lr=lr, bug=b, k=OF_K, steps=OF_STEPS) for C in (3, 4) for (lr, b) in ((0.3, 'none'), (10, 'none'), (0.003, 'none'), (0.3, 'nozero'), (0.3, 'twice'))]

with sync_playwright() as pw:
    b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.evaluate("()=>{location.hash='train-debug'}"); pg.wait_for_timeout(500)
    T = lambda expr, *a: pg.evaluate(expr, *a)
    cs = T("()=>window.__td.consts")
    check('constants', (cs['HID'], cs['STEPS'], cs['NTR'], cs['NVA'], cs['OF_K'], cs['OF_STEPS'], cs['OF_PASS'], cs['INIT_SEED'], cs['SHUF_SEED']) == (HID, STEPS, NTR, NVA, OF_K, OF_STEPS, OF_PASS, INIT_SEED, SHUF_SEED), cs)
    for C in (3, 4):
        d = T("C=>window.__td.data(C)", C); Xtr, ytr, Xva, yva = data_of(C)
        cmp_series(f'data {C}: training x', Xtr.ravel().tolist(), [v for row in d['Xtr'] for v in row], 1e-12)
        cmp_series(f'data {C}: validation x', Xva.ravel().tolist(), [v for row in d['Xva'] for v in row], 1e-12)
        check(f'data {C}: labels', d['ytr'] == ytr.tolist() and d['yva'] == yva.tolist())
        ip = T("C=>window.__td.init(C)", C); p0 = init_p(C)
        cmp_series(f'init {C}', np.concatenate([a.ravel() for a in p0]).tolist(), [v for row in ip['W1'] for v in row] + ip['b1'] + [v for row in ip['W2'] for v in row] + ip['b2'], 1e-12)
    for cf in CFGS:
        tag = json.dumps(cf); js = T("c=>window.__td.run(c)", cf); py = run(**cf)
        rt = 1e-7 if (cf.get('bug') == 'nozero' or cf.get('lr') == 10) else 1e-9   # chaotic runs: summation order matters a little
        lf = py['dead'] - 10 if py['dead'] >= 0 else None
        for k in ('tr', 'va', 'acc', 'gn'): cmp_series(f'{tag} {k}', py[k], js[k], rt, lf)
        check(f'{tag}: inf and NaN steps', (py['inf'], py['dead']) == (js['inf'], js['dead']), ((py['inf'], py['dead']), (js['inf'], js['dead'])))
    # the blow-ups the page names
    for C, inf_, dead_ in ((3, 3, 7), (4, 4, 6)):
        r = run(C, 10); check(f'learning rate 10, {C} groups: inf at {inf_}, NaN at {dead_}', (r['inf'], r['dead']) == (inf_, dead_), (r['inf'], r['dead']))
    check('zero_grad() missing: NaN at 257 (3 groups) and 194 (4 groups)', (run(3, 0.3, 'nozero')['dead'], run(4, 0.3, 'nozero')['dead']) == (257, 194))
    for C in (3, 4):
        check(f'ln {C}', abs(T("C=>window.__td.check1({C})", C)['lnC'] - math.log(C)) < 1e-15)
    # checks, categories, minimal sets
    for C in (3, 4):
        for k in KINDS + ['none']:
            cf = kind_cfg(k, C) if k != 'none' else dict(C=C, lr=0.3, bug='none'); r = run(**cf)
            j1 = T("c=>window.__td.check1(c)", cf); p1 = check1(r)
            check(f'check 1 {C} {k}', (j1['stays'], j1['startOk']) == (p1['stays'], p1['startOk']) and close(p1['start'], j1['start']), (j1, p1))
            j2 = T("c=>window.__td.check2(c)", cf); p2 = check2(C, cf['lr'], cf['bug'])
            check(f'check 2 {C} {k}', j2['pass'] == p2['pass_'] and j2['dead'] == p2['dead'] and close(p2['end'], j2['end'], 1e-7), (j2, p2))
            j3 = T("c=>window.__td.check3(c)", cf); p3 = check3(r)
            check(f'check 3 {C} {k}', j3['cat'] == p3['cat'] and close(p3['g0'], j3['g0']) and close(p3['tail'], j3['tail'], 1e-7), (j3, p3))
            check(f'curve {C} {k}', T("c=>window.__td.curveCat(c)", cf) == curve_cat(r))
        for k in KINDS:
            check(f'signature {C} {k}', T("([k,C])=>window.__td.sig(k,C)", [k, C]) == sig(k, C), (T("([k,C])=>window.__td.sig(k,C)", [k, C]), sig(k, C)))
            check(f'minimal checks {C} {k}', T("([k,C])=>window.__td.minChecks(k,C)", [k, C]) == min_checks(k, C))
    expect4 = {'twice': [], 'nozero': [], 'shuffle': [], 'low': [], 'high': []}
    check('challenge minimal sets (4 groups): the curve alone tells the five apart', {k: min_checks(k, 4) for k in KINDS} == expect4, {k: min_checks(k, 4) for k in KINDS})
    shapes = {C: {k: sig(k, C)['curve'] for k in KINDS} for C in (3, 4)}
    check('curve shapes (3 and 4 groups)', all(shapes[C] == {'high': 'now', 'low': 'slow', 'nozero': 'late', 'twice': 'stall', 'shuffle': 'flat'} for C in (3, 4)), shapes)
    check('curve shape of the healthy run, 15 examples and learning rate 3: it falls and levels off low', all(curve_shape(run(C, **c)) == 'falls' for C in (3, 4) for c in (dict(), dict(k=5), dict(lr=3))))
    for C in (3, 4):
        for k in KINDS:
            check(f'checks that confirm {k} ({C} groups)', T("([k,C])=>window.__td.confirmOf(k,C)", [k, C]) == confirm_of(k, C), (T("([k,C])=>window.__td.confirmOf(k,C)", [k, C]), confirm_of(k, C)))
            for a in KINDS:
                if a != k: check(f'hint check {a} against {k} ({C} groups)', T("([a,k,C])=>window.__td.pairCheck(a,k,C)", [a, k, C]) == pair_check(a, k, C))
    check('confirming checks (4 groups): check 3 for four of them, check 1 for the shuffled labels, checks 3 and 1 for the low rate',
          {k: confirm_of(k, 4) for k in KINDS} == {'high': ['c3'], 'low': ['c1', 'c3'], 'nozero': ['c3'], 'twice': ['c3'], 'shuffle': ['c1']}, {k: confirm_of(k, 4) for k in KINDS})
    check('every mystery run is told apart by its minimal set', all(T("([k,o])=>window.__td.tellsApart(k,4,o)", [k, min_checks(k, 4)]) for k in KINDS))
    check('the signatures that make the lab', sig('shuffle', 3)['curve'] == 'flat' and sig('high', 3)['c3'] == 'nan' and sig('nozero', 3)['c3'] == 'grows'
          and sig('low', 3)['c3'] == 'stays' and sig('twice', 3)['c3'] == 'fades' and check2(3, 0.3, 'shuffle')['pass_'] and not check2(3, 0.3, 'twice')['pass_'])
    # the numbers the text states
    h = run(3); sm = run(3, k=5)
    check('healthy: 1.084 → 0.135, 89.3% right', f"{h['tr'][0]:.3f}" == '1.084' and f"{h['tr'][300]:.3f}" == '0.135' and f"{h['acc'][300]*100:.1f}" == '89.3')
    check('shuffled: 1.102 at 0 and 300', f"{run(3, bug='shuffle')['tr'][0]:.3f}" == '1.102' and f"{run(3, bug='shuffle')['tr'][300]:.3f}" == '1.102')
    check('15 examples: training below 0.05, validation rises from its minimum', sm['tr'][300] < 0.05 and sm['va'][300] > min(sm['va']) + 0.1, (sm['tr'][300], min(sm['va']), sm['va'][300]))
    check('softmax twice stays above its floor ln(1 + 2/e)', min(run(3, bug='twice')['tr']) > math.log(1 + 2 / math.e))
    check('learning rate 3 trains (transfer answer “It trains”)', curve_cat(run(3, 3)) == 'falls' and T("()=>window.__td.exits().transferA") == [1, 2], T("()=>window.__td.exits().transferA"))
    for C in (3, 4):
        for nm in ('healthy', 'shuffle'):
            cf = dict(C=C) if nm == 'healthy' else dict(C=C, bug='shuffle')
            check(f'regions {C} {nm}', T("c=>window.__td.regions(c)", cf) == regions(run(**cf)['p']))
    for s in range(1, 9):
        d = T("s=>window.__td.drill(s)", s); check(f'drill {s}: ln C', abs(d['ln'] - math.log(d['C'])) < 1e-12 and d['C'] in (2, 5, 10, 100, 1000, 32000), d)
    # the teaser's sentences carry the engine's numbers
    reads = {k: T("k=>window.__tzTd?window.__tzTd.read(k):''", k) for k in ('none', 'high', 'nozero')}
    check('teaser: three picks, no shuffled labels (part 2 asks about them, R3.8)', T("()=>window.__tzTd.opts") == ['none', 'high', 'nozero'], T("()=>window.__tzTd.opts"))
    check('teaser: no bug', f"{h['tr'][300]:.3f}" in reads['none'], reads['none'])
    check('teaser: rate too high', f"step {run(3, 10)['dead']}" in reads['high'] and f"{max(v for v in run(3, 10)['tr'] if math.isfinite(v)):.3f}" in reads['high'], reads['high'])
    check('teaser: no zero_grad()', f"step {run(3, bug='nozero')['dead']}" in reads['nozero'] and f"{min(run(3, bug='nozero')['tr'][:60]):.3f}" in reads['nozero'], reads['nozero'])
    check('teaser sentences at most 25 words', all(len(v.split()) <= 25 for v in reads.values()), {k: len(v.split()) for k, v in reads.items()})
    # the minimal implementation
    code = T("()=>{const e=document.querySelector('#td-code'); return e?e.textContent:''}")
    lines = code.split('\n')
    check('minimal implementation: at most 40 lines, 74 characters a line', 0 < len(lines) <= 40 and max(len(l) for l in lines) <= 74, (len(lines), max(len(l) for l in lines) if lines else 0))
    check('no page errors', not errs, errs[:3])
    b.close()

print(f'{n - len(fails)}/{n} checks passed')
if fails: print('FAILED:', fails[:20]); sys.exit(1)
