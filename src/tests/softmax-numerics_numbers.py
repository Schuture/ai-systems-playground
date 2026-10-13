"""Numbers test for the softmax-numerics lab (CONTRACT §6).

An independent engine in Python (NumPy float32 and float16 rounding; BF16 by an exact round-to-nearest-even with 8 significant bits)
recomputes every number the lab states and compares it with the page's own engine (window.__sm):
- the safe window of exp per format, checked against NumPy's own float32 and float16 exp;
- the rounding of FP32, FP16 and BF16 on many values; naive and max-subtracted softmax, bit for bit, on every dataset;
- the mission (NaN at 92, 94.4% with the max subtracted; naive = stable from 88 down to −90; NaN from 89; −100 off by 2 points;
  NaN everywhere from −104), part 4 (attention scores in FP32, BF16 and FP16), part 5 (cross-entropy three ways), part 6 (online
  softmax, block by block), the challenge's six fixes and the smallest one, the scenarios, the exits' answers, the interview drill;
- the teaser against the lab (a temperature slider), its NaN range, and its sentence's length;
- the gates' numbers: part 3's range (88.5 down to −94), part 5's gap (103), the sum-fit values (88.67 FP32, 10.75 FP16);
- the Connect code: run with NumPy, its output equals its comment lines; at most 40 lines of at most 74 characters.
Run: python3 tests/softmax-numerics_numbers.py"""
import math, pathlib, re, subprocess, sys
import numpy as np
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/softmax-numerics-num.html', '--only', 'softmax-numerics'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'softmax-numerics-num-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'softmax-numerics-num.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'softmax-numerics-num-wrapped.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


# ---------------- the independent engine ----------------
def r32(x): return float(np.float32(x))


def r16(x):
    with np.errstate(over='ignore'): return float(np.float16(x))


def rbf(x):
    if x == 0 or math.isnan(x) or math.isinf(x): return x
    if abs(x) >= (2 - 2 ** -8) * 2 ** 127: return math.copysign(math.inf, x)
    m, e = math.frexp(abs(x)); E = max(e - 1, -126); q = 2.0 ** (E - 7); r = abs(x) / q; f = math.floor(r); d = r - f
    k = f + 1 if d > 0.5 else (f if d < 0.5 else (f if f % 2 == 0 else f + 1))
    return math.copysign(k * q, x)


R = {'fp32': r32, 'fp16': r16, 'bf16': rbf}


def ex(x):
    try: return math.exp(x)
    except OverflowError: return math.inf


def div(a, b):
    if math.isnan(a) or math.isnan(b): return math.nan
    if math.isinf(a) and math.isinf(b): return math.nan
    if b == 0: return math.nan if a == 0 else math.copysign(math.inf, a)
    return a / b


def sub(a, b):
    if math.isinf(a) and math.isinf(b) and a == b: return math.nan
    return a - b


def ssum(f, e):
    s = 0.0
    for v in e: s = R[f](s + v) if not (math.isinf(s) and math.isinf(v) and s != v) else math.nan
    return s


def naive(z, f):
    zz = [R[f](v) for v in z]; e = [R[f](ex(v)) if not math.isnan(v) else math.nan for v in zz]; s = ssum(f, e)
    return {'e': e, 's': s, 'p': [R[f](div(v, s)) for v in e]}


def stable(z, f):
    zz = [R[f](v) for v in z]; m = max(zz); d = [R[f](sub(v, m)) for v in zz]
    e = [R[f](ex(v)) if not math.isnan(v) else math.nan for v in d]; s = ssum(f, e)
    return {'m': m, 'd': d, 'e': e, 's': s, 'p': [R[f](div(v, s)) for v in e]}


def exact(z):
    m = max(z)
    if math.isinf(m): return [math.nan] * len(z)
    e = [math.exp(v - m) for v in z]; s = sum(e); return [v / s for v in e]


def err(p, ref):
    if any(math.isnan(v) for v in p + ref): return math.nan
    return max(abs(a - b) for a, b in zip(p, ref)) * 100


def num(v):
    if isinstance(v, str): return {'NaN': math.nan, 'Infinity': math.inf, '-Infinity': -math.inf}[v]
    return v


def nums(a): return [num(v) for v in a]


def same(a, b):
    a, b = nums(a), nums(b)
    return len(a) == len(b) and all((math.isnan(x) and math.isnan(y)) or x == y for x, y in zip(a, b))


with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300); pg.evaluate("()=>{location.hash='softmax-numerics'}"); pg.wait_for_timeout(500)
    J = lambda js, *a: pg.evaluate(js, *a)
    C = J("()=>window.__sm.consts")
    W = J("()=>window.__sm.win()")

    # ---- the safe window of exp, against NumPy's own exp ----
    with np.errstate(over='ignore', under='ignore'):
        hi32, lo32 = W['fp32']['hi'], W['fp32']['lo']
        check('FP32 window: 88.72 and −103.97', f"{hi32:.2f}" == '88.72' and f"{lo32:.2f}" == '-103.97', (hi32, lo32))
        check('FP32: NumPy exp finite just inside the top edge, Inf just outside', np.isfinite(np.exp(np.float32(hi32 - 1e-4))) and np.isinf(np.exp(np.float32(hi32 + 1e-4))))
        check('FP32: NumPy exp > 0 just inside the bottom edge, 0 just outside', np.exp(np.float32(lo32 + 0.02)) > 0 and np.exp(np.float32(lo32 - 0.02)) == 0)
        hi16, lo16 = W['fp16']['hi'], W['fp16']['lo']
        check('FP16 window: 11.09 and −17.33', f"{hi16:.2f}" == '11.09' and f"{lo16:.2f}" == '-17.33', (hi16, lo16))
        check('FP16: NumPy exp of 11.0859375 finite, of 11.09375 Inf', np.isfinite(np.exp(np.float16(11.0859375))) and np.isinf(np.exp(np.float16(11.09375))))
        check('FP16: NumPy exp > 0 at −17.25, 0 at −17.5', np.exp(np.float16(-17.25)) > 0 and np.exp(np.float16(-17.5)) == 0)
        check('FP16 largest number 65,504 and smallest 2^−24', W['fp16']['max'] == 65504 == float(np.finfo(np.float16).max) and W['fp16']['sub'] == 2 ** -24)
        check('FP32 largest number', W['fp32']['max'] == float(np.finfo(np.float32).max))
    hib, lob = W['bf16']['hi'], W['bf16']['lo']
    check('BF16 window: 88.72 and −92.88', f"{hib:.2f}" == '88.72' and f"{lob:.2f}" == '-92.88', (hib, lob))
    check('BF16: exp just inside the edges finite and above 0, just outside Inf and 0',
          math.isfinite(rbf(math.exp(hib - 1e-3))) and math.isinf(rbf(math.exp(hib + 1e-3))) and rbf(math.exp(lob + 1e-3)) > 0 and rbf(math.exp(lob - 1e-3)) == 0)
    check('BF16 largest number ≈ 3.3895 × 10^38', abs(W['bf16']['max'] - 3.3895313892515355e38) < 1e30, W['bf16']['max'])

    # ---- rounding, value by value ----
    rng = np.random.default_rng(7)
    vals = list(rng.normal(0, 1, 300) * 10.0 ** rng.integers(-40, 39, 300)) + [65504, 65519.99, 65520, 1e-8, 2 ** -25, 2 ** -24 * 1.5, 3.3895e38, 3.39e38, 1 / 3, -2.5e-41]
    bad = []
    for f in ('fp32', 'fp16', 'bf16'):
        got = J("([f,v])=>v.map(x=>{const r=window.__sm.rnd(f,x); return Number.isFinite(r)?r:String(r)})", [f, [float(v) for v in vals]])
        for v, g in zip(vals, got):
            w = R[f](float(v))
            if not ((isinstance(g, str) and math.isinf(w)) or (not isinstance(g, str) and float(g) == w)): bad.append((f, v, g, w))
    check('FP32, FP16 and BF16 rounding equal the independent rounding on 310 values', not bad, bad[:4])

    # ---- naive and stable softmax, bit for bit ----
    sets = [C['LM']['z'], C['AT']['z'], C['CHS']['z'], C['ON']['z'], [10.5, 10.5], [-1e9] * 6] + [list(rng.normal(0, 8, 6) + rng.integers(-120, 120)) for _ in range(40)]
    bad = []
    for z in sets:
        for f in ('fp32', 'fp16', 'bf16'):
            jn, js = J("([z,f])=>window.__sm.naive(z,f)", [z, f]), J("([z,f])=>window.__sm.stable(z,f)", [z, f])
            pn, ps = naive(z, f), stable(z, f)
            if not (same(jn['p'], pn['p']) and same(js['p'], ps['p']) and same([jn['s']], [pn['s']])): bad.append((f, z[:3], jn['p'][:2], pn['p'][:2]))
    check('naive and max-subtracted softmax equal the independent engine on 46 rows × 3 formats', not bad, bad[:3])

    # ---- the mission ----
    LMz = C['LM']['z']
    sh = lambda t: [v - max(LMz) + t for v in LMz]
    n92 = naive(sh(92), 'fp32')
    check('mission: naive at 92 = NaN, NaN, 0, 0, 0, 0', [str(v) for v in n92['p']] == ['nan', 'nan', '0.0', '0.0', '0.0', '0.0'], n92['p'])
    s92 = stable(sh(92), 'fp32')
    check('mission: max subtracted gives “Paris” 94.4%', J("x=>window.__sm.fmt.fP(x)", s92['p'][0]) == '94.4%', s92['p'][0])
    worst = max(err(naive(sh(t), 'fp32')['p'], stable(sh(t), 'fp32')['p']) for t in range(-90, 89))
    check('mission: naive equals max subtracted (< 10^−4 pts) for every integer top logit from 88 down to −90', worst < 1e-4, worst)
    check('mission: naive is NaN at every top logit from 89 to 130 (half steps)', all(any(math.isnan(v) for v in naive(sh(t / 2), 'fp32')['p']) for t in range(178, 261)))
    check('mission: 88.5 still works', not any(math.isnan(v) for v in naive(sh(88.5), 'fp32')['p']) and err(naive(sh(88.5), 'fp32')['p'], exact(sh(88.5))) < 1e-5)
    e100 = err(naive(sh(-100), 'fp32')['p'], exact(sh(-100)))
    check('mission: at −100 the naive one is 2.0 points off (96.4%)', abs(e100 - 2.03) < 0.01 and J("x=>window.__sm.fmt.fP(x)", naive(sh(-100), 'fp32')['p'][0]) == '96.4%', e100)
    check('mission: NaN everywhere from −104 down to −130', all(all(math.isnan(v) for v in naive(sh(t / 2), 'fp32')['p']) for t in range(-260, -207)))
    check('mission: max subtracted is within 10^−5 pts of exact at every half step', max(err(stable(sh(t / 2), 'fp32')['p'], exact(sh(t / 2))) for t in range(-260, 261)) < 1e-5)
    check('mission: the gate value 60 (Show me) works', err(naive(sh(60), 'fp32')['p'], exact(sh(60))) < 0.01)
    okh = [t / 2 for t in range(-260, 261) if not any(math.isnan(v) for v in naive(sh(t / 2), 'fp32')['p']) and err(naive(sh(t / 2), 'fp32')['p'], exact(sh(t / 2))) < 0.01]
    lo_ = 88.5
    while lo_ - 0.5 in okh: lo_ -= 0.5
    check('part 3: the naive softmax is within 0.01 pts from 88.5 down to −94 (the hint’s range), and not at −94.5', 88.5 in okh and 89 not in okh and lo_ == -94 and J("()=>window.__sm.okR()") == {'lo': -94, 'hi': 88.5}, (lo_, J("()=>window.__sm.okR()")))
    lo, hi = 88.0, 88.72
    for _ in range(60):
        mid = (lo + hi) / 2
        if math.isinf(naive(sh(mid), 'fp32')['s']): hi = mid
        else: lo = mid
    fit_lab = J("z=>window.__sm.fitTop(z,'fp32')", sh(92))
    check('Show the math: the naive sum is Inf from a largest logit of about 88.67 (FP32)', f"{hi:.2f}" == '88.67' and f"{fit_lab:.2f}" == '88.67', (hi, fit_lab))
    ATz = C['AT']['z']; sha = lambda t: [v - max(ATz) + t for v in ATz]
    lo, hi = 10.0, 11.09
    for _ in range(60):
        mid = (lo + hi) / 2
        if math.isinf(naive(sha(mid), 'fp16')['s']): hi = mid
        else: lo = mid
    fit_at = J("z=>window.__sm.fitTop(z,'fp16')", ATz)
    check('Show the math: the attention row’s FP16 sum is Inf from about 10.75', abs(hi - 10.75) < 0.01 and f"{fit_at:.2f}" == '10.75', (hi, fit_at))
    check('mission: “Paris” and “the” overflow at 92 (e^92, e^89 > FP32 max)', naive(sh(92), 'fp32')['e'][:3] == [math.inf, math.inf, r32(math.exp(87))])

    # ---- part 4: attention scores ----
    AT = C['AT']['z']; ea = exact(AT)
    check('part 4: the scores reach 13.5', max(AT) == 13.5)
    check('part 4: FP16 naive is NaN (two exps overflow)', any(math.isnan(v) for v in naive(AT, 'fp16')['p']) and sum(math.isinf(v) for v in naive(AT, 'fp16')['e']) == 2)
    e16 = err(stable(AT, 'fp16')['p'], ea); check('part 4: FP16 max subtracted 0.036 pts off', abs(e16 - 0.0359) < 0.0005, e16)
    eb_n, eb_s = err(naive(AT, 'bf16')['p'], ea), err(stable(AT, 'bf16')['p'], ea)
    check('part 4: BF16 naive 0.046, max subtracted 0.052 pts off', abs(eb_n - 0.0459) < 0.0005 and abs(eb_s - 0.0517) < 0.0005, (eb_n, eb_s))
    check('part 4: FP32 both below 10^−5 pts', err(naive(AT, 'fp32')['p'], ea) < 1e-5 and err(stable(AT, 'fp32')['p'], ea) < 1e-5)
    check('part 4: exact weight of “cat” 71.06%', abs(ea[1] - 0.71058) < 1e-4, ea[1])
    st = J("()=>{const s=window.__sm.stable([8,13.5,9,11,9.75,12.25],'fp16'); return s}")
    check('part 4: the lab agrees (FP16 max subtracted)', same(st['p'], stable(AT, 'fp16')['p']))

    # ---- part 5: cross-entropy ----
    top = C['CE']['top']; y = C['CE']['y']

    def ce(gap, f='fp32'):
        z = top + [top[0] - gap]; nn, ss = naive(z, f), stable(z, f)
        nl = lambda v: R[f](-math.log(v)) if v > 0 else (math.inf if v == 0 else math.nan)
        lse = R[f](R[f](ss['m'] + R[f](math.log(ss['s']))) - R[f](z[y]))
        m = max(z); exa = m + math.log(sum(math.exp(v - m) for v in z)) - z[y]
        return nl(nn['p'][y]), nl(ss['p'][y]), lse, exa
    a, s_, l, e = ce(120)
    check('part 5: at 120 below, naive and max-subtracted loss are Inf', math.isinf(a) and math.isinf(s_), (a, s_))
    check('part 5: log-sum-exp 120.09, exact 120.0861', abs(l - 120.08612060546875) < 1e-9 and abs(e - 120.08612108542543) < 1e-9, (l, e))
    jc = J("()=>window.__sm.ce([8,5.5,3,1,-0.5,-112],5,'fp32')")
    check('part 5: the lab agrees at 120', num(jc['naive']) == math.inf and num(jc['soft']) == math.inf and jc['lse'] == l, jc)
    check('part 5: naive loss Inf for every gap from 104 to 150', all(math.isinf(ce(g)[0]) for g in range(104, 151)))
    check('part 5: naive loss finite for every gap from 0 to 103', all(math.isfinite(ce(g)[0]) for g in range(0, 104)))
    check('part 5: Show me (90) gives 90.09, the same as log-sum-exp', abs(ce(90)[0] - 90.09) < 0.005 and abs(ce(90)[2] - 90.09) < 0.005, ce(90))
    check('part 5: the hint’s gap, 103, is the largest with a finite naive loss', J("()=>window.__sm.ceOK()") == 103 and not math.isfinite(ce(104)[0]))
    check('part 5: at 103 the naive loss is 103.28 (exact 103.09)', abs(ce(103)[0] - 103.2789) < 0.001, ce(103))
    check('part 5: log-sum-exp within 10^−4 of exact at every gap', max(abs(ce(g)[2] - ce(g)[3]) for g in range(0, 151)) < 1e-4)
    for g in (0, 50, 103, 120, 150):
        jj = J("g=>window.__sm.ce([8,5.5,3,1,-0.5,8-g],5,'fp32')", g); pa = ce(g)
        check(f'part 5: the lab’s three losses equal the engine at gap {g}', same([jj['naive'], jj['soft'], jj['lse']], list(pa[:3])), (jj, pa))

    # ---- part 6: online softmax ----
    Z, B = C['ON']['z'], C['ON']['B']
    m, s, rows = -math.inf, 0.0, []
    for k in range(0, len(Z), B):
        blk = [r32(v) for v in Z[k:k + B]]; mn = max([m] + blk); first = m == -math.inf
        sc = 0.0 if first else r32(math.exp(r32(m - mn))); s1 = 0.0 if first else r32(s * sc)
        for v in blk: s1 = r32(s1 + r32(math.exp(r32(v - mn))))
        rows.append((m, mn, sc, s1)); m, s = mn, s1
    jo = J("()=>window.__sm.online([2,4.5,3,7.5,6,1.5,9,5.5],2,'fp32')")
    check('part 6: the max rises in blocks 1, 2 and 4', [r[1] for r in rows] == [4.5, 7.5, 7.5, 9.0] and [st['raised'] for st in jo['steps']] == [False, True, False, True])
    check('part 6: the rescale factors e^−3 and e^−1.5', abs(rows[1][2] - 0.049787067) < 1e-8 and abs(rows[3][2] - 0.22313017) < 1e-8, rows)
    check('part 6: the running sums 1.082, 1.065, 1.291, 1.318', [round(r[3], 3) for r in rows] == [1.082, 1.065, 1.291, 1.318], [r[3] for r in rows])
    check('part 6: the lab’s steps equal the engine', [st['s2'] for st in jo['steps']] == [r[3] for r in rows] and [num(st['sc']) for st in jo['steps']] == [r[2] for r in rows])
    po = [r32(r32(math.exp(r32(r32(v) - m))) / s) for v in Z]
    check('part 6: online equals two-pass bit for bit (difference 0)', po == stable(Z, 'fp32')['p'] and jo['diff'] == 0, (po, stable(Z, 'fp32')['p']))
    check('part 6: “run” 75.9%', J("x=>window.__sm.fmt.fP(x)", po[6]) == '75.9%')
    # the picture of part 6 (R6.27): one piece per score read, each e^(z − m) against the current max; the pieces add up to s, the
    # max's own piece is 1, the dashed bar before a rescale adds up to the old sum, and every old piece shrinks by the same factor
    SC = J("()=>window.__sm.ONSC"); bad = []
    for k in range(1, len(Z) // B + 1):
        ob = J("k=>window.__sm.onBars(k)", k); m_, sc_ = rows[k - 1][1], rows[k - 1][2]
        want = [r32(math.exp(r32(r32(v) - m_))) for v in Z[:k * B]]
        if [q['e'] for q in ob['now']] != want: bad.append((k, 'pieces'))
        if abs(sum(want) - rows[k - 1][3]) > 1e-6 or max(want) != 1.0: bad.append((k, 'sum', sum(want)))
        if [q['nw'] for q in ob['now']] != [i >= (k - 1) * B for i in range(k * B)]: bad.append((k, 'new'))
        if ob['raised']:
            pre = [q['e'] for q in ob['before']]
            if abs(sum(pre) - ob['s0']) > 1e-6: bad.append((k, 'before'))
            if any(abs(q * sc_ - n_) > 1e-7 for q, n_ in zip(pre, want)): bad.append((k, 'factor'))
        elif ob['before'] is not None: bad.append((k, 'dashed bar without a rescale'))
        if max(sum(want), ob['s0']) >= SC: bad.append((k, 'off the scale'))
    check('part 6 picture: pieces e^(z − m) add up to s in every block, the dashed bar to the old sum, old pieces × the factor, all on the 1.5 scale', not bad, bad)

    # ---- the challenge ----
    CH = C['CHS']['z']; ec = exact(CH)
    fixes = {'fp32': lambda: naive(CH, 'fp32'), 'max': lambda: stable(CH, 'fp16'), 'bf16': lambda: naive(CH, 'bf16'),
             'half': lambda: naive([v / 2 for v in CH], 'fp16'), 'clamp': lambda: naive([min(v, 11) for v in CH], 'fp16'), 'both': lambda: stable(CH, 'fp32')}
    want = {'fp32': (True, 1.2e-6), 'max': (True, 0.0133), 'bf16': (False, 0.182), 'half': (False, 14.2), 'clamp': (False, 50.99), 'both': (True, 1e-6)}
    for k, fn in fixes.items():
        pr = fn(); e_ = err(pr['p'], ec); je = J("k=>window.__sm.chEval(k)", k)
        ok = e_ < 0.1
        check(f'challenge {k}: goal {"met" if want[k][0] else "missed"}, largest error ≈ {want[k][1]}', ok == want[k][0] and je['ok'] == ok and (abs(e_ - want[k][1]) < 0.01 * max(1, want[k][1]) if want[k][1] > 1e-3 else e_ < 1e-5), (e_, je))
        check(f'challenge {k}: the lab’s weights equal the engine', same(je['p'], pr['p']))
    check('challenge: the initial setup returns NaN', any(math.isnan(v) for v in naive(CH, 'fp16')['p']) and num(J("()=>window.__sm.chEval('start')")['err']) != num(J("()=>window.__sm.chEval('start')")['err']))
    check('challenge: clamping overflows the sum (three e^11 ≈ 59,900 each)', math.isinf(naive([min(v, 11) for v in CH], 'fp16')['s']) and abs(r16(math.exp(11)) - 59900) < 50 and all(v == 0 for v in naive([min(v, 11) for v in CH], 'fp16')['p']))
    check('challenge: e^13.5 and e^13 overflow FP16, e^12.25 does not exist below 65,504? (it overflows too)', math.isinf(r16(math.exp(13.5))) and math.isinf(r16(math.exp(13))) and math.isinf(r16(math.exp(12.25))))
    check('challenge: the smallest change is “subtract the max, stay in FP16”', J("()=>window.__sm.chBest()") == 'max')
    cho = {o['id']: o for o in C['CHO']}
    check('challenge: bytes per stored score: FP32 copy 4, FP32 arithmetic on FP16 scores 2, the others 2', cho['fp32']['bytes'] == 4 and cho['fp32']['store'] == 'fp32' and cho['both']['bytes'] == 2 and cho['both']['store'] == 'fp16' and all(cho[k]['bytes'] == 2 for k in ('max', 'bf16', 'half', 'clamp')), cho)
    check('challenge: the scores were never shown in the guided parts', sorted(CH) != sorted(AT) and sorted(CH) != sorted(LMz))
    s16 = 0.0
    for _ in range(4096): s16 = r16(s16 + 1)
    check('FP16 sum of 4,096 ones stops at 2,048', s16 == 2048)
    check('challenge: top weight is “time” (index 3)', CH.index(max(CH)) == 3 and C['CHS']['toks'][3] == 'time')

    # ---- scenarios ----
    pad = naive([-1e9] * 6, 'fp32')['p']; padi = stable([-math.inf] * 6, 'fp32')['p']
    check('scenario pad: −1e9 gives 1/6 each, −Inf gives NaN with the max subtracted', all(abs(v - 1 / 6) < 1e-7 for v in stable([-1e9] * 6, 'fp32')['p']) and all(math.isnan(v) for v in padi))
    check('scenario pad: in FP16, −1e9 rounds to −Inf', math.isinf(r16(-1e9)))
    check('scenario pad: with −1e9 the naive softmax is 0 ÷ 0 = NaN', all(math.isnan(v) for v in pad))
    jp = J("()=>window.__sm.stable([-1e9,-1e9,-1e9,-1e9,-1e9,-1e9],'fp32')")
    check('scenario pad: the lab agrees', same(jp['p'], stable([-1e9] * 6, 'fp32')['p']))
    check('scenarios: four presets', [x['id'] for x in C['PRESETS']] == ['samp', 'attn', 'loss', 'pad'])

    # ---- exits ----
    ea_ = J("()=>window.__sm.exits()")['transferA']
    q1 = naive([10.5, 10.5], 'fp16')['p']
    check('exits: FP16 two logits of 10.5 give 0 and 0 (the sum overflows)', q1 == [0.0, 0.0] and ea_[0] == 1, (q1, ea_))
    check('exits: running sum 2.5 × e^−2 ≈ 0.34 is option 3', abs(2.5 * math.exp(-2) - 0.338) < 0.001 and ea_[1] == 2, ea_)
    check('exits: e^10.5 in FP16 ≈ 36,320', r16(math.exp(10.5)) == 36320.0)

    # ---- the interview drill ----
    bad = []
    for sd in range(1, 41):
        d = J("s=>window.__sm.drill(s)", sd); z = d['z']
        if not (same(d['p'], naive(z, 'fp16')['p']) and same(d['ps'], stable(z, 'fp16')['p']) and 9 <= z[0] <= 15 and z[0] > z[1] >= z[2]): bad.append((sd, z))
    check('drill: 40 seeds equal the engine, logits 9 to 15, in order', not bad, bad[:3])
    kinds = set()
    for sd in range(1, 41):
        z = J("s=>window.__sm.drill(s)", sd)['z']; pn = naive(z, 'fp16')['p']
        kinds.add('nan' if any(math.isnan(v) for v in pn) else ('zero' if all(v == 0 for v in pn) else 'ok'))
    check('drill: seeds give NaN, all-zero and working cases', kinds == {'nan', 'zero', 'ok'}, kinds)

    # ---- the teaser against the lab: a temperature slider; the max-subtracted bar only where the naive one fails (R3.8) ----
    Z10 = C['LM']['z']
    zt = lambda k: [v * 10 / k for v in Z10]
    bad = []
    for k in range(5, 101):
        tm = J("t=>window.__tzSm.model(t)", k)
        if not (same(tm['n'], naive(zt(k), 'fp32')['p']) and same(tm['st'], stable(zt(k), 'fp32')['p'])): bad.append(k)
    check('teaser: its bars equal the lab’s engine at every temperature from 0.05 to 1.00', not bad, bad[:5])
    nank = [k for k in range(5, 101) if any(math.isnan(v) for v in naive(zt(k), 'fp32')['p'])]
    check('teaser: the naive softmax is NaN exactly at temperatures 0.05 to 0.10', nank == list(range(5, 11)), nank)
    for k in (5, 10, 11, 50, 100):
        rd = re.sub('<[^>]+>', '', J("t=>window.__tzSm.read(t)", k))
        check(f'teaser: sentence at temperature {k / 100:.2f} has at most 25 words', len(rd.split()) <= 25, rd)
    r10, r11 = J("()=>window.__tzSm.read(10)"), J("()=>window.__tzSm.read(11)")
    check('teaser: at 0.10 the largest logit is 92, the sentence says NaN and 94.4%', 'NaN' in r10 and '94.4%' in r10 and ' 92' in r10, r10)
    p11 = J("x=>window.__sm.fmt.fP(x)", naive(zt(11), 'fp32')['p'][0])
    check('teaser: at 0.11 the naive softmax works and the sentence names no subtraction', 'subtract' not in r11.lower() and ' max' not in r11 and p11 in r11, (r11, p11))
    check('teaser: no sentence calls the logits “top”', all('top logit' not in J("t=>window.__tzSm.read(t)", k) for k in (5, 10, 11, 100)))

    # ---- the Connect code ----
    code = C['CODE']; lines = code.split('\n')
    check('Connect: at most 40 lines', len(lines) <= 40, len(lines))
    check('Connect: at most 74 characters per line', max(len(x) for x in lines) <= 74, max(lines, key=len))
    (P / 'out' / 'softmax-numerics-code.py').write_text(code)
    out = subprocess.run([sys.executable, str(P / 'out' / 'softmax-numerics-code.py')], capture_output=True, text=True)
    printed = out.stdout.strip().split('\n'); comments = [x[2:] for x in lines if x.startswith('# ')]
    check('Connect: runs without error', out.returncode == 0 and not out.stderr.strip(), out.stderr[:300])
    check('Connect: the printed output equals the comment lines', printed == comments, (printed, comments))

    check('no page errors', not errs, errs[:3])
    b.close()

print(f'{n - len(fails)}/{n} checks passed')
sys.exit(1 if fails else 0)
