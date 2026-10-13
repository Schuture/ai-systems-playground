"""Golden numbers for the scaling-law lab (guide R9.2, CONTRACT §6).

Derived here independently of the JS engine, with NumPy:
- the fitted loss L(N, D) = E + A/N^α + B/D^β with Epoch AI's constants exactly as printed in epoch-research/analyzing-chinchilla
  data_analysis.ipynb (cell 25), and the original Approach-3 constants as quoted in karpathy/nanoGPT scaling_laws.ipynb (cell 14);
- C = 6ND; the lowest point of an iso-FLOP curve found by a fine grid search plus golden-section refinement over log N (not the
  closed form the page uses); the size that reaches a target loss with the least training + serving compute, 6N·D_need(N) + 2NT,
  found the same way; H100 GPU-days at 989.5 TFLOP/s and the assumed utilization.
Checks: part 1's three plans (same compute, their losses and terms), part 2 (GPU-days, the fleet sentence), part 3 (the 10× shrink
factors), part 4 (the lowest point for every budget and the 17-to-22 range), part 5 (Llama 3 8B's fitted loss, the cheapest sizes
for every served-token choice, the 5.5× ratio), part 6, the four scenarios (rendered text), the challenge (25 plans, 13 passing, the
cheapest and the training-only best), the fidelity card's ratios and GPU hours, the capability layer, the interview drill, the home
tile, and the Connect card's code (run here).
Run: python3 tests/scaling-law_numbers.py"""
import math, pathlib, subprocess, sys, io, contextlib
import numpy as np
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/scaling-law.html', '--only', 'scaling-law'], check=True, cwd=P, stdout=subprocess.DEVNULL)
OUT = P / 'out' / 'scaling-law'; OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'wrapped-num.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'scaling-law.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-num.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


rel = lambda a, b, tol=1e-6: abs(a - b) <= tol * max(abs(a), abs(b))

# ---------- the derivation
EPOCH = dict(E=1.81686, A=482.00572, B=2085.43420, al=0.34781, be=0.36585)     # notebook cell 25
CHIN = dict(E=1.69, A=406.4, B=410.7, al=0.34, be=0.28)                        # nanoGPT cell 14
PEAK, DAY = 989.5e12, 86400.0


def L(N, D, f=EPOCH): return f['E'] + f['A'] / N ** f['al'] + f['B'] / D ** f['be']


def argmin_log(fun, lo, hi):
    xs = np.linspace(lo, hi, 20001); v = np.array([fun(10 ** x) for x in xs]); i = int(np.argmin(v))
    a, b = xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]; g = (math.sqrt(5) - 1) / 2
    for _ in range(100):
        c, d = b - g * (b - a), a + g * (b - a)
        if fun(10 ** c) < fun(10 ** d): b = d
        else: a = c
    return 10 ** ((a + b) / 2)


def best(C, f=EPOCH):
    N = argmin_log(lambda N: L(N, C / (6 * N), f), 6, 14); D = C / (6 * N)
    return N, D, L(N, D, f)


def d_need(N, Lt):
    r = Lt - EPOCH['E'] - EPOCH['A'] / N ** EPOCH['al']
    return (EPOCH['B'] / r) ** (1 / EPOCH['be']) if r > 0 else math.inf


def cheapest(Lt, T):
    nmin = (EPOCH['A'] / (Lt - EPOCH['E'])) ** (1 / EPOCH['al'])
    N = argmin_log(lambda N: 6 * N * d_need(N, Lt) + 2 * N * T, math.log10(nmin) + 1e-4, 13.5)
    return N, d_need(N, Lt)


gpu_days = lambda C, u=0.4: C / (PEAK * u) / DAY
sig2 = lambda d: f'{d:,.1f}' if d < 10 else f'{int(round(d / 10 ** (math.floor(math.log10(d)) - 1)) * 10 ** (math.floor(math.log10(d)) - 1)):,}'


def fP(N):
    v, u = (N / 1e9, 'B') if N >= 1e9 else (N / 1e6, 'M')
    t = f'{v:.{max(0, 2 - math.floor(math.log10(v)))}f}'
    if '.' in t: t = t.rstrip('0').rstrip('.')
    return t + u


def fTok(D):
    v, u = (D / 1e12, 'T') if D >= 1e12 else (D / 1e9, 'B')
    t = f'{v:,.{max(0, 2 - math.floor(math.log10(v) + 1e-9))}f}'
    if '.' in t: t = t.rstrip('0').rstrip('.')
    return t + u


with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(250); pg.evaluate("()=>{location.hash='scaling-law'}"); pg.wait_for_timeout(450)
    J = lambda js, *a: pg.evaluate(js, *a)
    H = lambda js, *a: pg.evaluate("(a)=>{const H=window.__sl; return " + js + "}", a[0] if a else None)
    # constants as quoted
    fit = H("H.FIT"); fit0 = H("H.FIT0")
    check('the refit constants are the notebook’s', [fit[k] for k in ('E', 'A', 'B', 'al', 'be')] == [EPOCH[k] for k in ('E', 'A', 'B', 'al', 'be')], fit)
    check('the original constants are nanoGPT’s quote', [fit0[k] for k in ('E', 'A', 'B', 'al', 'be')] == [CHIN[k] for k in ('E', 'A', 'B', 'al', 'be')], fit0)
    a = EPOCH['be'] / (EPOCH['al'] + EPOCH['be'])
    check('a = β/(α+β) = 0.513 as the notebook prints', round(a, 3) == 0.513, a)
    # loss and optimum at many budgets
    for C in [1e20, 3e20, 1e21, 1e22, 1e23, 5.88e23, 1e24, 1e25, 1e26]:
        N, D, l = best(C); o = H("H.opt(a)", C)
        check(f'lowest point at {C:.0e}: N, D, loss', rel(o['N'], N, 1e-5) and rel(o['D'], D, 1e-5) and abs(o['L'] - l) < 1e-9, (o, N, D, l))
        check(f'loss formula at {C:.0e}', abs(H("H.lossOf(a[0],a[1])", [N, D]) - L(N, D)) < 1e-12)
    # ---- part 1: three plans
    plans = H("H.consts.PLANS")
    want = {'big': (280e9, 350e9), 'mid': (70e9, 1.4e12), 'small': (17.5e9, 5.6e12)}
    for k, (N, D) in want.items():
        check(f'plan {k}: {fP(N)} on {fTok(D)} costs 5.88e23 FLOPs', plans[k]['N'] == N and plans[k]['D'] == D and rel(6 * N * D, 5.88e23, 1e-12))
    los = {k: L(*v) for k, v in want.items()}
    check('part 1: losses 1.992 / 1.973 / 1.994; 70B lowest', [f'{los[k]:.3f}' for k in ('big', 'mid', 'small')] == ['1.992', '1.973', '1.994'] and min(los, key=los.get) == 'mid' and H("H.consts.PRED_RIGHT") == 'mid', los)
    tn = lambda N: EPOCH['A'] / N ** EPOCH['al']; td = lambda D: EPOCH['B'] / D ** EPOCH['be']
    check('part 3 terms: 280B 0.050 / 0.125; 70B 0.081 / 0.075; 17.5B 0.132 / 0.045', [f'{tn(N):.3f}/{td(D):.3f}' for N, D in want.values()] == ['0.050/0.125', '0.081/0.075', '0.132/0.045'])
    check('part 1: 5.9×10²³ as shown', f"{5.88e23 / 1e23:.1f}" == '5.9')
    # ---- part 2
    gd = gpu_days(1e24)
    check('part 2: 10²⁴ FLOPs = 29,000 GPU-days, 1,024 H100s for 29 days', sig2(gd) == '29,000' and sig2(gd / 1024) == '29' and gd / 1024 <= 60 and gd / 64 > 60, gd)
    check('part 2: the page’s GPU-days', abs(H("H.gpuDays(1e24,0.4)") - gd) < 1e-6)
    check('part 1 metric: 5.88e23 FLOPs = 17,000 GPU-days', sig2(gpu_days(5.88e23)) == '17,000')
    # ---- part 3
    check('part 3: ten times the parameters leave 45% of the model term; ten times the tokens 43% of the data term', round(10 ** -EPOCH['al'] * 100) == 45 and round(10 ** -EPOCH['be'] * 100) == 43)
    # ---- part 4
    rs = [best(C)[1] / best(C)[0] for C in [1e21, 1e22, 1e23, 1e24, 1e25]]
    check('part 4: every budget’s lowest point between 17 and 22 tokens per parameter', math.floor(min(rs)) == 17 and math.ceil(max(rs)) == 22, rs)
    N22, D22, _ = best(1e22)
    check('part 4: 10²² → 9.06B parameters, 184B tokens, about 20 per parameter', fP(N22) == '9.06B' and fTok(D22) == '184B' and round(D22 / N22) == 20, (N22, D22))
    check('part 4: the slider stop 39 (8.91B) counts as the lowest point', abs(math.log10(10 ** 9.95 / N22)) <= 0.0375 and abs(math.log10(10 ** 9.9 / N22)) > 0.0375)
    # ---- part 5
    L5 = L(8e9, 15e12)
    check('part 5: Llama 3 8B at 15T has fitted loss 2.022 and 1,875 tokens per parameter', f'{L5:.3f}' == '2.022' and abs(H("H.L5()") - L5) < 1e-12 and 15e12 / 8e9 == 1875)
    for T in [0, 1e12, 1e13, 1e14]:
        N, D = cheapest(L5, T); tb = H("H.totBest(a[0],a[1])", [L5, T])
        check(f'part 5: cheapest size serving {T:.0e} tokens', rel(tb['N'], N, 1e-4) and rel(tb['D'], D, 1e-4), (tb['N'], N))
    N0, D0 = cheapest(L5, 0)
    check('part 5: training only, Llama 3 8B takes about 5.5 times the compute', f'{7.2e23 / (6 * N0 * D0):.1f}' == '5.5', 7.2e23 / (6 * N0 * D0))
    N100, D100 = cheapest(L5, 1e14)
    check('part 5: at 100T served tokens the cheapest size is close to 8B (within 10%)', abs(N100 / 8e9 - 1) < 0.1, N100)
    check('part 5: Llama 3 8B serving 100T totals 2.3×10²⁴ FLOPs', f'{(7.2e23 + 2 * 8e9 * 1e14) / 1e24:.1f}' == '2.3')
    check('part 5: SmolLM 135M on 600B is over 4,000 tokens per parameter', 600e9 / 135e6 > 4000)
    # ---- part 6
    check('part 6: 10²³ FLOPs ≈ 2,900 GPU-days', sig2(gpu_days(1e23)) == '2,900')
    # ---- scenarios (rendered)
    pg.evaluate("()=>document.querySelector('#sl-app .modes button[data-mode=\"sandbox\"]:not([data-flow=free])').click()"); pg.wait_for_timeout(300)
    def ins(i):
        pg.evaluate(f"()=>document.querySelector('#sl-rail [data-preset=\"{i}\"]').click()"); pg.wait_for_timeout(250)
        return pg.evaluate("()=>document.getElementById('sl-insight').innerText")
    t = ins(0)
    check('scenario lab: 9.06B on 184B tokens, 290 GPU-days', '9.06B parameters trained on 184B tokens' in t and 'about 290 GPU-days' in t, t[:200])
    t = ins(1); N23, D23, L23 = best(1e23); Ns, Ds = cheapest(L23, 1e14)
    check('scenario serve: the cheapest size for 10²³’s best loss at 100T', f'can reach is {L23:.3f}' in t and f'a {fP(Ns)} model trained on {fTok(Ds)} tokens' in t and f'costs about {N23 / Ns:.1f} times as much' in t, (t[:300], fP(Ns), fTok(Ds), N23 / Ns))
    # S14: the small model's training is over 4 times the 29.5B model's (the 6ND of each, from the derivation)
    check('scenario serve: the small model trains on about 4.2 times the compute', f'takes about {6 * Ns * Ds / 1e23:.1f} times the compute' in t and f'{6 * Ns * Ds / 1e23:.1f}' == '4.2', (t[:400], 6 * Ns * Ds / 1e23))
    # deltas between scenarios: none, the preset sets the whole state (R6.6)
    t = ins(2); Nc = 1e22 / 6e11
    dshown = pg.evaluate("()=>[...document.querySelectorAll('#sl-metrics .d')].map(d=>d.textContent).join('')")
    check('scenarios: no ▲▼ after a scenario preset (R6.6)', dshown == '', dshown)
    d_shown = round(L(Nc, 1e11), 3) - round(L(N22, D22), 3)   # the difference of the values as shown (R6.6)
    check('scenario data: 16.7B on all 100B tokens, 0.007 above the unlimited best', f'{fP(Nc)} model' in t and f'loss is {L(Nc, 1e11):.3f}' in t and f'only {d_shown:.3f} above' in t and f'{d_shown:.3f}' == '0.007', t[:300])
    check('scenario data: the badge says the data is used up, the best size under the limit (S3)', pg.evaluate("()=>document.getElementById('sl-status').textContent") == 'All the data used: best size under the limit')
    check('scenario data: Data pipeline is a link', pg.evaluate("()=>!!document.querySelector('#sl-insight a[href=\"#data-pipeline\"]')"))
    t = ins(3); N21, D21, L21 = best(1e21); Nr = 4 * N21; Dr = 1e21 / 6 / Nr
    check('scenario redo: 10²¹, four times the best size, 0.061 above the lowest point', f'has about {fP(N21)} parameters' in t and f'the loss is {round(L(Nr, Dr), 3) - round(L21, 3):.3f} above the lowest point' in t, t[:300])
    # ---- the challenge
    Ns_, Ds_ = [1e9, 3e9, 8e9, 20e9, 70e9], [1e11, 3e11, 1e12, 3e12, 1e13]
    rows = [(i, j, L(N, D), 6 * N * D, 6 * N * D + 2 * N * 2e12) for i, N in enumerate(Ns_) for j, D in enumerate(Ds_)]
    ok = [r for r in rows if r[2] <= 2.10]
    cb = min(ok, key=lambda r: r[4]); tb_ = min(ok, key=lambda r: r[3])
    check('challenge: 25 plans, 13 meet loss ≤ 2.10', len(rows) == 25 and len(ok) == 13 and sum(e['ok'] for e in H("H.chAll()")) == 13, len(ok))
    check('challenge: the cheapest is 8B on 1T, 8.0×10²² FLOPs', (cb[0], cb[1]) == (2, 2) and f'{cb[4] / 1e22:.1f}' == '8.0' and (H("H.chBest()")['ni'], H("H.chBest()")['di']) == (2, 2))
    check('challenge: the training-only best is 20B on 300B, 1.2×10²³ in all', (tb_[0], tb_[1]) == (3, 1) and f'{tb_[4] / 1e23:.1f}' == '1.2' and (H("H.chTrainBest()")['ni'], H("H.chTrainBest()")['di']) == (3, 1))
    check('challenge: the initial setup (8B on 100B) fails at 2.187', f'{L(8e9, 1e11):.3f}' == '2.187' and L(8e9, 1e11) > 2.10)
    check('challenge: the cheapest trains 125 tokens per parameter; 20B costs 2.5 times per served token', 1e12 / 8e9 == 125 and 20 / 8 == 2.5)
    # the result's summary (the owner's review b8, R1): the one change that helps most, the main gap, the counterexample, each from the rows above
    def one(i, j):
        me = next(r for r in rows if r[:2] == (i, j)); c = [r for r in rows if (r[0] == i) != (r[1] == j)]
        c = [r for r in c if r[2] <= 2.10 and (me[2] > 2.10 or r[4] < me[4] * (1 - 1e-9))]
        return min(c, key=lambda r: r[4])[:2] if c else None
    hone = lambda i, j: (lambda o: (o['ni'], o['di']) if o else None)(H("H.chOne(a[0],a[1])", [i, j]))
    check('challenge summary: from the initial setup (8B on 100B), the best single change is 1T tokens, the cheapest plan', one(2, 0) == (2, 2) and hone(2, 0) == (2, 2))
    check('challenge summary: from 20B on 300B (passing), no single change saves compute', one(3, 1) is None and hone(3, 1) is None)
    check('challenge summary: from 70B on 10T, the best single change', one(4, 4) == hone(4, 4) and one(4, 4) is not None, (one(4, 4), hone(4, 4)))
    nmin = (EPOCH['A'] / (2.10 - EPOCH['E'])) ** (1 / EPOCH['al'])
    check('challenge summary: 1B can never reach 2.10 (smallest size about 1.95B); 8B on 100B falls short on data; 3B on 3T on the model term',
          1e9 < nmin < 3e9 and H("H.chGap(0,4)") == 'nmin' and H("H.chGap(2,0)") == 'data' and EPOCH['B'] / 1e11 ** EPOCH['be'] > EPOCH['A'] / 8e9 ** EPOCH['al']
          and H("H.chGap(1,3)") == 'model' and L(3e9, 3e12) > 2.10 and EPOCH['A'] / 3e9 ** EPOCH['al'] > EPOCH['B'] / 3e12 ** EPOCH['be'] and abs(H("H.nMin(2.1)") / nmin - 1) < 1e-9, nmin)
    check('challenge summary: the counterexample is the training-only best, unless it is the reader’s plan', (lambda o: (o['ni'], o['di']))(H("H.chCounter(2,0)")) == (3, 1) and H("H.chCounter(3,1)") is None)
    check('challenge: the 8B plan takes about 31% less in all than the 20B plan (8.0 against 11.6 × 10²² FLOPs as shown)', round((1 - cb[4] / tb_[4]) * 100) == 31 and round((1 - 8.0 / 11.6) * 100) == 31 and f'{tb_[4] / 1e22:.1f}' == '11.6')
    # ---- fidelity card
    r0 = [best(C, CHIN)[1] / best(C, CHIN)[0] for C in (1e21, 1e25)]
    check('fidelity: the original fit puts the best ratio at 50 to 122', [round(x) for x in r0] == [50, 122], r0)
    check('fidelity: the refit gives 17 to 22', round(rs[-1]) == 17 and round(rs[0]) == 22, rs)
    check('fidelity: 6ND at 40% predicts about 0.5M H100 hours for Llama 3 8B', f'{7.2e23 / (PEAK * 0.4) / 3600 / 1e6:.1f}' == '0.5')
    check('fidelity: DCLM 1x: 8.2B tokens for 412M parameters is about 20 per parameter', round(8.2e9 / 412e6) == 20)
    # S2: the range of the runs behind the fit, read here from Epoch's extracted points (245 rows)
    import csv
    rowsF = list(csv.DictReader(open(P / 'sources/b8src/scaling-law/epoch-research__analyzing-chinchilla__data_svg_extracted_data.csv')))
    fN = [float(r['Model Size']) for r in rowsF]; fC = [float(r['Training FLOP']) for r in rowsF]; fD = [c / 6 / m for c, m in zip(fC, fN)]
    fid = pg.evaluate("()=>{const b=document.getElementById('scaling-law-fid'); return b?b.textContent.replace(/\u00a0/g,' '):''}")   # the notes at the page end (2026-10-11)
    want_rng = f'go up to {fP(max(fN))} parameters, {max(fC) / 1e22:.1f}×10²² FLOPs, {fTok(max(fD))} training tokens and {round(max(d / m for d, m in zip(fD, fN)))} tokens per parameter'
    check('fidelity: the fitted runs go up to 16.2B, 1.3×10²², 318B, 341 per parameter; the page extrapolates', len(rowsF) == 245 and want_rng in fid and 'extrapolation' in fid and '1,875 tokens per parameter' in fid, (want_rng, fid[:200]))
    # S12: attention by nanoGPT's estimate_mfu, Llama 3 8B at 8,192 tokens: 12 × 32 × 4,096 × 8,192 ÷ (6 × 8,030,261,248)
    att = 12 * 32 * 4096 * 8192 / (6 * 8030261248)
    # the 27% sentence was in the Modeled list, deleted with it (owner's decision, 2026-10-11 19:28 UTC); only the arithmetic is kept
    check('attention adds about 27% to 6N for Llama 3 8B at 8K (arithmetic only)', round(att * 100) == 27, att)
    check('fidelity and Settings: the assumption is named MFU, 40%', 'The MFU is 40%' in fid and H("H.consts.BASE.u") == 0.4 and pg.evaluate("()=>{const l=document.querySelector('#sl-ctl-u'); return l?l.closest('.ctl').textContent:''}").startswith('MFU'))
    # ---- capability layer
    r4 = 4 ** a
    check('transfer: 4 × the budget → the best size × 2.04 (about 2)', f'{r4:.2f}' == '2.04' and H("H.exits()")['transferA'] == [1, 1])
    check('transfer: the loss floor E ≈ 1.82; 10²⁵’s best loss 1.911', f"{EPOCH['E']:.2f}" == '1.82' and f'{best(1e25)[2]:.3f}' == '1.911')
    check('explain: from 10²¹ to 10²⁵ the best tokens go from about 60B to about 5T', round(best(1e21)[1] / 1e10) * 10 == 60 and round(best(1e25)[1] / 1e12) == 5)
    check('explain: the training-only cheapest size for Llama 3 8B’s loss is 33.8B (feedback of question 2)', fP(N0) == '33.8B')
    check('part 6 and misconception: serving 100T moves the cheapest ratio to over a thousand per parameter', 1000 <= D100 / N100 < 2000, D100 / N100)
    check('part 3: the 280B plan’s data term is about 2.5 times its model term', f'{td(350e9) / tn(280e9):.1f}' == '2.5')
    # ---- drill
    for s in range(1, 9):
        d = H("H.drill(a)", s); N, D, _ = best(d['C'])
        check(f'drill {s}: size, tokens, GPU-days', rel(d['o']['N'], N, 1e-5) and rel(d['gd'], gpu_days(d['C'], d['u']), 1e-12), d)
    # ---- home tile
    T = J("()=>window.__tzSl&&window.__tzSl.BUD")
    check('tile: budgets 10²⁰ to 10²⁴', T and T[0] == 1e20 and T[-1] == 1e24 and len(T) == 9, T)
    trs = []
    for i, C in enumerate(T):
        m = J("i=>window.__tzSl.model(i)", i); N, D, l = best(C); trs.append(D / N)
        check(f'tile {C:.0e}: the ring at the lowest point', rel(m['N'], N, 1e-5) and abs(m['L'] - l) < 1e-9, (m, N))
    rd = J("i=>window.__tzSl.read(i)", 4)
    check('tile at 10²²: 9.06B parameters on 184B tokens', '9.06B parameters on about 184B tokens' in rd, rd)
    check('tile note: between 18 and 23 tokens per parameter', f'between {math.floor(min(trs))} and {math.ceil(max(trs))} tokens per parameter' in J("()=>window.__tzSl.notes()[1]") and (math.floor(min(trs)), math.ceil(max(trs))) == (18, 23), trs)
    # ---- Connect: run the code
    code = J("()=>document.getElementById('sl-code').textContent")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): exec(code, {})
    outl = buf.getvalue().splitlines()
    line22 = [l for l in outl if l.startswith('1e+22')]
    check('Connect code runs and its comment matches its own output at 1e22', line22 and ('# ' + line22[0]) in code, (line22, code[-120:]))
    check('Connect code: 1e+22 → 9.06B params, 184B tokens', line22 and '9.06B params, 184B tokens' in line22[0], line22)
    # the owner's clarification (2026-10-07 23:37 UTC): no hover only expands a number to its exact value, on the home tile either
    pg.evaluate("()=>{location.hash=''}"); pg.wait_for_timeout(600)
    tile = pg.evaluate("()=>{const t=document.querySelector('.tz-sl'); if(!t) return null; return {tips:[...t.querySelectorAll('[data-tipx],[data-tip]')].map(e=>e.getAttribute('data-tipx')||e.getAttribute('data-tip')), big:[...t.querySelectorAll('.num-big')].filter(e=>e.matches('[data-tipx],[tabindex],.def')).length}}")
    check('tile: drawn on the home page, no exact-value hover (no “精确值是” / “Exactly”, no .num-big with a hover)', tile is not None and tile['big'] == 0 and not any(('精确值是' in x or 'Exactly' in x) for x in tile['tips']), tile)
    # ---- G7 (the owner's message of 2026-10-08, from the model-parallel finding): walk every pick of the challenge, in both
    # languages, and check that a recommended single change makes the value the result shows strictly better at the precision shown
    def g7_walk(js):
        res = {}
        for lang in ('en', 'zh'):
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(150)
            res[lang] = pg.evaluate(js)
        pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(150)
        return res
    g7 = g7_walk("""()=>{const Z=window.__sl, C=Z.consts.CH; const R=/^The best single change is|^只改一项的话|^Of the single changes that meet the goal|^只改一项就能达标的改法里/; const out=[];
      C.Ns.forEach((_,i)=>C.Ds.forEach((_,j)=>{const t=Z.chNextText(i,j), s=Z.chShown(i,j), o=Z.chOne(i,j), rec=R.test(t); let ok=true, why='';
        if(rec){if(!o){ok=false; why='no engine change';} else {const r=Z.chShown(o.ni,o.di);
          if(s.ok){ok=r.ok&&r.tot<s.tot; why='compute '+s.totS+' -> '+r.totS;} else {ok=r.ok&&parseFloat(r.L)<parseFloat(s.L); why='loss '+s.L+' -> '+r.L;}}}
        if(/about 0%|约 0%|about less|约不到/.test(t)){ok=false; why+=' (a zero gain)';}
        out.push({pick:[i,j],rec,ok,why,t});})); return out;}""")
    for lang in ('en', 'zh'):
        rows = g7[lang]; bad = [r for r in rows if not r['ok']]
        check(f'G7 {lang}: all 25 plans walked; every recommended single change shows less compute (or a loss that meets the goal)', len(rows) == 25 and not bad, bad[:3])
    check('G7: the walk meets a passing plan with a single change and one without (3B on 10T: the sentence names both new settings)',
          any(r['rec'] and r['t'].startswith('只改一项的话') for r in g7['zh']) and any(r['t'].startswith('只改一项省不下来。最省的方案同时改了两项：模型换成 8B，训练 token 换成 1T') for r in g7['zh']), [r['t'] for r in g7['zh']][9:11])
    check('no page errors', not errs, errs[:3])
    b.close()
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:40] if fails else 'none')
sys.exit(1 if fails else 0)
