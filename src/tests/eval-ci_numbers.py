"""Golden numbers for the eval-ci lab (guide R9.2, CONTRACT §6).

Derived here independently of the JS engine:
- one score p on n questions: SE = sqrt(p(1-p)/n), error bar p ± 1.96 SE (normal approximation);
- two models on the same questions, from the joint mix of five kinds of question (both always right, both always wrong, only B always
  right, only A always right, and coin-flip questions that both models get right with chance 1/2 on every answer): the variance of one
  model's per-question average over k answers, and of B − A per question, by exact enumeration of the mix AND by Monte Carlo;
  each on its own: sqrt(SE_A² + SE_B²);
- questions needed: the smallest n whose error bar is at most h (found by search, not by the closed form).
Checks: every stop of the size slider, part 2's re-draws (their spread against SE within Monte Carlo tolerance, the count inside the
bar), part 3's first stop within ±1, the status badge's rule (B − A against 0, each on its own in guided parts 2, 3, 5), part 4 (both
ways, ends of the interval), part 5 (k = 1…16 and the floor), the four scenarios (all question by question), the challenge (16 plans,
the passing ones, the cheapest, the power of each passing plan and the error bar for 80% power), the fidelity card's coverage claim
(exact binomial sum), the capability layer's answers, the interview drill, the home tile (with its B − A row), and the Connect card's
code (run here with NumPy).
Run: python3 tests/eval-ci_numbers.py"""
import math, pathlib, subprocess, sys, io, contextlib
import numpy as np
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/eval-ci.html', '--only', 'eval-ci'], check=True, cwd=P, stdout=subprocess.DEVNULL)
OUT = P / 'out' / 'eval-ci'; OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'wrapped-num.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'eval-ci.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-num.html')
fails, n = [], 0
Z = 1.96


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ---------- the derivation
def mix(pA, pB, q, c):
    """shares of the five kinds of question; q = share with exactly one model right at one answer each"""
    D = pB - pA; d0 = q - c / 2
    onlyB, onlyA = (d0 + D) / 2, (d0 - D) / 2
    both = pA - onlyA - c / 2
    none = 1 - both - onlyA - onlyB - c
    return {'both': both, 'none': none, 'onlyB': onlyB, 'onlyA': onlyA, 'coin': c}


def per_question_var(pA, pB, q, c, k):
    """exact: enumerate the kinds; inside a coin-flip question each model's average of k answers has variance 1/4k"""
    w = mix(pA, pB, q, c)
    # (A's chance, B's chance) per kind
    kinds = {'both': (1, 1), 'none': (0, 0), 'onlyB': (0, 1), 'onlyA': (1, 0), 'coin': (.5, .5)}
    EA = sum(w[t] * kinds[t][0] for t in w); EB = sum(w[t] * kinds[t][1] for t in w)
    VA = sum(w[t] * (kinds[t][0] - EA) ** 2 for t in w) + sum(w[t] * kinds[t][0] * (1 - kinds[t][0]) / k for t in w)
    VB = sum(w[t] * (kinds[t][1] - EB) ** 2 for t in w) + sum(w[t] * kinds[t][1] * (1 - kinds[t][1]) / k for t in w)
    ED = EB - EA
    VD = sum(w[t] * (kinds[t][1] - kinds[t][0] - ED) ** 2 for t in w) + sum(w[t] * (kinds[t][0] * (1 - kinds[t][0]) + kinds[t][1] * (1 - kinds[t][1])) / k for t in w)
    return EA, EB, VA, VB, VD


def bars(n, pA, pB, q, c, k):
    EA, EB, VA, VB, VD = per_question_var(pA, pB, q, c, k)
    seA, seB = math.sqrt(VA / n), math.sqrt(VB / n)
    return {'hA': Z * seA, 'hB': Z * seB, 'hS': Z * math.hypot(seA, seB), 'hP': Z * math.sqrt(VD / n), 'D': EB - EA}


def need(p, h):
    m = 1
    while Z * math.sqrt(p * (1 - p) / m) > h: m += 1
    return m


def r1(x): return f'{x * 100:.1f}'


# Monte Carlo of the mix (2 million questions), seeded: the per-question variances must match the enumeration
rng = np.random.default_rng(20261003)
for (pA, pB, q, c) in [(0.8, 0.82, 0.10, 0.0), (0.8, 0.82, 0.12, 0.2), (0.8, 0.82, 0.16, 0.2)]:
    w = mix(pA, pB, q, c); names = list(w); N = 2_000_000
    kind = rng.choice(len(names), N, p=[w[t] for t in names])
    chA = np.array([{'both': 1, 'none': 0, 'onlyB': 0, 'onlyA': 1, 'coin': .5}[t] for t in names])[kind]
    chB = np.array([{'both': 1, 'none': 0, 'onlyB': 1, 'onlyA': 0, 'coin': .5}[t] for t in names])[kind]
    for k in (1, 4, 16):
        a = rng.binomial(k, chA) / k; b = rng.binomial(k, chB) / k
        EA, EB, VA, VB, VD = per_question_var(pA, pB, q, c, k)
        check(f'Monte Carlo: A’s per-question variance, q {q}, k {k}', abs(a.var() - VA) < 0.004 * max(VA, 0.05) * 10, (a.var(), VA))
        check(f'Monte Carlo: B − A per-question variance, q {q}, k {k}', abs((b - a).var() - VD) < 0.004 * max(VD, 0.05) * 10, ((b - a).var(), VD))
        check(f'Monte Carlo: mean score of A, q {q}, k {k}', abs(a.mean() - pA) < 0.002, a.mean())
    # the closed forms the page states equal the enumeration
    for k in (1, 2, 4, 8, 16):
        EA, EB, VA, VB, VD = per_question_var(pA, pB, q, c, k)
        check(f'closed form A: p(1-p) - c/4 (1 - 1/k), k {k}', close(VA, pA * (1 - pA) - c / 4 * (1 - 1 / k)))
        check(f'closed form B − A: q - Δ² - c/2 (1 - 1/k), k {k}', close(VD, q - (pB - pA) ** 2 - c / 2 * (1 - 1 / k)))

with sync_playwright() as pw:
    b = pw.chromium.launch(); pg = b.new_page()
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.evaluate("()=>{location.hash='eval-ci'}"); pg.wait_for_timeout(400)
    J = lambda s, *a: pg.evaluate(s, *a)
    C = J("()=>window.__evc.consts")
    cmpj = lambda o: J("o=>window.__evc.cmp(o)", o)
    BASE = C['BASE']

    # ---- every stop of the size slider (A at 80%, B at 82%, one answer)
    for nq in C['NS']:
        r = cmpj({'n': nq}); g = bars(nq, .8, .82, .12, .2, 1)
        check(f'size stop {nq}: A’s error bar', close(r['hA'], g['hA']), (r['hA'], g['hA']))
        check(f'size stop {nq}: B’s error bar', close(r['hB'], g['hB']), (r['hB'], g['hB']))
    check('HumanEval 164: ±6.1 (A), ±5.9 (B)', r1(bars(164, .8, .82, .12, .2, 1)['hA']) == '6.1' and r1(bars(164, .8, .82, .12, .2, 1)['hB']) == '5.9')
    check('GSM8K 1,319: ±2.2', r1(bars(1319, .8, .82, .12, .2, 1)['hA']) == '2.2')
    check('about 14,000: ±0.7', r1(bars(14000, .8, .82, .12, .2, 1)['hA']) == '0.7')
    check('the slider has the three benchmark ticks', all(x in C['NS'] for x in (164, 1319, 14000)))
    check('the slider has the challenge sizes', all(x in C['NS'] for x in C['CH']['Ns']), C['CH']['Ns'])
    # part 3: ±1 point needs 6,147 questions (shown as “about 6,100”); the first stop at or under ±1.0 is 6,200
    nd = need(.8, .01)
    check('±1 point at 80% needs 6,147 questions', nd == 6147, nd)
    check('the page’s nFor rounds to 6,147', math.ceil(J("()=>window.__evc.nFor(0.8,0.01)") - 1e-9) == 6147)
    check('shown as about 6,100', round(nd / 100) * 100 == 6100)
    first = [x for x in C['NS'] if bars(x, .8, .82, .12, .2, 1)['hA'] <= 0.01 + 1e-12][0]
    check('the first slider stop within ±1.0 is 6,200 (part 3’s gate and its Show me)', first == 6200, first)
    check('5,000 is still above ±1.0 (±1.1)', r1(bars(5000, .8, .82, .12, .2, 1)['hA']) == '1.1')
    # four times the questions halve the error bar
    check('4× questions halve the bar', close(bars(656, .8, .82, .12, .2, 1)['hA'] * 2, bars(164, .8, .82, .12, .2, 1)['hA']))
    # the status badge (B-1): B − A each on its own against 0, not the overlap of A's and B's bars
    clear = [x for x in C['NS'] if .02 - bars(x, .8, .82, .12, .2, 1)['hS'] > 0]
    check('badge: B − A each on its own first clears 0 at 3,000 questions (±2.0, shown as 2.00 ± 1.98)', clear and clear[0] == 3000 and all(x in clear for x in C['NS'] if x >= 3000), clear[:3])
    ovl = lambda x: .8 + bars(x, .8, .82, .12, .2, 1)['hA'] > .82 - bars(x, .8, .82, .12, .2, 1)['hB']
    check('badge: at 3,000 to 5,000 the bars still overlap while B − A is clear of 0 (the tooltip says so)', all(ovl(x) for x in (3000, 4000, 5000)) and not ovl(6200))
    for x in C['NS']:
        r = cmpj({'n': x}); check(f'badge at {x}: the page’s each-on-its-own verdict', r['real'] == (.02 - bars(x, .8, .82, .12, .2, 1)['hS'] > 0), r)
    iv = J("([D,h])=>window.__evc.ivl(D,h)", [.02, bars(3000, .8, .82, .12, .2, 1)['hS']])
    check('badge at 3,000: the interval is shown with two decimals and does not touch 0 (R6.10)', (iv['Ds'], iv['hs'], iv['lo']) == ('2.00', '1.98', '0.02'), iv)
    check('part 2: SE is about 3.1 points, the error bar ±6.1', f'{math.sqrt(.16 / 164) * 100:.1f}' == '3.1')

    # ---- part 2: 200 seeded re-runs of 164 questions at 80%
    rd = J("()=>window.__evc.redraws()"); RD = C['RD']
    check('part 2: 200 re-runs of 164 questions', len(rd) == RD['K'] == 200 and RD['n'] == 164 and all(0 <= x <= 164 for x in rd))
    sc = np.array(rd) / 164; se = math.sqrt(.8 * .2 / 164)
    check('part 2: the re-runs centre on 80% (within 3 SE/√200)', abs(sc.mean() - .8) < 3 * se / math.sqrt(200), sc.mean())
    check('part 2: their spread matches SE within Monte Carlo tolerance (±15%)', abs(sc.std(ddof=1) / se - 1) < 0.15, (sc.std(ddof=1), se))
    hR = Z * se
    inside = int(sum(abs(x / 164 - .8) <= hR + 1e-12 for x in rd))
    check('part 2: the count inside the bar matches the page', inside == J("()=>window.__evc.inside(200)"), (inside, J("()=>window.__evc.inside(200)")))
    check('part 2: about 95% inside (between 180 and 200)', 180 <= inside <= 200, inside)
    check('part 2: the bar runs from 73.9% to 86.1%', f'{(.8 - hR) * 100:.1f}' == '73.9' and f'{(.8 + hR) * 100:.1f}' == '86.1')
    print('part 2 inside:', inside, 'of 200; sd', round(sc.std(ddof=1) * 100, 2), 'points, SE', round(se * 100, 2))

    check('the base setup’s mix exists (q 12%, c 20%)', J("o=>window.__evc.feasible(o)", BASE) and BASE['q'] == .12 and BASE['c'] == .2, BASE)
    check('q 10% with 20% coin flips is impossible (coin flips alone give 10% + the 2-point gap)', not J("o=>window.__evc.feasible(o)", {'q': .10, 'c': .2}))
    # ---- part 4: 1,319 problems, 76 / 4 / 6 / 14, q = 10%
    P4 = C['P4']
    w = mix(.8, .82, .10, 0)
    check('part 4: the answer sheet is 76% / 4% / 6% / 14%', all(close(w[k], v) for k, v in (('both', .76), ('onlyA', .04), ('onlyB', .06), ('none', .14))), w)
    g = bars(1319, .8, .82, .10, 0, 1)
    check('part 4: its mix exists', J("o=>window.__evc.feasible(o)", P4))
    rs, rp = cmpj({**P4, 'paired': 0}), cmpj({**P4, 'paired': 1})
    check('part 4: each on its own ±3.0 (2.993)', close(rs['h'], g['hS']) and f"{g['hS'] * 100:.1f}" == '3.0', (rs['h'], g['hS']))
    check('part 4: question by question ±1.7 (1.703)', close(rp['h'], g['hP']) and f"{g['hP'] * 100:.1f}" == '1.7', (rp['h'], g['hP']))
    iv = J("([D,h])=>window.__evc.ivl(D,h)", [rp['D'], rp['h']])
    check('part 4: the interval runs from 0.3 to 3.7 (from the shown 2.0 and 1.7)', (iv['Ds'], iv['hs'], iv['lo'], iv['hi']) == ('2.0', '1.7', '0.3', '3.7'), iv)
    iv = J("([D,h])=>window.__evc.ivl(D,h)", [rs['D'], rs['h']])
    check('part 4: each on its own runs from −1.0 to 5.0', (iv['lo'], iv['hi']) == ('−1.0', '5.0'), iv)
    check('part 4: the two error bars overlap, yet question by question is clear of 0', rs['overlap'] and rp['real'] and not rs['real'])
    check('part 4: about 132 of 1,319 problems disagree', round(.10 * 1319) == 132)

    # ---- part 5: k answers at 164 problems, c = 20%
    exp5 = ['6.1', '5.6', '5.4', '5.2', '5.1']
    for k, e in zip(C['KS'], exp5):
        g = bars(164, .8, .82, .12, .2, k); r = cmpj({'k': k})
        check(f'part 5: {k} answers: ±{e}', close(r['hA'], g['hA']) and r1(g['hA']) == e, (r['hA'], g['hA']))
    fl = Z * math.sqrt((.8 * .2 - .2 / 4) / 164)
    check('part 5: the floor ±5.1 (k → ∞), page agrees', close(J("()=>window.__evc.floorH(0.8,0.2,164)"), fl) and r1(fl) == '5.1', fl)
    check('part 5: the floor is the limit of k answers', abs(bars(164, .8, .82, .12, .2, 10 ** 7)['hA'] - fl) < 1e-6)
    check('part 5: the mix is 70% / 20% / 10%', close(.8 - .2 / 2, .7) and close(1 - .8 - .2 / 2, .1))

    # ---- the four scenarios
    PR = {p['id']: p for p in C['PRESETS']}
    for pid, p in PR.items():
        g = bars(p['n'], p['pA'], p['pB'], p['q'], p['c'], p['k']); r = cmpj(p)
        check(f'scenario {pid}: the page’s bars', close(r['hA'], g['hA']) and close(r['hS'], g['hS']) and close(r['hP'], g['hP']), (r, g))
        check(f'scenario {pid}: the mix exists', J("o=>window.__evc.feasible(o)", p))
    check('scenarios: all four start question by question (both models answer the same questions)', all(p['paired'] == 1 for p in PR.values()), PR)
    g = bars(164, .8, .82, .12, .2, 1)
    check('scenario code: question by question 2.0 ± 5.3 crosses 0 (each on its own ±8.5)', f"{g['hP'] * 100:.1f}" == '5.3' and g['hP'] > .02 and f"{g['hS'] * 100:.1f}" == '8.5', g)
    check('scenario code: the page says it crosses 0', not cmpj(PR['code'])['real'])
    g = bars(14000, .8, .808, .1, 0, 1)
    check('scenario exam: question by question 0.8 ± 0.5 clear of 0; each on its own ± 0.9 crosses 0', f"{g['hP'] * 100:.1f}" == '0.5' and g['hP'] < .008 and f"{g['hS'] * 100:.1f}" == '0.9' and g['hS'] > .008, g)
    g = bars(200, .8, .82, .12, .2, 1); check('scenario team: ±5.5 at 200; B − A question by question ±4.8 crosses 0', r1(g['hA']) == '5.5' and r1(g['hP']) == '4.8' and g['hP'] > .02 and not cmpj(PR['team'])['real'], g)
    g1, g4 = bars(1319, .8, .82, .12, .2, 1), bars(1319, .8, .82, .12, .2, 4); flt = Z * math.sqrt((.16 - .05) / 1319)
    check('scenario temperature: ±2.2 with 1 answer, ±1.9 with 4, floor ±1.8', (r1(g1['hA']), r1(g4['hA']), r1(flt)) == ('2.2', '1.9', '1.8'), (g1['hA'], g4['hA'], flt))
    check('scenario temperature: question by question with 4 answers, 2.0 ± 1.1, clear of 0', r1(g4['hP']) == '1.1' and g4['hP'] < .02 and cmpj(PR['temp'])['real'], g4)
    iv = J("([D,h])=>window.__evc.ivl(D,h)", [.02, g4['hP']])
    check('scenario temperature: 2.0 ± 1.1 runs from 0.9 to 3.1', (iv['Ds'], iv['hs'], iv['lo'], iv['hi']) == ('2.0', '1.1', '0.9', '3.1'), iv)
    check('scenario exam: the page says question by question is clear of 0', cmpj(PR['exam'])['real'])

    # ---- the challenge: 200 / 500 / 1,000 / 4,000 × each on its own / question by question × 1 / 4 answers
    CH = C['CH']; allp = J("()=>window.__evc.chAll()")
    mine = []
    for nq in CH['Ns']:
        for m in (0, 1):
            for k in CH['Ks']:
                g = bars(nq, CH['pA'], CH['pB'], CH['q'], CH['c'], k); h = g['hP'] if m else g['hS']
                mine.append((nq, m, k, h, h < CH['goal'], nq * k * 2))
    check('challenge: 16 plans', len(allp) == 16 == len(mine))
    for (nq, m, k, h, ok, cost), e in zip(mine, allp):
        check(f'challenge {nq}/{m}/{k}: error bar and verdict', (e['n'], e['paired'], e['k']) == (nq, m, k) and close(e['h'], h) and e['ok'] == ok and e['cost'] == cost, (e, h))
        check(f'challenge {nq}/{m}/{k}: no plan sits on the goal when shown to one decimal (R6.10)', (h < CH['goal']) == (float(f'{h * 100:.1f}') < 2.0), h)
    passing = [x for x in mine if x[4]]
    check('challenge: 6 plans pass', len(passing) == 6, [x[:3] for x in passing])
    best = min(passing, key=lambda x: (x[5], x[3]))
    eb = J("()=>window.__evc.chBest()")
    check('challenge: the cheapest passing plan is 500 questions, question by question, 4 answers (4,000 graded)', best[:3] == (500, 1, 4) and best[5] == 4000 and (eb['n'], eb['paired'], eb['k'], eb['cost']) == (500, 1, 4, 4000), (best, eb))
    check('challenge: the cheapest plan is unique', sum(1 for x in passing if x[5] == best[5]) == 1)
    st = [x for x in mine if x[:3] == (200, 0, 1)][0]
    check('challenge: the initial setup misses (±7.7)', not st[4] and f'{st[3] * 100:.1f}' == '7.7', st)
    check('challenge: part 4’s own setup (1,319) is not an option', 1319 not in CH['Ns'])
    check('challenge: 1,000 question by question with 1 answer misses (±2.1)', [x for x in mine if x[:3] == (1000, 1, 1)][0][3] > .02)
    check('challenge: 500 each on its own with 4 answers gives ±4.2; 500 question by question with 1 answer ±3.0',
          f"{[x for x in mine if x[:3] == (500, 0, 4)][0][3] * 100:.1f}" == '4.2' and f"{[x for x in mine if x[:3] == (500, 1, 1)][0][3] * 100:.1f}" == '3.0')
    check('challenge: the assumed mix exists', J("o=>window.__evc.feasible(o)", {'pA': .8, 'pB': .82, 'q': .12, 'c': .2}))
    # power (S-1): the chance that B − A's error bar clears 0 when B is truly 2 points better, Φ(D/SE − z) + Φ(−D/SE − z)
    Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    pct = lambda pw: 'over 99%' if pw > .995 else f'about {round(pw * 100)}%'
    want_pct = {(500, 1, 4): 'about 56%', (1000, 1, 4): 'about 85%', (4000, 0, 1): 'about 63%', (4000, 0, 4): 'about 75%', (4000, 1, 1): 'about 96%', (4000, 1, 4): 'over 99%'}
    for (nq, m, k, h, ok, cost) in passing:
        se = h / Z; pw = Phi(.02 / se - Z) + Phi(-.02 / se - Z)
        pg_pw = J("h=>window.__evc.power(h,0.02)", h)
        check(f'challenge power {nq}/{m}/{k}: page {pg_pw:.4f} vs {pw:.4f}, shown “{pct(pw)}”', abs(pg_pw - pw) < 2e-6 and pct(pw) == want_pct[(nq, m, k)], (pg_pw, pw))
    check('challenge: the cheapest plan (±1.9) has only about 56% power', round((Phi(.02 / (best[3] / Z) - Z)) * 100) == 56)
    h80 = .02 * Z / (Z + 0.8416212)
    check('challenge: 80% power takes an error bar of about ±1.4', f'{h80 * 100:.1f}' == '1.4' and close(J("()=>window.__evc.H80(0.02)"), h80, 1e-9) and abs(Phi(.02 / (h80 / Z) - Z) - .8) < 1e-6)
    # the fidelity card's claim (S-3): exact coverage of the 95% bar, scores 50–90%, the slider's stops from 164 questions
    def cover(p, nq):
        lp, lq, tot = math.log(p), math.log(1 - p), 0.0
        for x in range(nq + 1):
            ph = x / nq; hh = Z * math.sqrt(ph * (1 - ph) / nq)
            if ph - hh <= p <= ph + hh: tot += math.exp(math.lgamma(nq + 1) - math.lgamma(x + 1) - math.lgamma(nq - x + 1) + x * lp + (nq - x) * lq)
        return tot
    worst = min(cover(i / 1000, nq) for nq in C['NS'] if 164 <= nq <= 3000 for i in range(500, 901, 5))
    big = min(cover(i / 1000, nq) for nq in C['NS'] if nq > 3000 for i in (500, 700, 800, 850, 900))
    cm = J("()=>window.__evc.covMin()")
    check(f'fidelity: lowest coverage on the page grid for 50–90% from 164 questions is {worst:.4f}, shown floored as “at least about 92%”', abs(cm - worst) < 1e-6 and math.floor(worst * 100) == 92 and big > worst, (cm, worst, big))
    fine = min(min(cover(i / 1000, nq) for i in range(860, 901)) for nq in (164, 165, 173, 200))
    check(f'fidelity: finer scores (0.1-point steps) and other question counts dip to {fine:.4f}, still at least 92% (the floor, not 93%)', .92 <= fine < .925, fine)
    # the floored figure was in the Fidelity card’s “Modeled” list, deleted with it (owner, 2026-10-11 19:28 UTC); the notes at the page end must not say 93%
    fid = J("()=>{const p=document.getElementById('eval-ci-fid'); return p?p.textContent:''}")
    check('the notes at the page end do not claim 93%', '93%' not in fid, fid[:120])
    check('fidelity: near 100% on few questions the bar is too narrow (95% at 100 questions covers under 90%)', cover(.95, 100) < .9 and abs(J("()=>window.__evc.coverage(0.95,100)") - cover(.95, 100)) < 1e-6, cover(.95, 100))

    # ---- capability layer
    ex = J("()=>window.__evc.exits()")
    g5 = bars(500, .8, .81, .12, .2, 1)
    check('new case 1: 500 questions, ±3.5, 1 point inside → answer “No” (index 1)', ex['transferA'][0] == 1 and r1(g5['hA']) == '3.5', (ex, g5['hA']))
    check('new case 1: even question by question, 500 questions cannot show 1 point (q 10%)', g5['hP'] > .01, g5['hP'])
    check('new case 1: question by question at 500 the error bar of B − A is still several points (q 10–12%: ±2.8 to ±3.0)',
          all(2.5 < bars(500, .8, .81, q, 0, 1)['hP'] * 100 < 3.5 for q in (.10, .12)), [bars(500, .8, .81, q, 0, 1)['hP'] for q in (.10, .12)])
    check('new case 2: halving the bar takes 4× the questions → index 1', ex['transferA'][1] == 1 and close(bars(4 * 300, .8, .8, .1, 0, 1)['hA'] * 2, bars(300, .8, .8, .1, 0, 1)['hA']))

    # ---- the interview drill, several seeds
    for s in range(1, 9):
        d = J("s=>window.__evc.drill(s)", s)
        g = bars(d['n'], d['pA'], d['pB'], d['q'], 0, 1)
        check(f'drill {s}: both error bars', close(d['s']['h'], g['hS']) and close(d['p']['h'], g['hP']), (d, g))
        check(f'drill {s}: its mix exists', J("o=>window.__evc.feasible(o)", {'pA': d['pA'], 'pB': d['pB'], 'q': d['q'], 'c': 0}))

    # ---- the home tile: 70% and 72%
    T = J("()=>window.__tzEvc&&window.__tzEvc.NS")
    check('tile: 7 stops from 100 to 10,000', T == [100, 200, 500, 1000, 2000, 5000, 10000], T)
    for i, nq in enumerate(T or []):
        m = J("i=>window.__tzEvc.model(i)", i)
        hA, hB = Z * math.sqrt(.21 / nq), Z * math.sqrt(.72 * .28 / nq); hS = math.hypot(hA, hB)
        ov = .70 + hA > .72 - hB; real = .02 - hS > 0
        check(f'tile {nq}: bars, overlap and B − A', close(m['hA'], hA) and close(m['hB'], hB) and close(m['hS'], hS) and m['overlap'] == ov and m['real'] == real, (m, hA, hB, hS))
        rdx = J("i=>window.__tzEvc.read(i)", i)
        want = 'could be noise' if not real else ('stays clear of 0' if ov else 'no longer overlap')
        check(f'tile {nq}: the sentence says “{want}”', want in rdx, rdx)
        check(f'tile {nq}: at most 25 words', len(rdx.replace('<b>', '').replace('</b>', '').split()) <= 25, rdx)
    check('tile: overlap at 5,000 but B − A clear of 0; no overlap at 10,000', J("()=>{const a=window.__tzEvc.model(5),b=window.__tzEvc.model(6); return a.overlap&&a.real&&!b.overlap}"))
    DX = J("()=>window.__tzEvc.DX")
    lo5, lo2 = 2 - Z * math.sqrt((.21 + .2016) / 5000) * 100, 2 - Z * math.sqrt((.21 + .2016) / 2000) * 100
    check('tile B − A row: on its axis (−2 to 6), 2,000 crosses 0 (from −0.8) and 5,000 starts right of 0 (0.2)', DX == [-2, 6] and f'{lo2:.1f}' == '-0.8' and f'{lo5:.1f}' == '0.2' and lo5 > 0, (DX, lo2, lo5))
    check('tile overlap at 5,000 is 0.5 points (drawn at least 8 px wide)', f"{(.70 + Z * math.sqrt(.21 / 5000) - (.72 - Z * math.sqrt(.2016 / 5000))) * 100:.1f}" == '0.5')
    nmin = next(m for m in range(1, 10000) if 1.96 * math.sqrt((.21 + .2016) / m) < .02)
    check('tile note: “from about 4,000 questions on” B − A stays clear of 0 (3,954)', nmin == 3954 and round(nmin, -3) == 4000, nmin)

    # ---- the Connect card's code, run here
    code = J("()=>document.getElementById('evc-code').textContent")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): exec(code, {})
    lines = buf.getvalue().split('\n')
    check('code: prints 6.1 and 6147', lines[0].strip() == '6.1' and lines[1].strip() == '6147', lines[:2])
    tup = lines[2]
    check('code: the simulated sheet gives both ways (paired narrower)', '(' in tup, tup)
    check('no page errors', not errs, errs)
    b.close()

print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
