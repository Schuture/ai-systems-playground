"""Golden numbers for the pass-k lab (guide R9.2, CONTRACT §6).

Everything is derived here independently of the JS engine:
- the tries: mulberry32 (ported, since it defines the seeded tries), seeded with (seed × 2654435761 + n × 97 + ring × 7919 + attempt
  × 104729) mod 2^32, one number per task and try, a pass when it is below 0.6; each ring of 10 tasks is redrawn until its tries pass
  exactly 60% of the time, rings 2 and 3 jointly until 12 of their 20 first tries pass (the rule the Fidelity popover states);
- which tasks always or never pass at each level (the first 6L always, the last 4L never), and part 1's shuffled order;
- pass@k per task as 1 − comb(n − c, k) / comb(n, k) and pass^k per task as comb(c, k) / comb(n, k) (math.comb), averaged over tasks;
  the true values from each task's chance (1, 0.6 or 0); the "every task equally hard" lines 1 − (1 − p̂)^k and p̂^k;
- the two estimators are unbiased: their exact expectation over a binomial number of passes, and a Monte Carlo run, give 1 − (1 − p)^k
  and p^k; human-eval's product form equals the comb form;
- the challenge in closed form, its fewest-calls answer by brute force; the home tile's three values; the interview drill.
Checks: the lab's engine (window.__pak) against this derivation at every level, n and three seeds; τ-bench's numbers against the local
copy of its README; the Connect card's code, run here, against the output printed under it; and the numbers the page shows in the
mission, parts 2 to 6, the scenarios, the challenge and the tile.
Run: python3 tests/pass-k_numbers.py"""
import io, contextlib, math, pathlib, random, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
SRC = P / 'sources' / 'b7src'   # local copies of the cited sources (tau-bench README, human-eval code and README)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/pass-k.html', '--only', 'pass-k'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'pass-k').mkdir(parents=True, exist_ok=True)
(P / 'out' / 'pass-k' / 'num.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'pass-k.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'pass-k' / 'num.html')
fails, n = [], 0


# 2026-10-09 (H1, H2): Show the math is typeset in chains (left, relation, right per row); read each chain row by row, spaces dropped
CHAINS = """()=>[...document.querySelectorAll('#pak-mathbody .pak-eqs')].map(q=>{const k=[...q.children], rows=[]; for(let i=0;i+2<k.length;i+=3) rows.push([k[i],k[i+1],k[i+2]].map(c=>c.textContent.replace(/\\s+/g,''))); return rows;})"""
# the calls chain of agent A: its first right-hand side split at “+”, a power written base^exponent; the result of its last row
CALLS = """()=>{const q=[...document.querySelectorAll('#pak-mathbody .pak-eqs')].find(q=>['calls','调用次数'].includes(q.children[0].textContent.replace(/\\s+/g,''))); if(!q) return null;
  const k=[...q.children], m=k[2].querySelector('math'), top=m.querySelector(':scope>mrow')||m, terms=[]; let cur=[];
  for(const c of top.children){const tg=c.tagName.toLowerCase(); if(tg==='mo'&&c.textContent==='+'){terms.push(cur.join('')); cur=[]; continue;}
    cur.push(tg==='msup'?c.children[0].textContent+'^'+c.children[1].textContent:c.textContent);}
  terms.push(cur.join('')); return {terms, res:k[k.length-1].textContent.replace(/\\s+/g,''), rel:k[k.length-2].textContent.trim()};}"""


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


close = lambda a, b, tol=1e-9: abs(a - b) <= tol
pct = lambda x: f'{x * 100:,.1f}%'
pctc = lambda x: '>99.9%' if 0.9995 <= x < 1 else pct(x)   # the challenge shows near-100% shares as >99.9%
f2 = lambda x: f'{x:.2f}'

# ---------- the derivation
NT, Q, NS, LVMAX = 40, 0.6, [5, 10, 20], 4
RING = [[0, 1, 2, 3, 4, 5, 36, 37, 38, 39], [6, 7, 8, 9, 10, 11, 32, 33, 34, 35], [12, 13, 14, 15, 16, 17, 28, 29, 30, 31], list(range(18, 28))]
TAU = [0.604, 0.491, 0.430, 0.383]
M32 = 0xFFFFFFFF


def mulberry(a):
    st = [a & M32]
    imul = lambda x, y: (x * y) & M32

    def r():
        st[0] = (st[0] + 0x6D2B79F5) & M32; a = st[0]
        t = imul(a ^ (a >> 15), 1 | a)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296
    return r


def tries(seed, nn):
    want = round(Q * nn * 10)

    def gen(r, att):
        rnd = mulberry((seed * 2654435761 + nn * 97 + r * 7919 + att * 104729) % 2 ** 32)
        return [[rnd() < Q for _ in range(nn)] for _ in RING[r]]
    tot = lambda g: sum(sum(row) for row in g)
    first = lambda g: sum(1 for row in g if row[0])
    G = [None] * 4
    for a3 in range(4000):
        g3 = gen(3, a3)
        if tot(g3) != want: continue
        need = 12 - first(g3)
        if need < 0 or need > 10: continue
        for a2 in range(4000):
            g2 = gen(2, 100000 + a3 * 4000 + a2)
            if tot(g2) == want and first(g2) == need: G[3], G[2] = g3, g2; break
        if G[2]: break
    for r in (1, 0):
        for att in range(20000):
            g = gen(r, att)
            if tot(g) == want: G[r] = g; break
    B = [None] * NT
    for r, ring in enumerate(RING):
        for m, slot in enumerate(ring): B[slot] = G[r][m]
    return B


kind = lambda i, lv: 'a' if i < 6 * lv else ('z' if i >= NT - 4 * lv else 's')


def p_at(nn, c, k):
    return 1.0 if nn - c < k else 1 - math.comb(nn - c, k) / math.comb(nn, k)


def p_hat(nn, c, k):
    return math.comb(c, k) / math.comb(nn, k)


def p_at_prod(nn, c, k):   # human-eval's estimate_pass_at_k, without NumPy
    if nn - c < k: return 1.0
    p = 1.0
    for j in range(nn - c + 1, nn + 1): p *= 1 - k / j
    return 1 - p


mean = lambda a: sum(a) / len(a)
TRI = {}


def derive(lv, nn, seed):
    if (seed, nn) not in TRI: TRI[(seed, nn)] = tries(seed, nn)
    B = TRI[(seed, nn)]
    ts = []
    for i in range(NT):
        kd = kind(i, lv)
        row = [True] * nn if kd == 'a' else ([False] * nn if kd == 'z' else B[i])
        ts.append({'kind': kd, 't': row, 'c': sum(row), 'p': 1.0 if kd == 'a' else (0.0 if kd == 'z' else Q)})
    p1 = mean([x['c'] / nn for x in ts])
    out = {'c': [x['c'] for x in ts], 'kind': [x['kind'] for x in ts], 'p1': p1, 'zero': sum(1 for x in ts if x['c'] == 0), 'first1': sum(1 for x in ts if x['t'][0]),
           'at': {}, 'hat': {}, 'firstK': {}, 'tAt': {}, 'tHat': {}, 'nAt': {}, 'nHat': {}}
    for k in range(1, nn + 1):
        out['at'][k] = mean([p_at(nn, x['c'], k) for x in ts]); out['hat'][k] = mean([p_hat(nn, x['c'], k) for x in ts])
        out['firstK'][k] = mean([1.0 if any(x['t'][:k]) else 0.0 for x in ts])
        out['tAt'][k] = mean([1 - (1 - x['p']) ** k for x in ts]); out['tHat'][k] = mean([x['p'] ** k for x in ts])
        out['nAt'][k] = 1 - (1 - p1) ** k; out['nHat'][k] = p1 ** k
    return out


def ch_eval(ag, k, p=0.7):
    done = 1 - (1 - p) ** k if ag == 'A' else p
    calls = sum((1 - p) ** j for j in range(k)) if ag == 'A' else p * 1 + (1 - p) * k
    return done, calls


# ---------- independent of the page: the estimators are unbiased
for nn in (5, 10, 20):
    for p in (0.1, 0.35, 0.6, 0.9):
        for k in range(1, nn + 1):
            pmf = [math.comb(nn, c) * p ** c * (1 - p) ** (nn - c) for c in range(nn + 1)]
            ea = sum(pmf[c] * p_at(nn, c, k) for c in range(nn + 1)); eh = sum(pmf[c] * p_hat(nn, c, k) for c in range(nn + 1))
            check(f'pass@k unbiased (exact) n={nn} p={p} k={k}', close(ea, 1 - (1 - p) ** k, 1e-12), (ea, 1 - (1 - p) ** k))
            check(f'pass^k unbiased (exact) n={nn} p={p} k={k}', close(eh, p ** k, 1e-12), (eh, p ** k))
        for c in range(nn + 1):
            for k in range(1, nn + 1): check(f'product form = comb form n={nn} c={c} k={k}', close(p_at_prod(nn, c, k), p_at(nn, c, k), 1e-12))
rng = random.Random(20261003)
for (nn, p, k) in ((10, 0.6, 5), (10, 0.6, 4), (5, 0.3, 3)):
    R = 60000; sa = sh = 0.0
    for _ in range(R):
        c = sum(1 for _ in range(nn) if rng.random() < p); sa += p_at(nn, c, k); sh += p_hat(nn, c, k)
    se = 4 * math.sqrt(0.25 / R)
    check(f'Monte Carlo pass@{k} n={nn} p={p}', abs(sa / R - (1 - (1 - p) ** k)) < se, (sa / R, 1 - (1 - p) ** k))
    check(f'Monte Carlo pass^{k} n={nn} p={p}', abs(sh / R - p ** k) < se, (sh / R, p ** k))
# the plug-in formula is biased when tasks differ: Jensen, the lab's whole point
check('plug-in overrates at lv 2: 1 − 0.4^5 = 99.0% > 79.5%', pct(1 - 0.4 ** 5) == '99.0%' and pct((12 + 20 * (1 - 0.4 ** 5)) / 40) == '79.5%')

# ---------- τ-bench, against the local README
if not SRC.exists(): sys.exit(f'missing local sources in {SRC}')
readme = (SRC / 'tau_readme.md').read_text()
retail = readme.split('Retail', 1)[1] if 'Retail' in readme else readme
m = re.search(r'\[TC \(gpt-4o\)\][^|]*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)', readme[readme.find('| Strategy', readme.find('0.420') + 1):])
check('τ-bench retail TC (gpt-4o) row found', bool(m))
if m: check('τ-bench numbers copied exactly', [float(x) for x in m.groups()] == TAU, m.groups())
check('τ-bench README: tasks outdated', 'outdated versions of the airline and retail tasks' in readme)
check('τ-bench README: simulated user, API tools and policy guidelines', 'a user (simulated by language models)' in readme and 'domain-specific API tools and policy guidelines' in readme)
check('τ-bench README: TC = tool-calling', 'TC = `tool-calling` strategy' in readme)
he = (SRC / 'he_eval.py').read_text()
check('human-eval: estimate_pass_at_k product form', '1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))' in he and 'if n - c < k:' in he, '')
check('human-eval README: no unbiased estimate with fewer samples than k', 'no unbiased way of estimating pass@k when there are fewer samples than k' in ' '.join((SRC / 'he_readme.md').read_text().split()))

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width': 1440, 'height': 1000})
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(100)
    pg.evaluate("()=>{location.hash='pass-k'}"); pg.wait_for_timeout(600)
    H = lambda js, *a: pg.evaluate(js, *a)
    C = H("()=>window.__pak.consts")
    check('consts', C['NT'] == NT and C['Q'] == Q and C['NS'] == NS and C['LVMAX'] == LVMAX and C['RING'] == RING and C['TAU'] == TAU, C)
    # part 1's shuffled order
    r7 = mulberry(7); ix = list(range(NT))
    for i in range(NT - 1, 0, -1):
        k = math.floor(r7() * (i + 1)); ix[i], ix[k] = ix[k], ix[i]
    check('part 1 order (mulberry32, seed 7)', C['SHUF'] == ix, (C['SHUF'][:8], ix[:8]))
    # the tries and every estimate, at every level, n and three seeds
    for seed in (1, 2, 7):
        for nn in NS:
            B = tries(seed, nn)
            check(f'tries seed {seed} n {nn}', H("([s,n])=>window.__pak.tries(s,n)", [seed, nn]) == B)
            for r, ring in enumerate(RING):
                check(f'ring {r} passes 60% (seed {seed}, n {nn})', sum(sum(B[i]) for i in ring) == round(0.6 * nn * 10))
            check(f'rings 2 and 3: 12 of 20 first tries (seed {seed}, n {nn})', sum(1 for i in RING[2] + RING[3] if B[i][0]) == 12)
            for lv in range(LVMAX + 1):
                d, js = derive(lv, nn, seed), H("(s)=>window.__pak.compute(s)", {'lv': lv, 'n': nn, 'seed': seed})
                check(f'c and kinds lv {lv} n {nn} seed {seed}', js['c'] == d['c'] and js['kind'] == d['kind'])
                check(f'pass@1 = 60% lv {lv} n {nn} seed {seed}', close(js['p1'], 0.6) and close(d['p1'], 0.6), (js['p1'], d['p1']))
                check(f'first tries lv {lv} n {nn} seed {seed}', d['first1'] == js['first1'], (d['first1'], js['first1']))
                if lv in (2, 4): check(f'first tries: 24 of 40 lv {lv} n {nn} seed {seed}', d['first1'] == 24, d['first1'])
                check(f'never passed lv {lv} n {nn} seed {seed}', js['zero'] == d['zero'])
                for key in ('at', 'hat', 'firstK', 'tAt', 'tHat', 'nAt', 'nHat'):
                    bad = [k for k in range(1, nn + 1) if not close(js[key][str(k)], d[key][k], 1e-9)]
                    check(f'{key} lv {lv} n {nn} seed {seed}', not bad, [(k, js[key][str(k)], d[key][k]) for k in bad[:3]])
                check(f'estimator per task matches the hook lv {lv} n {nn}', all(close(H("([n,c,k])=>window.__pak.pAt(n,c,k)", [nn, c, 3]), p_at(nn, c, 3)) for c in (0, 3, nn)))
    D = {lv: derive(lv, 10, 1) for lv in range(LVMAX + 1)}
    b2 = D[2]
    # the stated facts of the guided parts
    check('true pass@5 at lv 2 = 79.5%', pct(b2['tAt'][5]) == '79.5%', pct(b2['tAt'][5]))
    check('prediction: the right answer is “about 80%”', H("()=>window.__pak.predRight()") == 'mid' and 0.7 <= b2['at'][5] < 0.9)
    check('part 3: three ways', [pct(H("(i)=>window.__pak.estVal(i)", x)) for x in ('naive', 'first', 'full')] == [pct(b2['nAt'][5]), pct(b2['firstK'][5]), pct(b2['at'][5])])
    check('part 3: naive 99.0%', pct(b2['nAt'][5]) == '99.0%', pct(b2['nAt'][5]))
    check('part 5: products', pct(H("()=>window.__pak.prodVal('code')")) == pct(b2['at'][5]) and pct(H("()=>window.__pak.prodVal('support')")) == pct(b2['hat'][4]))
    best = min(range(LVMAX + 1), key=lambda lv: sum((D[lv]['hat'][i + 1] - TAU[i]) ** 2 for i in range(4)))
    check('part 4: the level closest to τ-bench', H("()=>window.__pak.bestTauLv()") == best == 2, (best,))
    check('τ-bench equal-hardness pass^4 = 0.133', f'{0.604 ** 4:.3f}' == '0.133')
    check('lv 0 true pass^4 = 13.0%', pct(D[0]['tHat'][4]) == '13.0%')
    # the challenge
    for ag in ('A', 'B'):
        for k in range(1, 9):
            e, (dn, cl) = H("([a,k])=>window.__pak.chEval(a,k)", [ag, k]), ch_eval(ag, k)
            check(f'challenge {ag} k {k}', close(e['done'], dn) and close(e['calls'], cl) and e['ok'] == (dn >= 0.95 - 1e-12), (e, dn, cl))
    opts = [(ag, k) + ch_eval(ag, k) for ag in ('A', 'B') for k in range(1, 9) if ch_eval(ag, k)[0] >= 0.95 - 1e-12]
    bb = min(opts, key=lambda o: o[3]); jb = H("()=>window.__pak.chBest()")
    check('challenge: fewest calls is A with 3 tries', (jb['ag'], jb['k']) == bb[:2] == ('A', 3) and pct(bb[2]) == '97.3%' and f2(bb[3]) == '1.39', (jb, bb))
    check('challenge: A with 2 tries misses (91.0%)', pct(ch_eval('A', 2)[0]) == '91.0%')
    check('challenge: B never passes 70%', all(ch_eval('B', k)[0] == 0.7 for k in range(1, 9)))
    # the interview drill
    for s in range(1, 12):
        d = H("(s)=>window.__pak.drill(s)", s)
        check(f'drill {s}', all(close(d['at'][i], p_at(d['n'], c, d['k'])) and close(d['hat'][i], p_hat(d['n'], c, d['k'])) for i, c in enumerate(d['cs'])), d)

    # ---------- the numbers on the page
    txt = lambda sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
    mets = lambda: pg.evaluate("()=>[...document.querySelectorAll('#pak-metrics .metric')].map(m=>(m.querySelector('.k').textContent+'='+m.querySelector('.v').textContent).replace(/\\u2011/g,'-'))")
    fast = lambda: (pg.evaluate("()=>{const b=document.querySelector('.seg-ctl[data-key=\"spd\"] button[data-v=\"fast\"]'); if(b) b.click()}"), pg.wait_for_function("()=>window.__pak.state().shown>=window.__pak.state().n", timeout=8000), pg.wait_for_timeout(150))
    go = lambda i: (pg.evaluate(f"()=>document.querySelector('#pak-rail [data-go=\"{i}\"]').click()"), pg.wait_for_timeout(250), fast())
    slide = lambda sel, v: (pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}))}}"), pg.wait_for_timeout(150), fast())
    t = txt('#pak-rail .step.current'); st = txt('#pak-st-sub')
    check('mission: 24 of 40 passed', '24 passed' in t and '24 of 40' in st, (t[:200], st))
    check('mission: pass@1 card 60.0%, pass@5 hidden', mets() == ['pass@1=60.0%', 'pass@5=?'], mets())
    go(1)
    for lv in (0, 2, 4):
        slide('#pak-ic-lv-in', lv)
        o, ms = txt('#pak-rail [data-obs]'), mets()
        check(f'part 2 lv {lv}: result', pct(D[lv]['at'][5]) in o, (o[:160], pct(D[lv]['at'][5])))
        check(f'part 2 lv {lv}: cards', ms == ['pass@1=60.0%', f'pass@5={pct(D[lv]["at"][5])}', f'Never passed={D[lv]["zero"]} of 40'], ms)
    o = (slide('#pak-ic-lv-in', 2), txt('#pak-rail [data-obs]'))[1]
    cap = NT - 4 * 2   # at level 2 the last 4L = 8 tasks never pass; the other 32 can get done at most
    check('part 2 lv 2: the ceiling, 32 tasks = 80% (S6)', f'other {cap} tasks' in o and f'{round(cap / NT * 100)}% of the 40' in o and '80.0%' not in o, o[:400])
    go(2)
    for lev, val in (('naive', b2['nAt'][5]), ('first', b2['firstK'][5]), ('full', b2['at'][5])):
        pg.click(f'#pak-rail [data-lever="{lev}"]'); pg.wait_for_timeout(200)
        r = txt(f'#pak-rail [data-lever="{lev}"] .lv-r'); o = txt('#pak-rail [data-obs]')
        check(f'part 3 {lev}: lever result', r == pct(val), (r, pct(val)))
        check(f'part 3 {lev}: result', pct(val) in o, o[:200])
    check('part 3: the true value', pct(b2['tAt'][5]) in txt('#pak-rail [data-obs]'))
    go(3)
    o = txt('#pak-rail [data-obs]')
    check('part 4 lv 0: τ-bench 38.3% (0.383 in the table) and 13.3%, the lab 13% (estimate)', pct(TAU[3]) == '38.3%' and pct(TAU[3]) in o and '(the table says 0.383)' in o and pct(TAU[0] ** 4) == '13.3%' and pct(TAU[0] ** 4) in o and '0.133' not in o and pct(D[0]['hat'][4]) in o, o[:300])
    slide('#pak-ic-lv4-in', 2); o = txt('#pak-rail [data-obs]')
    check('part 4 lv 2: closest', pct(b2['hat'][4]) in o and 'closest' in o, o[:300])
    check('part 4: τ-bench card in percent (S10)', f'τ-bench pass^4={pct(TAU[3])}' in mets(), mets())
    go(4)
    o = txt('#pak-rail [data-obs]'); check('part 5 code: pass@5', pct(b2['at'][5]) in o, o[:200])
    pg.click('#pak-rail [data-lever="support"]'); pg.wait_for_timeout(200); fast()
    o = txt('#pak-rail [data-obs]'); check('part 5 support: pass^4', pct(b2['hat'][4]) in o and '20 tasks' in o, o[:300])
    check('part 5 support: the lever shows pass^4', txt('#pak-rail [data-lever="support"] .lv-r') == f'look at pass^4: {pct(b2["hat"][4])}')
    go(5)
    t = txt('#pak-rail .step.current')
    # the owner, 2026-10-08 (G2): the three lines state the pattern, not the earlier parts' numbers; the title answer and the
    # pass@10 misconception keep the numbers the reader needs, restated in full
    for v in (b2['at'][5], b2['hat'][4], b2['at'][10], b2['hat'][10]):
        check(f'part 6 shows {pct(v)}', pct(v) in t, t[:400])
    check('part 6 (G2): no line leans on an earlier part (“In part …”), and the earlier parts’ numbers are gone', 'In part' not in t and 'in part' not in t
          and all(pct(v) not in t for v in (D[0]['at'][5], D[4]['at'][5], b2['nAt'][5], TAU[3], TAU[0] ** 4)), t[:600])
    check('part 6 (G2): the τ-bench line states the pattern', 'A real agent’s pass^k falls more slowly than if every task were equally hard.' in t, t[:800])
    ans = txt('#pak-rail .pak-answer')
    check('part 6: the title question answered with pass@5 and pass^4 (S28)', ans.startswith('Back to the opening question') and f'With up to 5 tries, this agent gets {pct(b2["at"][5])} of the tasks done' in ans and f'Only {pct(b2["hat"][4])} of the tasks are answered right all 4 times' in ans, ans)
    st6 = H("()=>window.__pak.state()")
    check('part 6 opens on the first line’s picture: all or nothing, its button pressed (S26)', st6['lv'] == LVMAX and st6['remView'] == 'uneven' and pg.evaluate("()=>document.querySelector('#pak-rail [data-rem=\"uneven\"]').getAttribute('aria-pressed')") == 'true', st6)
    pg.click('#pak-rail [data-rem="estimate"]'); pg.wait_for_timeout(200); fast()   # back to 12 / 20 / 8 for Show the math
    # show the math redoes the shown numbers
    pg.evaluate("()=>{document.getElementById('pak-math').open=true}"); pg.wait_for_timeout(150)
    # 2026-10-09 (H1, H2): the average is typeset, 1/40 × (12 × 1 + … two terms a line …) then “= 79.5%”; read row by row
    ch = pg.evaluate(CHAINS)
    at = None
    for rows in ch:
        for i, (l, e, r) in enumerate(rows):
            if l == 'pass@5' and e == '=' and r.startswith('140×('):
                body, j = r[len('140×('):], i + 1
                while j < len(rows) and rows[j][0] == '' and rows[j][1] == '' and rows[j][2].startswith('+'):
                    body += rows[j][2]; j += 1
                if j < len(rows) and rows[j][1] == '=' and body.endswith(')') and rows[j][2].endswith('%'): at = (body[:-1], rows[j][2][:-1])
                break
        if at: break
    check('math: pass@5 terms', bool(at), ch[:4])
    if at:
        tot = 0.0
        for part in at[0].split('+'):
            a, x = part.split('×'); tot += int(a) * float(x)
        check('math: the terms add up to the shown pass@5', f'{tot / 40 * 100:.1f}' == at[1], (tot / 40, at[1]))
        check('math: shown pass@5 = the estimate', at[1] + '%' == pct(b2['at'][5]))
    pg.evaluate("()=>{document.getElementById('pak-math').open=false}")
    # scenarios
    pg.click('#pak-app .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(300)
    want = [pct(b2['at'][5]), pct(b2['hat'][4]), pct(TAU[3]), pct(D[4]['at'][5])]
    for i in range(4):
        pg.click(f'#pak-rail [data-preset="{i}"]'); pg.wait_for_timeout(200); fast()
        o = txt('#pak-insight'); check(f'scenario {i + 1}', want[i] in o, (want[i], o[:200]))
    o = txt('#pak-insight') if (pg.click('#pak-rail [data-preset="2"]'), fast()) else ''
    check('scenario 3: τ-bench in percent, the dashed line about 13% (S10, O3)', pct(TAU[0]) in o and pct(TAU[3]) in o and f'about {round(b2["nHat"][4] * 100)}% (the dashed line)' in o and round(0.6 ** 4 * 100) == 13 and '0.133' not in o, o[:300])
    # the challenge
    pg.click('#pak-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(300)
    seg = lambda k, v: (pg.click(f'#pak-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]'), pg.wait_for_timeout(150))
    for ag in ('A', 'B'):
        for k in (1, 3, 8):
            seg('chAg', ag); seg('chK', str(k))
            dn, cl = ch_eval(ag, k)
            check(f'challenge card {ag} {k}', txt('#pak-ch-d') == pctc(dn) and txt('#pak-ch-c').startswith(f2(cl)), (txt('#pak-ch-d'), txt('#pak-ch-c'), pct(dn), f2(cl)))
    # B2: Show the math for agent A writes out the terms that exist, and they add up to the calls shown
    pg.evaluate("()=>{document.getElementById('pak-math').open=true}")
    for k in (1, 2, 3, 4, 6, 8):
        seg('chAg', 'A'); seg('chK', str(k))
        # the done line: \u201c= 97.3%\u201d, and for the shares that round to 100.0% \u201c> 99.9%\u201d, never \u201c= >99.9%\u201d
        # 2026-10-09 (H1, H2): the done chain ends in a row “= 97.3%” or “> 99.9%” (never “= >99.9%”); read row by row
        dc = next((rows for rows in pg.evaluate(CHAINS) if rows and rows[0][0] in ('sharedone', '做成的比例')), None)
        dl = (dc[-1][1] + ' ' + dc[-1][2]) if dc else ''
        dw = '> 99.9%' if pctc(ch_eval('A', k)[0]).startswith('>') else '= ' + pct(ch_eval('A', k)[0])
        check(f'challenge math A k {k}: the done line ends \u201c{dw}\u201d', dl == dw and not dc[-1][2].startswith('>'), dl)
        m = pg.evaluate(CALLS)
        check(f'challenge math A k {k}: a calls line', bool(m) and m['rel'] == '=', m)
        if not m: continue
        terms = m['terms']
        val = lambda t: 1.0 if t == '1' else (0.3 if t == '0.3' else 0.3 ** int(re.fullmatch(r'0\.3\^(\d+)', t).group(1)))
        if k <= 4:
            want_terms = ['1', '0.3'] + [f'0.3^{j}' for j in range(2, k)]
            check(f'challenge math A k {k}: every term written (B2)', terms == want_terms[:k], (terms, want_terms[:k]))
            check(f'challenge math A k {k}: the terms add up to the calls shown', f2(sum(val(t) for t in terms)) == m['res'] == f2(ch_eval('A', k)[1]), (terms, m['res']))
        else:
            check(f'challenge math A k {k}: 1 + 0.3 + 0.3² + … + 0.3^(k−1) (B2)', terms == ['1', '0.3', '0.3^2', '…', f'0.3^{k - 1}'], terms)
            check(f'challenge math A k {k}: the calls shown = the series', m['res'] == f2(sum(0.3 ** j for j in range(k))) == f2(ch_eval('A', k)[1]), m['res'])
    pg.evaluate("()=>{document.getElementById('pak-math').open=false}")
    seg('chAg', 'A'); seg('chK', '3'); pg.click('#pak-rail [data-act="submit"]'); pg.wait_for_timeout(250)
    o = txt('#pak-ch-res'); check('challenge: A 3 is the fewest calls', '97.3%' in o and '1.39' in o and 'fewest' in o, o[:300])
    q5 = pct(1 - 0.4 ** 5)
    check('1 − 0.4^5 = 99.0%', q5 == '99.0%' and pct(D[0]['at'][5]) != q5, (q5, pct(D[0]['at'][5])))
    card = "[...document.querySelectorAll('[data-exits]')].find(e=>e.offsetParent!==null)"
    for j in (0, 1):
        pg.evaluate(f"()=>{{const c={card}; c.querySelector('[data-ex=\"tab\"][data-i=\"0\"]').click()}}"); pg.wait_for_timeout(120)
        pg.evaluate(f"()=>{{const c={card}; const d=c.querySelector('.ex-qs .pb-dot[data-j=\"{j}\"]'); if(d) d.click()}}"); pg.wait_for_timeout(120)
        pg.evaluate(f"()=>{{const c={card}; c.querySelector('.ex-opts .choice[data-v=\"0\"]').click()}}"); pg.wait_for_timeout(150)
        w = pg.evaluate(f"()=>{{const c={card}; const e=c.querySelector('.ex-res'); return e?e.innerText:''}}")
        check(f'Explain it q{j + 1}: the per-task chance at 60% is {q5} (B1)', q5 in w and pct(D[0]['at'][5]) not in w, w[:300])
    # the brief chart (the owner's review of batch 8): the options that meet the goal, the pick (A, 3 tries) and the initial setup (B, 1 try)
    rows = pg.evaluate("()=>[...document.querySelectorAll('#pak-ch-res .cmp:not(details .cmp) .cmp-row')].map(r=>r.innerText.replace(/\\s+/g,' '))")
    want = [(a, k) for a in ('A', 'B') for k in range(1, 9) if ch_eval(a, k)[0] >= 0.95 or (a, k) in (('A', 3), ('B', 1))]
    check('challenge chart: 7 rows, the options that meet the goal, the pick and the initial setup', len(rows) == len(want) == 7, (len(rows), want))
    for i, (ag, k) in enumerate(want):
        if i < len(rows): check(f'challenge chart row {ag} {k}', pctc(ch_eval(ag, k)[0]) in rows[i] and f2(ch_eval(ag, k)[1]) in rows[i], rows[i])
    note = txt('#pak-ch-res .cmp-note')
    check('challenge chart: the note counts the 9 options left out', '9' in note, note)
    # the Connect card's code, run here
    code = pg.evaluate("()=>document.getElementById('pak-code').textContent")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): exec(compile(code, 'pak-code', 'exec'), {})
    printed = buf.getvalue().strip().splitlines(); shown = [l[2:] for l in code.splitlines() if l.startswith('# pass')]
    check('Connect code prints what it shows', printed == shown, (printed, shown))
    # the drill answer on the page
    pg.click('#pak-app [data-act="drill-reveal"]'); pg.wait_for_timeout(150)
    d = H("()=>window.__pak.drill(3)"); a = txt('#pak-drill-a')
    check('drill answer on the page', pct(mean([p_at(d['n'], c, d['k']) for c in d['cs']])) in a and pct(mean([p_hat(d['n'], c, d['k']) for c in d['cs']])) in a, a[:200])
    # the home tile's values
    T = H("()=>{const t=window.__tzPak; return t?{p1:t.opts.map(t.p1),p4:t.opts.map(t.p4),mix:t.mix,read:t.opts.map(t.read)}:null}")
    check('tile hook', T is not None)
    if T:
        mix = {'even': [.5] * 8, 'mix': [1, 1, .5, .5, .5, .5, 0, 0], 'split': [1, 1, 1, 1, 0, 0, 0, 0]}
        check('tile mixes', T['mix'] == mix)
        want4 = [mean([1 - (1 - q) ** 4 for q in mix[m]]) for m in ('even', 'mix', 'split')]
        check('tile pass@1 is 50% in all three', all(close(x, 0.5) for x in T['p1']))
        check('tile pass@4: 93.75%, 71.875%, 50%', [round(x, 6) for x in T['p4']] == [0.9375, 0.71875, 0.5] == [round(x, 6) for x in want4], T['p4'])
        check('tile sentences carry the values', all(pct(v) in re.sub('<[^>]+>', '', s) for v, s in zip(want4, T['read'])), T['read'])
    check('no page errors', not errs, errs)
    b.close()

print(f'{n - len(fails)}/{n} checks passed')
print('FAILED:', fails[:20] if fails else 'none')
sys.exit(1 if fails else 0)
