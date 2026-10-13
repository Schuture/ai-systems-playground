"""Numbers test for the agent-capstone lab (CONTRACT §6).

An independent Python copy of the lab's engine (the counter-based hash, the three suites, one run of a task under a harness, the
evaluation: the tasks' success shares, their mean, the 95% interval as lm-eval's mean_stderr over the tasks, pass^3 from the counts,
unsafe actions, cost) is compared with window.__acp:
- every task of the three suites; every run of the support suite under the initial and the final harness, field by field;
- the evaluations the page shows: part 1 (the initial setup and the four predictions; the right answer is not the stronger model),
  parts 2 to 4 (each lever), part 5 (1, 3 and 5 runs per task, the paired gain), the scenarios, two draws;
- the launch goal judged on the values as shown (one decimal);
- the challenge: all 108 combinations, the passing ones, the cheapest (minimal), and the tiers' combinations used by the flow test;
- the drill (seeds 1 to 6); the teaser's 36 cells against the engine's tile suite, and its sentence;
- the Python in Connect, run here: its printed tuple must equal its comment and the engine's numbers;
- the fix pass (reviews/agent-capstone-b7v2-review.md): what the interval is made of (the luck of each run vs which tasks are in
  the suite, from 400 draws per task) and the narrowing from 3 to 5 runs (B1); each half of “handle errors” alone (S1); the tasks
  that lose what step 1 looked up (B3, S2); confirmations per 100 runs (S6); the doubled cost (S11); differences of shown values
  (S5); the new case's ±7.9; the home tile's sentence and tooltip (S14);
- the product owner's check (2026-10-04): whose 60% the check finds (the 40% that are silent), in part 2, the fidelity popover and
  Settings, in both languages, from the engine's r and C (R6.6);
- the cited sources, from the local copies in sources/b7src/: lm-eval's sample_stddev and mean_stderr, AgentDojo's DEFENSES, the arXiv
  ids and authors of τ-bench, AgentDojo and Lost in the Middle.
Run: python3 tests/agent-capstone_numbers.py"""
import math, pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'agent-capstone'; OUT.mkdir(parents=True, exist_ok=True)
SRC = P / 'sources' / 'b7src'
subprocess.run([sys.executable, str(P / 'build.py'), 'out/agent-capstone.html', '--only', 'agent-capstone'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(OUT / 'numbers.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'agent-capstone.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'numbers.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


close = lambda a, b, t=1e-9: a is not None and b is not None and abs(a - b) <= t

# ------------------------------------------------------------------ the engine, again
M32 = 0xFFFFFFFF


def mix(h):
    h &= M32; h ^= h >> 16; h = (h * 0x7feb352d) & M32; h ^= h >> 15; h = (h * 0x846ca68b) & M32; h ^= h >> 16
    return h & M32


def u01(seed, a, b, c, d):
    h = mix(((seed * 0x9E3779B1) & M32) ^ 0x85ebca6b)
    for v, k in ((a, 0x27d4eb2f), (b, 0x165667b1), (c, 0x2545f491), (d, 0x68e31da4)):
        h = mix(h ^ (((v + 1) * k) & M32))
    return h / 4294967296


K = dict(B=3000, W=131072, TURN=200, ERR=300, TRIM=1000, SUMT=1000, NOTES=300, EVERY=8, LASTN=8, PAGE=6000, SMALL=600, C=0.6, F=0.5, D=0.8)
PRICE = {'cheap': 1, 'strong': 5}
SUITES = {
    'shop': dict(seed=11, kinds=[('L', 20), ('E', 20), ('U', 10)], n=dict(L=(3, 3), E=(12, 9), U=(5, 3)), q=dict(L=0.03, E=0.05, U=0.04), r=0.6, outE=(3000, 8), inj=lambda j: j % 2 == 0, need=0.7),
    'ops': dict(seed=23, kinds=[('L', 10), ('E', 25), ('U', 15)], n=dict(L=(3, 3), E=(14, 11), U=(5, 3)), q=dict(L=0.03, E=0.05, U=0.04), r=0.9, outE=(4000, 8), inj=lambda j: j % 3 != 2, need=0.7),
    'tile': dict(seed=89, kinds=[('L', 4), ('E', 5), ('U', 3)], n=dict(L=(3, 3), E=(12, 9), U=(5, 3)), q=dict(L=0.03, E=0.05, U=0.04), r=0.6, outE=(3000, 8), inj=lambda j: j % 2 == 0, need=0.7)}
INIT = dict(model='cheap', R=0, ck=0, ctx='all', gate='none', K=3, draw=1)
FINAL = dict(INIT, R=2, ck=1, ctx='notes', gate='confirm')
BUDGET = dict(shop=15, ops=12, tile=15)
TC = {}


def tasks(name):
    if name in TC: return TC[name]
    S, out, i = SUITES[name], [], 0
    for kind, cnt in S['kinds']:
        for j in range(cnt):
            u = lambda p: u01(S['seed'], i, p, 0, 0)
            lo, span = S['n'][kind]; x = u(1)
            nn = lo + math.floor(u(0) * span); m = 0.5 if x < 0.40 else (1 if x < 0.75 else (2 if x < 0.90 else 4))
            o = S['outE'][0] + math.floor(u(2) * S['outE'][1]) * 1000 if kind == 'E' else K['SMALL']
            out.append(dict(i=i, kind=kind, n=nn, m=m, out=o, need=kind == 'E' and u(3) < S['need'], inj=kind == 'U' and S['inj'](j))); i += 1
    TC[name] = out; return out


def run(name, t, k, h):
    S = SUITES[name]; seed = S['seed'] * 1000 + 7 + 101 * (h['draw'] - 1)
    p = 1 - S['q'][t['kind']] * t['m'] * (0.5 if h['model'] == 'strong' else 1)
    hist, summ, tokens, calls, conf, wrong, unsafe, res, steps = [], 0, 0, 0, 0, False, False, None, 0
    live = t['inj']
    if t['inj'] and h['gate'] == 'detect' and u01(seed, t['i'], k, 0, 4) < K['D']: live = False
    follows = live and u01(seed, t['i'], k, 0, 3) < K['F']
    for j in range(1, t['n'] + 1):
        o = K['PAGE'] if (t['kind'] == 'U' and j == 2) else t['out']
        if h['ctx'] == 'notes': o = min(o, K['TRIM'])
        ctx = K['B'] + (sum(hist) if h['ctx'] == 'all' else (sum(hist[-K['LASTN']:]) if h['ctx'] == 'last' else summ))
        if h['ctx'] == 'all' and ctx > K['W']: res = 'overflow'; break
        a, fl, sr = 0, 0, 'ok'
        while True:
            calls += 1; tokens += ctx + K['ERR'] * a
            if h['ck']: calls += 1; tokens += ctx + K['ERR'] * a + K['TURN']
            if u01(seed, t['i'], k, j * 4 + a, 0) < p: sr = 'ok'; break
            loud = u01(seed, t['i'], k, j * 4 + a, 1) < S['r'] or (bool(h['ck']) and u01(seed, t['i'], k, j * 4 + a, 2) < K['C'])
            if not loud: sr = 'silent'; break
            fl += 1
            if a >= h['R']: sr = 'stop'; break
            a += 1
        steps = j
        if sr == 'stop': res = 'stop'; break
        if sr == 'silent': wrong = True
        if t['kind'] == 'U' and j == 3 and follows:
            if h['gate'] == 'confirm': conf += 1
            else: unsafe = True
        if t['kind'] == 'U' and j == t['n'] and h['gate'] == 'confirm': conf += 1
        grow = o + K['TURN'] + K['ERR'] * fl
        if h['ctx'] == 'notes':
            summ += grow
            if j % K['EVERY'] == 0 and j < t['n']: calls += 1; tokens += K['B'] + summ; summ = K['SUMT'] + K['NOTES']
        else: hist.append(grow)
    if unsafe: res = 'unsafe'
    elif res is None: res = 'lost' if (t['need'] and h['ctx'] == 'last' and t['n'] > K['LASTN'] + 1) else ('wrong' if wrong else 'ok')
    return dict(res=res, unsafe=unsafe, tokens=tokens, calls=calls, conf=conf, steps=steps, cost=tokens * PRICE[h['model']] / 1e6 * 100)


def comb(nn, k):
    if k < 0 or k > nn: return 0
    r = 1
    for i in range(1, k + 1): r = r * (nn - k + i) / i
    return r


EC = {}


def evaluate(name, h):
    h = dict(INIT, **h); key = (name,) + tuple(h[x] for x in ('model', 'R', 'ck', 'ctx', 'gate', 'K', 'draw'))
    if key in EC: return EC[key]
    T, Kr = tasks(name), h['K']; runs, per, cnt = [], [], []
    for t in T:
        rs = [run(name, t, k, h) for k in range(Kr)]; runs.append(rs); c = sum(r['res'] == 'ok' for r in rs); cnt.append(c); per.append(c / Kr)
    nn = len(T); mean = 0
    for x in per: mean += x
    mean /= nn
    v = 0
    for x in per: v += (x - mean) * (x - mean)
    sd = math.sqrt(v / (nn - 1)); se = sd / math.sqrt(nn)
    flat = [r for rs in runs for r in rs]; N = len(flat)
    p3 = None
    if Kr >= 3:
        p3 = 0
        for c in cnt: p3 += comb(c, 3) / comb(Kr, 3)
        p3 /= nn
    cost = 0
    for r in flat: cost += r['cost']
    confs = 0
    for r in flat: confs += r['conf']
    e = dict(runs=runs, per=per, cnt=cnt, succ=mean, sd=sd, lo=mean - 1.96 * se, hi=mean + 1.96 * se, p3=p3, unsafe=sum(r['unsafe'] for r in flat), cost=cost / N, conf=confs / N,
             causes={c: sum(r['res'] == c for r in flat) for c in ('stop', 'wrong', 'lost', 'overflow', 'unsafe')}, N=N, n=nn, K=Kr)
    EC[key] = e; return e


r1 = lambda x: math.floor(x * 10 + 0.5) / 10   # JS Math.round(x*10)/10


def goal(e, budget):
    s, lo, c = r1(e['succ'] * 100), r1(max(0, e['lo']) * 100), r1(e['cost'])
    p3 = None if e['p3'] is None else r1(e['p3'] * 100)
    g = dict(succ=s >= 85 and lo >= 80, p3=p3 is not None and p3 >= 70, unsafe=e['unsafe'] == 0, cost=c <= budget)
    g['all'] = all(g.values()); g['miss'] = sum(not g[k] for k in ('succ', 'p3', 'unsafe', 'cost')); return g


def gain(name, h, h0):
    a, b = evaluate(name, h), evaluate(name, h0); d = [x - y for x, y in zip(a['per'], b['per'])]; nn = len(d)
    m = 0
    for x in d: m += x
    m /= nn
    v = 0
    for x in d: v += (x - m) * (x - m)
    sd = math.sqrt(v / (nn - 1)); half = 1.96 * sd / math.sqrt(nn)
    return dict(m=m, lo=m - half, hi=m + half)


fP = lambda x: f'{max(0, min(1, x)) * 100:.1f}%'

# ------------------------------------------------------------------ the sources (local copies)
lme = (SRC / 'lmeval_metrics.py').read_text()
check('source: lm-eval sample_stddev divides by n − 1', 'return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))' in lme)
check('source: lm-eval mean_stderr = sample_stddev / √n', 'return sample_stddev(arr) / math.sqrt(len(arr))' in lme)
dojo = (SRC / 'agentdojo_pipeline.py').read_text()
m = re.search(r'DEFENSES = \[(.*?)\]', dojo, re.S)
check('source: AgentDojo DEFENSES = tool_filter, transformers_pi_detector, spotlighting_with_delimiting, repeat_user_prompt',
      m and re.findall(r'"(\w+)"', m.group(1)) == ['tool_filter', 'transformers_pi_detector', 'spotlighting_with_delimiting', 'repeat_user_prompt'])
tau = (SRC / 'tau_readme.md').read_text()
check('source: τ-bench is arXiv 2406.12045 by Yao, Shinn, Razavi, Narasimhan', 'eprint={2406.12045}' in tau and 'Shunyu Yao and Noah Shinn and Pedram Razavi and Karthik Narasimhan' in tau)
check('source: τ-bench reports Pass^k', 'Pass^3' in tau)
ad = (SRC / 'agentdojo_readme.md').read_text()
check('source: AgentDojo is arXiv 2406.13352 by Debenedetti et al.', '2406.13352' in ad and 'author={Edoardo Debenedetti' in ad)
lm = (SRC / 'litm_readme.md').read_text()
check('source: Lost in the Middle is arXiv 2307.03172 by Liu et al.', '2307.03172' in lm and 'Nelson F. Liu' in lm and lm.startswith('# Lost in the Middle: How Language Models Use Long Contexts'))

# ------------------------------------------------------------------ the page
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.evaluate("()=>{location.hash='agent-capstone'}"); pg.wait_for_timeout(400)
    H = lambda js, *a: pg.evaluate(js, *a)
    check('hook present', H("()=>!!window.__acp"))
    C = H("()=>window.__acp.consts")
    check('constants: the engine sizes and rates', C['K'] == K, C['K'])
    check('constants: prices, budgets, goal', C['PRICE'] == PRICE and C['BUDGET'] == BUDGET and C['GOAL'] == {'succ': 85, 'lo': 80, 'p3': 70, 'unsafe': 0})
    check('constants: the initial and the final harness', C['INIT'] == INIT and C['FINAL'] == FINAL)
    check('constants: the six parts', C['STEP_IDS'] == ['mission', 'loud', 'context', 'gate', 'prove', 'remember'])
    for nm in SUITES:
        js = H("n=>window.__acp.tasks(n)", nm); py = tasks(nm)
        check(f'{nm}: {len(py)} tasks, identical', js == py, [x for x, y in zip(js, py) if x != y][:2])
        check(f'{nm}: suite seed and rates', C['SUITES'][nm]['seed'] == SUITES[nm]['seed'] and C['SUITES'][nm]['r'] == SUITES[nm]['r'])
    # every run of the support suite under the initial and the final harness, field by field
    for hn, h in (('initial', INIT), ('final', FINAL)):
        js = H("h=>{const a=window.__acp, o=[]; for(let i=0;i<50;i++) for(let k=0;k<3;k++){const r=a.run('shop',i,k,h); o.push([r.res,r.tokens,r.calls,r.conf,r.steps]);} return o}", h)
        py = [[r['res'], r['tokens'], r['calls'], r['conf'], r['steps']] for rs in evaluate('shop', h)['runs'] for r in rs]
        check(f'support suite, {hn} harness: all 150 runs identical (result, tokens, calls, confirmations, steps)', js == py, [(i, x, y) for i, (x, y) in enumerate(zip(js, py)) if x != y][:3])

    def same(name, h, label):
        j = H("([n,h])=>window.__acp.evaluate(n,h)", [name, h]); e = evaluate(name, h)
        ok = j['cnt'] == e['cnt'] and close(j['succ'], e['succ']) and close(j['lo'], e['lo']) and close(j['hi'], e['hi']) and j['unsafe'] == e['unsafe'] and close(j['cost'], e['cost'], 1e-6) \
            and ((j['p3'] is None and e['p3'] is None) or close(j['p3'], e['p3'])) and j['causes'] == e['causes'] and close(j['conf'], e['conf'])
        check(f'{label}: {fP(e["succ"])} [{fP(e["lo"])}, {fP(e["hi"])}], pass^3 {"—" if e["p3"] is None else fP(e["p3"])}, unsafe {e["unsafe"]}, {e["cost"]:.2f} cents', ok, (j, {k: e[k] for k in ('succ', 'lo', 'p3', 'unsafe', 'cost', 'causes')}))
        return e
    # part 1
    e0 = same('shop', INIT, 'part 1, initial setup')
    check('part 1: the initial setup misses all four conditions', goal(e0, 15)['miss'] == 4)
    preds = {}
    for v, d in C['PRED_H'].items():
        preds[v] = same('shop', dict(INIT, **d), f'part 1, prediction {v}')['succ']
    best = max(preds, key=preds.get)
    check('part 1: the right answer is “retries and a check”, not the stronger model', best == 'a' and H("()=>window.__acp.predRight()") == 'a', preds)
    check('part 1: the stronger model is the runner-up (the tempting answer), at more than five times the cost',
          sorted(preds, key=preds.get, reverse=True)[1] == 'd' and evaluate('shop', dict(INIT, model='strong'))['cost'] > 5 * e0['cost'])
    # parts 2 to 4
    P2, P3, P4 = dict(INIT), dict(INIT, R=2, ck=1), dict(INIT, R=2, ck=1, ctx='notes')
    for lab, h in (('part 2, 2 retries', dict(P2, R=2)), ('part 2, 2 retries and a check', P3), ('part 3, last 8 steps', dict(P3, ctx='last')), ('part 3, trim + notes', P4),
                   ('part 4, detector', dict(P4, gate='detect')), ('part 4, a person confirms', dict(P4, gate='confirm'))):
        same('shop', h, lab)
    check('part 2: retries alone leave more wrong runs than with a check', evaluate('shop', dict(P2, R=2))['causes']['wrong'] > evaluate('shop', P3)['causes']['wrong'])
    check('part 3: the last 8 steps lose early requests and score below keep-all', evaluate('shop', dict(P3, ctx='last'))['causes']['lost'] > 0 and evaluate('shop', dict(P3, ctx='last'))['succ'] < evaluate('shop', P3)['succ'])
    check('part 3: trim + notes: no overflow, cost below a third of keep-all', evaluate('shop', P4)['causes']['overflow'] == 0 and evaluate('shop', P4)['cost'] < evaluate('shop', P3)['cost'] / 3)
    check('part 4: the detector leaves unsafe actions (> 0); confirmation makes them 0', evaluate('shop', dict(P4, gate='detect'))['unsafe'] > 0 and evaluate('shop', dict(P4, gate='confirm'))['unsafe'] == 0)
    # part 5
    for k in (1, 3, 5):
        e = same('shop', dict(FINAL, K=k), f'part 5, {k} runs per task')
        g = goal(e, 15)
        check(f'part 5, {k} runs: goal {"met" if k > 1 else "missed (lower end, pass^3)"}', g['all'] == (k > 1) and (k > 1 or (not g['succ'] and not g['p3'])), g)
        gj = H("([h,h0])=>window.__acp.gain('shop',h,h0)", [dict(FINAL, K=k), dict(INIT, K=k)]); gp = gain('shop', dict(FINAL, K=k), dict(INIT, K=k))
        check(f'part 5, {k} runs: paired gain +{gp["m"] * 100:.1f} points [{gp["lo"] * 100:.1f}, {gp["hi"] * 100:.1f}], above 0', close(gj['m'], gp['m']) and close(gj['lo'], gp['lo']) and close(gj['hi'], gp['hi']) and gp['lo'] > 0)
    w3, w5 = evaluate('shop', FINAL), evaluate('shop', dict(FINAL, K=5))
    check('part 5: 5 runs narrow the interval only a little (less than a quarter)', (w5['hi'] - w5['lo']) > 0.75 * (w3['hi'] - w3['lo']))
    check('final harness, goal as judged by the page', H("h=>window.__acp.goal('shop',h)", FINAL) == goal(w3, 15))
    # B1: what the interval is made of. Each task's true success chance from 400 draws × 3 runs (Python, independent of the page);
    # the page's own estimate (200 draws) must agree, and at 3 runs most of the variance of a task's share is the luck of each run.
    def split(h, D):
        ps = []
        for t in tasks('shop'):
            c = 0
            for dr in range(1, D + 1):
                hd = dict(h, draw=dr)
                for k in range(h['K']): c += run('shop', t, k, hd)['res'] == 'ok'
            ps.append(c / (D * h['K']))
        m = sum(ps) / len(ps); bt = sum((x - m) ** 2 for x in ps) / len(ps); wi = sum(x * (1 - x) for x in ps) / len(ps)
        return bt, wi
    bt, wi = split(FINAL, 400); luck3 = (wi / 3) / (bt + wi / 3); luck5 = (wi / 5) / (bt + wi / 5)
    js = H("()=>window.__acp.lucksplit('shop',{R:2,ck:1,ctx:'notes',gate:'confirm'})")
    bt2, wi2 = split(FINAL, 200)
    check(f'B1: the page’s split (200 draws) equals the Python one: between {bt2:.4f}, luck {wi2:.4f}/k', close(js['between'], bt2, 1e-12) and close(js['within'], wi2, 1e-12), js)
    check(f'B1: at 3 runs most of the interval is the luck of each run ({luck3 * 100:.0f}%; 5 runs: {luck5 * 100:.0f}%)', luck3 > 0.6 and js['luck3'] > 0.6, (luck3, js['luck3']))
    floor = 1.96 * math.sqrt(bt / 50) * 100
    check(f'B1: with many runs the half-width would still be about {floor:.1f} points (from {(w3["hi"] - w3["lo"]) / 2 * 100:.1f}): task differences stay', 2 < floor < 4)
    wd = lambda e: r1(min(1, e['hi']) * 100) - r1(max(0, e['lo']) * 100)
    nar = 1 - wd(w5) / wd(w3)
    check(f'B1: 3 → 5 runs narrow the shown interval by about a fifth ({nar * 100:.0f}%), below the all-luck bound 1 − √(3/5) = {(1 - math.sqrt(3 / 5)) * 100:.1f}%', 0.15 <= nar < 0.25 and nar < 1 - math.sqrt(3 / 5))
    half = (w3['hi'] - w3['lo']) / 2 * 100
    check(f'new case: ±{half:.1f} × √(50/20) = ±{half * math.sqrt(2.5):.1f}, and from the shown ±{r1(half):.1f} too', f'{half * math.sqrt(2.5):.1f}' == '7.9' and f'{r1(half) * math.sqrt(2.5):.1f}' == '7.9')
    # S1: “handle errors” is two settings; each alone, and both
    eR, eC, eA, eD = evaluate('shop', dict(INIT, R=2)), evaluate('shop', dict(INIT, ck=1)), evaluate('shop', dict(INIT, R=2, ck=1)), evaluate('shop', dict(INIT, model='strong'))
    check(f'S1: retries alone {fP(eR["succ"])} < the stronger model {fP(eD["succ"])} < retries and a check {fP(eA["succ"])}', eR['succ'] < eD['succ'] < eA['succ'])
    check(f'S1: a check alone leaves success at {fP(eC["succ"])} (= the initial setup)', fP(eC['succ']) == fP(e0['succ']))
    check(f'S1: retries and a check cost less than half the stronger model ({eA["cost"]:.1f} vs {eD["cost"]:.1f} cents)', eA['cost'] < eD['cost'] / 2)
    # S5: differences of the values as shown
    pa, pd = r1(eA['succ'] * 100) - r1(e0['succ'] * 100), r1(eD['succ'] * 100) - r1(e0['succ'] * 100)
    check(f'S5: shown differences {pa:.1f} and {pd:.1f} points', f'{pa:.1f}' == '16.0' and f'{pd:.1f}' == '12.6')
    # S7: retries raise the unsafe actions (runs that stopped on an error reach the outside content)
    check(f'S7: unsafe actions {e0["unsafe"]} → {eR["unsafe"]} with retries', eR['unsafe'] > e0['unsafe'])
    # S11: the check doubles the cost of a run (retries alone → retries and a check)
    check(f'S11: the check doubles the cost ({eR["cost"]:.1f} → {eA["cost"]:.1f} cents, ×{eA["cost"] / eR["cost"]:.2f})', 1.8 <= eA['cost'] / eR['cost'] < 2.3)
    # B3, S2: the edits whose last step needs what step 1 looked up; every one of their runs is lost with the last 8 steps
    T = tasks('shop'); need = [t for t in T if t['need'] and t['n'] > K['LASTN'] + 1]; edits = [t for t in T if t['kind'] == 'E']
    for lab_, h in (('part 3', dict(INIT, R=2, ck=1, ctx='last')), ('scenario 2', dict(FINAL, ctx='last'))):
        e = evaluate('shop', h)
        check(f'B3/S2 {lab_}: {len(need)} of {len(edits)} edits need step 1 at the last step; all {e["causes"]["lost"]} of their runs lost', len(need) == 14 and len(edits) == 20 and e['causes']['lost'] == len(need) * 3
              and all(e['runs'][t['i']][k]['res'] == 'lost' for t in need for k in range(3)) and min(t['n'] for t in edits) > 9)
    # S6: confirmations per 100 runs
    check(f'S6: {w3["conf"] * 100:.1f} confirmations per 100 runs (about 25)', round(w3['conf'] * 100) == 25)
    # scenarios
    for ps in C['PRESETS']:
        h = {k: ps[k] for k in ('model', 'R', 'ck', 'ctx', 'gate', 'K')}
        e = same('shop', h, f'scenario {ps["id"]}')
        check(f'scenario {ps["id"]}: misses the goal', not goal(e, 15)['all'])
    # draws
    for d in (2, 3):
        same('shop', dict(FINAL, draw=d), f'final harness, draw {d}')
        same('shop', dict(INIT, draw=d), f'initial setup, draw {d}')
    # the challenge
    allj = H("()=>window.__acp.chAll()")
    check('challenge: 108 combinations', len(allj) == 108)
    bad, passing = [], []
    for x in allj:
        h = dict(INIT, **x['o']); e = evaluate('ops', h); g = goal(e, 12)
        if not (close(x['succ'], e['succ']) and close(x['lo'], e['lo']) and close(x['p3'], e['p3']) and x['unsafe'] == e['unsafe'] and close(x['cost'], e['cost'], 1e-6) and x['g'] == g): bad.append(x['o'])
        if g['all']: passing.append((r1(e['cost']), x['o']['R'], x['o'], e))
    check('challenge: every combination identical (rate, lower end, pass^3, unsafe, cost, goal)', not bad, bad[:3])
    passing.sort(key=lambda t: (t[0], t[1]))
    check('challenge: exactly 2 combinations pass', len(passing) == 2, [t[2] for t in passing])
    bj = H("()=>window.__acp.chBest()")
    check('challenge: the cheapest passing one is cheap, 1 retry, no check, trim + notes, a person confirms',
          bj and bj['o'] == {'model': 'cheap', 'R': 1, 'ck': 0, 'ctx': 'notes', 'gate': 'confirm'} == passing[0][2], bj)
    check('challenge: the guided parts’ harness (with a check) succeeds but is over the budget', (lambda g: g['succ'] and g['p3'] and g['unsafe'] and not g['cost'])(goal(evaluate('ops', dict(FINAL)), 12)))
    check('challenge: the stronger model with the same harness is over the budget', not goal(evaluate('ops', dict(INIT, model='strong', R=2, ck=0, ctx='notes', gate='confirm')), 12)['cost'])
    check('challenge: the initial setup misses', not goal(evaluate('ops', INIT), 12)['all'])
    for gname in ('none', 'detect'):
        check(f'challenge: without a person’s confirmation ({gname}) nothing passes', not any(g for (_, _, o, _) in passing for g in [o['gate'] == gname]))
    # the drill
    for s in range(1, 7):
        d = H("s=>window.__acp.drill(s)", s)
        cnt = []
        for i in range(10):
            u = u01(900 + s, i, 0, 0, 0); cnt.append(3 if u < 0.55 else (2 if u < 0.75 else (1 if u < 0.9 else 0)))
        sh = [c / 3 for c in cnt]; mean = sum(sh) / 10; sd = math.sqrt(sum((x - mean) ** 2 for x in sh) / 9); half = 1.96 * sd / math.sqrt(10)
        check(f'drill {s}: counts {cnt}: {fP(mean)} ± {half * 100:.1f}, pass^3 {cnt.count(3) / 10:.1f}', d['cnt'] == cnt and close(d['mean'], mean) and close(d['half'], half) and close(d['p3'], cnt.count(3) / 10))
    # the teaser
    tz = H("()=>window.__tzAcp?window.__tzAcp.RUNS:null")
    check('teaser hook present', tz is not None)
    if tz:
        for hn, h in (('old', INIT), ('new', FINAL)):
            e = evaluate('tile', h)
            rows = [''.join('o' if rs[k]['res'] == 'ok' else ('u' if rs[k]['res'] == 'unsafe' else 'f') for rs in e['runs']) for k in range(3)]
            check(f'teaser, {hn} harness: the 36 cells are the engine’s tile suite', tz[hn] == rows, (tz[hn], rows))
            st = H("h=>window.__tzAcp.stats(h)", hn)
            check(f'teaser, {hn}: {fP(e["succ"])} [{fP(e["lo"])}, {fP(e["hi"])}], pass^3 {fP(e["p3"])}', close(st['succ'], e['succ']) and close(st['lo'], e['lo']) and close(st['hi'], e['hi']) and close(st['p3'], e['p3']))
            rd = H("h=>window.__tzAcp.read(h)", hn)
            check(f'teaser, {hn}: the sentence shows the rate', fP(e['succ']) in rd, rd)
            hp = H("h=>window.__tzAcp.help(h)", hn)
            check(f'teaser, {hn}: the tooltip shows both intervals and both pass^3', all(x in hp for x in (fP(e['lo']), fP(e['hi']), fP(e['p3']))), hp)
        o, nw = evaluate('tile', INIT), evaluate('tile', FINAL)
        check('teaser: the old harness: fewer than one task in five succeeds all 3 times, and the sentence says so', o['p3'] < 0.2 and 'fewer than one task in five' in H("()=>window.__tzAcp.read('old')"))
        check('teaser: the two intervals do not overlap, and the sentence says so', nw['lo'] > o['hi'] and 'does not overlap the old one' in H("()=>window.__tzAcp.read('new')"))
        check('teaser: intervals inside 0% to 100%', 0 <= o['lo'] and nw['hi'] <= 1)
    # the Python in Connect
    code = H("()=>window.__acp.codeText()")
    (OUT / 'connect.py').write_text(code)
    r = subprocess.run([sys.executable, str(OUT / 'connect.py')], capture_output=True, text=True)
    printed = r.stdout.strip(); cm = re.search(r'#\s*(\(.*\))\s*$', code.strip().splitlines()[-1])
    e = evaluate('shop', FINAL)
    want = (round(e['succ'], 3), round(e['lo'], 3), round(sum(1 for c in e['cnt'] if c == 3) / 50, 3))
    check('connect: the code runs', r.returncode == 0, r.stderr)
    check('connect: printed tuple = its comment', cm and printed == cm.group(1), (printed, cm and cm.group(1)))
    check(f'connect: printed tuple = the engine’s part 5 numbers {want}', printed == str(want), (printed, want))
    check('connect: counts = part 5’s counts', re.sub(r'\s+', '', re.search(r'counts = (\[[^\]]*\])', code).group(1)) == '[' + ','.join(map(str, e['cnt'])) + ']')
    check('connect: no line longer than 74 characters', max(len(x) for x in code.splitlines()) <= 74, max(code.splitlines(), key=len))
    check('connect: at most 60 lines', len(code.splitlines()) <= 60, len(code.splitlines()))
    # the numbers shown: part 1's result, after the prediction and the runs
    H("()=>document.querySelector('#acp-rail [data-pred=\"d\"]').click()"); pg.wait_for_timeout(200)
    H("()=>document.querySelector('#acp-app [data-act=\"end\"]').click()"); pg.wait_for_timeout(400)
    obs = H("()=>document.querySelector('#acp-rail [data-obs]').innerText")
    check('part 1 result shows the initial rate, its interval and each prediction’s rate', all(x in obs for x in [fP(e0['succ']), fP(e0['lo']), fP(e0['hi'])] + [fP(v) for v in preds.values()]), obs)
    mt = H("()=>[...document.querySelectorAll('#acp-metrics .metric .v')].map(e=>e.textContent)")
    check('part 1 cards: rate, pass^3, unsafe, cost', mt == [fP(e0['succ']), fP(e0['p3']), str(e0['unsafe']), f'{e0["cost"]:.1f} cents'], mt)
    # part 5's cards
    H("()=>document.querySelector('#acp-rail [data-go=\"4\"]').click()"); pg.wait_for_timeout(300)
    H("()=>document.querySelector('#acp-app [data-act=\"end\"]').click()"); pg.wait_for_timeout(400)
    mt = H("()=>[...document.querySelectorAll('#acp-metrics .metric .v')].map(e=>e.textContent)")
    check('part 5 cards: 90.0%, 74.0%, 0, 13.4 cents', mt == [fP(w3['succ']), fP(w3['p3']), '0', f'{w3["cost"]:.1f} cents'], mt)
    mb = H("()=>document.querySelector('#acp-mathbody').textContent")
    check('Show the math: the mean, the half-width and the interval', all(x in mb for x in (f'{w3["sd"]:.4f}', f'{1.96 * w3["sd"] / math.sqrt(50):.4f}', fP(w3['lo']), fP(w3['hi']))), mb[:500])
    H("()=>document.querySelector('#acp-rail [data-go=\"5\"]').click()"); pg.wait_for_timeout(300)
    rem = H("()=>document.querySelector('#acp-rail .step.current').innerText")
    # the owner, 2026-10-08 (G2): the three points state the pattern, not part 1's 16.0 and 12.6 points or part 4's and 5's numbers
    check('part 6 (G2): the points state the pattern, no earlier part’s numbers', 'In part' not in rem and '16.0 points' not in rem and 'doubled the cost' not in rem
          and 'retries and a check raise success more than a stronger model that costs far more' in rem and 'only a person’s confirmation keeps them at zero' in rem, rem[:400])
    # the product owner (2026-10-04, R6.6): whose 60% the check finds. Of all errors r are reported and 1 − r are silent; the check finds C
    # of the silent ones, so (1 − r)(1 − C) stay unseen. The shares, worked out here from the Python engine's r and C, in both languages.
    rr, cc = SUITES['shop']['r'], K['C']
    rep, sil, ckp, left = (f'{round(x * 100)}%' for x in (rr, 1 - rr, cc, (1 - rr) * (1 - cc)))
    check(f'R6.6: the shares: {rep} reported, {sil} silent, the check finds {ckp} of those, {left} of all errors unseen (assumptions unchanged: r 0.6, C 0.6)',
          (rep, sil, ckp, left) == ('60%', '40%', '60%', '16%') and rr == 0.6 and cc == 0.6 and C['SUITES']['shop']['r'] == rr and C['K']['C'] == cc)
    for lang in ('en', 'zh'):
        H(f"()=>document.querySelector('.langsw button[data-lang=\"{lang}\"]').click()"); pg.wait_for_timeout(300)
        H("()=>document.querySelector('#acp-rail [data-go=\"1\"]').click()"); pg.wait_for_timeout(300)
        p2 = H("()=>document.querySelector('#acp-rail .step.current').innerText.replace(/\\s+/g,' ')")
        want = [f'{rep} of the errors are reported, and the other {sil} are silent', f'Of those {sil} silent errors, the check finds {ckp}'] if lang == 'en' \
            else [f'{rep} 的错会报出来，剩下 {sil} 是悄悄出的', f'这 {sil} 悄悄出的错里，检查能发现 {ckp}']
        check(f'R6.6 part 2 ({lang}): the check finds {ckp} of the {sil} silent errors, said in that order', all(x in p2 for x in want), p2[:600])
        # the shares in the Fidelity card’s “Modeled” list went with it (owner, 2026-10-11 19:28 UTC)
        fid = H("()=>{const p=document.getElementById('agent-capstone-fid'); return p?p.textContent.replace(/\\s+/g,' '):''}")
        ck = H("()=>document.querySelector('#acp-w-ck').textContent.replace(/\\s+/g,' ')"); asc = H("()=>document.querySelector('#acp-as-c').textContent.replace(/\\s+/g,' ')")
        check(f'R6.6 Settings ({lang}): the check note and the assumed value name the {sil} and the {ckp}, and {left} stay unseen', sil in ck and ckp in ck and all(x in asc for x in (sil, ckp, left)), (ck, asc))
        check(f'R6.6 ({lang}): no sentence says the check finds {ckp} of the errors without naming the {sil}', not any(x in p2 + fid + ck for x in ('makes 60% of the silent', '能让 60% 悄悄出的错', '其余的是悄悄出的')))
    check('no page errors', not errs, errs)
    b.close()

print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
