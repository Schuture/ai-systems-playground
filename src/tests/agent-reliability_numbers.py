"""Numbers test for the agent-reliability lab: every number on the page, recomputed independently in Python.

- the closed form (success, finished but wrong, stopped, model calls) against a step-by-step Markov chain written here;
- the seeded runs: the 32-bit hash ported to Python, every one of the chart's 30 tasks replayed cell by cell for several setups, the
  tallies within 3.5 standard errors of the closed form, and the default seed's 30 tasks typical in every guided part (owner check b7);
- the player: one illustration of 30 tasks, no batches, and the default speed (fast) plays a whole run in 10 s or less (R3.9);
- the numbers quoted in the spec and on the page: parts 1 to 6, the four scenarios, the challenge (exactly 3 of 32 combinations pass,
  the minimal one), the capability layer, the drill, the teaser, and the Python in Connect (run here; its printed lines must match its
  comments and SIMOUT);
- the page's own text: the part cards, the scenario cards and the challenge result read in the browser show these numbers.
Run: python3 tests/agent-reliability_numbers.py"""
import pathlib, subprocess, sys, math, io, contextlib, re
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'agent-reliability'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/agent-reliability.html', '--only', 'agent-reliability'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(OUT / 'numbers.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'agent-reliability.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'numbers.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


def close(a, b, tol=1e-9): return abs(a - b) <= tol * max(1.0, abs(b))
def fP(x):
    v = x * 100
    if x <= 0: return '0%'
    if x >= 1: return '100%'
    if v >= 99.95: return '>99.9%'
    if v < 0.05: return '<0.1%'
    return f'{v:.1f}%'
def fC(x): return f'{x:.1f}'


BASE = dict(n=20, p=0.95, r=0.7, R=0, ck=0, kind='i', c=0.8, pb=0.5, seed=377, price=1)
RUNS = 30


# ---------- the closed form, recomputed as a Markov chain over attempts and steps ----------
def step_chain(s):
    """One step: walk the attempts one by one; returns P(ends right), P(silent), P(stops), E[attempts]."""
    loud = s['r'] + (1 - s['r']) * s['c'] if s['ck'] else s['r']
    reach, ok, sil, att = 1.0, 0.0, 0.0, 0.0
    for a in range(s['R'] + 1):
        att += reach
        pa = s['p'] if a == 0 else (s['pb'] if s['kind'] == 'b' else s['p'])
        ok += reach * pa
        sil += reach * (1 - pa) * (1 - loud)
        reach = reach * (1 - pa) * loud            # failed loudly: retry if any left
    return ok, sil, reach, att                      # what is left in reach after the last attempt stops the task


def task_chain(s):
    ok, sil, stop, att = step_chain(s)
    clean, wrong, stopped, calls = 1.0, 0.0, 0.0, 0.0
    per = 2 if s['ck'] else 1
    for _ in range(s['n']):
        alive = clean + wrong
        calls += alive * att * per
        clean, wrong, stopped = clean * ok, wrong * (1 - stop) + clean * sil, stopped + alive * stop
    return dict(succ=clean, wrong=wrong, stopped=stopped, calls=calls, cost=calls * s.get('price', 1), ok=ok, stop=stop, att=att)


# ---------- the hash and one seeded task, ported ----------
M = 0xFFFFFFFF
def imul(a, b): return ((a & M) * (b & M)) & M
def mix(h):
    h &= M; h ^= h >> 16; h = imul(h, 0x7feb352d); h ^= h >> 15; h = imul(h, 0x846ca68b); h ^= h >> 16; return h & M
def u01(seed, t, j, a, k):
    h = mix(imul(seed, 0x9E3779B1) ^ 0x85ebca6b); h = mix(h ^ imul(t + 1, 0x27d4eb2f)); h = mix(h ^ imul(j + 1, 0x165667b1))
    h = mix(h ^ imul(a * 4 + k + 1, 0x2545f491)); return h / 4294967296


def sim_run(s, t):
    cells, p2, per = [], (s['pb'] if s['kind'] == 'b' else s['p']), (2 if s['ck'] else 1)
    doomed = stopped = False; calls = 0; end = s['n'] - 1
    for j in range(s['n']):
        if stopped: cells.append(('none', 0)); continue
        a = f = 0
        while True:
            calls += per
            if u01(s['seed'], t, j, a, 0) < (s['p'] if a == 0 else p2): res = 'ok'; break
            loud = u01(s['seed'], t, j, a, 1) < s['r'] or (bool(s['ck']) and u01(s['seed'], t, j, a, 2) < s['c'])
            if not loud: res = 'silent'; break
            f += 1
            if a >= s['R']: res = 'stop'; break
            a += 1
        if res == 'stop': stopped = True; end = j; cells.append(('stop', f))
        elif res == 'silent': cells.append(('pale' if doomed else 'silent', f)); doomed = True
        else: cells.append(('pale' if doomed else ('retry' if f else 'ok'), f))
    return dict(cells=cells, out='stop' if stopped else ('wrong' if doomed else 'ok'), calls=calls, end=end)


with sync_playwright() as pw:
    b = pw.chromium.launch()
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.evaluate("()=>{location.hash='agent-reliability'}"); pg.wait_for_timeout(500)
    H = lambda expr, *a: pg.evaluate(expr, *a)
    consts = H("()=>window.__arl.consts")
    check('hook: the chart runs 30 tasks, one illustration with no batches', consts['RUNS'] == RUNS and 'NB' not in consts and 'BATCH' not in consts, consts['RUNS'])
    check('hook: the default seed is 377', consts['BASE']['seed'] == 377, consts['BASE'])
    check('player: a whole run takes 6 / 12 / 24 s at fast / medium / slow', consts['SPD'] == {'f': 6, 'm': 12, 's': 24}, consts['SPD'])
    check('player: the default speed is fast, and a run at that speed takes 10 s or less (R3.9)', H("()=>window.__arl.state().speed") == 'f' and consts['SPD']['f'] <= 10)

    # ---- 1. the closed form against the chain, over a grid of setups ----
    grid = []
    for nn in (1, 5, 12, 20, 50):
        for p in (0.8, 0.9, 0.95, 0.99):
            for r in (0.0, 0.2, 0.7, 1.0):
                for R in (0, 1, 2, 3):
                    for ck in (0, 1):
                        for kind in ('i', 'b'):
                            grid.append(dict(BASE, n=nn, p=p, r=r, R=R, ck=ck, kind=kind))
    js = H("(g)=>g.map(s=>window.__arl.compute(s))", grid)
    bad = []
    for s, j in zip(grid, js):
        c = task_chain(s)
        for k in ('succ', 'wrong', 'stopped', 'calls'):
            if not close(j[k], c[k], 1e-9): bad.append((s, k, j[k], c[k]))
    check(f'closed form = Markov chain on {len(grid)} setups (success, wrong, stopped, calls)', not bad, bad[:2])
    check('success + wrong + stopped = 1', all(abs(j['succ'] + j['wrong'] + j['stopped'] - 1) < 1e-12 for j in js))

    # ---- 2. the seeded runs: replayed one by one; tallies near the formula ----
    setups = [dict(BASE), dict(BASE, R=2), dict(BASE, R=2, ck=1), dict(BASE, R=2, kind='b'), dict(BASE, n=50, p=0.97, R=2, ck=1),
              dict(BASE, n=15, r=0.2, R=2), dict(BASE, n=5, p=0.9, r=0.8, R=2), dict(BASE, seed=7, R=3, r=0.4)]
    for s in setups:
        js_runs = H("(s)=>Array.from({length:30},(_,t)=>window.__arl.simRun(s,t))", s)
        mism = []
        for t in range(RUNS):
            py = sim_run(s, t); jr = js_runs[t]
            jc = [(c['st'], c['f']) for c in jr['cells']]
            if jc != py['cells'] or jr['out'] != py['out'] or jr['calls'] != py['calls'] or jr['end'] != py['end']: mism.append(t)
        check(f'runs replayed cell by cell {s["n"]}/{s["p"]}/{s["r"]}/R{s["R"]}/ck{s["ck"]}/{s["kind"]}/seed{s["seed"]}', not mism, mism[:5])
        tl = H("(s)=>window.__arl.tally(s)", s); c = task_chain(s)
        for k, key in (('ok', 'succ'), ('wrong', 'wrong'), ('stop', 'stopped')):
            pr = c[key]; se = math.sqrt(max(pr * (1 - pr), 1e-12) / RUNS)
            check(f'tally {k} within 3.5 SE of the formula ({s["n"]} steps, R{s["R"]}, ck{s["ck"]})', abs(tl[k] / RUNS - pr) <= 3.5 * se + 1e-9, (tl[k], pr))
    # the default seed: the chart's 30 tasks come out typical of the formula in every guided part (within 2 tasks)
    for name, s in (('part 1', dict(BASE)), ('part 2 half', dict(BASE, n=10)), ('part 2 steady', dict(BASE, p=0.99)), ('part 3', dict(BASE, R=2)),
                    ('part 3 at 30% loud', dict(BASE, R=2, r=0.3)), ('part 4 check', dict(BASE, R=2, ck=1)), ('part 4 check, no retries', dict(BASE, ck=1)), ('part 5 same call', dict(BASE, R=2, kind='b'))):
        ok = sum(sim_run(s, t)['out'] == 'ok' for t in range(RUNS)); e = task_chain(s)['succ'] * RUNS
        check(f'seed 377, {name}: {ok} of 30 succeed, the formula gives {e:.1f}', abs(ok - e) <= 2, (ok, e))
    ok1 = sum(sim_run(dict(BASE), t)['out'] == 'ok' for t in range(RUNS))
    check('part 1: the chart’s share falls in the right range, 30 to 59 in 100', 30 <= ok1 / RUNS * 100 <= 59, ok1)
    # the hash itself at a few points
    pts = [(1, 0, 0, 0, 0), (1, 209, 49, 3, 2), (42, 17, 3, 1, 1), (9999, 5, 7, 2, 0)]
    jsu = H("(pts)=>pts.map(a=>window.__arl.u01(...a))", pts)
    check('hash ported: u01 equal at sample points', all(abs(a - u01(*p)) < 1e-15 for a, p in zip(jsu, pts)), (jsu, [u01(*p) for p in pts]))

    # ---- 3. the numbers the spec and the page quote ----
    want = {
        'part 1 / 2: 20 steps at 95%': (dict(BASE), 'succ', '35.8%'),
        'part 2: 10 steps at 95%': (dict(BASE, n=10), 'succ', '59.9%'),
        'part 2: 20 steps at 99%': (dict(BASE, p=0.99), 'succ', '81.8%'),
        'part 3: 70% loud, 2 retries': (dict(BASE, R=2), 'succ', '73.0%'),
        'part 4: check + 2 retries': (dict(BASE, R=2, ck=1), 'succ', '93.7%'),
        'part 4: check, no retries': (dict(BASE, R=0, ck=1), 'succ', '35.8%'),
        'part 5: same-call retries': (dict(BASE, R=2, kind='b'), 'succ', '58.6%'),
        'capability: 40 steps at 98%': (dict(BASE, n=40, p=0.98), 'succ', '44.6%'),
        'capability: 10 steps at 99%': (dict(BASE, n=10, p=0.99), 'succ', '90.4%'),
        'capability: 30 steps at 99.5%': (dict(BASE, n=30, p=0.995), 'succ', '86.0%'),
    }
    for name, (s, k, txt) in want.items():
        check(f'{name} = {txt}', fP(task_chain(s)[k]) == txt, fP(task_chain(s)[k]))
    calls = {'part 4: no check, 2 retries: 20.7 calls': (dict(BASE, R=2), '20.7'), 'part 4: check + 2 retries: 41.9 calls': (dict(BASE, R=2, ck=1), '41.9'),
             'part 4: check, no retries: 26.3 calls': (dict(BASE, ck=1), '26.3')}
    for name, (s, txt) in calls.items(): check(name, fC(task_chain(s)['calls']) == txt, fC(task_chain(s)['calls']))
    check('part 3: at 70% loud, the failures left are mostly silent (wrong > stopped)', task_chain(dict(BASE, R=2))['wrong'] > 5 * task_chain(dict(BASE, R=2))['stopped'])
    # part 4, a check without retries (B2): the check makes the failures known; it costs calls against part 1's agent
    n0, k0 = task_chain(dict(BASE)), task_chain(dict(BASE, ck=1))
    check('part 4 ck0: finished wrong 13.2% without the check, 2.3% with it', (fP(n0['wrong']), fP(k0['wrong'])) == ('13.2%', '2.3%'), (n0['wrong'], k0['wrong']))
    check('part 4 ck0: calls 14.6 without the check, 26.3 with it (the check adds calls)', (fC(n0['calls']), fC(k0['calls'])) == ('14.6', '26.3') and k0['calls'] > n0['calls'])
    check('part 4 ck0: the other failed tasks stop (stopped 51.0% -> 61.8%)', (fP(n0['stopped']), fP(k0['stopped'])) == ('51.0%', '61.8%'))
    # the retry ceiling (C13): unlimited retries at 70% loud leave about 73%
    cap = (0.95 / (1 - 0.05 * 0.7)) ** 20
    check('capability: with unlimited retries the success rate stays at about 73%', round(cap * 100) == 73 and task_chain(dict(BASE, R=3))['succ'] < cap, cap)
    check('part 1: the right range is 30 to 59 (formula 35.8 in 100)', consts['PRED_RIGHT'] == 'c' and 30 <= round(0.95 ** 20 * 100) <= 59)
    check('part 2: failed steps on average 1, 0.5, 0.2', (20 * 0.05, 10 * 0.05, round(20 * 0.01, 10)) == (1.0, 0.5, 0.2))
    check('capability: failed steps on average 0.8 (40 × 2%), A 0.1, B 0.15', (round(40 * 0.02, 10), round(10 * 0.01, 10), round(30 * 0.005, 10)) == (0.8, 0.1, 0.15))
    check('capability: the 40-step answer is “About 45%”', abs(0.98 ** 40 - 0.45) < 0.01)
    check('misconception: about 64 of 100 tasks fail', round((1 - 0.95 ** 20) * 100) == 64)
    check('capability: 30% of part 3’s failures are silent', abs(1 - BASE['r'] - 0.3) < 1e-12)

    # scenarios
    pre = {p['id']: p for p in consts['PRESETS']}
    for pid, s in pre.items():
        s = dict(BASE, **{k: v for k, v in s.items() if k != 'id'})
        c = task_chain(s); j = H("(s)=>window.__arl.compute(s)", s)
        check(f'scenario {pid}: {fP(c["succ"])}, {fC(c["calls"])} calls', fP(j['succ']) == fP(c['succ']) and fC(j['calls']) == fC(c['calls']))
    sc = {k: task_chain(dict(BASE, **{a: b for a, b in v.items() if a != 'id'})) for k, v in pre.items()}
    check('scenario research (15 steps) below code (20 steps): silent failures', sc['research']['succ'] < sc['code']['succ'])
    check('scenario short: the same assistant at 20 steps is at least 20 points lower', task_chain(dict(BASE, n=20, p=0.9, r=0.8, R=2))['succ'] < sc['short']['succ'] - 0.2)
    check('scenario long: about twice the calls of 50 steps', 1.9 * 50 < sc['long']['calls'] < 2.3 * 50, sc['long']['calls'])

    # ---- 4. the challenge ----
    CH = consts['CH']
    def ch_eval(strong, ck, R, nn):
        s = dict(BASE, n=nn, p=CH['pS'] if strong else CH['p'], r=CH['r'], R=R, ck=ck, c=CH['c'], kind='i', price=2 if strong else 1)
        c = task_chain(s); return dict(strong=strong, ck=ck, R=R, n=nn, succ=c['succ'], cost=c['cost'], ok=c['succ'] >= CH['goal'] and c['cost'] <= CH['budget'] + 1e-9)
    allc = [ch_eval(st, ck, R, nn) for st in (0, 1) for ck in (0, 1) for R in CH['Rs'] for nn in CH['ns']]
    passing = [e for e in allc if e['ok']]
    check('challenge: 32 combinations', len(allc) == 32)
    check('challenge: exactly 3 pass, all 95% model + check + 12 steps', len(passing) == 3 and all(e['strong'] == 0 and e['ck'] == 1 and e['n'] == 12 for e in passing), passing)
    best = sorted(passing, key=lambda e: (round(e['cost'], 1), e['R']))[0]
    check('challenge: minimal = 1 retry, 24.8 calls, 91.6%', (best['R'], fC(best['cost']), fP(best['succ'])) == (1, '24.8', '91.6%'), best)
    jb = H("()=>window.__arl.chBest()")
    check('challenge: the page agrees on the minimal one', (jb['R'], fC(jb['cost'])) == (1, '24.8'), jb)
    ja = H("()=>window.__arl.chAll()")
    check('challenge: the page agrees on every combination', all(fP(a['succ']) == fP(x['succ']) and fC(a['cost']) == fC(x['cost']) and a['ok'] == x['ok'] for a, x in zip(ja, allc)))
    check('challenge: the stronger model never passes', not any(e['ok'] for e in allc if e['strong']))
    st0 = ch_eval(0, 0, 0, 20)
    check('challenge: the initial setup misses (35.8%, 15.9 calls)', (fP(st0['succ']), fC(st0['cost'])) == ('35.8%', '15.9'), st0)
    r3 = ch_eval(0, 0, 3, 20)
    check('challenge: retries alone stop below 90%', r3['succ'] < 0.9 and max(e['succ'] for e in allc if e['ck'] == 0 and e['strong'] == 0) < 0.9)
    ck20 = ch_eval(0, 1, 1, 20); s12 = max((ch_eval(1, 0, R, 12) for R in (1, 2, 3)), key=lambda e: e['succ'])
    check('challenge: a check with retries at 20 steps costs at least the 1-retry figure', all(ch_eval(0, 1, R, 20)['cost'] >= ck20['cost'] for R in (1, 2, 3)))
    check('challenge: a check at 20 steps is over the budget', ck20['cost'] > 40, ck20['cost'])
    check('challenge: the 98% model at 12 steps (no check) stays below 90%', s12['succ'] < 0.9, s12['succ'])
    check('challenge: the 98% model at 20 steps with a check is over the budget', all(ch_eval(1, 1, R, 20)['cost'] > 40 for R in CH['Rs']))
    # S3: at 12 steps the 98% model without a check stays below 90%, and with a check (any retries) it is over the budget
    check('challenge: the 98% model at 12 steps with a check is over the budget for every retry count', all(ch_eval(1, 1, R, 12)['cost'] > 40 for R in CH['Rs']), [ch_eval(1, 1, R, 12)['cost'] for R in CH['Rs']])
    check('challenge: the 98% model at 12 steps without a check reaches 88.5% at most', fP(max(ch_eval(1, 0, R, 12)['succ'] for R in CH['Rs'])) == '88.5%')
    # B1: with no retries and no check, most failures are loud stops (either model, 20 or 12 steps); with retries, the 95% model's
    # failures are almost all silent, as the second hint says
    def ends(strong, ck, R, nn):
        c = task_chain(dict(BASE, n=nn, p=CH['pS'] if strong else CH['p'], r=CH['r'], R=R, ck=ck, c=CH['c'])); return c['wrong'], c['stopped']
    check('challenge hint, no retries: stopped > finished wrong', all(ends(st, 0, 0, nn)[1] > ends(st, 0, 0, nn)[0] for st in (0, 1) for nn in (20, 12)), [ends(st, 0, 0, nn) for st in (0, 1) for nn in (20, 12)])
    check('challenge hint, the initial setup: 24.4% wrong, 39.7% stopped', tuple(fP(v) for v in ends(0, 0, 0, 20)) == ('24.4%', '39.7%'))
    check('challenge hint, retries without a check: almost every failure left is silent', all(ends(0, 0, R, nn)[0] > 20 * ends(0, 0, R, nn)[1] for R in (1, 2, 3) for nn in (20, 12)))

    # ---- 5. the drill ----
    for sd in (1, 2, 3, 7, 20):
        d = H(f"()=>window.__arl.drill({sd})")
        c = task_chain(dict(BASE, n=d['n'], p=d['p'], r=d['r'], R=d['R']))
        check(f'drill {sd}: {d["n"]} steps, {d["p"]}, {d["r"]} loud, R{d["R"]}: {fP(c["succ"])}', fP(d['res']['succ']) == fP(c['succ']) and fP(d['res']['plain']) == fP(d['p'] ** d['n']))

    # ---- 6. the Python in Connect: run it ----
    code = consts['CODE']
    check('Connect code: at most 30 lines, at most 74 characters wide', len(code.split('\n')) <= 30 and max(len(l) for l in code.split('\n')) <= 74, max(len(l) for l in code.split('\n')))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): exec(code.replace('__SIM__', ''), {})
    lines = buf.getvalue().split()
    comments = [m.group(1) for m in re.finditer(r'#\s*([0-9.]+|__SIM__)\s*$', code, re.M)]
    check('Connect code prints three lines', len(lines) == 3, lines)
    check('Connect code: printed values match its comments', lines[:2] == comments[:2], (lines, comments))
    check('Connect code: the simulated share matches SIMOUT', lines[2] == consts['SIMOUT'], (lines[2], consts['SIMOUT']))
    check('Connect code: the simulation is close to the formula (73.0%)', abs(float(lines[2]) - task_chain(dict(BASE, R=2))['succ']) < 0.015)

    # ---- 7. the teaser ----
    tz = H("()=>!!window.__tzArl")
    check('teaser hook present', tz)
    if tz:
        for k in (1, 10, 20, 40):
            v = H(f"()=>window.__tzArl.succ({k})")
            check(f'teaser: {k} steps at 90% = {fP(0.9 ** k)}', close(v, 0.9 ** k) and fP(0.9 ** k) in H(f"()=>window.__tzArl.read({k})"))
        check('teaser: 90% per step, 1 to 40 steps', H("()=>[window.__tzArl.P,window.__tzArl.NMAX]") == [0.9, 40])

    # ---- 8. the page's text shows these numbers (EN) ----
    def obs(): return H("()=>{const o=document.querySelector('#arl-rail [data-obs]'); return o&&!o.hidden?o.innerText:''}")
    def go(i): H(f"()=>document.querySelector('#arl-rail [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(200)
    H("()=>document.querySelector('#arl-rail [data-pred=\"c\"]').click()"); pg.wait_for_timeout(100)
    H("()=>document.querySelector('#arl-player [data-act=\"end\"]').click()"); pg.wait_for_timeout(200)
    o = obs(); t = H("()=>window.__arl.tally({})")
    check('page part 1: the tally and the formula', f'{t["ok"]} of the 30 tasks in the chart succeeded' in o and '35.8%' in o and 'about 36 in 100' in o, o)
    go(1)
    for v, txt in (('half', '59.9%'), ('steady', '81.8%')):
        H(f"()=>document.querySelector('#arl-rail [data-lever=\"{v}\"]').click()"); pg.wait_for_timeout(150)
        check(f'page part 2 {v}: {txt}', txt in obs() and '35.8%' in obs(), obs())
    go(2); H("()=>document.querySelector('#arl-rail .seg-ctl[data-key=\"R\"] button[data-v=\"2\"]').click()"); pg.wait_for_timeout(150)
    check('page part 3: 35.8% to 73.0%', '35.8%' in obs() and '73.0%' in obs(), obs())
    go(3)
    for v, txts in (('ck', ('93.7%', '41.9', '20.7')), ('ck0', ('35.8%', fP(n0['wrong']), fP(k0['wrong']), fC(n0['calls']), fC(k0['calls'])))):
        H(f"()=>document.querySelector('#arl-rail [data-lever=\"{v}\"]').click()"); pg.wait_for_timeout(150)
        check(f'page part 4 {v}: {txts}', all(x in obs() for x in txts), obs())
    # Show the math (S10): the stop, the stopped share and the wrong share, one line each, with the numbers in
    mt = H("()=>document.querySelector('#arl-mathbody').textContent")
    st1 = 0.05 * (0.7 + 0.3 * 0.8)
    check('page math part 4 ck0: P_stop, 1 − (1 − P_stop)^n, (1 − P_stop)^n − P_task', f'{st1:.6f}' in mt and f'{1 - st1:.6f}' in mt and fP(k0['stopped']) in mt and fP(k0['wrong']) in mt and f'{k0["succ"]:.6f}' in mt, mt)
    go(4); H("()=>document.querySelector('#arl-rail [data-lever=\"blind\"]').click()"); pg.wait_for_timeout(150)
    check('page part 5: 73.0% to 58.6%', '73.0%' in obs() and '58.6%' in obs(), obs())
    go(5); rem = H("()=>document.querySelector('#arl-rail .step.current').innerText")
    # the owner, 2026-10-08 (G2): the three points state the pattern, not the numbers of parts 2 to 4; the first misconception keeps 35.8%
    check('page part 6 (G2): the points state the pattern, not the numbers of parts 2 to 4', '35.8%' in rem and 'In part' not in rem
          and not any(x in rem for x in ('81.8%', '59.9%', '73.0%', '93.7%', '41.9', '20.7')) and 'the silent failures stay, and they cap the success rate' in rem, rem)
    # metrics and the math of the current setup (part 6 shows part 3's agent with 2 retries)
    met = H("()=>[...document.querySelectorAll('#arl-metrics .metric .v')].map(e=>e.textContent)")
    c = task_chain(dict(BASE, R=2))
    check('page metrics: success, wrong, stopped, calls', met == [fP(c['succ']), fP(c['wrong']), fP(c['stopped']), fC(c['calls'])], (met, c))
    mt = H("()=>document.querySelector('#arl-mathbody').textContent")
    check('page math: rc 0.7000, P_step, the result, the calls', '0.7000' in mt and f'{c["ok"]:.6f}' in mt and fP(c['succ']) in mt and fC(c['calls']) in mt, mt)
    # scenarios
    H("()=>document.querySelector('#arl-app .modes button[data-mode=\"sandbox\"]:not([data-flow=free])').click()"); pg.wait_for_timeout(300)
    for i, pid in enumerate(['code', 'research', 'long', 'short']):
        H(f"()=>document.querySelector('#arl-rail [data-preset=\"{i}\"]').click()"); pg.wait_for_timeout(150)
        ins = H("()=>document.querySelector('#arl-insight').innerText"); c = sc[pid]
        extra = [fC(c['calls'])] if pid == 'long' else ([fP(task_chain(dict(BASE, n=20, p=0.9, r=0.8, R=2))['succ'])] if pid == 'short' else [])
        check(f'page scenario {pid}: {fP(c["succ"])}', fP(c['succ']) in ins and all(x in ins for x in extra), ins)
    # the challenge: the minimal pick and the full chart
    H("()=>document.querySelector('#arl-app .modes button[data-mode=\"challenge\"]').click()"); pg.wait_for_timeout(300)
    for k, v in (('chC', 1), ('chR', 1), ('chN', 12)):
        H(f"()=>document.querySelector('#arl-rail .seg-ctl[data-key=\"{k}\"] button[data-v=\"{v}\"]').click()"); pg.wait_for_timeout(120)
    H("()=>document.querySelector('#arl-rail [data-act=\"submit\"]').click()"); pg.wait_for_timeout(250)
    res = H("()=>document.querySelector('#arl-ch-res').innerText")
    check('page challenge: the minimal pick, 91.6% and 24.8 calls', '91.6%' in res and '24.8' in res and 'no passing combination costs less' in res, res)
    # the brief chart (the owner's review of batch 8): the failing alternatives the text names are in words, their numbers in hovers
    tips = H("()=>[...document.querySelectorAll('#arl-ch-res .def')].map(e=>e.getAttribute('data-tipx')).join(' | ')")
    check('page challenge: the chart text (35.8% in the text; retries alone, a check at 20 steps, the 98% model at 12 steps in hovers)', fP(st0['succ']) in res and fP(r3['succ']) in tips and fC(ck20['cost']) in tips and fP(s12['succ']) in tips, (res, tips))
    check('page challenge: 10 rows at most in the chart', H("()=>document.querySelectorAll('#arl-ch-res .cmp-row').length") <= 10)
    labels = H("()=>[...document.querySelectorAll('#arl-ch-res .cmp:not(details .cmp) .cmp-row')].map(r=>r.innerText.replace(/\\s+/g,' '))")
    want = [e for e in allc if e['ok'] or (e['strong'], e['ck'], e['R'], e['n']) == (0, 0, 0, 20)]
    check('page challenge: the chart holds the 3 passing combinations and the initial setup, in order, with their numbers', len(labels) == len(want) == 4
          and all(fP(e['succ']) in l and fC(e['cost']) in l for e, l in zip(want, labels)), (labels, [(fP(e['succ']), fC(e['cost'])) for e in want]))
    note = H("()=>{const n=document.querySelector('#arl-ch-res .cmp-note'); return n?n.innerText:''}")
    check('page challenge: the note counts the 28 combinations left out', 'The other 28 combinations miss at least one target' in note, note)
    check('no page errors', not errs, errs[:3])
    b.close()

print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
