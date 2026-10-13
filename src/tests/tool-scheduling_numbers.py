"""Numbers test for the tool-scheduling lab: an independent Python scheduler (an event loop over each batch, written from the rules in
specs/tool-scheduling.md, not ported from the JS) recomputes every schedule the page can show and compares it with window.__tls.
- every trip setup: 3 ways of sending × lock × k ∈ {1,2,3,none} × timeout {none, hotel search, flight booking} × key × model call
  {0.5, 1.5, 3} s, plus the part-4 “twice as fast” cases: total time, model calls, tool time added up, what goes wrong, the critical path;
- the named numbers of the copy (27.2, 6.0, 14.3, 16.3, 15.8, 14.8, 19.7, 16.8, 20.8, 21.3, 23.8, 42.2 s);
- the challenge: 4,683 distinct arrangements, which are right, which meet 14 s, the unique best (11.9 s), the initial setup (17.1 s);
- the teaser (19.1 s and 10.0 s) and the interview drill for seeds 1–12;
- the page's own text: the observations and the challenge result name the same numbers.
Run: python3 tests/tool-scheduling_numbers.py"""
import itertools, pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
LAB = 'tool-scheduling'
subprocess.run([sys.executable, str(P / 'build.py'), f'out/{LAB}.html', '--only', LAB], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / LAB).mkdir(parents=True, exist_ok=True)
(P / 'out' / LAB / 'num-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"></head><body>' + (P / 'out' / f'{LAB}.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / LAB / 'num-wrapped.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


# ---- the lab's setting, written down again from the spec ----
TO = 5.0
# name, seconds, needs (indices), after (indices), the thing it writes
TRIP = [('cal', .4, [], [], None), ('sfl', 2.5, [], [], None), ('sho', 2.0, [], [], None), ('wea', .6, [], [], None),
        ('cmp', .5, [1, 2], [], None), ('bfl', 3.0, [0, 4], [], 'sheet'), ('bho', 2.0, [4], [], 'sheet'),
        ('evt', .4, [5, 6], [], None), ('eml', .8, [3, 5, 6], [], None)]
CODE = [('A', .3, [], [], None), ('B', .3, [], [], None), ('C', 1.2, [], [], None), ('D', .4, [0, 2], [], None),
        ('E', .4, [1], [], None), ('F', 4.0, [], [3, 4], None)]


def depth(task):
    d = {}
    for i, c in enumerate(task):
        d[i] = 1 + max([d[j] for j in c[2] + c[3]], default=0)
    return [[i for i in d if d[i] == L] for L in range(1, max(d.values()) + 1)]


def batches(task, how):
    idx = list(range(len(task)))
    return [[i] for i in idx] if how == 'seq' else ([idx] if how == 'all' else depth(task))


def simulate(task, turns, k=0, lock=1, m=1.5, fail=-1, half=-1, idem=1):
    """time runs in a loop of events: at each moment start, in written order, every waiting call that has a free slot and,
    with the lock, no running or earlier waiting writer of the same thing; then jump to the next finish."""
    start, end, turn_of, T = {}, {}, {}, []
    clock = 0.0
    for j, calls in enumerate(turns):
        t0 = round(clock + m, 6); now = t0; waiting = list(calls); running = []
        while waiting or running:
            for i in list(waiting):
                if k and len(running) >= k: break
                w = task[i][4]
                if lock and w and (any(task[r][4] == w for r in running) or any(task[q][4] == w for q in waiting[:waiting.index(i)])):
                    continue
                d = (task[i][1] / 2 if half == i else task[i][1]) + (TO if fail == i else 0)
                start[i], end[i], turn_of[i] = now, round(now + d, 6), j
                running.append(i); waiting.remove(i)
            if not running: break
            now = min(end[r] for r in running)
            running = [r for r in running if end[r] > now + 1e-9]
        fin = max([t0] + [end[i] for i in calls]); T.append((clock, t0, fin, calls)); clock = fin
    total = round(clock + m, 6)
    wrong = []
    for i, c in enumerate(task):
        if i not in start: continue
        if any(not (turn_of.get(j, 99) < turn_of[i]) for j in c[2]): wrong.append((i, 'miss'))
        if any(not (turn_of.get(j, 99) < turn_of[i]) for j in c[3]): wrong.append((i, 'early'))
    for a, b in itertools.combinations(range(len(task)), 2):
        if a in start and b in start and task[a][4] and task[a][4] == task[b][4] and start[a] < end[b] - 1e-9 and start[b] < end[a] - 1e-9:
            wrong.append((b if end[b] > end[a] else a, 'lost'))
    if fail >= 0 and task[fail][4] and not idem: wrong.append((fail, 'double'))
    crit = set()
    for (ms, me, fin, calls) in T:
        cur = max(calls, key=lambda q: (end[q], -calls.index(q)))
        while True:
            crit.add(cur)
            if start[cur] <= me + 1e-9: break
            prev = [q for q in calls if q != cur and abs(end[q] - start[cur]) < 1e-9]
            if not prev: break
            cur = prev[0]
    return dict(total=total, n=len(turns) + 1, tools=round(sum(end[i] - start[i] for i in start), 6), wrong=sorted(wrong), crit=sorted(crit))


r1 = lambda v: round(v + 1e-9, 1)
NAMED = {('seq', 1, 0, 0, 1, 1.5, -1): 27.2, ('all', 0, 0, 0, 1, 1.5, -1): 6.0, ('dep', 0, 0, 0, 1, 1.5, -1): 14.3, ('dep', 1, 0, 0, 1, 1.5, -1): 16.3,
         ('dep', 1, 0, 0, 1, 1.5, 3): 16.3, ('dep', 1, 0, 0, 1, 1.5, 1): 15.8, ('dep', 1, 0, 0, 1, 1.5, 5): 14.8, ('dep', 1, 1, 0, 1, 1.5, -1): 19.7,
         ('dep', 1, 2, 0, 1, 1.5, -1): 16.8, ('dep', 1, 3, 0, 1, 1.5, -1): 16.3, ('dep', 1, 0, 1, 1, 1.5, -1): 20.8, ('dep', 1, 0, 2, 1, 1.5, -1): 21.3,
         ('dep', 1, 0, 2, 0, 1.5, -1): 21.3, ('dep', 1, 0, 0, 1, 3, -1): 23.8, ('seq', 1, 0, 0, 1, 3, -1): 42.2, ('seq', 1, 0, 0, 1, .5, -1): 17.2,
         ('dep', 1, 0, 0, 1, .5, -1): 11.3}
FAILI = [-1, 2, 5]


def mulberry(a):
    def rnd():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xffffffff
        t = a
        t = ((t ^ (t >> 15)) * (1 | t)) & 0xffffffff
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xffffffff)) & 0xffffffff) ^ t
        return ((t ^ (t >> 14)) & 0xffffffff) / 4294967296
    return rnd


def drill(seed):
    rnd = mulberry((seed * 7919 + 13) & 0xffffffff)
    m = [1, 1.5, 2][int(rnd() * 3)]; nb = 2 + int(rnd() * 2); bs = []
    for _ in range(nb):
        c = 1 + int(rnd() * 3); bs.append([(1 + int(rnd() * 30)) / 10 for _ in range(c)])
    allc = [x for b in bs for x in b]
    return m, bs, round((nb + 1) * m + sum(max(b) for b in bs), 3), round((len(allc) + 1) * m + sum(allc), 3)


with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300); pg.evaluate("()=>{location.hash='tool-scheduling'}"); pg.wait_for_timeout(500)
    D = pg.evaluate("()=>window.__tls.data()")
    check('data: trip times', [c['t'] for c in D['TRIP']['calls']] == [c[1] for c in TRIP], D['TRIP']['calls'])
    check('data: trip needs', [c['needs'] for c in D['TRIP']['calls']] == [c[2] for c in TRIP])
    check('data: code times and needs', [c['t'] for c in D['CODE']['calls']] == [c[1] for c in CODE] and [c['needs'] + c.get('after', []) for c in D['CODE']['calls']] == [c[2] + c[3] for c in CODE])
    check('data: timeout, goal', D['TO'] == TO and D['CH_GOAL'] == 14 and D['CH_K'] == 2 and D['CH_M'] == 1.5, D)
    check('levels', pg.evaluate("()=>window.__tls.levels('trip')") == depth(TRIP))
    # every trip setup
    cases = []
    for how, lock, k, fl, idem, m in itertools.product(['seq', 'all', 'dep'], [0, 1], [1, 2, 3, 0], [0, 1, 2], [0, 1], [.5, 1.5, 3]):
        cases.append((how, lock, k, fl, idem, m, -1))
    for h in (3, 1, 5, 8): cases.append(('dep', 1, 0, 0, 1, 1.5, h))
    js = pg.evaluate("cs=>cs.map(c=>window.__tls.run({sch:c[0],lock:c[1],k:c[2],fail:c[3],idem:c[4],m:c[5],half:c[6]}))", cases)
    bad_tot = bad_n = bad_w = bad_c = bad_t = 0
    for c, R in zip(cases, js):
        py = simulate(TRIP, batches(TRIP, c[0]), k=c[2], lock=c[1], m=c[5], fail=FAILI[c[3]], half=c[6], idem=c[4])
        if abs(py['total'] - R['total']) > 1e-6: bad_tot += 1; print('total', c, py['total'], R['total'])
        if py['n'] != R['n']: bad_n += 1
        if abs(py['tools'] - R['toolSum']) > 1e-6: bad_t += 1
        if py['wrong'] != sorted((x['i'], x['kind']) for x in R['bad']): bad_w += 1; print('wrong', c, py['wrong'], R['bad'])
        if py['crit'] != R['crit']: bad_c += 1; print('crit', c, py['crit'], R['crit'])
        if c in NAMED: check(f'named total {c}', r1(py['total']) == NAMED[c], (py['total'], NAMED[c]))
    check(f'{len(cases)} trip setups: total time', bad_tot == 0, bad_tot)
    check('trip setups: model calls', bad_n == 0, bad_n)
    check('trip setups: tool time added up', bad_t == 0, bad_t)
    check('trip setups: what goes wrong', bad_w == 0, bad_w)
    check('trip setups: critical path', bad_c == 0, bad_c)
    check('one at a time: 10 model calls, 12.2 s of tools', simulate(TRIP, batches(TRIP, 'seq'))['n'] == 10 and r1(simulate(TRIP, batches(TRIP, 'seq'))['tools']) == 12.2)
    a = simulate(TRIP, batches(TRIP, 'all'), lock=0)
    check('all at once: five calls without their inputs, one lost line', sum(1 for w in a['wrong'] if w[1] == 'miss') == 5 and any(w[1] == 'lost' for w in a['wrong']), a['wrong'])
    d = simulate(TRIP, batches(TRIP, 'dep'))
    check('by dependency: 5 model calls, critical path flights, compare, both bookings, email', d['n'] == 5 and d['crit'] == [1, 4, 5, 6, 8], d)
    # the challenge
    seen = {}
    for asg in itertools.product(range(1, 7), repeat=6):
        used = sorted(set(asg)); turns = [[i for i in range(6) if asg[i] == u] for u in used]
        key = '|'.join(''.join(map(str, t)) for t in turns)
        if key in seen: continue
        R = simulate(CODE, turns, k=2, lock=1, m=1.5)
        seen[key] = (R['total'], R['n'], not R['wrong'], not R['wrong'] and R['total'] <= 14 + 1e-9)
    jsch = pg.evaluate("()=>window.__tls.chAll()")
    check('challenge: 4,683 distinct arrangements', len(seen) == 4683 and len(jsch) == 4683, (len(seen), len(jsch)))
    mism = [x['key'] for x in jsch if x['key'] not in seen or abs(seen[x['key']][0] - x['total']) > 1e-6 or seen[x['key']][2] != x['right'] or seen[x['key']][3] != x['ok']]
    check('challenge: every arrangement agrees', not mism, mism[:5])
    okk = sorted((v[0], k) for k, v in seen.items() if v[3])
    check('challenge: 13 meet the goal, 63 are right', len(okk) == 13 and sum(1 for v in seen.values() if v[2]) == 63, (len(okk),))
    check('challenge: the unique best is 012|34|5 at 11.9 s', okk[0] == (11.9, '012|34|5') and okk[1][0] > 11.9 + 1e-9, okk[:2])
    check('challenge: the best is the one flagged best', [x['key'] for x in jsch if x['best']] == ['012|34|5'])
    check('challenge: initial setup 17.1 s, right but too slow', r1(seen['0|1|2|3|4|5'][0]) == 17.1 and seen['0|1|2|3|4|5'][2] and not seen['0|1|2|3|4|5'][3])
    check('challenge: 13.4 s arrangement 0|12|34|5', r1(seen['0|12|34|5'][0]) == 13.4 and seen['0|12|34|5'][3])
    check('challenge: without the limit the best takes 11.6 s', r1(simulate(CODE, [[0, 1, 2], [3, 4], [5]], k=0)['total']) == 11.6)
    check('challenge: starting C first under the limit also takes 11.6 s (the page says a real harness could)', r1(simulate(CODE, [[2, 0, 1], [3, 4], [5]], k=2, lock=1, m=1.5)['total']) == 11.6)
    # slowest first in every batch: the same 13 arrangements pass (review S8), so the start order does not change which ones pass
    def slow_first(t): return sorted(t, key=lambda i: -CODE[i][1])
    pass_sf = set(k for k, v in seen.items() if (lambda R: not R['wrong'] and R['total'] <= 14 + 1e-9)(simulate(CODE, [slow_first([int(c) for c in g]) for g in k.split('|')], k=2, lock=1, m=1.5)))
    check('challenge: slowest-first order passes the same 13 arrangements', pass_sf == set(k for k, v in seen.items() if v[3]), len(pass_sf))
    # the teaser: five calls, one at a time against by dependency
    T5 = [('w', .6, [], [], None), ('f', 2.5, [], [], None), ('h', 2.0, [], [], None), ('bf', 3.0, [1], [], None), ('bh', 2.0, [2], [], None)]
    check('teaser: 19.1 s one at a time, 10.0 s by dependency', r1(simulate(T5, batches(T5, 'seq'))['total']) == 19.1 and r1(simulate(T5, batches(T5, 'dep'))['total']) == 10.0)
    # the drill
    dj = pg.evaluate("()=>[...Array(12).keys()].map(i=>window.__tls.drill(i+1))")
    check('drill seeds 1-12', all(abs(drill(i + 1)[2] - x['total']) < 1e-6 and abs(drill(i + 1)[3] - x['seqT']) < 1e-6 and drill(i + 1)[0] == x['m'] for i, x in enumerate(dj)), [(drill(i + 1), x) for i, x in enumerate(dj)][:2])
    # the page's own text
    for lang in ('en', 'zh'):
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(200); pg.evaluate("()=>{location.hash='tool-scheduling~dep_1_0_0_1_1p5'}"); pg.wait_for_timeout(500)
        for i, want in ((0, ['16.3', '5']), (1, ['27.2']), (2, ['16.8']), (3, ['23.8', '42.2']), (4, ['21.3'])):
            pg.evaluate(f"()=>document.querySelector('#tls-rail [data-preset=\"{i}\"]').click()"); pg.wait_for_timeout(150)
            t = pg.evaluate("()=>document.querySelector('#tls-insight').innerText")
            check(f'{lang} scenario {i} names {want}', all(x in t for x in want), t[:200])
    b.close()
for e in errs: print('ERR', e)
print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails or errs else 0)
