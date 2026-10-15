"""Numbers test for the tool-scheduling lab: an independent Python scheduler (an event loop over each batch, written from the rules in
specs/tool-scheduling.md, not ported from the JS) recomputes every schedule the page can show and compares it with window.__tls.
- every trip setup: 3 ways of sending × lock × k ∈ {1,2,3,none} × timeout {none, hotel search, flight booking} × key × model call
  {0.5, 1.5, 3} s, plus the part-4 “twice as fast” cases: total time, model calls, tool time added up, what goes wrong, the critical path;
- the named numbers of the copy (27.2, 6.0, 14.3, 16.3, 15.8, 14.8, 19.7, 16.8, 20.8, 21.3, 23.8, 42.2 s);
- the challenge: 4,683 distinct arrangements, which are right, which meet 14 s, the unique best (12.7 s), the initial setup (17.9 s);
- the tests starting too early (polish r1, P1): in every arrangement the page flags the tests exactly when the timeline starts them
  before an edit they read has finished, and names exactly the edits still running; ABC → DEF says so of E, not D, in both languages;
  among those, the edits that had not even begun (polish r2, N10) are told apart: ABCF → DE says D and E “had not begun”, ABC → DF → E
  says D was still editing and E had not begun;
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
        ('E', 1.2, [1], [], None), ('F', 4.0, [], [3, 4], None)]


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
    wrong, early_by, not_begun = [], {}, {}
    for i, c in enumerate(task):
        if i not in start: continue
        if any(not (turn_of.get(j, 99) < turn_of[i]) for j in c[2]): wrong.append((i, 'miss'))
        # a call that must see another's changes is wrong when it starts before that call has finished (by the clock, as drawn)
        late = [j for j in c[3] if not (j in end and end[j] <= start[i] + 1e-9)]
        if late: wrong.append((i, 'early')); early_by[i] = late; not_begun[i] = [j for j in late if j not in start or start[j] > start[i] + 1e-9]
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
    return dict(total=total, n=len(turns) + 1, tools=round(sum(end[i] - start[i] for i in start), 6), wrong=sorted(wrong), crit=sorted(crit),
                start=start, end=end, turn_of=turn_of, early_by=early_by, not_begun=not_begun)


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
    check('challenge: 3 meet the goal, 63 are right', len(okk) == 3 and sum(1 for v in seen.values() if v[2]) == 63, (len(okk), okk))
    check('challenge: the three that pass are 012|34|5 12.7 s, 01|24|3|5 13.4 s, 1|024|3|5 13.7 s',
          [(r1(t), k) for t, k in okk] == [(12.7, '012|34|5'), (13.4, '01|24|3|5'), (13.7, '1|024|3|5')], okk)
    check('challenge: the unique best is 012|34|5 at 12.7 s', okk[0] == (12.7, '012|34|5') and okk[1][0] > 12.7 + 1e-9, okk[:2])
    check('challenge: the best is the one flagged best', [x['key'] for x in jsch if x['best']] == ['012|34|5'])
    check('challenge: initial setup 17.9 s, right but too slow', r1(seen['0|1|2|3|4|5'][0]) == 17.9 and seen['0|1|2|3|4|5'][2] and not seen['0|1|2|3|4|5'][3])
    check('challenge: 13.4 s arrangement 01|24|3|5 (the middle tier the flow test submits)', r1(seen['01|24|3|5'][0]) == 13.4 and seen['01|24|3|5'][3])
    check('challenge: without the limit the best takes 12.4 s', r1(simulate(CODE, [[0, 1, 2], [3, 4], [5]], k=0)['total']) == 12.4)
    check('challenge: starting C first under the limit also takes 12.4 s (the page says a real harness could)', r1(simulate(CODE, [[2, 0, 1], [3, 4], [5]], k=2, lock=1, m=1.5)['total']) == 12.4)
    # slowest first in every batch: the same 3 arrangements pass (review S8), so the start order does not change which ones pass
    def slow_first(t): return sorted(t, key=lambda i: -CODE[i][1])
    pass_sf = set(k for k, v in seen.items() if (lambda R: not R['wrong'] and R['total'] <= 14 + 1e-9)(simulate(CODE, [slow_first([int(c) for c in g]) for g in k.split('|')], k=2, lock=1, m=1.5)))
    check('challenge: slowest-first order passes the same 3 arrangements', pass_sf == set(k for k, v in seen.items() if v[3]), len(pass_sf))
    # P1 (polish r1): the verdict, the timeline and the card's rule agree. The JS schedule of every arrangement: each “started too
    # early” names exactly the edits whose bar is still running when the tests' bar starts, and every such overlap is flagged.
    keys = sorted(seen)
    jsr = pg.evaluate("ks=>ks.map(k=>{const R=window.__tls.sched('code',k.split('|').map(g=>[...g].map(Number)),{k:2,lock:1,m:1.5,fail:-1,half:-1,idem:1}); return {cs:R.cs,bad:R.bad};})", keys)
    bad_by, bad_wait, kinds = [], [], set()
    for k, R in zip(keys, jsr):
        cs = R['cs']; F = 5
        overlap = sorted(j for j in CODE[F][3] if cs[F]['s'] < cs[j]['e'] - 1e-9)
        flagged = sorted(j for x in R['bad'] if x['kind'] == 'early' and x['i'] == F for j in x['by'])
        py = simulate(CODE, [[int(c) for c in g] for g in k.split('|')], k=2, lock=1, m=1.5)
        if overlap != flagged or sorted(py['early_by'].get(F, [])) != flagged: bad_by.append((k, overlap, flagged))
        # N10: of those, the ones whose bar starts after the tests' bar has started had not begun
        wait = sorted(j for x in R['bad'] if x['kind'] == 'early' and x['i'] == F for j in x.get('wait', []))
        later = sorted(j for j in flagged if cs[j]['s'] > cs[F]['s'] + 1e-9)
        if wait != later or sorted(py['not_begun'].get(F, [])) != wait: bad_wait.append((k, later, wait))
        if flagged: kinds.add('not begun' if wait == flagged else 'still editing' if not wait else 'both')
    check('challenge P1: “started too early” names exactly the edits still running when the tests start, in all 4,683', not bad_by, bad_by[:5])
    check('challenge N10: the edits told “had not begun” are exactly those whose bar starts after the tests start, in all 4,683', not bad_wait, bad_wait[:5])
    check('challenge N10: all three wordings occur (not begun, still editing, both)', kinds == {'not begun', 'still editing', 'both'}, kinds)
    f1 = simulate(CODE, [[0, 1, 2, 5], [3, 4]], k=2, lock=1, m=1.5); mx = simulate(CODE, [[0, 1, 2], [3, 5], [4]], k=2, lock=1, m=1.5)
    check('challenge N10: ABCF → DE, the tests start at 1.8 s and both edits at 7.3 s; ABC → DF → E, D runs with the tests, E later',
          r1(f1['start'][5]) == 1.8 and r1(f1['start'][3]) == r1(f1['start'][4]) == 7.3 and f1['not_begun'] == {5: [3, 4]}
          and mx['early_by'] == {5: [3, 4]} and mx['not_begun'] == {5: [4]} and abs(mx['start'][3] - mx['start'][5]) < 1e-9, (f1['not_begun'], mx['not_begun']))
    # the card's rule (F starts after D and E are done) picks the same right arrangements as “F in a later model call than D and E”,
    # under letter order and slowest first, so either reading of the rule gives the same answers
    def turn_rule_right(k, order):
        t = [order([int(c) for c in g]) for g in k.split('|')]; R = simulate(CODE, t, k=2, lock=1, m=1.5)
        return not [w for w in R['wrong'] if w[1] != 'early'] and all(R['turn_of'][j] < R['turn_of'][5] for j in CODE[5][3])
    for nm, order in (('letter order', lambda x: x), ('slowest first', slow_first)):
        by_clock = set(k for k in keys if not simulate(CODE, [order([int(c) for c in g]) for g in k.split('|')], k=2, lock=1, m=1.5)['wrong'])
        check(f'challenge P1: the clock rule and the model-call rule pick the same {len(by_clock)} right arrangements ({nm})', by_clock == set(k for k in keys if turn_rule_right(k, order)), len(by_clock))
    a = simulate(CODE, [[0, 1, 2], [3, 4, 5]], k=2, lock=1, m=1.5)
    check('challenge P1: ABC → DEF, the tests start at 4.9 s, after D (4.9 s) and before E (5.7 s); 10.4 s, wrong',
          r1(a['start'][5]) == 4.9 and r1(a['end'][3]) == 4.9 and r1(a['end'][4]) == 5.7 and a['early_by'] == {5: [4]} and r1(a['total']) == 10.4, a['early_by'])
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
        pg.click('#tls-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(300)
        for i, v in enumerate([1, 1, 1, 2, 2, 2]): pg.click(f'#tls-ch-{i} button[data-v="{v}"]'); pg.wait_for_timeout(60)
        pg.click('#tls-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        t = pg.evaluate("()=>document.querySelector('#tls-ch-res').innerText")
        e_nm, d_nm = (('E edit test_parser.py', 'D edit parser.py') if lang == 'en' else ('E 改 test_parser.py', 'D 改 parser.py'))
        check(f'{lang} challenge P1: the ABC → DEF verdict names E still editing, not D', e_nm in t and d_nm not in t.split('\n')[0], t[:200])
        for i, v in enumerate([1, 1, 1, 2, 2, 1]): pg.click(f'#tls-ch-{i} button[data-v="{v}"]'); pg.wait_for_timeout(60)
        pg.click('#tls-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        t = pg.evaluate("()=>document.querySelector('#tls-ch-res').innerText").split('\n')[0]
        want = '“F run the tests” started before “D edit parser.py” and “E edit test_parser.py” had begun' if lang == 'en' else '“F 跑测试”开始时，“D 改 parser.py”和“E 改 test_parser.py”还没改，测试读到的是改之前的文件'
        check(f'{lang} challenge N10: ABCF → DE says both edits had not begun', want in t and ('still editing' not in t if lang == 'en' else '还没改完' not in t), t[:200])
        for i, v in enumerate([1, 1, 1, 2, 3, 2]): pg.click(f'#tls-ch-{i} button[data-v="{v}"]'); pg.wait_for_timeout(60)
        pg.click('#tls-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        t = pg.evaluate("()=>document.querySelector('#tls-ch-res').innerText").split('\n')[0]
        want = 'while “D edit parser.py” was still editing and before “E edit test_parser.py” had begun' if lang == 'en' else '“D 改 parser.py”还没改完，“E 改 test_parser.py”还没改，'
        check(f'{lang} challenge N10: ABC → DF → E says D was still editing and E had not begun', want in t, t[:200])
        pg.click('#tls-app .modes button[data-mode="sandbox"]'); pg.wait_for_timeout(300)
    b.close()
for e in errs: print('ERR', e)
print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails or errs else 0)
