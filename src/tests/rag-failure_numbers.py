"""Numbers test for the rag-failure lab: an independent Python twin of the page's engine (each run's four checks read from its record,
the 200-question breakdown with every set of fixes for both systems, the four rates, the added wait, the challenge's feasible pairs and
best pair) compared with the page; the numbers the page writes in its cards, charts, observations and results; the home tile; the
interview drill (mulberry32 re-implemented); and the quoted source wordings re-read from sources/b10src/rag-failure/raw/.
Run: python3 tests/rag-failure_numbers.py"""
import itertools, math, pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
(P / 'out' / 'rag-failure').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/rag-failure.html', '--only', 'rag-failure'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'rag-failure' / 'num-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'rag-failure.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'rag-failure' / 'num-wrapped.html')
RAW = P / 'sources' / 'b10src' / 'rag-failure' / 'raw'
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


# ---------- the twin (the lab's stated assumptions, written again here) ----------
KINDS = ['miss', 'cut', 'stale', 'unsup', 'multi', 'noans']
MIX = {'base': dict(ok=140, miss=6, cut=16, stale=12, unsup=12, multi=4, noans=10),
       'ch': dict(ok=118, miss=10, cut=8, stale=28, unsup=6, multi=20, noans=10)}
SHARE = {'hyb': dict(miss=.6, cut=.5, multi=.25), 'ctx': dict(cut=.75, multi=.25), 'ver': dict(stale=.9),
         'cite': dict(unsup=.5, noans=.8), 'dec': dict(multi=.75, miss=.2)}
WAIT = {'hyb': 200, 'ctx': 100, 'ver': 0, 'cite': 100, 'dec': 600}
FIXES = list(SHARE)
NRET, BUDGET, GOAL, LIMIT = 6, 3, 0.75, 300
half_up = lambda x: math.floor(x + 0.5 + 1e-9)


def twin(sys_, fx):
    mix = MIX[sys_]; N = sum(mix.values()); ans = N - mix['noans']
    left = {}
    for k in KINDS:
        stay = 1.0
        for f in fx: stay *= 1 - SHARE[f].get(k, 0)
        left[k] = half_up(mix[k] * stay)
    ok = N - sum(left.values())
    n_ret = ans - left['miss'] - left['multi']; n_ctx = n_ret - left['cut']; n_grd = N - left['unsup'] - left['noans']
    return dict(N=N, ans=ans, ok=ok, left=left, nRet=n_ret, nCtx=n_ctx, nGrd=n_grd, cor=ok / N, ret=n_ret / ans, ctx=n_ctx / ans, grd=n_grd / N,
                wait=sum(WAIT[f] for f in fx))


def run_checks(r):
    top = [x[0] for x in r['ret']]; sent = top[:BUDGET]; need = [g[0] for g in r['gold']]
    ret = all(g in top for g in need) if need else None
    ctx = all(g in sent for g in need) if need else None
    grd = all(c[2] is not None and c[2] in sent for c in r['ans'])
    return dict(ret=ret, ctx=ctx, grd=grd, cor=r['kind'] == 'ok')


# which checks a kind fails: the engine's FLAG table, derived here from the definitions
EXPECT = {'ok': dict(ret=True, ctx=True, grd=True, cor=True), 'miss': dict(ret=False, ctx=False, grd=True, cor=False),
          'cut': dict(ret=True, ctx=False, grd=True, cor=False), 'stale': dict(ret=True, ctx=True, grd=True, cor=False),
          'unsup': dict(ret=True, ctx=True, grd=False, cor=False), 'multi': dict(ret=False, ctx=False, grd=True, cor=False),
          'noans': dict(ret=None, ctx=None, grd=False, cor=False)}


def mulberry(a):
    a &= 0xFFFFFFFF
    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (1 | t)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        t = (t ^ (t >> 14)) & 0xFFFFFFFF
        return t / 4294967296
    return nxt


def drill(seed):
    r = mulberry(seed * 7919 + 13); pick = lambda a, b: a + math.floor(r() * (b - a + 1))
    N = 400; ans = pick(340, 380)
    while True:
        miss = pick(4, 40); cut = pick(4, 40); unsup = pick(4, 40); o = sorted([miss, cut, unsup], reverse=True)
        if o[0] - o[1] >= 3: break
    return dict(N=N, ans=ans, miss=miss, cut=cut, unsup=unsup, ret=(ans - miss) / ans, ctx=(ans - miss - cut) / ans, grd=(N - unsup - (N - ans)) / N)


pc = lambda v: f'{half_up(v * 100)}%'

with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300); pg.evaluate("()=>{location.hash='rag-failure'}"); pg.wait_for_timeout(500)
    D = pg.evaluate("()=>window.__rgf.data()")
    # the constants the page uses are the ones stated above
    check('mix', D['MIX'] == MIX, D['MIX'])
    check('shares', {f: {k: v for k, v in s.items()} for f, s in D['SHARE'].items()} == SHARE, D['SHARE'])
    check('waits', D['WAIT'] == WAIT and D['NRET'] == NRET and D['BUDGET'] == BUDGET and D['CH_GOAL'] == GOAL and D['CH_WAIT'] == LIMIT)
    check('each mix is 200 questions', all(sum(m.values()) == 200 for m in MIX.values()))
    # every run's checks: from its record, against the page and against its kind
    for i, r in enumerate(D['RUNS']):
        c = run_checks(r); pc_ = pg.evaluate("i=>window.__rgf.checks(i)", i)
        check(f'run {r["id"]}: checks match the page', c == pc_, (c, pc_))
        check(f'run {r["id"]}: checks match its kind ({r["kind"]})', c == EXPECT[r['kind']], c)
        check(f'run {r["id"]}: {NRET} chunks, scores falling', len(r['ret']) == NRET and all(r['ret'][j][1] > r['ret'][j + 1][1] for j in range(NRET - 1)))
        for g, rank in r['gold']:
            ids = [x[0] for x in r['ret']]
            check(f'run {r["id"]}: rank of {g}', (ids.index(g) + 1 == rank) if g in ids else rank > NRET, (g, rank))
        for cl in r['ans']:
            check(f'run {r["id"]}: claim source exists', cl[2] is None or cl[2] in D['CH'])
    check('one run per kind and one correct run', sorted(r['kind'] for r in D['RUNS']) == sorted(['ok'] + KINDS))
    # the breakdown for every set of fixes, both systems
    for s in ('base', 'ch'):
        for k in range(6):
            for fx in itertools.combinations(FIXES, k):
                t = twin(s, list(fx)); m = pg.evaluate("([s,fx])=>window.__rgf.compute(s,fx)", [s, list(fx)])
                same = all(abs(t[x] - m[x]) < 1e-12 for x in ('N', 'ans', 'ok', 'nRet', 'nCtx', 'nGrd', 'cor', 'ret', 'ctx', 'grd', 'wait')) and t['left'] == m['left']
                check(f'{s} {"+".join(fx) or "none"}', same, (t, m))
                # no count sits on a rounding edge (n × share ending in .5 would round differently in other languages)
                for kk in (KINDS if s == 'base' or len(fx) <= 2 else []):   # the challenge never shows more than two fixes
                    stay = 1.0
                    for f in fx: stay *= 1 - SHARE[f].get(kk, 0)
                    v = MIX[s][kk] * stay
                    check(f'{s} {fx} {kk}: no .5 edge', abs(v - math.floor(v) - 0.5) > 1e-6 or v == 0, v)
    b0 = twin('base', [])
    check('initial: 70% correct', pc(b0['cor']) == '70%' and b0['ok'] == 140)
    check('initial: retrieval recall 95%, context recall 86%, grounded 89%', (pc(b0['ret']), pc(b0['ctx']), pc(b0['grd'])) == ('95%', '86%', '89%'))
    check('predict: most wrong answers after retrieval', (b0['ans'] - b0['nRet']) * 2 < 200 - b0['ok'])
    singles = {f: twin('base', [f])['ok'] for f in FIXES}
    check('single fixes (hyb 153, ctx 153, ver 151, cite 154, dec 144)', singles == dict(hyb=153, ctx=153, ver=151, cite=154, dec=144), singles)
    check('the best single fix is cite', max(singles, key=singles.get) == 'cite')
    # the challenge
    pts = []
    for pr in itertools.combinations(FIXES, 2):
        t = twin('ch', list(pr)); pts.append((pr, t['ok'], t['cor'], t['wait'], t['cor'] >= GOAL - 1e-9 and t['wait'] <= LIMIT))
    met = [x for x in pts if x[4]]; best = max(x[1] for x in met)
    page = pg.evaluate("()=>window.__rgf.chAll()")
    check('challenge: 10 pairs', len(page) == 10)
    for x, y in zip(pts, page):
        check(f'challenge pair {x[0]}', list(x[0]) == y['p'] and x[1] == y['ok'] and x[4] == y['met'] and (x[1] == best and x[4]) == y['best'], (x, y))
    check('challenge: 3 pairs meet both limits', len(met) == 3, met)
    check('challenge: best pair is hybrid + version filter at 79%', [x[0] for x in met if x[1] == best] == [('hyb', 'ver')] and pc(best / 200) == '79%')
    check('challenge: version + split meets the bar but not the wait', any(x[0] == ('ver', 'dec') and x[2] >= GOAL and x[3] > LIMIT for x in pts))
    check('challenge: initial 59%', pc(twin('ch', [])['cor']) == '59%')

    # ---------- the numbers the page writes ----------
    def txt(sel):
        return pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' '):''}", sel)
    for lang in ('en', 'zh'):
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(150); pg.evaluate("()=>{location.hash='rag-failure'}"); pg.wait_for_timeout(400)
        pg.evaluate("()=>document.querySelector('#rgf-rail [data-go=\"0\"]').click()"); pg.wait_for_timeout(300)
        card = txt('#rgf-rail .step.current')
        check(f'{lang} mission names 140 and 70%', '140' in card and '70%' in card, card[:200])
        pg.evaluate("()=>document.querySelector('#rgf-rail [data-go=\"4\"]').click()"); pg.wait_for_timeout(300)
        cards = txt('#rgf-metrics')
        for v in ('70%', '95%', '86%', '89%', '140', '180', '164', '178'):
            check(f'{lang} metric cards show {v}', v in cards, cards)
        bars = txt('#rgf-bars')
        for k in KINDS:
            check(f'{lang} breakdown shows {k} = {MIX["base"][k]}', str(MIX['base'][k]) in bars)
        obs = txt('#rgf-rail .step.current')
        check(f'{lang} part 5 names 10 and 50', ' 10 ' in obs or '10 个' in obs, obs[:300])
        pg.evaluate("()=>document.querySelector('#rgf-rail [data-go=\"5\"]').click()"); pg.wait_for_timeout(300)
        pg.evaluate("()=>document.querySelector('#rgf-rail [data-lever=\"cite\"]').click()"); pg.wait_for_timeout(300)
        obs = txt('#rgf-rail [data-obs]')
        check(f'{lang} part 6: cite gives 77%, 14 more (154 of 200)', '14' in obs and '77%' in obs, obs)
        check(f'{lang} part 6: card shows 77%', '77%' in txt('#rgf-metrics'))
        pg.evaluate("()=>document.querySelector('#rgf-rail [data-go=\"6\"]').click()"); pg.wait_for_timeout(300)
        rem = txt('#rgf-rail .step.current')
        gain = {f: twin('base', [f])['ok'] - 140 for f in FIXES}
        check(f'{lang} part 7: split the question adds the fewest ({gain["dec"]}); rerank and the bigger budget add {gain["hyb"]} and {gain["ctx"]}; the cut kind is the largest',
              min(gain, key=gain.get) == 'dec' and re.search(rf'\b{gain["dec"]}\b', rem) and re.search(rf'\b{gain["hyb"]}\b', rem) and gain['hyb'] == gain['ctx'] and max(MIX['base'][k] for k in MIX['base'] if k != 'ok') == MIX['base']['cut'], (gain, rem[-300:]))
        # challenge: pick the best pair and read the result
        pg.evaluate("()=>document.querySelector('#rgf-app .modes button[data-mode=\"challenge\"]').click()"); pg.wait_for_timeout(300)
        for f in ('hyb', 'ver'): pg.evaluate("f=>document.querySelector(`#rgf-rail [data-chfx=\"${f}\"]`).click()", f); pg.wait_for_timeout(100)
        pg.evaluate("()=>document.querySelector('#rgf-rail [data-act=\"submit\"]').click()"); pg.wait_for_timeout(300)
        res = txt('#rgf-ch-res')
        check(f'{lang} challenge result: 158, 79%, 200 ms', '158' in res and '79%' in res and '200' in res, res[:300])
        for x in met:
            check(f'{lang} chart lists {x[0]} at {pc(x[2])}', pc(x[2]) in res)
        why_n = [twin('ch', ['ver'])['ok'] - 118, MIX['ch']['stale'], MIX['ch']['multi'], twin('ch', ['hyb', 'ver'])['ok'] - twin('ch', ['ver'])['ok']]
        check(f'{lang} challenge why names {why_n} (version filter repairs, the two largest kinds, hybrid on top)', all(re.search(rf'\b{v}\b', res) for v in why_n), res[-500:])
        pg.evaluate("()=>document.querySelector('#rgf-app .modes button[data-mode=\"guided\"]').click()"); pg.wait_for_timeout(200)
    # the drill
    for seed in range(1, 8):
        d = drill(seed); m = pg.evaluate("s=>window.__rgf.drill(s)", seed)
        check(f'drill {seed}', all(d[k] == m[k] if isinstance(d[k], int) else abs(d[k] - m[k]) < 1e-12 for k in d), (d, m))
        check(f'drill {seed}: losses add up', d['N'] - d['ans'] + d['unsup'] == round(d['N'] * (1 - d['grd'])))
    for seed in range(1, 201):   # the learner sees one decimal: the three losses rebuilt from it are exact and differ
        d = drill(seed); r1 = lambda v: half_up(v * 1000) / 1000
        got = [round(d['ans'] * (1 - r1(d['ret']))), round(d['ans'] * (r1(d['ret']) - r1(d['ctx']))), round(d['N'] * (1 - r1(d['grd']))) - (d['N'] - d['ans'])]
        o = sorted(got, reverse=True)
        if got != [d['miss'], d['cut'], d['unsup']] or o[0] - o[1] < 3:
            check(f'drill {seed}: losses rebuilt from the shown percentages', False, (got, d)); break
    else:
        check('drill seeds 1-200: losses rebuilt from the shown percentages are exact, the largest ahead by 3 or more', True)
    # the Connect code: run the page's Python on the seven runs; each run's label is its kind, and the rates match the engine's checks
    code = re.search(r"const CODE=`(.*?)`;", (P / 'labs' / 'rag-failure.js').read_text(), re.S).group(1).replace('\\\\', '\\')
    check('Connect code: at most 74 characters a line', max(len(x) for x in code.split('\n')) <= 74)
    ns = {}; exec(code, ns)
    runs = pg.evaluate("()=>window.__rgf.data().RUNS")
    want = {'ok': 'ok', 'miss': 'not retrieved', 'cut': 'cut by the budget', 'stale': 'outdated page? check versions',
            'unsup': 'not from the context', 'multi': 'needs two passages', 'noans': 'no answer in the docs'}
    res = []
    for r in runs:
        ids = [x[0] for x in r['ret']]; need = [x[0] for x in r['gold']]
        run = {'retrieved': ids, 'sent': ids[:3], 'sentences': [(c[0], c[2]) for c in r['ans']], 'correct': r['kind'] == 'ok'}
        lab, chk = ns['triage'](run, need, lambda s, sent: bool(s[1]) and s[1] in sent)
        jc = pg.evaluate("i=>window.__rgf.checks(i)", runs.index(r))
        check(f'Connect code: run {r["id"]} is labelled {want[r["kind"]]!r}', lab == want[r['kind']], lab)
        check(f'Connect code: run {r["id"]} checks match the page', [chk['retrieval'] if need else None, chk['context'] if need else None, chk['grounded'], chk['correct']] == [jc['ret'], jc['ctx'], jc['grd'], jc['cor']], (chk, jc))
        res.append((lab, chk, need))
    rt = ns['rates'](res)
    check('Connect code: rates over the seven runs', abs(rt['correct'] - 1 / 7) < 1e-9 and abs(rt['grounded'] - sum(c['grounded'] for _, c, _ in res) / 7) < 1e-9, rt)
    # Show the math: every substitution computes to the result it shows (fractions to 0.1 points; a product to its rounded count,
    # with “=” only when exact), in five fix states and in the drill for seeds 1-30
    def tex_ok(t):
        m = re.search(r'\\frac\{([\d\s\-]+)\}\{(\d+)\}\s*=\s*\\text\{([\d.]+)%\}', t)
        if m:
            v = eval(m.group(1)) / int(m.group(2)) * 100
            return abs(float(f'{v:.1f}') - float(m.group(3))) < 1e-9
        m = re.match(r'^(\d+)\s*((?:\\times \(1 - [\d.]+\)\s*)+)(=|\\approx)\s*(\d+)$', t.strip())
        if m:
            v = int(m.group(1))
            for s_ in re.findall(r'\(1 - ([\d.]+)\)', m.group(2)): v *= 1 - float(s_)
            return half_up(v) == int(m.group(4)) and ((m.group(3) == '=') == (abs(v - round(v)) < 1e-9))
        m = re.match(r'^(\d+) \\times \((?:1|([\d.]+)) - ([\d.]+)\)(?: - (\d+))? (=|\\approx) (\d+)$', t.strip())
        if m:
            a_ = float(m.group(2)) if m.group(2) else 1.0; v = int(m.group(1)) * (a_ - float(m.group(3))) - int(m.group(4) or 0)
            return round(v) == int(m.group(6)) and ((m.group(5) == '=') == (abs(v - round(v)) < 1e-6))
        return None
    bad, seen_n = [], 0
    for bits in ('00000', '10000', '00110', '01001', '11111', '10100'):
        pg.evaluate(f"()=>{{location.hash='rag-failure~f{bits}_r1'}}"); pg.wait_for_timeout(350)
        pg.evaluate("()=>{const m=document.querySelector('#rgf-math'); if(m) m.open=true}"); pg.wait_for_timeout(100)
        for t in pg.evaluate("()=>[...document.querySelectorAll('#rgf-math [data-tex]')].map(e=>e.getAttribute('data-tex'))"):
            r = tex_ok(t)
            if r is not None:
                seen_n += 1
                if not r: bad.append((bits, t))
    check(f'Show the math: {seen_n} substitution lines in six fix states compute to the shown result', seen_n >= 30 and not bad, bad)
    bad, seen_n = [], 0
    for seed in range(1, 31):
        pg.evaluate("s=>window.__rgf.drillShow(s)", seed); pg.wait_for_timeout(60)
        for t in pg.evaluate("()=>[...document.querySelectorAll('#rgf-drill-a [data-tex]')].map(e=>e.getAttribute('data-tex'))"):
            r = tex_ok(t); seen_n += r is not None
            if r is False: bad.append((seed, t))
    check(f'drill: {seen_n} substitution lines (seeds 1-30) compute to the shown result, “=” only when exact', seen_n == 90 and not bad, bad[:3])
    # scenario insights: each count they name, and the “most of them” claims, from the twin
    pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(200)
    for bits, fx in (('10000', ['hyb']), ('00110', ['ver', 'cite']), ('01001', ['ctx', 'dec'])):
        pg.evaluate(f"()=>{{location.hash='rag-failure~f{bits}_r1'}}"); pg.wait_for_timeout(350)
        ins = pg.evaluate("()=>document.querySelector('#rgf-insight').innerText")
        m, m0 = twin('base', fx), twin('base', [])
        gain = m['ok'] - m0['ok']; fixed = {k: MIX['base'][k] - m['left'][k] for k in m['left']}
        need = [str(gain)] + ([str(fixed['cut'])] if fx == ['hyb'] else []) + ([str(twin('base', ['hyb'])['ok'] - 140)] if fx == ['ver', 'cite'] else [])
        claim = {'10000': fixed['cut'] * 2 > gain, '00110': gain > twin('base', ['hyb'])['ok'] - 140, '01001': fixed['cut'] * 2 > gain}[bits]
        check(f'scenario {"+".join(fx)}: names {need} and its claim holds', all(re.search(rf'\b{x}\b', ins) for x in need) and claim, ins[:300])
    # the home tile: build with the map and read each option's sentence
    pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(300)
    pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(300)
    tile = pg.query_selector('#tile-rag-failure')
    if tile:
        want = [('ret', 10), ('ctx', 16), ('doc', 12), ('ans', 22)]
        for i, (k, v) in enumerate(want):
            pg.evaluate("i=>document.querySelector(`#tile-rag-failure .seg-ctl[data-key=\"tzs\"] button[data-v=\"${i}\"]`).click()", i); pg.wait_for_timeout(120)
            rd = pg.evaluate("()=>document.querySelector('#tile-rag-failure .tz-read').textContent")
            check(f'tile {k}: {v} of 60', f'{v} of the 60' in rd, rd)
        check('tile: 200 squares', pg.evaluate("()=>document.querySelectorAll('#tile-rag-failure .tz-rgf-q').length") == 200)
    else:
        check('tile present', False)
    check('no page errors', not errs, errs[:3])
    b.close()

# ---------- sources ----------
QUOTES = [('ragas_faithfulness.md', 'A response is considered **faithful** if all its claims can be supported by the retrieved context.'),
          ('ragas_faithfulness.md', 'Number of claims in the response supported by the retrieved context'),
          ('ragas_context_recall.md', 'Number of claims in the reference supported by the retrieved context'),
          ('trulens_rag_triad.md', 'The RAG triad is made up of 3 evaluations: context relevance, groundedness and'),
          ('ac_contextual_guide.ipynb', "Our evaluation dataset contains 248 queries - each of which contains a 'golden chunk.'"),
          ('ac_contextual_guide.ipynb', "Pass@k checks whether or not the 'golden document' was present in the first k documents retrieved for each query."),
          ('ac_contextual_guide.ipynb', 'from 12.85% failure rate down to 4.74%'),
          ('ac_contextual_guide.ipynb', 'reranking adds ~100-200ms per query'),
          ('ac_contextual_guide.ipynb', '| **+ Reranking** | 92.15% | 95.26% | 97.45% |'),
          ('ac_contextual_guide.ipynb', 'Reranking: Adds 100-200ms query latency')]
for f, q in QUOTES:
    t = (RAW / f).read_text() if (RAW / f).exists() else ''
    check(f'source {f}: “{q[:50]}…”', q in t)
src = (P / 'labs' / 'rag-failure.js').read_text()
for pin in ('298b68274234c060deacab3cf5fb52aa3a20e885', '96d858212b525450cd9dc091ab1fd9a7e6693eab', 'd7265d6ae994ccd8429db0594b000073b2f9ad43'):
    check(f'pin {pin[:7]} in the page and the README', pin in src and pin in (P / 'sources' / 'b10src' / 'rag-failure' / 'README.md').read_text())

print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
