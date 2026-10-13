"""Golden numbers for the product design capstone lab: every number the page shows, recomputed independently in Python and compared
with the page's model (window.__pdz) through Playwright, plus the numbers on the metric cards for key setups.

Independent derivations:
- the serving engine is the serving capstone's Python reference (capstone/engine.py), loaded twice and given the shapes of
  Llama-3.1-8B on one H100 and Llama-3.1-70B on four H100s (tensor parallelism: the step's FLOPs and bytes spread over n GPUs, the KV
  pool n x (80 GiB x 0.9 - 2 GiB) minus the weights), not the lab's JS port;
- the seeded chat load (mulberry32, seed 7, 200 requests) rebuilt here in the same draw order;
- goodput per replica by scanning the whole grid of rates (0.25 to 32 requests/s), which also checks that passing is monotone in the
  rate (the page finds it by bisection);
- the first-word budget, the fleet sizing and the cost sheet written out again from the lab's stated assumptions;
- the challenge: all 140 combinations, the ones that pass, the cheapest one; the drill's random draws.
Run: python3 tests/product-design_numbers.py"""
import importlib.util, math, pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/product-design.html', '--only', 'product-design'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'pdz').mkdir(parents=True, exist_ok=True)
(P / 'out' / 'pdz' / 'numbers.html').write_text('<!doctype html><html><head><meta charset="utf-8"></head><body>' + (P / 'out' / 'product-design.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'pdz' / 'numbers.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', info)


def close(a, b, rel=1e-9, ab=1e-12):
    return abs(a - b) <= max(ab, rel * max(abs(a), abs(b)))


# ---------- the engine, two models ----------
def engine(name, L, D, KVH, HD, FFN, VOCAB, ng):
    spec = importlib.util.spec_from_file_location(name, P / 'capstone' / 'engine.py')
    E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
    E.L, E.D, E.KVH, E.HD, E.FFN, E.VOCAB = L, D, KVH, HD, FFN, VOCAB
    E.P_LAYER = 2 * D * D + 2 * D * KVH * HD + 3 * D * FFN + 2 * D
    E.P_LAYERS = L * E.P_LAYER + D; E.P_HEAD = VOCAB * D; E.P_TOTAL = E.P_LAYERS + 2 * E.P_HEAD
    E.KV_TOK = 2 * L * KVH * HD * 2; E.W_READ = 2 * (E.P_LAYERS + E.P_HEAD); E.W_MEM = 2 * E.P_TOTAL
    E.FS = 1979e12 / 2 * 0.5 * ng; E.BS = 3.35e12 * 0.6 * ng
    E.POOL_T = math.floor((ng * (80 * E.GiB * 0.9 - 2 * E.GiB) - E.W_MEM) / E.KV_TOK); E.POOL_B = E.POOL_T // E.BLOCK
    return E


MOD = {'s': dict(E=engine('pdz_e8', 32, 4096, 8, 128, 14336, 128256, 1), n=1, budget=2048),
       'l': dict(E=engine('pdz_e70', 80, 8192, 8, 128, 28672, 128256, 4), n=4, budget=1024)}
check('8B parameters 8,030,261,248', MOD['s']['E'].P_TOTAL == 8030261248, MOD['s']['E'].P_TOTAL)
check('70B parameters 70,553,706,496', MOD['l']['E'].P_TOTAL == 70553706496, MOD['l']['E'].P_TOTAL)
SYS, OWN, RET, OUT, NREQ, SEED, CAP, STEP, GMAX = 2048, (500, 3500), 1536, (100, 500), 200, 7, 256, 0.25, 128


def load(E, rate, ret):
    rnd = E.mulberry(SEED); t = 0.0; out = []; add = RET if ret else 0
    for _ in range(NREQ):
        t += E.exp_gap(rnd, rate)
        own = E.uni(rnd, OWN[0] + add, OWN[1] + add); o = E.uni(rnd, OUT[0], OUT[1])
        out.append({'t': E.ms(t), 'cls': 'chat', 'app': 'sys', 'pre': SYS, 'own': own, 'out': o})
    return out


def run_at(m, i, cache, ret):
    E = MOD[m]['E']
    run = E.simulate(load(E, i * STEP, ret), {'policy': 'continuous', 'cap': CAP, 'budget': MOD[m]['budget'], 'cache': bool(cache), 'cache_blocks': None})
    E.check(run)
    mt = E.metrics(run, E.GOALS); R = run['R']
    ttft = [r.first - r.arr for r in R]; tpot = [(r.fin - r.first) / (r.out - 1) for r in R]
    return dict(ok=mt['chat']['ok_frac'], ttft90=E.pct(ttft, 90), tpot90=E.pct(tpot, 90), tpot50=E.pct(tpot, 50), gap=mt['chat']['gap_max'])


POOL = {}
for m in ('s', 'l'):
    for cache in (0, 1):
        for ret in (0, 1):
            ok = []
            for i in range(1, GMAX + 1):
                r = run_at(m, i, cache, ret); ok.append(r['ok'] >= 0.9 - 1e-9)
            k = next((j for j, v in enumerate(ok) if not v), len(ok))   # the leading run of passes: rates 1..k
            check(f'goodput {m} cache {cache} ret {ret}: passing is monotone in the rate', not any(ok[k:]), [j + 1 for j, v in enumerate(ok) if v and j >= k][:5])
            at = run_at(m, k, cache, ret)
            POOL[(m, cache, ret)] = dict(g=k * STEP, gpu=k * STEP / MOD[m]['n'], ttft90=at['ttft90'], tpot90=at['tpot90'], tpot50=at['tpot50'], idx=k)


def quiet(m, cache, ret):
    E = MOD[m]['E']; own = (OWN[0] + OWN[1]) / 2 + (RET if ret else 0)
    spec = [{'t': t, 'cls': 'chat', 'app': 'sys', 'pre': SYS, 'own': own, 'out': 1} for t in (0, 2)]
    r = E.simulate(spec, {'policy': 'continuous', 'cap': CAP, 'budget': MOD[m]['budget'], 'cache': bool(cache), 'cache_blocks': None})['R'][1]
    return dict(ttft=r.first - r.arr, prefill=SYS + own - r.hit, hit=r.hit, prompt=SYS + own)


# ---------- the product model, from the lab's stated assumptions ----------
LAT = dict(net=40, gw=5, hist=15, ctx=5, ret=150, inF=40, route=10, chk=20)
TTFT_GOAL, CHECK_GPU, OUTMEAN, EASY = 1.0, 0.001, 300, 0.7   # TTFT_GOAL only to check the engine's ttft90 against the sizing goal
BASE = dict(dau=1e6, msgs=10, peak=3, stream=1, cache=0, ret=0, share=0, filt=8, price=3)
CH = dict(dau=2e6, msgs=12, peak=3, ret=0, price=3, vis=1.5, budget=32000)
SHARES, FILTS = [0, 0.3, 0.5, 0.7, 0.9], ['off', 'after', 'whole', 32, 16, 8, 4]


def compute(s):
    lam = s['dau'] * s['msgs'] / 86400 * s['peak']
    Pm = {m: POOL[(m, s['cache'], s['ret'])] for m in ('s', 'l')}
    rate = {'s': lam * s['share'], 'l': lam * (1 - s['share'])}
    rep = {m: (math.ceil(rate[m] / Pm[m]['g'] - 1e-9) if rate[m] > 0 else 0) for m in ('s', 'l')}
    gpus = {m: rep[m] * MOD[m]['n'] for m in ('s', 'l')}
    f = s['filt']
    cps = OUTMEAN / f if isinstance(f, (int, float)) else (1 if f in ('whole', 'after') else 0)
    fG = math.ceil(lam * cps * CHECK_GPU - 1e-9) if cps else 0
    total = gpus['s'] + gpus['l'] + fG; cost = total * 24 * s['price']
    used = [m for m in ('s', 'l') if rate[m] > 0]
    tpot = max(Pm[m]['tpot90'] for m in used)
    t_model = max(Pm[m]['ttft90'] for m in used)   # the engine's 90th percentile TTFT at goodput, from capstone/engine.py
    seg = [('net', .04), ('gw', .005), ('hist', .015), ('ctx', .005)] + ([('ret', .15)] if s['ret'] else []) + [('inF', .04)] + ([('route', .01)] if s['share'] > 0 else []) + [('model', t_model)]
    tFirst = sum(d for _, d in seg)
    chk = f != 'off'
    gen = hold = 0
    if not s['stream'] or f == 'whole': gen, hold = (OUTMEAN - 1) * tpot, (0.02 if chk else 0)
    elif isinstance(f, (int, float)): hold = (f - 1) * tpot + 0.02
    if gen > 0: seg.append(('gen', gen))
    if hold > 0: seg.append(('hold', hold))
    seg.append(('back', .04))
    vis = sum(d for _, d in seg)
    nChk = (math.ceil(OUTMEAN / f) if isinstance(f, (int, float)) else 1) if chk else 0
    return dict(lam=lam, rep=rep, gpus=gpus, fG=fG, total=total, costDay=cost, perUserMonth=cost * 30 / s['dau'],
                storeGB=s['dau'] * s['msgs'] * 400 * 4 / 1e9, tpot=tpot, vis=vis, hold=hold, gen=gen, tModel=t_model, cps=cps, fRaw=lam * cps * CHECK_GPU, tFirst=tFirst, seg=seg,
                before=chk and (f != 'after' or not s['stream']), quality=s['share'] <= EASY + 1e-9, nChk=nChk)


def ch_eval(o):
    r = compute({**BASE, **CH, **o})
    okV, okB, okS, okQ = r['vis'] <= CH['vis'] + 1e-9, r['costDay'] <= CH['budget'] + 1e-9, r['before'], r['quality']
    return dict(o, r=r, okV=okV, okB=okB, okS=okS, okQ=okQ, ok=okV and okB and okS and okQ)


ALL = [ch_eval(dict(stream=st, cache=c, share=sh, filt=f)) for st in (1, 0) for c in (0, 1) for sh in SHARES for f in FILTS]

with sync_playwright() as pw:
    b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    J = lambda expr, *a: pg.evaluate(expr, *a)
    C = J('()=>window.__pdz.consts')
    check('constants: latencies', C['LAT'] == LAT, C['LAT'])
    check('constants: load', (C['WL']['sys'], C['WL']['own'], C['WL']['ret'], C['WL']['out'], C['WL']['n'], C['WL']['seed'], C['WL']['cap'], C['WL']['step'], C['WL']['gmax'])
          == (SYS, list(OWN), RET, list(OUT), NREQ, SEED, CAP, STEP, GMAX), C['WL'])
    check('constants: models', (C['MODELS']['s']['budget'], C['MODELS']['l']['budget'], C['MODELS']['s']['n'], C['MODELS']['l']['n']) == (2048, 1024, 1, 4))
    check('constants: base, brief, challenge', C['BASE'] == BASE and C['BRIEF'] == {'vis': 1.8, 'budget': 10000} and C['CH'] == CH)
    # goodput and the latencies at goodput
    for (m, c, rt), p in POOL.items():
        js = J('a=>window.__pdz.pool(...a)', [m, c, rt])
        check(f'pool {m} cache {c} ret {rt}: goodput {p["g"]}', close(js['g'], p['g']) and close(js['gpu'], p['gpu']), (js['g'], p['g']))
        for k in ('ttft90', 'tpot90', 'tpot50'):
            check(f'pool {m} {c} {rt} {k}', close(js[k], p[k], 1e-9), (js[k], p[k]))
        for i in (max(1, p['idx'] - 1), p['idx'], p['idx'] + 1, p['idx'] + 8):
            js = J('a=>window.__pdz.runAt(...a)', [m, i, c, rt]); py = run_at(m, i, c, rt)
            check(f'run {m} {i} {c} {rt}', all(close(js[k], py[k], 1e-9) for k in ('ok', 'ttft90', 'tpot90', 'tpot50', 'gap')), (js, py))
        q = quiet(m, c, rt); jq = J('a=>window.__pdz.quiet(...a)', [m, c, rt])
        check(f'quiet {m} {c} {rt}', all(close(jq[k], q[k], 1e-9) for k in q), (jq, q))
    # the numbers the copy states about the pools
    g = {k: v['g'] for k, v in POOL.items()}
    check('8B goodput: 6.25 / 14.5 (cache) / 4.25 (retrieval) / 6.25 (both)', (g[('s', 0, 0)], g[('s', 1, 0)], g[('s', 0, 1)], g[('s', 1, 1)]) == (6.25, 14.5, 4.25, 6.25), g)
    check('70B x 4 goodput: 2.25 / 5 / 1.25 / 2.5', (g[('l', 0, 0)], g[('l', 1, 0)], g[('l', 0, 1)], g[('l', 1, 1)]) == (2.25, 5, 1.25, 2.5), g)
    for c in (0, 1):
        ratio = (1 / POOL[('s', c, 0)]['gpu']) / (1 / POOL[('l', c, 0)]['gpu'])
        check(f'small model per message about a tenth of the large one (cache {c})', 0.07 <= ratio <= 0.13, ratio)
    q0, q1 = quiet('s', 0, 0), quiet('s', 1, 0)
    check('quiet 8B: caching computes 2,000 of 4,048 tokens and halves the prefill', q0['prefill'] == 4048 and q1['prefill'] == 2000 and 0.45 < q1['ttft'] / q0['ttft'] < 0.55, (q0, q1))
    # every guided setup and every challenge combination
    states = [dict(BASE)]
    for dau in (1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7):
        for st in (0, 1):
            for c in (0, 1):
                for rt in (0, 1):
                    for sh in SHARES:
                        for f in FILTS:
                            states.append(dict(BASE, dau=dau, stream=st, cache=c, ret=rt, share=sh, filt=f))
    states += [dict(BASE, msgs=ms_, peak=pk, price=pr, cache=1, share=0.7) for ms_ in (5, 10, 20, 40) for pk in (2, 3, 5) for pr in (1, 3, 8)]
    jsr = J('ss=>ss.map(s=>window.__pdz.compute(s))', states)
    bad = 0
    for s, js in zip(states, jsr):
        py = compute(s)
        okk = all(close(js[k], py[k], 1e-9) for k in ('lam', 'fG', 'total', 'costDay', 'perUserMonth', 'storeGB', 'tpot', 'vis', 'hold', 'gen', 'tModel', 'cps', 'fRaw', 'tFirst', 'nChk'))
        okk = okk and js['rep'] == py['rep'] and js['gpus'] == py['gpus'] and js['before'] == py['before'] and js['quality'] == py['quality']
        okk = okk and [g_['k'] for g_ in js['seg']] == [k for k, _ in py['seg']] and all(close(g_['d'], d) for g_, (_, d) in zip(js['seg'], py['seg']))
        if not okk:
            bad += 1
            if bad < 4: print('  mismatch', s, {k: (js[k], py[k]) for k in ('total', 'costDay', 'vis')})
    check(f'compute: {len(states)} setups agree', bad == 0, bad)
    # the guided parts' numbers
    base = compute(BASE); cache = compute(dict(BASE, cache=1)); route = compute(dict(BASE, cache=1, share=0.7))
    check('every pool meets the sizing goal: ttft90 at goodput within 1 s', all(p['ttft90'] <= TTFT_GOAL for p in POOL.values()), {k: round(p['ttft90'], 3) for k, p in POOL.items()})
    check('plain: the model segment is the 70B no-cache ttft90 (0.92 s)', f"{base['tModel']:.2f}" == '0.92' and base['tModel'] == POOL[('l', 0, 0)]['ttft90'], base['tModel'])
    check('plain: path 0.145 s outside the model, hold 368 ms', f"{base['vis'] - base['tModel'] - base['hold']:.3f}" == '0.145' and f"{base['hold'] * 1000:.0f}" == '368', (base['vis'] - base['tModel'] - base['hold'], base['hold']))
    check('plain: first word 1.44 s, 634 GPUs, $45,648 a day', f"{base['vis']:.2f}" == '1.44' and base['total'] == 634 and base['costDay'] == 45648, (base['vis'], base['total'], base['costDay']))
    check('plain is within the 1.8 s budget, over the $10,000 budget', base['vis'] <= 1.8 and base['costDay'] > 10000)
    ns = compute(dict(BASE, stream=0))
    check('stream off: about 16 s to the first word, almost all of it writing the answer (gen), 20 ms check', f"{ns['vis']:.1f}" == '16.0' and ns['gen'] > 14.5 and close(ns['hold'], 0.02), (ns['vis'], ns['gen'], ns['hold']))
    check('stream off: the same GPUs and checks as streaming', ns['total'] == base['total'] and ns['fG'] == base['fG'] and ns['nChk'] == base['nChk'] == 38, (ns['total'], ns['fG'], ns['nChk']))
    check('cache: first word 1.44 -> 1.51 s; TPOT p90 49.8 -> 62.6 ms; the hold grows 90 ms', f"{cache['vis']:.2f}" == '1.51' and f"{base['tpot']*1000:.1f}" == '49.8' and f"{cache['tpot']*1000:.1f}" == '62.6' and 0.085 < cache['hold'] - base['hold'] < 0.095, (cache['vis'], cache['tpot']))
    qa, qb = J('a=>window.__pdz.quiet(...a)', ['l', 0, 0]), J('a=>window.__pdz.quiet(...a)', ['l', 1, 0])
    check('cache: prefill tokens drop by more than half, quiet prefill about twice as fast, goodput more than doubles', 1 - qb['prefill'] / qb['prompt'] > 0.5 and 1.8 < qa['ttft'] / qb['ttft'] < 2.2 and 2 < POOL[('l', 1, 0)]['g'] / POOL[('l', 0, 0)]['g'] < 3, (qb, qa['ttft'] / qb['ttft']))
    check('route 70%: first word 1.54 s; 0.155 s path, 0.93 s TTFT, 459 ms hold', f"{route['vis']:.2f}" == '1.54' and f"{route['vis'] - route['tModel'] - route['hold']:.3f}" == '0.155' and f"{route['tModel']:.2f}" == '0.93' and f"{route['hold']*1000:.0f}" == '459', (route['vis'], route['tModel'], route['hold']))
    check('route 70%: within the $10,000 budget', route['costDay'] <= 10000)
    check('prediction parts: 70B 28.6%, 8B 5.8%, checks 4.8% of the all-large GPUs', [f"{v / cache['total'] * 100:.0f}" for v in (route['gpus']['l'], route['gpus']['s'], route['fG'])] == ['29', '6', '5'])
    d5, d6, d7 = (compute(dict(BASE, cache=1, share=0.7, dau=d)) for d in (1e5, 1e6, 1e7))
    check('size: 16 / 115 / 1,135 GPUs; 10x users -> about 10x GPUs, fewer than 10 x 115', (d5['total'], d6['total'], d7['total']) == (16, 115, 1135) and 9.5 < d7['total'] / d6['total'] < 10.5 and d7['total'] < 10 * d6['total'])
    check('size: per user per month $0.346 at 100,000, $0.245 at 10M', (f"{d5['perUserMonth']:.3f}", f"{d7['perUserMonth']:.3f}") == ('0.346', '0.245'))
    check('size: output checks 347.2 x 37.5 x 1 ms = 13.0 GPU-s/s -> 14', f"{route['fRaw']:.1f}" == '13.0' and route['fG'] == 14)
    check('cache: 294 GPUs, $21,168', cache['total'] == 294 and cache['costDay'] == 21168, (cache['total'], cache['costDay']))
    check('route 70%: 115 GPUs = 17 + 84 + 14, $8,280', (route['total'], route['gpus']['s'], route['gpus']['l'], route['fG'], route['costDay']) == (115, 17, 84, 14, 8280), route)
    x = route['costDay'] / cache['costDay']
    check('prediction: 37% is the nearest choice to the result', min((10, 37, 70), key=lambda v: abs(v / 100 - x)) == 37, x)
    check('the 30% left on the large model costs about 30% of before', 0.25 <= route['gpus']['l'] * 72 / cache['costDay'] <= 0.32, route['gpus']['l'] * 72 / cache['costDay'])
    r9 = compute(dict(BASE, cache=1, share=0.9))
    check('route 90%: $4,608 and fails quality', r9['costDay'] == 4608 and not r9['quality'], r9['costDay'])
    f16, f4, fa, fw = (compute(dict(BASE, cache=1, share=0.7, filt=v)) for v in (16, 4, 'after', 'whole'))
    check('chunks of 16: 2.04 s, over 1.8 s', f"{f16['vis']:.2f}" == '2.04' and f16['vis'] > 1.8, f16['vis'])
    check('chunks of 4: 1.29 s, $9,216', f"{f4['vis']:.2f}" == '1.29' and f4['costDay'] == 9216, (f4['vis'], f4['costDay']))
    check('check after sending: no hold, but shown before checked', fa['hold'] == 0 and not fa['before'])
    check('whole answer checked: as slow as no streaming', abs(fw['vis'] - compute(dict(BASE, cache=1, share=0.7, filt='whole', stream=0))['vis']) < 1e-9)
    # the challenge
    ja = J('()=>window.__pdz.chAll()')
    check('challenge: 140 combinations', len(ja) == len(ALL) == 140)
    agree = all(a['ok'] == p_['ok'] and close(a['cost'], p_['r']['costDay']) and close(a['vis'], p_['r']['vis']) and (a['okV'], a['okB'], a['okS'], a['okQ']) == (p_['okV'], p_['okB'], p_['okS'], p_['okQ']) for a, p_ in zip(ja, ALL))
    check('challenge: every combination agrees', agree)
    passing = [e for e in ALL if e['ok']]
    best = min(passing, key=lambda e: e['r']['costDay']) if passing else None
    jb = J('()=>window.__pdz.chBest()')
    check('challenge: two designs pass (70% and 50% to the small model, chunks of 4), so all three result tiers can be reached', sorted((e['stream'], e['cache'], e['share'], e['filt']) for e in passing) == [(1, 1, 0.5, 4), (1, 1, 0.7, 4)], [(e['stream'], e['cache'], e['share'], e['filt']) for e in passing])
    t2 = next(e for e in passing if e['share'] == 0.5)
    check('challenge tier 2: 50% to the small model costs $30,816', t2['r']['costDay'] == 30816, t2['r']['costDay'])
    check('challenge: the best is stream, cache, 70%, chunks of 4, $21,888, 1.29 s', best and (best['stream'], best['cache'], best['share'], best['filt'], best['r']['costDay'], f"{best['r']['vis']:.2f}") == (1, 1, 0.7, 4, 21888, '1.29'), best and best['r']['costDay'])
    f8 = ch_eval(dict(best, filt=8)) if best else None
    check('challenge why: chunks of 8 are cheaper but over 1.5 s (1.54 s)', f8 and f8['r']['costDay'] < best['r']['costDay'] and not f8['okV'] and f"{f8['r']['vis']:.2f}" == '1.54', f8 and f8['r']['vis'])
    check('challenge: 833.3 x 18.75 x 1 ms = 15.6 GPU-s/s -> 16 check GPUs at chunks of 16', (lambda r: f"{r['fRaw']:.1f}" == '15.6' and r['fG'] == 16)(ch_eval(dict(stream=1, cache=0, share=0, filt=16))['r']))
    check('challenge: page best agrees', jb and (jb['stream'], jb['cache'], jb['share'], jb['filt']) == (best['stream'], best['cache'], best['share'], best['filt']) and close(jb['cost'], best['r']['costDay']), jb)
    c0 = ch_eval(dict(stream=1, cache=0, share=0, filt=16))
    check('challenge start: $108,000 a day, 1.84 s', c0['r']['costDay'] == 108000 and f"{c0['r']['vis']:.2f}" == '1.84', (c0['r']['costDay'], c0['r']['vis']))
    check('challenge: 833.3 messages per second at peak', f"{c0['r']['lam']:.1f}" == '833.3')
    # the drill
    E8 = MOD['s']['E']
    for sd in range(8):
        rnd = E8.mulberry(900 + sd)
        dau = [2e5, 5e5, 2e6, 5e6][int(rnd() * 4)]; ms_ = [5, 10, 20][int(rnd() * 3)]; pk = [2, 3][int(rnd() * 2)]; sh = [0.5, 0.7][int(rnd() * 2)]
        s = dict(BASE, dau=dau, msgs=ms_, peak=pk, cache=1, share=sh, filt=8); r = compute(s)
        d = J('x=>window.__pdz.drill(x)', sd)
        check(f'drill {sd}', (d['s']['dau'], d['s']['msgs'], d['s']['peak'], d['s']['share']) == (dau, ms_, pk, sh) and d['gs'] == r['gpus']['s'] and d['gl'] == r['gpus']['l'] and close(d['lam'], r['lam']), (d, r['gpus']))
    # the metric cards show these numbers (Scenarios, EN)
    def cards(tok):
        pg.evaluate("t=>{location.hash='product-design~'+t}", tok); pg.wait_for_timeout(500)
        return pg.evaluate("()=>Object.fromEntries([...document.querySelectorAll('#pdz-metrics .metric')].map(e=>[e.dataset.k,e.querySelector('.v').textContent]))")
    pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(200)
    cd = cards('4_10_3_1_0_0_0_8_30')
    check('cards, plain setup: 1.44 s, 634, $45,648, $1.37', (cd.get('vis'), cd.get('gpu'), cd.get('cost'), cd.get('user')) == ('1.44 s', '634', '$45,648', '$1.37'), cd)
    pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(200)
    cd = cards('4_10_3_1_1_0_7_4_30')
    check('cards, cache + route + chunks of 4: 1.29 s, 128, $9,216', (cd.get('vis'), cd.get('gpu'), cd.get('cost')) == ('1.29 s', '128', '$9,216'), cd)
    check('no page errors', not errs, errs)
    b.close()
print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
