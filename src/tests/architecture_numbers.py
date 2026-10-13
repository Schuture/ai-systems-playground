"""Smoke checks of the Architecture overview v2 engine (first version for the product owner; rebuilt after approval).

python3 tests/architecture_numbers.py
Builds out/architecture-build.html (this lab only; wrapped as out/architecture-numbers.html) and reads window.__ar: model shapes, softmax and percentages, the illustrative
attention heads (causal, rows add up to 1), the mask explorer's pair counts, the residual/RMSNorm simulation, the challenge, and the
causal switch of guided part 5 (softmax with and without the mask, the copy rule, the training loss, the next-token percentages)."""
import math, pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/architecture-build.html', '--only', 'architecture'], check=True, cwd=P, stdout=subprocess.DEVNULL)
PAGE = P / 'out' / 'architecture-numbers.html'
PAGE.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'architecture-build.html').read_text() + '</body></html>')
n, fails = 0, []


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(f'{name} :: {info}')
        print('FAIL', name, '::', str(info)[:300])


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto('file://' + str(PAGE)); pg.wait_for_timeout(500)
    ar = 'window.__ar'
    m = pg.evaluate(f"()=>{ar}.MODELS")
    want = {'llama32-1b': (2048, 16, 32, 8192), 'llama31-8b': (4096, 32, 32, 14336), 'llama31-70b': (8192, 80, 64, 28672), 'llama31-405b': (16384, 126, 128, 53248)}
    for k, (d, L, H, f) in want.items():
        check(f'{k}: d, layers, heads, intermediate size from config.json', (m[k]['d'], m[k]['L'], m[k]['H'], m[k]['f']) == (d, L, H, f), m[k])
    sm = pg.evaluate(f"()=>{ar}.softmax([8,6,5.5,5,4])")
    check('softmax adds up to 1', abs(sum(sm) - 1) < 1e-9, sm)
    pc = pg.evaluate(f"()=>{ar}.pcts({ar}.softmax([8,6,5.5,5,4]))")
    check('percentages add up to exactly 100.0 (77.8, 10.5, 6.4, 3.9, 1.4)', pc == [77.8, 10.5, 6.4, 3.9, 1.4] and abs(sum(pc) - 100) < 1e-9, pc)
    for h in range(4):
        w = pg.evaluate(f"()=>{ar}.weightM({h},true)")
        check(f'head {h + 1}: rows add up to 1 and later tokens get 0 (causal)', all(abs(sum(r) - 1) < 1e-9 for r in w) and all(w[i][j] == 0 for i in range(5) for j in range(i + 1, 5)), w)
    fr = pg.evaluate(f"()=>{ar}.weightM(1,true)[4]")
    check('head 2: “ is” looks mostly at “ France”', max(range(5), key=lambda j: fr[j]) == 3, fr)
    counts = pg.evaluate(f"()=>Object.fromEntries({ar}.MASKS.map(mk=>{{let c=0; for(let i=0;i<16;i++) for(let j=0;j<16;j++) if(mk.f(i,j)) c++; return [mk.k,c];}}))")
    check('masks on 16 tokens: causal 136, bidirectional 256, sparse ones fewer than causal', counts['causal'] == 136 and counts['bi'] == 256 and all(counts[k] < 136 for k in ['window', 'sink', 'strided']), counts)
    # review 12: the pairs each mask computes at 1M tokens, recomputed from each rule (row by row, closed forms checked by brute force)
    def rows_count(n, row): return sum(row(i) for i in range(n))
    def brute(n, f): return sum(1 for i in range(n) for j in range(i + 1) if f(i, j))
    win = lambda n, w: rows_count(n, lambda i: min(i + 1, w))
    sink = lambda n, s_, w: rows_count(n, lambda i: (i - max(0, i - w + 1) + 1) + max(0, min(s_, max(0, i - w + 1))))
    strd = lambda n, l: rows_count(n, lambda i: min(i + 1, l) + (i // l if i >= l else 0))
    ok_small = win(40, 4) == brute(40, lambda i, j: i - j < 4) and sink(60, 4, 9) == brute(60, lambda i, j: j < 4 or i - j < 9) and strd(64, 8) == brute(64, lambda i, j: i - j < 8 or (i - j) % 8 == 0)
    check('mask rules: the row-by-row counts match brute force on small n', ok_small)
    N = 1 << 20
    want1m = {'causal': N * (N + 1) // 2, 'window': win(N, 4096), 'sink': sink(N, 4, 4092), 'strided': strd(N, 1024), 'bi': N * N}
    got1m = pg.evaluate(f"()=>{ar}.M1M")
    check('masks at 1M tokens: every count equals its rule (causal, window 4,096, 4 sinks + 4,092, local 1,024 + every 1,024th, n²)', all(got1m[k] == v for k, v in want1m.items()), (got1m, want1m))
    ratios = {k: round(want1m['causal'] / v) if v <= want1m['causal'] else -round(v / want1m['causal']) for k, v in want1m.items()}
    check('masks at 1M tokens: 128× fewer for the window and sink + window, 342× for strided, 2× as many for bidirectional', ratios == {'causal': 1, 'window': 128, 'sink': 128, 'strided': 342, 'bi': -2}, ratios)
    sim = pg.evaluate(f"()=>{{const a={ar}.simulate(true,true), b={ar}.simulate(false,true), c={ar}.simulate(true,false); return {{both:a.sz.at(-1), noNorm:b.sz.at(-1), noResSim:c.sim.at(-1), bothSim:a.sim.at(-1)}}}}")
    check('simulation: both on stays small, no RMSNorm explodes', sim['both'] < 1e3 and sim['noNorm'] > 1e6, sim)
    check('simulation: without the residual the input fades, with it some stays', sim['noResSim'] < 0.2 and sim['bothSim'] > 0.3, sim)
    chq = pg.evaluate(f"()=>{ar}.CHQ.map(q=>({{n:q.o.length,a:q.a,show:q.show[0]}}))")
    check('challenge: six structure questions (the sixth: training without the mask), four options each, an answer and a picture', len(chq) == 6 and all(q['n'] == 4 and 0 <= q['a'] < 4 and q['show'] for q in chq), chq)
    core = pg.evaluate(f"()=>{ar}.CHQ_CORE.map(q=>({{n:q.o.length,a:q.a,show:q.show[0]}}))")
    check('challenge: three core questions, four options each, an answer and a picture', len(core) == 3 and all(q['n'] == 4 and 0 <= q['a'] < 4 and q['show'] for q in core), core)
    check('six guided parts (three core ideas, then the causal switch, between the mission and the summary)', pg.evaluate(f"()=>{ar}.STEP_IDS") == ['mission', 'flow', 'mix', 'loop', 'mask', 'remember'])
    # the causal switch (guided part 5), recomputed here from its illustrative inputs: the scores, the probabilities, the copy strength C
    # and the flattening K; nothing below reads the page's results except to compare
    mk = pg.evaluate(f"()=>{ar}.MK")
    def softmax(z):
        m = max(v for v in z if v is not None); e = [0.0 if v is None else math.exp(v - m) for v in z]; t = sum(e); return [v / t for v in e]
    def pct1(p):   # largest remainder at 0.1%, so the five add up to exactly 100.0
        u = [v * 1000 for v in p]; fl = [math.floor(v) for v in u]; left = 1000 - sum(fl)
        for i in sorted(range(len(p)), key=lambda i: (-(u[i] - fl[i]), i))[:left]: fl[i] += 1
        return [v / 10 for v in fl]
    S = mk['S']; nk = len(S)
    won = [softmax([S[i][j] if j <= i else None for j in range(nk)]) for i in range(nk)]
    woff = [softmax(S[i]) for i in range(nk)]
    close = lambda a, b: all(abs(x - y) < 1e-9 for ra, rb in zip(a, b) for x, y in zip(ra, rb))
    check('switch: with the mask, each row adds up to 1 and every later token gets exactly 0', close(won, mk['wOn']) and all(mk['wOn'][i][j] == 0 for i in range(nk) for j in range(i + 1, nk)), mk["wOn"])
    check('switch: without the mask, the token each position must predict takes the most attention (the leak), and the last row is unchanged',
          close(woff, mk['wOff']) and all(max(range(nk), key=lambda j: woff[i][j]) == i + 1 for i in range(nk - 1)) and close([woff[nk - 1]], [won[nk - 1]]), [round(r[i + 1], 3) for i, r in enumerate(woff[:-1])])
    p_on = mk['P'] + [softmax(mk['Z0'])[0]]
    p_off = [p * math.exp(mk['C']) / (p * math.exp(mk['C']) + 1 - p) for p in p_on]   # +C on the right token's logit: its share becomes p·e^C / (p·e^C + 1 − p)
    check('switch: the five probabilities with the mask (4.0, 70.0, 12.0, 55.0, 77.8%) and, copying, without it (all above 99.7%)',
          all(abs(a - b) < 1e-12 for a, b in zip(p_on, mk['pOn'])) and all(abs(a - b) < 1e-12 for a, b in zip(p_off, mk['pOff'])) and mk['shown']['pOn'] == ['4.0%', '70.0%', '12.0%', '55.0%', '77.8%'] and min(p_off) > 0.997, (mk['shown'], p_off))
    loss_on = sum(-math.log(p) for p in p_on) / 5; loss_off = sum(-math.log(p) for p in p_off) / 5
    check('switch: training loss = mean of −ln p: 1.31 with the mask, below 0.01 without it', abs(loss_on - mk['lossOn']) < 1e-12 and mk['shown']['lossOn'] == '1.31' and loss_off < 0.01 and mk['shown']['lossOff'] == '< 0.01', (loss_on, loss_off, mk['shown']))
    g_on, g_off = pct1(softmax(mk['Z0'])), pct1(softmax([z / mk['K'] for z in mk['Z0']]))
    check('switch: when generating, “ Paris” 77.8% with the mask (the overview’s own numbers) and 33.5% after training without it',
          g_on == mk['genOn'] == [77.8, 10.5, 6.4, 3.9, 1.4] and g_off == mk['genOff'] == [33.5, 20.3, 18.0, 15.9, 12.3], (mk['genOn'], mk['genOff']))
    check('no page errors', not errs, errs[:3])
    b.close()
print(f'{n} checks, {len(fails)} failed')
sys.exit(1 if fails else 0)
