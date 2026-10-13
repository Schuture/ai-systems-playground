"""Numbers test for the context-eng lab: every number on the page, recomputed independently in Python.

- the seeded task generator: the 32-bit hash ported to Python, every step of the four tasks (turn size, output size, kind, re-read);
- the engine, written again here from the spec (every call re-reads its whole context; a call over the window fails and stops the run;
  trimming caps tool outputs and keeps only facts the agent knew mattered; the last N main-agent turns; a summary every M steps that
  keeps each fact with p; notes; a sub-agent with its own context; re-reads), against the page over a grid of 560+ setups;
- the numbers quoted in the spec and on the page: parts 1 to 6, the four scenarios, the challenge (exactly 4 of 56 combinations pass,
  the minimal one), the capability layer, the drill, the teaser, and the Python in Connect (run here; it must print its comment);
- the cited source text, when the local copies are there (sources/b7src/);
- the page's own text: the part cards, the scenario cards and the challenge result read in the browser show these numbers.
Run: python3 tests/context-eng_numbers.py"""
import pathlib, subprocess, sys, io, contextlib, re, json
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'context-eng'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/context-eng.html', '--only', 'context-eng'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(OUT / 'numbers.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'context-eng.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'numbers.html')
SRC = P / 'sources' / 'b7src'
fails, n = [], 0


# 2026-10-09 (H1, H2): Show the math is typeset in chains (left, relation, right per row); read each chain row by row, spaces dropped
CHAINS = """()=>[...document.querySelectorAll('#cte-mathbody .cte-eqs')].map(q=>{const k=[...q.children], rows=[]; for(let i=0;i+2<k.length;i+=3) rows.push([k[i],k[i+1],k[i+2]].map(c=>c.textContent.replace(/\\s+/g,''))); return rows;})"""


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


def fI(x): return f'{int(x):,}'
def fB(x):
    for d, s in ((1e9, 'B'), (1e6, 'M')):
        if x >= d:
            v = x / d
            return (f'{v:.2f}' if v < 10 else (f'{v:.1f}' if v < 100 else f'{v:.0f}')) + s
    return fI(x)
def fM(x):
    """one unit for read and cost (R6.12): 0.37M below a million, 2.78M above"""
    return fB(round(x)) if x >= 1e6 else f'{x / 1e6:.2f}M'
def fD(x):
    """owner b7 point 3: the cost in US dollars at the assumed $3 per million input tokens, 2 significant digits"""
    d = x * 3 / 1e6
    return '$0' if d <= 0 else '$' + (str(round(d)) if d >= 10 else (f'{d:.1f}' if d >= 1 else f'{d:.2f}'))
def near_frac(q):
    """the page's “about half / a third / a quarter / a fifth”: the nearest 1/k"""
    return min((2, 3, 4, 5), key=lambda k: abs(1 / k - q))
def fP(x):
    if x <= 0: return '0%'
    if x >= 1: return '100%'
    v = x * 100
    return (f'{v:.1f}' if v < 10 else str(int(v + 0.5))) + '%'


# ---- the generator: the page's counter-based hash, ported ----
M32 = 0xFFFFFFFF
def imul(a, b): return ((a & M32) * (b & M32)) & M32
def mix(h):
    h &= M32; h ^= h >> 16; h = imul(h, 0x7feb352d); h ^= h >> 15; h = imul(h, 0x846ca68b); h ^= h >> 16; return h & M32
def u01(seed, i, k):
    h = mix(imul(seed, 0x9E3779B1) ^ 0x85ebca6b); h = mix(h ^ imul(i + 1, 0x27d4eb2f)); h = mix(h ^ imul(k + 1, 0x165667b1)); return h / 4294967296


# the constants of the spec (specs/context-eng.md, "The model")
K = dict(rr=120, fetch=150, write=60, S=2000, Hs=3000, R=800, cr=0.1, cw=1.25, usd=3)  # usd: the assumed input price, $ per million tokens (owner b7 point 3)   # cr, cw: cache read and write, times the input price
BASE = dict(task='code', W=131072, hist='all', N=10, M=20, trim=0, notes=0, sub=0, p=0.8, cap=1500)


def steps_of(T):
    st = [None]
    for i in range(1, T['n'] + 1):
        o = T['o0'] + int(u01(T['seed'], i, 1) * T['o1']); kind = 'msg' if T['conv'] else 'small'
        lg = T['long'].get(str(i))
        if lg: kind = lg[0]; o = lg[1] + int(u01(T['seed'], i, 2) * (lg[2] - lg[1]))
        st.append(dict(m=T['m0'] + int(u01(T['seed'], i, 0) * T['m1']), o=o, kind=kind, ref=T['refs'].get(str(i), 0)))
    return st


def sim(s, TS):
    """One run, written from the spec: returns where it stops, the largest context, the tokens read and each fact's chance."""
    s = dict(BASE, **s); T = TS[s['task']]; st = steps_of(T); W = s['W']
    use_sub = bool(s['sub']) and bool(T['sub']); sa, sb = (T['sub'][0], T['sub'][1]) if use_sub else (0, -1)
    trims = s['trim'] and not T['conv']
    osz = lambda i: min(st[i]['o'], s['cap']) if trims else st[i]['o']
    keeps = lambda f, i: f['known'] or not s['trim'] or T['conv'] or st[i]['o'] <= s['cap']
    S = dict(items=[], summ=0, read=0, peak=0, n=0, cost=0.0, hit=0, prev=None, uid=0, sumid=0)
    ctx = lambda: T['H'] + S['summ'] + sum(x['size'] for x in S['items'])
    has_step = lambda j: any(j in x['steps'] for x in S['items'])
    has_fact = lambda fid: any(fid in x['facts'] for x in S['items'])

    def pieces():
        # what a call sends, in order, each piece with its identity: the head, the current summary, the kept items
        return [('H', T['H'])] + ([(('S', S['sumid']), S['summ'])] if S['summ'] else []) + [(('I', x['uid']), x['size']) for x in S['items']]

    def call():
        c = ctx()
        if c > W: return False
        cur, h = pieces(), 0
        if S['prev'] is not None:
            for a, b_ in zip(cur, S['prev']):
                if a != b_: break
                h += a[1]
        S['prev'] = cur; S['hit'] += h; S['cost'] += K['cr'] * h + K['cw'] * (c - h)
        S['read'] += c; S['peak'] = max(S['peak'], c); S['n'] += 1; return True
    def item(**kw):
        S['uid'] += 1; return dict(uid=S['uid'], **kw)
    sub_read, sub_cost, sub_hit, over, turn, nre, nsum, nsub, inS, avail = 0, 0.0, 0, 0, 0, 0, 0, 0, {}, {}
    for i in range(1, T['n'] + 1):
        if use_sub and sa < i <= sb: continue
        turn += 1; x = st[i]; j = x['ref']
        if j and not has_step(j):
            if not call(): over = i; break
            nre += 1
            S['items'].append(item(turn=turn, size=K['rr'] + osz(j), steps={j}, facts={f['id'] for f in T['facts'] if f['at'] == j and keeps(f, j)}))
        extra, now = 0, {}
        for f in T['facts']:
            if f['need'] == i:
                fid = f['id']
                now[fid] = 1.0 if has_fact(fid) else (1.0 if s['notes'] and f['known'] else (s['p'] ** inS[fid] if inS.get(fid) else 0.0))
                if s['notes']: extra += K['fetch']
        if not call(): over = i; break
        avail.update(now)
        if s['notes']: extra += K['write'] * sum(1 for f in T['facts'] if f['at'] == i and f['known'])
        if use_sub and i == sa:
            # the sub-agent only appends: each call repeats the whole previous one (cached), the first writes everything
            sizes, sc = [], 0
            for k in range(sa, sb + 1): sizes.append(K['Hs'] + sc); sc += st[k]['m'] + osz(k)
            hits = [0] + sizes[:-1]
            sub_read = sum(sizes); nsub = len(sizes); sub_hit = sum(hits)
            sub_cost = sum(K['cr'] * hh + K['cw'] * (cc - hh) for cc, hh in zip(sizes, hits))
            S['read'] += sub_read; S['cost'] += sub_cost; S['hit'] += sub_hit
            S['items'].append(item(turn=turn, size=x['m'] + extra + K['R'], steps=set(range(sa, sb + 1)), facts={f['id'] for f in T['facts'] if sa <= f['at'] <= sb and f['known']}))
        else:
            S['items'].append(item(turn=turn, size=x['m'] + extra + (x['o'] if T['conv'] else osz(i)), steps={i}, facts={f['id'] for f in T['facts'] if f['at'] == i and keeps(f, i)}))
        if s['hist'] == 'last': S['items'] = [it for it in S['items'] if it['turn'] > turn - s['N']]
        if s['hist'] == 'sum' and i % s['M'] == 0 and i < T['n']:
            if not call(): over = i; break
            nsum += 1
            for f in T['facts']:
                if has_fact(f['id']) or inS.get(f['id']): inS[f['id']] = inS.get(f['id'], 0) + 1
            S['summ'] = K['S']; S['sumid'] += 1; S['items'] = []
    for f in T['facts']: avail.setdefault(f['id'], 0.0)
    allp = 1.0
    for f in T['facts']: allp *= avail[f['id']]
    sure = sum(1 for f in T['facts'] if avail[f['id']] >= 1)
    return dict(over=over, peak=S['peak'], read=S['read'], subRead=sub_read, cost=S['cost'], hit=S['hit'], subCost=sub_cost, subHit=sub_hit, nre=nre, nsum=nsum, nsub=nsub, ncalls=S['n'] + nsub,
                avail=avail, succ=0.0 if over else allp, sure=sure, nf=len(T['facts']))


def same(py, js):
    for k in ('over', 'peak', 'read', 'subRead', 'hit', 'subHit', 'nre', 'nsum', 'nsub', 'ncalls', 'sure', 'nf'):
        if py[k] != js[k]: return f'{k}: {py[k]} vs {js[k]}'
    for k in ('cost', 'subCost'):
        if abs(py[k] - js[k]) > 1e-6: return f'{k}: {py[k]} vs {js[k]}'
    for fid, v in py['avail'].items():
        if abs(v - js['avail'][fid]) > 1e-12: return f'avail {fid}: {v} vs {js["avail"][fid]}'
    if abs(py['succ'] - js['succ']) > 1e-12: return f'succ {py["succ"]} vs {js["succ"]}'
    return ''


CH_HISTS = [('all', 0), ('last', 5), ('last', 10), ('last', 20), ('last', 30), ('sum', 10), ('sum', 20)]
def ch_state(h, trim, notes, sub):
    k, v = CH_HISTS[h]
    return dict(BASE, task='ops', hist=k, N=v if k == 'last' else 10, M=v if k == 'sum' else 20, trim=trim, notes=notes, sub=sub)


with sync_playwright() as pw:
    b = pw.chromium.launch()
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.evaluate("()=>{location.hash='context-eng'}"); pg.wait_for_timeout(500)
    H = lambda expr, *a: pg.evaluate(expr, *a)
    C = H("()=>window.__cte.consts")
    TS = C['TASKS']
    check('hook: the constants of the spec', C['K'] == K and C['BASE'] == BASE, (C['K'], C['BASE']))
    code = TS['code']
    check('hook: the coding task (60 steps, head 5,000, facts at 3/12/27 needed at 48/52/56, the sub-task 21 to 32)',
          code['n'] == 60 and code['H'] == 5000 and [(f['at'], f['need']) for f in code['facts']] == [(3, 48), (12, 52), (27, 56)] and code['sub'] == [21, 32], code['facts'])
    ops = TS['ops']
    check('hook: the challenge task (80 steps, fact ③ at 48 not known to matter, needed at 66)',
          ops['n'] == 80 and [(f['at'], f['need'], f['known']) for f in ops['facts']] == [(2, 72, 1), (15, 61, 1), (48, 66, 0)], ops['facts'])

    # ---- 1. the generator ----
    for tid, T in TS.items():
        js = H(f"()=>window.__cte.steps('{tid}')")[1:]
        py = steps_of(T)[1:]
        check(f'generator: every step of the {tid} task', js == py, next(((i + 1, a, b2) for i, (a, b2) in enumerate(zip(py, js)) if a != b2), None))
    stc = steps_of(code)
    check('generator: the coding task’s files are 7,000 to 10,000 tokens, pages 9,000 to 13,000, model turns 150 to 449',
          all(7000 <= stc[i]['o'] < 10000 for i in (5, 9, 14, 37, 44, 50)) and all(9000 <= stc[i]['o'] < 13000 for i in (22, 24, 26, 28, 30)) and all(150 <= stc[i]['m'] <= 449 for i in range(1, 61)))
    # S1: the fidelity note's ranges cover every task's steps (model turns 150 to 600, short outputs 200 to 1,500, logs 4,000 to 8,000, files, pages and dashboards 6,000 to 13,000)
    rng_ok = []
    for tid, T in TS.items():
        for x in steps_of(T)[1:]:
            k = x['kind']
            rng_ok.append(150 <= x['m'] <= 600 and ((k in ('small', 'msg') and (T['conv'] or 200 <= x['o'] <= 1500)) or (k == 'log' and 4000 <= x['o'] <= 8000) or (k in ('file', 'page', 'dash') and 6000 <= x['o'] <= 13000)))
    check('generator: every step of every task lies inside the ranges the fidelity note gives', all(rng_ok), sum(1 for v in rng_ok if not v))

    # ---- 2. the engine against the page, over a grid ----
    grid = []
    for tid in TS:
        for hist, v in [('all', 10), ('last', 5), ('last', 10), ('last', 20), ('last', 30), ('sum', 10), ('sum', 20), ('sum', 30)]:
            for trim in (0, 1):
                for notes in (0, 1):
                    for sub in (0, 1):
                        grid.append(dict(task=tid, hist=hist, N=v if hist == 'last' else 10, M=v if hist == 'sum' else 20, trim=trim, notes=notes, sub=sub))
    for tid in TS:
        for extra in (dict(W=262144), dict(W=262144, hist='sum', M=10, p=0.5), dict(p=0.9, hist='sum', M=10, trim=1), dict(cap=500, trim=1), dict(cap=3000, trim=1, hist='last', N=20, notes=1), dict(p=1.0, hist='sum', M=30)):
            grid.append(dict(task=tid, **extra))
    js_all = H("(g)=>g.map(s=>window.__cte.run(s))", grid)
    bad = [(g, same(sim(g, TS), j)) for g, j in zip(grid, js_all)]
    bad = [x for x in bad if x[1]]
    check(f'engine: Python and the page agree on {len(grid)} setups', not bad, bad[:3])

    R = lambda **s: sim(s, TS)
    # ---- 3. the numbers of the spec ----
    b0 = R()
    check('part 1: keep everything, the context is over the 128K window at step 38 (the right answer “Between step 30 and 49”)', b0['over'] == 38 and C['PRED_RIGHT'] == 'c' and 30 <= b0['over'] <= 49, b0['over'])
    # owner b7 point 2 (R3.21): the card's facts, computed here from the generator, and the estimate they give lands in the same option
    lg = C['TASKS']['code']['long']; small = [stc[i]['m'] + stc[i]['o'] for i in range(1, 61) if str(i) not in lg]
    avg = sum(stc[i]['m'] + stc[i]['o'] for i in range(1, 61)) / 60; avgR = float(f'{avg:.2g}'); est = int((131072 - 5000) // avgR)
    pages = sorted(int(k) for k, v in lg.items() if v[0] == 'page')
    check(f'part 1 facts: head 5,000; {avg:.0f} a step on average, shown {avgR:,.0f}; most steps a few hundred to 2,000; {len(lg)} long outputs of 4,000 to 13,000; pages at {pages[0]} to {pages[-1]}',
          avgR == 2900 and min(small) >= 300 and max(small) < 2000 and len(lg) == 14 and min(v[1] for v in lg.values()) == 4000 and max(v[2] for v in lg.values()) == 13000 and pages == [22, 24, 26, 28, 30], (avg, min(small), max(small), len(lg)))
    check(f'part 1 facts: (131,072 − 5,000) ÷ 2,900 ≈ {est}; the estimate and the run (38) are both “Between step 30 and 49”', est == 43 and 30 <= est <= 49, est)
    check('part 1: the last call that fits sends 127,352 tokens; 2.28M read over 37 calls', b0['peak'] == 127352 and b0['read'] == 2275493 and b0['ncalls'] == 37, b0)
    t = R(trim=1); s10 = R(trim=1, hist='last'); l10 = R(hist='last')
    check('part 3: trimming fits, 82,361 at most (63%), all 3 facts, 2.73M read', t['over'] == 0 and t['peak'] == 82361 and fP(t['peak'] / 131072) == '63%' and t['sure'] == 3 and fB(t['read']) == '2.73M', t)
    check('part 3: trim + the last 10 steps, 20,726 at most, 1.12M read, no fact left, 3 re-reads', s10['peak'] == 20726 and fB(s10['read']) == '1.12M' and s10['sure'] == 0 and s10['succ'] == 0 and s10['nre'] == 3, s10)
    check('part 3: the last 10 steps without trimming: 68,812 at most, 2.28M read', l10['peak'] == 68812 and l10['sure'] == 0, l10)
    check('part 3 (cost): trimming costs 0.37M with caching; the sliding window reads 1.12M (less than half) yet costs 0.93M',
          fM(t['cost']) == '0.37M' and fM(s10['cost']) == '0.93M' and s10['read'] < t['read'] / 2 and s10['cost'] > 2 * t['cost'], (t['cost'], s10['cost']))
    b0c = R()
    check('part 1 (cost): keeping everything until the overflow, 94% of the tokens read come from the cache', round(b0c['hit'] / b0c['read'] * 100) == 94, b0c['hit'] / b0c['read'])
    sm = R(trim=1, hist='sum'); nt = R(trim=1, hist='last', notes=1)
    check('part 4: a summary every 20 steps, 64%, 64%, 80%; all three 32.8% (shown 33%)', [round(sm['avail'][k], 6) for k in 'abc'] == [0.64, 0.64, 0.8] and fP(sm['succ']) == '33%' and sm['nsum'] == 2, sm['avail'])
    check('part 4: the summary call is the largest context (34,959)', sm['peak'] == 34959, sm['peak'])
    check('part 4: notes keep all 3 for sure, 1.12M to 1.13M read', nt['sure'] == 3 and fB(s10['read']) == '1.12M' and fB(nt['read']) == '1.13M', nt)
    sn = R(trim=1, hist='sum', notes=1)
    check('part 4 (cost): a summary every 20 steps costs 0.24M (from 0.93M); with notes all 3 facts for sure, still 0.24M',
          fM(sm['cost']) == '0.24M' and sn['sure'] == 3 and fM(sn['cost']) == '0.24M' and sn['peak'] == 35019, (sm['cost'], sn['cost'], sn['peak']))
    sb = R(sub=1); nsb = R(trim=1, hist='last', notes=1, sub=1)
    check('part 5: a sub-agent fits, 114,927 at most (88%), the sub-agent reads 428,783', sb['over'] == 0 and sb['peak'] == 114927 and fP(sb['peak'] / 131072) == '88%' and sb['subRead'] == 428783 and sb['sure'] == 3, sb)
    check('part 5: added to part 4’s setup, 1.13M to 1.01M read', fB(nt['read']) == '1.13M' and fB(nsb['read']) == '1.01M', nsb)
    snb = R(trim=1, hist='sum', notes=1, sub=1)
    check('part 5 (cost): added to part 4’s last setup (summary + notes), the cost goes from 0.24M to 0.22M', fM(sn['cost']) == '0.24M' and fM(snb['cost']) == '0.22M', snb['cost'])
    big = R(W=262144); l5 = R(trim=1, hist='last', N=5)
    check('part 6: a 256K window fits, 179,718 at most, 5.93M read, about 5 times the notes setup', big['over'] == 0 and big['peak'] == 179718 and fB(big['read']) == '5.93M' and round(big['read'] / nt['read']) == 5, big)
    check('part 6: summaries keep a fact 80%, 64%, 51%', [fP(0.8 ** k) for k in (1, 2, 3)] == ['80%', '64%', '51%'])
    check('part 6 (cost): the 256K window costs 0.80M with caching, more than 2 times trimming alone (0.37M)', fM(big['cost']) == '0.80M' and int(big['cost'] // t['cost']) == 2, big['cost'] / t['cost'])
    check('part 6: the last 5 steps re-read 3 files', l5['nre'] == 3 and l5['sure'] == 0, l5)
    # owner b7 point 3 (R6.20): the cost in dollars, and the words that compare two costs
    check('dollars: trimming $1.1, the sliding window $2.8, a summary every 20 $0.71, with notes still $0.71, 256K $2.4',
          [fD(t['cost']), fD(s10['cost']), fD(sm['cost']), fD(sn['cost']), fD(big['cost'])] == ['$1.1', '$2.8', '$0.71', '$0.71', '$2.4'], [fD(x['cost']) for x in (t, s10, sm, sn, big)])
    check('words: the sliding window costs about 2.5 times trimming; a summary about a quarter of the sliding window; 256K more than 2 times trimming; a sub-agent lowers part 4’s cost a little',
          f"{s10['cost'] / t['cost']:.1f}" == '2.5' and near_frac(sm['cost'] / s10['cost']) == 4 and int(big['cost'] // t['cost']) == 2 and snb['cost'] < sn['cost'], (s10['cost'] / t['cost'], sm['cost'] / s10['cost']))
    # part 2: the shares at the last call that fits (step 37)
    calls = H("()=>window.__cte.run({}).calls.filter(c=>c.k!=='over')")
    last = calls[-1]
    st_items = sum(stc[i]['m'] + stc[i]['o'] for i in range(1, 37))
    check('part 2: the last call that fits is step 37, head + every earlier step', last['step'] == 37 and last['ctx'] == 5000 + st_items, (last['step'], last['ctx'], 5000 + st_items))
    tool = sum(stc[i]['o'] for i in range(1, 37)); conv = sum(stc[i]['m'] for i in range(1, 37))
    check(f'part 2: tool outputs {fP(tool / last["ctx"])}, conversation {fP(conv / last["ctx"])}, head {fP(5000 / last["ctx"])}', last['seg'] == [5000, 0, conv, tool], last['seg'])

    # ---- 4. the scenarios ----
    PR = {p['id']: p for p in C['PRESETS']}
    sc = {k: R(**{kk: vv for kk, vv in v.items() if kk != 'id'}) for k, v in PR.items()}
    check('scenario code: trim + a summary every 20 + notes, 35,019 at most, all 3 facts, 0.24M; kept whole it is full at 38',
          PR['code']['hist'] == 'sum' and PR['code']['M'] == 20 and sc['code']['peak'] == 35019 and sc['code']['sure'] == 3 and fM(sc['code']['cost']) == '0.24M' and b0['over'] == 38, sc['code'])
    rs_all = R(task='research')
    check('scenario research: sub-agent + summary 20 + notes, 84,001 at most, all 3; kept whole it is full at 41', sc['research']['peak'] == 84001 and sc['research']['sure'] == 3 and rs_all['over'] == 41, (sc['research'], rs_all['over']))
    ch_all = R(task='chat'); ch_n = R(task='chat', hist='last', N=20, notes=1); ch_s = R(task='chat', hist='sum')
    check('scenario chat: the last 20 turns lose the request; kept whole 63,908 at most; notes keep it', sc['chat']['sure'] == 0 and ch_all['peak'] == 63908 and ch_n['sure'] == 1, (sc['chat'], ch_all['peak']))
    check('scenario chat (cost): the last 20 turns cost 1.34M, about 3 times keeping everything (0.46M)', fM(sc['chat']['cost']) == '1.34M' and fM(ch_all['cost']) == '0.46M' and round(sc['chat']['cost'] / ch_all['cost']) == 3, (sc['chat']['cost'], ch_all['cost']))
    check('words: the coding scenario costs about a third of the 256K one, which costs more than 3 times it; the chat about 3 times keeping everything',
          near_frac(sc['code']['cost'] / sc['big']['cost']) == 3 and int(sc['big']['cost'] // sc['code']['cost']) == 3 and round(sc['chat']['cost'] / ch_all['cost']) == 3, sc['big']['cost'] / sc['code']['cost'])
    check('scenario big: 256K fits, 5.93M read, 179,718 for the last call, 0.80M with caching against 0.24M for the coding scenario',
          sc['big']['over'] == 0 and fB(sc['big']['read']) == '5.93M' and sc['big']['peak'] == 179718 and fM(sc['big']['cost']) == '0.80M' and fM(sc['code']['cost']) == '0.24M', sc['big'])

    # ---- 5. the challenge: exactly 4 of 56 pass, the minimal one ----
    allc = [(h, tr, no, su, R(**{k: v for k, v in ch_state(h, tr, no, su).items()})) for h in range(7) for tr in (0, 1) for no in (0, 1) for su in (0, 1)]
    ok = [(h, tr, no, su, r) for h, tr, no, su, r in allc if not r['over'] and r['sure'] == 3]
    check('challenge: exactly 4 of 56 combinations pass, none with trimming, all with notes, all with 20 or 30 raw steps',
          len(ok) == 4 and all(tr == 0 and no == 1 and h in (3, 4) for h, tr, no, su, r in ok), [(h, tr, no, su) for h, tr, no, su, r in ok])
    best = min(ok, key=lambda x: x[4]['cost'])
    check('challenge: the cheapest passing one (cost with caching) is the last 20 steps + notes + a sub-agent, 2.78M, 3.48M read, 63,631 at most; it also reads the fewest',
          best[:4] == (3, 0, 1, 1) and fM(best[4]['cost']) == '2.78M' and fB(best[4]['read']) == '3.48M' and best[4]['peak'] == 63631 and best[:4] == min(ok, key=lambda x: x[4]['read'])[:4], best)
    check('challenge: the page agrees on all 56 and on the cheapest one',
          H("()=>window.__cte.chAll().filter(e=>e.ok).length") == 4 and H("()=>{const b=window.__cte.chBest(); return [b.h,b.trim,b.notes,b.sub,b.read]}") == [3, 0, 1, 1, best[4]['read']] and abs(H("()=>window.__cte.chBest().cost") - best[4]['cost']) < 1e-6)
    st0 = R(**ch_state(0, 0, 0, 0)); n20 = R(**ch_state(3, 0, 1, 0)); t20 = R(**ch_state(3, 1, 1, 1)); s20 = R(**ch_state(6, 0, 1, 1))
    bc = best[4]['cost']
    check('challenge in words: the cheapest pick about $8.3; without the sub-agent about 1.6 times; trimming about half; a summary every 20 about a fifth',
          fD(bc) == '$8.3' and f"{n20['cost'] / bc:.1f}" == '1.6' and near_frac(t20['cost'] / bc) == 2 and near_frac(s20['cost'] / bc) == 5, (n20['cost'] / bc, t20['cost'] / bc, s20['cost'] / bc))
    ok_d = sorted(fD(r['cost']) for h, tr, no, su, r in allc if not r['over'] and r['sure'] == 3)
    check(f'challenge: the 4 passing combinations show different dollar values ({ok_d})', len(set(ok_d)) == len(ok_d) == 4, ok_d)
    check('challenge: the initial setup is full at step 45; 20 raw steps without a sub-agent cost 4.46M; with trimming 1.49M, but fact ③ lost',
          st0['over'] == 45 and fM(n20['cost']) == '4.46M' and fM(t20['cost']) == '1.49M' and t20['avail']['c'] == 0, (st0['over'], n20['cost'], t20['cost']))
    check('challenge: a summary every 20 steps + notes + sub-agent costs only 0.52M, but fact ③ survives with 80% only', fM(s20['cost']) == '0.52M' and abs(s20['avail']['c'] - 0.8) < 1e-12 and s20['sure'] == 2, (s20['cost'], s20['avail']))
    byk = {(h, tr, no, su): r for h, tr, no, su, r in allc}
    check('challenge: trimming always loses fact ③; notes never change its chance', all(r['avail']['c'] == 0 for h, tr, no, su, r in allc if tr) and all(byk[(h, tr, 1, su)]['avail']['c'] == byk[(h, tr, 0, su)]['avail']['c'] for h in range(7) for tr in (0, 1) for su in (0, 1) if not byk[(h, tr, 1, su)]['over'] and not byk[(h, tr, 0, su)]['over']))

    # ---- 6. the capability layer, the drill ----
    q = H("()=>window.__cte.quad()")
    ctx_, r60, r120 = 5000, 0, 0
    for k in range(1, 121):
        x = stc[(k - 1) % 60 + 1]
        if k <= 60: r60 += ctx_
        r120 += ctx_; ctx_ += x['m'] + x['o']
    check(f'exits: the 120-step task reads {fB(r120)}, {r120 / r60:.1f} times the 60-step {fB(r60)} (about four times)', q['r60'] == r60 == big['read'] and q['r120'] == r120 and 3.5 < r120 / r60 < 4.5, (q, r60, r120))
    check('exits: a chat summarized every 20 turns keeps the request 0.8^5 = 32.8%, about 33%', abs(ch_s['avail']['a'] - 0.8 ** 5) < 1e-12 and fP(ch_s['avail']['a']) == '33%' and ch_s['nsum'] == 5, ch_s)
    for seed in range(1, 31):
        d = H(f"()=>window.__cte.drill({seed})")
        pk = lambda a, k: a[int(u01(seed + 900, 1, k) * len(a))]
        Hh, dd, nn = pk([2000, 3000, 4000, 5000, 6000], 0), pk([800, 1000, 1500, 2000, 2500, 3000], 1), pk([20, 30, 40, 50, 60], 2)
        tot = sum(Hh + dd * (i - 1) for i in range(1, nn + 1))
        if not (d['H'] == Hh and d['d'] == dd and d['n'] == nn and d['total'] == tot and d['last'] == Hh + dd * (nn - 1)):
            check(f'drill seed {seed}', False, (d, Hh, dd, nn, tot)); break
    else: check('drill: 30 seeds, the total is the sum of the calls and the last context is H + (n − 1) d', True)

    # ---- 7. the teaser ----
    tz = H("()=>!!window.__tzCte")
    check('teaser hook present', tz)
    if tz:
        c = H("()=>window.__tzCte.consts")
        ctx_all = lambda k: 2000 + 3000 * k
        over_all = next(k for k in range(1, 41) if ctx_all(k) > 65536)
        check('teaser: the toy run (40 steps, 3,000 a step, 64K, the fact at step 2 needed at step 36)', c['N'] == 40 and c['PER'] == 3000 and c['W'] == 65536 and c['FACT'] == 2 and c['NEED'] == 36)
        check(f'teaser: keep everything is over at step {over_all} (not the lab’s part 1 answer)', H("()=>window.__tzCte.overAt('all')") == over_all == 22 and over_all != b0['over'])
        check('teaser: the last 10 steps stay at 32,000 and lose the fact', H("()=>[window.__tzCte.ctxAt('last',36),window.__tzCte.overAt('last'),window.__tzCte.factOk('last')]") == [32000, 0, False])
        check('teaser: summary + notes send 22,150 at step 36 and keep the fact', H("()=>[window.__tzCte.ctxAt('notes',36),window.__tzCte.overAt('notes'),window.__tzCte.factOk('notes')]") == [2000 + 2000 + 6 * 3000 + 150, 0, True])
        for pid, txt in (('all', 'step 22'), ('last', '32,000'), ('notes', '22,150')):
            rd = H(f"()=>window.__tzCte.read('{pid}')")
            words = len(re.sub('<[^>]+>', '', rd).split())
            check(f'teaser read {pid}: shows {txt}, at most 25 words ({words})', txt in rd and words <= 25, rd)

    # ---- 8. the Python in Connect ----
    src = C['CODE']
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): exec(compile(src, 'connect', 'exec'), {})
    out = buf.getvalue().strip()
    check('connect: the Python runs and prints its comment (2 7)', out == '2 7' and src.rstrip().endswith('# 2 7'), out)

    # ---- 9. the cited sources (local copies) ----
    lm = SRC / 'litm_readme.md'
    if lm.exists(): check('source: Lost in the Middle is arXiv 2307.03172', '2307.03172' in lm.read_text())
    sw = SRC / 'sweagent_readme.md'
    if sw.exists(): check('source: SWE-agent is arXiv 2405.15793', '2405.15793' in sw.read_text())
    mi = SRC / 'mini_readme.md'
    if mi.exists(): check('source: mini-swe-agent keeps a linear history', 'completely linear history' in mi.read_text())
    hp = SRC / 'swe_history_processors.py'
    if hp.exists():
        t_ = hp.read_text()
        check('source: LastNObservations, “Old environment output: (n lines omitted)”, 5 in the paper, breaks prompt caching',
              'class LastNObservations' in t_ and 'Old environment output: (n lines omitted)' in t_ and 'last 5 observations' in t_ and 'break prompt caching' in t_)
    else: print('note: sources/b7src/swe_history_processors.py not found; the SWE-agent history processor check is skipped (copy in out/context-eng/src/)')

    # ---- 10. the page's text shows these numbers (EN) ----
    def obs(): return H("()=>{const o=document.querySelector('#cte-rail [data-obs]'); return o&&!o.hidden?o.innerText:''}")
    def go(i): H(f"()=>document.querySelector('#cte-rail [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(200)
    def end(): H("()=>document.querySelector('#cte-player [data-act=\"end\"]').click()"); pg.wait_for_timeout(150)
    def lever(v): H(f"()=>document.querySelector('#cte-rail [data-lever=\"{v}\"]').click()"); pg.wait_for_timeout(150); end()
    H("()=>document.querySelector('#cte-rail [data-pred=\"c\"]').click()"); pg.wait_for_timeout(100); end()
    over_ctx = H("()=>window.__cte.run({}).calls.pop().ctx")
    check('page part 1: the over call is step 38 with its size, and the verdict', f'At step 38 the context would be {fI(over_ctx)} tokens' in obs() and 'was right' in obs() and over_ctx > 131072, obs())
    card = H("()=>document.querySelector('#cte-rail .step.current').innerText")
    check('page part 1: the card gives the facts (5,000; 2,900 on average; 14 of the 60 steps; 4,000 to 13,000)', all(x in card for x in ('about 5,000 tokens', 'about 2,900 tokens on average', '14 of the 60 steps', '4,000 to 13,000')), card[:600])
    check('page part 1: the reveal ties back to them (about 43 steps fit; the pages in steps 22 to 30)', 'About 43 steps fit' in obs() and 'steps 22 to 30' in obs() and H("()=>!!document.querySelector('#cte-rail [data-obs] math')"), obs())
    go(1); lever('tool')
    check('page part 2: the tool share', fP(tool / last['ctx']) in obs() and fI(last['ctx']) in obs(), obs())
    go(2); lever('trim')
    check('page part 3 trim: 82,361 and 63%', '82,361' in obs() and '63%' in obs(), obs())
    lever('slide')
    check('page part 3 slide: 20,726, less than half the tokens read, about 2.5 times the cost, 3 re-read files, no exact cost', all(x in obs() for x in ('20,726', 'less than half', 'about 2.5 times as much as trimming alone', '3 old files')) and not re.search(r'\d\.\d\dM', obs()), obs())
    go(3); lever('sum')
    check('page part 4 summary: 64%, 64% and 80%, all three 33%, about a quarter of the sliding window’s cost', '64%, 64% and 80%' in obs() and '33%' in obs() and 'about a quarter of the sliding window' in obs() and not re.search(r'\d\.\d\dM', obs()), obs())
    lever('notes')
    check('page part 4 notes: all 3 for sure, the cost barely moves', 'for sure' in obs() and 'barely moves' in obs(), obs())
    go(4); lever('sub')
    check('page part 5: 114,927, 88%, the sub-agent lowers part 4’s cost a little, no exact value', all(x in obs() for x in ('114,927', '88%', 'also lowers the cost a little')) and '428,783' not in obs() and not re.search(r'\d\.\d\dM', obs()), obs())
    go(5); end(); prod = H("()=>document.querySelector('#cte-rail .step.current').innerText")
    links = H("()=>[...document.querySelectorAll('#cte-rail .step.current .cte-srcs a')].map(a=>a.href)")
    check('page part 6 (owner point 4): Codex and Claude Code, five points, each with its sources', all(x in prod for x in ('Claude Code', 'Codex', '90% of the window', 'handoff summary', 'The Σ call')) and len(links) >= 9 and all(('code.claude.com/docs' in u) or ('github.com/openai/codex' in u) or ('openai.com/index' in u) for u in links), (prod[:400], links))
    go(6); end(); rem = H("()=>document.querySelector('#cte-rail .step.current').innerText")
    # the owner, 2026-10-08 (G2): the three points state the pattern, not the numbers of parts 2 to 4 (63%, less than half, about 2.5
    # times, 33%, barely moved); the misconceptions keep their computed numbers
    check('page part 7 (G2): the points state the pattern; the misconceptions keep more than 2 times, 80% 64% 51%', all(x in rem for x in ('more than 2 times', '80%', '64%', '51%', 'misses the cache'))
          and not any(x in rem for x in ('In part', 'less than half', 'about 2.5 times as much', 'barely moved')) and not re.search(r'\d\.\d\dM', rem), rem)
    # metrics and the math of part 6's setup (trim, a summary every 20 steps, notes)
    met = H("()=>[...document.querySelectorAll('#cte-metrics .metric .v')].map(e=>e.textContent)")
    check('page metrics: largest context, facts, cost in dollars, read, calls', met == [fI(sn['peak']), '3 / 3', fD(sn['cost']), fM(sn['read']), fI(sn['ncalls'])], (met, sn))
    mt = H("()=>document.querySelector('#cte-mathbody').textContent")
    # 2026-10-09 (H1, H2): the dollars are a typeset chain, “cost in dollars = N × 3/1,000,000 ≈ 3.70”; read it row by row
    cc = next((rows for rows in H(CHAINS) if rows and rows[0][0] in ('costindollars', '花费（美元）')), None)
    check('page math: the exact total read and cost, then the dollars', fI(sn['read']) in mt and fI(round(sn['cost'])) in mt and bool(cc) and len(cc) == 2
          and cc[0][1] == '=' and cc[0][2] == f"{fI(round(sn['cost']))}×31,000,000" and cc[1][1] == '≈' and cc[1][2] == fD(sn['cost']).replace('$', ''), (cc, mt[-400:]))
    # scenarios
    H("()=>document.querySelector('#cte-app .modes button[data-mode=\"sandbox\"]:not([data-flow=free])').click()"); pg.wait_for_timeout(300)
    exp = {'code': [fI(sc['code']['peak']), 'step 38', 'about a third of'], 'research': [fI(sc['research']['peak']), 'step 41'], 'chat': ['turn 115', fI(ch_all['peak']), 'about 3 times as much'],
           'big': ['more than 3 times as much', fI(sc['big']['peak'])]}
    for i, pid in enumerate(['code', 'research', 'chat', 'big']):
        H(f"()=>document.querySelector('#cte-rail [data-preset=\"{i}\"]').click()"); pg.wait_for_timeout(150)
        ins = H("()=>document.querySelector('#cte-insight').innerText")
        check(f'page scenario {pid}: {exp[pid]}', all(x in ins for x in exp[pid]), ins)
    # the challenge: the minimal pick and the full chart
    H("()=>document.querySelector('#cte-app .modes button[data-mode=\"challenge\"]').click()"); pg.wait_for_timeout(300)
    for k, v in (('chH', 3), ('chN', 1), ('chS', 1)):
        H(f"()=>document.querySelector('#cte-rail .seg-ctl[data-key=\"{k}\"] button[data-v=\"{v}\"]').click()"); pg.wait_for_timeout(120)
    H("()=>document.querySelector('#cte-rail [data-act=\"submit\"]').click()"); pg.wait_for_timeout(250)
    res = H("()=>document.querySelector('#cte-ch-res').innerText")
    check(f'page challenge: the cheapest pick, 63,631 and {fD(bc)}', '63,631' in res and f'about {fD(bc)}' in res and 'no passing combination costs less' in res, res)
    check(f'page challenge: the chart in dollars ({fD(st0["cost"])} full at step 45) and the comparisons in words', f'{fD(st0["cost"])}, full at step 45' in res and all(x in res for x in ('about 1.6 times as much', 'about half as much', 'about a fifth as much')) and not re.search(r'\d\.\d\dM', res), res)
    check('page challenge: one unit in the chart (every value in M, R6.12)', not re.search(r'\b\d{3},\d{3}\b(?!.*tokens)', H("()=>[...document.querySelectorAll('#cte-ch-res .cmp-row')].map(e=>e.innerText).join(' | ')")), H("()=>[...document.querySelectorAll('#cte-ch-res .cmp-row')].map(e=>e.innerText).join(' | ')"))
    check('page challenge: 9 rows at most in the chart', H("()=>document.querySelectorAll('#cte-ch-res .cmp-row').length") <= 9)
    # the brief chart (the owner's review of batch 8): the 4 passing combinations and the initial setup, in the engine's order, nothing else
    labels = H("()=>[...document.querySelectorAll('#cte-ch-res .cmp:not(details .cmp) .cmp-row')].map(r=>r.innerText.replace(/\\s+/g,' '))")
    jall = H("()=>window.__cte.chAll()")
    want = [x for x in jall if x['ok'] or (x['h'], x['trim'], x['notes'], x['sub']) == (0, 0, 0, 0)]
    check('page challenge: the chart holds the 4 passing combinations and the initial setup, in order, with their dollars', len(labels) == len(want) == 5
          and all(fD(x['cost']) in l for x, l in zip(want, labels)), (labels, [fD(x['cost']) for x in want]))
    note = H("()=>{const n=document.querySelector('#cte-ch-res .cmp-note'); return n?n.innerText:''}")
    check('page challenge: the note counts the 51 combinations left out', 'The other 51 combinations fail at least one target' in note, note)
    check('no page errors', not errs, errs[:3])
    b.close()

print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
