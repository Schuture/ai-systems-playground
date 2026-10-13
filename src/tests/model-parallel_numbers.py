"""Numbers test for the model-parallel lab (CONTRACT §6): an independent Python recomputation of every number the lab shows, compared
with the page's read-only hook window.__mp (and the teaser's window.__tzMp).

Recomputed here from the stated assumptions, not copied from the lab:
- parameters per layer from the config.json shapes (70B: 855,654,400) and the totals of app.js (70B 70,553,706,496);
- model states at 18 bytes per parameter (70B: 1.27 TB), ZeRO-1 6 + 12/d (integrator decision, Megatron-LM dist_optimizer.md);
- the bubble (p − 1)/(m + p − 1) for p 8: 88%, 47%, 18%, 10% (zero-bubble README form); the teaser p 4, m 16: 16%;
- a separate event simulation of GPipe and of Megatron's non-interleaved 1F1B (stage i: p − i − 1 warm-up forwards), forward 1 unit,
  backward 2 units: the step length, the idle share and the micro-batches GPU 1 holds at once (GPipe m, 1F1B min(m, p));
- one 70B layer at 4,096 tokens: 53.1 ms on one GPU; TP 8: 6.64 ms compute, 4 all-reduces of 67.1 MB: NVLink 1.04 ms (14%),
  InfiniBand 9.40 ms (59%); the activations of 10 layers per micro-batch: 10.7 GB;
- part 6 (TP 8 × PP 4 × DP 2, 128 micro-batches): 37.2 GB per GPU, bubble 2.3%, TP 13%;
- the four scenarios: 169 GB (does not fit), 67.0 GB, bubble 32% for Llama 3's layout, 179 GB at 128K tokens (does not fit);
- the challenge: 57 layouts, 6 fit, the fastest 37.2 s, minimal (within 1%) TP 4 × PP 4 × DP 2 × 128 and TP 4 × PP 8 × 256; the
  initial setup TP 8 × PP 1 × 8 does not fit (165 GB);
- G7 (Codex on b9 preview v5): every p, t and m the task card offers (57 layouts of the 96 combinations): the single change the result
  names, recomputed with the task card's own rule for m; a layout that fits gets a single change only when the step time the card shows
  drops (PP 2 × TP 8 × 128 at 33.2 s: PP 4 alone gives 33.3 s, so none); the sentence in EN and ZH shows the step before and after as
  the card shows them, never “about 0 s”; a change that makes a layout fit really fits and its memory and step time are named; when no
  single change helps, the sentence says so and names every setting the fastest layout changes, and only those;
- the transfer questions' answers and the seeded drill.
Run: python3 tests/model-parallel_numbers.py"""
import pathlib, subprocess, sys, math
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/model-parallel.html', '--only', 'model-parallel'], check=True, cwd=P, stdout=subprocess.DEVNULL)
OUT = P / 'out' / 'model-parallel'; OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'wrapped-numbers.html').write_text('<!doctype html><html><head><meta charset="utf-8"><style>[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'model-parallel.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-numbers.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


close = lambda a, b, r=1e-9: abs(a - b) <= r * max(1.0, abs(a), abs(b))
# ---------- the lab's stated assumptions ----------
# SHARE: the share of the peak a GPU reaches while computing (integrator decision 1: one name, default 50%; MFU only as a result).
# NODE: one 8-GPU machine's network bandwidth (8 × 400 Gb/s); IB: one GPU's own port, the machine's 400 GB/s shared by its 8 GPUs.
PEAK, SHARE, HBM, NV, IB, NODE, PER_NODE, SEQ = 989.5e12, 0.50, 80e9, 450e9, 50e9, 400e9, 8, 4096
SHAPES = {'8b': dict(N=8030261248, L=32, h=4096, I=14336, kv=8, hd=128),
          '70b': dict(N=70553706496, L=80, h=8192, I=28672, kv=8, hd=128),
          '405b': dict(N=405853388800, L=126, h=16384, I=53248, kv=8, hd=128)}


def per_layer(m):  # q and o: h·h each; k and v: h·(kv·hd) each; gate, up, down: h·I each; two RMSNorm weights: h each
    return 2 * m['h'] * m['h'] + 2 * m['h'] * m['kv'] * m['hd'] + 3 * m['h'] * m['I'] + 2 * m['h']


def config(model, t, p, d, m, b=1, seq=SEQ, place='in', zero=0, sched='1f1b'):
    sh = SHAPES[model]; Pl = per_layer(sh); tokens = b * seq; layers = sh['L'] / p
    compute = 6 * Pl * tokens * layers / t / (PEAK * SHARE)
    ar_bytes = tokens * sh['h'] * 2
    ar = 2 * (t - 1) / t * ar_bytes / (IB if place == 'out' else NV) if t > 1 else 0.0
    comm = 4 * layers * ar
    pipe = (m + p - 1) * (compute + comm)
    shard = sh['N'] / (t * p)
    # data parallel (integrator decisions 8 and 10): gradients move in BF16, 2 bytes per value. One machine (t·d ≤ 8): a ring over
    # NVLink, 2(d − 1)/d × (2 bytes × N/(t·p)) ÷ 450 GB/s. Across machines (the scaling book): an all-reduce of an array of B bytes
    # takes 2 × B ÷ 400 GB/s, B the whole array = the stage's gradient, 2 bytes × N/p.
    if d == 1: dp = 0.0
    elif t * d <= PER_NODE: dp = 2 * (d - 1) / d * 2 * shard / NV
    else: dp = 2 * (2 * sh['N'] / p) / NODE
    bpp = 6 + 12 / d if zero else 18
    held = m if sched == 'gpipe' else min(m, p)
    act_mb = layers * tokens * sh['h'] * (18 + 4 * sh['I'] / sh['h']) / t
    mem = bpp * shard + held * act_mb
    step = pipe + dp
    return dict(compute=compute, comm=comm, ar=ar, ar_bytes=ar_bytes, pipe=pipe, dp=dp, step=step, mem=mem, states=bpp * shard, act=held * act_mb,
                act_mb=act_mb, held=held, bubble=(p - 1) / (m + p - 1), tp_share=m * comm / step, fits=mem <= HBM)


def schedule(p, m, sched):
    """Own event simulation: each stage runs its list of operations in order, each as soon as its input is ready."""
    order = []
    for i in range(p):
        if sched == 'gpipe':
            o = [('F', j) for j in range(m)] + [('B', j) for j in range(m)]
        else:
            w = min(p - i - 1, m); o = [('F', j) for j in range(w)]
            for k in range(m - w): o += [('F', w + k), ('B', k)]
            o += [('B', j) for j in range(m - w, m)]
        order.append(o)
    end, free, rows = {}, [0] * p, [[] for _ in range(p)]
    pending = [list(o) for o in order]
    while any(pending):
        moved = False
        for i in range(p):
            while pending[i]:
                k, j = pending[i][0]
                dep = (('F', i - 1, j) if i > 0 else None) if k == 'F' else (('F', i, j) if i == p - 1 else ('B', i + 1, j))
                if dep is not None and dep not in end: break
                s = max(free[i], end.get(dep, 0)); e = s + (1 if k == 'F' else 2)
                end[(k, i, j)] = free[i] = e; rows[i].append((k, j, s, e)); pending[i].pop(0); moved = True
        assert moved, 'deadlock'
    T = max(free)
    ev = sorted([(s, 1) for k, j, s, e in rows[0] if k == 'F'] + [(e, -1) for k, j, s, e in rows[0] if k == 'B'])
    cur = peak = 0
    for _, dlt in ev: cur += dlt; peak = max(peak, cur)
    busy = sum(e - s for r in rows for _, _, s, e in r)
    return dict(T=T, peak=peak, idle=1 - busy / (p * T))


pct = lambda x: f'{x * 100:.1f}%' if x < 0.1 else f'{x * 100:.0f}%'
with sync_playwright() as pw:
    br = pw.chromium.launch(); pg = br.new_page()
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300); pg.evaluate("()=>{location.hash='model-parallel'}"); pg.wait_for_timeout(400)
    H = lambda js: pg.evaluate(js)
    K = H("()=>window.__mp.consts")
    # ---------- constants ----------
    check('GPU constants', K['GPU']['peak'] == PEAK and K['GPU']['mem'] == HBM and K['GPU']['nv'] == NV and K['GPU']['ib'] == IB and K['GPU']['node'] == NODE and K['SHARE'] == SHARE and K['SEQ'] == SEQ, K['GPU'])
    for k, sh in SHAPES.items():
        check(f'shape {k}', all(K['MD'][k][x] == sh[x] for x in sh), K['MD'][k])
    check('70B per layer 855,654,400', per_layer(SHAPES['70b']) == 855654400)
    check('70B layers hold 68.5B of 70.6B', round(80 * per_layer(SHAPES['70b']) / 1e9, 1) == 68.5 and round(SHAPES['70b']['N'] / 1e9, 1) == 70.6)
    check('70B model states 1.27 TB', f"{18 * SHAPES['70b']['N'] / 1e12:.2f}" == '1.27')
    check('part order', K['STEP_IDS'] == ['mission', 'gpipe', '1f1b', 'tp', 'link', '3d', 'remember'], K['STEP_IDS'])
    check('prediction answer: only 1', K['PRED_RIGHT'] == 'one')
    # owner b8 (1): the old Fast (8 s) is the new Medium, the old Medium (20 s) the new Slow, and Fast is 1.5 to 2 times Medium's pace
    check('replay speeds: slow 20 s, medium 8 s (the default), fast 1.5–2× medium', K['DUR']['slow'] == 20000 and K['DUR']['mid'] == 8000 and 1.5 <= K['DUR']['mid'] / K['DUR']['fast'] <= 2, K['DUR'])
    # ---------- the bubble ----------
    for m, want in [(1, '88%'), (8, '47%'), (32, '18%'), (64, '9.9%')]:
        b = 7 / (m + 7); check(f'bubble p 8, m {m}: {want}', pct(b) == want, pct(b))
    # ---------- schedules: our own simulation against the page's ----------
    for p_, m_ in [(8, 1), (8, 8), (8, 16), (8, 32), (4, 7), (4, 16), (16, 32), (2, 3)]:
        for sch in ('gpipe', '1f1b'):
            mine, page = schedule(p_, m_, sch), H(f"()=>window.__mp.sim({p_},{m_},'{sch}')")
            check(f'sim {sch} p {p_} m {m_}: length, peak, idle', mine['T'] == page['T'] and mine['peak'] == page['peak'] and close(mine['idle'], page['idle'], 1e-9),
                  (mine, {k: page[k] for k in ('T', 'peak', 'idle')}))
            check(f'sim {sch} p {p_} m {m_}: step (m + p − 1) × 3, idle = bubble', mine['T'] == (m_ + p_ - 1) * 3 and close(mine['idle'], (p_ - 1) / (m_ + p_ - 1), 1e-9), mine)
            check(f'sim {sch} p {p_} m {m_}: GPU 1 holds {"m" if sch == "gpipe" else "min(m, p)"}', mine['peak'] == (m_ if sch == 'gpipe' else min(m_, p_)), mine)
    # ---------- one layer (parts 4 and 5) ----------
    l1, l8, l8o = config('70b', 1, 80, 1, 1), config('70b', 8, 80, 1, 1), config('70b', 8, 80, 1, 1, place='out')
    check('one layer on one GPU: 42.5 ms (50% of the peak)', f"{l1['compute'] * 1e3:.1f}" == '42.5', l1['compute'])
    check('TP 8: 5.31 ms of compute', f"{l8['compute'] * 1e3:.2f}" == '5.31', l8['compute'])
    check('all-reduce 67.1 MB', f"{l8['ar_bytes'] / 1e6:.1f}" == '67.1')
    check('TP 8 NVLink: 4 all-reduces 1.04 ms, 16%', f"{l8['comm'] * 1e3:.2f}" == '1.04' and pct(l8['comm'] / (l8['compute'] + l8['comm'])) == '16%')
    check('TP 8 InfiniBand: 9.40 ms, 64%, longer than the compute', f"{l8o['comm'] * 1e3:.2f}" == '9.40' and pct(l8o['comm'] / (l8o['compute'] + l8o['comm'])) == '64%' and l8o['comm'] > l8o['compute'])
    sh5 = l8o['comm'] / (l8o['compute'] + l8o['comm'])
    check('part 5 sentence: 9 times as long, "nearly two thirds"', round(l8o['comm'] / l8['comm']) == 9 and 0.6 <= sh5 < 2 / 3, (l8o['comm'] / l8['comm'], sh5))
    check('part 1: 8 stages alone still hold 159 GB of model states per GPU', f"{config('70b', 1, 8, 1, 1)['states'] / 1e9:.0f}" == '159')
    check('fidelity: embedding table and LM head about 3% of 70B', round(2 * 128256 * 8192 / SHAPES['70b']['N'] * 100) == 3 and K['MD']['70b'].get('V') == 128256, K['MD']['70b'].get('V'))
    for t in (1, 2, 4, 8):
        for pl in ('in', 'out'):
            mine, page = config('70b', t, 80, 1, 1, place=pl), H(f"()=>window.__mp.layer({t},'{pl}')")
            check(f'layer t {t} {pl}', close(mine['compute'], page['comp']) and close(mine['comm'], page['comm']) and close(mine['comm'] / (mine['compute'] + mine['comm']), page['share']), (mine['compute'], mine['comm'], page))
    # ---------- part 3: activations of 10 layers per micro-batch ----------
    c3 = config('70b', 1, 8, 1, 16, sched='gpipe')
    check('part 3: one micro-batch on 10 layers 10.7 GB; GPipe 16, 1F1B 8', f"{c3['act_mb'] / 1e9:.1f}" == '10.7' and c3['held'] == 16 and config('70b', 1, 8, 1, 16)['held'] == 8)
    # ---------- full configurations against the page ----------
    def same(name, **c):
        mine = config(**c); args = dict(c); args.setdefault('b', 1); args.setdefault('seq', SEQ); args.setdefault('place', 'in'); args.setdefault('zero', 0); args.setdefault('sched', '1f1b')
        js = '{' + ','.join(f"{k}:{repr(v) if isinstance(v, str) else v}" for k, v in args.items()) + '}'
        page = H(f"()=>window.__mp.cost({js})")
        ok = all(close(mine[a], page[b_]) for a, b_ in [('mem', 'mem'), ('step', 'step'), ('bubble', 'bubble'), ('tp_share', 'tpShare'), ('dp', 'dp'), ('states', 'states'), ('act', 'act')]) and mine['fits'] == page['fits']
        check(f'{name}: page = recomputation', ok, ({k: mine[k] for k in ('mem', 'step', 'tp_share')}, {k: page[k] for k in ('mem', 'step', 'tpShare')}))
        return mine
    c6 = same('part 6', model='70b', t=8, p=4, d=2, m=128, zero=1)
    check('part 6: 37.2 GB, bubble 2.3%, TP 16%', f"{c6['mem'] / 1e9:.1f}" == '37.2' and pct(c6['bubble']) == '2.3%' and pct(c6['tp_share']) == '16%', c6)
    s1 = same('scenario one machine', model='70b', t=8, p=1, d=1, m=16, zero=1)
    s2 = same('scenario 2D', model='70b', t=8, p=1, d=32, m=16, zero=1)
    s3 = same('scenario Llama 3', model='405b', t=8, p=16, d=128, m=32, zero=1)
    s4 = same('scenario 128K', model='405b', t=8, p=8, d=8, m=1, seq=131072, zero=1)
    check('scenario one machine: 169 GB, does not fit', f"{s1['mem'] / 1e9:.0f}" == '169' and not s1['fits'], s1['mem'])
    check('scenario 2D: 67.0 GB, fits', f"{s2['mem'] / 1e9:.1f}" == '67.0' and s2['fits'], s2['mem'])
    check('scenario Llama 3: bubble 32%, fits', pct(s3['bubble']) == '32%' and s3['fits'], (s3['bubble'], s3['mem']))
    check('scenario 128K: 179 GB, does not fit', f"{s4['mem'] / 1e9:.0f}" == '179' and not s4['fits'], s4['mem'])
    pre = {x['id']: x for x in K['PRESETS']}
    check('presets carry these layouts', [pre[k]['t'] * pre[k]['p'] * pre[k]['d'] for k in ('one', '2d', 'llama3', 'long')] == [8, 256, 16384, 512] and pre['long']['seq'] == 131072, pre)
    same('sandbox 8B, GPipe, across machines', model='8b', t=4, p=2, d=4, m=8, place='out', sched='gpipe')
    # ---------- the challenge ----------
    rows = []
    for t in (1, 2, 4, 8):
        for p_ in (1, 2, 4, 8):
            for m_ in (8, 16, 32, 64, 128, 256):
                d = 32 / (t * p_)
                if d != int(d) or d < 1: continue
                b = 256 / (d * m_)
                if b != int(b) or b < 1: continue
                r = config('70b', t, p_, int(d), m_, b=int(b), zero=1); rows.append((t, p_, m_, int(d), int(b), r))
    fit = [x for x in rows if x[5]['fits']]; best = min(x[5]['step'] for x in fit)
    minimal = sorted((x[0], x[1], x[2]) for x in fit if x[5]['step'] <= best * 1.01)
    page = H("()=>window.__mp.chAll()")
    check('challenge: 57 layouts, 6 fit', len(rows) == 57 and len(fit) == 6 and len(page) == 57 and sum(1 for x in page if x['fits']) == 6, (len(rows), len(fit), len(page)))
    check('challenge: fastest 30.2 s', f'{best:.1f}' == '30.2', best)
    check('challenge: minimal TP 4 × PP 4 × 128 and TP 4 × PP 8 × 256', minimal == [(4, 4, 128), (4, 8, 256)] and sorted((x['t'], x['p'], x['m']) for x in page if x['min']) == minimal, minimal)
    pmap = {(x['t'], x['p'], x['m']): x for x in page}
    check('challenge: every layout matches', all(close(r['step'], pmap[(t, p_, m_)]['step']) and close(r['mem'], pmap[(t, p_, m_)]['mem']) for t, p_, m_, d, b, r in rows))
    tp8 = [x[5]['step'] for x in fit if x[0] == 8]
    check('challenge: TP 8 layouts that fit are 9% slower or more', tp8 and min(tp8) > best * 1.08, tp8)
    st = config('70b', 8, 1, 4, 8, b=8, zero=1)
    check('challenge: initial setup TP 8 × PP 1 × 8 does not fit (165 GB)', not st['fits'] and f"{st['mem'] / 1e9:.0f}" == '165', st['mem'])
    # R1 (owner b8) and G7 (Codex on b9 preview v5): the single change the result names, recomputed. The task card's single changes: a
    # new t or p keeps m when the batch still divides, else the card takes the nearest valid m (by ratio, the smaller on a tie); a new m
    # alone. A layout that does not fit: of the changes that make it fit, the shortest step. A layout that fits but misses the goal:
    # the change with the shortest step that fits, and only when the step time the task card shows (one decimal) gets strictly
    # shorter; otherwise none, and the sentence says that no single change helps and names what the fastest layout changes together.
    allm = (8, 16, 32, 64, 128, 256)
    def valid(t, p_, m_):
        d = 32 / (t * p_)
        return d == int(d) and d >= 1 and (256 / (d * m_)) == int(256 / (d * m_)) and 256 / (d * m_) >= 1
    def cfg_ok(t, p_, m_):
        d = int(32 / (t * p_)); return config('70b', t, p_, d, m_, b=int(256 / (d * m_)), zero=1)
    def near(t2, p2, m_):
        if valid(t2, p2, m_): return m_
        ok = [x for x in allm if valid(t2, p2, x)]
        return min(ok, key=lambda x: abs(math.log(x / m_))) if ok else None
    from decimal import Decimal, ROUND_HALF_UP
    half_up = lambda x, q: Decimal(x).quantize(Decimal(q), rounding=ROUND_HALF_UP)   # what toLocaleString does with the exact double
    shown_s = lambda s_: str(half_up(s_, '1') if s_ >= 100 else half_up(s_, '0.1'))   # the task card: 33.2 s, 123 s
    shown_gb = lambda b_: str(half_up(b_ / 1e9, '1') if b_ / 1e9 >= 100 else half_up(b_ / 1e9, '0.1')) + ' GB'
    is_min = lambda r: r['fits'] and r['step'] <= best * 1.01
    def changes(t, p_, m_):
        out = []
        for t2 in (1, 2, 4, 8):
            if t2 != t and (32 / (t2 * p_)) == int(32 / (t2 * p_)) and 32 / (t2 * p_) >= 1:
                m2 = near(t2, p_, m_)
                if m2: out.append(dict(k='t', v=t2, t=t2, p=p_, m=m2))
        for p2 in (1, 2, 4, 8):
            if p2 != p_ and (32 / (t * p2)) == int(32 / (t * p2)) and 32 / (t * p2) >= 1:
                m2 = near(t, p2, m_)
                if m2: out.append(dict(k='p', v=p2, t=t, p=p2, m=m2))
        for m2 in allm:
            if m2 != m_ and valid(t, p_, m2): out.append(dict(k='m', v=m2, t=t, p=p_, m=m2))
        for c in out: c['r'] = cfg_ok(c['t'], c['p'], c['m'])
        return out
    def one_change(t, p_, m_):
        me = cfg_ok(t, p_, m_)
        fit_c = sorted((c for c in changes(t, p_, m_) if c['r']['fits']), key=lambda c: c['r']['step'])
        if not fit_c: return None
        c = fit_c[0]
        if me['fits'] and not float(shown_s(c['r']['step'])) < float(shown_s(me['step'])): return None
        return c
    fastest = min(fit, key=lambda x: x[5]['step'])
    BT, BP, BM = fastest[0], fastest[1], fastest[2]
    def targets(lang, t, p_, m_, bt, bp, bm):
        z = lang == 'zh'
        pz = (f'改用 {bp} 段流水线' if p_ == 1 else f'把流水线改成 {bp} 段') if z else (f'a pipeline of {bp} stages instead of none' if p_ == 1 else f'{bp} pipeline stages instead of {p_}')
        tz = (f'改用张量并行 {bt} 份' if t == 1 else f'把张量并行改成 {bt} 份') if z else (f'tensor parallelism {bt} ways instead of none' if t == 1 else f'tensor parallelism {bt} ways instead of {t}')
        return ([pz] if bp != p_ else []) + ([tz] if bt != t else []) + \
               ([(f'把 micro-batch 数改成 m = {bm}' if z else f'm = {bm} micro-batches instead of {m_}')] if bm != m_ else [])
    def nearest_goal(t, p_, m_):
        mins = [x for x in fit if is_min(x[5])]
        return min(mins, key=lambda x: (sum(a != b for a, b in ((x[0], t), (x[1], p_), (x[2], m_))), x[5]['step']))
    import re as _re
    nums = lambda s_: [float(x) for x in _re.findall(r'\d+(?:\.\d+)?', s_)]
    walked, kinds = 0, {}
    texts = {}
    for lang in ('en', 'zh'):
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(150)
        texts[lang] = H("()=>{const o={}; for(const t of [1,2,4,8]) for(const p of [1,2,4,8]) for(const m of [8,16,32,64,128,256]) o[t+'_'+p+'_'+m]=window.__mp.nextText(t,p,m); return o}")
    pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(150)
    for t in (1, 2, 4, 8):
        for p_ in (1, 2, 4, 8):
            for m_ in allm:
                key = f'{t}_{p_}_{m_}'; name = f'G7: TP {t} × PP {p_} × {m_}'
                if not valid(t, p_, m_):
                    check(f'{name}: the card cannot offer it (no layout, no sentence)', H(f"()=>window.__mp.chEval({t},{p_},{m_})") is None and texts['en'][key] is None and texts['zh'][key] is None)
                    continue
                walked += 1
                me = cfg_ok(t, p_, m_); mine = one_change(t, p_, m_); page = H(f"()=>window.__mp.oneChange({t},{p_},{m_})")
                ok = (mine is None and page is None) or (mine is not None and page is not None and (page['k'], page['v'], page['t'], page['p'], page['m']) == (mine['k'], mine['v'], mine['t'], mine['p'], mine['m']))
                check(f'{name}: the single change named = recomputation', ok, (mine and {k: mine[k] for k in 'kvtpm'}, page))
                kind = 'goal' if is_min(me) else ('better' if me['fits'] and mine else 'none-fits' if me['fits'] else 'fit' if mine else 'none-nofit')
                kinds[kind] = kinds.get(kind, 0) + 1
                for lang in ('en', 'zh'):
                    z = lang == 'zh'; tx_ = texts[lang][key]; nm = f'{name} ({lang})'
                    check(f'{nm}: never “about 0 s”, never “少约”', not _re.search(r'about 0 s|约 0 秒|少约|shorter by about 0', tx_), tx_)
                    if kind == 'goal':
                        check(f'{nm}: meets the goal, no sentence', tx_ == '', tx_); continue
                    if mine:
                        a, b_ = shown_s(me['step']), shown_s(mine['r']['step'])
                        cm = mine['k'] != 'm' and mine['m'] != m_
                        note = (f'任务卡会把 m 随之改成 {mine["m"]}' if z else f'the card then sets m = {mine["m"]}')
                        check(f'{nm}: “the card then sets m” exactly when the card changes m, to the card\'s own nearest m', (note in tx_) == cm and (('任务卡会把 m' in tx_ or 'the card then sets' in tx_) == cm), tx_)
                        if kind == 'better':
                            want = (f'每步从 {a} 秒缩短到 {b_} 秒' if z else f'a step goes from {a} s to {b_} s')
                            check(f'{nm}: names the change with the step before and after as the card shows them, and the shown time drops', want in tx_ and float(b_) < float(a)
                                  and (('只改一项的话' in tx_ and '最划算' in tx_) if z else 'The best single change is' in tx_), (want, tx_))
                            goal = is_min(mine['r'])
                            if goal:
                                check(f'{nm}: the change meets the goal and says so', ('改完就达标了' if z else 'which meets the goal') in tx_, tx_)
                            else:
                                g = nearest_goal(mine['t'], mine['p'], mine['m'])
                                tg = targets(lang, mine['t'], mine['p'], mine['m'], g[0], g[1], g[2])
                                check(f'{nm}: still short of the goal; names what else to change', ('但还没达标' if z else 'still short of the goal') in tx_ and all(x in tx_ for x in tg), (tg, tx_))
                        else:   # makes it fit: the change really fits, and the sentence says what it costs
                            r2 = mine['r']; goal = is_min(r2)
                            cost_ok = r2['fits'] and r2['mem'] <= HBM and shown_gb(r2['mem']) in tx_ and (f'每步 {b_} 秒' if z else f'a step takes {b_} s') in tx_
                            slow = max(1, math.floor((r2['step'] / best - 1) * 100 + 0.5))
                            check(f'{nm}: the change that makes it fit really fits; its memory and step time are named', cost_ok and ('只改一项就放得下的改法里' if z else 'Of the single changes that make it fit') in tx_, (shown_gb(r2['mem']), b_, tx_))
                            check(f'{nm}: meets the goal, or how much slower than the fastest', (('就达标了' if z else 'which meets the goal') in tx_) if goal else ((f'比最快的排法慢约 {slow}%' if z else f'about {slow}% slower than the fastest layout') in tx_), tx_)
                    else:
                        lead = ('不管只改哪一项，每步都不会变短。' if z else 'No single change makes a step shorter.') if me['fits'] else ('只改一项还放不下。' if z else 'No single change makes it fit.')
                        tg = targets(lang, t, p_, m_, BT, BP, BM)
                        # a setting the fastest layout keeps must not be named at all
                        others = [w for (k, w) in zip((BP != p_, BT != t, BM != m_), (('流水线', 'pipeline'), ('张量并行', 'tensor parallelism'), ('micro-batch', 'micro-batch'))) if not k]
                        others = [w[0] if z else w[1] for w in others]
                        check(f'{nm}: says no single change helps, and names every setting the fastest layout changes, and only those', tx_.startswith(lead) and all(x in tx_ for x in tg) and not any(x in tx_ for x in others), (tg, tx_))
                        if me['fits']:   # walk every single change: none fits with a shorter shown step
                            better = [c for c in changes(t, p_, m_) if c['r']['fits'] and float(shown_s(c['r']['step'])) < float(shown_s(me['step']))]
                            check(f'{nm}: truly no single change shortens the shown step', not better, better)
    check(f'G7: walked every pick the task card offers (57 layouts of 96 combinations)', walked == 57, walked)
    check('G7: every kind of sentence occurs (goal 2, better 3, none-fits 1, fit 18, none-nofit 33)', kinds == {'goal': 2, 'better': 3, 'none-fits': 1, 'fit': 18, 'none-nofit': 33}, kinds)
    oc = H("()=>window.__mp.oneChange(8,4,256)")
    check('challenge: from TP 8 × PP 4 × 256 the best single change is TP 4 (the card sets m = 128), a layout that passes', oc and (oc['k'], oc['v'], oc['m']) == ('t', 4, 128) and oc['step'] <= best * 1.01, oc)
    check('challenge: from the initial setup no single change fits', H("()=>window.__mp.oneChange(8,1,8)") is None)
    # the Codex finding: PP 2 × TP 8 × 128 fits at 33.2 s; PP 4 alone gives 33.3 s, so no single change helps (it used to say “about 0 s”)
    check('G7 (Codex): from PP 2 × TP 8 × 128 no single change is named; PP 4 alone shows 33.3 s', H("()=>window.__mp.oneChange(8,2,128)") is None and shown_s(cfg_ok(8, 2, 128)['step']) == '33.2' and shown_s(cfg_ok(8, 4, 128)['step']) == '33.3',
          (shown_s(cfg_ok(8, 2, 128)['step']), shown_s(cfg_ok(8, 4, 128)['step'])))
    # ---------- transfer and drill ----------
    check('transfer: 16 stages, 64 micro-batches about 20%', pct(15 / 79) == '19%')
    ex = H("()=>window.__mp.exits()")
    check('transfer answers', ex['transferA'] == [1, 1], ex)
    for seed in (1, 2, 3, 7, 42):
        d = H(f"()=>window.__mp.drill({seed})"); mine = config('70b', d['t'], 80, 1, 1, place=d['place'])
        check(f'drill {seed}', close(d['bubble'], (d['p'] - 1) / (d['m'] + d['p'] - 1)) and d['gp'] == d['m'] and d['f1'] == min(d['m'], d['p']) and close(d['comm'], mine['comm']), d)
    # ---------- teaser ----------
    tz = H("()=>window.__tzMp?{P:window.__tzMp.P,MS:window.__tzMp.MS,m:window.__tzMp.MS.map((_,i)=>window.__tzMp.model(i)),r:window.__tzMp.read(4)}:null")
    check('teaser hook', tz is not None)
    if tz:
        check('teaser: p 4, m 1 to 32', tz['P'] == 4 and tz['MS'] == [1, 2, 4, 8, 16, 32])
        for x in tz['m']:
            mine = schedule(4, x['m'], 'gpipe')
            check(f"teaser m {x['m']}", x['T'] == mine['T'] and close(x['bubble'], 3 / (x['m'] + 3)) and close(1 - x['busy'], mine['idle']), (x, mine))
        check('teaser sentence: 16 micro-batches, 16%', '16%' in tz['r'] and '16' in tz['r'], tz['r'])
    check('no page errors', not errs, errs)
    br.close()
print(f'{n - len(fails)}/{n} passed'); sys.exit(1 if fails else 0)
