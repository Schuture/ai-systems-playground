"""Golden numbers for the kv-basics lab (Transformer series ③), recomputed independently in Python.

- Toy: a numpy re-implementation of the 2-layer transformer, vectorized with a −inf mask (the page loops token by token), fed with
  the weights the page exposes read-only (window.__kvb.weights()). Checks: keys and values equal the page's for every step and both
  masks; the recompute check (exactly 0 on the page with the causal mask; with the mask off layer 1 is 0 and layer 2 matches);
  the counts n(n + 1) / 2, n, n(n − 1) / 2 and 32 n.
- Llama-3.1-8B on one H100 SXM: the step-time model rebuilt from the config and the datasheet, not from the page's constants;
  every context stop, the challenge answer, the four scenarios, the interview drill.
- The prefill-decode tile (teasers/set-a.js): a private build with both files; the lab's prefill of C tokens (the first step of an
  answer with the cache) equals the tile's prefill, the lab's step without the cache at C does the same arithmetic but writes no cache
  (review 8: nothing is cached), and the lab's step with the cache at C + 1 equals the tile's decode step after C tokens.
- What the page shows: metric cards, speed bars, the part-4 result at every step with the mask off (the largest change and the
  average size of layer 2's numbers, both from numpy), the part-6 result, the challenge verdicts and chart, each scenario's result
  box, the math lines (parsed and redone from the shown numbers, R6.6), the tile sentence.
- The Connect code sample: at most 40 lines, and, when PyTorch is installed, it runs and its two loops agree.

Run: python3 tests/kv-basics_numbers.py
"""
import json, math, pathlib, re, subprocess, sys, tempfile
import numpy as np
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/kv-basics.html', '--only', 'kv-basics'], check=True, cwd=P, stdout=subprocess.DEVNULL)
WRAP = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>'
(P / 'out' / 'wrapped-kv-basics.html').write_text(WRAP + (P / 'out' / 'kv-basics.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-kv-basics.html')
TMP = pathlib.Path(tempfile.mkdtemp(prefix='kvb-'))
subprocess.run([sys.executable, str(P / 'build.py'), str(TMP / 'kvb-seta.html'), '--only', 'kv-basics,set-a'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(TMP / 'wrapped.html').write_text(WRAP + (TMP / 'kvb-seta.html').read_text() + '</body></html>')
URL2 = 'file://' + str(TMP / 'wrapped.html')

n_checks, fails = 0, []
def check(name, ok, info=''):
    global n_checks
    n_checks += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])

# ---------------------------------------------------------------- independent models
# Llama-3.1-8B (config.json): vocab 128,256, hidden 4,096, 32 layers, 32 heads, 8 KV heads, head_dim 128, MLP 14,336, untied.
V, d, L, H, KVH, hd, f = 128256, 4096, 32, 32, 8, 128, 14336
layer = d * H * hd + 2 * d * KVH * hd + H * hd * d + 3 * d * f + 2 * d        # q, k, v, o, three MLP matrices, two RMSNorm vectors
multiplied = L * layer + d                                                     # every token goes through these (+ final norm)
head = V * d                                                                    # the LM head, once per step
embed = V * d                                                                   # a lookup: one row per token
total = multiplied + head + embed
kv_tok = 2 * L * KVH * hd * 2                                                   # K and V, BF16
peak, bw = 1979e12 / 2 * 0.5, 3.35e12 * 0.6                                     # dense BF16 at 50%, HBM3 at 60% (assumptions)
check('Llama-3.1-8B has 8,030,261,248 parameters (the facts file’s safetensors count)', total == 8030261248, total)
check('KV cache per token is 131,072 bytes = 128 KiB', kv_tok == 131072)

def attn_pairs(first, last):
    """(query, key) pairs for query positions first..last (1-indexed), causal: query i meets i keys"""
    return sum(range(first, last + 1))

def step(tokens, pairs, kv_read):
    flops = 2 * multiplied * tokens + 2 * head + 4 * L * d * pairs              # QKᵀ and AV: 2 + 2 FLOPs per (query, key) pair per hidden number
    bytes_ = 2 * (multiplied + head) + kv_tok * kv_read
    return max(flops / peak, bytes_ / bw), flops, bytes_

# review 8 (r14): without the cache nothing is cached, so the step reads the weights only (the keys and values it computes are
# activations, left out like the others); with the cache it reads the weights and the cache of the C − 1 earlier tokens; the first step
# of an answer with the cache (the prefill) writes the prompt's keys and values.
def nocache(C): return step(C, attn_pairs(1, C), 0)
def prefill(P): return step(P, attn_pairs(1, P), P)
def cached(C): return step(1, C, C - 1)

def tS(sec):
    v = sec * 1e3
    return f'{v:,.1f} ms' if v < 999.95 else f'{v / 1000:,.2f} s'
def shown_sec(sec):
    v = sec * 1e3
    return round(v, 1) / 1000 if v < 999.95 else round(v / 1000, 2)
def tps(sec):
    v = 1 / shown_sec(sec)
    return f'{round(v):,}' if v >= 99.5 else (f'{v:.1f}' if v >= 0.995 else f'{v:.2f}')
def totS(sec):
    if sec < 0.9995: return f'{round(sec * 1e3):,} ms'
    if sec < 9.995: return f'{sec:.2f} s'
    if sec < 59.95: return f'{sec:.1f} s'
    return f'{sec / 60:.1f} min'
def tot_sec(sec):
    if sec < 0.9995: return round(sec * 1e3) / 1000
    if sec < 9.995: return round(sec, 2)
    if sec < 59.95: return round(sec, 1)
    return round(sec / 60, 1) * 60
def ratioS(r): return f'{round(r):,}' if r >= 9.95 else f'{r:.1f}'
rnd = lambda x: int(math.floor(x + 0.5))                                         # JavaScript's Math.round (half up)
def fR(sec):
    """a time in prose (R6.20): about two significant digits, as the lab's fR"""
    v = sec * 1e3
    if v >= 9950: return f'{rnd(v / 1e3):,} s'
    if v >= 950: return f'{v / 1e3:.1f} s'
    if v >= 99.5: return f'{rnd(v / 10) * 10:,} ms'
    if v >= 9.95: return f'{rnd(v):,} ms'
    return f'{v:.1f} ms'
def totR(sec):
    if sec < 59.95: return fR(sec)
    m = sec / 60
    return (f'{rnd(m):,}' if m >= 9.95 else f'{m:.1f}') + ' min'
def ratioW(q):
    """how many times as long, in words, as the lab's ratioW"""
    if q < 9.95: return f'{q:.1f} times'
    f = (q // 100) * 100 if q >= 100 else (q // 10) * 10
    return f'over {int(f):,} times' if q > f + 1e-9 else f'{int(f):,} times'
def ctxK(C): return (f'{C // 1024}K' if C % 1024 == 0 else f'{C / 1024:g}K') if C >= 1024 and C % 512 == 0 else f'{C:,}'
def gbS(b):
    sig3 = lambda v: f'{v:.0f}' if v >= 100 else (f'{v:.1f}' if v >= 10 else f'{v:.2f}')
    return (sig3(b / 1e9) + ' GB') if b >= 1e9 else (sig3(b / 1e6) + ' MB')
sp = lambda x: x.replace('\u00a0', ' ').replace('\u2011', '-')
def memS(b):
    g = b / 2**30
    if g >= 1: return (f'{int(g):,}' if g == int(g) else f'{g:.2f}') + ' GiB'
    m = b / 2**20
    return (f'{int(m):,}' if m == int(m) else f'{m:.1f}') + ' MiB'

CTX = [256, 512, 1024, 1536, 2048, 3072, 4096, 8192, 16384, 32768, 65536, 131072]
STOPS = CTX[:10]

# toy, in numpy
RELU = lambda x: np.maximum(x, 0)
GELU = lambda x: 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

def np_forward(W, s, mask, act=RELU):
    h = np.array(W['tok'][:s], float) + np.array(W['pos'][:s], float)
    rms = lambda x: x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)
    Ks, Vs = [], []
    for Pm in W['layers']:
        Wq, Wk, Wv, Wo, W1, W2 = (np.array(Pm[k], float) for k in ('q', 'k', 'v', 'o', 'w1', 'w2'))
        a = rms(h); q, k, v = a @ Wq, a @ Wk, a @ Wv
        Ks.append(k); Vs.append(v)
        o = np.zeros_like(h)
        allowed = np.tril(np.ones((s, s), bool)) if mask else np.ones((s, s), bool)
        for hh in range(2):
            sl = slice(4 * hh, 4 * hh + 4)
            sc = np.where(allowed, q[:, sl] @ k[:, sl].T / 2.0, -np.inf)
            w = np.exp(sc - sc.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
            o[:, sl] = w @ v[:, sl]
        h1 = h + o @ Wo
        h = h1 + act(rms(h1) @ W1) @ W2
    return Ks, Vs, h

def np_check(W, n, mask, act=RELU):
    now = np_forward(W, n, mask, act)
    per = [[], []]
    for j in range(n - 1):
        first = np_forward(W, j + 1, mask, act)
        for l in range(2):
            per[l].append(max(np.abs(now[0][l][j] - first[0][l][j]).max(), np.abs(now[1][l][j] - first[1][l][j]).max()))
    return per

def sdiff(v): return '0' if v == 0 else ('<0.01' if v < 0.005 else f'{v:.2f}')

def go(pg, hash_):
    pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(120)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(350)

def metrics(pg):
    return pg.evaluate("()=>Object.fromEntries([...document.querySelectorAll('#kvb-metrics .metric')].map(m=>[m.dataset.k,{v:m.querySelector('.v').textContent,sub:m.querySelector('.sub').textContent}]))")

def set_range(pg, sel, v):
    pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}))}}"); pg.wait_for_timeout(120)

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width': 1440, 'height': 1000})
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    K = 'window.__kvb'
    W = pg.evaluate(f'()=>{K}.weights()')
    c = pg.evaluate(f'()=>{K}.consts')

    # ------------------------------------------------ toy: shapes, forward pass, recompute check, counts
    check('toy shapes: 9 input vectors of 8, 2 layers of 8×8 q/k/v/o and an 8→16→8 MLP',
          np.array(W['tok']).shape == (9, 8) and np.array(W['pos']).shape == (9, 8) and len(W['layers']) == 2
          and all(np.array(l[k]).shape == (8, 8) for l in W['layers'] for k in 'qkvo') and np.array(W['layers'][0]['w1']).shape == (8, 16)
          and np.array(W['layers'][1]['w2']).shape == (16, 8))
    check('toy: the per-token cache is 2 layers × K and V × 2 heads × 4 = 32 numbers', c['PER_TOK'] == 2 * 2 * 2 * 4 == 32)
    worst = 0.0
    for mask in (True, False):
        for s in range(1, 10):
            page = pg.evaluate(f'()=>{K}.forward({s},{str(mask).lower()})')
            Ks, Vs, h = np_forward(W, s, mask)
            for l in range(2):
                worst = max(worst, np.abs(np.array(page['K'][l]) - Ks[l]).max(), np.abs(np.array(page['V'][l]) - Vs[l]).max())
            worst = max(worst, np.abs(np.array(page['h']) - h).max())
    check('toy: keys, values and outputs equal numpy’s for every step, with and without the mask', worst < 1e-12, worst)
    for n in range(2, 10):
        on = pg.evaluate(f'()=>{K}.check({n},true)'); off = pg.evaluate(f'()=>{K}.check({n},false)')
        ref_on, ref_off = np_check(W, n, True), np_check(W, n, False)
        check(f'toy check at step {n}, causal mask on: the page computes exactly 0 in both layers (numpy: below 1e-12)',
              on['max'] == [0, 0] and all(v == 0 for l in on['per'] for v in l) and max(max(r) for r in ref_on) < 1e-12, (on['max'], max(max(r) for r in ref_on)))
        check(f'toy check at step {n}, mask off: layer 1 exactly 0, layer 2 equals numpy per token',
              all(v == 0 for v in off['per'][0]) and max(abs(a - b) for a, b in zip(off['per'][1], ref_off[1])) < 1e-12 and max(off['per'][1]) > 0.05,
              (off['per'], ref_off))
    off9 = np_check(W, 9, False)
    j2 = int(np.argmax(off9[1]))
    check('toy check at step 9, mask off: the largest layer-2 difference is 1.16, at token 2', sdiff(max(off9[1])) == '1.16' and j2 == 1, (max(off9[1]), j2))
    row9 = ' '.join(sdiff(v) for v in off9[1])
    check('toy check row at step 9, mask off (spec): 1.00 1.16 0.68 0.75 0.39 0.24 0.26 0.11', row9 == '1.00 1.16 0.68 0.75 0.39 0.24 0.26 0.11', row9)
    # r12 B N8: the Fidelity card names what numpy needs to reproduce the page: a ReLU in the MLP and an RMSNorm without learned
    # weights (numpy above has none). The activation matters: with a GELU the step-9 row would not read as on the page.
    gelu9 = ' '.join(sdiff(v) for v in np_check(W, 9, False, GELU)[1])
    fid = {}
    for lg in ('en', 'zh'):
        pg.evaluate(f"()=>document.querySelector('.langsw button[data-lang=\"{lg}\"]').click()"); pg.wait_for_timeout(150)
        go(pg, 'kv-basics'); fid[lg] = pg.evaluate("()=>document.getElementById('kv-basics-fid').textContent")
    pg.evaluate("()=>document.querySelector('.langsw button[data-lang=\"en\"]').click()"); pg.wait_for_timeout(150)
    # r15 (ARCH #2, #15): the residual connection adds back the vector from before the block's RMSNorm, as the engine does
    # (h1 = h + attention(rms(h)), then h1 + mlp(rms(h1))), and the RMSNorm has no learned scales
    # the small model's description was in the Fidelity card's “Modeled” list, deleted with it (owner, 2026-10-11 19:28 UTC); the numbers still have to follow that model
    check('small model: a ReLU, an RMSNorm without learned scales, each block’s output added back to the vector from before its RMSNorm reproduce the page (a GELU would give another row)',
          gelu9 != row9 and set(W) == {'tok', 'pos', 'layers'} and all(set(l) == {'q', 'k', 'v', 'o', 'w1', 'w2'} for l in W['layers']), (gelu9, fid['en'][:300]))
    for n in range(0, 10):
        for cache in (False, True):
            cc = pg.evaluate(f'()=>{K}.counts({n},{str(cache).lower()})')
            exp = {'passes': n if cache else sum(range(1, n + 1)), 'reads': sum(range(0, n)) if cache else 0, 'nums': 32 * n if cache else 0, 'p100': 100 if cache else 5050}
            check(f'toy counts, {n} steps, cache {cache}', all(cc[k] == v for k, v in exp.items()), (cc, exp))

    # ------------------------------------------------ toy: what the stage shows (metric cards, check row)
    for n in (0, 4, 9):
        for cache in (0, 1):
            go(pg, f'kv-basics~t_{n}_{cache}_1_8192')
            m = metrics(pg)
            passes = n if cache else n * (n + 1) // 2
            check(f'toy metrics at {n} steps, cache {cache}', m['passes']['v'] == f'{passes:,}' and m['p100']['v'] == ('100' if cache else '5,050')
                  and m['reads']['v'] == f'{(n * (n - 1) // 2) if cache else 0:,}' and m['nums']['v'] == f'{32 * n if cache else 0:,}', m)
    go(pg, 'kv-basics~t_9_0_0_8192')
    rows = pg.evaluate("()=>[...document.querySelectorAll('#kvb-grid .kvb-chk')].map(r=>[...r.querySelectorAll('.kvb-d')].map(d=>d.textContent))")
    check('the stage’s check rows at step 9, mask off: layer 1 all 0, layer 2 the numpy numbers', rows[0][:8] == ['0'] * 8 and ' '.join(rows[1][:8]) == row9 and rows[0][8] == '—', rows)
    cells = pg.evaluate("()=>[...document.querySelectorAll('#kvb-grid .kvb-layer')].map(l=>[l.querySelectorAll('.kvb-comp').length,l.querySelectorAll('.kvb-read').length])")
    check('the grid without the cache at 9 steps: 45 computed cells per layer, none read', cells == [[45, 0], [45, 0]], cells)
    go(pg, 'kv-basics~t_9_1_1_8192')
    cells = pg.evaluate("()=>[...document.querySelectorAll('#kvb-grid .kvb-layer')].map(l=>[l.querySelectorAll('.kvb-comp').length,l.querySelectorAll('.kvb-read').length])")
    check('the grid with the cache at 9 steps: 9 computed and 36 read per layer', cells == [[9, 36], [9, 36]], cells)

    # ------------------------------------------------ the part-4 result at every step, mask off: the largest difference and the scale
    go(pg, 'kv-basics'); pg.evaluate("()=>document.querySelector('#kvb-app .modes [data-mode=guided]').click()"); pg.wait_for_timeout(200)
    pg.evaluate("()=>document.querySelector('#kvb-rail [data-go=\"3\"]').click()"); pg.wait_for_timeout(250)
    pg.evaluate("()=>document.querySelector('#kvb-ic-mask button[data-v=\"0\"]').click()"); pg.wait_for_timeout(200)
    for n in range(2, 10):
        set_range(pg, '#kvb-ctl-n-in', n)
        t = pg.inner_text('#kvb-rail [data-obs]').replace(' ', ' ')
        Ks, Vs, _ = np_forward(W, n, False)
        size = np.abs(np.concatenate([Ks[1].ravel(), Vs[1].ravel()])).mean()
        per = np_check(W, n, False)
        jm = int(np.argmax(per[1])); word = pg.evaluate(f'()=>{K}.words().en[{jm}]')
        # R6.20: the sentence keeps one number (the largest change, on the stage) and compares it in words with the average size of
        # layer 2's numbers, chosen by the ratio at this step; the average itself is on hover
        tip = pg.evaluate("()=>{const d=[...document.querySelectorAll('#kvb-rail [data-obs] .def')].find(e=>e.textContent.startsWith('the average size')); return d?d.getAttribute('data-tipx'):''}")
        q = max(per[1]) / size
        cmp = 'larger than' if q >= 1.05 else ('about as large as' if q >= 0.95 else 'more than half of')
        want = f'layer 1’s did not. In layer 2 the largest change is {sdiff(max(per[1]))} (at “{word}”), {cmp} the average size of the numbers in its keys and values.'
        want_tip = f'At step {n}, the numbers in layer 2’s keys and values average {size:.1f}, ignoring sign.'
        check(f'part 4 result at step {n}, mask off: the largest layer-2 difference in the sentence, its size against layer 2’s numbers in words ({q:.2f}) and their average on hover (numpy)', want in t and sp(tip) == want_tip, (want, want_tip, t[:400], tip))
    check('part 4 at step 9: the largest change (1.16) is larger than the average size of layer 2’s numbers (0.9, on hover)', '1.16' in t and 'larger than the average size' in t and 'average 0.9, ignoring sign' in tip, (t[:400], tip))

    # ------------------------------------------------ Llama-3.1-8B: constants and every context stop
    check('page constants equal the independent derivation (multiplied weights, LM head, weights read, KV per token, memory of the weights)',
          c['P_LAYERS'] == multiplied and c['P_HEAD'] == head and c['W_READ'] == 2 * (multiplied + head) and c['KV_TOK'] == kv_tok and c['W_MEM'] == 2 * total,
          {k: c[k] for k in ('P_LAYERS', 'P_HEAD', 'W_READ', 'KV_TOK', 'W_MEM')})
    for C in CTX:
        pw, po = pg.evaluate(f'()=>{K}.cached({C})'), pg.evaluate(f'()=>{K}.nocache({C})')
        tw, flw, byw = cached(C); to, flo, byo = nocache(C)
        check(f'step times at {C:,} tokens (with / without the cache)', abs(pw['t'] - tw) < 1e-12 and abs(po['t'] - to) < 1e-12 and pw['fl'] == flw and po['fl'] == flo and pw['by'] == byw and po['by'] == byo,
              (pw, tw, po, to))
        # review 8 point 4: both steps read the weights once; only the step with the cache also reads the cache
        check(f'memory terms at {C:,}: the weights ({2 * (multiplied + head):,} bytes) in both, plus the cache of {C - 1:,} tokens with it, nothing more without it',
              po['by'] == 2 * (multiplied + head) and pw['by'] - po['by'] == (C - 1) * kv_tok, (pw['by'], po['by']))
        check(f'deciding term at {C:,}: memory with the cache, arithmetic without it from 512 tokens on (memory at 256)',
              pw['tm'] > pw['tc'] and ((po['tc'] > po['tm']) == (C >= 512)), (pw['tm'], pw['tc'], po['tm'], po['tc']))
        go(pg, f'kv-basics~r_9_1_1_{C}')
        m = metrics(pg)
        check(f'real metrics at {C:,}', m['kvtok']['v'] == '128 KiB' and m['kvctx']['v'] == memS(C * kv_tok) and m['tw']['v'] == tS(tw) and m['two']['v'] == tS(to)
              and m['tw']['sub'].startswith(tps(tw)) and m['two']['sub'].startswith(tps(to)), (m, memS(C * kv_tok), tS(tw), tS(to), tps(tw), tps(to)))
        bars = pg.evaluate("()=>[...document.querySelectorAll('#kvb-sp .kvb-sprow')].map(r=>r.querySelector('.kvb-spv').textContent)")
        check(f'speed bars at {C:,}', bars[0] == f'{tps(tw)} tokens per second ({tS(tw)} each)' and bars[1] == f'{tps(to)} tokens per second ({tS(to)} each)', bars)
        # Show the math: redo each line from the numbers it shows
        pg.evaluate("()=>{document.getElementById('kvb-math').open=true}")
        # H2 (2026-10-09): Show the math is small tables (name and formula, the numbers under it, the result); read as one line of text
        mt = ' '.join(pg.inner_text('#kvb-mathbody').split())
        ok = True
        for mm in re.finditer(r'= \(([^()=]+)\) ÷ ([\d.]+) TFLOP/s = ([\d,.]+) TFLOP ÷ ([\d.]+) TFLOP/s = ([\d,.]+) ms', mt):
            expr, pk, tf, pk2, ms = mm.groups(); tf, ms = tf.replace(',', ''), ms.replace(',', '')
            val = eval(expr.replace(',', '').replace('×', '*'))
            dec = len(tf.split('.')[1])
            ok &= f'{val / 1e12:.{dec}f}' == tf and pk == pk2 == '494.75' and f'{float(tf) / 494.75 * 1000:.2f}' == ms   # the commas in 1,234.56 are only grouping
        for mm in re.finditer(r'= \(([^()=]+)\) ÷ ([\d.]+) TB/s = ([\d,.]+) GB ÷ ([\d.]+) TB/s = ([\d,.]+) ms', mt):
            expr, bwv, gb, bwv2, ms = mm.groups(); gb, ms = gb.replace(',', ''), ms.replace(',', '')
            val = eval(expr.replace(',', '').replace('×', '*'))
            ok &= f'{val / 1e9:.3f}' == gb and bwv == bwv2 == '2.01' and f'{float(gb) / 2.01:.2f}' == ms
        nfl = len(re.findall(r'TFLOP/s = [\d,.]+ TFLOP ÷', mt)); nby = len(re.findall(r'TB/s = [\d,.]+ GB ÷', mt))
        check(f'Show the math at {C:,}: every compute and memory time redoes from its shown numbers', ok and mt.count('Compute time =') == 2 and mt.count('Memory time =') == 2 and nfl == 2 and nby == 2, mt[:600])
        # the memory line without the cache is the weights plus 0 × 131,072: the two steps side by side, term for term (review 8)
        check(f'Show the math at {C:,}: with the cache the memory line adds {C - 1:,} × 131,072, without it 0 × 131,072',
              f'(15,009,849,344 + {C - 1:,} × 131,072) ÷' in mt and '(15,009,849,344 + 0 × 131,072) ÷ 2.01 TB/s = 15.010 GB' in mt, mt[:900])
        # which term decides each step, named after the step time (review 8 point 4)
        dec = [(b, a) for a, b in re.findall(r'= ([\d.,]+ (?:ms|s)), set by (memory|arithmetic)', mt)]
        want_dec = [('memory', ' '.join(tS(tw).split())), ('memory' if nocache(C)[2] / bw >= nocache(C)[1] / peak else 'arithmetic', ' '.join(tS(to).split()))]
        check(f'Show the math at {C:,}: each step names the term that decides it, and its time', dec == want_dec, (dec, want_dec))
        kvline = re.search(r'KV cache for the context = .*? = ([\d,]+) × 131,072 bytes = ([\d,]+) bytes \(([\d.]+) (GB|MB), ([\d.,]+ (?:MiB|GiB))\)', mt)
        gb = C * kv_tok / 1e9
        sig3 = lambda v: f'{v:.0f}' if v >= 100 else (f'{v:.1f}' if v >= 10 else f'{v:.2f}')
        gbs = sig3(gb) if gb >= 1 else sig3(C * kv_tok / 1e6)
        check(f'Show the math at {C:,}: the context line, bytes to GB and GiB (R6.12)', bool(kvline) and int(kvline.group(1).replace(',', '')) == C and int(kvline.group(2).replace(',', '')) == C * kv_tok
              and kvline.group(3) == gbs and kvline.group(4) == ('GB' if gb >= 1 else 'MB') and kvline.group(5) == memS(C * kv_tok).replace(' ', ' '), kvline.groups() if kvline else mt[:300])

    # ------------------------------------------------ the part-6 result at every stop
    go(pg, 'kv-basics'); pg.evaluate("()=>document.querySelector('#kvb-app .modes [data-mode=guided]').click()"); pg.wait_for_timeout(200)
    pg.evaluate("()=>document.querySelector('#kvb-rail [data-go=\"5\"]').click()"); pg.wait_for_timeout(250)
    for i, C in enumerate(CTX):
        set_range(pg, '#kvb-ic-ctx-in', i)
        t = pg.inner_text('#kvb-rail [data-obs]').replace(' ', ' ')
        tw, _, _ = cached(C); to, _, _ = nocache(C)
        k = shown_sec(to) / shown_sec(tw)
        same = k < 1.05
        big = C * kv_tok > 2 * total
        # R6.12: where bytes first become GiB the GB is shown beside it
        gbtxt = gbS(C * kv_tok)
        # R6.20: the result is the two step times, rounded, as one pair (the exact ones are on the speed bars and the metric cards, checked above)
        at = f'At {ctxK(C)} tokens of context, one decode step takes about {fR(tw)}'
        hl = at + (' with or without the KV cache, by this lab’s estimate.' if same else f' with the KV cache and about {fR(to)} without it, by this lab’s estimate.')
        mem = f'The price is memory: the cache now holds {sp(memS(C * kv_tok))} ({gbtxt})' + (', more than all the weights.' if big else ', and it grows with every token.')
        check(f'part 6 result at {C:,} tokens: the two step times rounded ({fR(tw)}, {fR(to)}), and the cache with {gbtxt} (more than the weights: {big})', hl in t and mem in t, (hl, mem, t[:420]))
        # review 8 points 2, 4, 5: the reason says both steps read the weights once, and which term decides each step, in words chosen by
        # the state (R6.10): the cache's share of the reads with the cache, arithmetic without it
        w_read = 2 * (multiplied + head) / bw
        share = (C - 1) * kv_tok / (2 * (multiplied + head) + (C - 1) * kv_tok)
        if same:
            ok6 = 'every step reads the weights once, however many tokens it computes. At this short context, recomputing every token and reading the cache both take less time than that, so the cache saves nothing yet' in t \
                  and nocache(C)[1] / peak < w_read and (C - 1) * kv_tok / bw < w_read
        else:
            wd = 'but the cache it reads is now long: over half of the step goes to reading it' if share > 0.5 else ('so memory decides, and its time grows only slowly as the cache gets longer' if share > 0.15 else 'so memory decides and its time barely grows')
            ok6 = 'every step reads the weights once, however many tokens it computes. With the cache it computes one token, ' + wd in t \
                  and 'Without it, the step recomputes the whole context, so arithmetic decides and the time grows with the context.' in t and nocache(C)[1] / peak > w_read
        check(f'part 6 at {C:,}: the reason says both steps read the weights once and which term decides each, in words for the state (cache share of the reads {share:.0%})', ok6, t[-520:])
        # r15 N12: what one read of the weights is (14.0 GiB) is on hover over “reads the weights”
        gW = f'{2 * (multiplied + head) / 2 ** 30:.1f} GiB'
        rw = sp(pg.evaluate("()=>{const d=[...document.querySelectorAll('#kvb-rail [data-obs] .def')].find(e=>e.textContent==='reads the weights'); return d?d.getAttribute('data-tipx'):''}"))
        check(f'part 6 at {C:,}: “reads the weights” says on hover what one read is ({gW} outside the embedding table)', f'all but the embedding table: {gW} for Llama-3.1-8B' in rw, rw)
        # K2: the speed badge is the ratio of the shown step times, and its tip shows the division
        tip = pg.get_attribute('#kvb-status', 'data-tipx').replace(' ', ' ')
        want_tip = f'{tS(to)} ÷ {tS(tw)} = {ratioS(k)}'.replace(' ', ' ')
        stat = pg.inner_text('#kvb-status').replace(' ', ' ')
        check(f'status at {C:,}: the tip redoes the badge from the shown times ({want_tip})', want_tip in tip and ((ratioS(k) in stat) if k >= 1.05 else 'Same speed' in stat), (tip, stat))
        # A3 and review 8: “How these are computed” (r15: shorter, so the 128K card fits) answers on hover with both terms of both steps and the one that decides
        how = pg.get_attribute('#kvb-rail [data-jump="math"]', 'data-tipx').replace(' ', ' ')
        tw_, flw, byw = cached(C); to_, flo, byo = nocache(C)
        ms2 = lambda x: f'{x * 1e3:,.2f} ms'
        tfl = lambda f: f'{f / 1e12:,.4f} TFLOP' if f < 1e12 else f'{f / 1e12:,.3f} TFLOP'
        parts = [f'{byw / 1e9:,.3f} GB', ms2(byw / bw), tfl(flw), ms2(flw / peak), f'{byo / 1e9:,.3f} GB', ms2(byo / bw), tfl(flo), ms2(flo / peak),
                 'so memory decides: ' + tS(tw).replace(' ', ' '), ('so memory decides: ' if byo / bw >= flo / peak else 'so arithmetic decides: ') + tS(to).replace(' ', ' '),
                 'at the same time', '494.75 TFLOP/s', '2.01 TB/s', 'reads only the weights']
        check(f'part 6 at {C:,}: the how-tip gives both terms of both steps and the one that decides', all(x in how for x in parts), ([x for x in parts if x not in how], how))

    # ------------------------------------------------ part 7: the misconception's short-context case holds on the shown times
    pg.evaluate("()=>document.querySelector('#kvb-rail [data-go=\"6\"]').click()"); pg.wait_for_timeout(250)
    c7 = pg.inner_text('#kvb-rail .step.current').replace('\u00a0', ' ')
    # R6.20: misconception 2 says the trend and one ratio, the one the stage's badge shows at 8K (33 times as fast), in words
    k8 = shown_sec(nocache(8192)[0]) / shown_sec(cached(8192)[0])
    check(f'part 7: at 256 tokens both steps show the same time (7.5 ms), so “only at the shortest contexts” holds; at 8K a step without the cache takes {ratioW(k8)} as long ({k8:.1f})',
          tS(cached(256)[0]) == tS(nocache(256)[0]) == '7.5\u00a0ms' and 'Only at the shortest contexts.' in c7 and f'Without it each step recomputes the whole context, so at 8K tokens it takes {ratioW(k8)} as long, by this lab’s estimate.' in c7, c7[-600:])

    # ------------------------------------------------ the challenge: answer, tiers, chart
    pts = [(C, nocache(C)[0]) for C in STOPS]
    best = max(C for C, t in pts if 1 / shown_sec(t) >= 20)
    check('challenge answer: 1,536 tokens is the longest stop under 50 ms without the cache (44.6 ms; 2,048 takes 60.0 ms)',
          best == 1536 and tS(nocache(1536)[0]) == '44.6 ms' and tS(nocache(2048)[0]) == '60.0 ms', (best, tS(nocache(1536)[0]), tS(nocache(2048)[0])))
    check('with the cache every stop takes 7.5 to 9.6 ms', tS(cached(256)[0]) == '7.5 ms' and tS(cached(32768)[0]) == '9.6 ms')
    for i, C in enumerate(STOPS):
        pg.evaluate("()=>document.querySelector('#kvb-app .modes [data-mode=guided]').click()"); pg.wait_for_timeout(120)
        pg.evaluate("()=>document.querySelector('#kvb-app .modes [data-mode=challenge]').click()"); pg.wait_for_timeout(200)
        set_range(pg, '#kvb-ic-chctx-in', i)
        now = pg.inner_text('#kvb-ch-now')
        pg.click('#kvb-rail [data-act="submit"]'); pg.wait_for_timeout(200)
        r = pg.inner_text('#kvb-ch-res').replace(' ', ' ')
        t = nocache(C)[0]; tier = 'exact' if C == best else ('short' if C < best else 'miss')
        head_ = {'exact': 'Goal met', 'short': 'Goal met', 'miss': 'Not yet'}[tier]
        extra = {'exact': 'longest context here that keeps up: the next one is already too slow', 'short': 'But a longer context still keeps up.', 'miss': 'Hint:'}[tier]
        # R6.20: the verdict's one number is the speed, in the goal's unit; the time per token is on the metric card (next check)
        check(f'challenge at {C:,}: tier {tier}, with the speed, marked as this lab’s estimate', r.startswith(head_) and extra in r and f'at {C:,} tokens of context it writes {"only " if tier == "miss" else ""}{tps(t)} tokens per second' in r and now == tps(t) and 'by this lab’s estimate' in r, (now, r[:240]))
        m = metrics(pg)
        check(f'challenge at {C:,}: the stage shows the two times only (the cache is off), the memory chart is hidden',
              list(m) == ['two', 'tw'] and m['two']['v'] == tS(t) and m['tw']['v'] == tS(cached(C)[0]) and not pg.is_visible('#kvb-mem'), (m, pg.is_visible('#kvb-mem')))
        if tier == 'miss':
            rv = pg.query_selector('#kvb-ch-res [data-act="reveal"]')
            check(f'challenge at {C:,}: the “Show the answer” link is there when the goal is missed', rv is not None and rv.inner_text() == 'Show the answer')
            if rv: rv.click(); pg.wait_for_timeout(150)
        rows = pg.evaluate("()=>[...document.querySelectorAll('#kvb-ch-res .cmp-row')].map(r=>({lab:r.querySelector('.cmp-lab span').textContent,val:r.querySelector('.cmp-val').textContent,ok:r.classList.contains('ok'),me:r.classList.contains('me')}))")
        # the owner's review of batch 8: the chart keeps the stops that pass, the pick and the initial setup (4K); the note counts the rest
        want = [(f'{S:,} tokens: {tS(nocache(S)[0])} per token', tps(nocache(S)[0]), 1 / shown_sec(nocache(S)[0]) >= 20, S == C) for S in STOPS
                if 1 / shown_sec(nocache(S)[0]) >= 20 or S == C or S == 4096]
        check(f'challenge chart at {C:,}: only the stops that reach 20 per second, your pick and the initial setup, in order', [(r['lab'], r['val'], r['ok'], r['me']) for r in rows] == want, (rows, want))
        note = pg.inner_text('#kvb-ch-res .cmp-note') if pg.query_selector('#kvb-ch-res .cmp-note') else ''
        check(f'challenge chart at {C:,}: the note counts the {len(STOPS) - len(want)} stops left out', note == f'The other {len(STOPS) - len(want)} context lengths give fewer than 20 tokens per second and are not shown.', note)
        # baseline → effect → reason under the chart (R3.10, R6.20): the answer as the one number, the trend in words
        r2 = sp(pg.inner_text('#kvb-ch-res'))
        base_t = nocache(4096)[0]
        check(f'challenge at {C:,}: under the chart, the initial setup is {"far " if 1 / shown_sec(base_t) < 10 else ""}too slow, the answer ({best:,} tokens) is named, and the time per token with the cache grows only a little',
              f'Initial setup, 4K tokens: {"far too slow" if 1 / shown_sec(base_t) < 10 else "too slow"}.' in r2 and f'The longest context here that meets the goal is {best:,} tokens.' in r2
              and 'With the KV cache, the time per token would grow only a little from the shortest context here to the longest.' in r2 and max(cached(S)[0] for S in STOPS) / min(cached(S)[0] for S in STOPS) < 1.5, r2[-900:])

    # ------------------------------------------------ scenarios
    pg.evaluate("()=>document.querySelector('#kvb-app .modes [data-mode=guided]').click()"); pg.wait_for_timeout(120)
    pg.evaluate("()=>document.querySelector('#kvb-app .modes [data-mode=sandbox]').click()"); pg.wait_for_timeout(200)
    for i, (Pp, Aa) in enumerate([(1000, 500), (32768, 1000), (4096, 50), (65536, 500)]):
        first = prefill(Pp)[0]                       # with the cache the answer starts with a prefill, which writes the prompt's keys and values
        w = first + sum(cached(C)[0] for C in range(Pp + 1, Pp + Aa))
        wo = sum(nocache(C)[0] for C in range(Pp, Pp + Aa))
        avg = (w - first) / (Aa - 1)
        page = pg.evaluate(f'()=>{K}.scen({Pp},{Aa})')
        check(f'scenario {i + 1}: totals equal the independent sums', abs(page['with'] - w) < 1e-9 and abs(page['without'] - wo) < 1e-7, (page, w, wo))
        pg.click(f'#kvb-rail [data-preset="{i}"]'); pg.wait_for_timeout(200)
        t = pg.inner_text('#kvb-insight').replace(' ', ' ')
        k = tot_sec(wo) / tot_sec(w)
        last = nocache(Pp + Aa - 1)[0]
        # R6.20: the sentence gives the two whole-answer times rounded; each carries on hover its exact value and its parts (the stage shows
        # one step, not the whole answer), the one without the cache also the ratio; the cache at the end in GiB with GB beside it (R6.12)
        tips = {sp(d[0]): sp(d[1]) for d in pg.evaluate("()=>[...document.querySelectorAll('#kvb-insight .def')].map(e=>[e.textContent,e.getAttribute('data-tipx')||''])")}
        tip_w, tip_wo = tips.get(f'about {totR(w)}', ''), tips.get(f'about {totR(wo)}', '')
        want = [f'by this lab’s estimate: with the KV cache the whole answer takes about {totR(w)}; without it, about {totR(wo)}.',
                f'by the last one it holds {sp(memS((Pp + Aa - 1) * kv_tok))} ({gbS((Pp + Aa - 1) * kv_tok)}).', f'The chart shows one step at {Pp:,} tokens of context']
        want_w = [sp(totS(w)), f'The model first runs the {Pp:,}-token prompt through once (the prefill), which takes {sp(tS(first))} and gives the first token',
                  f'each of the other {Aa - 1:,} tokens takes one decode step of about {avg * 1e3:.2f} ms', 'by this lab’s estimate']
        want_wo = [f'{sp(totS(wo))}, {ratioS(k)} times as long as with it', f'Every step, the first included, runs the whole conversation through the model: {Pp:,} tokens at the first step ({sp(tS(nocache(Pp)[0]))}), {Pp + Aa - 1:,} at the last ({sp(tS(last))})']
        check(f'scenario {i + 1} result box: the whole answer with and without the cache, rounded, and the cache at the end', all(x in t for x in want), (want, t[:400]))
        check(f'scenario {i + 1}: the rounded times carry on hover the exact totals, the first token, the per-token time, the steps without the cache and the ratio',
              all(x in tip_w for x in want_w) and all(x in tip_wo for x in want_wo), (want_w, tip_w, want_wo, tip_wo))
        # the with-cache total redoes from the shown first token and per-token time (to the shown precision)
        redo = shown_sec(first) + (Aa - 1) * round(avg * 1e3, 2) / 1e3
        check(f'scenario {i + 1}: first token + {Aa - 1} × per-token rounds to the shown total', totS(redo) == totS(w), (totS(redo), totS(w)))
    check('the four scenarios as in the spec (3.80 s / 18.0 s; 11.1 s / 25.4 min; 504 ms / 6.26 s; 9.99 s / 34.6 min)',
          [(totS(s['with']), totS(s['without'])) for s in [pg.evaluate(f'()=>{K}.scen({a},{b})') for a, b in [(1000, 500), (32768, 1000), (4096, 50), (65536, 500)]]]
          == [('3.80 s', '18.0 s'), ('11.1 s', '25.4 min'), ('504 ms', '6.26 s'), ('9.99 s', '34.6 min')])

    # ------------------------------------------------ interview drill
    for seed in range(1, 25):
        dr = pg.evaluate(f'()=>{K}.drill({seed})')
        m = dr['m']; per = 2 * m['L'] * m['kv'] * m['hd'] * 2; end = dr['P'] + dr['A'] - 1
        check(f'drill {seed}: {m["name"]}, P {dr["P"]}, A {dr["A"]}', dr['per'] == per and dr['endTok'] == end and dr['bytes'] == per * end
              and dr['without'] == sum(dr['P'] + k - 1 for k in range(1, dr['A'] + 1)) and dr['with'] == dr['P'] + dr['A'] - 1, dr)
    known = {'Llama-3.2-1B': (16, 8, 64), 'Llama-3.1-8B': (32, 8, 128), 'Mistral-7B-v0.1': (32, 8, 128), 'Llama-3.1-70B': (80, 8, 128)}
    ms = {pg.evaluate(f'()=>{K}.drill({s})')['m']['name']: pg.evaluate(f'()=>{K}.drill({s})')['m'] for s in range(1, 40)}
    check('drill models match their configs (facts file A1–A4)', all((m['L'], m['kv'], m['hd']) == known[n] for n, m in ms.items()) and len(ms) >= 3, ms)
    # the answer as shown redoes line by line
    pg.evaluate("()=>{const b=document.querySelector('[data-act=\"drill-reveal\"]'); b.click()}"); pg.wait_for_timeout(150)
    at = pg.inner_text('#kvb-drill-a').replace(' ', ' ')
    m1 = re.search(r'= 2 \(a key and a value\) × (\d+) layers × (\d+) KV heads × (\d+) × 2 bytes = ([\d,]+) bytes = ([\d,]+) KiB', at)
    m2 = re.search(r'([\d,]+) \+ ([\d,]+) − 1 = ([\d,]+) tokens .*?: ([\d,]+) × ([\d,]+) KiB = ([\d.,]+) (MiB|GiB)', at)
    m3 = re.search(r'([\d,]+) steps cost ([\d,]+) × ([\d,]+) \+ ([\d,]+) × ([\d,]+) ÷ 2 = ([\d,]+) \+ ([\d,]+) = ([\d,]+)', at)
    num = lambda s_: int(s_.replace(',', ''))
    ok = bool(m1 and m2 and m3)
    if ok:
        Lx, kvx, hdx, byx, kib = map(num, m1.groups())
        ok &= 2 * Lx * kvx * hdx * 2 == byx and byx / 1024 == kib
        a_, b_, e_, e2, k2, val, unit = m2.groups()
        ok &= num(a_) + num(b_) - 1 == num(e_) == num(e2) and num(k2) == kib and memS(num(e_) * kib * 1024).replace(' ', ' ') == f'{val} {unit}'
        g = list(map(num, m3.groups()))
        ok &= g[1] * g[2] == g[5] and g[3] * g[4] // 2 == g[6] and g[5] + g[6] == g[7]
    check('the drill answer redoes line by line from the numbers it shows', ok, at[:500])

    # ------------------------------------------------ the tile
    go(pg, 'map')
    pg.evaluate("()=>{const g=[...document.querySelectorAll('#domStrip [data-world=\"models\"], #domGrid [data-world=\"models\"]')].find(e=>e.offsetParent); if(g) g.click(); else {const m=document.querySelector('#mapSvg .world[data-world=\"models\"]'); if(m) m.dispatchEvent(new MouseEvent('click',{bubbles:true}));}}"); pg.wait_for_timeout(300)
    for n in (1, 2, 9, 32, 64):
        set_range(pg, '#tile-kv-basics input[type=range]', n)
        t = pg.inner_text('#tile-kv-basics .tz-read')
        check(f'tile at {n} tokens: {n * (n + 1) // 2:,} token-passes without a cache, {n} with one', f'{n * (n + 1) // 2:,}' in t and t.count(f'{n:,}') >= 1 and ('Writing 1 token' in t if n == 1 else f'Writing {n:,} tokens' in t), t)
    chip = pg.evaluate("()=>{const c=document.querySelector('#tile-kv-basics .chip.tz-kind'); return c?c.textContent:''}")
    check('the tile’s chip says Formula (the counts are a formula; the notes say so)', chip == 'Formula', chip)
    check('no page errors (lab-only build)', not errs, errs[:3])
    pg.close()

    # ------------------------------------------------ cross-check with the prefill-decode tile (teasers/set-a.js)
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs2 = []
    pg.on('pageerror', lambda e: errs2.append(str(e)))
    pg.goto(URL2); pg.wait_for_timeout(300)
    opts = pg.evaluate('()=>window.__tzA.pd.options')
    tz = pg.evaluate('()=>window.__tzA.consts')
    check('same constants as set-a (peak, bandwidth, efficiencies, weights, KV per token, bytes read)',
          c['H100']['peak'] == tz['peak'] and c['H100']['bw'] == tz['bw'] and c['EFF']['mfu'] == tz['mfu'] and c['EFF']['mbu'] == tz['mbu']
          and c['P_LAYERS'] == tz['P_LAYERS'] and c['P_HEAD'] == tz['P_HEAD'] and c['KV_TOK'] == tz['KV_TOK'] and c['W_READ'] == tz['W_READ'] and c['P_TOTAL'] == tz['P_TOTAL'], (c, tz))
    for N in opts:
        r = pg.evaluate(f'()=>window.__tzA.pd.compute({N})')
        pf = pg.evaluate(f'()=>window.__kvb.prefill({N})'); no = pg.evaluate(f'()=>window.__kvb.nocache({N})'); ca = pg.evaluate(f'()=>window.__kvb.cached({N + 1})')['t']
        check(f'prefill-decode tile at a {N:,}-token prompt: its prefill = this lab’s prefill (the first step with the cache); its decode step = this lab’s step with the cache at {N + 1:,}',
              pf['t'] == r['ttft'] and ca == r['tpot'], (pf['t'], r['ttft'], ca, r['tpot']))
        check(f'at {N:,} tokens this lab’s step without the cache does the tile prefill’s arithmetic but writes no cache: {N:,} × 131,072 bytes less memory traffic',
              no['fl'] == pf['fl'] and pf['by'] - no['by'] == N * c['KV_TOK'], (no, pf))
    check('no page errors (build with set-a)', not errs2, errs2[:3])
    b.close()

# ------------------------------------------------ the Connect code sample: at most 40 lines, and it runs (when PyTorch is installed)
code = re.search(r"const CODE=`(.*?)`;", (P / 'labs' / 'kv-basics.js').read_text(), re.S).group(1)
check('the minimal implementation is at most 40 lines', len(code.splitlines()) <= 40, len(code.splitlines()))
try:
    import torch  # noqa: F401
    run = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=300)
    out = run.stdout.strip()
    check('the minimal implementation runs and its two loops agree (largest difference under 1e-6, as its comment says)',
          run.returncode == 0 and float(out) < 1e-6 and 'under 1e-6' in code, (run.returncode, out, run.stderr[-300:]))
except ImportError:
    print('note: PyTorch is not installed, so the code sample was not run')

print(f'{n_checks} checks, {len(fails)} failed')
if fails: print('FAILED:', fails[:30])
sys.exit(1 if fails else 0)
