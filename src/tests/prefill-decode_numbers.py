"""Golden numbers for the prefill-decode lab (guide R9.2, CONTRACT §6).

The model is derived here independently of the JS: Llama-3.1-8B's parameters are counted component by component from its config
(q, k, v, o, gate, up, down and two RMSNorms per layer, the final RMSNorm, an untied embedding table and LM head); attention FLOPs are
counted pair by pair on small prompts and checked against the closed form; a step is priced as the slower of its arithmetic (at 50% of
the H100 SXM's dense BF16 peak, 1,979 TFLOPS "with sparsity" ÷ 2) and its memory traffic (at 60% of 3.35 TB/s).
Checks:
- the spec's verified numbers (prefill 512 → 14.6 ms … 32K → 1,493.5 ms; decode 7.60 / 8.00 ms; the batch at 4K: 7.73, 16.01, 19.75,
  20.02, 24.56 ms; ridge ≈ 246; intensities ≈ 1,943 and ≈ 1);
- the page's engine (window.__pd) against this derivation, and against the home tile (window.__tzA from teasers/set-a.js) where they
  overlap: the same constants, and for every tile prompt the same time to first token and time per output token;
- the most requests that fit, against the KV cache memory lab's own engine (window.__kv.compute) and a formula here;
- the challenge (46 users within 20 ms; 47 over), its caveat (70% of bandwidth), the scenarios' whole answers, the exits' A100 case,
  the interview drill, and the Connect card's code, run here;
- the numbers the page shows at the mission, at 8K, at 64 requests, and in the challenge;
- part 6, the experiment (twice the compute, or twice the bandwidth): every GPU (the H100, the H100 with twice the peak FLOP/s, the H100
  with twice the bandwidth) × both loads (chat: 2K prompt, 300-token answer; long document: 32K prompt, 20-token answer), recomputed
  here from the step formula: the time to first token, the time per output token (the first decode step, the lab's convention), the
  whole answer (every decode step as the context grows), which limit sets each phase, the pattern the spec expects (chat: compute × 2
  halves the first token and leaves the whole answer about the same, bandwidth × 2 halves every later token and nearly halves the whole
  answer; the long document flips it), and the cards and their change against the original as the part shows them.
Run: python3 tests/prefill-decode_numbers.py"""
import math, pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/prefill-decode.html', '--only', 'prefill-decode,set-a'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-prefill-decode-num.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'prefill-decode.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-prefill-decode-num.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


def close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


# ---------- the model, derived from the config (hidden 4,096, 32 layers, 32 heads, 8 KV heads, head_dim 128, FFN 14,336, vocab 128,256)
D, LAYERS, HEADS, KVH, HD, FFN, VOCAB = 4096, 32, 32, 8, 128, 14336, 128256
comp = {'q': D * HEADS * HD, 'k': D * KVH * HD, 'v': D * KVH * HD, 'o': HEADS * HD * D, 'gate': D * FFN, 'up': D * FFN, 'down': FFN * D, 'norms': 2 * D}
per_layer = sum(comp.values())
layers = LAYERS * per_layer + D                       # the 32 layers and the final RMSNorm: multiplied (or scaled) by every token
emb = head = VOCAB * D                                # untied: the embedding table is a lookup, the LM head a matmul per new token
total = layers + emb + head
check('parameters: 8,030,261,248 (the Hugging Face total, as A.kv.MODELS)', total == 8_030_261_248, total)
W_READ = 2 * (layers + head)                          # bytes per step: every parameter a step multiplies, BF16
W_MEM = 2 * total
KV_TOK = 2 * LAYERS * KVH * HD * 2                    # K and V, BF16
check('KV cache per token: 131,072 bytes = 128 KiB', KV_TOK == 131072)
PEAK, BW, MFU, MBU = 1979e12 / 2, 3.35e12, 0.5, 0.6


def attn_pairs_loop(a, b):
    """query–key pairs for query positions a..b-1 under the causal mask, counted one by one"""
    return sum(i + 1 for i in range(a, b))


def attn_flops(a, b):
    # 4 FLOPs per pair per dimension of the hidden size (2 for q·k, 2 for adding the value), in every layer
    pairs = (b * (b + 1) - a * (a + 1)) // 2
    return 4 * LAYERS * D * pairs


for a, b in [(0, 1), (0, 7), (0, 64), (5, 6), (100, 101), (10, 50)]:
    check(f'attention pairs {a}..{b}: closed form equals the count', (b * (b + 1) - a * (a + 1)) // 2 == attn_pairs_loop(a, b))


def step(tokens, logits, attn, kv, peak=PEAK, bw=BW):
    fl = 2 * layers * tokens + 2 * head * logits + attn
    by = W_READ + KV_TOK * kv
    tc, tm = fl / (peak * MFU), by / (bw * MBU)
    return {'t': max(tc, tm), 'tc': tc, 'tm': tm, 'fl': fl, 'by': by}


def prefill(N, **g): return step(N, 1, attn_flops(0, N), N, **g)
def decode(B, c, **g): return step(B, B, B * attn_flops(c, c + 1), B * c, **g)


ms = lambda s: s * 1e3
# ---------- the spec's verified numbers
for N, want in [(512, 14.6), (2048, 60.0), (8192, 266.7), (32768, 1493.5)]:
    p = prefill(N)
    check(f'spec: prefill {N} → {want} ms, arithmetic sets it', round(ms(p['t']), 1) == want and p['tc'] > p['tm'], ms(p['t']))
d2, d8 = decode(1, 2048), decode(1, 8192)
check('spec: decode at 2K, one request → 7.60 ms (memory 7.60 ms, arithmetic 0.033 ms)', round(ms(d2['t']), 2) == 7.60 and round(ms(d2['tm']), 2) == 7.60 and round(ms(d2['tc']), 3) == 0.033, (ms(d2['tm']), ms(d2['tc'])))
check('spec: decode at 8K → 8.00 ms', round(ms(d8['t']), 2) == 8.00, ms(d8['t']))
for B, want, tps in [(1, 7.73, 129), (32, 16.01, 1998), (46, 19.75, None), (47, 20.02, None), (64, 24.56, 2606)]:
    d = decode(B, 4096)
    check(f'spec: {B} requests at 4K → {want} ms' + (f' ({tps} tokens/s)' if tps else ''), round(ms(d['t']), 2) == want and (tps is None or round(B / d['t']) == tps), (ms(d['t']), B / d['t']))
RIDGE = PEAK * MFU / (BW * MBU)
check('spec: the ridge ≈ 246 FLOPs per byte', round(RIDGE) == 246, RIDGE)
p2 = prefill(2048)
check('spec: prefill of 2K ≈ 1,943 FLOPs per byte; a decode step ≈ 1', round(p2['fl'] / p2['by']) == 1943 and round(d2['fl'] / d2['by']) == 1, (p2['fl'] / p2['by'], d2['fl'] / d2['by']))
check('a single-token step does exactly 1 FLOP per weight byte before attention and the KV cache (2 FLOPs, 2 bytes per parameter)', (2 * layers + 2 * head) == W_READ)


# ---------- the most requests that fit: the KV cache memory lab's model (80 GB read as 80 GiB, 90% usable, 2 GiB overhead)
GiB = 2 ** 30
def mem_limit(c): return math.floor((80 * GiB * 0.9 - W_MEM - 2 * GiB) / (KV_TOK * c) + 1e-9)
check('memory limit at 4K: 110 requests (the spec’s “about 106” left out the embedding table and the overhead, and read 80 GB as 10^9 bytes)', mem_limit(4096) == 110)


def answer(N, M, B):
    pre = prefill(N)
    steps = [decode(B, N + k)['t'] for k in range(M - 1)]
    return pre['t'], steps, pre['t'] + sum(steps)


def chat_best(goal=0.020, mbu=0.6):
    best = 0
    for B in range(1, mem_limit(4096) + 1):
        if decode(B, 4096, bw=BW * mbu / MBU)['t'] <= goal:
            best = B
    return best


check('challenge: 46 users within 20 ms at 4K tokens each', chat_best() == 46)
check('challenge caveat: 59 users if the kernels reached 70% of the bandwidth', chat_best(mbu=0.7) == 59, chat_best(mbu=0.7))

A100 = {'peak': 312e12, 'bw': 2.039e12}
a_dec, a_pre = decode(1, 2048, **A100), prefill(2048, **A100)
check('exits: A100 80GB SXM decode step at 2K ≈ 12.49 ms, memory sets it (answer: 10 to 18 ms)', round(ms(a_dec['t']), 2) == 12.49 and 10 <= ms(a_dec['t']) < 18 and a_dec['tm'] > a_dec['tc'], ms(a_dec['t']))
check('exits: A100 80GB SXM prefill of 2K ≈ 190.3 ms, arithmetic sets it (answer: 140 to 250 ms)', round(ms(a_pre['t']), 1) == 190.3 and 140 <= ms(a_pre['t']) < 250 and a_pre['tc'] > a_pre['tm'], ms(a_pre['t']))

# ---------- part 6, the experiment: one number of the H100 doubled, everything else the same (the spec's "Model")
XV = {'orig': {}, 'c2': {'peak': 2 * PEAK}, 'b2': {'bw': 2 * BW}}
XL = {'chat': (2048, 300), 'doc': (32768, 20)}
X = {}
for v, g in XV.items():
    for l, (N, M) in XL.items():
        pre, dec = prefill(N, **g), decode(1, N, **g)
        steps = [decode(1, N + k, **g)['t'] for k in range(M - 1)]
        X[v, l] = {'ttft': pre['t'], 'tpot': dec['t'], 'e2e': pre['t'] + sum(steps), 'pre': pre, 'dec': dec, 'avg': sum(steps) / (M - 1), 'N': N, 'M': M}
for l in XL:
    for v in XV:
        x = X[v, l]
        check(f'part 6 {v} {l}: arithmetic sets the prefill, memory sets a decode step', x['pre']['tc'] > x['pre']['tm'] and x['dec']['tm'] > x['dec']['tc'], (x['pre'], x['dec']))
o, c2, b2 = X['orig', 'chat'], X['c2', 'chat'], X['b2', 'chat']
check('part 6 chat: the original equals the mission (60.0 ms, 7.60 ms, 2.34 s)', round(ms(o['ttft']), 1) == 60.0 and round(ms(o['tpot']), 2) == 7.60 and round(o['e2e'], 2) == 2.34, (o['ttft'], o['tpot'], o['e2e']))
check('part 6 chat: compute × 2 halves the first token (60.0 → 30.0 ms), leaves every later token (7.60 ms) and the whole answer about the same (2.34 → 2.31 s)',
      close(c2['ttft'], o['ttft'] / 2) and close(c2['tpot'], o['tpot']) and round(ms(c2['ttft']), 1) == 30.0 and round(c2['e2e'], 2) == 2.31 and 0.98 < c2['e2e'] / o['e2e'] < 1, (c2['ttft'], c2['e2e']))
check('part 6 chat: bandwidth × 2 leaves the first token (60.0 ms), halves every later token (7.60 → 3.80 ms) and nearly halves the whole answer (2.34 → 1.20 s)',
      close(b2['ttft'], o['ttft']) and close(b2['tpot'], o['tpot'] / 2) and round(ms(b2['tpot']), 2) == 3.80 and round(b2['e2e'], 2) == 1.20 and 0.5 < b2['e2e'] / o['e2e'] < 0.55, (b2['tpot'], b2['e2e']))
od, c2d, b2d = X['orig', 'doc'], X['c2', 'doc'], X['b2', 'doc']
check('part 6 long document: mostly prefill (89%); compute × 2 wins the whole answer (1.68 → 0.93 s), bandwidth × 2 only a little (1.68 → 1.58 s)',
      0.85 < od['ttft'] / od['e2e'] < 0.95 and round(od['e2e'], 2) == 1.68 and round(c2d['e2e'], 2) == 0.93 and round(b2d['e2e'], 2) == 1.58 and c2d['e2e'] < b2d['e2e'] < od['e2e'] and close(c2d['ttft'], od['ttft'] / 2) and close(b2d['tpot'], od['tpot'] / 2),
      (od['e2e'], c2d['e2e'], b2d['e2e']))


def f_pre(sec):
    v = sec * 1e3
    return f'{v:,.1f} ms' if v < 999.95 else f'{v / 1e3:,.2f} s'


def f_dec(sec):
    v = sec * 1e3
    return f'{v:,.3f} ms' if v < 0.9995 else f'{v:,.2f} ms'


def f_tot(sec):
    return f'{round(sec * 1e3):,} ms' if sec < 0.9995 else (f'{sec:.2f} s' if sec < 9.995 else f'{sec:.1f} s')


def shown_s(t):
    v, u = t.replace(',', '').split(' ')
    return float(v) / (1000 if u == 'ms' else 1)


def x_s3(sec):
    """part 6, the long document: the first token and the whole answer in s, 3 significant digits (one unit per column, review A4)"""
    return f'{sec:.3f} s' if sec < 0.9995 else (f'{sec:.2f} s' if sec < 9.995 else f'{sec:.1f} s')


def x_ft(sec, l): return x_s3(sec) if l == 'doc' else f_pre(sec)
def x_wa(sec, l): return x_s3(sec) if l == 'doc' else f_tot(sec)


def cmp_txt(a, b):
    """the card's change against the original, from the two values as shown, as the guide writes a change (R6.6): a decrease as a
    percentage rounded half away from zero, “▼ −50% vs the original”"""
    if a == b: return 'same as the original'
    va, ua = a.replace(',', '').split(' '); vb, ub = b.replace(',', '').split(' ')
    assert ua == ub, (a, b)
    dp = max(len(va.split('.')[1]) if '.' in va else 0, len(vb.split('.')[1]) if '.' in vb else 0)
    x, y = round(float(va), dp), round(float(vb), dp)
    r = (y / x - 1) * 100; pc = int(math.copysign(math.floor(abs(r) + 0.5 + 1e-9), r))
    return ('▲ +' if pc > 0 else '▼ −') + f'{abs(pc)}% vs the original'


check('part 6 the cards’ changes, recomputed from the shown values: chat −50% / −1% (compute × 2), −50% / −49% (bandwidth × 2); long document −50% / −45%, −50% / −6%',
      [cmp_txt(x_ft(o['ttft'], 'chat'), x_ft(c2['ttft'], 'chat')), cmp_txt(x_wa(o['e2e'], 'chat'), x_wa(c2['e2e'], 'chat')), cmp_txt(f_dec(o['tpot']), f_dec(b2['tpot'])), cmp_txt(x_wa(o['e2e'], 'chat'), x_wa(b2['e2e'], 'chat'))]
      == ['▼ −50% vs the original', '▼ −1% vs the original', '▼ −50% vs the original', '▼ −49% vs the original']
      and [cmp_txt(x_ft(X['orig', 'doc']['ttft'], 'doc'), x_ft(X['c2', 'doc']['ttft'], 'doc')), cmp_txt(x_wa(X['orig', 'doc']['e2e'], 'doc'), x_wa(X['c2', 'doc']['e2e'], 'doc')), cmp_txt(x_wa(X['orig', 'doc']['e2e'], 'doc'), x_wa(X['b2', 'doc']['e2e'], 'doc'))]
      == ['▼ −50% vs the original', '▼ −45% vs the original', '▼ −6% vs the original'])


with sync_playwright() as pw:
    b = pw.chromium.launch()
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(80)
    pg.evaluate("()=>{location.hash='prefill-decode'}"); pg.wait_for_timeout(500)
    J = lambda expr: pg.evaluate(expr)
    c = J('()=>window.__pd.consts')
    for k, v in [('P_TOTAL', total), ('P_LAYERS', layers), ('P_HEAD', head), ('P_EMB', emb), ('KV_TOK', KV_TOK), ('W_READ', W_READ), ('W_MEM', W_MEM)]:
        check(f'page constant {k} = {v}', c[k] == v, c[k])
    check('page ridge equals the derivation', close(c['RIDGE'], RIDGE))
    # ---------- the lab and the home tile: the same step model, the same constants, the same numbers where they overlap
    tz = J('()=>window.__tzA&&window.__tzA.consts')
    check('the home tile (set-a) is in the page', tz is not None)
    if tz:
        for k in ['P_TOTAL', 'P_LAYERS', 'P_EMB', 'P_HEAD', 'KV_TOK', 'W_READ']:
            check(f'tile and lab share {k}', tz[k] == c[k], (tz[k], c[k]))
        check('tile and lab share the peaks and efficiencies', tz['peak'] == c['H100']['peak'] and tz['bw'] == c['H100']['bw'] and tz['mfu'] == c['EFF']['mfu'] and tz['mbu'] == c['EFF']['mbu'])
        for N in J('()=>window.__tzA.pd.options'):
            t = J(f'()=>window.__tzA.pd.compute({N})')
            lp, ld = J(f'()=>window.__pd.prefill({N})'), J(f'()=>window.__pd.decode(1,{N})')
            check(f'tile and lab agree at a {N}-token prompt: TTFT and TPOT', t['ttft'] == lp['t'] and t['tpot'] == ld['t'], (t['ttft'], lp['t'], t['tpot'], ld['t']))
    # ---------- the page's engine against the derivation
    for N in c['N_STOPS']:
        lp = J(f'()=>window.__pd.prefill({N})'); pp = prefill(N)
        check(f'prefill {N}: FLOPs, bytes and time', lp['fl'] == pp['fl'] and lp['by'] == pp['by'] and close(lp['t'], pp['t']), (lp['t'], pp['t']))
        for B in (1, 7, 64):
            ld, pd_ = J(f'()=>window.__pd.decode({B},{N})'), decode(B, N)
            check(f'decode {B} at {N}: FLOPs, bytes and time', ld['fl'] == pd_['fl'] and ld['by'] == pd_['by'] and close(ld['t'], pd_['t']), (ld['t'], pd_['t']))
        lim = J(f'()=>window.__pd.memLimit({N})')
        kv = J(f"()=>window.__kv.compute({{model:'llama31-8b',gpu:'h100',n:1,ctx:{N},users:1,wp:'bf16',kp:'bf16',attn:'gqa',util:0.9,overhead:2}}).maxUsers")
        check(f'memory limit at {N}: the page, the KV cache memory lab’s engine and the formula agree', lim == kv == mem_limit(N), (lim, kv, mem_limit(N)))
    for N, M, B in [(1024, 300, 1), (32768, 300, 1), (4096, 20, 1), (2048, 300, 64), (2048, 300, 1), (8192, 1000, 16)]:
        r = J(f'()=>window.__pd.compute({{N:{N},M:{M},B:{B}}})')
        ttft, steps, e2e = answer(N, M, B)
        check(f'whole answer N={N} M={M} B={B}: the sum of every step as the context grows', close(r['e2e'], e2e) and close(r['avg'], sum(steps) / (M - 1)) and len(r['ticks']) == min(M - 1, c['TL_K']), (r['e2e'], e2e))
        check(f'N={N} M={M} B={B}: throughput = B ÷ TPOT', close(r['thr'], B / decode(B, N)['t']))
    check('page: challenge best = 46', J('()=>window.__pd.chBest()') == 46)
    check('page: challenge caveat at 70% = 59', J('()=>window.__pd.chBestAt(0.7)') == 59)
    for B in (46, 47):
        cp = J(f'()=>window.__pd.chPoint({B})')
        check(f'page: challenge point {B}', close(cp['t'], decode(B, 4096)['t']) and cp['ok'] == (B == 46))
    la, lp2 = J("()=>window.__pd.decode(1,2048,'a100')"), J("()=>window.__pd.prefill(2048,'a100')")
    check('page: the A100 case equals the derivation', close(la['t'], a_dec['t']) and close(lp2['t'], a_pre['t']))
    for s in range(1, 21):
        d = J(f'()=>window.__pd.drill({s})')
        check(f'drill {s}: prefill and decode from the derivation, within the memory limit', close(d['pre']['t'], prefill(d['N'])['t']) and close(d['dec']['t'], decode(d['B'], d['N'])['t']) and d['B'] <= mem_limit(d['N']) and d['N'] != 4096, d)
    # ---------- the Connect card's code, run here: its comments state its own output
    code = J("()=>document.getElementById('pd-code').textContent")
    check('Connect code: at most 40 lines, none longer than 74 characters', len(code.splitlines()) <= 40 and max(len(l) for l in code.splitlines()) <= 74, (len(code.splitlines()), max(len(l) for l in code.splitlines())))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True).stdout.strip().splitlines()
    promised = [l[2:] for l in code.splitlines() if l.startswith('# ') and ('TTFT' in l or 'tokens/s' in l)]
    check('Connect code: runs and prints what its comments say', out == promised, (out, promised))
    # ---------- what the page shows (the displayed values, from the derivation)
    def cards():
        return J("()=>Object.fromEntries([...document.querySelectorAll('#pd-metrics .metric')].map(m=>[m.dataset.k,m.querySelector('.v').textContent]))")
    got = cards()
    want = {'ttft': '60.0 ms', 'tpot': '7.60 ms', 'tps': '132', 'thr': '132', 'e2e': f"{answer(2048, 300, 1)[2]:.2f} s"}
    check('mission cards: 60.0 ms, 7.60 ms, 132, 132, 2.34 s', got == want and want['e2e'] == '2.34 s', got)
    tl = J("()=>document.querySelector('.pd-tl-labs').innerText")
    check('timeline labels: first token at 60.0 ms, then one every 7.60 ms', 'first token at 60.0' in tl and 'every 7.60' in tl, tl)
    # 8K and 64 requests, in Scenarios (no masking there)
    J("()=>document.querySelector('#pd-app .modes button[data-mode=\"sandbox\"]:not([data-flow=free])').click()"); pg.wait_for_timeout(300)
    def set_state(N, M, B):
        J(f"""()=>{{const f=(k,v)=>{{const i=document.querySelector('#pd-controls input[data-key="'+k+'"]'); i.value=v; i.dispatchEvent(new Event('input',{{bubbles:true}}));}};
          f('N',{c['N_STOPS'].index(N)}); f('M',{c['M_STOPS'].index(M)}); f('B',{B});}}"""); pg.wait_for_timeout(200)
    set_state(8192, 300, 1)
    got = cards()
    check('8K cards: 266.7 ms and 8.00 ms, 125 tokens per second', got['ttft'] == '266.7 ms' and got['tpot'] == '8.00 ms' and got['tps'] == str(round(1 / decode(1, 8192)['t'])), got)
    set_state(2048, 300, 64)
    got = cards(); d64 = decode(64, 2048)
    check('64 requests at 2K: 16.01 ms and 3,996 tokens per second in all (64 ÷ 16.0148 ms)', got['tpot'] == '16.01 ms' and got['thr'] == f"{round(64 / d64['t']):,}" == '3,996', got)
    J("()=>document.querySelector('#pd-app .modes button[data-mode=\"challenge\"]').click()"); pg.wait_for_timeout(300)
    J("()=>{const i=document.querySelector('#pd-ic-chB-in'); i.value=47; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(200)
    got = cards()
    check('challenge at 47 users: 20.02 ms and 49.9 per user (never “50.0”)', got['tpot'] == '20.02 ms' and got['tps'] == '49.9', got)
    J("()=>{const i=document.querySelector('#pd-ic-chB-in'); i.value=46; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(200)
    got = cards()
    check('challenge at 46 users: 19.75 ms, 50.6 per user, 2,329 in all', got['tpot'] == '19.75 ms' and got['tps'] == '50.6' and got['thr'] == '2,329', got)
    # the goal is checked at 4K tokens of context: no whole-answer figure (its growing context averages 20.20 ms) in the challenge
    tlc = J("()=>document.getElementById('pd-tl-cap').innerText")
    check('challenge: no whole-answer card or caption (an answer’s growing context is not part of the goal)', 'e2e' not in got and 'whole answer' not in tlc.lower(), (list(got), tlc))
    # the fidelity note: for any efficiencies from 20% to 100% the ridge stays between a decode step and a 2K prefill
    lo, hi = PEAK * 0.2 / BW, PEAK / (BW * 0.2)
    ai_p, ai_d = p2['fl'] / p2['by'], d2['fl'] / d2['by']
    fid = J("()=>document.getElementById('prefill-decode-fid').textContent").replace('\u00a0', ' ').replace('\u202f', ' ')
    want_f = f"between {round(lo)} and {round(hi):,} FLOPs per byte, while the prefill of 2K tokens does {round(ai_p):,} and one decode step {ai_d:.2f}"
    check(f'fidelity note: {want_f}', want_f in fid and ai_d < lo and hi < ai_p, fid[:300])
    # ---------- part 6: the page's engine, and what the part shows for every GPU × both loads
    cx = J('()=>window.__pd.consts')
    check('part 6: the page’s variants double one number each (peak FLOP/s, bandwidth), the loads are 2K/300 and 32K/20',
          {v['id']: (v['peak'], v['bw']) for v in cx['XV']} == {'orig': (PEAK, BW), 'c2': (2 * PEAK, BW), 'b2': (PEAK, 2 * BW)} and {l['id']: (l['N'], l['M']) for l in cx['XL']} == XL
          and cx['STEP_IDS'] == ['mission', 'predict', 'prefill', 'decode', 'batch', 'double', 'remember'], (cx['XV'], cx['XL'], cx['STEP_IDS']))
    for (v, l), x in X.items():
        r = J(f"()=>window.__pd.xrun('{v}','{l}')")
        check(f'part 6 engine {v} {l}: TTFT, TPOT, whole answer, average decode step and both limits of each phase equal the derivation',
              close(r['ttft'], x['ttft']) and close(r['tpot'], x['tpot']) and close(r['e2e'], x['e2e']) and close(r['avg'], x['avg'])
              and close(r['pre']['tc'], x['pre']['tc']) and close(r['pre']['tm'], x['pre']['tm']) and close(r['dec']['tc'], x['dec']['tc']) and close(r['dec']['tm'], x['dec']['tm'])
              and r['pre']['cb'] and not r['dec']['cb'] and len(r['times']) == x['M'] - 1 and close(r['times'][-1], x['e2e']), (r['ttft'], r['e2e'], x['ttft'], x['e2e']))
    for l, (N, M) in XL.items():
        base = J(f'()=>window.__pd.compute({{N:{N},M:{M},B:1}})'); r = J(f"()=>window.__pd.xrun('orig','{l}')")
        check(f'part 6 {l}: the original equals the rest of the lab (compute) exactly', r['ttft'] == base['ttft'] and r['tpot'] == base['tpot'] and r['e2e'] == base['e2e'], (r['e2e'], base['e2e']))
    J("()=>document.querySelector('#pd-app .modes button[data-mode=\"guided\"]').click()"); pg.wait_for_timeout(300)
    J("()=>document.querySelector('#pd-rail [data-go=\"5\"]').click()"); pg.wait_for_timeout(400)
    J("()=>document.querySelector('#pd-rail [data-xpred=\"first\"]').click()"); pg.wait_for_timeout(250)
    for l in XL:
        J(f"()=>document.querySelector('#pd-rail .seg-ctl[data-key=\"xl\"] button[data-v=\"{l}\"]').click()"); pg.wait_for_timeout(150)
        for v in XV:
            J(f"()=>document.querySelector('#pd-rail [data-lever=\"{v}\"]').click()"); pg.wait_for_timeout(150)
            x, o = X[v, l], X['orig', l]
            got = J("()=>Object.fromEntries([...document.querySelectorAll('#pd-metrics .metric')].map(m=>[m.dataset.k,[m.querySelector('.v').textContent,m.querySelector('.d').textContent]]))")
            got = {k: [a.replace('\xa0', ' ') for a in vv] for k, vv in got.items()}
            want = {'ttft': [x_ft(x['ttft'], l), ''], 'tpot': [f_dec(x['tpot']), ''], 'e2e': [x_wa(x['e2e'], l), '']}
            if v != 'orig':
                want['ttft'][1] = cmp_txt(x_ft(o['ttft'], l), x_ft(x['ttft'], l)); want['tpot'][1] = cmp_txt(f_dec(o['tpot']), f_dec(x['tpot'])); want['e2e'][1] = cmp_txt(x_wa(o['e2e'], l), x_wa(x['e2e'], l))
            check(f'part 6 cards {v} {l}: TTFT, TPOT and whole answer, each with its change against the original', got == want, (got, want))
            row = J(f"()=>{{const r=document.querySelector('.pd-xrow[data-xv=\"{v}\"]'); return [r.querySelector('.pd-xrow-f').textContent, r.querySelector('.pd-xrow-e').textContent]}}")
            check(f'part 6 timeline {v} {l}: first token and whole answer on its row, one unit per load', [a.replace('\xa0', ' ') for a in row] == [f'first token {x_ft(x["ttft"], l)}', f'whole answer {x_wa(x["e2e"], l)}'], row)
            res = J("()=>[...document.querySelectorAll('#pd-rail [data-lever] .lv-r')].map(e=>e.textContent.replace(/\\u00a0/g,' '))")
            check(f'part 6 options {v} {l}: each GPU’s whole answer on its title row, in the load’s one unit', res == [x_wa(X[g, l]['e2e'], l) for g in XV], res)
            lim = J("()=>[...document.querySelectorAll('#pd-lim .pd-ph')].map(p=>[p.querySelector('[data-k=\"c\"]').classList.contains('pd-set'), p.querySelector('[data-k=\"m\"]').classList.contains('pd-set')])")
            check(f'part 6 bars {v} {l}: “sets the time” on prefill’s arithmetic and on the decode step’s memory', lim == [[True, False], [False, True]], lim)
    check('no page errors', not errs, errs[:3])
    b.close()
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails if fails else 'none')
sys.exit(1 if fails else 0)
