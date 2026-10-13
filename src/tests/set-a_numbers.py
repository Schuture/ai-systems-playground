"""Set A teasers (prefill-decode, chunked-prefill, prefix-caching, quantization, spec-decoding):
recompute every number the five tiles can show, for every control position, independently in Python,
and compare with the page: raw values through the read-only hook window.__tzA, and the rendered text
(sentence, visual labels, notes, tooltips) after operating each control the way a user does.

Run from the prototype folder:  python3 tests/set-a_numbers.py
It builds out/set-a.html (only teasers/set-a.*) and wraps it as out/wrapped-set-a.html."""
import pathlib, subprocess, re, bisect, sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run(['python3', 'build.py', 'out/set-a.html', '--only', 'set-a'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'wrapped-set-a.html').write_text(
    '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>'
    + (P / 'out' / 'set-a.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-set-a.html')

FAILS, CHECKS = [], [0]
def check(cond, what):
    CHECKS[0] += 1
    if not cond:
        FAILS.append(what)
def close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-30)

# ---------------------------------------------------------------- facts, from the primary sources
# NVIDIA H100 datasheet, H100 SXM column: BF16 Tensor Core 1,979 TFLOPS "* With sparsity"; memory bandwidth 3.35 TB/s.
PEAK_DENSE = 1979e12 / 2
HBM_BW = 3.35e12
MFU, MBU = 0.5, 0.6            # the tiles' stated assumptions
# meta-llama/llama-models, models/sku_list.py, Llama 3.1 8B: dim 4096, n_layers 32, n_heads 32, n_kv_heads 8,
# vocab 128256, ffn_dim_multiplier 1.3, multiple_of 1024. FFN width as the reference code computes it:
DIM, LAYERS, HEADS, KV_HEADS, VOCAB = 4096, 32, 32, 8, 128256
hidden = int(2 * (4 * DIM) / 3); hidden = int(1.3 * hidden); FFN = 1024 * ((hidden + 1024 - 1) // 1024)
HEAD_DIM = DIM // HEADS
TENSORS = {'embed_tokens': (VOCAB, DIM), 'lm_head': (VOCAB, DIM), 'norm': (DIM,)}
for l in range(LAYERS):
    TENSORS.update({f'{l}.q_proj': (HEADS * HEAD_DIM, DIM), f'{l}.k_proj': (KV_HEADS * HEAD_DIM, DIM), f'{l}.v_proj': (KV_HEADS * HEAD_DIM, DIM),
                    f'{l}.o_proj': (DIM, HEADS * HEAD_DIM), f'{l}.gate_proj': (FFN, DIM), f'{l}.up_proj': (FFN, DIM), f'{l}.down_proj': (DIM, FFN),
                    f'{l}.input_layernorm': (DIM,), f'{l}.post_attention_layernorm': (DIM,)})
size = lambda s: s[0] * (s[1] if len(s) > 1 else 1)
N_TOTAL = sum(size(s) for s in TENSORS.values())
N_EMBED, N_HEAD = size(TENSORS['embed_tokens']), size(TENSORS['lm_head'])
N_BODY = N_TOTAL - N_EMBED - N_HEAD                  # everything each token is multiplied by, except the output layer
KV_PER_TOKEN = LAYERS * 2 * KV_HEADS * HEAD_DIM * 2  # K and V, 2 bytes each (BF16)
WEIGHT_BYTES_PER_STEP = 2 * (N_BODY + N_HEAD)        # the embedding table is looked up, not streamed
check(FFN == 14336, f'FFN width {FFN}')
check(N_TOTAL == 8_030_261_248, f'Llama-3.1-8B total {N_TOTAL} != 8,030,261,248 (Hugging Face safetensors count)')
check(KV_PER_TOKEN == 128 * 1024, f'KV per token {KV_PER_TOKEN}')

def attention_flops(first, last):
    """QK^T and AV for query positions first..last-1 (causal: position i sees i+1 keys), all layers: 2 FLOPs per multiply-add."""
    keys = sum(range(first + 1, last + 1))
    return LAYERS * (2 * DIM * keys + 2 * DIM * keys)

def gpu_step(tokens, logit_tokens, attn, kv_entries):
    flops = 2 * N_BODY * tokens + 2 * N_HEAD * logit_tokens + attn
    bytes_ = WEIGHT_BYTES_PER_STEP + KV_PER_TOKEN * kv_entries
    t_compute, t_memory = flops / (PEAK_DENSE * MFU), bytes_ / (HBM_BW * MBU)
    return max(t_compute, t_memory), t_compute, t_memory, flops, bytes_

# ---------------------------------------------------------------- JS number formatting, emulated exactly
def js_fixed(x, d):  # Number.prototype.toFixed: exact binary value, ties away from zero
    return str(Decimal(x).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
def num(x, d):       # ASP.fmt.num: toFixed, then drop one trailing ".0"
    return re.sub(r'\.0(?=\s|$)', '', js_fixed(x, d), count=1)
def js_str(x):       # String(number) for the magnitudes used here
    r = repr(float(x)); return r[:-2] if r.endswith('.0') else r
def js_prec(x, p):   # Number(x.toPrecision(p))
    if x == 0: return 0.0
    dx = Decimal(x); e = dx.adjusted()
    return float(dx.quantize(Decimal(1).scaleb(e - p + 1), rounding=ROUND_HALF_UP))
NB = ' '
def ms(sec):
    v = sec * 1e3
    if v >= 999.5: return num(v / 1000, 2) + NB + 's'
    if v >= 99.95: return num(v, 0) + NB + 'ms'
    return num(v, 1) + NB + 'ms'
def fint(n): return f'{round(n):,}'
def fbig(n):   # R6.18: more than 6 digits -> 3 significant digits, decimal M/B (the kit's A.fmt.big)
    import math
    if n < 1e6: return fint(n)
    d, u = (1e9, 'B') if n >= 1e9 else (1e6, 'M'); v = n / d; dp = max(0, 2 - math.floor(math.log10(v)))
    return f'{v:.{dp}f}{u}'
def ms1(sec):                         # 1,494.0 ms: the old first-token format, which must no longer appear
    i, f = js_fixed(sec * 1e3, 1).split('.'); return f'{int(i):,}.{f}' + NB + 'ms'
def ms0(sec):                         # 1,494 ms: first-token times in whole ms (R8 re-check 13a); JS Math.round, then en-US grouping
    return f'{int(Decimal(sec * 1e3).quantize(Decimal(1), rounding=ROUND_HALF_UP)):,}' + NB + 'ms'
def tenths(sec): return int(Decimal(js_fixed(sec * 1e3, 1)) * 10)
def words(s): return len(s.split())
def tokK(n): return f'{n // 1024}K' if n >= 1024 and n % 1024 == 0 else fint(n)
def stored(x):
    s = js_str(x); d = len(re.sub(r'^0\.0*|\.', '', s))
    return s if d <= 8 else '≈' + js_str(js_prec(x, 6))
def stored_short(x):
    s = stored(x); return s if len(s) <= 8 else '≈' + js_str(js_prec(x, 4))
def norm(s): return ' '.join(s.replace(NB, ' ').split())
def has(text, piece): return norm(piece) in norm(text)

# ---------------------------------------------------------------- 1. prefill-decode
PD_N = [512, 1024, 2048, 4096, 8192, 16384, 32768]
def pd_expect(n):
    pre = gpu_step(n, 1, attention_flops(0, n), n)          # the whole prompt in one pass
    dec = gpu_step(1, 1, attention_flops(n, n + 1), n)      # one new token: weights + the prompt's KV cache
    return dict(ttft=pre[0], tpot=dec[0], pre_compute=pre[1], pre_memory=pre[2], flops=pre[3], dec_bytes=dec[4], dec_compute=dec[1], dec_memory=dec[2])

# ---------------------------------------------------------------- 2. chunked-prefill
USERS, USER_CTX, LONG = 16, 4096, 32768
def cp_expect(chunk):
    size_ = chunk or LONG
    decode_attn = USERS * attention_flops(USER_CTX, USER_CTX + 1)
    normal = gpu_step(USERS, USERS, decode_attn, USERS * USER_CTX)
    durations, done = [], 0
    while done < LONG:
        nxt = min(LONG, done + size_)
        durations.append(gpu_step(USERS + nxt - done, USERS + (1 if nxt == LONG else 0), decode_attn + attention_flops(done, nxt), USERS * USER_CTX + nxt))
        done = nxt
    return dict(normal=normal[0], normal_compute=normal[1], normal_memory=normal[2], longest=max(d[0] for d in durations), ttft=sum(d[0] for d in durations),
                steps=len(durations), all_compute_bound=all(d[1] >= d[2] for d in durations))

# ---------------------------------------------------------------- 3. prefix-caching
PC_N = [1, 10, 100, 1000]
def pc_expect(n):
    without = sum(10_000 + 200 for _ in range(n))
    with_ = 10_000 + sum(200 for _ in range(n))
    return dict(without=without, with_=with_, pct=100 * (without - with_) / without, kv=10_000 * KV_PER_TOKEN)

# ---------------------------------------------------------------- 4. quantization: enumerate each grid from its bit patterns
def grid(e_bits, m_bits, bias, skip=()):
    vals = []
    for code in range(1 << (e_bits + m_bits)):
        if code in skip: continue
        E, M = code >> m_bits, code & ((1 << m_bits) - 1)
        v = Fraction(M, 1 << m_bits) * Fraction(2) ** (1 - bias) if E == 0 else (1 + Fraction(M, 1 << m_bits)) * Fraction(2) ** (E - bias)
        vals.append((v, code))
    return sorted(vals)
GRIDS = {'fp4': grid(2, 1, 1),                        # OCP MX spec, FP4 E2M1: bias 1, max 6, no Inf/NaN
         'fp8': grid(4, 3, 7, skip={0x7F}),           # OCP OFP8 spec, E4M3: bias 7, S.1111.111 is NaN, max 448
         'bf16': [g for g in grid(8, 7, 127) if g[0] < 8]}   # bfloat16: 8 exponent bits, 7 mantissa bits, bias 127
check([float(v) for v, _ in GRIDS['fp4']] == [0, 0.5, 1, 1.5, 2, 3, 4, 6], 'FP4 E2M1 grid')
check(float(GRIDS['fp8'][-1][0]) == 448, 'E4M3 max 448')
def round_to(x, g):   # nearest representable value; a tie goes to the even bit pattern (roundTiesToEven)
    vals = [v for v, _ in g]; i = bisect.bisect_left(vals, x)
    if i < len(vals) and vals[i] == x: return x, False
    lo, hi = g[i - 1], g[i]
    dl, dh = x - lo[0], hi[0] - x
    if dl != dh: return (lo[0] if dl < dh else hi[0]), False
    return (lo[0] if lo[1] % 2 == 0 else hi[0]), True
def qz_expect(i):
    x = Fraction(i, 100); out = []
    for f in ['fp4', 'fp8', 'bf16']:
        q, tie = round_to(x, GRIDS[f])
        out.append(dict(id=f, q=float(q), tie=tie, err=float(abs(q - x) / x * 100) if x else 0.0))
    return out

# ---------------------------------------------------------------- 5. spec-decoding (Leviathan et al., 2023)
K, C_COST = 4, 0.17
def sd_expect(p):
    a = p / 100
    # the j-th token of a target pass exists only if the first j-1 guesses were all accepted: P = a^(j-1)
    per_slot = [a ** j for j in range(K + 1)]
    E = sum(per_slot)
    closed = (1 - a ** (K + 1)) / (1 - a)                   # the paper's closed form, Eq. (1)
    check(close(E, closed, 1e-12), f'series vs closed form at {p}%')
    return dict(E=E, speed=E / (K * C_COST + 1), slots=per_slot)

# ---------------------------------------------------------------- the spec's worked examples, re-derived
ex = pd_expect(4096)
print(f'prefill-decode @4K: first token {ms(ex["ttft"])}, then every {ms(ex["tpot"])} (ratio {ex["ttft"]/ex["tpot"]:.1f})')
print(f'prefix-caching @100: {fint(pc_expect(100)["without"])} vs {fint(pc_expect(100)["with_"])} tokens, {num(pc_expect(100)["pct"],0)}% fewer, KV {num(pc_expect(100)["kv"]/2**30,1)} GiB')
print(f'quantization @2.70: ' + '; '.join(f'{r["id"]} {stored(r["q"])} ({js_str(js_prec(r["err"],2))}%)' for r in qz_expect(270)))
print(f'spec-decoding @80%: {num(sd_expect(80)["E"],2)} tokens per pass, {num(sd_expect(80)["speed"],1)}x')
for c in [0, 8192, 2048, 512]:
    e = cp_expect(c); print(f'chunked-prefill {c or "none"}: longest gap {ms(e["longest"])}, first token {ms(e["ttft"])}, steps {e["steps"]}')

# ---------------------------------------------------------------- compare with the page
READ = """(sel)=>{const t=document.querySelector(sel); const g=s=>{const e=t.querySelector(s); return e?e.textContent:null;};
  return {out:g('output'), label:g('.tz-lab span'), read:g('.tz-read'), notes:[...t.querySelectorAll('.tz-note')].map(n=>n.textContent),
   lpre:g('.tz-pfd-lpre'), ldec:g('.tz-pfd-ldec'), v0:g('.tz-pfx-v0'), v1:g('.tz-pfx-v1'), e:g('.tz-spec-e'),
   pr:[...t.querySelectorAll('.tz-spec-pr span')].map(s=>s.textContent), qv:[...t.querySelectorAll('.tz-quant-val')].map(s=>s.textContent),
   pressed:[...t.querySelectorAll('.seg-ctl button[aria-pressed="true"]')].map(b=>b.dataset.v)};}"""
SET_RANGE = "([id,v])=>{const i=document.querySelector('#tile-'+id+' input[type=range]'); i.value=v; i.dispatchEvent(new Event('input',{bubbles:true})); i.dispatchEvent(new Event('change',{bubbles:true}));}"
LONGEST = {}                          # longest sentence per tile and language, reported at the end
CP_NOTES = {}                         # chunked-prefill notes per language: the same for every chunk size

def sentence_ok(tile, lang, txt):
    """guide R11.5: one or two sentences, about 35 English words at most; R6.14: no "·" joins, no arrow fragments"""
    k = (tile, lang); n = words(txt) if lang == 'en' else len(norm(txt))
    if n > LONGEST.get(k, (0, ''))[0]: LONGEST[k] = (n, norm(txt))
    if lang == 'en':
        check(n <= 35, f'{tile}: sentence has {n} words (max 35): {txt}')
        check(norm(txt).count('. ') <= 1 and norm(txt).endswith('.'), f'{tile}: one or two sentences: {txt}')
    else:
        check(norm(txt).count('。') in (1, 2) and norm(txt).endswith('。'), f'{tile}: one or two sentences (zh): {txt}')
    check('·' not in txt and '→' not in txt, f'{tile}: "·" or "→" in the sentence: {txt}')

def sentence_count(text, lang):
    t = norm(text)
    if lang == 'zh': return len(re.findall('[。！？]', t))
    return len(re.findall(r'[.!?](?=\s|$)', t.replace('et al.', 'et al')))
def notes_ok(tile, lang, notes):
    """guide R11.5: the Assumptions & sources text is 2 to 4 short sentences; no "·" joins (R6.14)"""
    t = ' '.join(notes); n = sentence_count(t, lang)
    check(2 <= n <= 4, f'{tile} {lang}: notes have {n} sentences (2 to 4): {t[:200]}')
    check('·' not in t, f'{tile} {lang}: "·" in the notes')
    check(all(norm(x).endswith(('.', '。')) for x in notes if x), f'{tile} {lang}: a note paragraph does not end a sentence')

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    for lang in ['en', 'zh']:
        page = browser.new_page(viewport={'width': 1440, 'height': 900})
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(URL + '#map'); page.wait_for_timeout(300)
        if lang == 'zh':
            page.click('.langsw button[data-lang="zh"]'); page.wait_for_timeout(200)
        H = page.evaluate('()=>window.__tzA.consts')
        check(H['P_TOTAL'] == N_TOTAL and H['P_EMB'] == N_EMBED and H['P_HEAD'] == N_HEAD, 'parameter counts in the page')
        check(H['KV_TOK'] == KV_PER_TOKEN and H['W_READ'] == WEIGHT_BYTES_PER_STEP, 'KV per token / weight bytes per step')
        check(H['peak'] == PEAK_DENSE and H['bw'] == HBM_BW and H['mfu'] == MFU and H['mbu'] == MBU, 'H100 constants and efficiencies')
        check(page.evaluate('()=>Object.isFrozen(window.__tzA)'), 'hook is frozen (read-only)')
        W_GB, W_GIB = num(WEIGHT_BYTES_PER_STEP / 1e9, 1) + ' GB', num(WEIGHT_BYTES_PER_STEP / 2**30, 1) + ' GiB'

        # 1. prefill-decode: every slider position
        for i, n in enumerate(PD_N):
            page.evaluate(SET_RANGE, ['prefill-decode', i]); e = pd_expect(n)
            r = page.evaluate('(n)=>window.__tzA.pd.compute(n)', n); d = page.evaluate(READ, '#tile-prefill-decode')
            for k, pk in [('ttft', 'ttft'), ('tpot', 'tpot'), ('flops', 'preFlops'), ('dec_bytes', 'decBytes')]:
                check(close(e[k], r[pk]), f'pd {n} {k}: py {e[k]} page {r[pk]}')
            check(e['pre_compute'] >= e['pre_memory'], f'pd {n}: prefill should be compute-bound in this model')
            check(e['dec_memory'] > 50 * e['dec_compute'], f'pd {n}: decode should be memory-bound')
            T, P_ = ms(e['ttft']), ms(e['tpot'])
            check(norm(d['out']) == norm(f'{tokK(n)} tokens' if lang == 'en' else f'{tokK(n)} token'), f'pd {n} output {d["out"]!r}')
            sentence_ok('prefill-decode', lang, d['read'])
            for piece in [T, P_, 'Llama-3.1-8B', 'H100 SXM', tokK(n), 'one token every' if lang == 'en' else '出一个 token']:
                check(has(d['read'], piece), f'pd {n} {lang} sentence lacks {piece!r}: {d["read"]}')
            check(has(d['lpre'], T) and has(d['ldec'], P_), f'pd {n} timeline labels {d["lpre"]!r} {d["ldec"]!r}')
            nt = ' '.join(d['notes']); notes_ok('prefill-decode', lang, d['notes'])
            # SITE-A-r7 item 12: the prompt length and the bytes one decode step reads follow the slider; no formula lines
            for piece in [fint(n), num(e['dec_bytes'] / 1e9, 1) + ' GB', num(PEAK_DENSE / 1e12, 1) + ' TFLOP/s', '3.35 TB/s', '50%', '60%', 'Databricks',
                          'H100 SXM', 'meta-llama/llama-models']:
                check(has(nt, piece), f'pd {n} notes lack {piece!r}')
            for gone in ['TFLOP,', 'N²', '1,979', W_GIB]:
                check(not has(nt, gone), f'pd {n} notes still have {gone!r}')
        # R8 re-check 13e: every legend tooltip names its subject, and the thick line has one too
        tips = page.evaluate("()=>[...document.querySelectorAll('#tile-prefill-decode .tz-leg > span')].map(s=>s.getAttribute('data-tipx'))")
        want = (['The GPU computes every prompt token in one pass before the first token appears.',
                 'The thick line is the first token: it appears only after prefill has processed the whole prompt.',
                 'Each thin tick is a decoded token: every new token takes one full pass over the weights.'] if lang == 'en' else
                ['GPU 先一次算完所有 prompt token，之后才出首 token。',
                 '粗线是首 token：prefill 算完整个 prompt 之后才出现。',
                 '细线是 decode 出的 token：每出一个新 token，都要把全部权重读一遍。'])
        check(tips == want, f'pd legend tooltips {tips}')

        # 2. chunked-prefill: every button
        base = cp_expect(0)
        for c in [0, 8192, 2048, 512]:
            page.click(f'#tile-chunked-prefill button[data-v="{c}"]'); e = cp_expect(c)
            r = page.evaluate('(c)=>window.__tzA.cp.compute(c)', c); d = page.evaluate(READ, '#tile-chunked-prefill')
            check(d['pressed'] == [str(c)], f'cp {c} pressed {d["pressed"]}')
            for k, pk in [('normal', 'normal'), ('longest', 'longest'), ('ttft', 'ttft')]:
                check(close(e[k], r[pk]), f'cp {c} {k}: py {e[k]} page {r[pk]}')
            check(r['n'] == e['steps'], f'cp {c} steps')
            check(e['all_compute_bound'] and e['normal_memory'] > e['normal_compute'], f'cp {c}: chunk steps compute-bound, decode-only step memory-bound')
            sentence_ok('chunked-prefill', lang, d['read'])
            for piece in ['Llama-3.1-8B', 'H100 SXM', '16', '32K']:
                check(has(d['read'], piece), f'cp {c} sentence lacks {piece!r}')
            nt = ' '.join(d['notes']); notes_ok('chunked-prefill', lang, d['notes'])
            if c == 0:
                for piece in [ms0(e['ttft']), ms(e['normal'])]:
                    check(has(d['read'], piece), f'cp none sentence lacks {piece!r}: {d["read"]}')
            else:
                for piece in [tokK(c), ms(e['longest']), ms0(base['ttft'])]:
                    check(has(d['read'], piece), f'cp {c} sentence lacks {piece!r}: {d["read"]}')
            if lang == 'zh':
                check(has(d['read'], '每个用户两个 token 之间'), f'cp {c} zh sentence: {d["read"]}')
            # R8 re-check 13a: whole milliseconds, no ".0", in the sentence, the notes and the screen-reader label
            aria = page.evaluate("()=>document.querySelector('#tile-chunked-prefill .tz-chunk-vis').getAttribute('aria-label')")
            for where, text in [('sentence', d['read']), ('notes', nt), ('aria-label', aria)]:
                check('.0 ms' not in norm(text) and not has(text, ms1(base['ttft'])), f'cp {c} {where} still shows a ".0 ms" time: {text[:160]}')
            check(has(aria, ms0(e['ttft'])), f'cp {c} aria-label lacks the long prompt\'s first token {ms0(e["ttft"])!r}: {aria}')
            # SITE-A-r7 item 12: the same four sentences for every chunk size, with the unchunked run's numbers
            for piece in [ms0(base['ttft']), ms(base['normal']), '16', '4K', '50%', '60%', '3.35 TB/s', 'H100 SXM', 'Databricks']:
                check(has(nt, piece), f'cp {c} notes lack {piece!r}: {nt[:200]}')
            check(nt == ' '.join(CP_NOTES.setdefault(lang, d['notes'])), f'cp {c}: the notes should not change with the chunk size')
            # SITE-A-r7 item 13: in ZH an engine step is 一步 / 每一步, never "step"
            if lang == 'zh':
                zh_all = page.evaluate("()=>{const t=document.getElementById('tile-chunked-prefill'); return t.querySelector('.tile-tz').textContent+' '+[...t.querySelectorAll('[data-tipx]')].map(e=>e.getAttribute('data-tipx')).join(' ')+' '+[...t.querySelectorAll('[aria-label]')].map(e=>e.getAttribute('aria-label')).join(' ')}")
                check('step' not in zh_all.lower(), f'cp {c} zh: "step" in the tile: {zh_all[:200]}')
                for piece in ['Prefill 的 chunk 大小（每一步 prefill 的 token 数）', '每种深浅是一步，这一步 prefill 长 prompt 的一个 chunk。']:
                    check(has(zh_all, piece), f'cp {c} zh lacks {piece!r}')

        # 3. prefix-caching: every button
        for n in PC_N:
            page.click(f'#tile-prefix-caching button[data-v="{n}"]'); e = pc_expect(n)
            r = page.evaluate('(n)=>window.__tzA.pc.compute(n)', n); d = page.evaluate(READ, '#tile-prefix-caching')
            check(r['without'] == e['without'] and r['with'] == e['with_'] and r['kvBytes'] == e['kv'] and close(r['pct'], e['pct']), f'pc {n} raw')
            check(d['v0'] == fbig(e['without']) and d['v1'] == fbig(e['with_']), f'pc {n} bar values {d["v0"]} {d["v1"]}')
            sentence_ok('prefix-caching', lang, d['read'])
            for piece in [fbig(e['without']), '10,000', '200']:
                check(has(d['read'], piece), f'pc {n} sentence lacks {piece!r}: {d["read"]}')
            check('10K' not in d['read'], f'pc {n}: decimal count must be written in full')
            if n > 1:
                for piece in [fint(n), fbig(e['with_']), num(e['pct'], 0) + '%']:
                    check(has(d['read'], piece), f'pc {n} sentence lacks {piece!r}: {d["read"]}')
                # R8 re-check 13b: one shared prompt, and each request's own 200 tokens
                lead = (f'{fint(n)} requests share one 10,000-token system prompt, and each adds 200 tokens of its own:' if lang == 'en' else
                        f'{fint(n)} 个请求共用一段 10,000 个 token 的 system prompt，每个请求还有自己的 200 个 token：')
                check(norm(d['read']).startswith(norm(lead)), f'pc {n} sentence should start {lead!r}: {d["read"]}')
            else:
                check(e['without'] == e['with_'] and has(d['read'], 'nothing is cached yet' if lang == 'en' else '还没有可复用的缓存'), f'pc 1 must say why there is no saving: {d["read"]}')
                lead = ('1 request with a 10,000-token system prompt and 200 tokens of its own:' if lang == 'en' else
                        '1 个请求用一段 10,000 个 token 的 system prompt，还有自己的 200 个 token：')
                check(norm(d['read']).startswith(norm(lead)), f'pc 1 sentence should start {lead!r}: {d["read"]}')
            check('own tokens' not in d['read'] and '各带' not in d['read'], f'pc {n}: old wording left: {d["read"]}')
            nt = ' '.join(d['notes']); notes_ok('prefix-caching', lang, d['notes'])
            # SITE-A-r7 item 12: one plain formula sentence that names N; the per-token size says what it is
            for piece in [num(e['kv'] / 2**30, 1) + ' GiB', '128 KiB', 'Llama-3.1-8B', 'automatic prefix caching', 'N × 10,200', '10,000 + N × 200',
                          'For N requests' if lang == 'en' else 'N 个请求时', '128 KiB per token' if lang == 'en' else '每个 token 占 128 KiB']:
                check(has(nt, piece), f'pc {n} notes lack {piece!r}')
            for gone in ['(10,000 + 200)', '2 × 32', '8 KV heads' if lang == 'en' else '8 个 KV head']:
                check(not has(nt, gone), f'pc {n} notes still have {gone!r}')

        # 4. quantization: all 601 slider positions, read in one pass
        rows = page.evaluate("""()=>{const t=document.getElementById('tile-quantization'), i=t.querySelector('input'), out=[];
          for(let v=0;v<=600;v++){i.value=v; i.dispatchEvent(new Event('input',{bubbles:true}));
            out.push({v, out:t.querySelector('output').textContent, read:t.querySelector('.tz-read').textContent,
              qv:[...t.querySelectorAll('.tz-quant-val')].map(s=>s.textContent), hook:window.__tzA.qz.compute(v).rows});}
          return out;}""")
        check(len(rows) == 601, 'quantization positions')
        TIE = 'a tie, rounded to even' if lang == 'en' else '正好居中，取偶数档'
        for row in rows:
            v = row['v']; exp = qz_expect(v); vs = js_fixed(v / 100, 2)
            check(row['out'] == vs, f'qz {v} output {row["out"]}')
            for ex_, hk, qv in zip(exp, row['hook'], row['qv']):
                check(ex_['q'] == hk['q'] and ex_['tie'] == hk['tie'], f'qz {v} {ex_["id"]}: py {ex_["q"]} tie {ex_["tie"]} page {hk["q"]} {hk["tie"]}')
                check(close(ex_['err'], hk['err'], 1e-12) if ex_['err'] else hk['err'] == 0, f'qz {v} {ex_["id"]} error')
                check(qv == stored_short(ex_['q']), f'qz {v} {ex_["id"]} value column {qv!r} vs {stored_short(ex_["q"])!r}')
            txt = row['read']
            sentence_ok('quantization', lang, txt)
            check(txt.startswith(vs), f'qz {v} sentence starts with the value: {txt}')
            if all(x['err'] == 0 for x in exp):
                check(has(txt, 'stored exactly' if lang == 'en' else '都能精确存下'), f'qz {v} all exact: {txt}')
            else:
                for x in exp:
                    check(has(txt, stored(x['q'])), f'qz {v} sentence lacks {x["id"]} value {stored(x["q"])}: {txt}')
                    if x['err'] and x['q']:   # SITE-A-r7 item 14c: every percentage says it is the error
                        e2 = js_str(js_prec(x['err'], 2))
                        check(has(txt, f'{e2}% error' if lang == 'en' else f'误差 {e2}%'), f'qz {v} sentence lacks {x["id"]} error {e2}%: {txt}')
                    if x['q'] == 0 and v:
                        check(has(txt, 'value lost' if lang == 'en' else '这个数丢了'), f'qz {v} lost value: {txt}')
                ties = sum(x['tie'] for x in exp)
                check(norm(txt).count(TIE) == ties, f'qz {v} ties {ties}: {txt}')
        d = page.evaluate(READ, '#tile-quantization'); nt = ' '.join(d['notes']); notes_ok('quantization', lang, d['notes'])
        # SITE-A-r7 item 12: the FP4 values, the values per power of two for FP8 (2^3) and BF16 (2^7), the MX block of 32
        fp4 = [js_str(float(x)) for x, _ in GRIDS['fp4']]
        for piece in [', '.join(fp4[:-1]) + ' and ' + fp4[-1] if lang == 'en' else '、'.join(fp4),
                      '8 values between two powers of two' if lang == 'en' else '之间有 8 档', 'BF16 has 128' if lang == 'en' else 'BF16 有 128 档', '32', 'OCP']:
            check(has(nt, piece), f'qz notes lack {piece!r}')
        # the grids the visual draws are the true grids
        for f in ['fp4', 'fp8']:
            g = page.evaluate('(f)=>window.__tzA.qz.grid(f)', f)
            check(g == sorted(float(x) for x, _ in GRIDS[f] if x <= 6), f'{f} ticks drawn = representable values in [0, 6]')

        # 5. spec-decoding: every slider position
        for p in range(50, 96, 5):
            page.evaluate(SET_RANGE, ['spec-decoding', p]); e = sd_expect(p)
            kept = sum((p / 100) ** j for j in range(1, K + 1))       # accepted guesses per pass, independently
            r = page.evaluate('(p)=>window.__tzA.sd.compute(p)', p); d = page.evaluate(READ, '#tile-spec-decoding')
            check(close(e['E'], r['E']) and close(e['speed'], r['speed']) and close(kept, r['kept']), f'sd {p} raw E/speed/kept')
            check(d['out'] == f'{p}%' and d['e'] == num(e['E'], 2), f'sd {p} output {d["out"]} / E {d["e"]}')
            check(d['label'] == ('Chance each guess is accepted, given the earlier ones were (α)' if lang == 'en' else '每个猜测被接受的概率 α（前面都被接受时）'), f'sd label {d["label"]!r}')
            check(d['pr'] == [num(s * 100, 0) + '%' for s in e['slots']], f'sd {p} slot chances {d["pr"]}')
            sentence_ok('spec-decoding', lang, d['read'])
            # SITE-A-r7 item 14d: the speed-up says what it is compared with
            for piece in [f'α = {p}%', num(kept, 2), num(e['E'], 2), '4',
                          num(e['speed'], 1) + (' times as fast as without drafts' if lang == 'en' else ' 倍'), 'the target model' if lang == 'en' else '不猜时']:
                check(has(d['read'], piece), f'sd {p} sentence lacks {piece!r}: {d["read"]}')
            check('×' not in d['read'], f'sd {p}: a bare "×" in the sentence: {d["read"]}')
            nt = ' '.join(d['notes']); notes_ok('spec-decoding', lang, d['notes'])
            for piece in ['α + α² + α³ + α⁴', f'{round(C_COST * 100)}%', f'{K + 1} ', 'Leviathan', '2211.17192']:
                check(has(nt, piece), f'sd {p} notes lack {piece!r}')
        check(num(sd_expect(80)['E'] - 1, 2) == '2.36', 'the coordinator\'s example: on average 2.36 of 4 guesses kept at α = 80%')

        check(not errors, f'{lang}: page errors {errors}')
        page.close()
    browser.close()

for (tile, lang), (n, txt) in sorted(LONGEST.items()):
    print(f'longest {lang} {tile}: {n} {"words" if lang == "en" else "chars"} | {txt}')
print(f'{CHECKS[0]} checks, {len(FAILS)} failed')
for f in FAILS[:40]: print('FAIL', f)
sys.exit(1 if FAILS else 0)
