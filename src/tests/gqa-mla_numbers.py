"""Numbers of the GQA and MLA lab (labs/gqa-mla.js), recomputed independently in Python and compared with the page.

- Shapes from config.json (specs/transformer-facts.md A2, A4, A5, A8); parameter counts are the Hugging Face totals, the same as the
  KV cache memory lab's (checked against ASP.kv.MODELS): 70,553,706,496 parameters are 131.4 GiB in BF16.
- KV cache per token from the definitions: MHA/GQA/MQA store a key and a value of head_dim numbers per KV head per layer; MLA stores
  d_c + d_R numbers per layer (DeepSeek-V2 §2.1.3).
- What each GPU stores follows vLLM's QKVParallelLinear (facts I): num_kv_heads per GPU = 1 if tp >= G else G / tp. MLA: the whole vector.
- Users: the KV cache lab's memory model: usable = memory x 0.9; weights / GPUs; 2 GiB overhead; floor(room / one user's KV cache on a GPU).
- The page's engine (window.__gm) is compared on every combination of the settings, and the KV cache lab's own engine (ASP.kv.compute)
  on every combination both labs know, so the two labs give the same users and the same bytes per token.
- Show the math: every line is re-done from its own shown operands (R6.6) for every combination, including the weights note:
  parameters × bytes = GB = GiB (review 8, R6.12: GB next to GiB) and the total shown ÷ the GPUs gives the weights per GPU shown
  (review B r12 N4: 378.0 ÷ 8 did not give 47.2).
- Real FP8 checkpoints (review B r12 N9): Meta's Llama-3.1-405B-Instruct-FP8 keeps 81,314,922,496 parameters in BF16 (every
  attention matrix, the MLPs of the first and last layers, the embedding and the output layer, recomputed here from config.json),
  324,538,466,304 in FP8 and 15,237,120 FP32 scales (Hugging Face API, checked 2026-09-26): 56.7 GiB per GPU on 8, 1 user at 128K.
  DeepSeek-V3's checkpoint adds about 0.24 GiB per GPU (review B's count), and its 5 users still fit.
- The spec's table, the GQA paper's Table 1 for the three designs on T5-XXL (from the facts file; review 8: no T5-Large), the derived
  numbers in the text, the drill, the tile, the scenarios (the initial setup, MHA, first) and the challenge answer.
- Review 8: the initial setup is MHA (1 user at 128K), not GQA-8; numbers over 6 digits are written 70.6B, 81.3B, 9.40B (R6.18).
python3 tests/gqa-mla_numbers.py"""
import pathlib, re, subprocess, sys, math, itertools
from decimal import Decimal, ROUND_HALF_UP
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/gqa-mla.html', '--only', 'gqa-mla'], cwd=P, check=True, capture_output=True)
(P / 'out/wrapped-gqa-mla.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                                            '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out/gqa-mla.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out/wrapped-gqa-mla.html')
KiB, MiB, GiB = 1024, 1024 ** 2, 1024 ** 3
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(f'{name} :: {info}')
        print('FAIL', name, '::', str(info)[:300])


# ---------- independent model ----------
CFG = {  # config.json: layers, attention heads, KV heads, head_dim (hidden / heads when no head_dim key)
    'llama31-8b': dict(L=32, H=32, G=8, hd=128, d=4096),
    'llama31-70b': dict(L=80, H=64, G=8, hd=8192 // 64, d=8192),
    'llama31-405b': dict(L=126, H=128, G=8, hd=16384 // 128, d=16384),
    'deepseek-v3': dict(L=61, H=128, dc=512, dr=64, d=7168),
}
PARAMS = {'llama31-8b': 8030261248, 'llama31-70b': 70553706496, 'llama31-405b': 405853388800, 'deepseek-v3': 671e9}   # HF totals; DeepSeek-V3's paper figure
MEM = {'h100': 80, 'h200': 141}
BYTES_W = {'bf16': 2, 'fp8': 1}
UTIL, OH, KVB = 0.9, 2, 2
CTXS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
DES = ['mha', 'gqa16', 'gqa8', 'gqa4', 'mqa', 'mla']


def kv_heads(model, a):
    c = CFG[model]
    return {'mha': c['H'], 'gqa16': 16, 'gqa8': 8, 'gqa4': 4, 'mqa': 1}.get(a, 0)


def per_token(model, a):
    c = CFG[model]
    if a == 'mla':
        return c['L'] * (c.get('dc', 512) + c.get('dr', 64)) * KVB
    return 2 * c['L'] * kv_heads(model, a) * c['hd'] * KVB


def per_gpu_token(model, a, tp):
    """vLLM: each GPU holds max(1, G / tp) whole KV heads; MLA's vector is not split."""
    c = CFG[model]
    if a == 'mla':
        return per_token(model, a)
    G = kv_heads(model, a)
    heads_here = 1 if tp >= G else G // tp
    return 2 * c['L'] * heads_here * c['hd'] * KVB


def users(model, a, gpu, tp, wp, ctx, params=None):
    usable = MEM[gpu] * GiB * UTIL
    w = (params or PARAMS[model]) * BYTES_W[wp] / tp
    room = usable - w - OH * GiB
    if room <= 0:
        return 0
    one_user = per_gpu_token(model, a, tp) * ctx
    return int(math.floor(room / one_user + 1e-9))


def designs(model):
    return ['mla'] if model == 'deepseek-v3' else DES


# ---------- the spec's table and the facts ----------
T = {a: (per_token('llama31-70b', a) / KiB, per_gpu_token('llama31-70b', a, 8) / KiB, users('llama31-70b', a, 'h100', 8, 'bf16', 131072)) for a in DES}
SPEC = {'mha': (2560, 320, 1), 'gqa16': (640, 80, 5), 'gqa8': (320, 40, 10), 'gqa4': (160, 40, 10), 'mqa': (40, 40, 10), 'mla': (90, 90, 4)}
for a in DES:
    check(f'spec table row {a}', T[a] == SPEC[a], (T[a], SPEC[a]))
check('DeepSeek-V3: 68.625 KiB per token (61 x 576 x 2 bytes)', per_token('deepseek-v3', 'mla') == 70272 and per_token('deepseek-v3', 'mla') / KiB == 68.625)
check('Falcon-7B: 8 KiB per token (2 x 32 x 1 x 64 x 2)', 2 * 32 * 1 * (4544 // 71) * 2 == 8192)
check('MLA stores as much as 2.25 KV heads: (512 + 64) / (2 x 128)', (512 + 64) / 256 == 2.25)
check('the weights as shown: 131.4 GiB for Llama-3.1-70B in BF16, 16.4 GiB per GPU; 47.2 GiB per GPU for the 405B in FP8; 15.0 GiB for the 8B',
      [round(PARAMS['llama31-70b'] * 2 / GiB, 1), round(PARAMS['llama31-70b'] * 2 / 8 / GiB, 1), round(PARAMS['llama31-405b'] / 8 / GiB, 1), round(PARAMS['llama31-8b'] * 2 / GiB, 1)] == [131.4, 16.4, 47.2, 15.0])
facts = (P / 'specs/transformer-facts.md').read_text()
T1 = {}
for name, key in [('MHA-XXL', 'mha'), ('GQA-8-XXL', 'gqa8'), ('MQA-XXL', 'mqa')]:
    m = re.search(r'^\| ' + re.escape(name) + r' \| ([\d.]+) \| ([\d.]+) \|', facts, re.M)
    T1[key] = (float(m.group(2)), float(m.group(1)))       # (average, time)
check('GQA paper Table 1 (T5-XXL) read from the facts file', T1 == {'mha': (47.2, 1.51), 'gqa8': (47.1, 0.28), 'mqa': (46.6, 0.24)}, T1)
d1, d2 = round(T1['mha'][0] - T1['gqa8'][0], 1), round(T1['mha'][0] - T1['mqa'][0], 1)
fast = math.floor(min(T1['mha'][1] / T1['gqa8'][1], T1['mha'][1] / T1['mqa'][1]))
check('part 3: 0.1 and 0.6 points below MHA', (d1, d2) == (0.1, 0.6), (d1, d2))
check('part 3: GQA-8 and MQA each took under a fifth of MHA’s time per sample (0.28 / 1.51, 0.24 / 1.51)', fast == 5 and T1['gqa8'][1] / T1['mha'][1] < 0.2 and T1['mqa'][1] / T1['mha'][1] < 0.2, fast)
check('the WMT exception: MQA 28.5 BLEU against MHA 28.4 (facts file, Table 1)', re.search(r'^\| MQA-XXL \|.*\| 28\.5 \|', facts, re.M) is not None and re.search(r'^\| MHA-XXL \|.*\| 28\.4 \|', facts, re.M) is not None)
dW = 2 * CFG['llama31-70b']['d'] * (64 - 8) * 128 * 80
check('MHA version: about 9.40B more parameters in its key and value matrices (3 significant digits, R6.18)', f'{dW / 1e9:.2f}' == '9.40', dW)
check('MHA version with those weights still fits 1 user at 128K', users('llama31-70b', 'mha', 'h100', 8, 'bf16', 131072, params=PARAMS['llama31-70b'] + dW) == 1)

# ---------- scenarios and challenge ----------
SCN = [('llama31-70b', 'mha', 'h100', 8, 'bf16'), ('llama31-70b', 'gqa8', 'h100', 8, 'bf16'), ('llama31-8b', 'gqa8', 'h100', 1, 'bf16'),
       ('llama31-405b', 'gqa8', 'h100', 8, 'fp8'), ('deepseek-v3', 'mla', 'h200', 8, 'fp8')]
SCN_U = [users(*s, 131072) for s in SCN]
check('scenario users at 128K: 1 (the initial setup, MHA), 10, 3, 2, 5', SCN_U == [1, 10, 3, 2, 5], SCN_U)
check('part 1: MHA on 8 x H100 at 128K: 2,560 KiB per token, 1 user; part 4: MQA stores one sixty-fourth of it',
      per_token('llama31-70b', 'mha') == 2560 * KiB and SCN_U[0] == 1 and per_token('llama31-70b', 'mha') // per_token('llama31-70b', 'mqa') == 64)
check('part 5: per token MQA stores one sixty-fourth of MHA, on each GPU one eighth (40 KiB against 320 KiB)',
      per_gpu_token('llama31-70b', 'mha', 8) // per_gpu_token('llama31-70b', 'mqa', 8) == 8 and per_gpu_token('llama31-70b', 'mqa', 8) == 40 * KiB)
check('part 4 options: 1 / 2 to 20 / 21 to 63 / 64 or more; MQA gives 10, in "2 to 20"', 2 <= users('llama31-70b', 'mqa', 'h100', 8, 'bf16', 131072) <= 20)
ch = {a: users('llama31-70b', a, 'h100', 8, 'bf16', 131072) for a in DES}
met = [a for a in DES if ch[a] >= 10]
best = max(met, key=lambda a: kv_heads('llama31-70b', a))
check('challenge: GQA-8, GQA-4 and MQA fit 10; the best is GQA-8', met == ['gqa8', 'gqa4', 'mqa'] and best == 'gqa8', (met, best))

# ---------- real FP8 checkpoints (review B r12 N9) ----------
M405 = dict(bf16=81314922496, f8=324538466304, f32=15237120)   # Hugging Face API, meta-llama/Llama-3.1-405B-Instruct-FP8, checked 2026-09-26
check('Meta 405B FP8: BF16 + FP8 numbers = the 405B parameter count; the FP32 numbers are the scales', M405['bf16'] + M405['f8'] == PARAMS['llama31-405b'])
c405 = CFG['llama31-405b']; ffn405, vocab405 = 53248, 128256
attn405 = c405['L'] * (2 * c405['d'] * c405['d'] + 2 * c405['d'] * c405['G'] * c405['hd'])        # q, o: d x d; k, v: d x (8 x 128)
bf16_405 = attn405 + 2 * 3 * c405['d'] * ffn405 + 2 * vocab405 * c405['d'] + (2 * c405['L'] + 1) * c405['d']
check('Meta 405B FP8: the BF16 part is every attention matrix, the MLPs of layers 0 and 125, the embedding, the output layer and the norms (config.json)', bf16_405 == M405['bf16'], (bf16_405, M405['bf16']))
w405 = (M405['bf16'] * 2 + M405['f8'] + M405['f32'] * 4) / 8
room405 = MEM['h100'] * GiB * UTIL - w405 - OH * GiB
u405 = math.floor(room405 / (per_gpu_token('llama31-405b', 'gqa8', 8) * 131072) + 1e-9)
check('Meta 405B FP8 on 8 x H100: 56.7 GiB per GPU, 1 user at 128K (2 at 1 byte per parameter)', (round(w405 / GiB, 1), u405, SCN_U[3]) == (56.7, 1, 2), (w405 / GiB, u405))
check('Meta 405B FP8: the BF16 part is a fifth of the parameters (scenario 4, review A r13 M4)', round(PARAMS['llama31-405b'] / M405['bf16']) == 5 and abs(M405['bf16'] / PARAMS['llama31-405b'] - 0.2) < 0.005, M405['bf16'] / PARAMS['llama31-405b'])
attn_only = (attn405 * 2 + (PARAMS['llama31-405b'] - attn405)) / 8 / GiB
check('Meta 405B FP8: attention alone in BF16 would give 55.6 GiB per GPU, not 56.7, so scenario 4 names the whole BF16 part (review A r13 M4)', round(attn_only, 1) == 55.6 and round(w405 / GiB, 1) == 56.7, attn_only)
room_ds = MEM['h200'] * GiB * UTIL - PARAMS['deepseek-v3'] / 8 - 0.24 * GiB - OH * GiB
u_ds = math.floor(room_ds / (per_gpu_token('deepseek-v3', 'mla', 8) * 131072) + 1e-9)
check('DeepSeek-V3 with about 0.24 GiB more per GPU still fits 5 users at 128K', u_ds == SCN_U[4] == 5, u_ds)


# ---------- Show the math, re-done from its shown operands ----------
def num(s):
    return float(s.replace(',', ''))


def dec_of(s):
    return len(s.split('.')[1]) if '.' in s else 0


def half_up(x, d):
    """a reader's rounding: halves go up"""
    return Decimal(x).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)


def check_math(tag, lines, st, exp_users):
    """lines: the text of each math line as shown"""
    L1 = lines[0]
    m = re.search(r'= ([\d,]+) (?:bytes|字节) = ([\d,.]+) KiB', L1)
    if not m:
        check(f'{tag}: line 1 has bytes and KiB', False, L1); return
    ops = [num(x) for x in re.findall(r'(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?= ?(?:\(|（|layers|层|KV|个 KV|head_dim|bytes|字节|latent|RoPE|\)|）))', L1.split('=')[1])]
    b = num(m.group(1))
    if st['attn'] == 'mla':
        lay, dc, dr, by = [x for x in ops][:4]
        check(f'{tag}: line 1 layers x (latent + RoPE) x bytes', lay * (dc + dr) * by == b, (L1, ops))
    else:
        two, lay, g, hd, by = ops[:5]
        check(f'{tag}: line 1 2 x layers x KV heads x head_dim x bytes', two * lay * g * hd * by == b, (L1, ops))
    check(f'{tag}: line 1 bytes = KiB', abs(b / KiB - num(m.group(2))) < 1e-9, L1)
    if len(lines) < 5:
        return
    L2, L3, L4, L5 = lines[1], lines[2], lines[3], lines[4]
    tokk = num(m.group(2))
    m2 = re.search(r'= ([\d,.]+) KiB ÷ (\d+).*= ([\d,.]+) KiB', L2)
    if m2:
        check(f'{tag}: line 2 divides', abs(num(m2.group(1)) / int(m2.group(2)) - num(m2.group(3))) < 1e-9, L2)
        pg = num(m2.group(3))
    else:
        pg = num(re.search(r'= ([\d,.]+) KiB', L2).group(1))
        check(f'{tag}: line 2 not divided (MLA)', st['attn'] == 'mla' and pg == tokk, L2)
    m3 = re.search(r'= ([\d,.]+) KiB × ([\d,]+) .*= ([\d,.]+) GiB', L3)
    per_user = num(m3.group(3))
    exact = num(m3.group(1)) * KiB * num(m3.group(2)) / GiB
    dec = len(m3.group(3).split('.')[1]) if '.' in m3.group(3) else 0
    check(f'{tag}: line 3 KiB x tokens = GiB (as rounded)', abs(round(exact, dec) - per_user) < 1e-9 or abs(exact - per_user) < 0.5 * 10 ** -dec + 1e-12, (L3, exact))
    m4 = re.search(r'= ([\d.]+) GiB × ([\d.]+) − ([\d,.]+) GiB of weights − (\d+) GiB of overhead = ([\d,.]+) − ([\d,.]+) − (\d+) = (−?[\d,.]+) GiB', L4)
    if not m4:   # ZH order: 权重 16.4 GiB − 开销 2 GiB
        m4 = re.search(r'= ([\d.]+) GiB × ([\d.]+) − 权重 ([\d,.]+) GiB − 开销 (\d+) GiB = ([\d,.]+) − ([\d,.]+) − (\d+) = (−?[\d,.]+) GiB', L4)
    if not m4:
        check(f'{tag}: line 4 parses', False, L4); return
    mem, ut, w, oh, use, w2, oh2, room = [num(x.replace('−', '-')) for x in m4.groups()]
    dr = len(m4.group(8).split('.')[1]) if '.' in m4.group(8) else 0
    check(f'{tag}: line 4 usable = memory x utilization', round(mem * ut, 1) == use, L4)
    check(f'{tag}: line 4 room = usable − weights − overhead (shown values)', round(use - w2 - oh2, dr) == room and w == w2 and oh == oh2, L4)
    # the weights note: parameters x bytes = GB = GiB (review 8, R6.12), and total ÷ GPUs rounds to the weights shown (review B r12 N4)
    me = re.search(r'\((BF16|FP8) weights: ([\d,.]+B?) × (\d) bytes? = ([\d,.]+) GB = ([\d,.]+) GiB, ÷ (\d+) GPUs?\)', L4)
    mz = re.search(r'（(BF16|FP8) 权重：([\d,.]+B?) × (\d) 字节 = ([\d,.]+) GB = ([\d,.]+) GiB，÷ (\d+) 张 GPU）', L4)
    mm = me or mz
    if not mm:
        check(f'{tag}: line 4 weights note parses', False, L4); return
    prec, par_s, byt, gb_s, tot_s, nn = mm.group(1), mm.group(2), int(mm.group(3)), mm.group(4), mm.group(5), int(mm.group(6))
    want_par = '671B' if st['model'] == 'deepseek-v3' else f"{PARAMS[st['model']]:,}"
    check(f'{tag}: line 4 weights: the parameter count and bytes per parameter', par_s == want_par and byt == BYTES_W[st['wp']], (L4, want_par))
    check(f'{tag}: line 4 weights in GB = parameters x bytes / 10^9 (1 decimal)', half_up(Decimal(repr(PARAMS[st['model']] * BYTES_W[st['wp']] / 1e9)), 1) == Decimal(gb_s.replace(',', '')), L4)
    dt, dw = dec_of(tot_s), dec_of(m4.group(3))
    check(f'{tag}: line 4 total weights = parameters x bytes per parameter (as rounded)', half_up(Decimal(repr(PARAMS[st['model']] * BYTES_W[st['wp']] / GiB)), dt) == Decimal(tot_s.replace(',', '')) and prec == st['wp'].upper() and nn == st['n'], (L4, PARAMS[st['model']] * BYTES_W[st['wp']] / GiB))
    check(f'{tag}: line 4 total ÷ GPUs = the weights per GPU shown, same decimals', half_up(Decimal(tot_s.replace(',', '')) / nn, dw) == Decimal(m4.group(3).replace(',', '')) and dt == dw, L4)
    if room <= 0:
        check(f'{tag}: line 5 says 0', re.search(r'= 0\b', L5) is not None and exp_users == 0, L5); return
    m5 = re.search(r'⌊([\d,.]+) GiB ÷ ([\d,.]+) GiB⌋ = ⌊([\d,.]+)⌋ = ([\d,]+)', L5)
    q = num(m5.group(1)) / num(m5.group(2))
    check(f'{tag}: line 5 floor(room / one user) = users', math.floor(q + 1e-9) == int(num(m5.group(4))) == exp_users, (L5, q, exp_users))
    q2 = round(q + 1e-12, 2)
    if math.floor(q2 + 1e-9) > math.floor(q + 1e-9):
        q2 = math.floor(q * 100 + 1e-7) / 100
    check(f'{tag}: line 5 quotient shown to 2 decimals, never rounded up past the floor', abs(q2 - num(m5.group(3))) < 1e-9, (L5, q))


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL + '#gqa-mla'); pg.wait_for_timeout(600)
    # the KV cache lab's model table
    kvm = pg.evaluate("()=>Object.fromEntries(Object.entries(ASP.kv.MODELS).map(([k,m])=>[k,{params:m.params,layers:m.layers,heads:m.heads,kv:m.kvHeads||0,hd:m.headDim||0,lora:m.kvLora||0,rope:m.ropeDim||0}]))")
    gm = pg.evaluate("()=>window.__gm.MODELS")
    for k in PARAMS:
        check(f'{k}: parameters equal the KV cache lab and this test', kvm[k]['params'] == gm[k]['params'] == PARAMS[k], (kvm[k], gm[k]))
        check(f'{k}: layers and heads equal the KV cache lab and config.json', kvm[k]['layers'] == gm[k]['layers'] == CFG[k]['L'] and kvm[k]['heads'] == gm[k]['heads'] == CFG[k]['H'], (kvm[k], gm[k]))
    # every combination of the settings
    combos = [(m, a, g, t, w, c) for m in CFG for a in designs(m) for g in MEM for t in [1, 2, 4, 8] for w in BYTES_W for c in CTXS]
    res = pg.evaluate("""cs=>cs.map(([model,attn,gpu,n,wp,ctx])=>{const x=window.__gm.compute({model,attn,gpu,n,wp,ctx}); return [x.perTok,x.perGpuTok,x.users,x.copies,x.headsPerGpu]})""", combos)
    bad = 0
    for (m, a, g, t, w, c), (pt, pgt, u, cp, hpg) in zip(combos, res):
        ok = pt == per_token(m, a) and pgt == per_gpu_token(m, a, t) and u == users(m, a, g, t, w, c)
        if a != 'mla':
            G = kv_heads(m, a)
            ok = ok and cp == max(1, t // G) and hpg == max(1, G // t)
        else:
            ok = ok and cp == t
        if not ok:
            bad += 1
            if bad < 6:
                print('  mismatch', (m, a, g, t, w, c), (pt, pgt, u, cp, hpg), (per_token(m, a), per_gpu_token(m, a, t), users(m, a, g, t, w, c)))
    check(f'page engine = independent model on {len(combos)} combinations', bad == 0, bad)
    # the KV cache lab's own engine on every combination both labs know (native design, and "No GQA" = MHA)
    kvc = [(m, a, g, t, w, c) for m in CFG for a in (['mla'] if m == 'deepseek-v3' else ['gqa8', 'mha']) for g in MEM for t in [1, 2, 4, 8] for w in BYTES_W for c in CTXS]
    kr = pg.evaluate("""cs=>cs.map(([model,attn,gpu,n,wp,ctx])=>{const r=ASP.kv.compute({...ASP.kv.DEFAULT,model,gpu,n,wp,kp:'bf16',ctx,users:1,attn:attn==='mha'?'mha':'gqa',util:0.9,overhead:2});
        const x=window.__gm.compute({model,attn,gpu,n,wp,ctx}); return [r.kvPerToken,x.perTok,r.maxUsers,x.users,r.kvPerGpu,x.perGpuTok*ctx]})""", kvc)
    badk = [(c, r) for c, r in zip(kvc, kr) if not (r[0] == r[1] and r[2] == r[3] and abs(r[4] - r[5]) < 1e-6)]
    check(f'this lab = the KV cache lab (bytes per token, users, KV cache per user on a GPU) on {len(kvc)} combinations', not badk, badk[:3])
    g8 = pg.evaluate("()=>{const r=ASP.kv.compute({...ASP.kv.DEFAULT,ctx:131072,users:1}); return [r.maxUsers,r.kvPerToken]}")
    check('the KV cache lab’s initial setup (Llama-3.1-70B with its GQA-8) gives 10 users and 320 KiB at 128K, as here', g8 == [10, 327680] and T['gqa8'][2] == 10, g8)
    # constants on the page
    t1 = pg.evaluate("()=>({rows:window.__gm.GQA_T1, large:window.__gm.GQA_LARGE, k:window.__gm.fracK, init:window.__gm.INIT})")
    got = {x['d']: (x['score'], x['time']) for x in t1['rows']}
    check('page GQA Table 1 = facts file, the three T5-XXL rows only (review 8: no T5-Large)', got == T1 and [x['d'] for x in t1['rows']] == ['mha', 'gqa8', 'mqa'] and t1.get('large') is None, t1)
    check('page: the initial setup is MHA on Llama-3.1-70B’s shape, 8 x H100, BF16, 128K (review 8)', t1['init'] == dict(model='llama31-70b', attn='mha', gpu='h100', n=8, wp='bf16', ctx=131072), t1['init'])
    check('page: "under a fifth" uses k = 5', t1['k'] == 5, t1['k'])
    check('page MHA extra parameters', pg.evaluate("()=>window.__gm.mhaExtraParams()") == dW)
    check('page presets = the scenarios above', [tuple(x[k] for k in ['model', 'attn', 'gpu', 'n', 'wp']) for x in pg.evaluate("()=>window.__gm.presets()")] == SCN)
    fr = pg.evaluate("()=>({c:window.__gm.FP8_REAL, f:window.__gm.fp8Real()})")
    check('page FP8 checkpoint counts = this test', (fr['c']['bf16'], fr['c']['f8'], fr['c']['f32'], fr['c']['dsExtra']) == (M405['bf16'], M405['f8'], M405['f32'], 0.24), fr)
    check('page FP8 checkpoints: 405B 56.7 GiB per GPU and 1 user instead of 2; DeepSeek-V3 still 5 users', abs(fr['f']['w4'] - w405) < 1 and (fr['f']['u4'], fr['f']['u4base'], fr['f']['uds'], fr['f']['udsBase']) == (1, 2, 5, 5), fr)
    for lang in ['en', 'zh']:
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(250)
        fid = pg.evaluate("()=>document.getElementById('gqa-mla-fid').innerText").replace('\u2011', '-').replace('\u00a0', ' ')
        want = ['FP8 weights count 1 byte per parameter.', 'about 0.24 GiB per GPU, and its 5 users at 128K each still fit.', 'keeps 81.3B parameters in BF16, every attention matrix among them: 56.7 GiB per GPU instead of 47.2 GiB, so at 128K each only 1 user fits instead of 2.'] if lang == 'en' else \
               ['FP8 权重按每个参数 1 字节算。', '约多 0.24 GiB，每人 128K 上下文时还是装下 5 个用户。', '有 81.3B 个参数是 BF16，所有 attention 矩阵都在其中：每张 GPU 56.7 GiB，而不是 47.2 GiB，每人 128K 上下文时只装得下 1 个用户，而不是 2 个。']
        check(f'{lang}: fidelity card states the FP8 simplification with both real checkpoints', all(x in fid for x in want), [x for x in want if x not in fid])
        tip = pg.evaluate("()=>window.__gm.wpTip({model:'llama31-405b',wp:'fp8'})")
        check(f'{lang}: the weight-precision tip shows the same total as the 405B math line (377.98 GiB)', '405,853,388,800 × 1' in tip and '377.98 GiB' in tip, tip)
    chp = pg.evaluate("()=>({all:window.__gm.challenge(),best:window.__gm.best()})")
    check('page challenge users and best', {x['a']: x['u'] for x in chp['all']} == ch and chp['best'] == 'gqa8', chp)
    # Show the math on every combination, both languages
    for lang in ['en', 'zh']:
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(250)
        pg.evaluate("()=>{const b=document.querySelector('.dyn-root .modes button[data-mode=\"sandbox\"]:not([data-flow=free])'); b.click()}"); pg.wait_for_timeout(250)
        texts = pg.evaluate("""cs=>cs.map(([model,attn,gpu,n,wp,ctx])=>window.__gm.mathText({model,attn,gpu,n,wp,ctx}))""", combos)
        before = len(fails)
        for (m, a, g, t, w, c), lines in zip(combos, texts):
            check_math(f'{lang} math {m} {a} {g} {t} {w} {c}', lines, dict(attn=a, model=m, wp=w, n=t), users(m, a, g, t, w, c))
        print(f'{lang}: math lines re-done on {len(combos)} combinations, {len(fails) - before} failures')
    # the drill: every answer from its own question's numbers
    for k in range(80):
        d = pg.evaluate(f"()=>window.__gm.drill({k})")
        if d['mla']:
            check(f'drill {k} (MLA)', d['per'] == d['L'] * (d['dc'] + d['dr']) * 2, d)
        else:
            per = 2 * d['L'] * d['G'] * d['hd'] * 2
            here = 1 if d['n'] >= d['G'] else d['G'] // d['n']
            check(f'drill {k}', d['per'] == per and d['pg'] == 2 * d['L'] * here * d['hd'] * 2 and d['H'] % d['G'] == 0 and d['H'] % d['n'] == 0, d)
    # the tile
    for a in ['mha', 'gqa8', 'mqa', 'mla']:
        tz = pg.evaluate(f"()=>window.__gm.teaser('{a}')")
        check(f'tile {a}', tz['perTok'] == per_token('llama31-70b', a) and tz['perGpu'] == per_gpu_token('llama31-70b', a, 8), tz)
    check('no page errors', not errs, errs[:3])
    b.close()
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:30] if fails else 'none')
sys.exit(1 if fails else 0)
