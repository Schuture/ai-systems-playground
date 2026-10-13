"""Numbers test for the MoE lab (CONTRACT §6): every number the page states is recomputed here, independently, from Mixtral-8x7B's
config.json values and prefill-decode's step model, and compared with the page's read-only hooks (window.__moe, window.__tzMoe) and
with the text the page shows.

- Parameters counted matrix by matrix (Wq, Wk, Wv, Wo, gate, up, down, router, norms, embedding table, LM head), checked against
  Mistral AI's published 46.7B total and 12.9B per token, and against the lab Model parameters' Mixtral new case (46,702,792,704).
- Memory in GiB for BF16 and FP8, the KV cache budget and the most requests that fit at every context (checked against the KV cache
  memory lab's engine A.kv.compute with its rounded 46.7e9 parameters: the same counts at every context the lab offers).
- Experts read per layer, E·(1 − (1 − k/E)^B), also brute-forced over every routing for small B, and checked by simulation.
- Decode step bytes, FLOPs and time for every scenario and a grid of batches and contexts; every decode step is memory-bound.
- The challenge's best answer (25 users at 4K, 30 ms) by brute force; part 7's layer times; the new cases (DeepSeek-V3 at batch 8,
  top-1 routing).
- The home tile (teasers/moe.js) at 1, 4, 16 and 64 requests of 2K tokens: Mixtral's step equals the lab's decode step; two hypothetical
  dense models with Mixtral's attention (so its KV cache) and every parameter used by every token, 46.7B (part 6's "Dense 46.7B") and
  12.9B (what one Mixtral token uses), counted here from the parameter totals; the times shown, the shared scale, the comparison the
  sentence makes (from the times as shown), the ratios 3.6 and 2.4, the experts read, no memory size (R3.8); at 16 and 64 requests a
  second sentence on what MoE still saves (arithmetic, not memory reads), checked on the numbers; at most 30 words a sentence and 38 in
  all; the notes (at most 4, no "^") reproduce every time from what they say (home review v4 B-08); and the rendered tile on the home
  page shows the same.
- Part 6 (the advantage): Llama 2 13B and 70B counted from their config.json (13,015,864,320 and 68,976,648,192); arithmetic per token,
  weights read per decode step at 1 and 64 requests, and memory for all four models; the benchmark scores are quoted constants from the
  Mixtral paper's Table 2 (arXiv 2401.04088), and the page's summary counts (higher than 13B on 12 of 12, than 70B on 9 of 12) follow.
- Round 1b: the curve's Llama 2 70B line counts its KV cache, like Mixtral's curve, and ends where one H100 is full.
- Round 1: part 6's times (a 4K-token prompt's prefill at the step model's arithmetic rate; the weight reads at 2.01 TB/s, with one
  decimal so they reproduce from the GB shown), the KV cache per token and at 64 requests of 2K, the requests that fit at 2K on one H100,
  Mistral 7B (config.json and Table 2), the ratios the text rounds (1.5, over five, over six, over ten, about 5, about half), the router's
  FLOPs, a real FP8 checkpoint's extra read, part 7's whole token copies.
Run: python3 tests/moe_numbers.py"""
import itertools, math, pathlib, random, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/moe-num.html', '--only', 'moe'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'moe-num-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'moe-num.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'moe-num-wrapped.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


close = lambda a, b, rel=1e-9: abs(a - b) <= rel * max(1.0, abs(a), abs(b))

# ---------- independent model: Mixtral-8x7B config.json (mistralai/Mixtral-8x7B-v0.1) ----------
cfg = dict(hidden_size=4096, intermediate_size=14336, num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
           num_local_experts=8, num_experts_per_tok=2, vocab_size=32000, tie_word_embeddings=False)
d, f, L, H, KV = cfg['hidden_size'], cfg['intermediate_size'], cfg['num_hidden_layers'], cfg['num_attention_heads'], cfg['num_key_value_heads']
E, K, V = cfg['num_local_experts'], cfg['num_experts_per_tok'], cfg['vocab_size']
hd = d // H
mats = {'Wq': d * H * hd, 'Wk': d * KV * hd, 'Wv': d * KV * hd, 'Wo': H * hd * d}
attn = sum(mats.values())
expert = d * f + d * f + f * d                      # gate, up, down
router = d * E
norms = d + d                                       # input and post-attention RMSNorm
emb, head, fnorm = V * d, V * d, d
layer = attn + router + norms + E * expert
total = emb + head + fnorm + L * layer
active = emb + head + fnorm + L * (attn + router + norms + K * expert)
shared = total - L * E * expert
sread = shared - emb                                # a decode step does not read the whole embedding table
mult = sread + L * K * expert
check('attention block per layer', attn == 41_943_040, attn)
check('one expert', expert == 176_160_768, expert)
check('total = 46,702,792,704 (Hugging Face count; the Model parameters lab)', total == 46_702_792_704, total)
check('active = 12,879,925,248', active == 12_879_925_248, active)
check('Mistral AI: 46.7B total, 12.9B per token', round(total / 1e9, 1) == 46.7 and round(active / 1e9, 1) == 12.9)
check('8x7B is not 56B: the total is far below 8 x Mistral-7B (7,241,732,096)', total < 8 * 7_241_732_096)
# Mistral-7B-v0.1 + 7 more MLP blocks per layer + routers (the Model parameters lab's new case)
mistral7b = emb + head + fnorm + L * (attn + norms + expert)
check('Mistral-7B-v0.1 = 7,241,732,096', mistral7b == 7_241_732_096, mistral7b)
check('Mixtral = Mistral-7B + 32 x 7 experts + routers', total == mistral7b + L * (E - 1) * expert + L * router)
top1 = total - L * (E - 1) * expert
check('top-1 active = Mistral-7B + routers = 7,242,780,672', top1 == mistral7b + L * router == 7_242_780_672, top1)

GiB = 2 ** 30
KVT = 2 * L * KV * hd * 2
check('KV cache per token 128 KiB', KVT == 131072)
PEAK, BW, MFU, MBU = 1979e12 / 2, 3.35e12, 0.5, 0.6
FS, BS = PEAK * MFU, BW * MBU
USABLE, OH = 80 * GiB * 0.9, 2 * GiB
BPP = {'bf16': 2, 'fp8': 1}


def mem(p, c):
    w = BPP[p] * total; budget = USABLE - w - OH
    return w, budget, (math.floor(budget / (c * KVT) + 1e-9) if budget > 0 else 0)


def expected_read(B, e=E, k=K):
    return e * (1 - (1 - k / e) ** B)


def brute_read(B, e=E, k=K):
    """average number of distinct experts over every choice of k-subsets for B tokens (exact, small B)"""
    pairs = list(itertools.combinations(range(e), k)); tot = 0
    for picks in itertools.product(pairs, repeat=B): tot += len(set(x for pk in picks for x in pk))
    return tot / len(pairs) ** B


def step(B, c, p='fp8', T=None):
    T = expected_read(B) if T is None else T
    fl = B * (2 * mult + 4 * L * d * (c + 1))
    by = BPP[p] * (sread + L * T * expert) + B * c * KVT
    return dict(T=T, fl=fl, by=by, tc=fl / FS, tm=by / BS, t=max(fl / FS, by / BS))


for B in (1, 2, 3):
    check(f'expected experts read equals brute force at B={B}', close(expected_read(B), brute_read(B), 1e-12), (expected_read(B), brute_read(B)))
rng = random.Random(7)
sim = sum(len({x for _ in range(8) for x in rng.sample(range(8), 2)}) for _ in range(40000)) / 40000
check('expected experts read at B=8 matches a simulation (7.2)', abs(sim - expected_read(8)) < 0.02, (sim, expected_read(8)))

with sync_playwright() as pw:
    b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300); pg.click('.langsw button[data-lang="en"]'); pg.evaluate("()=>{location.hash='moe'}"); pg.wait_for_timeout(600)
    ev = lambda js: pg.evaluate(js)
    C = ev('()=>window.__moe.consts')
    for k_, v_ in (('P_ATTN', attn), ('P_EXPERT', expert), ('P_ROUTER', router), ('P_TOTAL', total), ('P_ACTIVE', active), ('P_SHARED', shared), ('P_SREAD', sread), ('P_MULT', mult), ('KV_TOK', KVT)):
        check(f'page {k_}', C[k_] == v_, (C[k_], v_))
    # memory, in both precisions, at every context the lab offers; and the KV cache memory lab's engine gives the same counts
    for p in ('bf16', 'fp8'):
        for c in C['C_STOPS']:
            w, bud, mx = mem(p, c); m = ev(f"()=>window.__moe.memOf('{p}',{c})")
            check(f'memory {p} {c}: weights', close(m['w'], w)); check(f'memory {p} {c}: max requests {mx}', m['max'] == mx, (m['max'], mx))
            kvl = ev(f"()=>{{const r=window.ASP.kv.compute({{model:'mixtral-8x7b',gpu:'h100',n:1,ctx:{c},users:1,wp:'{p}',kp:'bf16',attn:'gqa',util:0.9,overhead:2}}); return r.maxUsers}}")
            check(f'memory {p} {c}: equals the KV cache memory lab ({kvl})', kvl == mx, (kvl, mx))
    check('BF16 weights 87.0 GiB, over one H100 (72 GiB usable)', f'{2 * total / GiB:.1f}' == '87.0' and 2 * total > USABLE)
    check('FP8 weights 43.5 GiB, 26.5 GiB left', f'{total / GiB:.1f}' == '43.5' and f'{mem("fp8", 2048)[1] / GiB:.1f}' == '26.5')
    check('the prediction: 87.0 GiB is in "80 to 100 GiB"', C['PRED_RIGHT'] == '80-100' and 80 <= 2 * total / GiB < 100)
    # decode steps: a grid and every scenario
    grid = [(B, c) for c in (512, 2048, 4096, 16384, 32768) for B in (1, 2, 4, 8, 16, 32, 64, 106) if B <= mem('fp8', c)[2]]
    for B, c in grid:
        s = step(B, c); pgv = ev(f"()=>window.__moe.decode({B},{c},'fp8')")
        check(f'decode B={B} c={c}: time', close(pgv['t'], s['t']), (pgv['t'], s['t']))
        check(f'decode B={B} c={c}: bytes', close(pgv['by'], s['by']))
        check(f'decode B={B} c={c}: memory-bound', s['tm'] > s['tc'], s)
    for c in C['C_STOPS']:   # every decode step the lab can show is memory-bound (the fidelity note says the BF16 arithmetic rate changes none)
        mx = mem('fp8', c)[2]; s = step(mx, c)
        check(f'the largest batch at {c} is memory-bound', s['tm'] > s['tc'], (s['tm'], s['tc']))
    shown = {1: '6.48', 8: '21.99', 64: '31.72', 106: '37.33'}
    for B, v in shown.items():
        check(f'2K, B={B}: {v} ms', f"{step(B, 2048)['t'] * 1e3:.2f}" == v, step(B, 2048)['t'] * 1e3)
    check('batch 64 reads as much as all 8 experts (to 0.01 ms)', f"{step(64, 2048)['t'] * 1e3:.2f}" == f"{step(64, 2048, T=8)['t'] * 1e3:.2f}")
    check('batch 1 reads as much as 2 experts', close(step(1, 2048)['t'], step(1, 2048, T=2)['t']))
    for sc in C['PRESETS']:
        r = ev(f"()=>window.__moe.compute({{p:'{sc['p']}',c:{sc['c']},B:{sc['B']}}})")
        w, bud, mx = mem(sc['p'], sc['c'])
        if sc['p'] == 'bf16':
            check(f'scenario {sc["id"]}: does not fit', not r['m']['fits'] and bud < 0)
        else:
            s = step(sc['B'], sc['c'])
            check(f'scenario {sc["id"]}: {s["t"] * 1e3:.2f} ms, {s["T"]:.1f} experts', close(r['d']['t'], s['t']) and f"{r['d']['T']:.1f}" == f"{s['T']:.1f}")
    check('long documents: at most 13 requests at 16K', mem('fp8', 16384)[2] == 13)
    # the challenge: brute force over every batch that fits
    mx4 = mem('fp8', 4096)[2]
    best = max(B for B in range(1, mx4 + 1) if step(B, 4096)['t'] <= 0.030)
    check('challenge: 53 users fit at 4K', mx4 == 53 and ev('()=>window.__moe.chMax()') == 53)
    check('challenge: best 25 users', best == 25 and ev('()=>window.__moe.chBest()') == 25, best)
    check('challenge: 25 → 29.83 ms, 26 → 30.10 ms', f"{step(25, 4096)['t'] * 1e3:.2f}" == '29.83' and f"{step(26, 4096)['t'] * 1e3:.2f}" == '30.10')
    # with a real FP8 checkpoint (embedding table, LM head, routers in BF16) the answer stays 25
    extra = emb + head + L * router
    real = lambda B: (sread + extra - emb + L * expected_read(B) * expert - 0 + B * 4096 * KVT) / BS   # LM head and routers read in BF16: one more byte each
    check('challenge answer unchanged with a real FP8 checkpoint', max(B for B in range(1, 53) if max(real(B), step(B, 4096)['tc']) <= 0.030) == 25)
    check('a real FP8 checkpoint: one request fewer at 2K and 4K', math.floor((mem('fp8', 2048)[1] - extra) / (2048 * KVT)) == 105 and math.floor((mem('fp8', 4096)[1] - extra) / (4096 * KVT)) == 52)
    # part 6: expert parallelism, 8,192 copies; balanced 1,024 each
    for sid, share, want in (('bal', 1 / 8, '0.73'), ('q', 1 / 4, '1.46'), ('h', 1 / 2, '2.92')):
        hot = share * 4096 * 2; t = max(hot * 2 * expert / FS, expert / BS)
        r = ev(f"()=>window.__moe.epRun('{sid}')")
        check(f'part 6 {sid}: {want} ms', close(r['t'], t) and f'{t * 1e3:.2f}' == want, (r['t'], t))
        check(f'part 6 {sid}: {share * 8:.0f}x the balanced time', close(r['ratio'], share * 8))
    # part 7: whole token copies that add up to 8,192 (B-O1)
    for sid, want in (('bal', [1024] * 8), ('q', [878, 878, 2048, 878, 878, 878, 877, 877]), ('h', [586, 585, 4096, 585, 585, 585, 585, 585])):
        r = ev(f"()=>window.__moe.epRun('{sid}')")
        check(f'part 7 {sid}: whole copies {want}', r['loads'] == want and sum(r['loads']) == 8192, r['loads'])
    # the new cases
    ds = 256 * (1 - (248 / 256) ** 8)
    check('DeepSeek-V3 at batch 8: about 22% of 256 routed experts (10% to 40%)', 0.10 <= ds / 256 < 0.40 and round(ds / 256 * 100) == 22, ds)
    check('DeepSeek-V3 at batch 64: about 87%', round(256 * (1 - (248 / 256) ** 64) / 256 * 100) == 87)
    x = ev('()=>window.__moe.exits()')
    check('new case 1: right option is "10% to 40%"', x['transfer'][0]['o'][x['transfer'][0]['a']] == '10% to 40%')
    check('new case 2: right option is "About 7.2B"', x['transfer'][1]['o'][x['transfer'][1]['a']] == 'About 7.2B' and round(top1 / 1e9, 1) == 7.2)
    check('new case 2: 12.9B − 5.64B', round(L * expert / 1e9, 2) == 5.64)
    check('teaser hook present', ev('()=>!!window.__tzMoe'))
    # the page shows the numbers it computes: part 1's cards, part 5 at 8 requests
    cards = ev("()=>[...document.querySelectorAll('#moe-metrics .metric .v')].map(e=>e.textContent)")
    check('part 1 cards: 46.7B, 12.9B, memory hidden', cards[:3] == ['46.7B', '12.9B', '?'], cards)
    ev("()=>document.querySelector('#moe-rail [data-go=\"4\"]').click()"); pg.wait_for_timeout(300)
    ev("()=>{const x=document.querySelector('#moe-ic-B-in'); x.value=8; x.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(300)
    cards = ev("()=>[...document.querySelectorAll('#moe-metrics .metric .v')].map(e=>e.textContent)")
    check('part 5 cards at 8 requests: 7.2 of 8, 44.2 GB, 21.99 ms, 364 tokens/s', cards == ['7.2 of 8', '44.2 GB', '21.99 ms', '364'], cards)
    # ---------- part 6: Mixtral next to dense models ----------
    def dense(V, dd, LL, HH, KVh, ff):
        hdd = dd // HH; layer_ = 2 * dd * dd + 2 * dd * KVh * hdd + 3 * dd * ff + 2 * dd; embd = V * dd
        return dict(total=2 * embd + LL * layer_ + dd, read=embd + LL * layer_ + dd, kv=2 * LL * KVh * hdd * 2)
    l13 = dense(32000, 5120, 40, 40, 40, 13824)     # meta-llama/Llama-2-13b-hf config.json (NousResearch mirror)
    l70 = dense(32000, 8192, 80, 64, 8, 28672)      # meta-llama/Llama-2-70b-hf config.json (NousResearch mirror)
    check('Llama 2 13B = 13,015,864,320', l13['total'] == 13_015_864_320, l13)
    check('Llama 2 70B = 68,976,648,192', l70['total'] == 68_976_648_192, l70)
    check('KV per token: 13B 800 KiB, 70B 320 KiB (the fidelity note)', l13['kv'] == 800 * 1024 and l70['kv'] == 320 * 1024)
    want = {'l13': dict(total=l13['total'], mult=l13['read'], read1=l13['read'], readB=l13['read']),
            'mx': dict(total=total, mult=mult, read1=mult, readB=sread + L * expected_read(64) * expert),
            'l70': dict(total=l70['total'], mult=l70['read'], read1=l70['read'], readB=l70['read']),
            'd47': dict(total=total, mult=total - emb, read1=total - emb, readB=total - emb)}
    pm = {m['id']: m for m in ev('()=>window.__moe.cmpModels()')}
    for k_, w_ in want.items():
        for f_ in ('total', 'mult', 'read1', 'readB'):
            check(f'part 6 {k_}: {f_}', close(pm[k_][f_], w_[f_]), (pm[k_][f_], w_[f_]))
        check(f'part 6 {k_}: FLOPs per token = 2 × multiplied parameters', close(pm[k_]['fl'], 2 * w_['mult']))
    gfl = {k_: round(2 * w_['mult'] / 1e9, 1) for k_, w_ in want.items()}
    check('part 6: 25.7 / 25.5 / 137.4 / 93.1 GFLOP per token', [gfl[k_] for k_ in ('l13', 'mx', 'l70', 'd47')] == [25.7, 25.5, 137.4, 93.1], gfl)
    check('part 6: Llama 2 70B computes 5.4 times as much per token (the paper: 5x fewer active parameters)', f"{want['l70']['mult'] / want['mx']['mult']:.1f}" == '5.4')
    check('part 6: the hypothetical dense 46.7B computes 3.7 times as much', f"{want['d47']['mult'] / want['mx']['mult']:.1f}" == '3.7')
    check('part 6: weights read per step (FP8): 12.9, 12.7 → 46.6, 68.7, 46.6 GB',
          [f"{want[k_][f_] / 1e9:.1f}" for k_, f_ in (('l13', 'read1'), ('mx', 'read1'), ('mx', 'readB'), ('l70', 'read1'), ('d47', 'read1'))] == ['12.9', '12.7', '46.6', '68.7', '46.6'])
    check('part 6: memory (FP8): 12.1, 43.5, 64.2, 43.5 GiB', [f"{want[k_]['total'] / GiB:.1f}" for k_ in ('l13', 'mx', 'l70', 'd47')] == ['12.1', '43.5', '64.2', '43.5'])
    check('part 6: 13B memory ratio 3.6 and read ratio at 64 requests 3.6', f"{total / l13['total']:.1f}" == '3.6' and f"{want['mx']['readB'] / l13['read']:.1f}" == '3.6')
    check('part 6: one request reads about a quarter of the dense 46.7B', 0.25 <= want['mx']['read1'] / want['d47']['read1'] < 0.3)
    # the Mixtral paper, Table 2 (arXiv 2401.04088), as printed: MMLU, HellaS, WinoG, PIQA, Arc-e, Arc-c, NQ, TriQA, HumanE, MBPP, Math, GSM8K
    T2 = {'l13': [55.6, 80.7, 72.9, 80.8, 75.2, 48.8, 16.7, 64.0, 18.9, 35.4, 6.0, 34.3],
          'mx': [70.6, 84.4, 77.2, 83.6, 83.1, 59.7, 30.6, 71.5, 40.2, 60.7, 28.4, 74.4],
          'l70': [69.9, 85.4, 80.4, 82.6, 79.9, 56.5, 25.4, 73.0, 29.3, 49.8, 13.8, 69.6],
          'm7': [62.5, 81.0, 74.2, 82.2, 80.5, 54.9, 23.2, 62.5, 26.2, 50.2, 12.7, 50.0]}
    check('Table 2 constants on the page equal the paper', C['BENCH'] == T2, C['BENCH'])
    b13 = sum(a > b for a, b in zip(T2['mx'], T2['l13'])); b70 = sum(a > b for a, b in zip(T2['mx'], T2['l70']))
    check('Mixtral higher than Llama 2 13B on 12 of 12, than Llama 2 70B on 9 of 12', b13 == 12 and b70 == 9 and ev("()=>[window.__moe.beats('l13'),window.__moe.beats('l70')]") == [12, 9], (b13, b70))
    check('MMLU gap to Llama 2 13B: +15.0', f"{T2['mx'][0] - T2['l13'][0]:.1f}" == '15.0')
    check('GSM8K gap to Llama 2 13B: +40.1; HellaSwag +3.7', f"{T2['mx'][11] - T2['l13'][11]:.1f}" == '40.1' and f"{T2['mx'][1] - T2['l13'][1]:.1f}" == '3.7')
    # ---------- round 1: part 6 in time, with the KV cache, Mistral 7B ----------
    m7 = dense(32000, 4096, 32, 32, 8, 14336)       # mistralai/Mistral-7B-v0.1 config.json
    check('Mistral 7B = 7,241,732,096 (its config.json)', m7['total'] == 7_241_732_096 == mistral7b, m7)
    check('Mistral 7B: 14.2 GFLOP per token, about half of Llama 2 13B (0.55)', f"{2 * m7['read'] / 1e9:.1f}" == '14.2' and 0.5 <= m7['read'] / l13['read'] < 0.6)
    check('Table 2, Mistral 7B: MMLU 62.5, GSM8K 50.0, HumanEval 26.2, HellaSwag 81.0', [T2['m7'][i] for i in (0, 11, 8, 1)] == [62.5, 50.0, 26.2, 81.0])
    check('Mixtral +8.1 on MMLU over Mistral 7B at 1.8 times the arithmetic', f"{T2['mx'][0] - T2['m7'][0]:.1f}" == '8.1' and f"{mult / m7['read']:.1f}" == '1.8')
    kv = {'l13': l13['kv'], 'mx': KVT, 'l70': l70['kv'], 'd47': KVT}
    for k_, w_ in want.items():
        pre = 4096 * 2 * w_['mult'] / FS; t1 = w_['read1'] / BS; tB = w_['readB'] / BS
        fit = math.floor((USABLE - OH - w_['total']) / (2048 * kv[k_]))
        for f_, v_ in (('pre', pre), ('t1', t1), ('tB', tB), ('kv', kv[k_]), ('fit', fit), ('kvB', 64 * 2048 * kv[k_])):
            check(f'part 6 {k_}: {f_}', close(pm[k_][f_], v_), (pm[k_][f_], v_))
    pre_s = [f"{4096 * round(2 * want[k_]['mult'] / 1e9, 1) * 1e9 / FS:.2f}" for k_ in ('l13', 'mx', 'l70', 'd47')]
    check('part 6: a 4K-token prefill, 0.21 / 0.21 / 1.14 / 0.77 s, also from the GFLOP shown', pre_s == ['0.21', '0.21', '1.14', '0.77'] and [f"{pm[k_]['pre']:.2f}" for k_ in ('l13', 'mx', 'l70', 'd47')] == pre_s, pre_s)
    shown_gb = {('l13', 'read1'): 12.9, ('mx', 'read1'): 12.7, ('mx', 'readB'): 46.6, ('l70', 'read1'): 68.7, ('d47', 'read1'): 46.6}
    for (k_, f_), g in shown_gb.items():
        exact = f"{want[k_][f_] / BS * 1e3:.1f}"; from_shown = f"{g * 1e9 / BS * 1e3:.1f}"
        check(f'part 6 {k_} {f_}: the time with one decimal reproduces from the GB shown ({exact} ms)', exact == from_shown, (exact, from_shown))
    check('part 6: read times 6.4, 6.3 → 23.2, 34.2, 23.2 ms', [f"{want[k_][f_] / BS * 1e3:.1f}" for k_, f_ in shown_gb] == ['6.4', '6.3', '23.2', '34.2', '23.2'])
    check('part 6: requests of 2K that fit: 37 / 106 / 9 / 106', [pm[k_]['fit'] for k_ in ('l13', 'mx', 'l70', 'd47')] == [37, 106, 9, 106])
    check('part 6: the KV cache at 64 requests of 2K: 107 / 17 / 43 GB', [round(pm[k_]['kvB'] / 1e9) for k_ in ('l13', 'mx', 'l70')] == [107, 17, 43])
    check('part 6: with the KV cache, Llama 2 13B reads more than Mixtral at 64 × 2K (120.2 vs 63.8 GB)', f"{(l13['read'] + 64 * 2048 * l13['kv']) / 1e9:.1f}" == '120.2' and f"{(want['mx']['readB'] + 64 * 2048 * KVT) / 1e9:.1f}" == '63.8')
    check('part 6: KV per token, Llama 2 13B over six times Mixtral’s (6.25), Llama 2 70B 2.5 times', l13['kv'] / KVT == 6.25 and l70['kv'] / KVT == 2.5)
    check('part 6: one H100 holds over ten times as many Mixtral requests as Llama 2 70B ones (106 vs 9)', 10 < 106 / 9 < 12)
    check('busy scenario: Llama 2 70B reads about 1.5 times the weights and computes over five times as much', f"{l70['read'] / want['mx']['readB']:.1f}" == '1.5' and 5 < l70['read'] / mult < 6)
    check('part 5 at 64: Mixtral computes about a fifth of Llama 2 70B’s arithmetic', 0.17 < mult / l70['read'] < 0.21)
    check('misconception 3: a 64-request step takes about 5 times one request’s', round(step(64, 2048)['t'] / step(1, 2048)['t']) == 5)
    check('part 4: 2 × 12.75B = 25.5 GFLOP (the tooltip)', f"{mult / 1e9:.2f}" == '12.75' and f"{2 * 12.75:.1f}" == '25.5' and f"{2 * mult / 1e9:.1f}" == '25.5')
    check('the router: 65,536 FLOPs per token per layer, under 0.01% of the token’s arithmetic', 2 * d * E == 65536 and 2 * d * E * L / (2 * mult) < 1e-4)
    check('a real FP8 checkpoint reads the LM head and routers in BF16: about 0.07 ms more per step', f"{(head + L * router) / BS * 1e3:.2f}" == '0.07')
    check('FP8 dense rate 1,979 TFLOP/s halves part 7’s times (0.73 → 0.36 ms)', f"{1024 * 2 * expert / (1979e12 * 0.5) * 1e3:.2f}" == '0.36' and f"{1024 * 2 * expert / FS * 1e3:.2f}" == '0.73')
    check('team scenario: more than three times as long as alone', 3 < step(8, 2048)['t'] / step(1, 2048)['t'] < 3.5)
    check('docs scenario: about five and a half experts per layer', f"{expected_read(4):.1f}" == '5.5')
    # ---------- round 1b: the curve's 70B line counts the same things as Mixtral's curve (weights and its own KV cache) ----------
    cl = ev('()=>window.__moe.curveLines()')       # drawn in part 5 at 8 requests, 2K
    t70 = lambda B, c: max((l70['read'] + B * c * l70['kv']) / BS, B * (2 * l70['read'] + 4 * 80 * 8192 * (c + 1)) / FS)
    fit70 = math.floor((USABLE - OH - l70['total']) / (2048 * l70['kv']))
    check('curve: the 70B line includes its KV cache: 34.52 ms at 1 request, 37.19 ms at 9, and ends at 9 (one H100 is full)',
          bool(cl and cl['c'] == 2048 and cl['l70']) and close(cl['l70']['t0'], t70(1, 2048)) and close(cl['l70']['t1'], t70(9, 2048)) and cl['l70']['B1'] == 9 == fit70
          and f"{t70(1, 2048) * 1e3:.2f}" == '34.52' and f"{t70(9, 2048) * 1e3:.2f}" == '37.19', cl)
    check('curve: Mixtral’s curve includes its KV cache too, and stays below the 70B line wherever both are drawn',
          all(step(B, 2048)['by'] == BPP['fp8'] * (sread + L * expected_read(B) * expert) + B * 2048 * KVT for B in range(1, 10)) and all(step(B, 2048)['t'] < t70(B, 2048) for B in range(1, 10)))
    check('curve: the 70B line ends where one H100 is full at every context (512: 36, 4K: 4, 16K: 1, 32K: none)', [ev(f'()=>window.__moe.l70Fit({c})') for c in (512, 4096, 16384, 32768)] == [36, 4, 1, 0])
    check('round 1c: Llama 2 70B takes at most 4,096 tokens, so the curve draws it at 2K and below, not at 4K and above',
          ev("()=>[512,1024,2048,4096,8192,16384,32768].map(c=>window.__moe.l70Takes(c))") == [True, True, True, False, False, False, False])
    check('the challenge: Llama 2 70B’s weights alone take 34 ms per step, over the 30 ms goal at any context', f"{l70['read'] / BS * 1e3:.2f}" == '34.19' and round(l70['read'] / BS * 1e3) == 34 > 30)
    # ---------- the home tile (teasers/moe.js): Mixtral-8x7B next to two hypothetical dense models, 1 to 64 requests of 2K tokens ----------
    # Both dense models keep Mixtral's attention, so its KV cache (128 KiB per token), and use every parameter for every token: one with all
    # 46.7B (part 6's "Dense 46.7B"), one with the 12.9B that one Mixtral token uses. Same step model, FP8 weights.
    def dense_step(P, B, c=2048):
        m = P - emb                                  # every parameter but the embedding table (a lookup) is multiplied and read
        fl = B * (2 * m + 4 * L * d * (c + 1)); by = m + B * c * KVT
        return dict(fl=fl, by=by, tc=fl / FS, tm=by / BS, t=max(fl / FS, by / BS))
    ms = lambda s: f'{s * 1e3:.2f}'
    TZN = [1, 4, 16, 64]
    check('tile: 1, 4, 16 and 64 requests of 2K tokens', ev('()=>window.__tzMoe.NS') == TZN and ev('()=>window.__tzMoe.CTX') == 2048)
    want_tz = {}
    for B in TZN:
        tz = ev(f'()=>window.__tzMoe.step({B})'); lab = ev(f"()=>window.__moe.decode({B},2048,'fp8')")
        mx, a, z = step(B, 2048), dense_step(total, B), dense_step(active, B)
        want_tz[B] = dict(mx=mx['t'], d47=a['t'], d13=z['t'])
        check(f'tile {B}: Mixtral {ms(mx["t"])} ms reading {mx["T"]:.2f} of 8 experts per layer, the lab’s decode step',
              all(close(tz['mx'][k_], mx[k_]) for k_ in ('fl', 'by', 't')) and close(tz['T'], mx['T']) and close(lab['t'], mx['t']), (tz['mx'], mx))
        check(f'tile {B}: dense 46.7B {ms(a["t"])} ms', all(close(tz['d47'][k_], a[k_]) for k_ in ('fl', 'by', 't')), (tz['d47'], a))
        check(f'tile {B}: dense 12.9B {ms(z["t"])} ms', all(close(tz['d13'][k_], z[k_]) for k_ in ('fl', 'by', 't')), (tz['d13'], z))
        check(f'tile {B}: the dense 46.7B reads what part 6’s reads, plus the requests’ KV caches', close(a['by'], pm['d47']['read1'] + B * 2048 * pm['d47']['kv']) and close(2 * (total - emb), pm['d47']['fl']))
        check(f'tile {B}: every step is limited by its reads (the note)', all(s_['tm'] > s_['tc'] for s_ in (mx, a, z)), [(s_['tm'], s_['tc']) for s_ in (mx, a, z)])
    m1, d1 = step(1, 2048), dense_step(active, 1)
    check('tile 1: one request reads 2 of 8 experts, so Mixtral’s step is the dense 12.9B’s: the same bytes and the same FLOPs', expected_read(1) == 2 and m1['by'] == d1['by'] and m1['fl'] == d1['fl'], (m1, d1))
    check('tile 64: the requests read all but a few hundred bytes of the dense 46.7B’s weights', 0 <= dense_step(total, 64)['by'] - step(64, 2048)['by'] < 1024 and f'{expected_read(64):.1f}' == '8.0')
    SHW = {B: {k_: ms(v_) for k_, v_ in want_tz[B].items()} for B in TZN}
    check('tile: the times shown (dense 46.7B / Mixtral / dense 12.9B): 23.30 / 6.48 / 6.48, 23.70 / 16.61 / 6.88, 25.31 / 25.08 / 8.48, 31.72 / 31.72 / 14.89 ms',
          [[SHW[B][k_] for k_ in ('d47', 'mx', 'd13')] for B in TZN] == [['23.30', '6.48', '6.48'], ['23.70', '16.61', '6.88'], ['25.31', '25.08', '8.48'], ['31.72', '31.72', '14.89']], SHW)
    check('tile: Mixtral’s times are the lab’s (6.48 ms alone, 31.72 ms at 64 requests)', SHW[1]['mx'] == shown[1] and SHW[64]['mx'] == shown[64])
    scale = max(v_ for B in TZN for v_ in want_tz[B].values())
    check('tile: one scale for every setting, set by the longest bar (the dense 46.7B at 64 requests)', close(ev('()=>window.__tzMoe.SCALE'), scale) and scale == want_tz[64]['d47'])
    for B in TZN:
        bs = ev(f'()=>window.__tzMoe.bars({B})')
        check(f'tile {B}: bars top to bottom dense 46.7B, Mixtral, dense 12.9B, each with its time and its length on the shared scale',
              [x['id'] for x in bs] == ['d47', 'mx', 'd13'] and all(x['text'] == SHW[B][x['id']] + ' ms' and close(x['w'], want_tz[B][x['id']] / scale * 100) for x in bs), bs)

    def verdict(B):      # the comparison the sentence makes, from the times as shown
        mx_, a_, z_ = (float(SHW[B][k_]) for k_ in ('mx', 'd47', 'd13'))
        if mx_ == z_: return ['as13', f'{a_ / mx_:.1f}']
        if mx_ == a_: return ['as47', None]
        if mx_ / a_ >= 0.98: return ['near47', None]
        return ['times13', f'{mx_ / z_:.1f}']
    V = {B: verdict(B) for B in TZN}
    check('tile: 1 request as fast as the dense 12.9B and 3.6 times the dense 46.7B’s speed; 4 requests 2.4 times the dense 12.9B’s time; 16 almost the dense 46.7B’s; 64 the dense 46.7B’s',
          V == {1: ['as13', '3.6'], 4: ['times13', '2.4'], 16: ['near47', None], 64: ['as47', None]}, V)
    check('tile: the two ratios hold on the exact times too', f"{want_tz[1]['d47'] / want_tz[1]['mx']:.1f}" == '3.6' and f"{want_tz[4]['mx'] / want_tz[4]['d13']:.1f}" == '2.4')
    check('tile 16: almost, not quite (25.08 against 25.31 ms, within 2%)', SHW[16]['mx'] != SHW[16]['d47'] and want_tz[16]['mx'] / want_tz[16]['d47'] > 0.98)
    plain = lambda s: re.sub(r'<[^>]+>', '', s).replace(' ', ' ')
    for B in TZN:
        pv = ev(f'()=>window.__tzMoe.verdict({B})')
        check(f'tile {B}: the page picks the same comparison', [pv['k'], pv.get('x')] == V[B], (pv, V[B]))
        Tw = '2' if B == 1 else f'{expected_read(B):.1f}'
        want_s = {'as13': ['as fast as a dense 12.9B', f'{V[B][1]} times a dense 46.7B’s speed'], 'times13': [f'{V[B][1]} times as long per token as a dense 12.9B'],
                  'near47': ['almost as slowly as a dense 46.7B'], 'as47': ['decodes as slowly as a dense 46.7B']}[V[B][0]]
        s_ = plain(ev(f'()=>window.__tzMoe.read({B})'))
        lead = '1 request: Mixtral‑8x7B reads 2 of 8 experts per layer' if B == 1 else f'{B} requests pick different experts, so Mixtral‑8x7B reads {Tw} of 8 per layer'
        # R6.26 (home review v4 A #6): where the batch reads (almost) every expert, a second sentence says what MoE still saves
        two = {'near47': 'It still saves arithmetic, but hardly any memory reads: each token computes with only 12.9B parameters.',
               'as47': 'It still saves arithmetic, not memory reads: each token computes with only 12.9B parameters.'}.get(V[B][0])
        sents = [x for x in re.split(r'(?<=\.) ', s_) if x]
        check(f'tile {B}: the sentence: “{lead} …”, its comparison{", then what it still saves" if two else ""}; {2 if two else 1} sentence(s) of at most 30 words, at most 38 in all (R11.5)',
              s_.startswith(lead) and all(x in s_ for x in want_s) and len(sents) == (2 if two else 1) and (not two or sents[1] == two)
              and all(len(x.split()) <= 30 for x in sents) and len(s_.split()) <= 38, ([len(x.split()) for x in sents], s_))
    # the second sentence holds on the numbers shown: at 16 and 64 requests each Mixtral token does the dense 12.9B's arithmetic, under 30% of
    # the dense 46.7B's; the step's reads are within 1% of the dense 46.7B's at 16 ("hardly any") and within 1 KiB at 64 ("not"); and
    # every step is limited by its reads, so the arithmetic saved does not shorten it
    for B in (16, 64):
        mx, a, z = step(B, 2048), dense_step(total, B), dense_step(active, B)
        saved = 1 - mx['by'] / a['by']
        check(f'tile {B}: Mixtral saves arithmetic ({mx["fl"] / a["fl"] * 100:.0f}% of the dense 46.7B’s, the same as the dense 12.9B’s) but {"hardly any" if B == 16 else "no"} memory reads ({saved * 100:.2f}% fewer bytes)',
              mx['fl'] == z['fl'] and mx['fl'] / a['fl'] < 0.30 and ((0 < saved < 0.01) if B == 16 else (0 <= a['by'] - mx['by'] < 1024)) and mx['tm'] > mx['tc'], (mx['fl'] / a['fl'], saved))
        check(f'tile {B}: “12.9B parameters” per token is the active count Mistral AI gives (12,879,925,248)', round(active / 1e9, 1) == 12.9)
    # the rendered tile on the home page shows the same numbers
    pg.evaluate("()=>{location.hash='map~models'}"); pg.wait_for_timeout(800)
    for B in TZN:
        pg.evaluate(f"()=>document.querySelector('#tile-moe .seg-ctl button[data-v=\"{B}\"]').click()"); pg.wait_for_timeout(120)
        r_ = ev("()=>{const t=document.querySelector('#tile-moe'); return {v:[...t.querySelectorAll('.tz-moe-v')].map(e=>e.textContent), n:[...t.querySelectorAll('.tz-moe-n')].map(e=>e.textContent), read:t.querySelector('.tz-read').textContent,"
                "ex:[...t.querySelectorAll('.tz-moe-ex i')].map(i=>parseFloat(i.style.width)), w:[...t.querySelectorAll('.tz-moe-tr i')].map(i=>parseFloat(i.style.width))}}")
        check(f'tile on the home page, {B}: the values, the bars, the experts filled and the sentence', r_['n'] == ['Dense 46.7B', 'Mixtral‑8x7B', 'Dense 12.9B'] and r_['v'] == [SHW[B][k_] + ' ms' for k_ in ('d47', 'mx', 'd13')]
              and all(abs(w_ - want_tz[B][k_] / scale * 100) < 0.01 for w_, k_ in zip(r_['w'], ('d47', 'mx', 'd13'))) and abs(sum(r_['ex']) / 100 - expected_read(B)) < 0.005
              and plain(r_['read']) == plain(ev(f'()=>window.__tzMoe.read({B})')), r_)
    # the notes (home review v4 A #7, A #8, B-08): at most 4, one sentence each, no "^"; with them alone a reader redoes every time on the tile:
    # bytes = FP8 weights at 1 byte per parameter, without the embedding table (one row per request), plus each request's 2K tokens of BF16
    # KV cache; time = bytes ÷ (60% of 3.35 TB/s). The numbers are read from the notes' own text
    nts = [x.replace('\u00a0', ' ') for x in ev('()=>window.__tzMoe.notes()')]
    check('tile notes: at most 4, one sentence each, no “^” (the routing formula in words)', len(nts) <= 4 and all(x.endswith('.') and x.count('. ') == 0 for x in nts)
          and not any('^' in x for x in nts) and '8 × (1 − (6/8) to the power N)' in ' '.join(nts), nts)
    nt = ' '.join(nts)
    m_bw = re.search(r'assumed (\d+)% of an H100 SXM’s ([\d.]+) TB/s', nt); m_kv = re.search(r'KV cache of (\d+)K tokens, in BF16 at (\d+) KiB per token in all three models', nt)
    check('tile notes: they name FP8 weights, the embedding table read as one row per request, BF16 KV cache at 128 KiB per token, 2K tokens, 60% of 3.35 TB/s',
          bool(m_bw and m_kv) and 'FP8 weights (of the embedding table only one row per request)' in nt and 'always take longer than its arithmetic' in nt, nt)
    if m_bw and m_kv:
        bw_n = int(m_bw.group(1)) / 100 * float(m_bw.group(2)) * 1e12; ctx_n = int(m_kv.group(1)) * 1024; kv_n = int(m_kv.group(2)) * 1024
        check('tile notes: the KV cache per token they give is Mixtral’s (2 × 32 layers × 8 KV heads × 128 × 2 bytes)', kv_n == KVT)
        redo = {}
        for B in TZN:
            kvb = B * ctx_n * kv_n
            redo[B] = {'mx': (sread + L * expected_read(B) * expert + kvb) / bw_n, 'd47': (total - emb + kvb) / bw_n, 'd13': (active - emb + kvb) / bw_n}
        check('tile notes: with only what they say, every time on the tile comes out as shown (B-08)', all(ms(redo[B][k_]) == SHW[B][k_] for B in TZN for k_ in ('mx', 'd47', 'd13')),
              {B: {k_: (ms(redo[B][k_]), SHW[B][k_]) for k_ in ('mx', 'd47', 'd13')} for B in TZN})
        check('tile notes: reading the embedding rows (4 KiB per request in FP8) changes no time shown', all(ms(redo[B]['mx'] + B * d / bw_n) == SHW[B]['mx'] for B in TZN))
    nz = [x.replace('\u00a0', ' ') for x in ev("()=>{document.querySelector('.langsw button[data-lang=\"zh\"]').click(); const n=window.__tzMoe.notes(); document.querySelector('.langsw button[data-lang=\"en\"]').click(); return n}")]
    check('tile notes in Chinese: the same four, the formula as “8 ×（1 − 6/8 的 N 次方）”, BF16 and 128 KiB, no “^”', len(nz) == len(nts) and '8 ×（1 − 6/8 的 N 次方）' in ''.join(nz)
          and 'BF16' in ''.join(nz) and '128 KiB' in ''.join(nz) and '只查其中一行' in ''.join(nz) and not any('^' in x for x in nz), nz)
    txt_all = ev("()=>{const t=document.querySelector('#tile-moe'); return [t.innerText, ...[...t.querySelectorAll('[data-tipx]')].map(e=>e.getAttribute('data-tipx')), ...[...t.querySelectorAll('.tz-note')].map(e=>e.textContent), t.querySelector('.tz-moe-vis').getAttribute('aria-label')].join('\\n')}")
    check('tile: no memory size anywhere on the tile, its tooltips or its notes (the lab’s part 2 asks for it, R3.8)', not re.search(r'\d\s*(GiB|GB)\b|87\.0|43\.5', txt_all), re.findall(r'.{30}\d\s*(?:GiB|GB)\b.{10}', txt_all))
    check('no page errors', not errs, errs)
    b.close()

print(f'{n - len(fails)} / {n} checks passed')
sys.exit(1 if fails else 0)
