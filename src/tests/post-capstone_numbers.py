"""Numbers test for the post-capstone lab: every number the page shows, recomputed independently in Python and compared with
the page's engine (window.__pcap) and the teaser (window.__pcapTz) through Playwright.

Derived here from first principles, not ported from the JS:
- the Llama-3.1 parameter counts from the layer shapes (linear, embedding + LM head, (2L + 1) RMSNorm vectors), which must add up
  to the published 8,030,261,248 and 70,553,706,496; LoRA parameters r × (in + out) per matrix; NF4 with double quantization as
  4 bits + one 8-bit scale per 64 weights + one FP32 scale per 256 blocks;
- memory per GPU = (sum of the bytes of every model held) / GPUs + one 1,024-token sequence's activations and logits;
- GPU hours from FLOPs (6N / 4N per trained token, 2N per token read by a frozen model) at 50% of 989.5 TFLOP/s, and the writing
  time from the memory-bound step (64 answers per GPU, 2N bytes read per step at 3.35 TB/s);
- the metric rule m(x) = m_inf - (m_inf - m0) * 2^(-x/h) and the 95% error bar 1.96 * sqrt(p(1 - p)/n);
- which methods can run: a method is blocked only when the card's data cannot form its training examples. SFT's completion is a
  whole example answer when the card has one, else a final answer alone (8 tokens), else the chosen answer of a pair (TRL turns a
  preference dataset into prompt-completion data by dropping the rejected answer). SFT on final answers alone or on the chosen
  answers alone gets no gain (m_inf = m0, the lab's conservative rule, review codex-b9v3 item 8);
- brief S5's totals (321.2 / 176.7 / 160.6 GB), part 3's ratio (more than 10 times), part 4's fits, part 5's passes and margins,
  part 6's budgets, the challenge's minimal answer, the exits' answers, the seeded drill (mulberry32) and the teaser's rows;
- the owner's item 2 (2026-10-09): 70B QLoRA's memory blocks by what they hold (4.2 GB is the BF16 embedding table, LM head and norms,
  35.3 GB the 4-bit linear layers), the 0.73 GPU hours from 4.61M tokens at about 1,750 tokens/s per GPU, and the measured check
  (torchtune's 70B LoRA on 8 A100s: 3,497 tokens/s, about 40% of the peak, so about 0.92 GPU hours).
Run: python3 tests/post-capstone_numbers.py"""
import math, pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/post-capstone.html', '--only', 'post-capstone'], check=True, cwd=P, stdout=subprocess.DEVNULL)
OUT = P / 'out' / 'post-capstone'; OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'wrapped-numbers.html').write_text('<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0}</style></head><body>' + (P / 'out' / 'post-capstone.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-numbers.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', info)


close = lambda a, b, rel=1e-9: abs(a - b) <= rel * max(1.0, abs(a), abs(b))

# ---------------------------------------------------------------- shapes and parameters
V = 128256
M = {'8b': dict(N=8030261248, L=32, d=4096, H=32, KV=8, hd=128, f=14336),
     '70b': dict(N=70553706496, L=80, d=8192, H=64, KV=8, hd=128, f=28672)}
R16 = 16
for k, m in M.items():
    mats = [(m['d'], m['H'] * m['hd']), (m['d'], m['KV'] * m['hd']), (m['d'], m['KV'] * m['hd']), (m['H'] * m['hd'], m['d']),
            (m['d'], m['f']), (m['d'], m['f']), (m['f'], m['d'])]
    m['lin'] = m['L'] * sum(a * b for a, b in mats)
    m['eo'] = 2 * V * m['d']
    m['nrm'] = (2 * m['L'] + 1) * m['d']
    m['lp'] = m['L'] * R16 * sum(a + b for a, b in mats)
    check(f'{k}: linear + embedding + head + norms = N', m['lin'] + m['eo'] + m['nrm'] == m['N'], (m['lin'], m['eo'], m['nrm']))
check('8B LoRA parameters r = 16 on 7 matrices = 41,943,040', M['8b']['lp'] == 41943040, M['8b']['lp'])
check('70B LoRA parameters = 207,093,760', M['70b']['lp'] == 207093760, M['70b']['lp'])
ACT = {'8b': 167936, '70b': 331776}       # training-memory's bytes per token per layer (BF16, no recomputation)
NF4 = 4 / 8 + 1 / 64 + 4 / (64 * 256)
check('NF4 + double quantization = 0.515869140625 bytes per weight', NF4 == 0.515869140625, NF4)

# ---------------------------------------------------------------- engine, written from the formulas
PEAK, U, HBM, CAP, SEQ, GB_, G8, EP = 989.5e12, 0.5, 3.35e12, 80e9, 1024, 64, 8, 4
CARDS = {
    'fmt': dict(m='8b', gpus=1, budget=1, data={'demo'}, n=3000, p=128, r=384, m0=40, target=95, ne=200, rule={'sft': (98, .5)}),
    'pref': dict(m='8b', gpus=8, budget=3, data={'pairs'}, n=20000, p=128, r=256, m0=50, target=60, ne=500, rule={'sft': (50, .5), 'dpo': (70, .5), 'ppo': (70, .5), 'grpo': (70, .5)}),
    'math': dict(m='8b', gpus=16, budget=10, data={'demo', 'checker'}, n=7473, p=128, r=256, m0=39.7, target=70, ne=1319, rule={'sft': (60, .5), 'grpo': (79.2, 1)}),
    'big': dict(m='70b', gpus=1, budget=2, data={'demo'}, n=3000, p=128, r=384, m0=40, target=95, ne=200, rule={'sft': (98, .5)}),
    'ch': dict(m='8b', gpus=1, budget=3, data={'ans', 'checker'}, n=7473, p=128, r=256, a=8, m0=39.7, target=65, ne=1319, rule={'sft': (39.7, .5), 'grpo': (79.2, 1)}),
}
NEEDS = {'sft': [{'demo'}, {'ans'}, {'pairs'}], 'dpo': [{'pairs'}], 'ppo': [{'pairs'}], 'grpo': [{'checker'}, {'pairs'}]}


def sft_tokens(c):
    """tokens of one SFT example: the prompt plus its completion (a whole answer, else a final answer alone, else the chosen answer)"""
    if 'demo' in c['data']: return c['p'] + c['r']
    if 'ans' in c['data']: return c['p'] + c['a']
    return c['p'] + c['r']


def gen_tokens_per_s(m):
    step = max(2 * m['N'] / HBM, GB_ * 2 * m['N'] / (PEAK * U))   # one step: read the BF16 weights, or the FLOPs if slower
    return GB_ / step


def model_bytes(m, mode):
    """bytes of one trained copy (policy or value model)"""
    if mode == 'full': return 18 * m['N']
    if mode == 'lora': return 2 * m['N'] + 16 * m['lp']
    return NF4 * m['lin'] + 2 * (m['eo'] + m['nrm']) + 16 * m['lp']


def ref_plan(card, meth, mode, x, G=G8, gpus=None, rc=False, noref=False):
    """rc: activations recomputed in the backward pass (training-memory: 2·d bytes per token per layer kept);
    noref: GRPO with β = 0, no reference model (TRL's default since v0.19.0)"""
    c = CARDS[card]; m = M[c['m']]; N = m['N']; g = gpus or c['gpus']
    can = any(need <= c['data'] for need in NEEDS[meth])
    rm = meth == 'ppo' or (meth == 'grpo' and 'checker' not in c['data'])
    ref = not (noref and meth == 'grpo')
    total = model_bytes(m, mode)
    if meth in ('dpo', 'ppo', 'grpo') and mode == 'full' and ref: total += 2 * N   # the reference: a BF16 copy, or the base with the adapter off
    if meth == 'ppo': total += model_bytes(m, mode)                               # the value model
    if rm: total += 2 * N                                                         # the reward model, BF16
    mem = total / g + SEQ * m['L'] * (2 * m['d'] if rc else ACT[c['m']]) + SEQ * V * 4
    k = 6 if mode == 'full' else 4
    t = c['p'] + c['r']; fl = PEAK * U
    if meth == 'sft': per = k * N * sft_tokens(c) / fl
    elif meth == 'dpo': per = (k * N + 2 * N) * 2 * t / fl                       # chosen and rejected, policy trained, reference reads
    elif meth == 'ppo': per = c['r'] / gen_tokens_per_s(m) + (2 * EP * k * N + 2 * 2 * N) * t / fl
    else: per = G * c['r'] / gen_tokens_per_s(m) + (k * N + 2 * N * ((1 if ref else 0) + (1 if rm else 0))) * G * t / fl
    sec = per * c['n'] * x + (k * N * 2 * t * c['n'] / fl if rm else 0)
    gpuh = sec / 3600
    if can and meth in c['rule']:
        mi, h = c['rule'][meth]; met = mi - (mi - c['m0']) * 2 ** (-x / h)
    else: met = c['m0']
    band = 1.96 * math.sqrt(met / 100 * (1 - met / 100) / c['ne']) * 100
    fits, inb, hits = mem <= CAP, gpuh <= c['budget'] + 1e-9, met >= c['target'] - 1e-9
    return dict(can=can, mem=mem, modelBytes=total, gpuh=gpuh, metric=met, band=band, fits=fits, inBudget=inb, hits=hits,
                ok=can and fits and inb and hits, useRM=rm)


def mulberry32(a):
    st = [a & 0xffffffff]
    def imul(x, y): return ((x & 0xffffffff) * (y & 0xffffffff)) & 0xffffffff
    def nxt():
        st[0] = (st[0] + 0x6D2B79F5) & 0xffffffff; a_ = st[0]
        t = imul(a_ ^ (a_ >> 15), 1 | a_)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & 0xffffffff) ^ t
        return ((t ^ (t >> 14)) & 0xffffffff) / 4294967296
    return nxt


with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(); pg.goto(URL); pg.wait_for_timeout(300)
    H = lambda js, *a: pg.evaluate(js, *a)
    C = H("()=>window.__pcap.consts")
    # constants and shapes
    for k in ('8b', '70b'):
        for f in ('lin', 'eo', 'nrm', 'lp'):
            check(f'{k} {f}', C['PT'][k][f] == M[k][f], (C['PT'][k][f], M[k][f]))
        check(f'{k} activation bytes per token per layer', C['PT'][k]['aL'] == ACT[k], C['PT'][k]['aL'])
    check('NF4 constant', C['NF4'] == NF4)
    for k in ('PEAK', 'U0', 'HBM', 'CAP', 'SEQ', 'GEN_B', 'G_ANS', 'PPO_EP', 'RANK'):
        check(f'const {k}', C[k] == {'PEAK': PEAK, 'U0': U, 'HBM': HBM, 'CAP': CAP, 'SEQ': SEQ, 'GEN_B': GB_, 'G_ANS': G8, 'PPO_EP': EP, 'RANK': R16}[k])
    for cid, c in CARDS.items():
        pc_ = C['CARDS'][cid]
        same = pc_['gpus'] == c['gpus'] and pc_['budget'] == c['budget'] and pc_['n'] == c['n'] and pc_['tok'] == {'p': c['p'], 'r': c['r'], **({'a': c['a']} if 'a' in c else {})} \
            and pc_['m0'] == c['m0'] and pc_['target'] == c['target'] and pc_['nEval'] == c['ne'] and {k for k, v in pc_['data'].items() if v} == c['data']
        check(f'card {cid} matches', same, pc_)
    gr = gen_tokens_per_s(M['8b'])
    check('writing rate 8B ≈ 13,350 tokens/s per GPU', round(gr / 10) * 10 == 13350 and close(H("()=>window.__pcap.genRate('8b')"), gr), gr)
    check('one writing step of 8B = 4.79 ms (16.1 GB / 3.35 TB/s)', round(2 * M['8b']['N'] / HBM * 1e3, 2) == 4.79)

    # every card × method × mode × passes
    bad = []
    for cid in CARDS:
        for meth in ('sft', 'dpo', 'ppo', 'grpo'):
            for mode in ('full', 'lora', 'qlora'):
                for x in (0.5, 1, 2, 3, 4):
                    r = ref_plan(cid, meth, mode, x); q = H("s=>window.__pcap.plan(s)", dict(card=cid, meth=meth, mode=mode, x=x))
                    for f in ('mem', 'modelBytes', 'gpuh', 'metric', 'band'):
                        if not close(r[f], q[f], 1e-9): bad.append((cid, meth, mode, x, f, r[f], q[f]))
                    for f in ('can', 'fits', 'inBudget', 'hits', 'ok', 'useRM'):
                        if r[f] != q[f]: bad.append((cid, meth, mode, x, f, r[f], q[f]))
    check('plan(): 300 setups agree on memory, GPU hours, metric, error bar and verdicts', not bad, bad[:5])
    gb = lambda v: round(v / 1e9, 1)
    bad2, flips = [], []
    for cid in CARDS:
        for meth in ('sft', 'dpo', 'ppo', 'grpo'):
            for mode in ('full', 'lora', 'qlora'):
                for kw in ({'rc': True}, {'noRef': True}):
                    r = ref_plan(cid, meth, mode, 1, rc=kw.get('rc', False), noref=kw.get('noRef', False)); q = H("s=>window.__pcap.plan(s)", dict(card=cid, meth=meth, mode=mode, x=1, **kw))
                    if not (close(r['mem'], q['mem']) and close(r['modelBytes'], q['modelBytes']) and close(r['gpuh'], q['gpuh'])): bad2.append((cid, meth, mode, kw, r['mem'], q['mem']))
                if ref_plan(cid, meth, mode, 1)['fits'] != ref_plan(cid, meth, mode, 1, rc=True)['fits']: flips.append((cid, meth, mode))
    check('plan() with recomputed activations or without the reference model: 120 setups agree', not bad2, bad2[:3])
    check('fidelity card: recomputing the activations changes no fit / no-fit on any of the 60 setups', not flips, flips)
    qn, qr = ref_plan('big', 'sft', 'qlora', 1), ref_plan('big', 'sft', 'qlora', 1, rc=True)
    check('fidelity card and Connect: 70B QLoRA 70.5 GB without recomputation, 44.7 GB with it (QLoRA’s 65B script recomputes)', (gb(qn['mem']), gb(qr['mem'])) == (70.5, 44.7), (qn['mem'], qr['mem']))
    g_ref, g_no = ref_plan('math', 'grpo', 'full', 1), ref_plan('math', 'grpo', 'full', 1, noref=True)
    check('fidelity card (brief S5a): GRPO with a checker 160.6 GB, 144.5 GB with β = 0 (no reference model)', (gb(g_ref['modelBytes']), gb(g_no['modelBytes'])) == (160.6, 144.5))
    k0 = g_no['gpuh'] / ref_plan('math', 'sft', 'full', 1)['gpuh']
    check('part 3 tooltip: without the reference model GRPO costs about 12 times SFT, still more than 10', round(k0) == 12 and k0 > 10 and close(H("()=>window.__pcap.p3NoRef()"), k0), k0)
    kv = GB_ * (128 + 256) * 2 * M['8b']['L'] * M['8b']['KV'] * M['8b']['hd'] * 2
    check('Settings: the KV cache of 64 answers (128 + 256 tokens, 8B, BF16) is about 3.2 GB', round(kv / 1e9, 1) == 3.2 and close(H("()=>window.__pcap.kvGen('8b')"), kv), kv)
    check('Settings: one generating step of the 8B takes 4.8 ms', round(2 * M['8b']['N'] / HBM * 1e3, 1) == 4.8 and close(H("()=>window.__pcap.stepMs('8b')"), 2 * M['8b']['N'] / HBM * 1e3))

    # brief S5 totals (8B, full fine-tuning, all models)
    gb = lambda v: round(v / 1e9, 1)
    check('S5: reward model + PPO hold 321.2 GB', gb(ref_plan('pref', 'ppo', 'full', 1)['modelBytes']) == 321.2)
    check('S5: GRPO with a reward model holds 176.7 GB', gb(ref_plan('pref', 'grpo', 'full', 1)['modelBytes']) == 176.7)
    check('S5: GRPO with a checker holds 160.6 GB', gb(ref_plan('math', 'grpo', 'full', 1)['modelBytes']) == 160.6)
    check('training-memory: 18 bytes × 8B = 144.5 GB', gb(18 * M['8b']['N']) == 144.5)

    # part 3: GRPO vs SFT, one pass over the math card, full, 16 GPUs
    s1, g1 = ref_plan('math', 'sft', 'full', 1), ref_plan('math', 'grpo', 'full', 1)
    k = g1['gpuh'] / s1['gpuh']
    check('part 3: GRPO / SFT ≈ 14.8, more than 10 times (the right answer)', round(k, 1) == 14.8 and C['PRED_RIGHT'] == 'gt10' and close(H("()=>window.__pcap.p3Ratio()"), k), k)
    check('part 3: SFT 0.078 GPU h, GRPO 1.15 GPU h', (round(s1['gpuh'], 3), round(g1['gpuh'], 2)) == (0.078, 1.15), (s1['gpuh'], g1['gpuh']))
    tr = lambda m: H("s=>window.__pcap.plan(s).seg.train", dict(card='math', meth=m, mode='full', x=1))
    check('part 3: training 8 answers alone is 8 times SFT', close(tr('grpo') / tr('sft'), 8.0), tr('grpo') / tr('sft'))
    check('part 3 card: a frozen read costs a third of training a token (2N vs 6N); generating a token costs about as much as training it (0.5 to 1.5 times)',
          0.5 < (1 / gr) / (6 * M['8b']['N'] / (PEAK * U)) < 1.5, (1 / gr) / (6 * M['8b']['N'] / (PEAK * U)))
    check('part 7: GRPO costs “over ten times” (十几倍) SFT', 10 < k < 20, k)

    # part 4: one H100
    lv4 = {o['v']: o for o in C['LV4']}
    want = {'8full': (150.6, False), '8lora': (22.8, True), '70lora': (172.1, False), '70qlora': (70.5, True)}
    for v, (g_, f_) in want.items():
        r = ref_plan(lv4[v]['card'], lv4[v]['meth'], lv4[v]['mode'], lv4[v]['x'])
        check(f'part 4 {v}: {g_} GB, fits {f_}', (gb(r['mem']), r['fits']) == (g_, f_), (r['mem'], r['fits']))
    check('QLoRA 70B frozen 4-bit linear weights 35.3 GB + BF16 embedding/head/norms 4.2 GB = 39.5 GB',
          gb(NF4 * M['70b']['lin'] + 2 * (M['70b']['eo'] + M['70b']['nrm'])) == 39.5)
    # item 2 (the owner, 2026-10-09: “BF16 权重只有 4.2GB”, “训练3遍才0.73GPU小时”): each memory block of 70B QLoRA is what it holds,
    # and the 0.73 GPU hours come from the tokens, 4N per token and the assumed 50% of the peak
    m70, big3 = M['70b'], dict(card='big', meth='sft', mode='qlora', x=3)
    q70q = H("s=>window.__pcap.plan(s)", big3); blk = {k: v for k, v in q70q['blocks']}
    want_b = {'w': 2 * (m70['eo'] + m70['nrm']), 'q4': NF4 * m70['lin'], 'lr': 16 * m70['lp'], 'a': SEQ * m70['L'] * ACT['70b'], 'l': SEQ * V * 4}
    check('item 2: 70B QLoRA blocks: embedding table, LM head and norms (BF16) 4.2 GB, frozen linear layers (4-bit) 35.3, LoRA 3.3, activations 27.2, logits 0.5',
          set(blk) == set(want_b) and all(close(blk[k], v) for k, v in want_b.items()) and [gb(want_b[k]) for k in ('w', 'q4', 'lr', 'a', 'l')] == [4.2, 35.3, 3.3, 27.2, 0.5]
          and close(sum(blk.values()), q70q['mem']), blk)
    check('item 2: the embedding table and LM head that QLoRA keeps in BF16 are 3.0% of the 70B parameters (the hover says so)', round(m70['eo'] / m70['N'] * 100, 1) == 3.0, m70['eo'] / m70['N'])
    tok3 = CARDS['big']['n'] * sft_tokens(CARDS['big']) * 3; rate70 = PEAK * U / (4 * m70['N'])
    check('item 2: 3 passes × 3,000 examples × 512 tokens = 4.61M tokens; 4N = 282 GFLOP per token; about 1,750 tokens/s per GPU at 50% of 989.5 TFLOP/s; 0.73 GPU h',
          sft_tokens(CARDS['big']) == 512 and q70q['tok']['train'] == 512 and round(tok3 / 1e6, 2) == 4.61 and round(4 * m70['N'] / 1e9) == 282 and round(rate70, -1) == 1750
          and close(q70q['trainRate'], rate70) and round(tok3 / rate70 / 3600, 2) == 0.73 and close(q70q['gpuh'], tok3 / rate70 / 3600), (q70q['tok'], q70q['trainRate'], q70q['gpuh']))
    # the measured check: torchtune's README, Llama 3.1 70B with LoRA on 8 A100s, 3,497 tokens/s in all: 437 per GPU × 4N ÷ 312 TFLOP/s
    tt_u = 3497 / 8 * 4 * m70['N'] / 312e12; tt_h = q70q['gpuh'] * U / tt_u
    check('item 2: torchtune measured about 40% of the peak; at that speed the 70B QLoRA run takes about 0.92 GPU h, not 0.73 (realistic within 25%)',
          round(tt_u * 100) == 40 and round(tt_h, 2) == 0.92 and close(H("()=>window.__pcap.ttU()"), tt_u) and close(H("s=>window.__pcap.ttGpuh(s)", big3), tt_h) and tt_h / q70q['gpuh'] < 1.3, (tt_u, tt_h))
    g6b = {k: v for k, v in H("s=>window.__pcap.plan(s)", dict(card='pref', meth='grpo', mode='full', x=1))['blocks']}
    check('item 2: part 6 GRPO full on 8 GPUs: the reference and reward models (BF16) are their own block, 2 × 2N ÷ 8 = 4.0 GB, apart from the policy (2N ÷ 8 = 2.0 GB)',
          close(g6b['fz'], 2 * 2 * M['8b']['N'] / 8) and close(g6b['w'], 2 * M['8b']['N'] / 8) and gb(g6b['fz']) == 4.0, g6b)

    # part 5: format check by passes, SFT with LoRA on the support card
    exp = {0.5: 69.0, 1: 83.5, 2: 94.4, 3: 97.1, 4: 97.8}
    for x, v in exp.items():
        r = ref_plan('fmt', 'sft', 'lora', x)
        check(f'part 5: {x} passes → {v}%', round(r['metric'], 1) == v and r['hits'] == (x >= 3), r['metric'])
    m3, m4, m2 = ref_plan('fmt', 'sft', 'lora', 3), ref_plan('fmt', 'sft', 'lora', 4), ref_plan('fmt', 'sft', 'lora', 2)
    check('part 5: 2 passes 94.4% is inside its error bar of 95%', m2['metric'] + m2['band'] >= 95 > m2['metric'])
    check('part 5: 3 passes, the bottom of the error bar (94.8%) is below 95%', round(m3['metric'] - m3['band'], 1) == 94.8)
    check('part 5: 4 passes, the bottom of the error bar (95.7%) is above 95%', round(m4['metric'] - m4['band'], 1) == 95.7)
    check('part 5: error bar at 3 passes ±2.3 points', round(m3['band'], 1) == 2.3)

    # part 6: the preference card, 8 GPUs, 3 GPU hours
    d6, p6, g6 = ref_plan('pref', 'dpo', 'full', 1), ref_plan('pref', 'ppo', 'full', 1), ref_plan('pref', 'grpo', 'full', 1)
    check('part 6: DPO 0.55 GPU h within budget', round(d6['gpuh'], 2) == 0.55 and d6['inBudget'], d6['gpuh'])
    check('part 6: reward model + PPO 2.32 GPU h within budget, 4.2 times DPO', round(p6['gpuh'], 2) == 2.32 and p6['inBudget'] and round(p6['gpuh'] / d6['gpuh'], 1) == 4.2, p6['gpuh'])
    check('part 6: GRPO with a reward model 4.04 GPU h, over budget, about 7 times DPO', round(g6['gpuh'], 2) == 4.04 and not g6['inBudget'] and round(g6['gpuh'] / d6['gpuh']) == 7, g6['gpuh'])
    check('part 6: all three reach a 65.0% win rate after one pass', all(round(r['metric'], 1) == 65.0 for r in (d6, p6, g6)))
    check('part 6: error bar of 500 judged prompts at 65% ±4.2 points', round(d6['band'], 1) == 4.2)
    check('scenario 3: reward model + PPO costs about 4 times DPO', round(p6['gpuh'] / d6['gpuh']) == 4)
    f8, l8, q70 = ref_plan('fmt', 'sft', 'full', 3), ref_plan('fmt', 'sft', 'lora', 3), ref_plan('big', 'sft', 'lora', 3)
    check('part 7: 8B full does not fit one H100, LoRA does; 70B does not fit even with LoRA', not f8['fits'] and l8['fits'] and not q70['fits'])

    # scenarios
    pres = {o['id']: o for o in C['PRESETS']}
    for pid, (memv, ghv, met) in {'fmt': (22.8, 0.11, 97.8), 'dpo': (26.1, 0.55, 65.0), 'ppo': (46.2, 2.32, 65.0), 'grpo': (16.1, 3.44, 74.3), 'big': (70.5, 0.97, 97.8)}.items():
        o = pres[pid]; r = ref_plan(o['card'], o['meth'], o['mode'], o['x'])
        ghs = round(r['gpuh'], 3) if r['gpuh'] < 0.1 else round(r['gpuh'], 2)      # the page's rounding
        check(f'scenario {pid}: {memv} GB, {ghv} GPU h, {met}%', (gb(r['mem']), ghs, round(r['metric'], 1)) == (memv, ghv, met) and r['ok'], (r['mem'], r['gpuh'], r['metric']))
    check('scenarios: every preset meets its target with the whole error bar above it',
          all(ref_plan(o['card'], o['meth'], o['mode'], o['x'])['metric'] - ref_plan(o['card'], o['meth'], o['mode'], o['x'])['band'] >= CARDS[o['card']]['target'] for o in C['PRESETS']))

    # challenge
    allr = [(meth, mode, x, ref_plan('ch', meth, mode, x)) for meth in ('sft', 'dpo', 'ppo', 'grpo') for mode in ('full', 'lora', 'qlora') for x in (0.5, 1, 2, 3, 4)]
    oks = sorted([(r['gpuh'], meth, mode, x) for meth, mode, x, r in allr if r['ok']])
    best = oks[0][0]
    mins = {(m_, o_, x_) for g_, m_, o_, x_ in oks if g_ <= best * 1.01}
    pa = H("()=>window.__pcap.chAll()")
    check('challenge: the minimal answers are GRPO with LoRA or QLoRA, 2 passes', mins == {('grpo', 'lora', 2), ('grpo', 'qlora', 2)}, mins)
    check('challenge: page agrees on every option', all((e['ok'], e['min']) == (ref_plan('ch', e['meth'], e['mode'], e['x'])['ok'], (e['meth'], e['mode'], e['x']) in mins) for e in pa))
    st = ref_plan('ch', 'sft', 'full', 1); fl2 = ref_plan('ch', 'grpo', 'full', 2); l1 = ref_plan('ch', 'grpo', 'lora', 1); l2 = ref_plan('ch', 'grpo', 'lora', 2); l4 = ref_plan('ch', 'grpo', 'lora', 4)
    # review codex-b9v3 item 8: final answers alone are SFT data (prompt = the problem, completion = the answer); the four checks are separate
    check('challenge: the initial setup (SFT, full) can run on the final answers, but needs 150.6 GB (does not fit) and stays at 39.7%',
          st['can'] and gb(st['mem']) == 150.6 and not st['fits'] and st['inBudget'] and round(st['metric'], 1) == 39.7 and not st['hits'] and not st['ok'], st)
    sl = ref_plan('ch', 'sft', 'lora', 1)
    sl_h = 4 * M['8b']['N'] * (128 + 8) * 7473 / (PEAK * U) / 3600                     # 4N per trained token, 136 tokens per problem
    check('challenge: SFT with LoRA passes data, memory (22.8 GB) and budget (0.018 GPU h) but not the target (39.7%, as before training)',
          sl['can'] and gb(sl['mem']) == 22.8 and sl['fits'] and close(sl['gpuh'], sl_h) and round(sl['gpuh'], 3) == 0.018 and sl['inBudget'] and sl['metric'] == 39.7 and not sl['hits'], (sl['gpuh'], sl['metric']))
    sfts = [r for meth, mode, x, r in allr if meth == 'sft']
    check('challenge: every SFT option can run and none reaches 65% (no gain at any pass, m∞ = m₀)', all(r['can'] and r['metric'] == 39.7 and not r['ok'] for r in sfts) and len(sfts) == 15)
    check('challenge: DPO and reward model + PPO cannot form their examples (no pairs)', not any(r['can'] for meth, mode, x, r in allr if meth in ('dpo', 'ppo')))
    ps = [ref_plan('pref', 'sft', mode, x) for mode in ('full', 'lora', 'qlora') for x in (0.5, 1, 2, 3, 4)]
    pf = ref_plan('pref', 'sft', 'full', 1)
    check('preference card: SFT on the chosen answers can run (24.1 GB per GPU on 8, 0.21 GPU h) and stays at a 50.0% win rate, below 60%',
          all(r['can'] and r['metric'] == 50 and not r['hits'] for r in ps) and gb(pf['mem']) == 24.1 and round(pf['gpuh'], 2) == 0.21 and pf['inBudget'], (pf['mem'], pf['gpuh']))
    check('the other blocks stand: on the support and 70B cards only SFT can run; on the math card DPO and PPO cannot',
          all(not ref_plan(cid, m_, 'lora', 1)['can'] for cid in ('fmt', 'big') for m_ in ('dpo', 'ppo', 'grpo')) and not ref_plan('math', 'dpo', 'lora', 1)['can'] and not ref_plan('math', 'ppo', 'lora', 1)['can'])
    q = H("s=>window.__pcap.plan(s)", dict(card='ch', meth='sft', mode='lora', x=1))
    check('page: SFT on the challenge is marked as trained on final answers and flat; GRPO is not flat', q['from'] == 'ans' and q['flat'] and not H("s=>window.__pcap.plan(s)", dict(card='ch', meth='grpo', mode='lora', x=2))['flat'] and H("s=>window.__pcap.plan(s)", dict(card='pref', meth='sft', mode='full', x=1))['from'] == 'chosen')
    check('challenge: full GRPO needs 166.6 GB on one GPU', gb(fl2['mem']) == 166.6 and not fl2['fits'])
    check('challenge: LoRA 1 pass 59.5%, 2 passes 69.3% for 1.88 GPU h, 4 passes over budget',
          (round(l1['metric'], 1), round(l2['metric'], 1), round(l2['gpuh'], 2), l4['inBudget']) == (59.5, 69.3, 1.88, False), (l1['metric'], l2['metric'], l2['gpuh']))
    check('challenge: the error bar at 69.3% on 1,319 problems ±2.5 points', round(l2['band'], 1) == 2.5)
    check('challenge feedback: at 2 passes the whole error bar is above 65% (the margin is enough)', l2['metric'] - l2['band'] >= 65, l2['metric'] - l2['band'])
    check('challenge: QLoRA 12.4 GB, LoRA 22.8 GB on one GPU', (gb(ref_plan('ch', 'grpo', 'qlora', 2)['mem']), gb(l2['mem'])) == (12.4, 22.8))

    # exits
    ex = H("()=>window.__pcap.exits()")
    one_full = ref_plan('pref', 'dpo', 'full', 1, gpus=1); one_lora = ref_plan('pref', 'dpo', 'lora', 1, gpus=1)
    check('exit 1: DPO on one H100, full 166.6 GB does not fit, LoRA 22.8 GB fits (answer 2)',
          ex['transferA'][0] == 1 and gb(one_full['mem']) == 166.6 and not one_full['fits'] and gb(one_lora['mem']) == 22.8 and '166.6 GB' in ex['transferWhy'][0] and '22.8 GB' in ex['transferWhy'][0], ex['transferWhy'][0])
    d8 = ref_plan('pref', 'dpo', 'full', 1)
    check('exit 1: the reference given (8 GPUs, 26.1 GB each) and the models alone on one GPU (160.6 GB)', gb(d8['mem']) == 26.1 and '26.1 GB' in ex['transferQ'][0] and gb(one_full['modelBytes']) == 160.6 and '160.6 GB' in ex['transferWhy'][0], (ex['transferQ'][0], ex['transferWhy'][0]))
    r8, r16 = ref_plan('math', 'grpo', 'full', 1), ref_plan('math', 'grpo', 'full', 1, G=16)
    check('exit 2: 16 answers cost exactly twice 8 (answer 2)', close(r16['gpuh'] / r8['gpuh'], 2.0) and ex['transferA'][1] == 1 and '1.15' in ex['transferQ'][1], r16['gpuh'] / r8['gpuh'])
    check('exit 2: the options are multiples, no exact values (R3.19): the right one says 2 times', ' 2 ' in ex['transferO'][1][1] and not any(ch.isdigit() and '.' in o for o in ex['transferO'][1] for ch in o), ex['transferO'][1])

    # drill, seeded
    dbad = []
    for seed in range(1, 31):
        rnd = mulberry32(seed * 7919 + 3); pk = lambda a: a[int(rnd() * len(a))]
        meth, mode, gpus, x = pk(['dpo', 'grpo']), pk(['full', 'lora']), pk([1, 2, 4, 8]), pk([1, 2])
        card = 'pref' if meth == 'dpo' else 'math'
        r = ref_plan(card, meth, mode, x, gpus=gpus); d = H("s=>window.__pcap.drill(s)", seed)
        if (d['meth'], d['mode'], d['gpus'], d['x']) != (meth, mode, gpus, x) or not close(d['mem'], r['mem']) or not close(d['gpuh'], r['gpuh']) or d['fits'] != r['fits']:
            dbad.append((seed, d, meth, mode, gpus, x))
    check('drill: 30 seeds agree (mulberry32, same seed rule)', not dbad, dbad[:2])

    # teaser
    tz = lambda d, m: H("([d,m])=>window.__pcapTz.cost(d,m)", [d, m])
    card_of = {'demo': 'fmt', 'pairs': 'pref', 'checker': 'ch'}
    tbad = []
    for d_, cid in card_of.items():
        for meth in ('sft', 'dpo', 'ppo', 'grpo'):
            r = ref_plan(cid, meth, 'full', 1); v = tz(d_, meth)
            if (v is None) == r['can'] or (v is not None and not close(v, r['gpuh'])): tbad.append((d_, meth, v, r['gpuh'], r['can']))
    check('teaser: every row matches the lab (one pass, full fine-tuning)', not tbad, tbad)
    check('teaser: SFT on 3,000 conversations 0.042 GPU h; DPO 0.55, RM + PPO 2.32, GRPO 4.04; GRPO with a checker 1.15',
          (round(tz('demo', 'sft'), 3), round(tz('pairs', 'dpo'), 2), round(tz('pairs', 'ppo'), 2), round(tz('pairs', 'grpo'), 2), round(tz('checker', 'grpo'), 2)) == (0.042, 0.55, 2.32, 4.04, 1.15))
    check('teaser: SFT runs on all three kinds of data: 0.208 GPU h on the chosen answers, 0.027 on the final answers alone',
          (round(tz('pairs', 'sft'), 3), round(tz('checker', 'sft'), 3)) == (0.208, 0.027) and close(tz('checker', 'sft'), 6 * M['8b']['N'] * 136 * 7473 / (PEAK * U) / 3600), (tz('pairs', 'sft'), tz('checker', 'sft')))
    check('teaser: everything fits its 4.5-hour scale', max(v for v in (tz('pairs', 'grpo'), tz('pairs', 'ppo'))) < 4.5)
    check('teaser: same writing rate as the lab', close(H("()=>window.__pcapTz.genRate"), gr))
    b.close()

print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
