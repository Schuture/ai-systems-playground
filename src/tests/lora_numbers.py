"""Golden numbers for the lora lab (guide R9.2, CONTRACT §6). Analytical lab: no toy, so nothing to replay with a seed except the drill.

Derived here independently of the JS engine:
- the parameter counts of Llama-3.1-8B / 70B / 405B summed tensor by tensor from the Hugging Face config values (hidden size,
  intermediate size, heads, KV heads, layers, vocabulary; untied output), which must give 8,030,261,248 / 70,553,706,496 /
  405,853,388,800 (brief S1);
- a LoRA on an nn.Linear(in, out) adds r·in (lora_A) + out·r (lora_B) parameters (PEFT layer.py), summed over the chosen projections;
- bytes: full fine-tuning 18 per parameter; LoRA 2 per frozen parameter + 16 per LoRA parameter (FP32 weight, FP32 gradient, m, v;
  PEFT autocast_adapter_dtype=True); QLoRA: NF4 4 bits + an FP32 absmax per 64 weights compressed to 1 byte + an FP32 constant per
  256 absmax values (bitsandbytes quantize_4bit / compress_statistics), on every nn.Linear of the blocks, the rest in BF16;
- activations as tests/training-memory_numbers.py: the tensors one Llama layer keeps, listed by name, 2 bytes each; recomputation
  keeps the layer input (2d bytes per token); logits tokens × vocabulary × 4; the peak is the sum at the end of the forward pass;
- the challenge by brute force over every plan; the capability layer's answers; the drill; the home tile; the Connect card's code
  (its lora_params part, run here without torch).
Run: python3 tests/lora_numbers.py"""
import pathlib, subprocess, sys, io, contextlib, itertools
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/lora.html', '--only', 'lora'], check=True, cwd=P, stdout=subprocess.DEVNULL)
OUT = P / 'out' / 'lora'; OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'wrapped-num.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'lora.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-num.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


def close(a, b, rel=1e-12):
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


g1 = lambda b: f'{b / 1e9:,.1f} GB'

# ---------- configs (Hugging Face config.json values of meta-llama/Llama-3.1-*)
CFG = {'8b': dict(hidden=4096, inter=14336, heads=32, kv=8, layers=32, vocab=128256),
       '70b': dict(hidden=8192, inter=28672, heads=64, kv=8, layers=80, vocab=128256),
       '405b': dict(hidden=16384, inter=53248, heads=128, kv=8, layers=126, vocab=128256)}
WANT_P = {'8b': 8030261248, '70b': 70553706496, '405b': 405853388800}


def projections(c):
    hd = c['hidden'] // c['heads']
    q, kv, d, f = c['heads'] * hd, c['kv'] * hd, c['hidden'], c['inter']
    # name: (in_features, out_features) as nn.Linear(in, out)
    return {'q_proj': (d, q), 'k_proj': (d, kv), 'v_proj': (d, kv), 'o_proj': (q, d), 'gate_proj': (d, f), 'up_proj': (d, f), 'down_proj': (f, d)}


def params(c):
    pr = projections(c)
    per_layer = sum(i * o for i, o in pr.values()) + 2 * c['hidden']          # 7 matrices + input/post-attention RMSNorm
    return c['vocab'] * c['hidden'] * 2 + c['layers'] * per_layer + c['hidden']  # embed + lm_head (untied) + final norm


for k, c in CFG.items():
    check(f'parameter count {k}', params(c) == WANT_P[k], (params(c), WANT_P[k]))

TARGETS = {'qv': ['q_proj', 'v_proj'], 'attn': ['q_proj', 'k_proj', 'v_proj', 'o_proj'], 'all': list(projections(CFG['8b']).keys())}


def lora_count(k, r, tg):
    pr = projections(CFG[k])
    return CFG[k]['layers'] * sum(r * pr[m][0] + pr[m][1] * r for m in TARGETS[tg])


def linear_count(k):
    pr = projections(CFG[k]); return CFG[k]['layers'] * sum(i * o for i, o in pr.values())


NF4_BYTES = (4 / 8) + (1 / 64) + (4 / (64 * 256))   # 4-bit code; uint8 absmax per 64; FP32 offset/scale per 256 absmax values
check('NF4 bytes per parameter 0.515869140625', NF4_BYTES == 0.515869140625, NF4_BYTES)


def layer_kept_bytes(c):
    """Tensors one Llama layer keeps for the backward pass, per token, BF16 (as tests/training-memory_numbers.py lists them)."""
    d, f, hd = c['hidden'], c['inter'], c['hidden'] // c['heads']
    q, kv = c['heads'] * hd, c['kv'] * hd
    kept = {'attn_norm_in': d, 'attn_norm_out': d, 'q': q, 'k': kv, 'v': kv, 'attn_out': q, 'mlp_norm_in': d, 'mlp_norm_out': d,
            'gate': f, 'up': f, 'act(gate)': f, 'act*up': f}
    return 2 * sum(kept.values())


check('8B layer keeps 167,936 bytes per token (training-memory)', layer_kept_bytes(CFG['8b']) == 167936, layer_kept_bytes(CFG['8b']))
check('70B layer keeps 331,776 bytes per token', layer_kept_bytes(CFG['70b']) == 331776, layer_kept_bytes(CFG['70b']))
CAP = {'g24': 24e9, 'g48': 48e9, 'g80': 80e9}


def peak(k, meth, r, tg, seq, rc):
    c = CFG[k]; N = WANT_P[k]
    lp = 0 if meth == 'full' else lora_count(k, r, tg)
    if meth == 'full': states = 18 * N
    elif meth == 'lora': states = 2 * N + 16 * lp
    else: states = linear_count(k) * NF4_BYTES + 2 * (N - linear_count(k)) + 16 * lp
    acts = seq * c['layers'] * (2 * c['hidden'] if rc else layer_kept_bytes(c))
    logits = seq * c['vocab'] * 4
    return dict(tot=states + acts + logits, lp=lp, acts=acts, logits=logits, states=states, trained=N if meth == 'full' else lp)


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.evaluate("()=>{location.hash='lora'}"); pg.wait_for_timeout(500)
    ev = lambda js, *a: pg.evaluate(js, *a)
    C = ev("()=>window.__lora.consts")
    for k in ('8b', '70b', '405b'):
        m = C['MOD'][k]; c = CFG[k]
        check(f'shapes {k} match the config', (m['P'], m['L'], m['d'], m['H'], m['KV'], m['f'], m['V']) == (WANT_P[k], c['layers'], c['hidden'], c['heads'], c['kv'], c['inter'], c['vocab']), m)
    check('NF4 constant on the page', C['NF4'] == NF4_BYTES and close(C['NF4_NODQ'], 0.5625), (C['NF4'], C['NF4_NODQ']))
    # ---------- the engine over a grid of states
    bad = []
    for k, meth, r, tg, seq, rc in itertools.product(('8b', '70b'), ('full', 'lora', 'qlora'), C['RANKS'], ('qv', 'attn', 'all'), C['SEQS'], (0, 1)):
        js = ev("o=>window.__lora.compute(o)", dict(m=k, meth=meth, r=r, tg=tg, seq=seq, rc=rc, gpu='g80'))
        py = peak(k, meth, r, tg, seq, rc)
        if not (close(js['tot'], py['tot'], 1e-12) and js['trained'] == py['trained'] and close(js['acts'], py['acts']) and close(js['logits'], py['logits'])):
            bad.append((k, meth, r, tg, seq, rc, js['tot'], py['tot']))
    check('engine: every state of the grid (2 models × 3 methods × 7 ranks × 3 targets × 6 lengths × 2) matches', not bad, bad[:3])
    for k in ('8b', '70b', '405b'):
        for r in C['RANKS']:
            for tg in ('qv', 'attn', 'all'):
                if ev("a=>window.__lora.loraOf(a[0],a[1],a[2])", [k, r, tg]) != lora_count(k, r, tg): bad.append((k, r, tg))
    check('LoRA parameter counts for every rank and target, 8B / 70B / 405B', not bad, bad[:3])
    # ---------- the numbers the page states (guided parts)
    N8 = WANT_P['8b']
    check('part 1: 18 × 8B = 144.5 GB', g1(18 * N8) == '144.5 GB')
    q = ev("()=>window.__lora.matInfo('8b','q',16)"); kk = ev("()=>window.__lora.matInfo('8b','k',16)"); dn = ev("()=>window.__lora.matInfo('8b','down',16)")
    check('part 2: q: W 16,777,216; B and A 131,072, under 1%', (q['W'], q['BA']) == (4096 * 4096, 16 * (4096 + 4096)) and q['BA'] / q['W'] < 0.01, q)
    check('part 2: k and v about 2% (81,920 / 4,194,304 = 1.95%)', kk['BA'] == 16 * (4096 + 1024) and round(kk['BA'] / kk['W'] * 100) == 2, kk)
    check('part 2: down 0.50%', f"{dn['BA'] / dn['W'] * 100:.2f}" == '0.50', dn)
    # the owner's review of 2026-10-08, item 1: the picture labels W d_out × d_in, B d_out × r, A r × d_in; the shapes come from nn.Linear(in, out)
    bad = []
    for k in ('8b', '70b'):
        for name, (i, o) in projections(CFG[k]).items():
            for r in (8, 16, 64):
                mi = ev("a=>window.__lora.matInfo(a[0],a[1],a[2])", [k, name.split('_')[0], r])
                if (mi['din'], mi['dout'], mi['W'], mi['BA']) != (i, o, o * i, o * r + r * i): bad.append((k, name, r, mi))
    check('item 1: every matrix’s shape: W is d_out × d_in, B·A has d_out × r + r × d_in parameters (8B and 70B, r = 8 / 16 / 64)', not bad, bad[:3])
    # item 2: compute per token in units of N. Forward 2N; the backward pass computes the input gradient (2N) and, for weights that are
    # trained, the weight gradient (2N); recomputation runs the forward pass again (2N). LoRA and QLoRA train no frozen weight.
    want_fl = {('full', 0): 2 + 2 + 2, ('full', 1): 2 + 2 + 2 + 2, ('lora', 0): 2 + 2, ('lora', 1): 2 + 2 + 2, ('qlora', 0): 2 + 2, ('qlora', 1): 2 + 2 + 2}
    got_fl = {(me, rc): ev("o=>window.__lora.compute(o).fl", dict(m='8b', meth=me, r=16, tg='all', seq=4096, rc=rc, gpu='g80')) for me, rc in want_fl}
    check('item 2: compute per token: full 6N, 8N with recomputation; LoRA and QLoRA 4N, 6N with recomputation', got_fl == want_fl, got_fl)
    l16 = lora_count('8b', 16, 'all')
    check('part 3: rank 16 on all 7: 41,943,040 = 0.52% (the answer “under 2%”)', l16 == 41943040 and f'{l16 / N8 * 100:.2f}' == '0.52' and C['PRED_RIGHT'] == 'lt2', l16)
    full8, lora8 = peak('8b', 'full', 16, 'all', 4096, 0), peak('8b', 'lora', 16, 'all', 4096, 0)
    check('part 4: full 168.7 GB → LoRA 40.8 GB; gradients + optimizer state 128.5 GB', (g1(full8['tot']), g1(lora8['tot']), g1(16 * N8)) == ('168.7 GB', '40.8 GB', '128.5 GB'), (full8['tot'], lora8['tot']))
    check('part 4: LoRA is about a quarter of full', 0.2 < lora8['tot'] / full8['tot'] < 0.3, lora8['tot'] / full8['tot'])
    r64 = lora_count('8b', 64, 'all')
    check('part 5: r = 64: 167,772,160 (168M), 2.7 GB; on q and v 27,262,976 (27.3M)', (r64, g1(16 * r64), lora_count('8b', 64, 'qv')) == (167772160, '2.7 GB', 27262976))
    l16k = peak('8b', 'lora', 16, 'all', 16384, 0); l16r = peak('8b', 'lora', 16, 'all', 16384, 1)
    check('part 6: 16,384 tokens: 113.2 GB (activations 88.0 GB), recomputed 29.4 GB', (g1(l16k['tot']), g1(l16k['acts']), g1(l16r['tot'])) == ('113.2 GB', '88.0 GB', '29.4 GB'), (l16k, l16r))
    check('part 6: 8,192 tokens without recomputation still fits (65.0 GB)', g1(peak('8b', 'lora', 16, 'all', 8192, 0)['tot']) == '65.0 GB')
    l70, q70, q70n = peak('70b', 'lora', 16, 'all', 4096, 1), peak('70b', 'qlora', 16, 'all', 4096, 1), peak('70b', 'qlora', 16, 'all', 4096, 0)
    check('part 7: 70B LoRA 151.9 GB (frozen 141.1 GB), QLoRA 50.3 GB, without recomputation 153.6 GB (activations 108.7 GB)',
          (g1(l70['tot']), g1(2 * WANT_P['70b']), g1(q70['tot']), g1(q70n['tot']), g1(q70n['acts'])) == ('151.9 GB', '141.1 GB', '50.3 GB', '153.6 GB', '108.7 GB'))
    w4 = linear_count('70b') * NF4_BYTES
    check('part 7: 4-bit weights about a quarter of BF16; about 0.52 bytes', 0.25 < w4 / (2 * linear_count('70b')) < 0.27 and f'{NF4_BYTES:.2f}' == '0.52', w4)
    check('part 8: frozen BF16 weights 16.1 GB', g1(2 * N8) == '16.1 GB')
    # ---------- scenarios
    q8 = peak('8b', 'qlora', 16, 'all', 4096, 1)
    check('scenario l8: 40.8 GB, 39.2 GB to spare, activations 22.0 GB', (g1(lora8['tot']), g1(80e9 - lora8['tot']), g1(lora8['acts'])) == ('40.8 GB', '39.2 GB', '22.0 GB'))
    check('scenario q8: 9.5 GB on 24 GB; 4-bit 3.6 GB; BF16 embedding + head 2.1 GB', (g1(q8['tot']), g1(linear_count('8b') * NF4_BYTES), g1(2 * (N8 - linear_count('8b')))) == ('9.5 GB', '3.6 GB', '2.1 GB'))
    ms = 18 * WANT_P['70b']
    check('scenario f70: 1,270.0 GB of model states, at least 16 H100s', g1(ms) == '1,270.0 GB' and -(-ms // 80e9) == 16, ms)
    pres = {p_['id']: p_ for p_ in C['PRESETS']}
    check('scenarios: the four setups', (pres['l8']['meth'], pres['q8']['gpu'], pres['q70']['m'], pres['f70']['meth']) == ('lora', 'g24', '70b', 'full'), pres)
    # ---------- the challenge, by brute force
    plans = []
    for meth in ('full', 'lora', 'qlora'):
        for rc in (0, 1):
            for r, tg in ([(16, 'all')] if meth == 'full' else itertools.product(C['RANKS'], ('qv', 'attn', 'all'))):
                pk = peak('70b', meth, r, tg, 2048, rc)
                plans.append(dict(meth=meth, r=r, tg=tg, rc=rc, tot=pk['tot'], ok=pk['tot'] <= 48e9 and pk['trained'] >= 1e8, trained=pk['trained']))
    good = sorted([x for x in plans if x['ok']], key=lambda x: x['tot'])
    best = good[0]
    check('challenge: the unique best is QLoRA, recompute, all 7, r = 8: 44.9 GB, 103,546,880 (104M)', (best['meth'], best['rc'], best['tg'], best['r'], g1(best['tot']), best['trained']) == ('qlora', 1, 'all', 8, '44.9 GB', 103546880) and good[1]['tot'] - best['tot'] > 1e8, good[:3])
    jb = ev("()=>window.__lora.chBest()")
    check('challenge: the page finds the same best', (jb['meth'], jb['r'], jb['tg'], jb['rc']) == (best['meth'], best['r'], best['tg'], best['rc']) and close(jb['tot'], best['tot']), jb)
    check('challenge: the page evaluates the same plans', len(ev("()=>window.__lora.chAll()")) == len(plans) and sum(x['ok'] for x in ev("()=>window.__lora.chAll()")) == len(good), (len(plans), len(good)))
    check('challenge tier 2 example: QLoRA r = 16 all 7 recompute 46.6 GB', g1(peak('70b', 'qlora', 16, 'all', 2048, 1)['tot']) == '46.6 GB')
    st = peak('70b', 'lora', 8, 'qv', 2048, 0)
    check('challenge: the initial setup needs 196.8 GB and trains 16,384,000 (16.4M)', (g1(st['tot']), st['trained']) == ('196.8 GB', 16384000), (st['tot'], st['trained']))
    rows = ev("()=>window.__lora.chPass()")
    check('challenge list (the owner, batch 8): only the 6 plans that pass, least memory first (ties in either order), the same as by brute force', len(rows) == 6 == len(good) and all(r_['ok'] for r_ in rows)
          and {(r_['meth'], r_['r'], r_['tg'], r_['rc']) for r_ in rows} == {(x['meth'], x['r'], x['tg'], x['rc']) for x in good} and all(rows[i]['tot'] <= rows[i + 1]['tot'] for i in range(5)), [(r_['meth'], r_['r'], r_['tg'], r_['rc'], r_['ok']) for r_ in rows])
    check('challenge chart: q and v need r = 64, the 4 attention r = 32, both 131,072,000', lora_count('70b', 64, 'qv') == lora_count('70b', 32, 'attn') == 131072000 and lora_count('70b', 32, 'qv') < 1e8 and lora_count('70b', 16, 'attn') < 1e8)
    lo_all = peak('70b', 'lora', 8, 'all', 2048, 1); q_norc = peak('70b', 'qlora', 8, 'all', 2048, 0)
    jl, jq = ev("()=>window.__lora.chEval('lora',8,'all',1)"), ev("()=>window.__lora.chEval('qlora',8,'all',0)")
    check('challenge why (on hover): LoRA all 7 with recomputation 146.5 GB, QLoRA without recomputation 96.6 GB', close(jl['tot'], lo_all['tot']) and close(jq['tot'], q_norc['tot']) and (g1(lo_all['tot']), g1(q_norc['tot'])) == ('146.5 GB', '96.6 GB'), (g1(lo_all['tot']), g1(q_norc['tot'])))
    # ---------- the summary after Submit (review codex-b9v3 应修 5): three blocks, each with the setting that sets it, recomputed here
    def blocks(meth, r, tg, rc):
        N = WANT_P['70b']; pk = peak('70b', meth, r, tg, 2048, rc)
        w = 2 * N if meth != 'qlora' else linear_count('70b') * NF4_BYTES + 2 * (N - linear_count('70b'))
        t = 16 * N if meth == 'full' else 16 * pk['lp']
        return dict(w=w, t=t, a=pk['acts'] + pk['logits'], tot=pk['tot'])
    bad = []
    for x in plans:
        mine, js = blocks(x['meth'], x['r'], x['tg'], x['rc']), ev("e=>window.__lora.chParts(e.meth,e.r,e.tg,e.rc)", dict(meth=x['meth'], r=x['r'], tg=x['tg'], rc=x['rc']))
        if not all(close(js[k], mine[k], 1e-9) for k in 'wta') or not close(mine['w'] + mine['t'] + mine['a'], mine['tot'], 1e-9): bad.append((x, js, mine))
    check(f'challenge summary: the three blocks of all {len(plans)} plans match and add up to the total', not bad, bad[:2])
    s0, sb = blocks('lora', 8, 'qv', 0), blocks('qlora', 8, 'all', 1)
    check('challenge summary: the initial setup 141.1 + 0.3 + 55.4 GB; the best plan 39.5 + 1.7 + 3.7 GB',
          [g1(s0[k]) for k in 'wta'] == ['141.1 GB', '0.3 GB', '55.4 GB'] and [g1(sb[k]) for k in 'wta'] == ['39.5 GB', '1.7 GB', '3.7 GB'], ([g1(s0[k]) for k in 'wta'], [g1(sb[k]) for k in 'wta']))
    s2 = blocks('qlora', 16, 'all', 1)
    check('challenge summary, tier 2 (r = 16): the gap is the LoRA block, 3.3 GB against 1.7 GB, twice the parameters', g1(s2['t']) == '3.3 GB' and lora_count('70b', 16, 'all') == 2 * lora_count('70b', 8, 'all')
          and max(('w', 't', 'a'), key=lambda k: s2[k] - sb[k]) == 't', (g1(s2['t']), g1(sb['t'])))
    # ---------- the capability layer
    ex = ev("()=>window.__lora.exits()")
    l405, l70c = lora_count('405b', 16, 'all'), lora_count('70b', 16, 'all')
    ratio = l405 / l70c
    check('transfer 1: 405B LoRA r = 16 all 7 = 623,443,968, about 3 times the 70B → option “about 600M”', l405 == 623443968 and 2.8 < ratio < 3.2 and ex['transferA'][0] == 1, (l405, ratio))
    check('transfer 1: the premises (1.6 times the layers, about twice the sides)', round(126 / 80, 1) == 1.6 and 1.8 < 53248 / 28672 < 2.01 and 16384 / 8192 == 2)
    w405 = linear_count('405b') * NF4_BYTES + 2 * (WANT_P['405b'] - linear_count('405b'))
    w70 = linear_count('70b') * NF4_BYTES + 2 * (WANT_P['70b'] - linear_count('70b'))
    check('transfer 2: 405B QLoRA frozen weights 215.6 GB > 200 GB, so no on one H100; 70B 39.5 GB; 5.8 times the parameters',
          g1(w405) == '215.6 GB' and w405 > 200e9 and ex['transferA'][1] == 0 and g1(w70) == '39.5 GB' and round(WANT_P['405b'] / WANT_P['70b'], 1) == 5.8, (w405, w70))
    check('explain answers', ex['explainA'] == [0, 0], ex)
    check('explain 1 premises: about 169 GB and about 41 GB', round(full8['tot'] / 1e9) == 169 and round(lora8['tot'] / 1e9) == 41)
    # ---------- the fidelity card's left-out bounds
    check('left out: the largest temporary BF16 matrix of the 70B is 0.5 GB', g1(2 * 8192 * 28672) == '0.5 GB')
    # x·A: r numbers per token per matrix with a LoRA. Base case at its largest: the 70B (80 layers), rank 16, all 7, 4,096 tokens, in FP32
    # (PEFT runs lora_A in the adapter dtype); the 8B is smaller. The largest case the lab allows: 70B, 32,768 tokens, rank 256, all 7, BF16.
    xa8, xa70b = 4096 * 32 * 7 * 16 * 4, 4096 * 80 * 7 * 16 * 4
    xa_max = 32768 * 80 * 7 * 256 * 2
    check('left out (S17): LoRA’s own activations under 0.2 GB at rank 16 and 4,096 tokens, both models, FP32', max(xa8, xa70b) < 0.2e9 and xa70b > 0.1e9, (xa8, xa70b))
    check('left out (S17): about 9 GB for the 70B at rank 256 and 32,768 tokens in BF16 (C shows the same bound)', round(xa_max / 1e9) == 9 and max(C['RANKS']) == 256 and max(C['SEQS']) == 32768, xa_max)
    # ---------- the fixes of the b9v1 review
    lo_rc8 = peak('8b', 'lora', 16, 'all', 4096, 1)
    check('scenario q8 (S18): LoRA with recomputation fits the 24 GB card at 19.9 GB; QLoRA saves about 10 GB more',
          g1(lo_rc8['tot']) == '19.9 GB' and lo_rc8['tot'] <= 24e9 and round((lo_rc8['tot'] - q8['tot']) / 1e9) == 10, (lo_rc8['tot'], q8['tot']))
    more = lora_count('70b', 64, 'qv') / best['trained'] - 1
    check('challenge reveal (S13): q and v at r = 64, or the 4 attention matrices at r = 32, train about a quarter more than the best plan',
          abs(more - 0.25) < min(abs(more - 0.2), abs(more - 1 / 3), abs(more - 0.5)) and lora_count('70b', 32, 'attn') == lora_count('70b', 64, 'qv'), more)
    w4_70, fb_70 = linear_count('70b') * NF4_BYTES, 2 * (WANT_P['70b'] - linear_count('70b'))
    check('transfer 2 (S14): 35.3 GB of 4-bit weights + 4.2 GB of BF16 embedding and LM head, about 40 GB; 5.8 × 40 GB over 200 GB',
          (g1(w4_70), g1(fb_70), round((w4_70 + fb_70) / 1e10) * 10) == ('35.3 GB', '4.2 GB', 40) and 5.8 * 40 > 200, (w4_70, fb_70))
    # part 4 result (S9): the trained parameters as a multiple or a share of the initial setup (rank 16 on all 7)
    rel = lambda r, tg: lora_count('8b', r, tg) / l16
    check('part 5 (S9): r = 64 on all 7 trains 4 times the initial setup; on q and v 65% of it', rel(64, 'all') == 4 and f'{rel(64, "qv") * 100:.0f}' == '65', (rel(64, 'all'), rel(64, 'qv')))
    check('part 5 (S8): torchtune’s 8B LoRA config adapts 6 of the 7 matrices (q, v, o and the MLP: all but k)',
          "lora_attn_modules: ['q_proj', 'v_proj', 'output_proj']" in (P / 'sources/b9src/lora/pinned/tt__8B_lora_single_device.yaml').read_text() and 'apply_lora_to_mlp: True' in (P / 'sources/b9src/lora/pinned/tt__8B_lora_single_device.yaml').read_text())
    lp_frac = 6 * l16 / (6 * N8)
    check('left out: LoRA arithmetic under 1% of the compute (rank 16 on all 7)', lp_frac < 0.01, lp_frac)
    # ---------- the drill (seeded)
    bad = []
    for seed in range(1, 41):
        d = ev("s=>window.__lora.drill(s)", seed); s = d['s']
        py = peak(s['m'], s['meth'], s['r'], s['tg'], s['seq'], s['rc'])
        if not (close(d['tot'], py['tot']) and d['lp'] == py['lp'] and d['fits'] == (py['tot'] <= 80e9)): bad.append((seed, d, py['tot']))
    check('drill: 40 seeds recomputed', not bad, bad[:2])
    # ---------- the home tile
    tz = {m_: ev("m=>window.__tzLora.model(m)", m_) for m_ in ('full', 'lora', 'qlora')}
    full8n, q8n = peak('8b', 'full', 16, 'all', 4096, 0), peak('8b', 'qlora', 16, 'all', 4096, 0)
    check('tile: full 168.7, LoRA 40.8, QLoRA 30.5 GB, same as the lab', all(close(tz[k]['tot'], v['tot']) for k, v in (('full', full8n), ('lora', lora8), ('qlora', q8n))) and [g1(tz[k]['tot']) for k in ('full', 'lora', 'qlora')] == ['168.7 GB', '40.8 GB', '30.5 GB'], {k: v['tot'] for k, v in tz.items()})
    check('tile: only full fine-tuning goes over 80 GB', [tz[k]['fits'] for k in ('full', 'lora', 'qlora')] == [False, True, True])
    check('tile: the gray part of LoRA is 0.7 GB; QLoRA mostly activations', g1(16 * l16) == '0.7 GB' and q8n['acts'] / q8n['tot'] > 0.5)
    rd = ev("()=>['full','lora','qlora'].map(m=>window.__tzLora.read(m).replace(/<[^>]+>/g,''))")
    check('tile: the read sentences carry the numbers, at most 25 words', all(x in rd[i] for i, x in enumerate(('168.7', '40.8', '30.5'))) and all(len(x.split()) <= 25 for x in rd), rd)
    check('tile: no share of trained parameters (R3.8)', not any('%' in x for x in rd))
    # ---------- the Connect card's code (its lora_params part; torch is not needed for it)
    code = C['CODE']
    part = code[code.index('def lora_params'):]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): exec(part, {})
    outl = buf.getvalue().strip()
    check('Connect code prints 41943040 0.67108864, as its comment says', outl == '41943040 0.67108864' and '# 41943040 0.67108864' in code, outl)
    check('Connect code: B = zeros, (xA)B order, merge', 'torch.zeros(d_out, r)' in code and '@ self.A.T) @ self.B.T' in code and 'def merge' in code)
    # ---------- the torchtune figures quoted on the page are in the source file
    ttr = (P / 'sources' / 'b8src' / 'training-memory' / 'torchtune__README.md').read_text()
    for s_ in ('| Llama 3.1 8B | LoRA | 1x 4090 |  16.2 GiB', '| Llama 3.1 8B | LoRA | 1x A6000 | 30.3 GiB', '| Llama 3.1 8B | QLoRA | 1x 4090 | 7.4 GiB', '| Llama 3.1 405B | QLoRA | 8x A100', 'batch size 2 (without gradient accumulation), a dataset packed to sequence length 2048'):
        check(f'torchtune README quote: {s_[:40]}', s_ in ttr)
    src = P / 'sources' / 'b9src' / 'lora' / 'pinned'   # the pinned copies the quotes are checked against
    quotes = [('peft__src_peft_tuners_lora_config.py', 'r: int = field(default=8'), ('peft__src_peft_tuners_lora_config.py', 'lora_alpha: int = field(default=8'),
              ('peft__src_peft_utils_constants.py', '"llama": ["q_proj", "v_proj"]'), ('peft__src_peft_tuners_lora_layer.py', 'self.scaling[adapter_name] = lora_alpha / r'),
              ('peft__src_peft_tuners_lora_layer.py', 'nn.init.zeros_(self.lora_B[adapter_name].weight)'),
              ('qlora__README.md', 'finetune a 65B parameter model on a single 48GB GPU'), ('lora__README.md', 'without introducing inference latency'),
              ('tt__8B_lora_single_device.yaml', 'lora_rank: 8  # higher increases accuracy and memory'), ('tt__8B_lora_single_device.yaml', 'enable_activation_checkpointing: True'),
              ('bnb__bitsandbytes_functional.py', 'blocksize=256'), ('tf__bitsandbytes.py', 'lm_head')]
    for f_, s_ in quotes:
        t = (src / f_).read_text(errors='ignore') if (src / f_).exists() else ''
        check(f'source quote in {f_}: {s_[:50]}', s_ in t, f_)
    # the quotes the fixes added (S10, S15, S16)
    more_quotes = [('pinned/lora__README.md', 'We obtain result comparable or superior to full finetuning on the GLUE benchmark'), ('pinned/lora__README.md', 'DeBERTa (He et al., 2020)](https://arxiv.org/abs/2006.03654) XXL 1.5B'),
                   ('pinned/lora__README.md', 'This vastly reduces the storage requirement for large language models adapted to specific tasks and enables efficient task-switching during deployment'),
                   ('pinned/qlora__README.md', 'while preserving full 16-bit finetuning task performance'), ('pinned/qlora__README.md', 'multiple model types (LLaMA, T5)'),
                   ('pinned/peft__src_peft_tuners_lora_layer.py', 'x = self._cast_input_dtype(x, lora_A.weight.dtype)')]
    for f_, s_ in more_quotes:
        t = (P / 'sources' / 'b9src' / 'lora' / f_).read_text(errors='ignore')
        check(f'source quote in {f_}: {s_[:50]}', s_ in t, f_)
    tfull = (P / 'sources' / 'b8src' / 'training-memory' / 'torchtune__llama3_1_8B_full_single_device.yaml').read_text()
    check('Connect (S15): torchtune’s 8B full fine-tune on one 4090, 18.9 GiB, with BF16 weights, 8-bit paged AdamW, the update in the backward pass',
          '| Llama 3.1 8B | Full finetune | 1x 4090 | 18.9 GiB' in ttr and 'bitsandbytes.optim.PagedAdamW8bit' in tfull and 'optimizer_in_bwd: True' in tfull and 'dtype: bf16' in tfull)
    check('no page errors', not errs, errs[:3])
    b.close()
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails if fails else 'none')
sys.exit(1 if fails else 0)
