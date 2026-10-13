"""Numbers test for the roofline lab (CONTRACT §6): every key number the page shows, recomputed independently in Python.

The Python side derives everything from first principles: Llama-3.1-8B's shapes from its config.json, the datasheet peaks, and the
roofline (time = max(FLOPs / peak, bytes / bandwidth)). It then checks the page's engine through the read-only hook window.__rf, the
page's formatted strings, the teaser's hook window.__tzRf, prefill-decode's own engine (labs/prefill-decode.js loaded into the same
page, window.__pd) for the shared constants and the 50% / 60% times the Fidelity popover quotes, and runs the Connect section's
minimal implementation to compare its printed lines with the comments under it.
Run: python3 tests/roofline_numbers.py"""
import math, pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/roofline.html', '--only', 'roofline'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-roofline.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'roofline.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-roofline.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


def close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-30)


# ---------------- first principles ----------------
cfg = dict(vocab=128256, hidden=4096, layers=32, heads=32, kv_heads=8, head_dim=128, ffn=14336)   # Llama-3.1-8B config.json
d, L, kvd = cfg['hidden'], cfg['layers'], cfg['kv_heads'] * cfg['head_dim']
attn_w = d * d + d * kvd + d * kvd + d * d            # Wq, Wk, Wv, Wo
mlp_w = 3 * d * cfg['ffn']                            # gate, up, down
norms = 2 * d                                         # two RMSNorm weights per layer
emb = cfg['vocab'] * d; head = cfg['vocab'] * d       # untied embedding and LM head
layers = L * (attn_w + mlp_w + norms) + d             # + the final norm
total = layers + emb + head
W = total - emb                                       # a decode step multiplies every weight but the embedding table (looked up)
KV = 2 * L * kvd * 2                                  # K and V, all layers, BF16: bytes per token
GPU = {'a100': (312e12, 2039e9), 'h100': (1979e12 / 2, 3.35e12), 'h200': (1979e12 / 2, 4.8e12)}


def attn_flops(q_from, q_to):
    """QK^T and AV over all layers for query positions q_from..q_to-1, each attending to itself and every earlier position."""
    return sum(2 * 2 * L * d * (q + 1) for q in range(q_from, q_to)) if q_to - q_from < 5000 else \
        2 * 2 * L * d * ((q_to * (q_to + 1) - q_from * (q_from + 1)) // 2)


def kernel(k, B=1, N=2048, c=4096, n=256, p='bf16'):
    wb = 2 if p == 'bf16' else 1
    if k == 'dec': return 2 * W * B, wb * W
    if k == 'pre': return 2 * layers * N + 2 * head + attn_flops(0, N), wb * W + KV * N
    if k == 'att': return B * attn_flops(c, c + 1), B * KV * c
    if k == 'mm': return 2 * n ** 3, 2 * 3 * n * n
    return 2048 * d, 3 * 2 * 2048 * d                  # residual add: read two numbers, write one, 2 bytes each, 1 FLOP


def roof(fl, by, gpu):
    peak, bw = GPU[gpu]
    tc, tm = fl / peak, by / bw
    t = max(tc, tm)
    return dict(fl=fl, by=by, I=fl / by, tc=tc, tm=tm, t=t, att=fl / t, share=fl / t / peak, ridge=peak / bw,
                bound='compute' if tc >= tm else 'memory')


def py_ev(k='dec', gpu='h100', **kw):
    return roof(*kernel(k, **kw), gpu)


# ---------------- the page ----------------
with sync_playwright() as pw:
    br = pw.chromium.launch()
    pg = br.new_page(viewport={'width': 1440, 'height': 900})
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.evaluate("()=>{location.hash='roofline'}"); pg.wait_for_timeout(500)
    rf = lambda js, *a: pg.evaluate(js, *a)
    C = rf("()=>window.__rf.consts")

    # constants shared with prefill-decode
    check('P_TOTAL is 8,030,261,248', total == 8030261248 == C['P_TOTAL'], (total, C['P_TOTAL']))
    check('W_PAR (every parameter but the embedding table) is 7,504,924,672', W == 7504924672 == C['W_PAR'], C['W_PAR'])
    check('embedding table 525,336,576', emb == C['P_EMB'] == 525336576, C['P_EMB'])
    check('KV cache 131,072 bytes per token', KV == C['KV_TOK'] == 131072, C['KV_TOK'])
    for g, (pk, bw) in GPU.items():
        check(f'{g} peak and bandwidth', close(C['GPUS'][g]['peak'], pk) and close(C['GPUS'][g]['bw'], bw), C['GPUS'][g])

    # the engine against Python at every slider stop and kernel
    cases = [dict(k='dec', B=B, gpu=g, p=p) for B in C['BST'] for g in ('a100', 'h100') for p in ('bf16', 'fp8')]
    cases += [dict(k='pre', N=N, gpu=g, p=p) for N in C['NST'] for g in ('a100', 'h100') for p in ('bf16', 'fp8')]
    cases += [dict(k='att', B=B, c=c, gpu=g) for B in (1, 16, 64, 256) for c in C['CST'] for g in ('a100', 'h100')]
    cases += [dict(k='mm', n=m, gpu=g) for m in C['MST'] for g in ('a100', 'h100')]
    cases += [dict(k='ew', gpu=g) for g in ('a100', 'h100')]
    bad = []
    for cs in cases:
        js = rf("s=>window.__rf.ev(s)", cs)
        py = py_ev(**cs)
        for key in ('fl', 'by', 'I', 't', 'att', 'share', 'ridge'):
            if not close(js[key], py[key], 1e-9): bad.append((cs, key, js[key], py[key]))
        if js['bound'] != py['bound']: bad.append((cs, 'bound', js['bound'], py['bound']))
    check(f'engine = Python for {len(cases)} kernels (FLOPs, bytes, intensity, time, TFLOP/s, share, ridge, bound)', not bad, bad[:3])

    # the numbers the spec, the text and the cards state
    F = lambda kind, v: rf(f"v=>window.__rf.fmt.{kind}(v)", v)
    b1 = py_ev('dec', B=1)
    check('mission: 4.48 ms, 0.34% of the peak, 15.0 GB, 15.0 GFLOP', (F('time', b1['t']), F('share', b1['share']), F('bytes', b1['by']), F('flops', b1['fl'])) == ('4.48 ms', '0.34%', '15.0 GB', '15.0 GFLOP'),
          (F('time', b1['t']), F('share', b1['share'])))
    exp = {16: '5.4%', 32: '10.8%', 64: '21.7%', 256: '86.7%', 512: '100%'}
    got = {B: F('share', py_ev('dec', B=B)['share']) for B in exp}
    check('decode share of peak at batch 16, 32, 64, 256, 512', got == exp, got)
    check('decode at batch 64 takes the same 4.48 ms; batch 512 takes 7.77 ms, compute-bound',
          F('time', py_ev('dec', B=64)['t']) == '4.48 ms' and F('time', py_ev('dec', B=512)['t']) == '7.77 ms' and py_ev('dec', B=512)['bound'] == 'compute')
    check('ridges: H100 295, A100 153, H200 206', [round(GPU[g][0] / GPU[g][1]) for g in ('h100', 'a100', 'h200')] == [295, 153, 206])
    check('first compute-bound batch on the H100 is past 295 (384 among the stops)',
          min(B for B in C['BST'] if py_ev('dec', B=B)['bound'] == 'compute') == 384)
    pre = py_ev('pre', N=2048)
    check('prefill of 2,048 tokens: 29.7 TFLOP, 15.3 GB, about 1,943 FLOPs per byte, 30.0 ms',
          (F('flops', pre['fl']), F('bytes', pre['by']), F('I', pre['I']), F('time', pre['t'])) == ('29.7 TFLOP', '15.3 GB', '1,943', '30.0 ms'),
          (F('flops', pre['fl']), F('bytes', pre['by']), F('I', pre['I']), F('time', pre['t'])))
    at = py_ev('att', B=64, c=4096)
    check('attention 64 users × 4K: 34.4 GB, 10.3 ms, about 4.0 FLOPs per byte, 1.4% of the peak',
          (F('bytes', at['by']), F('time', at['t']), F('I', at['I']), F('share', at['share'])) == ('34.4 GB', '10.3 ms', '4.0', '1.4%'),
          (F('bytes', at['by']), F('time', at['t']), F('I', at['I']), F('share', at['share'])))
    check('attention intensity is about 4 at any batch (GQA group of 4)', all(abs(py_ev('att', B=B, c=4096)['I'] - 4) < 0.01 for B in (1, 64, 1024)))
    ew = py_ev('ew')
    check('residual add: 1 FLOP per 6 bytes, 15.0 µs', close(ew['I'], 1 / 6) and F('time', ew['t']) == '15.0 µs', F('time', ew['t']))
    nr = 3 * GPU['h100'][0] / GPU['h100'][1]
    check('square matmul: I = n / 3, the ridge at n ≈ 886, compute-bound from n = 1,024 on the stops',
          round(nr) == 886 and min(m for m in C['MST'] if py_ev('mm', n=m)['bound'] == 'compute') == 1024 and close(py_ev('mm', n=768)['I'], 256))

    mb = [pr['id'] for pr in C['PRESETS'] if py_ev(**{k: v for k, v in pr.items() if k in ('k', 'B', 'N', 'c', 'n', 'gpu', 'p')})['bound'] == 'memory']
    check('part 6 and the dock: four of the five scenarios are memory-bound (all but the prompt)', mb == ['chat1', 'busy', 'attn', 'resid'], mb)
    # part 5: one change at a time
    duo = {k: rf("id=>window.__rf.duo(id)", k) for k in ('base', 'a100', 'fp8')}
    want = {'base': ('4.48 ms', '30.0 ms'), 'a100': ('7.36 ms', '95.2 ms'), 'fp8': ('2.24 ms', '30.0 ms')}
    got = {k: (F('time', v['dec']['t']), F('time', v['pre']['t'])) for k, v in duo.items()}
    check('part 5: decode at batch 16 and prefill of 2,048 for H100, A100, FP8 weights', got == want, got)
    pyd = {'base': (py_ev('dec', B=16), py_ev('pre', N=2048)), 'a100': (py_ev('dec', B=16, gpu='a100'), py_ev('pre', N=2048, gpu='a100')),
           'fp8': (py_ev('dec', B=16, p='fp8'), py_ev('pre', N=2048, p='fp8'))}
    check('part 5: the engine matches Python', all(close(duo[k]['dec']['t'], pyd[k][0]['t']) and close(duo[k]['pre']['t'], pyd[k][1]['t']) for k in duo))
    r_dec = pyd['a100'][0]['t'] / pyd['base'][0]['t']; r_pre = pyd['a100'][1]['t'] / pyd['base'][1]['t']
    check('part 5 / misconception: A100 slows decode 1.6 times (bandwidth ratio) and prefill 3.2 times (FLOP/s ratio)',
          f'{r_dec:.1f}' == '1.6' and f'{r_pre:.1f}' == '3.2' and close(r_dec, 3.35e12 / 2039e9) and close(r_pre, 989.5 / 312), (r_dec, r_pre))

    # the challenge on the H200
    rows = {(B, p): rf("([B,p])=>window.__rf.chEval(B,p)", [B, p]) for B in C['CH']['Bs'] for p in C['CH']['Ps']}
    pyc = {(B, p): py_ev('dec', B=B, p=p, gpu='h200') for B in C['CH']['Bs'] for p in C['CH']['Ps']}
    check('challenge: engine = Python for the 10 options', all(close(rows[k]['share'], pyc[k]['share']) for k in rows))
    passing = sorted(k for k, v in pyc.items() if v['share'] >= 0.5)
    check('challenge: passing = BF16 256, 512 and FP8 64, 256, 512', passing == sorted([(256, 'bf16'), (512, 'bf16'), (64, 'fp8'), (256, 'fp8'), (512, 'fp8')]), passing)
    check('challenge: page marks the same options as passing', sorted(k for k, v in rows.items() if v['ok']) == passing)
    best = min(passing, key=lambda k: (k[0], (k[0] != 1) + (k[1] != 'bf16')))
    jb = rf("()=>window.__rf.chBest()")
    check('challenge: the minimal answer is FP8 weights at batch 64 (62.1%)', best == (64, 'fp8') == (jb['B'], jb['p']) and F('share', pyc[best]['share']) == '62.1%', (jb, F('share', pyc[best]['share'])))
    check('challenge: BF16 at 64 reaches 31.0%, the initial setup 0.49%', (F('share', pyc[(64, 'bf16')]['share']), F('share', pyc[(1, 'bf16')]['share'])) == ('31.0%', '0.49%'),
          (F('share', pyc[(64, 'bf16')]['share']), F('share', pyc[(1, 'bf16')]['share'])))

    # misconception 3: with each user's KV cache read added (4K tokens each), a step for 64 users against one user alone
    one = py_ev('dec', B=1)['t'] + py_ev('att', B=1, c=4096)['t']; many = py_ev('dec', B=64)['t'] + py_ev('att', B=64, c=4096)['t']
    kv = rf("()=>window.__rf.kv()")
    check('misconception 3: weights + KV cache at 4K, 64 users 14.7 ms against 4.64 ms alone, about 3 times (not 64)',
          close(kv['t1'], one) and close(kv['t64'], many) and (F('time', one), F('time', many)) == ('4.64 ms', '14.7 ms') and round(many / one) == 3 and f'{many / one:.1f}' == '3.2',
          (F('time', one), F('time', many), many / one))
    # the KV cache of 64 users at 4K fits next to the weights on one 80 GB H100 (the scenario is a real setup)
    check('64 users × 4K of KV cache (34.4 GB) plus 16.1 GB of weights fit in 80 GB', 64 * 4096 * KV + 2 * total < 80e9)

    # capability layer
    x32 = rf("()=>window.__rf.x32()")
    check('new case 1: batch 32 reaches 10.8%; a model twice as large keeps the share (FLOPs and bytes both double)',
          F('share', x32['share']) == '10.8%' and close((2 * 2 * W * 32 / (2 * 2 * W)) * 3.35e12 / 989.5e12, x32['share']))
    xi = 64 * 4096 * 4096 / (64 * 4096 + 4096 * 4096 + 64 * 4096)
    check('new case 2: the (64 × 4,096)·(4,096 × 4,096) matmul does 62.1 FLOPs per byte, memory-bound', close(rf("()=>window.__rf.xmmI()"), xi) and f'{xi:.1f}' == '62.1' and xi < 295)

    # the Fidelity popover's activations note: about 9% more bytes at batch 256; the ridge moves from 295 to about 331
    per_tok = 2 * (L * ((d + d + d + d + d + d + cfg['ffn']) + (d + kvd + kvd + d + cfg['ffn'] + cfg['ffn'] + d)) + (d + cfg['vocab']))
    extra = per_tok * 256 / (2 * W)
    ridge = GPU['h100'][0] / GPU['h100'][1]
    b_act = ridge * 2 * W / (2 * W - ridge * per_tok)
    check('activations: about 9% more bytes at batch 256, about 9% fewer FLOPs per byte (spec: ridge batch about 331)',
          round(extra * 100) == 9 and round((1 - 1 / (1 + extra)) * 100) == 9 and round(b_act) == 331, (extra, b_act))

    # the interview drill: 30 seeds, recomputed
    bad = []
    for s in range(1, 31):
        dr = rf("s=>window.__rf.drill(s)", s)
        fl, by = 2 * dr['m'] * dr['k'] * dr['n'], 2 * (dr['m'] * dr['k'] + dr['k'] * dr['n'] + dr['m'] * dr['n'])
        r = roof(fl, by, dr['gpu'])
        if not (dr['fl'] == fl and dr['by'] == by and close(dr['r']['t'], r['t']) and dr['r']['bound'] == r['bound']): bad.append(s)
    check('drill: 30 seeds recomputed', not bad, bad)

    # the teaser: every stop
    T = rf("()=>window.__tzRf.BST")
    check('teaser uses the lab batch stops', T == C['BST'])
    bad = [B for B in T if not (close(rf("B=>window.__tzRf.model(B)", B)['share'], py_ev('dec', B=B)['share']) and rf("B=>window.__tzRf.model(B)", B)['bound'] == py_ev('dec', B=B)['bound'])]
    check('teaser: share and bound at every batch = Python', not bad, bad)
    rd16 = re.sub('<[^>]+>', '', rf("()=>window.__tzRf.read(16)"))
    check('teaser sentence at batch 16: memory-bound, 5.4%, at most 25 words', 'memory-bound' in rd16 and '5.4%' in rd16 and len(rd16.split()) <= 25, rd16)
    rd384 = re.sub('<[^>]+>', '', rf("()=>window.__tzRf.read(384)"))
    check('teaser sentence at batch 384: compute-bound, 100%', 'compute-bound' in rd384 and '100%' in rd384, rd384)
    check('teaser never states a step time (part 2 asks for it, R3.8)', all(' ms' not in re.sub('<[^>]+>', '', rf("B=>window.__tzRf.read(B)", B)) for B in T))

    # prefill-decode's engine, loaded into the same page: the same constants, and the 50% / 60% times the popover quotes
    try:
        pg.add_script_tag(path=str(P / 'labs' / 'prefill-decode.js')); pg.wait_for_timeout(100)
        pd = rf("()=>window.__pd&&window.__pd.consts")
    except Exception as e:
        pd = None; print('note: could not load prefill-decode.js:', e)
    check('prefill-decode loads in the same page', bool(pd))
    if pd:
        check('prefill-decode: same H100 and A100 peaks and bandwidths', close(pd['H100']['peak'], GPU['h100'][0]) and close(pd['H100']['bw'], GPU['h100'][1]) and close(pd['A100']['peak'], GPU['a100'][0]) and close(pd['A100']['bw'], GPU['a100'][1]))
        check('prefill-decode: W_READ = 2 bytes × W_PAR', pd['W_READ'] == 2 * W and pd['P_TOTAL'] == total and pd['KV_TOK'] == KV)
        check('prefill-decode: efficiencies 50% / 60%', pd['EFF'] == {'mfu': 0.5, 'mbu': 0.6}, pd['EFF'])
        pp = rf("()=>window.__pd.prefill(2048,'h100')")
        check('prefill-decode: same prefill FLOPs for 2,048 tokens', close(pp['fl'], pre['fl']), (pp['fl'], pre['fl']))
        check('prefill-decode: 60.0 ms for that prefill (30.0 ms here at 100%)', F('time', pp['t']) == '60.0 ms', F('time', pp['t']))
        check('prefill-decode: ridge at 50% / 60% about 246; decode weights 7.47 ms (4.48 here)',
              round(0.5 * GPU['h100'][0] / (0.6 * GPU['h100'][1])) == 246 and F('time', 2 * W / (0.6 * GPU['h100'][1])) == '7.47 ms')

    # the popover and the parts show those numbers
    pg2 = br.new_page(viewport={'width': 1440, 'height': 900}); pg2.on('pageerror', lambda e: errs.append(str(e)))
    pg2.goto(URL); pg2.wait_for_timeout(250); pg2.evaluate("()=>{location.hash='roofline'}"); pg2.wait_for_timeout(450)
    body = pg2.evaluate("()=>document.querySelector('#rf-app').innerText")
    check('mission card shows 4.48 ms and 0.34%', '4.48 ms' in body and '0.34%' in body)
    html = pg2.content()
    check('Fidelity popover quotes 60.0 ms, 30.0 ms, 7.47 ms, 4.48 ms and a ridge of 246',
          all(x in html for x in ('60.0 ms', '30.0 ms', '7.47 ms', '4.48 ms', 'about 246')))

    # the minimal implementation runs and prints the lines in its comments
    code = C['CODE']
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    want = [ln[2:] for ln in code.splitlines() if ln.startswith('# ') and ('TFLOP/s' in ln and ' us ' in ln)]
    check('Connect code runs and prints exactly its comment lines', out == want, (out, want))
    check('Connect code: lines at most 74 characters, at most 40 lines', max(len(x) for x in code.splitlines()) <= 74 and len(code.splitlines()) <= 40,
          (max(len(x) for x in code.splitlines()), len(code.splitlines())))
    check('Connect code agrees with the engine (n = 4,096 matmul 138.9 µs, decode at batch 1 4,480.6 µs)',
          f"{py_ev('mm', n=4096)['t'] * 1e6:.1f}" == '138.9' and f"{py_ev('dec', B=1)['t'] * 1e6:.1f}" == '4480.6')
    check('Connect code prints no decode at a batch above 1 (part 2 asks for batch 64, R3.8)', not re.search(r'decode\(([2-9]|\d\d+)\)', code) and 'B=64' not in code)
    check('no page errors', not errs, errs)
    br.close()

print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
