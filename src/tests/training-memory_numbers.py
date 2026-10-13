"""Golden numbers for the training-memory lab (guide R9.2, CONTRACT §6).

Derived here independently of the JS engine:
- model shapes from meta-llama/llama-models (models/sku_list.py, Llama 3.1 8B / 70B / 405B; Llama 3.2 1B), the FFN size from the
  FeedForward rule (int(2·4d/3) × multiplier, rounded up to multiple_of), and the parameter count summed matrix by matrix
  (untied output for 3.1, tied for 3.2 1B);
- model states with the 18-byte accounting (BF16 weights 2, FP32 gradients 4, FP32 copy 4, m 4, v 4; Megatron-LM dist_optimizer.md,
  Hugging Face model_memory_anatomy.md), 8-bit AdamW 12, SGD with momentum 14;
- activations: the tensors one Llama layer keeps for the backward pass, listed by name, 2 bytes each; full recomputation keeps 2d;
  logits T × V × 4; the peak = states + kept activations + logits, which a step-by-step walk over one update confirms (the walk is
  written here, layer by layer, not ported from the JS timeline);
- the challenge: 16 plans, which fit in 24 × 10⁹ bytes, the relative time (6N or 8N per token ÷ the assumed utilisation), the goal;
- the popover's GiB claim, the capability questions' answers, the drill, the home tile, and the Connect card's code (run here).
Run: python3 tests/training-memory_numbers.py"""
import math, pathlib, subprocess, sys, io, contextlib, re, itertools
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/training-memory.html', '--only', 'training-memory'], check=True, cwd=P, stdout=subprocess.DEVNULL)
OUT = P / 'out' / 'training-memory'; OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'wrapped-num.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'training-memory.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-num.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


def close(a, b, rel=1e-12):
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


g1 = lambda b: f'{b / 1e9:,.1f}'   # the page's GB with one decimal

# ---------- shapes (sku_list.py) and the FeedForward rule (llama3/model.py)
SKU = {'8b': dict(d=4096, L=32, H=32, KV=8, mult=1.3, mo=1024, tied=False),
       '70b': dict(d=8192, L=80, H=64, KV=8, mult=1.3, mo=4096, tied=False),
       '405b': dict(d=16384, L=126, H=128, KV=8, mult=1.2, mo=4096, tied=False),
       '1b': dict(d=2048, L=16, H=32, KV=8, mult=1.5, mo=256, tied=True),
       '3b': dict(d=3072, L=28, H=24, KV=8, mult=1.0, mo=256, tied=True)}   # the home tile only (sku_list.py, llama3_2_3b)
V = 128256


def ffn(d, mult, mo):
    h = int(2 * (4 * d) / 3); h = int(mult * h); return mo * ((h + mo - 1) // mo)


def params(s):
    d, L, H, KV = s['d'], s['L'], s['H'], s['KV']; hd = d // H; f = ffn(d, s['mult'], s['mo'])
    layer = d * H * hd + 2 * d * KV * hd + H * hd * d + 3 * d * f + 2 * d
    return V * d * (1 if s['tied'] else 2) + L * layer + d


def acts_per_token_layer(s):
    d, H, KV = s['d'], s['H'], s['KV']; hd = d // H; f = ffn(d, s['mult'], s['mo'])
    kept = {'attn norm in': d, 'attn norm out': d, 'q': H * hd, 'k': KV * hd, 'v': KV * hd, 'attn out': H * hd,
            'mlp norm in': d, 'mlp norm out': d, 'gate': f, 'silu(gate)': f, 'up': f, 'silu(gate)*up': f}
    return 2 * sum(kept.values())


BPP = {'adamw': (2, 4, 4, 4, 4), 'adam8': (2, 4, 4, 1, 1), 'sgdm': (2, 4, 4, 4, 0)}


def peak(m, opt, seq, mb, rc):
    s = SKU[m]; T = seq * mb
    states = params(s) * sum(BPP[opt]); kept = T * s['L'] * (2 * s['d'] if rc else acts_per_token_layer(s)); lg = T * V * 4
    return states, kept, lg, states + kept + lg


def walk_max(m, opt, seq, mb, acc, rc):
    """one update, step by step: per micro-batch the forward adds each layer's kept tensors, the LM head adds the logits, the
    backward frees them layer by layer (with recomputation a layer's full activations come back for that layer only)"""
    s = SKU[m]; T = seq * mb; st = params(s) * sum(BPP[opt]); a = acts_per_token_layer(s) * T; i = 2 * s['d'] * T
    kept_l = i if rc else a; hi = 0; where = None
    for _ in range(acc):
        kept = 0
        for _l in range(s['L']):
            kept += kept_l
            if st + kept > hi: hi, where = st + kept, 'forward'
        tot = st + kept + T * V * 4
        if tot > hi: hi, where = tot, 'end of forward'
        for _l in range(s['L']):
            if rc and st + kept + a > hi: hi, where = st + kept + a, 'recompute'
            kept -= kept_l
    return hi, where


with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300); pg.evaluate("()=>{location.hash='training-memory'}"); pg.wait_for_timeout(500)
    H = lambda expr, *a: pg.evaluate(expr, *a)
    K = H("()=>window.__tmem.consts")
    # ---- shapes and parameter counts
    for m, s in SKU.items():
        if m not in K['MOD']: continue
        jm = K['MOD'][m]
        check(f'{m}: FFN size from the FeedForward rule', ffn(s['d'], s['mult'], s['mo']) == jm['f'], (ffn(s['d'], s['mult'], s['mo']), jm['f']))
        check(f'{m}: parameters summed matrix by matrix', params(s) == jm['P'], (params(s), jm['P']))
        check(f'{m}: shape (d, L, H, KV, head_dim)', (s['d'], s['L'], s['H'], s['KV'], s['d'] // s['H']) == (jm['d'], jm['L'], jm['H'], jm['KV'], jm['hd']), jm)
        check(f'{m}: activations per token per layer', acts_per_token_layer(s) == H(f"()=>window.__tmem.aLayer('{m}')"))
    check('8B: 41·d bytes per token per layer', acts_per_token_layer(SKU['8b']) == 41 * 4096 == 167936)
    check('bytes per parameter 18 / 12 / 14', [H(f"()=>window.__tmem.bpp('{o}')") for o in ('adamw', 'adam8', 'sgdm')] == [18, 12, 14])
    # ---- every combination the sandbox allows: the engine's peak, and the walk's maximum at the end of the forward pass
    bad = []
    for m, opt, seq, mb, acc, rc in itertools.product(['8b', '70b'], ['adamw', 'adam8', 'sgdm'], K['SEQS'], K['MBS'], [1, 2], [0, 1]):
        c = H("o=>window.__tmem.compute(o)", dict(m=m, opt=opt, seq=seq, mb=mb, acc=acc, rc=rc))
        st, kp, lg, pk = peak(m, opt, seq, mb, rc); hi, where = walk_max(m, opt, seq, mb, acc, rc)
        if not (close(c['states'], st) and close(c['acts'], kp) and close(c['logits'], lg) and close(c['peak'], pk) and close(hi, pk) and where == 'end of forward'):
            bad.append((m, opt, seq, mb, acc, rc, c['peak'], pk, hi, where))
    check('sandbox: every combination (states, activations, logits, peak at the end of the forward pass)', not bad, bad[:3])
    tl_bad = []
    for o in [dict(), dict(rc=1), dict(mb=4, acc=2), dict(opt='adam8', rc=1, acc=8), dict(m='70b', seq=32768, rc=1)]:
        c = H("o=>window.__tmem.compute(o)", o); tl = H("o=>window.__tmem.timeline(o)", o)
        top = max(c['states'] + v['acts'] + v['lg'] for v in tl['V'])
        if not close(top, c['peak']) or tl['V'][tl['pk']]['ph'] != 'h': tl_bad.append((o, top, c['peak']))
    check('the curve’s highest point is the peak, at the LM head', not tl_bad, tl_bad)
    # ---- the numbers the guided parts print
    st8 = params(SKU['8b']) * 18
    check('8B weights 16.1 GB (title: 16 GB)', g1(params(SKU['8b']) * 2) == '16.1')
    check('8B model states 144.5 GB = 9 × the weights', g1(st8) == '144.5' and 18 / 2 == 9)
    pk0 = peak('8b', 'adamw', 4096, 1, 0)
    check('part 2: peak 168.7 GB, 88.7 GB over 80 GB, prediction “130 to 200 GB”', (g1(pk0[3]), g1(pk0[3] - 80e9)) == ('168.7', '88.7') and 130e9 < pk0[3] < 200e9 and K['PRED_RIGHT'] == '130-200')
    # S1: the "=" hovers show the parts at 2 decimals and the total as the sum of the parts as shown
    s1 = []
    for o_, rc_, mb_ in (('adamw', 0, 1), ('adam8', 1, 1), ('adamw', 0, 4), ('adamw', 0, 2)):
        st_, kp_, lg_, pk_ = peak('8b', o_, 4096, mb_, rc_)
        shown = sum(round(v / 1e7) for v in (st_, kp_, lg_)) / 100
        if abs(shown - pk_ / 1e9) > 0.015: s1.append((o_, rc_, mb_, shown, pk_))
    check('S1: parts at 2 decimals add up to within 0.01 GB of the exact peak (168.66, 99.55, 240.96, 192.82)', not s1, s1)
    check('part 2: 8-bit AdamW 96.4 GB of states, peak 120.5 GB', (g1(peak('8b', 'adam8', 4096, 1, 0)[0]), g1(peak('8b', 'adam8', 4096, 1, 0)[3])) == ('96.4', '120.5'))
    check('part 2: SGD with momentum 112.4 GB of states', g1(peak('8b', 'sgdm', 4096, 1, 0)[0]) == '112.4')
    check('part 3: 4,096 tokens 22.0 GB of activations, logits 2.1 GB; 16,384 tokens 88.0 GB', (g1(pk0[1]), g1(pk0[2]), g1(peak('8b', 'adamw', 16384, 1, 0)[1])) == ('22.0', '2.1', '88.0'))
    check('part 3: a = 168 KB for the 8B', round(acts_per_token_layer(SKU['8b']) / 1000) == 168)
    r = peak('8b', 'adamw', 4096, 1, 1)
    check('part 4: recomputation 1.1 GB of activations, peak 147.7 GB', (g1(r[1]), g1(r[3])) == ('1.1', '147.7'))
    check('part 4: compute 6N → 8N, a third more', close(8 / 6 - 1, 1 / 3))
    ac = {k: g1(peak('8b', 'adamw', 4096, mb, 0)[3]) for k, mb in (('a8x1', 8), ('a4x2', 4), ('a2x4', 2), ('a1x8', 1))}
    check('part 5: 8 at once 337.4, 2 × 4 241.0, 4 × 2 192.8, 8 × 1 168.7 GB', ac == {'a8x1': '337.4', 'a4x2': '241.0', 'a2x4': '192.8', 'a1x8': '168.7'}, ac)
    allp = peak('8b', 'adam8', 4096, 1, 1)
    check('part 5: all three tricks 99.5 GB, 19.5 GB over; states alone 96.4 GB > 80 GB', (g1(allp[3]), g1(allp[3] - 80e9)) == ('99.5', '19.5') and allp[0] > 80e9)
    # ---- scenarios
    check('scenario 32K: activations 176.1 GB > 144.5 GB of states; 8.6 GB with recomputation; peak 337.4 GB',
          (g1(peak('8b', 'adamw', 32768, 1, 0)[1]), g1(peak('8b', 'adamw', 32768, 1, 1)[1]), g1(peak('8b', 'adamw', 32768, 1, 0)[3])) == ('176.1', '8.6', '337.4'))
    s70 = params(SKU['70b']) * 18
    check('scenario 70B: 1,270.0 GB of states, at least 16 H100s; about 9 times the 8B', g1(s70) == '1,270.0' and math.ceil(s70 / 80e9) == 16 and round(params(SKU['70b']) / params(SKU['8b'])) == 9)
    # ---- the popover: read as GiB nothing fits on one H100, and the challenge's best plan is the same
    gib = 80 * 2 ** 30
    low = min(peak(m, o, s, mb, rc)[3] for m in ('8b', '70b') for o in BPP for s in K['SEQS'] for mb in K['MBS'] for rc in (0, 1))
    check('popover: read as 85.9 GB, still no 8B or 70B case fits', low > gib and f'{gib / 1e9:.1f}' == '85.9', low)
    check('popover: the final norm and LM head inputs are under 0.1 GB at 4,096 tokens', 2 * 2 * 4096 * 4096 < 0.1e9)
    check('popover: 12 bytes or more never fit the 8B on one H100; pure BF16 (8 bytes) states 64.2 GB < 80 GB', params(SKU['8b']) * 12 > 80e9 and g1(params(SKU['8b']) * 8) == '64.2')
    prod = H("()=>{const e=document.getElementById('tmem-prod-card'); return e?e.textContent:''}")
    check('connect, torchtune: BF16 weights + 8-bit m and v of the 8B are 32.1 GB, more than the 24 GB card even read as GiB', g1(params(SKU['8b']) * 4) == '32.1' and params(SKU['8b']) * 4 > 24 * 2 ** 30 and '32.1 GB' in prod, prod[-400:])
    # ---- the challenge
    CH = K['CH']; util = {int(k): v for k, v in CH['util'].items()}
    plans = []
    for o in CH['opts']:
        for rc in CH['rcs']:
            for mb in CH['mbs']:
                pkc = peak('1b', o, CH['seq'], mb, rc)[3]; t = ((8 if rc else 6) / util[mb]) / (6 / util[1])
                plans.append((o, rc, mb, pkc, t, pkc <= 24e9, pkc <= 24e9 and t <= CH['goal'] + 1e-9))
    jall = H("()=>window.__tmem.chAll()")
    check('challenge: 16 plans, engine = derivation', len(plans) == 16 and all(close(j['peak'], q[3]) and close(j['t'], q[4]) and j['fits'] == q[5] and j['ok'] == q[6] for j, q in zip(jall, plans)), jall[:2])
    passing = [(q[0], q[1], q[2]) for q in plans if q[6]]
    check('challenge: exactly 2 plans meet the goal (8-bit no recompute 2 at a time; 8-bit recompute 4 at a time)', sorted(passing) == sorted([('adam8', 0, 2), ('adam8', 1, 4)]), passing)
    best = min((q for q in plans if q[5]), key=lambda q: (q[4], q[3]))
    jb = H("()=>window.__tmem.chBest()")
    check('challenge: the fastest plan that fits is 8-bit AdamW, no recomputation, 2 at a time: 23.0 GB, 0.79', (best[0], best[1], best[2]) == ('adam8', 0, 2) == (jb['opt'], jb['rc'], jb['mb']) and g1(best[3]) == '23.0' and f'{best[4]:.2f}' == '0.79')
    start = [q for q in plans if (q[0], q[1], q[2]) == ('adamw', 0, 1)][0]
    check('challenge: the initial setup does not fit (26.3 GB), time 1.00', not start[5] and g1(start[3]) == '26.3' and f'{start[4]:.2f}' == '1.00')
    r4 = [q for q in plans if (q[0], q[1], q[2]) == ('adam8', 1, 4)][0]; r8 = [q for q in plans if (q[0], q[1], q[2]) == ('adam8', 1, 8)][0]
    a2 = [q for q in plans if (q[0], q[1], q[2]) == ('adamw', 0, 2)][0]
    check('challenge texts: recompute 4 at a time 19.6 GB, 0.91; 8 at a time 24.3 GB does not fit; AdamW 2 at a time 30.4 GB',
          (g1(r4[3]), f'{r4[4]:.2f}', g1(r8[3]), r8[5], g1(a2[3])) == ('19.6', '0.91', '24.3', False, '30.4'))
    check('challenge chart: max 56 GB holds every plan', max(q[3] for q in plans) < 56e9)
    best_gib = min((q for q in plans if q[3] <= 24 * 2 ** 30), key=lambda q: (q[4], q[3]))
    check('popover: read as 25.8 GB, the challenge’s fastest plan stays the same', (best_gib[0], best_gib[1], best_gib[2]) == ('adam8', 0, 2) and f'{24 * 2 ** 30 / 1e9:.1f}' == '25.8')
    # ---- capability questions
    st405 = params(SKU['405b']) * 18; acts405 = peak('405b', 'adamw', 4096, 1, 0)[1]
    check('transfer 1: 405B ≈ 50 × the 8B; 7,305.4 GB of states, at least 92 GPUs: “about 90” (option 2)',
          abs(params(SKU['405b']) / params(SKU['8b']) - 50) < 1 and g1(st405) == '7,305.4' and math.ceil(st405 / 80e9) == 92 and H("()=>window.__tmem.exits().transferA[0]") == 1)
    check('transfer 1: 50 × 144.5 GB ≈ 7.2 TB, ÷ 80 GB ≈ 90', round(50 * 144.5 / 1000, 1) == 7.2 and round(7200 / 80) == 90)
    check('transfer 2: hidden 4×, layers ≈ 4×; activations about 15 × (option 2)', 16384 / 4096 == 4 and round(126 / 32) == 4 and round(acts405 / pk0[1]) == 15 and H("()=>window.__tmem.exits().transferA[1]") == 1, acts405 / pk0[1])
    check('explain 2: 8-bit AdamW saves 6 of 18 bytes (m and v 8 → 2)', 18 - 12 == 6 == (4 + 4) - (1 + 1))
    check('BF16 spacing near 0.02 is 2⁻¹³ ≈ 0.00012, so +0.00003 rounds away', math.floor(math.log2(0.02)) == -6 and round(2 ** -13, 5) == 0.00012 and 0.00003 < 2 ** -14)
    # ---- the drill
    dbad = []
    for seed in range(1, 9):
        d = H(f"()=>window.__tmem.drill({seed})"); s = d['s']; q = peak(s['m'], s['opt'], s['seq'], s['mb'], s['rc'])
        if not (close(d['peak'], q[3]) and close(d['states'], q[0])): dbad.append((seed, d, q))
    check('drill: eight seeds', not dbad, dbad[:2])
    # ---- the home tile (the owner's review of batch 8, item 1): four real Llama models, opening at the lab's 8B
    tz = H("()=>({M:window.__tzTmem.MODS, S:window.__tzTmem.SIZES, D:window.__tzTmem.DEF_IDX, P:window.__tzTmem.PARTS, m:window.__tzTmem.MODS.map((_,i)=>window.__tzTmem.model(i))})")
    want = [('1B', '1b'), ('3B', '3b'), ('8B', '8b'), ('70B', '70b')]
    check('tile: stops 1B, 3B, 8B, 70B, each a real Llama model with its parameter count summed from its config',
          tz['S'] == [k for k, _ in want] and all(o['P'] == params(SKU[m]) for o, (_, m) in zip(tz['M'], want)), [(o['k'], o['P']) for o in tz['M']])
    check('tile: the 3B is 3,212,749,824 parameters (tied embeddings), the 8B the lab’s own count', params(SKU['3b']) == 3212749824 and tz['M'][2]['P'] == K['MOD']['8b']['P'] == 8030261248)
    check('tile: opens at the 8B, as its title (R2 / item 1)', tz['D'] == 2 and tz['S'][tz['D']] == '8B')
    check('tile: the lab’s bytes per parameter (2 + 4 + 4 + 4 + 4 = 18, inference 2), 9 times', tz['P'] == [2, 4, 4, 4, 4] and all(m['inf'] == 2 * m['P'] and m['tr'] == 18 * m['P'] and m['ratio'] == 9 for m in tz['m']))
    check('tile: fits one 80 GB GPU only at 1B and 3B', [m['fits'] for m in tz['m']] == [True, True, False, False] and [m['fits'] for m in tz['m']] == [18 * params(SKU[k]) <= 80e9 for _, k in want])
    def sig3(b):   # GB to 3 significant digits, half up (R6.18): 2.47, 16.1, 145, 1,270
        v = b / 1e9
        if v >= 1000:
            q = 10 ** (math.floor(math.log10(v)) - 2); return f'{int(math.floor(v / q + 0.5) * q):,}'
        d = 0 if v >= 100 else 1 if v >= 10 else 2
        return f'{math.floor(v * 10 ** d + 0.5) / 10 ** d:,.{d}f}'
    exp = {k: (sig3(2 * params(SKU[m])), sig3(18 * params(SKU[m]))) for k, m in want}
    check('tile: sizes to 3 significant digits (R6.18): 2.47 / 22.2, 6.43 / 57.8, 16.1 / 145, 141 / 1,270 GB', exp == {'1B': ('2.47', '22.2'), '3B': ('6.43', '57.8'), '8B': ('16.1', '145'), '70B': ('141', '1,270')}, exp)
    rd = [H(f"()=>window.__tzTmem.read({i})") for i in range(4)]
    check('tile: each sentence names its stop and its two sizes', all(f'{k} model' in r and f'needs {a} GB' in r and f'needs {t} GB' in r for r, (k, _), (a, t) in zip(rd, want, [exp[k] for k, _ in want])), rd)
    check('tile: the 8B sentence agrees with the title and the lab (16.1 GB of weights, 145 GB of model states)', 'An 8B model' in rd[2] and '16.1 GB' in rd[2] and '145 GB' in rd[2] and 'more than one 80 GB GPU holds' in rd[2], rd[2])
    hp = [H(f"()=>window.__tzTmem.help({i})") for i in range(4)]
    check('tile: the hover says what the chart shows, with the tile’s own numbers (8.03B, 16.1 GB, 145 GB), no exact value',
          '8.03B parameters' in hp[2] and '16.1 GB' in hp[2] and '145 GB' in hp[2] and 'dashed line' in hp[2]
          and not any(re.search(r'\d,\d{3},\d{3}|\d{2,}\.\d\d GB|[Ee]xactly|精确', h) for h in hp), hp)
    # ---- R1 (the owner's review of batch 8): the result shows the reader's plan, the fastest, and the plans that pass
    jp = H("()=>window.__tmem.chPass()")
    check('R1: the closed list holds exactly the 2 plans that pass, fastest first', [(e['opt'], e['rc'], e['mb']) for e in jp] == [('adam8', 0, 2), ('adam8', 1, 4)], jp)
    def one_change(o, rc, mb):
        me = [q for q in plans if (q[0], q[1], q[2]) == (o, rc, mb)][0]; c = []
        for q in plans:
            d = sum(x != y for x, y in zip(q[:3], (o, rc, mb)))
            if d == 1 and q[6] and (not me[6] or q[4] < me[4] - 1e-9): c.append(q)
        c.sort(key=lambda q: (q[4], q[3]))
        return (c[0][0], c[0][1], c[0][2]) if c else None
    oc_bad = []
    for q in plans:
        j = H("a=>window.__tmem.chOneChange(a[0],a[1],a[2])", [q[0], q[1], q[2]])
        got = (j['e']['opt'], j['e']['rc'], j['e']['mb']) if j else None
        if got != one_change(q[0], q[1], q[2]): oc_bad.append((q[:3], got, one_change(q[0], q[1], q[2])))
    check('R1: the single change that helps most, for all 16 plans (engine = derivation)', not oc_bad, oc_bad)
    check('R1: from the initial setup no single change meets the goal; from 8-bit AdamW with 1 at a time, 2 at a time does (the fastest plan)',
          one_change('adamw', 0, 1) is None and one_change('adam8', 0, 1) == ('adam8', 0, 2) and one_change('adam8', 1, 2) == ('adam8', 0, 2) and one_change('adam8', 1, 4) is None and one_change('adam8', 1, 8) == ('adam8', 1, 4))
    # the main-gap sentence: which part is larger when a plan does not fit
    gap = {}
    for (o, rc, mb) in (('adamw', 0, 1), ('adam8', 0, 4), ('adam8', 1, 8)):
        st_, kp_, lg_, pk_ = peak('1b', o, CH['seq'], mb, rc); gap[(o, rc, mb)] = 'states' if st_ >= kp_ + lg_ else 'acts'
    check('R1: the gap sentence blames the model states for AdamW 1 at a time (22.2 GB) and 8-bit with recompute 8 at a time, the activations for 8-bit 4 at a time (16.3 GB)',
          gap == {('adamw', 0, 1): 'states', ('adam8', 0, 4): 'acts', ('adam8', 1, 8): 'states'} and g1(params(SKU['1b']) * 18) == '22.2'
          and g1(sum(peak('1b', 'adam8', 2048, 4, 0)[1:3])) == '16.3', gap)
    # ---- the Connect card's code, run here
    code = H("()=>document.getElementById('tmem-code').textContent")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): exec(code, {})
    got = [float(x) for x in buf.getvalue().split()]
    wants = [float(x) for x in re.findall(r'#\s*([\d.]+)', code)]
    check('Connect code: prints 168.7 and 99.5, as its comments say', [round(x, 1) for x in got] == wants == [168.7, 99.5], (got, wants))
    # ---- G7 (the owner's message of 2026-10-08, from the model-parallel finding): walk every pick of the challenge, in both
    # languages, and check that a recommended single change makes the value the result shows strictly better at the precision shown
    def g7_walk(js):
        res = {}
        for lang in ('en', 'zh'):
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(150)
            res[lang] = pg.evaluate(js)
        pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(150)
        return res
    g7 = g7_walk("""()=>{const Z=window.__tmem; const R=/^The one change that helps most is|^只改一项的话/; const f=s=>parseFloat(String(s).replace(/,/g,''));
      return Z.chAll().map(e=>{const t=Z.chNextText(e.opt,e.rc,e.mb), s=Z.chShown(e.opt,e.rc,e.mb), o=Z.chOneChange(e.opt,e.rc,e.mb), rec=R.test(t); let ok=true, why='';
        if(rec){if(!o){ok=false; why='a sentence without an engine change';} else {const r=Z.chShown(o.e.opt,o.e.rc,o.e.mb);
          if(!r.ok){ok=false; why='the change misses the goal';}
          else if(!s.fits){ok=f(r.peak)<f(s.peak); why='peak '+s.peak+' -> '+r.peak;}
          else {ok=f(r.t)<f(s.t)&&t.includes(r.t); why='time '+s.t+' -> '+r.t;}}}
        if(/about 0[ .]|约 0[ 。]|less than 1%|不到 1%/.test(t)){ok=false; why+=' (a zero or under-1% gain)';}
        return {pick:[e.opt,e.rc,e.mb],rec,ok,why,t};});}""")
    for lang in ('en', 'zh'):
        rows = g7[lang]; bad = [r for r in rows if not r['ok']]
        check(f'G7 {lang}: all 16 plans walked; every recommended single change shows a strictly better time (or makes the plan fit)', len(rows) == 16 and not bad, bad[:3])
    check('G7: the walk meets both kinds of sentence (a single change that helps, and none that does)',
          all(any(r['rec'] for r in g7[l]) and any(r['t'] and not r['rec'] for r in g7[l]) for l in ('en', 'zh')), [r['t'] for r in g7['zh']][:4])
    check('no page errors', not errs, errs[:3])
    b.close()
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails if fails else 'none')
sys.exit(1 if fails else 0)
