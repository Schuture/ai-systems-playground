"""Numbers test for the number-formats lab (CONTRACT §6): every number on the page against an independent Python derivation.

- each format's largest number, smallest normal, smallest subnormal and gap at 1, from its bit layout by enumerating every bit pattern
  (exact fractions); FP32 and FP16 also against numpy's finfo;
- a reference rounding (nearest representable value, ties to the even code, overflow past the largest code: Inf for IEEE formats, no
  code for E4M3 / E2M1), FP32 and FP16 also against numpy's casts; the lab's quant() against it at every value stop, for every format,
  with and without every block scale, and at 1.0 + every update stop;
- the guided parts' facts: 0.02 in each format, 1e‑8 lost in FP16 and kept in BF16, which updates BF16 and FP16 lose, the four fixes;
- the challenge (Llama 2 7B late in pretraining: lr 3e-5 = 10% of the 3e-4 peak, Adam's step ≈ lr on weights of 0.02 and gains of 1.0):
  each recipe's checks, bytes and extra parts, recomputed; the best answer (BF16 + FP32 master copy);
- FP4 overflow clamps to ±6 (OCP MX); Scenario 4's MX power-of-two scale (0.02 in a block whose largest is 0.2);
- the capability card's answers (E5M2's smallest number, the smallest update a BF16 weight of 4.0 keeps);
- the drill for 30 seeds, the home tile at every stop;
- the minimal implementation: its code runs and prints the output written under it;
- the shared names (Llama‑3.1‑8B written with the no-break hyphen); no throughput figure without its FP32 baseline (review A11).
Run: python3 tests/number-formats_numbers.py"""
import pathlib, re, subprocess, sys, warnings
from bisect import bisect_left
from fractions import Fraction as F
import numpy as np
from playwright.sync_api import sync_playwright

warnings.filterwarnings('ignore')
P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/number-formats-numbers.html', '--only', 'number-formats'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'number-formats-numbers-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'number-formats-numbers.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'number-formats-numbers-wrapped.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


# ---------- the formats from their bit layouts ----------
LAY = {'fp32': (8, 23, 127, 'ieee'), 'bf16': (8, 7, 127, 'ieee'), 'fp16': (5, 10, 15, 'ieee'), 'e4m3': (4, 3, 7, 'fn'), 'e5m2': (5, 2, 15, 'ieee'), 'fp4': (2, 1, 1, 'all')}


def table(fid):
    """every finite non-negative value with its code, plus the first code past the largest (where a value overflows)"""
    e, m, b, k = LAY[fid]
    vals = []
    for E in range(2 ** e):
        for M in range(2 ** m):
            if k == 'ieee' and E == 2 ** e - 1: continue
            if k == 'fn' and E == 2 ** e - 1 and M == 2 ** m - 1: continue
            v = F(M, 2 ** m) * F(2) ** (1 - b) if E == 0 else (1 + F(M, 2 ** m)) * F(2) ** (E - b)
            vals.append((v, (E << m) | M))
    if k == 'ieee': past = (F(2) ** (2 ** e - 1 - b), (2 ** e - 1) << m)
    elif k == 'fn': past = ((2 - F(1, 2 ** m)) * F(2) ** (2 ** e - 1 - b), 2 ** (e + m) - 1)
    else: past = (F(2) ** (2 ** e - b), 2 ** (e + m))
    assert all(vals[i][0] < vals[i + 1][0] for i in range(len(vals) - 1))
    return vals, past


TAB = {f: table(f) for f in LAY if f != 'fp32'}


def props(fid):
    if fid == 'fp32':
        fi = np.finfo(np.float32)
        return {'max': float(fi.max), 'minN': float(fi.tiny), 'minS': float(np.nextafter(np.float32(0), np.float32(1))), 'gap1': float(fi.eps)}
    vals, _ = TAB[fid]; m = LAY[fid][1]
    one = [v for v, c in vals].index(1)
    return {'max': float(vals[-1][0]), 'minN': float(vals[1 << m][0]), 'minS': float(vals[1][0]), 'gap1': float(vals[one + 1][0] - 1)}


def ref(x, fid):
    """(value, kind) of x stored in fid: nearest, ties to the even code; float64 in, exact"""
    if x == 0: return 0.0, 'zero'
    if fid == 'fp32':
        v = float(np.float32(x))
        if np.isinf(v): return (float('inf') if x > 0 else float('-inf')), 'over'
        if v == 0: return 0.0, 'lost'
        return v, ('sub' if abs(v) < 2.0 ** -126 else 'norm')
    vals, past = TAB[fid]; ext = vals + [past]; a = F(abs(x)); vs = [v for v, c in ext]
    i = bisect_left(vs, a)
    if i < len(vs) and vs[i] == a: j = i
    elif i >= len(vs): j = len(vs) - 1
    else:
        lo, hi = ext[i - 1], ext[i]; dl, dh = a - lo[0], hi[0] - a
        j = i - 1 if dl < dh else (i if dh < dl else (i - 1 if lo[1] % 2 == 0 else i))
    if j == len(ext) - 1 and ext[j] is past:
        if LAY[fid][3] == 'all': mx = float(vals[-1][0]); return (mx if x > 0 else -mx), 'over'   # FP4: OCP MX clamps to ±6
        return (float('inf') if x > 0 else float('-inf')) if LAY[fid][3] == 'ieee' else float('nan'), 'over'
    v = float(ext[j][0])
    if v == 0: return 0.0, 'lost'
    kind = 'sub' if ext[j][1] < (1 << LAY[fid][1]) else 'norm'
    return (v if x > 0 else -v), kind


def js(v):
    return {'Infinity': float('inf'), '-Infinity': float('-inf'), 'NaN': float('nan')}.get(v, v) if isinstance(v, str) else float(v)


def same(a, b):
    return (np.isnan(a) and np.isnan(b)) or a == b


# numpy cross-checks of the reference
for x in [0.02, 1e-8, 3e-8, 5e-8, 1e-7, 65504, 65520, 1e5, 6.5536e-4, 1 + 3e-4, 1 + 1e-3, 1.00048828125, 1.000732421875] + [float(k) * 2.0 ** -30 for k in range(1, 40)]:
    a, _ = ref(x, 'fp16'); b = float(np.float16(x))
    check(f'reference FP16 matches numpy at {x!r}', same(a, b), (a, b))

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width': 1200, 'height': 900})
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.evaluate("()=>{location.hash='number-formats'}"); pg.wait_for_timeout(400)
    C = pg.evaluate('()=>window.__nf.consts')
    # ---------- each format's numbers ----------
    for fid in LAY:
        pr, fm = props(fid), C['FM'][fid]
        for k in ('max', 'minN', 'minS', 'gap1'):
            check(f'{fid} {k}', js(fm[k]) == pr[k], (fm[k], pr[k]))
        check(f'{fid} bits', fm['bits'] == 1 + LAY[fid][0] + LAY[fid][1])
    check('FP16 largest 65,504, BF16 3.3895e38, E4M3 448, E5M2 57,344, E2M1 6', [props(f)['max'] for f in ('fp16', 'e4m3', 'e5m2', 'fp4')] == [65504.0, 448.0, 57344.0, 6.0] and abs(props('bf16')['max'] - 3.3895e38) < 1e34)
    check('FP16 against numpy finfo', props('fp16')['max'] == float(np.finfo(np.float16).max) and props('fp16')['minN'] == float(np.finfo(np.float16).tiny) and props('fp16')['gap1'] == float(np.finfo(np.float16).eps))
    # ---------- the slider stops ----------
    XS, US = C['XS'], C['US']
    check('value stops: 1-2-5 from 1e-10 to 5e5', len(XS) == 48 and XS[0] == 1e-10 and XS[-1] == 5e5 and XS[C['X_W']] == 0.02 and XS[C['X_G']] == 1e-8)
    check('update stops', US == [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2] and US[C['U_0']] == 3e-4)
    Q = pg.evaluate("xs=>{const out={}; for(const f of ['fp32','bf16','fp16','e4m3','e5m2','fp4']) out[f]=xs.map(x=>window.__nf.quant(x,f)); return out}", XS)
    bad_ = []
    for fid in LAY:
        for x, r in zip(XS, Q[fid]):
            v, k = ref(x, fid)
            if not (same(js(r['v']), v) and r['kind'] == k): bad_.append((fid, x, r['v'], r['kind'], v, k))
    check(f'quant() at every value stop and format ({len(XS) * 6})', not bad_, bad_[:6])
    # bits decode back to the stored value
    dec = []
    for fid in LAY:
        e, m, bb, k = LAY[fid]
        for x, r in zip(XS, Q[fid]):
            if r['kind'] in ('sub', 'norm'):
                val = (F(r['M'], 2 ** m) * F(2) ** (1 - bb)) if r['E'] == 0 else (1 + F(r['M'], 2 ** m)) * F(2) ** (r['E'] - bb)
                if float(val) != abs(js(r['v'])) or not (0 <= r['M'] < 2 ** m and 0 <= r['E'] < 2 ** e): dec.append((fid, x, r))
    check('the bit boxes decode to the stored value', not dec, dec[:4])
    # block scales on the FP8 / FP4 rows
    S = pg.evaluate("a=>{const out=[]; for(const blk of a.b) for(const f of ['e4m3','e5m2','fp4','bf16']) for(const x of a.x) out.push([blk,f,x,window.__nf.store(x,f,blk)]); return out}", {'b': C['BLKS'], 'x': XS})
    bad_ = []
    for blk, fid, x, r in S:
        if blk and fid != 'bf16':
            s = abs(x) * blk / props(fid)['max']; v, k = ref(x / s, fid); v = v * s
        else:
            v, k = ref(x, fid)
        if not (same(js(r['v']), v) and r['kind'] == k): bad_.append((blk, fid, x, r['v'], v))
    check(f'store() with every block scale ({len(S)})', not bad_, bad_[:6])
    # 1.0 + update
    U = pg.evaluate("us=>us.map(u=>['fp32','bf16','fp16'].map(f=>window.__nf.quant(1+u,f).v))", US)
    check('1.0 + update at every stop', all(same(js(r[i]), ref(1 + u, f)[0]) for u, r in zip(US, U) for i, f in enumerate(('fp32', 'bf16', 'fp16'))), U)
    lostB = [u for u in US if ref(1 + u, 'bf16')[0] == 1]; lostH = [u for u in US if ref(1 + u, 'fp16')[0] == 1]
    check('BF16 loses 1e-5 … 3e-3, FP16 loses 1e-5 … 3e-4 (part 4)', lostB == [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3] and lostH == [1e-5, 3e-5, 1e-4, 3e-4], (lostB, lostH))
    # ---------- the guided parts' facts ----------
    check('0.02: BF16 0.02001953125, FP16 0.0200042724609375, FP32 0.019999999552965164, E4M3 0.01953125, FP4 0 (lost)',
          [ref(0.02, f)[0] for f in ('bf16', 'fp16', 'fp32', 'e4m3', 'e5m2')] == [0.02001953125, 0.0200042724609375, 0.019999999552965164, 0.01953125, 0.01953125] and ref(0.02, 'fp4')[1] == 'lost')
    check('1e-8: FP16 0 (lost), BF16 1.0011717677e-8, FP32 9.99999994e-9', ref(1e-8, 'fp16')[1] == 'lost' and abs(ref(1e-8, 'bf16')[0] - 1.0011717677e-8) < 1e-18 and abs(ref(1e-8, 'fp32')[0] - 9.99999994e-9) < 1e-17)
    check('1e5 overflows FP16, BF16 stores 99,840; 500 overflows E4M3, E5M2 512', ref(1e5, 'fp16')[1] == 'over' and ref(1e5, 'bf16')[0] == 99840 and ref(500, 'e4m3')[1] == 'over' and ref(500, 'e5m2')[0] == 512)
    check('the prediction’s right answer is “exactly 0”', C['PRED_RIGHT'] == 'zero' and ref(1e-8, 'fp16')[0] == 0)
    fmt = pg.evaluate("()=>[window.__nf.fmt.fv(1.0011717677e-8),window.__nf.fmt.fv(0.02001953125),window.__nf.fmt.fv(0.0200042724609375),window.__nf.fmt.fv(99840),window.__nf.fmt.fin(1e-8),window.__nf.fmt.fin(500000),window.__nf.fmt.fv(1/0)]")
    check('display: 1.001e‑8, 0.020019531, 0.020004272, 99,840, 1e‑8, 500,000, Inf', fmt == ['1.001e‑8', '0.020019531', '0.020004272', '99,840', '1e‑8', '500,000', 'Inf'], fmt)

    def rec(fid_w, fid_g, ls, ms, T):
        f32 = lambda v: float(np.float32(v))
        g, gk = ref(T['g'] * ls, fid_g)
        lost = gk in ('lost', 'over') or g == 0
        gUsed = 0 if lost else f32(f32(g) / ls)
        _, bk = ref(T['gmax'] * ls, fid_g)
        w0 = f32(T['w']) if ms else ref(T['w'], fid_w)[0]
        w1 = f32(w0 + f32(T['u'])) if ms else ref(w0 + T['u'], fid_w)[0]
        if T.get('w2') is not None:
            v0 = f32(T['w2']) if ms else ref(T['w2'], fid_w)[0]
            v1 = f32(v0 + f32(T['u'])) if ms else ref(v0 + T['u'], fid_w)[0]
        else: v0, v1 = 0, 1
        bits = {'fp32': 32, 'bf16': 16, 'fp16': 16}
        sixteen = bits[fid_w] == 16 and bits[fid_g] == 16
        return {'lost': lost, 'gUsed': gUsed, 'over': bk == 'over', 'w0': w0, 'w1': w1, 'kept': w1 != w0, 'sixteen': sixteen,
                'bytes': bits[fid_w] // 8 + bits[fid_g] // 8 + (4 if ms else 0), 'extras': (1 if ls > 1 else 0) + (1 if ms else 0),
                'kept2': v1 != v0, 'ok': sixteen and bk != 'over' and not lost and w1 != w0 and v1 != v0}
    RCP = {'fp32': ('fp32', 'fp32', 1, 0), 'fp16': ('fp16', 'fp16', 1, 0), 'fp16ls': ('fp16', 'fp16', 65536, 0), 'fp16m': ('fp16', 'fp16', 1, 1),
           'fp16lsm': ('fp16', 'fp16', 65536, 1), 'bf16': ('bf16', 'bf16', 1, 0), 'bf16m': ('bf16', 'bf16', 1, 1)}
    T5 = {'g': 1e-8, 'gmax': 1e-3, 'w': 1, 'u': 3e-4}
    exp5 = {'fp16': (True, False), 'fp16ls': (False, False), 'fp16m': (True, True), 'fp16lsm': (False, True)}
    for id_, (lost, kept) in exp5.items():
        e = pg.evaluate("a=>window.__nf.recipe(a[0],a[1])", [id_, T5]); r = rec(*RCP[id_], T5)
        check(f'part 5 {id_}: gradient {"lost" if lost else "kept"}, update {"kept" if kept else "lost"}', e['lost'] == r['lost'] == lost and e['kept'] == r['kept'] == kept and js(e['gUsed']) == r['gUsed'] and js(e['w1']) == r['w1'], (e['lost'], e['kept'], e['gUsed'], r))
    check('part 5: loss scaling stores 1e-8 × 65,536 as 6.5517e-4 in FP16 and gives back 9.997e-9', abs(ref(1e-8 * 65536, 'fp16')[0] - 6.5517e-4) < 1e-8 and abs(rec(*RCP['fp16ls'], T5)['gUsed'] - 9.997e-9) < 1e-12)
    check('part 5 levers in order', C['LV5'] == ['fp16', 'fp16ls', 'fp16m', 'fp16lsm'] and C['STEP_IDS'] == ['mission', 'predict', 'range', 'prec', 'fix', 'remember'])
    # ---------- the challenge ----------
    CHT = C['CHT']
    check('challenge tensors: weight 0.02 and gain 1.0, update 3e-5 (10% of the 3e-4 peak), gradients 1e-8 to 1e-3', CHT['w'] == 0.02 and CHT['w2'] == 1 and CHT['u'] == 3e-5 and C['LR_PEAK'] == 3e-4 and CHT['g'] == 1e-8 and CHT['gmax'] == 1e-3, CHT)
    check('the premise: 3e-5 is below half of BF16’s gap at 0.02 (2^-14) and above half of FP16’s (2^-17); below half of both gaps at 1.0', 2.0 ** -17 < 3e-5 < 2.0 ** -14 and 3e-5 < 2.0 ** -11)
    want = {'fp32': (False, 8, 0), 'fp16': (False, 4, 0), 'fp16lsm': (True, 8, 2), 'bf16': (False, 4, 0), 'bf16m': (True, 8, 1)}
    for id_ in C['CHO']:
        e = pg.evaluate("id=>window.__nf.chEval(id)", id_); r = rec(*RCP[id_], CHT)
        check(f'challenge {id_}: ok {want[id_][0]}, {want[id_][1]} bytes, {want[id_][2]} extra parts', (e['ok'], e['bytes'], e['extras']) == (r['ok'], r['bytes'], r['extras']) == want[id_] and e['lost'] == r['lost'] and e['kept'] == r['kept'] and e['kept2'] == r['kept2'] and e['over'] == r['over'], (e['ok'], e['bytes'], e['extras'], r))
    best = pg.evaluate('()=>window.__nf.chBest()')
    passing = sorted([i for i in C['CHO'] if rec(*RCP[i], CHT)['ok']], key=lambda i: (rec(*RCP[i], CHT)['extras'], rec(*RCP[i], CHT)['bytes']))
    check('challenge best: BF16 + FP32 master copy (fewest extra parts)', best['id'] == passing[0] == 'bf16m', (best, passing))
    f16, b16 = rec(*RCP['fp16'], CHT), rec(*RCP['bf16'], CHT)
    check('challenge: all FP16 loses the gradient and the gain’s update but keeps the update on 0.02; all BF16 loses both updates, not the gradient', f16['lost'] and f16['kept'] and not f16['kept2'] and not b16['lost'] and not b16['kept'] and not b16['kept2'], (f16, b16))
    check('challenge: FP16 + loss scaling alone would still lose the gain’s update (so the tier-1 text holds)', not rec(*RCP['fp16ls'], CHT)['ok'] and not rec(*RCP['fp16ls'], CHT)['kept2'])
    # FP4 clamps to 6 (OCP MX), Scenario 4's MX scale
    q4 = pg.evaluate("()=>[window.__nf.quant(500,'fp4'),window.__nf.mx(0.02,'fp4',10),window.__nf.mx(0.02,'e4m3',10),window.__nf.store(0.02,'fp4',10),window.__nf.store(0.02,'e4m3',10)]")
    check('FP4: 500 clamps to 6 (overflow, 98.8% off), bits 0 11 1', js(q4[0]['v']) == 6.0 and q4[0]['kind'] == 'over' and q4[0]['E'] == 3 and q4[0]['M'] == 1 and round(abs(6 - 500) / 500 * 100, 1) == 98.8, q4[0])
    mx4 = ref(0.02 / 2.0 ** -5, 'fp4')[0] * 2.0 ** -5; mx8 = ref(0.02 / 2.0 ** -11, 'e4m3')[0] * 2.0 ** -11
    check('Scenario 4: MX scale 2^-5 (FP4) and 2^-11 (E4M3) for a block whose largest is 0.2: 0.015625 (21.9% off), 0.01953125', js(q4[1]['v']) == mx4 == 0.015625 and js(q4[2]['v']) == mx8 == 0.01953125 and round((0.02 - mx4) / 0.02 * 100, 1) == 21.9, (q4[1], q4[2]))
    check('Scenario 4: the exact scale gives 0.016666667 (FP4) and 0.019642857 (E4M3)', abs(js(q4[3]['v']) - 0.2 / 6 * ref(0.02 / (0.2 / 6), 'fp4')[0]) < 1e-15 and abs(js(q4[4]['v']) - 0.019642857142857) < 1e-12, (q4[3]['v'], q4[4]['v']))
    # ---------- the capability card ----------
    ex = pg.evaluate('()=>window.__nf.exits()')
    t1, t2 = ex['transfer']
    check('transfer 1: E5M2’s smallest number ≈ 1.5e-5 (2^-16)', props('e5m2')['minS'] == 2.0 ** -16 and '1.5e' in t1['o'][t1['a']].replace('‑', '-'), t1['o'][t1['a']])
    gap4 = ref(4.0 + 0.015625, 'bf16')[0] == 4.0 and ref(4.0 + 0.0157, 'bf16')[0] > 4.0 and ref(4.01, 'bf16')[0] == 4.0
    check('transfer 2: a BF16 weight of 4.0 keeps updates above 0.015625 (≈ 0.016)', gap4 and '0.016' in t2['o'][t2['a']], t2['o'][t2['a']])
    check('explain: the right options are the range and the precision ones', ex['explain'][0]['a'] == 0 and 'exponent' in ex['explain'][0]['o'][0] and ex['explain'][1]['a'] == 0 and '0.0078' in ex['explain'][1]['o'][0])
    check('practice ids', [x['id'] for x in ex['practice']] == ['cs-fundamentals', 'matmul-roofline'])
    # ---------- drill and tile ----------
    dr = pg.evaluate('()=>Array.from({length:30},(_,i)=>window.__nf.drill(i+1))')
    check('drill: 30 seeds, every answer from the reference rounding', all(same(js(d['r']['v']), ref(d['x'], d['f'])[0]) for d in dr) and len({(d['f'], d['x']) for d in dr}) > 15, [(d['f'], d['x'], d['r']['v']) for d in dr[:4]])
    tz = pg.evaluate('()=>window.__tzNf?{xs:window.__tzNf.xs,start:window.__tzNf.start,m:window.__tzNf.xs.map((_,i)=>window.__tzNf.model(i))}:null')
    check('tile: the same stops, opens at 0.02', tz and tz['xs'] == XS and tz['xs'][tz['start']] == 0.02, tz and tz['start'])
    if tz:
        badt = [(m['x'], r['id'], r['v']) for m in tz['m'] for r in m['rows'] if not same(js(r['v']), ref(m['x'], r['id'])[0])]
        check('tile: every stored value from the reference rounding', not badt, badt[:5])
        rd = pg.evaluate('()=>[window.__tzNf.read(window.__tzNf.xs.indexOf(1e-8)),window.__tzNf.read(window.__tzNf.xs.indexOf(0.02)),window.__tzNf.read(window.__tzNf.xs.indexOf(100000))]')
        rd = [re.sub('<[^>]+>', '', x).replace('‑', '-') for x in rd]
        check('tile sentences: 1e-8 → 0 in FP16 and E4M3, BF16 1.001e-8; 0.02 → E4M3 rounds most; 1e5 overflows FP16 and E4M3', 'becomes 0 in FP16 and FP8 E4M3' in rd[0] and '1.001e-8' in rd[0] and 'FP8 E4M3 rounds it most, to 0.019531' in rd[1] and 'overflows FP16 and FP8 E4M3' in rd[2] and '99,840' in rd[2], rd)
        check('tile sentences ≤ 25 words', all(len(x.split()) <= 25 for x in rd), [len(x.split()) for x in rd])
        r3 = re.sub('<[^>]+>', '', pg.evaluate('()=>window.__tzNf.read(window.__tzNf.xs.indexOf(0.001))'))
        e3 = abs(ref(0.001, 'e4m3')[0] - 0.001) / 0.001
        check('tile: 0.001 says E4M3 keeps only a rough 0.001953125 (95.3% off)', 'keeps only a rough 0.001953125 (95.3% off)' in r3 and round(e3 * 100, 1) == 95.3 and len(r3.split()) <= 25, r3)
    # ---------- the minimal implementation runs and prints what it says ----------
    code = pg.evaluate('()=>window.__nf.code')
    lines = code.split('\n'); expect = [l[2:] for l in lines if l.startswith('# ')]
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30).stdout.strip().split('\n')
    check('the code prints the output written under it', out == expect, (out, expect))
    check('the code: ≤ 40 lines, ≤ 74 characters', len(lines) <= 40 and max(len(l) for l in lines) <= 74, (len(lines), max(len(l) for l in lines)))
    # ---------- shared numbers and names ----------
    src = (P / 'labs' / 'number-formats.js').read_text()
    check('no H100 throughput figure without an FP32 baseline (review A11)', '989.5' not in src)
    check('Llama‑3.1‑8B with the no-break hyphen, never an ASCII one', 'Llama‑3.1‑8B' in src and 'Llama-3.1-8B' not in src)
    check('GradScaler: 65,536, ×0.5, 2,000 steps', 'init_scale=2.0**16' in src and 'backoff_factor=0.5' in src and 'growth_interval=2000' in src)
    check('no page errors', not errs, errs[:3])
    b.close()
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails if fails else 'none')
sys.exit(1 if fails else 0)
