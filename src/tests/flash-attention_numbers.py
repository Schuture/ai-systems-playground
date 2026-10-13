"""Golden numbers for the flash-attention lab (guide R9.2, CONTRACT §6).

Everything is derived here independently of the JS engine:
- the toy: Q, K, V from mulberry32 (ported, since it defines the example), seeded with 9 × 7919 + 17, uniform in [−1.5, 1.5] rounded
  to 0.1 (JS Math.round); standard attention with NumPy (S = QKᵀ ÷ √d, softmax by row, O = PV); FlashAttention-2's loop with the
  online softmax written again here; the numbers moved: standard 4Nd + 4N², tiled 2Nd + (N/Br)·2Nd;
- HBM traffic per head in BF16 (the papers' count): standard 2·(2·Nq·d + 2·N·d + 4·Nq·N), FlashAttention 2·(2·Nq·d + ⌈Nq/Br⌉·2·N·d);
- SRAM for one tile pair: 2·Br·d + 4·Bc·d + 4·Br·Bc + 4·Br·d + 8·Br bytes against 227 KiB = 232,448 bytes;
- the times at H100 peak: 989.5 TFLOP/s, 3.35 TB/s, 3.9 T exps/s.
Checks: the engine (window.__fa) against this derivation (toy, every traffic and SRAM case, times, the challenge grid, its pass set and
minimal answer, the capability layer's answers, the drill), the home tile (window.__tzFa) against the same formulas, the Connect
card's code (run here with NumPy, at most 40 lines of at most 74 characters), and the numbers the page shows in every part, the
scenarios and the challenge.
Run: python3 tests/flash-attention_numbers.py"""
import math, pathlib, subprocess, sys
import numpy as np
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/flash-attention-num.html', '--only', 'flash-attention'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'flash-attention-num-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'flash-attention-num.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'flash-attention-num-wrapped.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


# ---------- the derivation
M32 = 0xFFFFFFFF
def imul(a, b): return ((a & M32) * (b & M32)) & M32
def mulberry(a):
    st = [a & M32]
    def f():
        st[0] = (st[0] + 0x6D2B79F5) & M32; x = st[0]
        t = imul(x ^ (x >> 15), 1 | x)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296
    return f


N, D, B = 12, 4, 4
r = mulberry(9 * 7919 + 17)
jsround = lambda x: math.floor(x + 0.5)
mats = [[[jsround((r() * 3 - 1.5) * 10) / 10 for _ in range(D)] for _ in range(N)] for _ in range(3)]
Q, K, V = (np.array(m) for m in mats)
S = Q @ K.T / math.sqrt(D)
Pm = np.exp(S - S.max(1, keepdims=True)); Pm /= Pm.sum(1, keepdims=True)
O_std = Pm @ V
steps, O_til = [], np.zeros((N, D))
for i in range(N // B):
    m = np.full(B, -np.inf); l = np.zeros(B); acc = np.zeros((B, D)); q = Q[i * B:(i + 1) * B]
    for j in range(N // B):
        s = q @ K[j * B:(j + 1) * B].T / math.sqrt(D); mn = np.maximum(m, s.max(1))
        f = np.where(np.isinf(m), 0.0, np.exp(m - mn)); p = np.exp(s - mn[:, None])
        l = l * f + p.sum(1); acc = acc * f[:, None] + p @ V[j * B:(j + 1) * B]
        steps.append({'i': i, 'j': j, 'tmax': s.max(1).copy(), 'mOld': m.copy(), 'm': mn.copy(), 'f': f.copy(), 'l': l.copy()}); m = mn
    O_til[i * B:(i + 1) * B] = acc / l[:, None]
STD_TOTAL, FA_TOTAL, SP = 4 * N * D + 4 * N * N, 2 * N * D + (N // B) * 2 * N * D, 4 * N * N
KiB, MiB, GiB = 1024, 1024 ** 2, 1024 ** 3


def traffic(Nn, d, Br, Nq=None):
    Nq = Nn if Nq is None else Nq
    std = 2 * (2 * Nq * d + 2 * Nn * d + 4 * Nq * Nn); fa = 2 * (2 * Nq * d + math.ceil(Nq / Br) * 2 * Nn * d)
    return {'std': std, 'fa': fa, 's': 2 * Nq * Nn, 'passes': math.ceil(Nq / Br)}


def sram(Br, Bc, d): return 2 * Br * d + 4 * Bc * d + 4 * Br * Bc + 4 * Br * d + 8 * Br


BUDGET = 227 * 1024
fx = lambda v, k: f'{v:,.{k}f}'
def fB(b, u=None):
    u = u or ('G' if b >= 10 * GiB else 'M')
    if u == 'G': return fx(b / GiB, 1) + ' GiB'
    m = b / MiB
    return (fx(m, 2) if m < 10 else f'{round(m):,}') + ' MiB'
fK = lambda b: (f'{round(b / KiB):,}' if b % KiB == 0 else fx(b / KiB, 2)) + ' KiB'
NS = [1024 * 2 ** k for k in range(8)]
TILES = [32, 64, 128, 256]

with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(250); pg.evaluate("()=>{location.hash='flash-attention'}"); pg.wait_for_timeout(450)
    ev = lambda js, *a: pg.evaluate(js, *a)
    # ---- the toy
    toy = ev("()=>window.__fa.toy()")
    check('toy: Q, K, V from the seed', toy['Q'] == mats[0] and toy['K'] == mats[1] and toy['V'] == mats[2], (toy['Q'][0], mats[0][0]))
    check('toy: standard output', np.abs(np.array(toy['stdO']) - O_std).max() < 1e-12)
    check('toy: tiled output', np.abs(np.array(toy['tilO']) - O_til).max() < 1e-12)
    md = float(np.abs(np.array(toy['stdO']) - np.array(toy['tilO'])).max())
    check('toy: largest difference is the page’s own, about 1.7e-16', toy['maxdiff'] == md and 1e-17 < md < 1e-15, (toy['maxdiff'], md))
    check('toy: numbers moved 768 / 384, S and P 576', (toy['stdTotal'], toy['faTotal'], toy['spToy']) == (STD_TOTAL, FA_TOTAL, SP) == (768, 384, 576))
    check('toy: standard steps', [s['n'] for s in toy['stdSteps']] == [0, 2 * N * D, N * N, N * N, N * N, N * N + N * D, N * D])
    ok = len(toy['steps']) == 9
    for a, s in zip(toy['steps'], steps):
        ok &= a['i'] == s['i'] and a['j'] == s['j'] and np.allclose(a['tmax'], s['tmax'], atol=1e-12) and np.allclose(a['m'], s['m'], atol=1e-12) and np.allclose(a['f'], s['f'], atol=1e-12) and np.allclose(a['l'], s['l'], atol=1e-12)
    check('toy: every tile’s max, m, rescale and ℓ', ok)
    row0 = [s for s in steps if s['i'] == 0]; k = next(j for j in range(1, 3) if row0[j]['f'][0] < 1)
    rise = (row0[k]['mOld'][0], row0[k]['m'][0], row0[k]['f'][0])
    check('toy: row 1 rises at K/V tile 3 by a factor of about 0.19', k == 2 and f'{rise[2]:.2f}' == '0.19', rise)
    # ---- real size
    ok, bad = True, []
    for Nn in NS:
        for d in (64, 128, 256):
            for Br in TILES:
                for Nq in (None, 1):
                    a = ev("a=>window.__fa.traffic(...a)", [Nn, d, Br, Nq]); e = traffic(Nn, d, Br, Nq)
                    if (a['std'], a['fa'], a['s'], a['passes']) != (e['std'], e['fa'], e['s'], e['passes']): ok = False; bad.append((Nn, d, Br, Nq))
    check('traffic: 384 cases', ok, bad[:3])
    t8 = traffic(8192, 128, 128)
    check('8K, one head: S 128 MiB, standard 520 MiB, FlashAttention 260 MiB; a layer’s S 4 GiB', (fB(t8['s']), fB(t8['std']), fB(t8['fa']), t8['s'] * 32 / GiB) == ('128 MiB', '520 MiB', '260 MiB', 4.0))
    check('8K tile sizes: 32 → 1,028 MiB, 64 → 516, 256 → 132', [fB(traffic(8192, 128, x)['fa'], 'M') for x in TILES] == ['1,028 MiB', '516 MiB', '260 MiB', '132 MiB'])
    t128 = traffic(131072, 128, 128)
    check('128K: 128.1 / 64.1 GiB, S 32 GiB', (fB(t128['std']), fB(t128['fa']), t128['s'] / GiB) == ('128.1 GiB', '64.1 GiB', 32.0))
    check('every doubling: both about 4×', all(3.75 < traffic(2 * x, 128, 128)['std'] / traffic(x, 128, 128)['std'] < 4.01 and 3.75 < traffic(2 * x, 128, 128)['fa'] / traffic(x, 128, 128)['fa'] < 4.01 for x in NS[:-1]))
    ok = True
    for Br in TILES:
        for Bc in TILES:
            for d in (64, 128):
                a = ev("a=>window.__fa.sram(...a)", [Br, Bc, d]); ok &= a['total'] == sram(Br, Bc, d) and a['fits'] == (sram(Br, Bc, d) <= BUDGET)
    check('SRAM: 32 cases', ok)
    check('SRAM, square tiles at d 128: 44.25 / 96.50 / 225 / 578 KiB, only 256 too big', [fK(sram(x, x, 128)) for x in TILES] == ['44.25 KiB', '96.50 KiB', '225 KiB', '578 KiB'] and [sram(x, x, 128) <= BUDGET for x in TILES] == [True, True, True, False])
    tm = ev("()=>window.__fa.times(8192,128)")
    e = {'qk': 2 * 8192 ** 2 * 128 / 989.5e12, 'exp': 8192 ** 2 / 3.9e12, 'sp': 8 * 8192 ** 2 / 3.35e12}
    check('times at 8K: QKᵀ 17 µs, exps 17 µs, S and P 160 µs', all(abs(tm[k] - e[k]) < 1e-15 for k in e) and [round(e[k] * 1e6) for k in ('qk', 'exp', 'sp')] == [17, 17, 160], tm)
    # ---- the challenge
    grid = {(Br, Bc): (traffic(32768, 64, Br)['fa'], sram(Br, Bc, 64)) for Br in TILES for Bc in TILES}
    passing = {k for k, (t, m) in grid.items() if t <= 1.5 * GiB and m <= BUDGET}
    page_pass = {(Br, Bc) for Br in TILES for Bc in TILES if ev("a=>window.__fa.chEval(...a).ok", [Br, Bc])}
    check('challenge: the pass set is 256 × 32 and 256 × 64', passing == page_pass == {(256, 32), (256, 64)}, (passing, page_pass))
    best = ev("()=>window.__fa.chBest()")
    ch = lambda Br, Bc: (Br != 64) + (Bc != 64)
    exp_best = min(passing, key=lambda k: (ch(*k), grid[k][1]))
    check('challenge: the minimal answer is 256 × 64', (best['Br'], best['Bc']) == exp_best == (256, 64), best)
    check('challenge: 64 × 64 moves 4.01 GiB; 256 × 64 1.01 GiB in 178 KiB; 256 × 128 needs 258 KiB', (f"{grid[(64, 64)][0] / GiB:.2f}", f"{grid[(256, 64)][0] / GiB:.2f}", fK(grid[(256, 64)][1]), fK(grid[(256, 128)][1])) == ('4.01', '1.01', '178 KiB', '258 KiB'))
    # ---- capability layer
    ex = ev("()=>window.__fa.exits()")
    s8 = traffic(8192, 128, 128)['s']
    check('new case 1: a 64-head layer’s S at 8K is 8 GiB', 64 * s8 == 8 * GiB and '8 GiB' in ex['transferO'][0][ex['transferA'][0]], ex['transferO'][0])
    check('new case 2: S does not depend on the head size: 128 MiB', '128 MiB' in ex['transferO'][1][ex['transferA'][1]], ex['transferO'][1])
    check('new cases: the right answers sit at different places', ex['transferA'][0] != ex['transferA'][1], ex['transferA'])
    ridge = 989.5e12 / 3.35e12
    check('explain 2: 64 FLOPs per byte against a ridge of 295', round(ridge) == 295 and '64 FLOPs for each byte it moves from HBM' in ex['explainO'][1][ex['explainA'][1]] and '295' in ex['explainO'][1][ex['explainA'][1]], ex['explainO'][1][0][:200])
    # ---- the drill
    ok = True
    for sd in range(1, 30):
        q = ev("s=>window.__fa.drill(s)", sd); e = traffic(q['N'], q['d'], q['Br'])
        ok &= q['N'] in (2048, 4096, 16384, 32768) and q['d'] in (64, 128) and q['Br'] in (64, 128) and q['t']['std'] == e['std'] and q['t']['fa'] == e['fa']
    check('drill: 29 seeds', ok)
    # ---- the player's speeds (Yixiong: even Fast was too slow): a whole run, both paths, in at most 10 s at Fast
    sp = ev("()=>window.__fa.consts.SPEEDS")
    check('player: Fast plays a whole run in at most 10 s, Medium and Slow longer, in order', sp['fast'] <= 10000 and sp['fast'] < sp['med'] < sp['slow'] <= 30000, sp)
    # ---- the home tile
    ok = True
    for i in range(8):
        a = ev("i=>window.__tzFa.traffic(i)", i); e = traffic(NS[i], 128, 128)
        ok &= a['std'] == e['std'] and a['fa'] == e['fa'] and a['s'] == e['s']
    check('home tile: the same formulas at 8 lengths', ok)
    rd = ev("()=>window.__tzFa.read(3)")
    check('home tile at 8K: 520 MiB and 260 MiB', '520 MiB' in rd and '260 MiB' in rd, rd)
    # Yixiong's review: the bars must change length, not only their ratio. Both share one log axis, 2 MiB to 256 GiB (17 doublings)
    bars = [ev("i=>window.__tzFa.bars(i)", i) for i in range(8)]
    want = [tuple(math.log2(traffic(NS[i], 128, 128)[k] / (2 * 2 ** 20)) / 17 for k in ('std', 'fa')) for i in range(8)]
    check('home tile: both bars on one log axis, longer at every step, FlashAttention exactly one doubling (half) shorter',
          all(abs(b['std'] - w[0]) < 1e-9 and abs(b['fa'] - w[1]) < 1e-9 for b, w in zip(bars, want))
          and all(bars[i + 1]['std'] > bars[i]['std'] and bars[i + 1]['fa'] > bars[i]['fa'] for i in range(7))
          and all(0 < b['fa'] < b['std'] <= 1 and abs((b['std'] - b['fa']) * 17 - 1) < 1e-9 for b in bars), bars)
    ax_ = ev("()=>window.__tzFa.axis")
    check('home tile: the axis ticks 4 MiB, 64 MiB, 1 GiB, 16 GiB, 256 GiB sit at 1, 5, 9, 13 and 17 doublings', all(abs(t - k / 17) < 1e-9 for t, k in zip(ax_['ticks'], (1, 5, 9, 13, 17))), ax_)
    # ---- the Connect code, run here
    code = ev("()=>document.getElementById('fa-code').textContent")
    lines = code.split('\n')
    check('code: at most 40 lines of at most 74 characters', len(lines) <= 40 and max(len(x) for x in lines) <= 74, (len(lines), max(len(x) for x in lines)))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60)
    v = float(out.stdout.strip() or 'nan')
    check('code: runs and prints a difference below 1e-14', out.returncode == 0 and v < 1e-14, (out.stdout, out.stderr[-300:]))
    # ---- what the page shows
    txt = lambda sel: ev("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
    def go(i): ev(f"()=>document.querySelector('#fa-rail [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(250)
    o = txt('#fa-rail .step.current')
    check('part 1 shows 128 MiB per head (the 4 GiB per layer left part 1 in v6 S-2)', '128 MiB' in o, o[:300])
    ev("()=>document.querySelector('#fa-rail [data-nav=\"next\"]').click()"); pg.wait_for_timeout(200)
    ev("()=>document.querySelector('#fa-rail [data-pred=\"move\"]').click()"); pg.wait_for_timeout(200)
    go(2); ev("()=>document.querySelector('#fa-rail [data-nav=\"showme\"]').click()"); pg.wait_for_timeout(300)
    o = txt('#fa-rail [data-obs]')
    check('part 3 shows 768, 576, 160 µs, 17 µs', all(x in o for x in ('768', '576', '160 µs', '17 µs')), o[:400])
    go(3); ev("()=>document.querySelector('#fa-rail [data-nav=\"showme\"]').click()"); pg.wait_for_timeout(300)
    o = txt('#fa-rail [data-obs]')
    want = f'from {rise[0]:.2f} to {rise[1]:.2f} at K/V tile 3'
    check('part 4 shows 384, the rise and the factor', '384' in o and want in o and f'multiplied by {rise[2]:.2f}' in o and f'{md:.1e}'.split('e')[0] in o, (want, o[:500]))
    tab = ev("()=>[...document.querySelectorAll('#fa-row tbody tr')].map(r=>[...r.cells].map(c=>c.textContent))")
    check('part 4 row table: tile max, m and ℓ of row 1', [x[1] for x in tab] == [f'{s["tmax"][0]:.2f}' for s in row0] and [x[4] for x in tab] == [f'{s["l"][0]:.3f}' for s in row0], (tab, [(s['tmax'][0], s['l'][0]) for s in row0]))
    go(4); ev("()=>{const i=document.querySelector('#fa-ic-N input'); i.value=7; i.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(300)
    o = txt('#fa-rail [data-obs]')
    check('part 5 shows 128.1 GiB and 64.1 GiB, about 252 times the 8K traffic', all(x in o for x in ('128.1 GiB', '64.1 GiB', '252')), o[:400])
    go(5)
    for lvv, exp in ((32, '1,028 MiB'), (64, '516 MiB'), (256, '578 KiB')):
        ev(f"()=>document.querySelector('#fa-rail [data-lever=\"{lvv}\"]').click()"); pg.wait_for_timeout(200)
        o = txt('#fa-rail [data-obs]'); check(f'part 6 tiles of {lvv}: {exp}', exp in o, o[:300])
    for i, exp in enumerate((fB(traffic(1024, 64, 128)['std']), '520 MiB', '128.1 GiB', fB(traffic(8192, 128, 128, 1)['std']))):
        if i == 0: ev("()=>document.querySelector('#fa-app .modes button[data-mode=\"sandbox\"]:not([data-flow=free])').click()"); pg.wait_for_timeout(300)
        ev(f"()=>document.querySelector('#fa-rail [data-preset=\"{i}\"]').click()"); pg.wait_for_timeout(250)
        o = txt('#fa-insight'); check(f'scenario {i + 1}: {exp}', exp in o, o[:300])
    o = txt('#fa-insight')
    check('decode: one row of S is 16 KiB', '16 KiB' in o and traffic(8192, 128, 128, 1)['s'] == 16 * KiB, o[:400])
    ev("()=>document.querySelector('#fa-app .modes button[data-mode=\"challenge\"]').click()"); pg.wait_for_timeout(300)
    check('challenge progress at 64 × 64: 56.50 KiB and 4.01 GiB', txt('#fa-ch-m') == fK(sram(64, 64, 64)) == '56.50 KiB' and txt('#fa-ch-t') == '4.01 GiB', (txt('#fa-ch-m'), txt('#fa-ch-t')))
    check('no page errors', not errs, errs[:3])
    b.close()

print(f'{n - len(fails)}/{n} checks passed')
sys.exit(1 if fails else 0)
