"""Golden numbers for the rope lab, v2 (guide R9.2, CONTRACT §6).

Derived here independently of the JS:
- the speeds θᵢ = base^(−2i/128) for i = 0..63, and Llama‑3.1‑8B's "llama3" scaling written from its definition (wavelength 2π/θ:
  above 8,192 tokens ÷ 8, below 2,048 unchanged, in between blended by (8192/λ − 1)/(4 − 1)), cross-checked on a few pairs by hand;
- the made-up head of the spec (128 numbers per query and key; pair 0 as [length, angle]; pairs 1–63 from the spec's formulas; the keys
  of the nouns one pair-speed ahead of the query of "bites"), built here with complex numbers: a pair (x, y) of neighbours 2i, 2i + 1 is
  x + iy, and turning it by a is multiplying by e^{ia};
- the pairing: q·k over 128 numbers equals the sum of the 64 pairs' dot products; the engine pairs neighbours (as Meta's Llama code),
  not j with j + 64 (Hugging Face's rotate_half), which gives other scores on the same vectors;
- the relative property on random 128-number vectors (score at m, n equals the score at m + s, n + s);
- every number the page states: part 1 (8.12 in both orders), part 3 (pair 0: 77.0°, 0.44 without RoPE; 19.7°, 1.84 and 134.3°, −1.36
  with it, the angle moved by +θ and −θ; the whole head 9.64 vs 6.00), part 4 (102 and 101: 19.7°, 9.64; the curve at ±1), part 5 (pairs
  under one turn at every length and base), part 6 (pairs past training and the slowest pair's ratio, with and without the scaling), the
  scenarios, the challenge (turns over 1M tokens per base; 500,000 the smallest under one turn), the exits' new cases, the drill, the
  Connect code (run here) and the home tile (window.__tzRope);
- the page's engine (window.__rp) against all of these, and numbers read from the page itself (metric cards, observations, readouts).
Run: python3 tests/rope_numbers.py"""
import cmath, math, pathlib, random, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/rope.html', '--only', 'rope'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-rope-num.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'rope.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-rope-num.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


def close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


TAU = 2 * math.pi
D = 128
NP = D // 2
R2D = 180 / math.pi


def speeds(base):
    return [base ** (-(2 * i) / D) for i in range(NP)]


def llama3(base, factor=8.0, lo=1.0, hi=4.0, old=8192):
    out = []
    for f in speeds(base):
        lam = TAU / f
        if lam < old / hi: out.append(f)                       # short wavelength: kept
        elif lam > old / lo: out.append(f / factor)            # long wavelength: 8 times slower
        else:
            s = (old / lam - lo) / (hi - lo)                   # blended
            out.append((1 - s) * f / factor + s * f)
    return out


# by hand: pair 0 is 1 rad/token for every base; pair 63 of base 500,000 is 500000^(-126/128); llama3 divides it by 8
check('pair 0 turns 1 radian per token (57.3°)', all(speeds(b)[0] == 1.0 for b in (1e4, 5e5, 1e6)) and f'{R2D:.1f}' == '57.3')
check('pair 63, base 500,000', close(speeds(5e5)[63], math.exp(-126 / 128 * math.log(5e5))))
l3 = llama3(5e5)
check('llama3: pairs 0–28 unchanged, 35–63 divided by 8', all(l3[i] == speeds(5e5)[i] for i in range(29)) and all(close(l3[i] * 8, speeds(5e5)[i]) for i in range(35, 64)))
check('llama3: pairs 29–34 blended (between ÷8 and ÷1)', all(speeds(5e5)[i] / 8 < l3[i] < speeds(5e5)[i] for i in range(29, 35)))
check('29 pairs of a 500,000 head need more than 8,192 tokens for one turn', sum(TAU / f > 8192 for f in speeds(5e5)) == 29)

# ---------------------------------------------------------------- the made-up head (spec: "Engine")
TH5 = speeds(5e5)
frac = lambda x: x - math.floor(x)
ALPHA = lambda i: (i * 137.508) % 360


def mk(p0, lenf, angf):
    """64 complex pairs; pair i = numbers 2i and 2i + 1 (neighbours); pair 0 is p0 = (length, angle in degrees)"""
    out = []
    for i in range(NP):
        length, ang = p0 if i == 0 else (lenf(i), angf(i))
        out.append(cmath.rect(length, math.radians(ang)))
    return out


Q = {'bites': mk((1.5, -47), lambda i: 0.22 + 0.26 * frac(i * 0.618034), ALPHA),
     'dog': mk((1.0, 80), lambda i: 0.22 + 0.26 * frac(i * 0.7071 + 0.1), lambda i: (i * 97.3 + 20) % 360),
     'man': mk((1.1, 110), lambda i: 0.22 + 0.26 * frac(i * 0.5772 + 0.6), lambda i: (i * 61.1 + 200) % 360)}
nounK = lambda p0: mk(p0, lambda i: 0.22 + 0.26 * frac(i * 0.414214 + 0.3), lambda i: ALPHA(i) + TH5[i] * R2D)
K = {'dog': nounK((1.3, 30)), 'man': nounK((1.2, 40)),
     'bites': mk((0.9, 150), lambda i: 0.22 + 0.26 * frac(i * 0.3183 + 0.45), lambda i: (i * 151.7 + 75) % 360)}
flat = lambda z: [c for p_ in z for c in (p_.real, p_.imag)]            # 128 numbers: x0, y0, x1, y1, ...


def share(qw, kw, i, m, nn, rope):
    q, k = Q[qw][i], K[kw][i]
    if rope: q, k = q * cmath.exp(1j * m * TH5[i]), k * cmath.exp(1j * nn * TH5[i])
    return (q * k.conjugate()).real                                   # x·u + y·v


def score(qw, kw, m, nn, rope):
    return sum(share(qw, kw, i, m, nn, rope) for i in range(NP))


ORD = {'a': ['dog', 'bites', 'man'], 'b': ['man', 'bites', 'dog']}   # positions 1, 2, 3


def grid(o, rope):
    return [[score(qw, kw, a + 1, b + 1, rope) for b, kw in enumerate(ORD[o])] for a, qw in enumerate(ORD[o])]


def half_pair_score(qv, kv, m, nn):
    """Hugging Face's rotate_half pairing (j with j + 64) on the same 128 numbers"""
    s = 0
    for j in range(NP):
        a, b = complex(qv[j], qv[j + 64]) * cmath.exp(1j * m * TH5[j]), complex(kv[j], kv[j + 64]) * cmath.exp(1j * nn * TH5[j])
        s += (a * b.conjugate()).real
    return s


deg = lambda r: r * R2D
fmt2 = lambda v: ('−' if v < 0 and f'{abs(v):.2f}' != '0.00' else '') + f'{abs(v):.2f}'
wrap = lambda d: (d + 180) % 360 - 180

# ---------------------------------------------------------------- the relative property, random vectors, Llama‑3.1‑8B's speeds
rng = random.Random(7)
th31 = llama3(5e5)
for _ in range(5):
    q = [rng.gauss(0, 1) for _ in range(D)]; k = [rng.gauss(0, 1) for _ in range(D)]
    def full(m, nn):
        return sum((complex(q[2 * i], q[2 * i + 1]) * cmath.exp(1j * m * th31[i]) * (complex(k[2 * i], k[2 * i + 1]) * cmath.exp(1j * nn * th31[i])).conjugate()).real for i in range(64))
    m, nn, s = rng.randint(0, 50), rng.randint(0, 50), rng.randint(1, 100000)
    check('relative property: score(m, n) = score(m + s, n + s)', abs(full(m, nn) - full(m + s, nn + s)) < 1e-7, (m, nn, s))

# ---------------------------------------------------------------- the page's key numbers, derived here
g0a, g0b, gra, grb = grid('a', False), grid('b', False), grid('a', True), grid('b', True)
check('part 1: bites → dog 8.12 without positions, in both orders', fmt2(g0a[1][0]) == '8.12' and fmt2(g0b[1][2]) == '8.12', (g0a[1][0], g0b[1][2]))
check('part 1: without positions every score of one order appears in the other with the same words',
      all(close(g0a[ORD['a'].index(x)][ORD['a'].index(y)], g0b[ORD['b'].index(x)][ORD['b'].index(y)]) for x in Q for y in K))
check('part 1 grid “dog bites man”: 0.82 −0.07 0.91 / 8.12 −2.50 7.77 / 0.22 0.69 0.42', [[fmt2(v) for v in r] for r in g0a] == [['0.82', '−0.07', '0.91'], ['8.12', '−2.50', '7.77'], ['0.22', '0.69', '0.42']], g0a)
check('part 7 grid with RoPE: bites → dog 9.64 in “dog bites man”, 6.00 in “man bites dog”', fmt2(gra[1][0]) == '9.64' and fmt2(grb[1][2]) == '6.00', (gra[1][0], grb[1][2]))
check('with RoPE the head scores the token right before “bites” highest in both sentences (dog 9.64, man 9.37)',
      max(range(3), key=lambda j: gra[1][j]) == 0 and max(range(3), key=lambda j: grb[1][j]) == 0 and fmt2(grb[1][0]) == '9.37', (gra[1], grb[1]))
qa0, ka0 = -47.0, 30.0
b_off = wrap(qa0 - ka0); b_a = wrap((qa0 + 2 * R2D) - (ka0 + 1 * R2D)); b_b = wrap((qa0 + 2 * R2D) - (ka0 + 3 * R2D))
check('part 3, RoPE off: pair 0 is 77.0° apart and gives 0.44 in both sentences', f'{abs(b_off):.1f}' == '77.0' and fmt2(share('bites', 'dog', 0, 2, 1, False)) == '0.44')
check('part 3, RoPE on: 19.7° apart (1.84) in “dog bites man”, 134.3° apart (−1.36) in “man bites dog”',
      f'{abs(b_a):.1f}' == '19.7' and f'{abs(b_b):.1f}' == '134.3' and fmt2(share('bites', 'dog', 0, 2, 1, True)) == '1.84' and fmt2(share('bites', 'dog', 0, 2, 3, True)) == '−1.36', (b_a, b_b))
check('part 3: the angle between moves by +θ in one sentence and −θ in the other (θ = 57.3°)', close(b_a - b_off, R2D) and close(b_b - b_off, -R2D), (b_a - b_off, b_b - b_off))
check('part 3: 1.5 × 1.3 × cos 19.7° = 1.84 (the pair’s share is its dot product)', close(1.5 * 1.3 * math.cos(math.radians(b_a)), share('bites', 'dog', 0, 2, 1, True)))
check('part 3: whole head 9.64 vs 6.00 with RoPE, 8.12 without', fmt2(score('bites', 'dog', 2, 1, True)) == '9.64' and fmt2(score('bites', 'dog', 2, 3, True)) == '6.00' and fmt2(score('bites', 'dog', 2, 1, False)) == '8.12')
check('part 3 metric: difference between the orders 3.64 with RoPE, 0.00 without', f'{abs(9.64 - 6.00):.2f}' == '3.64')
check('part 4: moved 100 later, 102 and 101: still 19.7° apart, whole head still 9.64', fmt2(score('bites', 'dog', 102, 101, True)) == '9.64' and f'{abs(wrap((qa0 + 102 * R2D) - (ka0 + 101 * R2D))):.1f}' == '19.7')
check('part 4: the whole head depends on the positions only through m − n', all(abs(score('bites', 'dog', 2 + s, 1 + s, True) - score('bites', 'dog', 2, 1, True)) < 1e-9 for s in (1, 7, 50, 100)))
check('part 4 curve: +1 gives 9.64, −1 gives 6.00', fmt2(score('bites', 'dog', 1, 0, True)) == '9.64' and fmt2(score('bites', 'dog', 0, 1, True)) == '6.00')
check('part 4 curve range: every value between 3 and 11 over −8 … 8', all(3 < score('bites', 'dog', d / 10, 0, True) < 11 for d in range(-80, 81)))

# pairing
qb, kd = flat(Q['bites']), flat(K['dog'])
check('pairing: q·k over the 128 numbers = the sum of the 64 pairs’ dot products', close(sum(a * c for a, c in zip(qb, kd)), score('bites', 'dog', 0, 0, False)))
check('pairing: part 2 shows pair 0 of “bites” as (1.023, −1.097)', f'{qb[0]:.3f}' == '1.023' and f'{qb[1]:.3f}' == '-1.097', qb[:2])
check('pairing: with j + 64 pairs (rotate_half) the same numbers give another score, so the page must say which pairing it draws',
      abs(half_pair_score(qb, kd, 2, 1) - score('bites', 'dog', 2, 1, True)) > 0.1)


def turns(th, L, th_train=None, T=None):
    rows = []
    for i, f in enumerate(th):
        t = L * f / TAU; tt = None if T is None else T * th_train[i] / TAU
        rows.append(dict(t=t, tt=tt, under=t < 1, beyond=tt is not None and tt < 1 and t > tt * (1 + 1e-9)))
    nu = sum(r['under'] for r in rows); nb = sum(r['beyond'] for r in rows); s = rows[-1]
    first = next((i for i, r in enumerate(rows) if r['under']), -1)
    return dict(nUnder=nu, nBeyond=nb, first=first, slow_t=s['t'], ratio=(s['t'] / s['tt']) if s['tt'] else None)


L5 = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
L6 = [8192, 16384, 32768, 65536, 131072, 262144]
P5 = {(b, L): turns(speeds(b), L) for b in (1e4, 5e5) for L in L5}
check('part 5, base 500,000: 32, 29, 25, 22, 18, 15, 12 pairs under one turn from 4K to 256K', [P5[(5e5, L)]['nUnder'] for L in L5] == [32, 29, 25, 22, 18, 15, 12])
check('part 5, base 10,000: 18, 14, 9, 4, 0, 0, 0', [P5[(1e4, L)]['nUnder'] for L in L5] == [18, 14, 9, 4, 0, 0, 0])
check('part 5 start (8K, 500,000): pairs 35 to 63', P5[(5e5, 8192)]['first'] == 35)
check('part 5 at 128K: the slowest pair turns 0.051 (18.4°) with 500,000, 2.41 turns with 10,000',
      f"{P5[(5e5, 131072)]['slow_t']:.3f}" == '0.051' and f"{P5[(5e5, 131072)]['slow_t'] * 360:.1f}" == '18.4' and f"{P5[(1e4, 131072)]['slow_t']:.2f}" == '2.41')
check('part 5 at 4K: 0.0016 turns (0.6°) with 500,000', f"{P5[(5e5, 4096)]['slow_t']:.2g}" == '0.0016' and f"{P5[(5e5, 4096)]['slow_t'] * 360:.1f}" == '0.6')
P6 = {(L, sc): turns(llama3(5e5) if sc else speeds(5e5), L, speeds(5e5), 8192) for L in L6 for sc in (False, True)}
check('part 6 as trained: 0 then 29 pairs past training; the slowest 1×, 2×, 4×, 8×, 16×, 32×',
      [P6[(L, False)]['nBeyond'] for L in L6] == [0, 29, 29, 29, 29, 29] and [round(P6[(L, False)]['ratio'], 6) for L in L6] == [1, 2, 4, 8, 16, 32])
check('part 6 with the scaling: none up to 64K, then 29 at 2× and 4×',
      [P6[(L, True)]['nBeyond'] for L in L6] == [0, 0, 0, 0, 29, 29] and [round(P6[(L, True)]['ratio'], 6) for L in L6] == [0.125, 0.25, 0.5, 1, 2, 4])
check('part 6: 128K is 16 times 8K, so 8 times slower leaves 2 times', 131072 / 8192 == 16 and 16 / 8 == 2)
blend = [speeds(5e5)[i] / l3[i] for i in range(29, 35)]
check('llama3: the 6 blended pairs 29–34 are slowed 1.2 to 5.3 times', f'{blend[0]:.1f}' == '1.2' and f'{blend[-1]:.1f}' == '5.3' and all(1 < x < 8 for x in blend), blend)
both128 = sum(1 for i in range(NP) if 131072 * speeds(5e5)[i] / TAU < 1 and 8192 * speeds(5e5)[i] / TAU < 1)
check('part 6: at 128K, 15 of the 29 pairs past training still stay under one turn', both128 == 15)
both32 = sum(1 for i in range(NP) if 32768 * speeds(5e5)[i] / TAU < 1 and 8192 * speeds(5e5)[i] / TAU < 1)
check('scenario 3: at 32K, 22 of the 29 pairs past training stay under one turn', both32 == 22)
check('the chart’s end speeds: pair 0 57.3° per token; pair 63 0.000141° (base 500,000), 0.00662° (base 10,000)',
      f'{R2D:.1f}' == '57.3' and f'{speeds(5e5)[63] * R2D:.3g}' == '0.000141' and f'{speeds(1e4)[63] * R2D:.3g}' == '0.00662')
top = lambda g: max(range(3), key=lambda j: g[1][j])
check('the row of “bites”: without RoPE its top is “dog” in both orders; with RoPE the token just before “bites”',
      ORD['a'][top(g0a)] == 'dog' and ORD['b'][top(g0b)] == 'dog' and ORD['a'][top(gra)] == 'dog' and ORD['b'][top(grb)] == 'man' and fmt2(grb[1][0]) == '9.37' and fmt2(gra[1][2]) == '5.90')
check('part 3 turns: 2 × 57.3° = 114.6°, 3 × 57.3° = 171.9°, both under a full turn (the arcs are drawn)', f'{2 * R2D:.1f}' == '114.6' and f'{3 * R2D:.1f}' == '171.9' and 3 * R2D < 360)
check('part 5: 5,000 tokens back, pair 0 has gone round about 800 times', round(5000 * speeds(5e5)[0] / TAU, -2) == 800)
check('part 6: 100,000 tokens back, the slowest pair turns 12 times as far as at the 8K training length', f'{100000 / 8192:.1f}' == '12.2' and 100000 * speeds(5e5)[63] / TAU < 1)
CH = {b: 1048576 * speeds(b)[63] / TAU for b in (1e4, 1e5, 5e5, 1e6)}
check('challenge: turns over 1M tokens 19.27, 2.00, 0.41, 0.21', [f'{CH[b]:.2f}' for b in (1e4, 1e5, 5e5, 1e6)] == ['19.27', '2.00', '0.41', '0.21'], CH)
check('challenge: the smallest base under one turn is 500,000', min(b for b in CH if CH[b] < 1) == 5e5)
lam = {b: TAU / speeds(b)[63] for b in (1e4, 1e5, 5e5, 1e6)}
check('challenge chart: tokens for one full turn 54,410 / 525K / 2.56M / 5.06M', round(lam[1e4]) == 54410 and f'{lam[1e5] / 1e3:.0f}' == '525' and f'{lam[5e5] / 1e6:.2f}' == '2.56' and f'{lam[1e6] / 1e6:.2f}' == '5.06', lam)
check('the part 5 slider stops at 256K, below the challenge’s 1M', max(L5) == 262144 < 1048576)
a4, a16 = 4096 * speeds(1e4)[63], 16384 * speeds(1e4)[63]
check('exits: Llama‑2‑7B slowest 27.1° at 4K, 108.4° at 16K', f'{deg(a4):.1f}' == '27.1' and f'{deg(a16):.1f}' == '108.4', (deg(a4), deg(a16)))
check('exits: position interpolation ÷ 4 → 14.3° between neighbours', f'{deg(1) / 4:.1f}' == '14.3')
check('remember: base 10,000 over 128K, the slowest pair turns 2.41 turns', f"{131072 * speeds(1e4)[63] / TAU:.2f}" == '2.41')

# the Connect card's code, run here
code_ns = {}
exec("""
import numpy as np
def speeds(base=500_000.0, d=128):
    return base ** (-np.arange(0, d, 2) / d)
def rope(x, pos, theta):
    a = pos * theta
    c, s = np.cos(a), np.sin(a)
    out = np.empty_like(x)
    out[0::2] = x[0::2] * c - x[1::2] * s
    out[1::2] = x[0::2] * s + x[1::2] * c
    return out
rng = np.random.default_rng(0)
q, k = rng.normal(size=128), rng.normal(size=128)
th = speeds()
s1 = rope(q, 5, th) @ rope(k, 2, th)
s2 = rope(q, 105, th) @ rope(k, 102, th)
same = bool(np.isclose(s1, s2))
turns = 131_072 * th[-1] / (2 * np.pi)
""", code_ns)
check('Connect code: prints True and 0.051 turns', code_ns['same'] and f"{code_ns['turns']:.3f}" == '0.051', (code_ns['same'], code_ns['turns']))

with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300); pg.click('.langsw button[data-lang="en"]'); pg.evaluate("()=>{location.hash='rope'}"); pg.wait_for_timeout(600)
    RP = lambda js: pg.evaluate(f"()=>{{const r=window.__rp; return {js};}}")
    # the engine's speeds against this derivation
    for base, sc in ((1e4, False), (5e5, False), (1e6, False), (5e5, True), (1e5, False)):
        js = RP(f"r.thetas({base},{str(sc).lower()})")
        py = llama3(base) if sc else speeds(base)
        check(f'engine speeds, base {base:.0f}{" llama3" if sc else ""}', len(js) == 64 and all(close(a, c, 1e-12) for a, c in zip(js, py)))
    # the engine's head against this derivation, number by number
    for w in ('bites', 'dog', 'man'):
        check(f'engine query of “{w}”: 128 numbers as the spec builds them', all(close(a, c, 1e-12) for a, c in zip(RP(f"r.qv('{w}')"), flat(Q[w]))) and len(RP(f"r.qv('{w}')")) == 128)
        check(f'engine key of “{w}”: 128 numbers as the spec builds them', all(close(a, c, 1e-12) for a, c in zip(RP(f"r.kv('{w}')"), flat(K[w]))) and len(RP(f"r.kv('{w}')")) == 128)
    check('engine pairs neighbours (2i, 2i + 1): its score equals the neighbour pairing, not rotate_half',
          close(RP("r.score('bites','dog',2,1,true)"), score('bites', 'dog', 2, 1, True)) and not close(RP("r.score('bites','dog',2,1,true)"), half_pair_score(qb, kd, 2, 1), 1e-3))
    check('engine: every pair’s share, both sentences', all(close(RP(f"r.share('bites','dog',{i},2,{nn},true)"), share('bites', 'dog', i, 2, nn, True)) for i in (0, 1, 17, 40, 63) for nn in (1, 3)))
    check('engine grid (no RoPE, both orders)', all(close(RP(f"r.grid('{o}',false)")[i][j], grid(o, False)[i][j]) for o in 'ab' for i in range(3) for j in range(3)))
    check('engine grid (RoPE, both orders)', all(close(RP(f"r.grid('{o}',true)")[i][j], grid(o, True)[i][j]) for o in 'ab' for i in range(3) for j in range(3)))
    for m, nn, rope in ((2, 1, False), (2, 1, True), (2, 3, True), (102, 101, True), (37, 36, True)):
        p0 = RP(f"r.pair0({m},{nn},{str(rope).lower()})")
        check(f'engine pair0({m},{nn},{rope}): share, head score, angle between',
              close(p0['share'], share('bites', 'dog', 0, m, nn, rope)) and close(p0['head'], score('bites', 'dog', m, nn, rope))
              and f"{abs(p0['between']):.1f}" == f"{abs(wrap((qa0 + (m * R2D if rope else 0)) - (ka0 + (nn * R2D if rope else 0)))):.1f}", p0)
    for (bb, L), v in P5.items():
        js = RP(f"r.p5({bb:.0f},{L})")
        check(f'part 5 at {L}, base {bb:.0f}: pairs under one turn', js['nUnder'] == v['nUnder'] and js['first'] == v['first'] and close(js['slow']['t'], v['slow_t']), (js['nUnder'], v['nUnder']))
    for (L, sc), v in P6.items():
        js = RP(f"r.p6({L},{str(sc).lower()})")
        check(f'part 6 at {L}{" scaled" if sc else ""}: pairs past training, ratio', js['nBeyond'] == v['nBeyond'] and close(js['ratio'], v['ratio']), (js['nBeyond'], js['ratio'], v))
    for pr in RP("r.consts.PRESETS"):
        th_ = llama3(pr['base']) if pr['sc'] else speeds(pr['base'])
        v = turns(th_, pr['ctx'], th_, pr['train']); js = RP(f"r.sb({{base:{pr['base']},sc:{str(pr['sc']).lower()},ctx:{pr['ctx']},train:{pr['train']}}})")
        check(f"scenario {pr['id']}", js['nUnder'] == v['nUnder'] and js['nBeyond'] == v['nBeyond'] and close(js['slow']['t'], v['slow_t']), (js, v))
    check('scenario presets: Llama‑2‑7B 10,000 at 4K, Llama‑3‑8B 500,000 at 8K and 32K, Llama‑3.1‑8B 500,000 + llama3 at 128K, Code Llama 1,000,000 at 16K',
          [(p_['id'], p_['base'], p_['sc'], p_['ctx'], p_['train']) for p_ in RP("r.consts.PRESETS")] == [('l2', 10000, False, 4096, 4096), ('l3', 500000, False, 8192, 8192), ('l3x', 500000, False, 32768, 8192), ('l31', 500000, True, 131072, 131072), ('cl', 1000000, False, 16384, 16384)])
    check('part sliders: L5 and L6 as in the spec', RP("r.consts.L5") == L5 and RP("r.consts.L6") == L6 and RP("r.consts.B5") == [10000, 500000])
    for bb in (1e4, 1e5, 5e5, 1e6):
        check(f'challenge turns, base {bb:.0f}', close(RP(f"r.chTurns({bb:.0f})"), CH[bb]))
    check('challenge best = 500,000', RP("r.chBest()") == 500000)
    for s in range(1, 6):
        dr = RP(f"r.drill({s})")
        th_i = dr['b'] ** (-(2 * dr['i']) / dr['d']); th_s = dr['b'] ** (-(dr['d'] - 2) / dr['d'])
        check(f'drill {s}: pair i full turn and the slowest pair over L', close(dr['lam'], TAU / th_i) and close(dr['aS'], dr['L'] * th_s) and not (dr['d'] == 128 and dr['L'] == 1048576), dr)
    # the home tile against the lab's part 3, pair 0 ("bites" at 2, "dog" at 1 or 3); it shows part 3's result, not part 4's (R3.8)
    tz = pg.evaluate("()=>window.__tzRope.model(false)")
    check('tile, RoPE off: 77.0° apart and 0.44 in both orders', fmt2(tz['a']['score']) == '0.44' and fmt2(tz['b']['score']) == '0.44' and f"{abs(tz['a']['between']):.1f}" == '77.0')
    tz = pg.evaluate("()=>window.__tzRope.model(true)")
    check('tile, RoPE on: 19.7° apart (1.84) and 134.3° apart (−1.36), as the lab’s pair 0',
          close(tz['a']['score'], share('bites', 'dog', 0, 2, 1, True)) and close(tz['b']['score'], share('bites', 'dog', 0, 2, 3, True)) and f"{abs(tz['a']['between']):.1f}" == '19.7' and f"{abs(tz['b']['between']):.1f}" == '134.3', tz)
    check('tile sentences: at most 35 words (R11.5), the "on" one names pair 0, and none says that moving both tokens keeps the score',
          all(len(re.sub('<[^>]+>', '', pg.evaluate(f"()=>window.__tzRope.read({str(o).lower()})")).split()) <= 35 for o in (False, True))
          and 'pair 0 scores' in pg.evaluate("()=>window.__tzRope.read(true)")
          and not any(re.search(r'stay|same score|only the distance', pg.evaluate(f"()=>window.__tzRope.read({str(o).lower()})")) for o in (False, True)))
    # numbers shown on the page
    cards = lambda: pg.evaluate("()=>[...document.querySelectorAll('#rp-metrics .metric')].map(m=>m.querySelector('.v').textContent)")
    go = lambda i: (pg.evaluate(f"()=>document.querySelector('#rp-rail [data-go=\"{i}\"]').click()"), pg.wait_for_timeout(250))
    check('part 1 cards: 8.12, 8.12, 0.00', cards() == ['8.12', '8.12', '0.00'], cards())
    go(1)
    check('part 2: no metric cards (the stage shows the head)', pg.evaluate("()=>document.getElementById('rp-metrics').hidden"))
    go(2)
    check('part 3 cards, RoPE off: 8.12, 8.12, 0.00', cards() == ['8.12', '8.12', '0.00'], cards())
    pg.evaluate("()=>document.querySelector('#rp-rail .seg-ctl[data-key=\"rope3\"] button[data-v=\"on\"]').click()"); pg.wait_for_timeout(250)
    check('part 3 cards, RoPE on: 9.64, 6.00, 3.64', cards() == ['9.64', '6.00', '3.64'], cards())
    obs = pg.inner_text('#rp-rail [data-obs]')
    check('part 3 result: the row of “bites” tops at “dog” 9.64 and “man” 9.37 with RoPE, at “dog” 8.12 without', all(x in obs for x in ('9.64', '9.37', '8.12')), obs[:200])
    subs = pg.evaluate("()=>[...document.querySelectorAll('#rp-metrics .metric .sub')].map(e=>e.textContent)")
    check('part 3 card notes: pair 0 gives 1.84 and −1.36', '1.84' in subs[0] and '−1.36' in subs[1], subs)
    svgt = pg.evaluate("()=>[...document.querySelectorAll('#rp-vis svg text')].map(t=>t.textContent).join('|')")
    check('part 3 dials: “bites” at 2 turns +2 × 57.3°, “dog” +1 × and +3 × 57.3°; the bars read 9.64, 5.90, 9.37, 6.00',
          all(x in svgt for x in ('+2 × 57.3°', '+1 × 57.3°', '+3 × 57.3°', '9.64', '5.90', '9.37', '6.00')), svgt[:300])
    go(3)
    pg.evaluate("()=>document.querySelector('#rp-rail [data-pred=\"same\"]').click()"); pg.wait_for_timeout(250)
    pg.evaluate("()=>{const i=document.querySelector('#rp-ic-shift-in'); i.value=100; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(250)
    check('part 4 cards after moving 100: distance 1, 19.7°, 9.64', cards() == ['1', '19.7°', '9.64'], cards())
    obs = pg.inner_text('#rp-rail [data-obs]')
    check('part 4 observation states 19.7° and 9.64', '19.7°' in obs and '9.64' in obs, obs[:200])
    go(4)
    pg.evaluate("()=>{const i=document.querySelector('#rp-ic-l5-in'); i.value=0; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(250)
    check('part 5 at 4K: cards 4K, 32 of 64, 0.00160 (3 significant digits)', cards() == ['4K', '32 of 64', '0.00160'], cards())
    check('part 5 result compares with the start: 8K → 4K, 29 → 32', '8K → 4K' in pg.inner_text('#rp-rail [data-obs]') and '29 → 32' in pg.inner_text('#rp-rail [data-obs]'), pg.inner_text('#rp-rail [data-obs]')[:200])
    pg.evaluate("()=>document.querySelector('#rp-rail [data-nav=\"showme\"]').click()"); pg.wait_for_timeout(250)
    check('part 5 show me (128K, base 10,000): cards 128K, 0 of 64, 2.41', cards() == ['128K', '0 of 64', '2.41'], cards())
    go(5)
    pg.evaluate("()=>document.querySelector('#rp-rail [data-nav=\"showme\"]').click()"); pg.wait_for_timeout(250)
    check('part 6 at 128K as trained: cards 128K, 29 of 64, 0.0512', cards() == ['128K', '29 of 64', '0.0512'], cards())
    check('part 6: 15 of the 29 pairs past training still stay under one turn', '15 of these pairs' in pg.inner_text('#rp-rail [data-obs]'), pg.inner_text('#rp-rail [data-obs]')[:300])
    pg.evaluate("()=>document.querySelector('#rp-rail .seg-ctl[data-key=\"s6\"] button[data-v=\"1\"]').click()"); pg.wait_for_timeout(250)
    obs = pg.inner_text('#rp-rail [data-obs]')
    check('part 6 scaled at 128K: still 29 pairs, 2 times instead of 16, and the 6 blended pairs slowed 1.2 to 5.3 times', '29' in obs and '2 times' in obs and 'instead of 16' in obs and '1.2 to 5.3' in obs, obs[:300])
    pg.evaluate("()=>document.querySelector('#rp-app .modes button[data-mode=\"challenge\"]').click()"); pg.wait_for_timeout(350)
    for bb, want in ((100000, '2.00 turns'), (500000, '0.41 turns')):
        pg.evaluate(f"()=>document.querySelector('#rp-ic-ch [data-lever=\"{bb}\"]').click()"); pg.wait_for_timeout(200)
        check(f'challenge readout at base {bb}', pg.inner_text('#rp-ch-now') == want, pg.inner_text('#rp-ch-now'))
    check('no page errors', not errs, errs)
    b.close()

print(f'{n - len(fails)} / {n} passed')
sys.exit(1 if fails else 0)
