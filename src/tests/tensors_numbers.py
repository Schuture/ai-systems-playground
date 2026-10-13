"""Numbers test for the tensors lab (CONTRACT §6): every number the page shows is recomputed here, independently, with NumPy.

- broadcasting: every pair of the lab's shapes against np.broadcast_shapes (result or error, and the axis the lab names: the first
  mismatch from the right); the stretched axes; the small arrays' numbers (a[i][j] = 10i + j, b = 100, 200, …) and a + b against NumPy;
- einsum: each string's output shape and summed letters against np.einsum on small stand-in sizes, the multiply-adds as the product of
  every letter's size (17,179,869,184 for the attention scores), the 4 × 3 by 3 × 2 example against A @ B;
- 1-NN: the largest array and the loop runs of each way at every slider stop, the mission's 2 KiB, 954 GiB (1.02 TB), 1.86 GiB and 500M,
  the interview sizes (763 MiB against 15.3 MiB), and the identity ‖x − y‖² = ‖x‖² − 2x·y + ‖y‖² against brute force;
- the challenge: every option's array, loop runs and verdict, the pass set {chunks of 32, matmul expansion}, the best, the ratio 768;
- the capability card's answers ((8, 1, 6, 1) + (7, 1, 5) and the 16 times), the drill against np.broadcast_shapes for 300 seeds, the
  home tile's four pairs (and that it never shows the prediction's pair);
- Connect: the code has at most 40 lines of at most 74 characters, runs, and prints exactly its last three comment lines (its double loop
  keeps a running minimum, so its largest array is one (D,) difference, as the chart says);
- shared numbers: one H100 = 80 GiB (80 GB read as GiB); Llama‑3.1‑8B's 32 heads of 128 numbers, written with U+2011.
Run: python3 tests/tensors_numbers.py"""
import pathlib, re, subprocess, sys, math
import numpy as np
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/tensors-num.html', '--only', 'tensors'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'tensors-num-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'tensors-num.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'tensors-num-wrapped.html')
fails, n, errs = [], 0, []
GiB, MiB, KiB = 2 ** 30, 2 ** 20, 2 ** 10


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


def shape(k): return tuple(int(x) for x in k.split('x'))


def bc(a, b):
    try: return np.broadcast_shapes(a, b)
    except ValueError: return None


def first_bad_from_right(a, b):
    for k in range(max(len(a), len(b))):
        x = a[-1 - k] if k < len(a) else 1
        y = b[-1 - k] if k < len(b) else 1
        if x != y and x != 1 and y != 1: return k, x, y
    return None


with sync_playwright() as p:
    br = p.chromium.launch()
    pg = br.new_page(viewport={'width': 1440, 'height': 900})
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.evaluate("()=>{location.hash='tensors'}"); pg.wait_for_timeout(600)
    C = pg.evaluate("()=>window.__ts.consts")
    ev = lambda js, *a: pg.evaluate(js, *a)

    # ---------- broadcasting ----------
    for ka in C['SH_A']:
        for kb in C['SH_B']:
            a, b = shape(ka), shape(kb)
            r = ev("([a,b])=>window.__ts.bshape(a,b)", [list(a), list(b)])
            want = bc(a, b)
            if want is None:
                k, x, y = first_bad_from_right(a, b)
                check(f'broadcast {a} + {b}: an error on axis {k} from the right ({x} vs {y})', not r['ok'] and r['err']['k'] == k and (r['err']['x'], r['err']['y']) == (x, y), r['err'])
            else:
                check(f'broadcast {a} + {b} = {want}', r['ok'] and tuple(r['out']) == want, r['out'])
                n_ = len(want)
                for c in r['ax']:
                    k = c['k']; x = a[-1 - k] if k < len(a) else 1; y = b[-1 - k] if k < len(b) else 1
                    if c['sa'] != (x == 1 and want[-1 - k] > 1) or c['sb'] != (y == 1 and want[-1 - k] > 1):
                        check(f'broadcast {a} + {b}: stretched axes', False, c)
            # the small arrays with their numbers
            if want is not None and all(v <= 6 for v in a + b):
                A = np.add.outer(10 * np.arange(a[0]), np.arange(a[1]))
                B = (100 * (np.arange(int(np.prod(b))) + 1)).reshape(b)
                got = ev("([a,b])=>window.__ts.bsum(a,b)", [list(a), list(b)])
                check(f'a + b numbers for {a} + {b}', ev("s=>window.__ts.arrA(s)", list(a)) == A.tolist() and np.array_equal(np.array(got), A + B), got)
    check('broadcast: (3, 1) + (1, 4) is the prediction’s pair, answer (3, 4)', C['PRED_RIGHT'] == '34' and bc((3, 1), (1, 4)) == (3, 4))

    # ---------- einsum ----------
    small = [2, 3, 5, 7, 11]
    for e in C['ES']:
        r = ev("id=>window.__ts.es(id)", e['id'])
        x, y, z = re.match(r'^(\w+),(\w+)->(\w*)$', e['spec']).groups()
        letters = sorted(set(x + y))
        stand = {c: small[i] for i, c in enumerate(letters)}
        out = np.einsum(e['spec'], np.zeros([stand[c] for c in x]), np.zeros([stand[c] for c in y])).shape
        check(f'einsum {e["spec"]}: output letters map to NumPy’s output shape', out == tuple(stand[c] for c in z) and tuple(r['outShape']) == tuple(e['sizes'][c] for c in z), (out, r['outShape']))
        check(f'einsum {e["spec"]}: summed letters', sorted(r['summed']) == sorted(set(x + y) - set(z)), r['summed'])
        check(f'einsum {e["spec"]}: multiply-adds = product of every letter’s size', r['macs'] == math.prod(e['sizes'][c] for c in letters), r['macs'])
    sc = ev("()=>window.__ts.es('sc')")
    check('attention scores: 17,179,869,184 multiply-adds, d summed, 134,217,728 outputs', sc['macs'] == 17179869184 and sc['summed'] == ['d'] and sc['outN'] == 134217728, sc)
    check('batch of matmuls: 262,144 multiply-adds', ev("()=>window.__ts.es('bm')")['macs'] == 262144)
    MA, MB, MC = (np.array(ev(f"()=>window.__ts.{k}")) for k in ('MA', 'MB', 'MC'))
    check('the 4 × 3 by 3 × 2 example: a[i][k] = i + k, b[k][j] = 2k + j + 1, and the product', np.array_equal(MA, np.add.outer(np.arange(4), np.arange(3))) and np.array_equal(MB, 2 * np.arange(3)[:, None] + np.arange(2) + 1)
          and np.array_equal(MC, np.einsum('ik,kj->ij', MA, MB)) and MC.tolist() == [[13, 16], [22, 28], [31, 40], [40, 52]], MC.tolist())
    att = C['ES'][1]['sizes']
    check('attention sizes are Llama‑3.1‑8B’s: 32 heads of 128 numbers, 2,048 tokens', (att['h'], att['d'], att['q'], att['k'], att['b']) == (32, 128, 2048, 2048, 1), att)

    # ---------- 1-NN ----------
    bad = []
    for N in C['N_ST']:
        for M in C['M_ST']:
            for D in C['D_ST']:
                r = ev("([N,M,D])=>window.__ts.nn(N,M,D)", [N, M, D])
                if (r['loop']['bytes'], r['loop']['iters'], r['bc']['bytes'], r['mm']['bytes'], r['mm']['macs']) != (4 * D, N * M, 4 * N * M * D, 4 * N * M, N * M * D + (N + M) * D): bad.append((N, M, D))
    check('1-NN: every slider stop (180 combinations), arrays and loop runs', not bad, bad[:5])
    F = lambda b: ev("b=>window.__ts.fmt.B(b)", b)
    G = lambda b: ev("b=>window.__ts.fmt.GB(b)", b)
    r = ev("()=>window.__ts.nn(10000,50000,512)")
    check('mission: 2 KiB, 954 GiB (1.02 TB), 1.86 GiB, 500M runs', (F(r['loop']['bytes']), F(r['bc']['bytes']), G(r['bc']['bytes']), F(r['mm']['bytes'])) == ('2 KiB', '954 GiB', '1.02 TB', '1.86 GiB')
          and abs(r['bc']['bytes'] / GiB - 953.674) < 0.001 and r['loop']['iters'] == 500_000_000, (F(r['bc']['bytes']), G(r['bc']['bytes'])))
    r = ev("()=>window.__ts.nn(2000,2000,50)")
    check('interview sizes: 763 MiB against 15.3 MiB, ratio 50', (F(r['bc']['bytes']), F(r['mm']['bytes'])) == ('763 MiB', '15.3 MiB') and r['bc']['bytes'] // r['mm']['bytes'] == 50, (F(r['bc']['bytes']), F(r['mm']['bytes'])))
    r = ev("()=>window.__ts.nn(10000,50000,1024)")
    check('part 5 at D 1,024: 1,907 GiB, the matmul still 1.86 GiB', (F(r['bc']['bytes']), F(r['mm']['bytes'])) == ('1,907 GiB', '1.86 GiB'), (F(r['bc']['bytes']), F(r['mm']['bytes'])))
    # part 5's term table (review round 3): what each term of each way holds, at every slider stop and at the mission's sizes
    bad = []
    for N in C['N_ST']:
        for M in C['M_ST']:
            for D in C['D_ST']:
                tm = {x['k']: x for x in ev("([N,M,D])=>window.__ts.terms(N,M,D)", [N, M, D])}
                r = ev("([N,M,D])=>window.__ts.nn(N,M,D)", [N, M, D])
                want_t = {'bcd': (4 * N * M * D, '(N, M, D)'), 'bcs': (4 * N * M, '(N, M)'), 'xx': (4 * N, '(N, 1)'), 'xy': (4 * N * M, '(N, M)'), 'yy': (4 * M, '(1, M)')}
                if {k: (x['b'], x['sh']) for k, x in tm.items()} != want_t or tm['bcd']['b'] != r['bc']['bytes'] or max(x['b'] for x in tm.values() if x['w'] == 'mm') != r['mm']['bytes']: bad.append((N, M, D))
    check('part 5 terms: every slider stop, each term’s bytes and shape; the largest term of each way is its bar', not bad, bad[:5])
    tm = {x['k']: F(x['b']) for x in ev("()=>window.__ts.terms(10000,50000,512)")}
    check('part 5 terms at the mission’s sizes: 954 GiB, 1.86 GiB; 39.1 KiB, 1.86 GiB, 195 KiB', tm == {'bcd': '954 GiB', 'bcs': '1.86 GiB', 'xx': '39.1 KiB', 'xy': '1.86 GiB', 'yy': '195 KiB'}, tm)
    rng = np.random.default_rng(3); X = rng.standard_normal((60, 24)); Y = rng.standard_normal((90, 24))
    brute = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1); expand = (X ** 2).sum(1)[:, None] - 2 * X @ Y.T + (Y ** 2).sum(1)[None, :]
    check('the identity ‖x − y‖² = ‖x‖² − 2x·y + ‖y‖² against brute force, same nearest neighbours', np.allclose(brute, expand) and (brute.argmin(1) == expand.argmin(1)).all())
    check('one H100 = 80 GiB (80 GB read as GiB)', C['H100'] == 80 * GiB)

    # ---------- the challenge ----------
    N, M, D = C['CH']['N'], C['CH']['M'], C['CH']['D']
    want = {'loop': (4 * D, N * M), 'bc': (4 * N * M * D, 0), 'c1': (4 * M * D, N), 'c32': (4 * 32 * M * D, math.ceil(N / 32)), 'c256': (4 * 256 * M * D, math.ceil(N / 256)), 'mm': (4 * N * M, 0)}
    passing = []
    for o in C['CHO']:
        e = ev("id=>window.__ts.chEval(id)", o['id'])
        bts, it = want[o['id']]
        ok = bts <= 80 * GiB and it <= C['CH']['maxIters']
        check(f'challenge {o["id"]}: {bts} bytes, {it} loop runs, {"meets" if ok else "misses"} the goal', (e['bytes'], e['iters'], e['ok']) == (bts, it, ok), e)
        if ok: passing.append(o['id'])
    check('challenge: chunks of 32 and the matmul expansion pass; the best is the expansion', passing == ['c32', 'mm'] and ev("()=>window.__ts.chBest()")['id'] == 'mm', passing)
    check('challenge: 18.3 GiB and 313 runs for chunks of 32, 146 GiB for 256, 586 MiB for 1, 7.45 GiB and 5,722 GiB', (F(want['c32'][0]), want['c32'][1], F(want['c256'][0]), F(want['c1'][0]), F(want['mm'][0]), F(want['bc'][0]))
          == ('18.3 GiB', 313, '146 GiB', '586 MiB', '7.45 GiB', '5,722 GiB'), (F(want['c32'][0]), F(want['c256'][0]), F(want['c1'][0])))
    e = ev("()=>window.__ts.chEval('c32')")
    check('challenge: chunks write N × M × D in total, 768 times the expansion', e['written'] == 4 * N * M * D and e['written'] // ev("()=>window.__ts.chEval('mm')")['written'] == 768)

    # ---------- capability card, drill, home tile ----------
    ex = ev("()=>window.__ts.exits()")
    o1 = ex['transferO'][0]
    check('new case 1: (8, 1, 6, 1) + (7, 1, 5) = (8, 7, 6, 5), the answer the card marks', o1[ex['transferA'][0]] == '(8, 7, 6, 5)' and bc((8, 1, 6, 1), (7, 1, 5)) == (8, 7, 6, 5), ex)
    check('new case 2: 8,192 tokens instead of 2,048: 16 times the multiply-adds', ex['transferA'][1] == 1 and (8192 * 8192) // (2048 * 2048) == 16, ex)
    big = lambda v: f'{v / 1e9:.3g}B'
    w2 = ex['transferW'][1]
    check('new case 2, why: 32 heads × 2,048² × 128 = 17.2B, × 8,192² = 275B', big(32 * 2048 ** 2 * 128) == '17.2B' and big(32 * 8192 ** 2 * 128) == '275B' and '17.2B' in w2 and '275B' in w2, w2[:300])
    bad = []
    for s in range(1, 301):
        d = ev("s=>window.__ts.drill(s)", s)
        w = bc(tuple(d['a']), tuple(d['b']))
        if (w is None) == d['r']['ok'] or (w is not None and tuple(d['r']['out']) != w): bad.append((s, d['a'], d['b']))
        if w is None:
            k, x, y = first_bad_from_right(tuple(d['a']), tuple(d['b']))
            if d['r']['err']['k'] != k: bad.append(('axis', s))
    check('drill: 300 seeds against np.broadcast_shapes', not bad, bad[:5])
    pg.evaluate("()=>{location.hash=''}"); pg.wait_for_timeout(300)
    tz = ev("()=>window.__tzTs?window.__tzTs.pairs():null")
    check('home tile: four pairs, none of them the prediction’s (3, 1) + (1, 4)', bool(tz) and len(tz) == 4 and not any(t['a'] == [3, 1] and t['b'] == [1, 4] for t in tz), tz)
    for t in tz or []:
        m = ev("id=>window.__tzTs.model(id)", t['id'])
        w = bc(tuple(t['a']), tuple(t['b']))
        check(f'home tile {t["a"]} + {t["b"]}', (w is None and not m['ok']) or (w is not None and m['ok'] and tuple(m['out']) == w), m)

    # ---------- Connect ----------
    pg.evaluate("()=>{location.hash='tensors'}"); pg.wait_for_timeout(500)
    code = ev("()=>document.getElementById('ts-code').textContent")
    lines = code.split('\n')
    check('Connect code: at most 40 lines of at most 74 characters', len(lines) <= 40 and max(len(l) for l in lines) <= 74, (len(lines), max(len(l) for l in lines)))
    run = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=120)
    want_out = [l[2:] for l in lines[-3:]]
    check('Connect code runs and prints its last three comment lines', run.returncode == 0 and run.stdout.strip().split('\n') == want_out, (run.stdout, run.stderr[-300:], want_out))

    # ---------- what the page shows ----------
    bv = ev("()=>[...document.querySelectorAll('#ts-nn .ts-bv')].map(e=>e.textContent)")
    card = ev("()=>document.querySelector('#ts-rail .step.current').innerText")
    jt = ev("()=>document.querySelector('#ts-rail .step.current [data-jump=\"nn\"]').dataset.tipx")
    check('page, part 1: the bars read 2 KiB, 954 GiB (does not fit), 1.86 GiB; the card 500M; the link’s tooltip 954 GiB (1.02 TB) and 1.86 GiB (2.00 GB)', bv == ['2 KiB', '954 GiB, does not fit', '1.86 GiB'] and '500M' in card
          and '954 GiB (1.02 TB)' in jt and '1.86 GiB (2.00 GB)' in jt and '2 KiB' in jt, (bv, card[:200], jt))
    # A4, B6: every conversion once, decimal units with 3 significant digits
    check('formatting: 3 significant digits in GB and MB, no repeated values', (G(100_000_000_000), G(800_000_000), G(16_000_000), G(6_144_000_000_000)) == ('100 GB', '800 MB', '16.0 MB', '6.14 TB'), (G(100_000_000_000), G(800_000_000)))
    r = ev("()=>window.__ts.nn(2000,2000,50)")
    check('B3: NumPy’s float64 doubles the interview sizes: 1.49 GiB and 30.5 MiB', (F(2 * r['bc']['bytes']), F(2 * r['mm']['bytes'])) == ('1.49 GiB', '30.5 MiB'), (F(2 * r['bc']['bytes']), F(2 * r['mm']['bytes'])))
    note = ev("()=>document.getElementById('ts-controls').textContent")
    check('page: Llama‑3.1‑8B written with no-break hyphens (U+2011)', 'Llama‑3.1‑8B' in note and 'Llama-3.1-8B' not in note)
    br.close()

check('no page errors', not errs, errs[:5])
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:40] if fails else 'none')
sys.exit(1 if fails else 0)
