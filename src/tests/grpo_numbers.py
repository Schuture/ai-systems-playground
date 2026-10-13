"""Recompute every number the GRPO lab shows, independently, and compare with the page.

The toy is re-derived here from its stated rules, not ported from labs/grpo.js:
- a policy over 10 candidate answers (true sum - 4 ... + 5) per question; near misses start at -0.35 * |offset|, the right answer's
  logit is chosen so that softmax gives the question's starting accuracy;
- one GRPO step = sample G answers per question with mulberry32 (seed 20261004, one draw per answer, inverse CDF), reward 1 / 0,
  advantage (r - mean) / (std_{n-1} + 1e-4) (or r - mean), then one gradient step of size 4 (one update per batch, TRL's
  num_iterations = 1, so rho = 1 and the clip never acts) on
  L = -mean_i min(rho A, clip(rho, 0.8, 1.2) A) + 0.05 * mean_i (e^u - u - 1), u = log pi_ref - log pi, averaged over all 8 G answers.
  The gradient of each term with respect to a question's logits is derived by hand below (d log pi(o) / dz = onehot(o) - pi).
Also: the demo groups, p^G + (1 - p)^G, the G = 64 figures of part 6, the matched step size of the no-division scenario (review B1),
the memory figures of brief S5, the challenge tiers, the capability-layer numbers, the home tile's nested groups (review B2), and the
numbers printed in the page's own text.
Run: python3 tests/grpo_numbers.py
"""
import math, pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/grpo.html', '--only', 'grpo'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'grpo').mkdir(parents=True, exist_ok=True)
(P / 'out' / 'grpo' / 'wrapped-num.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'grpo.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'grpo' / 'wrapped-num.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(f'{name} :: {info}'); print('FAIL', name, '::', info)


# ---------- the toy, from its rules ----------
SEED, ETA, BETA, EPS, MU, ADV_EPS, T_MAX, NQ = 20261004, 4.0, 0.05, 0.2, 1, 1e-4, 30, 8
OFF = list(range(-4, 6)); R = OFF.index(0)
SETS = {'mix': [0.95, 0.85, 0.7, 0.5, 0.3, 0.15, 0.07, 0.03], 'easy': [0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.9, 0.88],
        'hard': [0.08, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02], 'ch': [0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04]}
U32 = 0xFFFFFFFF


class Mulberry:
    def __init__(self, seed): self.a = seed & U32
    def __call__(self):
        self.a = (self.a + 0x6D2B79F5) & U32
        t = ((self.a ^ (self.a >> 15)) * (1 | self.a)) & U32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & U32)) & U32) ^ t
        return ((t ^ (t >> 14)) & U32) / 4294967296


def start_logits(p):
    near = [-0.35 * abs(o) for o in OFF]
    rest = sum(math.exp(v) for j, v in enumerate(near) if j != R)
    # softmax(z)[R] = e^c / (e^c + rest) = p  =>  c = log(p rest / (1 - p))
    near[R] = math.log(p * rest / (1 - p)); return near


def probs(z):
    m = max(z); e = [math.exp(v - m) for v in z]; s = 0.0
    for v in e: s += v
    return [v / s for v in e]


def logprobs(z):
    m = max(z); s = 0.0
    for v in z: s += math.exp(v - m)
    c = m + math.log(s); return [v - c for v in z]


def sample(p, u):
    c = 0.0
    for j, v in enumerate(p):
        c += v
        if u < c: return j
    return len(p) - 1


def advantages(r, scale):
    G = len(r); mean = sum(r) / G
    sd = math.sqrt(sum((x - mean) ** 2 for x in r) / (G - 1))
    return mean, sd, [((x - mean) / (sd + ADV_EPS)) if scale else (x - mean) for x in r]


def run(set_, G, scale, eta=ETA):
    rnd = Mulberry(SEED); ref = [start_logits(p) for p in SETS[set_]]; Z = [z[:] for z in ref]
    out = {'acc': [], 'kl': [], 'eq': [0.0], 'gen': [0], 'o': [], 'clip': 0, 'act': 0, 'amp': []}

    def stats():
        a = 0.0; k = 0.0
        for q in range(NQ):
            p = probs(Z[q]); a += p[R]
            lp, lr = logprobs(Z[q]), logprobs(ref[q]); k += sum(math.exp(lp[j]) * (lp[j] - lr[j]) for j in range(10))
        out['acc'].append(a / NQ); out['kl'].append(k / NQ)
    stats()
    for t in range(T_MAX):
        batch = []
        for q in range(NQ):
            p = probs(Z[q]); lp = logprobs(Z[q]); o = [sample(p, rnd()) for _ in range(G)]
            r = [1.0 if j == R else 0.0 for j in o]; mean, sd, A = advantages(r, scale)
            batch.append((o, A, [lp[j] for j in o], len(set(r)) == 1))
            if len(set(r)) > 1: out['amp'].append(1 / (sd + ADV_EPS))   # how much the division enlarges a mixed group's advantages
        out['o'].append([b[0] for b in batch])
        for _ in range(MU):
            grads = []
            for q in range(NQ):
                o, A, old, _eq = batch[q]; p = probs(Z[q]); lp = logprobs(Z[q]); lr = logprobs(ref[q]); g = [0.0] * 10
                for i in range(G):
                    rho = math.exp(lp[o[i]] - old[i]); a = A[i]
                    clipped = (a > 0 and rho > 1 + EPS) or (a < 0 and rho < 1 - EPS)
                    if a != 0:
                        out['act'] += 1; out['clip'] += clipped
                    # d/dz of -(rho a)/(Q G) is -(a rho)(onehot - p)/(Q G) unless the clip holds it; d/dz of beta k3/(QG): beta (1 - e^u)(onehot - p)/(QG)
                    u = lr[o[i]] - lp[o[i]]
                    coef = (0.0 if clipped else -a * rho / (NQ * G)) + BETA * (1 - math.exp(u)) / (NQ * G)
                    for j in range(10): g[j] += coef * ((1.0 if j == o[i] else 0.0) - p[j])
                grads.append(g)
            Z = [[Z[q][j] - eta * grads[q][j] for j in range(10)] for q in range(NQ)]
        stats(); out['eq'].append(sum(b[3] for b in batch) / NQ); out['gen'].append((t + 1) * NQ * G)
    out['pR'] = [probs(z)[R] for z in Z]
    return out


def demo(p, seed, G, scale):
    rnd = Mulberry(seed); pr = probs(start_logits(p)); o = [sample(pr, rnd()) for _ in range(G)]
    r = [1.0 if j == R else 0.0 for j in o]; mean, sd, A = advantages(r, scale)
    return o, r, mean, sd, A


with sync_playwright() as pw:
    b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 1000}); errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300); pg.evaluate("()=>{location.hash='grpo'}"); pg.wait_for_timeout(600)
    check('page hook present', pg.evaluate('()=>!!window.__grpo'))
    C = pg.evaluate('()=>window.__grpo.consts')
    check('constants', (C['SEED'], C['ETA'], C['BETA'], C['EPS'], C['MU'], C['MAXS'], C['NQ']) == (SEED, ETA, BETA, EPS, MU, T_MAX, NQ), C)
    for k, v in SETS.items(): check(f'set {k} starting accuracies', C['SETS'][k]['p'] == v, C['SETS'][k]['p'])
    # starting policy
    for p in (0.95, 0.3, 0.03):
        js = pg.evaluate(f'()=>window.__grpo.initP({p})'); py = probs(start_logits(p))
        check(f'start policy p={p}', max(abs(a - c) for a, c in zip(js, py)) < 1e-12 and abs(py[R] - p) < 1e-12, (js[R], py[R]))
    # every run the page can show
    runs = [(s, G, sc) for s in ('mix', 'easy', 'hard') for G in (2, 4, 8, 16) for sc in ('on', 'off')] + [('ch', G, 'on') for G in (2, 4, 8, 16, 32)]
    for s, G, sc in runs:
        js = pg.evaluate(f"()=>window.__grpo.train('{s}',{G},'{sc}')"); py = run(s, G, sc == 'on')
        check(f'{s} G={G} std={sc}: sampled answers identical', js['o'] == py['o'], 'first differing step: ' + str(next((t for t in range(T_MAX) if js['o'][t] != py['o'][t]), None)))
        check(f'{s} G={G} std={sc}: accuracy curve', max(abs(a - c) for a, c in zip(js['acc'], py['acc'])) < 1e-9, (js['acc'][-1], py['acc'][-1]))
        check(f'{s} G={G} std={sc}: KL curve', max(abs(a - c) for a, c in zip(js['kl'], py['kl'])) < 1e-9, (js['kl'][-1], py['kl'][-1]))
        check(f'{s} G={G} std={sc}: all-equal share', max(abs(a - c) for a, c in zip(js['eq'], py['eq'])) < 1e-12)
        check(f'{s} G={G} std={sc}: answers generated', js['gen'] == py['gen'])
        check(f'{s} G={G} std={sc}: clip count', (js['clip'], js['act']) == (py['clip'], py['act']), (js['clip'], js['act'], py['clip'], py['act']))
    mix8 = run('mix', 8, True); mix2 = run('mix', 2, True)
    check('default run: the clip never acts (the page says so)', mix8['clip'] == 0, mix8['clip'])
    print(f"default run: accuracy {mix8['acc'][0]:.4f} -> {mix8['acc'][-1]:.4f}, KL {mix8['kl'][-1]:.4f}; G=2 -> {mix2['acc'][-1]:.4f}; answers {mix8['gen'][-1]}")
    check('default run: 44% -> 84% with 1,920 answers; G = 2 reaches 69%', [f"{x * 100:.0f}" for x in (mix8['acc'][0], mix8['acc'][-1], mix2['acc'][-1])] == ['44', '84', '69'] and mix8['gen'][-1] == 1920, (mix8['acc'][0], mix8['acc'][-1], mix2['acc'][-1]))
    # review B1: the no-division run with the step enlarged by the mean enlargement of the mixed groups
    nost = run('mix', 8, False); amp = sum(mix8['amp']) / len(mix8['amp']); mt = run('mix', 8, False, ETA * amp)
    jm = pg.evaluate("()=>window.__grpo.matched('mix',8)")
    check('matched step: the division enlarges the mixed groups about 2.4 times on average', abs(jm['amp'] - amp) < 1e-9 and f'{amp:.1f}' == '2.4', (jm['amp'], amp))
    check('matched step: the page runs the same no-division run with step 4 x 2.4', abs(jm['acc'] - mt['acc'][-1]) < 1e-9, (jm['acc'], mt['acc'][-1]))
    check('no division: 71% against 84% at the same step; under 1 point apart at the matched step', f"{nost['acc'][-1] * 100:.0f}" == '71' and abs(mt['acc'][-1] - mix8['acc'][-1]) * 100 < 1, (nost['acc'][-1], mt['acc'][-1], mix8['acc'][-1]))
    print(f"no division: {nost['acc'][-1]:.4f}; matched step {ETA * amp:.3f} (x{amp:.3f}): {mt['acc'][-1]:.4f}")
    # part 6, G = 64: the best-trained question after the default run, and the share of its groups of 64 that are all right
    pm = max(mix8['pR'])
    check('part 6: best question at 98% after 30 steps; 31% of its groups of 64 all right', f'{pm * 100:.0f}' == '98' and f'{pm ** 64 * 100:.0f}' == '31', (pm, pm ** 64))
    # demo groups
    o, r, mean, sd, A = demo(0.3, 6, 8, True)
    js = pg.evaluate("()=>window.__grpo.demo('g2','on')")
    check('demo g2: same answers', js['o'] == o, (js['o'], o))
    check('demo g2: 2 of 8 right, mean 0.25', sum(r) == 2 and mean == 0.25, (sum(r), mean))
    check('demo g2: std 0.46, advantages +1.62 / -0.54', f'{sd:.2f}' == '0.46' and f'{A[r.index(1)]:.2f}' == '1.62' and f'{A[r.index(0)]:.2f}' == '-0.54', (sd, A))
    check('demo g2: right answers 3 times as far from the baseline', abs((1 - mean) / mean - 3) < 1e-12)
    o7, r7, m7, sd7, A7 = demo(0.85, 3, 8, True); _, _, _, _, A7off = demo(0.85, 3, 8, False)
    js7 = pg.evaluate("()=>window.__grpo.demo('g7','on')")
    check('demo g7: same answers, 7 of 8 right', js7['o'] == o7 and sum(r7) == 7, (js7['o'], o7))
    check('demo g7: mean 0.875, std 0.35, wrong answer -2.47 (about -2.5), right +0.35', f'{m7:.3f}' == '0.875' and f'{sd7:.2f}' == '0.35' and f'{A7[r7.index(0)]:.2f}' == '-2.47' and f'{A7[r7.index(0)]:.1f}' == '-2.5', (m7, sd7, A7))
    check('demo g7 without std: -0.88 and +0.125, about a third', f'{A7off[r7.index(0)]:.2f}' == '-0.88' and f'{A7off[r7.index(1)]:.3f}' == '0.125' and 0.3 < A7off[r7.index(0)] / A7[r7.index(0)] < 0.4, A7off)
    check('demo g7: -2.47 is larger than any advantage in part 4', abs(A7[r7.index(0)]) > max(abs(x) for x in A))
    # p^G + (1 - p)^G
    pe = lambda p, G: p ** G + (1 - p) ** G
    check('p=0.5, G=2: 50%', f'{pe(0.5, 2) * 100:.0f}' == '50')
    check('p=0.9, G=8: 43%', f'{pe(0.9, 8) * 100:.0f}' == '43' and f'{pe(0.9, 8) * 100:.1f}' == '43.0', pe(0.9, 8))
    check('p=0.5, G=8: under 1%', pe(0.5, 8) < 0.01)
    check('G=64 generates 8 times as many answers as G=8', 64 / 8 == 8)
    check('page pEq', abs(pg.evaluate('()=>window.__grpo.pEq(0.9,8)') - pe(0.9, 8)) < 1e-15)
    # memory (brief S5)
    NPAR = 8030261248; t, f = NPAR * 18 / 1e9, NPAR * 2 / 1e9
    mem = pg.evaluate('()=>window.__grpo.mem()')
    exp = {'ppo': 2 * t + 2 * f, 'rm': t + 2 * f, 'chk': t + f}
    for row in mem: check(f"memory {row['k']}", abs(row['tot'] - exp[row['k']]) < 1e-9, row)
    check('memory figures 321.2 / 176.7 / 160.6 / 144.5 / 16.1 GB', [f'{v:.1f}' for v in (2 * t + 2 * f, t + 2 * f, t + f, t, f)] == ['321.2', '176.7', '160.6', '144.5', '16.1'])
    check('saved = the value model', f'{(2 * t + 2 * f) - (t + 2 * f):.1f}' == '144.5')
    check('block split 16.06 / 32.12 / 96.36 GB', [f'{NPAR * k / 1e9:.2f}' for k in (2, 4, 12)] == ['16.06', '32.12', '96.36'])
    # challenge
    chs = {G: run('ch', G, True) for G in (2, 4, 8, 16, 32)}
    page_ch = pg.evaluate('()=>[2,4,8,16,32].map(G=>window.__grpo.chPoint(G))')
    for c in page_ch: check(f"challenge G={c['G']}", abs(c['acc'] - chs[c['G']]['acc'][-1]) < 1e-9 and c['ok'] == (chs[c['G']]['acc'][-1] >= 0.8) and c['gen'] == 8 * c['G'] * 30, c)
    oks = [G for G in (2, 4, 8, 16, 32) if chs[G]['acc'][-1] >= 0.8]
    check('challenge: default G=2 fails, G=4 fails, G=8 is the minimal pass, larger pass', oks == [8, 16, 32], {G: round(chs[G]['acc'][-1], 4) for G in chs})
    check('challenge starting accuracy 12%', f"{chs[8]['acc'][0] * 100:.0f}" == '12', chs[8]['acc'][0])
    # home tile (review B2): one sequence of 32 draws, a group of G takes the first G; the mean gets closer to 40% at every larger G
    tile = pg.evaluate('()=>window.__grpoTile?{r:window.__grpoTile.rewards(),seed:window.__grpoTile.seed,p:window.__grpoTile.p}:null')
    rnd = Mulberry(20261010); pr = probs(start_logits(0.4)); tr = [1 if sample(pr, rnd()) == R else 0 for _ in range(32)]
    ks = [sum(tr[:G]) for G in (2, 4, 8, 16, 32)]; errs_ = [abs(k / G - 0.4) for k, G in zip(ks, (2, 4, 8, 16, 32))]
    check('tile: the page draws the same 32 answers (seed 20261010, accuracy 40%)', tile is not None and tile['r'] == tr and tile['seed'] == 20261010 and tile['p'] == 0.4, tile)
    check('tile: 2 of 2, 3 of 4, 4 of 8, 7 of 16, 13 of 32 right; each larger group is closer to 40%', ks == [2, 3, 4, 7, 13] and all(errs_[i + 1] < errs_[i] for i in range(4)), (ks, errs_))
    print('challenge:', {G: round(chs[G]['acc'][-1], 4) for G in chs})
    # capability layer: 1 of 4 right
    m1 = 0.25; sd1 = math.sqrt(sum((x - m1) ** 2 for x in [1, 0, 0, 0]) / 3); a1 = (1 - m1) / (sd1 + 1e-4)
    check('new case 1: std 0.50, advantage +1.50', f'{sd1:.2f}' == '0.50' and f'{a1:.2f}' == '1.50', (sd1, a1))
    # drill: seeded groups give advantages that sum to 0 and the stated formula
    for seed in (1, 2, 3, 4, 5):
        d = pg.evaluate(f'()=>window.__grpo.drill({seed})'); k = sum(d['r']); G = d['G']
        m = k / G; sdd = math.sqrt(k * (G - k) / (G * (G - 1))); ar = (1 - m) / (sdd + 1e-4)
        check(f'drill {seed}', 0 < k < G and abs(d['adv'][d['r'].index(1)] - ar) < 1e-12 and abs(sum(d['adv'])) < 1e-9, d)
    # numbers in the page's own text, in both languages
    def text_of_part(i, lang):
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(150)
        pg.evaluate(f"()=>{{const b=document.querySelector('#grpo-rail [data-go=\"{i}\"]'); if(b) b.click();}}"); pg.wait_for_timeout(250)
        return pg.inner_text('#grpo-rail')
    for (vw, vh) in ((1440, 1000), (390, 844)):
      pg.set_viewport_size({'width': vw, 'height': vh})
      for lang in ('en', 'zh'):
        pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(200); pg.evaluate("()=>{location.hash='grpo'}"); pg.wait_for_timeout(500)
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(150)
        t1 = pg.inner_text('#grpo-rail')
        check(f'{lang} {vw} part 1: 144.5 GB and 321.2 GB', '144.5 GB' in t1 and '321.2 GB' in t1, t1[:200])
        memtxt = pg.inner_text('#grpo-pan-mem')
        check(f'{lang} {vw} memory chart totals', all(x in memtxt for x in ('321.2 GB', '176.7 GB', '160.6 GB')), memtxt[:200])
        check(f'{lang} {vw} memory note: beta = 0 figures', '160.6 GB' in pg.inner_text('#grpo-m-note') and '144.5 GB' in pg.inner_text('#grpo-m-note'))
        ov = pg.evaluate('()=>document.documentElement.scrollWidth-innerWidth')
        check(f'{lang} {vw} no horizontal overflow', ov <= 0, ov)
    # the owner's review of 2026-10-09: Show the math substitutes the numbers of the group in the chart (H2), and part 8 and the memory
    # note name what a reward model adds (item 5): its frozen BF16 weights, 2 bytes per parameter
    pg.set_viewport_size({'width': 1440, 'height': 1000})
    pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(200); pg.evaluate("()=>{location.hash='grpo'}"); pg.wait_for_timeout(500)
    pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(150)
    pg.evaluate("()=>{const b=document.querySelector('#grpo-rail [data-go=\"3\"]'); if(b) b.click();}"); pg.wait_for_timeout(400)
    mt = ''.join(pg.evaluate("()=>document.querySelector('#grpo-mathbody').textContent").split()).replace('\u2212', '-')
    o, r, mean, sd, A = demo(0.3, 6, 8, True); k = int(sum(r)); nn = len(r)
    sd3 = f'{math.sqrt(k * (nn - k) / (nn * (nn - 1))):.3f}'; p2 = 0.30; Pz = p2 ** nn + (1 - p2) ** nn
    check(f'Show the math, part 4: mean 0.25, std {sd3}, advantages +1.62 and -0.54 recomputed from the shown std, P = {Pz:.3f} for p = 0.30, G = 8',
          all(x in mt for x in ('0.25', sd3, '+1.62', '-0.54', f'{Pz:.3f}')) and f'{(1 - mean) / (float(sd3) + 1e-4):.2f}' == '1.62' and f'{(0 - mean) / (float(sd3) + 1e-4):.2f}' == '-0.54', (mt[-300:], sd3, Pz))
    pg.evaluate("()=>{const b=document.querySelector('#grpo-rail [data-go=\"7\"]'); if(b) b.click();}"); pg.wait_for_timeout(400)
    t8 = pg.inner_text('#grpo-rail'); rm = f'{NPAR * 2 / 1e9:.1f} GB'
    check(f'part 8: the reward model adds {rm}; GRPO takes {t + 2 * f:.1f} GB with it and {t + f:.1f} GB with a checker', all(x in t8 for x in (rm, f'{t + 2 * f:.1f} GB', f'{t + f:.1f} GB')), t8[:400])
    check(f'memory note: the reward model\'s {rm}', rm in pg.inner_text('#grpo-m-note'), pg.inner_text('#grpo-m-note'))
    stage_txt = pg.inner_text('#grpo-stage')
    check('no NaN or undefined on the stage', 'NaN' not in stage_txt and 'undefined' not in stage_txt)
    check('no page errors', not errs, errs[:3])
    b.close()

print(f'{n - len(fails)} / {n} passed')
sys.exit(1 if fails else 0)
