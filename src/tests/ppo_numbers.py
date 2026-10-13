"""Golden numbers for the PPO lab, recomputed independently in Python.

The page trains a toy policy (3 prompts x 5 candidate responses; one logit per response plus one weight w shared by all prompts
that multiplies each response's length) with real policy-gradient steps on PPO's clipped objective, against a toy reward model,
with seeded sampling (mulberry32, seed 7). This script re-implements that toy from the rules stated in the fidelity card, in numpy,
with its own sampler, its own gradient (written from the objective, not copied from the page) and its own bookkeeping, then
compares every trajectory with the page through the read-only hook window.__ppo. It also checks part 6's captions and option results
(the owner's message of 2026-10-08, item 3), the memory numbers (brief S5),
the challenge table, the 6-seed fan of part 4, the exit questions, and every claim the copy makes about the runs.

Run: python3 tests/ppo_numbers.py   (builds out/ppo.html)
"""
import json, math, pathlib, subprocess, sys, itertools
import numpy as np
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/ppo.html', '--only', 'ppo'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'ppo').mkdir(parents=True, exist_ok=True)
(P / 'out' / 'ppo' / 'wrapped-ppo.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'ppo.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'ppo' / 'wrapped-ppo.html')

# ---------- the toy, as stated on the page ----------
REF = np.array([0.23, 0.25, 0.25, 0.24, 0.03])                    # the SFT model = the reference, the same for every prompt
RM = np.array([[0.90, 0.85, 0.80, 0.88, 0.94], [0.60, 0.55, 0.50, 0.58, 0.64], [0.27, 0.22, 0.17, 0.25, 0.31]])
QL = np.array([0.95, 0.75, 0.55, 0.50, 0.20])                     # true quality of response kinds A..E
LEN = np.array([0.2, 0.4, 0.1, 0.7, 1.0])
NP_, NY, NS, MAXS, SEED, EPS, LRV = 3, 5, 4, 200, 7, 0.2, 0.5
ETA = {'n': 8, 'l': 16}
BETAS, KS, FAN = [0, 0.05, 0.2], [1, 4], [7, 1, 2, 3, 5, 11]
P8, GPU = 8030261248, 80e9

def mulberry32(seed):
    """mulberry32 (Tommy Ettinger's public-domain generator), 32-bit unsigned arithmetic."""
    s = seed & 0xFFFFFFFF
    def nxt():
        nonlocal s
        s = (s + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((s ^ (s >> 15)) * (s | 1)) & 0xFFFFFFFF
        t = (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return nxt

def policy(theta, w):
    z = theta + w * LEN[None, :]
    z = z - z.max(axis=1, keepdims=True); e = np.exp(z); return e / e.sum(axis=1, keepdims=True)

def run(base=1, beta=0.05, K=4, clip=1, eta='n', seed=SEED):
    """One run from the SFT model. Per step: 4 responses per prompt from the current policy; reward R = r - beta*log(pi_old/pi_ref);
    advantage A = R - V(prompt) (or R without the value model); K passes of gradient ascent on the clipped surrogate
    sum_prompts mean_n min(rho*A, clip(rho)*A) for the logits, and on the mean over all 12 responses for the shared w;
    then V(prompt) moves halfway to the batch's mean reward."""
    rnd, lr = mulberry32(seed), ETA[eta]
    theta, w = np.tile(np.log(REF), (NP_, 1)), 0.0
    V = RM @ REF
    out = {'rm': [], 'q': [], 'kl': [], 'w': [], 'rho': [], 'pi': [], 'V': []}
    for t in range(MAXS + 1):
        pi = policy(theta, w)
        out['rm'].append(float((pi * RM).sum(axis=1).mean())); out['q'].append(float((pi @ QL).mean()))
        with np.errstate(divide='ignore', invalid='ignore'): klt = np.where(pi > 0, pi * (np.log(pi) - np.log(REF)), 0.0)
        out['kl'].append(float(klt.sum(axis=1).mean())); out['w'].append(w); out['pi'].append(pi.copy()); out['V'].append(V.copy())
        if t == MAXS: break
        ks, ys = [], []
        for k in range(NP_):
            for _ in range(NS):
                u = rnd(); c = np.cumsum(pi[k]); idx = int(np.searchsorted(c, u, side='right'))
                ks.append(k); ys.append(min(idx, NY - 1))
        ks, ys = np.array(ks), np.array(ys)
        old = pi[ks, ys]; R = RM[ks, ys] - beta * (np.log(old) - np.log(REF[ys])); A = R - V[ks] if base else R
        rp, lv = [], []
        for _ in range(K):
            pn = policy(theta, w); rho = pn[ks, ys] / old
            # d/dx of min(rho*A, clip(rho)*A) is rho*A*dlog(pi_y)/dx, or 0 where the clipped term is the smaller (flat) one
            live = np.ones(len(ks), bool) if not clip else ~(((A > 0) & (rho > 1 + EPS)) | ((A < 0) & (rho < 1 - EPS)))
            if t == 0: rp.append(rho.copy()); lv.append(live.copy())
            coef = np.where(live, rho * A, 0.0)
            onehot = np.zeros((len(ks), NY)); onehot[np.arange(len(ks)), ys] = 1
            dlog_th = onehot - pn[ks]                                   # d log pi_y / d theta_k
            dlog_w = LEN[ys] - (pn[ks] * LEN[None, :]).sum(axis=1)       # d log pi_y / d w
            g = np.zeros((NP_, NY)); np.add.at(g, ks, coef[:, None] * dlog_th / NS)
            gw = float((coef * dlog_w).sum() / (NS * NP_))
            theta = theta + lr * g; w = w + lr * gw
        pe = policy(theta, w); r_end = pe[ks, ys] / old
        if t == 0: out['b0'] = dict(ks=ks, ys=ys, A=A, rp=rp, live=lv, end=r_end)   # step 1's batch, pass by pass (part 6's picture)
        out['rho'].append((float(r_end.min()), float(r_end.max())))
        for k in range(NP_): V[k] += LRV * (R[ks == k].mean() - V[k])
    return out

def peak(q, t): return max(range(t + 1), key=lambda u: (q[u], -u))
def hack_from(o, t=MAXS):
    k = peak(o['q'], t); return k if (o['q'][k] - o['q'][t] >= 0.05 and o['rm'][t] > o['rm'][k]) else -1
def runaway(o):
    for u in range(MAXS):
        if o['rho'][u][1] > 3: return u
    return -1

fails, notes = [], []
def check(name, ok, info=''):
    notes.append(('OK   ' if ok else 'FAIL ') + name + (f'  [{info}]' if info else ''))
    if not ok: fails.append(name)

CONFIGS = {
    'loop (no baseline, no KL, 1 pass)': dict(base=0, beta=0, K=1, clip=0, eta='n'),
    'baseline, no KL': dict(base=1, beta=0, K=1, clip=0, eta='n'),
    'baseline, beta 0.05': dict(base=1, beta=0.05, K=1, clip=0, eta='n'),
    'baseline, beta 0.2': dict(base=1, beta=0.2, K=1, clip=0, eta='n'),
    'full PPO (default)': dict(base=1, beta=0.05, K=4, clip=1, eta='n'),
    'drop the KL (scenario 3)': dict(base=1, beta=0, K=4, clip=1, eta='n'),
    'large step, 1 pass': dict(base=1, beta=0.05, K=1, clip=0, eta='l'),
    'large step, 4 passes, no clip': dict(base=1, beta=0.05, K=4, clip=0, eta='l'),
    'large step, 4 passes, clipped': dict(base=1, beta=0.05, K=4, clip=1, eta='l'),
    'challenge start': dict(base=1, beta=0, K=4, clip=0, eta='l'),
}
PY = {k: run(**c) for k, c in CONFIGS.items()}
Q0 = PY['full PPO (default)']['q'][0]

with sync_playwright() as pw:
    b = pw.chromium.launch(); pg = b.new_page(); errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300); pg.evaluate("()=>{location.hash='ppo'}"); pg.wait_for_timeout(300)
    C = pg.evaluate("()=>window.__ppo.consts")
    check('constants: REF, RM, true quality, lengths', np.allclose(C['REF'], REF) and np.allclose(C['RM'], RM) and np.allclose(C['QL'], QL) and np.allclose(C['LEN'], LEN))
    check('constants: 4 responses per prompt, 200 steps, seed 7, eps 0.2, value step 0.5, step sizes 8/16',
          (C['NS'], C['MAXS'], C['SEED'], C['EPS'], C['LRV'], C['ETA']) == (NS, MAXS, SEED, EPS, LRV, ETA))
    for name, c in CONFIGS.items():
        js = pg.evaluate("(c)=>{const T=window.__ppo.train(c); return {rm:T.rm,q:T.q,kl:T.kl,w:T.w,rho:T.rho,V:T.V}}", c)
        o = PY[name]
        d = max(max(abs(a - b_) for a, b_ in zip(js[k], o[k])) for k in ('rm', 'q', 'kl', 'w'))
        dr = max(abs(math.log(a[i]) - math.log(b_[i])) for a, b_ in zip(js['rho'], o['rho']) for i in (0, 1))
        dv = max(abs(a - b_) for va, vb in zip(js['V'], o['V']) for a, b_ in zip(va, vb))
        check(f'trajectory matches: {name}', d < 1e-9 and dr < 1e-7 and dv < 1e-9, f'max |diff| {d:.1e}, log ratio {dr:.1e}, V {dv:.1e}')
    fan = {bs: pg.evaluate(f"()=>window.__ppo.fan({bs})") for bs in (0, 1)}
    pyfan = {bs: [run(base=bs, beta=0, K=1, clip=0, eta='n', seed=s)['q'][30] for s in FAN] for bs in (0, 1)}
    check('part 4 fan: 6 seeds x 2 settings match', all(abs(a - b_) < 1e-9 for bs in (0, 1) for a, b_ in zip(fan[bs], pyfan[bs])))
    mem = {a: pg.evaluate(f"()=>window.__ppo.memOf('{a}')") for a in ('ppo', 'grpo', 'dpo', 'grpo0')}
    # B2 of the review: every beta = 0 setting the page shows, on 40 seeds (the page computes these counts in the browser)
    MANY = {'part 5, beta 0': dict(base=1, beta=0, K=1, clip=0, eta='n'), 'part 5, beta 0.05': dict(base=1, beta=0.05, K=1, clip=0, eta='n'),
            'scenario 3 (beta 0, 4 clipped passes)': dict(base=1, beta=0, K=4, clip=1, eta='n'), 'full PPO': dict(base=1, beta=0.05, K=4, clip=1, eta='n')}
    many_js = {k: pg.evaluate("(c)=>window.__ppo.many(c)", c) for k, c in MANY.items()}
    FANS = {'part 5, beta 0': MANY['part 5, beta 0'], 'part 5, beta 0.05': MANY['part 5, beta 0.05'], 'part 5, beta 0.2': dict(base=1, beta=0.2, K=1, clip=0, eta='n'),
            'tile, score only': dict(base=0, beta=0, K=1, clip=0, eta='n')}
    fan_js = {k: pg.evaluate("(c)=>window.__ppo.fanQ(c,200)", c) for k, c in FANS.items()}
    easy_js = pg.evaluate("()=>window.__ppo.easyTops()")
    tz_js = pg.evaluate("()=>window.__ppo.tzSets")
    chall = pg.evaluate("()=>window.__ppo.chAll()")
    chq = {(beta, K, clip): pg.evaluate("(c)=>window.__ppo.train(c).q", dict(base=1, beta=beta, K=K, clip=clip, eta='l')) for beta, K, clip in itertools.product(BETAS, KS, (0, 1))}
    ex = pg.evaluate("()=>window.__ppo.exits()")
    drills = [pg.evaluate(f"()=>window.__ppo.drill({sd})") for sd in range(7, 37)]
    comp = {k: pg.evaluate("(s)=>window.__ppo.compute(s)", s) for k, s in {
        'kl0': dict(base=1, beta=0, K=1, clip=0, eta='n', steps=200), 'kl05': dict(base=1, beta=0.05, K=1, clip=0, eta='n', steps=200)}.items()}
    # review codex-b9v3, item 7: the lab's own objective and its slope, and part 6's picture of step 1's batch
    GRID = [(a, r) for a in (0.5, -0.5, 0.03, -0.27) for r in (0.5, 0.7, 0.79, 0.81, 1.0, 1.19, 1.21, 1.3, 1.6)]
    objs = {(a, r, c): (pg.evaluate("([a,r,c])=>window.__ppo.obj(a,r,c)", [a, r, c]), pg.evaluate("([a,r,c])=>window.__ppo.slope(a,r,c)", [a, r, c])) for a, r in GRID for c in (0, 1)}
    P6 = {'clipped': dict(base=1, beta=0.05, K=4, clip=1, eta='l'), 'not clipped': dict(base=1, beta=0.05, K=4, clip=0, eta='l')}
    b1 = {k: pg.evaluate("(c)=>window.__ppo.batch1(c)", c) for k, c in P6.items()}
    b1one = pg.evaluate("(c)=>window.__ppo.batch1(c)", dict(base=1, beta=0.05, K=1, clip=0, eta='l'))
    b.close()
check('no page errors', not errs, '; '.join(errs[:3]))

# ---------- review codex-b9v3, item 7: which responses the clipped objective still pushes ----------
def L_obj(a, r, clip=1): return min(r * a, min(max(r, 1 - EPS), 1 + EPS) * a) if clip else r * a
def dL(a, r, clip=1, h=1e-6):   # d L / d log pi_y = d L / d log rho, by central differences on the objective itself
    return (L_obj(a, r * math.exp(h), clip) - L_obj(a, r * math.exp(-h), clip)) / (2 * h)
bad = [(a, r, c, objs[(a, r, c)]) for a, r in GRID for c in (0, 1) if abs(objs[(a, r, c)][0] - L_obj(a, r, c)) > 1e-12 or abs(objs[(a, r, c)][1] - dL(a, r, c)) > 1e-6]
check('the page objective min(ρA, clip(ρ)A) and its slope match the finite-difference slope (36 points, clipped and not)', not bad, str(bad[:3]))
A_ = 0.5
for a, r, push in ((A_, 0.7, True), (-A_, 1.3, True), (A_, 1.3, False), (-A_, 0.7, False)):
    g, gpy = objs[(a, r, 1)][1], dL(a, r)
    check(f'review 验收: advantage {"+" if a > 0 else "−"}{A_} at ratio {r} {"still has a gradient" if push else "has no gradient"}', (abs(g) > 0.1 and abs(gpy) > 0.1 and g * a > 0) if push else (g == 0 and abs(gpy) < 1e-9),
          f'page {g:+.3f}, finite difference {gpy:+.3f}')
for name, c in P6.items():
    o = run(**c)['b0']; worst = 0.0; same = True
    for d in b1[name]:
        ix = np.where((o['ks'] == d['k']) & (o['ys'] == d['y']))[0]
        same &= len(ix) == d['n'] and abs(float(o['A'][ix[0]]) - d['a']) < 1e-12 and abs(float(o['end'][ix[0]]) - d['end']) < 1e-9 * d['end']
        same &= all(bool(o['live'][p][ix[0]]) == d['on'][p] for p in range(4))
        worst = max(worst, max(abs(math.log(float(o['rp'][p][ix[0]]) / d['rp'][p])) for p in range(4)))
    check(f'part 6 picture, {name}: step 1’s responses, advantages, ratios at each pass and pushed or not match', same and worst < 1e-9 and sum(d['n'] for d in b1[name]) == NS * NP_, f'{len(b1[name])} distinct responses, log ratio {worst:.1e}')
D = b1['clipped']
flat2 = [d for d in D if not d['on'][1]]; push2 = [d for d in D if d['on'][1]]
check('part 6 picture, clipped: from pass 2 some responses sit on the plateau and the others are still pushed',
      flat2 and push2 and all((d['a'] > 0 and d['rp'][1] > 1 + EPS) or (d['a'] < 0 and d['rp'][1] < 1 - EPS) for d in flat2), f'{len(flat2)} on the plateau, {len(push2)} pushed at pass 2')
back = [d for d in D for p in range(4) if (d['a'] > 0 and d['rp'][p] < 1 - EPS) or (d['a'] < 0 and d['rp'][p] > 1 + EPS)]
check('part 6 picture, clipped: no response of step 1 leaves the band against its advantage (the text says none is pushed back)', not back, str(back[:1]))
moved = [d for d in D for p in range(1, 4) if not d['on'][p] and not any(d['on'][p:]) and abs(d['end'] - d['rp'][p]) > 0.01]
check('part 6 picture, clipped: a response on the plateau still moves (shared parameters: clipping does not lock a ratio)', bool(moved), ', '.join(f"{'ABCDE'[d['y']]}@{d['k']}" for d in moved))
over = max(D, key=lambda d: d['rp'][1])
check('part 6 picture, clipped: the first pass alone carries a response well past 1.2 (only the crossing pass overshoots)', over['rp'][1] > 1.4 and not over['on'][1], f"{'ABCDE'[over['y']]} of prompt {over['k']}: {over['rp'][1]:.2f}")
red = [d for d in b1['not clipped'] if (d['a'] > 0 and d['rp'][1] > 1 + EPS) or (d['a'] < 0 and d['rp'][1] < 1 - EPS)]
check('part 6 picture, not clipped: at pass 2 responses already past the band in their advantage’s direction are still pushed (red)', red and all(d['on'][1] for d in red), f'{len(red)} responses')

# ---------- the owner's message of 2026-10-08, item 3: what part 6's captions, observation and option results say ----------
NAMES = lambda d: f"{'ABCDE'[d['y']]} of prompt {d['k']}"
o1 = run(base=1, beta=0.05, K=1, clip=0, eta='l')['b0']
check('part 6, one pass: the page’s step-1 batch matches (advantage, end ratio); its ratios end between 0.65 and 1.55 (the “after” caption)',
      all(abs(float(o1['end'][np.where((o1['ks'] == d['k']) & (o1['ys'] == d['y']))[0][0]]) - d['end']) < 1e-9 * d['end'] for d in b1one) and sum(d['n'] for d in b1one) == NS * NP_
      and f"{min(o1['end']):.2f}" == '0.65' and f"{max(o1['end']):.2f}" == '1.55', f"{min(o1['end']):.3f}-{max(o1['end']):.3f}")
mx = {'1-0': max(d['end'] for d in b1one), '4-0': max(d['end'] for d in b1['not clipped']), '4-1': max(d['end'] for d in b1['clipped'])}
pymx = {'1-0': float(max(o1['end'])), '4-0': float(max(run(**P6['not clipped'])['b0']['end'])), '4-1': float(max(run(**P6['clipped'])['b0']['end']))}
check('part 6 option results: step 1’s batch ends with ρ up to 1.55 (1 pass), 24 times (4 passes, no clipping), 1.57 (4 passes, clipped)',
      all(abs(mx[k] - pymx[k]) < 1e-9 * pymx[k] for k in mx) and f"{pymx['1-0']:.2f}" == '1.55' and round(pymx['4-0']) == 24 and f"{pymx['4-1']:.2f}" == '1.57', str({k: round(v, 3) for k, v in pymx.items()}))
top = max(b1['not clipped'], key=lambda d: d['end'])
check('part 6, no clipping: the response pushed furthest is E of the easy prompt, 24 times as likely after 4 passes (observation and caption)', (top['k'], top['y']) == (0, 4) and round(top['end']) == 24, f"{NAMES(top)}: {top['end']:.2f}")
red2 = [d for d in b1['not clipped'] if d['on'][1] and ((d['a'] > 0 and d['rp'][1] > 1 + EPS) or (d['a'] < 0 and d['rp'][1] < 1 - EPS))]
check('part 6, no clipping, pass 2: 5 responses already out of the band the way their advantage pushes are still pushed (the caption’s count)', len(red2) == 5, f'{len(red2)}')
D = b1['clipped']; K4 = 4
hollow = [d for d in D if not d['on'][K4 - 1]]; filled = [d for d in D if d['on'][K4 - 1]]
n_out = [d for d in hollow if d['end'] > 1 + EPS or d['end'] < 1 - EPS]
check('part 6, clipped, after the 4 passes: 6 responses end outside [0.8, 1.2] (hollow) and 2 were pushed at the last pass (filled); no hollow one ends inside the band',
      len(n_out) == 6 and len(filled) == 2 and len(hollow) == len(n_out) and len(D) == 8, f'{len(n_out)} out, {len(hollow)} hollow, {len(filled)} filled')
stay = [d for d in hollow if any(d['on'][p] for p in range(next(q for q in range(K4) if not d['on'][q]), K4))]
check('part 6, clipped: once a response is no longer pushed, it is not pushed again in that batch (“from the pass that took it out”)', not stay, ', '.join(NAMES(d) for d in stay))
topc = max(D, key=lambda d: d['end'])
check('part 6, clipped: the furthest response is E of the easy prompt at 1.57, carried out of the band by one pass (pushed at pass 1, not from pass 2)',
      (topc['k'], topc['y']) == (0, 4) and f"{topc['end']:.2f}" == '1.57' and topc['on'][0] and not topc['on'][1] and topc['rp'][1] > 1 + EPS, f"{NAMES(topc)}: pass 2 starts at {topc['rp'][1]:.2f}, ends {topc['end']:.2f}")
check('part 6, clipped, pass 2: 5 responses no longer pushed (hollow), 3 still learning (filled)', len(flat2) == 5 and len(push2) == 3, f'{len(flat2)} / {len(push2)}')
oc, o1r = PY['large step, 4 passes, clipped'], PY['large step, 1 pass']
check('part 6 observation, clipped: “true quality 0.69, KL 0.25 nats, the same as one pass” (both runs print the same)',
      (f"{oc['q'][MAXS]:.2f}", f"{oc['kl'][MAXS]:.2f}") == (f"{o1r['q'][MAXS]:.2f}", f"{o1r['kl'][MAXS]:.2f}") == ('0.69', '0.25'), f"{oc['q'][MAXS]:.4f} / {oc['kl'][MAXS]:.4f} vs {o1r['q'][MAXS]:.4f} / {o1r['kl'][MAXS]:.4f}")

# ---------- memory (brief S5) ----------
tr, fr = P8 * 18 / 1e9, P8 * 2 / 1e9
check('trained model 144.5 GB, frozen BF16 model 16.1 GB', round(tr, 1) == 144.5 and round(fr, 1) == 16.1, f'{tr:.3f} / {fr:.3f}')
want = {'ppo': (2 * tr + 2 * fr, 321.2, 5), 'grpo': (tr + 2 * fr, 176.7, 3), 'dpo': (tr + fr, 160.6, 3)}
for a, (tot, gb, g) in want.items():
    check(f'{a.upper()}: {gb} GB, at least {g} H100s of 80 GB', round(tot, 1) == gb and math.ceil(tot * 1e9 / GPU) == g and abs(mem[a]['tot'] - tot) < 1e-9 and mem[a]['gpus'] == g,
          f"page {mem[a]['tot']:.3f} GB, {mem[a]['gpus']} GPUs")

# ---------- challenge ----------
CH_KL = C['CH_KL']
rows = {}
for beta, K, clip in itertools.product(BETAS, KS, (0, 1)):
    o = run(base=1, beta=beta, K=K, clip=clip, eta='l'); q, kl = o['q'][MAXS], o['kl'][MAXS]
    n = (beta != 0) + (K != 4) + (clip != 0)
    rows[(beta, K, clip)] = dict(q=q, kl=kl, n=n, ok=q > Q0 and kl <= CH_KL)
exact, chaotic = [], []
for r in chall:
    key = (r['beta'], r['K'], r['clip']); py = rows[key]
    same = abs(r['q'] - py['q']) < 1e-9 and abs(r['kl'] - py['kl']) < 1e-9
    if same: exact.append(key); continue
    # the large step with beta 0.2 and 4 clipped passes is chaotic: last-bit differences in float order (numpy vs the page)
    # grow until one sampled response flips; both runs must agree exactly before that and give the same verdict
    pyq = run(base=1, beta=key[0], K=key[1], clip=key[2], eta='l')['q']
    first = next(i for i, (x, y) in enumerate(zip(chq[key], pyq)) if abs(x - y) > 1e-9)
    chaotic.append((key, first, round(r['q'], 3), round(py['q'], 3)))
    if not (first >= 50 and r['ok'] == py['ok'] and r['n'] == py['n'] and min(r['q'], py['q']) > Q0 + 0.005): exact.append(None)
check('challenge table: 12 combinations, values equal (verdicts equal for a chaotic run)', len(chall) == 12 and None not in exact and all(r['ok'] == rows[(r['beta'], r['K'], r['clip'])]['ok'] for r in chall),
      f'chaotic rows (key, first differing step, page q, python q): {chaotic}')
start = rows[(0, 4, 0)]
check('challenge start fails', not start['ok'], f"q {start['q']:.3f}, KL {start['kl']:.2f}")
okn = [r['n'] for r in rows.values() if r['ok']]
check('no single change passes; the fewest that pass is 2', min(okn) == 2 and not any(r['ok'] for r in rows.values() if r['n'] <= 1), f'passing: {sorted((k, round(v["q"], 3), round(v["kl"], 2)) for k, v in rows.items() if v["ok"])}')
check('Q0 (SFT true quality) as on the page', abs(C['Q0'] - Q0) < 1e-12, f'{Q0:.4f}')
for k, v in sorted(rows.items()): notes.append(f'      challenge {k}: q {v["q"]:.3f}  KL {v["kl"]:.3f}  n {v["n"]}  {"PASS" if v["ok"] else "-"}')

# ---------- claims in the copy ----------
o = PY['loop (no baseline, no KL, 1 pass)']; pi0, pi1 = o['pi'][0], o['pi'][1]
rnd = mulberry32(SEED); cnt = np.zeros((NP_, NY), int)
for k in range(NP_):
    for _ in range(NS):
        u = rnd(); cnt[k][min(int(np.searchsorted(np.cumsum(pi0[k]), u, side='right')), NY - 1)] += 1
down = all(pi1[k][y] < pi0[k][y] for k in range(NP_) for y in range(NY) if cnt[k][y] == 0)
top = int(np.argmax(cnt[1])); once = [y for y in range(NY) if cnt[1][y] == 1 and pi1[1][y] < pi0[1][y]]
check('part 2: unwritten responses lost probability; on the medium prompt the most-written one jumped and a once-written one fell',
      down and pi1[1][top] > pi0[1][top] + 0.3 and len(once) >= 1, f'medium: {"ABCDE"[top]} x{cnt[1][top]} {pi0[1][top]:.0%}->{pi1[1][top]:.0%}; once and fell: {["ABCDE"[y] + f" {pi1[1][y]:.0%}" for y in once]}')
lo1, hi1 = min(pyfan[1]), max(pyfan[1]); lo0, hi0 = min(pyfan[0]), max(pyfan[0])
check('part 4: with the value model all 6 seeds end above the SFT model at step 30', lo1 > Q0, f'{lo1:.3f}-{hi1:.3f} vs Q0 {Q0:.3f}')
check('part 4: without it the seeds spread wider', hi0 - lo0 > 2 * (hi1 - lo1), f'no baseline {lo0:.3f}-{hi0:.3f}, with {lo1:.3f}-{hi1:.3f}')
h = PY['baseline, no KL']; hk = hack_from(h)
check('part 5: no KL -> score rises, quality peaks then falls (red segment)', hk >= 0 and h['rm'][MAXS] > h['rm'][0] and h['q'][MAXS] < Q0, f'peak {h["q"][hk]:.3f} @ {hk}, end q {h["q"][MAXS]:.3f}, rm {h["rm"][0]:.3f}->{h["rm"][MAXS]:.3f}')
check('part 5: no KL -> E is the top response on every prompt', all(int(np.argmax(p)) == 4 for p in h['pi'][MAXS]))
k5 = PY['baseline, beta 0.05']
check('part 5: beta 0.05 ends a little above the SFT model and stays there', Q0 < k5['q'][MAXS] < Q0 + 0.1 and max(k5['q'][100:]) - min(k5['q'][100:]) < 0.02, f'{k5["q"][MAXS]:.3f}')
e5, e0 = float(np.mean([p[4] for p in k5['pi'][MAXS]])), float(np.mean([p[4] for p in h['pi'][MAXS]]))
check('part 5: E is written far less with beta 0.05 than with no KL penalty', e5 < 0.2 and e0 > 0.9, f'{e5:.0%} vs {e0:.0%} (SFT 3%)')
k2 = PY['baseline, beta 0.2']
check('part 5: beta 0.2 barely moves (KL below 0.05 nats)', k2['kl'][MAXS] < 0.05 and hack_from(k2) < 0, f'KL {k2["kl"][MAXS]:.3f}, q {k2["q"][MAXS]:.3f}')
one, nc, cl = PY['large step, 1 pass'], PY['large step, 4 passes, no clip'], PY['large step, 4 passes, clipped']
u = runaway(nc)
check('part 6: 4 passes without clipping run off (a ratio above 3)', u >= 0, f'step {u + 1}: {nc["rho"][u][1]:.1f}x; KL {nc["kl"][MAXS]:.2f} vs 1 pass {one["kl"][MAXS]:.2f}')
check('part 6: no clipping ends with a larger KL than one pass', nc['kl'][MAXS] > one['kl'][MAXS] + 0.3)
mx = max(r[1] for r in cl['rho'])
check('part 6: clipped, the largest ratio stays below 3 (only the crossing pass overshoots)', mx < 3 and runaway(cl) < 0, f'{mx:.2f}')
check('part 6: clipped ends like one pass (true quality within 0.01, KL within 0.02 nats)', abs(cl['q'][MAXS] - one['q'][MAXS]) < 0.01 and abs(cl['kl'][MAXS] - one['kl'][MAXS]) < 0.02,
      f'q {cl["q"][MAXS]:.3f} vs {one["q"][MAXS]:.3f}; KL {cl["kl"][MAXS]:.3f} vs {one["kl"][MAXS]:.3f}')
check('part 8 recap numbers equal the runs', abs(comp['kl0']['rm'] - h['rm'][MAXS]) < 1e-9 and abs(comp['kl0']['q'] - h['q'][MAXS]) < 1e-9 and abs(comp['kl05']['q'] - k5['q'][MAXS]) < 1e-9)
full = PY['full PPO (default)']
check('scenario 1: full PPO ends above the SFT model and near it', full['q'][MAXS] > Q0 and full['kl'][MAXS] < 0.5, f'q {full["q"][MAXS]:.3f}, KL {full["kl"][MAXS]:.3f}')
s2 = PY['loop (no baseline, no KL, 1 pass)']
check('scenario 2: without a baseline each prompt locks onto one response (above 90%) and E stays rare', all(max(p) > 0.9 for p in s2['pi'][MAXS]) and max(p[4] for p in s2['pi'][MAXS]) < 0.05,
      f'tops {["ABCDE"[int(np.argmax(p))] for p in s2["pi"][MAXS]]}, q {s2["q"][MAXS]:.3f}')
d3 = PY['drop the KL (scenario 3)']
check('scenario 3: dropping the KL penalty hacks', hack_from(d3) >= 0, f'end q {d3["q"][MAXS]:.3f}')

# ---------- exit questions ----------
r0, v, beta = RM[2][0], 0.24, 0.05
R = r0 - beta * math.log(0.30 / 0.25); Aq = R - v
fmt = lambda x: ('+' if x >= 0 else '−') + f'{abs(x):.3f}'
f2s = lambda x: ('+' if x >= 0 else '−') + f'{abs(x):.2f}'
check('transfer 1: options “a little less than +0.03 / exactly +0.03 / about +0.26”, and the right one is the first', f2s(r0 - v) in ex['t0'][0] and f2s(r0 - v) in ex['t0'][1] and f2s(R) in ex['t0'][2]
      and 0 < Aq < r0 - v and (r0 - v) - Aq < 0.015, f'{ex["t0"]}; advantage {Aq:+.4f}, without the KL term {r0 - v:+.4f}, reward {R:.4f}')
check('transfer 1: the reveal carries the arithmetic (log 1.2, the reward, the advantage)', f'{math.log(1.2):.4f}' in ex['why0'] and f'{R:.4f}' in ex['why0'] and fmt(Aq) in ex['why0'], ex['why0'][:160])
check('transfer 2: clipped objective 2.4 vs unclipped 3.0', ex['t1'][0].startswith('2.4') and ex['t1'][1].startswith('3.0,'), str(ex['t1']))

# ---------- B2 of the review: the beta = 0 settings over many seeds, and what the page says about them ----------
S40 = list(range(1, 41))
many_py = {k: [run(seed=s, **c)['q'][MAXS] for s in S40] for k, c in MANY.items()}
for k in MANY:
    check(f'40 seeds match the page: {k}', max(abs(a - b_) for a, b_ in zip(many_js[k], many_py[k])) < 1e-9)
low = sum(q <= 0.30 for q in many_py['part 5, beta 0'])
check('part 5, beta 0: about half of 40 seeds collapse to 0.30 or below, the rest end higher (the page prints the count)', 10 <= low <= 30 and max(many_py['part 5, beta 0']) > Q0, f'{low} of 40 at or below 0.30; best {max(many_py["part 5, beta 0"]):.3f}')
check('part 5 and full PPO, beta 0.05: every one of 40 seeds ends at the same point, a little above the SFT model', all(abs(q - 0.6854) < 0.001 for q in many_py['part 5, beta 0.05'] + many_py['full PPO']),
      f'{min(many_py["part 5, beta 0.05"]):.4f}-{max(many_py["part 5, beta 0.05"]):.4f}')
below3 = sum(q < Q0 for q in many_py['scenario 3 (beta 0, 4 clipped passes)'])
check('scenario 3: some of 40 seeds end below the SFT model and the rest do not (the page prints the count)', 0 < below3 < 40, f'{below3} of 40 below {Q0:.3f}')
d3 = PY['drop the KL (scenario 3)']; k3 = peak(d3['q'], MAXS)
check('scenario 3, seed 7: true quality first rises above the KL-penalised end point, then falls from its peak; E takes the medium prompt only',
      d3['q'][k3] > PY['full PPO (default)']['q'][MAXS] + 0.1 and hack_from(d3) == k3 and [int(np.argmax(p)) == 4 for p in d3['pi'][MAXS]] == [False, True, False], f'peak {d3["q"][k3]:.3f} at step {k3}')
fan_py = {k: [run(seed=s, **c)['q'][MAXS] for s in FAN] for k, c in FANS.items()}
check('part 5 fan and tile: the 6 seeds match the page', all(max(abs(a - b_) for a, b_ in zip(fan_js[k], fan_py[k])) < 1e-9 for k in FANS))
f0 = [run(seed=s, **MANY['part 5, beta 0']) for s in FAN]
check('part 5, beta 0: all 6 seeds of the fan end at 0.30 or below with E on top for every prompt', all(o['q'][MAXS] <= 0.30 and all(int(np.argmax(p)) == 4 for p in o['pi'][MAXS]) for o in f0),
      f'{[round(o["q"][MAXS], 3) for o in f0]}')
check('part 5, beta 0.05 and 0.2: the 6 seeds end at one value each (shown as “all 6 seeds”)', round(min(fan_py['part 5, beta 0.05']), 2) == round(max(fan_py['part 5, beta 0.05']), 2) and round(min(fan_py['part 5, beta 0.2']), 2) == round(max(fan_py['part 5, beta 0.2']), 2))
check('tile, score only: the 6 seeds spread widely', max(fan_py['tile, score only']) - min(fan_py['tile, score only']) > 0.3, f'{min(fan_py["tile, score only"]):.3f}-{max(fan_py["tile, score only"]):.3f}')
check('tile: the three settings are score only / + value model / + KL penalty (1 pass, normal step)', [(t['base'], t['beta'], t['K'], t['clip'], t['eta']) for t in tz_js] == [(0, 0, 1, 0, 'n'), (1, 0, 1, 0, 'n'), (1, 0.05, 1, 0, 'n')])
tops = []
for s in FAN:
    p = run(base=0, beta=0, K=1, clip=0, eta='n', seed=s)['pi'][30][0]; tops.append(('ABCDE'[int(np.argmax(p))], float(max(p))))
check('part 4: after 30 steps of push by the score the easy prompt is locked (above 90%) on one response per seed, as printed', easy_js == [t for t, _ in tops] and all(m > 0.9 for _, m in tops) and len(set(easy_js)) >= 3,
      f'{tops}')
chz = [r for r in rows.items() if r[0][0] == 0]
check('challenge: every beta = 0 row ends with a KL far over 0.5 nats; three land on E, the clipped one on A (seed 7)', all(v['kl'] > 1.0 for _, v in chz)
      and sum(not v['q'] > Q0 for _, v in chz) == 3 and rows[(0, 4, 1)]['q'] > 0.9, f'{[(k, round(v["q"], 3), round(v["kl"], 2)) for k, v in chz]}')
chz20 = {k: [run(base=1, beta=0, K=k[1], clip=k[2], eta='l', seed=s) for s in range(1, 21)] for k, _ in chz}
check('challenge: with beta = 0 no seed of 20 passes (true quality and KL), for each of the four beta = 0 rows', all(not (o['q'][MAXS] > Q0 and o['kl'][MAXS] <= 0.5) for v in chz20.values() for o in v),
      f'min KL {min(o["kl"][MAXS] for v in chz20.values() for o in v):.2f}; below the SFT model: {[(k, sum(o["q"][MAXS] < Q0 for o in v)) for k, v in chz20.items()]}')
tz20 = [run(base=0, beta=0, K=1, clip=0, eta='n', seed=s)['q'][MAXS] for s in range(1, 21)]
check('tile, score only, on 20 seeds: the end points spread widely, as on its 6 seeds', max(tz20) - min(tz20) > 0.3, f'{min(tz20):.3f}-{max(tz20):.3f}')
check('part 7: GRPO with beta = 0 (no reference model) 160.6 GB', round(tr + fr, 1) == 160.6 and abs(mem['grpo0']['tot'] - (tr + fr)) < 1e-9, f"page {mem['grpo0']['tot']:.3f}")

# ---------- interview drill: every answer line recomputed from the numbers the question prints ----------
bad = []
for d in drills:
    x = round(math.log(d['pOld'] / d['pRef']), 4); R = round(d['r'] - 0.05 * x, 4); Av = round(R - d['V'], 4)
    c = min(max(d['rho'], 0.8), 1.2); u = round(d['rho'] * Av, 4); k = round(c * Av, 4)
    flat = (Av > 0 and d['rho'] > 1.2) or (Av < 0 and d['rho'] < 0.8)
    if not (abs(x - d['x']) < 1e-9 and abs(R - d['R']) < 1e-9 and abs(Av - d['A']) < 1e-9 and abs(u - d['u']) < 1e-9 and abs(k - d['k']) < 1e-9 and abs(min(u, k) - d['obj']) < 1e-9 and flat == d['flat'] and abs(Av) >= 0.01):
        bad.append(d)
check('interview drill: 30 seeds, each answer recomputed from the printed numbers', not bad and len(drills) == 30, str(bad[:2]))
check('interview drill: both cases occur (still pushed / clipped flat)', any(d['flat'] for d in drills) and not all(d['flat'] for d in drills))

print('\n'.join(notes))
print(f'\n{len(notes) - sum(1 for n in notes if n.startswith("      "))} checks, {len(fails)} failed')
sys.exit(1 if fails else 0)
