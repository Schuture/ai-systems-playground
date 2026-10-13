"""Recompute every number the Embeddings lab shows, independently, and compare with the page.

The small encoder is re-derived here from its stated rules with numpy, not ported from labs/embeddings.js:
- tokens: English words [a-z0-9]+ joined by - or . (lower case); Chinese: every two adjacent characters of a run of CJK characters
  (a lone character by itself), ASCII runs as words; the vocabulary is the tokens of the 24 passages, then of the 48 training
  questions, in order; a held-out question keeps only known tokens;
- table: V x d numbers, 2u - 1 with u from mulberry32(seed 20261009), row by row; a text's vector = mean of its rows / its length;
- a step: mulberry32(seed + 1) draws B different passages (partial Fisher-Yates) and for each one of its two questions; logits =
  cosine / tau; loss = mean over rows of -log softmax(row)[own]; the exact gradient (softmax - I) / (tau B) through the dot products,
  the normalisation and the mean, then Adam (lr 0.05, 0.9, 0.999, 1e-8, bias-corrected);
- recall@k of the 24 held-out questions against the 24 passages (strictly more similar passages counted), the PCA of the last step
  (numpy eigh, sign: largest entry positive), the keyword ranks, the challenge grid and tiers, the trap cosines, the capability-layer
  arithmetic, the interview drill, the numbers printed in the page's own text and on the home tile, and the facts quoted from the
  pinned sources in sources/b10src/embeddings/raw/;
- every conclusion the page draws from its runs is also checked over many random starts (seeds 20261009 + 1000k): the challenge's
  unique answer in 3 disjoint groups of 16 starts, the batch-size and per-pair statements, the temperature and dimension orders, the
  untrained loss above ln 8, the rising own cosine, the negation pair; the page's own 16-start averages must equal the twin's.
Run: python3 tests/embeddings_numbers.py   (EMB_V=1 also prints every passing check with its numbers)
"""
import os, math, pathlib, re, subprocess, sys
import numpy as np
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/embeddings.html', '--only', 'embeddings'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'embeddings').mkdir(parents=True, exist_ok=True)
(P / 'out' / 'embeddings' / 'wrapped-num.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'embeddings.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'embeddings' / 'wrapped-num.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(f'{name} :: {info}'); print('FAIL', name, '::', str(info)[:300])
    elif os.environ.get('EMB_V'): print('ok', name)


U32 = 0xFFFFFFFF


class Mulberry:
    def __init__(self, seed): self.a = seed & U32
    def __call__(self):
        self.a = (self.a + 0x6D2B79F5) & U32
        t = ((self.a ^ (self.a >> 15)) * (1 | self.a)) & U32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & U32)) & U32) ^ t
        return ((t ^ (t >> 14)) & U32) / 4294967296


SEED, MAXS, LR = 20261009, 200, 0.05
WRE = re.compile(r'[a-z0-9]+(?:[-.][a-z0-9]+)*')
ZRE = re.compile(r'[一-鿿]+|[a-z0-9]+(?:[-.][a-z0-9]+)*')


def toks(s, lang):
    s = s.lower()
    if lang == 'en': return WRE.findall(s)
    out = []
    for w in ZRE.findall(s):
        if '一' <= w[0] <= '鿿': out += [w] if len(w) == 1 else [w[i:i + 2] for i in range(len(w) - 1)]
        else: out.append(w)
    return out


class World:
    def __init__(self, data, lang):
        li = 1 if lang == 'zh' else 0; self.lang = lang; self.vocab = {}
        def ids(s):
            r = []
            for t in toks(s, lang): self.vocab.setdefault(t, len(self.vocab)); r.append(self.vocab[t])
            return r
        self.P = [ids(x[2][li]) for x in data]
        self.T = [ids(q[li]) for x in data for q in x[3]]
        self.H = [[self.vocab[t] for t in toks(x[4][li], lang) if t in self.vocab] for x in data]
        self.V = len(self.vocab)

    def enc(self, E, ids):
        u = E[ids].mean(0); return u / np.linalg.norm(u)

    def known(self, s): return [self.vocab[t] for t in toks(s, self.lang) if t in self.vocab]


def train(w, B, tau, d, seed=SEED, maxs=MAXS, light=False):
    """light: keep only recall@1, the loss, the mean own cosine and the mean Adam step per step (for the many-start checks)"""
    r = Mulberry(seed); E = np.array([[r() * 2 - 1 for _ in range(d)] for _ in range(w.V)])
    m = np.zeros_like(E); v = np.zeros_like(E); rb = Mulberry(seed + 1); N = len(w.P)
    out = {'r1': [], 'r5': [], 'loss': [], 'rank': [], 'Pv': [], 'Hv': [], 'pd': [], 'own': [], 'upd': []}
    for s in range(maxs + 1):
        Pv = np.array([w.enc(E, x) for x in w.P]); Hv = np.array([w.enc(E, x) for x in w.H]); Sm = Hv @ Pv.T
        rk = [int((Sm[i] > Sm[i, i]).sum()) for i in range(N)]
        out['r1'].append(sum(k < 1 for k in rk) / N); out['own'].append(float(np.diag(Sm).mean()))
        if not light:
            out['rank'].append(rk); out['r5'].append(sum(k < 5 for k in rk) / N); out['Pv'].append(Pv); out['Hv'].append(Hv)
        idx = list(range(N))
        for i in range(B):
            j = i + int(rb() * (N - i)); idx[i], idx[j] = idx[j], idx[i]
        ps = idx[:B]; qs = [2 * p + int(rb() * 2) for p in ps]
        uq = np.array([E[w.T[q]].mean(0) for q in qs]); up = np.array([E[w.P[p]].mean(0) for p in ps])
        nq = np.linalg.norm(uq, axis=1, keepdims=True); npn = np.linalg.norm(up, axis=1, keepdims=True); q = uq / nq; p = up / npn
        L = q @ p.T / tau; L = L - L.max(1, keepdims=True); sm = np.exp(L); sm /= sm.sum(1, keepdims=True)
        out['loss'].append(float(-np.log(np.diag(sm)).mean())); out['pd'].append(float(np.diag(sm).mean()))
        if s == maxs: out['S_last'] = q @ p.T; break
        dS = (sm - np.eye(B)) / (tau * B); dq = dS @ p; dp = dS.T @ q
        duq = (dq - q * (q * dq).sum(1, keepdims=True)) / nq; dup = (dp - p * (p * dp).sum(1, keepdims=True)) / npn
        G = np.zeros_like(E)
        for i, qq in enumerate(qs): np.add.at(G, w.T[qq], duq[i] / len(w.T[qq]))
        for i, pp in enumerate(ps): np.add.at(G, w.P[pp], dup[i] / len(w.P[pp]))
        m = 0.9 * m + 0.1 * G; v = 0.999 * v + 0.001 * G * G; st = s + 1
        stp = LR * (m / (1 - 0.9 ** st)) / (np.sqrt(v / (1 - 0.999 ** st)) + 1e-8)
        out['upd'].append(float(np.abs(stp).mean())); E = E - stp
    out['E'] = E; return out


def pca(o):
    X = np.vstack([o['Pv'][MAXS], o['Hv'][MAXS]]); mu = X.mean(0); C = (X - mu).T @ (X - mu) / len(X)
    val, vec = np.linalg.eigh(C); order = np.argsort(val)[::-1][:2]; ax = []
    for i in order:
        e = vec[:, i].copy(); k = int(np.argmax(np.abs(e)))
        if e[k] < 0: e = -e
        ax.append(e)
    ax = np.array(ax).T
    return mu, ax, (val[order].sum() / val.sum())


def sig2(v):
    if v >= 100: return f'{v:,.0f}'
    if v >= 0.1: return f'{v:.2f}'
    e = math.floor(math.log10(v))
    if e >= -3: return f'{v:.{1 - e}f}'
    m = round(v / 10 ** e, 1)
    if m >= 10: m, e = 1.0, e + 1
    return f'{m:.1f}×10{e}'   # as the typeset line reads in textContent


GROUPS = 3; NSEED = 16
SEEDS = [SEED + 1000 * k for k in range(GROUPS * NSEED)]


with sync_playwright() as pw:
    b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.evaluate("()=>{location.hash='embeddings'}"); pg.wait_for_timeout(600)
    data = pg.evaluate("()=>window.__emb.data()")
    C = pg.evaluate("()=>window.__emb.consts")
    check('24 passages, 2 training questions each, 6 topics of 4', len(data) == 24 and all(len(x[3]) == 2 for x in data) and sorted(set(x[0] for x in data)) == sorted('tlvaew') and all(sum(1 for x in data if x[0] == t) == 4 for t in 'tlvaew'))
    check('constants: seed, 200 steps, lr 0.05, options, 16 starts', C['SEED'] == SEED and C['MAXS'] == MAXS and C['LR'] == LR and C['BS'] == [2, 4, 8, 16] and C['TAUS'] == [1, 0.2, 0.05, 0.01] and C['DS'] == [4, 8, 16, 32] and C['CH_STEPS'] == [10, 25, 50, 100] and C['CH_GOAL'] == 0.55 and C['NSEED'] == NSEED and C['SEEDS'] == SEEDS[:NSEED], C)
    W = {l: World(data, l) for l in ('en', 'zh')}
    for l in ('en', 'zh'):
        check(f'{l}: vocabulary', pg.evaluate(f"()=>window.__emb.vocab('{l}')") == list(W[l].vocab.keys()))
        check(f'{l}: every held-out question keeps at least 2 known tokens', all(len(h) >= 2 for h in W[l].H))
    # ---------- training runs of the page (seed 20261009) ----------
    CFG = [('en', 8, 0.05, 16), ('zh', 8, 0.05, 16), ('en', 2, 0.05, 16), ('en', 4, 0.05, 16), ('en', 16, 0.05, 16), ('zh', 2, 0.05, 16), ('zh', 4, 0.05, 16), ('zh', 16, 0.05, 16),
           ('en', 8, 1, 16), ('en', 8, 0.2, 16), ('en', 8, 0.01, 16), ('zh', 8, 1, 16), ('zh', 8, 0.01, 16), ('en', 8, 0.05, 4), ('zh', 8, 0.05, 4), ('en', 8, 0.05, 32), ('zh', 8, 0.05, 32)]
    PY = {}
    for (l, B, tau, d) in CFG:
        o = train(W[l], B, tau, d); PY[(l, B, tau, d)] = o
        js = pg.evaluate(f"()=>window.__emb.run('{l}',{B},{tau},{d})")
        tag = f'{l} B={B} tau={tau} d={d}'
        check(f'{tag}: recall@1 at every step', js['r1'] == o['r1'], [(s, js['r1'][s], o['r1'][s]) for s in range(MAXS + 1) if js['r1'][s] != o['r1'][s]][:3])
        check(f'{tag}: recall@5 at every step', js['r5'] == o['r5'])
        check(f'{tag}: rank of every held-out question at every step', js['rank'] == o['rank'])
        mx = max(abs(a - c) / max(1, abs(c)) for a, c in zip(js['loss'], o['loss']))
        check(f'{tag}: loss of every next batch (rel. 1e-6)', mx < 1e-6, mx)
        dv = max(np.abs(np.array(js['P'][1]) - o['Pv'][MAXS].ravel()).max(), np.abs(np.array(js['H'][1]) - o['Hv'][MAXS].ravel()).max())
        check(f'{tag}: vectors after 200 steps (1e-6)', dv < 1e-6, dv)
        if (l, B, tau, d) in (('en', 8, 0.05, 16), ('zh', 8, 0.05, 16), ('en', 16, 0.05, 16)):
            mu, ax, share = pca(o); XY = np.vstack([o['Pv'][MAXS], o['Hv'][MAXS]]) - mu
            jx = np.array(js['xy'][2]); px = XY @ ax
            check(f'{tag}: PCA map at step 200 (1e-6)', np.abs(jx - px).max() < 1e-6, np.abs(jx - px).max())
            X0 = np.vstack([o['Pv'][0], o['Hv'][0]]) - mu
            check(f'{tag}: PCA map at step 0 uses the same axes', np.abs(np.array(js['xy'][0]) - X0 @ ax).max() < 1e-6)
    en, zh = PY[('en', 8, 0.05, 16)], PY[('zh', 8, 0.05, 16)]
    # ---------- many starts: every conclusion the page draws must hold on average over 16 starts, in 3 disjoint groups of 16 ----------
    RS = {}
    def run_s(l, B, tau, d, sd, maxs):
        k = (l, B, tau, d, sd); o = RS.get(k)
        if o is None or len(o['r1']) < maxs + 1: o = train(W[l], B, tau, d, seed=sd, maxs=maxs, light=True); RS[k] = o
        return o
    def cnt(o, s): return round(o['r1'][s] * 24)
    def avg(l, B, s, grp, tau=0.05, d=16, maxs=100):
        return sum(cnt(run_s(l, B, tau, d, sd, maxs), s) for sd in SEEDS[NSEED * grp:NSEED * grp + NSEED]) / (NSEED * 24)
    LEN = {2: 200, 4: 200, 8: 100, 16: 100}
    for l in ('en', 'zh'):
        for g in range(GROUPS):
            A = {(B, s): avg(l, B, s, g, maxs=LEN[B]) for B in (2, 4, 8, 16) for s in (10, 25, 40, 50, 100, 200) if s <= LEN[B] and not (B == 8 and s == 200)}
            ok = [(B, s) for B in (2, 4, 8, 16) for s in (10, 25, 50, 100) if A[(B, s)] >= 0.55 - 1e-9]; mn = min(s for B, s in ok)
            check(f'{l} group {g}: challenge: the only choice reaching an average of 55% in the fewest steps is B 16 for 10 steps', [x for x in ok if x[1] == mn] == [(16, 10)], [(B, s, round(A[(B, s)], 3)) for B, s in ok])
            check(f'{l} group {g}: challenge: the initial pick B 2 for 25 steps misses', A[(2, 25)] < 0.55)
            a25 = [A[(B, 25)] for B in (2, 4, 8, 16)]
            check(f'{l} group {g}: part 5: after 25 steps the average rises with B {[round(x * 100) for x in a25]}', all(x < y for x, y in zip(a25, a25[1:])))
            p800 = [A[(4, 200)], A[(8, 100)], A[(16, 50)]]
            check(f'{l} group {g}: part 5: at 800 pairs B 4, 8, 16 are within 8 points {[round(x * 100) for x in p800]}', max(p800) - min(p800) <= 0.08)
            p160 = [A[(4, 40)], avg(l, 8, 20, g), A[(16, 10)]]
            check(f'{l} group {g}: challenge why: at 160 pairs B 4, 8, 16 are within 6 points {[round(x * 100) for x in p160]}', max(p160) - min(p160) <= 0.06)
            if g == 0:
                js = {B: pg.evaluate(f"()=>window.__emb.avg('{l}',{B},0.05,16,{LEN[B]})") for B in (2, 4, 8, 16)}
                bad = [(B, s) for (B, s), v in A.items() if abs(js[B][s] - v) > 1e-12]
                check(f'{l}: the page averages of the 16 starts equal the Python twin (B 2..16, up to 200 / 100 steps)', not bad, bad[:5])
                for B in (2, 4, 8, 16):
                    for s in (10, 25, 50, 100):
                        jp = pg.evaluate(f"()=>window.__emb.chPoint('{l}',{B},{s})")
                        if abs(jp['r1'] - A[(B, s)]) > 1e-12 or jp['ok'] != (A[(B, s)] >= 0.55 - 1e-9): check(f'{l}: challenge point B {B} {s} steps', False, (jp, A[(B, s)]))
                PY[('avg', l)] = A
                check(f'{l}: scenario b2: over 16 starts B 2 ends below B 8 after 200 steps', A[(2, 200)] < sum(cnt(run_s(l, 8, 0.05, 16, sd, 200), 200) for sd in SEEDS[:NSEED]) / (NSEED * 24))
        # temperature over 24 starts (B 8, d 16, 200 steps)
        S24 = SEEDS[:24]; T = {tau: [run_s(l, 8, tau, 16, sd, 200) for sd in S24] for tau in (1, 0.05, 0.01)}
        m = {tau: np.mean([cnt(o, 200) for o in T[tau]]) for tau in T}
        check(f'{l}: part 6: over 24 starts the mean recall@1 after 200 steps falls with tau: 1 {m[1]:.1f} > 0.05 {m[0.05]:.1f} > 0.01 {m[0.01]:.1f}', m[1] > m[0.05] > m[0.01])
        check(f'{l}: part 6: tau 1 beats tau 0.05 in at least 20 of 24 starts', sum(cnt(a, 200) > cnt(c, 200) for a, c in zip(T[1], T[0.05])) >= 20)
        for tau in (0.05, 0.01):
            zs = [next((s for s in range(MAXS + 1) if o['loss'][s] < 0.01), 999) for o in T[tau]]
            check(f'{l}: part 6: tau {tau}: the loss falls under 0.01 within 50 steps in all 24 starts (latest {max(zs)})', max(zs) <= 50)
            check(f'{l}: part 6: tau {tau}: after that recall@1 still rises on average and the Adam step shrinks', np.mean([cnt(o, 200) - cnt(o, z) for o, z in zip(T[tau], zs)]) > 0 and all(o['upd'][150] < o['upd'][z] for o, z in zip(T[tau], zs)))
        check(f'{l}: tau 1: the loss never falls below the bound -ln(e^2/(e^2+7)) = 0.67', all(min(o['loss']) >= -math.log(math.exp(2) / (math.exp(2) + 7)) - 1e-9 for o in T[1]))
        js = {tau: pg.evaluate(f"()=>window.__emb.avg('{l}',8,{tau},16,200)[200]") for tau in (1, 0.05, 0.01)}
        check(f'{l}: the page averages for part 6 equal the twin', all(abs(js[tau] - sum(cnt(o, 200) for o in T[tau][:NSEED]) / (NSEED * 24)) < 1e-12 for tau in js), js)
        PY[('tavg', l)] = js
        # dimensions over 16 starts
        md = {d: np.mean([cnt(run_s(l, 8, 0.05, d, sd, 200), 200) for sd in SEEDS[:NSEED]]) for d in (4, 16, 32)}
        check(f'{l}: part 8: over 16 starts more dimensions find more: d 4 {md[4]:.1f} < 16 {md[16]:.1f} < 32 {md[32]:.1f}', md[4] < md[16] < md[32])
        # the untrained loss, the own cosine, the reworded questions
        l0 = [o['loss'][0] for o in T[0.05]]
        check(f'{l}: part 4: the untrained loss is above ln 8 in all 24 starts (lowest {min(l0):.2f})', min(l0) > math.log(8))
        rise = [o['own'][200] - o['own'][0] for o in T[0.05]]
        check(f'{l}: part 4: the mean own cosine rises by more than 0.3 in all 24 starts (least {min(rise):.2f})', min(rise) > 0.3)
        li = 1 if l == 'zh' else 0; w = W[l]
        sh = []
        for x in data:
            hk = {t for t in toks(x[4][li], l) if t in w.vocab}; tq = {t for q in x[3] for t in toks(q[li], l)}; sh.append(len(hk & tq) / len(hk))
        check(f'{l}: part 2: held-out questions are rewordings: on average {np.mean(sh):.0%} of their known tokens are in their passage\'s training questions, every one shares some', np.mean(sh) >= 0.75 and min(sh) > 0)
        # the negation pair stays close over 16 starts
        tr = [x for x in C['TRAPS'] if x['k'] == 'neg'][0]; a, c = tr['a'][li], tr['b'][li]
        cs = []
        for sd in SEEDS[:NSEED]:
            E = run_s(l, 8, 0.05, 16, sd, 200)['E']; cs.append(float(w.enc(E, w.known(a)) @ w.enc(E, w.known(c))))
        check(f'{l}: part 7: the negation pair stays above 0.6 in all 16 starts (lowest {min(cs):.2f})', min(cs) > 0.6)
    # the story the page tells about its own run, in both languages
    for l, o in (('en', en), ('zh', zh)):
        check(f'{l}: untrained recall@1 is near chance (at most 2 of 24)', o['r1'][0] <= 2 / 24, o['r1'][0])
        check(f'{l}: untrained recall@5 above the 21% of a random pick', o['r5'][0] > 5 / 24, o['r5'][0])
        check(f'{l}: initial setup reaches at least 75% after 200 steps', o['r1'][MAXS] >= 0.75, o['r1'][MAXS])
        check(f'{l}: the mission question finds its passage first after 200 steps', o['rank'][MAXS][0] == 0)
        b2, b8, b16 = PY[(l, 2, 0.05, 16)], o, PY[(l, 16, 0.05, 16)]
        check(f'{l}: scenarios: B 16 above B 8 after 25 steps, B 2 below B 8 after 200 (this run)', b16['r1'][25] > b8['r1'][25] and b2['r1'][MAXS] < b8['r1'][MAXS])
        for tau in (0.05, 0.01):
            t = PY[(l, 8, tau, 16)]; z = next(s for s in range(MAXS + 1) if t['loss'][s] < 0.01)
            check(f'{l}: tau {tau} (this run): the loss falls under 0.01 at step {z}; later it jumps above 0.1 again (spikes), recall@1 still rises', any(v > 0.1 for v in t['loss'][z + 1:]) and t['r1'][MAXS] > t['r1'][z])
        t1 = PY[(l, 8, 1, 16)]
        check(f'{l}: tau 1 (this run) ends above the initial setup', t1['loss'][MAXS] > 1 and t1['r1'][MAXS] > o['r1'][MAXS])
        check(f'{l}: 4 dimensions end below 16, 32 above (this run)', PY[(l, 8, 0.05, 4)]['r1'][MAXS] < o['r1'][MAXS] < PY[(l, 8, 0.05, 32)]['r1'][MAXS])
        # traps
        w = W[l]; E = o['E']; li = 1 if l == 'zh' else 0
        for tr in C['TRAPS']:
            a, c = tr['a'][li], tr['b'][li]
            ca = w.enc(E, w.known(a)) @ w.enc(E, w.known(c)); jc = pg.evaluate("([l,a,b])=>window.__emb.cosOf(l,a,b)", [l, a, c])
            check(f'{l}: trap {tr["k"]}: cosine {ca:.4f}', abs(ca - jc) < 1e-9 and len(w.known(a)) > 0 and len(w.known(c)) > 0, (ca, jc))
            if tr['k'] == 'order': check(f'{l}: word-order pair has the same known tokens, cosine 1.00', sorted(w.known(a)) == sorted(w.known(c)) and abs(ca - 1) < 1e-12)
            else:
                ta, tb = toks(a, l), toks(c, l); neg = [t for t in tb if t not in ta and t in w.vocab]; skip = sorted({t for t in ta + tb if t not in w.vocab})
                PY[('neg', l)] = (ca, neg, len([t for t in tb if t in w.vocab]), skip)
                check(f'{l}: negation pair: the added tokens {neg} are all known, skipped {skip}', neg and all(t in w.vocab for t in tb if t not in ta))
        # keyword search for the mission question
        sc = [len(set(toks(data[0][4][li], l)) & set(toks(x[2][li], l))) for x in data]
        ahead = sum(1 for j in range(24) if j != 0 and sc[j] > sc[0])
        check(f'{l}: the mission question shares no token with its passage', sc[0] == 0)
        kw = pg.evaluate(f"()=>window.__emb.kwRank('{l}',0)")
        check(f'{l}: keyword scores', kw['sc'] == sc)
        PY[('ahead', l)] = ahead
    # ---------- numbers printed on the page ----------
    def text(sel): return pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' '):''}", sel)
    def go(i):
        pg.evaluate(f"()=>{{const b=document.querySelector('#emb-rail [data-go=\"{i}\"]'); if(b) b.click();}}"); pg.wait_for_timeout(300)
    def showme():
        pg.evaluate("()=>document.querySelector('#emb-rail .step.current [data-nav=\"showme\"]').click()")
        pg.wait_for_function("()=>!window.__emb.animState()", timeout=30000); pg.wait_for_timeout(300)
    def obs_ready():
        pg.wait_for_function("()=>{const o=document.querySelector('#emb-rail [data-obs]'); return o&&!/Averaging 16|正在算 16/.test(o.innerText)}", timeout=60000)
    pct = lambda v: f'{round(v * 100)}%'
    cap = lambda B: math.exp(2) / (math.exp(2) + B - 1)
    def same_topic(B):
        none = 1.0
        for k in range(B - 1): none *= (20 - k) / (23 - k)
        return 1 - none
    check('facts: tau 1 caps the own passage at 51% (B 8, loss 0.67) and about a third (B 16)', pct(cap(8)) == '51%' and f'{-math.log(cap(8)):.2f}' == '0.67' and abs(cap(16) - 1 / 3) < 0.01)
    check('facts: same topic in the batch: 13% with B 2 (3/23), 97% with B 16', pct(same_topic(2)) == '13%' and abs(same_topic(2) - 3 / 23) < 1e-12 and pct(same_topic(16)) == '97%' and abs(1 - same_topic(16) - math.comb(20, 15) / math.comb(23, 15)) < 1e-12)
    for l in ('en', 'zh'):
        pg.click(f'.langsw button[data-lang="{l}"]'); pg.wait_for_timeout(400)
        pg.evaluate("()=>{const b=document.querySelector('#emb-app .modes button[data-mode=\"guided\"]'); b.click()}"); pg.wait_for_timeout(300)
        o = en if l == 'en' else zh; ahead = PY[('ahead', l)]; A = PY[('avg', l)]; li = 1 if l == 'zh' else 0; w = W[l]
        go(0); t = text('#emb-rail .step.current')
        check(f'{l}: part 1 names the {ahead} passages a keyword search returns first', (f'puts {ahead} other passages first' if l == 'en' else f'先给出另外 {ahead} 段') in t, t[:400])
        check(f'{l}: stage names the same count', (f'returns {ahead} other passages before it' if l == 'en' else f'先给出另外 {ahead} 段') in text('#emb-pair'))
        if l == 'zh': check('zh: part 1 says character pairs and 1M', '字对' in t and '共同的字对' in text('#emb-pair') and '1M 个以上' in t, t[:500])
        go(1); showme()
        t = text('#emb-rail [data-obs]')
        k0 = sum(1 for k in o['rank'][0] if k == 0)
        check(f'{l}: part 2 untrained recall@1 {pct(o["r1"][0])} and recall@5 {pct(o["r5"][0])}', pct(o['r1'][0]) in t and pct(o['r5'][0]) in t and (f'for {k0} of the 24' if l == 'en' else (f'只有 {k0} 个' if k0 else '没有一个的正确段落排第一')) in t, t[:300])
        rk = o['rank'][0][9]
        check(f'{l}: part 2 rank of the shown question ({rk + 1})', (f'ranks {rk + 1}' if l == 'en' else f'排第 {rk + 1}') in t)
        check(f'{l}: part 2 says the held-out questions reword the training questions', ('asks the same thing as the training questions of its passage, in other words' if l == 'en' else '和这一段说明的训练问题一样，只是换了一种说法') in text('#emb-rail .step.current'))
        check(f'{l}: parts 2 and 3 show only the two recall cards', pg.evaluate("()=>document.querySelectorAll('#emb-metrics .metric').length") == 2)
        go(2); ht = pg.evaluate("()=>document.querySelector('#emb-heat').textContent")
        check(f'{l}: part 3 shows no probability column', 'own p' not in ht and '正确的 p' not in ht and not pg.query_selector('#emb-heat .emb-pb'), ht[:200])
        go(3); t = text('#emb-rail .step.current')
        check(f'{l}: part 4 names contrastive learning and InfoNCE, the untrained loss {o["loss"][0]:.2f} above ln 8', ('contrastive learning' in t and 'InfoNCE' in t and 'natural logarithm' in t) if l == 'en' else ('对比学习（contrastive learning）' in t and 'InfoNCE' in t and '自然对数' in t) and f'{o["loss"][0]:.2f}' in t and '2.08' in t, t[:1500])
        check(f'{l}: part 4 card: {o["loss"][0]:.2f}', f'{o["loss"][0]:.2f}' in t)
        showme(); t = text('#emb-rail [data-obs]')
        ow0, ow1 = o['own'][0], o['own'][MAXS]
        fx2 = lambda v: f'{v:.2f}'.replace('-', '−')
        check(f'{l}: part 4 recall@1 {pct(o["r1"][0])} to {pct(o["r1"][MAXS])}, own cosine {ow0:.2f} to {ow1:.2f}', pct(o['r1'][0]) in t and pct(o['r1'][MAXS]) in t and fx2(ow0) in t and fx2(ow1) in t, t[:400])
        hk = list(dict.fromkeys(t_ for t_ in toks(data[0][4][li], l) if t_ in w.vocab)); best = max(data[0][3], key=lambda q: len([x for x in hk if x in set(toks(q[li], l))]))
        nb = len([x for x in hk if x in set(toks(best[li], l))])
        check(f'{l}: part 4 names the training question {best[li]!r}: {nb} of {len(hk)} known tokens', best[li] in t and ((f'{len(hk)} appear in the training data' in t and (f'holds {nb} of them' if nb < len(hk) else 'holds all of them')) if l == 'en' else f'有 {len(hk)} 个在训练数据里出现过、在 embedding 表里有自己的一行，这 {len(hk)} 个' + ('全都' if nb == len(hk) else f'里有 {nb} 个')) in t, t)
        mt = text('#emb-metrics')
        check(f'{l}: metric cards at step 200', pct(o['r1'][MAXS]) in mt and pct(o['r5'][MAXS]) in mt and f"{o['loss'][MAXS]:.2f}" in mt and pct(o['pd'][MAXS]) in mt, mt)
        check(f'{l}: baseline ln 8 = 2.08 on the card, named as equal probabilities', '2.08' in mt and ('all passages equal' in mt if l == 'en' else '每段概率一样时' in mt))
        # the worked loss of row 1 of the next batch at step 200
        pg.evaluate("()=>{document.querySelector('#emb-math').open=true}"); pg.wait_for_timeout(100)
        row = o['S_last'][0]; l1 = math.log1p(sum(math.exp((row[j] - row[0]) / 0.05) for j in range(1, 8)))
        mtx = pg.evaluate("()=>[...document.querySelectorAll('#emb-mathbody .emb-math-line')].map(x=>x.textContent).join(' || ')")
        check(f'{l}: math block 3: the loss of row 1 is {sig2(l1)}', f'={sig2(l1)}' in mtx.replace(' ', '').replace('−', '-'), mtx[-300:])
        pg.evaluate("()=>{document.querySelector('#emb-math').open=false}")
        go(4); showme(); obs_ready()
        t = text('#emb-rail [data-obs]')
        p8 = [A[(4, 200)], A[(8, 100)], A[(16, 50)]]
        check(f'{l}: part 5 averages after 25 steps: B 8 {pct(A[(8, 25)])}, B 16 {pct(A[(16, 25)])}; 800 pairs between {pct(min(p8))} and {pct(max(p8))}', all(x in t for x in (f'B = 8{": " if l == "en" else "："}{pct(A[(8, 25)])}', f'B = 16{": " if l == "en" else "："}{pct(A[(16, 25)])}', (f'between {pct(min(p8))} and {pct(max(p8))}' if l == 'en' else f'都在 {pct(min(p8))} 到 {pct(max(p8))} 之间'))) and '16' in t, t[:900])
        check(f'{l}: part 5 no longer claims a bigger batch wins at equal pairs', ('little difference' in t) if l == 'en' else ('差别不大' in t), t)
        go(5); showme(); obs_ready()
        t = text('#emb-rail [data-obs]')
        for tau in (0.05, 0.01):
            tt = PY[(l, 8, tau, 16)]; z = next(s for s in range(MAXS + 1) if tt['loss'][s] < 0.01)
            check(f'{l}: part 6 tau {tau}: under 0.01 at step {z} with recall@1 {pct(tt["r1"][z])}, {pct(tt["r1"][MAXS])} after 200', ((f'τ = {tau}: the loss first falls under 0.01 at step {z}, when recall@1 is {pct(tt["r1"][z])}; recall@1 after 200 steps {pct(tt["r1"][MAXS])}') if l == 'en' else (f'τ = {tau}：第 {z} 步 loss 第一次降到 0.01 以下，这时 recall@1 是 {pct(tt["r1"][z])}；200 步后 recall@1 是 {pct(tt["r1"][MAXS])}')) in t, t[:500])
        ta = PY[('tavg', l)]
        check(f'{l}: part 6 averages {pct(ta[0.05])} / {pct(ta[0.01])} and 1B pairs, no "learning stops"', pct(ta[0.05]) in t and pct(ta[0.01]) in t and ('1B pairs' in t if l == 'en' else '1B 对' in t) and 'stops' not in t and '学习就停了' not in t, t)
        go(6); t = text('#emb-rail .step.current')
        check(f'{l}: part 7 has no storage paragraph, names the E5 v2 English models and BGE v1.5', '3,072' not in t and ('three English E5 v2 models' in t and 'also work without it' in t if l == 'en' else 'E5 v2 的三个英文模型' in t and '不加也能用' in t), t[-700:])
        pg.evaluate("()=>document.querySelector('#emb-rail [data-lever=\"neg\"]').click()"); pg.wait_for_timeout(300)
        t = text('#emb-rail [data-obs]'); ca, neg, nkb, skip = PY[('neg', l)]
        exp_ = (f'Similarity {ca:.2f}' in t and f'the average of {nkb} table rows' in t and all(f'“{x}”' in t for x in neg + skip)) if l == 'en' else (f'相似度 {ca:.2f}' in t and f'第二句的向量是 embedding 表里 {nkb} 行的平均' in t and f'其中 {len(neg)} 行' in t and all(f'“{x}”' in t for x in neg + skip))
        check(f'{l}: part 7 negation: {ca:.2f}, {neg} among {nkb} rows, skipped {skip}', exp_, t)
        go(7); t = text('#emb-rail .step.current')
        k0 = round(o['r1'][0] * 24); k16 = round(o['r1'][MAXS] * 24); k32 = round(PY[(l, 8, 0.05, 32)]['r1'][MAXS] * 24)
        check(f'{l}: part 8: 13% / 97%, d 32 finds {k32 - k16} more of the 24 ({k32} vs {k16}), BEIR named', all(x in t for x in ('13%', '97%', 'BEIR')) and k32 > k16 and (f'finds {k32 - k16} more of the 24' in t if l == 'en' else f'24 个新问题里多找对 {k32 - k16} 个' in t), t[-900:])
        bt = pg.evaluate("()=>[...document.querySelectorAll('#emb-rail .step.current [data-tipx]')].map(x=>x.getAttribute('data-tipx')).join(' | ')")
        check(f'{l}: part 8 BEIR tooltip: 50.6 / 49.0, 1,024 / 384 dimensions, 24 / 12 layers', all(x in bt for x in ('50.6', '49.0', '1,024', '384', '24', '12')), bt[-500:])
        dn = pg.evaluate("()=>document.querySelector('#emb-ctl-d').closest('.lockable').textContent")
        check(f'{l}: settings d note: 768 x 4 = 3,072 bytes, 1M passages = 3.07 GB, E5 v2 384 / 768 / 1,024', all(x in dn for x in ('3,072', '3.07 GB', '1M', '384', '768', '1,024')) and 768 * 4 == 3072 and abs(1e6 * 3072 / 1e9 - 3.072) < 1e-12, dn)
        check(f'{l}: part 8 misconception 1 counts {k0} untrained', (('not one' if k0 == 0 else f'only {k0}') + ' of the 24 held-out questions') in t if l == 'en' else ('24 个新问题里' + ('没有一个' if k0 == 0 else f'只有 {k0} 个')) in t, t)
        # sandbox scenarios: the initial setup and tau 0.01 name the step and the recall at which the loss falls under 0.01
        pg.evaluate("()=>document.querySelector('#emb-app .modes button[data-mode=\"sandbox\"]:not([data-flow=free])').click()"); pg.wait_for_timeout(300)
        for i, tau in ((0, 0.05), (4, 0.01)):
            pg.evaluate(f"()=>document.querySelector('#emb-rail [data-preset=\"{i}\"]').click()"); pg.wait_for_function("()=>!window.__emb.animState()", timeout=30000); pg.wait_for_timeout(300)
            tt = PY[(l, 8, tau, 16)]; z = next(s for s in range(MAXS + 1) if tt['loss'][s] < 0.01); t = text('#emb-insight')
            check(f'{l}: scenario tau {tau}: step {z}, {pct(tt["r1"][z])} then, {pct(tt["r1"][MAXS])} after 200', (f'step {z}, when recall@1 is {pct(tt["r1"][z])}' in t if l == 'en' else f'第 {z} 步' in t and f'这时 recall@1 是 {pct(tt["r1"][z])}' in t) and pct(tt['r1'][MAXS]) in t and 'stop' not in t and '不再变好' not in t, t)
        pg.evaluate("()=>document.querySelector('#emb-rail [data-preset=\"3\"]').click()"); pg.wait_for_function("()=>!window.__emb.animState()", timeout=30000); pg.wait_for_timeout(300)
        t = text('#emb-insight')
        tb = pg.evaluate("()=>[...document.querySelectorAll('#emb-insight [data-tipx]')].map(x=>x.getAttribute('data-tipx')).join(' | ')")
        r1t1, r1b = PY[(l, 8, 1, 16)]['r1'][MAXS], PY[(l, 8, 0.05, 16)]['r1'][MAXS]
        check(f'{l}: scenario tau 1: about half (tooltip 51%, bound 0.67), loss {PY[(l, 8, 1, 16)]["loss"][MAXS]:.2f}, ends {pct(r1t1)} against {pct(r1b)}', ('about half' in t if l == 'en' else '约一半' in t) and '51%' in tb and '0.67' in tb and f'{PY[(l, 8, 1, 16)]["loss"][MAXS]:.2f}' in t and pct(r1t1) in t and pct(r1b) in t and r1t1 > r1b, t + ' || ' + tb)
        # the home tile reads the lab's own run: k of 24 held-out questions ranked first at each tile step
        pg.evaluate("()=>{location.hash='map~apps'}"); pg.wait_for_timeout(600)
        if not pg.query_selector('[data-teaser="embeddings"] .tz-read'):
            check(f'{l}: the home tile renders', False)
        else:
            k0 = sum(1 for k in o['rank'][0] if k == 0)
            for i, s in enumerate((10, 25, 50, 100, 200), 1):
                pg.evaluate(f"()=>document.querySelector('[data-teaser=\"embeddings\"] button[data-v=\"{i}\"]').click()"); pg.wait_for_timeout(80)
                t = text('[data-teaser="embeddings"] .tz-read'); k = sum(1 for r in o['rank'][s] if r == 0)
                check(f'{l}: tile at {s} steps says {k} of 24 (untrained {k0})', (f'After {s} training steps, {k} of 24' in t and f'(untrained: {k0})' in t) if l == 'en' else (f'训练 {s} 步后，24 个没见过的问题里有 {k} 个' in t and f'（训练前 {k0} 个）' in t), t)
            pg.evaluate("()=>document.querySelector('[data-teaser=\"embeddings\"] button[data-v=\"0\"]').click()"); pg.wait_for_timeout(80)
            t = text('[data-teaser="embeddings"] .tz-read')
            check(f'{l}: tile at 0 steps says {k0} of 24', (f'Before training, {k0} of 24' if l == 'en' else f'训练前，24 个没见过的问题里有 {k0} 个') in t, t)
        pg.evaluate("()=>{location.hash='embeddings'}"); pg.wait_for_timeout(500)
    # challenge: submit the best answer in EN and read the verdict
    pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(300)
    pg.evaluate("()=>document.querySelector('#emb-app .modes button[data-mode=\"challenge\"]').click()"); pg.wait_for_timeout(300)
    pg.evaluate("()=>document.querySelector('#emb-rail .seg-ctl[data-key=\"B\"] button[data-v=\"16\"]').click()")
    pg.evaluate("()=>document.querySelector('#emb-rail .seg-ctl[data-key=\"chN\"] button[data-v=\"10\"]').click()")
    pg.evaluate("()=>document.querySelector('#emb-rail [data-act=\"submit\"]').click()")
    pg.wait_for_function("()=>!window.__emb.animState()", timeout=20000); pg.wait_for_timeout(300)
    t = text('#emb-ch-res'); A = PY[('avg', 'en')]
    check(f'challenge: B 16 x 10 is met ({pct(A[(16, 10)])}) and minimal', 'Goal met' in t and pct(A[(16, 10)]) in t and 'no choice reaches 55% in fewer steps' in t, t[:300])
    p160 = [A[(4, 40)], sum(cnt(run_s('en', 8, 0.05, 16, sd, 100), 20) for sd in SEEDS[:NSEED]) / (NSEED * 24), A[(16, 10)]]
    check(f'challenge: the why: only B = 16 after 10 steps; at 160 pairs between {pct(min(p160))} and {pct(max(p160))}', 'only B = 16 reaches 55%' in t and f'between {pct(min(p160))} and {pct(max(p160))}' in t and all(A[(B, 10)] < 0.55 for B in (2, 4, 8)), t)
    check('challenge: the note no longer cites "real teams"', 'Real teams' not in t and 'real teams' not in t)
    # capability layer and drill arithmetic
    check('exits: 1M x 384 x 4 bytes = 1.54 GB, x 1,024 = 4.10 GB', f'{1e6 * 384 * 4 / 1e9:.2f}' == '1.54' and f'{1e6 * 1024 * 4 / 1e9:.2f}' == '4.10')
    check('exits: ln 8 = 2.08, ln 64 = 4.16', f'{math.log(8):.2f}' == '2.08' and f'{math.log(64):.2f}' == '4.16')
    for sd in range(1, 6):
        dj = pg.evaluate(f"()=>window.__emb.drill({sd})"); s = dj['s']; tau = dj['tau']
        z = [math.exp(v / tau) for v in s]; p = z[0] / sum(z)
        check(f'drill {sd}: own passage first and probability / loss', s[0] > max(s[1:]) and abs(p - dj['p']) < 1e-12 and abs(-math.log(p) - dj['loss']) < 1e-12, (s, p, dj['p']))
    # facts quoted from the pinned sources
    R = P / 'sources' / 'b10src' / 'embeddings' / 'raw'
    mn = (R / 'st_mnrl.py').read_text(); e5 = (R / 'e5_readme.md').read_text(); ev = (R / 'e5_mteb_beir_eval.py').read_text(); fe = (R / 'fe_readme.md').read_text(); se = (R / 'st_semantic_search.md').read_text(); pre = (R / 'st_pretrained.md').read_text()
    check('source: MNRL scale 20.0 = temperature 0.05, cos_sim, 10 to 100 common, NO_DUPLICATES', 'scale: float = 20.0' in mn and '``scale=20.0`` is equivalent to ``temperature=0.05``' in mn and 'values between 10 and 100 are common' in mn and 'similarity_fct: Callable[[Tensor, Tensor], Tensor] = cos_sim' in mn and 'BatchSamplers.NO_DUPLICATES' in mn)
    check('source: MNRL is also known as InfoNCE; much higher batch sizes, better performance', 'also known as the InfoNCE loss' in mn and 'much higher batch sizes (and thus better performance)' in mn)
    check('source: all-* models trained on more than 1 billion pairs', 'trained on all available training data (more than 1 billion training pairs)' in pre)
    check('source: E5 small/base/large v2 dims 384/768/1024, BEIR 49.0/50.3/50.6, 12/12/24 layers', re.search(r'E5-small-v2\s*\|\s*49\.0\s*\|\s*12\s*\|\s*384', e5) and re.search(r'E5-base-v2\s*\|\s*50\.3\s*\|\s*12\s*\|\s*768', e5) and re.search(r'E5-large-v2\s*\|\s*50\.6\s*\|\s*24\s*\|\s*1024', e5))
    check('source: E5 query: / passage: prefixes and normalize; not for instruct models; symmetric tasks prefix both sides', "f'query: {q}'" in ev and "'passage: {}'.format(t)" in ev and 'F.normalize(embeds, p=2, dim=-1)' in ev and 'no need to add prefix for instruct models' in ev and 'use query prefix for passages, only used for Quora as it is a symmetric task' in ev)
    check('source: BGE v1.5 query instruction, works without it', 'Represent this sentence for searching relevant passages: ' in fe and 'enhance its retrieval ability without instruction' in fe)
    check('source: semantic search normalise then dot product', 'normalize the corpus embeddings so that each corpus embeddings is of length 1' in se and 'dot_score' in se)
    check('no page errors', not errs, errs[:3])
    b.close()

print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
