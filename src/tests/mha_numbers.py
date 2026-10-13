"""Golden numbers for the MHA lab (labs/mha.js), recomputed independently in numpy.

1. The hand-set matrices are rebuilt here from the patterns written in specs/mha-copy.md (a rotary code for the previous word,
   pronoun -> animal, first token, nouns) and must equal what the page exposes (window.__mha.mats(), read-only).
2. Multi-head attention is re-implemented here from scratch (q = Wq x, k = Wk x, v = Wv x, split into heads,
   q.k / sqrt(head_dim), causal mask, softmax per head, weighted sum, concat, Wo) and compared with the page's compute() for
   both sentences, all 9 query words, 1 / 2 / 4 heads and all 16 on/off sets of the 4 heads.
3. The claims the copy makes are asserted on the numpy numbers: the calibration targets, the language symmetry at "it",
   the challenge's unique minimal answer, the margins to the one threshold the page tests against (80%); the spec's
   calibration (H1 and H3 at 85% or more with 4 heads, each word in its own head with 2) is checked too.
4. Show the math: every chain line recomputes from the rounded values it shows, and its weights round to the stage's.
5. Every number the page prints is read from the rendered page, in English and Chinese, and compared with numpy or with
   Llama-3.1-8B's config values: the home tile, the chart and its tooltips, the metrics, the badge, the scenario card, the
   mini result bar, Show the math, the hand-set parameters note, guided parts 3-8, the challenge (goal lines, chart rows,
   feedback, summary) and the interview drill. Since LAB_UX_GUIDE R6.20 the explanations say the trend in words with at most one
   key number: their words are checked against numpy ("almost all", "splits between", "far below"), and the numbers they no longer
   state are read where they now are, on the chart, the metric cards and Show the math.
6. The PyTorch code under "Minimal implementation" is run as printed (when torch is installed) and must give the same
   weights and outputs as numpy for both sentences and 1 / 2 / 4 heads.

Run: python3 tests/mha_numbers.py
"""
import json, math, pathlib, re, subprocess, sys, itertools
import numpy as np
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/mha.html', '--only', 'mha'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-mha.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'mha.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-mha.html')
fails, n = [], 0


def rhu(v, d):
    """round half away from zero, as a reader does by hand (the page does the same)"""
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-d)
    v = float(v)
    return float(Decimal(repr(v)).quantize(q, rounding=ROUND_HALF_UP)) if v >= 0 else -float(Decimal(repr(-v)).quantize(q, rounding=ROUND_HALF_UP))


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(f'{name} :: {info}')
        print('FAIL', name, '::', str(info)[:400])


# ---------- 1. the toy, from the spec ----------
SENT = {'en': ['<s>', 'The', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it'], 'zh': ['<s>', '小猫', '坐', '在', '垫子', '上', '，', '因为', '它']}
NOUN = {'en': {'cat', 'mat'}, 'zh': {'小猫', '垫子'}}
ANIMAL = {'en': {'cat'}, 'zh': {'小猫'}}
PRON = {'en': {'it'}, 'zh': {'它'}}
CONJ = {'en': {'because'}, 'zh': {'因为'}}
OM = 2 * math.pi / 11
a, b, b2, c, d = 5.4, 7.28, 3.0, 3.0, 2.0


def x_of(lang, p):
    w = SENT[lang][p]
    return np.array([math.cos(p * OM), math.sin(p * OM), float(p == 0), float(w in NOUN[lang]), float(w in ANIMAL[lang]), float(w in PRON[lang]), 1.0, float(w in CONJ[lang])])


# feature columns: 0 cos, 1 sin, 2 first, 3 noun, 4 animal, 5 pronoun, 6 one, 7 because
def rot(t):   # the 2 x 2 rotation that turns a position code back by t
    return np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])


Wq = np.zeros((8, 8)); Wk = np.zeros((8, 8)); Wv = np.zeros((8, 8)); Wo = np.eye(8)
# the four patterns sit in the slots (rows) 1-2, 3-4, 5-6, 7-8: previous word, first token, who "it" is, nouns (H1-H4 with 4 heads;
# head A = slots 1-2 and head B = slots 3-4 with 2 heads, so the two words "it" needs fall in different heads)
Wq[0:2, 0:2] = a * rot(OM); Wk[0:2, 0:2] = a * np.eye(2)             # P1: q(p) . k(j) = a^2 cos((p-1-j) w)
Wq[2, 6] = c; Wk[2, 2] = c                                             # P3: first token c^2
Wq[4, 5] = b; Wk[4, 4] = b + d * d / b; Wk[4, 3] = -d * d / b          # P2: pronoun -> animal b^2, other nouns -d^2
Wq[5, 6] = b2; Wq[5, 5] = -b2; Wk[5, 2] = b2                           # P2: any other query -> <s> b2^2
Wq[6, 6] = d; Wk[6, 3] = d                                             # P4: nouns d^2
for h in range(4):
    Wv[2 * h, 7] = 1; Wv[2 * h + 1, 4] = 1                             # values: (is "because", is an animal) per head


def attend(lang, p, H, on=(1, 1, 1, 1)):
    """numpy MHA for the query at position p: returns per-head weights over 9 tokens (0 after p), per-head outputs, block output."""
    X = np.stack([x_of(lang, j) for j in range(9)])
    Q, K, V = X @ Wq.T, X @ Wk.T, X @ Wv.T
    hd = 8 // H
    Wts, outs = [], []
    for h in range(H):
        sl = slice(h * hd, (h + 1) * hd)
        s = K[:p + 1, sl] @ Q[p, sl] / math.sqrt(hd)
        e = np.exp(s - s.max()); w = np.zeros(9); w[:p + 1] = e / e.sum()
        Wts.append(w); outs.append(w @ V[:, sl] if (H != 4 or on[h]) else np.zeros(hd))
    return np.array(Wts), outs, Wo @ np.concatenate(outs)


# ---------- the claims, on numpy numbers ----------
IT, BEC = 8, 7
NT = 9   # the sentence's tokens, <s> included
CAT = {'en': 2, 'zh': 1}
W4 = {l: attend(l, IT, 4)[0] for l in SENT}
W1 = {l: attend(l, IT, 1)[0] for l in SENT}
W2 = {l: attend(l, IT, 2)[0] for l in SENT}
HC = 2   # H3 finds who "it" is
for l in SENT:
    check(f'{l}: 4 heads at "it", H1 >= 85% on "because"', W4[l][0, BEC] >= 0.85, W4[l][0, BEC])
    check(f'{l}: 4 heads at "it", H3 >= 85% on "cat"', W4[l][HC, CAT[l]] >= 0.85, W4[l][HC, CAT[l]])
    check(f'{l}: 2 heads at "it", head A >= 82% on "because" and head B >= 85% on "cat" (each word in its own head)', W2[l][0, BEC] >= 0.82 and W2[l][1, CAT[l]] >= 0.85, (W2[l][0, BEC], W2[l][1, CAT[l]]))
    check(f'{l}: 1 head at "it", neither word above half', W1[l][0, BEC] <= 0.5 and W1[l][0, CAT[l]] <= 0.5, (W1[l][0, BEC], W1[l][0, CAT[l]]))
    top2 = set(np.argsort(-W1[l][0])[:2])
    check(f'{l}: 1 head at "it", "because" and "cat" are the two largest shares (a blend of the two)', top2 == {BEC, CAT[l]}, top2)
    check(f'{l}: every head row adds up to 1', all(abs(W.sum() - 1) < 1e-12 for WW in (W4[l], W2[l], W1[l]) for W in WW))
    for p in range(9):
        w4 = attend(l, p, 4)[0]
        check(f'{l}: causal mask at word {p}: no weight after it', np.all(w4[:, p + 1:] == 0))
        if p >= 1:
            check(f'{l}: H1 puts >= 85% on the previous word at word {p}', w4[0, p - 1] >= 0.85, w4[0, p - 1])
# the language symmetry: the same multiset of weights at "it" with 4, 2 and 1 heads (so the fidelity card needs no language note)
for H, WW in ((4, W4), (2, W2), (1, W1)):
    same = all(np.allclose(np.sort(WW['en'][h]), np.sort(WW['zh'][h]), atol=1e-12) for h in range(H))
    check(f'EN and ZH give the same weights at "it" with {H} head(s)', same)
    check(f'{H} head(s): "because" gets the same share in both languages', abs(WW['en'][0, BEC] - WW['zh'][0, BEC]) < 1e-12)
# challenge: 80% on both words from heads that are on; the unique set with the most heads off is {H1, H3}
res = {}
for on in itertools.product([1, 0], repeat=4):
    bb = max([W4['en'][h, BEC] for h in range(4) if on[h]], default=0)
    bc = max([W4['en'][h, CAT['en']] for h in range(4) if on[h]], default=0)
    res[on] = (bb, bc, bb >= 0.8 and bc >= 0.8)
    for v in (bb, bc):
        check(f'challenge {on}: no value within 1 point of the 80% goal', abs(v - 0.8) > 0.01, v)
met = [on for on, r in res.items() if r[2]]
check('challenge: met exactly when H1 and H3 are on', sorted(met) == sorted(on for on in res if on[0] and on[2]), met)
best = max(met, key=lambda on: 4 - sum(on))
check('challenge: the most heads off is 2, with H1 and H3 on', best == (1, 0, 1, 0) and sum(1 for on in met if 4 - sum(on) == 2) == 1, best)
# the badge at "it" and the prediction test 80%: no shown weight sits within 1 point of it, with 1, 2 or 4 heads
for l in SENT:
    for WW in (W4[l], W2[l], W1[l]):
        for v in (WW[:, BEC].max(), WW[:, CAT[l]].max()):
            check(f'{l}: best weight at "it" is not within 1 point of 80%', abs(v - 0.80) > 0.01, v)
    check(f'{l}: one head at "it" gives neither word 80% (the in-lab answer part 4 states)', W1[l][0, BEC] < 0.8 and W1[l][0, CAT[l]] < 0.8)

# ---------- 2. against the page ----------
with sync_playwright() as pw:
    br = pw.chromium.launch(); pg = br.new_page(); errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    mats = pg.evaluate("()=>__mha.mats()")
    for name, Wn in (('Wq', Wq), ('Wk', Wk), ('Wv', Wv), ('Wo', Wo)):
        check(f'the page\'s {name} equals the matrix built from the spec', np.allclose(np.array(mats[name]), Wn, atol=1e-12), np.abs(np.array(mats[name]) - Wn).max())
    for l in SENT:
        for p in range(9):
            xs = pg.evaluate(f"()=>__mha.x('{l}',{p})")
            check(f'{l} x({p}) equals the features of "{SENT[l][p]}"', np.allclose(xs, x_of(l, p), atol=1e-12), xs)
    worst = 0.0
    for l in SENT:
        for p in range(9):
            for H in (4, 2, 1):
                ons = list(itertools.product([1, 0], repeat=4)) if H == 4 else [(1, 1, 1, 1)]
                for on in ons:
                    r = pg.evaluate("(s)=>{const r=__mha.compute(s); return {w:r.heads.map(h=>h.w), y:r.y, prev:r.prev, bec:r.bec, cat:r.catw}}",
                                    {'lang': l, 'q': p, 'H': H, 'on': [bool(v) for v in on]})
                    Wn, outs, y = attend(l, p, H, on)
                    diff = max(np.abs(np.array(r['w']) - Wn).max(), np.abs(np.array(r['y']) - y).max())
                    worst = max(worst, diff)
                    if diff > 1e-12:
                        check(f'{l} word {p} {H} heads on={on}: weights and output match numpy', False, diff)
                    # best head still on, as the metrics and the challenge use it
                    heads_on = [h for h in range(H) if H != 4 or on[h]]
                    for key, j in (('bec', BEC), ('cat', CAT[l])):
                        v = max([Wn[h, j] for h in heads_on], default=0)
                        check(f'{l} word {p} {H} heads on={on}: best {key} matches', abs(r[key]['v'] - v) < 1e-12, (r[key], v))
    check('all weights and outputs agree with numpy to 1e-12', worst < 1e-12, worst)
    # ---------- 4. Show the math chains ----------
    SQ = {2: 1.4142, 4: 2.0, 8: 2.8284}
    bad = []
    for l in SENT:
        for p in range(9):
            for H in (4, 2, 1):
                for h in range(H):
                    ch = pg.evaluate("([s,h])=>__mha.chain(s,h)", [{'lang': l, 'q': p, 'H': H}, h])
                    Wn = attend(l, p, H)[0][h]
                    for row in ch['rows']:
                        dot = rhu(sum(q * k for q, k in zip(ch['qs'], row['ks'])), 3)
                        if abs(dot - row['dot']) > 1e-9: bad.append(('dot', l, p, H, h, row['j'], dot, row['dot']))
                        if abs(rhu(row['dot'] / SQ[8 // H], 3) - row['sc']) > 1e-9: bad.append(('score', l, p, H, h, row['j']))
                        if abs(rhu(math.exp(row['sc'] - ch['mx']), 4) - row['e']) > 1e-9: bad.append(('exp', l, p, H, h, row['j']))
                    if abs(rhu(sum(r['e'] for r in ch['rows']), 4) - ch['sum']) > 1e-9: bad.append(('sum', l, p, H, h))
                    for row in ch['rows']:
                        shown = rhu(row['e'] / ch['sum'] * 100, 1)
                        if rhu(shown, 0) != rhu(Wn[row['j']] * 100, 0) and not abs(Wn[row['j']] * 100 % 1 - 0.5) < 0.06:
                            bad.append(('weight', l, p, H, h, row['j'], shown, Wn[row['j']] * 100))
                        if abs(shown - Wn[row['j']] * 100) > 0.15: bad.append(('weight1', l, p, H, h, row['j'], shown, Wn[row['j']] * 100))
    check('Show the math: every line recomputes from the shown values, and its weights round like the stage', not bad, bad[:6])
    # ---------- part 7: the real model ----------
    rm = pg.evaluate("()=>__mha.realModel()")
    check('Llama-3.1-8B: 32 heads x 128 = 4,096', rm['heads'] * rm['hd'] == 4096 and rm['d'] == 4096, rm)
    check('Llama-3.1-8B keys and values per token and layer: 2 x 8 x 128 = 2,048 (MHA 8,192)', rm['gqa'] == 2 * 8 * 128 == 2048 and rm['mha'] == 2 * 32 * 128 == 8192, rm)
    check('Llama-3.1-8B KV cache per token in BF16: 131,072 bytes = 128 KiB (MHA 512 KiB)', rm['perTok'] == 131072 == 128 * 1024 and rm['perTokMha'] == 512 * 1024, rm)
    check('parameters of the toy block: 4 x 8 x 8 = 256', sum(np.array(mats[k]).size for k in ('Wq', 'Wk', 'Wv', 'Wo')) == 256)

    # ---------- 5. every number the page prints, read from the rendered page and compared with numpy ----------
    # Llama-3.1-8B from its config.json (specs/transformer-facts.md A2): hidden 4,096, 32 layers, 32 heads, 8 KV heads, head_dim 128
    LD, LH, LKV, LHD, LL = 4096, 32, 8, 128, 32

    def pct(w): return f'{int(rhu(w * 100, 0))}%'
    def pct1(w): return f'{rhu(w * 100, 1):.1f}%'
    def best(W, H, on, j): return max([W[h, j] for h in range(H) if H != 4 or on[h]] + [0.0])
    def pcts(t): return set(re.findall(r'\d+(?:\.\d+)?%', t))
    def dec2(t): return [float(x.replace('−', '-')) for x in re.findall(r'−?\d+\.\d\d(?!\d)', t)]
    def fx(v, d): s = f'{abs(v):.{d}f}'; return ('−' if v < 0 and float(s) != 0 else '') + s
    def wname(l, j): return '<s>' if j == 0 else SENT[l][j]
    def qname(l, j): return '<s>' if j == 0 else ('逗号“，”' if l == 'zh' and SENT[l][j] == '，' else f'“{SENT[l][j]}”')   # as the running text quotes it (M8)
    def ljoin(l, a): return a[0] if len(a) < 2 else (', '.join(a[:-1]) + (',' if len(a) > 2 else '') + ' and ' + a[-1] if l == 'en' else '、'.join(a[:-1]) + '和' + a[-1])

    def look(ws, n):
        """the lab's look() (R6.20): how a head spreads its attention weight over words 0..n-1, from the top share as the chart shows it:
        all (100%), almost all (90% or more), most (60% or more), else split among its words of 10% or more (up to 3)"""
        t = [j for j in sorted(range(n), key=lambda j: (-ws[j], j)) if ws[j] >= 0.1]
        v = int(rhu(ws[t[0]] * 100, 0))
        if v >= 60: return ('all' if v >= 100 else 'almost' if v >= 90 else 'most'), t[:1]
        return ('split', t[:3]) if len(t) >= 2 else ('top', t[:1])
    def nz(t): return re.sub(r'(?<=[\u4e00-\u9fff]) (?=<s>)|(?<=<s>) (?=[\u4e00-\u9fff])', '', t)   # the page spaces <s> from Chinese; compare without

    def lead1(l, kind, js, W0):
        """the one head's lead as the lab writes it: its largest share, and for a split the words it splits its attention between"""
        n = pct(W0[js[0]])
        if kind == 'split':
            return ((f'spreads its attention over ' if len(js) > 2 else 'splits its attention between ') + ljoin(l, [qname(l, j) for j in js]) + f', {n} on {qname(l, js[0])}') if l == 'en' else \
                   f'把注意力分给了{ljoin(l, [qname(l, j) for j in js])}，{qname(l, js[0])}最多，{n}'
        return f'puts {n} of its attention weight on {qname(l, js[0])}' if l == 'en' else f'把 {n} 的注意力权重放在{qname(l, js[0])}上'
    # the owner's review of 2026-10-09 (R6.37: no Show the math row scrolls sideways): the one six-column table became two that fit their
    # box, the score block's (word, k, q·k, score) and the softmax block's (word, e^, attention weight). They are read back as one row per
    # word in the old order (word, q·k, score, e^, attention weight, k), and only when both tables list the same words in the same order
    MATHROWS = """()=>{const T=s=>[...document.querySelectorAll('#mha-mathbody '+s+' tr')].slice(1).map(tr=>[...tr.children].map(td=>td.textContent.trim()));
      const a=T('.mha-mtab-s'), b=T('.mha-mtab-w');
      if(!a.length||a.length!==b.length||a.some((r,i)=>r.length!==4||b[i].length!==3||r[0]!==b[i][0])) return [['the two tables differ',JSON.stringify(a),JSON.stringify(b),'','','']];
      return a.map((r,i)=>[r[0],r[2],r[3],b[i][1],b[i][2],r[1]]);}"""
    OBSLINK = "()=>!!document.querySelector('#mha-rail [data-obs] [data-jump=\"math\"]')"

    def fmt_big(n):
        # the kit's A.fmt.big: 3 significant digits with M or B above 6 digits (R6.18)
        if abs(n) < 1e6: return f'{n:,}'
        dv, u = (1e9, 'B') if abs(n) >= 1e9 else (1e6, 'M')
        v = n / dv; dp = max(0, 2 - math.floor(math.log10(abs(v)))); t = f'{v:.{dp}f}'
        if u == 'M' and abs(float(t)) >= 1000: v = n / 1e9; dp = max(0, 2 - math.floor(math.log10(abs(v)))); t, u = f'{v:.{dp}f}', 'B'
        return t + u

    ui = br.new_page(viewport={'width': 1440, 'height': 1000}); ui.set_default_timeout(8000)
    ui.on('pageerror', lambda e: errs.append(str(e)))
    ui.goto(URL); ui.wait_for_timeout(300)

    def TX(sel): return ui.evaluate("(s)=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\s+/g,' ').trim():''}", sel)
    def hashgo(tok): ui.evaluate("(t)=>{location.hash=t}", tok); ui.wait_for_timeout(260)
    def go(k): ui.evaluate(f"()=>document.querySelector('#mha-rail [data-go=\"{k}\"]').click()"); ui.wait_for_timeout(300)
    def home_tile():
        hashgo('')
        ui.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="models"], #domStrip [data-world="models"]')].find(e=>e.offsetParent);
          if(g) g.click(); else {const m=document.querySelector('#mapSvg .world[data-world="models"]'); if(m) m.dispatchEvent(new MouseEvent('click',{bubbles:true}));}}""")
        ui.wait_for_timeout(350)

    for l in ('en', 'zh'):
        ci = CAT[l]; B_, C_ = SENT[l][BEC], SENT[l][ci]
        hashgo(''); ui.click(f'.langsw button[data-lang="{l}"]'); ui.wait_for_timeout(250)
        # --- home tile: the sentence and the bars at 1 and 4 heads ---
        home_tile()
        if l == 'en':
            check('UI tile: opens on 2 heads (the one-head split is part 2’s prediction)', ui.evaluate("()=>document.querySelector('#tile-mha .seg-ctl button[aria-pressed=\"true\"]').dataset.v") == '2')
        for v in (1, 2):
            ui.click(f'#tile-mha .seg-ctl button[data-v="{v}"]'); ui.wait_for_timeout(150)
            W = attend(l, IT, v)[0]
            exp = {pct(W[0, BEC]), pct(W[0, ci])} if v == 1 else {pct(W[0, BEC]), pct(W[1, ci])}
            check(f'UI {l} tile, {v} head(s): the sentence', pcts(TX('#tile-mha .tz-read')) == exp, (TX('#tile-mha .tz-read'), exp))
            cells = ui.evaluate("()=>[...document.querySelectorAll('#tile-mha .mha-row')].map(r=>[...r.querySelectorAll('.mha-pv')].map(c=>c.textContent.trim()))")
            expc = [[pct(W[h, j]) if rhu(W[h, j] * 100, 0) >= 1 else '' for j in range(9)] for h in range(v)]
            check(f'UI {l} tile, {v} head(s): the bars', cells == expc, (cells, expc))
        # --- Scenarios and Settings: every state through the link token (sandbox) ---
        states = [(p, H, (1, 1, 1, 1)) for p in range(9) for H in (4, 2, 1)]
        states += [(IT, 4, on) for on in itertools.product([1, 0], repeat=4) if on != (1, 1, 1, 1)] + [(3, 4, (0, 1, 1, 1)), (5, 4, (1, 0, 0, 1)), (0, 4, (0, 1, 0, 1))]
        for (p, H, on) in states:
            tag = f'UI {l} {p}_{H}_{"".join(map(str, on))}'
            hashgo(f'mha~{p}_{H}_{"".join(map(str, on))}')
            W = attend(l, p, H, on)[0]
            g = ui.evaluate("()=>[...document.querySelectorAll('#mha-grid .mha-row')].map(r=>[...r.querySelectorAll('.mha-cell')].map(c=>[c.querySelector('.mha-pv').textContent.trim(), c.dataset.tipx||'']))")
            exp = [[(pct(W[h, j]) if rhu(W[h, j] * 100, 0) >= 1 else '') if j <= p else '' for j in range(9)] for h in range(H)]
            check(f'{tag}: chart percentages', [[c[0] for c in row] for row in g] == exp, (g, exp))
            # N8: a cell's hover gives the attention weight to 0.1% as Show the math has it (its rounded chain), within 0.15 points of numpy
            chs = [{r_['j']: r_['w'] for r_ in ui.evaluate("([s,h])=>__mha.chain(s,h).rows", [{'lang': l, 'q': p, 'H': H, 'on': list(on)}, h])} for h in range(H)]
            tipbad = [(h, j, g[h][j][1][-60:], pct1(chs[h][j])) for h in range(H) for j in range(p + 1) if pct1(chs[h][j]) not in g[h][j][1] or abs(chs[h][j] - W[h, j]) * 100 > 0.15]
            check(f'{tag}: cell hovers give the attention weight to 0.1% as Show the math does (N8)', not tipbad, tipbad[:3])
            met = ui.evaluate("()=>Object.fromEntries([...document.querySelectorAll('#mha-metrics .metric')].map(m=>[m.dataset.k,m.querySelector('.v').textContent.trim()]))")
            expm = {'H': str(H), 'prev': '—' if p == 0 else pct(best(W, H, on, p - 1)), 'cat': '0%' if ci > p else pct(best(W, H, on, ci)), 'par': '256'}
            check(f'{tag}: metrics', met == expm, (met, expm))
            st = TX('#mha-status')
            if p == IT:
                kb, kc = best(W, H, on, BEC) >= 0.80, best(W, H, on, ci) >= 0.80
                if kb and kc: want = 'Both words at 80% or more' if l == 'en' else '两个词都在 80% 以上'
                elif kb or kc: want = f'Only “{B_ if kb else C_}” at 80% or more' if l == 'en' else f'只有“{B_ if kb else C_}”在 80% 以上'
                else: want = 'Neither word at 80% or more' if l == 'en' else '两个词都不到 80%'
                check(f'{tag}: badge', st == want, (st, want))
            else:
                check(f'{tag}: badge counts {p + 1} tokens', st.endswith(f'sees {p + 1} token' + ('s' if p else '')) if l == 'en' else st.endswith(f'能看到 {p + 1} 个 token'), st)
            ins = TX('#mha-insight')
            extra = pcts(ins) - {pct(W[h, j]) for h in range(H) for j in range(p + 1)} - {'100%'}
            check(f'{tag}: every % in the scenario card is a weight of this state', not extra, (extra, ins[:300]))
            if (p, H, on) == (IT, 4, (1, 1, 1, 1)):
                # R6.20: the card says in words where each head looks; the weights are the chart's (checked above). The words hold for numpy:
                # H1, H2 and H3 at 90% to 99% as shown (“almost only”) on their word, H4 split between the two nouns
                need = [f'H1 looks almost only at “{B_}”, H2 at <s> and H3 at “{C_}”', 'H4 splits its attention between'] if l == 'en' else \
                       [f'H1、H2、H3 各自几乎只看一个词，分别是“{B_}”、<s> 和“{C_}”', 'H4 把注意力分给了']
                lv = [int(rhu(W[h, j] * 100, 0)) for h, j in ((0, BEC), (1, 0), (2, ci))]
                k4 = look(W[3], 9)
                holds = 90 <= min(lv) and min(lv) < 100 and k4[0] == 'split' and set(k4[1]) == set(j for j in range(9) if SENT[l][j] in NOUN[l])
                check(f'{tag}: the “it” scenario says where H1 to H4 look, and the words hold for numpy ({lv}, H4 {k4})', all(s in ins for s in need) and holds and all(qname(l, j) in ins.split(need[1])[-1] for j in k4[1]), (need, ins[:300]))
            if (p, H, on) in ((IT, 4, (1, 1, 1, 1)), (IT, 1, (1, 1, 1, 1)), (3, 2, (1, 1, 1, 1)), (IT, 4, (1, 0, 1, 0))):
                # the mini result bar: at phone width it shows once the chart has scrolled away
                ui.set_viewport_size({'width': 390, 'height': 844}); ui.wait_for_timeout(200)
                # phones (m2, Batch 1): Settings is a closed drawer at the end of the bench, so scroll to the card under the stage instead
                # since 2026-10-11 an address with a setup can open on 自由调参, where the card is hidden: scroll the chart away instead
                ui.evaluate("()=>{window.scrollTo(0, document.getElementById('mha-stage').getBoundingClientRect().bottom+scrollY+20); window.dispatchEvent(new Event('scroll'))}"); ui.wait_for_timeout(400)
                dk = ui.evaluate("()=>({hidden:document.getElementById('dock').hidden, t:document.getElementById('dock').innerText})")
                want = {pct(best(W, H, on, BEC)), pct(best(W, H, on, ci))} if p == IT else {pct(best(W, H, on, p - 1))}
                check(f'{tag}: mini result bar (80% only in the badge)', not dk['hidden'] and pcts(dk['t']) - {'80%'} == want, (dk, want))
                ui.evaluate("()=>{window.scrollTo(0,0); window.dispatchEvent(new Event('scroll'))}"); ui.set_viewport_size({'width': 1440, 'height': 1000}); ui.wait_for_timeout(200)
        # --- Show the math, rendered table against the chain and numpy ---
        for (p, H) in ((3, 4), (IT, 4), (IT, 2), (IT, 1)):
            tag = f'UI {l} math {p}_{H}'
            hashgo(f'mha~{p}_{H}_1111'); ui.evaluate("()=>{document.getElementById('mha-math').open=true}"); ui.wait_for_timeout(150)
            rows = ui.evaluate(MATHROWS)
            ch = ui.evaluate("([s,h])=>__mha.chain(s,h)", [{'lang': l, 'q': p, 'H': H}, 0])
            expt = [[fx(r['dot'], 3), fx(r['sc'], 3), f"{r['e']:.4f}", f"{rhu(r['w'] * 100, 1):.1f}%"] for r in ch['rows']]
            check(f'{tag}: table rows equal the chain (word, q·k, score, e^, weight, then k)', [r[1:5] for r in rows] == expt and all(r[5] == '(' + ', '.join(fx(v, 3) for v in c['ks']) + ')' for r, c in zip(rows, ch['rows'])), (rows[:2], expt[:2]))
            Wn = attend(l, p, H)[0][0]
            check(f'{tag}: table weights within 0.15 points of numpy', len(rows) == p + 1 and all(abs(float(r[4][:-1]) - Wn[j] * 100) <= 0.15 for j, r in enumerate(rows)), rows)
            body = TX('#mha-mathbody')
            hd = 8 // H
            sq = ui.evaluate("()=>{const s=document.querySelector('#mha-mathbody .mha-sq math msqrt'); return s?s.textContent.trim():''}")   # 2026-10-09 (H1): √ typeset
            check(f'{tag}: head_dim, √head_dim (typeset) and the sum line', f'head_dim = 8 ÷ {H} = {hd}' in body and sq == str(hd) and f'= {"2" if hd == 4 else f"{math.sqrt(hd):.4f}"}' in body and f"= {ch['sum']:.4f}" in body, (sq, body[:200]))
            ui.evaluate("()=>{document.getElementById('mha-math').open=false}")
        # --- Settings: the hand-set parameters note (N7, R6.18: 41.9M, shapes written out; the owner's clarification of 2026-10-07:
        #     no hover that only gives the exact value, so 41.9M is plain text) ---
        ui.evaluate("()=>{const d=document.getElementById('mha-adv'); if(d) d.open=true}"); ui.wait_for_timeout(120)
        note = TX('#mha-adv'); att = 2 * LD * LD + 2 * LD * LKV * LHD
        tipn = ui.evaluate("()=>[...document.querySelectorAll('#mha-adv .def')].map(d=>d.getAttribute('data-tipx')).join(' ')")
        big_ok = (f'The real block is far bigger: Llama-3.1-8B’s attention block has {fmt_big(att)} parameters per layer (Wq and Wo {LD:,} × {LD:,} each, Wk and Wv {LD:,} × {LKV * LHD:,} each).' in note) if l == 'en' else \
                 (f'真实的块大得多：Llama-3.1-8B 的 attention 块每层有 {fmt_big(att)} 个参数（Wq 和 Wo 各 {LD:,} × {LD:,}，Wk 和 Wv 各 {LD:,} × {LKV * LHD:,}）。' in note)
        nb = ui.evaluate("()=>[...document.querySelectorAll('#mha-adv .num-big')].map(e=>({t:e.textContent,hover:e.classList.contains('def')||!!e.closest('.def')||e.hasAttribute('data-tipx')||e.hasAttribute('tabindex')}))")
        tip_ok = nb == [{'t': '41.9M', 'hover': False}] and '精确值' not in tipn and 'Exactly' not in tipn
        check(f'UI {l} settings note: 41.9M with no exact-value hover, the shapes written out, 80% with H1 and H3 at their attention weights', att == 41943040 and fmt_big(att) == '41.9M' and big_ok and tip_ok and '41,943,040' not in note and '4K' not in note and '80%' in note and pct(W4[l][0, BEC]) in note and pct(W4[l][HC, ci]) in note, (note, tipn[:200], nb))
        # --- guided parts: 1 head, then 2, then 4 (R3.13) ---
        ui.click('.dyn-root .modes button[data-mode="guided"]'); ui.wait_for_timeout(300)
        go(1); ui.click('#mha-rail [data-pred="one"]'); ui.wait_for_timeout(200)
        go(2)
        for p in range(9):
            q_before = ui.evaluate("()=>document.querySelector('#mha-grid .mha-tok.q').dataset.q")
            ui.click(f'#mha-ic-q button[data-q="{p}"]', force=(p == IT)); ui.wait_for_timeout(200)   # “it” is aria-disabled in part 3: a real click only shows why
            ob = TX('#mha-rail [data-obs]'); W = attend(l, p, 1)[0]; tag = f'UI {l} part 3 at word {p}'
            if p == IT:
                check(f'{tag}: “it” is kept for part 4 (the current word does not change)', ui.evaluate("()=>document.querySelector('#mha-grid .mha-tok.q').dataset.q") == q_before, q_before)
                continue
            # R6.20: the later words are counted on the stage (hatched cells); the card only says they get none
            nmask = ui.evaluate("()=>document.querySelector('#mha-grid .mha-row').querySelectorAll('.mha-cell.masked').length")
            msk = ('words after it get no attention weight' if p < NT - 2 else 'word after it gets no attention weight') if l == 'en' else '后面的词分不到注意力权重'
            if p == 0:
                check(f'{tag}: 100% on <s>; the 8 later words hatched on the stage, and the card says they get none', pcts(ob) == {'100%'} and nmask == 8 and msk in ob, (nmask, ob))
                continue
            kind, js = look(W[0], p + 1)
            check(f'{tag}: one number, the one head’s largest share ({pct(W[0, js[0]])}), and for a split the words it goes to ({kind}: {[wname(l, j) for j in js]}); the later words hatched',
                  pcts(ob) - {'100%'} == {pct(W[0, js[0]])} - {'100%'} and nz(lead1(l, kind, js, W[0])) in nz(ob) and nmask == NT - 1 - p and msk in ob, (kind, js, lead1(l, kind, js, W[0]), ob))
            # the scores behind the shares are Show the math's: its table lists them as the chain has them, and the card's reason names the
            # highest (for a split: the close ones), with a link there. The words hold: “clearly highest” leads by 0.5 or more, “close” within 1
            ch = ui.evaluate("([s,h])=>__mha.chain(s,h)", [{'lang': l, 'q': p, 'H': 1}, 0])
            ui.evaluate("()=>{document.getElementById('mha-math').open=true}"); ui.wait_for_timeout(120)
            rows = ui.evaluate(MATHROWS); ui.evaluate("()=>{document.getElementById('mha-math').open=false}")
            sc = sorted((r['sc'] for r in ch['rows']), reverse=True); scj = {r['j']: r['sc'] for r in ch['rows']}
            why = ob[ob.find('Why' if l == 'en' else '原因'):]
            words_ok = (max(scj[j] for j in js) - min(scj[j] for j in js) < 1 and ('have close scores, so they share most of the attention weight' if l == 'en' else '的得分很接近，所以它们分走了大部分注意力权重') in why) if kind == 'split' else \
                       (sc[0] - sc[1] >= 0.5 and nz(f'{qname(l, js[0])} has the highest score, so it gets the largest share' if l == 'en' else f'{qname(l, js[0])}的得分最高，所以份额最大') in nz(why))
            check(f'{tag}: the scores are in Show the math, as the chain has them; the reason names {[wname(l, j) for j in js]} (highest, or close for a split: the chain’s scores) and links there',
                  [r[2] for r in rows] == [fx(r['sc'], 3) for r in ch['rows']] and all(qname(l, j) in why for j in js) and words_ok and ui.evaluate(OBSLINK), (rows[:3], sc[:3], why[:220]))
        go(3)
        if l == 'en':   # the first visit (what the learner did in a part is kept across a language switch)
            check(f'UI {l} part 4: no result before a word is picked', not ui.is_visible('#mha-rail [data-obs]'))
        ui.click('#mha-ic-q button[data-q="8"]'); ui.wait_for_timeout(250)
        ob = TX('#mha-rail [data-obs]'); W = W1[l]
        rest1 = [j for j in range(9) if W[0, j] >= 0.05 and j not in (BEC, ci)]
        want = {pct(W[0, BEC]), pct(W[0, ci])}   # R6.20: the pair in the result line; the other words' shares are on the chart
        check(f'UI {l} part 4 at “it”: percentages (the pair; 80% only in the verdict, 100% as the total)', pcts(ob) - {'80%', '100%'} == want, (pcts(ob), want, ob))
        X = np.stack([x_of(l, j) for j in range(9)]); Qn, Kn = X @ Wq.T, X @ Wk.T
        dots = Kn @ Qn[IT]
        check(f'UI {l} part 4: the one head gives the rest to {[wname(l, j) for j in rest1]}', rest1 and all(wname(l, j) in ob for j in rest1), ob)
        # the q·k values moved to Show the math (R6.20): read them there (the chain's, within 0.01 of numpy); the card calls the two words
        # “almost equal” (numpy: less than 0.5 apart) and links there
        ui.evaluate("()=>{document.getElementById('mha-math').open=true}"); ui.wait_for_timeout(120)
        qk = {r[0]: r[1] for r in ui.evaluate(MATHROWS)}; ui.evaluate("()=>{document.getElementById('mha-math').open=false}")
        chn = ui.evaluate("([s,h])=>__mha.chain(s,h)", [{'lang': l, 'q': IT, 'H': 1}, 0])
        tab_ok = all(qk.get(wname(l, r['j'])) == fx(r['dot'], 3) and abs(r['dot'] - dots[r['j']]) < 0.01 for r in chn['rows'])
        eq = ('come out almost equal' in ob) if l == 'en' else ('几乎一样高' in ob)   # q·k is typeset since 2026-10-09 (H1)
        check(f'UI {l} part 4: one head’s q·k in Show the math ({rhu(dots[BEC], 1)} and {rhu(dots[ci], 1)} for the two words), called almost equal on the card, with a link there',
              tab_ok and eq and abs(dots[BEC] - dots[ci]) < 0.5 and ui.evaluate(OBSLINK), (qk, ob[-320:]))
        rn = ljoin(l, [qname(l, j) for j in rest1])
        goes = (f'what is left goes to {rn}' in ob) if l == 'en' else (f'剩下的给了{rn}' in ob)
        split = ('almost equal shares' in ob) if l == 'en' else ('差不多的份额' in ob)
        half = (('less than half of each one’s value' in ob) if l == 'en' else ('都只有不到一半' in ob)) and W[0, BEC] < 0.5 and W[0, ci] < 0.5
        check(f'UI {l} part 4: the one softmax gives the two words almost equal shares ({pct(W[0, BEC])}, {pct(W[0, ci])}), the rest goes to the other words (N4), each keeps under half', goes and split and half, ob)
        if l == 'zh':
            check('UI zh part 4: the comma token is written 逗号“，” (M8)', '剩下的给了逗号“，”和“它”' in ob, ob)
        go(4)
        ui.click('#mha-ic-H button[data-v="2"]'); ui.wait_for_timeout(250)
        ob = TX('#mha-rail [data-obs]'); W = W2[l]
        check(f'UI {l} part 5 with 2 heads: head A on “{B_}” and head B on “{C_}”, and the output carries them', pcts(ob) == {pct(W[0, BEC]), pct(W[1, ci])} and '256' in ob and 'head A' in ob and 'head B' in ob, (pcts(ob), ob))
        mostw = ('carries most of both words’ values' in ob and 'head A gets the first half' in ob) if l == 'en' else ('两个词的大部分 value' in ob and 'head A 分到前一半' in ob)
        check(f'UI {l} part 5: the reason in words (R6.20), “most of both words’ values” holding for numpy', mostw and W[0, BEC] >= 0.5 and W[1, ci] >= 0.5, ob[-400:])
        go(5)
        ui.click('#mha-ic-H button[data-v="4"]'); ui.wait_for_timeout(250)
        ob = TX('#mha-rail [data-obs]'); W = W4[l]
        # R6.20: the card says where each head looks in words (no percentages); the words hold for numpy, and the two weights the goal needs
        # are on the metric cards (H1's on the previous word, H3's on the word “it” refers to)
        need = [f'H1 looks almost only at “{B_}”, H2 at <s> and H3 at “{C_}”', 'H4 splits its attention between'] if l == 'en' else \
               [f'H1、H2、H3 各自几乎只看一个词，分别是“{B_}”、<s> 和“{C_}”', 'H4 把注意力分给了']
        lv = [int(rhu(W[h, j] * 100, 0)) for h, j in ((0, BEC), (1, 0), (2, ci))]; k4 = look(W[3], 9)
        met = ui.evaluate("()=>Object.fromEntries([...document.querySelectorAll('#mha-metrics .metric')].map(m=>[m.dataset.k,m.querySelector('.v').textContent.trim()]))")
        check(f'UI {l} part 6 with 4 heads: H1 on “{B_}”, H2 on <s>, H3 on “{C_}”, H4 split between the nouns, in words that hold for numpy ({lv}), and 256',
              not pcts(ob) and all(s_ in ob for s_ in need) and 90 <= min(lv) < 100 and k4[0] == 'split' and all(qname(l, j) in ob.split(need[1])[-1] for j in k4[1]) and '256' in ob, (pcts(ob), lv, k4, ob[:300]))
        check(f'UI {l} part 6: the metric cards show H1’s {pct(W[0, BEC])} and H3’s {pct(W[2, ci])}', met.get('prev') == pct(W[0, BEC]) and met.get('cat') == pct(W[2, ci]), met)
        go(6)
        card = TX('#mha-rail .step.current')
        need = [f'each token’s query is cut into {LH} heads of {LHD} numbers', f'also lets every {LH // LKV} heads share one key and one value, which is why its config.json lists {LKV} KV heads', 'this is called grouped-query attention (GQA), and the lab GQA and MLA covers it.'] if l == 'en' else \
               [f'每个 token 的 query 切成 {LH} 个 head，每个 {LHD} 维', f'还让每 {LH // LKV} 个 head 共用一份 key 和 value，所以 config.json 里写着 {LKV} 个 KV head', '这种做法叫 grouped-query attention（GQA），在“GQA 和 MLA”这个 Lab 里讲。']
        miss = [s_ for s_ in need if s_ not in card]
        check(f'UI {l} part 7: Llama-3.1-8B’s heads from its config, and one sentence on sharing keys and values (N9)', not miss, (miss, card[:500]))
        check(f'UI {l} part 7: no MHA, query head, KV cache or byte counts on the card (N9, R6.16)', all(x not in card for x in ('query head', 'MHA', 'KV cache', '131,072', 'KiB')), card[-300:])
        # the kit's KV head and GQA definitions; GQA followed by this lab's model sentence computed from the real-model helper
        tips = ui.evaluate("()=>{const o={}; document.querySelectorAll('#mha-rail .step.current .def').forEach(d=>{o[d.textContent.trim()]=d.getAttribute('data-tipx')||''}); return o}")
        kvh_t, gqa_t = tips.get('KV heads' if l == 'en' else 'KV head', ''), tips.get('GQA', '')
        gqa_end = (f' Llama-3.1-8B, used here, has {LH} query heads in {LKV} groups of {LH // LKV}, so each token stores {LKV} keys and {LKV} values per layer instead of {LH} of each.' if l == 'en'
                   else f'这里用的 Llama-3.1-8B 有 {LH} 个 query head，分成 {LKV} 组、每组 {LH // LKV} 个，所以每个 token 每层只存 {LKV} 个 key 和 {LKV} 个 value，而不是各 {LH} 个。')
        check(f'UI {l} part 7: the KV head hover is the kit text', kvh_t.startswith('A KV head is one key vector' if l == 'en' else 'KV head 是每个 token 在每一层存的一个 key 向量'), kvh_t[:120])
        check(f'UI {l} part 7: the GQA hover ends with the model sentence (N5)', gqa_t.endswith(gqa_end) and len(gqa_t) > len(gqa_end) + 40, gqa_t[-220:])
        go(7)
        card = TX('#mha-rail .step.current')
        masked3 = attend(l, 3, 4)[0][:, 4:].sum(axis=1).max()
        want = {'100%', pct(masked3)}   # R6.20: the sum rule and the masked words; the shares are parts 4 and 5's
        # G2 (the owner, 2026-10-08): the points state the pattern (one head splits its attention; each head takes from its own word; the
        # mask gives later words 0%), no detail of an earlier part; numpy below still confirms the pattern holds in this lab.
        # M9: misconception 1 names its condition (every head has its own key and value here; Llama keeps 8 KV heads); misconception 2: the
        # slices are cut from q, k and v, each computed from the whole token. R6.16: no Llama shapes on this card (they are part 7's)
        need = ([f'only Wk and Wv change with the head count, because its {LKV} KV heads stay fixed', 'Every head here keeps its own key and value', f'Llama-3.1-8B has {LH} in every layer', 'computed from all 8 of the token’s numbers',
                 'one head has to split its attention between them, and neither gets most of it', 'each head from its own', 'every later word 0% of the attention weight'] if l == 'en'
                else [f'只有 Wk 和 Wv 会变，因为它的 {LKV} 个 KV head 不变', '这里每个 head 各有 key 和 value', f'Llama-3.1-8B 每层有 {LH} 个', '都由 token 的全部 8 维算出',
                      '一个 head 只能把注意力分给两边，哪一边都拿不到大头', '每个 head 取自己的那个', '后面每个词的注意力权重都是 0%'])
        holds = 0.35 <= min(W1[l][0, BEC], W1[l][0, ci]) and max(W1[l][0, BEC], W1[l][0, ci]) < 0.5 and look(W2[l][0], 9) in (('most', [BEC]), ('almost', [BEC]), ('all', [BEC])) and look(W2[l][1], 9) in (('most', [ci]), ('almost', [ci]), ('all', [ci]))
        check(f'UI {l} part 8: percentages (100% the total, 0% the masked words), 256, Llama’s head count, no shapes, and its words hold for numpy', pcts(card) == want and masked3 == 0 and '256' in card and all(s_ in card for s_ in need) and f'{LD:,} × {LD:,}' not in card and holds, (pcts(card), want, [s_ for s_ in need if s_ not in card]))
        # --- challenge: goal lines, chart rows and feedback for several choices ---
        ui.click('.dyn-root .modes button[data-mode="challenge"]'); ui.wait_for_timeout(300)
        ui.click('#mha-rail [data-act="submit"]'); ui.wait_for_timeout(250)
        cur = [1, 1, 1, 1]; W = W4[l]
        ib, ic = best(W, 4, (1, 1, 1, 1), BEC), best(W, 4, (1, 1, 1, 1), ci)
        o1b, o3c = best(W, 4, (0, 1, 1, 1), BEC), best(W, 4, (1, 1, 0, 1), ci)
        for on in [(1, 1, 1, 1), (1, 0, 1, 1), (1, 0, 1, 0), (1, 1, 0, 1), (0, 1, 1, 1), (1, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0)]:
            for h in range(4):
                if cur[h] != on[h]:
                    ui.click(f'#mha-ic-on button[data-on="{h}"]'); ui.wait_for_timeout(120); cur[h] = on[h]
            ui.wait_for_timeout(150); tag = f'UI {l} challenge {"".join(map(str, on))}'
            bb, bc = best(W, 4, on, BEC), best(W, 4, on, ci)
            gl = (TX('#mha-ch-b b'), TX('#mha-ch-c b'), TX('#mha-ch-o b'))
            check(f'{tag}: goal lines', gl == (pct(bb), pct(bc), str(4 - sum(on))), (gl, pct(bb), pct(bc)))
            rows = ui.evaluate("()=>[...document.querySelectorAll('#mha-ch-res .mha-cr')].map(r=>[r.querySelector('.cmp-lab span').textContent,[...r.querySelectorAll('.cmp-val')].map(v=>v.textContent.trim())])")
            bad = [(lab, vals) for lab, vals in rows if vals != [pct(best(W, 4, tuple(int(f'H{h + 1}' in lab) for h in range(4)), BEC)), pct(best(W, 4, tuple(int(f'H{h + 1}' in lab) for h in range(4)), ci))]]
            check(f'{tag}: every chart row matches numpy, and the pick is on the chart', rows and not bad and any(tuple(int(f'H{h + 1}' in lab) for h in range(4)) == on for lab, _ in rows), (bad, rows))
            paras = ui.evaluate("()=>[...document.querySelectorAll('#mha-ch-res > p')].map(p=>p.innerText.replace(/\\s+/g,' ').trim())")
            ok = bb >= 0.8 and bc >= 0.8
            lead = (pcts(paras[0]) if paras else set()) - {'80%'}   # 80% is the goal itself
            looks = {pct(W[h, j]) for h in (1, 3) for j in range(9)}
            check(f'{tag}: feedback line', (({pct(bb), pct(bc)} <= lead <= {pct(bb), pct(bc)} | looks) if ok else ({pct(v) for v in (bb, bc) if v < 0.8} <= lead <= {pct(bb), pct(bc)})), (lead, paras[:1]))
            # M2 (R4.4): “Goal met” only for the fewest heads; met with more heads on says so, and a miss says “Not yet”
            fewest = ok and on == (1, 0, 1, 0)
            head = (('Goal met: with only H1 and H3 on' if fewest else 'Both still at 80% or more:') if ok else 'Not yet:') if l == 'en' else \
                   (('达成目标：只开着 H1 和 H3' if fewest else '两个词仍都在 80% 以上：') if ok else '还没达到：')
            check(f'{tag}: feedback opens with “{head}”', bool(paras) and paras[0].startswith(head) and (fewest or ('Goal met' not in paras[0] and '达成目标' not in paras[0])), paras[:1])
            summ = next((t for t in paras if 'H1' in t and 'H2' in t and ('Initial' in t or '初始' in t)), '')
            # R6.20: baseline, effect and reason in words; the words hold for numpy (“far below” under 40%), and the weights the summary no
            # longer states are the chart's rows for H1 off and H3 off (every row is checked against numpy above)
            f1, f3 = o1b < 0.4, o3c < 0.4
            want = ['Initial setup, all 4 heads on: both words clear the 80% line.', 'Turning off H2 or H4 changes neither.', f'Turning off H1 drops “{B_}” {"far below" if f1 else "below"} that line', f'turning off H3 drops “{C_}” {"far below" if f3 else "below"} it too'] if l == 'en' else \
                   ['初始设置是 4 个 head 全开：两个词都过了 80% 这条线。', '关掉 H2 或 H4，两个词都不变。', f'关掉 H1，“{B_}”{"远远掉到" if f1 else "掉到"}这条线以下', f'关掉 H3，“{C_}”也{"远远掉到" if f3 else "掉到"}线以下']
            same = all(pct(best(W, 4, o_, j)) == pct(best(W, 4, (1, 1, 1, 1), j)) for o_ in ((1, 0, 1, 1), (1, 1, 1, 0)) for j in (BEC, ci))
            onc = {tuple(int(f'H{h + 1}' in lab) for h in range(4)): vals for lab, vals in rows}
            chart = onc.get((0, 1, 1, 1), [None])[0] == pct(o1b) and onc.get((1, 1, 0, 1), [None, None])[1] == pct(o3c)
            check(f'{tag}: summary of the initial setup and of H1 / H3 off, in words that hold for numpy, the weights on the chart ({pct(o1b)}, {pct(o3c)})',
                  all(s in summ for s in want) and ib >= 0.8 and ic >= 0.8 and same and o1b < 0.8 and o3c < 0.8 and chart, (want, summ, onc))
        # --- interview drill: every answer redone from its question ---
        for i in range(10):
            ui.click('.dyn-root [data-act="drill-new"]'); ui.wait_for_timeout(80); ui.click('.dyn-root [data-act="drill-reveal"]'); ui.wait_for_timeout(80)
            q, a = TX('#mha-drill-q'), TX('#mha-drill-a')
            if l == 'en':
                m1, m2, m3 = re.search(r'^A layer has hidden size d = ([\d,]+) and (\d+) heads in its attention block, with no biases\.', q), re.search(r'With (\d+) heads instead', q), re.search(r'with (\d+) KV heads \(grouped-query attention, GQA\)', q)
                ma = re.search(r'\(1\) (\d+)\. \(2\) ([\d.]+[MB]|[\d,]+)\. \(3\) (\d+) and still ([\d.]+[MB]|[\d,]+)\. \(4\) ([\d,]+) and ([\d,]+)\.', a)
            else:
                m1, m2, m3 = re.search(r'^一层的隐藏维度 d = ([\d,]+)，attention 块里有 (\d+) 个 head，没有 bias。', q), re.search(r'改成 (\d+) 个 head', q), re.search(r'用 (\d+) 个 KV head（grouped-query attention，GQA）', q)
                ma = re.search(r'（1）(\d+)。（2）([\d.]+[MB]|[\d,]+)。（3）(\d+)，参数还是 ([\d.]+[MB]|[\d,]+)。（4）([\d,]+) 和 ([\d,]+)。', a)
            if not (m1 and m2 and m3 and ma):
                check(f'UI {l} drill {i}: question and answer parse', False, (q, a)); continue
            dd, hh, h2, gg = int(m1.group(1).replace(',', '')), int(m1.group(2)), int(m2.group(1)), int(m3.group(1))
            g_ = ma.groups(); hdd = dd // hh
            got = [int(g_[0]), g_[1], int(g_[2]), g_[3], int(g_[4].replace(',', '')), int(g_[5].replace(',', ''))]
            exact = ui.evaluate("()=>[...document.querySelectorAll('#mha-drill-a .num-big')].map(e=>e.getAttribute('data-tipx')||(e.closest('.def')?'def':null))")
            check(f'UI {l} drill {i}: d={dd}, heads={hh}, 2x heads={h2}, KV heads={gg}; the parameter count as {fmt_big(4 * dd * dd)}, with no exact-value hover (the owner, 2026-10-07)',
                  h2 == 2 * hh and dd % hh == 0 and got == [hdd, fmt_big(4 * dd * dd), dd // h2, fmt_big(4 * dd * dd), 2 * hh * hdd, 2 * gg * hdd] and exact and all(t is None for t in exact), (got, dd, hh, gg, exact[:1]))
    # ---------- 6. the PyTorch code in "Minimal implementation" runs and gives the toy's numbers ----------
    code = ui.evaluate("()=>document.querySelector('#mha-code').textContent")
    try:
        import torch
    except ImportError:
        torch = None
        print('skipped: the PyTorch sample (torch is not installed)')
    if torch is not None:
        ns = {}; exec(code, ns)
        for l in SENT:
            X = torch.tensor(np.stack([x_of(l, j) for j in range(9)]))
            for H in (4, 2, 1):
                out, w = ns['mha'](X, *(torch.tensor(Wn) for Wn in (Wq, Wk, Wv, Wo)), H)
                ok = all(np.allclose(w[:, p, :].numpy(), attend(l, p, H)[0], atol=1e-12) and np.allclose(out[p].numpy(), attend(l, p, H)[2], atol=1e-12) for p in range(9))
                check(f'the PyTorch sample gives the page\'s weights and outputs ({l}, {H} heads)', ok)
    ui.close()
    check('no page errors', not errs, errs[:3])
    br.close()

print(f'{n} checks, {len(fails)} failed')
if fails:
    print('FAILED:', fails[:30])
sys.exit(1 if fails else 0)
