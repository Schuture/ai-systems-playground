"""Golden numbers for the DPO lab, recomputed independently in Python.

The page trains a softmax policy over four answers with exact gradient steps on the DPO loss.
This script does NOT port the page's JavaScript: it writes the loss straight from the paper
(Rafailov et al., 2023, Eq. 7), takes its gradient by central finite differences (and, as a
cross-check, through the full softmax Jacobian), runs the same update rule, and compares every
key number with the page through the read-only hook window.__dpo and with the rendered text.

It also checks the round-5 rules on formula lines: every line under "Show the math" and in part 4
is recomputed from the numbers printed in that line (not from the engine), and the final values
must agree with the unrounded values on the metric cards.

Run: python3 tests/dpo_numbers.py   (builds out/dpo.html and out/wrapped-dpo.html)
"""
import json, math, pathlib, re, subprocess, sys, types, itertools
import numpy as np
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/dpo.html', '--only', 'dpo'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-dpo.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'dpo.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-dpo.html')

# ---------- the toy, as stated on the page ----------
REFS = {'b': [0.18, 0.52, 0.18, 0.12], 's': [0.05, 0.85, 0.06, 0.04]}
DETAIL = [1.0, 0.2, 0.5, 2.0]
DATA = {'one': [(0, 1)], 'three': [(0, 1), (0, 2), (0, 3)], 'mixed': [(0, 1), (0, 1), (1, 0)]}
BETAS, ETAS, MAXS = [0.1, 0.5, 1.0, 2.0], [0.05, 0.1, 0.2], 100
LET = 'ABCD'

def log_softmax(z):
    z = np.asarray(z, float); m = z.max(); return z - (m + math.log(np.exp(z - m).sum()))

def dpo_loss(logits, ref_logp, pairs, beta):
    """Eq. 7 of the DPO paper, averaged over the pairs: -log sigma(beta*[log pi(w)/ref(w) - log pi(l)/ref(l)])."""
    lp = log_softmax(logits)
    vals = []
    for w, l in pairs:
        u = beta * ((lp[w] - ref_logp[w]) - (lp[l] - ref_logp[l]))
        vals.append(np.logaddexp(0.0, -u))          # -log sigma(u)
    return float(np.mean(vals))

def fd_grad(fun, x, h=1e-6):
    x = np.array(x, float); g = np.zeros_like(x)
    for k in range(len(x)):
        e = np.zeros_like(x); e[k] = h
        g[k] = (fun(x + e) - fun(x - e)) / (2 * h)
    return g

def jacobian_grad(logits, ref_logp, pairs, beta):
    """Paper's gradient: -beta * sigma(r_l - r_w) * [grad log pi(w) - grad log pi(l)], with the full softmax Jacobian."""
    lp = log_softmax(logits); pi = np.exp(lp); n = len(pairs); g = np.zeros(4)
    for w, l in pairs:
        rw, rl = beta * (lp[w] - ref_logp[w]), beta * (lp[l] - ref_logp[l])
        weight = 1.0 / (1.0 + math.exp(-(rl - rw)))
        gw = np.eye(4)[w] - pi; gl = np.eye(4)[l] - pi          # d log pi(y) / d logits
        g += -beta * weight * (gw - gl) / n
    return g

def run(ds, ref, mode, beta, eta, steps=MAXS):
    """Plain gradient descent from the SFT model; step = (eta / beta) * gradient (the page's stated rule)."""
    ref_p = np.array(REFS[ref]); ref_logp = np.log(ref_p); pairs = DATA[ds]
    theta = ref_logp.copy(); w = 0.0; out = []
    for t in range(steps + 1):
        logits = theta if mode == 'sep' else ref_logp + w * np.array(DETAIL)
        lp = log_softmax(logits); pi = np.exp(lp)
        loss = dpo_loss(logits, ref_logp, pairs, beta)
        kl = float(np.sum(pi * (lp - ref_logp)))
        lr = lp - ref_logp
        if mode == 'sep':
            g = fd_grad(lambda th: dpo_loss(th, ref_logp, pairs, beta), theta)
        else:
            g = np.array([fd_grad(lambda v: dpo_loss(ref_logp + v[0] * np.array(DETAIL), ref_logp, pairs, beta), [w])[0]])
        out.append(dict(pi=pi, loss=loss, kl=kl, hAB=lr[0] - lr[1], rhat=beta * lr, logits=logits.copy(), w=w, g=g.copy()))
        if mode == 'sep': theta = theta - (eta / beta) * g
        else: w = w - (eta / beta) * g[0]
    return out

sigma = lambda x: 1.0 / (1.0 + math.exp(-x))
softplus = lambda x: float(np.logaddexp(0.0, x))            # log(1 + e^x) = -log sigma(-x)
trim0 = lambda s: re.sub(r'\.0(?=\s|$)', '', s)            # the kit's number formatter drops a lone ".0"
def pct(p, d=None):
    """A percentage as the page prints it: whole percent, one decimal below 1% or above 99%."""
    v = p * 100
    if d is None: d = 1 if (v < 1 or v > 99) else 0
    if 0 < v < 0.05: return '<0.1%'
    if 99.95 < v < 100: return '>99.9%'
    return trim0(f'{v:.{d}f}') + '%'
M = '−'
fz = lambda v, d: f'{v:.{d}f}'.replace('-0.' + '0' * d, '0.' + '0' * d)   # no "-0.00"
sgn = lambda v, d: (M if v < 0 and float(f'{abs(v):.{d}f}') != 0 else ('+' if float(f'{abs(v):.{d}f}') != 0 else '')) + f'{abs(v):.{d}f}'
num = lambda s: float(s.replace(M, '-').replace('(', '').replace(')', '').replace('%', '').replace('+', ''))
r3 = lambda v: float(f'{v:.3f}')
shown_margin = lambda q: sgn(r3(q['rhat'][0]) - r3(q['rhat'][1]), 3)   # the card shows r̂A − r̂B from the two rewards it lists (R6.6)
def kit_delta(a, b, pts=False, dp=None):
    """The kit's rule for a ▲▼ badge (app.js deltaVs): both shown values rounded to the card's decimals; percentage points for
    percentages; otherwise the relative change, "from 0" / "to 0", or "n times" from 1.995×."""
    if dp is not None: k = 10 ** dp; a = math.floor(a * k + 0.5) / k; b = math.floor(b * k + 0.5) / k
    if a == b: return ''
    if pts:
        d = b - a
        if abs(d) < 0.5 * 10 ** -dp: return ''
        return ('▲ ' if d > 0 else '▼ ') + trim0(f'{abs(d):.{dp}f}') + ' pts vs last'
    if a == 0: return '▲ from 0 vs last' if b > 0 else ''
    if b == 0: return '▼ to 0 vs last'
    q = b / a
    if q >= 1.995: n_ = re.sub(r'\.0$', '', trim0(f'{q:.{0 if q >= 9.95 else 1}f}')); return f'▲ {n_} times last'
    r = (q - 1) * 100; pp = int(math.copysign(math.floor(abs(r) + 1e-9 + 0.5), r))
    if pp == 0: return ''
    return ('▲ +' if pp > 0 else '▼ ' + M) + str(abs(pp)) + '% vs last'
fails, notes = [], []
def check(name, ok, info=''):
    notes.append(('OK   ' if ok else 'FAIL ') + name + (f'  [{info}]' if info else ''))
    if not ok: fails.append(name)

# ---------- 1. the gradient itself: finite differences vs the paper's formula vs "only A and B move" ----------
rng = np.random.default_rng(0)
for trial in range(20):
    ref_logp = np.log(np.array(REFS['b'])); logits = ref_logp + rng.normal(0, 1.0, 4)
    for ds in DATA:
        for beta in BETAS:
            g1 = fd_grad(lambda th: dpo_loss(th, ref_logp, DATA[ds], beta), logits); g2 = jacobian_grad(logits, ref_logp, DATA[ds], beta)
            if not np.allclose(g1, g2, atol=1e-8): check(f'gradient formula {ds} beta {beta}', False, f'{g1} vs {g2}')
check('paper gradient == finite differences (20 random points x 3 datasets x 4 betas)', not any('gradient formula' in f for f in fails))
ref_logp = np.log(np.array(REFS['b'])); logits = ref_logp + np.array([0.3, -0.2, 0.1, 0.05])
g = jacobian_grad(logits, ref_logp, DATA['one'], 0.1)
check('one pair: gradient on the unlabeled answers C and D is exactly 0', abs(g[2]) < 1e-15 and abs(g[3]) < 1e-15, str(g))

# ---------- 2. whole trajectories: page engine vs independent run ----------
configs = [(ds, ref, mode, b, e) for ds in DATA for ref in REFS for mode in ('sep', 'shared') for b in BETAS for e in ETAS]
CHECK_STEPS = [0, 1, 2, 10, 20, 40, 60, 100]
py = {c: run(*c) for c in configs}
def st(ds='one', ref='b', mode='sep', beta=0.1, eta=0.1): return py[(ds, ref, mode, float(beta), eta)]

with sync_playwright() as p:
    br = p.chromium.launch(); pg = br.new_page(viewport={'width': 1440, 'height': 900})
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL + '#dpo'); pg.wait_for_timeout(400)
    page = pg.evaluate("""(cfgs)=>cfgs.map(([ds,ref,mode,beta,eta,steps])=>{const r=window.__dpo.compute({ds,ref,mode,beta,eta,steps});
        return {pi:r.pi,loss:r.loss,kl:r.kl,hAB:r.hAB,rhat:r.rhat,wAB:r.wAB,w:r.w,logits:r.z,g:r.g}})""",
        [[*c, t] for c in configs for t in CHECK_STEPS])
    worst = 0.0; k = 0; bad = []
    for c in configs:
        for t in CHECK_STEPS:
            a, q = page[k], py[c][t]; k += 1
            diffs = [np.max(np.abs(np.array(a['pi']) - q['pi'])), abs(a['loss'] - q['loss']), abs(a['kl'] - q['kl']), abs(a['hAB'] - q['hAB']),
                     np.max(np.abs(np.array(a['rhat']) - q['rhat'])), abs(a['wAB'] - sigma(-c[3] * q['hAB'])), np.max(np.abs(np.array(a['logits']) - q['logits'])),
                     np.max(np.abs(np.array(a['g']) - q['g']))]
            d = max(diffs); worst = max(worst, d)
            if d > 1e-6: bad.append((c, t, d))
    check(f'{len(configs)} configs x {len(CHECK_STEPS)} steps: probabilities, loss, KL, log-ratio gap, implicit rewards, push weights, logits, gradients match', not bad, f'worst abs diff {worst:.2e}; first bad {bad[:2]}')

    # ---------- 3. the specific numbers the page states ----------
    base = st(); r20 = base[20]; r40 = base[40]
    check('SFT start: A 18%, B 52%, C 18%, D 12%', [pct(x) for x in base[0]['pi']] == ['18%', '52%', '18%', '12%'])
    check('loss at step 0 = log 2 = 0.693', abs(base[0]['loss'] - math.log(2)) < 1e-12 and f"{base[0]['loss']:.3f}" == '0.693')
    check('step 3 (20 steps): A 18% -> 48%, B 52% -> 21%', (pct(r20['pi'][0]), pct(r20['pi'][1])) == ('48%', '21%'), f"{r20['pi']}")
    cpk = max(range(21), key=lambda t: base[t]['pi'][2])
    check('step 3: C peaks at 19.7% at step 11; after 11 steps A 34%, B 33%, C 19.7%, D 13.2%', (pct(base[cpk]['pi'][2], 1), cpk) == ('19.7%', 11)
          and [pct(base[11]['pi'][0]), pct(base[11]['pi'][1]), pct(base[11]['pi'][2], 1), pct(base[11]['pi'][3], 1)] == ['34%', '33%', '19.7%', '13.2%'], f'{cpk} {base[cpk]["pi"][2]}')
    b1 = st(beta=1.0)[40]
    check('part 6 (40 steps): beta 0.1 -> A 74%, KL 0.85; beta 1 -> A 49%, KL 0.31', (pct(r40['pi'][0]), f"{r40['kl']:.2f}", pct(b1['pi'][0]), f"{b1['kl']:.2f}") == ('74%', '0.85', '49%', '0.31'))
    w_b1, w_b01 = f"{sigma(-1.0 * b1['hAB']):.3f}", f"{sigma(-0.1 * r40['hAB']):.3f}"
    check(f'part 6: push weights at step 40: beta 1 -> {w_b1}, beta 0.1 -> {w_b01} (3 decimals)', (w_b1, w_b01) == ('0.122', '0.410'))
    f60 = st()[60]; w60 = sigma(-0.1 * f60['hAB'])
    check('part 5: beta 0.1 at step 60: push weight 0.373, 75% of the first step (from the shown 0.373 and from the unrounded weight)',
          (f'{w60:.3f}', round(float(f'{w60:.3f}') / 0.5 * 100), round(w60 / 0.5 * 100)) == ('0.373', 75, 75), f'{w60}')
    # the page's "x% as far" uses the shown 3-decimal weight; prove it always equals the unrounded percentage
    ws = np.linspace(0.0005, 0.5, 200001)
    check('"x% as far" from a 3-decimal weight equals the unrounded percentage for every weight in (0, 0.5]',
          all(round(float(f'{w:.3f}') / 0.5 * 100) == round(w / 0.5 * 100) for w in ws if abs(w / 0.5 * 100 % 1 - 0.5) > 1e-9))
    sh = st(mode='shared'); pk = max(range(61), key=lambda t: sh[t]['pi'][0])
    dz = sh[60]['logits'] - sh[0]['logits']
    check('step 7 shared weight, 60 steps: A 12% (peak 21.4% at step 21), D 12% -> 79%, loss 0.693 -> 0.606, separate 88%, logit changes A +2.29, D +4.58',
          (pct(sh[60]['pi'][0]), pct(sh[pk]['pi'][0], 1), pk, pct(sh[60]['pi'][3]), f"{sh[60]['loss']:.3f}", pct(base[60]['pi'][0]), f'{dz[0]:+.2f}', f'{dz[3]:+.2f}') == ('12%', '21.4%', 21, '79%', '0.606', '88%', '+2.29', '+4.58'),
          f"{sh[60]['pi']} peak {pk} dz {dz}")
    w8a, w8b = sigma(-0.1 * base[60]['hAB']), sigma(-2.0 * st(beta=2.0)[60]['hAB'])
    check('step 8: push weight after 60 steps: beta 0.1 -> 0.373, beta 2 -> 0.043', (f'{w8a:.3f}', f'{w8b:.3f}') == ('0.373', '0.043'), f'{w8a} {w8b}')
    ch = {b: st(beta=b)[60] for b in BETAS}
    tbl = {b: (pct(ch[b]['pi'][0]), f"{ch[b]['kl']:.2f}") for b in BETAS}
    check('challenge (60 steps): beta 0.1 88%/1.23, 0.5 71%/0.76, 1 56%/0.44, 2 42%/0.20', tbl == {0.1: ('88%', '1.23'), 0.5: ('71%', '0.76'), 1.0: ('56%', '0.44'), 2.0: ('42%', '0.20')}, str(tbl))
    ok = [b for b in BETAS if ch[b]['pi'][0] >= 0.5 and ch[b]['kl'] <= 0.8]
    check('challenge: options that meet the goal are beta 0.5 and 1; closest to SFT is beta 1', ok == [0.5, 1.0] and min(ok, key=lambda b: ch[b]['kl']) == 1.0, str(ok))
    check('part 6 baseline: the initial setup (beta 0.1, 40 steps) ends at KL 0.85 nats', f"{r40['kl']:.2f}" == '0.85')
    mixed = st(ds='mixed', beta=1.0)[100]
    check('mixed labels, beta 1, 100 steps: reward margin 0.686 unrounded, shown as 0.424 − (−0.263) = 0.687, close to log 2 = 0.693', abs(1.0 * mixed['hAB'] - math.log(2)) < 0.01 and f"{mixed['hAB']:.3f}" == '0.686' and shown_margin(mixed) == '+0.687', f"{mixed['hAB']} {shown_margin(mixed)}")
    u = np.linspace(0.3, 1.1, 80001); lossu = (2 * np.logaddexp(0, -u) + np.logaddexp(0, u)) / 3
    check('mixed labels: loss minimum sits at reward margin = log 2', abs(u[np.argmin(lossu)] - math.log(2)) < 2e-5)
    three = st(ds='three')[40]; one40 = base[40]
    check('three pairs, 40 steps: A 73%, B 17% (one pair: A 74%, B 6%); A logit rises about as fast, B drops a third as fast',
          (pct(three['pi'][0]), pct(three['pi'][1]), pct(one40['pi'][0]), pct(one40['pi'][1])) == ('73%', '17%', '74%', '6%')
          and abs((three['logits'][0] - math.log(0.18)) / (one40['logits'][0] - math.log(0.18)) - 1) < 0.15, f"{three['pi']} dA3 {three['logits'][0] - math.log(0.18):.3f} dA1 {one40['logits'][0] - math.log(0.18):.3f}")
    stub = st(ref='s')
    first = lambda tr: next(t for t in range(MAXS + 1) if tr[t]['pi'][0] > tr[t]['pi'][1])
    check('stubborn SFT model, 40 steps: A 5% -> 56%, B 85% -> 25%; A passes B at step 31 (default start: step 11)',
          (pct(stub[40]['pi'][0]), pct(stub[40]['pi'][1]), first(stub), first(base)) == ('56%', '25%', 31, 11))
    big = st(beta=2.0)[100]; wbig = sigma(-2.0 * big['hAB'])
    check('large beta (2), 100 steps: A 47%, KL 0.27, push weight 0.026 = 5% of the first step',
          (pct(big['pi'][0]), f"{big['kl']:.2f}", f'{wbig:.3f}', round(float(f'{wbig:.3f}') / 0.5 * 100)) == ('47%', '0.27', '0.026', 5), f'{wbig}')
    check('teaser at 20 steps: A 18% -> 48%, B 52% -> 21%', (pct(base[20]['pi'][0]), pct(base[20]['pi'][1])) == ('48%', '21%'))

    # ---------- reading the typeset Show the math (H2, the owner's review of 2026-10-09) ----------
    # every block has a data-k; every aligned display row a data-r. A row reads as its MathML tokens; a braced signed number (−0.93) is one token.
    TOK_JS = r"""(el)=>{const t=[]; const walk=(n)=>{if(n.nodeType===3){const s=n.textContent.trim(); if(s) t.push(s); return;}
        if(n.localName==='mrow'&&n.children.length===2&&n.children[0].localName==='mo'&&/^[−+]$/.test(n.children[0].textContent)&&n.children[1].localName==='mn'){t.push(n.children[0].textContent+n.children[1].textContent); return;}
        for(const c of n.childNodes) walk(c);}; walk(el); return t.join(' ');}"""
    EQS_JS = r"""(sel)=>{const tok=""" + "TOK" + r"""; return [...document.querySelectorAll(sel+' .dpo-eq')].map(eq=>{const rows={}; for(const c of eq.querySelectorAll(':scope > [data-r]')){const r=c.dataset.r; rows[r]=(rows[r]?rows[r]+' ':'')+tok(c);} return Object.keys(rows).sort((a,b)=>a-b).map(k=>rows[k]);});}"""
    EQS_JS = EQS_JS.replace('TOK', TOK_JS)
    BLOCKS_JS = r"""()=>{const tok=""" + TOK_JS + r"""; const out={}; for(const b of document.querySelectorAll('#dpo-mathbody > .dpo-mb')){const eq=b.querySelector('.dpo-eq'); if(!eq) continue; const rows={}; for(const c of eq.querySelectorAll(':scope > [data-r]')){const r=c.dataset.r; rows[r]=(rows[r]?rows[r]+' ':'')+tok(c);} out[b.dataset.k]=Object.keys(rows).sort((a,b)=>a-b).map(k=>rows[k]);} return out;}"""
    NUMS = lambda s_: re.findall(r'(?<![\w.])[−+]?\d+(?:\.\d+)?', s_)
    blocks = lambda: pg.evaluate(BLOCKS_JS)

    # ---------- 4. the numbers as rendered on the page ----------
    PART = {k: i for i, k in enumerate(pg.evaluate("()=>window.__dpo.consts.STEP_IDS"))}
    pg.evaluate(f"()=>document.querySelector('#dpo-rail [data-go=\"{PART['beta']}\"]').click()"); pg.wait_for_timeout(200)
    pg.click('#dpo-ic-beta button[data-v="1"]'); pg.wait_for_timeout(250)
    obs = pg.inner_text('#dpo-rail [data-obs]')
    # R6.20: the observation keeps one key number, the KL against the initial setup's (0.31 against 0.85); π(A) and the push weights left
    # the sentence, so the push weight at step 40 is read from Show the math (β = 1: 0.122) and π(A) from its metric card (49%)
    pw40 = blocks().get('pw', [])
    mpw = (NUMS(pw40[1])[0], NUMS(pw40[2])[0]) if len(pw40) == 3 and NUMS(pw40[1]) and NUMS(pw40[2]) else None
    check('rendered part 6 text: “Its KL from the SFT model is 0.31 nats instead of 0.85” (β 1 against 0.1, 40 steps), π(A) in words (49% on its card), and β = 1’s push weight 0.122 in Show the math, from the shown reward margin',
          'Its KL from the SFT model is 0.31 nats instead of 0.85, and π(A) rises less.' in obs and '49%' not in obs and '0.122' not in obs
          and pg.inner_text('#dpo-metrics [data-k="pA"] .v') == pct(b1['pi'][0]) == '49%' and bool(mpw) and mpw[1] == w_b1 == '0.122' and f'{sigma(num(mpw[0])):.3f}' == '0.122',
          obs[:300].replace('\n', ' | ') + ' | ' + (str(pw40) if mpw else 'no push-weight block'))
    # a KL above 0 but under 0.005 shows 4 decimals (0.0003, not 0.00), as the page's klS does (site review r13 #10)
    kl_shown = lambda k: fz(k, 4) if 0 < k < 0.005 else fz(k, 2)
    card = lambda q, beta: [pct(q['pi'][0]), pct(q['pi'][1]), shown_margin(q), f"{q['loss']:.3f}", kl_shown(q['kl'])]
    mets = pg.evaluate("()=>[...document.querySelectorAll('#dpo-metrics .metric')].map(m=>m.querySelector('.v').textContent)")
    check('rendered metric cards at beta 1, 40 steps equal the unrounded values', mets == card(b1, 1.0), f'{mets} vs {card(b1, 1.0)}')
    subs = pg.evaluate("()=>document.querySelector('#dpo-metrics [data-k=\"margin\"] .sub').textContent")
    check('reward-margin card: the two implicit rewards in words, and the margin is their difference as shown (D2)', subs == f"A’s implicit reward {sgn(b1['rhat'][0], 3)}, B’s {sgn(b1['rhat'][1], 3)}" and abs(num(mets[2]) - (num(sgn(b1['rhat'][0], 3)) - num(sgn(b1['rhat'][1], 3)))) < 1e-9, subs + ' | ' + mets[2])
    bars = pg.evaluate("()=>[...document.querySelectorAll('#dpo-bars .dpo-val')].map(e=>e.textContent)")
    check('rendered bar values at beta 1, 40 steps (now, then before training)', bars == [f"{pct(b1['pi'][i])} (was {pct(REFS['b'][i])})" for i in range(4)], str(bars))
    # metric badges follow the kit's rule, computed from the values shown before and after the learner's click
    dec = lambda v: len(v.split('.')[1].rstrip('%')) if '.' in v else 0
    before = pg.evaluate("()=>[...document.querySelectorAll('#dpo-metrics .metric .v')].map(e=>e.textContent)")
    pg.click('#dpo-ic-beta button[data-v="2"]'); pg.wait_for_timeout(250)
    after = pg.evaluate("()=>[...document.querySelectorAll('#dpo-metrics .metric')].map(m=>[m.querySelector('.v').textContent,m.querySelector('.d').textContent])")
    dl = [(dtxt, kit_delta(num(bv), num(av), pts=k2 < 2, dp=max(dec(bv), dec(av)))) for k2, (bv, (av, dtxt)) in enumerate(zip(before, after))]
    core = lambda t: (t[:1], re.findall(r'\d+(?:\.\d+)?', t.replace(',', '')))   # the arrow and the number; the kit's wording may change
    check('metric-card badges follow the kit rule from the shown values (β 1 → 2 at 40 steps): the arrow and the number, points for π, relative change otherwise', all(a and core(a) == core(b) for a, b in dl), str(dl))
    pg.click('#dpo-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(250)
    pg.click('#dpo-rail [data-act="submit"]'); pg.wait_for_timeout(200); pg.click('#dpo-rail [data-act="reveal"]'); pg.wait_for_timeout(200)
    rows = pg.evaluate("()=>[...document.querySelectorAll('#dpo-ch-res .cmp-row')].map(r=>r.querySelector('.cmp-lab span').textContent+' = '+r.querySelector('.cmp-val').textContent)")
    # brief (the owner's review of batch 8): the baseline, the initial β (0.1, also the pick here) and the β values that meet both conditions
    keep = [b for b in BETAS if b == 0.1 or (ch[b]['pi'][0] >= 0.5 and ch[b]['kl'] <= 0.8)]
    exp_rows = ['The SFT model before training = 18%'] + [f"β = {b:g}, KL {ch[b]['kl']:.2f} nats = {pct(ch[b]['pi'][0])}" for b in keep]
    check('rendered challenge chart rows: the baseline, the initial β and the passing ones (0.1, 0.5, 1)', rows == exp_rows and keep == [0.1, 0.5, 1], str(rows))
    note = pg.evaluate("()=>(document.querySelector('#dpo-ch-res .cmp-note')||{}).textContent||''")
    check(f'the note under the chart counts the {len(BETAS) - len(keep)} β value left out', note == 'The other β value misses at least one condition and is not shown.' and len(BETAS) - len(keep) == 1, note)
    why = pg.inner_text('#dpo-ch-res')
    # R6.20: the push weights at step 60 are said in words; the words must match the weights (golden: 0.373 and 0.043 above), as shares of
    # the first step's 0.5: at least half left reads "still has most of its strength", under a fifth "has almost faded"
    check('challenge “Why” says the push weights at step 60 in words that match them (0.373 = 75% of the first push, 0.043 = 9%)',
          'By step 60 the push still has most of its strength with β = 0.1 but has almost faded with β = 2.' in why and '0.373' not in why and '0.043' not in why
          and float(f'{w8a:.3f}') / 0.5 >= 0.5 and float(f'{w8b:.3f}') / 0.5 < 0.2, why[-600:].replace('\n', ' | '))

    # ---------- 4b. formula lines reproduce from the numbers they show ----------
    def cards():
        return pg.evaluate("()=>[...document.querySelectorAll('#dpo-metrics .metric .v')].map(e=>e.textContent)")
    NUM = re.compile(r'(?<![\w.])[−+]?\d+(?:\.\d+)?')
    def goto_state(tok):
        """Open a setup through the page's own hash link and wait until the lab has mounted it."""
        pg.evaluate("(t)=>{location.hash='dpo~'+t}", tok)
        pg.wait_for_function("(t)=>{const s=window.__dpo.state(); return [s.ds,s.ref,s.logits,s.beta,s.eta,s.steps].join('_')===t && !!document.querySelector('#dpo-mathbody > .dpo-mb')}", arg=tok, timeout=5000)
    def parse_math(B):
        """The numbers of each block, row by row (row 0 of a block is the formula in symbols)."""
        o = {'pi': [], 'ref': [], 'l': []}
        for s_ in B['lr'][1:5]:
            v = NUMS(s_); o['pi'].append((float(v[0]), len(v[0].split('.')[1]) if '.' in v[0] else 0)); o['ref'].append(float(v[1])); o['l'].append(num(v[2]))
        o['r'] = {re.match(r'r \^ ([ABCD])', s_).group(1): [num(x) for x in NUMS(s_)] for s_ in B['r'][1:]}
        o['m_rows'] = [([m_.group(1), m_.group(2)] if (m_ := re.search(r'([ABCD]) ≻ ([ABCD])', s_)) else None, [num(x) for x in NUMS(s_)]) for s_ in B['m']]
        o['loss_rows'] = [([m_.group(1), m_.group(2)] if (m_ := re.search(r'([ABCD]) ≻ ([ABCD])', s_)) else None, [num(x) for x in NUMS(s_)]) for s_ in B['loss']]
        o['pw'] = [num(NUMS(B['pw'][1])[0]), num(NUMS(B['pw'][2])[0])]
        o['next'] = [[num(x) for x in NUMS(s_)] for s_ in B['next']]
        o['kl_rows'] = [[num(x) for x in NUMS(s_)] for s_ in B['kl'][1:5]]; o['kl'] = num(NUMS(B['kl'][5])[0])
        return o
    def verify(o, cfg, t, q):
        """Recompute every row from the numbers printed in it; compare finals with the unrounded values q."""
        ds, ref, mode, beta, eta = cfg; pairs = DATA[ds]; errs_ = []
        E = lambda name, ok: (None if ok else errs_.append(name))
        for i, (v, d) in enumerate(o['pi']):
            sig = d + 1 + math.floor(math.log10(v)); E(f'pi {LET[i]} has 6 significant digits', sig == 6)
            E(f'pi {LET[i]} is the rounded probability', abs(v - q['pi'][i] * 100) <= 0.5 * 10 ** -d + 1e-12)
        E('SFT percentages', [round(x * 100, 6) for x in REFS[ref]] == o['ref'])
        for i in range(4): E(f'log-ratio {LET[i]} = ln(shown pi / shown ref)', abs(math.log(o['pi'][i][0] / o['ref'][i]) - o['l'][i]) <= 5e-6 + 1e-12)
        inp = sorted({k for p_ in pairs for k in p_})
        E('implicit rewards for exactly the answers in a pair', sorted(LET.index(k) for k in o['r']) == inp)
        for k, (bb, ll, rr) in o['r'].items():
            E(f'r_{k} = beta x shown log-ratio', bb == beta and ll == o['l'][LET.index(k)] and abs(beta * ll - rr) < 1e-9)
        rA, rB = o['r']['A'][2], o['r']['B'][2]
        if len(pairs) == 1:
            (_, v1), (_, v2) = o['m_rows'][1], o['m_rows'][2]; m = v2[0]
            E('reward margin row: the shown implicit rewards', v1 == [rA, rB]); E('reward margin = r_A - r_B', abs(rA - rB - m) < 1e-9)
            (_, l1), (_, l2) = o['loss_rows'][1], o['loss_rows'][2]
            E('loss = -log sigma(shown margin)', l1 == [m] and abs(softplus(-m) - l2[0]) <= 5e-4 + 1e-12); loss = l2[0]
        else:
            per = {}
            for pr, v in o['m_rows']:
                a_, c_ = pr; per[(a_, c_)] = v[2]
                E(f'pair {a_}>{c_} margin = shown r_{a_} - r_{c_}', v[:2] == [o['r'][a_][2], o['r'][c_][2]] and abs(v[0] - v[1] - v[2]) < 1e-9)
            m = per[('A', 'B')]
            tot, n = 0.0, 0
            for pr, v in o['loss_rows'][:-1]:
                a_, c_ = pr; k_ = sum(1 for p_ in pairs if (LET[p_[0]], LET[p_[1]]) == (a_, c_))
                E(f'pair {a_}>{c_} loss = -log sigma(shown margin)', v[0] == per[(a_, c_)] and abs(softplus(-v[0]) - v[1]) <= 5e-5 + 1e-12)
                tot += k_ * softplus(-v[0]); n += k_
            last = o['loss_rows'][-1][1]; loss = last[-1]
            E('pair multiplicities add up', n == len(pairs) and last[-2] == len(pairs)); E('loss = mean of -log sigma(pair margins)', abs(tot / n - loss) <= 5e-4 + 1e-12)
        mm, pw = o['pw']; E('push weight = sigma(-shown margin)', abs(mm + m) < 1e-12 and abs(sigma(-m) - pw) <= 5e-4 + 1e-12)
        nx = o['next']
        if mode == 'shared':
            v = nx[1]; res = nx[2][0]
            E('next w = shown w - (eta/beta) x shown gradient', v[1] == eta and v[2] == beta and abs(v[0] - q['w']) <= 5e-4 + 1e-12 and abs(v[3] - q['g'][0]) <= 5e-5 + 1e-12 and abs(v[0] - (eta / beta) * v[3] - res) <= 5e-5 + 1e-12)
        elif len(pairs) == 1:
            v = nx[1]; res = nx[2][0]; E('next step = eta x shown push weight', v[0] == eta and v[1] == pw and abs(eta * pw - res) < 1e-12)
        else:
            ch_ = [r_[0] for r_ in nx[1:5]]; E('logit changes = -(eta/beta) x gradient', all(abs(-(eta / beta) * q['g'][k] - ch_[k]) <= 5e-5 + 1e-12 for k in range(4)))
        E('KL rows: shown pi (as a fraction) and shown log-ratio', all(abs(o['kl_rows'][i][0] - o['pi'][i][0] / 100) < 1e-12 and o['kl_rows'][i][1] == o['l'][i] for i in range(4)))
        E('KL = sum of shown pi x shown log-ratio (as on the card)', abs(sum(o['kl_rows'][i][0] * o['kl_rows'][i][1] for i in range(4)) - o['kl']) <= 5e-3 + 1e-12)
        ex = {'margin': beta * q['hAB'], 'loss': q['loss'], 'pw': sigma(-beta * q['hAB']), 'kl': q['kl']}
        E('chain margin within 3e-5 of the unrounded margin', abs(m - ex['margin']) < 3e-5)
        E('chain KL within rounding of the unrounded KL', abs(o['kl'] - ex['kl']) < 5e-3 + 2e-4)
        bound = []
        if f"{m:.3f}" != f"{ex['margin']:.3f}": bound.append('margin')
        if f"{loss:.3f}" != f"{ex['loss']:.3f}": bound.append('loss')
        if f"{pw:.3f}" != f"{ex['pw']:.3f}": bound.append('push weight')
        if fz(o['kl'], 2) != fz(ex['kl'], 2): bound.append('KL')
        return errs_, bound
    STATES = [(ds, ref, mode, b, 0.1, t) for ds in DATA for ref in REFS for mode in ('sep', 'shared') for b in BETAS for t in (0, 1, 11, 40, 100)]
    STATES += [(ds, 'b', mode, b, e, 60) for ds in DATA for mode in ('sep', 'shared') for b in BETAS for e in (0.05, 0.2)]
    bad, boundary, cards_bad, zh_bad = [], [], [], []
    for (ds, ref, mode, b, e, t) in STATES:
        goto_state(f'{ds}_{ref}_{mode}_{b:g}_{e:g}_{t}')
        cfg = (ds, ref, mode, float(b), e); q = py[cfg][t]
        try:
            o = parse_math(blocks()); er, bd = verify(o, cfg, t, q)
        except Exception as ex_:
            er, bd = [f'parse: {ex_!r}'], []
        if er: bad.append(((ds, ref, mode, b, e, t), er[:3]))
        if bd: boundary.append(((ds, ref, mode, b, e, t), bd))
        cv = cards(); exp_c = card(q, float(b))
        if cv != exp_c or abs(num(cv[2]) - float(b) * q['hAB']) > 1.5e-3: cards_bad.append(((ds, ref, mode, b, e, t), cv, exp_c))
    check(f'{len(STATES)} setups: every row of “Show the math” reproduces from the numbers it shows (π 6 significant digits, log-ratios 5 decimals, implicit rewards, reward margin, loss, push weight, next step, KL)', not bad, str(bad[:3]))
    check(f'{len(STATES)} setups: metric cards equal the unrounded values (π, loss, KL); the reward margin is the difference of the two rewards shown, within 0.001 of the unrounded one', not cards_bad, str(cards_bad[:3]))
    check(f'{len(STATES)} setups: math finals round to the same value as the unrounded numbers, except {len(boundary)} rounding-boundary cases (one unit in the last digit)', len(boundary) <= len(STATES) // 50, str(boundary[:6]))
    notes.append(f'     boundary cases: {boundary}')
    # the same numbers in Chinese
    pg.click('.langsw button[data-lang="zh"]'); pg.wait_for_timeout(150)
    for (ds, ref, mode, b, e, t) in STATES[::7]:
        tok = f'{ds}_{ref}_{mode}_{b:g}_{e:g}_{t}'
        goto_state(tok); zh = {k: [NUMS(x) for x in v] for k, v in blocks().items()}
        pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(60); en = {k: [NUMS(x) for x in v] for k, v in blocks().items()}
        pg.click('.langsw button[data-lang="zh"]'); pg.wait_for_timeout(60)
        if zh != en: zh_bad.append((tok, zh, en))
    check(f'{len(STATES[::7])} setups: the Chinese math panel shows the same numbers in the same order', not zh_bad, str(zh_bad[:1]))
    pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(100)

    # part 4 keeps the idea in its paragraphs; the formula box that repeated Show the math and the metric cards is gone
    pg.evaluate("()=>{location.hash='dpo'}"); pg.wait_for_timeout(200)
    pg.click('#dpo-app .modes button[data-mode="guided"]'); pg.wait_for_timeout(150)
    pg.evaluate(f"()=>document.querySelector('#dpo-rail [data-go=\"{PART['loss']}\"]').click()"); pg.wait_for_timeout(200)
    check('part 4 has no “Where these numbers come from” box (the numbers are in Show the math and on the metric cards)', not pg.evaluate("()=>{const o=document.querySelector('#dpo-rail [data-obs]'); return !!o&&!o.hidden}") and 'Where these numbers come from' not in pg.inner_text('#dpo-rail'))
    # part 5: push weight readout and "x% as far" (β 0.1)
    pg.evaluate(f"()=>document.querySelector('#dpo-rail [data-go=\"{PART['fade']}\"]').click()"); pg.wait_for_timeout(200)
    s6bad = []
    for t in (60, 1, 20, 80, 100):          # the observation appears once 60 steps were reached
        pg.evaluate(f"()=>{{const i=document.getElementById('dpo-ic-steps-in'); i.value={t}; i.dispatchEvent(new Event('input',{{bubbles:true}}))}}"); pg.wait_for_timeout(60)
        o6 = pg.inner_text('#dpo-rail [data-obs]'); live = pg.inner_text('#dpo-pw-live'); w = sigma(-0.1 * st()[t]['hAB'])
        m = re.search(r'By step (\d+), each step moves the logits only (\d+)% as far as the first step did\.', o6); ml = re.search(r'push weight is ([\d.]+) \(0\.500 at step 0\)', live)
        if not (m and ml and int(m.group(1)) == t and ml.group(1) == f'{w:.3f}' and int(m.group(2)) == round(float(ml.group(1)) / 0.5 * 100) == round(w / 0.5 * 100) and f'At step {t} the push weight is {w:.3f} (0.500 at step 0).' == live and not any('push weight' in ln for ln in o6.split('\n') if 'By step' in ln)):
            s6bad.append((t, o6[:140], live))
    check('part 5: live push weight and “x% as far” reproduce from the shown weight and match the unrounded one', not s6bad, str(s6bad[:2]))

    # drill: the page's generated question and answer, recomputed from the stated inputs
    for seed in range(1, 40):
        d = pg.evaluate("(s)=>window.__dpo.makeDrill(s)", seed)
        lw, ll = d['piW'] - d['refW'], d['piL'] - d['refL']; m = d['beta'] * (lw - ll)
        if not (abs(d['loss'] - np.logaddexp(0, -m)) < 1e-12 and abs(d['wt'] - sigma(-m)) < 1e-12 and abs(d['coef'] - round(d['beta'] * round(sigma(-m), 3), 4)) < 1e-12 and abs(m) > 1e-9):
            check(f'drill seed {seed}', False, json.dumps(d))
    check('interview drill answers (39 seeds): loss, push weight and coefficient recomputed from the stated log-probs', not any(f.startswith('drill') for f in fails))
    dbad = []
    for k2 in range(6):
        pg.click('#dpo-app [data-act="drill-reveal"]'); pg.wait_for_timeout(60)
        qd = pg.inner_text('#dpo-drill-q'); ad = pg.inner_text('#dpo-drill-a'); eqs = pg.evaluate(EQS_JS, '#dpo-drill-a')
        pw_, pl_, rw_, rl_ = [num(x) for x in re.findall(r'−?\d+(?:\.\d+)?', re.sub(r'β = [\d.]+', '', qd))[:4]]
        beta_d = float(re.search(r'β = (\d+(?:\.\d+)?)', qd).group(1))
        lw, ll = round(pw_ - rw_, 1), round(pl_ - rl_, 1); m = round(beta_d * (lw - ll), 4); wt = sigma(-m)
        # the typeset answer (H2): [log-ratios], [implicit rewards, reward margin], [loss], [push weight, coefficient]
        try:
            n_ = [[[num(x) for x in NUMS(r_)] for r_ in e_] for e_ in eqs]
            ok = (len(n_) == 4 and n_[0][0] == [pw_, rw_, lw] and n_[0][1] == [pl_, rl_, ll]
                  and n_[1][0][0] == beta_d and n_[1][0][1] == lw and n_[1][1][1] == ll and abs(n_[1][2][0] - n_[1][2][1] - n_[1][2][2]) < 1e-9 and abs(n_[1][2][2] - m) < 1e-9
                  and n_[2][0] == [m, float(f'{softplus(-m):.3f}')] and n_[3][0] == [-m, float(f'{wt:.3f}')] and n_[3][1] == [float(f'{wt:.3f}'), round(beta_d * round(wt, 3), 4)]
                  and f'loss = {softplus(-m):.3f}' in ad and 'r̂_w' not in ad)
        except Exception as ex_:
            ok = False; n_ = repr(ex_)
        if not ok: dbad.append((qd, n_))
        pg.click('#dpo-app [data-act="drill-new"]'); pg.wait_for_timeout(60)
    check('rendered drill answers (6 questions) reproduce from the log-probs in the question', not dbad, str(dbad[:1]))
    code = pg.inner_text('#dpo-code')
    br.close()
check('no page errors while testing', not errs, str(errs))

# ---------- 5. the minimal implementation on the page runs and matches the formula ----------
class Tn:
    """A tiny numpy-backed stand-in for torch.Tensor with only the methods the snippet uses."""
    def __init__(self, a): self.a = np.asarray(a, float)
    def _w(self, x): return x.a if isinstance(x, Tn) else x
    def __sub__(self, o): return Tn(self.a - self._w(o))
    def __rsub__(self, o): return Tn(self._w(o) - self.a)
    def __add__(self, o): return Tn(self.a + self._w(o))
    __radd__ = __add__
    def __mul__(self, o): return Tn(self.a * self._w(o))
    __rmul__ = __mul__
    def __neg__(self): return Tn(-self.a)
    def detach(self): return Tn(self.a.copy())
    def mean(self): return Tn(self.a.mean())
    def sum(self, dim): return Tn(self.a.sum(axis=dim))
    def log_softmax(self, dim): z = self.a - self.a.max(axis=dim, keepdims=True); return Tn(z - np.log(np.exp(z).sum(axis=dim, keepdims=True)))
    def gather(self, dim, idx): return Tn(np.take_along_axis(self.a, idx.a.astype(int), axis=dim))
    def unsqueeze(self, dim): return Tn(np.expand_dims(self.a, dim))
    def squeeze(self, dim): return Tn(np.squeeze(self.a, axis=dim))
torch = types.ModuleType('torch'); nn = types.ModuleType('torch.nn'); F = types.ModuleType('torch.nn.functional')
F.logsigmoid = lambda x: Tn(-np.logaddexp(0, -x.a)); nn.functional = F; torch.nn = nn
sys.modules.update({'torch': torch, 'torch.nn': nn, 'torch.nn.functional': F})
ns = {}; exec(code, ns)
check('minimal implementation fits in 20 lines', len(code.strip().splitlines()) <= 20, f'{len(code.strip().splitlines())} lines')
pw, pl, rw, rl = (rng.normal(-20, 5, 8) for _ in range(4))
loss, a, b = ns['dpo_loss'](Tn(pw), Tn(pl), Tn(rw), Tn(rl), beta=0.1)
exp_loss = np.mean(np.logaddexp(0, -0.1 * ((pw - rw) - (pl - rl))))
check('minimal implementation: dpo_loss matches -mean log sigma(beta * log-ratio gap)', abs(loss.a - exp_loss) < 1e-12 and np.allclose(a.a, 0.1 * (pw - rw)) and np.allclose(b.a, 0.1 * (pl - rl)))
logits = rng.normal(0, 1, (2, 5, 7)); toks = rng.integers(0, 7, (2, 5)); mask = np.array([[0, 0, 1, 1, 1], [0, 1, 1, 1, 0]], float)
lp = ns['seq_logp'](Tn(logits), Tn(toks), Tn(mask)).a
ref_lp = [sum(float(log_softmax(logits[i, t])[toks[i, t]]) for t in range(5) if mask[i, t]) for i in range(2)]
check('minimal implementation: seq_logp sums the masked token log-probs', np.allclose(lp, ref_lp))

print('\n'.join(notes))
print('FAILED:', fails if fails else 'none')
sys.exit(1 if fails else 0)
