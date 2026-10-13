"""Numbers test for the sft lab (CONTRACT §6): every number the page shows, recomputed independently in Python.

The toy is rebuilt here in matrix form, not ported line by line: each position becomes a one-hot feature row over (how far back,
which token) for the 8 tokens before it, logits = X·W + b, and one full-batch step is W ← W − η·Xᵀ(P − Y)/N. Same vocabulary order,
same data, same zero start, no random numbers (the page uses none), so the runs must agree with the page's (window.__sft) to
floating-point accuracy: generations token by token with their probabilities, the loss on the answers, the counted positions, the
bridge to DPO (logits, probabilities, log-probabilities, their sum, the reference), the token counts of every example, the 16
combinations of the challenge and the fewest changes, the drill (mulberry32 reimplemented), the exit answers. Then it reads the
numbers the page prints (part 3's share and factor, part 4's share, the bridge's table, the challenge's chart, the teaser's notes).
Run: python3 tests/sft_numbers.py"""
import math, pathlib, subprocess, sys
import numpy as np
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/sft.html', '--only', 'sft'], check=True, cwd=P, stdout=subprocess.DEVNULL)
URL = 'file://' + str(P / 'out' / 'sft.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


# ---------------- the toy, from its stated assumptions ----------------
VOC = ['bot', 'sh', 'eh', 'eot', 'pad', 'user', 'asst', 'nl', 'what', 'qm', 'dot', 'is', 'kv', 'batch', 'token', 'gpu', 'softmax',
       'logit', 'd_kv', 'd_batch', 'd_token', 'd_gpu', 'd_softmax', 'd_logit', 'U', 'As']
V = len(VOC); IX = {k: i for i, k in enumerate(VOC)}
K, ETA, PRE, MAXG = 8, 1.0, 300, 16
TERMS = ['kv', 'batch', 'token', 'gpu', 'softmax', 'logit']
ORDERS = [[0, 1, 2, 3, 4, 5], [3, 0, 5, 2, 1, 4], [1, 4, 0, 3, 5, 2], [5, 2, 4, 1, 0, 3]]


def qa(t):
    return ['what', t, 'qm'], [t, 'is', 'd_' + t, 'dot']


def conv(t):
    q, a = qa(t)
    return ['bot', 'sh', 'user', 'eh', 'nl'] + q + ['eot', 'sh', 'asst', 'eh', 'nl'], a + ['eot']


PAGES = [sum((sum(qa(TERMS[i]), []) for i in o), []) for o in ORDERS]
CONVS = [conv(t) for t in TERMS]


def feats(seq, t):
    """one-hot row over (j back, token) for the K tokens before position t"""
    x = np.zeros(K * V)
    for j in range(1, K + 1):
        if t - j >= 0:
            x[(j - 1) * V + IX[seq[t - j]]] = 1.0
    return x


def dataset(seqs_positions):
    X = np.array([feats(s, t) for s, t in seqs_positions]); y = np.array([IX[s[t]] for s, t in seqs_positions])
    return X, y


def softmax_rows(Z):
    Z = Z - Z.max(axis=1, keepdims=True); E = np.exp(Z); return E / E.sum(axis=1, keepdims=True)


def fit(W, b, X, y, steps):
    Y = np.zeros((len(y), V)); Y[np.arange(len(y)), y] = 1.0
    for _ in range(steps):
        G = softmax_rows(X @ W + b) - Y
        W = W - ETA * (X.T @ G) / len(y); b = b - ETA * G.sum(axis=0) / len(y)
    return W, b


def positions(count, mk):
    out = []
    for p, r in CONVS:
        s = p + r
        for t in range(1, len(s)):
            if count != 'all' and t < len(p): continue
            if s[t] == 'eot' and (mk == 'id' or (mk == 'ans' and t >= len(p))): continue
            out.append((s, t))
    return out


W0, b0 = fit(np.zeros((K * V, V)), np.zeros(V), *dataset([(pg, t) for pg in PAGES for t in range(1, len(pg))]), PRE)
RUNS = {}


def run(count, mk, steps):
    key = (count, mk, steps)
    if key not in RUNS:
        RUNS[key] = fit(W0.copy(), b0.copy(), *dataset(positions(count, mk)), steps)
    return RUNS[key]


def probs(Wb, seq, t):
    W, b = Wb; return softmax_rows((feats(seq, t) @ W + b)[None, :])[0]


def generate(Wb, prompt):
    seq, out = list(prompt), []
    for _ in range(MAXG):
        pr = probs(Wb, seq, len(seq)); k = int(np.argmax(pr))   # first maximum wins, as on the page
        out.append((VOC[k], pr[k])); seq.append(VOC[k])
        if VOC[k] == 'eot': break
    return out


def resp_loss(Wb):
    ls = [-math.log(probs(Wb, p + r, t)[IX[(p + r)[t]]]) for p, r in CONVS for t in range(len(p), len(p + r))]
    return sum(ls) / len(ls)


def mk_of(s):
    return 'id' if s['pad'] == 'eot' else ('ans' if s['eotl'] == 'no' else '')


P_TMPL, P_PLAIN, P_DIFF = conv('kv')[0], ['what', 'kv', 'qm'], ['U', 'what', 'kv', 'qm', 'As']
DEF = dict(ex='toy', count='resp', eotl='yes', pad='sep', tmpl='same', steps=60)


def gen_for(s, gv='sft'):
    if gv == 'basePlain': return generate((W0, b0), P_PLAIN)
    if gv == 'baseTmpl': return generate((W0, b0), P_TMPL)
    return generate(run(s['count'], mk_of(s), s['steps']), P_DIFF if s['tmpl'] == 'diff' else P_TMPL)


def stops(g):
    ks = [k for k, _ in g]
    i = ks.index('dot') if 'dot' in ks else -1
    return i >= 0 and i + 1 < len(ks) and ks[i + 1] == 'eot'


# ---------------- token counts, from the template ----------------
def count_parts(ex):
    """(prompt tokens, response tokens, [(who, kind, n)]) in order; kind 'eot' marks an end token"""
    if ex == 'toy':
        p, r = conv('kv'); return [('p', k, 1) for k in p] + [('r', k, 1) for k in r]
    turns = {'chat': [(30, 199)], 'article': [(1790, 119)], 'multi': [(30, 150), (20, 120), (25, 180)]}[ex]
    out = [('p', 'bot', 1)]
    for u, a in turns:
        out += [('p', 'hdr', 4), ('p', 'text', u), ('p', 'eot', 1), ('p', 'hdr', 4), ('r', 'text', a), ('r', 'eot', 1)]
    return out


def counts(ex, count, mk):
    parts = count_parts(ex)
    Pn = sum(n_ for w, _, n_ in parts if w == 'p'); Rn = sum(n_ for w, _, n_ in parts if w == 'r')
    cp = cr = 0
    for i, (w, k, n_) in enumerate(parts):
        c = n_ - 1 if i == 0 else n_          # the first token is never predicted
        if w == 'p':
            if count == 'all' and not (mk == 'id' and k == 'eot'): cp += c
        elif not (mk and k == 'eot'): cr += c
    N = cp + cr
    return dict(P=Pn, R=Rn, cp=cp, cr=cr, N=N, share=cr / N, wAll=Pn + Rn - 1, ratio=(Pn + Rn - 1) / Rn)


def mulberry32(a):
    def r():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (1 | t)) & 0xFFFFFFFF
        t = (t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF) ^ t) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return r


def js_round(x):
    return math.floor(x + 0.5)


def pc_share(p):
    v = p * 100
    if abs(v - round(v)) < 1e-9: return f'{round(v)}%'
    return f'{v:.2f}%' if v < 10 else f'{v:.1f}%'


with sync_playwright() as pw:
    b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900}); errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    T = lambda js, *a: pg.evaluate('(a)=>{const T=window.__sft; return (' + js + ')(a)}', list(a))
    # ---- constants
    check('vocabulary: 26 tokens in the same order', T('()=>T.vocab()') == VOC, T('()=>T.vocab()'))
    check('parameters: 8 × 26 × 26 + 26 = 5,434', T('()=>T.NPAR') == K * V * V + V == 5434)
    check('K, step size, pretraining steps, max tokens', [T('()=>T.K'), T('()=>T.LR'), T('()=>T.PRE'), T('()=>T.MAXG')] == [8, 1, 300, 16])
    # ---- generations: base model and every SFT run the page can show
    def same_gen(name, page_g, mine):
        ok = len(page_g) == len(mine) and all(a['k'] == k and abs(a['p'] - p_) < 1e-7 for a, (k, p_) in zip(page_g, mine))
        check(name, ok, ([(a['k'], round(a['p'], 4)) for a in page_g][:8], [(k, round(p_, 4)) for k, p_ in mine][:8]))
    for gv in ('basePlain', 'baseTmpl'):
        same_gen(f'generation: {gv}', T('([gv])=>T.gen({},gv)', gv), gen_for(DEF, gv))
    for c in ('all', 'resp'):
        for e in ('yes', 'no'):
            for pd in ('sep', 'eot'):
                for st_ in (20, 60, 150):
                    for tm in ('same', 'diff'):
                        s = dict(DEF, count=c, eotl=e, pad=pd, steps=st_, tmpl=tm)
                        same_gen(f'generation: {c}/{e}/{pd}/{st_}/{tm}', T('([o])=>T.gen(o)', s), gen_for(s))
                    s = dict(DEF, count=c, eotl=e, pad=pd, steps=st_)
                    rl = T('([o])=>T.respLoss(o)', s); mine = resp_loss(run(c, mk_of(s), st_))
                    check(f'loss on the answers: {c}/{e}/{pd}/{st_}', abs(rl - mine) < 1e-9, (rl, mine))
                    check(f'counted positions: {c}/{e}/{pd}/{st_}', T('([o])=>T.nCounted(o)', s) == len(positions(c, mk_of(s))))
    # ---- what the page says about those runs
    g = [k for k, _ in gen_for(DEF, 'basePlain')]
    check('part 1: the base model answers once, then writes the next question and never stops', g[:4] == ['kv', 'is', 'd_kv', 'dot'] and g[4] == 'what' and 'eot' not in g, g)
    g = [k for k, _ in gen_for(DEF, 'baseTmpl')]
    check('part 2: the base model asked in the template never stops', 'eot' not in g, g)
    g1 = gen_for(dict(DEF, count='all')); g2 = gen_for(DEF)
    check('part 4: every token counts, 60 steps: another term, stops', g1[0][0] != 'kv' and stops(g1), g1)
    check('part 4: only the answer, 60 steps: KV cache, stops', g2[0][0] == 'kv' and stops(g2), g2)
    g3 = [k for k, _ in gen_for(dict(DEF, pad='eot'))]
    check('part 5: pad = end token: never stops, red at the 5th token, writes a new question', 'eot' not in g3 and g3[4] != 'eot' and g3[3] == 'dot' and 'what' in g3[4:], g3)
    check('part 5: a separate pad token stops', stops(gen_for(DEF)))
    g4 = [k for k, _ in gen_for(dict(DEF, count='all', pad='eot', steps=150))]
    check('challenge statement: it writes the next user header and a question', 'eot' not in g4 and 'sh' in g4 and 'user' in g4 and 'what' in g4, g4)
    check('template, different: never stops', not stops(gen_for(dict(DEF, tmpl='diff'))))
    # ---- the bridge to DPO
    br = T('()=>T.bridge()')
    Wb = run('resp', '', 60); p, r = conv('kv'); s = p + r
    rows = []
    for t in range(len(p), len(s)):
        W, bb = Wb; z = feats(s, t) @ W + bb; pr = softmax_rows(z[None, :])[0]
        rows.append((s[t], z[IX[s[t]]], pr[IX[s[t]]], math.log(pr[IX[s[t]]])))
    ok = all(a['k'] == k and abs(a['z'] - z_) < 1e-8 and abs(a['p'] - p_) < 1e-9 and abs(a['lp'] - l_) < 1e-9 for a, (k, z_, p_, l_) in zip(br['rows'], rows)) and len(rows) == len(br['rows'])
    check('bridge: each answer token’s logit, probability and log p', ok, (br['rows'][:2], rows[:2]))
    lp = sum(x[3] for x in rows)
    ref = sum(math.log(probs((W0, b0), s, t)[IX[s[t]]]) for t in range(len(p), len(s)))
    check('bridge: log π(y|x) and log π_ref(y|x)', abs(br['sum'] - lp) < 1e-9 and abs(br['ref'] - ref) < 1e-9, (br['sum'], lp, br['ref'], ref))
    refrows = [math.log(probs((W0, b0), s, t)[IX[s[t]]]) for t in range(len(p), len(s))]
    check('bridge: the reference model’s log p of each answer token (the stage’s last column)', len(br['refRows']) == len(refrows) and all(a['k'] == s[len(p) + i] and abs(a['lp'] - l_) < 1e-9 for i, (a, l_) in enumerate(zip(br['refRows'], refrows))), (br['refRows'][:2], refrows[:2]))
    check('bridge: SFT made the answer far more likely (log-ratio > 5)', lp - ref > 5, lp - ref)
    # ---- token counts
    for ex in ('toy', 'chat', 'article', 'multi'):
        for c in ('all', 'resp'):
            for mk in ('', 'ans', 'id'):
                pc_ = T('([ex,c,m])=>T.counts(ex,c,m)', ex, c, mk); mine = counts(ex, c, mk)
                check(f'counts: {ex}/{c}/{mk or "-"}', all(abs(pc_[k] - v) < 1e-12 for k, v in mine.items()), (pc_, mine))
    art = counts('article', 'all', '')
    check('article: 1,800 prompt, 120 answer tokens, 1,919 counted, 6.25%, about 16 times', (art['P'], art['R'], art['N'], pc_share(art['share']), round(art['ratio'])) == (1800, 120, 1919, '6.25%', 16), art)
    check('toy conversation: 17 counted positions (every token), 5 (answer only), 29.4%', (counts('toy', 'all', '')['N'], counts('toy', 'resp', '')['N'], pc_share(counts('toy', 'all', '')['share'])) == (17, 5, '29.4%'))
    # ---- the challenge
    CH0 = dict(count='all', eotl='yes', pad='eot', tmpl='same')
    allc = T('()=>T.chAll()'); good = []
    for x in allc:
        s = dict(DEF, steps=150, **x['c']); st_ = stops(gen_for(s)); sh = counts('toy', s['count'], mk_of(s))['share']
        ch = sum(1 for k in CH0 if s[k] != CH0[k]); ok = st_ and sh >= 0.9
        if ok: good.append(x['c'])
        check(f'challenge: {x["c"]}', x['stop'] == st_ and abs(x['share'] - sh) < 1e-12 and x['ok'] == ok and x['changes'] == ch, (x, st_, sh, ok, ch))
    check('challenge: 16 combinations, exactly one passes (answer only, separate pad), 2 changes', len(allc) == 16 and good == [dict(count='resp', eotl='yes', pad='sep', tmpl='same')] and T('()=>T.chMin()') == 2, good)
    # ---- the drill and the exits
    for seed in range(1, 13):
        r_ = mulberry32(seed); Pd = 10 * js_round(20 + r_() * 380); Rd = 5 * js_round(10 + r_() * 150); Nd = Pd + Rd - 1
        d = T('([s])=>T.drill(s)', seed)
        check(f'drill {seed}', (d['P'], d['R'], d['N']) == (Pd, Rd, Nd) and abs(d['share'] - Rd / Nd) < 1e-12 and abs(d['ratio'] - Nd / Rd) < 1e-12, (d, Pd, Rd))
    ex_ = T('()=>T.exCases()'); ea = T('()=>T.exits()')
    s1 = 600 / (200 + 600 - 1); r2 = (3000 + 100 - 1) / 100
    check('exit transfer 1: 600 ÷ 799 = 75.1%, “About 75%” (index 2)', abs(ex_['s1'] - s1) < 1e-12 and ea['transferA'][0] == 2 and pc_share(s1) == '75.1%')
    check('exit transfer 2: 3,099 ÷ 100 = 31.0, “About 30 times” (index 1)', abs(ex_['r2'] - r2) < 1e-12 and ea['transferA'][1] == 1)
    check('exit explain answers: the first option', ea['explainA'] == [0, 0])
    # ---- numbers as the page prints them
    pg.evaluate("()=>{location.hash='sft'}"); pg.wait_for_timeout(500)
    for lang in ('en', 'zh'):
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(300)
        pg.evaluate("()=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"guided\"]')].find(e=>e.offsetParent!==null); b.click();}"); pg.wait_for_timeout(200)
        pg.evaluate("()=>{document.querySelector('#sft-rail [data-go=\"2\"]')?.click()}"); pg.wait_for_timeout(250)
        if T('()=>T.state().step') != 2:
            for _ in range(2):
                pg.evaluate("()=>document.querySelector('#sft-rail [data-nav=\"next\"]').click()"); pg.wait_for_timeout(250)
        pg.evaluate("()=>document.querySelector('#sft-rail [data-pred=\"10to50\"]').click()"); pg.wait_for_timeout(300)
        obs = pg.evaluate("()=>document.querySelector('#sft-rail [data-obs]').innerText")
        frac16 = f'1/{round(art["ratio"])}'
        check(f'{lang} part 3 prints 6.25% and the weight “about {frac16}” (review S13: a fraction, not “16 times smaller”)', '6.25%' in obs and frac16 in obs and '倍' not in obs and 'times smaller' not in obs, obs[:200])
        bar = pg.evaluate("()=>document.querySelector('#sft-sharebar').innerText")
        check(f'{lang} part 3 bar: 1,799 prompt and 120 answer tokens', '1,799' in bar and '120' in bar and '93.75%' in bar and '6.25%' in bar, bar)
        pg.evaluate("()=>document.querySelector('#sft-rail [data-nav=\"next\"]').click()"); pg.wait_for_timeout(300)
        sh = pg.evaluate("()=>document.querySelector('#sft-metrics [data-k=\"sh\"] .v').textContent")
        check(f'{lang} part 4 card: the answer is 29.4% of the counted tokens', sh == '29.4%', sh)
        for _ in range(2):
            pg.evaluate("()=>document.querySelector('#sft-rail [data-nav=\"showme\"]')?.click()"); pg.wait_for_timeout(200)
            pg.evaluate("()=>{const i=document.querySelector('#sft-player input[data-key=\"scrub\"]'); i.value=99; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(200)
            pg.evaluate("()=>document.querySelector('#sft-rail [data-nav=\"next\"]').click()"); pg.wait_for_timeout(300)
        # Codex R11 (round 3): the table is on the stage and fills in step by step; step 3 prints the sum and the SFT loss, step 4 the reference
        sgn = lambda v, d: ('−' if v < 0 else '') + f'{abs(v):.{d}f}'
        step6 = lambda: pg.evaluate("()=>({tbl:[...document.querySelectorAll('#sft-dpobox .sft-tbl tbody tr')].map(r=>[...r.cells].map(c=>c.textContent)), foot:(document.querySelector('#sft-dpobox .sft-tbl tfoot')||{}).innerText||'', now:document.querySelector('#sft-dpobox .sft-dp-now').innerText})")
        pg.evaluate("()=>document.querySelector('#sft-dp-k [data-v=\"0\"]').click()"); pg.wait_for_timeout(150)   # the step is kept per lab: start from step 1
        a0 = step6()
        check(f'{lang} part 6, step 1: each token’s logit and probability; the first row’s numbers in the text', [r_[1:] for r_ in a0['tbl']] == [[f'{z_:.2f}', f'{p_ * 100:.2f}%'] for _, z_, p_, l_ in rows] and f'{rows[0][1]:.2f}' in a0['now'] and f'{rows[0][2] * 100:.2f}%' in a0['now'], a0)
        pg.evaluate("()=>document.querySelector('#sft-dpobox [data-act=\"bk-next\"]').click()"); pg.wait_for_timeout(150)
        a1 = pg.evaluate("()=>document.querySelector('#sft-dpobox .sft-dp-now .sft-math-line').textContent.replace(/\\s+/g,'')")
        check(f'{lang} part 6, step 2: log {rows[0][2]:.4f} = {sgn(rows[0][3], 3)}', f'log{rows[0][2]:.4f}={sgn(rows[0][3], 3)}' in a1, a1)
        pg.evaluate("()=>document.querySelector('#sft-dpobox [data-act=\"bk-next\"]').click()"); pg.wait_for_timeout(150)
        a2 = step6()
        check(f'{lang} part 6, step 3: the sum {sgn(lp, 3)} and the SFT loss {-lp / 5:.3f}', ('−' + f'{abs(lp):.3f}') in a2['foot'] and f'{-lp / 5:.3f}' in a2['now'], a2['foot'] + ' | ' + a2['now'][-200:])
        pg.evaluate("()=>document.querySelector('#sft-dpobox [data-act=\"bk-next\"]').click()"); pg.wait_for_timeout(150)
        a3 = step6()
        want = [[f'{z_:.2f}', f'{p_ * 100:.2f}%', sgn(l_, 3), sgn(r_l, 3)] for (_, z_, p_, l_), r_l in zip(rows, refrows)]
        check(f'{lang} part 6 table: logits, probabilities, log p, the reference model’s log p', [r_[1:] for r_ in a3['tbl']] == want, (a3['tbl'], want))
        check(f'{lang} part 6: the sum and the reference sum', ('−' + f'{abs(lp):.3f}') in a3['foot'] and ('−' + f'{abs(ref):.2f}') in a3['foot'], a3['foot'])
        txt = a3['now']
        check(f'{lang} part 6: reference {ref:.2f}, log-ratio {lp - ref:.2f}', ('−' + f'{abs(ref):.2f}') in txt and f'{lp - ref:.2f}' in txt, txt[-400:])
        # H2 (owner 2026-10-09): the log-ratio is a named formula block whose lines break at “=”; read each block row by row,
        # the “=” cell then the formula cell, so consecutive lines give “−lp−(−ref)=ratio”
        ml = pg.evaluate("()=>[...document.querySelectorAll('#sft-dpobox .sft-eq')].map(q=>[...q.children].filter(c=>!c.classList.contains('sft-eq-n')).map(c=>c.textContent.replace(/\\s+/g,'')).join('')).join(' | ')")
        check(f'{lang} part 6: the log-ratio in its own formula block, {lp:.3f} − ({ref:.2f}) = {lp - ref:.2f}', f'−{abs(lp):.3f}−(−{abs(ref):.2f})={lp - ref:.2f}' in ml, ml)
    # the challenge's reason and the home tile's notes
    pg.evaluate("()=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"challenge\"]')].find(e=>e.offsetParent!==null); b.click();}"); pg.wait_for_timeout(300)
    pg.evaluate("()=>document.querySelector('#sft-rail [data-act=\"submit\"]').click()"); pg.wait_for_timeout(200)
    pg.evaluate("()=>document.querySelector('#sft-rail [data-act=\"reveal\"]').click()"); pg.wait_for_timeout(200)
    res = pg.evaluate("()=>document.querySelector('#sft-ch-res').innerText"); st0 = counts('toy', 'all', 'id')
    check('challenge: the start’s share 4 ÷ 15 = 26.7% in the goal row and the reason', (st0['cr'], st0['N']) == (4, 15) and pc_share(st0['share']) == '26.7%' and '26.7%' in res and '26.7%' in pg.evaluate("()=>document.querySelector('#sft-ch-hv').textContent"), res[-300:])
    pg.evaluate("()=>{location.hash=''}"); pg.wait_for_timeout(300)
    pg.evaluate("()=>{const b=[...document.querySelectorAll('[data-world=\"post\"]')].find(e=>e.getClientRects().length); b.dispatchEvent(new MouseEvent('click',{bubbles:true}));}"); pg.wait_for_timeout(400)
    notes = pg.evaluate("()=>[...document.querySelectorAll('#tile-sft .tz-note')].map(e=>e.textContent).join(' ')")
    rd = pg.evaluate("()=>{const b=document.querySelector('#tile-sft button[data-v=\"all\"]'); b.click(); return document.querySelector('#tile-sft .tz-read').textContent}")
    check('home tile: 26 tokens, 5,434 parameters, 17 and 5 counted positions, 29.4%, GPU', all(x in notes for x in ('26', '5,434', '17', ' 5 ')) and '29.4%' in rd and 'GPU' in rd, (notes[:200], rd))
    # the scenarios: the chat example's share, and the small model's own data named with its share (review S1)
    pg.evaluate("()=>{location.hash='sft'}"); pg.wait_for_timeout(400)
    pg.evaluate("()=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"sandbox\"]:not([data-flow=free])')].find(e=>e.offsetParent!==null); b.click();}"); pg.wait_for_timeout(300)
    pg.evaluate("()=>document.querySelector('#sft-rail [data-preset=\"0\"]').click()"); pg.wait_for_timeout(300)
    pg.evaluate("()=>{const i=document.querySelector('#sft-player input[data-key=\"scrub\"]'); i.value=99; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(300)
    ins = pg.evaluate("()=>document.querySelector('#sft-insight').innerText")
    chat, toy_all = counts('chat', 'all', ''), counts('toy', 'all', '')
    check('scenario 1: the chat example’s share and the small model’s own share', pc_share(chat['share']) == '83.7%' and '83.7%' in ins and pc_share(toy_all['share']) in ins and '6' in ins, ins[:300])
    # the drill (seed 3 on entry): the share and the weight as a fraction
    pg.evaluate("()=>document.querySelector('#sft-app [data-act=\"drill-reveal\"]').click()"); pg.wait_for_timeout(200)
    da = pg.evaluate("()=>document.querySelector('#sft-drill-a').innerText")
    r_ = mulberry32(3); Pd = 10 * js_round(20 + r_() * 380); Rd = 5 * js_round(10 + r_() * 150); Nd = Pd + Rd - 1; rt = Nd / Rd
    fr = '1/' + (f'{round(rt)}' if rt >= 10 else f'{rt:.1f}')
    check(f'drill (seed 3): {pc_share(Rd / Nd)} and about {fr}', pc_share(Rd / Nd) in da and fr in da and f'{rt:.2f}' in da, da[:200])
    # the capability card, a new case 2: the hover under the answer holds the arithmetic, 3,099 ÷ 100 ≈ 31
    pg.evaluate("()=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"challenge\"]')].find(e=>e.offsetParent!==null); b.click();}"); pg.wait_for_timeout(300)
    pg.evaluate("()=>document.querySelector('#sft-rail [data-act=\"submit\"]').click()"); pg.wait_for_timeout(200)
    pg.evaluate("()=>document.querySelector('#sft-rail [data-exits] [data-ex=\"tab\"][data-i=\"1\"]').click()"); pg.wait_for_timeout(150)
    pg.evaluate("()=>document.querySelector('#sft-rail [data-exits] [data-ex=\"q\"][data-j=\"1\"]').click()"); pg.wait_for_timeout(150)
    pg.evaluate("()=>document.querySelector('#sft-rail [data-exits] [data-ex=\"ans\"][data-v=\"1\"]').click()"); pg.wait_for_timeout(150)
    tip2 = pg.evaluate("()=>{const d=document.querySelector('#sft-rail [data-exits] .ex-res .def'); return d?d.getAttribute('data-tipx'):''}")
    n2 = 3000 + 100 - 1
    check(f'new case 2: {n2:,} ÷ 100 ≈ {round(n2 / 100)} in the answer’s hover', f'{n2:,} ÷ 100 ≈ {round(n2 / 100)}' in tip2, tip2)
    check('no page errors', not errs, errs[:2])
    b.close()
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:30] if fails else 'none')
sys.exit(1 if fails else 0)
