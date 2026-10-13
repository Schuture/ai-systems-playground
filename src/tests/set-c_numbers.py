"""Set C teasers (rag-failure, agent-reliability, agent-anatomy, prompt-injection, eval-ci, pass-k):
recompute every number the tiles can show, for every control position, independently in Python,
and compare with the page (the read-only hook window.__tzC and the rendered text, in EN and ZH).

    python3 tests/set-c_numbers.py        (from the prototype folder or anywhere)

Independent derivations: exact fractions for p^n, pass@k and pass^k; the tau-bench / Codex estimators
checked against the closed forms; the eval interval from the binomial variance with the exact normal
quantile; a seeded Monte Carlo for the agent chain and for the interval's coverage; the agent trace's
context sizes from cumulative sums over a trace table written here. The page side checks the one-sentence
result (.tz-read), the plain-text notes (.tz-note, shown behind "Assumptions & sources") and the legend."""
import math, pathlib, re, subprocess, sys, random
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction as F
from itertools import accumulate
from math import comb
from statistics import NormalDist
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/set-c.html', '--only', 'set-c'], check=True, cwd=P, capture_output=True)
WR = P / 'out' / 'wrapped-set-c.html'
WR.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
              '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'set-c.html').read_text() + '</body></html>')
URL = 'file://' + str(WR)

fails, notes, n_ok = [], [], 0
def check(name, cond, info=''):
    global n_ok
    if cond: n_ok += 1
    else: fails.append(f'{name} {info}')

# ---------- display rules (the page's format, restated): percent to 0.1, or 0.01 within 0.05 of 100 ----------
def rnd(x, d):
    """Round an exact rational half-up to d decimals; also say whether it sat exactly on a tie."""
    x = F(x); q = x * 10 ** d; fl = math.floor(q); frac = q - fl
    return (fl + (1 if frac >= F(1, 2) else 0)) / F(10 ** d), frac == F(1, 2)
def s_dec(v, d): return f'{float(v):.{d}f}'
def pcts(v):
    """Acceptable strings for a probability v (exact). At an exact tie either neighbour is correct to that precision."""
    v = F(v)
    if v >= 1: return {'100%'}
    p = v * 100; d = 2 if p >= F(9995, 100) else 1
    r, tie = rnd(p, d); out = {s_dec(r, d) + '%'}
    if tie: out.add(s_dec(r - F(1, 10 ** d), d) + '%')
    return {('> 99.99%' if s == '100.00%' else s) for s in out}
MINUS = '−'
def signed(v, d=1):
    r, _ = rnd(abs(F(v)), d); s = s_dec(r, d)
    return (MINUS if v < 0 and s != '0.0' else '+') + s
def thou(n): return f'{int(n):,}'
def has_any(text, opts): return any(o in text for o in opts)

# ---------- independent models ----------
# agent-reliability: the task succeeds only if all n independent steps do
REL_P = [F(90, 100), F(95, 100), F(98, 100), F(99, 100), F(995, 1000)]
def rel(n, p): return p ** n
# pass-k with p = 3/5: exact closed forms, checked below against the estimators' expectations
PK_P = F(3, 5)
def pass_at(k): return 1 - (1 - PK_P) ** k
def pass_hat(k): return PK_P ** k
# eval-ci: two independent binomial accuracies, normal approximation
CI_A, CI_B, CI_NS = F(70, 100), F(72, 100), [100, 500, 1000, 5000, 10000]
Z = 1.96
def ci_half(n, z=Z): return z * math.sqrt(float(CI_A * (1 - CI_A) / n + CI_B * (1 - CI_B) / n))
# agent-anatomy: the illustrative trace, written out here as a table (tool, tokens written by the model, tokens the tool returns)
BASE = 3200
TRACE = [('run_tests', 150, 1900), ('search_code', 120, 650), ('read_file', 110, 3400), ('read_file', 130, 2200),
         ('search_code', 160, 700), ('read_file', 120, 2500), ('edit_file', 450, 120), ('run_tests', 140, 3800),
         ('read_file', 170, 1100), ('edit_file', 380, 120), ('run_tests', 120, 600), (None, 260, 0)]
mod_before = [0] + list(accumulate(m for _, m, _ in TRACE))[:-1]
tool_before = [0] + list(accumulate(o for _, _, o in TRACE))[:-1]
CTX = [BASE + a + b for a, b in zip(mod_before, tool_before)]          # tokens the model reads at step i
CUM_IN = list(accumulate(CTX)); CUM_OUT = list(accumulate(m for _, m, _ in TRACE))
def share_pct(i):  # tool outputs' share of step i's context, rounded half-up to a whole percent
    r, tie = rnd(F(tool_before[i - 1], CTX[i - 1]) * 100, 0); return int(r), tie

# ---------- theory checks, no browser ----------
for k in range(1, 11):   # tau-bench: pass^k = E_task[C(c,k)/C(n,k)]; Codex: pass@k = E[1 - C(n-c,k)/C(n,k)]; c ~ Binomial(n, p)
    for n in (k, 10, 16):
        e_hat = sum(comb(n, c) * PK_P ** c * (1 - PK_P) ** (n - c) * F(comb(c, k), comb(n, k)) for c in range(n + 1))
        e_at = sum(comb(n, c) * PK_P ** c * (1 - PK_P) ** (n - c) * (1 - F(comb(n - c, k), comb(n, k))) for c in range(n + 1))
        check(f'tau-bench estimator = p^k (k={k}, n={n})', e_hat == pass_hat(k))
        check(f'Codex estimator = 1-(1-p)^k (k={k}, n={n})', e_at == pass_at(k))
check('1.96 is the 97.5% normal quantile to 2 decimals', abs(NormalDist().inv_cdf(0.975) - Z) < 5e-3)
zq = NormalDist().inv_cdf(0.975)
for n in CI_NS:   # the shown numbers do not depend on rounding the quantile
    check(f'eval-ci n={n}: same display with the exact quantile', f'{ci_half(n) * 100:.1f}' == f'{ci_half(n, zq) * 100:.1f}')
n_tgt = Z ** 2 * float(CI_A * (1 - CI_A) + CI_B * (1 - CI_B)) / 0.02 ** 2
check('eval-ci: questions for +-2 points', 3952 < n_tgt < 3954, n_tgt)
check('eval-ci: +-2 reached between 3,953 and 3,954 questions', ci_half(3953) > 0.02 > ci_half(3954))
N_TGT_SHOWN = int(round(n_tgt / 50) * 50)
check('eval-ci: shown target rounds to 3,950', N_TGT_SHOWN == 3950)
rng = random.Random(7)   # seeded Monte Carlo: 20 independent steps at 95%
sims = 200_000; hits = sum(all(rng.random() < 0.95 for _ in range(20)) for _ in range(sims))
se = math.sqrt(0.3585 * 0.6415 / sims)
check('Monte Carlo 0.95^20', abs(hits / sims - float(rel(20, F(95, 100)))) < 4 * se, hits / sims)
try:   # coverage of the eval interval at n = 500 (numpy, seeded)
    import numpy as np
    g = np.random.default_rng(11); R = 40_000; n = 500
    a = g.binomial(n, 0.70, R) / n; b = g.binomial(n, 0.72, R) / n
    h = 1.96 * np.sqrt(a * (1 - a) / n + b * (1 - b) / n); cov = float(np.mean((b - a - h <= 0.02) & (0.02 <= b - a + h)))
    check('eval-ci interval covers the true gap about 95% of the time (n=500)', 0.94 < cov < 0.96, cov)
except ImportError:
    notes.append('numpy missing: coverage check skipped')
# the anatomy numbers quoted in the report
check('anatomy step 8 context', CTX[7] == 15910); check('anatomy step 8 share', share_pct(8)[0] == 72)
check('anatomy step 8 cumulative', (CUM_IN[7], CUM_OUT[7]) == (79830, 1380))

# ---------- page ----------
WORLD = {'rag-failure': 'apps', 'agent-reliability': 'agents', 'agent-anatomy': 'agents', 'prompt-injection': 'agents', 'eval-ci': 'eval', 'pass-k': 'eval'}
def show(pg, lab): pg.evaluate(f"()=>document.querySelector('#domStrip .ds-btn[data-world=\"{WORLD[lab]}\"]').click()"); pg.wait_for_timeout(60)
def set_range(pg, lab, v): pg.evaluate(f"()=>{{const i=document.querySelector('#tile-{lab} input[type=range]'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}}));}}")
def texts(pg, lab):
    return pg.evaluate(f"""()=>{{const t=document.querySelector('#tile-{lab}');
      return {{read:t.querySelector('.tz-read').textContent, out:(t.querySelector('.tz-lab output')||{{}}).textContent||'',
        vis:[...t.querySelectorAll('.tz-vis text')].map(x=>x.textContent), visText:t.querySelector('.tz-vis').innerText,
        note:[...t.querySelectorAll('.tz-note')].map(x=>x.textContent).join(' || '),
        leg:[...t.querySelectorAll('.tz-leg > span')].filter(x=>!x.hidden).map(x=>x.textContent.trim())}};}}""")
def need(name, text, parts):
    for w in parts: check(f'{name} has {w!r}', w in text, text[:160])
def notes_ok(name, t, zh):
    """guide R11.5: 2 to 4 short sentences behind Assumptions & sources, plain sentences without "·" (R6.14)"""
    paras = [x for x in t['note'].split(' || ') if x.strip()]
    whole = ' '.join(paras)
    n = len(re.findall('[。！？]', whole)) if zh else len(re.findall(r'[.!?](?=\s|$)', whole.replace('et al.', 'et al')))
    check(f'{name} notes: 2 to 4 sentences', 2 <= n <= 4, (n, whole[:200]))
    check(f'{name} notes: plain sentences', '·' not in whole and all(x.strip().endswith(('.', '。')) for x in paras), whole[:200])
def sentence_ok(name, r, zh):
    """guide R11.5 and R6.14: the tile's text has no "·" joins or arrow fragments"""
    check(f'{name} sentence has no "·" or "→"', '·' not in r and '→' not in r, r)

with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL + '#map'); pg.wait_for_timeout(400)
    for lang in ['en', 'zh']:
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(150)
        zh = lang == 'zh'
        # --- rag-failure: k = 1..10, the answer ranked 6th of 10
        show(pg, 'rag-failure')
        for k in range(1, 11):
            set_range(pg, 'rag-failure', k); t = texts(pg, 'rag-failure'); r = t['read']
            m = pg.evaluate(f'()=>__tzC.rag.model({k})')
            inp = k >= 6; a = k - 6
            check(f'[{lang}] rag k={k} model', m['inPrompt'] == inp and (m['after'] == (a if inp else None)), m)
            check(f'[{lang}] rag k={k} output', t['out'] == f'k = {k}', t['out'])
            check(f'[{lang}] rag k={k} cells', len([x for x in t['vis'] if x.isdigit()]) == 10 + k, t['vis'])
            check(f'[{lang}] rag k={k} legend shows the miss only when missed', any(('被漏掉' if zh else 'left out') in x for x in t['leg']) == (not inp), t['leg'])
            if not inp:
                need(f'[{lang}] rag k={k}', r, [f'top-{k}', '排名第 1 的 chunk' if k == 1 else f'排名前 {k} 的 chunk', '第 6 个里', '到不了模型', '检索失败'] if zh else
                     [f'Top-{k} retrieval', 'only the best-ranked chunk' if k == 1 else f'the {k} best-ranked chunks', 'the answer is in chunk 6', 'retrieval failure'])
                check(f'[{lang}] rag k={k} footer is the miss', any(('不在 prompt 里' if zh else 'not in the prompt') in x for x in t['vis']))
            elif k == 6:
                need(f'[{lang}] rag k=6', r, ['top-6', '第 6 个 chunk', '排在最后、紧挨问题'] if zh else ['Top-6 retrieval', 'chunk 6', 'last, right before the question'])
            elif k == 7:   # TILES-B #9: position 6 of 7 is one from the end, not the middle
                need(f'[{lang}] rag k=7', r, ['top-7', '第 6 个 chunk', '离末尾只差一个'] if zh else ['Top-7 retrieval', 'chunk 6', 'one chunk from the end'])
                check(f'[{lang}] rag k=7 does not call it the middle', ('用不好中间' not in r) if zh else ('less reliably' not in r), r)
            else:
                need(f'[{lang}] rag k={k}', r, [f'top-{k}', '第 6 个 chunk', f'后面还有 {a} 个', '推向中间'] if zh else [f'Top-{k} retrieval', 'chunk 6', f'{a} chunks follow it', 'toward the middle'])
            if inp and k > 6:
                check(f'[{lang}] rag k={k} footer', any((f'第 6 / {k} 个，后面还有 {a} 个' if zh else f'6th of {k}, {a} chunk') in x for x in t['vis']), t['vis'])
            sentence_ok(f'[{lang}] rag k={k}', r, zh); notes_ok(f'[{lang}] rag k={k}', t, zh)
        need(f'[{lang}] rag notes', t['note'], ['Lost in the Middle', 'TACL 2024', '10、20 或 30' if zh else '10, 20 or 30', '示意' if zh else 'Illustrative'])
        # SITE-A-r7 item 12: no clause about the tile's own design, no repeat of the diagram's tooltip
        for gone in (['一个滑块', '按排名顺序放进 prompt'] if zh else ['one slider', 'in rank order']):
            check(f'[{lang}] rag notes drop {gone!r}', gone not in t['note'], t['note'][:200])
        # --- agent-reliability: n = 1..50
        show(pg, 'agent-reliability')
        for n in range(1, 51):
            set_range(pg, 'agent-reliability', n); t = texts(pg, 'agent-reliability'); r = t['read']
            m = pg.evaluate(f'()=>__tzC.rel.model({n})')
            check(f'[{lang}] rel n={n} hook', abs(m['succ'] - float(rel(n, F(95, 100)))) < 1e-12 and abs(m['cmpSucc'] - float(rel(n, F(99, 100)))) < 1e-12)
            s95, s99 = pcts(rel(n, F(95, 100))), pcts(rel(n, F(99, 100)))
            check(f'[{lang}] rel n={n} sentence 95%', has_any(r, s95), (s95, r))
            check(f'[{lang}] rel n={n} sentence 99%', has_any(r, s99), (s99, r))
            check(f'[{lang}] rel n={n} chart values', any(x in s95 for x in t['vis']) and any(x in s99 for x in t['vis']), t['vis'])
            # the per-step rate is named as a success rate, as SITE-A-r7 item 14e asks for pass-k
            check(f'[{lang}] rel n={n} setup named', ((f'{n} 步的 agent 任务，' + ('每步' if n > 1 else '这一步') + '成功率 95%') if zh else
                                                     (f'An agent task of {n} step' + ('s' if n > 1 else '') + ' at a 95% success rate per step')) in r, r)
            sentence_ok(f'[{lang}] rel n={n}', r, zh); notes_ok(f'[{lang}] rel n={n}', t, zh)
            st = f'{n} 步' if zh else ('1 step' if n == 1 else f'{n} steps')
            check(f'[{lang}] rel n={n} output and legend', t['out'] == st and st in t['leg'], (t['out'], t['leg']))
            if any(len(pcts(rel(n, p_))) > 1 for p_ in (F(95, 100), F(99, 100))): notes.append(f'rel n={n}: a value sits exactly on a rounding tie; page shows {r[:90]}')
        for p_ in REL_P:
            check(f'[{lang}] rel notes have {p_}^50', has_any(t['note'], pcts(rel(50, p_))), t['note'])
        need(f'[{lang}] rel notes', t['note'], ['pⁿ', '每步成功率为 p' if zh else 'a success rate p per step'])
        check(f'[{lang}] rel notes: no "→" pairs', '→' not in t['note'], t['note'][:200])
        # --- agent-anatomy: steps 1..12
        show(pg, 'agent-anatomy')
        data = pg.evaluate('()=>({base:__tzC.anat.data.base, steps:__tzC.anat.data.steps.map(s=>[s.tool,s.m,s.o])})')
        check(f'[{lang}] anatomy trace table matches', data['base'] == BASE and [tuple(x) for x in data['steps']] == TRACE, data)
        rng_of = lambda tool: f"{thou(min(o for tl, _, o in TRACE if tl == tool))}–{thou(max(o for tl, _, o in TRACE if tl == tool))}"
        for i in range(1, 13):
            set_range(pg, 'agent-anatomy', i); t = texts(pg, 'agent-anatomy'); r = t['read']
            m = pg.evaluate(f'()=>__tzC.anat.model({i})')
            sh, tie = share_pct(i)
            check(f'[{lang}] anatomy {i} hook', m['cur']['total'] == CTX[i - 1] and m['cumIn'] == CUM_IN[i - 1] and m['cumOut'] == CUM_OUT[i - 1]
                  and [bb['total'] for bb in m['bars']] == CTX, m['cur'])
            check(f'[{lang}] anatomy {i} context tokens', thou(CTX[i - 1]) in r and thou(CTX[i - 1]) in t['vis'], (r, t['vis']))
            # R8 re-check 13c: the readout names the step in words, no "·"
            check(f'[{lang}] anatomy {i} output', t['out'] == (f'第 {i} / 12 步：' if zh else f'Step {i} of 12: ') + (TRACE[i - 1][0] or ('最终答复' if zh else 'final answer')), t['out'])
            check(f'[{lang}] anatomy {i} step named', (f'修 bug 第 {i} / 12 步' if zh else f'Bug-fix step {i} of 12') in r, r)
            check(f'[{lang}] anatomy {i} bars drawn so far', len(pg.eval_on_selector_all('#tile-agent-anatomy .tz-anat-tool', 'e=>e')) == max(0, i - 1)
                  and len(pg.eval_on_selector_all('#tile-agent-anatomy .tz-anat-sys', 'e=>e')) == i)
            # SITE-A-r7 item 14a: every count says it counts tokens
            if i == 1:
                check(f'[{lang}] anatomy 1 first call writes', (f'写出 {thou(CUM_OUT[0])} 个 token' if zh else f'writes {thou(CUM_OUT[0])} tokens') in r, r)
            else:
                check(f'[{lang}] anatomy {i} tool share', f'{sh}%' in r or (tie and f'{sh - 1}%' in r), (sh, r))
                check(f'[{lang}] anatomy {i} cumulative', (f'共读入 {thou(CUM_IN[i - 1])} 个 token、写出 {thou(CUM_OUT[i - 1])} 个 token' if zh else
                                                         f'the model has read {thou(CUM_IN[i - 1])} tokens and written {thou(CUM_OUT[i - 1])}') in r, r)
            check(f'[{lang}] anatomy {i} note names the step', (f'第 {i} 步' if zh else f'At step {i} the model') in t['note'], t['note'][:120])
            sentence_ok(f'[{lang}] anatomy {i}', r, zh); notes_ok(f'[{lang}] anatomy {i}', t, zh)
        need(f'[{lang}] anatomy notes', t['note'], [thou(BASE), 'Effective context engineering for AI agents', '示意' if zh else 'illustrative'])
        # SITE-A-r7 item 12: the two opening sentences are merged and the per-tool ranges are cut
        for gone in [rng_of('read_file'), rng_of('run_tests')]:
            check(f'[{lang}] anatomy notes drop {gone!r}', gone not in t['note'], t['note'][:200])
        # --- prompt-injection: two setups, no numbers; the text must differ and name addresses only at example.com
        show(pg, 'prompt-injection')
        seen = []
        for v in ['paste', 'isolate', 'paste']:
            pg.click(f'#tile-prompt-injection .seg-ctl button[data-v="{v}"]'); t = texts(pg, 'prompt-injection')
            check(f'[{lang}] injection {v} hook', pg.evaluate('()=>__tzC.inj.mode') == v)
            check(f'[{lang}] injection {v} pressed', pg.evaluate(f"()=>document.querySelector('#tile-prompt-injection button[data-v=\"{v}\"]').getAttribute('aria-pressed')") == 'true')
            emails = set(re.findall(r'[\w.]+@[\w-]+(?:\.[\w-]+)+', t['read'] + '\n' + t['visText'] + '\n' + t['note']))
            check(f'[{lang}] injection {v} only example.com addresses', emails and all(e.rstrip('.').endswith('example.com') for e in emails), emails)
            check(f'[{lang}] injection {v} legend matches the setup', len(t['leg']) == 1 and (('被注入' if zh else 'injected') in t['leg'][0]) == (v == 'paste'), t['leg'])
            seen.append(t['read']); sentence_ok(f'[{lang}] injection {v}', t['read'], zh); notes_ok(f'[{lang}] injection {v}', t, zh)
        check(f'[{lang}] injection sentences differ', seen[0] != seen[1] and seen[0] == seen[2])
        need(f'[{lang}] injection notes', t['note'], ['OWASP LLM01:2025', 'Greshake', 'RFC 2606', '示意' if zh else 'Illustrative'])
        # --- eval-ci: n in 100, 500, 1,000, 5,000, 10,000
        show(pg, 'eval-ci')
        check(f'[{lang}] ci buttons write counts in full', pg.eval_on_selector_all('#tile-eval-ci .seg-ctl button', 'bs=>bs.map(b=>b.textContent)') == [thou(n) for n in CI_NS])
        for n in CI_NS:
            pg.click(f'#tile-eval-ci .seg-ctl button[data-v="{n}"]'); t = texts(pg, 'eval-ci'); r = t['read']
            h = ci_half(n); lo, hi = 0.02 - h, 0.02 + h
            m = pg.evaluate(f'()=>__tzC.ci.model({n})')
            check(f'[{lang}] ci n={n} hook', abs(m['h'] - h) < 1e-12 and abs(m['lo'] - lo) < 1e-12 and m['includes0'] == (lo <= 0), m)
            check(f'[{lang}] ci n={n} interval', signed(lo * 100) in r and signed(hi * 100) in r, (signed(lo * 100), signed(hi * 100), r))
            check(f'[{lang}] ci n={n} verdict', (((('含 0' in r and '不含 0' not in r) if zh else 'includes 0' in r)) if lo <= 0 else (('不含 0' if zh else 'excludes 0') in r)), r)
            check(f'[{lang}] ci n={n} names the setup', thou(n) in r and ('70%' in r and '72%' in r) and t['out'] == f'n = {thou(n)}', (r, t['out']))
            check(f'[{lang}] ci n={n} row half-widths', all(f'±{ci_half(m_) * 100:.1f}' in t['vis'] for m_ in CI_NS), t['vis'])
            sentence_ok(f'[{lang}] ci n={n}', r, zh); notes_ok(f'[{lang}] ci n={n}', t, zh)
        need(f'[{lang}] ci notes', t['note'], [thou(N_TGT_SHOWN), '1.96', '0.72 × 0.28', 'Miller', 'Adding Error Bars to Evals'])
        # --- pass-k: k = 1..10
        show(pg, 'pass-k')
        for k in range(1, 11):
            set_range(pg, 'pass-k', k); t = texts(pg, 'pass-k'); r = t['read']
            m = pg.evaluate(f'()=>__tzC.pk.model({k})')
            check(f'[{lang}] pk k={k} hook', abs(m['at'] - float(pass_at(k))) < 1e-12 and abs(m['hat'] - float(pass_hat(k))) < 1e-12)
            A_, H_ = pcts(pass_at(k)), pcts(pass_hat(k))
            check(f'[{lang}] pk k={k} pass@k', has_any(r, A_) and any(x.startswith(tuple(A_)) for x in t['vis']), (A_, r, t['vis']))
            check(f'[{lang}] pk k={k} pass^k', has_any(r, H_) and any(x.startswith(tuple(H_)) for x in t['vis']), (H_, r))
            check(f'[{lang}] pk k={k} labels', (f'pass@{k}' in r and f'pass^{k}' in r) if k > 1 else ('pass@1 = pass^1' in r), r)
            check(f'[{lang}] pk k={k} output and legend', t['out'] == f'k = {k}' and f'k = {k}' in t['leg'], (t['out'], t['leg']))
            check(f'[{lang}] pk k={k} note: never sees a failure', f'pass^{k} = ' in t['note'] and has_any(t['note'], H_), t['note'][:200])
            check(f'[{lang}] pk k={k} never shows 100%', '100.0%' not in r and '100.00%' not in r)
            # SITE-A-r7 item 14e: the 60% is named as the success rate per try
            check(f'[{lang}] pk k={k} names the success rate', ('每次成功率 60%' if zh else 'At a 60% success rate per try') in r, r)
            sentence_ok(f'[{lang}] pk k={k}', r, zh); notes_ok(f'[{lang}] pk k={k}', t, zh)
        tb = pcts(F(604, 1000) ** 4)   # what 0.604^4 gives, next to the leaderboard's 60.4% and 38.3%
        check(f'[{lang}] pk notes cite tau-bench', 'τ-bench' in t['note'] and '60.4%' in t['note'] and '38.3%' in t['note'] and has_any(t['note'], tb), (tb, t['note']))
        need(f'[{lang}] pk notes', t['note'], ['Chen', 'Yao', '1 − (1 − p)ᵏ', 'pᵏ'])
    check('no page errors', not errs, errs)
    b.close()

for n_ in notes: print('note:', n_)
print(f'{n_ok} checks passed, {len(fails)} failed')
for f_ in fails[:40]: print('FAIL', f_)
sys.exit(1 if fails else 0)
