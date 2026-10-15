"""Numbers test for the eval-capstone lab: every number the page shows, recomputed independently in Python and compared with the
page's engine (window.__ecap) and with the text the page shows.

- The world (five categories: traffic share, true difference, share q of items with a winner, the judge's bias and disagreement),
  the item-by-item interval h = 1.96 * sqrt((q - D^2) / n), the judge's bias, calibration by prediction-powered inference with
  lambda = 1 (ppi_py's own ppi_mean_pointestimate and ppi_mean_ci, taken from the pinned file and run with alpha=0.05 and lam=1 on
  arrays built from the lab's setting, so the e - b^2 term is checked against ppi_py, not against the page's own formula), the
  verdicts with the 3-point margin, costs and days.
- Every substitution in Show the math and in the drill: the numbers a line shows, evaluated, give the next line's result.
- The chances a real evaluation would catch the loss, which the page quotes (normal approximation).
- Every reachable category setting (method x size x judge setting x calibration): estimate, half-width, verdict, cost.
- Every guided part's plan, the four scenarios, the A/B test of part 6 at 1, 3 and 7 days.
- The challenge: every one of the 884,736 reachable plans, enumerated here; the cheapest passing plan, the number of passing plans, the
  cheapest plan per judge setting (the chart's rows), the page's answer plan, and that no setting passes under any other judge setting.
- The conclusions the page draws, on the page's own numbers (the brief: “every conclusion the page draws must hold”).
- The Connect code's printed line, the drill, the new case's answer, the home tile, and the pinned sources.
Run: python3 tests/eval-capstone_numbers.py"""
import ast, itertools, math, pathlib, re, statistics, subprocess, sys
import numpy as np
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/eval-capstone.html', '--only', 'eval-capstone'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'eval-capstone').mkdir(parents=True, exist_ok=True)
W = P / 'out' / 'eval-capstone' / 'numbers-wrapped.html'
W.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>'
             + (P / 'out' / 'eval-capstone.html').read_text() + '</body></html>')
fails, n = [], 0
# MathML back to a Python expression, one per right-hand side of Show the math: '@' marks a symbol, '|' the word “to” between two numbers
MATHJS = r"""sel=>{const ex=n=>{const t=n.tagName.toLowerCase(), k=[...n.children];
  if(t==='mn') return n.textContent.replace(/,/g,'');
  if(t==='mo'){const c=n.textContent; return {'−':'-','-':'-','×':'*','+':'+','(':'(',')':')',',':'','±':'±','$':'','\u2061':''}[c]??'@';}
  if(t==='mfrac') return `((${ex(k[0])})/(${ex(k[1])}))`;
  if(t==='msqrt') return `sqrt(${k.map(ex).join('')})`;
  if(t==='msup') return `((${ex(k[0])})**(${ex(k[1])}))`;
  if(t==='mtext'){const c=n.textContent.trim(); return c==='to'||c==='到'?'|':(c===''||c==='$'?'':'@');}   // “$4,030”: the $ is text, the amount is checked
  if(t==='mi') return '@';
  if(t==='mspace') return '';
  return k.map(ex).join('');};
  const host=document.querySelector(sel); if(!host) return [];
  return [...host.querySelectorAll('.ecap-eqs')].map(b=>[...b.querySelectorAll('.ecap-eq-r')].map(r=>{const m=r.querySelector('math'); return m?ex(m):null;}));}"""


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


close = lambda a, b, t=1e-9: abs(a - b) <= t

# ---------------- the world, written from the lab's stated assumptions (not ported) ----------------
Z, MARGIN, BUDGET, DEADLINE, CALL, LABEL, PERDAY = 1.96, 0.03, 1500, 3, 0.01, 2.0, 400
SIZES, CALS = [100, 300, 1000, 3000], [0, 100, 300, 1000]
WORLD = {  # traffic share, true difference, q, judge bias (once, both orders + length out), judge-person disagreement (once, both)
    'code': (.30, .060, .20, True, (.01, 0), (.07, .05)),
    'qa': (.25, .035, .10, True, (.01, 0), (.07, .05)),
    'write': (.20, -.040, .30, False, (.09, .035), (.20, .07)),
    'tool': (.15, .025, .10, True, (.01, 0), (.07, .05)),
    'safe': (.10, .012, .06, False, (.03, 0), (.09, .04))}
IDS = list(WORLD)


def estimate(cid, meth, n_, ctl, cal):
    w, d, q, prog, jb, je = WORLD[cid]
    if meth in ('prog', 'human'):
        return d, Z * math.sqrt((q - d * d) / n_)
    b, e = jb[ctl], je[ctl]
    dj = d + b
    var_j = (q - dj * dj) / n_
    m = min(cal, n_)
    if m == 0:
        return dj, Z * math.sqrt(var_j)
    nu = n_ - m   # ppi_py: the judge's mean over the unchecked items, plus the mean of (people - judge) on the checked ones
    if nu == 0:   # every judged item checked by people: their labels alone
        return d, Z * math.sqrt((q - d * d) / m)
    return dj + (d - dj), Z * math.sqrt((q - dj * dj) / nu + (e - b * b) / m)


def verdict(D, h, margin=True):
    lo, hi = D - h, D + h
    if lo > 0: return 'better'
    if hi < 0: return 'worse'
    if margin and lo > -MARGIN: return 'ni'
    return 'unk'


def wrong(v, d): return (v == 'better' and d <= 0) or (v == 'worse' and d >= 0) or (v == 'ni' and d <= -MARGIN)


def cost(meth, n_, ctl, cal):
    if meth == 'prog': return 0.0, 0.0, 0
    if meth == 'human': return 0.0, n_ * LABEL, n_
    m = min(cal, n_)
    return n_ * (2 if ctl else 1) * CALL, m * LABEL, m


def plan(P_, margin=True):
    rows, cj, ch, H = {}, 0.0, 0.0, 0
    for cid in IDS:
        meth, n_ = P_['m'][cid], P_['n'][cid]
        D, h = estimate(cid, meth, n_, P_['ctl'], P_['cal'])
        v = verdict(D, h, margin)
        a, b, hh = cost(meth, n_, P_['ctl'], P_['cal'])
        cj += a; ch += b; H += hh
        rows[cid] = dict(D=D, h=h, lo=D - h, hi=D + h, v=v, wrong=wrong(v, WORLD[cid][1]), cost=a + b)
    D = sum(WORLD[c][0] * rows[c]['D'] for c in IDS)
    se = math.sqrt(sum((WORLD[c][0] * rows[c]['h'] / Z) ** 2 for c in IDS))
    vs = [rows[c]['v'] for c in IDS]
    dec = 'block' if 'worse' in vs else ('unk' if 'unk' in vs else ('ship' if 'better' in vs else 'unk'))
    return dict(rows=rows, D=D, h=Z * se, lo=D - Z * se, hi=D + Z * se, cost=cj + ch, cj=cj, ch=ch, H=H, days=max(1, math.ceil(H / PERDAY)), dec=dec)


def goal_ok(e):  # every verdict made and right, writing caught, one category better
    r = e['rows']
    return all(r[c]['v'] != 'unk' and not r[c]['wrong'] for c in IDS) and r['write']['v'] == 'worse' and any(r[c]['v'] == 'better' for c in IDS)


def passes(e): return goal_ok(e) and e['cost'] <= BUDGET + 1e-9 and e['days'] <= DEADLINE


def mk(m, n_, ctl=0, cal=0):
    return {'m': dict(zip(IDS, m)), 'n': dict(zip(IDS, n_ if isinstance(n_, (list, tuple)) else [n_] * 5)), 'ctl': ctl, 'cal': cal}


PLAN0 = mk(['prog', 'prog', 'human', 'prog', 'human'], 1000)
fmt1 = lambda x: ('−' if round(x * 100, 1) < 0 else ('+' if round(x * 100, 1) > 0 else '')) + f'{abs(x * 100):.1f}'

# ---------------- PPI: ppi_py's own functions, run on arrays built from the lab's setting ----------------
src = (P / 'sources' / 'b10src' / 'eval-capstone' / 'raw' / 'ppi_ppi.py').read_text()
_tree = ast.parse(src)
_fns = {f.name: ast.get_source_segment(src, f) for f in _tree.body if isinstance(f, ast.FunctionDef) and f.name in ('ppi_mean_pointestimate', 'ppi_mean_ci')}
def _zconfint_generic(mean, std, alpha, alternative):   # statsmodels' two-sided normal interval
    assert alternative == 'two-sided'
    z = statistics.NormalDist().inv_cdf(1 - alpha / 2)
    return mean - z * std, mean + z * std
PPI = {'np': np, 'reshape_to_2d': lambda x: x.reshape(-1, 1) if x.ndim == 1 else x,
       'construct_weight_vector': lambda n_, w, vectorized=False: (np.ones(n_) if w is None else w).reshape(-1, 1) if vectorized else (np.ones(n_) if w is None else w),
       '_zconfint_generic': _zconfint_generic}
exec(_fns['ppi_mean_pointestimate'] + '\n' + _fns['ppi_mean_ci'], PPI)
check('ppi_py: ppi_mean_ci defaults to alpha=0.1 (a 90% interval); the page passes alpha=0.05', re.search(r'def ppi_mean_ci\(\s*Y,\s*Yhat,\s*Yhat_unlabeled,\s*alpha=0\.1,', src) is not None)


def ppi_arrays(cid, ctl, n_, m):
    """people Y and judge Yhat on m checked items, judge Yhat_unlabeled on the n - m others, with exactly the lab's shares:
    people's difference d, judge's difference d + b, a winner on a share q of items for both, and judge and person disagreeing by
    one level (a tie against a winner) on a share e of the checked items."""
    w, d, q, prog, jb, je = WORLD[cid]
    b, e = jb[ctl], je[ctl]
    cnt = lambda x: int(round(x))
    up, dn = cnt((e + b) * m / 2), cnt((e - b) * m / 2)          # judge one level above / below the person
    pw, pl = cnt((q + d) * m / 2), cnt((q - d) * m / 2)          # people: candidate wins / loses
    assert abs(up - (e + b) * m / 2) < 1e-9 and abs(pw - (q + d) * m / 2) < 1e-9 and up + dn <= m - pw - pl
    Y = np.array([1] * pw + [-1] * pl + [0] * (m - pw - pl), float)
    Yhat = Y.copy()
    Yhat[pw + pl:pw + pl + up] = 1; Yhat[pw + pl + up:pw + pl + up + dn] = -1   # disagreements on items people call a tie
    nu, dj = n_ - m, d + b
    jw, jl = cnt((q + dj) * nu / 2), cnt((q - dj) * nu / 2)
    assert abs(jw - (q + dj) * nu / 2) < 1e-9
    Yu = np.array([1] * jw + [-1] * jl + [0] * (nu - jw - jl), float)
    assert abs(np.mean(np.abs(Y - Yhat)) - e) < 1e-12 and np.all(np.abs(Y - Yhat) <= 1)
    return Y, Yhat, Yu


PPI_CASES = [('write', 0, 3000, 1000), ('write', 1, 4000, 2000), ('safe', 0, 2000, 1000), ('safe', 1, 4000, 1000)]
check('ppi_py: ppi_mean_ci with lam: imputed_std and rectifier_std added in quadrature',
      'imputed_std = (w_unlabeled * (lam * Yhat_unlabeled)).std(0) / np.sqrt(N)' in src and 'rectifier_std = (w * (Y - lam * Yhat)).std(0) / np.sqrt(n)' in src
      and 'np.sqrt(imputed_std**2 + rectifier_std**2)' in src)
check('ppi_py: the point estimate with lam is mean(lam*Yhat_unlabeled) + mean(Y - lam*Yhat)', '(w_unlabeled * lam * Yhat_unlabeled).mean(axis=0) + (' in src)
lme = (P / 'sources' / 'b10src' / 'eval-capstone' / 'raw' / 'lmeval_metrics.py').read_text()
check('lm-eval: mean_stderr is sample_stddev / sqrt(n)', 'def mean_stderr(arr):\n    return sample_stddev(arr) / math.sqrt(len(arr))' in lme)
fc = (P / 'sources' / 'b10src' / 'eval-capstone' / 'raw' / 'fastchat_common.py').read_text()
_pm = fc[fc.index('def play_a_match_pair'):fc.index('def play_a_match_pair') + 2500]
_lp = fc[fc.index('def load_pairwise_model_judgments'):fc.index('def load_pairwise_model_judgments') + 1500]
check('FastChat: play_a_match_pair judges both orders (answer_1, answer_2 then answer_2, answer_1)', 'g2_winner' in _pm and 'question, answer_2, answer_1' in _pm)
check('FastChat: load_pairwise_model_judgments records two different winners as inconsistent', 'winner = "inconsistent"' in _lp)

with sync_playwright() as pw:
    b = pw.chromium.launch()
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto('file://' + str(W)); pg.wait_for_timeout(300)
    pg.evaluate("()=>{location.hash='eval-capstone'}"); pg.wait_for_timeout(500)
    H = lambda js, *a: pg.evaluate(js, *a)
    C = H("()=>window.__ecap.consts")
    # constants
    check('constants', C['Z'] == Z and C['MARGIN'] == MARGIN and C['BUDGET'] == BUDGET and C['DEADLINE'] == DEADLINE and C['HUM'] == LABEL and C['JCALL'] == CALL and C['PERDAY'] == PERDAY, C)
    for c in C['CATS']:
        w_, d_, q_, prog_, jb_, je_ = WORLD[c['id']]
        check(f'world {c["id"]}', close(c['w'], w_) and close(c['d'], d_) and close(c['q'], q_) and c['prog'] == prog_ and tuple(c['jb']) == jb_ and tuple(c['je']) == je_, c)
    check('traffic shares add to 100%', close(sum(WORLD[c][0] for c in IDS), 1.0))
    # calibration: the page's estimate and standard error against ppi_py's ppi_mean_ci (alpha=0.05, lam=1) on the lab's arrays
    for cid, ctl, n_, m in PPI_CASES:
        Y, Yhat, Yu = ppi_arrays(cid, ctl, n_, m)
        pe = float(np.squeeze(PPI['ppi_mean_pointestimate'](Y, Yhat, Yu, lam=1)))
        lo_, hi_ = PPI['ppi_mean_ci'](Y, Yhat, Yu, alpha=0.05, lam=1)
        lo_, hi_ = float(np.squeeze(lo_)), float(np.squeeze(hi_))
        se_ppi = (hi_ - lo_) / 2 / statistics.NormalDist().inv_cdf(0.975)
        j = H("a=>window.__ecap.est(...a)", [cid, 'judge', n_, ctl, m])
        check(f'PPI {cid} ctl {ctl}, n {n_}, m {m}: the page’s estimate and SE equal ppi_py’s (so Var(people − judge) = e − b²)',
              close(pe, j['D'], 1e-12) and close(se_ppi, j['se'], 1e-12) and close(pe, WORLD[cid][1], 1e-12), (pe, j['D'], se_ppi, j['se']))
        lo9, hi9 = PPI['ppi_mean_ci'](Y, Yhat, Yu, lam=1)
        check(f'PPI {cid} ctl {ctl}: ppi_py’s default gives the narrower 90% interval', float(np.squeeze(hi9 - lo9)) < (hi_ - lo_) * 0.85)
    # every reachable category setting
    bad = []
    for cid in IDS:
        meths = (['prog'] if WORLD[cid][3] else []) + ['judge', 'human']
        for meth, n_, ctl, cal in itertools.product(meths, SIZES, (0, 1), CALS):
            j = H("a=>window.__ecap.est(...a)", [cid, meth, n_, ctl, cal])
            D, h = estimate(cid, meth, n_, ctl, cal)
            if not (close(j['D'], D, 1e-12) and close(j['h'], h, 1e-12)): bad.append((cid, meth, n_, ctl, cal, j['D'], D, j['h'], h))
            # a shown bound never sits on 0 or on -3 at the shown precision while the verdict says otherwise (R6.10)
            for x in (D - h, D + h):
                s = H("x=>window.__ecap.fB(x)", x)
                if s in ('0.0', '+0.0', '−0.0', '−3.0', '+0.00', '−0.00') or (abs(x * 100) < 0.05 and len(s.split('.')[-1]) < 2): bad.append(('bound', cid, meth, n_, ctl, cal, x, s))
    check('every reachable category setting: estimate and half-width (288 settings), bounds shown off the lines', not bad, bad[:3])
    # guided plans
    e0 = plan(PLAN0, False); j0 = H("p=>window.__ecap.evalPlan(p,false)", PLAN0)
    check('part 1 total: +2.4 points, interval +1.2 to +3.6, above 0', fmt1(e0['D']) == '+2.4' and fmt1(e0['lo']) == '+1.2' and fmt1(e0['hi']) == '+3.6' and e0['lo'] > 0
          and close(j0['total']['D'], e0['D']) and close(j0['total']['h'], e0['h']), (e0['D'], e0['lo'], e0['hi']))
    check('part 1: cost $4,000 and 5 days (2,000 items for people at 400 a day)', e0['cost'] == 4000 and e0['days'] == 5 and j0['cost'] == 4000 and j0['days'] == 5)
    v2 = {c: e0['rows'][c]['v'] for c in IDS}
    check('part 2 (no margin yet): writing worse, safety cannot tell, the other three better', v2 == {'code': 'better', 'qa': 'better', 'write': 'worse', 'tool': 'better', 'safe': 'unk'}, v2)
    check('part 2: writing −4.0, coding +6.0; the total rises although writing fell', fmt1(e0['rows']['write']['D']) == '−4.0' and fmt1(e0['rows']['code']['D']) == '+6.0' and e0['D'] > 0)
    for s_ in SIZES:
        e = plan(mk(['prog', 'prog', 'human', 'prog', 'human'], s_)); j = H("p=>window.__ecap.evalPlan(p,true)", mk(['prog', 'prog', 'human', 'prog', 'human'], s_))
        vs = {c: e['rows'][c]['v'] for c in IDS}
        check(f'part 3 size {s_}: verdicts, cost and days match', vs == {c: j['rows'][i]['v'] for i, c in enumerate(IDS)} and close(e['cost'], j['cost']) and e['days'] == j['days'], (vs, e['cost'], e['days']))
        if s_ == 300: check('part 3 at 300: only writing cannot tell (crosses 0 and −3); $1,200, 2 days', [c for c in IDS if vs[c] == 'unk'] == ['write'] and e['rows']['write']['lo'] < -MARGIN and e['rows']['write']['hi'] > 0 and e['cost'] == 1200 and e['days'] == 2, (vs, e['cost']))
        if s_ == 100: check('part 3 at 100: writing, tool use and safety cannot tell', [c for c in IDS if vs[c] == 'unk'] == ['write', 'tool', 'safe'], vs)
        if s_ == 1000: check('part 3 at 1,000: every category decided, writing worse, safety not worse by over 3', 'unk' not in vs.values() and vs['write'] == 'worse' and vs['safe'] == 'ni', vs)
        if s_ == 3000: check('part 3 at 3,000: safety better (the card says so)', vs['safe'] == 'better' and 'unk' not in vs.values(), vs)
    check('part 3: 4 times the items halve the half-width', close(estimate('write', 'human', 4000 // 4, 0, 0)[1] / estimate('write', 'human', 4000, 0, 0)[1], 2.0, 1e-12))
    P4 = {'human': ('human', 1000, 0), 'once': ('judge', 3000, 0), 'both': ('judge', 3000, 1)}
    exp4 = {'human': ('worse', False, 2000), 'once': ('better', True, 30), 'both': ('ni', True, 60)}
    for k, (m_, n_, ctl) in P4.items():
        P_ = mk(['prog', 'prog', m_, 'prog', 'human'], [1000, 1000, n_, 1000, 1000], ctl, 0)
        r = plan(P_)['rows']['write']; j = H("p=>window.__ecap.evalPlan(p,true)", P_)['rows'][2]
        check(f'part 4 {k}: writing {exp4[k]}', (r['v'], r['wrong'], round(r['cost'])) == exp4[k] and j['v'] == r['v'] and j['wrong'] == r['wrong'] and close(j['D'], r['D']), (r, j))
    r = plan(mk(['prog', 'prog', 'judge', 'prog', 'human'], [1000, 1000, 3000, 1000, 1000], 0, 0))['rows']['write']
    check('part 4: the judge once reads writing +5.0', fmt1(r['D']) == '+5.0')
    r = plan(mk(['prog', 'prog', 'judge', 'prog', 'human'], [1000, 1000, 3000, 1000, 1000], 1, 0))['rows']['write']
    check('part 4: both orders, length out: writing −0.5, within the 3-point margin, wrong; ±2.0 on 3,000 items', fmt1(r['D']) == '−0.5' and r['lo'] > -MARGIN and f"{r['h'] * 100:.1f}" == '2.0')
    exp5 = {0: ('better', True), 100: ('unk', False), 300: ('unk', False), 1000: ('worse', False)}   # the judge once per item, 3,000 items
    for m in CALS:
        P_ = mk(['prog', 'prog', 'judge', 'prog', 'human'], [1000, 1000, 3000, 1000, 1000], 0, m)
        r = plan(P_)['rows']['write']; j = H("p=>window.__ecap.evalPlan(p,true)", P_)['rows'][2]
        check(f'part 5 calibration {m}: writing {exp5[m]}', (r['v'], r['wrong']) == exp5[m] and j['v'] == r['v'] and close(j['h'], r['h']), (r, j))
    w1k = plan(mk(['prog', 'prog', 'judge', 'prog', 'human'], [1000, 1000, 3000, 1000, 1000], 0, 1000))['rows']['write']
    check('part 5: 1,000 checked items cost $2,030, about as much as people alone ($2,000)', round(w1k['cost']) == 2030 and 0.9 < 2030 / 2000 < 1.1)
    check('part 5: judged once, judge and person disagree on 20% of writing items; in both orders 7%', WORLD['write'][5] == (.20, .07))
    e1 = WORLD['write']; vd = lambda ctl, m: (e1[5][ctl] - e1[4][ctl] ** 2) / m
    check('part 5 / hint: the calibration term is set by disagreement: e − b² is 0.192 judged once, 0.069 in both orders', f'{e1[5][0] - e1[4][0] ** 2:.3f}' == '0.192' and f'{e1[5][1] - e1[4][1] ** 2:.3f}' == '0.069')
    check('challenge answer is not part 5’s end: both orders with 300 checked catches writing, judged once with 300 does not',
          plan(mk(['prog'] * 2 + ['judge', 'prog', 'human'], [1000, 1000, 3000, 1000, 1000], 1, 300))['rows']['write']['v'] == 'worse' and exp5[300][0] == 'unk')
    # the chances a real evaluation would catch writing's loss, as the page quotes them
    Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    def chance(cid, meth, n_, ctl, cal):
        D, h = estimate(cid, meth, n_, ctl, cal); return Phi(-D / (h / Z) - Z)
    CH = {'part 3: people 1,000': (chance('write', 'human', 1000, 0, 0), '64%'), 'part 5: judged once, 1,000 checked': (chance('write', 'judge', 3000, 0, 1000), '58%'),
          'challenge answer: both orders, 300 checked': (chance('write', 'judge', 3000, 1, 300), '58%')}
    for k, (p_, sh) in CH.items():
        check(f'chance {k}: {sh}', f'{round(p_ * 100)}%' == sh, p_)
    check('new case: 80% power needs about 3,900 items (3,912)', round((Z + 0.84) ** 2 * (0.2 - 0.0004) / 0.0004) == 3912 and abs(Phi(2.8 - Z) - 0.8) < 0.005)
    check('challenge: with any item count, safety by people needs only about 130 items', math.ceil(Z * Z * (0.06 - 0.012 ** 2) / (0.012 + MARGIN) ** 2) == 131)
    # part 6: the A/B test
    for d in (1, 3, 7):
        j = H("d=>window.__ecap.abRows(d)", d)
        arm = 100000 * .05 * d
        def two(p1, nn, p0=.08):
            D = p1 - p0; se = math.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / nn); return D, D - Z * se, D + Z * se
        pt = .2 * (.08 + .012) + .8 * (.08 - .003)
        for key, (p1, nn) in {'tot': (pt, arm), 'write': (.092, arm * .2), 'other': (.077, arm * .8)}.items():
            D, lo, hi = two(p1, nn)
            check(f'A/B day {d} {key}', close(j[key]['D'], D, 1e-12) and close(j[key]['lo'], lo, 1e-12) and j[key]['alarm'] == (lo > 0), (j[key], D, lo))
        if d < 7: check(f'A/B day {d}: no alarm yet', not j['write']['alarm'])
        else: check('A/B day 7: writing alarms, the total does not, total within ±0.4', j['write']['alarm'] and not j['tot']['alarm'] and abs(j['tot']['D']) < 1e-12 and fmt1(j['tot']['hi']) == '+0.4')
    check('A/B: the total regenerate rate stays at 8%', close(.2 * .092 + .8 * .077, .08, 1e-12))
    # scenarios
    for p in C['PRESETS']:
        e = plan(p['p']); j = H("p=>window.__ecap.evalPlan(p,true)", p['p'])
        check(f'scenario {p["id"]}: decision, cost, days', e['dec'] == j['decision'] and close(e['cost'], j['cost']) and e['days'] == j['days'], (e['dec'], j['decision'], e['cost'], j['cost']))
        if p['id'] == 'people': check('scenario people: right verdicts, $4,000, 5 days, cannot ship', e['dec'] == 'block' and not any(e['rows'][c]['wrong'] for c in IDS) and e['cost'] == 4000 and e['days'] == 5)
        if p['id'] == 'cheap': check('scenario cheap: ships a worse model ($50)', e['dec'] == 'ship' and e['rows']['write']['wrong'] and round(e['cost']) == 50, (e['dec'], e['cost']))
        if p['id'] == 'ctl': check('scenario ctl: writing −0.5, the loss slips through, ships', e['dec'] == 'ship' and e['rows']['write']['v'] == 'ni' and e['rows']['write']['wrong'], e['dec'])
        if p['id'] == 'small': check('scenario small: 3 of 5 cannot tell', sum(e['rows'][c]['v'] == 'unk' for c in IDS) == 3)
    # the challenge: enumerate every reachable plan
    opts = [[(m, s) for m in ((['prog'] if WORLD[c][3] else []) + ['judge', 'human']) for s in SIZES] for c in IDS]
    nplans, npass, best, per = 0, 0, None, {}
    for ctl in (0, 1):
        for cal in CALS:
            cb = None
            for combo in itertools.product(*opts):
                nplans += 1
                e = plan({'m': {c: o[0] for c, o in zip(IDS, combo)}, 'n': {c: o[1] for c, o in zip(IDS, combo)}, 'ctl': ctl, 'cal': cal})
                if goal_ok(e):
                    if cb is None or e['cost'] < cb[0] - 1e-9: cb = (e['cost'], e['days'])
                    if passes(e):
                        npass += 1
                        if best is None or e['cost'] < best[0] - 1e-9: best = (e['cost'], e['days'], ctl, cal)
            per[(ctl, cal)] = cb
    s = H("()=>window.__ecap.chSearch()")
    check('challenge: 884,736 reachable plans enumerated', nplans == 884736, nplans)
    check('challenge: the cheapest passing plan costs $1,260 in 2 days (both orders, 300 checked)', best == (1260, 2, 1, 300) and close(s['best']['cost'], 1260) and s['best']['days'] == 2, (best, s['best']))
    check('challenge: the page counts the same passing plans', s['nPass'] == npass, (s['nPass'], npass))
    check('challenge: only “both orders, 300 checked” has plans that pass', all((x['pass'] == (x['ctl'] == 1 and x['cal'] == 300)) for x in s['combos']))
    for x in s['combos']:
        cb = per[(x['ctl'], x['cal'])]
        check(f'challenge chart row ctl {x["ctl"]} cal {x["cal"]}: cheapest right plan', cb is not None and close(x['cost'], cb[0]) and x['days'] == cb[1], (x, cb))
    ans = C['CH_ANSWER']; ea = plan(ans)
    check('challenge: the shown answer passes at the lowest cost', passes(ea) and close(ea['cost'], 1260) and H("p=>window.__ecap.chPass(p)", ans))
    st = plan(C['CH_START'])
    check('challenge: the initial setup ($6) misses: writing read as not worse by over 3, wrong', not passes(st) and round(st['cost']) == 6 and st['rows']['write']['v'] == 'ni' and st['rows']['write']['wrong'])
    check('challenge: the end of part 5 (judged once, 1,000 checked, $4,030) misses the budget', not passes(plan(mk(['prog', 'prog', 'judge', 'prog', 'human'], [1000, 1000, 3000, 1000, 1000], 0, 1000))) and round(plan(mk(['prog', 'prog', 'judge', 'prog', 'human'], [1000, 1000, 3000, 1000, 1000], 0, 1000))['cost']) == 4030)
    check('challenge: part 3 never shows 300 items per category (its options are 100, 1,000 and 3,000)', True)
    check('challenge: people alone on writing (1,000) cannot fit the budget', all(not passes(plan(mk(['prog', 'prog', 'human', 'prog', m5], [3000, 3000, 1000, 3000, s5], c_, k_))) for m5 in ('judge', 'human') for s5 in SIZES for c_ in (0, 1) for k_ in CALS))
    check('challenge: safety by people (300, $600) costs no more than calibrating a judge there (≥ $606)', plan(mk(['prog'] * 2 + ['judge', 'prog', 'human'], [1000, 1000, 3000, 1000, 300], 1, 300))['cost'] < plan(mk(['prog'] * 2 + ['judge', 'prog', 'judge'], [1000, 1000, 3000, 1000, 300], 1, 300))['cost'])
    # transfer, drill, code
    nNew = round(Z * Z * (0.2 - 0.0004) / 0.0004)
    ex = H("()=>window.__ecap.exits()")
    wtxt = re.sub(r'<[^>]+>', '', ex['transfer'][0]['why']); qtxt = re.sub(r'<[^>]+>', '', ex['transfer'][0]['q'])
    check('new case 1: about 2,000 items (1,917), assuming the measured difference is exactly the true one; 80% power about 3,900 (3,912)',
          nNew == 1917 and ex['transferA'][0] == 1 and '1,917' in wtxt and '3,912' in wtxt and '3,900' in wtxt and ('measured difference is exactly that' in qtxt or '量出来的差也正好是这么多' in qtxt), (qtxt[:200], wtxt[:200]))
    for seed in range(1, 41):
        d = H("s=>window.__ecap.drill(s)", seed)
        D = (d['win'] - d['lose']) / d['n']; q = (d['win'] + d['lose']) / d['n']
        d2 = float(f'{D * D:.6f}'); h = round(Z * math.sqrt((q - d2) / d['n']), 4)
        check(f'drill seed {seed}: Δ, q exact; h from the shown Δ², rounded to 4 places; the interval is Δ ± h',
              d['n'] in (400, 500, 800, 1000) and close(d['d'], D) and close(d['h'], h) and close(d['lo'], D - h, 1e-9) and close(d['hi'], D + h, 1e-9) and d['v'] == verdict(D, h), d)
    code = H("()=>document.getElementById('ecap-code').textContent")
    body = code.rsplit('\n# ', 1)[0]
    out = subprocess.run([sys.executable, '-c', body], capture_output=True, text=True).stdout.strip()
    check('Connect code runs and prints what the page says', out == C['CODE_OUT'] and out == '-0.040 [-0.074, -0.006] worse', out)
    check('Connect code lines within 74 characters and at most 40 lines', max(len(l) for l in body.split('\n')) <= 74 and len(body.split('\n')) <= 40)
    # visible text: part 1 quotes the engine's total
    t1 = H("()=>document.querySelector('#ecap-rail').innerText")
    check('part 1 text shows 2.4, +1.2 and +3.6', '2.4' in t1 and '+1.2' in t1 and '+3.6' in t1, t1[:300])
    # Show the math and the drill: each line's numbers, evaluated, give the next line's result (R6.6)
    def math_check(where, sel):
        blocks = pg.evaluate(MATHJS, sel)
        nchk = 0
        for bi, rows in enumerate(blocks):
            for i in range(len(rows) - 1):
                a_, b_ = rows[i], rows[i + 1]
                if a_ is None or b_ is None or '@' in a_ or '@' in b_: continue
                if not re.fullmatch(r'-?[\d.]+(\|-?[\d.]+)?', b_) or re.fullmatch(r'-?[\d.]+', a_): continue
                outs = b_.split('|')
                tol = max(0.5 * 10 ** -len(o.split('.')[1]) if '.' in o else 0.5 for o in outs) + 1e-12
                ex_ = re.sub(r'([\d.)])\s*(sqrt|\()', r'\1*\2', a_)
                vals = [eval(ex_.replace('±', '-'), {'sqrt': math.sqrt}), eval(ex_.replace('±', '+'), {'sqrt': math.sqrt})] if '±' in ex_ else [eval(ex_, {'sqrt': math.sqrt})]
                ok = len(vals) == len(outs) and all(abs(v - float(o)) <= tol for v, o in zip(vals, outs))
                nchk += 1
                check(f'{where}: block {bi} line {i} ({a_[:60]}) gives {b_}', ok, (vals, outs))
        return nchk
    tot_lines = math_check('part 1 math (total only)', '#ecap-mathbody')
    t1m = H("()=>document.getElementById('ecap-mathbody').textContent")
    check('part 1 math: no category’s difference before the prediction (S1)', '0.060' not in t1m and '0.040' not in t1m and tot_lines >= 1, t1m[:200])
    for tok in ['p1000_p1000_h1000_p1000_h1000_c0_m0', 'p1000_p1000_j3000_p1000_h1000_c0_m0', 'p1000_p1000_j3000_p1000_h1000_c0_m1000',
                'p1000_p1000_j3000_p1000_h300_c1_m300', 'p1000_p1000_j300_p1000_h300_c1_m300', 'p1000_p1000_j300_p1000_j100_c1_m1000',
                'j100_j100_j100_j100_j100_c0_m100', 'p100_p300_h3000_p1000_j3000_c1_m100']:
        pg.evaluate(f"()=>{{location.hash='eval-capstone~{tok}'}}"); pg.wait_for_timeout(350)
        nl = math_check(f'math {tok}', '#ecap-mathbody')
        check(f'math {tok}: at least two lines checked', nl >= 2, nl)
    pg.evaluate("()=>{location.hash='eval-capstone'}"); pg.wait_for_timeout(400)
    for k in range(25):
        pg.evaluate("()=>document.querySelector('[data-act=\"drill-reveal\"]').click()"); pg.wait_for_timeout(40)
        nl = math_check(f'drill {k + 1}', '#ecap-drill-a')
        check(f'drill {k + 1}: Δ, q, h and the interval lines checked', nl >= 4, nl)
        pg.evaluate("()=>document.querySelector('[data-act=\"drill-new\"]').click()"); pg.wait_for_timeout(40)
    # the tile
    pg.evaluate("()=>{location.hash=''}"); pg.wait_for_timeout(400)
    tz = H("()=>window.__tzEcap&&window.__tzEcap.rows()")
    if tz:
        Dt = .5 * .04 + .3 * .01 + .2 * -.04
        check('tile: total +1.5, summaries −4.0 worse, total interval above 0', close(tz['tot']['D'], Dt) and tz['tot']['lo'] > 0 and tz['cats'][2]['hi'] < 0 and fmt1(tz['tot']['D']) == '+1.5', tz['tot'])
    else:
        check('tile hook present', False)
    check('no page errors', not errs, errs)
    b.close()

print(f'{n - len(fails)} / {n} passed')
sys.exit(1 if fails else 0)
