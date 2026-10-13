"""Numbers test for the llm-judge lab: an independent Python model of the judge (only the random number generator is ported) checked
against the page's engine (window.__ljg) and its home tile (window.__tzLjg), then every number the copy relies on.

- the seeded data (mulberry32, seven stratified columns, then the second person's draws) and the length shift;
- every data set × judge × order (B first, A first, random order, both orders): wins, ties, losses, inconsistent pairs, win rate,
  standard error, the length-controlled win rate (a fractional logistic fit by Newton's method), people's win rate, the agreement on the
  100 labelled pairs and the second person's agreement on the same pairs (the judge's ties left out of both), the cost;
- the calibration the parts rely on (R3.18): judged once B wins about 63% while people give about 49%; A first about 47%; both orders
  about 55% with 26% inconsistent; length taken out about 50%; agreement 64% / 72% / 79% against 62% between two people;
- random order: once per pair, a seeded coin per pair picks the order; it lands near both orders on the mission data;
- the scenarios' claims, the challenge (12 setups; exactly the four with length taken out and a random order or both orders pass; the
  cheapest is the small judge, random order, length taken out, 300 calls; every other setup misses by 5 points or more), the new case's
  answer, the drill, the tile's counts; the featured pair of each data set flips with the order and shows B about as much longer as the
  set's mean ratio;
- the Connect code runs and prints what its comments say;
- the quoted source lines exist in the local copies of the sources (b7src) or, failing that, are skipped with a note.
Run: python3 tests/llm-judge_numbers.py"""
import json, math, pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
(P / 'out' / 'llm-judge').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/llm-judge.html', '--only', 'llm-judge'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'llm-judge' / 'num-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"></head><body>' + (P / 'out' / 'llm-judge.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'llm-judge' / 'num-wrapped.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


# ---------------- the independent model ----------------
M32 = 0xffffffff
imul = lambda a, b: ((a & M32) * (b & M32)) & M32


def mulberry(a):
    st = [a & M32]
    def f():
        st[0] = (st[0] + 0x6D2B79F5) & M32; x = st[0]
        t = imul(x ^ (x >> 15), 1 | x)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296
    return f


sig = lambda x: 1 / (1 + math.exp(-x))
LQ = lambda u: math.log(u / (1 - u))
SN = math.sqrt(3) / math.pi
N, NL = 300, 100
SC = {'def': dict(seed=4, Q=0, Sq=2, R=1.4, Sl=0.4), 'chatty': dict(seed=4, Q=0, Sq=2, R=2.0, Sl=0.4), 'better': dict(seed=4, Q=0.6, Sq=2, R=1.0, Sl=0.4),
      'self': dict(seed=4, Q=0, Sq=2, R=1.0, Sl=0.4, self=0.6), 'math': dict(seed=4, Q=0, Sq=2, R=1.4, Sl=0.4, math=0.6),
      'ch': dict(seed=104, Q=0, Sq=2, R=1.8, Sl=0.4), 'nw': dict(seed=4, Q=-0.4, Sq=2, R=2.0, Sl=0.4)}
JUDGES = {'small': dict(lam=1, pi=0.5, st=0.9, se=0.4, tau=0.4, cost=1), 'big': dict(lam=1, pi=0.5, st=0.45, se=0.2, tau=0.4, cost=5)}


def perm(r, n):
    a = list(range(n))
    for i in range(n - 1, 0, -1):
        k = int(r() * (i + 1)); a[i], a[k] = a[k], a[i]
    return a


def data(seed, n=N):
    r = mulberry(seed * 9973 + 11)
    Pm = [perm(r, n) for _ in range(7)]
    u2 = [r() for _ in range(n)]
    ro = [r() for _ in range(n)]   # the random order's coin: B first when below 0.5
    q = lambda p, i: (p[i] + 0.5) / n
    return [dict(zq=LQ(q(Pm[0], i)) * SN, zl=LQ(q(Pm[1], i)) * SN, za=LQ(q(Pm[2], i)) * SN, u1=q(Pm[3], i), eAB=LQ(q(Pm[4], i)), eBA=LQ(q(Pm[5], i)),
                 et=LQ(q(Pm[6], i)), u2=u2[i], ro=ro[i]) for i in range(n)]


def lshift(Sl, n=N):
    return math.log(sum(math.exp(Sl * LQ((k + 0.5) / n) * SN) for k in range(n)) / n)


def base(sc):
    D = SC[sc]; rows = data(D['seed']); c = lshift(D['Sl']) if D['Sl'] else 0.0; out = []
    for i, x in enumerate(rows):
        if D.get('math'):
            cB = 1 if x['u1'] < D['math'] else 0; cA = 1 if x['u2'] < D['math'] else 0; g = 3 * (cB - cA); h = 0.5 if cB == cA else cB; h2 = h
        else:
            g = D['Q'] + D['Sq'] * x['zq']; h = 1 if x['u1'] < sig(g) else 0; h2 = 1 if x['u2'] < sig(g) else 0
        out.append(dict(g=g, l=math.log(D['R']) + D['Sl'] * x['zl'] - c, h=h, h2=h2, x=x))
    return out


def fit(v):
    a = b = 0.0
    for _ in range(60):
        ga = gb = haa = hab = hbb = 0.0
        for s, l in v:
            p = sig(a + b * l); w = p * (1 - p); ga += s - p; gb += (s - p) * l; haa += w; hab += w * l; hbb += w * l * l
        det = haa * hbb - hab * hab; da = (hbb * ga - hab * gb) / det; db = (-hab * ga + haa * gb) / det; a += da; b += db
        if abs(da) + abs(db) < 1e-13: break
    return a, b


def run(sc, j, order):
    B = base(sc); J = JUDGES[j]; slf = SC[sc].get('self', 0); v = []
    for p in B:
        x = p['x']
        def vote(o, e):
            s = p['g'] + J['lam'] * p['l'] + J['pi'] * o + slf + J['st'] * x['et'] + J['se'] * e
            return 1.0 if s > J['tau'] else (0.0 if s < -J['tau'] else 0.5)
        a, b = vote(1, x['eBA']), vote(-1, x['eAB'])
        if order == 'B': s = a
        elif order == 'A': s = b
        elif order == 'R': s = a if x['ro'] < 0.5 else b
        else: s = a if a == b else 0.5
        v.append(dict(s=s, a=a, b=b, l=p['l'], inc=order == 'both' and a != b))
    W = sum(1 for x in v if x['s'] == 1); T = sum(1 for x in v if x['s'] == 0.5); Lo = sum(1 for x in v if x['s'] == 0); inc = sum(1 for x in v if x['inc'])
    w = (W + T / 2) / N; sd = math.sqrt(sum((x['s'] - w) ** 2 for x in v) / (N - 1)); se = sd / math.sqrt(N)
    fa, fb = fit([(x['s'], x['l']) for x in v])
    keep = [i for i, x in enumerate(v[:NL]) if x['s'] != 0.5]          # the judge's ties are left out, for the judge and the second person
    ag = sum(1 for i in keep if v[i]['s'] == B[i]['h']); pag = sum(1 for i in keep if B[i]['h2'] == B[i]['h'])
    return dict(w=w, lc=sig(fa), a=fa, b=fb, H=sum(p['h'] for p in B) / N, W=W, T=T, Lo=Lo, inc=inc, sd=sd, se=se, agree=ag / len(keep), ag=ag, agN=len(keep),
                pag=pag, pagree=pag / len(keep), cost=N * (2 if order == 'both' else 1) * J['cost'])


pp = lambda sc: sum(1 for p in base(sc)[:NL] if p['h'] == p['h2']) / NL
R = lambda x: round(x * 100)
# ---------------- compare with the page ----------------
with sync_playwright() as pw:
    b = pw.chromium.launch(); pg = b.new_page(); errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    check('the lab registered its hook', pg.evaluate('()=>!!window.__ljg'))
    js = pg.evaluate('()=>window.__ljg.data(4).slice(0,8)'); py = data(4)[:8]
    check('the seeded data matches (mulberry32, stratified columns, second person)', all(abs(js[i][k] - py[i][k]) < 1e-12 for i in range(8) for k in py[i]), (js[0], py[0]))
    check('the length shift matches', abs(pg.evaluate('()=>window.__ljg.lshift(0.4)') - lshift(0.4)) < 1e-12)
    cache = {}
    for sc in SC:
        for j in JUDGES:
            for o in ('B', 'A', 'R', 'both'):
                a = pg.evaluate(f"()=>window.__ljg.run('{sc}','{j}','{o}')"); e = run(sc, j, o); cache[(sc, j, o)] = e
                for k in ('W', 'T', 'Lo', 'inc', 'ag', 'agN', 'pag', 'cost'):
                    check(f'{sc} {j} {o}: {k}', a[k] == e[k], (a[k], e[k]))
                for k, tol in (('w', 1e-12), ('H', 1e-12), ('se', 1e-12), ('agree', 1e-12), ('pagree', 1e-12), ('lc', 1e-9), ('a', 1e-8), ('b', 1e-8)):
                    check(f'{sc} {j} {o}: {k}', abs(a[k] - e[k]) < tol, (a[k], e[k]))
    check('two people agree as computed', abs(pg.evaluate("()=>window.__ljg.pp('def')") - pp('def')) < 1e-12)
    C = lambda sc, j, o: cache[(sc, j, o)]
    d1, dA, d2, g2 = C('def', 'small', 'B'), C('def', 'small', 'A'), C('def', 'small', 'both'), C('def', 'big', 'both')
    # ---------------- the guided parts' numbers (R3.18) ----------------
    check('part 1: judged once with B first, B wins about 63%', R(d1['w']) == 63, d1['w'])
    check('part 1: people give B about 49%, so “about 50%” is the right prediction', R(d1['H']) == 49 and pg.evaluate('()=>window.__ljg.consts.PRED_RIGHT') == '50', d1['H'])
    check('part 2: the 95% interval sits above 50% (the judge reliably prefers B)', d1['w'] - 1.96 * d1['se'] > 0.5, d1['w'] - 1.96 * d1['se'])
    check('part 3: A first gives about 47%', R(dA['w']) == 47, dA['w'])
    check('part 3: both orders give about 55%, with 26% inconsistent', R(d2['w']) == 55 and R(d2['inc'] / N) == 26, (d2['w'], d2['inc']))
    check('part 3: both orders still above 50% (“one preference is left”)', d2['w'] > 0.53)
    check('part 4: length taken out gives about 50%, within 1 point of people', R(d2['lc']) == 50 and abs(d2['lc'] - d2['H']) < 0.01, d2['lc'])
    dR = C('def', 'small', 'R')
    nB = sum(1 for x in data(4) if x['ro'] < 0.5)
    check('part 3: random order gives about 55% (near both orders), with 150 pairs B first', R(dR['w']) == 55 and abs(dR['w'] - d2['w']) < 0.01 and nB == 150 and pg.evaluate("()=>window.__ljg.nBfirst('def')") == nB, (dR['w'], nB))
    check('part 5: agreement once 64% (55/86), both orders 72% (48/67), large judge 79% (48/61)', (R(d1['agree']), R(d2['agree']), R(g2['agree'])) == (64, 72, 79)
          and (d1['ag'], d1['agN'], d2['ag'], d2['agN'], g2['ag'], g2['agN']) == (55, 86, 48, 67, 48, 61), (d1['agree'], d2['agree'], g2['agree']))
    check('part 5: the second person on the same pairs: 63% (54/86), 69% (46/67), 69% (42/61)', (R(d1['pagree']), R(d2['pagree']), R(g2['pagree'])) == (63, 69, 69)
          and (d1['pag'], d2['pag'], g2['pag']) == (54, 46, 42), (d1['pagree'], d2['pagree'], g2['pagree']))
    check('part 5: on the same pairs each judge agrees at least as often as the second person', all(x['agree'] > x['pagree'] for x in (d1, d2, g2)))
    check('part 5: the judge judged once agrees more often than a second person, yet is more than 10 points off', d1['agree'] > d1['pagree'] and abs(d1['w'] - d1['H']) > 0.10)
    # the featured pair of every data set: flips with the order, and B's shown length is within 10% of the set's mean ratio
    for sc in ('def', 'chatty', 'better', 'self', 'math', 'ch'):
        k = pg.evaluate(f"()=>window.__ljg.feat('{sc}')"); bs = base(sc); x = data(SC[sc]['seed'])[k]
        la = round(150 * math.exp(0.3 * x['za']) / 10) * 10; lb = round(la * math.exp(bs[k]['l']) / 10) * 10
        v = run(sc, 'small', 'both'); J = JUDGES['small']; p = bs[k]; slf = SC[sc].get('self', 0)
        sc_ = lambda o, e: p['g'] + p['l'] + 0.5 * o + slf + 0.9 * x['et'] + 0.4 * e
        a1 = 1 if sc_(1, x['eBA']) > 0.4 else 0; b1 = 1 if sc_(-1, x['eAB']) > 0.4 else 0
        w = pg.evaluate(f"()=>window.__ljg.words('{sc}',{k})")
        check(f'featured pair {sc}: flips with the order, B {lb / la:.2f}× A (mean {SC[sc]["R"]})', a1 == 1 and b1 == 0 and abs(lb / la / SC[sc]['R'] - 1) <= 0.1 and w == dict(la=la, lb=lb), (k, la, lb, w))
    check('the featured pair of the mission is part 2’s', pg.evaluate("()=>window.__ljg.consts.FEAT") == pg.evaluate("()=>window.__ljg.feat('def')"))
    # ---------------- scenarios ----------------
    ch_, cb_ = C('chatty', 'small', 'both'), C('chatty', 'small', 'B')
    check('chatty: once > both > people, and only length out comes close', cb_['w'] > ch_['w'] > ch_['H'] + 0.05 and abs(ch_['lc'] - ch_['H']) < abs(ch_['w'] - ch_['H']) and abs(ch_['lc'] - ch_['H']) <= 0.05,
          (cb_['w'], ch_['w'], ch_['lc'], ch_['H']))
    bt = C('better', 'small', 'both')
    check('better: both orders within 3 points of people, length out within 4', abs(bt['w'] - bt['H']) <= 0.03 and abs(bt['lc'] - bt['H']) <= 0.04 and bt['H'] > 0.55, (bt['w'], bt['lc'], bt['H']))
    sf = C('self', 'small', 'both')
    check('own family: neither fix removes it (5+ points above people)', sf['w'] - sf['H'] >= 0.05 and sf['lc'] - sf['H'] >= 0.05, (sf['w'], sf['lc'], sf['H']))
    mt = C('math', 'small', 'B')
    check('math: the checker gives about 50%, the judge once 5+ points more', R(mt['H']) == 50 and mt['w'] - mt['H'] >= 0.05, (mt['H'], mt['w']))
    # ---------------- the challenge ----------------
    opts = [(j, o, lc) for j in ('small', 'big') for o in ('B', 'R', 'both') for lc in (False, True)]
    res = {}
    for j, o, lc in opts:
        e = C('ch', j, o); v = e['lc'] if lc else e['w']; gap = abs(v - e['H']) * 100; res[(j, o, lc)] = (gap, gap <= 3 + 1e-9, e['cost'])
        a = pg.evaluate(f"()=>window.__ljg.chEval('{j}','{o}',{str(lc).lower()})")
        check(f'challenge {j} {o} {lc}: gap and pass match', abs(a['gap'] - gap) < 1e-9 and a['ok'] == (gap <= 3 + 1e-9), (a, gap))
    passing = [k for k, v in res.items() if v[1]]
    check('challenge: 12 setups; exactly the four “random order or both orders, length out” setups pass', len(opts) == 12 and sorted(passing) == sorted([(j, o, True) for j in ('small', 'big') for o in ('R', 'both')]), passing)
    check('challenge: every other setup misses by 5 points or more', all(v[0] >= 5 for k, v in res.items() if not v[1]), {k: round(v[0], 1) for k, v in res.items()})
    bst = pg.evaluate('()=>window.__ljg.chBest()')
    check('challenge: the cheapest passing setup is the small judge, random order, length out, 300 calls', (bst['j'], bst['o'], bst['lc'], bst['cost']) == ('small', 'R', True, 300), bst)
    check('challenge: its gap is 1.6 points; random order alone 8.2; both orders + length out 0.1 at 600', (round(res[('small', 'R', True)][0], 1), round(res[('small', 'R', False)][0], 1), round(res[('small', 'both', True)][0], 1), res[('small', 'both', True)][2]) == (1.6, 8.2, 0.1, 600),
          {k: round(v[0], 2) for k, v in res.items()})
    check('challenge: the initial setup fails', not res[('small', 'B', False)][1])
    check('challenge data differs from the guided data (R3.8)', C('ch', 'small', 'B')['w'] != d1['w'])
    # ---------------- capability layer, drill ----------------
    nw = C('nw', 'small', 'B'); idx = max(0, min(3, round(nw['w'] * 10) - 4))
    check('new case: the engine’s answer is “about 60%” and the page marks it', idx == 2 and pg.evaluate('()=>window.__ljg.exits().transferA[0]') == 2, (nw['w'], idx))
    check('new case: people give the new version less than the mission’s B', nw['H'] < d1['H'], nw['H'])
    # the new case as written (R3.19, the product owner's check of 2026-10-04): the reference the reader saw first (the mission: B 40%
    # longer, as good, the judge once gives 63%), then the new case (twice as long, a little worse, people 42%), then one question; no
    # “For reference” / “参照” label and no condition in parentheses; the reveal names the marked choice and the engine's numbers
    nwt = C('nw', 'small', 'both')
    for lang in ('en', 'zh'):
        pg.evaluate(f"()=>document.querySelector('.langsw button[data-lang=\"{lang}\"]').click()"); pg.wait_for_timeout(150)
        tq = pg.evaluate('()=>window.__ljg.exits().transfer[0]'); q, why, opts = tq['q'], tq['why'], tq['o']
        qm = '?' if lang == 'en' else '？'
        ref, new = q.find(f"{R(d1['w'])}%"), q.find(f"{R(nw['H'])}%")
        check(f'new case ({lang}): starts from the mission, the reference ({R(d1["w"])}%) before the new case ({R(nw["H"])}%), one question at the end',
              q.startswith('In the mission' if lang == 'en' else '任务里') and 0 <= ref < new and q.rstrip().endswith(qm) and q.count(qm) == 1, q)
        check(f'new case ({lang}): no reference label, no parentheses', not re.search(r'参照|for reference|reference:', q, re.I) and not re.search(r'[()（）]', q), q)
        check(f'new case ({lang}): the lengths are the data sets’ (mission B {round((SC["def"]["R"] - 1) * 100)}% longer, the new version twice as long)',
              f"{round((SC['def']['R'] - 1) * 100)}%" in q and SC['nw']['R'] == 2.0 and (('twice as long' in q) if lang == 'en' else ('长一倍' in q)), q)
        check(f'new case ({lang}): the reveal names the marked choice and the engine’s numbers (63, 62, 55, 41, 42)',
              opts[idx].lower() in why.lower() and all(f'{R(x)}%' in why for x in (d1['w'], nw['w'], nwt['w'], nwt['lc'], nw['H'])) and
              (R(d1['w']), R(nw['w']), R(nwt['w']), R(nwt['lc']), R(nw['H'])) == (63, 62, 55, 41, 42), why)
    pg.evaluate("()=>document.querySelector('.langsw button[data-lang=\"en\"]').click()"); pg.wait_for_timeout(100)
    for s in (2, 3, 4):
        dr = pg.evaluate(f'()=>window.__ljg.drill({s})')
        check(f'drill {s}: counts add up and the win rate is (bb + cc/2)/n', dr['bb'] + dr['aa'] + dr['cc'] == dr['n'] and abs(dr['w'] - (dr['bb'] + dr['cc'] / 2) / dr['n']) < 1e-12 and dr['cc'] > 0, dr)
    # ---------------- the home tile ----------------
    t = pg.evaluate('()=>window.__tzLjg&&window.__tzLjg.results()')
    rows = data(502, 100); once, both = [], []
    for x in rows:
        g = 2 * x['zq']   # g = Sq · zq with Sq = 2, Q = 0; equal lengths
        def vote(o, e):
            s = g + 0.5 * o + 0.9 * x['et'] + 0.4 * e
            return 1.0 if s > 0.4 else (0.0 if s < -0.4 else 0.5)
        a, b2 = vote(1, x['eBA']), vote(-1, x['eAB']); once.append(b2); both.append(a if a == b2 else 0.5)
    cnt = lambda v: dict(W=sum(1 for s in v if s == 1), T=sum(1 for s in v if s == 0.5), Lo=sum(1 for s in v if s == 0))
    co, cb = cnt(once), cnt(both)
    check('tile: once with A first, counts match', t and all(t['once'][k] == co[k] for k in co), (t, co))
    check('tile: both orders, counts match', t and all(t['both'][k] == cb[k] for k in cb), (t, cb))
    ao = (co['Lo'] + co['T'] / 2) / 100; ab = (cb['Lo'] + cb['T'] / 2) / 100
    check('tile: about 60 / 40 once, about 50 / 50 in both orders', 0.55 <= ao <= 0.62 and abs(ab - 0.5) <= 0.02, (ao, ab))
    check('tile: its numbers are not part 1’s (R3.8)', R(ao) != R(d1['w']) and R(1 - ao) != R(d1['w']))
    check('no page errors', not errs, errs)
    # ---------------- the Connect code ----------------
    src = (P / 'labs' / 'llm-judge.js').read_text()
    code = re.search(r"const CODE=`([\s\S]*?)`;", src).group(1); outs = re.search(r"const CODE_OUT=`([\s\S]*?)`;", src).group(1)
    lines = code.split('\n') + outs.split('\n')
    check('Connect code: at most 40 lines of at most 74 characters', len(lines) <= 40 and max(len(x) for x in lines) <= 74, (len(lines), max(len(x) for x in lines)))
    got = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True).stdout.split()
    want = [x.split()[1] for x in outs.split('\n')]
    check('Connect code prints what its comments say', got == want, (got, want))
    b.close()
# ---------------- the quoted sources (local copies) ----------------
S = P / 'sources' / 'b7src'
if S.exists():
    rd = lambda f: (S / f).read_text(errors='ignore')
    check('MT-Bench README: 3.3K annotations, over 80% agreement, as between humans', '3.3K human annotations' in rd('mtb_readme.md') and 'over 80\\% agreement, the same level of agreement between humans' in rd('mtb_readme.md'))
    check('common.py: two different winners are “inconsistent”', 'winner = "inconsistent"' in rd('mtb_common.py') and 'def play_a_match_pair' in rd('mtb_common.py'))
    check('show_result.py: win_rate_adjusted = (win + 0.5 tie) / all', 'df["win_rate_adjusted"] = (df["win"] + 0.5 * df["tie"])' in rd('show_result.py'))
    check('compute_agreement.py leaves ties out', 'ban=["tie"]' in rd('mtb_agree.py'))
    al = rd('alpaca_readme.md')
    check('AlpacaEval: 65.7 (humans), 62.2 (longest), 0.64 humans prefer longer, 0.93 → 0.98', re.search(r'\|\s*humans\s*\|\s*65\.7', al) and re.search(r'\|\s*longest\s*\|\s*62\.2', al) and 'from 0.93 to 0.98' in al and re.search(r'humans[^\n]*0\.64\s*\|', al))
    check('rating_systems.py: compute_bt and compute_style_control', 'def compute_bt(' in rd('rating_systems.py') and 'def compute_style_control(' in rd('rating_systems.py'))
else:
    print('note: local source copies not found; source checks skipped')
print(f'{n - len(fails)}/{n} checks passed')
sys.exit(1 if fails else 0)
