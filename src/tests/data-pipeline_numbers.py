"""Golden numbers for the data-pipeline lab (guide R9.2, CONTRACT §6).

Derived here independently of the JS engine:
- the three made-up crawls: mulberry32 and the page generator re-implemented in Python from the lab's stated assumptions (the counts
  of each kind of page, the ranges of each page's numbers, the seeds), then the four stages with FineWeb's thresholds: no main text;
  English score < threshold; the five rules in datatrove's order (repeated lines > 30%, words < 50, symbols per word > 0.1, stop
  words < 2 of 8, lines ending in punctuation < 12%); exact copies by hash, near copies by MinHash (caught iff u < P(s));
- MinHash: P(s) = 1 − (1 − s^r)^b and the steepest point (1/b)^(1/r), checked against datatrove's own comment (0.72 and 0.924) and a
  Monte Carlo of real signatures (sha1 over 5-grams, as in the Connect card);
- the mix: epochs = share × 15T ÷ tokens on hand.
Checks: part 1 (the count left, the right answer of the prediction, the bins), part 2 (threshold 0.65 and 0.90), part 3 (rule
counts, the switch numbers, one rule off), part 4 (exact only vs MinHash 14 × 8), part 5 (four mixes), part 6, the four scenarios,
the challenge (20 settings, the passing ones, the cheapest, FineWeb's; the result after Submit of owner's review b8, R1: the shares,
the main gap and the one change that helps most, recomputed here for all 20; no exact-value hover, the owner's clarification at
23:37 UTC), the capability layer's numbers (new case 1
asks how many of 100 reposts each setting misses, owner's review b8), the interview drill, the
home tile, the Connect card's code (run here) and every number the page prints in those places.
Run: python3 tests/data-pipeline_numbers.py"""
import math, pathlib, subprocess, sys, io, contextlib, re, hashlib, random
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/data-pipeline.html', '--only', 'data-pipeline'], check=True, cwd=P, stdout=subprocess.DEVNULL)
URL = 'file://' + str(P / 'out' / 'data-pipeline.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ---------------------------------------------------------------- the crawls, re-implemented
M32 = 0xFFFFFFFF


def imul(a, b):
    r = (a & M32) * (b & M32) & M32
    return r - (1 << 32) if r & 0x80000000 else r


def mulberry(seed):
    st = [seed & M32]
    def nxt():
        st[0] = (st[0] + 0x6D2B79F5) & M32
        a = st[0] if st[0] < 0x80000000 else st[0] - (1 << 32)
        u = a & M32
        t = imul(u ^ (u >> 15), 1 | u) & M32
        t = ((t + (imul(t ^ (t >> 7), 61 | t) & M32)) & M32) ^ t
        t &= M32
        return ((t ^ (t >> 14)) & M32) / 4294967296
    return nxt


def jsround(x):
    return math.floor(x + 0.5)


CORPORA = {
    'web': (11, dict(nomain=150, foreign=330, mixed=20, enlow=10, rep=35, short=55, sym=20, kw=30, menu=40, seo=6, uniq=182, alike=20, exact=45, near=45, border=12)),
    'foreign': (23, dict(nomain=150, foreign=640, mixed=30, enlow=5, rep=10, short=15, sym=5, kw=10, menu=10, seo=2, uniq=91, alike=5, exact=12, near=12, border=3)),
    'code': (37, dict(code=600, gen=60, cexact=220, cnear=120))}
JUNK = {'nomain', 'foreign', 'mixed', 'rep', 'short', 'sym', 'kw', 'menu', 'seo', 'gen'}
GOODU = {'enlow', 'uniq', 'alike', 'code'}
COPY = {'exact', 'near', 'border', 'cexact', 'cnear'}
LANGS = ['es', 'de', 'fr', 'ru', 'zh', 'ja', 'pt', 'it']


def corpus(key):
    seed, cnt = CORPORA[key]
    r = mulberry(seed)
    U = lambda a, b: a + (b - a) * r()
    I = lambda a, b: math.floor(a + (b - a + 1) * r())
    F = lambda v, d: jsround(v * 10 ** d) / 10 ** d
    docs = []

    def base(k):
        d = dict(kind=k, noMain=False, lang='en')
        d['en'] = F(U(0.72, 0.99), 2); d['words'] = I(250, 1600); d['rep'] = F(U(0, 0.12), 2); d['sym'] = F(U(0, 0.03), 3); d['stop'] = I(5, 8); d['punct'] = F(U(0.45, 0.95), 2)
        d.update(dupOf=-1, sim=0, u=0)
        return d

    def add(k, f=None):
        for _ in range(cnt.get(k, 0)):
            d = base(k)
            if f: f(d)
            docs.append(d)

    def f_nomain(d): d['noMain'] = True; d['words'] = 0
    def f_foreign(d): d['lang'] = LANGS[I(0, 7)]; d['en'] = F(U(0, 0.25), 2); d['stop'] = 0; d['words'] = I(200, 1200)
    def f_mixed(d): d['lang'] = LANGS[I(0, 7)]; d['en'] = F(U(0.3, 0.8), 2); d['stop'] = I(0, 3); d['words'] = I(100, 800)
    def f_enlow(d): d['en'] = F(U(0.45, 0.64), 2); d['words'] = I(150, 600)
    def f_rep(d):
        d['en'] = F(U(0.8, 0.99), 2); d['words'] = I(80, 400); d['rep'] = F(U(0.35, 0.8), 2); d['stop'] = I(3, 6)
        d['punct'] = F(U(0.03, 0.11), 2) if r() < 0.5 else F(U(0.2, 0.5), 2)
    def f_short(d): d['en'] = F(U(0.7, 0.97), 2); d['words'] = I(8, 49); d['stop'] = I(0, 4); d['punct'] = F(U(0.3, 0.9), 2)
    def f_sym(d): d['sym'] = F(U(0.12, 0.45), 3); d['words'] = I(60, 300); d['stop'] = I(1, 4); d['punct'] = F(U(0.05, 0.4), 2)
    def f_kw(d): d['stop'] = I(0, 1); d['words'] = I(60, 400); d['punct'] = F(U(0, 0.3), 2); d['sym'] = F(U(0, 0.05), 3)
    def f_menu(d): d['punct'] = F(U(0, 0.11), 2); d['stop'] = I(2, 5); d['words'] = I(60, 300); d['rep'] = F(U(0.05, 0.25), 2)
    def f_seo(d): d['en'] = F(U(0.85, 0.97), 2)
    def f_code(d): d['en'] = F(U(0.3, 0.9), 2); d['words'] = I(50, 1500); d['rep'] = F(U(0, 0.35), 2); d['sym'] = F(U(0, 0.25), 3); d['stop'] = I(0, 4); d['punct'] = F(U(0, 0.35), 2)
    def f_gen(d): d['en'] = F(U(0.1, 0.6), 2); d['words'] = I(500, 5000); d['rep'] = F(U(0.3, 0.9), 2); d['sym'] = F(U(0, 0.4), 3); d['stop'] = 0; d['punct'] = F(U(0, 0.05), 2)
    for k, f in [('nomain', f_nomain), ('foreign', f_foreign), ('mixed', f_mixed), ('enlow', f_enlow), ('rep', f_rep), ('short', f_short), ('sym', f_sym), ('kw', f_kw),
                 ('menu', f_menu), ('seo', f_seo), ('code', f_code), ('gen', f_gen), ('uniq', None), ('alike', None)]:
        add(k, f)
    orig = [i for i, d in enumerate(docs) if d['kind'] in ('uniq', 'code')]

    def cp(k, s0, s1):
        for _ in range(cnt.get(k, 0)):
            o = orig[I(0, len(orig) - 1)]
            d = dict(docs[o]); d['kind'] = k; d['dupOf'] = o
            d['sim'] = 1 if s0 >= 1 else F(U(s0, s1), 2); d['u'] = r()
            docs.append(d)
    cp('exact', 1, 1); cp('near', 0.8, 0.98); cp('border', 0.65, 0.79); cp('cexact', 1, 1); cp('cnear', 0.8, 0.98)
    for d in docs:
        if d['kind'] == 'alike':
            d['dupOf'] = orig[I(0, len(orig) - 1)]; d['sim'] = F(U(0.45, 0.6), 2); d['u'] = r()
    ordr = list(range(len(docs)))
    for i in range(len(ordr) - 1, 0, -1):
        j = math.floor(r() * (i + 1)); ordr[i], ordr[j] = ordr[j], ordr[i]
    pos = [0] * len(docs)
    for p_, o in enumerate(ordr): pos[o] = p_
    out = [dict(docs[o]) for o in ordr]
    for d in out:
        if d['dupOf'] >= 0: d['dupOf'] = pos[d['dupOf']]
    return out


def PofS(s, b, r):
    return 1 - (1 - s ** r) ** b


RULES = [('rep', lambda d: d['rep'] > 0.3), ('short', lambda d: d['words'] < 50), ('sym', lambda d: d['sym'] > 0.1), ('stop', lambda d: d['stop'] < 2), ('punct', lambda d: d['punct'] < 0.12)]
BASE = dict(corpus='web', thr=0.65, rules=dict(rep=1, short=1, sym=1, stop=1, punct=1), exact=1, mh=1, b=14, r=8)
DOCS = {k: corpus(k) for k in CORPORA}


def fates(**o):
    S = {**BASE, **o}; S['rules'] = {**BASE['rules'], **o.get('rules', {})}
    D = DOCS[S['corpus']]; n_ = len(D); st = [5] * n_; why = ['kept'] * n_
    for i, d in enumerate(D):
        if d['noMain']: st[i] = 1; why[i] = 'nomain'; continue
        if d['en'] < S['thr'] - 1e-9: st[i] = 2; why[i] = 'lang'; continue
        for k, f in RULES:
            if S['rules'][k] and f(d): st[i] = 3; why[i] = k; break
    for i, d in enumerate(D):
        if st[i] != 5 or d['dupOf'] < 0 or st[d['dupOf']] < 4: continue
        if S['exact'] and d['sim'] >= 1: st[i] = 4; why[i] = 'exact'
        elif S['mh'] and d['u'] < PofS(d['sim'], S['b'], S['r']): st[i] = 4; why[i] = 'mh'
    red = [(d['kind'] in GOODU) if st[i] < 5 else (d['kind'] in JUNK or (d['kind'] in COPY and d['sim'] >= 0.8 and st[d['dupOf']] == 5)) for i, d in enumerate(D)]
    rm = [0] * 5
    for s in st:
        if s < 5: rm[s] += 1
    by = {k: sum(1 for i in range(n_) if st[i] == 3 and why[i] == k) for k, _ in RULES}
    dup = sum(1 for i, d in enumerate(D) if st[i] == 5 and d['kind'] in COPY and d['sim'] >= 0.8 and st[d['dupOf']] == 5)
    wall = sum(d['words'] for d in D); wk = sum(d['words'] for i, d in enumerate(D) if st[i] == 5)
    return dict(st=st, why=why, rm=rm, kept=n_ - sum(rm), byRule=by, dupLeft=dup, redKept=sum(1 for i in range(n_) if red[i] and st[i] == 5),
                redBin=sum(1 for i in range(n_) if red[i] and st[i] < 5), wShare=wk / wall, red=red)


def pred_of(k):
    return 'half' if k > 500 else ('third' if k >= 300 else ('tenth' if k >= 100 else 'few'))


# ---------------------------------------------------------------- MinHash, independently
check('datatrove comment: threshold (1/14)^(1/8) ≈ 0.72', round((1 / 14) ** (1 / 8), 2) == 0.72, (1 / 14) ** (1 / 8))
check('datatrove comment: P(0.8) at 14 × 8 = 0.924', round(PofS(0.8, 14, 8), 3) == 0.924, PofS(0.8, 14, 8))
check('RedPajama 9 × 13 catches about 40% at 0.8', round(PofS(0.8, 9, 13), 3) == 0.399, PofS(0.8, 9, 13))
check('9 × 13 rises near 0.84', round((1 / 9) ** (1 / 13), 2) == 0.84)
check('0.8^13 = 0.055', round(0.8 ** 13, 3) == 0.055)
check('0.8^8 = 0.17, 0.5^8 = 0.004', round(0.8 ** 8, 2) == 0.17 and round(0.5 ** 8, 3) == 0.004)


# Monte Carlo with real signatures: two sets of Jaccard s, B×R sha1 minhashes, candidate when one band agrees
def mc(s, b, r, trials=300):
    hits = 0
    for t in range(trials):
        nA = 60; shared = round(2 * nA * s / (1 + s)); a = set(range(shared)) | {('a', t, i) for i in range(nA - shared)}; bb = set(range(shared)) | {('b', t, i) for i in range(nA - shared)}
        def sig(x): return [min(int.from_bytes(hashlib.sha1(f'{k}:{t}:{e}'.encode()).digest()[:8], 'big') for e in x) for k in range(b * r)]
        sa, sb = sig(a), sig(bb)
        if any(sa[i * r:(i + 1) * r] == sb[i * r:(i + 1) * r] for i in range(b)): hits += 1
    return hits / trials


for s, tol in ((0.8, 0.07), (0.5, 0.06)):
    m = mc(s, 14, 8)
    nA = 60; shared = round(2 * nA * s / (1 + s)); sj = shared / (2 * nA - shared)
    check(f'Monte Carlo of real signatures agrees with P({s}) at 14 × 8', abs(m - PofS(sj, 14, 8)) < tol, (m, PofS(sj, 14, 8)))

# ---------------------------------------------------------------- the challenge grid
BS, RS = [14, 20, 34, 40, 50], [6, 8, 11, 12]
grid = [(b, r, b * r, PofS(0.8, b, r), PofS(0.5, b, r)) for b in BS for r in RS]
passing = [g for g in grid if g[3] >= 0.95 - 1e-12 and g[4] <= 0.02 + 1e-12]
best = sorted(passing, key=lambda g: g[2])[0]
check('challenge: three settings pass (34 × 11, 40 × 11, 50 × 12)', sorted((g[0], g[1]) for g in passing) == [(34, 11), (40, 11), (50, 12)], passing)
check('challenge: the cheapest is 34 × 11 = 374 hashes', (best[0], best[1], best[2]) == (34, 11, 374), best)
check('challenge: FineWeb 14 × 8 misses (92.4%, 5.3%)', round(PofS(0.8, 14, 8) * 100, 1) == 92.4 and round(PofS(0.5, 14, 8) * 100, 1) == 5.3)


# the result after Submit (owner's review b8, R1), recomputed here: a share as the lab writes it everywhere (one decimal, the same as the
# goal rows and the chart), the main gap, and the one change (only b or only r) that meets both goals with the fewest hashes (for a
# setting that already passes, only a change that saves hashes)
def ok_(b, r): return PofS(0.8, b, r) >= 0.95 - 1e-12 and PofS(0.5, b, r) <= 0.02 + 1e-12


def say(v, goal=None, up=None):
    return f'{v * 100:.1f}%'


def one_change(b, r):
    c = [('b', b2, r) for b2 in BS if b2 != b] + [('r', b, r2) for r2 in RS if r2 != r]
    c = [x for x in c if ok_(x[1], x[2]) and (not ok_(b, r) or x[1] * x[2] < b * r)]
    return min(c, key=lambda x: x[1] * x[2]) if c else None


def gap(b, r):
    hi, lo = PofS(0.8, b, r) >= 0.95 - 1e-12, PofS(0.5, b, r) <= 0.02 + 1e-12
    return None if hi and lo else ('steep' if not hi and not lo else ('right' if not hi else 'left'))


check('challenge result: from FineWeb’s 14 × 8 no single change meets both goals; from 20 × 11, 34 groups; from 40 × 11, 34 groups saves hashes; from 50 × 12 none saves',
      one_change(14, 8) is None and one_change(20, 11) == ('b', 34, 11) and one_change(40, 11) == ('b', 34, 11) and one_change(50, 12) is None)
check('challenge result: the main gap of 14 × 8 is a curve not steep enough; 20 × 11 too far right; 40 × 8 too far left', (gap(14, 8), gap(20, 11), gap(40, 8)) == ('steep', 'right', 'left'))

# ---------------------------------------------------------------- the mix
TARGET = 15e12; HAVE = [12e12, 4e12, 1.5e12, 0.3e12]
MIXES = {'prop': [h / sum(HAVE) for h in HAVE], 'more': [.65, .15, .15, .05], 'math15': [.55, .15, .15, .15], 'math40': [.35, .10, .15, .40]}
EP = {k: [sh * TARGET / h for sh, h in zip(v, HAVE)] for k, v in MIXES.items()}
check('mix: shares add to 1', all(abs(sum(v) - 1) < 1e-12 for v in MIXES.values()))
check('mix: math 40% → 20 epochs; 15% → 7.5; 5% → 2.5; in proportion 0.84', [round(EP[k][3], 2) for k in ('math40', 'math15', 'more', 'prop')] == [20.0, 7.5, 2.5, 0.84], [EP[k][3] for k in EP])
check('transfer: 20% of 10T over 0.5T = 4 epochs', 0.2 * 10 / 0.5 == 4)

# ---------------------------------------------------------------- the Connect card's code
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.evaluate("()=>{location.hash='data-pipeline'}"); pg.wait_for_function("()=>!!window.__dl&&!!document.querySelector('#view-dyn #dl-app #dl-rail')", timeout=30000); pg.wait_for_timeout(600)   # wait for the mounted lab, not only a fixed time
    H = lambda js: pg.evaluate('()=>' + js)
    C = H('window.__dl.consts')
    code = C['CODE']
    lines = code.split('\n')
    check('code: at most 30 lines, each at most 74 characters', len(lines) <= 30 and max(map(len, lines)) <= 74, (len(lines), max(map(len, lines))))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, {})
    check('code: prints {2}', buf.getvalue().strip() == '{2}', buf.getvalue())
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, ns)
    a = ns['a']; sh = ns['shingles']; j3 = len(sh(a) & sh(a[:900] + ' new ending')) / len(sh(a) | sh(a[:900] + ' new ending'))
    check('code comment: page 3 shares about 40% of 5-grams', round(j3, 1) == 0.4, j3)

    # the crawls: every page's numbers, as generated in Python
    for k in CORPORA:
        js = H(f"window.__dl.docs('{k}')")
        keys = ['kind', 'noMain', 'lang', 'en', 'words', 'rep', 'sym', 'stop', 'punct', 'dupOf', 'sim']
        same = len(js) == len(DOCS[k]) == 1000 and all(all((abs(x[q] - y[q]) < 1e-9 if isinstance(y[q], float) else x[q] == y[q]) for q in keys) and abs(x['u'] - y['u']) < 1e-12 for x, y in zip(js, DOCS[k]))
        check(f'crawl {k}: the 1,000 pages match the Python generator', same)

    def jsf(o):
        import json
        return H(f'window.__dl.fates({json.dumps(o)})')
    cases = [('web default', {}), ('thr 0.9', {'thr': 0.9}), ('thr 0.5', {'thr': 0.5}), ('stop off', {'rules': {'stop': 0}}), ('punct off', {'rules': {'punct': 0}}),
             ('exact only', {'mh': 0}), ('no dedup', {'mh': 0, 'exact': 0}), ('20 × 6', {'b': 20, 'r': 6}), ('foreign', {'corpus': 'foreign'}), ('code', {'corpus': 'code'})]
    PY = {}
    for name, o in cases:
        f = fates(**o); PY[name] = f; g = jsf(o)
        check(f'{name}: stage counts, kept, rules, copies left, wrong calls agree', f['rm'] == g['rm'] and f['kept'] == g['kept'] and f['byRule'] == g['byRule'] and f['dupLeft'] == g['dupLeft']
              and f['redKept'] == g['redKept'] and f['redBin'] == g['redBin'] and abs(f['wShare'] - g['wShare']) < 1e-12, (name, f['rm'], g['rm'], f['kept'], g['kept']))
    f0 = PY['web default']
    print('default:', f0['rm'], 'kept', f0['kept'], 'by rule', f0['byRule'], 'copies left', f0['dupLeft'], 'red', f0['redKept'], f0['redBin'], 'words', round(f0['wShare'], 3))
    check('part 1: the answer is “10% to 30%”', pred_of(f0['kept']) == 'tenth' and H('window.__dl.consts.PRED').count('tenth') == 1, f0['kept'])
    check('part 1: the largest bin is “not English”', max(range(1, 5), key=lambda k: f0['rm'][k]) == 2, f0['rm'])

    # part 1 on the page: the count, the share and the RedPajama-V2 numbers
    pg.evaluate("()=>document.querySelector('#dl-rail [data-pred=\"tenth\"]').click()")
    pg.wait_for_function('()=>window.__dl.state().g>=4', timeout=20000); pg.wait_for_timeout(300)
    obs = H("document.querySelector('#dl-rail [data-obs]').innerText")
    check('part 1 text: kept and its share', f"{f0['kept']} of the 1,000 pages are left, about {round(f0['kept'] / 10)}%" in obs, obs[:200])
    check('part 1 text: RedPajama-V2 100B docs, 84 snapshots, 30B with quality signals, 20B after dedup (README)', '100B' in obs and '84' in obs and '30B' in obs and '20B are left after dedup' in obs, obs[:400])
    mt = H("document.querySelector('#dl-mathbody').textContent")
    check('part 1 math: 1,000 − four bins = kept', f"1,000 − {f0['rm'][1]} − {f0['rm'][2]} − {f0['rm'][3]} − {f0['rm'][4]} = {f0['kept']}" in mt and 1000 - sum(f0['rm']) == f0['kept'], mt[:200])
    met = H("[...document.querySelectorAll('#dl-metrics .metric')].map(e=>e.querySelector('.v').textContent)")
    check('part 1 metrics: kept, words, copies left, wrong calls', met == [str(f0['kept']), f"{round(f0['wShare'] * 100)}%", str(f0['dupLeft']), str(f0['redKept'] + f0['redBin'])], met)

    def go(i):
        pg.evaluate(f"()=>document.querySelector('#dl-rail [data-go=\"{i}\"]')?.click()"); pg.wait_for_timeout(300)
    # part 2 at 0.65 and 0.9
    pg.evaluate("()=>document.querySelector('#dl-rail [data-nav=\"next\"]').click()"); pg.wait_for_timeout(300)
    D = DOCS['web']

    def lang_counts(thr):
        f = fates(thr=thr)
        wo = sum(1 for i, d in enumerate(D) if not d['noMain'] and d['lang'] == 'en' and d['kind'] in GOODU and f['why'][i] == 'lang')
        wi = sum(1 for d in D if not d['noMain'] and d['lang'] != 'en' and d['en'] >= thr - 1e-9)
        return f['rm'][2], wo, wi
    pg.evaluate("()=>{const i=document.querySelector('#dl-rail input[data-key=thr]'); i.value=90; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(300)
    obs = H("document.querySelector('#dl-rail [data-obs]').innerText")
    r90, wo, wi = lang_counts(0.9); r65 = lang_counts(0.65)[0]
    q = r90 / r65   # the change against 0.65 in words (fix S7): 676 / 352 = 1.92, “almost twice as many”
    word = 'almost twice as many' if 1.8 <= q < 1.97 else ('twice as many' if 1.97 <= q < 2.05 else f'{q:.1f} times as many')
    leak = 'No page in another language gets through.' if wi == 0 else f'{wi} pages in other languages get through.'
    check('part 2 text at 0.90: the change against 0.65 in words, good English thrown away, foreign let through', f'At a threshold of 0.90, {word} pages as at 0.65 are dropped as not English, and {wo} of them are good English pages. {leak}' in obs and (r90, wo, wi) == (676, 145, 0), (r90, wo, wi, obs[:300]))
    # part 3: one rule off
    pg.evaluate("()=>document.querySelector('#dl-rail [data-nav=\"next\"]').click()"); pg.wait_for_timeout(300)
    sw = H("[...document.querySelectorAll('#dl-rail .dl-sw em')].map(e=>e.textContent)")
    check('part 3 switches: the count each rule caught first', sw == [str(f0['byRule'][k]) for k in ('rep', 'short', 'sym', 'stop', 'punct')], sw)
    pg.evaluate("()=>document.querySelector('#dl-rail .dl-sw[data-rule=\"stop\"]').click()"); pg.wait_for_timeout(300)
    obs = H("document.querySelector('#dl-rail [data-obs]').innerText")
    fo = fates(rules={'stop': 0})
    was = [i for i in range(1000) if f0['st'][i] == 3 and f0['why'][i] == 'stop']
    later = sum(1 for i in was if fo['st'][i] == 3); kept = sum(1 for i in was if fo['st'][i] == 5)
    few = 'Only a few' if later / len(was) < 0.35 else 'Some'
    check('part 3 text with “too few stop words” off (fix S8)', kept > later and f"most of the pages that rule dropped get through. {few} are caught by a later rule, and the other {kept} stay in the data" in obs and (len(was), later, kept) == (34, 10, 24), (len(was), later, kept, obs[:300]))
    seo = sum(1 for i, d in enumerate(D) if d['kind'] == 'seo' and f0['red'][i] and f0['st'][i] == 5)
    check('part 3 text: red pages that pass every rule, and the SEO pages among them (fix S15)', f"{f0['redKept']} red pages stay even with every rule on, and {seo} of them are stitched-together SEO text" in obs and seo == 6, obs[-400:])
    # part 4
    pg.evaluate("()=>document.querySelector('#dl-rail [data-nav=\"next\"]').click()"); pg.wait_for_timeout(300)
    fe = PY['exact only']
    lv = H("[...document.querySelectorAll('#dl-rail [data-lever] .lv-r')].map(e=>e.textContent)")
    pg.evaluate("()=>document.querySelector('#dl-rail [data-lever=\"exact\"]').click()"); pg.wait_for_timeout(300)
    obs = H("document.querySelector('#dl-rail [data-obs]').innerText")
    check('part 4: exact copies only leaves the near copies (the example’s B first, review codex-b9v3)', f"Comparing whole-text hashes keeps B, the repost. Among the 1,000 pages, {fe['dupLeft']} near copies stay" in obs and fe['dupLeft'] > 30, (fe['dupLeft'], obs[:200]))
    pg.evaluate("()=>document.querySelector('#dl-rail [data-lever=\"both\"]').click()"); pg.wait_for_timeout(300)
    # fix S23: both sides of MinHash's trade-off, counted here: pairs that reach dedup; a heavily edited version (alike) caught = wrongly dropped
    def bad(b, r):
        f = fates(b=b, r=r); fn = fates(exact=0, mh=0)
        return sum(1 for i, d in enumerate(D) if d['kind'] == 'alike' and fn['st'][i] == 5 and fn['st'][d['dupOf']] == 5 and f['st'][i] == 4), f['dupLeft']
    b8, l8 = bad(14, 8); b3, l3 = bad(14, 3)
    obs = H("document.querySelector('#dl-rail [data-obs]').innerText")
    check('part 4 text at 14 × 8: no near copy left, only 2 heavily edited versions thrown away', (b8, l8) == (2, 0) and f'With MinHash at 14 groups of 8 hashes, no near copy stays, and only {b8} heavily edited versions that should stay are thrown away.' in obs, obs[:300])
    met4 = H("[...document.querySelectorAll('#dl-metrics .metric')].map(e=>e.querySelector('.v').textContent)")
    check('part 4 metrics: copies left, pages wrongly thrown away, hashes', met4 == [str(l8), str(b8), '112'], met4)
    pg.evaluate("()=>{const i=document.querySelector('#dl-rail input[data-key=r]'); i.value=3; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(300)
    obs = H("document.querySelector('#dl-rail [data-obs]').innerText")
    check('part 4 text at 14 × 3: also names the 16 pages wrongly thrown away (review: 16 of 20)', (b3, l3) == (16, 0) and f'but {b3} heavily edited versions that should stay are thrown away' in obs, obs[:300])
    pg.evaluate("()=>{const i=document.querySelector('#dl-rail input[data-key=r]'); i.value=8; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(300)
    mt = H("document.querySelector('#dl-mathbody').textContent")
    check('part 4 math: 0.8^8 and P(0.8), P(0.5), the steepest point', f"{0.8 ** 8:.4f}" in mt and f"{PofS(0.8, 14, 8):.3f}" in mt and f"{PofS(0.5, 14, 8):.3f}" in mt and f"{(1 / 14) ** (1 / 8):.3f}" in mt, mt[:300])
    # part 4's example (review codex-b9v3, 应修 3): three pages; 5-grams, Jaccard, the whole-text hash (FNV-1a, 32 bits) and a
    # seeded MinHash of 12 values in 3 groups of 4 (value j = min over 5-grams of fnv1a("seed:j:5-gram") mod 1000), recomputed here
    def fnv(x):
        h = 0x811c9dc5
        for c in x.encode('utf-8'): h ^= c; h = (h * 0x01000193) & 0xFFFFFFFF
        return h
    XPG = H('window.__dl.xpages()'); XSEED = H('window.__dl.xseed'); XR_ = {}
    def xpage(lg, k):
        p_ = XPG[lg][k]; body = p_.get('b') or XPG[lg]['A']['b']; toks = p_['t'] + body; w = [re.sub(r'[^\w]', '', t.lower()) for t in toks]
        st = {' '.join(w[i:i + 5]) for i in range(len(w) - 4)}; sep = '' if lg == 'zh' else ' '
        return dict(toks=toks, nt=len(p_['t']), set=st, sig=[min(fnv(f'{XSEED}:{j}:{x}') % 1000 for x in st) for j in range(12)], hash=fnv(sep.join(p_['t']) + '\n' + sep.join(body)))
    for lg in ('en', 'zh'):
        xa, xb, xc = (xpage(lg, k) for k in 'ABC')
        jb = len(xa['set'] & xb['set']) / len(xa['set'] | xb['set']); jc = len(xa['set'] & xc['set']) / len(xa['set'] | xc['set'])
        agB = [a_ == b_ for a_, b_ in zip(xa['sig'], xb['sig'])]; agC = [a_ == c_ for a_, c_ in zip(xa['sig'], xc['sig'])]
        grB = [all(agB[4 * g:4 * g + 4]) for g in range(3)]
        check(f'example {lg}: B is A with only the title changed, and its whole-text hash differs', xb['toks'][xb['nt']:] == xa['toks'][xa['nt']:] and xb['toks'][:xb['nt']] != xa['toks'][:xa['nt']] and xb['hash'] != xa['hash'] and xc['hash'] != xa['hash'])
        check(f'example {lg}: B shares 28 of 34 different 5-grams with A (Jaccard 0.82); C shares none', (len(xa['set'] & xb['set']), len(xa['set'] | xb['set'])) == (28, 34) and round(jb, 2) == 0.82 and jc == 0, (jb, jc))
        check(f'example {lg}: MinHash 3 × 4: B agrees with A in 10 values, in full in group 2 only; C in none', sum(agB) == 10 and grB == [False, True, False] and sum(agC) == 0, (agB, agC))
        js = H(f"window.__dl.x('{lg}')")
        check(f'example {lg}: the page computes the same hashes, signatures and shared 5-grams', [[v['v'] for v in js['pg'][k]['sig']] for k in 'ABC'] == [xa['sig'], xb['sig'], xc['sig']] and [js['pg'][k]['hash'] for k in 'ABC'] == [xa['hash'], xb['hash'], xc['hash']] and (js['B']['shared'], js['B']['union'], js['C']['shared']) == (28, 34, 0), js['B']['shared'])
        XR_[lg] = (xa, xb, xc, agB)
    xa, xb, xc, agB = XR_['en']
    # the similarity's hover (checker codex-b9v3): the idea, then the formula on its own line (MathML), with the counts recomputed here
    strip = lambda t: ' '.join(re.sub(r'<[^>]+>', ' ', t).split())
    tj, tc = H("window.__dl.xtip('xjB')"), H("window.__dl.xtip('xjC')")
    uB, uC = len(xa['set'] | xb['set']), len(xa['set'] | xc['set'])
    check('part 4 example: the similarity hovers show the formula with B’s 28 of 34 = 0.82 and C’s 0 of its union = 0', '<math' in tj and '<math' in tc
          and f"= {len(xa['set'] & xb['set'])} {uB} = {round(len(xa['set'] & xb['set']) / uB, 2)}" in strip(tj) and f"= 0 {uC} = 0" in strip(tc), (strip(tj)[-60:], strip(tc)[-60:]))
    cards = H("[...document.querySelectorAll('#dl-xbox .dl-xpg')].map(e=>e.innerText)")
    check('part 4 example on the page: the three hashes, B’s 28 shared 5-grams and 0.82, B thrown away by group 2, C kept', f"{xa['hash']:08x}" in cards[0] and f"{xb['hash']:08x}" in cards[1] and 'not A’s' in cards[1] and '28 of its 5-grams are also in A' in cards[1] and 'similarity 0.82' in cards[1]
          and 'All 4 values of group 2 agree' in cards[1] and f"{xc['hash']:08x}" in cards[2] and 'C stays' in cards[2], cards)
    cells = H("[...document.querySelectorAll('#dl-xbox .dl-xrow:not(.dl-xax)')].map(r=>[...r.querySelectorAll('.dl-xc')].map(c=>[+c.textContent,c.classList.contains('same')]))")
    check('part 4 example: the 12 values per page as printed, filled where B or C agrees with A, one group boxed', [[v for v, _ in r] for r in cells] == [xa['sig'], xb['sig'], xc['sig']] and [x for _, x in cells[1]] == agB and not any(x for _, x in cells[2])
          and H("[...document.querySelectorAll('#dl-xbox .dl-xg.hit')].map(g=>[...g.parentNode.children].indexOf(g))") == [2], cells)
    pg.evaluate("()=>document.querySelector('#dl-rail [data-lever=\"exact\"]').click()"); pg.wait_for_timeout(300)
    cards = H("[...document.querySelectorAll('#dl-xbox .dl-xpg')].map(e=>e.className+' | '+e.innerText)")
    obs = H("document.querySelector('#dl-rail [data-obs]').innerText")
    check('part 4 example, only exact copies: B stays, outlined in red, a wrong call; no group boxed', ' bad' in cards[1] and 'A wrong call: B is a repost.' in cards[1] and 'so C stays, which is right' in cards[2] and H("document.querySelectorAll('#dl-xbox .dl-xg.hit').length") == 0
          and f"Comparing whole-text hashes keeps B, the repost. Among the 1,000 pages, {fe['dupLeft']} near copies stay" in obs, (cards[1], obs[:200]))
    pg.evaluate("()=>document.querySelector('#dl-rail [data-lever=\"both\"]').click()"); pg.wait_for_timeout(300)
    obs = H("document.querySelector('#dl-rail [data-obs]').innerText")
    check('part 4 result with MinHash: the example first (B found by group 2, C stays), then the 1,000 pages', 0 <= obs.find('In the example, MinHash finds B: group 2 agrees in full. C stays.') < obs.find('no near copy stays'), obs[:200])
    # part 5
    pg.evaluate("()=>document.querySelector('#dl-rail [data-nav=\"next\"]').click()"); pg.wait_for_timeout(300)
    for v in ('more', 'math15', 'math40'):
        pg.evaluate(f"()=>document.querySelector('#dl-rail [data-lever=\"{v}\"]').click()"); pg.wait_for_timeout(250)
        js = H(f"window.__dl.epochs('{v}')")
        check(f'part 5 {v}: epochs per domain', all(abs(a['ep'] - b_) < 1e-12 for a, b_ in zip(js, EP[v])), js)
        ep = EP[v][3]; txt = H("document.querySelector('#dl-rail [data-obs]').innerText")
        cmp_ = {'more': 'not more than The Pile', 'math15': 'more than twice The Pile', 'math40': 'more than 6 times The Pile'}[v]   # 2.5, 7.5, 20 against 3
        check(f'part 5 {v}: the text prints math’s epochs and compares them with The Pile’s 3', (f'used {ep:.1f} times' if ep < 10 else f'used {ep:.0f} times') in txt and cmp_ in txt, txt[:300])
        tips = H("[...document.querySelectorAll('#dl-mix .dl-mrow')].map(e=>e.getAttribute('data-tipx'))")
        use = EP[v][3] * HAVE[3] / 1e12   # tokens of math the run takes, in T
        check(f'part 5 {v}: the math hover gives the tokens it takes exactly ({use:g}T, fix S9)', f'is {use:g}T tokens' in tips[3], tips[3])
    # part 6
    pg.evaluate("()=>document.querySelector('#dl-rail [data-nav=\"next\"]').click()"); pg.wait_for_timeout(300)
    txt = H("document.querySelector('#dl-rail .step.current').innerText")
    # G2 (the owner's message of 2026-10-08): the recap no longer quotes part 1's, part 4's and part 5's numbers; each rule says why it holds
    check('part 6: the three rules, each with why it holds, and no number that leans on an earlier part (G2)',
          'Expect most of a crawl to go.' in txt and 'start from several times as much raw web text' in txt
          and 'Remove near copies, not only exact ones.' in txt and 'Comparing hashes keeps such reposts; MinHash finds them.' in txt
          and 'A small domain can be raised, a little.' in txt and 'each extra pass teaches the model less' in txt
          and 'In part ' not in txt and 'of 1,000 pages were left' not in txt, txt[:400])
    # scenarios
    pg.evaluate("()=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"sandbox\"]:not([data-flow=free])')].find(e=>e.offsetParent!==null); b.click();}"); pg.wait_for_timeout(400)
    for i, (name, key) in enumerate((('web', 'web default'), ('foreign', 'foreign'), ('code', 'code'), ('exact', 'exact only'))):
        pg.evaluate(f"()=>document.querySelector('#dl-rail [data-preset=\"{i}\"]').click()"); pg.wait_for_timeout(400)
        if i > 0:   # fix S1: a new picture shows its result only once the run has ended (R3.8)
            t = H("document.querySelector('#dl-insight').innerText"); mv = H("[...document.querySelectorAll('#dl-metrics .metric .v')].map(e=>e.textContent)")
            check(f'scenario {name}: no result while the run replays (card and metrics)', 'once the run ends' in t and all(x == '—' for x in mv) and H("document.querySelector('#dl-status').textContent") == 'Running', (t[:120], mv))
            pg.evaluate("()=>{const i=document.querySelector('#dl-player input[data-key=scrub]'); i.value=400; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(300)
            dl = H("[...document.querySelectorAll('#dl-metrics .metric .d')].map(e=>e.textContent).join('')")
            check(f'scenario {name}: no ▲▼ after picking a scenario (the lab sets the state, R6.6)', dl == '', dl)
        t = H("document.querySelector('#dl-insight').innerText"); f = PY[key]
        if i == 0: ok = f"{f['kept']} of the 1,000 pages are left" in t
        elif i == 1: ok = f"Only {f['kept']} of 1,000 pages are left; the language stage alone drops {f['rm'][2]}" in t
        elif i == 2:
            Dc = DOCS['code']; good = sum(1 for d in Dc if d['kind'] == 'code'); lost = sum(1 for j, d in enumerate(Dc) if d['kind'] == 'code' and f['st'][j] < 5)
            ok = f'throw away {lost} of the {good} good source files' in t
        else: ok = f"{f['dupLeft']} near copies stay" in t and '69M' in t and '17M' in t
        check(f'scenario {name}: its numbers', ok, t[:240])
    # challenge
    pg.evaluate("()=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"challenge\"]')].find(e=>e.offsetParent!==null); b.click();}"); pg.wait_for_timeout(400)
    js = H('window.__dl.chAll()')
    check('challenge: 20 settings, as computed', len(js) == 20 and all(abs(a['ph'] - g[3]) < 1e-12 and abs(a['pl'] - g[4]) < 1e-12 and a['h'] == g[2] for a, g in zip(js, grid)))
    check('challenge: the cheapest', H('window.__dl.chBest()')['h'] == 374)
    card = H("document.querySelector('#dl-rail .railcard').innerText")
    check('challenge premise: FineWeb’s setting misses about 8% at 0.8 and catches about 5% at 0.5', f'misses about {round((1 - PofS(0.8, 14, 8)) * 100)}% of such pairs' in card and f'catches about {round(PofS(0.5, 14, 8) * 100)}% of the heavily edited versions' in card and 'only on the same topic' not in card, card[:400])
    jf = 2 * 0.5 / (1 + 0.5)   # fix B1: two pages of n 5-grams sharing x have J = x / (2n - x), so x / n = 2J / (1 + J)
    check('challenge goal: at similarity 0.5 two pages of the same length share about two thirds of their 5-grams (fix B1)', abs(jf - 2 / 3) < 1e-12 and 'share about two thirds of their 5-grams' in card, (jf, card[-500:]))
    # every setting: the engine's one change and main gap agree with the ones recomputed here
    jo = [(b_, r_, H(f'window.__dl.chOne({b_},{r_})'), H(f'window.__dl.chGap({b_},{r_})')) for b_ in BS for r_ in RS]
    bad1 = [(b_, r_, o, g) for b_, r_, o, g in jo if (None if o is None else (o['k'], o['e']['b'], o['e']['r'])) != one_change(b_, r_) or g != gap(b_, r_)]
    check('challenge result: the one change and the main gap of all 20 settings agree with the ones recomputed here', not bad1, bad1[:3])
    check('challenge result: the settings that pass, cheapest first', [(e['b'], e['r']) for e in H('window.__dl.chPass()')] == [(g[0], g[1]) for g in sorted(passing, key=lambda g: g[2])])
    for b_, r_, tier in ((14, 8, 'Not yet.'), (40, 11, 'Goal met, with 440 hashes per page. But a cheaper setting needs only 374.'), (34, 11, 'Goal met, with the fewest hashes: 374 per page.')):
        pg.evaluate(f"()=>document.querySelector('#dl-rail .seg-ctl[data-key=\"chb\"] button[data-v=\"{b_}\"]').click()"); pg.wait_for_timeout(150)
        v0 = H("document.querySelector('#dl-metrics .metric[data-k=ph] .v').textContent")
        pg.evaluate(f"()=>document.querySelector('#dl-rail .seg-ctl[data-key=\"chr\"] button[data-v=\"{r_}\"]').click()"); pg.wait_for_timeout(150)
        if (b_, r_) == (40, 11):   # fix S2: a change of a percentage is in points, against the last value shown (40 × 8 → 40 × 11)
            v1 = H("document.querySelector('#dl-metrics .metric[data-k=ph] .v').textContent"); dl = H("document.querySelector('#dl-metrics .metric[data-k=ph] .d').textContent")
            a_, b2 = float(v0.rstrip('%')), float(v1.rstrip('%'))
            check('challenge: ▲▼ of “caught at 0.8” in points (R6.6)', abs(a_ - PofS(0.8, 40, 8) * 100) < 0.051 and abs(b2 - PofS(0.8, 40, 11) * 100) < 0.051 and dl == f"▼ {abs(b2 - a_):.1f} pts vs last", (v0, v1, dl))
        pg.evaluate("()=>document.querySelector('#dl-rail [data-act=\"submit\"]')?.click()"); pg.wait_for_timeout(250)
        if (b_, r_) == (14, 8):   # a miss: the verdict and its gap first; the comparison on demand
            t = H("document.querySelector('#dl-ch-res').innerText")
            check('challenge 14 × 8: the verdict names both misses (92.4%, 5.3%), then the gap: a curve not steep enough', f"caught only {say(PofS(0.8, 14, 8))} of the time; the goal is at least 95%" in t
                  and f"caught {say(PofS(0.5, 14, 8))} of the time; the goal is at most 2%" in t and 'the curve is not steep enough' in t and 'Show the cheapest setting' in t, t[:400])
            pg.evaluate("()=>document.querySelector('#dl-rail [data-act=\"reveal\"]')?.click()"); pg.wait_for_timeout(250)
        t = H("document.querySelector('#dl-ch-res').innerText")
        best_ = sorted(passing, key=lambda g: g[2])[0]
        vals = [say(PofS(0.8, b_, r_), .95, True), say(PofS(0.5, b_, r_), .02, False), str(b_ * r_)]
        cells = H("[...document.querySelectorAll('#dl-ch-res .dl-c3r')].map(r=>[r.dataset.role,[...r.querySelectorAll('.dl-c3c b')].map(b=>b.textContent)])")
        check(f'challenge {b_} × {r_}: the right tier, your row with the shares and the hashes', tier in t and cells and cells[0][1] == vals and (cells[0][0] == ('mebest' if (b_, r_) == best_[:2] else 'me')), (t[:200], cells, vals))
        oc = one_change(b_, r_)
        nxt_ = {(14, 8): 'No single change meets both goals: the cheapest setting changes both.',
                (40, 11): f"The best single change is cutting the groups from 40 to {oc[1] if oc else '?'}: {40 * 11 - (oc[1] * oc[2] if oc else 0)} fewer hashes per page, and that is the cheapest setting."}.get((b_, r_))
        if nxt_: check(f'challenge {b_} × {r_}: the one change, computed', nxt_ in t, t[-500:])
        # the owner's clarification (23:37 UTC): no hover that only expands a shown number to its exact value; the shares carry no hover look
        tips_ = H("[...document.querySelectorAll('#dl-ch-res [data-tipx]')].map(e=>e.getAttribute('data-tipx'))")
        check(f'challenge {b_} × {r_}: no exact-value hover in the result, the shares are plain text', not any(re.search(r'精确值是|Exactly', x) for x in tips_)
              and H("document.querySelectorAll('#dl-ch-res .dl-c3c .def, #dl-ch-res .dl-tbl .def').length") == 0, tips_)
    check('challenge: the settings that pass, behind a closed disclosure; never all 20', H("(()=>{const d=document.querySelector('#dl-ch-res details.dl-all'); return !!d&&!d.open&&d.querySelectorAll('tbody tr').length===3&&/See all settings that pass \\(3\\)/.test(d.querySelector('summary').textContent)})()")
          and H("document.querySelectorAll('#dl-ch-res .cmp-row').length") == 0)

    # capability layer: the transfer numbers
    ex = H('window.__dl.exits()')
    # the new cases (fixes S5, S6): the reference first, RedPajama's 9 × 13 typeset, part 5's 7.5 epochs as the reference
    pg.evaluate("()=>{const b=[...document.querySelectorAll('#dl-rail [data-exits] .ex-tabs button')][1]; b&&b.click();}"); pg.wait_for_timeout(200)
    q1 = H("document.querySelector('#dl-rail [data-exits] .ex-body p b').textContent")
    # owner's review b8: the question asks how many of 100 reposts each setting misses, so the setting that is harder to match (9 × 13)
    # gets the bigger number: about 8 for FineWeb, about 60 for RedPajama's setting
    fwm, rpm = jsround((1 - PofS(0.8, 14, 8)) * 100), jsround((1 - PofS(0.8, 9, 13)) * 100)
    check('new case 1: FineWeb misses about 8 of 100 reposts at 0.8, 9 × 13 about 60: the harder setting gives the bigger number', (fwm, rpm) == (8, 60) and rpm > fwm, (fwm, rpm))
    check('new case 1: the 100 reposts first, then FineWeb’s setting and its misses, then RedPajama’s setting, then the question', q1.startswith('Take 100 reposts, each 0.8 similar to its original.') and f'FineWeb’s setting (14 groups of 8 hashes) misses about {fwm} of them' in q1 and q1.endswith('about how many does it miss?'), q1)
    opts1 = H("[...document.querySelectorAll('#dl-rail [data-exits] .ex-opts button')].map(b=>b.textContent)")
    check('new case 1: the options in the same unit (reposts missed of 100), the right one computed', len(opts1) == 3 and opts1[1].startswith(f'About {fwm},') and opts1[2].startswith(f'About {rpm}:'), opts1)
    pg.evaluate("()=>{const b=[...document.querySelectorAll('#dl-rail [data-exits] .ex-opts button')][2]; b&&b.click();}"); pg.wait_for_timeout(200)
    w1 = H("document.querySelector('#dl-rail [data-exits] .ex-res').textContent")
    check('new case 1: the answer as a rule, the curve rising near 0.84, about 60 missed, the arithmetic typeset with a name (60.1%)', f'near similarity {(1 / 9) ** (1 / 13):.2f}' in w1 and f'about {rpm} of the 100 reposts are missed' in w1
          and f'{(1 - PofS(0.8, 9, 13)) * 100:.1f}' in w1 and 'share missed' in w1 and '^' not in w1 and H("!!document.querySelector('#dl-rail [data-exits] .ex-res math')") and '✓' in w1, w1[:300])
    pg.evaluate("()=>{const b=[...document.querySelectorAll('#dl-rail [data-exits] button')].find(x=>/Next question/.test(x.textContent)); b&&b.click();}"); pg.wait_for_timeout(200)
    q2 = H("document.querySelector('#dl-rail [data-exits] .ex-body p b').textContent")
    check('new case 2: part 5’s 7.5 epochs given as the reference', f'the same math text is used {EP["math15"][3]:.1f} times' in q2, q2)
    check('exits: right answers (explain 0, 1; transfer 2, 1, 1)', ex['explainA'] == [0, 1] and ex['transferA'] == [2, 1, 1], ex)
    # drill
    for sd in range(1, 6):
        d = H(f'window.__dl.drill({sd})')
        check(f'drill {sd}: P and the steepest point', close(d['p'], PofS(d['s'], d['b'], d['r'])) and close(d['t'], (1 / d['b']) ** (1 / d['r'])), d)
    # home tile
    tz = H('window.__tzDl')
    if tz:
        for rr in range(2, 13):
            check(f'tile r = {rr}: P(0.8) and P(0.5)', close(pg.evaluate(f'()=>window.__tzDl.P(0.8,{rr})'), PofS(0.8, 20, rr)) and close(pg.evaluate(f'()=>window.__tzDl.P(0.5,{rr})'), PofS(0.5, 20, rr)))
        t6 = pg.evaluate('()=>window.__tzDl.read(6)')
        check('tile r = 6: 99.8% and 27.0%', '99.8%' in t6 and '27.0%' in t6, t6)
    # ---- G7 (the owner's message of 2026-10-08, from the model-parallel finding): walk every pick of the challenge, in both
    # languages, and check that a recommended single change makes the value the result shows strictly better at the precision shown
    def g7_walk(js):
        res = {}
        for lang in ('en', 'zh'):
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(150)
            res[lang] = pg.evaluate(js)
        pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(150)
        return res
    g7 = g7_walk("""()=>{const Z=window.__dl, C=Z.consts.CH; const R=/^The best single change is|^只改一项的话|^One change is enough|^只改一项就够了/; const f=s=>parseFloat(s); const out=[];
      C.Bs.forEach(b=>C.Rs.forEach(r=>{const t=Z.chNextText(b,r), s=Z.chShown(b,r), o=Z.chOne(b,r), rec=R.test(t); let ok=true, why='';
        if(rec){if(!o){ok=false; why='no engine change';} else {const q=Z.chShown(o.e.b,o.e.r);
          if(s.ok){ok=q.ok&&+q.h<+s.h&&t.includes(String(+s.h-+q.h)); why='hashes '+s.h+' -> '+q.h;}
          else {ok=q.ok&&(s.okHi||f(q.ph)>f(s.ph))&&(s.okLo||f(q.pl)<f(s.pl)); why='0.8: '+s.ph+' -> '+q.ph+', 0.5: '+s.pl+' -> '+q.pl;}}}
        if(/ 0 fewer|少算 0 /.test(t)){ok=false; why+=' (a zero gain)';}
        out.push({pick:[b,r],rec,ok,why,t});})); return out;}""")
    for lang in ('en', 'zh'):
        rows = g7[lang]; bad = [r for r in rows if not r['ok']]
        check(f'G7 {lang}: all 20 settings walked; every recommended single change saves hashes (or meets both goals where the setting missed)', len(rows) == 20 and not bad, bad[:3])
    check('no page errors', not errs, errs[:2])
    b.close()

print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:40] if fails else 'none')
sys.exit(1 if fails else 0)
