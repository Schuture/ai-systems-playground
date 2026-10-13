"""Numbers test for the chunk-rerank lab: an independent Python twin of the page's engine (tokens, chunks, BM25 as Lucene computes it,
the small hand-written encoder, reciprocal rank fusion, the stand-in reranker, recall@k, context tokens, the latency assumption, the
challenge's feasible settings) compared with the page for every chunking in both languages; the numbers in the page's text; the
interview drill; the lesson the challenge must teach (only hybrid search + rerank is cheapest, in both languages, and no guided state or
scenario shows it); and the quoted source wordings re-read from sources/b10src/chunk-rerank/raw/. The English stop words are read from
the pinned copy of Lucene's EnglishAnalyzer, not from the page.
Run: python3 tests/chunk-rerank_numbers.py"""
import math, pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
(P / 'out' / 'chunk-rerank').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/chunk-rerank.html', '--only', 'chunk-rerank'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'chunk-rerank' / 'num-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'chunk-rerank.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'chunk-rerank' / 'num-wrapped.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


# ---------- the twin ----------
K1, B, RRFK, NCAND, RRMS = 1.2, 0.75, 60, 5, 0.2
GOAL, CTX, LATMAX = 0.95, 200, 80
LAT = {'kw': 5, 'vec': 10, 'hyb': 15}
SIZES, OVF, RETS, KS = [16, 32, 64, 128, 'h'], [0, 0.25, 0.5], ['kw', 'vec', 'hyb'], [1, 3, 5]
EN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
ZH_RE = re.compile(r"[一-鿿]|[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
RAW = P / 'sources' / 'b10src' / 'chunk-rerank' / 'raw'
_ea = (RAW / 'lucene_english_analyzer.java').read_text()
STOP = set(re.findall(r'"([a-z]+)"', _ea[_ea.index('Arrays.asList('):_ea.index('final CharArraySet stopSet')]))
assert len(STOP) == 33, len(STOP)


class Twin:
    def __init__(self, data):
        self.docs, self.qs, self.lex = data['DOCS'], data['QS'], data['LEX']
        self.en, self.zh = {}, {}
        for i, (_, en, zh) in enumerate(self.lex):
            for w in en.split(): self.en[w] = i
            for w in zh.split(): self.zh[w.lower()] = i
        self.zmax = max(len(w) for w in self.zh)
        self.cache = {}

    @staticmethod
    def toks(s, lang):
        ms = list((ZH_RE if lang == 'zh' else EN_RE).finditer(s))
        return [(m.group(0), s[m.end():(ms[i + 1].start() if i + 1 < len(ms) else len(s))], i == len(ms) - 1) for i, m in enumerate(ms)]

    def build(self, lang):
        li = 1 if lang == 'zh' else 0; out = []
        for d in self.docs:
            t, secs, sents = [], [], []
            for h, ss in d[2]:
                s0 = len(t); t += self.toks(h[li], lang); row = []
                for x in ss:
                    a = len(t); t += self.toks(x[li], lang); row.append((a, len(t)))
                sents.append(row); secs.append((s0, len(t)))
            out.append((t, secs, sents))
        return out

    @staticmethod
    def cjk(w): return '一' <= w[0] <= '鿿'

    def terms(self, tk, lang):
        if lang != 'zh': return [w.lower() for w, _, _ in tk if w.lower() not in STOP]
        out, run = [], []
        def flush():
            if len(run) == 1: out.append(run[0])
            out.extend(run[i] + run[i + 1] for i in range(len(run) - 1)); run.clear()
        for w, tl, end in tk:
            if self.cjk(w):
                run.append(w)
                if tl.strip() or end: flush()
            else:
                flush(); out.append(w.lower())
        flush(); return out

    def meanings(self, tk, lang):
        v = [0] * len(self.lex)
        if lang != 'zh':
            for w, _, _ in tk:
                if w.lower() in self.en: v[self.en[w.lower()]] += 1
            return v
        s = ''
        for w, tl, end in tk:
            if self.cjk(w): s += w
            else:
                if w.lower() in self.zh: v[self.zh[w.lower()]] += 1
                s += '|'
            if tl.strip() or end: s += '|'
        i = 0
        while i < len(s):
            for L in range(min(self.zmax, len(s) - i), 0, -1):
                if s[i:i + L] in self.zh: v[self.zh[s[i:i + L]]] += 1; i += L; break
            else: i += 1
        return v

    @staticmethod
    def codes(tk): return {w.lower() for w, _, _ in tk if re.fullmatch(r'[A-Za-z]-\d+', w)}

    @staticmethod
    def unit(v):
        nn = math.sqrt(sum(x * x for x in v)); return [x / nn for x in v] if nn else list(v)

    def setup(self, lang, size, ov, tie='cos'):
        key = (lang, size, ov, tie)
        if key in self.cache: return self.cache[key]
        docs = self.build(lang); C = []
        for di, (t, secs, _) in enumerate(docs):
            if size == 'h':
                C += [(di, a, b) for a, b in secs]; continue
            step = size - round(size * OVF[ov]); s = 0
            while True:
                C.append((di, s, min(s + size, len(t))))
                if s + size >= len(t): break
                s += step
        ch = []
        for di, a, b in C:
            tk = docs[di][0][a:b]
            whole = [x for sa_, sb_ in sum(docs[di][2], []) if sa_ >= a and sb_ <= b for x in docs[di][0][sa_:sb_]]   # codes count only inside whole sentences
            ch.append({'d': di, 'a': a, 'b': b, 'terms': self.terms(tk, lang), 'm': self.meanings(tk, lang), 'codes': self.codes(whole)})
        N = len(ch); avg = sum(len(c['terms']) for c in ch) / N; df = {}
        for c in ch:
            for w in set(c['terms']): df[w] = df.get(w, 0) + 1
        Q = []
        ids = [d[0] for d in self.docs]; li = 1 if lang == 'zh' else 0
        for q in self.qs:
            qt = [(w, tl, e) for w, tl, e in self.toks(q[li], lang)]
            qterms = set(self.terms(qt, lang)); qm = self.meanings(qt, lang); qv = self.unit(qm); qc = self.codes(qt)
            kw = []
            for c in ch:
                sc = 0.0; dl = len(c['terms'])
                for w in qterms:
                    fq = c['terms'].count(w)
                    if fq: sc += math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5)) * fq / (fq + K1 * (1 - B + B * dl / avg))
                kw.append(sc)
            vec = [sum(x * y for x, y in zip(self.unit(c['m']), qv)) for c in ch]
            srt = lambda sc: [j for j in sorted(range(N), key=lambda j: (-sc[j], j)) if sc[j] > 0]   # a search returns only chunks scoring above 0
            lk, lv = srt(kw), srt(vec); rrf = [0.0] * N
            for lst in (lk, lv):
                for r, j in enumerate(lst[:NCAND], start=1): rrf[j] += 1 / (RRFK + r)   # RRF over each search's first NCAND
            lists = {'kw': lk, 'vec': lv, 'hyb': srt(rrf)}
            want = [i for i, x in enumerate(qm) if x]; den = max(1, len(want) + len(qc))
            cov = [(sum(1 for i in want if c['m'][i]) + len(qc & c['codes'])) / den for c in ch]
            rr, nc = {}, {}
            for r_, l in lists.items():   # rerank reads the first NCAND chunks the search returned
                cand = l[:NCAND]
                key2 = (lambda j: (-cov[j], -vec[j], cand.index(j))) if tie == 'cos' else (lambda j: (-cov[j], cand.index(j)))
                rr[r_] = sorted(cand, key=key2) + l[NCAND:]; nc[r_] = len(cand)
            d = ids.index(q[2]); sa = docs[d][2][q[3]][q[4]][0]; sb = docs[d][2][q[3]][q[5]][1]
            right = [j for j, c in enumerate(ch) if c['d'] == d and c['a'] <= sa and c['b'] >= sb]
            part = [j for j, c in enumerate(ch) if c['d'] == d and c['a'] < sb and c['b'] > sa]
            rk = {}
            for r_ in RETS:
                rk[r_ + '0'] = min((lists[r_].index(j) for j in right if j in lists[r_]), default=-1)
                rk[r_ + '1'] = min((rr[r_].index(j) for j in right if j in rr[r_]), default=-1)
            Q.append({'kw': kw, 'vec': vec, 'rrf': rrf, 'cov': cov, 'right': right, 'part': part, 'rk': rk, 'sa': sa, 'sb': sb, 'lists': lists, 'rr': rr, 'nc': nc})
        R = {'C': ch, 'N': N, 'longest': max(b - a for _, a, b in C), 'Q': Q, 'splits': sum(1 for q in Q if not q['right'])}
        self.cache[key] = R; return R

    def measure(self, lang, s, tie='cos'):
        R = self.setup(lang, s['size'], s['ov'], tie); key = s['ret'] + ('1' if s['rr'] else '0')
        hits = sum(1 for q in R['Q'] if 0 <= q['rk'][key] < s['k'])
        # rerank time: count, for every question, the candidates the first search really returned (at most NCAND); the slowest question counts
        nc = max(len(q['lists'][s['ret']][:NCAND]) for q in R['Q']) if s['rr'] else 0
        rer = nc * R['longest'] * RRMS
        return {'hits': hits, 'rec': hits / len(R['Q']), 'ctx': s['k'] * R['longest'], 'lat': LAT[s['ret']] + rer, 'rer': rer, 'nc': nc, 'splits': R['splits']}

    def challenge(self, lang, tie='cos'):
        feas = [s for s in settings([16, 32, 64, 128]) if (lambda m: m['rec'] >= GOAL - 1e-9 and m['ctx'] <= CTX and m['lat'] <= LATMAX + 1e-9)(self.measure(lang, s, tie))]
        minc = min(self.measure(lang, s, tie)['ctx'] for s in feas)
        return feas, [s for s in feas if self.measure(lang, s, tie)['ctx'] == minc], minc


def settings(sizes=SIZES):
    for size in sizes:
        for ov in ([0] if size == 'h' else [0, 1, 2]):
            for ret in RETS:
                for rr in (0, 1):
                    for k in KS:
                        yield {'size': size, 'ov': ov, 'ret': ret, 'rr': rr, 'k': k}


GUIDED_AND_SCENARIOS = []   # every state the guided parts and the scenarios put on the stage (read from the page below)

with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce')
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300); pg.evaluate("()=>{location.hash='chunk-rerank'}"); pg.wait_for_timeout(500)
    T = Twin(pg.evaluate("()=>window.__ckr.data()"))
    consts = pg.evaluate("()=>window.__ckr.consts")
    check('constants (k1, b, RRF 60, 5 candidates, 0.2 ms per token)', (consts['K1'], consts['BB'], consts['RRF_K'], consts['NCAND'], consts['RR_MS']) == (K1, B, RRFK, NCAND, RRMS), consts)
    check('challenge limits (95%, 200 tokens, 80 ms)', (consts['CH_GOAL'], consts['CH_CTX'], consts['CH_LAT']) == (GOAL, CTX, LATMAX), consts)
    check('20 questions', len(T.qs) == 20 and consts['NQ'] == 20)
    check(f'encoder dimensions {len(T.lex)}', consts['DIM'] == len(T.lex))
    for lang in ('en', 'zh'):
        lens = [len(d[0]) for d in T.build(lang)]
        check(f'{lang} document lengths', pg.evaluate(f"()=>window.__ckr.docLen('{lang}')") == lens, lens)
        for size in SIZES:
            for ov in ([0] if size == 'h' else [0, 1, 2]):
                R = T.setup(lang, size, ov); J = pg.evaluate(f"()=>window.__ckr.setup('{lang}',{repr(size) if size == 'h' else size},{ov})")
                tag = f'{lang} {size}/{ov}'
                check(f'{tag} chunks', [(c['d'], c['a'], c['b']) for c in R['C']] == [(c['d'], c['a'], c['b']) for c in J['C']])
                check(f'{tag} longest, splits', (R['longest'], R['splits']) == (J['longest'], J['splits']), (R['longest'], R['splits'], J['longest'], J['splits']))
                bad = []
                for qi, (q, jq) in enumerate(zip(R['Q'], J['Q'])):
                    if q['right'] != jq['right'] or q['rk'] != jq['rk'] or q['part'] != jq['part']: bad.append(('rank', qi, q['rk'], jq['rk']))
                    if q['lists'] != jq['lists'] or q['rr'] != jq['rr'] or q['nc'] != jq['nc']: bad.append(('lists', qi))
                    for f in ('kw', 'vec', 'rrf', 'cov'):
                        if max(abs(x - y) for x, y in zip(q[f], jq[f])) > 1e-9: bad.append((f, qi))
                check(f'{tag} scores, returned lists, rerank order and ranks of all 20 questions', not bad, bad[:3])
        for s_ in settings():
            m = T.measure(lang, s_); j = pg.evaluate(f"(s)=>window.__ckr.measure('{lang}',s)", s_)
            if (m['hits'], m['ctx'], m['splits'], m['nc']) != (j['hits'], j['ctx'], j['splits'], j['nc']) or abs(m['lat'] - j['lat']) > 1e-9 or abs(m['rer'] - j['rerankMs']) > 1e-9:
                check(f'{lang} measure {s_}', False, (m, j)); break
        else: check(f'{lang} every setting: recall, context, latency (rerank charged by the candidates really returned)', True)
        # B2: no rerank charge counts candidates that were never returned
        check(f'{lang} the candidate pool ({NCAND}) is smaller than the index at every chunking', all(T.setup(lang, sz, ov)['N'] > NCAND for sz in SIZES for ov in ([0] if sz == 'h' else [0, 1, 2])))
        # the challenge: feasible settings and the fewest context tokens
        feas, best, minc = T.challenge(lang)
        J = pg.evaluate(f"()=>window.__ckr.chAll('{lang}')")
        jok = [{k: x[k] for k in ('size', 'ov', 'ret', 'rr', 'k')} for x in J if x['ok']]
        check(f'{lang} challenge feasible set ({len(feas)})', sorted(map(str, feas)) == sorted(map(str, jok)))
        jb = [{k: x[k] for k in ('size', 'ov', 'ret', 'rr', 'k')} for x in J if x['best']]
        check(f'{lang} challenge best: {minc} tokens, {len(best)} setting(s)', sorted(map(str, best)) == sorted(map(str, jb)) and minc == 64, best)
        # A1: the lesson. Only hybrid search + rerank, keeping 1, is cheapest; keyword or vector search + rerank stays under the goal
        check(f'{lang} lesson: every cheapest setting is hybrid search + rerank, keep 1, with overlap', best and all(s_['ret'] == 'hyb' and s_['rr'] == 1 and s_['k'] == 1 and s_['ov'] > 0 for s_ in best), best)
        bb = sorted(best, key=lambda x: -x['ov'])[0]
        for r_ in ('kw', 'vec'):
            alt = T.measure(lang, {**bb, 'ret': r_})
            check(f'{lang} lesson: {r_} search + rerank at the best chunking stays under {round(GOAL * 100)}% ({alt["hits"]}/20)', alt['rec'] < GOAL - 1e-9)
            # and every question it misses has its answer outside that search's first 5: the pool, not a tie, decides
            R = T.setup(lang, bb['size'], bb['ov'])
            miss = [qi for qi, q in enumerate(R['Q']) if q['rk'][r_ + '1'] != 0]
            check(f'{lang} lesson: {r_} + rerank misses only answers outside its first {NCAND} (questions {[x + 1 for x in miss]})', all(not (0 <= R['Q'][qi]['rk'][r_ + '0'] < NCAND) for qi in miss))
        check(f'{lang} lesson: only hybrid search has every answer in its first {NCAND} at the best chunking', sum(1 for q in T.setup(lang, bb['size'], bb['ov'])['Q'] if 0 <= q['rk']['hyb0'] < NCAND) == 20
              and all(sum(1 for q in T.setup(lang, bb['size'], bb['ov'])['Q'] if 0 <= q['rk'][r_ + '0'] < NCAND) < 20 for r_ in ('kw', 'vec')))
        check(f'{lang} lesson: no setting without rerank meets the goal keeping 1', not [s_ for s_ in feas if s_['k'] == 1 and not s_['rr']])
        # the same cheapest set if ties in the stand-in's coverage are broken by the first search's order instead of the cosine
        _, best_pos, minc_pos = T.challenge(lang, tie='pos')
        check(f'{lang} lesson holds when coverage ties go by first-search order', sorted(map(str, best_pos)) == sorted(map(str, best)) and minc_pos == minc, best_pos)
        # no right chunk sits in or out of a 5-chunk pool only because its score equals the score at the pool's edge
        edge = []
        for sz, ov in ((64, 0), (64, 1), (64, 2)):
            R = T.setup(lang, sz, ov)
            for qi, q in enumerate(R['Q']):
                for r_, sc in (('kw', q['kw']), ('vec', q['vec']), ('hyb', q['rrf'])):
                    l = q['lists'][r_]
                    if len(l) <= NCAND: continue
                    for j in q['right']:
                        if j in l and abs(sc[j] - (sc[l[NCAND]] if l.index(j) < NCAND else sc[l[NCAND - 1]])) < 1e-12: edge.append((sz, ov, qi, r_))
        check(f'{lang} no ties at the edge of the {NCAND}-chunk pool', not edge, edge[:4])
        init = T.measure(lang, {'size': 64, 'ov': 0, 'ret': 'vec', 'rr': 0, 'k': 1})
        check(f'{lang} the initial setup misses the challenge goal', init['rec'] < GOAL)
        # B1: with k = 1 a cut answer is at most half in the context; with k = 3 some cut answers have every part in the context
        R0 = T.setup(lang, 64, 0)
        allin = lambda k: [qi for qi, q in enumerate(R0['Q']) if not q['right'] and all(j in q['lists']['vec'][:k] for j in q['part'])]
        check(f'{lang} B1: keeping 1, no cut answer has all its parts in the context', not allin(1))
        print(f'   info {lang}: cut answers with every part in the vector top 3 at 64/0: {len(allin(3))} of {R0["splits"]}')
        # teaching facts the copy states
        R2 = T.setup(lang, 64, 2); toks = T.build(lang)
        q5 = R0['Q'][5]; c0 = R0['C'][q5['part'][0]] if q5['part'] else None
        check(f'{lang} part 1: the E-203 answer is cut right after its code at 64 tokens', not q5['right'] and len(q5['part']) == 2 and c0['b'] == q5['sa'] + 1
              and re.fullmatch(r'[A-Za-z]-\d+', toks[c0['d']][0][q5['sa']][0]) and toks[c0['d']][0][q5['sa']][0] not in [w for w, _, _ in toks[c0['d']][0][R0['C'][q5['part'][1]]['a']:R0['C'][q5['part'][1]]['b']]])
        check(f'{lang} 64 tokens with half overlap: no answer is cut apart', R2['splits'] == 0)
        q0, q8, q17 = R2['Q'][0], R2['Q'][8], R2['Q'][17]
        check(f'{lang} E-201: keyword first, vector outside the {NCAND} candidates', q0['rk']['kw0'] == 0 and q0['rk']['vec0'] >= NCAND, q0['rk'])
        check(f'{lang} the stairs question: keyword search does not return it, vector first', q8['rk']['kw0'] == -1 and q8['rk']['vec0'] == 0, q8['rk'])
        check(f'{lang} the thermostat-battery question: vector rank 3, first after rerank', q17['rk']['vec0'] == 2 and q17['rk']['vec1'] == 0, q17['rk'])
        a1, b1 = T.measure(lang, {'size': 64, 'ov': 2, 'ret': 'vec', 'rr': 0, 'k': 1}), T.measure(lang, {'size': 64, 'ov': 2, 'ret': 'vec', 'rr': 1, 'k': 1})
        check(f'{lang} rerank raises vector recall@1', b1['rec'] > a1['rec'], (a1, b1))
        sm = T.measure(lang, {'size': 16, 'ov': 0, 'ret': 'vec', 'rr': 0, 'k': 1})
        check(f'{lang} 16-token chunks: more answers cut apart', sm['splits'] > init['splits'])
        li = 1 if lang == 'zh' else 0; qt = T.toks(T.qs[8][li], lang); d = [x[0] for x in T.docs].index('W'); ans = T.docs[d][2][1][1][0][li]
        check(f'{lang} the stairs question shares no keyword term with its answer', not (set(T.terms(qt, lang)) & set(T.terms(T.toks(ans, lang), lang))))
    # A2: no guided state and no scenario is a cheapest challenge setting
    def txt(sel): return pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' '):''}", sel)
    st = lambda: {k: v for k, v in pg.evaluate("()=>window.__ckr.state()").items() if k in ('size', 'ov', 'ret', 'rr', 'k')}
    for i in range(7):
        pg.evaluate(f"()=>{{const b=document.querySelector('#ckr-rail [data-go=\"{i}\"]'); if(b) b.click();}}"); pg.wait_for_timeout(120)
        if pg.evaluate("()=>window.__ckr.state().step") != i:
            while pg.evaluate("()=>window.__ckr.state().step") < i:
                if pg.query_selector('#ckr-rail [data-pred="kw"]') and not pg.evaluate("()=>window.__ckr.state().pred"): pg.evaluate("()=>document.querySelector('#ckr-rail [data-pred=\"kw\"]').click()")
                if pg.query_selector('#ckr-rail [data-nav="showme"]:not([hidden])'): pg.evaluate("()=>document.querySelector('#ckr-rail [data-nav=\"showme\"]').click()")
                pg.evaluate("()=>{const b=document.querySelector('#ckr-rail .step.current [data-nav=\"next\"]'); if(b&&!b.disabled) b.click();}"); pg.wait_for_timeout(120)
        GUIDED_AND_SCENARIOS.append(('part %d' % (i + 1), st()))
        if pg.query_selector('#ckr-rail [data-nav="showme"]:not([hidden])'):
            pg.evaluate("()=>document.querySelector('#ckr-rail [data-nav=\"showme\"]').click()"); pg.wait_for_timeout(120); GUIDED_AND_SCENARIOS.append(('part %d, Show me' % (i + 1), st()))
        for key, vals in (('rr', ('0', '1')), ('k', ('1', '3', '5')), ('size', ('16', '32', '64', '128', 'h')), ('ov', ('0', '1', '2'))):
            for v in vals:
                if pg.query_selector(f'#ckr-rail .seg-ctl[data-key="{key}"] button[data-v="{v}"]'):
                    pg.evaluate(f"()=>document.querySelector('#ckr-rail .seg-ctl[data-key=\"{key}\"] button[data-v=\"{v}\"]').click()"); pg.wait_for_timeout(60)
                    GUIDED_AND_SCENARIOS.append((f'part {i + 1}, {key} = {v}', st()))
    pg.evaluate("()=>document.querySelector('#ckr-app .modes button[data-mode=\"sandbox\"]:not([data-flow=free])').click()"); pg.wait_for_timeout(200)
    for i in range(6):
        pg.evaluate(f"()=>document.querySelector('#ckr-rail [data-preset=\"{i}\"]').click()"); pg.wait_for_timeout(80); GUIDED_AND_SCENARIOS.append((f'scenario {i + 1}', st()))
    print('   info: states seen in the guided parts and scenarios:', len(GUIDED_AND_SCENARIOS), sorted({tuple(x.values()) for _, x in GUIDED_AND_SCENARIOS}, key=str)[:40])
    for lang in ('en', 'zh'):
        _, best, _ = T.challenge(lang)
        shown = [(w, x) for w, x in GUIDED_AND_SCENARIOS if any(all(x[k] == s_[k] for k in ('size', 'ov', 'ret', 'rr', 'k')) for s_ in best)]
        check(f'{lang} A2: no guided state or scenario ({len(GUIDED_AND_SCENARIOS)} seen) is a cheapest challenge setting', not shown, shown[:3])
    pg.evaluate("()=>document.querySelector('#ckr-app .modes button[data-mode=\"guided\"]').click()"); pg.wait_for_timeout(200)
    # page text in both languages
    for lang in ('en', 'zh'):
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(200); pg.evaluate("()=>{location.hash='chunk-rerank'}"); pg.wait_for_timeout(400)
        pg.evaluate("()=>{const b=document.querySelector('#ckr-rail [data-go=\"0\"]'); if(b) b.click();}"); pg.wait_for_timeout(150)
        init = T.measure(lang, {'size': 64, 'ov': 0, 'ret': 'vec', 'rr': 0, 'k': 1})
        body = txt('#ckr-rail .step.current')
        low = 20 - init['hits'] - init['splits']
        check(f'{lang} part 1 names {init["hits"]} found, {init["splits"]} cut, {low} below first', ((f'that is {init["hits"]} of the 20' in body and f'{init["splits"]} are cut apart and {low} rank below first' in body) if lang == 'en' else
              (f'找到了 {init["hits"]} 个' in body and f'{init["splits"]} 个答案被切开，{low} 个没排到第一' in body)), body[-400:])
        sumL = lambda l: sum(len(d[0]) for d in T.build(l))
        more = round((sumL('zh') / sumL('en') - 1) * 20) * 5
        # the share of extra tokens on the Chinese pages was in the Fidelity card’s “Modeled” list, deleted with it (owner, 2026-10-11 19:28 UTC)
        # the challenge result
        pg.evaluate("()=>document.querySelector('#ckr-app .modes button[data-mode=\"challenge\"]').click()"); pg.wait_for_timeout(300)
        pg.evaluate("()=>document.querySelector('#ckr-rail [data-act=\"submit\"]').click()"); pg.wait_for_timeout(200)
        res = txt('#ckr-ch-res')
        check(f'{lang} challenge verdict for the initial setup', ('Not yet' in res or '还没达到' in res) and f'{round(init["rec"] * 100)}%' in res and f'{round(GOAL * 100)}%' in res, res[:200])
        pg.evaluate("()=>document.querySelector('#ckr-rail [data-act=\"reveal\"]').click()"); pg.wait_for_timeout(200)
        res = txt('#ckr-ch-res')
        rows = pg.evaluate("()=>[...document.querySelectorAll('#ckr-ch-res .cmp-row')].map(r=>r.innerText.replace(/\\s+/g,' '))")
        check(f'{lang} answer chart lists the feasible settings', len(rows) >= 3 and any('64' in r for r in rows), rows[:4])
        _, best, _ = T.challenge(lang); bb = sorted(best, key=lambda x: -x['ov'])[0]; R = T.setup(lang, bb['size'], bb['ov'])
        pool = {r_: sum(1 for q in R['Q'] if 0 <= q['rk'][r_ + '0'] < NCAND) for r_ in RETS}
        check(f'{lang} the why names the pools (kw {pool["kw"]}, vec {pool["vec"]}, hyb {pool["hyb"]})', all(re.search(r'[:：]\s*%d\b' % pool[r_], res) for r_ in RETS), res[-600:])
        for r_ in ('kw', 'vec'):
            alt = T.measure(lang, {**bb, 'ret': r_})
            check(f'{lang} the why lists {r_} + rerank at {round(alt["rec"] * 100)}%', f'{round(alt["rec"] * 100)}%' in pg.evaluate("()=>document.querySelector('#ckr-ch-res details').textContent"))
        pg.evaluate("()=>document.querySelector('#ckr-app .modes button[data-mode=\"guided\"]').click()"); pg.wait_for_timeout(200)
    # the interview drill
    for seed in (1, 2, 3, 7):
        dd = pg.evaluate(f"()=>window.__ckr.drill({seed})")
        sc = {c['n']: 1 / (60 + c['kw']) + 1 / (60 + c['vec']) for c in dd['ch']}
        bestn = sorted(sc, key=lambda k: (-sc[k], k))[0]
        check(f'drill {seed}: chunk {bestn}', dd['best']['n'] == bestn and abs(dd['best']['s'] - sc[bestn]) < 1e-12)
    check('rank 10 gets about 13% less than rank 1 in RRF (the drill’s claim)', round((1 / 61 - 1 / 70) / (1 / 61) * 100) == 13)
    check('page errors', not errs, errs[:3])
    b.close()

# the quoted sources, re-read from the local copies
QUOTES = [('lucene_bm25.java', 'this(1.2f, 0.75f, true, -1f);'), ('lucene_bm25.java', 'log(1 + (docCount - docFreq + 0.5)/(docFreq + 0.5))'),
          ('lc_ensemble2.py', 'c: int = 60'), ('lc_ensemble2.py', 'Reciprocal Rank Fusion'), ('lc_splitter_base.py', 'chunk_size: int = 4000,'), ('lc_splitter_base.py', 'chunk_overlap: int = 200,'),
          ('lc_splitter_base.py', 'length_function: Callable[[str], int] = len,'),
          ('oa_static_chunk.py', 'The default value is `800`.'), ('oa_static_chunk.py', 'The default value is `400`.'), ('oa_static_chunk.py', 'must not exceed half of `max_chunk_size_tokens`'),
          ('st_retrieve_rerank.md', 'retrieves a large list of e.g. 100 possible hits'), ('st_retrieve_rerank.md', 'the 5 passages with the highest score from the Cross-Encoder'),
          ('st_ce_readme.md', 'CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")'),
          ('ac_contextual_guide.ipynb', 'can miss exact keyword matches'), ('ac_contextual_guide.ipynb', 'Get top 150 results from both semantic search and BM25'),
          ('ac_contextual_guide.ipynb', 'reranking adds ~100-200ms per query'), ('rank_bm25.py', 'k1=1.5, b=0.75'),
          ('lucene_english_analyzer.java', '"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is",'),
          ('lucene_cjk_bigram.java', 'when a CJK character has no adjacent characters to form a bigram, it is output in'),
          ('lc_ensemble2.py', 'Defaults to equal'), ('ac_contextual_guide.ipynb', 'These weights are tunable'),
          ('st_retrieve_rerank.md', "If you only have a small set of paragraphs, we don't do the retrieval stage."),
          ('st_retrieve_rerank.md', 'The bi-encoder produces embeddings independently for your paragraphs and for your search queries.')]
for f, q in QUOTES:
    check(f'source {f}: “{q[:50]}”', q in (RAW / f).read_text())
check('BGE README: rerankers for top-k documents', 're-rank top-k documents returned by embedding models' in (P / 'sources' / 'b10src' / 'embeddings' / 'raw' / 'fe_readme.md').read_text())
print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
