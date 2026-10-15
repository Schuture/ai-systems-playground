"""Numbers test for the chunk-rerank lab: an independent Python twin of the page's engine (tokens, chunks, BM25 as Lucene computes it,
the small hand-written encoder, reciprocal rank fusion, the stand-in reranker, recall@k, context tokens, the latency assumption, the
challenge's feasible settings) compared with the page for every chunking in both languages; the numbers in the page's text (part 1,
part 2's cards, and each scenario's conclusion against the ranking on the stage beside it, P1 of the polish review of 2026-10-15); the
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
        low = 20 - init['hits'] - init['splits']; en = lang == 'en'
        R0 = T.setup(lang, 64, 0); q5 = R0['Q'][5]; vl = q5['lists']['vec']
        pr = sorted(vl.index(j) + 1 for j in q5['part'] if j in vl); one = vl[0] in q5['part']; hr = (' and ' if en else '、').join(map(str, pr))
        want = ([f'{init["hits"]} of the 20 are found now', f'the recall@1 of {round(init["rec"] * 100)}%', f'The halves rank {hr}', 'holds only one of them' if one else 'holds neither'] if en else
                [f'现在找到 {init["hits"]} 个', f'recall@1 {round(init["rec"] * 100)}%', f'两半在舞台上的列表里排第 {hr} 名', '只有其中一半' if one else '一样也没有'])
        check(f'{lang} part 1 names {init["hits"]} found (recall@1 {round(init["rec"] * 100)}%) and the halves at {pr} of the vector list, from the twin', len(pr) == 2 and all(w in body for w in want), [w for w in want if w not in body])
        # the counts part 1 no longer names live in part 2's cards and in the initial setup's scenario
        pg.evaluate("()=>{const b=document.querySelector('#ckr-rail [data-go=\"1\"]'); if(b) b.click();}"); pg.wait_for_timeout(150)
        if pg.evaluate("()=>window.__ckr.state().step") != 1: pg.evaluate("()=>document.querySelector('#ckr-rail .step.current [data-nav=\"next\"]').click()"); pg.wait_for_timeout(150)
        met = txt('#ckr-metrics')
        check(f'{lang} part 2: the cards name {init["splits"]} of 20 answers cut and {R0["N"]} chunks, from the twin', ((f'{init["splits"]} of 20' in met and f'the help center is cut into {R0["N"]} chunks' in met) if en else
              (f'20 个里 {init["splits"]} 个' in met and f'帮助中心一共切成 {R0["N"]} 块' in met)), met)
        Z_ = lambda e, a, b: a if e else b
        # part 4 (N3 and N2 of the polish check): the sentence under the small table and the stairs question's reason, from the twin
        pg.evaluate("()=>{const b=document.querySelector('#ckr-rail [data-go=\"3\"]'); if(b) b.click();}"); pg.wait_for_timeout(200)
        R2 = T.setup(lang, 64, 2); Q2 = R2['Q']; two = lambda q, x: x in q['lists']['kw'][:NCAND] and x in q['lists']['vec'][:NCAND]
        wonQ = [qi for qi in range(20) if Q2[qi]['rk']['hyb0'] == 0 and Q2[qi]['rk']['vec0'] != 0]; lostQ = [qi for qi in range(20) if Q2[qi]['rk']['vec0'] == 0 and Q2[qi]['rk']['hyb0'] != 0]
        codes_ = bool(wonQ) and all(T.qs[qi][6] == 'code' for qi in wonQ); lr = sorted({Q2[qi]['rk']['hyb0'] + 1 for qi in lostQ})
        def how(q):   # 2: every chunk above the right one is on both lists; 1: or ties with it (and comes first by order); 0: neither
            r = q['rk']['hyb0']; j = q['lists']['hyb'][r]; ab = q['lists']['hyb'][:r]
            return 2 if all(two(q, x) for x in ab) else (1 if all(two(q, x) or abs(q['rrf'][x] - q['rrf'][j]) < 1e-12 for x in ab) else 0)
        hw = [how(Q2[qi]) for qi in lostQ]; both = bool(lostQ) and all(h == 2 for h in hw); ortie = bool(lostQ) and not both and all(h >= 1 for h in hw)
        orl = lambda a: str(a[0]) if len(a) < 2 else ', '.join(map(str, a[:-1])) + ' or ' + str(a[-1])
        if en: want = [f'puts {len(wonQ)} more {"error-code " if codes_ else ""}question', f'{len(lostQ)} question{"s" if len(lostQ) != 1 else ""} that vector search ranks first drop to rank {orl(lr)}',
                       'the chunks above them are on both lists, with two shares that add up to at least the right chunk’s score' if both else ('each chunk above them is on both lists with two shares, or ties with the right chunk and comes first in the help center' if ortie else 'the ranks of other chunks can add up')]
        else: want = [f'多把 {len(wonQ)} 道{"错误码题" if codes_ else "题"}排到第一', f'有 {len(lostQ)} 道向量检索排第一的题掉到第 {"、".join(map(str, lr))} 名',
                      '排在它们前面的块在两个列表里都有，两份分加起来不低于正确的块' if both else ('排在它们前面的块，要么两个列表里都有、拿两份分，要么和正确的块同分、按块的先后排在前面' if ortie else '别的块的名次加起来可能压过正确的块')]
        sm = txt('#ckr-sum')
        check(f'{lang} part 4: under the table, hybrid search gains {len(wonQ)} questions (error codes: {codes_}) and drops {len(lostQ)} to rank {lr}, behind chunks on both lists: {both}, or those and ties: {ortie}, from the twin (N3)', bool(wonQ) and bool(lostQ) and all(w in sm for w in want), ([w for w in want if w not in sm], sm))
        pg.evaluate("()=>document.querySelector('#ckr-rail .step.current .seg-ctl[data-key=\"q\"] button[data-v=\"8\"]').click()"); pg.wait_for_timeout(200)
        q8 = Q2[8]; r8 = q8['rk']['hyb0']; j8 = q8['lists']['hyb'][r8]; ab = q8['lists']['hyb'][:r8]; nn = len(ab); n2 = sum(1 for x in ab if two(q8, x)); ob = txt('#ckr-rail [data-obs]')
        tie8 = any(abs(q8['rrf'][x] - q8['rrf'][j8]) < 1e-12 for x in ab)
        if en: w8 = 'In RRF it has a rank on the vector list only, so it gets one share' + ('; ' + (('the chunk above it is on both lists and gets two shares' if nn == 1 else f'the {nn} chunks above it are on both lists and get two shares each') if n2 == nn else f'{n2} of the {nn} chunks above it {"is" if n2 == 1 else "are"} on both lists and get{"s" if n2 == 1 else ""} two shares') if n2 else '')
        else: w8 = 'RRF 合并时，它只在向量列表里有名次，拿一份分' + ('；' + (('排在它前面的那一块两个列表里都有，拿两份分' if nn == 1 else f'排在它前面的 {nn} 块两个列表里都有，各拿两份分') if n2 == nn else f'排在它前面的 {nn} 块里，有 {n2} 块两个列表里都有，{"拿" if n2 == 1 else "各拿"}两份分') if n2 else '')
        check(f'{lang} part 4, the stairs question: keyword search does not return its answer; hybrid search ranks it {r8 + 1}, behind {n2} of {nn} chunks on both lists{" and a tie" if tie8 else ""}, and the reason says so, from the twin (N2)', q8['rk']['kw0'] < 0 and r8 >= 1 and w8 in ob and (Z_(en, 'it ties with', '同分，按块的先后排在后面') in ob) == tie8, (w8, ob))
        # the scenarios: each conclusion is the page's own computation, for the question on the stage (P1 of the polish review)
        pg.evaluate("()=>document.querySelector('#ckr-app .modes button[data-mode=\"sandbox\"]:not([data-flow=free])').click()"); pg.wait_for_timeout(200)
        setq = lambda qi: (pg.evaluate(f"()=>{{const s=document.querySelector('#ckr-qsel'); s.value='{qi}'; s.dispatchEvent(new Event('change',{{bubbles:true}}));}}"), pg.wait_for_timeout(80))
        preset = lambda i: (pg.evaluate(f"()=>document.querySelector('#ckr-rail [data-preset=\"{i}\"]').click()"), pg.wait_for_timeout(150))
        setq(5); preset(0); ins = txt('#ckr-insight')
        check(f'{lang} scenario “initial setup”: {init["splits"]} cut apart, {low} below first, from the twin', ((f'{init["splits"]} answers are cut apart' in ins and f'Of the others, {low} rank below first' in ins) if en else
              (f'有 {init["splits"]} 个答案被切开' in ins and f'有 {low} 个在向量检索里没排到第一' in ins)), ins)
        R16 = T.setup(lang, 16, 0); ln = [q['sb'] - q['sa'] for q in R16['Q']]; lng = sum(1 for x in ln if x > 16); sh = [q for q, x in zip(R16['Q'], ln) if x <= 16]; shc = sum(1 for q in sh if not q['right'])
        preset(1); ins = txt('#ckr-insight')
        check(f'{lang} scenario “small chunks”: {lng} answers longer than 16 tokens, {shc} of the {len(sh)} no longer than a chunk cut, from the twin', ((f'{lng} answers here are longer than a 16-token chunk' in ins and f'of the {len(sh)} no longer than a chunk, the cut still falls inside {shc}' in ins) if en else
              (f'这里有 {lng} 个答案比一块（16 个 token）还长' in ins and f'不比一块长的 {len(sh)} 个里，也有 {shc} 个' in ins)), ins)
        mh = T.measure(lang, {'size': 'h', 'ov': 0, 'ret': 'vec', 'rr': 0, 'k': 1}); i0, ih = round(init['rec'] * 100), round(mh['rec'] * 100)
        preset(2); ins = txt('#ckr-insight')
        # N8 of the polish check: which of the answers the initial setup cuts apart now rank first, and how many whole answers miss first place
        Rh = T.setup(lang, 'h', 0); cut0 = [qi for qi in range(20) if not R0['Q'][qi]['right']]; now = sum(1 for qi in cut0 if Rh['Q'][qi]['rk']['vec0'] == 0); lowh = 20 - mh['hits'] - mh['splits']
        wh = ([f'No answer is cut apart', (f'recall@1 rises from {i0}% (the initial setup) to {ih}%' if mh['rec'] > init['rec'] else f'the initial setup reaches {i0}%'),
               f'Of the {len(cut0)} answer{"" if len(cut0) == 1 else "s"} the initial setup cuts apart, {now} now {"ranks" if now == 1 else "rank"} first' + (f', but whole answers that do not rank first grow from {low} to {lowh}.' if lowh > low else '.')] if en else
              ['没有答案被切开', (f'recall@1 从初始设置的 {i0}% 升到 {ih}%' if mh['rec'] > init['rec'] else f'初始设置的 recall@1 是 {i0}%'),
               f'原来被切开的 {len(cut0)} 个里有 {now} 个现在排第一' + (f'，可答案完整却没排第一的，从 {low} 个变成了 {lowh} 个。' if lowh > low else '。')])
        check(f'{lang} scenario “cut by heading”: no answer cut, recall@1 {i0}% → {ih}%, {now} of the {len(cut0)} cut before now first, {low} → {lowh} whole but not first, from the twin', mh['splits'] == 0 and len(cut0) == init['splits'] and all(w in ins for w in wh), ([w for w in wh if w not in ins], ins))
        R2 = T.setup(lang, 64, 2); kind = lambda x: [qi for qi in range(20) if T.qs[qi][6] == x]; codes, paras = kind('code'), kind('para')
        first = sum(1 for qi in codes if R2['Q'][qi]['rk']['kw0'] == 0); pf = sum(1 for qi in paras if R2['Q'][qi]['rk']['kw0'] == 0)
        for qi in (5, 0, 8):
            setq(qi); preset(3); ins = txt('#ckr-insight'); tg = f'{lang} scenario “keyword search only”, question {qi + 1} on the stage'
            head = (f'Of the {len(codes)} error-code questions, keyword search puts {"all" if first == len(codes) else first} first.' if en else
                    f'{len(codes)} 道错误码题里，关键词检索{"都排第一" if first == len(codes) else f"把 {first} 道排第一"}。')
            check(f'{tg}: the conclusion counts {first} of {len(codes)} error codes first, from the twin, and says “all” only then', head in ins and (('puts all first' in ins) if en else ('都排第一' in ins)) == (first == len(codes)), ins)
            rows = pg.evaluate("()=>[...document.querySelectorAll('#ckr-cols .ckr-col[data-c=\"kw0\"] .ckr-li')].map(li=>({c:li.className.split(' '), b:li.querySelector('b').textContent}))")
            q = R2['Q'][qi]; r = q['rk']['kw0']
            if T.qs[qi][6] == 'code':
                code = re.search(r'[A-Za-z]-\d+', T.qs[qi][1 if lang == 'zh' else 0]).group(0).upper()
                say = ((f'The question on the stage, {code}, ranks first too.' if r == 0 else f'The question on the stage, {code}, ranks {r + 1}.') if en else
                       (f'舞台上这一问（{code}）也排第一。' if r == 0 else f'舞台上这一问（{code}）排第 {r + 1}。'))
                seen = 0 <= r < len(rows) and ('ok' if r == 0 else 'bad') in rows[r]['c'] and not any('ok' in x['c'] for x in rows[:r])
                check(f'{tg}: “{say}” and the keyword list beside it agree (rank {r + 1})', say in ins and seen, (ins, rows))
                if r > 0: check(f'{tg}: the chunk it names first is the list’s first row ({rows[0]["b"]})', (f'The first chunk, {rows[0]["b"]},' in ins) if en else (f'排第一的{rows[0]["b"]}讲的是' in ins), ins)
            else:
                check(f'{tg}: no sentence about the stage for a question without a code', 'The question on the stage' not in ins and '舞台上这一问' not in ins, ins)
            para = (f'of {len(paras)}, it puts {pf} first' if en else (f'{len(paras)} 道里只有 {pf} 道排第一' if pf < first else f'{len(paras)} 道里有 {pf} 道排第一'))
            check(f'{tg}: questions in other words, {pf} of {len(paras)} first, from the twin', para in ins, ins)
        g = {'both': 0, 'kw': 0, 'vec': 0}; gk = {'kw': set(), 'vec': set()}
        for qi, q in enumerate(R2['Q']):
            r = q['rk']['hyb0']
            if not 0 <= r < 3: continue
            j = q['lists']['hyb'][r]; a_, v_ = j in q['lists']['kw'][:NCAND], j in q['lists']['vec'][:NCAND]; k_ = 'both' if a_ and v_ else ('kw' if a_ else 'vec'); g[k_] += 1
            if k_ in gk: gk[k_].add(T.qs[qi][6])
        preset(4); ins = txt('#ckr-insight')
        want = ([f'on both lists: {g["both"]}', f'on the keyword list only: {g["kw"]}', f'on the vector list only: {g["vec"]}'] if en else
                [f'两个列表都有：{g["both"]} 个', f'只在关键词列表：{g["kw"]} 个', f'只在向量列表：{g["vec"]} 个'])
        tags = ([('all error codes', gk['kw'] == {'code'}), ('all in the customer’s own words', gk['vec'] == {'para'})] if en else [('都是错误码题', gk['kw'] == {'code'}), ('都是换了说法的题', gk['vec'] == {'para'})])
        check(f'{lang} scenario “hybrid, keep 3”: where the answers in the context come from {g}, from the twin', all(w in ins for w in want) and all((t in ins) == ok for t, ok in tags), (ins, gk))
        Rb = T.setup(lang, 128, 0); cut = [q for q in Rb['Q'] if not q['right']]; allp = [q for q in cut if all(j in q['lists']['vec'][:5] for j in q['part'])]
        mb = T.measure(lang, {'size': 128, 'ov': 0, 'ret': 'vec', 'rr': 0, 'k': 5})
        preset(5); ins = txt('#ckr-insight')
        if en: wb = [f'from {round(init["rec"] * 100)}% (the initial setup) to {round(mb["rec"] * 100)}%'] + ([f'{len(cut)} answer{"s are" if len(cut) > 1 else " is"} still cut apart'] if cut else []) + (['all the pieces are among the 5 chunks kept' if len(allp) == len(cut) else f'for {len(allp)} of them all the pieces are'] if allp else [])
        else: wb = [f'召回率从初始设置的 {round(init["rec"] * 100)}% 升到 {round(mb["rec"] * 100)}%'] + ([f'还有 {len(cut)} 个答案被切开' if allp else f'而且还有 {len(cut)} 个答案被切开'] if cut else []) + ([('不过它们的' if len(allp) == len(cut) else f'其中 {len(allp)} 个的') + ('两半' if all(len(q['part']) == 2 for q in allp) else '几部分') + '都在取回的 5 块里'] if allp else [])
        check(f'{lang} scenario “big chunks, keep 5”: {len(cut)} cut apart, {len(allp)} of them with every piece kept, from the twin (P17)', all(w in ins for w in wb), ([w for w in wb if w not in ins], ins))
        pg.evaluate("()=>document.querySelector('#ckr-app .modes button[data-mode=\"guided\"]').click()"); pg.wait_for_timeout(200)
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
