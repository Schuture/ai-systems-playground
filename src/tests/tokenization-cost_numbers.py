"""Golden numbers for the tokenization-cost lab (guide R9.2, CONTRACT §6).

The byte-level BPE is re-derived here, independently of the JS:
- the training texts are built literally (the English paragraphs joined by spaces, then " " + the Chinese paragraph repeated w times),
  where the page weights the Chinese pieces' counts by w;
- pieces come from Python's `regex` module with the same pattern (letters, digits, other symbols, each with an optional leading space);
- every merge recounts all pairs from scratch (the page keeps its counts up to date incrementally); ties go to the smaller pair;
  training stops at 400 merges or when the best pair occurs fewer than 2 times;
- encoding merges, again and again, the adjacent pair with the lowest merge rank (as tiktoken and minbpe do), where the page applies
  the merges one after another in the order they were learned.
Checks: the corpus equals the home tile's (window.__tzB); the tile mix's merges equal the tile's; every mix's merges and counts; the
token counts of both sentence pairs at every number of merges from 0 to 400 in every mix; the mission, the part-5 options, the
scenarios, the challenge (every option, the minimal answer, the tiers), the exits' numbers, the interview drill, UTF-8 byte counts
(against Python's encoder), and the Connect card's code, run here, printing what its last comment lines say.
Round 1 (reviews A and B): the challenge's “why” (B1): on mostly English text, 100 → 400 merges shrink the Chinese sentence more
than the English one (38 → 25 against 18 → 14), and the page's chTrend agrees; half and half and mostly Chinese learn the same
400 merges (A32); “我” is a whole token of the Chinese sentence from merge 210 on the initial text (part 7); every “131,072 ÷ n”
line of Show the math shows the quotient before rounding down (S2); the new case 2's answer range (A10); the drill's candidates.
Run: python3 tests/tokenization-cost_numbers.py"""
import json, math, pathlib, re, subprocess, sys
import regex
from collections import Counter
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/tokenization-cost-num.html', '--only', 'tokenization-cost,set-b'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'tokenization-cost-num-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'tokenization-cost-num.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'tokenization-cost-num-wrapped.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


PAT = regex.compile(r' ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+')
K_MAX = 400
MIXW = {'en': (1, 0), 'tile': (1, 1), 'bal': (1, 16), 'zh': (1, 48)}


def train(text, kmax=K_MAX):
    words = Counter(tuple(m.group(0).encode()) for m in PAT.finditer(text))
    merges = []
    for k in range(kmax):
        pairs = Counter()
        for w, f in words.items():
            for p in zip(w, w[1:]):
                pairs[p] += f
        if not pairs: break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        if pairs[best] < 2: break
        merges.append((best[0], best[1], pairs[best]))
        new = 256 + k
        nw = Counter()
        for w, f in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and w[i] == best[0] and w[i + 1] == best[1]: out.append(new); i += 2
                else: out.append(w[i]); i += 1
            nw[tuple(out)] += f
        words = nw
    return merges


def encode(text, merges, k):
    """lowest-rank pair first (tiktoken / minbpe style), with the first k merges"""
    rank = {(a, b): i for i, (a, b, c) in enumerate(merges[:k])}
    out = []
    for m in PAT.finditer(text):
        ids = list(m.group(0).encode())
        while len(ids) >= 2:
            cand = [(rank[(x, y)], j) for j, (x, y) in enumerate(zip(ids, ids[1:])) if (x, y) in rank]
            if not cand: break
            r = min(cand)[0]; a, b = merges[r][0], merges[r][1]; nid = 256 + r
            res, j = [], 0
            while j < len(ids):
                if j + 1 < len(ids) and ids[j] == a and ids[j + 1] == b: res.append(nid); j += 2
                else: res.append(ids[j]); j += 1
            ids = res
        out += ids
    return out


def tok_bytes(merges):
    t = {i: bytes([i]) for i in range(256)}
    for i, (a, b, c) in enumerate(merges): t[256 + i] = t[a] + t[b]
    return t


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(); errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.evaluate("()=>{location.hash='tokenization-cost'}"); pg.wait_for_timeout(600)
    TK = lambda js: pg.evaluate('()=>' + js)
    corpus = TK('window.__tk.corpus()')
    tile_corpus = TK('window.__tzB.tok.corpus()')
    check('the training text equals the home tile’s (English and Chinese)', corpus['en'] == tile_corpus['en'] and corpus['zh'] == tile_corpus['zh'])
    EN, ZH = corpus['en'], corpus['zh']
    check('corpus: the English paragraphs are 6,077 bytes; the Chinese paragraph 126 characters, 378 bytes',
          len(EN.encode()) == 6077 and len(ZH) == 126 and len(ZH.encode()) == 378, (len(EN.encode()), len(ZH), len(ZH.encode())))
    consts = TK('window.__tk.consts')
    PAIRS = consts['PAIRS']
    # ---------- the merges of every mix ----------
    MER = {}
    for mix, (we, wz) in MIXW.items():
        text = (EN if we else '') + (' ' + ZH) * wz
        if not we: text = text[1:]
        MER[mix] = train(text)
        page = [tuple(x) for x in TK(f"window.__tk.merges('{mix}')")]
        check(f'{mix}: {K_MAX} merges, the same pairs and counts as the page', len(MER[mix]) == K_MAX and page == MER[mix],
              next(((i, page[i], MER[mix][i]) for i in range(min(len(page), len(MER[mix]))) if page[i] != MER[mix][i]), (len(page), len(MER[mix]))))
        zp = TK(f"window.__tk.zhPiece('{mix}')"); tb = tok_bytes(MER[mix])
        check(f'{mix}: a merge counts as a piece of Chinese when its token has a byte ≥ 0x80', zp == [any(x >= 128 for x in tb[256 + i]) for i in range(len(MER[mix]))])
    tile_m = [tuple(x) for x in TK('window.__tzB.tok.merges()')]
    check('the tile mix’s first 300 merges equal the home tile’s', tile_m == MER['tile'][:300], (len(tile_m),))
    # ---------- token counts of both pairs, every mix, every k ----------
    page_counts = TK("""(()=>{const o={}; for(const m of ['en','tile','bal','zh']) for(const pr of ['main','ch']){o[m+'|'+pr]=[]; for(let k=0;k<=400;k++){const r=window.__tk.compute(m,k,pr); o[m+'|'+pr].push([r.en,r.zh]);}} return o;})()""")
    bad = []
    for mix in MIXW:
        for pr in ('main', 'ch'):
            for k in range(K_MAX + 1):
                mine = [len(encode(PAIRS[pr]['en'], MER[mix], k)), len(encode(PAIRS[pr]['zh'], MER[mix], k))]
                if mine != page_counts[f'{mix}|{pr}'][k]: bad.append((mix, pr, k, mine, page_counts[f'{mix}|{pr}'][k]))
    check('token counts of both sentence pairs at every k from 0 to 400 in every mix (6,416 counts)', not bad, bad[:5])
    tile_page = TK("[0,10,25,50,100,200,300].map(k=>{const r=window.__tzB.tok.compute(k); return [r.en,r.zh]})")
    check('the tile’s counts at its seven stops equal the lab’s', tile_page == [page_counts['tile|main'][k] for k in (0, 10, 25, 50, 100, 200, 300)], tile_page)
    # ---------- UTF-8 ----------
    for s, nb in [(PAIRS['main']['en'], 36), (PAIRS['main']['zh'], 27), (PAIRS['ch']['en'], 41), (PAIRS['ch']['zh'], 39), ('привет', 12)]:
        check(f'UTF-8 bytes of “{s}”: {nb}', len(s.encode('utf-8')) == nb, len(s.encode('utf-8')))
    check('each Chinese character of the pairs is 3 bytes', all(len(c.encode()) == 3 for c in PAIRS['main']['zh'] + PAIRS['ch']['zh']))
    # ---------- the mission and its numbers ----------
    r = TK("window.__tk.compute('tile',100,'main')")
    en, zh = len(encode(PAIRS['main']['en'], MER['tile'], 100)), len(encode(PAIRS['main']['zh'], MER['tile'], 100))
    check('mission: English 10 tokens, Chinese 26, 2.6 times', (en, zh) == (10, 26) and (r['en'], r['zh']) == (10, 26) and abs(r['ratio'] - 2.6) < 1e-12, r)
    check('mission: 131,072 ÷ 10 = 13,107 and ÷ 26 = 5,041 copies; vocabulary 356', r['ctxEn'] == 131072 // 10 == 13107 and r['ctxZh'] == 131072 // 26 == 5041 and r['vocab'] == 356, r)
    tb = tok_bytes(MER['tile'])
    zc = sum(1 for i in range(100) if any(x >= 128 for x in tb[256 + i]))
    check(f'mission: {zc} of the first 100 merges join Chinese bytes (the page: {r["zhMerges"]})', zc == r['zhMerges'] == 1, (zc, r['zhMerges']))
    first3 = [tb[256 + i].decode() for i in range(3)]
    check('the first three merges of the initial text are “ t”, “he”, “ a”', first3 == [' t', 'he', ' a'], first3)
    toks = [tb[i].decode('utf-8', 'replace') for i in encode(PAIRS['main']['en'], MER['tile'], 100)]
    check('at 100 merges “books” is two tokens, “ book” and “s” (the misconception “a token is a word”)', ' book' in toks and toks[toks.index(' book') + 1] == 's', toks)
    shares = {m: 6077 * we / (6077 * we + 378 * wz) for m, (we, wz) in MIXW.items()}
    page_sh = TK("Object.fromEntries(['en','tile','bal','zh'].map(m=>[m,window.__tk.compute(m,0,'main').share]))")
    check('shares of English by bytes: 100%, 94%, 50%, 25%', all(abs(page_sh[m] - shares[m]) < 1e-12 for m in MIXW) and [round(shares[m] * 100) for m in ('en', 'tile', 'bal', 'zh')] == [100, 94, 50, 25], page_sh)
    # ---------- part 5 (100 merges) and the scenarios (300 merges) ----------
    for k, want in [(100, {'tile': (10, 26), 'en': (10, 27), 'bal': (18, 17), 'zh': (28, 15)}), (300, {'en': (9, 27), 'tile': (9, 18), 'bal': (17, 10), 'zh': (20, 10)})]:
        for mix, w in want.items():
            mine = (len(encode(PAIRS['main']['en'], MER[mix], k)), len(encode(PAIRS['main']['zh'], MER[mix], k)))
            check(f'{mix} at {k} merges: English {w[0]}, Chinese {w[1]} (ratio {w[1] / w[0]:.2f})', mine == w and tuple(page_counts[f'{mix}|main'][k]) == w, mine)
    bal_chars = [tok_bytes(MER['bal'])[i].decode('utf-8', 'replace') for i in encode(PAIRS['main']['zh'], MER['bal'], 100)]
    check('half and half, 100 merges: “我” and “在” are single tokens', '我' in bal_chars and '在' in bal_chars, bal_chars)
    check('half and half, 100 merges: the English sentence grows from 10 to 18 while the Chinese one falls from 26 to 17', page_counts['bal|main'][100] == [18, 17])
    zb = sum(1 for i in range(100) if any(x >= 128 for x in tok_bytes(MER['bal'])[256 + i]))
    check(f'half and half: {zb} of the first 100 merges join Chinese bytes (the page agrees)', zb == TK("window.__tk.compute('bal',100,'main').zhMerges"), zb)
    # ---------- the challenge ----------
    rows = TK('window.__tk.chAll()')
    exp = []
    for mix in ('tile', 'bal', 'zh'):
        for k in (50, 100, 200, 400):
            e_, z_ = len(encode(PAIRS['ch']['en'], MER[mix], k)), len(encode(PAIRS['ch']['zh'], MER[mix], k))
            exp.append({'mix': mix, 'k': k, 'en': e_, 'zh': z_, 'okEn': e_ <= 25, 'okR': 2 * z_ <= 3 * e_})
    check('challenge: every option’s counts and verdicts', [(x['mix'], x['k'], x['en'], x['zh'], x['okEn'], x['okR']) for x in rows] == [(x['mix'], x['k'], x['en'], x['zh'], x['okEn'], x['okR']) for x in exp], rows)
    oks = [(x['mix'], x['k']) for x in exp if x['okEn'] and x['okR']]
    check('challenge: the options that meet both targets are half and half at 100, 200 and 400, and mostly Chinese at 400', oks == [('bal', 100), ('bal', 200), ('bal', 400), ('zh', 400)], oks)
    check('challenge: the fewest merges is 100, with half and half only', TK('window.__tk.chMinK()') == 100 and [(x['mix'], x['k']) for x in TK('window.__tk.chBest()')] == [('bal', 100)])
    check('challenge: at 50 merges no training text meets both targets', not any(x['k'] == 50 and x['okEn'] and x['okR'] for x in exp))
    st = next(x for x in exp if x['mix'] == 'tile' and x['k'] == 100)
    check('challenge: the initial setup (mostly English, 100 merges) gives English 18, Chinese 38, 2.11 times', (st['en'], st['zh']) == (18, 38) and f"{st['zh'] / st['en']:.2f}" == '2.11', st)
    t4 = next(x for x in exp if x['mix'] == 'tile' and x['k'] == 400); z4 = next(x for x in exp if x['mix'] == 'zh' and x['k'] == 400)
    check('challenge feedback: mostly English at 400 merges still 1.79 times; mostly Chinese meets at 400', f"{t4['zh'] / t4['en']:.2f}" == '1.79' and z4['okEn'] and z4['okR'], (t4, z4))
    # B1: from the initial setup (100 merges) to the most merges offered (400), on mostly English text
    drop_en, drop_zh = st['en'] - t4['en'], st['zh'] - t4['zh']
    tr = TK('window.__tk.chTrend()')
    check(f'B1: 100 → 400 merges on mostly English text: English {st["en"]} → {t4["en"]}, Chinese {st["zh"]} → {t4["zh"]}; Chinese drops more',
          (st['en'], t4['en'], st['zh'], t4['zh']) == (18, 14, 38, 25) and drop_zh > drop_en)
    check('B1: the page’s trend (which sentence the challenge text says shrinks more) agrees', tr['zhMore'] == (drop_zh > drop_en) and (tr['a']['en'], tr['b']['en'], tr['a']['zh'], tr['b']['zh']) == (st['en'], t4['en'], st['zh'], t4['zh']), tr)
    late = sum(1 for i in range(100, 400) if any(x >= 128 for x in tok_bytes(MER['tile'])[256 + i]))
    check(f'B1: merges 101–400 of the mostly English text include {late} that join Chinese bytes (the text’s “later merges reach the Chinese paragraph”)', late == 36, late)
    eng_whole = min(k for k in range(401) if all(len(encode(w, MER['tile'], k)) == 1 for w in [' like', ' read', ' books', ' morning']) or k == 400)
    check(f'B1: the English words of the main pair are whole tokens by merge {eng_whole} (“once those are whole tokens”)', eng_whole == 153, eng_whole)
    # A32: half and half and mostly Chinese end with the same merges at 400
    tbb, tbz = tok_bytes(MER['bal']), tok_bytes(MER['zh'])
    same = {tbb[256 + i] for i in range(400)} == {tbz[256 + i] for i in range(400)}
    check('A32: half and half and mostly Chinese learn the same 400 merges, in a different order', same and [tbb[256 + i] for i in range(400)] != [tbz[256 + i] for i in range(400)] and TK("window.__tk.sameMerges('bal','zh',400)") is True)
    check('A32: so both give English 16, Chinese 9 on the challenge pair at 400', [(x['en'], x['zh']) for x in exp if x['k'] == 400 and x['mix'] in ('bal', 'zh')] == [(16, 9), (16, 9)])
    # ---------- the exits ----------
    ex = TK('window.__tk.exits()')
    ru = len(encode('привет', MER['tile'], 100))
    check('new case 1: “привет” is 12 bytes and 12 tokens on the initial tokenizer; the right option is “12”', ru == 12 and ex['transfer'][0]['a'] == 1 and ex['transfer'][0]['o'][1].startswith('12'), (ru, ex['transfer'][0]['a']))
    gain = 2 * (128256 - 32000) * 4096
    check(f'new case 2: two tables gain 2 × 96,256 × 4,096 = {gain:,}, in the option “0.6B to 0.9B” (option 3)', gain == 788529152 and 0.6e9 <= gain < 0.9e9 and ex['transfer'][1]['a'] == 2 and '0.6B' in ex['transfer'][1]['o'][2] and '0.9B' in ex['transfer'][1]['o'][2], (ex['transfer'][1]['a'], ex['transfer'][1]['o']))
    check('new case 2: the question gives the reference value, about 1.05B (A10)', '1.05B' in ex['transfer'][1]['q'], ex['transfer'][1]['q'])
    check('new case 2: 32,000 is about a quarter of 128,256 (0.25), so the gain is about three quarters of 1.05B', abs(32000 / 128256 - 0.25) < 0.01 and abs(gain / 1050673152 - 0.75) < 0.01)
    check('Llama 3’s two tables: 2 × 128,256 × 4,096 = 1,050,673,152, 13% of 8,030,261,248', 2 * 128256 * 4096 == 1050673152 and round(100 * 1050673152 / 8030261248) == 13)
    check('explain answers are written first (a = 0)', all(q['a'] == 0 for q in ex['explain']))
    check('the two new cases put the right answer at different positions', ex['transfer'][0]['a'] != ex['transfer'][1]['a'])
    # ---------- the interview drill ----------
    dbad = []
    for s in range(1, 25):
        d = TK(f'window.__tk.drill({s})')
        tbt = tok_bytes(MER['tile'])
        mine = [tbt[i].decode('utf-8', 'replace').replace(' ', '·') for i in encode(d['w'], MER['tile'], d['k'])]
        if mine != d['final']: dbad.append((s, d['w'], d['k'], mine, d['final']))
        steps = [i for i in range(d['k']) if len(encode(d['w'], MER['tile'], i + 1)) < len(encode(d['w'], MER['tile'], i))]
        if steps != d['steps']: dbad.append(('steps', s, steps, d['steps']))
    check('interview drill: the tokens left and the merges that apply, for 24 seeds', not dbad, dbad[:3])
    cbad = []
    for s in range(1, 25):
        d = TK(f'window.__tk.drill({s})'); tbt = tok_bytes(MER['tile']); wb = d['w'].encode()
        cand = [i for i in range(d['k']) if tbt[256 + i] in wb]
        if cand != d['cand'] or not set(d['steps']) <= set(cand): cbad.append((s, d['w'], d['k'], cand, d['cand'], d['steps']))
        if d['w'].strip() not in 'I like to read books in the morning.': cbad.append(('word not in the sentence', d['w']))
    check('interview drill (A11): the question lists every merge among the first k whose token is part of the word, a superset of the ones that apply; the word is from the lab’s sentence', not cbad, cbad[:3])
    # part 7: “我” becomes a whole token of the Chinese sentence on the initial text at merge 210 (the page says “about 200”)
    wo = next(k for k in range(401) if any(tok_bytes(MER['tile'])[i] == '我'.encode() for i in encode(PAIRS['main']['zh'], MER['tile'], k)))
    check(f'part 7: “我” is a whole token from merge {wo} on the initial text; the page agrees and rounds it to {round(wo / 100) * 100}', wo == 210 == TK("window.__tk.wholeAt('tile','我喜欢在早上看书。','我')"), wo)
    # A4 (and N3, round 1b): a merge row names the characters its hex bytes come from. A new token that sits inside one character
    # and occurs in several different characters says how many (“start of 6 characters, such as “上” and “一””), and its two parts
    # then carry no note; otherwise the two parts are placed inside the new token's first place.
    ZB, offs = ZH.encode(), []
    o = 0
    for ch in ZH: offs.append((ch, o, len(ch.encode()))); o += len(ch.encode())
    def span(i, n):
        e = i + n - 1; cs = [c for c in offs if c[1] + c[2] > i and c[1] <= e]; f, l = cs[0], cs[-1]
        if len(cs) == 1: return f'part of “{f[0]}”'
        ms, me = i > f[1], e < l[1] + l[2] - 1
        inner = ''.join(c[0] for c in cs[(1 if ms else 0):(len(cs) - 1 if me else len(cs))])
        return ' + '.join(x for x in [f'end of “{f[0]}”' if ms else '', f'“{inner}”' if inner else '', f'start of “{l[0]}”' if me else ''] if x)
    def shared(bs):
        """the different characters of the paragraph holding bs inside them, most frequent first (ties: first seen), and where"""
        cnt, pos = {}, set()
        for ch, b0, nb in offs:
            cb = ch.encode()
            if nb <= len(bs): continue
            for o_ in range(nb - len(bs) + 1):
                if cb[o_:o_ + len(bs)] == bs:
                    cnt[ch] = cnt.get(ch, 0) + 1; pos.add('s' if o_ == 0 else ('e' if o_ + len(bs) == nb else 'm')); break
        order = sorted(cnt, key=lambda c: (-cnt[c], list(cnt).index(c)))
        return order, (pos.pop() if len(pos) == 1 else 'm')
    def alone(bs):
        i = ZB.find(bs)
        if i < 0: return None
        cs = [c for c in offs if c[1] + c[2] > i and c[1] < i + len(bs)]
        if len(cs) == 1 and cs[0][2] > len(bs):
            chars, ps = shared(bs)
            if len(chars) > 1:
                return f"{'inside' if ps == 'm' else {'s': 'start', 'e': 'end'}[ps] + ' of'} {len(chars)} characters, such as “{chars[0]}” and “{chars[1]}”", True
        return span(i, len(bs)), False
    def ok_utf8(bs):
        try: bs.decode('utf-8'); return True
        except UnicodeDecodeError: return False
    hbad, hn, notes, nshared = [], 0, 0, 0
    for mix, k in (('zh', 100), ('bal', 300), ('bal', 100), ('zh', 300), ('tile', 80), ('tile', 82)):
        pg.evaluate(f"()=>{{location.hash='tokenization-cost~{mix}_{k}'}}"); pg.wait_for_timeout(400)
        rows = pg.evaluate("()=>[...document.querySelectorAll('#tk-mg-list li:not(.tk-mg-h)')].map(l=>[...l.querySelectorAll('.tk-where')].map(w=>w.textContent))")
        first = pg.evaluate("()=>[...document.querySelectorAll('#tk-mg-first .tk-where')].map(w=>w.textContent)")
        tbm = tok_bytes(MER[mix])
        for r, idx in zip(rows, range(k - 1, k - 4, -1)):
            a, b_, _ = MER[mix][idx]; R = tbm[256 + idx]
            want = []
            if any(x >= 128 for x in R):
                note = alone(R) if not ok_utf8(R) else None
                i = -1 if (note and note[1]) else ZB.find(R)
                for bs, at in ((tbm[a], i), (tbm[b_], i + len(tbm[a]) if i >= 0 else -1)):
                    if not ok_utf8(bs) and at >= 0: want.append('(' + span(at, len(bs)) + ')')
                if note: want.append('(' + note[0] + ')'); nshared += note[1]
            hn += 1; notes += len(want)
            if r != want: hbad.append((mix, k, idx + 1, r, want))
        fw = [f'({alone(tbm[256 + i])[0]})' for i in range(3) if not ok_utf8(tbm[256 + i])]
        if first != fw: hbad.append((mix, k, 'first merges', first, fw))
        notes += len(fw)
    check(f'A4/N3: the {notes} character notes of {hn} merge rows and the first merges (6 setups) match the bytes’ place in the Chinese paragraph, {nshared} of them shared by several characters',
          not hbad and hn == 18 and notes >= 14 and nshared >= 1, (notes, nshared, hbad[:3]))
    chars, _ = shared(bytes.fromhex('e4b8'))
    check('N3: e4 b8 starts 6 different characters of the paragraph: 上 (3 times), 一 (3), 个 (2), 下, 万 and 不 (11 in all)',
          chars == ['上', '一', '个', '下', '万', '不'] and ZB.count(bytes.fromhex('e4b8')) == 11 and MER['tile'][79][:2] == (0xe4, 0xb8) and MER['tile'][79][2] == 11, (chars, MER['tile'][79]))
    # N4 (round 1b): the characters that are whole tokens of the Chinese sentence at half and half, 100 merges
    tbb = tok_bytes(MER['bal'])
    whole = [tbb[t].decode() for t in encode(PAIRS['main']['zh'], MER['bal'], 100) if ok_utf8(tbb[t]) and len(tbb[t].decode()) == 1]
    check('N4: at half and half, 100 merges, “我”, “在”, “上”, “书” and “。” are single tokens (part 7 lists all five)', whole == ['我', '在', '上', '书', '。'], whole)
    # S2 (N1, round 1b): Show the math rounds the quotient to one decimal, then writes the whole copies (the floor)
    def q1(x):
        for d in range(1, 5):
            t_ = f'{x:,.{d}f}'
            if not re.search(r'\.0+$', t_): return t_
        return f'{x:,.4f}'
    mbad, mseen = [], set()
    for h in ('tile_0', 'tile_100', 'bal_100', 'zh_300', 'en_37', 'tile_400', 'bal_400', 'zh_50', 'tile_30'):
        pg.evaluate(f"()=>{{location.hash='tokenization-cost~{h}'}}"); pg.wait_for_timeout(400)
        mt = pg.evaluate("()=>document.getElementById('tk-mathbody').textContent")
        for m in re.finditer(r'131,072 ÷ (\d+) = ([\d,]+(?:\.\d+)?)(?:, so ([\d,]+) whole copies)?', mt):
            nn, q, fl = int(m.group(1)), m.group(2), m.group(3); mseen.add(nn)
            if 131072 % nn:
                if q != q1(131072 / nn) or fl is None or int(fl.replace(',', '')) != 131072 // nn: mbad.append((h, m.group(0), q1(131072 / nn)))
            elif q.replace(',', '') != str(131072 // nn): mbad.append((h, m.group(0)))
        if len(re.findall(r'131,072 ÷', mt)) != 2: mbad.append((h, 'lines', mt[:200]))
    check(f'S2/N1: every “131,072 ÷ n” in Show the math is rounded to one decimal, then the whole copies ({len(mseen)} divisors in 9 setups, 36 → 3,640.9 among them)',
          not mbad and len(mseen) >= 10 and 36 in mseen, (sorted(mseen), mbad[:3]))
    hk = pg.evaluate("()=>Array.from({length:400},(_,i)=>window.__tk.q1(131072/(i+1)))")
    qbad = [(n_ + 1, hk[n_], q1(131072 / (n_ + 1))) for n_ in range(400) if 131072 % (n_ + 1) and hk[n_] != q1(131072 / (n_ + 1))]
    check('N1: the page’s quotient for every divisor from 1 to 400 equals Python’s rounding to one decimal (more digits only where one would read x.0)', not qbad, qbad[:5])
    # ---------- the Connect card's code ----------
    code = consts['CODE']
    lines = code.split('\n')
    import unicodedata
    width = lambda l: sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in l)
    check('Connect code: at most 40 lines, each at most 74 columns wide', len(lines) <= 40 and max(width(l) for l in lines) <= 74, (len(lines), max(width(l) for l in lines)))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True).stdout.strip().split('\n')
    want = [l[2:] for l in lines if l.startswith('# ') and '->' in l]
    check('Connect code runs and prints what its last comment lines say', out == want, (out, want))
    check('no page errors', not errs, errs[:3])
    b.close()
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:30] if fails else 'none')
sys.exit(1 if fails else 0)
