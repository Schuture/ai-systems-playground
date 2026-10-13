"""Set B teasers: recompute every number the two tiles can show, independently in Python, and compare with
the page (the read-only hook window.__tzB and the text actually rendered in each tile), for every control
position, in English and Chinese.

  python3 tests/set-b_numbers.py

Independent derivations (not ports of the JS):
  sampling         Hugging Face's TopPLogitsWarper rule (drop tokens whose ascending cumulative mass <= 1 - top_p)
  tokenization     a plain, non-incremental byte-level BPE with the `regex` module's \\p{L} pre-tokenizer
The training-memory and parallelism tiles left set B in batch 8, and the two post-training tiles in batch 9 (those
labs' own tiles replaced them).
"""
import math, pathlib, subprocess, sys
from decimal import Decimal, ROUND_HALF_UP
import regex
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/set-b.html', '--only', 'set-b'], check=True, cwd=P, capture_output=True)
WRAP = P / 'out' / 'wrapped-set-b.html'
WRAP.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'set-b.html').read_text() + '</body></html>')
URL = 'file://' + str(WRAP)
GiB, GB, MINUS = 2 ** 30, 1e9, '−'

# ---------- formatting exactly as the page does (JS toFixed: exact binary value, ties away from zero) ----------
def fixed(v, d):
    return str(Decimal(abs(v)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
def nf(v, d=0):
    s = fixed(v, d); neg = v < 0 and Decimal(s) != 0
    i, _, f = s.partition('.')
    i = f'{int(i):,}'
    return (MINUS if neg else '') + i + ('.' + f if f else '')
def pct(x):
    v = x * 100
    d = 1 if (v < 10 or v > 90) and abs(v - math.floor(v + 0.5)) > 1e-9 else 0
    return nf(v, d) + '%'

checks, fails = 0, []
def check(ok, msg):
    global checks
    checks += 1
    if not ok: fails.append(msg)
def close(a, b, rel=1e-9, tol=1e-12):
    return abs(a - b) <= max(tol, rel * max(abs(a), abs(b)))
def has_all(text, needles, where):
    for n in needles: check(n in text, f'{where}: expected “{n}” in: {text[:260]}')

# ---------- independent models ----------
# sampling: toy logits for 6 words; temperature 0.1 .. 2.0; top-p 0.9 applied the way Hugging Face does it
LOGITS, TOP_P = [2.0, 1.5, 1.0, 0.3, -0.5, -1.0], 0.9
def softmax(z):
    m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e); return [v / s for v in e]
def samp(T):
    p = softmax([l / T for l in LOGITS])
    asc = sorted(range(6), key=lambda i: (p[i], -i))           # ascending, ties: later index first (removed first)
    cum, drop = 0.0, set()
    for i in asc:
        cum += p[i]
        if cum <= 1 - TOP_P: drop.add(i)
    keep = [i not in drop for i in range(6)]
    if not any(keep): keep[max(range(6), key=lambda i: p[i])] = True
    z = sum(p[i] for i in range(6) if keep[i])
    fin = [p[i] / z if keep[i] else 0.0 for i in range(6)]
    top = max(range(6), key=lambda i: (p[i], -i))
    return dict(p=p, keep=keep, fin=fin, n=sum(keep), top=top, raw=softmax(LOGITS))
WORDS = {'en': ['noodles', 'rice', 'dumplings', 'hot pot', 'salad', 'a burger'], 'zh': ['面条', '米饭', '饺子', '火锅', '沙拉', '汉堡']}

# tokenization: plain byte-level BPE
PRE = regex.compile(r' ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+')
def bpe_train(text, kmax):
    freq = {}
    for c in PRE.findall(text): freq[c] = freq.get(c, 0) + 1
    words = [[list(c.encode('utf-8')), f] for c, f in freq.items()]
    merges = []
    for i in range(kmax):
        cnt = {}
        for seq, f in words:
            for a, b in zip(seq, seq[1:]): cnt[(a, b)] = cnt.get((a, b), 0) + f
        if not cnt: break
        (a, b), c = min(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
        if c < 2: break
        merges.append((a, b, c)); nid = 256 + i
        for w in words: w[0] = apply(w[0], a, b, nid)
    return merges
def apply(seq, a, b, nid):
    out, j = [], 0
    while j < len(seq):
        if j + 1 < len(seq) and seq[j] == a and seq[j + 1] == b: out.append(nid); j += 2
        else: out.append(seq[j]); j += 1
    return out
def bpe_count(sentence, merges, k):
    seqs = [list(c.encode('utf-8')) for c in PRE.findall(sentence)]
    for i, (a, b, _) in enumerate(merges[:k]): seqs = [apply(s, a, b, 256 + i) for s in seqs]
    return sum(len(s) for s in seqs)

# ---------- page helpers ----------
def world(pg, w):
    pg.evaluate("w=>document.querySelector('#mapSvg .world[data-world=\"'+w+'\"]').dispatchEvent(new MouseEvent('click',{bubbles:true}))", w)
    pg.wait_for_timeout(120)
def set_range(pg, lab, v):
    pg.evaluate("([l,v])=>{const i=document.querySelector('#tile-'+l+' input[type=range]'); i.value=v; i.dispatchEvent(new Event('input',{bubbles:true}))}", [lab, v])
def text(pg, sel): return pg.evaluate("s=>{const e=document.querySelector(s); return e?e.textContent:''}", sel)
def attr(pg, sel, a): return pg.evaluate("([s,a])=>{const e=document.querySelector(s); return e?e.getAttribute(a)||'':''}", [sel, a])

with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900}); errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL + '#map'); pg.wait_for_timeout(300)
    hook = 'window.__tzB'

    # ===== hook values vs independent values =====
    for t in range(1, 21):
        got, want = pg.evaluate(f"T=>{hook}.samp.compute(T,{TOP_P})", t / 10), samp(t / 10)
        check(got['keep'] == want['keep'], f'samp T={t/10} kept set {got["keep"]} vs {want["keep"]}')
        check(all(close(x, y) for x, y in zip(got['fin'], want['fin'])), f'samp T={t/10} final probs')
        check(all(close(x, y) for x, y in zip(got['p'], want['p'])), f'samp T={t/10} probs')
    corpus = pg.evaluate(f"()=>{hook}.tok.corpus()")
    sent = pg.evaluate(f"()=>{hook}.tok.SENTENCES")
    TK_STOPS = pg.evaluate(f"()=>{hook}.tok.STOPS")
    merges = bpe_train(corpus['text'], max(TK_STOPS))
    got_m = pg.evaluate(f"()=>{hook}.tok.merges()")
    check([tuple(m) for m in got_m] == merges, f'BPE merge lists differ: first mismatch at '
          f'{next((i for i, (x, y) in enumerate(zip(got_m, merges)) if tuple(x) != y), min(len(got_m), len(merges)))}')
    counts = {k: (bpe_count(sent['en'], merges, k), bpe_count(sent['zh'], merges, k)) for k in range(max(TK_STOPS) + 1)}
    got_c = pg.evaluate(f"K=>Array.from({{length:K+1}},(_,k)=>{{const r={hook}.tok.compute(k); return [r.en,r.zh]}})", max(TK_STOPS))
    check([tuple(x) for x in got_c] == [counts[k] for k in range(max(TK_STOPS) + 1)], 'BPE token counts differ for some merge count')
    check(counts[0] == (len(sent['en'].encode()), len(sent['zh'].encode())), 'at 0 merges the counts must equal UTF-8 bytes')
    check(all(len(ch.encode()) == 3 for ch in sent['zh']), 'every character of the Chinese sentence is 3 bytes in UTF-8')
    for a, b2 in zip(TK_STOPS, TK_STOPS[1:]):
        check(counts[a] != counts[b2], f'slider stops {a} -> {b2} give the same counts {counts[a]}: the sentence would not change')
    en_b, zh_b = len(corpus['en'].encode()), len(corpus['zh'].encode())

    # ===== what each tile shows, for every control position, in both languages =====
    # each tile (guide R11.5): one or two sentences (.tz-read, about 35 English words at most), a legend (.tz-leg),
    # notes (.tz-note, read through the tile's "Assumptions & sources" button) of 2 to 4 short sentences;
    # every number in them must match the independent values
    def sentence(lab, lang, where):
        rd = text(pg, f'#tile-{lab} .tz-read')
        if lang == 'en':
            words = len(rd.split())
            check(words <= 35, f'{where}: sentence has {words} words (max 35): {rd}')
        check(rd.count('. ') <= 1 and rd.count('。') <= 2, f'{where}: more than two sentences: {rd}')
        check('·' not in rd and '→' not in rd, f'{where}: "·" or "→" in the sentence: {rd}')
        return rd
    def notes(lab):
        paras = [x for x in pg.evaluate(f"()=>Array.from(document.querySelectorAll('#tile-{lab} .tz-note'),n=>n.textContent)") if x]
        t = ' '.join(paras); lang_ = pg.evaluate('()=>document.documentElement.lang')
        n = len(regex.findall('[。！？]', t)) if lang_.startswith('zh') else len(regex.findall(r'[.!?](?=\s|$)', t.replace('et al.', 'et al')))
        check(2 <= n <= 4, f'{lab} {lang_}: notes have {n} sentences (guide R11.5: 2 to 4): {t[:200]}')
        check('·' not in t and all(x.strip().endswith(('.', '。')) for x in paras), f'{lab} {lang_}: a note is not plain sentences: {t[:200]}')
        return t
    for lang in ('en', 'zh'):
        if lang == 'zh': pg.click('.langsw button[data-lang="zh"]'); pg.wait_for_timeout(200)
        world(pg, 'models')
        for t in range(1, 21):
            set_range(pg, 'sampling', t); pg.wait_for_timeout(20)
            where = f'samp T={t/10} {lang}'
            rd = sentence('sampling', lang, where); s_ = samp(t / 10); top = WORDS[lang][s_['top']]
            need = [nf(t / 10, 1), '0.9', top, pct(s_['raw'][s_['top']])]
            need += [pct(s_['p'][s_['top']])] if s_['n'] == 1 else [str(s_['n']), '6', pct(s_['fin'][s_['top']])]
            has_all(rd, need, where)
            vals = pg.evaluate("()=>Array.from(document.querySelectorAll('#tile-sampling .tz-samp-v'),e=>e.textContent)")
            want = [pct(s_['fin'][i]) if s_['keep'][i] else ('cut' if lang == 'en' else '去掉') for i in range(6)]
            check(vals == want, f'{where} bar values {vals} vs {want}')
        nt = notes('sampling'); has_all(nt, ['0.9', '6', 'Holtzman'], f'samp notes {lang}')
        check(('so you can see' if lang == 'en' else '好让你看到') not in nt, f'samp notes {lang}: a sentence about the tile itself (R3.4)')
        for i, k in enumerate(TK_STOPS):
            set_range(pg, 'tokenization-cost', i); pg.wait_for_timeout(30)
            where = f'tok k={k} {lang}'
            rd = sentence('tokenization-cost', lang, where); e, z = counts[k]; e0, z0 = counts[0]
            need = [str(e), str(z)]
            if k:
                need += [str(k), nf(en_b / (en_b + zh_b) * 100) + '%', str(e0)]
                if z != z0: need += [str(z0)] + ([nf(z / e, 1)] if z > e else [])
                # SITE-A-r7 item 14b: counts in words, each saying it counts tokens
                need += [f'English drops from {e0} tokens to {e}' if lang == 'en' else f'英文从 {e0} 个 token 减到 {e} 个']
                if z != z0: need += [f'from {z0} to {z}' if lang == 'en' else f'从 {z0} 个减到 {z} 个']
            has_all(rd, need, where)
            bars = pg.evaluate("()=>[document.querySelectorAll('#tile-tokenization-cost .tz-tokn-en .tz-tokn-boxes i').length,document.querySelectorAll('#tile-tokenization-cost .tz-tokn-zh .tz-tokn-boxes i').length]")
            check(tuple(bars) == (e, z), f'{where}: bars {bars} vs counts {(e, z)}')
        nt = notes('tokenization-cost')
        has_all(nt, [nf(en_b), nf(zh_b), nf(en_b / (en_b + zh_b) * 100) + '%', '256', '15', 'Petrov'], f'tok notes {lang}')
        check('RFC 3629' not in nt, f'tok notes {lang}: the UTF-8 sentence should be cut (SITE-A-r7 item 12)')
    b.close()

check(not errs, f'page errors: {errs}')
print(f'set-b numbers: {checks} checks, {len(fails)} failed')
for f in fails[:40]: print('  FAIL', f)
print('key values:',
      f"sampling T=0.7 {samp(0.7)['n']} kept, {pct(samp(0.7)['fin'][0])};",
      f"BPE {counts[0]} -> {counts[100]} at 100 -> {counts[300]} at 300")
sys.exit(1 if fails else 0)
