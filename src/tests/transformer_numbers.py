"""Golden numbers for the Transformer ① lab (`transformer`), recomputed independently in Python.

Not a port of labs/transformer.js: the counts are derived here from the layer shapes of the reference implementations
(Meta's models/llama3/model.py, transformers' modeling_llama.py: Wq d×(H·hd), Wk/Wv d×(KV·hd), Wo (H·hd)×d, gate/up d×f,
down f×d, two RMSNorm vectors of d per layer, a final RMSNorm, an embedding table V×d and an LM head V×d unless tied), and
compared with (1) the totals in specs/transformer-series.md and the Hugging Face safetensors counts in specs/transformer-facts.md
(A9), (2) the page's engine through its read-only hook window.__tf, and (3) the text the page shows in every guided part, the
scenarios, the challenge, Show the math, the drill and the home tile. Every formula line on the page is parsed and re-evaluated
(R6.6: it must reproduce from the numbers it shows). Counts over 6 digits are shown as 8.03B / 525M with the exact value on hover
(R6.18, `A.ui.big`): the checks read the visible text for the short form and the hover values for the exact arithmetic.

Run: python3 tests/transformer_numbers.py
Optional: TF_LLAMA_TOKENIZER=/path/to/llama3/tokenizer.model also checks the token ids 374 (" is") and 12366 (" Paris") with tiktoken.
"""
import math, os, pathlib, re, subprocess, sys
from fractions import Fraction
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/transformer.html', '--only', 'transformer'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-transformer.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'transformer.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-transformer.html')
GiB = 2 ** 30
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name)
        print('FAIL', name, '::', str(info)[:400])


# ---------------------------------------------------------------- independent model
# config.json values (specs/transformer-facts.md A1–A5)
CFG = {
    'llama32-1b': dict(V=128256, d=2048, L=16, H=32, KV=8, f=8192, tied=True),
    'llama31-8b': dict(V=128256, d=4096, L=32, H=32, KV=8, f=14336, tied=False),
    'mistral-7b': dict(V=32000, d=4096, L=32, H=32, KV=8, f=14336, tied=False),
    'llama31-70b': dict(V=128256, d=8192, L=80, H=64, KV=8, f=28672, tied=False),
    'llama31-405b': dict(V=128256, d=16384, L=126, H=128, KV=8, f=53248, tied=False),
}
NAMES = {'llama32-1b': 'Llama-3.2-1B', 'llama31-8b': 'Llama-3.1-8B', 'mistral-7b': 'Mistral-7B-v0.1', 'llama31-70b': 'Llama-3.1-70B', 'llama31-405b': 'Llama-3.1-405B'}
# the spec's table and the Hugging Face safetensors metadata (facts A9)
TOTAL = {'llama32-1b': 1235814400, 'llama31-8b': 8030261248, 'mistral-7b': 7241732096, 'llama31-70b': 70553706496, 'llama31-405b': 405853388800}
BYTES = {'bf16': 2, 'fp8': 1, 'fp4': 0.5}


def tensors(c):
    """every weight tensor of the model, by name and shape, as the reference code creates them"""
    hd = c['d'] // c['H']
    t = [('embed_tokens', c['V'], c['d'])]
    for i in range(c['L']):
        t += [(f'L{i}.q_proj', c['H'] * hd, c['d']), (f'L{i}.k_proj', c['KV'] * hd, c['d']), (f'L{i}.v_proj', c['KV'] * hd, c['d']),
              (f'L{i}.o_proj', c['d'], c['H'] * hd), (f'L{i}.gate_proj', c['f'], c['d']), (f'L{i}.up_proj', c['f'], c['d']),
              (f'L{i}.down_proj', c['d'], c['f']), (f'L{i}.input_layernorm', c['d'], 1), (f'L{i}.post_attention_layernorm', c['d'], 1)]
    t += [('norm', c['d'], 1)]
    if not c['tied']:
        t += [('lm_head', c['V'], c['d'])]
    return t


def count(c):
    ts = tensors(c)
    tot = sum(a * b for _, a, b in ts)
    kind = lambda nm: ('attn' if re.search(r'[qkvo]_proj$', nm) else 'mlp' if re.search(r'(gate|up|down)_proj$', nm) else
                       'emb' if nm == 'embed_tokens' else 'head' if nm == 'lm_head' else 'norm')
    parts = {k: 0 for k in ('attn', 'mlp', 'emb', 'head', 'norm')}
    for nm, a, b in ts:
        parts[kind(nm)] += a * b
    layer = sum(a * b for nm, a, b in ts if nm.startswith('L0.'))
    attn1 = sum(a * b for nm, a, b in ts if nm.startswith('L0.') and kind(nm) == 'attn')
    mlp1 = sum(a * b for nm, a, b in ts if nm.startswith('L0.') and kind(nm) == 'mlp')
    return dict(total=tot, parts=parts, layer=layer, attn=attn1, mlp=mlp1, emb=c['V'] * c['d'], hd=c['d'] // c['H'], kvw=c['KV'] * (c['d'] // c['H']))


def fast(c):
    """closed form (used for the challenge sweep; checked against count() below)"""
    hd = c['d'] // c['H']
    layer = 2 * c['d'] * c['d'] + 2 * c['d'] * c['KV'] * hd + 3 * c['d'] * c['f'] + 2 * c['d']
    return c['V'] * c['d'] * (1 if c['tied'] else 2) + c['L'] * layer + c['d']


def share_str(frac):
    """three significant digits, at least one decimal, at most three (the page's rule, restated)"""
    v = frac * 100
    if v <= 0:
        return '0%'
    if v < 0.0005:
        return '<0.001%'
    d = max(1, min(3, 2 - math.floor(math.log10(v))))
    return f'{round(v + 1e-12, d):.{d}f}%'


def delta_txt(a, b, lang, init=False):
    """the kit's A.fmt.delta rules (app.js deltaVs), written again: a factor from 2 up, else a whole percent, half away from zero.
    init: compared with the initial setup (parts 5 and 6) instead of the last value"""
    if a == b:
        return ''
    q = b / a
    if q >= 1.995:
        n = f'{round(q):,}' if q >= 9.95 else f'{q:.1f}'.removesuffix('.0')
        if lang == 'en':
            return f'▲ {n} times ' + ('the initial setup' if init else 'the last value')
        return f'▲ 是{"初始设置" if init else "上次"}的 {n} 倍'
    r = (q - 1) * 100
    p = int(math.copysign(math.floor(abs(r) + 1e-9 + 0.5), r))
    if p == 0:
        return ''
    return ('▲ +' if p > 0 else '▼ −') + str(abs(p)) + '%' + ((' vs the initial setup' if init else ' vs last') if lang == 'en' else ('（比初始设置）' if init else '（比上次）'))


def pts_txt(a, b, lang, init=False):
    """a share's change in points from the two shown values, with the decimals they are shown with"""
    dp = max(len(x.rstrip('%').split('.')[1]) if '.' in x else 0 for x in (a, b))
    dd = round(float(b.rstrip('%')) - float(a.rstrip('%')), dp)
    if abs(dd) < 0.5 * 10 ** -dp:
        return ''
    n = f'{abs(dd):.{dp}f}'
    return ('▲ ' if dd > 0 else '▼ ') + n + ((' pt' if n == '1' else ' pts') + (' vs the initial setup' if init else ' vs last') if lang == 'en' else ' 个百分点' + ('（比初始设置）' if init else '（比上次）'))


def g1(x):
    return f'{x:,.1f}'


def mem(total, b=2):
    return f'{g1(total * b / GiB)} GiB', f'{g1(total * b / 1e9)} GB'


R = {k: count(c) for k, c in CFG.items()}
for k in CFG:
    check(f'{k}: tensor sum = spec table / safetensors count', R[k]['total'] == TOTAL[k], (R[k]['total'], TOTAL[k]))
    check(f'{k}: closed form = tensor sum', fast(CFG[k]) == R[k]['total'])
r8 = R['llama31-8b']
check('8B per layer: attention 41,943,040, MLP 176,160,768, layer 218,112,000, embedding 525,336,576',
      (r8['attn'], r8['mlp'], r8['layer'], r8['emb']) == (41943040, 176160768, 218112000, 525336576), (r8['attn'], r8['mlp'], r8['layer'], r8['emb']))
check('8B shares as the spec states them (70.2%, 16.7%, 6.5% each)', [round(r8['parts'][k] / r8['total'] * 100, 1) for k in ('mlp', 'attn', 'emb', 'head')] == [70.2, 16.7, 6.5, 6.5])
check('8B weights in BF16: 15.0 GiB (16.1 GB)', mem(r8['total']) == ('15.0 GiB', '16.1 GB'), mem(r8['total']))

# ---------------------------------------------------------------- the challenge, swept independently
CH = dict(CFG['llama31-8b'])
fits = [L for L in range(1, 129) if fast({**CH, 'L': L}) * 2 <= 20 * GiB]
best = max(fits)
check('challenge: the most layers in 20 GiB of BF16 weights is 44 (spec)', best == 44, best)
check('challenge: 44 layers 19.8 GiB, 45 layers 20.2 GiB', (g1(fast({**CH, 'L': 44}) * 2 / GiB), g1(fast({**CH, 'L': 45}) * 2 / GiB)) == ('19.8', '20.2'))
per_layer = r8['layer'] * 2 / GiB
fixed = (2 * r8['emb'] + CH['d']) * 2 / GiB
check('challenge hint: each layer 0.406 GiB in BF16, the fixed part 1.957 GiB', (f'{per_layer:.3f}', f'{fixed:.3f}') == ('0.406', '1.957'))
q = (20 - round(fixed, 3)) / round(per_layer, 3)
check('challenge reason: (20 − 1.957) ÷ 0.406 = 44.4 from the shown values, floor 44', f'{math.floor(q * 10) / 10:.1f}' == '44.4' and math.floor(q) == best, q)

# ---------------------------------------------------------------- part 6 and part 5
wide = count({**CH, 'd': 8192, 'f': 28672})
deep = count({**CH, 'L': 64})
check('part 6 wide: 30,019,166,208 total, 872,431,616 per layer, 3.7 and 4.0 times', (wide['total'], wide['layer'], f"{wide['total'] / r8['total']:.1f}", f"{wide['layer'] / r8['layer']:.1f}") == (30019166208, 872431616, '3.7', '4.0'))
check('part 6 deep: 15,009,845,248 total, layers 13,959,168,000, 1.9 times', (deep['total'], 64 * r8['layer'], f"{deep['total'] / r8['total']:.1f}") == (15009845248, 13959168000, '1.9'))
diff = R['llama31-8b']['total'] - R['mistral-7b']['total']
check('part 5: 788,529,152 = (128,256 − 32,000) × 4,096 × 2', diff == 788529152 == (128256 - 32000) * 4096 * 2)
untied1 = count({**CFG['llama32-1b'], 'tied': False})['total']
check('part 5: tying saves 262,668,288 in Llama-3.2-1B', untied1 - R['llama32-1b']['total'] == 262668288)
mult8 = r8['total'] - r8['emb']
check('Show the math: 7,504,924,672 weights multiplied per token, 2 × that ≈ 15.0 billion FLOPs', mult8 == 7504924672 and f'{2 * mult8 / 1e9:.1f}' == '15.0')
h16 = count({**CH, 'H': 16})
check('part 7: with 16 heads head_dim goes from 128 to 256, Wk and Wv double to 4,096 × 2,048 and the model has 8,298,696,704 parameters', (CH['d'] // 32, CH['d'] // 16, h16['kvw'], h16['total']) == (128, 256, 2048, 8298696704), (h16['kvw'], h16['total']))
kv_tok = 2 * CH['L'] * CH['KV'] * (CH['d'] // CH['H']) * 2
check('KV cache definition: 128 KiB per token for Llama-3.1-8B in BF16, each GiB holds 8,192 tokens (4 GiB, the challenge’s room, 32,768)', (kv_tok, GiB // kv_tok, GiB % kv_tok, 4 * GiB // kv_tok) == (131072, 8192, 0, 32768), kv_tok)
check('challenge note: 176 KiB per token at 44 layers', 2 * 44 * 1024 * 2 // 1024 == 176)
check('scenario 1B: 0.6 GiB in FP4', g1(R['llama32-1b']['total'] * 0.5 / GiB) == '0.6')
check('scenario 7B: 1.5 GiB less than Llama-3.1-8B, from the shown 15.0 and 13.5', (g1(R['llama31-8b']['total'] * 2 / GiB), g1(R['mistral-7b']['total'] * 2 / GiB), f"{float(g1(R['llama31-8b']['total'] * 2 / GiB)) - float(g1(R['mistral-7b']['total'] * 2 / GiB)):.1f}") == ('15.0', '13.5', '1.5'))
check('scenario 405B: 756.0 GiB is more than 8 × 80 GiB = 640 GiB', R['llama31-405b']['total'] * 2 / GiB > 640 and g1(R['llama31-405b']['total'] * 2 / GiB) == '756.0')
SCALE = CFG['llama31-70b']['V'] * CFG['llama31-70b']['d']
check('stage scale: a full-length bar is 1,050,673,152 = Llama-3.1-70B’s embedding table = the “twice as wide” one, the longest in the guided parts',
      SCALE == 1050673152 == 128256 * 8192 and max(wide['emb'], wide['layer'], r8['emb'], r8['layer'], R['mistral-7b']['emb']) == SCALE)
mlps = {k: R[k]['parts']['mlp'] / R[k]['total'] for k in CFG}
attns = {k: R[k]['parts']['attn'] / R[k]['total'] for k in CFG}
check('part 7: MLP share from 65.2% (1B) to 81.3% (405B), attention 13.6% to 18.5%',
      (share_str(min(mlps.values())), min(mlps, key=mlps.get), share_str(max(mlps.values())), max(mlps, key=mlps.get), share_str(min(attns.values())), share_str(max(attns.values())))
      == ('65.2%', 'llama32-1b', '81.3%', 'llama31-405b', '13.6%', '18.5%'))

# ---------------------------------------------------------------- R6.6: every formula line reproduces from its own numbers
NUM = r'\d[\d,]*(?:\.\d+)?'


def eval_line(line):
    """evaluate the arithmetic of a line like 'model = embedding table (128,256 × 4,096) + 32 × 218,112,000 + … = 8,030,261,248'.
    Labels and brackets that only name a term are dropped; the right-most '= N' is the stated result."""
    t = line.replace('‑', '-').replace(' ', ' ').replace('−', '-').replace('（', '(').replace('）', ')')
    t = re.sub(r'\s*\([^()]*=[^()]*\)\s*$', '', t)                   # a trailing note like "(1,024 = 8 KV heads × 128)" is checked on its own
    if '=' not in t:
        return None
    lhs, res = t.rsplit('=', 1)
    lhs = lhs.split('=', 1)[1] if lhs.count('=') else lhs
    m = re.match(r'\s*(' + NUM + r')([MB](?![A-Za-z]))?', res)
    if not m:
        return None
    stated = Fraction(m.group(1).replace(',', ''))
    expr = re.sub(r'[^\d,.\s×÷+\-()]', ' ', lhs)                     # drop words (labels), keep numbers and operators
    expr = re.sub(r'(?<=\d),(?=\d{3})', '', expr).replace('×', '*').replace('÷', '/')
    expr = re.sub(r'\(\s*\)', '', expr)
    expr = re.sub(r'\s+', ' ', expr).strip()
    try:
        val = eval(re.sub(r'(\d+(?:\.\d+)?)', r'Fraction("\1")', expr), {'Fraction': Fraction})
    except Exception as e:
        return ('error', str(e), expr)
    return (val, stated, expr, m.group(1).replace(',', '') + (m.group(2) or ''))


def reproduces(line, tol_rel=0):
    r = eval_line(line)
    if r is None or r[0] == 'error':
        return False, r
    val, stated, expr, shown = r
    # a result written with M or B (R6.18; no exact-value hover since the owner's clarification of 2026-10-07): it must be the
    # exact result at 3 significant digits
    if shown[-1:] in ('M', 'B'):
        return big_str(round(val)) == shown, (float(val), shown, expr)
    # a rounded result: equal when rounded to the decimals it is shown with ("44.4" has one)
    dec = len(shown.split('.')[1]) if '.' in shown else 0
    ok = val == stated or round(float(val), dec) == float(stated)
    return ok, (float(val), float(stated), expr)


# ---------------------------------------------------------------- the page
def page_text(pg, sel):
    return pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u2011/g,'-').replace(/\\u00a0/g,' '):''}", sel)


EXACT_JS = """s=>{const e=document.querySelector(s); if(!e) return '';
  const bs=[...e.querySelectorAll('.num-big')], keep=bs.map(x=>x.textContent);
  bs.forEach(x=>{const m=(x.getAttribute('data-tipx')||'').match(/\\d[\\d,]*/); if(m) x.textContent=m[0];});
  const t=e.innerText; bs.forEach((x,i)=>{x.textContent=keep[i];});
  return t.replace(/\\u2011/g,'-').replace(/\\u00a0/g,' ');}"""


def exact_text(pg, sel):
    """the text with every big number (R6.18) replaced by the exact value its hover gives"""
    return pg.evaluate(EXACT_JS, sel)


def no_big_hover(pg, sel):
    """the owner's clarification (2026-10-07 23:37 UTC): a big number is plain text, with no hover that gives its exact value"""
    return pg.evaluate("s=>[...document.querySelectorAll(s+' .num-big')].every(x=>!x.hasAttribute('data-tipx')&&!x.closest('.def')&&!x.hasAttribute('tabindex'))", sel)


def bigs(pg, sel):
    """[(shown, exact)] for every big number in sel"""
    return pg.evaluate("s=>[...document.querySelectorAll(s+' .num-big')].map(x=>[x.textContent,(x.getAttribute('data-tipx')||'')])", sel)


def big_str(n):
    """the kit's A.fmt.big, restated: 3 significant digits with M or B from a million up"""
    a = abs(n)
    if a < 1e6:
        return f'{n:,}'
    d, u = (1e9, 'B') if a >= 1e9 else (1e6, 'M')
    v = n / d
    dp = max(0, 2 - math.floor(math.log10(abs(v))))
    t = f'{v:.{dp}f}'
    if u == 'M' and abs(float(t)) >= 1000:
        v = n / 1e9; u = 'B'; dp = max(0, 2 - math.floor(math.log10(abs(v)))); t = f'{v:.{dp}f}'
    return t + u


LONG = re.compile(r'\d{1,3}(?:,\d{3}){2,}')   # 7 digits or more, written out
nth = lambda s, i: ([l for l in s.split('\n') if l.strip()] + ['', '', ''])[i]      # the i-th non-empty line


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width': 1440, 'height': 1000})
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(100)
    pg.evaluate("()=>{location.hash='transformer'}"); pg.wait_for_timeout(500)
    # the engine, through the hook, for every preset and precision
    for k, c in CFG.items():
        for prec, by in BYTES.items():
            js = pg.evaluate("([c,p])=>{const r=window.__tf.compute({...c,p}); return {total:r.total,parts:r.parts,layer:r.layer,attn:r.attn,mlp:r.mlp,emb:r.emb,bytes:r.bytes,gib:r.gib,mult:r.mult}}", [c, prec])
            mine = R[k]
            check(f'{k} {prec}: engine total and parts', js['total'] == mine['total'] and all(js['parts'][x] == mine['parts'][x] for x in mine['parts']) and js['layer'] == mine['layer'] and js['attn'] == mine['attn'] and js['mlp'] == mine['mlp'], (js, mine))
            check(f'{k} {prec}: engine bytes', js['bytes'] == mine['total'] * by, (js['bytes'], mine['total'] * by))
            check(f'{k}: weights multiplied per token (all but an untied embedding table)', js['mult'] == mine['total'] - (0 if c['tied'] else mine['emb']))
        shown = pg.evaluate("c=>{const r=window.__tf.compute({...c,p:'bf16'}), f=window.__tf.fmt; return {mlp:f.pctS(r.parts.mlp/r.total),attn:f.pctS(r.parts.attn/r.total),emb:f.pctS(r.parts.emb/r.total),norm:f.pctS(r.parts.norm/r.total),voc:f.vocShare(r)}}", c)
        mine = R[k]; t = mine['total']
        check(f'{k}: shares as shown', (shown['mlp'], shown['attn'], shown['emb'], shown['norm']) == (share_str(mine['parts']['mlp'] / t), share_str(mine['parts']['attn'] / t), share_str(mine['parts']['emb'] / t), share_str(mine['parts']['norm'] / t)), shown)
        # the embedding table and LM head card is the sum of the two shown shares (R6.6)
        e = share_str(mine['parts']['emb'] / t)
        if c['tied']:
            check(f'{k}: tied table card = its one share', shown['voc'] == e, shown['voc'])
        else:
            dd = len(e.rstrip('%').split('.')[1])
            check(f'{k}: vocabulary card = shown embedding + shown LM head share', shown['voc'] == f"{2 * float(e.rstrip('%')):.{dd}f}%", (shown['voc'], e))
    ch = pg.evaluate("()=>window.__tf.CH")
    check('engine: challenge answer 44, rows 32–52, budget 20 GiB on a 24 GiB RTX 4090', (ch['best'], ch['rows'], ch['budget'], ch['gpu'], ch['min'], ch['max']) == (44, [32, 36, 40, 44, 48, 52], 20, 24, 32, 56), ch)
    for L in range(ch['min'], ch['max'] + 1):
        pt = pg.evaluate("l=>window.__tf.chPoint(l)", L)
        mine = fast({**CH, 'L': L}) * 2 / GiB
        check(f'challenge {L} layers: {g1(mine)} GiB, fits {mine <= 20}', abs(pt['gib'] - mine) < 1e-9 and pt['fits'] == (mine <= 20), pt)

    # ---------------- guided parts, EN and ZH
    def go(i):
        pg.evaluate(f"()=>document.querySelector('#tf-rail [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(200)

    def cards():
        return pg.evaluate("()=>Object.fromEntries([...document.querySelectorAll('#tf-metrics .metric')].map(m=>[m.dataset.k,{v:m.querySelector('.v').textContent.replace(/\\u00a0/g,' '),d:m.querySelector('.d').textContent,tip:m.dataset.tipx}]))")

    def mathlines():
        pg.evaluate("()=>{document.getElementById('tf-math').open=true}"); pg.wait_for_timeout(100)
        return pg.evaluate("()=>[...document.querySelectorAll('#tf-mathbody p')].map(p=>p.innerText.replace(/\\s*\\n\\s*/g,' ').replace(/\\u2011/g,'-').replace(/\\u00a0/g,' '))")   # 2026-10-09: a long chain breaks at “=” onto aligned lines (.tf-chain), read as one line

    def check_math(lang, mlines, tag):
        for l in mlines:
            if '=' not in l or 'FLOP' in l or 'attention block + MLP block' in l or 'attention 块 + MLP 块' in l:   # the hidden split names no numbers to redo
                continue
            if 'GiB' in l:
                mm = re.search(r'(' + NUM + r') × \S+ \S* ?= (' + NUM + r') (?:bytes|字节) = (' + NUM + r') GB = (' + NUM + r') GiB', l)
                check(f'{lang} {tag} math: weights line reproduces, bytes → GB → GiB (R6.12): {l[:50]}', bool(mm) and int(mm.group(1).replace(',', '')) * 2 == int(mm.group(2).replace(',', ''))
                      and g1(int(mm.group(2).replace(',', '')) / 1e9) == mm.group(3) and g1(int(mm.group(2).replace(',', '')) / GiB) == mm.group(4), l)
                continue
            ok, info = reproduces(l)
            check(f'{lang} {tag} math line reproduces: {l[:70]}', ok, info)

    for lang in ['en', 'zh']:
        Z = (lambda en, zh: zh) if lang == 'zh' else (lambda en, zh: en)
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        pg.click('#tf-app .modes button[data-mode="guided"]'); pg.wait_for_timeout(200)
        go(0)
        t1 = page_text(pg, '#tf-rail .step.current')
        t1x = exact_text(pg, '#tf-rail .step.current')
        # R6.20: the mission keeps the count (exact on hover) and says the memory in words; the weights card gives 15.0 GiB, and its
        # tooltip the bytes → GB → GiB with both units defined (R6.12)
        memtip = pg.evaluate("()=>document.querySelector('#tf-metrics .metric[data-k=\"mem\"]').dataset.tipx")
        check(f'{lang} part 1: the count as 8.03B (no exact-value hover), no memory arithmetic in the card; the weights card’s tooltip: 16,060,522,496 bytes = 16.1 GB = 15.0 GiB, GiB and GB defined (R6.18, R6.12, R6.20)',
              '8.03B' in t1 and no_big_hover(pg, '#tf-rail .step.current') and '8,030,261,248' not in t1x and not LONG.search(t1) and 'GiB' not in t1 and '16,060,522,496' in memtip and '16.1 GB = 15.0 GiB' in memtip and '2^30' not in memtip and '10^9' not in memtip and memtip.count('<math') >= 3
              and 8030261248 * 2 == 16060522496 and g1(16060522496 / 1e9) == '16.1' and g1(16060522496 / GiB) == '15.0', (t1[:300], memtip))
        check(f'{lang} part 1: the seven numbers', all(f'"{k}": {v}' in t1 for k, v in [('vocab_size', 128256), ('hidden_size', 4096), ('num_hidden_layers', 32), ('num_attention_heads', 32), ('num_key_value_heads', 8), ('intermediate_size', 14336), ('tie_word_embeddings', 'false')]), t1[:400])
        tips = pg.evaluate("()=>Object.fromEntries([...document.querySelectorAll('#tf-rail .tf-cfg .def')].map(d=>[d.textContent,d.dataset.tipx]))")
        check(f'{lang} part 1: num_attention_heads gives head_dim 4,096 ÷ 32 = 128', '4,096 ÷ 32' in tips.get('num_attention_heads', '') and '128' in tips.get('num_attention_heads', ''), tips.get('num_attention_heads'))
        ht = tips.get('num_attention_heads', '')
        check(f'{lang} part 1: heads work with numbers that the attention block’s matrices Wq, Wk and Wv compute from all 4,096 numbers, not a slice of the token (as lab ② says; r13 M6)',
              Z('computed by the attention block’s matrices Wq, Wk and Wv from all 4,096 numbers of the token', 'attention 块的矩阵 Wq、Wk、Wv 用 token 的全部 4,096 维算出新的向量') in ht
              and Z('pieces of 128 numbers (head_dim = 4,096 ÷ 32)', '每个 head 分到其中 128 维（head_dim = 4,096 ÷ 32）') in ht and not any(x in ht for x in ['splits', 'split', '切成', '每份', '个数']), ht)
        check(f'{lang} part 1: num_key_value_heads gives 8 × 128 = 1,024 for Wk and Wv, a quarter of 4,096', '8 × 128 = 1,024' in tips.get('num_key_value_heads', '') and 1024 * 4 == 4096, tips.get('num_key_value_heads'))
        check(f'{lang} part 1: num_key_value_heads ends with what a KV head is, 32 ÷ 8 = 4 heads per KV head (R14 term table)',
              tips.get('num_key_value_heads', '').replace('\u2011', '-').endswith(Z('A KV head is one key and one value each token stores in every layer, which several heads can share; Llama-3.1-8B lets every 4 heads share one, as the lab GQA and MLA explains.',
                                                            'KV head 是每个 token 在每一层存的一组 key 和 value，可以由几个 head 共用；Llama-3.1-8B 让每 4 个 head 共用一组，“GQA 和 MLA”这个 Lab 里讲。')) and 32 // 8 == 4, tips.get('num_key_value_heads'))
        check(f'{lang} part 1: intermediate_size starts with the kit’s words (common r15 (a))', tips.get('intermediate_size', '').startswith(pg.evaluate("()=>window.ASP.terms.inter()")) and '14,336' in tips.get('intermediate_size', '') and 14336 / 4096 == 3.5, tips.get('intermediate_size'))
        met = pg.evaluate("()=>[...document.querySelectorAll('#tf-metrics .metric')].map(m=>(m.querySelector('.v').textContent+' | '+m.querySelector('.sub').textContent).replace(/\\u00a0/g,' '))")
        hidden = not pg.evaluate("()=>window.__tf.state().revealed")      # the second language runs after part 4 has shown the answer
        mtip = pg.evaluate("()=>document.querySelector('#tf-metrics .metric[data-k=\"total\"]').dataset.tipx")
        check(f'{lang} part 1 metrics: 8.03B on the card, its tooltip says what is counted with no exact count; memory 15.0 GiB and 16.1 GB; shares {"hidden" if hidden else "shown after part 4"}', met[0].startswith('8.03B') and not LONG.search(met[0]) and '8,030,261,248' not in mtip and Z('Counted here: the embedding table', '这里数进去的有 embedding 表') in mtip
              and met[1].startswith('15.0 GiB') and '16.1 GB' in met[1] and all(m.startswith('?') == hidden for m in met[2:]), (met, mtip))
        cap = page_text(pg, '#tf-cap')
        check(f'{lang} stage caption: one scale, a full-length bar is 1.05B (no exact-value hover)', '1.05B' in cap and no_big_hover(pg, '#tf-cap') and not LONG.search(cap), cap)
        stk = page_text(pg, '#tf-stack')
        check(f'{lang} stage labels: 525M, 218M, 6.98B, no count over 6 digits written out (R6.18)', all(x in stk for x in ['525M', '218M', '6.98B']) and not LONG.search(stk)
              and all(big_str(n) in stk for n in [525336576, 218112000, 6979584000]) and no_big_hover(pg, '#tf-stack'), stk[:300])
        check(f'{lang} no big number in the lab has an exact-value hover (the owner, 2026-10-07)', no_big_hover(pg, '#tf-app'), bigs(pg, '#tf-app')[:3])
        widths = pg.evaluate("()=>{const g=document.querySelector('#tf-stack .tf-sec .tf-bar'); const W=g.getBoundingClientRect().width; return [...document.querySelectorAll('#tf-stack .tf-sec .tf-bar')].map(b=>{const i=b.querySelector('.tf-seg,.tf-lw'); return i?i.getBoundingClientRect().width/W:0})}")
        check(f'{lang} stage: the embedding table is drawn at 525,336,576 ÷ 1,050,673,152 = 50% of full length, a layer at 218,112,000 ÷ 1,050,673,152 = 20.8%',
              abs(widths[0] - 0.5) < 0.01 and abs(widths[1] - 218112000 / SCALE) < 0.01, widths[:3])
        if hidden:
            ml = mathlines()
            check(f'{lang} part 1 math: no head_dim line while the split is hidden (it only names heads)', not any('head_dim' in l for l in ml[:-1]), ml)
            check_math(lang, ml, 'part 1')
        go(1); pg.click('#tf-rail [data-pred="q2"]'); pg.wait_for_timeout(150)
        go(3)
        t4v = page_text(pg, '#tf-rail [data-obs]')
        t4 = exact_text(pg, '#tf-rail [data-obs]')
        # R6.20: one key number in the result (the attention share the prediction asked for); the counts are on the metric cards
        # (5.64B, 1.34B) and the stage's layer label (41.9M, 176M, exact on hover); the reason says the shapes in words
        c4 = cards(); each = page_text(pg, '#tf-stack .tf-each'); eachx = exact_text(pg, '#tf-stack .tf-each')
        check(f'{lang} part 4: the result gives 16.7% and says the MLP blocks hold most; the counts 5.64B and 1.34B on the cards, 41.9M and 176M on the stage (exact on hover, R6.18)',
              nth(t4v, 1).find('16.7%') >= 0 and Z('the MLP blocks hold most of them', '大部分参数在 MLP 块里') in nth(t4v, 1) and not LONG.search(t4v)
              and c4['mlp']['v'] == '70.2%' and '5.64B' in pg.evaluate("()=>document.querySelector('#tf-metrics .metric[data-k=\"mlp\"] .sub').textContent")
              and '1.34B' in pg.evaluate("()=>document.querySelector('#tf-metrics .metric[data-k=\"attn\"] .sub').textContent")
              and all(x in each for x in ['41.9M', '176M']) and big_str(41943040) == '41.9M' and big_str(176160768) == '176M' and no_big_hover(pg, '#tf-stack .tf-each')
              and big_str(5637144576) == '5.64B' and big_str(1342177280) == '1.34B', (t4v, each))
        # the reason: the intermediate size is 3.5 times the hidden size (14,336 ÷ 4,096), so about four fifths of a layer is MLP (176,160,768 ÷ 218,112,000 = 0.81)
        FR = [(1 / 10, 'a tenth', '十分之一'), (1 / 5, 'a fifth', '五分之一'), (1 / 4, 'a quarter', '四分之一'), (1 / 3, 'a third', '三分之一'), (2 / 5, 'two fifths', '五分之二'), (1 / 2, 'half', '一半'),
              (3 / 5, 'three fifths', '五分之三'), (2 / 3, 'two thirds', '三分之二'), (3 / 4, 'three quarters', '四分之三'), (4 / 5, 'four fifths', '五分之四'), (9 / 10, 'nine tenths', '十分之九')]
        near = lambda x: min(FR, key=lambda f: abs(f[0] - x))
        fr4 = near(176160768 / 218112000)
        inter = pg.evaluate("()=>{const d=[...document.querySelectorAll('#tf-rail [data-obs] .def')].find(e=>/intermediate size|中间维度/.test(e.textContent)); return d?d.dataset.tipx:''}")
        check(f'{lang} part 4 reason: 3.5 times the hidden size, “{fr4[1]}” of every layer in the MLP block, the Wk and Wv narrow; the intermediate size’s hover gives 14,336',
              fr4[1] == 'four fifths' and Z('3.5 times the hidden size', '是隐藏维度的 3.5 倍') in t4 and Z(f'So about {fr4[1]} of every layer is MLP.', f'所以每层约{fr4[2]}的参数在 MLP 块里。') in t4
              and 'Wk' in t4 and '14,336' in inter and 14336 / 4096 == 3.5, (t4, inter))
        check(f'{lang} part 4 verdict quotes both options', Z('You predicted “1/4 to 1/2”; the right answer is “Under 1/4”.', '你预测的是“1/4 到 1/2”，正确答案是“不到 1/4”。') in t4, t4)
        met = pg.evaluate("()=>[...document.querySelectorAll('#tf-metrics .metric .v')].map(v=>v.textContent)")
        check(f'{lang} part 4 metrics: 8B shares shown', met[2:] == ['70.2%', '16.7%', '13.08%'], met)
        # a change under Settings does not change the answer to a question about Llama-3.1-8B (review r11, #4)
        pg.evaluate("()=>{const i=document.getElementById('tf-ctl-f-in'); i.value=0; i.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(250)
        t4b = exact_text(pg, '#tf-rail [data-obs]')
        check(f'{lang} part 4 after a change under Settings: still about Llama-3.1-8B, same verdict, the reason stays true',
              Z('You changed something under Settings', '你在“自由调参”里改了设置') in t4b and '16.7%' in t4b and Z('the right answer is “Under 1/4”', '正确答案是“不到 1/4”') in t4b
              and Z('3.5 times the hidden size', '是隐藏维度的 3.5 倍') in t4b and Z('about four fifths', '约五分之四') in t4b, t4b[:400])
        pg.click('#tf-controls [data-act="reset"]'); pg.wait_for_timeout(200)
        check_math(lang, mathlines(), 'part 4')
        go(4)
        k128 = pg.evaluate("()=>[...document.querySelectorAll('#tf-rail .step.current .def')].filter(d=>/128K/.test(d.textContent)).map(d=>[d.textContent,d.dataset.tipx])")
        check(f'{lang} part 5: “about 128K” says on hover that its K is 1,000, 128,000 + 256 = 128,256, and that a context length’s K is 1,024 (R14 term table)',
              len(k128) == 1 and k128[0][0] == Z('about 128K', '约 128K') and k128[0][1] == Z('Here K is 1,000: 128,000 text tokens plus 256 special tokens. For context length K is 1,024, so 128K tokens are 131,072.',
                                                                                              '这里的 K 是 1,000：128,000 个文本 token 加 256 个特殊 token。上下文长度的 K 是 1,024，128K 个 token 就是 131,072 个。')
              and 128000 + 256 == 128256 and 128 * 1024 == 131072, k128)
        pg.click('#tf-ic-model button[data-v="mistral-7b"]'); pg.wait_for_timeout(250)
        t5v = page_text(pg, '#tf-rail [data-obs]')
        check(f'{lang} part 5: 7.24B and 789M shown (R6.18)', '7.24B' in t5v and '789M' in t5v and not LONG.search(t5v), t5v)
        t5 = exact_text(pg, '#tf-rail [data-obs]')
        # R6.20: the reason in words (Mistral's vocabulary is a quarter of Llama's: 32,000 ÷ 128,256 = 0.2495); the arithmetic is a line in Show the math
        check(f'{lang} part 5: 7.24B and 789M with no exact-value hover; the reason says the vocabulary is “a quarter” the size, no arithmetic in the sentence; one new thing only (no tying here)',
              all(x in t5 for x in [big_str(7241732096), big_str(788529152)]) and no_big_hover(pg, '#tf-rail [data-obs]') and near(32000 / 128256)[1] == 'a quarter' and abs(32000 / 128256 - 0.25) < 0.005
              and Z('the vocabulary, a quarter the size in Mistral-7B-v0.1', 'Mistral-7B-v0.1 的词表只有四分之一大') in t5 and '(128,256' not in t5 and '262,668,288' not in t5, t5)
        ml5 = [l for l in mathlines() if '788,529,152' in l and '(128,256' in l]
        ok, info = reproduces(ml5[0] if ml5 else '')
        check(f'{lang} part 5: Show the math has the vocabulary line, Llama-3.1-8B − Mistral-7B-v0.1 = (128,256 − 32,000) × 4,096 × 2 = 788,529,152, and it reproduces', ok and (128256 - 32000) * 4096 * 2 == 788529152, (ml5, info))
        vt = pg.evaluate("()=>[...document.querySelectorAll('#tf-mathbody .def')].map(d=>[d.textContent,d.dataset.tipx])")
        check(f'{lang} part 5: the two vocabulary sizes in that line say on hover whose they are (R1.4)', any(a == '128,256' and 'vocab_size' in b and 'Llama' in b for a, b in vt) and any(a == '32,000' and 'vocab_size' in b and 'Mistral' in b for a, b in vt), vt)
        pg.evaluate("()=>{document.getElementById('tf-math').open=false}")
        c5 = cards()
        want = delta_txt(TOTAL['llama31-8b'], TOTAL['mistral-7b'], lang, init=True)
        check(f'{lang} part 5: “{want}” on the parameter and weights cards, against the initial setup', c5['total']['d'] == want == c5['mem']['d'], {k: c5[k]['d'] for k in c5})
        check(f'{lang} part 5: the parameter card’s tooltip gives the exact counts it compares', '7,241,732,096' in c5['total']['tip'] and '8,030,261,248' in c5['total']['tip'], c5['total']['tip'])
        for mk, (a, b_) in {'mlp': ('70.2%', '77.8%'), 'attn': ('16.7%', '18.5%'), 'voc': ('13.08%', '3.62%')}.items():
            check(f'{lang} part 5: {mk} share {a} → {b_}, “{pts_txt(a, b_, lang, True)}”', c5[mk]['v'] == b_ and c5[mk]['d'] == pts_txt(a, b_, lang, True), c5[mk])
        widths = pg.evaluate("()=>{const W=document.querySelector('#tf-stack .tf-sec .tf-bar').getBoundingClientRect().width; return [...document.querySelectorAll('#tf-stack .tf-sec .tf-bar')].map(b=>{const i=b.querySelector('.tf-seg,.tf-lw'); return i?i.getBoundingClientRect().width/W:0})}")
        check(f'{lang} part 5 stage: Mistral’s table at 131,072,000 ÷ 1,050,673,152 = 12.5%, its layers as long as Llama’s (20.8%)',
              abs(widths[0] - 131072000 / SCALE) < 0.01 and abs(widths[1] - 218112000 / SCALE) < 0.01, widths[:3])
        go(5)
        for i, (tot, name) in enumerate([(8030261248, 'initial'), (30019166208, 'wide'), (15009845248, 'deep')]):
            pg.click(f'#tf-rail [data-lever="{i}"]'); pg.wait_for_timeout(250)
            c6 = cards()      # first: the ▲▼ badge fades after 2.6 s
            obv = page_text(pg, '#tf-rail [data-obs]')
            ob = exact_text(pg, '#tf-rail [data-obs]')
            check(f'{lang} part 6 {name}: total {big_str(tot)} shown, no exact-value hover', big_str(tot) in obv and no_big_hover(pg, '#tf-rail [data-obs]') and not LONG.search(obv), obv[:200])
            # R6.20: the result says the size and the reason in words; the count, line by line, is in Show the math, linked from the result
            nl = pg.evaluate("()=>document.querySelectorAll('#tf-rail [data-obs] .tf-lines, #tf-rail [data-obs] p.mono').length")
            check(f'{lang} part 6 {name}: no formula lines in the result, and a link to Show the math', nl == 0 and '=' not in obv and pg.evaluate("()=>!!document.querySelector('#tf-rail [data-obs] [data-jump=\"math\"]')"), obv)
            want = delta_txt(TOTAL['llama31-8b'], tot, lang, init=True)
            check(f'{lang} part 6 {name}: “{want or "no badge"}” on the parameter and weights cards, against the initial setup', c6['total']['d'] == want == c6['mem']['d'], {k: c6[k]['d'] for k in c6})
            ml6 = mathlines()
            check_math(lang, ml6, f'part 6 {name}')
            if name == 'initial':   # R14 N14: the LM head is hidden size × vocabulary in Show the math
                mm_ = [l for l in ml6 if l.startswith(Z('model =', '模型 ='))]
                check(f'{lang} Show the math: the LM head term is 4,096 × 128,256 (N14)', bool(mm_) and Z('4,096 × 128,256 (LM head)', '4,096 × 128,256（LM head）') in mm_[0], mm_)
            if name == 'wide':
                hdl = [l for l in ml6 if l.startswith('head_dim')]
                kvn = [re.search(r'[(（](' + NUM + r') = (\d+) (?:KV heads|个 KV head) × (\d+)[)）]', l) for l in ml6]
                kvn = [m for m in kvn if m]
                check(f'{lang} part 6 wide, Show the math: head_dim = 8,192 ÷ 32 heads = 256, and 2,048 = 8 KV heads × 256', bool(hdl) and '8,192' in hdl[0] and hdl[0].endswith('256')
                      and bool(kvn) and kvn[0].group(1) == '2,048' and int(kvn[0].group(2)) * int(kvn[0].group(3)) == 2048 and 8192 // 32 == 256, (hdl, [m.group(0) for m in kvn]))
                tw = round(872431616 / 218112000)
                check(f'{lang} part 6 wide: 3.7 times the initial setup; each layer {tw} times as big (872,431,616 ÷ 218,112,000), in words',
                      '3.7' in obv and f'{30019166208 / 8030261248:.1f}' == '3.7' and abs(872431616 / 218112000 - tw) < 0.05 and tw == 4
                      and Z('each layer becomes four times as big', '每层变成原来的四倍') in obv, obv)
                stk = page_text(pg, '#tf-stack'); stkx = exact_text(pg, '#tf-stack')
                check(f'{lang} part 6 wide stage: a layer 872M (no exact-value hover)', '872M' in stk and big_str(872431616) == '872M' and no_big_hover(pg, '#tf-stack'), stk[:300])
                widths = pg.evaluate("()=>{const W=document.querySelector('#tf-stack .tf-sec .tf-bar').getBoundingClientRect().width; return [...document.querySelectorAll('#tf-stack .tf-sec .tf-bar')].map(b=>{const i=b.querySelector('.tf-seg,.tf-lw'); return i?i.getBoundingClientRect().width/W:0})}")
                check(f'{lang} part 6 wide stage: the table doubles to full length, a layer grows to 872,431,616 ÷ 1,050,673,152 = 83%',
                      abs(widths[0] - 1) < 0.01 and abs(widths[1] - 872431616 / SCALE) < 0.01, widths[:3])
            if name == 'deep':
                stk = page_text(pg, '#tf-stack'); stkx = exact_text(pg, '#tf-stack')
                check(f'{lang} part 6 deep: 1.9 times; the layers double but the tables do not, so “just short of twice”; the stage shows the layers at 14.0B (no exact-value hover)',
                      '1.9' in obv and f'{15009845248 / 8030261248:.1f}' == '1.9' and Z('falls just short of twice', '总数不到两倍') in obv and '14.0B' in stk and big_str(13959168000) == '14.0B' and no_big_hover(pg, '#tf-stack'), (obv, stk[:300]))
        pg.click('#tf-rail [data-lever="1"]'); pg.wait_for_timeout(250)
        go(6)
        t7v = page_text(pg, '#tf-rail .step.current')
        check(f'{lang} part 7: weights in memory = parameters × bytes, 2 bytes in BF16, 15.0 GiB (16.1 GB) with GiB defined; 8.03B to 8.30B, 789M; nothing over 6 digits written out (R6.12, R6.18)',
              all(x in t7v for x in [Z(': 15.0 GiB (16.1 GB) for Llama-3.1-8B in BF16.', '。Llama-3.1-8B 在 BF16 下是 15.0 GiB（16.1 GB）。'),
                                     Z('15.0 GiB (16.1 GB)', '15.0 GiB（16.1 GB）'), '8.30B', '789M']) and not LONG.search(t7v)
              and pg.evaluate("()=>[...document.querySelectorAll('#tf-rail .step.current .def')].some(d=>d.textContent==='GiB'&&/2(\\^30|³⁰)/.test(d.dataset.tipx))"), t7v[-800:])
        check(f'{lang} part 7 ends with “See the scenarios”, not the challenge (R3.12)', pg.evaluate("()=>{const b=[...document.querySelectorAll('#tf-rail .step.current .stepnav [data-mode-go]')]; return b.length===1&&b[0].dataset.modeGo==='sandbox'}")
              and Z('See the scenarios', '去看应用场景') in t7v, t7v[-200:])
        t7 = exact_text(pg, '#tf-rail .step.current')
        # R6.20: the summary says the principle in words; the five models' shares are checked here against the independent count
        min_mlp = min(mlps.values())
        ratios = [c['f'] / c['d'] for c in CFG.values()]
        check(f'{lang} part 7: the MLP blocks hold most in all five models (at least {share_str(min_mlp)}), their intermediate size “several times” the hidden size ({min(ratios)} to {max(ratios)})',
              min_mlp > 0.5 and min(ratios) >= 3
              and Z('The MLP blocks hold most of the parameters in all five models', '大部分参数都在 MLP 块里') in t7
              and Z('several times the hidden size', '隐藏维度的好几倍') in t7 and '8.03B' in t7 and '15.0 GiB' in t7, t7[:700])
        q7 = R['llama31-8b']['parts']['mlp'] / R['llama31-8b']['parts']['attn']
        check(f'{lang} part 7 misconception: the attention blocks 16.7%, the MLP blocks “about four times as much” (5,637,144,576 ÷ 1,342,177,280 = {q7:.2f})',
              share_str(R['llama31-8b']['parts']['attn'] / R['llama31-8b']['total']) == '16.7%' and '16.7%' in t7 and round(q7) == 4 and abs(q7 - 4) >= 0.05
              and Z('the MLP blocks about four times as much', 'MLP 块约是它的四倍') in t7, t7[:700])
        check(f'{lang} part 7: halving the heads with the KV heads unchanged doubles head_dim (4,096 ÷ 32 = 128, 4,096 ÷ 16 = 256), and the model, not Wk and Wv, grows to 8.30B; 789M shown, no exact-value hover (no “heads of 128” slices; r13 M7)',
              Z('With the same KV heads, halving the heads doubles head_dim', 'KV head 数不变时，head 减半，head_dim 就翻倍') in t7
              and Z(f'the model grows from {big_str(8030261248)} to {big_str(8298696704)} parameters', f'整个模型从 {big_str(8030261248)} 个参数变成 {big_str(8298696704)} 个') in t7
              and 4096 // 32 == 128 and 4096 // 16 == 256 and big_str(788529152) in t7 and no_big_hover(pg, '#tf-rail .step.current') and not any(x in t7 for x in ['heads of 128', '个数的 head']), t7[-700:])
        check(f'{lang} part 7: the 7B misconception says what it compares with (r13 M7)', Z('“A 7B model has smaller layers than an 8B one.”', '“7B 模型的层比 8B 模型的小。”') in t7, t7[-400:])
        # Show the math: every line reproduces (initial setup)
        mlines = mathlines()
        check_math(lang, mlines, 'part 7')
        fl = [l for l in mlines if 'FLOP' in l and '=' in l]
        check(f'{lang} math: multiplied per token 8,030,261,248 − 525,336,576 = 7,504,924,672, ≈ 15.0B FLOPs', fl and '8,030,261,248 − 525,336,576' in fl[0] and '7,504,924,672' in fl[0] and '15.0B' in fl[0] and big_str(2 * 7504924672) == '15.0B', fl)
        check(f'{lang} math: head_dim = 4,096 ÷ 32 heads = 128', any(l.startswith('head_dim = 4,096') and Z('32 heads', '32 个 head') in l and l.endswith('128') for l in mlines), mlines[:2])
        kvn = [re.search(r'\((' + NUM + r') = (\d+) KV heads? × (\d+)\)', l) or re.search(r'（(' + NUM + r') = (\d+) 个 KV head × (\d+)）', l) for l in mlines]
        kvn = [m for m in kvn if m]
        check(f'{lang} math: the KV width note reproduces (1,024 = 8 × 128)', bool(kvn) and all(int(m.group(1).replace(',', '')) == int(m.group(2)) * int(m.group(3)) for m in kvn), [m.group(0) for m in kvn])
        # ---------------- scenarios
        pg.click('#tf-app .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(250)
        prev = None
        for i, k in enumerate(['llama32-1b', 'llama31-8b', 'mistral-7b', 'llama31-70b', 'llama31-405b']):
            pg.click(f'#tf-rail [data-preset="{i}"]'); pg.wait_for_timeout(200)
            cs = cards()
            if prev is not None:   # R6.6: ▲▼ say what they compare with, and one change shows one factor on both cards
                want = delta_txt(R[prev[0]]['total'], R[k]['total'], lang)
                check(f'{lang} {prev[0]} → {k}: parameter change “{want}” from the exact counts, which the card’s tooltip gives', cs['total']['d'] == want and f"{R[k]['total']:,}" in cs['total']['tip'] and f"{R[prev[0]]['total']:,}" in cs['total']['tip'], cs['total'])
                check(f'{lang} {prev[0]} → {k}: the weights card shows the same change as the parameter count', cs['mem']['d'] == cs['total']['d'], (cs['mem'], cs['total']))
                for mk in ['mlp', 'attn', 'voc']:
                    va, vb = prev[1][mk]['v'], cs[mk]['v']
                    check(f'{lang} {prev[0]} → {k}: {mk} share change {va} → {vb} in points from the shown values, as shown (“{pts_txt(va, vb, lang)}”)', cs[mk]['d'] == pts_txt(va, vb, lang), cs[mk])
            prev = (k, cs)
            insv = page_text(pg, '#tf-insight')
            ins = exact_text(pg, '#tf-insight')
            mine = R[k]; t = mine['total']; gi, gb = mem(t)
            # R6.20: the result line keeps the count and the memory; the shares are on the metric cards (no line repeating them)
            check(f'{lang} scenario {k}: total ({big_str(t)} shown, no exact-value hover) and memory in the result; the shares on the cards', big_str(t) in insv and not LONG.search(insv) and no_big_hover(pg, '#tf-insight') and gi in ins and gb in ins
                  and cs['mlp']['v'] == share_str(mine['parts']['mlp'] / t) and cs['attn']['v'] == share_str(mine['parts']['attn'] / t) and share_str(mine['parts']['mlp'] / t) not in ins, (insv[:300], cs['mlp']['v'], cs['attn']['v']))
            cap = exact_text(pg, '#tf-cap')
            longest = max(mine['emb'], mine['layer'])
            check(f'{lang} scenario {k}: the caption states the scale ({big_str(max(longest, SCALE))} full length)', big_str(max(longest, SCALE)) in cap and ((big_str(SCALE) in cap) if longest <= SCALE else True), cap)
            voc = share_str(mine['parts']['emb'] / t)
            voc2 = f"{2 * float(voc.rstrip('%')):.{len(voc.rstrip('%').split('.')[1])}f}%"
            if k == 'llama31-70b':
                check(f'{lang} scenario 70B: the tables only {voc2} (more layers, 80 vs 32, each much bigger, {mine["layer"] / r8["layer"]:.1f} times); 16.4 GiB on each of 8 GPUs; the lab that counts users',
                      voc2 in ins and Z('there are more layers, and each is much bigger', '但层更多，每层也大得多') in ins and CFG[k]['L'] > 32 and mine['layer'] / r8['layer'] > 2
                      and g1(t * 2 / GiB / 8) + ' GiB' in ins and g1(t * 2 / GiB / 8) == '16.4' and Z('KV cache memory', 'KV cache 显存') in ins, ins)
            if k == 'llama31-8b':
                check(f'{lang} scenario RTX 4090: 15.0 GiB in the result, “more than half” of the RTX 4090’s 24 GiB in words (15.0 ÷ 24 = {15.0 / 24:.2f})', '15.0 GiB' in ins and 0.5 < 15.0 / 24 < 2 / 3
                      and Z('its weights fill more than half of the RTX 4090’s memory', '它的权重占了 RTX 4090 一半以上的显存') in ins, ins)
                kd = pg.evaluate("()=>[...document.querySelectorAll('#tf-insight .def')].map(d=>d.dataset.tipx)")
                check(f'{lang} scenario RTX 4090: the KV cache is defined with its size, 128 KiB per token, 8,192 tokens per GiB (no 4 GiB here)', any('128 KiB' in x and Z('each GiB holds 8,192 tokens', '每 GiB 能放 8,192 个 token') in x and '4 GiB' not in x for x in kd), kd)
            if k == 'llama32-1b':
                stx = exact_text(pg, '#tf-stack')
                check(f'{lang} scenario 1B: the table alone {voc} (tied); fewer and much smaller layers (16 vs 32, 60,821,504 vs 218,112,000 on the stage); 2.3 GiB in BF16, FP4 “a quarter” of that (0.6 GiB)',
                      voc in ins and '2.3 GiB' in ins and Z('the layers are fewer and much smaller', '层却更少，每层也小得多') in ins and Z('In FP4 the weights take a quarter of that', 'FP4 下权重只有 BF16 的四分之一') in ins
                      and big_str(60821504) in stx and CFG[k]['L'] < 32 and 60821504 / 218112000 < 0.5 and 0.5 / 2 == 0.25 and g1(t * 0.5 / GiB) == '0.6' and g1(t * 2 / GiB * 0.25) == '0.6', ins)
            if k == 'mistral-7b':
                check(f'{lang} scenario 7B: 13.5 GiB, a vocabulary “a quarter the size” (32,000 ÷ 128,256), 1.5 GiB less than the 8B', '13.5 GiB' in ins and '1.5 GiB' in ins and abs(32000 / 128256 - 0.25) < 0.005
                      and Z('the vocabulary is a quarter the size', '词表只有它的四分之一') in ins, ins)
            if k == 'llama31-405b':
                nd = pg.evaluate("()=>[...document.querySelectorAll('#tf-insight .def')].map(d=>[d.textContent,d.dataset.tipx])")
                check(f'{lang} scenario 405B: the tables only {voc2}; 756.0 GiB, more than an 8 × H100 node, whose hover gives 640 GiB (8 × 80 GB)',
                      voc2 in ins and '756.0 GiB' in ins and 756.0 > 8 * 80 and any('8' in a and 'H100' in a.replace(' ', ' ') and '640 GiB' in b and '80 GB' in b for a, b in nd), (ins, nd))
        # ---------------- challenge
        pg.click('#tf-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(250)
        for L in [40, 48, 44]:
            pg.evaluate(f"()=>{{const i=document.getElementById('tf-ic-L-in'); i.value={L}; i.dispatchEvent(new Event('input',{{bubbles:true}}))}}"); pg.wait_for_timeout(150)
            now = page_text(pg, '#tf-ch-now')
            check(f'{lang} challenge {L} layers: progress shows {g1(fast({**CH, "L": L}) * 2 / GiB)}', now == g1(fast({**CH, 'L': L}) * 2 / GiB), now)
            st = page_text(pg, '#tf-status')
            spare = round(20 - float(now), 1)
            check(f'{lang} challenge {L}: status margin from the shown value', g1(abs(spare)) in st, (st, spare))
        mlb = [l for l in mathlines() if '÷' in l and ('budget' in l or '预算' in l)]
        check(f'{lang} challenge before Submit: no budget line in Show the math (R3.8)', not mlb, mlb)
        pg.click('#tf-rail [data-act="submit"]'); pg.wait_for_timeout(250)
        res = page_text(pg, '#tf-ch-res')
        # R6.20: the verdict and the reason in words; the division that gives 44 is in Show the math once the chart shows
        check(f'{lang} challenge 44: goal met at 19.8 GiB, one more layer does not fit (45 layers: {g1(fast({**CH, "L": 45}) * 2 / GiB)} GiB), the reason in words, no arithmetic in the sentences',
              all(x in res for x in ['19.8 GiB', Z('one more would go over', '再加一层就超了'), Z('each layer adds the same amount', '每加一层多占的也都一样'), Z('the same memory at any depth', '不管多少层都占同样多的显存')])
              and not any(x in res for x in ['1.957', '0.406', '44.4', '20.2']) and fast({**CH, 'L': 45}) * 2 / GiB > 20, res[:600])
        mlb = [l for l in mathlines() if '÷' in l and ('budget' in l or '预算' in l)]
        ok, info = reproduces(mlb[0] if mlb else '')
        check(f'{lang} challenge, Show the math: (20 − 1.957) ÷ 0.406 = 44.4, so 44 layers fit, and it reproduces from its shown terms', ok and bool(mlb) and '(20 − 1.957) ÷ 0.406 = 44.4' in mlb[0] and Z('so 44 layers fit', '所以放得下 44 层') in mlb[0], (mlb, info))
        bt = pg.evaluate("()=>Object.fromEntries([...document.querySelectorAll('#tf-mathbody .def')].map(d=>[d.textContent,d.dataset.tipx]))")
        check(f'{lang} challenge, Show the math: the hovers of 0.406 (one layer, 218,112,000 × 2 bytes = 436,224,000 bytes = 0.436 GB) and 1.957 (tables and final RMSNorm, 1,050,677,248 × 2 bytes = 2,101,354,496 bytes = 2.101 GB) and of the budget 20',
              '218,112,000' in bt.get('0.406', '') and '436,224,000' in bt.get('0.406', '') and '0.436 GB' in bt.get('0.406', '') and '0.406 GiB' in bt.get('0.406', '')
              and '1,050,677,248' in bt.get('1.957', '') and '2,101,354,496' in bt.get('1.957', '') and '2.101 GB' in bt.get('1.957', '') and '1.957 GiB' in bt.get('1.957', '')
              and '20 GiB' in bt.get('20', '') and 218112000 * 2 == 436224000 and 2 * 525336576 + 4096 == 1050677248 and f'{1050677248 * 2 / GiB:.3f}' == '1.957' and f'{436224000 / GiB:.3f}' == '0.406', bt)
        rows = pg.evaluate("()=>[...document.querySelectorAll('#tf-ch-res .cmp-row')].map(r=>r.innerText.replace(/\\s+/g,' '))")
        want = [(L, g1(fast({**CH, 'L': L}) * 2 / GiB)) for L in [32, 36, 40, 44]]
        # the owner's review of batch 8 (2026-10-07): the chart keeps the layer counts that fit (and the pick, 44); 48 and 52 go over 20 GiB
        # and are only counted in the note under it
        check(f'{lang} challenge chart: rows 32–44 (the counts that fit) with their GiB; 48 and 52 ({g1(fast({**CH, "L": 48}) * 2 / GiB)} and {g1(fast({**CH, "L": 52}) * 2 / GiB)} GiB) are left out and counted in the note',
              len(rows) == 4 and all(str(L) in r and f'{v} GiB' in r for (L, v), r in zip(want, rows)) and all(fast({**CH, 'L': L}) * 2 / GiB > 20 for L in (48, 52))
              and page_text(pg, '#tf-ch-res .cmp-note') == Z('The other 2 layer counts do not fit and are not shown.', '其余 2 种层数放不下，图里没有列出。'), rows)
        check(f'{lang} challenge note: the KV cache grows with every layer too, in words (128 KiB per token at 32 layers, 176 KiB at 44, recomputed here)',
              Z('which grows with every layer too', '而它也随层数增加') in res and 'KiB' not in res and 2 * 32 * 1024 * 2 // 1024 == 128 and 2 * 44 * 1024 * 2 // 1024 == 176, res[-400:])
        kd = pg.evaluate("()=>[...document.querySelectorAll('#tf-rail .railcard .def')].map(d=>d.dataset.tipx)")
        check(f'{lang} challenge: KV cache and runtime overhead are defined where the card names the memory kept for them; 8,192 tokens per GiB', len(kd) >= 2 and any(Z('each GiB holds 8,192 tokens', '每 GiB 能放 8,192 个 token') in x for x in kd), kd)
        # ---------------- the capability layer (R6.20): "Explain it" in words, "A new case" as short steps that reproduce (R6.6)
        EXJS = """([tab,picks])=>{const card=()=>[...document.querySelectorAll('[data-exits]')].find(e=>e.offsetParent!==null), out=[];
          card().querySelector(`[data-ex="tab"][data-i="${tab}"]`).click();
          const nq=card().querySelectorAll('.ex-qs .pb-dot').length||1;
          for(let j=0;j<nq;j++){const dot=card().querySelector(`.ex-qs .pb-dot[data-j="${j}"]`); if(dot) dot.click();   // the card remembers the last question, also across a language switch
            const vs=picks[j]!=null?[String(picks[j])]:[...card().querySelectorAll('.ex-opts .choice')].map(b=>b.dataset.v);
            for(const v of vs){card().querySelector(`.ex-opts .choice[data-v="${v}"]`).click(); const r=card().querySelector('.ex-res');
              if(r&&(picks[j]!=null||r.classList.contains('ok'))){const bs=[...r.querySelectorAll('.num-big')], keep=bs.map(x=>x.textContent);
                bs.forEach(x=>{const m=(x.getAttribute('data-tipx')||'').match(/\\d[\\d,]*/); if(m) x.textContent=m[0];});
                out.push(r.innerText.replace(/\\u2011/g,'-').replace(/\\u00a0/g,' ')); bs.forEach((x,i)=>{x.textContent=keep[i];}); break;}}}
          return out;}"""
        ex = pg.evaluate(EXJS, [0, [2, 1]])      # a wrong pick shows its note, then the explanation
        mha_attn = 4 * 4096 * 4096
        check(f'{lang} Explain it, question 1: the note says “less than half” ({mha_attn:,} ÷ 176,160,768 = {mha_attn / 176160768:.2f}) and the explanation “3.5 times” and “about four fifths”, with no count written out',
              len(ex) == 2 and mha_attn / 176160768 < 0.5 and Z('less than half as many parameters as an MLP block', '不到一个 MLP 块的一半') in ex[0]
              and Z('the intermediate size is 3.5 times the hidden size', '中间维度是隐藏维度的 3.5 倍') in ex[0] and Z('So about four fifths of every layer is MLP.', '所以每层约五分之四的参数在 MLP 块里。') in ex[0]
              and not LONG.search(ex[0]), ex[:1])
        check(f'{lang} Explain it, question 2: the explanation says the model grows a little, from 8.03B to 8.30B (no exact-value hover), with no arithmetic in the sentence',
              len(ex) == 2 and Z(f'The model grows a little, from {big_str(8030261248)} to {big_str(8298696704)} parameters.', f'模型只长了一点，从 {big_str(8030261248)} 个参数变成 {big_str(8298696704)} 个。') in ex[1]
              and '×' not in nth(ex[1], 1).replace('KV heads × head_dim', '').replace('KV head 数 × head_dim', '') and (8298696704 - 8030261248) / 8030261248 < 0.05, ex[1:])
        nc = pg.evaluate(EXJS, [1, [None, None]])
        check(f'{lang} A new case: both answers shown, each the conclusion first, then one step per line (R6.20)', len(nc) == 2 and all(len([l for l in t_.split('\n') if l.strip()]) >= 4 and '=' not in t_.split('\n')[0] for t_ in nc), nc)
        m7 = count(CFG['mistral-7b'])
        x_tot = m7['total'] + 32 * 7 * m7['mlp'] + 32 * 4096 * 8
        x_act = m7['total'] + 32 * m7['mlp'] + 32 * 4096 * 8
        check(f'{lang} A new case: Mixtral-8x7B about {big_str(x_tot)} ({x_tot:,}, as Hugging Face reports), about {big_str(x_act)} per token; Llama-3.2-1B’s table 1.6 times its attention blocks',
              len(nc) == 2 and x_tot == 46702792704 and f'{x_tot:,}' in nc[0] and big_str(x_tot) in nc[0] and big_str(x_act) in nc[0]
              and f"{R['llama32-1b']['emb'] / R['llama32-1b']['parts']['attn']:.1f}" == '1.6' and '1.6' in nc[1] and big_str(R['llama32-1b']['parts']['attn']) in nc[1], nc)
        for t_ in nc:
            for l in [l.strip() for l in t_.split('\n') if l.strip()]:
                for part in re.split(r'[;；]', l):
                    if '=' not in part:
                        continue
                    seg = re.split(r'[:：]', part)[-1] if re.search(r'[:：]', part.split('=')[0]) else part
                    seg = re.sub(r'[(（][^()（）]*[A-Za-z][^()（）]*[)）]', '', seg)      # a note in brackets, "(Mistral-7B-v0.1)", "(8 KV heads × 64)"
                    ok, info = reproduces(seg)
                    check(f'{lang} A new case, step reproduces: {seg.strip()[:70]}', ok, info)
        # ---------------- drill
        for seed in range(5, 12):
            d = pg.evaluate("s=>window.__tf.makeDrill(s)", seed)
            c = d['cfg']
            mine = count(c)
            check(f'drill seed {seed}: engine = independent count', d['r']['total'] == mine['total'] and d['r']['layer'] == mine['layer'], (d['r']['total'], mine['total']))
            check(f'drill seed {seed}: head_dim is whole and at least 64', c['d'] % c['H'] == 0 and c['d'] // c['H'] >= 64 and c['KV'] <= c['H'])
        pg.click('#tf-app [data-act="drill-reveal"]'); pg.wait_for_timeout(150)
        dl = pg.evaluate("()=>[...document.querySelectorAll('#tf-drill-a p')].map(p=>p.innerText.replace(/\\u00a0/g,' '))")
        for l in dl[1:]:
            if '=' not in l:
                continue
            if 'GiB' in l:
                mm = re.search(r'(' + NUM + r') × 2 (?:bytes|字节) = (' + NUM + r') (?:bytes|字节) = (' + NUM + r') GB = (' + NUM + r') GiB', l)
                check(f'{lang} drill: memory line reproduces, bytes → GB → GiB', bool(mm) and int(mm.group(1).replace(',', '')) * 2 == int(mm.group(2).replace(',', ''))
                      and g1(int(mm.group(2).replace(',', '')) / 1e9) == mm.group(3) and g1(int(mm.group(2).replace(',', '')) / GiB) == mm.group(4), l)
                continue
            for part in re.split(r'[;；]', l):
                if '=' in part and re.search(r'\d', part.split('=')[0]):
                    ok, info = reproduces(part)
                    check(f'{lang} drill line reproduces: {part[:60]}', ok, info)
    # ---------------- home tile: where the parameters are, drawn to scale by area (EN and ZH)
    # Every number the tile shows, recomputed from count() above: the total in the caption (3 significant digits, the exact value on
    # hover), the share in the sentence (a whole percent), and in the tooltips the embedding table and the LM head (shape, count, share)
    # and every layer (count, share). The geometry: each table is a column of its share of the frame's width, the layers take the rest
    # as L equal cells, column by column, in the rows the rule below picks. Nothing counts the attention or MLP blocks (R3.8).
    def rows_rule(c, r):
        """the divisor of L that makes a cell closest to square in a 300 × 90 px frame"""
        L_ = c['L']; lay = L_ * r['layer'] / r['total']
        return min((x for x in range(1, L_ + 1) if L_ % x == 0), key=lambda x: abs(math.log((300 * lay / (L_ / x)) / (90 / x))))
    ROWS = {'llama32-1b': 2, 'llama31-8b': 4, 'llama31-70b': 5}
    for k in ROWS:
        check(f'tile {k}: {ROWS[k]} rows of cells, the divisor of {CFG[k]["L"]} closest to square cells in 300 × 90', rows_rule(CFG[k], R[k]) == ROWS[k], rows_rule(CFG[k], R[k]))
    TILE_JS = """()=>{const t=document.getElementById('tile-transformer'), q=s=>t.querySelector(s), rc=e=>{const r=e.getBoundingClientRect(); return {x:r.left,y:r.top,w:r.width,h:r.height};};
      const nb=q('.tf-tz-cap .num-big'), grid=q('.tf-tz-grid:not(.tf-tz-old)');
      return {cap:q('.tf-tz-cap').innerText, capBig:nb?nb.textContent:'', capTip:nb?nb.dataset.tipx:'', inner:rc(q('.tf-tz-in')), emb:rc(q('.tf-tz-emb')), head:rc(q('.tf-tz-head')),
        lay:rc(q('.tf-tz-lay')), embTip:q('.tf-tz-emb').dataset.tipx||'', headTip:q('.tf-tz-head').dataset.tipx||null, grids:t.querySelectorAll('.tf-tz-grid').length,
        cells:[...grid.querySelectorAll('.tf-tz-cell')].map(c=>({...rc(c), tip:c.dataset.tipx, op:getComputedStyle(c).opacity})),
        leg:[...t.querySelectorAll('.tz-leg > span')].map(s=>s.textContent), aria:q('.tf-tz-frame').getAttribute('aria-label'),
        legTips:[...t.querySelectorAll('.tz-leg > span')].map(s=>[s.dataset.k, s.dataset.tipx===undefined?null:s.dataset.tipx, s.tabIndex]),
        read:q('.tz-read').innerText.replace(/\\u2011/g,'-'), notes:[...t.querySelectorAll('.tz-note')].map(n=>n.textContent),
        anims:t.getAnimations({subtree:true}).filter(a=>a.playState==='running').length,
        all:t.innerText+' '+[...t.querySelectorAll('[data-tipx],[aria-label]')].map(e=>(e.dataset.tipx||'')+' '+(e.getAttribute('aria-label')||'')).join(' ')}}"""
    near = lambda a, b, tol=0.6: abs(a - b) <= tol
    # the home hero's words for the attention block (labs/architecture.js), which the tile repeats (home review v4 A #20, B-19)
    # since 2026-10-11 the hero names attention (the owner's review: titles use the real terms)
    hero = re.search(r"attn:\{en:'Through attention, each token takes information (from itself and the tokens before it)\.',zh:'每个 token 借助注意力，(从它自己和前面的 token 取信息)。'\}", (P / 'labs' / 'architecture.js').read_text())
    check('tile: the home hero says each token takes information from itself and the tokens before it (EN and ZH)', bool(hero))
    HERO = {'en': 'from itself and the tokens before it', 'zh': '从它自己和前面的 token 取信息'}
    pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(300)
    for lang in ('en', 'zh'):
        Z = (lambda en, zh: zh) if lang == 'zh' else (lambda en, zh: en)
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(150)
        pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="models"], #domStrip [data-world="models"]')].find(e=>e.offsetParent); if(g) g.click(); else {const m=document.querySelector('#mapSvg .world[data-world="models"]'); if(m) m.dispatchEvent(new MouseEvent('click',{bubbles:true}));}}"""); pg.wait_for_timeout(300)
        pg.evaluate("()=>document.getElementById('tile-transformer').scrollIntoView({block:'center'})"); pg.wait_for_timeout(1300)   # the build-up plays once (< 1 s)
        for k, (v0, l0) in {'llama32-1b': (21, 79), 'llama31-8b': (13, 87), 'llama31-70b': (3, 97)}.items():
            pg.click(f'#tile-transformer .seg-ctl button[data-v="{k}"]'); pg.wait_for_timeout(1100)   # the re-tiling ends by 0.96 s
            T = pg.evaluate(TILE_JS)
            c = CFG[k]; r = R[k]; t = r['total']; P_ = r['parts']; L_ = c['L']; layers = L_ * r['layer']; tied = c['tied']; nm = NAMES[k]
            js = pg.evaluate("c=>{const r=window.__tf.compute({...c,p:'bf16'}); return [r.total,r.layer,r.emb,r.head,r.layers]}", c)
            check(f'tile {lang} {k}: the engine gives the counts the tile uses (total, layer, embedding table, LM head, all layers)', js == [t, r['layer'], r['emb'], 0 if tied else r['emb'], layers], (js, t, r['layer']))
            rd = T['read']
            mine = (round((P_['emb'] + P_['head']) / t * 100), round(layers / t * 100))
            # R6.20: one key number, the vocabulary's share; the layers "hold the rest" ({l0}%, the final RMSNorm under 0.001% aside)
            check(f'tile {lang} {k}: {v0}% in the embedding table and LM head, the rest ({l0}%) in the {L_} layers', mine == (v0, l0) and v0 + l0 == 100 and f'{v0}%' in rd and f'{l0}%' not in rd
                  and Z('hold the rest', '其余都在') in rd and Z(f'{L_} layers', f'{L_} 层') in rd and nm in rd, (rd, mine))
            check(f'tile {lang} {k}: no attention or MLP share in the sentence (part 2’s answer, R3.8)', 'attention' not in rd and 'MLP' not in rd, rd)
            check(f'tile {lang} {k}: the note states the exact total and no longer speaks of a bar', f'{t:,}' in ' '.join(T['notes']) and not re.search(r'\bbar\b|条里', ' '.join(T['notes'])), T['notes'])
            # the caption: the whole frame is the model's parameters, 3 significant digits with the exact value on hover (R6.18)
            check(f'tile {lang} {k}: caption “{big_str(t)}” is all the parameters, with no exact-value hover (the owner, 2026-10-07)', T['capBig'] == big_str(t) and not T['capTip']
                  and Z(f'All {big_str(t)} parameters, drawn to scale by area', f'全部 {big_str(t)} 个参数，按面积等比例画') in T['cap'].replace(' ', ' '), (T['cap'], T['capTip']))
            # geometry: the tables are columns of their share of the frame's width; the layers take the rest, 2 px from each table
            Wi, Hi, x0 = T['inner']['w'], T['inner']['h'], T['inner']['x']
            e, h = T['emb'], T['head']
            check(f'tile {lang} {k}: the embedding table is {share_str(r["emb"] / t)} of the frame’s width ({e["w"]:.1f} of {Wi:.0f} px), at the left edge, full height',
                  near(e['w'], Wi * r['emb'] / t) and near(e['x'], x0) and near(e['h'], Hi), (e, Wi))
            if tied:
                check(f'tile {lang} {k}: tied, so no LM head block (one shared table, no tooltip)', h['w'] < 0.6 and T['headTip'] is None, (h, T['headTip']))
            else:
                check(f'tile {lang} {k}: the LM head is {share_str(r["emb"] / t)} of the width, at the right edge', near(h['w'], Wi * P_['head'] / t) and near(h['x'] + h['w'], x0 + Wi) and near(h['h'], Hi), (h, Wi))
            lay = T['lay']; gl, gr = 2, (0 if tied else 2)
            check(f'tile {lang} {k}: the layers take the rest of the width, {share_str(layers / t)} (the final RMSNorm, {share_str(c["d"] / t)}, is not drawn)',
                  near(lay['x'], e['x'] + e['w'] + gl) and near(lay['x'] + lay['w'], (x0 + Wi) if tied else (h['x'] - gr)) and near(lay['w'] + gl + gr, Wi * (1 - (P_['emb'] + P_['head']) / t))
                  and abs((1 - (P_['emb'] + P_['head']) / t) - layers / t) < 1e-5, (lay, Wi))
            cl = T['cells']; Rw = ROWS[k]; Cn = L_ // Rw
            xs = sorted({round(q['x'], 1) for q in cl}); ys = sorted({round(q['y'], 1) for q in cl})
            col = lambda q: min(range(len(xs)), key=lambda i: abs(xs[i] - q['x'])); row = lambda q: min(range(len(ys)), key=lambda i: abs(ys[i] - q['y']))
            check(f'tile {lang} {k}: {L_} equal cells, {Rw} rows × {Cn} columns, layer i at column i ÷ {Rw}, row i mod {Rw}',
                  len(cl) == L_ and len(xs) == Cn and len(ys) == Rw and max(q['w'] for q in cl) - min(q['w'] for q in cl) < 0.6 and max(q['h'] for q in cl) - min(q['h'] for q in cl) < 0.6
                  and all(col(q) * Rw + row(q) == i for i, q in enumerate(cl)), (len(cl), len(xs), len(ys)))
            check(f'tile {lang} {k}: the gaps between cells take under 15% of the layers’ area (visible cells {sum(q["w"] * q["h"] for q in cl) / (Wi * Hi) * 100:.1f}% of the frame for {layers / t * 100:.1f}%)',
                  sum(q['w'] * q['h'] for q in cl) >= 0.85 * layers / t * Wi * Hi and near(min(q['y'] for q in cl), T['inner']['y']) and near(max(q['y'] + q['h'] for q in cl), T['inner']['y'] + Hi))
            check(f'tile {lang} {k}: after the re-tiling, one grid, every cell fully drawn, no animation running', T['grids'] == 1 and all(q['op'] == '1' for q in cl) and T['anims'] == 0, (T['grids'], T['anims']))
            # tooltips: every layer, the embedding table and the LM head, with counts and shares recomputed here
            lt = lambda i: Z(f'Layer {i} of {L_} holds {big_str(r["layer"])} parameters, {share_str(r["layer"] / t)} of the model’s {big_str(t)}. All {L_} layers have the same shapes, so each holds the same number.',
                             f'第 {i} 层（共 {L_} 层）有 {big_str(r["layer"])} 个参数，占模型 {big_str(t)} 个参数的 {share_str(r["layer"] / t)}。{L_} 层的形状都一样，所以每层的参数一样多。')
            bad = [i for i, q in enumerate(cl, 1) if q['tip'] != lt(i)]
            check(f'tile {lang} {k}: every cell’s tooltip, e.g. “{lt(min(12, L_))[:70]}…”', not bad, (bad[:3], cl[bad[0] - 1]['tip'] if bad else ''))
            V_, d_ = c['V'], c['d']
            et = Z(f'The embedding table, a {V_:,} × {d_:,} matrix (vocabulary size × hidden size), holds {big_str(r["emb"])} parameters, {share_str(r["emb"] / t)} of the model’s {big_str(t)}.',
                   f'embedding 表是一个 {V_:,} × {d_:,} 的矩阵（词表大小 × 隐藏维度），有 {big_str(r["emb"])} 个参数，占模型 {big_str(t)} 个参数的 {share_str(r["emb"] / t)}。')
            et += Z(' The LM head reuses the embedding table’s parameters, so it adds none.', 'LM head 直接用 embedding 表的参数，不另加参数。') if tied else ''
            check(f'tile {lang} {k}: the embedding table’s tooltip ({V_:,} × {d_:,}, {big_str(r["emb"])}, {share_str(r["emb"] / t)})', T['embTip'] == et, (T['embTip'], et))
            if not tied:
                ht = Z(f'The LM head, {"an" if str(d_).startswith("8") else "a"} {d_:,} × {V_:,} matrix (hidden size × vocabulary size), holds {big_str(P_["head"])} parameters, {share_str(P_["head"] / t)} of the model’s {big_str(t)}.',
                       f'LM head 是一个 {d_:,} × {V_:,} 的矩阵（隐藏维度 × 词表大小），有 {big_str(P_["head"])} 个参数，占模型 {big_str(t)} 个参数的 {share_str(P_["head"] / t)}。')
                check(f'tile {lang} {k}: the LM head’s tooltip ({d_:,} × {V_:,}, {big_str(P_["head"])})', T['headTip'] == ht, (T['headTip'], ht))
            check(f'tile {lang} {k}: the picture’s label names the total and the per-layer count', nm in T['aria'] and big_str(t) in T['aria'] and big_str(r['layer']) in T['aria'] and str(L_) in T['aria'], T['aria'])
            leg = [Z('embedding table (tied to the LM head)', 'embedding 表（和 LM head 共享参数）') if tied else Z('embedding table', 'embedding 表'), Z(f'{L_} layers, one per cell', f'{L_} 层，每格一层')] + ([] if tied else ['LM head'])
            check(f'tile {lang} {k}: the legend names the blocks in the picture’s order', T['leg'] == leg, T['leg'])
            # the definitions that were the first note now live in the legend's tooltips (home review v4 A #7), each item reachable by keyboard
            emb_def = Z('The embedding table turns each token ID into a vector: it has one row for every token in the vocabulary, and the model looks up the token’s row.',
                        'embedding 表把每个 token ID 变成一个向量：词表里每个 token 占一行，模型查出这个 token 的那一行。')
            tied_def = Z(f'The embedding table turns each token ID into a vector, and the LM head (LM: language model) turns the final vector into one score for every token in the vocabulary. In {nm} the LM head reuses the embedding table’s parameters.',
                         f'embedding 表把每个 token ID 变成一个向量，LM head（LM 即语言模型）把最后的向量变成词表里每个 token 的一个得分。{nm} 的 LM head 直接用 embedding 表的参数。')
            lay_def = Z(f'Each layer has an attention block, where each token takes information {HERO["en"]}, and an MLP block (multi-layer perceptron), which transforms each token on its own.',
                        f'每层有一个 attention 块（每个 token 在这里{HERO["zh"]}）和一个 MLP 块（多层感知机，单独变换每个 token）。')
            head_def = Z('The LM head (LM: language model) turns the final vector into one score for every token in the vocabulary.', 'LM head（LM 即语言模型）把最后的向量变成词表里每个 token 的一个得分。')
            want_tips = [['emb', tied_def if tied else emb_def, 0], ['lay', lay_def, 0]] + ([] if tied else [['head', head_def, 0]])
            got_tips = [[a_, (b_ or '').replace('‑', '-'), c_] for a_, b_, c_ in T['legTips']]
            check(f'tile {lang} {k}: each legend item says what its part does, on hover and keyboard focus; the attention block in the hero’s words', got_tips == want_tips, (got_tips, want_tips))
            # the notes: at most 4 (here 2), one question each; no definitions left in them, and no claim that config.json gives the shapes (B-04)
            nt = ' '.join(T['notes'])
            check(f'tile {lang} {k}: two notes (the counts’ source and exactness; what the picture leaves out), no definitions in them',
                  len([x for x in T['notes'] if x.strip()]) == 2 and len(T['notes']) <= 4 and not re.search(r'takes information|transforms each token|turns (a|each) token|取信息|单独变换|变成向量|变成一个向量', nt)
                  and Z('the sizes in each model’s config.json on Hugging Face fix the shape of every matrix in Llama’s design', '各模型在 Hugging Face 上的 config.json 里的维数决定每个矩阵的形状') in nt
                  and Z('final RMSNorm, under 0.001% of the parameters', '最后的 RMSNorm（不到参数的 0.001%）') in nt and 'config.json gives the shape' not in nt and '三个' not in nt and 'three matrices' not in nt, T['notes'])
            # B-19: wherever the tile speaks of attention, a token takes information from itself and the tokens before it
            scan = (T['all'] + ' ' + nt).replace('‑', '-')
            olds = re.findall(r'takes information from (?!itself and the tokens before it).{0,30}', scan) + re.findall(r'.{0,20}(?<!从它自己和前面的 token )取信息', scan)
            check(f'tile {lang} {k}: no “from the tokens before it” without “itself” anywhere on the tile (B-19)', not olds and (HERO['en'] in scan if lang == 'en' else HERO['zh'] in scan), olds)
            # R3.8: no count or share of the attention or MLP blocks anywhere on the tile, its tooltips or its labels
            forbid = {big_str(r['attn']), big_str(r['mlp']), f"{r['attn']:,}", f"{r['mlp']:,}", big_str(P_['attn']), big_str(P_['mlp']), share_str(P_['attn'] / t), share_str(P_['mlp'] / t),
                      share_str(r['attn'] / r['layer']), share_str(r['mlp'] / r['layer'])}
            check(f'tile {lang} {k}: nothing counts the attention or MLP blocks (part 2’s prediction, R3.8)', not [x for x in forbid if x in T['all']], [x for x in forbid if x in T['all']])
            check(f'tile {lang} {k}: no count over 6 digits written out on the tile (R6.18)', not LONG.search(pg.evaluate("()=>document.getElementById('tile-transformer').innerText")))
    check('no page errors', not errs, errs[:3])
    b.close()

# ---------------------------------------------------------------- optional: token ids from Meta's Llama 3 tokenizer
tok = os.environ.get('TF_LLAMA_TOKENIZER')
if tok and pathlib.Path(tok).exists():
    import base64, tiktoken
    ranks = {}
    for line in open(tok, 'rb'):
        if line.strip():
            a, r_ = line.split(); ranks[base64.b64decode(a)] = int(r_)
    pat = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    enc = tiktoken.Encoding(name='llama3', pat_str=pat, mergeable_ranks=ranks, special_tokens={})
    check('tokenizer: "The capital of France is" ends with token 374, " Paris" is 12366, 128,000 base tokens + 256 special = 128,256',
          enc.encode('The capital of France is')[-1] == 374 and enc.encode(' Paris') == [12366] and len(ranks) + 256 == 128256)
else:
    print('(token ids not re-checked: set TF_LLAMA_TOKENIZER to models/llama3/tokenizer.model from meta-llama/llama-models)')

print(f'{n} checks, {len(fails)} failed')
if fails:
    print('FAILED:', fails[:40])
sys.exit(1 if fails else 0)
