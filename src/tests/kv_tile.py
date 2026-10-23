"""The KV cache memory lab's home tile (#tile-kv-cache): one GPU's memory against each user's context.

python3 tests/kv_tile.py            -> checks, plus screenshots in out/v4/shots-kv2/
python3 tests/kv_tile.py noshots    -> checks only

Builds its own page (python3 build.py out/v4/kv-tile.html --only kv-basics; app.js is always in) and wraps it in
out/v4/wrapped-kv-tile.html. For every slider position (2K to 32K in 2K steps), at 1440×900, 390×844 and 280×800, in EN and ZH, in the
light and the dark theme, it checks:
- the tile's own setup (R3.8): 64 users, a number no part, prediction, scenario or challenge of the lab uses (the lab's parts use
  100 users, its levers 100 or 10, its scenarios 100, 8, 32, 16, 16 and 4, its challenge 20 at 128K: read from app.js below);
  the slider starts at 16K, where the KV cache is already larger than the weights and everything still fits;
- every number in the sentence and on the chart against an independent recomputation from Llama-3.1-70B's shape
  (70,553,706,496 parameters, 80 layers, 8 KV heads × 128, BF16) on 8 × H100 with 64 users, 90% of 80 GiB usable, 2 GiB overhead;
  the KV cache's size always with one decimal (40.0 GiB at 16K, 80.0 GiB at 32K; B-14);
- which of the three sentences shows (fits / over / KV cache alone over);
- the chart's geometry against the same numbers, with the scale read from the drawn ticks: the weights band 16.4 GiB and the
  overhead band 2 GiB, both to scale (B-15), so at every position the wedge at the marker is thinner than the weights band exactly
  when the sentence says "still less"; the KV cache drawn whole as one straight wedge up to 98.4 GiB at 32K on a 0–100 GiB axis
  (nothing clipped), red stripes from the point where the line passes the usable limit (inside the axis), the dashed line, the
  marker and its dot; the kit's colour for each part;
- the axis caption is HTML over the chart's top margin and its "GiB" carries the kit's GiB definition (A.terms.gib) on hover and
  focus; it is the first GiB in the tile (B-13, R6.12); the caption overlaps no chart label;
- the KV cache's label sits inside the plain amber next to the marker (measured glyphs), and is left out at 2K where the wedge is
  too thin (at 4K only on the widest tiles); no label overlaps another or leaves the chart; no horizontal overflow;
- the sentence is one sentence of at most 30 English words (R11.5) with one key number, the KV cache's size (R6.20), and
  comparisons in words that hold for the numbers as shown; no "·", no two colons in a sentence (R6.7, R6.14);
- every legend item has a tooltip and the swatch the chart uses; the notes are at most 4 short sentences, one per question
  (A #7): what the chart holds, why 16.4 GiB (BF16, 2 bytes per parameter, GB and GiB; B-13), the per-token size and the growth
  in one sentence, why 72 GiB; each formula redoes from the numbers as written (R6.6);
- no label states where the line crosses the limit (R3.8): no 21.4K or 21,943 (this tile) and no 13.7K or 14,04x (the lab's
  part 3) anywhere in the tile, its notes or its tooltips.
"""
import math, pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
SHOTS = 'noshots' not in sys.argv
OUT = P / 'out/v4'; SH = OUT / 'shots-kv2'; SH.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/v4/kv-tile.html', '--only', 'kv-basics'], cwd=P, check=True, capture_output=True)
W = OUT / 'wrapped-kv-tile.html'
W.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
             '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (OUT / 'kv-tile.html').read_text() + '</body></html>')
URL = 'file://' + str(W)

fails, checks = [], 0
def check(name, ok, info=''):
    global checks; checks += 1
    if not ok: fails.append(f'{name}: {info}'); print('FAIL', name, info)

# ---------- independent numbers ----------
GiB = 2 ** 30
PARAMS, LAYERS, KVH, HD, BYTES, N, USERS, MEM, UTIL, OH = 70_553_706_496, 80, 8, 128, 2, 8, 64, 80, 0.9, 2.0
def r1(v): return math.floor(v * 10 + 0.5 + 1e-7) / 10           # half up, as the page rounds
def g1(v):                                                           # "72 GiB": a limit, at most one decimal
    s = f'{r1(v):,.1f}'; return (s[:-2] if s.endswith('.0') else s) + ' GiB'
def gk(v): return f'{r1(v):,.1f} GiB'                               # a KV cache size: always one decimal (B-14)
W_GPU = PARAMS * BYTES / N / GiB                                     # 16.427 GiB of weights on each GPU
W_SHOWN = r1(r1(PARAMS * BYTES / GiB) / N)                           # the lab shows 131.4 GiB in all, so 16.4 on each GPU
KV_TOK = 2 * LAYERS * KVH * HD * BYTES                               # 327,680 bytes = 320 KiB per token
PER_1K = KV_TOK * 1024 * USERS / N / GiB                             # 2.5 GiB per GPU for each 1K of context per user
USE = MEM * UTIL                                                     # 72 GiB usable per GPU
BASE = W_GPU + OH
CROSS = (USE - BASE) / PER_1K                                        # 21.43K: where the line passes the usable limit
ALONE = USE / PER_1K                                                 # 28.8K: where the KV cache alone passes it
NEED32 = BASE + 32 * PER_1K                                          # 98.4 GiB at 32K
YMAX = math.ceil(NEED32 / 50) * 50                                   # the axis holds the whole line: 0–100 GiB
BREAK = int((USE * GiB - PARAMS * BYTES / N - OH * GiB) * N / (USERS * KV_TOK) + 1e-9)   # 21,943 tokens, as the lab would count it
assert KV_TOK == 320 * 1024 and PER_1K == 2.5 and W_SHOWN == 16.4 and YMAX == 100 and BREAK == 21943 and ALONE == 28.8
check('the product owner’s ask holds on the axis: the total passes the usable limit inside 0–32K, the KV cache ends far above the weights',
      2 < CROSS < 32 and 32 * PER_1K > 3 * W_GPU, (CROSS, 32 * PER_1K))
check('the KV cache alone passes the limit inside the axis too (30K and 32K), after the total did', CROSS < ALONE < 32, (CROSS, ALONE))
def expect(k):
    kv = PER_1K * k; need = BASE + kv; kvs = r1(kv)
    return dict(kv=kv, kvs=kvs, need=need, fits=need <= USE, alone=kvs > USE, more=kv > W_GPU)
check('default 16K: the KV cache already larger than the weights, and everything fits', expect(16)['more'] and expect(16)['fits'] and gk(expect(16)['kv']) == '40.0 GiB', expect(16))

# ---------- R3.8: the tile's setup is not one the lab uses (read from app.js, not typed here) ----------
SRC = (P / 'app.js').read_text()
A0, A1 = SRC.index('/* ---------- KV teaser for the home tiles ---------- */'), SRC.index('/* ---------- kit API: what lab and teaser modules may use ---------- */')
LAB_SRC = SRC[:A0] + SRC[A1:]                                                                # the lab and the rest of the core, without the tile
lab_users = set(int(x) for x in re.findall(r'\busers:(\d+)', LAB_SRC))                   # DEFAULT, levers, scenarios, exits
lab_users |= set(int(x) for x in re.findall(r'/ (\d+) \$\{tx\(\'users\'', LAB_SRC))       # the challenge goal ("/ 20 users")
lab_users |= set(int(x) for x in re.findall(r'goal:(\d+)', LAB_SRC))
check('R3.8: the lab’s user counts were found (100 in DEFAULT, 10 in a lever, 8 / 32 / 16 / 4 in scenarios, 20 in the challenge)',
      {100, 10, 8, 32, 16, 4, 20} <= lab_users, sorted(lab_users))
tz = re.search(r'const KVTZ=\{users:(\d+),min:(\d+),max:(\d+),step:(\d+),k0:(\d+)', SRC)
check('R3.8: the tile’s own users: 64, used by no part, prediction, scenario or challenge of the lab', tz and int(tz.group(1)) == USERS and USERS not in lab_users, (tz and tz.groups(), sorted(lab_users)))
check('the slider: 2K to 32K in 2K steps, starting at 16K', tz and tuple(map(int, tz.groups()[1:])) == (2, 32, 2, 16), tz and tz.groups())

GEOM = """el=>{const s=el.querySelector('svg.tz-kvc-vis'); const vb=s.viewBox.baseVal; const q=c=>[...s.querySelectorAll(c)];
  const ln=e=>({x1:+e.getAttribute('x1'),y1:+e.getAttribute('y1'),x2:+e.getAttribute('x2'),y2:+e.getAttribute('y2')});
  const bb=e=>{const b=e.getBBox(); return {x:b.x,y:b.y,w:b.width,h:b.height,t:e.textContent,c:e.getAttribute('class')||''};};
  const dot=s.querySelector('.tz-kvc-dot'), sr=s.getBoundingClientRect(), cap=el.querySelector('.tz-kvc-cap');
  // the HTML caption in the SVG's coordinates (the SVG is drawn 1:1); its text box, and the GiB with its dotted underline
  const rr=(r)=>({x:r.left-sr.left,y:r.top-sr.top,w:r.width,h:r.height});
  const range=document.createRange(); range.selectNodeContents(cap);
  return {W:vb.width,H:vb.height,cw:el.clientWidth,sw:sr.width,
    ticks:q('.tz-kvc-tick').map(ln),grid:ln(s.querySelector('.tz-kvc-grid')),lim:ln(s.querySelector('.tz-kvc-lim')),ovl:ln(s.querySelector('.tz-kvc-ovl')),
    mk:ln(s.querySelector('.tz-kvc-mk')),dot:{x:+dot.getAttribute('cx'),y:+dot.getAttribute('cy')},texts:q('text').map(bb),
    cap:{...rr(range.getBoundingClientRect()),t:cap.textContent},capDef:rr(cap.querySelector('.def').getBoundingClientRect()),
    paths:Object.fromEntries(q('path').map(p=>[p.getAttribute('class'),p.getAttribute('d')])),
    fills:Object.fromEntries(['w','oh','kv','ov','hs'].map(c=>{const e=s.querySelector('.tz-kvc-'+c); return [c,e?getComputedStyle(e).fill:'']})),
    ovl_stroke:getComputedStyle(s.querySelector('.tz-kvc-ovl')).stroke,
    tokens:Object.fromEntries(['weights','overhead','kv','oom'].map(c=>[c,getComputedStyle(document.documentElement).getPropertyValue('--c-'+c).trim()])),
    tag:(()=>{const e=s.querySelector('.tz-kvc-tag'); if(!e) return null; const cs=getComputedStyle(e), cv=document.createElement('canvas').getContext('2d');
      cv.font=`${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`; const m=cv.measureText(e.textContent);
      const a=e.getAttribute('text-anchor')||cs.textAnchor, x=+e.getAttribute('x')-(a==='end'?m.width:a==='middle'?m.width/2:0);   // x: the glyphs' left end, wherever the anchor puts them
      return {t:e.textContent,x,y:+e.getAttribute('y'),w:m.width,asc:m.actualBoundingBoxAscent,desc:m.actualBoundingBoxDescent,fill:cs.fill}})()}}"""
def rgb(h):
    h = h.lstrip('#'); return f'rgb({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)})'


def tile_state(pg):
    return pg.evaluate("""()=>{const t=document.querySelector('#tile-kv-cache'); const i=t.querySelector('input[type=range]');
      const d=t.querySelector('.tz-kvc-cap .def');
      // the text of the tile in DOM order up to the caption's GiB: no GiB may come before its definition
      const walk=document.createTreeWalker(t,NodeFilter.SHOW_TEXT); let before='', n; while((n=walk.nextNode())){if(d&&d.contains(n)) break; before+=n.textContent+' ';}
      return {out:t.querySelector('.tz-lab output').textContent, read:t.querySelector('.tz-read').innerText.replace(/\\u2011/g,'-').replace(/\\u00a0/g,' '),
        bold:[...t.querySelectorAll('.tz-read b')].map(b=>b.textContent.replace(/ /g,' ')), raw:t.querySelector('.tz-read').innerHTML, vt:i.getAttribute('aria-valuetext'), aria:t.querySelector('svg.tz-kvc-vis').getAttribute('aria-label'),
        leg:[...t.querySelectorAll('.tz-leg>span')].map(s=>({t:s.innerText.trim(), tip:!!(s.dataset.tip||s.dataset.tipx), hidden:s.hidden, sw:(s.querySelector('i')||{}).className||''})),
        notes:[...t.querySelectorAll('.tz-note')].map(n=>n.textContent.replace(/\\u2011/g,'-')),
        cap:t.querySelector('.tz-kvc-cap').textContent, def:d?{t:d.textContent, tip:d.getAttribute('data-tipx'), tab:d.getAttribute('tabindex')}:null, before,
        tips:[...t.querySelectorAll('[data-tipx]')].map(e=>e.getAttribute('data-tipx')).join(' '),
        ov:document.documentElement.scrollWidth-innerWidth, tzov:t.querySelector('.tile-tz').scrollWidth-t.querySelector('.tile-tz').clientWidth,
        all:t.innerText.replace(/\\u2011/g,'-')}}""")

def set_k(pg, k):
    pg.evaluate("k=>{const e=document.querySelector('#tile-kv-cache input[type=range]'); e.value=k; e.dispatchEvent(new Event('input',{bubbles:true}))}", k)
    pg.wait_for_timeout(60)

# the kit's GiB definition, read once from the page (A.terms.gib), in each language
GIB_DEF = {}
NO_BREAK = re.compile(r'21\.4\s?K|21\.43|21,?94[34]|28\.8\s?K|13\.7\s?K|14,?04[0-9]')   # this tile's crossings and the lab's part 3 break point

def check_state(tag, k, s, g, lang):
    e = expect(k); kk = f'{k}K'
    # ---- the sentence ----
    rd = s['read']
    check(f'{tag} {kk}: output', s['out'] == (f'{kk} tokens' if lang == 'en' else f'{kk} 个 token'), s['out'])
    check(f'{tag} {kk}: aria-valuetext', s['vt'] == s['out'], s['vt'])
    head = (f'Llama-3.1-70B on 8 × H100, {USERS} users with {kk} of context each: ' if lang == 'en' else f'8 × H100 跑 Llama-3.1-70B，{USERS} 个用户，每人 {kk} 上下文：')
    check(f'{tag} {kk}: condition written in full (model, GPUs, users, context)', rd.startswith(head), rd)
    check(f'{tag} {kk}: KV cache in bold, one decimal (B-14)', s['bold'] == [gk(e['kvs'])] and re.fullmatch(r'\d+\.\d GiB', s['bold'][0] if s['bold'] else ''), s['bold'])
    # one key number, the KV cache's size; the comparisons in words must hold for the numbers as shown (16.4 GiB of weights, 72 usable)
    if e['fits']:
        check(f'{tag} {kk}: "more/less than the weights" holds for the shown numbers', e['more'] == (e['kvs'] > W_SHOWN), (e['kvs'], W_SHOWN))
        check(f'{tag} {kk}: "fits" holds for the shown numbers (16.4 + 2 + KV cache ≤ 72; R6.6)', r1(W_SHOWN + e['kvs'] + OH) <= USE, e)
        if lang == 'en':
            want = f"the KV cache takes {gk(e['kvs'])} per GPU, {'already more' if e['more'] else 'still less'} than the weights, {'and' if e['more'] else 'but'} grows linearly with context."
        else:
            want = f"每张卡的 KV cache 占 {gk(e['kvs'])}，{'已经比权重多，而且' if e['more'] else '还比权重少，但'}随上下文线性增长。"
    elif not e['alone']:
        check(f'{tag} {kk}: the shown parts pass the limit (16.4 + 2 + KV cache > 72; R6.6)', r1(W_SHOWN + e['kvs'] + OH) > USE and e['kvs'] <= USE, e)
        want = (f"the KV cache needs {gk(e['kvs'])} per GPU and pushes the total past the usable {g1(USE)}." if lang == 'en'
                else f"每张卡的 KV cache 要 {gk(e['kvs'])}，总量因此超过了可用的 {g1(USE)}。")
    else:
        check(f'{tag} {kk}: the KV cache alone passes the limit as shown', e['kvs'] > USE, e)
        want = (f"the KV cache alone needs {gk(e['kvs'])} per GPU, more than the usable {g1(USE)}." if lang == 'en'
                else f"每张卡光 KV cache 就要 {gk(e['kvs'])}，超过了可用的 {g1(USE)}。")
    check(f'{tag} {kk}: sentence', rd == head + want, f'\n   got  {rd}\n   want {head + want}')
    check(f'{tag} {kk}: no size breaks from its unit (no-break space before GiB)', not re.search(r'\d [GK]iB', s['raw']) and len(re.findall('(?: |&nbsp;)[GK]iB', s['raw'])) == len(re.findall(r'[GK]iB', rd)), s['raw'])
    if lang == 'en':
        words = [w for w in rd.split() if re.search(r'[A-Za-z0-9]', w)]
        check(f'{tag} {kk}: at most 30 words (R11.5)', len(words) <= 30, len(words))
    check(f'{tag} {kk}: one sentence, no "·", at most one colon (R6.7, R6.14)', '·' not in rd and rd.count(':') + rd.count('：') <= 1 and len(re.findall(r'[.。](?=\s|$)', rd)) == 1, rd)
    # ---- the chart: scale from the drawn ticks ----
    xt = sorted(g['ticks'], key=lambda t: t['x1']); x0, x32 = xt[0]['x1'], xt[-1]['x1']; b0 = xt[0]['y1']; top = g['grid']['y1']
    X = lambda c: x0 + c / 32 * (x32 - x0); Y = lambda gi: b0 - gi / YMAX * (b0 - top)
    near = lambda a, b, tol=0.15: abs(a - b) <= tol
    check(f'{tag} {kk}: x ticks at 0, 8K, … 32K', all(near(t['x1'], X(c), 0.11) for t, c in zip(xt, [0, 8, 16, 24, 32])), [t['x1'] for t in xt])
    check(f'{tag} {kk}: y labels: the usable 72 and the top of the axis, 100 (plus the x axis’ 0)', sorted(t['t'] for t in g['texts'] if re.fullmatch(r'\d+', t['t'])) == ['0', '100', '72'], [t['t'] for t in g['texts']])
    check(f'{tag} {kk}: marker at the chosen context', near(g['dot']['x'], X(k)) and near(g['mk']['x1'], X(k)) and near(g['mk']['y1'], b0), (g['dot'], X(k)))
    check(f'{tag} {kk}: dot on top of the stack (weights + overhead + KV cache)', near(g['dot']['y'], Y(e['need'])), (g['dot']['y'], Y(e['need'])))
    check(f'{tag} {kk}: dashed line at the usable 72 GiB', near(g['lim']['y1'], Y(USE)) and near(g['lim']['y2'], Y(USE)), g['lim'])
    pts = lambda cls: [(float(u), float(v)) for u, v in re.findall(r'[ML](-?[\d.]+) (-?[\d.]+)', g['paths'].get(cls, ''))]
    same = lambda got, want: len(got) == len(want) and all(near(a, c) and near(b, d) for (a, b), (c, d) in zip(got, want))
    yb, yw, ye = Y(BASE), Y(W_GPU), Y(NEED32)
    check(f'{tag} {kk}: weights band to scale, 16.4 GiB (B-15)', same(pts('tz-kvc-w'), [(X(0), b0), (X(32), b0), (X(32), yw), (X(0), yw)]), (pts('tz-kvc-w'), yw))
    check(f'{tag} {kk}: overhead band on top of it to scale, 2 GiB (about 2 px; B-15)', same(pts('tz-kvc-oh'), [(X(0), yw), (X(32), yw), (X(32), yb), (X(0), yb)]) and near(yw - yb, OH / YMAX * (b0 - top), 0.2), (pts('tz-kvc-oh'), yw - yb))
    # B-15: the picture says what the sentence says; the wedge's thickness at the marker against the weights band's
    wedge, band = yb - Y(BASE + e['kv']), b0 - yw
    check(f'{tag} {kk}: the wedge at the marker is thicker than the weights band exactly when the KV cache is larger (B-15)', (wedge > band) == e['more'], (wedge, band, e['more']))
    check(f'{tag} {kk}: the KV cache drawn whole, one straight wedge from 0 to 80 GiB at 32K (not capped at the limit)', same(pts('tz-kvc-kv'), [(X(0), yb), (X(32), ye), (X(32), yb)]) and ye > top, (pts('tz-kvc-kv'), ye, top))
    check(f'{tag} {kk}: red stripes over the part above the usable limit, from where the line passes it', same(pts('tz-kvc-ov'), [(X(CROSS), Y(USE)), (X(32), ye), (X(32), Y(USE))]), (pts('tz-kvc-ov'), X(CROSS)))
    check(f'{tag} {kk}: red edge along the top of that part', near(g['ovl']['x1'], X(CROSS)) and near(g['ovl']['y1'], Y(USE)) and near(g['ovl']['x2'], X(32)) and near(g['ovl']['y2'], ye), g['ovl'])
    tk = g['tokens']; fl = g['fills']
    check(f'{tag} {kk}: kit colours: blue weights, violet overhead, amber KV cache, red stripes and edge (R5.6)',
          fl['w'] == rgb(tk['weights']) and fl['oh'] == rgb(tk['overhead']) and fl['kv'] == rgb(tk['kv']) and fl['hs'] == rgb(tk['oom'])
          and g['ovl_stroke'] == rgb(tk['oom']) and 'tz-kvc-hatch' in fl['ov'], (fl, tk, g['ovl_stroke']))
    # ---- labels: inside the chart, apart (the HTML caption included); the KV cache's size inside the amber next to the marker ----
    T = g['texts'] + [{'x': g['cap']['x'], 'y': g['cap']['y'], 'w': g['cap']['w'], 'h': g['cap']['h'], 't': g['cap']['t'], 'c': 'caption'},
                      {'x': g['capDef']['x'], 'y': g['capDef']['y'], 'w': g['capDef']['w'], 'h': g['capDef']['h'], 't': 'GiB, with its dotted underline', 'c': 'caption'}]
    check(f'{tag} {kk}: chart as wide as the tile', abs(g['W'] - g['cw']) <= 1 and abs(g['sw'] - g['cw']) <= 1, (g['W'], g['cw'], g['sw']))
    check(f'{tag} {kk}: the caption above the plot', g['capDef']['y'] + g['capDef']['h'] <= top - 1 and g['cap']['y'] >= -0.5, (g['cap'], g['capDef'], top))
    for t in T:
        check(f'{tag} {kk}: label “{t["t"]}” inside the chart', t['x'] >= -0.5 and t['x'] + t['w'] <= g['W'] + 0.5 and t['y'] >= -0.5 and t['y'] + t['h'] <= g['H'] + 0.5, t)
    for i in range(len(T)):
        for j in range(i + 1, len(T)):
            a, b = T[i], T[j]
            if a['c'] == 'caption' and b['c'] == 'caption': continue
            hit = a['x'] < b['x'] + b['w'] - 0.5 and b['x'] < a['x'] + a['w'] - 0.5 and a['y'] < b['y'] + b['h'] - 1 and b['y'] < a['y'] + a['h'] - 1
            check(f'{tag} {kk}: “{a["t"]}” and “{b["t"]}” apart', not hit, (a, b))
    tg = g['tag']
    if k == 2:
        check(f'{tag} {kk}: no label where the wedge is too thin for it (the sentence gives the number)', tg is None, tg)
    elif k >= 6:
        check(f'{tag} {kk}: one label, the KV cache’s size, one decimal', tg is not None and tg['t'] == 'KV ' + gk(e['kvs']), tg)
    if tg:
        l, r, gt, gb = tg['x'], tg['x'] + tg['w'], tg['y'] - tg['asc'], tg['y'] + tg['desc']
        cl = (l - x0) / (x32 - x0) * 32; xd = X(k); gap = l - xd if l > xd else xd - r
        check(f'{tag} {kk}: the label inside the plain amber (above the overhead, under the wedge’s edge and the limit)',
              gb <= yb - 0.5 and gt >= max(Y(BASE + PER_1K * cl), Y(USE)) + 1 and X(0) <= l and r <= X(32), dict(tag=tg, yb=yb, edge=Y(BASE + PER_1K * cl)))
        check(f'{tag} {kk}: the label next to the marker, not on it', 3 <= gap <= 14, (gap, tg, xd))
        check(f'{tag} {kk}: the label in dark text on the amber', tg['fill'] in ('rgb(26, 18, 0)',), tg['fill'])
    # ---- page ----
    check(f'{tag} {kk}: no horizontal overflow', s['ov'] <= 0 and s['tzov'] <= 0, (s['ov'], s['tzov']))
    check(f'{tag} {kk}: no label names a break point (R3.8)', not NO_BREAK.search(s['all'] + ' ' + ' '.join(t['t'] for t in T) + ' ' + ' '.join(s['notes']) + ' ' + s['tips']), NO_BREAK.search(s['all'] + ' '.join(s['notes'])))
    check(f'{tag} {kk}: every legend item has a tooltip, none hidden', len(s['leg']) == 6 and all(x['tip'] and not x['hidden'] for x in s['leg']), s['leg'])
    check(f'{tag} {kk}: legend swatches as drawn: blue, violet, amber, amber with red stripes, dashed line, line with a dot (R5.7)',
          [x['sw'] for x in s['leg']] == ['sw w', 'sw oh', 'sw kv', 'sw tz-kvc-swov', 'sw lim', 'tz-kvc-swmk'], [x['sw'] for x in s['leg']])

def check_caption(tag, s, lang):
    want = 'Memory per GPU (GiB)' if lang == 'en' else '每张 GPU 的显存（GiB）'
    check(f'{tag}: axis caption', s['cap'] == want, s['cap'])
    d = s['def']
    check(f'{tag}: the caption’s GiB carries the kit’s GiB definition, focusable (B-13, R6.12)', d and d['t'] == 'GiB' and d['tip'] == GIB_DEF[lang] and d['tab'] == '0', d)
    check(f'{tag}: no GiB in the tile before its definition', 'GiB' not in s['before'], s['before'][-200:])
    check(f'{tag}: the definition says what a GiB is and how GPU memory is read', ('2³⁰' in GIB_DEF[lang]) and ('GB' in GIB_DEF[lang]), GIB_DEF[lang][:80])

def check_notes(tag, s, lang):
    n = s['notes']
    check(f'{tag}: 4 notes at most, one sentence each (A #7, R11.5)', len(n) == 4 and all(len(re.findall(r'[.。](?=\s|$)', x)) == 1 for x in n), n)
    wgb = f'{PARAMS * BYTES / N / 1e9:.1f} GB'; p4 = f'{PARAMS / 1e9:.2f}B'; per = f'{PER_1K:g} GiB'
    # R6.6: each formula redoes from the numbers as written
    check(f'{tag}: weights formula redoes: {p4} × 2 ÷ 8 = {wgb} = 16.4 GiB', f'{float(p4[:-1]) * 2 / 8:.1f} GB' == wgb and r1(float(wgb[:-3]) * 1e9 / GiB) == W_SHOWN == r1(W_GPU), (p4, wgb))
    check(f'{tag}: per-token formula redoes: 2 × 80 × 8 × 128 × 2 = 320 KiB', 2 * 80 * 8 * 128 * 2 == 320 * 1024, '')
    check(f'{tag}: growth redoes: 320 KiB × 1,024 × {USERS} ÷ 8 = {per}, exactly, so every KV cache size is 2.5 × the context in K', 320 * 1024 * 1024 * USERS / 8 / GiB == 2.5 and per == '2.5 GiB', per)
    check(f'{tag}: usable redoes: 80 × 90% = 72', g1(MEM * UTIL) == '72 GiB', '')
    if lang == 'en':
        want = ['The chart shows one of the 8 GPUs: 1/8 of the weights, 1/8 of every user’s KV cache, and an estimated 2 GiB of runtime overhead.',
                f'The weights are in BF16, 2 bytes per parameter: each GPU holds {p4} parameters × 2 bytes ÷ 8 = {wgb} = 16.4 GiB.',
                f'Each token needs 320 KiB (2 × 80 layers × 8 KV heads × 128 numbers × 2 bytes), so 1K more context for each of the {USERS} users adds {per} to every GPU.',
                'An H100 has 80 GB, read here as 80 GiB, and the server uses only 90% of it, so each GPU can use 72 GiB.']
        for i, x in enumerate(n):
            words = [w for w in x.split() if re.search(r'[A-Za-z0-9]', w)]
            check(f'{tag}: note {i + 1} short (at most 32 words)', len(words) <= 32, len(words))
    else:
        want = ['图里画的是 8 张 GPU 里的一张：1/8 的权重、每个用户 1/8 的 KV cache，以及估计 2 GiB 的运行时开销。',
                f'权重用 BF16，每个参数 2 字节：每张卡存 {p4} 个参数 × 2 字节 ÷ 8 = {wgb} = 16.4 GiB。',
                f'每个 token 要 320 KiB（2 × 80 层 × 8 个 KV head × 128 维 × 2 字节），所以 {USERS} 个用户的上下文每人多 1K，每张卡就多 {per}。',
                'H100 有 80 GB（这里按 80 GiB 计），服务端只用其中 90%，所以每张卡可用 72 GiB。']
    for i, (x, w) in enumerate(zip(n, want)):
        check(f'{tag}: note {i + 1}', x == w, f'\n   got  {x}\n   want {w}')
    for x in n:
        check(f'{tag}: note without "·" and at most one colon (R6.7, R6.14)', '·' not in x and x.count(':') + x.count('：') <= 1, x)

errors = []
with sync_playwright() as p:
    b = p.chromium.launch()
    for (w, h), langs, themes in [((1440, 900), ['en', 'zh'], ['light', 'dark']), ((390, 844), ['en', 'zh'], ['light', 'dark']), ((280, 800), ['en', 'zh'], ['light', 'dark'])]:
        for lang in langs:
            for theme in themes:
                tag = f'{w} {lang} {theme}'
                ctx = b.new_context(viewport={'width': w, 'height': h}, color_scheme=theme)
                pg = ctx.new_page(); pg.on('pageerror', lambda e: errors.append(str(e)))
                pg.goto(URL); pg.wait_for_timeout(400)
                pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
                if lang not in GIB_DEF: GIB_DEF[lang] = pg.evaluate('()=>window.ASP.terms.gib()')
                pg.click('#domStrip .ds-btn[data-world="models"]'); pg.wait_for_timeout(400)   # since 2026-10-12 the lab sits on the trunk, after “Why a KV cache”
                tile = pg.locator('#tile-kv-cache'); tile.scroll_into_view_if_needed(); pg.wait_for_timeout(200)
                s0 = tile_state(pg)
                check(f'{tag}: default 16K', s0['out'].startswith('16K'), s0['out'])
                check_caption(tag, s0, lang)
                check_notes(tag, s0, lang)
                for k in range(2, 33, 2):
                    set_k(pg, k)
                    check_state(tag, k, tile_state(pg), pg.evaluate(GEOM, pg.query_selector('#tile-kv-cache .tile-tz')), lang)
                    if SHOTS and k in (2, 8, 16, 28, 32):
                        tile.screenshot(path=str(SH / f'{w}_{lang}_{theme}_{k}K.png'))
                # tooltips: the caption's GiB (hover and keyboard focus), each legend item, the notes
                if theme == 'light':
                    set_k(pg, 16)
                    pg.hover('#tile-kv-cache .tz-kvc-cap .def'); pg.wait_for_timeout(200)
                    tip = pg.evaluate("()=>{const t=document.getElementById('tip'); return t&&!t.hidden?t.innerText.trim():''}")
                    check(f'{tag}: hovering the caption’s GiB shows its definition', tip == GIB_DEF[lang].strip(), tip[:120])
                    if SHOTS and w in (1440, 390): pg.screenshot(path=str(SH / f'{w}_{lang}_{theme}_gib-tip.png'))
                    pg.mouse.move(2, 2); pg.wait_for_timeout(150); pg.evaluate("()=>{const t=document.getElementById('tip'); if(t) t.hidden=true}")
                    pg.focus('#tile-kv-cache .tz-kvc-cap .def'); pg.wait_for_timeout(200)
                    tip = pg.evaluate("()=>{const t=document.getElementById('tip'); return t&&!t.hidden?t.innerText.trim():''}")
                    check(f'{tag}: focusing the caption’s GiB shows its definition', tip == GIB_DEF[lang].strip(), tip[:120])
                    pg.evaluate('()=>document.activeElement&&document.activeElement.blur()'); pg.wait_for_timeout(100)
                if w == 1440 and theme == 'light':
                    for i in range(6):
                        it = pg.locator('#tile-kv-cache .tz-leg > span').nth(i); it.hover(); pg.wait_for_timeout(150)
                        tip = pg.evaluate("()=>{const t=document.getElementById('tip'); return t&&!t.hidden?t.innerText.trim():''}")
                        check(f'{tag}: legend item {i + 1} tooltip', len(tip) > (30 if lang == 'en' else 15) and tip[-1] in '.。', tip)
                        if i == 4: check(f'{tag}: limit tooltip names no direction (the rest is above, not to the right)', '右边' not in tip and 'right' not in tip, tip)
                    check(f'{tag}: no “Assumptions & sources” on the tile (owner 2026-10-13: the tile is one button, no “Assumptions & sources” on it)', pg.evaluate("()=>!document.querySelector('#tile-kv-cache .tz-more')&&!/假设和来源|Assumptions (&|and) sources/.test(document.querySelector('#tile-kv-cache').innerText)"))
                # state survives a language switch and a branch switch (module scope)
                set_k(pg, 20); other = 'zh' if lang == 'en' else 'en'
                pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(250)
                v = pg.evaluate("()=>document.querySelector('#tile-kv-cache input[type=range]').value")
                check(f'{tag}: context kept across a language switch', v == '20', v)
                pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(250)
                pg.click('#domStrip .ds-btn[data-world="inference"]'); pg.wait_for_timeout(250); pg.click('#domStrip .ds-btn[data-world="models"]'); pg.wait_for_timeout(300)
                v = pg.evaluate("()=>document.querySelector('#tile-kv-cache input[type=range]').value")
                check(f'{tag}: context kept across a branch switch', v == '20', v)
                set_k(pg, 16)
                ctx.close()
    # resize: the chart redraws in real pixels when the tile changes width
    ctx = b.new_context(viewport={'width': 1440, 'height': 900}); pg = ctx.new_page(); pg.on('pageerror', lambda e: errors.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(400); pg.click('#domStrip .ds-btn[data-world="models"]'); pg.wait_for_timeout(400)   # since 2026-10-12 the lab sits on the trunk, after “Why a KV cache”
    for vw in (1100, 760, 390, 1440):
        pg.set_viewport_size({'width': vw, 'height': 900}); pg.wait_for_timeout(450)
        g = pg.evaluate(GEOM, pg.query_selector('#tile-kv-cache .tile-tz'))
        check(f'resize to {vw}: chart redrawn at the tile’s width', abs(g['W'] - g['cw']) <= 1, (g['W'], g['cw']))
        ov = pg.evaluate('()=>document.documentElement.scrollWidth-innerWidth'); check(f'resize to {vw}: no horizontal overflow', ov <= 0, ov)
    ctx.close()
    b.close()
check('no page errors', not errors, errors[:3])
print(f'{checks - len(fails)}/{checks} checks passed' + (f'; screenshots in {SH.relative_to(P)}' if SHOTS else ''))
sys.exit(1 if fails else 0)
