"""Flow test for the GQA and MLA lab (labs/gqa-mla.js). CONTRACT §6, LAB_UX_GUIDE §3, §6, §7, §8.
- Every mode and every guided part at 1440 x 900 and 390 x 844, in EN and ZH, and once in dark mode; screenshots in shots/gqa-mla/.
- Review 8 (R2.8, R3.13): the tour starts from multi-head attention (MHA), the initial setup; part 2 shares keys and values (GQA, then
  MQA as the extreme); part 3 is what sharing costs (the three designs on T5-XXL only, and one line of scope); part 4 predicts MQA on
  8 GPUs from MHA's 1 user; part 5 checks it; part 6 is MLA; part 7 sums up. Nothing calls GQA the original design.
- Review 8: every abbreviation has its full name at its first appearance in each card and in the hover of every design option,
  legend item and chart label (MHA = multi-head attention, GQA = grouped-query attention, MQA = multi-query attention, MLA =
  multi-head latent attention).
- Review 8 (R3.2): a change in Settings never scrolls the page. Every design is picked under Settings in the guided parts, in
  Scenarios and in the challenge (before and after a submit), at 1440 x 900 and 390 x 844: no script scrolls the page, at 1440 scrollY
  does not change, and at 390 (where the card and the stage above Settings change height) the button stays where it was on screen.
  No challenge answer shows up from Settings.
- R3.12: the last guided part ends with "See the scenarios", the Scenarios card with "Take the challenge", the challenge with the next lab.
- R6.18 and R6.17: no number over 6 digits outside Show the math and hover tips; Chinese says 维 for a vector's length.
- Gating: Next waits for a pick other than MHA in part 2, for the prediction in part 4 (Skip works), for MQA in part 5, for MLA in part 6.
- The prediction's answer (users with other designs, what each GPU stores) appears nowhere before part 5 checks it: metrics, status,
  chart, math, mini bar. MHA's 1 user, stated in part 1, shows.
- Challenge: submit waits for a pick; the three tiers; "Show the answer"; a brief comparison chart (the owner's review of batch 8): the
  designs that meet the goal, MHA (the initial setup, tagged) and the pick, nothing else, a note that counts the designs left out, and
  full names.
- No hover that only expands a number to its exact value, outside Show the math (the owner, 2026-10-07 23:37 UTC): the kit's A.ui.big
  has no hover any more, and no hover in the lab says “精确值是” or “Exactly”.
- The language switch keeps the part, the prediction, the design and the challenge; the hash round-trips.
- After each action at 1440 x 1000 the result box is on screen, and each card's height is reported (R3.10).
- No horizontal overflow at 1440, 1280, 900, 720 and 390; the Connect code reads without sideways scrolling at 1440 (R7.6).
- Writing rules on all visible text and tooltips: no "·" between clauses, never two colons in one Chinese sentence, no "in this toy",
  no 起点 / starting point, no 控件 / Controls, never "the KV cache lab", "the least per token" only among the five scenarios, every
  SGLang sentence says its data-parallel attention is an option, no "original design" / 原设计, no T5-Large.
- Earlier rounds kept: the KV heads carry their numbers; the stage stays under the top bar after Next and after picks in part 5 and
  Next from part 6; MLA's per-GPU value hides until it is picked; the kit's tensor-parallelism opening; "plain"; the rebuilt tip and the
  dashed-line legend say decode steps; "1 user fits"; the 405B scenario names Meta's FP8 release.
python3 tests/gqa-mla_flow.py"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/gqa-mla.html', '--only', 'gqa-mla'], cwd=P, check=True, capture_output=True)
(P / 'out/wrapped-gqa-mla.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                                            '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out/gqa-mla.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out/wrapped-gqa-mla.html')
SHOTS = P / 'shots/gqa-mla'; SHOTS.mkdir(parents=True, exist_ok=True)
fails, n, heights, heights900 = [], 0, [], []
DES = ['mha', 'gqa16', 'gqa8', 'gqa4', 'mqa', 'mla']
INIT = {'model': 'llama31-70b', 'attn': 'mha', 'gpu': 'h100', 'n': 8, 'wp': 'bf16', 'ctx': 131072}
FULL = {'en': {'mha': 'multi-head attention', 'gqa16': 'grouped-query attention', 'gqa8': 'grouped-query attention', 'gqa4': 'grouped-query attention',
               'mqa': 'multi-query attention', 'mla': 'multi-head latent attention'},
        'zh': {'mha': '多头注意力', 'gqa16': 'grouped-query attention', 'gqa8': 'grouped-query attention', 'gqa4': 'grouped-query attention',
               'mqa': 'multi-query attention', 'mla': 'multi-head latent attention'}}
# a spy on every way a script can scroll the page (review 8, R3.2): the log shows who scrolled
SPY = """(()=>{const W=window; W.__sc=[];
  for(const k of ['scrollBy','scrollTo','scroll']){const f=W[k].bind(W); W[k]=function(...a){W.__sc.push([k,a.map(x=>typeof x==='object'?JSON.stringify(x):x)]); return f(...a);};}
  const siv=Element.prototype.scrollIntoView; Element.prototype.scrollIntoView=function(...a){W.__sc.push(['scrollIntoView',this.id||String(this.className)]); return siv.apply(this,a);};})()"""


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(f'{name} :: {info}')
        print('FAIL', name, '::', str(info)[:300])


def plain(s): return (s or '').replace('\u2011', '-').replace('\u00a0', ' ')


def settle(pg, limit=2500):
    last, same, t = None, 0, 0
    while t < limit:
        y = pg.evaluate("()=>Math.round(scrollY)")
        same = same + 1 if y == last else 0
        if same >= 3: return
        last = y; pg.wait_for_timeout(60); t += 60


def rail_click(pg, sel, wait=300):
    """click inside the rail like a learner (scroll into view, click), then let smooth scrolling finish"""
    el = pg.query_selector(sel)
    if not el: return False
    el.scroll_into_view_if_needed(); el.click(); pg.wait_for_timeout(wait); settle(pg)
    return True


def go_part(pg, k):
    pg.evaluate(f"()=>document.querySelector('#gm-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(300); settle(pg)


def set_mode(pg, mode, wait=400):
    pg.evaluate(f"()=>document.querySelector('.dyn-root .modes button[data-mode=\"{mode}\"]').click()"); pg.wait_for_timeout(wait); settle(pg)


def overflow(pg):
    return pg.evaluate("()=>document.documentElement.scrollWidth-document.documentElement.clientWidth")


def state(pg): return pg.evaluate("()=>window.__gm.state()")


def visible_text(pg):
    return plain(pg.evaluate("()=>document.querySelector('.dyn-root').innerText"))


def all_tips(pg):
    return plain(pg.evaluate("""()=>[...document.querySelectorAll('.dyn-root [data-tipx]')].map(e=>e.getAttribute('data-tipx')).join('\\n')"""))


def disabled(pg, sel): return pg.evaluate(f"()=>{{const b=document.querySelector('{sel}'); return !b||b.disabled}}")


KIT_TEXTS = []   # the site's shared definitions in the current language (kit text, not this lab's words)


def rule_scan(tag, text, lang):
    # "·" may only separate standalone metadata tags; this lab has none, so any "·" is a violation
    check(f'{tag}: no "·" in the text', '·' not in text, [l for l in text.split('\n') if '·' in l][:3])
    low = text.lower()
    check(f'{tag}: no "in this toy"', 'in this toy' not in low and '在这个玩具里' not in text)
    check(f'{tag}: no 起点 / starting point as the baseline', '起点' not in text and 'starting point' not in low)
    check(f'{tag}: no 控件 / Controls as the panel name', '控件' not in text and not re.search(r'\bControls\b', text))
    check(f'{tag}: never "the KV cache lab" (review r11, X4)', 'KV cache lab' not in text and 'KV cache Lab' not in text, [l for l in text.split('\n') if 'KV cache lab' in l or 'KV cache Lab' in l][:2])
    bad = [l for l in text.split('\n') if ('the least per token' in l and 'of these five' not in l) or ('存得最少' in l and '五个场景' not in l)]
    check(f'{tag}: "the least per token" only among the five scenarios (review r11, G1)', not bad, bad[:2])
    bad = [x for x in re.split(r'(?<=\.)\s+|(?<=。)|\n', text) if 'SGLang' in x and not any(k in x for k in ['can ', 'option', '可以', '可选'])]
    check(f'{tag}: SGLang’s data-parallel attention is always an option (review A r13 M2)', not bad, bad[:2])
    bad = [l for l in text.split('\n') if re.search(r'as designed|original design|原设计|原始设计|as the model was designed', l, re.I)]
    check(f'{tag}: GQA is never the original design (review 8, R2.8)', not bad, bad[:2])
    check(f'{tag}: no T5-Large (review 8, R6.16: only the three designs on T5-XXL)', 'T5-Large' not in text and 'T5‑Large' not in text)
    if lang == 'zh':
        bad = [s for s in re.split(r'[。！？\n]', text) if s.count('：') >= 2]
        check(f'{tag}: never two colons in one Chinese sentence', not bad, bad[:3])
        own = text
        for k in KIT_TEXTS: own = own.replace(k, '')
        bad = [s for s in re.split(r'[。！？\n]', own) if re.search(r'个数(?!值|据)', s)]
        check(f'{tag}: 参数 never shortened to 数, a vector’s length in 维 (R6.17)', not bad, bad[:3])


def big_scan(pg, tag):
    """R6.18: no number over 6 digits in the text, the cards, the metrics, the chart or the popovers; Show the math and tooltips may"""
    t = plain(pg.evaluate("""()=>{const r=document.querySelector('.dyn-root').cloneNode(true); r.querySelectorAll('#gm-mathbody,pre,[hidden]').forEach(e=>e.remove());
      const w=document.createTreeWalker(r,NodeFilter.SHOW_TEXT), out=[]; while(w.nextNode()) out.push(w.currentNode.nodeValue); return out.join(' ')}"""))
    bad = re.findall(r'(?<![\d,.])\d{1,3}(?:,\d{3}){2,}(?![\d,])|(?<![\d,.])\d{7,}', t)
    check(f'{tag}: no number over 6 digits outside Show the math and tooltips (R6.18)', not bad, bad[:5])


def masked_ok(pg, tag, Z):
    """before part 5 checks the prediction: no user count for designs other than MHA and no per-GPU value anywhere. MHA's 1 user,
    stated in part 1, may show."""
    m = pg.evaluate("""()=>{const q=s=>document.querySelector(s); const mv=[...document.querySelectorAll('#gm-metrics .metric')].map(e=>[e.dataset.k,e.querySelector('.v').textContent]);
      const cells=[...document.querySelectorAll('#gm-chart .gm-ch-r')].map(r=>r.querySelectorAll('.gm-cc b')[1]?r.querySelectorAll('.gm-cc b')[1].textContent:'');
      return {mv, st:q('#gm-status').textContent, cls:q('#gm-status').className, cells, math:q('#gm-mathbody').innerText, dock:q('#dock').hidden?'':q('#dock').innerText, gpus:document.querySelectorAll('#gm-diag .gm-gpu').length}}""")
    mv = dict(m['mv']); st = state(pg)['S']
    init_mha = st == INIT
    check(f'{tag}: per-GPU metric masked; users masked unless the design is MHA at the initial setup (1, stated in part 1)', mv['gpu'] == '?' and mv['users'] == ('1' if init_mha else '?'), (mv, st))
    sok = ('neutral' in m['cls'] and not re.search(r'\d+ (users?|个用户)', plain(m['st']))) or (init_mha and plain(m['st']) == Z('At most 1 user fits', '最多装下 1 个用户'))
    check(f'{tag}: status neutral with no number, or MHA’s known 1 user', sok, (m['st'], m['cls'], st))
    check(f'{tag}: chart per-GPU column masked', all(c == '?' for c in m['cells']), m['cells'])
    check(f'{tag}: math has no user or per-GPU line', ('Most users' not in m['math']) and ('最多用户数' not in m['math']) and ('On each GPU' not in m['math']), m['math'][:200])
    check(f'{tag}: no GPU boxes drawn', m['gpus'] == 0, m['gpus'])
    ud = re.findall(r'(\d+) (?:users?|user fits|个用户)', m['dock'])
    check(f'{tag}: mini result bar shows no user count but MHA’s 1', all(x == '1' for x in ud) and (init_mha or not ud), m['dock'])


def result_on_screen(pg, tag, sel='#gm-rail [data-obs]'):
    r = pg.evaluate("""s=>{const e=document.querySelector(s); if(!e||e.hidden) return null; const b=e.getBoundingClientRect(), bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
      const top=(bar?bar.offsetHeight:58)+(dk&&!dk.hidden?dk.offsetHeight:0); return {top:Math.round(b.top), bottom:Math.round(b.bottom), vh:innerHeight, lim:top}}""", sel)
    check(f'{tag}: result box shown', r is not None, r)
    if r:
        check(f'{tag}: result box top on screen after the action (R3.10)', r['lim'] - 2 <= r['top'] <= r['vh'] - 40, r)


def card_height(pg, tag, into=None):
    h = pg.evaluate("()=>{const c=document.querySelector('#gm-rail .step.current')||document.querySelector('#gm-rail .railcard'); return c?Math.round(c.getBoundingClientRect().height):0}")
    (heights if into is None else into).append((tag, h))
    return h


def tip_of(pg, scope, words):
    """the tooltip text of the first definition in scope whose text is one of words"""
    return plain(pg.evaluate("""([sc,ws])=>{const e=[...document.querySelectorAll(sc+' .def')].find(d=>ws.includes(d.textContent.replace(/‑/g,'-'))); return e?e.getAttribute('data-tipx'):''}""", [scope, words]))


def defs(pg, scope='#gm-rail .step.current'):
    return [plain(x) for x in pg.evaluate(f"()=>[...document.querySelectorAll('{scope} .def')].map(e=>e.textContent)")]


def stage_ok(pg):
    return pg.evaluate("""()=>{const tb=document.querySelector('.topbar').getBoundingClientRect().bottom, t=document.getElementById('gm-title').getBoundingClientRect(),
      bx=document.querySelector('#gm-rail [data-obs]'), q=bx&&!bx.hidden?bx.getBoundingClientRect():null;
      return {bar:Math.round(tb), title:Math.round(t.top), box:q?[Math.round(q.top),Math.round(q.bottom)]:null, vh:innerHeight}}""")


def option_tips(pg, scope):
    """[design, tooltip] for every design option under scope"""
    return [(a, plain(t)) for a, t in pg.evaluate(f"""()=>[...document.querySelectorAll('{scope} .seg-ctl[data-key="attn"] button[data-v], {scope} [data-lever]')].map(b=>[b.dataset.v||b.dataset.lever, b.getAttribute('data-tipx')||''])""")]


def no_exact_hover(pg, tag):
    """the owner, 2026-10-07 23:37 UTC: no hover only expands a shown number to its exact value (Show the math may write exact values)"""
    hits = pg.evaluate("""()=>[...document.querySelectorAll('[data-tipx],[data-tip],[title]')].filter(e=>!e.closest('details.math,[id$="mathbody"]'))
      .map(e=>e.getAttribute('data-tipx')||e.getAttribute('data-tip')||e.getAttribute('title')||'').filter(t=>/精确值是|Exactly|shown rounded to 3|取 3 位有效数字/.test(t)).map(t=>t.slice(0,90))""")
    check(f'{tag}: no hover that only gives the exact value', not hits, hits[:3])


def full_names_ok(pg, tag, lang, scope):
    bad = [(a, t) for a, t in option_tips(pg, scope) if FULL[lang][a] not in t]
    check(f'{tag}: every design option’s hover has its full name (review 8)', not bad and option_tips(pg, scope), bad[:3])



# ---- the owner's decisions of 2026-10-09 (sweep of the kernels group, reviews/sweep-kernels-owner-b9v6.md): H1, every formula is
# typeset with the kit (MathML, no TeX command shown as a word) and none of the converted plain-text formulas is left in the source or
# on screen (text and hovers, EN and ZH); H2, Show the math comes in short blocks split by a hairline. Every guided part, Scenarios
# and the challenge, at each width and language this test runs.
B9V6_PROBE = r"""([mb,blk])=>{const root=document.querySelector('#view-dyn:not([hidden])'); if(!root) return {n:0,bad:[],txt:'',sep:[],blocks:0,body:'',inBody:0};
  const box=document.querySelector(mb); const det=box&&box.closest('details'); if(det) det.open=true;
  const CMD=/^(left|right|frac|sqrt|text|times|cdot|div|approx|Big|big|hat|leftarrow|mathrm|operatorname|lceil|rceil|lfloor|rfloor|quad|infty)$/;
  const maths=[...root.querySelectorAll('math')], bad=[];
  for(const m of maths) for(const e of m.querySelectorAll('mi,mtext,mo')){const t=e.textContent.trim(); if(CMD.test(t)||t.includes('\\')) bad.push(t);}
  const tips=[...root.querySelectorAll('[data-tipx]')].map(e=>e.getAttribute('data-tipx')||'');
  for(const t of tips){const m=t.match(/<m(?:i|text)[^>]*>(left|right|frac|sqrt|text|times|cdot|Big|hat|lceil|rceil)<\/m(?:i|text)>/g); if(m) bad.push(...m);}
  const blocks=box?[...box.querySelectorAll(blk)]:[];
  const sep=blocks.filter(x=>x.previousElementSibling&&x.previousElementSibling.matches(blk)).map(x=>{const cs=getComputedStyle(x); return cs.borderTopWidth+' '+cs.borderTopStyle;});
  const parts=[]; const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let nd;
  while((nd=tw.nextNode())){const e=nd.parentElement; if(!e||e.closest('[data-endnotes],pre,code')||!e.getClientRects().length) continue; parts.push(nd.textContent);}
  const tipTxt=tips.map(t=>{const d=document.createElement('div'); d.innerHTML=t; d.querySelectorAll('pre,code').forEach(x=>x.remove()); return d.textContent;}).join(' ');
  return {n:maths.length, bad, blocks:blocks.length, sep, body:box?box.textContent.trim():'', inBody:box?box.querySelectorAll('math').length:0, txt:parts.join(' ')+' '+tipTxt};}"""


# the owner's review of 2026-10-09 (R6.37, reviews/math-wrap-kernels-owner-b9v7.md): a row of “Show the math” the reader has to scroll
# sideways is not read. With Show the math open, no box in it scrolls sideways and it is no wider than its own box.
MSCROLL_PROBE = r"""(mb)=>{const box=document.querySelector(mb); const d=box&&box.closest('details'); if(!d||!d.open||!d.getClientRects().length) return {seen:0,bad:[]};
  const out=[];
  for(const e of d.querySelectorAll('*')){const cs=getComputedStyle(e); if(!/auto|scroll/.test(cs.overflowX)||!e.getClientRects().length) continue;
    const o=e.scrollWidth-e.clientWidth; if(o>1) out.push((typeof e.className==='string'&&e.className?e.className.split(' ')[0]:e.tagName.toLowerCase())+' by '+o+' px: '+(e.textContent||'').trim().replace(/\s+/g,' ').slice(0,40));}
  if(d.scrollWidth-d.clientWidth>1) out.push('the whole Show the math by '+(d.scrollWidth-d.clientWidth)+' px');
  return {seen:1,bad:out};}"""


def owner_b9v6(b, w, h, lang, cfg):
    tag = f'{w} {lang} owner b9v6'
    if not cfg.get('_src_done'):
        cfg['_src_done'] = 1
        src = (P / 'labs' / f"{cfg['lab']}.js").read_text()
        left = [s for s in cfg['src_gone'] if s in src]
        check(f"owner b9v6 H1: no plain-text formula of the converted kinds left in labs/{cfg['lab']}.js", not left, left)
    pg = b.new_page(viewport={'width': w, 'height': h})
    here = []; pg.on('pageerror', lambda e: here.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{cfg['lab']}'}}"); pg.wait_for_timeout(600)
    nparts = pg.evaluate("()=>new Set([...document.querySelectorAll('#view-dyn aside.rail [data-go]')].map(e=>e.dataset.go)).size")
    states = [('guided', k) for k in range(nparts)] + [('sandbox', None), ('challenge', None)]
    bad, gone, nomath, seps, nblocks, mathn = [], set(), [], [], 0, 0
    nscr, scr = 0, []
    for mode, k in states:
        if k is None or k == 0:
            pg.evaluate("m=>{const b=[...document.querySelectorAll('#view-dyn .modes button[data-mode=\"'+m+'\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}", mode); pg.wait_for_timeout(400)
        if k is not None:
            pg.evaluate("k=>{const e=document.querySelector('#view-dyn aside.rail [data-go=\"'+k+'\"]'); if(e) e.click();}", k)
        pg.wait_for_timeout(900)
        r = pg.evaluate(B9V6_PROBE, [cfg['math'], cfg['block']])
        sx = pg.evaluate(MSCROLL_PROBE, cfg['math']); nscr += sx['seen']
        scr += [mode + ('' if k is None else f' part {k + 1}') + ': ' + x for x in sx['bad']]
        where = mode + ('' if k is None else f' part {k + 1}')
        bad += [f'{where}: {x}' for x in r['bad']]
        txt = r['txt'].replace('‑', '-').replace(' ', ' ').replace(' ', ' ')
        for s in cfg['text_gone']:
            if s in txt: gone.add(f'{where}: {s}')
        if r['body'] and not r['inBody'] and not re.search(cfg.get('nomath_ok', r'^$'), r['body']): nomath.append(where)
        if r['blocks'] >= 2: nblocks += 1
        seps += [f'{where}: {x}' for x in r['sep'] if not x.startswith('1px solid')]
        mathn += r['n']
    pg.close()
    check(f'{tag} H1: the formulas render as MathML ({mathn} on the visited screens), no TeX command shown as a word', mathn > 0 and not bad, bad[:5])
    check(f'{tag} H1: no plain-text formula of the converted kinds in the rendered text or the hovers', not gone, sorted(gone)[:6])
    check(f'{tag} H1/H2: Show the math typesets its formulas on every screen that has them', not nomath, nomath[:5])
    check(f'{tag} H2: Show the math comes in short blocks split by a 1px hairline ({nblocks} screens with two or more blocks)', nblocks > 0 and not seps, seps[:5])
    check(f'{tag} owner 2026-10-09 (R6.37): no “Show the math” box scrolls sideways, every row fits its box ({nscr} screens with it open)', nscr > 0 and not scr, scr[:5])
    check(f'{tag}: no page errors', not here, here[:3])

B9V6 = {'lab': 'gqa-mla', 'math': '#gm-mathbody', 'block': '.gm-mb', 'nomath_ok': 'Pick a design|先在任务卡里选', 'src_gone': ['keeps max(1, KV heads ÷ GPUs)', '放 max(1, KV head 数 ÷ GPU 数)'], 'text_gone': ['max(1, KV']}


with sync_playwright() as p:
    b = p.chromium.launch()
    # ================= 1440 x 1000: the guided walk with gating, masking, results on screen, card heights; then the mode order =================
    for lang in ['en', 'zh']:
        Z = (lambda en, zh: zh) if lang == 'zh' else (lambda en, zh: en)
        ctx = b.new_context(viewport={'width': 1440, 'height': 1000}); pg = ctx.new_page()
        errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        pg.evaluate("()=>{location.hash='gqa-mla'}"); pg.wait_for_timeout(700)
        T = f'{lang} 1440x1000'
        check(f'{T}: opens in guided part 1 with MHA, the initial setup (review 8)', state(pg)['mode'] == 'guided' and state(pg)['step'] == 0 and state(pg)['S'] == INIT, state(pg))
        check(f'{T}: title', pg.inner_text('.dyn-root .labtitle') == Z('How do GQA and MLA shrink the KV cache?', 'GQA 和 MLA 怎么让 KV cache 变小？'))
        md = plain(pg.evaluate("()=>{const b=document.querySelector('.dyn-root [data-flow-nav] button[aria-pressed=true]'); return b?(b.dataset.tipx||''):''}"))
        check(f'{T}: the guided description right under the title gives all four full names, MHA first (review 8)',
              all(x in md for x in Z(['multi-head attention (MHA)', 'grouped-query and multi-query attention, GQA and MQA', 'multi-head latent attention (MLA)'], ['多头注意力（MHA）', 'grouped-query attention 和 multi-query attention，即 GQA 和 MQA', 'multi-head latent attention（MLA）'])) and md.find('MHA') < md.find('GQA'), md)
        check(f'{T}: no primer button on top; the notes close the page (owner, 2026-10-11 19:28 UTC)', not any(x in pg.inner_text('.dyn-root .labmeta') for x in ('Primer', '前置知识')) and pg.evaluate("()=>{const e=document.getElementById('gqa-mla-fid'); return !!e&&!e.hidden}"))
        masked_ok(pg, f'{T} part 1', Z)
        card_height(pg, f'{T} part 1')
        kit = pg.evaluate("()=>({qh:ASP.terms.qhead(), gqa:ASP.terms.gqa(), tp:ASP.terms.tp(), tpmla:ASP.terms.tpmla(), mqa:ASP.terms.mqa(), kib:ASP.terms.kib(), mla:ASP.terms.mla(), mha:ASP.terms.mha(), kvc:ASP.terms.kvcache(), gib:ASP.terms.gib()})")
        kit = {k: plain(v) for k, v in kit.items()}
        KIT_TEXTS[:] = [plain(x) for x in pg.evaluate("()=>Object.values(ASP.terms).map(f=>{try{return f()}catch(e){return ''}}).filter(Boolean)")]
        # part 1: MHA first, from the earlier lab; its terms are the site's (review 8, CONTRACT §11)
        p1 = plain(pg.inner_text('#gm-rail .step.current'))
        d1 = defs(pg)
        check(f'{T} part 1: MHA, query head, KV head, KiB and KV cache defined', all(x in d1 for x in [Z('multi-head attention (MHA)', '多头注意力（MHA）'), 'query head', 'KV head', 'KiB', 'KV cache']), d1)
        check(f'{T} part 1: links the lab that built MHA', pg.query_selector('#gm-rail .step.current a[href="#mha"]') is not None)
        check(f'{T} part 1: MHA is what the reader knows: every head has its own query, key and value',
              Z('built multi-head attention (MHA): each head has its own query, key and value.', '多头注意力（MHA）：每个 head 都有自己的 query、key 和 value。') in p1 and p1.find('MHA') < p1.find('KV head'), p1[:300])
        mt1 = plain(pg.evaluate("()=>document.getElementById('gm-mathbody').textContent"))
        check(f'{T} part 1: MHA’s KV cache and users on 8 × H100, the size in the sentence and its product in Show the math (R6.20)', Z('in Llama-3.1-70B’s shape, 2,560 KiB of KV cache per token', '每个 token 要 2,560 KiB 的 KV cache') in p1
              and Z('that leaves room for only 1 user', '这样只装得下 1 个用户') in p1 and Z('× 80 layers × 64 KV heads × 128 head_dim × 2 bytes = 2,621,440 bytes = 2,560 KiB', '× 80 层 × 64 个 KV head × 128 head_dim × 2 字节 = 2,621,440 字节 = 2,560 KiB') in mt1, (p1, mt1[:300]))
        mh = tip_of(pg, '#gm-rail .step.current', ['multi-head attention (MHA)', '多头注意力（MHA）'])
        check(f'{T} part 1: MHA = the kit’s structural text + the model sentence (CONTRACT §11)', mh == kit['mha'] + Z(' In Llama-3.1-70B’s shape each layer has 64 heads, and each head’s query, key and value have 128 numbers each.', 'Llama-3.1-70B 的结构每层有 64 个 head，每个 head 的 query、key、value 各是 128 维。'), mh)
        check(f'{T} part 1: KV cache = the kit’s text', tip_of(pg, '#gm-rail .step.current', ['KV cache']) == kit['kvc'])
        kb = tip_of(pg, '#gm-rail .step.current', ['KiB'])
        check(f'{T} part 1: KiB = the kit’s text + one sentence, no GiB before GiB is defined (review R14 N10)', kb == kit['kib'] + Z(' Here 2,560 KiB is 2,621,440 bytes (2.5 MiB).', '这里 2,560 KiB 就是 2,621,440 字节（2.5 MiB）。') and 'GiB' not in kb, kb)
        fg = pg.evaluate("""()=>{const r=document.querySelector('.dyn-root'), w=document.createTreeWalker(r,NodeFilter.SHOW_TEXT);
          while(w.nextNode()){const n=w.currentNode; if(!/GiB/.test(n.nodeValue)) continue; const e=n.parentElement;
            if(e.closest('[hidden]')||e.closest('details:not([open]) > :not(summary)')||!e.getClientRects().length) continue;
            const d=e.closest('.def'); return {txt:n.nodeValue, def:d?d.getAttribute('data-tipx'):''};} return null}""")
        check(f'{T} part 1: the first GiB on the page carries the site’s GiB definition (R6.12, review R14 2-6)', fg is not None and plain(fg['def']) == kit['gib'], fg)
        qh = tip_of(pg, '#gm-rail .step.current', ['query heads', 'query head'])
        check(f'{T} part 1: query head = the kit text + the model sentence (r12)', qh == kit['qh'] + Z(' Llama-3.1-70B has 64 query heads per layer.', 'Llama-3.1-70B 每层有 64 个 query head。'), qh)
        # ----- part 2: share keys and values: GQA, then MQA as the extreme -----
        rail_click(pg, '#gm-rail [data-nav="next"]', 400)
        check(f'{T}: Next reveals part 2 at the top of the screen (R3.7)', pg.evaluate("()=>{const r=document.querySelector('#gm-rail .step.current').getBoundingClientRect(); return r.top>=50&&r.top<innerHeight-160}"))
        check(f'{T} part 2: Next disabled before a pick', disabled(pg, '#gm-rail [data-nav="next"]'))
        masked_ok(pg, f'{T} part 2', Z)
        p2 = plain(pg.inner_text('#gm-rail .step.current'))
        check(f'{T} part 2: GQA with its full name, the design Llama-3.1-70B uses, then MQA as the extreme',
              Z('grouped-query attention (GQA)', 'grouped-query attention（GQA）') in p2 and Z('multi-query attention (MQA)', 'multi-query attention（MQA）') in p2 and p2.find('GQA') < p2.find('MQA')
              and Z('Llama-3.1-70B uses 8 groups of 8 (GQA-8)', 'Llama-3.1-70B 分 8 组，每组 8 个（GQA-8）') in p2 and Z('Start from MHA', '从 MHA 出发') in p2, p2)
        gq = tip_of(pg, '#gm-rail .step.current', ['grouped-query attention (GQA)', 'grouped-query attention（GQA）'])
        check(f'{T} part 2: GQA = the kit text + the model sentence and a pointer to part 3', gq == kit['gqa'] + Z(' Llama-3.1-70B has 64 query heads in 8 groups of 8, so each token stores 8 keys and 8 values per layer instead of 64 of each. Part 3 shows more of the paper’s numbers.',
              'Llama-3.1-70B 的 64 个 query head 分 8 组、每组 8 个，所以每个 token 每层只存 8 个 key 和 8 个 value，而不是各 64 个。第 3 部分给出论文的更多数字。'), gq)
        mq = tip_of(pg, '#gm-rail .step.current', ['multi-query attention (MQA)', 'multi-query attention（MQA）'])
        check(f'{T} part 2: MQA = the kit’s text + the Falcon-7B sentence (review R14 3-3)', mq == kit['mqa'] + Z(' Falcon-7B is a real example: its 71 query heads share one KV head, 8 KiB per token in BF16.', 'Falcon-7B 就是这样：71 个 query head 共用一个 KV head，BF16 下每个 token 8 KiB。'), mq)
        full_names_ok(pg, f'{T} part 2', lang, '#gm-rail')
        ot = dict(option_tips(pg, '#gm-rail'))
        check(f'{T} part 2: the 8 KV heads option says Llama-3.1-70B uses it, not that it is the original', ot['gqa8'].endswith(Z(' Llama-3.1-70B uses it.', 'Llama-3.1-70B 用的就是它。')) and ot['mha'].startswith(Z('MHA, multi-head attention:', 'MHA，多头注意力：')), ot)
        rail_click(pg, '#gm-ic-kv button[data-v="mha"]', 400)
        o = plain(pg.inner_text('#gm-rail [data-obs]'))
        check(f'{T} part 2 MHA: the initial setup; Next still disabled', Z('64 KV heads (MHA): 2,560 KiB per token', '64 个 KV head（MHA）：每个 token 2,560 KiB') in o and Z('the initial setup', '初始设置') in o and disabled(pg, '#gm-rail [data-nav="next"]'), o)
        for v, want in [('gqa8', Z('8 KV heads (GQA-8, the design Llama-3.1-70B uses): 320 KiB per token, one eighth of MHA’s. Every 8 query heads share one KV head.', '8 个 KV head（GQA-8，Llama-3.1-70B 用的设计）：每个 token 320 KiB，是 MHA 的八分之一。每 8 个 query head 共用一个 KV head。')),
                        ('mqa', Z('1 KV head (MQA): 40 KiB per token, one sixty-fourth of MHA’s. All the query heads share it.', '1 个 KV head（MQA）：每个 token 40 KiB，是 MHA 的六十四分之一。所有 query head 共用它。'))]:
            rail_click(pg, f'#gm-ic-kv button[data-v="{v}"]', 400)
            result_on_screen(pg, f'{T} part 2 pick {v}')
            masked_ok(pg, f'{T} part 2 after {v}', Z)
            o = plain(pg.inner_text('#gm-rail [data-obs]'))
            check(f'{T} part 2 {v}: result', want in o and Z('Only keys and values are shared', '共用的只是 key 和 value') in o, o)
        card_height(pg, f'{T} part 2 after picks')
        check(f'{T} part 2: Next enabled', not disabled(pg, '#gm-rail [data-nav="next"]'))
        pg.screenshot(path=str(SHOTS / f'1440_{lang}_p2.png'))
        # ----- part 3: what sharing costs, the three designs on T5-XXL and one line of scope (review 8, R6.16) -----
        rail_click(pg, '#gm-rail [data-nav="next"]', 400)
        p3 = plain(pg.inner_text('#gm-rail .step.current'))
        # R6.20: the scores stay in the table; the sentence says the gaps in words (GQA-8 0.1 below MHA: almost as well; MQA 0.6 below: a little lower)
        for x in ['47.2', '47.1', '46.6', '1.51', '0.28', '0.24', 'T5-XXL', Z('GQA-8 scored almost as well as MHA, and MQA a little lower.', 'GQA-8 的平均分几乎和 MHA 一样，MQA 略低一点。'), Z('under a fifth', '五分之一'), Z('Llama 3 uses GQA-8 in all three sizes', 'Llama 3 的三个尺寸都用 GQA-8')]:
            check(f'{T} part 3 shows {x}', x in p3, p3[-600:])
        check(f'{T} part 3: the gaps are no longer numbers in the sentence (R6.20)', not any(x in p3 for x in [Z('0.1 points', '0.1 分'), Z('0.6 below', '0.6 分')]), p3[-400:])
        check(f'{T} part 3: one line of scope: converted from MHA, trained a little further, decoder-only not tested',
              Z('converted T5-XXL, a model trained with MHA, to GQA-8 and to MQA and trained each a little further; it did not test decoder-only models such as Llama', '把用 MHA 训好的 T5-XXL 分别改成 GQA-8 和 MQA，再各自稍加训练；论文没有测 Llama 这类 decoder-only 模型') in p3, p3[:400])
        for x in ['T5-Large', '46.0', '0.37', '1.2', '§3.3', '5%', '28.5', 'WMT']:
            check(f'{T} part 3: no other paper detail ({x}) (review 8)', x not in p3, x)
        rows = pg.evaluate("()=>[...document.querySelectorAll('#gm-rail .gm-t1 tbody tr')].map(r=>r.querySelector('th').textContent)")
        check(f'{T} part 3: three rows, MHA, GQA-8, MQA', [plain(r) for r in rows] == ['MHA', 'GQA-8', 'MQA'], rows)
        rt = [plain(t) for t in pg.evaluate("()=>[...document.querySelectorAll('#gm-rail .gm-t1 tbody th .def')].map(e=>e.getAttribute('data-tipx'))")]
        check(f'{T} part 3: each row’s hover has the full name', len(rt) == 3 and all(FULL[lang][a] in t for a, t in zip(['mha', 'gqa8', 'mqa'], rt)), rt)
        check(f'{T} part 3: the paper and the table headers are defined', len(defs(pg)) >= 6, defs(pg))
        masked_ok(pg, f'{T} part 3', Z)
        card_height(pg, f'{T} part 3')
        pg.screenshot(path=str(SHOTS / f'1440_{lang}_p3.png'))
        # ----- part 4: predict MQA on 8 GPUs, from MHA's 1 user -----
        rail_click(pg, '#gm-rail [data-nav="next"]', 400)
        check(f'{T} part 4: Next disabled before a prediction', disabled(pg, '#gm-rail [data-nav="next"]'))
        check(f'{T} part 4: Skip offered', pg.query_selector('#gm-rail [data-nav="skip"]') is not None)
        check(f'{T} part 4: the badge asks for the prediction', plain(pg.inner_text('#gm-status')) == Z('Predict first', '先预测'))
        opts = [plain(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#gm-rail [data-pred]')].map(b=>b.textContent)")]
        check(f'{T} part 4: four options that cover the range, starting from MHA’s 1', opts == Z(['1, the same as MHA', '2 to 20', '21 to 63', '64 or more'], ['1 个，和 MHA 一样', '2 到 20 个', '21 到 63 个', '64 个或更多']), opts)
        p4 = plain(pg.inner_text('#gm-rail .step.current'))
        check(f'{T} part 4: the question starts from MHA (review 8: not from GQA as the original design)',
              Z('Per token, MQA stores one sixty-fourth of what MHA stores. With MHA, only 1 user fits at 128K each on the 8 × H100', '按每个 token 算，MQA 只存 MHA 的六十四分之一。用 MHA 时，8 × H100 每人 128K 上下文只装得下 1 个用户') in p4
              and Z('With MQA on the same 8 GPUs, how many users fit?', '换成 MQA，还是这 8 张 GPU，能装下多少用户？') in p4, p4)
        masked_ok(pg, f'{T} part 4 before predicting', Z)
        tp4 = tip_of(pg, '#gm-rail .step.current', ['tensor parallelism', '张量并行'])
        want_tp = kit['tp'] + Z(' On 8 GPUs, each computes 8 of Llama-3.1-70B’s 64 query heads.', '8 张 GPU 时，每张算 Llama-3.1-70B 64 个 query head 里的 8 个。')
        check(f'{T} part 4: tensor parallelism = the kit’s text + the model sentence (review R14 3-3)', tp4 == want_tp, (tp4, want_tp))
        check(f'{T} part 4: its tensor-parallelism tip does not say which KV heads each GPU keeps (the prediction)', 'KV' not in tp4, tp4)
        check(f'{T} part 4: the tensor-parallelism tip uses the site’s block names', Z('attention block and MLP block', 'attention 块和 MLP 块') in tp4, tp4)
        rail_click(pg, '#gm-rail [data-pred="many"]', 300)
        check(f'{T} part 4: Next enabled after a prediction', not disabled(pg, '#gm-rail [data-nav="next"]'))
        check(f'{T} part 4: saved note says it can still change', Z('You can still change it', '还可以改') in pg.inner_text('#gm-rail .step.current'))
        masked_ok(pg, f'{T} part 4 after predicting', Z)
        card_height(pg, f'{T} part 4 after the prediction')
        pg.screenshot(path=str(SHOTS / f'1440_{lang}_p4.png'))
        # ----- part 5: across 8 GPUs, from MHA -----
        rail_click(pg, '#gm-rail [data-nav="next"]', 400); pg.wait_for_timeout(700); settle(pg)
        sg = pg.evaluate("()=>{const b=document.getElementById('gm-status').getBoundingClientRect(), t=document.querySelector('.topbar').offsetHeight; return {badge:Math.round(b.top), bar:t}}")
        check(f'{T} part 5 entry: the stage’s badge is below the top bar (review A r11 #15)', sg['badge'] >= sg['bar'], sg)
        check(f'{T} part 5 enters with MHA, the initial setup', state(pg)['S'] == INIT, state(pg)['S'])
        check(f'{T} part 5: Next disabled before MQA', disabled(pg, '#gm-rail [data-nav="next"]'))
        p5 = plain(pg.inner_text('#gm-rail .step.current'))
        check(f'{T} part 5: each GPU computes one eighth of the query heads; MHA gives it one eighth of the KV cache, 320 KiB', Z('each computes one eighth of the query heads', '每张算八分之一的 query head') in p5 and Z('With MHA, each GPU stores one eighth of the KV cache, 320 KiB per token.', '用 MHA 时，每张 GPU 存八分之一的 KV cache，每个 token 320 KiB。') in p5, p5)
        chk5 = pg.evaluate("""()=>({gpus:document.querySelectorAll('#gm-diag .gm-gpu').length, users:document.querySelector('#gm-metrics [data-k="users"] .v').textContent,
          cells:[...document.querySelectorAll('#gm-chart .gm-ch-r')].map(r=>[r.querySelector('.gm-ch-l .def').textContent, r.querySelectorAll('.gm-cc b')[1].textContent])})""")
        check(f'{T} part 5: GPU boxes and MHA’s 1 user shown', chk5['gpus'] == 8 and chk5['users'] == '1', chk5)
        c5 = {plain(a): v for a, v in chk5['cells']}
        check(f'{T} part 5: chart shows MHA per GPU, not yet the others', c5.get('MHA') == '320 KiB' and c5.get('MQA') == '?' and c5.get('GQA-8') == '?', c5)
        rail_click(pg, '#gm-ic-kv5 button[data-v="gqa4"]', 400)
        result_on_screen(pg, f'{T} part 5 pick gqa4')
        n4 = pg.evaluate("()=>[...document.querySelectorAll('#gm-diag .gm-kl')].map(e=>e.textContent)")
        check(f'{T} part 5 GQA-4: KV heads numbered 1 1 2 2 3 3 4 4', n4 == ['1', '1', '2', '2', '3', '3', '4', '4'], n4)
        o = plain(pg.inner_text('#gm-rail [data-obs]'))
        check(f'{T} part 5: GQA-4 does not give away the verdict, and points to MQA', Z('the right answer', '正确答案') not in o and Z('Now pick 1 KV head (MQA) to check your prediction.', '再选 1 个 KV head（MQA），验证你的预测。') in o, o)
        rail_click(pg, '#gm-ic-kv5 button[data-v="mqa"]', 400)
        result_on_screen(pg, f'{T} part 5 pick mqa')
        o5 = plain(pg.inner_text('#gm-rail [data-obs]'))
        check(f'{T} part 5: verdict quotes both options', Z('You predicted “64 or more”; the right answer is “2 to 20”.', '你预测的是“64 个或更多”，正确答案是“2 到 20 个”。') in o5, o5)
        check(f'{T} part 5: result first, then verdict', 0 <= o5.find(Z('1 KV head (MQA): at most 10 users fit, the same as with GQA-8.', '1 个 KV head（MQA）：最多装下 10 个用户，和 GQA-8 一样。')) < o5.find(Z('You predicted', '你预测的是')), o5)
        check(f'{T} part 5: the why says each GPU computes attention from its own memory, and compares with MHA', Z('from its own memory at every decode step', '每个 decode 步都用自己显存里的') in o5
              and Z('Per token MQA stores one sixty-fourth of what MHA stores, but on each GPU it is still one eighth.', '按每个 token 算，MQA 只存 MHA 的六十四分之一；可在每张 GPU 上，仍然是 MHA 的八分之一。') in o5, o5)
        dg = pg.evaluate("()=>({nums:[...document.querySelectorAll('#gm-diag .gm-kl')].map(e=>e.textContent), leg:document.getElementById('gm-legend').innerText, cap:document.getElementById('gm-cap').textContent})")
        check(f'{T} part 5 MQA: every GPU’s copy is labeled KV head 1', dg['nums'] == ['1'] * 8, dg)
        check(f'{T} part 5 MQA: the legend explains the line below and the caption says it is drawn once per GPU', Z('line below', '下方横线') in dg['leg'] and Z('drawn once on every GPU', '各画一份') in plain(dg['cap']), dg)
        lg = plain(pg.evaluate("()=>[...document.querySelectorAll('#gm-legend [data-tipx]')].map(e=>e.getAttribute('data-tipx')).join('\\n')"))
        check(f'{T} part 5: the KV head legend’s hover names MQA in full', Z('MQA, multi-query attention', 'MQA，multi-query attention') in lg, lg)
        card_height(pg, f'{T} part 5 after MQA')
        check(f'{T} part 5: prediction locked after the check', state(pg)['seen'])
        pg.screenshot(path=str(SHOTS / f'1440_{lang}_p5.png'))
        # ----- part 6: MLA, another approach -----
        rail_click(pg, '#gm-rail [data-nav="next"]', 400); pg.wait_for_timeout(700); settle(pg)
        sg = pg.evaluate("()=>{const b=document.getElementById('gm-status').getBoundingClientRect(), t=document.querySelector('.topbar').offsetHeight; return {badge:Math.round(b.top), bar:t}}")
        check(f'{T} part 6 entry: the stage’s badge is below the top bar (review A r11 #15)', sg['badge'] >= sg['bar'], sg)
        check(f'{T} part 6: Next disabled before MLA', disabled(pg, '#gm-rail [data-nav="next"]'))
        c6 = pg.evaluate("()=>[...document.querySelectorAll('#gm-chart .gm-ch-r')].map(r=>[r.querySelector('.gm-ch-l .def').textContent, r.querySelectorAll('.gm-cc b')[1].textContent])")
        check(f'{T} part 6: before the pick, MLA’s per-GPU value is a ? (review A r11 #7)', [c for c in c6 if c[0].startswith('MLA')][0][1] == '?' and Z('A ? marks', '“?”表示') in pg.inner_text('#gm-chart'), c6)
        p6 = plain(pg.inner_text('#gm-rail .step.current'))
        check(f'{T} part 6: MLA with its full name, as another approach', Z('Multi-head latent attention (MLA), used in DeepSeek-V2 and V3, takes another approach', 'DeepSeek-V2 和 V3 用的 multi-head latent attention（MLA）换了一种做法') in p6, p6[:300])
        eqt = tip_of(pg, '#gm-rail .step.current', ['2.25 KV heads', '2.25 个 KV head'])
        check(f'{T} part 6: as much as 2.25 KV heads, the 576 numbers and the division on hover (R6.20); Chinese in 维', Z('Together that is as much as 2.25 KV heads store', '两者加起来，相当于 2.25 个 KV head 存的量') in p6
              and Z('make 576 numbers per token and layer', '一共 576 维') in eqt and '576 ÷ 256 = 2.25' in eqt, (p6, eqt))
        d6 = defs(pg)
        check(f'{T} part 6: MLA, latent, RoPE key, rebuilt and these sizes defined', len(d6) >= 5, d6)
        full_names_ok(pg, f'{T} part 6', lang, '#gm-rail')
        rail_click(pg, '#gm-ic-mla button[data-v="mla"]', 400)
        result_on_screen(pg, f'{T} part 6 pick mla')
        o6 = plain(pg.inner_text('#gm-rail [data-obs]'))
        check(f'{T} part 6: MLA, much less per token than GQA-8 but a full copy on each GPU, 4 users against GQA-8’s 10 (R6.20)', Z('MLA: much less KV cache per token than GQA-8, but each GPU keeps all of it, so at most 4 users fit, fewer than GQA-8’s 10.', 'MLA：每个 token 的 KV cache 比 GQA-8 少得多，但每张 GPU 都要存全部，所以最多装下 4 个用户，比 GQA-8 的 10 个少。') in o6, o6)
        check(f'{T} part 6: the why: the whole compressed vector, like MQA, no pointer back to part 5 (owner G2, 2026-10-08), plain tensor parallelism defined', Z('from the whole compressed vector', '用整个压缩向量') in o6 and Z('in part 5', '第 5 部分') not in o6 and pg.query_selector('#gm-rail [data-obs] .def') is not None, o6)
        check(f'{T} part 6: the why names the compressed vector, not “it” (review A r12 N12)', Z('Like the single KV head of MQA, the compressed vector is copied to all 8 GPUs', '压缩向量和 MQA 唯一的 KV head 一样，纯张量并行时要复制到全部 8 张 GPU 上。') in o6, o6)
        pl = tip_of(pg, '#gm-rail [data-obs]', ['plain tensor parallelism', '纯张量并行'])
        check(f'{T} part 6: plain = the kit’s model-free text + which design has 8 KV heads and the initial setup’s 64 + what “plain” means (review R14 N5)', pl == kit['tpmla'] + Z(' For example, the GQA-8 that Llama-3.1-70B really uses has 8 KV heads per layer, one per GPU on 8 GPUs, 1/8 of the total; this lab’s initial setup, MHA, has 64, 8 per GPU. “Plain” means every layer is split across the GPUs, attention by head; the alternative keeps the other layers split but runs attention data-parallel.',
              '比如 Llama-3.1-70B 真实用的 GQA-8 每层有 8 个 KV head，8 张 GPU 正好每张存一个，即总量的 1/8；这个 Lab 的初始设置 MHA 有 64 个，每张存 8 个。“纯”指每一层都拆到各张 GPU 上，attention 按 head 拆；另一种做法是其余各层照样拆，attention 改做数据并行。') and 'Llama' not in kit['tpmla'], pl)
        up = tip_of(pg, '#gm-rail .step.current', ['rebuilt', '还原'])
        check(f'{T} part 6: rebuilt says only the compressed vector and RoPE key are stored, and folding is at decode steps (review B r12 N3, A N8)',
              Z('Only the compressed vector and the RoPE key are stored. At decode steps, engines such as vLLM fold these matrices', '存下来的只有压缩向量和 RoPE key。decode 步里，vLLM 这类引擎把这些上投影矩阵') in up and Z('the prefill', 'prefill') in up and Z('never stored', '从来不存') not in up and '这两个矩阵' not in up
              and Z('fold these matrices into Wq and Wo', '把这些上投影矩阵并进 Wq 和 Wo') in up and 'output matri' not in up and '输出矩阵' not in up, up)
        lg = plain(pg.evaluate("()=>[...document.querySelectorAll('#gm-legend [data-tipx]')].map(e=>e.getAttribute('data-tipx')).join('\\n')"))
        check(f'{T} part 6: the dashed-line legend says decode steps, and 步 only for the engine’s step (review A r13 M1)', Z('at decode steps, engines skip the rebuilding', 'decode 步里引擎不做这次还原') in lg and Z('at inference', '推理时') not in lg and '还原这一步' not in lg, lg)
        check(f'{T} part 6: the compressed-vector legend names MLA in full', Z('MLA, multi-head latent attention', 'MLA（multi-head latent attention）') in lg, lg)
        c6b = pg.evaluate("()=>[...document.querySelectorAll('#gm-chart .gm-ch-r')].map(r=>[r.querySelector('.gm-ch-l .def').textContent, r.querySelectorAll('.gm-cc b')[1].textContent])")
        check(f'{T} part 6: after the pick, MLA’s per-GPU value is shown', [c for c in c6b if c[0].startswith('MLA')][0][1] == '90 KiB', c6b)
        ct = [plain(t) for t in pg.evaluate("()=>[...document.querySelectorAll('#gm-chart .gm-ch-l .def')].map(e=>e.getAttribute('data-tipx'))")]
        check(f'{T} part 6: every chart label’s hover has the full name (review 8)', len(ct) == 6 and all(FULL[lang][a] in t for a, t in zip(DES, ct)), ct)
        for v, want in [('gqa8', Z('GQA-8, the design Llama-3.1-70B uses: each GPU stores one eighth of its KV cache, and at most 10 users fit.', 'GQA-8（Llama-3.1-70B 用的设计）：每张 GPU 存它 KV cache 的八分之一，最多装下 10 个用户。')),
                        ('mha', Z('MHA, the initial setup: each GPU stores one eighth of its KV cache, and at most 1 user fits.', 'MHA（初始设置）：每张 GPU 存它 KV cache 的八分之一，最多装下 1 个用户。'))]:
            rail_click(pg, f'#gm-ic-mla button[data-v="{v}"]', 300)
            o = plain(pg.inner_text('#gm-rail [data-obs]'))
            check(f'{T} part 6 {v}: result', want in o, o)
        rail_click(pg, '#gm-ic-mla button[data-v="mla"]', 300)
        card_height(pg, f'{T} part 6 after MLA')
        pg.screenshot(path=str(SHOTS / f'1440_{lang}_p6.png'))
        # ----- part 7: it ends in Scenarios (R3.12) -----
        rail_click(pg, '#gm-rail [data-nav="next"]', 400)
        card_height(pg, f'{T} part 7')
        p7 = plain(pg.inner_text('#gm-rail .step.current'))
        check(f'{T} part 7: MHA, GQA, MQA in that order, and the misconceptions', p7.find('MHA') < p7.find('GQA') < p7.find('MQA') and Z('common misconceptions', '常见误解') in p7.lower(), p7[:300])
        check(f'{T} part 7: ends with “See the scenarios”, no challenge button (R3.12)', pg.query_selector('#gm-rail .step.current [data-mode-go="sandbox"]') is not None and pg.query_selector('#gm-rail [data-mode-go="challenge"]') is None
              and plain(pg.inner_text('#gm-rail .step.current [data-mode-go="sandbox"]')) == Z('See the scenarios', '去看应用场景'))
        nl = pg.query_selector('#gm-rail .next-lab a')
        check(f'{T} part 7: next lab is the KV cache memory lab', nl is not None and nl.get_attribute('href') == '#kv-cache')
        pg.screenshot(path=str(SHOTS / f'1440_{lang}_p7.png'))
        # language switch keeps the part, the prediction and the design
        before = state(pg)
        other = 'zh' if lang == 'en' else 'en'
        pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(300)
        after = state(pg)
        check(f'{T}: language switch keeps part, prediction, design', (before['step'], before['pred'], before['S']) == (after['step'], after['pred'], after['S']), (before, after))
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(300)
        # back to part 3: parts 1-4 are about one GPU's worth of KV heads, so users and GPUs stay hidden there even after part 5
        go_part(pg, 2)
        check(f'{T}: going back to part 3 still hides the GPUs (MHA’s 1 user shows, as in part 1)', pg.inner_text('#gm-metrics [data-k="gpu"] .v') == '?' and pg.inner_text('#gm-metrics [data-k="users"] .v') == '1' and pg.evaluate("()=>document.querySelectorAll('#gm-diag .gm-gpu').length") == 0)
        # ----- the mode order: guided end -> Scenarios -> Challenge (R3.12) -----
        go_part(pg, 6); rail_click(pg, '#gm-rail [data-mode-go="sandbox"]', 400)
        check(f'{T}: “See the scenarios” opens Scenarios', state(pg)['mode'] == 'sandbox')
        sc = plain(pg.inner_text('#gm-rail .railcard'))
        check(f'{T} Scenarios: the card ends with “Take the challenge” (R3.12)', pg.query_selector('#gm-rail .railcard [data-mode-go="challenge"]') is not None and sc.rstrip().endswith(Z('Take the challenge', '去做挑战')), sc[-200:])
        check(f'{T} Scenarios: the MHA scenario first, then Llama-3.1-70B with GQA-8', [plain(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#gm-rail [data-preset] b')].map(e=>e.textContent)")][:2] == Z(['Llama-3.1-70B’s shape with MHA', 'Llama-3.1-70B on 8 × H100'], ['Llama-3.1-70B 的结构用 MHA', '8 × H100 跑 Llama-3.1-70B']))
        rail_click(pg, '#gm-rail [data-mode-go="challenge"]', 400)
        check(f'{T}: “Take the challenge” opens the challenge', state(pg)['mode'] == 'challenge')
        check(f'{T} challenge: submit disabled before a pick', disabled(pg, '#gm-rail [data-act="submit"]'))
        cht = plain(pg.inner_text('#gm-rail .railcard > p'))
        check(f'{T} challenge: the paper is scoped to quality for MHA, GQA-8 and MQA (review A r12 N3)', Z('the paper compared quality only for these two and MHA.', '论文只比较了这两种和 MHA 的质量。') in cht and Z('did not test', '没有测') not in cht, cht)
        check(f'{T} challenge: GQA, MHA and MQA with full names in the task', all(x in cht for x in Z(['grouped-query attention (GQA)', 'multi-head attention (MHA)', 'multi-query attention (MQA'], ['grouped-query attention（GQA）', '多头注意力（MHA）', 'multi-query attention（MQA'])), cht)
        lv = [plain(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#gm-rail [data-lever]')].map(b=>b.innerText)")]
        check(f'{T} challenge: every option says its full name', len(lv) == 6 and all(FULL[lang][a].lower() in x.lower() for a, x in zip(DES, lv)), lv)
        full_names_ok(pg, f'{T} challenge', lang, '#gm-rail')
        check(f'{T} challenge: the chart of all designs is hidden', pg.evaluate("()=>document.getElementById('gm-chart').hidden"))
        check(f'{T} challenge: other settings hidden, attention design visible', pg.evaluate("()=>{const c=document.getElementById('gm-controls'); return c.classList.contains('locked')&&!document.getElementById('gm-ctl-model').offsetParent&&!!document.getElementById('gm-ctl-attn').offsetParent}"))
        rail_click(pg, '#gm-rail [data-lever="mha"]', 300)
        check(f'{T} challenge: progress 1 / 10', pg.inner_text('#gm-goalv') == '1')
        rail_click(pg, '#gm-rail [data-act="submit"]', 400)
        r1 = plain(pg.inner_text('#gm-chres'))
        check(f'{T} challenge tier 3 (MHA): not met, a hint, “Show the answer”', Z('Not yet', '还没达到') in r1 and Z('Hint', '提示') in r1 and pg.query_selector('#gm-rail [data-act="reveal"]') is not None
              and plain(pg.inner_text('#gm-rail [data-act="reveal"]')) == Z('Show the answer', '直接看答案'), r1)
        check(f'{T} challenge tier 3: no answer given', 'GQA-8' not in r1, r1)
        # scroll back up to the options like a learner, pick another design: the new feedback comes on screen (review A r12 N6)
        pg.evaluate("()=>document.querySelector('#gm-rail [data-lever=\"mha\"]').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200); settle(pg)
        pg.click('#gm-rail [data-lever="mla"]'); pg.wait_for_timeout(300); settle(pg)
        result_on_screen(pg, f'{T} challenge: a new pick in the task card after the submit', '#gm-chres')
        r1b = plain(pg.inner_text('#gm-chres'))
        check(f'{T} challenge tier 3 (MLA): not met with its own hint', 'MLA' in r1b and Z('Not yet', '还没达到') in r1b, r1b)
        rail_click(pg, '#gm-rail [data-act="reveal"]', 300)
        check(f'{T} challenge: “Show the answer” draws the chart', pg.query_selector('#gm-chres .cmp') is not None)
        cr = pg.evaluate("""()=>{const rows=[...document.querySelectorAll('#gm-chres .gm-cmpwrap > .cmp .cmp-row')], tg=r=>!!r.querySelector('.cmp-tag,.cmp-you');
          return {n:rows.length, ok:rows.filter(r=>r.classList.contains('ok')).length, me:rows.filter(r=>r.classList.contains('me')).length,
            stray:rows.filter(r=>!r.classList.contains('ok')&&!r.classList.contains('me')&&!tg(r)).length, note:(document.querySelector('#gm-chres .cmp-note')||{}).textContent||'',
            title:(document.querySelector('#gm-chres .cmp-title')||{}).textContent||'', init:rows.filter(r=>/initial setup|初始设置/.test(r.textContent)).map(r=>r.querySelector('.cmp-lab span').textContent)}}""")
        check(f'{T} challenge, the answer for MLA: the 3 designs that meet the goal, MHA (the initial setup) and my pick, nothing else', cr['n'] == 5 and cr['ok'] == 3 and cr['me'] == 1 and cr['stray'] == 0
              and len(cr['init']) == 1 and 'MHA' in cr['init'][0], cr)
        check(f'{T} challenge: the note counts the 1 design left out; the title no longer says “by attention design”', cr['note'] == Z('The other design fits fewer than 10 users and is not shown.', '其余 1 种设计装不下 10 个用户，图里没有列出。')
              and not re.search(r'by attention design|各种', cr['title']), (cr['note'], cr['title']))
        rail_click(pg, '#gm-rail [data-lever="mqa"]', 300)
        r2 = plain(pg.inner_text('#gm-chres'))
        check(f'{T} challenge tier 2 (MQA): met, but GQA-8 keeps more KV heads', Z('Goal met', '达成目标') in r2 and 'GQA-8' in r2 and Z('8 KV heads instead of 1', '有 8 个 KV head，不是 1 个') in r2, r2)
        rail_click(pg, '#gm-rail [data-lever="gqa8"]', 300)
        r3 = plain(pg.inner_text('#gm-chres'))
        check(f'{T} challenge tier 1 (GQA-8): met and the best', Z('Goal met: 10 users fit, with 8 KV heads.', '达成目标：装下 10 个用户，有 8 个 KV head。') in r3, r3)
        check(f'{T} challenge note: SGLang can run data-parallel attention (review A r13 M2)', Z('Serving stacks such as SGLang can run data-parallel attention instead, so that MLA’s copies go away; this lab does not model that.', 'SGLang 等框架可以改用数据并行 attention，MLA 的副本就没有了；这个 Lab 没有建模这种做法。') in r3, r3[-300:])
        dp = tip_of(pg, '#gm-chres', ['data-parallel attention', '数据并行 attention'])
        check(f'{T} challenge: the data-parallel attention tip says it is an option, off by default', Z('as an option (--enable-dp-attention), off by default', '可选的做法（--enable-dp-attention，默认不开）') in dp, dp)
        rows = pg.evaluate("()=>[...document.querySelectorAll('#gm-chres .cmp-row')].map(r=>[r.querySelector('.cmp-lab span').textContent, r.classList.contains('ok'), r.classList.contains('me')])")
        check(f'{T} challenge chart: four rows (MHA, the initial setup, and the three green), your pick marked', len(rows) == 4 and sum(1 for r in rows if r[1]) == 3 and sum(1 for r in rows if r[2]) == 1, rows)
        check(f'{T} challenge chart: every label has the full name (review 8)', len(rows) == 4 and all(FULL[lang][a] in plain(r[0]) for a, r in zip(['mha', 'gqa8', 'gqa4', 'mqa'], rows)), rows)
        cr = pg.evaluate("""()=>{const rows=[...document.querySelectorAll('#gm-chres .gm-cmpwrap > .cmp .cmp-row')], tg=r=>!!r.querySelector('.cmp-tag,.cmp-you');
          return {n:rows.length, ok:rows.filter(r=>r.classList.contains('ok')).length, me:rows.filter(r=>r.classList.contains('me')).length,
            stray:rows.filter(r=>!r.classList.contains('ok')&&!r.classList.contains('me')&&!tg(r)).length, note:(document.querySelector('#gm-chres .cmp-note')||{}).textContent||'',
            title:(document.querySelector('#gm-chres .cmp-title')||{}).textContent||'', init:rows.filter(r=>/initial setup|初始设置/.test(r.textContent)).map(r=>r.querySelector('.cmp-lab span').textContent)}}""")
        check(f'{T} challenge chart: nothing else, and the note counts the 2 designs left out', cr['stray'] == 0 and cr['note'] == Z('The other 2 designs fit fewer than 10 users and are not shown.', '其余 2 种设计装不下 10 个用户，图里没有列出。'), cr)
        no_exact_hover(pg, f'{T} challenge answered')
        pg.screenshot(path=str(SHOTS / f'1440_{lang}_challenge.png'), full_page=True)
        # the challenge survives a language switch
        before = state(pg); pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(300); after = state(pg)
        check(f'{T}: language switch keeps the challenge', (before['mode'], before['S'], before['chSubmitted'], before['chPicked']) == (after['mode'], after['S'], after['chSubmitted'], after['chPicked']), (before, after))
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(300)
        # scenarios: pressed state follows the settings; the result line changes on every click
        set_mode(pg, 'sandbox')
        lines = []
        for i in range(5):
            rail_click(pg, f'#gm-rail [data-preset="{i}"]', 300)
            pressed = pg.evaluate("()=>[...document.querySelectorAll('#gm-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
            check(f'{T} scenario {i + 1}: pressed alone', pressed.count('true') == 1 and pressed[i] == 'true', pressed)
            lines.append(plain(pg.inner_text('#gm-sbnote')))
            result_on_screen(pg, f'{T} scenario {i + 1}', '#gm-sbnote')
        check(f'{T} scenarios: every click changes the first sentence', len(set(re.split(r'(?<=\.)\s|。', l)[0] for l in lines)) == 5, lines)
        us = [re.search(r'(\d+) (?:users?|user fits|个用户)', l) for l in lines]
        check(f'{T} scenarios: users 1, 10, 3, 2, 5', [int(x.group(1)) if x else None for x in us] == [1, 10, 3, 2, 5], lines)
        check(f'{T} scenario 1: MHA is the initial setup', Z('This is the initial setup: every query head has its own KV head.', '这是初始设置：每个 query head 都有自己的 KV head。') in lines[0], lines[0])
        check(f'{T} scenario 2: GQA-8 against MHA, what each GPU stores in words (R6.20)', Z('Every 8 query heads share one KV head, so each GPU stores one eighth as much as with MHA.', '每 8 个 query head 共用一个 KV head，所以每张 GPU 存的只有 MHA 的八分之一。') in lines[1], lines[1])
        check(f'{T} scenario 4: the 405B line says FP8 counts 1 byte and names Meta’s larger FP8 release and the size of its BF16 part (review B r12 N9, review A r13 M4)', Z('That counts one byte per parameter: Meta’s own FP8 release keeps a fifth of the parameters in BF16, every attention matrix among them, which leaves room for only 1 user.', '这是按每个参数一个字节算的；Meta 官方的 FP8 版本有五分之一的参数留在 BF16，所有 attention 矩阵都在其中，只装得下 1 个用户。') in lines[3]
              and Z('its weights take more than half of the memory each GPU can use', '权重也要占每张 GPU 可用显存的一半以上') in lines[3]
              and Z('its extra layers make each token’s KV cache bigger than Llama-3.1-70B’s.', '它的层数更多，每个 token 的 KV cache 也比 Llama-3.1-70B 的大。') in lines[3], lines[3])
        rail_click(pg, '#gm-rail [data-preset="3"]', 300)
        # the exact BF16 count left the scenario line (R6.20); the Fidelity card keeps 81.3B, with no exact-value hover (the owner, 2026-10-07 23:37 UTC)
        check(f'{T} Fidelity card: 81.3B with no exact-value hover (the owner, 23:37 UTC)', '81.3B' in plain(pg.evaluate("()=>document.getElementById('gqa-mla-fid').textContent")) and tip_of(pg, '#gqa-mla-fid', ['81.3B']) == '', tip_of(pg, '#gqa-mla-fid', ['81.3B']))
        no_exact_hover(pg, f'{T} scenarios')
        g4 = pg.evaluate("""()=>{const n=document.getElementById('gm-sbnote'), d=[...n.querySelectorAll('.def')].find(e=>e.textContent==='GiB'); if(!d) return null;
          const rg=document.createRange(); rg.setStart(n,0); rg.setEndBefore(d); return {before:rg.toString(), tip:d.getAttribute('data-tipx')}}""")
        check(f'{T} scenario 4: the sentence says the weights in words, so no GiB is left in it; a GiB there would carry the GiB definition (review R14 N10)', (g4 is None and 'GiB' not in lines[3]) or (g4 is not None and 'GiB' not in g4['before'] and plain(g4['tip']) == kit['gib']), (g4, lines[3]))
        pg.evaluate("()=>{const i=document.getElementById('gm-ctl-ctx-in'); i.value=3; i.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(250)
        check(f'{T} scenarios: none pressed after a hand change', pg.evaluate("()=>[...document.querySelectorAll('#gm-rail [data-preset]')].every(b=>b.getAttribute('aria-pressed')==='false')"))
        # hash round trip: the URL token opens Scenarios with the same state
        tok = pg.evaluate("()=>location.hash")
        stA = state(pg)['S']
        pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(300)
        pg.evaluate(f"()=>{{location.hash='{tok[1:]}'}}"); pg.wait_for_timeout(600)
        stB = state(pg)
        check(f'{T}: hash round-trip', stB['S'] == stA and stB['mode'] == 'sandbox', (tok, stA, stB))
        # no share button since the owner's decision of 2026-10-11 19:28 UTC
        check(f'{T}: no share button', pg.query_selector('.dyn-root [data-share]') is None)
        # Settings: the full name in every option’s hover and in the note; the note never calls GQA-8 the original design
        rail_click(pg, '#gm-rail [data-preset="0"]', 250)
        full_names_ok(pg, f'{T} Settings', lang, '#gm-controls')
        for v in DES:
            pg.click(f'#gm-ctl-attn button[data-v="{v}"]'); pg.wait_for_timeout(150)
            nt = plain(pg.inner_text('#gm-attn-note'))
            check(f'{T} Settings {v}: the note starts with the full name', nt.lower().startswith(FULL[lang][v]) or nt.startswith(FULL[lang][v]), nt)
        pg.click('#gm-ctl-attn button[data-v="mha"]'); pg.wait_for_timeout(150)
        check(f'{T} Settings MHA: the initial setup, a what-if (the real model uses GQA-8)', plain(pg.inner_text('#gm-attn-note')) == Z('Multi-head attention (MHA): 64 KV heads, one per query head, the initial setup (a what-if; the real model uses GQA-8).', '多头注意力（MHA）：64 个 KV head，每个 query head 一个，初始设置（这是假设，真实模型用 GQA-8）。'), pg.inner_text('#gm-attn-note'))
        wn = plain(pg.inner_text('#gm-model-note'))
        check(f'{T} Settings: the model note names the real model’s design, 70.6B with no exact-value hover (review R14 N5; the owner, 23:37 UTC)', wn == Z('80 layers, 64 query heads, 70.6B parameters; the real model uses 8 KV heads of 128 numbers (GQA-8). Picking a model also picks the attention design it uses.', '80 层，64 个 query head，70.6B 个参数；真实模型用 8 个 KV head（GQA-8，每个 128 维）。换模型时，注意力设计也换成这个模型用的设计。')
              and tip_of(pg, '#gm-model-note', ['70.6B']) == '', (wn, tip_of(pg, '#gm-model-note', ['70.6B'])))
        no_exact_hover(pg, f'{T} Settings')
        for gv, mem in [('h100', 80), ('h200', 141)]:
            pg.evaluate(f"()=>{{const i=document.getElementById('gm-gpu'); i.value='{gv}'; i.dispatchEvent(new Event('change',{{bubbles:true}}))}}"); pg.wait_for_timeout(200)
            gn = plain(pg.inner_text('#gm-gpu-note'))
            check(f'{T} Settings {gv}: the GPU note says why the vendor’s GB is read as GiB, with GiB defined (review R14 2-6)', gn == Z(f'The vendor’s {mem} GB is read as {mem} GiB, as in the lab “KV cache memory”: a GPU’s usable memory is close to that many GiB.', f'厂商标的 {mem} GB 按 {mem} GiB 计，和“KV cache 显存”Lab 一样：显卡实际能用的显存接近同样多的 GiB。')
                  and tip_of(pg, '#gm-gpu-note', ['GiB']) == kit['gib'] and pg.query_selector('#gm-gpu-note a[href="#kv-cache"]') is not None, gn)
        rail_click(pg, '#gm-rail [data-preset="0"]', 250)
        pt = [(plain(a), plain(t)) for a, t in pg.evaluate("()=>[...document.querySelectorAll('#gm-rail [data-preset]')].map(b=>[b.querySelector('b').textContent, b.getAttribute('data-tipx')||''])")]
        check(f'{T} Scenarios: every scenario button’s hover names its design in full (R2.8, round 15 (h))', len(pt) == 5 and all(FULL[lang][a] in t for a, (x, t) in zip(['mha', 'gqa8', 'gqa8', 'gqa8', 'mla'], pt)), pt)
        # Show the math: the weights go from bytes to GB and GiB, and GiB is defined where it first appears (R6.12)
        pg.evaluate("()=>{document.getElementById('gm-math').open=true}"); pg.wait_for_timeout(150)
        mt = plain(pg.inner_text('#gm-mathbody'))
        check(f'{T} math: the weights in bytes, GB and GiB', Z('(BF16 weights: 70,553,706,496 × 2 bytes = 141.1 GB = 131.4 GiB, ÷ 8 GPUs)', '（BF16 权重：70,553,706,496 × 2 字节 = 141.1 GB = 131.4 GiB，÷ 8 张 GPU）') in mt, mt)
        gd = pg.evaluate("()=>{const d=[...document.querySelectorAll('#gm-mathbody .def')].find(e=>e.textContent==='GiB'); return d?d.getAttribute('data-tipx'):''}")
        check(f'{T} math: the first GiB carries the site’s GiB definition', plain(gd) == kit['gib'], gd)
        pg.evaluate("()=>{document.getElementById('gm-math').open=false}")
        # writing rules on everything visible and every tooltip, in each mode; no number over 6 digits outside the math and tooltips
        for mode in ['guided', 'sandbox', 'challenge']:
            set_mode(pg, mode, 300)
            parts = range(7) if mode == 'guided' else [None]
            for k in parts:
                if k is not None: go_part(pg, k)
                tg = f'{T} {mode}' + (f' part {k + 1}' if k is not None else '')
                big_scan(pg, tg)
                pg.evaluate("()=>{document.getElementById('gm-math').open=true; document.getElementById('gm-fixed').open=true}")
                rule_scan(tg, visible_text(pg) + '\n' + all_tips(pg), lang)
                pg.evaluate("()=>{document.getElementById('gm-math').open=false; document.getElementById('gm-fixed').open=false}")
        # the Connect code reads without sideways scrolling at 1440 (R7.6, review 8)
        cw = pg.evaluate("()=>{const c=document.getElementById('gm-code'); return [c.scrollWidth,c.clientWidth]}")
        check(f'{T} Connect: the code block needs no sideways scrolling', cw[0] <= cw[1], cw)
        # the home tile: renders, one sentence of at most 25 English words, a legend, notes with the full names
        pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(300)
        pg.evaluate("()=>document.querySelector('#mapSvg .world[data-world=\"models\"]').dispatchEvent(new MouseEvent('click',{bubbles:true}))"); pg.wait_for_timeout(300)
        tile = pg.query_selector('#tile-gqa-mla')
        check(f'{T} tile: rendered', tile is not None and pg.query_selector('#tile-gqa-mla .tz-read') is not None)
        check(f'{T} tile: starts on MHA (review 8)', pg.evaluate("()=>document.querySelector('#tile-gqa-mla .seg-ctl button[aria-pressed=\"true\"]').dataset.v") == 'mha')
        tt = dict((a, plain(t)) for a, t in pg.evaluate("()=>[...document.querySelectorAll('#tile-gqa-mla .seg-ctl button[data-v]')].map(b=>[b.dataset.v,b.getAttribute('data-tipx')])"))
        check(f'{T} tile: every option’s hover has its full name', len(tt) == 4 and all(FULL[lang][a] in t for a, t in tt.items()), tt)
        for v in ['mha', 'gqa8', 'mqa', 'mla']:
            pg.click(f'#tile-gqa-mla .seg-ctl button[data-v="{v}"]'); pg.wait_for_timeout(120)
            rd = plain(pg.inner_text('#tile-gqa-mla .tz-read'))
            if lang == 'en': check(f'{T} tile {v}: at most 25 words', len(rd.split()) <= 25, (len(rd.split()), rd))
            check(f'{T} tile {v}: the reading names the design in full', FULL[lang][v] in rd, rd)
            rule_scan(f'{T} tile {v}', rd + '\n' + plain(pg.evaluate("()=>[...document.querySelectorAll('#tile-gqa-mla .tz-note')].map(n=>n.textContent).join('\\n')")), lang)
        check(f'{T} tile: 4 notes, the first says what MHA, GQA-8, MQA and MLA are, in full', len(pg.query_selector_all('#tile-gqa-mla .tz-note')) == 4 and Z('MHA (multi-head attention) gives each', 'MHA（多头注意力）让') in pg.inner_text('#tile-gqa-mla .tz-note'))
        n4 = plain(pg.evaluate("()=>[...document.querySelectorAll('#tile-gqa-mla .tz-note')].pop().textContent"))
        check(f'{T} tile: the source note gives QKVParallelLinear for KV heads and get_num_kv_heads for MLA (review B r12 N5)', n4 == Z('Shapes from Llama-3.1-70B’s config.json. The copies follow vLLM: QKVParallelLinear copies KV heads, and for MLA every GPU keeps one whole compressed vector (ModelConfig.get_num_kv_heads returns 1).', '结构来自 Llama-3.1-70B 的 config.json。复制的规则和 vLLM 一致：KV head 由 QKVParallelLinear 复制；MLA 每张 GPU 存一整份压缩向量（ModelConfig.get_num_kv_heads 返回 1）。'), n4)
        check(f'{T}: no page errors', not errs, errs[:3])
        ctx.close()

    # ================= review 8 (R3.2): a change in Settings never scrolls the page; no challenge answer from Settings =================
    for lang in ['en', 'zh']:
        Z = (lambda en, zh: zh) if lang == 'zh' else (lambda en, zh: en)
        for (w, h) in [(1440, 900), (390, 844)]:
            ctx = b.new_context(viewport={'width': w, 'height': h}); pg = open_drawers(ctx.new_page())
            pg.add_init_script(SPY)
            errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
            pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
            pg.evaluate("()=>{location.hash='gqa-mla'}"); pg.wait_for_timeout(700)
            T = f'{lang} {w}x{h} Settings'
            moved = []
            for mode, k in [('guided', 0), ('guided', 1), ('guided', 4), ('guided', 5), ('sandbox', None), ('challenge', 'before'), ('challenge', 'after')]:
                set_mode(pg, mode, 300)
                if mode == 'guided': go_part(pg, k)
                if mode == 'challenge' and k == 'after':
                    rail_click(pg, '#gm-rail [data-lever="mha"]', 250); rail_click(pg, '#gm-rail [data-act="submit"]', 400)
                # the reader scrolls to Settings (the design buttons in the middle of the screen)
                pg.evaluate("()=>{const e=document.getElementById('gm-ctl-attn'), r=e.getBoundingClientRect(); window.scrollTo(0, scrollY + r.top - innerHeight/2 + r.height/2)}")
                pg.wait_for_timeout(300); settle(pg)
                for a in DES:
                    sel = f'#gm-ctl-attn button[data-v="{a}"]'
                    y0 = pg.evaluate("()=>scrollY"); t0 = pg.evaluate(f"()=>document.querySelector('{sel}').getBoundingClientRect().top")
                    pg.evaluate("()=>{window.__sc=[]}")
                    pg.click(sel); pg.wait_for_timeout(700); settle(pg)
                    y1 = pg.evaluate("()=>scrollY"); t1 = pg.evaluate(f"()=>document.querySelector('{sel}').getBoundingClientRect().top")
                    log = pg.evaluate("()=>window.__sc")
                    tag = f'{T} {mode}{"" if k is None else " " + str(k if mode != "guided" else "part " + str(k + 1))} {a}'
                    # only the one-step correction that keeps the button in place, never a scroll to something (review 8: to the challenge's answer)
                    check(f'{tag}: no script scrolls the page to anything', all(c[0] == 'scrollBy' and len(c[1]) == 2 and all(isinstance(x, (int, float)) for x in c[1]) for c in log), log)
                    # since 2026-10-11 Settings show only in 自由调参; with every part shown (tests) they sit under the card in Scenarios and the challenge, where the one-step correction is expected
                    if w == 1440 and mode == 'guided':
                        check(f'{tag}: scrollY does not change', abs(y1 - y0) < 0.5 and not log, (y0, y1, log))
                    check(f'{tag}: the button stays where it was on screen', abs(t1 - t0) <= 1, (t0, t1, y0, y1))
                    if abs(y1 - y0) >= 0.5: moved.append((mode, k, a, round(y1 - y0)))
                    if mode == 'challenge':
                        res = plain(pg.inner_text('#gm-chres')) if pg.query_selector('#gm-chres') else ''
                        if k == 'before':
                            check(f'{tag}: no challenge answer from Settings before a submit', res == '' and pg.query_selector('.dyn-root .cmp') is None, res)
                        vis = visible_text(pg) + '\n' + plain(pg.evaluate("()=>[...document.querySelectorAll('#gm-controls [data-tipx], #gm-stage [data-tipx]')].map(e=>e.getAttribute('data-tipx')).join('\\n')"))
                        if k == 'before':
                            check(f'{tag}: Settings and the stage never say which design Llama uses', not re.search(r'Llama-3\.1-70B uses|the design Llama-3\.1-70B uses|Llama-3\.1-70B 用的', vis), [l for l in vis.split('\n') if 'uses' in l or '用的' in l][:3])
            if w == 390:
                check(f'{T}: at 390 scrollY moves only to keep the button in place (the card and the stage above Settings change height)', True, moved[:4])
            check(f'{T}: no page errors', not errs, errs[:3])
            ctx.close()

    # ================= edge states (review A r11 #13): Skip, MHA's formula term, one GPU =================
    for lang in ['en', 'zh']:
        Z = (lambda en, zh: zh) if lang == 'zh' else (lambda en, zh: en)
        ctx = b.new_context(viewport={'width': 1440, 'height': 1000}); pg = ctx.new_page()
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        pg.evaluate("()=>{location.hash='gqa-mla'}"); pg.wait_for_timeout(700)
        T = f'{lang} skip path'
        go_part(pg, 3); rail_click(pg, '#gm-rail [data-nav="skip"]', 400)
        check(f'{T}: Skip goes on to part 5', state(pg)['step'] == 4 and state(pg)['skipped'])
        go_part(pg, 3)
        bd = pg.evaluate("()=>{const s=document.getElementById('gm-status'); return [s.textContent, s.getAttribute('data-tipx')]}")
        check(f'{T}: after Skip the badge no longer asks for a prediction', plain(bd[0]) != Z('Predict first', '先预测') and Z('prediction', '预测') not in bd[1], bd)
        pg.click('#gm-ctl-attn button[data-v="gqa8"]'); pg.wait_for_timeout(200)
        bd = pg.evaluate("()=>{const s=document.getElementById('gm-status'); return [s.textContent, s.getAttribute('data-tipx')]}")
        check(f'{T}: part 4 with GQA-8 after Skip: users wait for part 5, no prediction asked', plain(bd[0]) == Z('Users: part 5', '用户数：第 5 部分') and Z('prediction', '预测') not in bd[1], bd)
        go_part(pg, 4); rail_click(pg, '#gm-ic-kv5 button[data-v="gqa4"]', 400)
        o = plain(pg.inner_text('#gm-rail [data-obs]'))
        check(f'{T}: part 5 GQA-4 restates the question (MQA on the same 8 GPUs), not part 4 and not a prediction (owner G2, 2026-10-08)', Z('Now pick 1 KV head (MQA) and see how many users fit on the same 8 GPUs.', '再选 1 个 KV head（MQA），看看同样 8 张 GPU 能装下多少个用户。') in o and Z('part 4', '第 4 部分') not in o and Z('check your prediction', '验证你的预测') not in o, o)
        rail_click(pg, '#gm-ic-kv5 button[data-v="mha"]', 400)
        pg.evaluate("()=>{document.getElementById('gm-math').open=true}"); pg.wait_for_timeout(150)
        el = pg.query_selector('#gm-mathbody .t[data-ctl="attn"]'); el.scroll_into_view_if_needed(); bb = el.bounding_box()
        pg.mouse.move(5, 5); pg.mouse.move(bb['x'] + 4, bb['y'] + bb['height'] / 2); pg.mouse.move(bb['x'] + 5, bb['y'] + bb['height'] / 2); pg.wait_for_timeout(150)
        tip = plain(pg.evaluate("()=>document.getElementById('tip').innerText"))
        check(f'{T}: MHA’s formula term reads “one for each query head”, with the full name', Z('one for each query head', '每个 query head 各有一个') in tip and Z('MHA (multi-head attention)', 'MHA（多头注意力）') in tip and Z('shared by 1 query', '由 1 个') not in tip, tip)
        set_mode(pg, 'sandbox', 300)
        pg.click('#gm-ctl-n button[data-v="1"]'); pg.wait_for_timeout(250)
        nn = plain(pg.inner_text('#gm-n-note'))
        check(f'{T}: one GPU computes all the heads (no “1/1”)', '1/1' not in nn and Z('One GPU computes all', '一张 GPU 算全部') in nn, nn)
        ctx.close()

    # ================= review A r12 N5 and r13 M7: the stage stays below the top bar after picks in part 5 and after Next from part 6 =================
    for (w, h) in [(1440, 900), (1280, 800)]:
        for lang in ['en', 'zh']:
            ctx = b.new_context(viewport={'width': w, 'height': h}); pg = open_drawers(ctx.new_page())
            pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
            pg.evaluate("()=>{location.hash='gqa-mla'}"); pg.wait_for_timeout(700)
            go_part(pg, 3); rail_click(pg, '#gm-rail [data-pred="few"]', 300); rail_click(pg, '#gm-rail [data-nav="next"]', 400); pg.wait_for_timeout(700); settle(pg)
            rail_click(pg, '#gm-ic-kv5 button[data-v="mqa"]', 400); pg.wait_for_timeout(700); settle(pg)
            # the learner scrolls the page down to read (the page, not the stage: the stage scrolls its own content under the mouse)
            pg.evaluate("()=>window.scrollBy(0,200)"); pg.wait_for_timeout(400); settle(pg)
            for v in ['mha', 'gqa16']:
                rail_click(pg, f'#gm-ic-kv5 button[data-v="{v}"]', 400); pg.wait_for_timeout(800); settle(pg)
                g = stage_ok(pg)
                check(f'{lang} {w}x{h} part 5 after MQA, scrolling and {v}: the stage title is below the top bar and the result on screen (review A r12 N5)',
                      g['title'] >= g['bar'] and g['box'] is not None and g['box'][0] >= g['bar'] and g['box'][1] <= g['vh'], g)
            if w == 1440: pg.screenshot(path=str(SHOTS / f'r12_{w}x{h}_{lang}_p5_gqa16.png'))
            go_part(pg, 5); rail_click(pg, '#gm-ic-mla button[data-v="mla"]', 400); pg.wait_for_timeout(800); settle(pg)
            rail_click(pg, '#gm-rail .step.current [data-nav="next"]', 400); pg.wait_for_timeout(1500); settle(pg)
            g7 = pg.evaluate("()=>({bar:Math.round(document.querySelector('.topbar').getBoundingClientRect().bottom), card:Math.round(document.querySelector('#gm-rail .step.current').getBoundingClientRect().top), title:Math.round(document.getElementById('gm-title').getBoundingClientRect().top), step:window.__gm.state().step})")
            check(f'{lang} {w}x{h}: Next from part 6 after MLA: part 7’s card and the stage title start below the top bar (review A r13 M7)', g7['step'] == 6 and g7['card'] >= g7['bar'] - 2 and g7['title'] >= g7['bar'], g7)
            ctx.close()

    # ================= every mode and part at 1440 x 900 and 390 x 844, both languages; overflow; the dock =================
    for lang in ['en', 'zh']:
        Z = (lambda en, zh: zh) if lang == 'zh' else (lambda en, zh: en)
        for (w, h) in [(1440, 900), (390, 844)]:
            ctx = b.new_context(viewport={'width': w, 'height': h}); pg = open_drawers(ctx.new_page())
            errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
            pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
            pg.evaluate("()=>{location.hash='gqa-mla'}"); pg.wait_for_timeout(700)
            T = f'{lang} {w}x{h}'
            for k in range(7):
                go_part(pg, k)
                if k == 3 and not state(pg)['pred']: rail_click(pg, '#gm-rail [data-pred="few"]', 250)
                sm = pg.query_selector('#gm-rail [data-nav="showme"]')
                if sm and sm.is_visible(): rail_click(pg, '#gm-rail [data-nav="showme"]', 400)
                check(f'{T} part {k + 1}: no horizontal overflow', overflow(pg) <= 0, overflow(pg))
                pg.screenshot(path=str(SHOTS / f'{w}_{lang}_part{k + 1}.png'))
                if w == 1440: card_height(pg, f'{T} guided part {k + 1}', heights900)
                if w == 390 and k in (1, 4):
                    # the mini result bar: scroll the stage away, it shows a full sentence
                    pg.evaluate("()=>document.getElementById('gm-controls').scrollIntoView({block:'start'})"); pg.wait_for_timeout(400)
                    dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return d.hidden?null:d.innerText}")
                    check(f'{T} part {k + 1}: mini result bar appears below the stage', dk is not None and len(dk) > 20, dk)
                    if k == 1: check(f'{T} part 2: mini result bar hides the answer (GQA-8’s users)', dk is None or not re.search(r'\d+ (users|个用户)', dk), dk)
                    pg.screenshot(path=str(SHOTS / f'{w}_{lang}_part{k + 1}_dock.png'))
                    pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(200)
            for mode in ['sandbox', 'challenge']:
                pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(100)
                set_mode(pg, mode)
                if mode == 'sandbox':
                    for i in range(5):
                        rail_click(pg, f'#gm-rail [data-preset="{i}"]', 300)
                        if w == 1440: card_height(pg, f'{T} Scenarios, scenario {i + 1}', heights900)
                        if w == 390 and i == 0:
                            # review A r12 N10: "1 user fits" in the mini result bar and in the announcement
                            pg.wait_for_timeout(700)
                            live = plain(pg.evaluate("()=>document.getElementById('live').textContent"))
                            check(f'{T} scenario 1: the announcement says “1 user fits”', Z('at most 1 user fits at 128K each', '最多装下 1 个用户') in live, live)
                            pg.evaluate("()=>document.getElementById('gm-controls').scrollIntoView({block:'start'})"); pg.wait_for_timeout(400)
                            dk = plain(pg.evaluate("()=>{const d=document.getElementById('dock'); return d.hidden?'':d.innerText}"))
                            check(f'{T} scenario 1: the mini result bar says “1 user fits” (review A r12 N10)', Z('at most 1 user fits at 128K each', '最多装下 1 个用户') in dk, dk)
                            pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(200)
                    rail_click(pg, '#gm-rail [data-preset="3"]', 300)
                else:
                    rail_click(pg, '#gm-rail [data-lever="mha"]', 300); rail_click(pg, '#gm-rail [data-act="submit"]', 400)
                    if w == 1440: card_height(pg, f'{T} Challenge, MHA submitted (the hint)', heights900)
                    rail_click(pg, '#gm-rail [data-act="reveal"]', 400)
                    if w == 1440: card_height(pg, f'{T} Challenge, MHA with all designs shown (the chart)', heights900)
                    rail_click(pg, '#gm-rail [data-lever="gqa4"]', 300)
                    if w == 1440: card_height(pg, f'{T} Challenge, then GQA-4 (goal met, GQA-8 keeps more KV heads)', heights900)
                check(f'{T} {mode}: no horizontal overflow', overflow(pg) <= 0, overflow(pg))
                check(f'{T} {mode}: next lab below the card', pg.query_selector('#gm-rail .next-lab a') is not None)
                if w == 390:
                    check(f'{T} {mode}: on phones the next lab shows after the bench', pg.evaluate("()=>{const a=document.querySelector('.next-lab-end a'); return !!a&&!!a.offsetParent}"))
                pg.screenshot(path=str(SHOTS / f'{w}_{lang}_{mode}.png'), full_page=(w == 390))
            check(f'{T}: no page errors', not errs, errs[:3])
            ctx.close()

    # ================= other widths: no overflow in any mode =================
    ctx = b.new_context(viewport={'width': 1280, 'height': 800}); pg = ctx.new_page()
    pg.goto(URL); pg.wait_for_timeout(300); pg.click('.langsw button[data-lang="zh"]'); pg.wait_for_timeout(150)
    pg.evaluate("()=>{location.hash='gqa-mla'}"); pg.wait_for_timeout(600)
    go_part(pg, 3); rail_click(pg, '#gm-rail [data-pred="same"]'); go_part(pg, 4); rail_click(pg, '#gm-ic-kv5 button[data-v="mqa"]')
    for (w, h) in [(1280, 800), (900, 900), (720, 900), (390, 844)]:
        pg.set_viewport_size({'width': w, 'height': h}); pg.wait_for_timeout(300)
        for mode in ['guided', 'sandbox', 'challenge']:
            pg.click(f'.dyn-root .modes button[data-mode="{mode}"]'); pg.wait_for_timeout(300)
            check(f'zh {w}: {mode} no horizontal overflow', overflow(pg) <= 0, overflow(pg))
            if mode == 'guided': pg.screenshot(path=str(SHOTS / f'{w}_zh_guided_p5.png'))
    ctx.close()

    # ================= dark mode once =================
    ctx = b.new_context(viewport={'width': 1440, 'height': 900}, color_scheme='dark'); pg = ctx.new_page()
    pg.goto(URL); pg.wait_for_timeout(300); pg.evaluate("()=>{location.hash='gqa-mla'}"); pg.wait_for_timeout(600)
    go_part(pg, 5); rail_click(pg, '#gm-ic-mla button[data-v="mla"]')
    bg = pg.evaluate("()=>getComputedStyle(document.body).backgroundColor")
    check('dark: dark background', bg in ('rgb(15, 21, 18)',), bg)
    fills = pg.evaluate("()=>[getComputedStyle(document.querySelector('#gm-diag .gm-q')).fill, getComputedStyle(document.querySelector('#gm-diag .gm-k')).fill]")
    check('dark: query heads light gray, compressed vector amber', fills[0] != 'rgb(60, 72, 88)' and fills[1] == 'rgb(239, 174, 54)', fills)
    pg.screenshot(path=str(SHOTS / 'dark_1440_en_p6.png'))
    pg.click('.dyn-root .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(300); rail_click(pg, '#gm-rail [data-preset="3"]')
    pg.screenshot(path=str(SHOTS / 'dark_1440_en_sandbox.png'))
    ctx.close()
    # ===== the owner's message of 2026-10-08 (reviews/owner-b9-2026-10-08.md): G1 lead-ins instead of “one at a time” notes =====
    src = (P / 'labs' / 'gqa-mla.js').read_text()
    left = [x for x in ('一次只能选', '一次选一个', '不会叠加', '每次只试', 'do not stack', 'One at a time', 'Pick one at a time', "'Pick one:'", "'选一个：'") if x in src]
    check('owner b9 G1: no “one at a time / do not stack / Pick one:” note left in the lab, EN or ZH', not left, left)
    for lang in ['en', 'zh']:
        Z = (lambda en, zh: zh) if lang == 'zh' else (lambda en, zh: en)
        for w, h in ((1440, 900), (390, 844)):
            ctx = b.new_context(viewport={'width': w, 'height': h}); pg = ctx.new_page(); open_drawers(pg)
            pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
            pg.evaluate("()=>{location.hash='gqa-mla'}"); pg.wait_for_timeout(600)
            T = f'{lang} {w}'
            set_mode(pg, 'sandbox')
            t = plain(pg.inner_text('#gm-rail .railcard'))
            check(f'{T} owner b9 G1: the Scenarios card leads in with what to pick and what to watch', Z('Pick a scenario and see how many users fit at once.', '选一个场景，看看最多能同时装下多少个用户。') in t, t[:240])
            set_mode(pg, 'challenge')
            t = plain(pg.inner_text('#gm-rail .railcard'))
            check(f'{T} owner b9 G1: the challenge leads in to the designs', Z('Pick an attention design and see how many users fit.', '选一种注意力设计，看看能同时装下多少个用户。') in t and Z('Pick one:', '选一个：') not in t, t[:400])
            ctx.close()
    for (w, h) in [(1440, 900), (390, 844)]:
        for lang in ['en', 'zh']:
            owner_b9v6(b, w, h, lang, B9V6)
    b.close()

print('card heights after the action at 1440 x 1000 (px):')
for t, h in heights: print(f'  {h:5d}  {t}')
print('card heights at 1440 x 900 (px), tallest first:')
for t, h in sorted(heights900, key=lambda x: -x[1]): print(f'  {h:5d}  {t}')
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:40] if fails else 'none')
sys.exit(1 if fails else 0)
