"""Flow test for the MoE lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH (plus one dark run at 1440 EN):
- no page errors and no horizontal overflow in any state;
- the phone first screen (m2 template, R7.3): the parts' bar with Next, then the stage's router picture;
- the gates: part 2 (a prediction), part 3 (three tokens routed), part 4 (FP8 tried), part 5 (more than one request), part 6 (the
  same-quality dense model looked at), part 7 (a skewed router); Show me opens each gate; “Saved …” after the pick (R2.5);
- part 6, the advantage (R6.26): Mixtral next to Llama 2 13B, Llama 2 70B and a hypothetical dense 46.7B on quality (the paper's
  Table 2), arithmetic per token, weights read per decode step and memory; the result answers “where is the advantage?” and “do fewer
  active parameters hurt quality?”; the benchmark switch changes the scores; What to remember has the misconception “MoE saves nothing”;
- the scenarios' curve: moving the pointer reads out a point in a tooltip, and a click sets the batch there (R1.1);
- R1.1 and R7.8 in every state: every element whose cursor is a question mark explains itself (data-tipx, data-tip, .t or .def on it
  or an ancestor); no text overflows its capsule, button, badge or card value, in HTML (scrollWidth) or SVG (text length in its box);
- the prediction's answer (87.0 GiB) appears nowhere, in text or tooltips, before part 4; part 4 gives the verdict with the right answer;
- ▲▼ badges only after the learner's own change (R6.6), not on entering a part;
- guided → Scenarios → Challenge by the cards' buttons (R3.12); the scenarios' pressed state follows the settings (R1.8);
- the challenge: only the number of users in the card, Settings locked with a note (R4.2), the three feedback tiers, “Show the answer”,
  a brief comparison chart (the owner's review of batch 8): the user counts that meet the goal (green, up to 25 users), the initial setup
  and the pick, nothing else, and a note that counts the 3 left out; the capability card; the next-lab footer in every mode;
- no hover that only expands a number to its exact value, outside Show the math (the owner, 2026-10-07 23:37 UTC);
- the language switch keeps the part, the prediction, the settings and the challenge's answer; the hash round-trip in Scenarios;
- the mini result bar at 390 once the chart scrolls away, with a short status;
- the home tile at 1, 4, 16 and 64 requests (the product owner: show how the 47B model's speed compares with a 13B one's): the experts
  read and the time per token of Mixtral-8x7B next to a hypothetical dense 46.7B and dense 12.9B, values from window.__tzMoe, bars on
  one scale, a sentence that changes on every click (at 16 and 64 requests a second one on what MoE still saves, R6.26; at most 30
  words each and 38 in all, Chinese about as short), the tile's height the same at every setting (R7.9), a visual at most 130 px tall,
  no overflow and no broken names at 1440 and 390 px and in a 280 px tile, grayscale bars and blue experts (R5.6) in light and dark, a
  legend for every colour, four one-sentence notes in the page language (B1) that name what the times assume (FP8 weights, the
  embedding table, BF16 KV cache, 2K tokens, 60% of 3.35 TB/s; home review v4 B-08) with no "^" (A #8), the tooltips, the Formula
  chip, no memory size (R3.8), the writing rules, and the language switch keeping the setting; `python3 tests/moe_flow.py tile` runs
  only these;
- round 1: part 6 shows times (prefill and read), the KV cache note and
  the requests that fit, Mistral 7B as a faded row, DeepSeekMoE in the dense 46.7B answer; the options, the status and the result follow
  the benchmark; the curve's marker is drawn (computed display) and a 70B line stands next to it outside the challenge; a click sets the
  batch and Scenarios says what was set; the scope line is right for each part; why Next is greyed out, under the buttons and in its
  tooltip (the phone keeps the line's height); option titles on one line; the status badge stays inside the stage; part 7's waiting
  badge in percentage points; the parameter bars fill the experts of the token shown; no "part 6" left for part 7;
- the result box on screen after each action at 1440 (R3.10);
- the writing rules on every visible text and tooltip (no “·” between clauses, at most one colon per Chinese sentence, a space between
  Chinese and Latin letters or digits, full-width punctuation in Chinese, no 亿 / 万, no number over 6 digits outside the math and
  tooltips, no “toy”, 起点, 控件), and blue, amber, red and green only for what they mean (R5.6).
Screenshots: shots/moe/.
At 1440 × 900 parts 5, 6 and 8 fit the screen (polish round 4), and Next clicked with the mouse into part 7 keeps the stage below the
top bar.
Run: python3 tests/moe_flow.py [1440|390] [en|zh]  (no arguments: everything)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/moe-flow.html', '--only', 'moe'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'moe-flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'moe-flow.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'moe-flow-wrapped.html')
SH = P / 'shots' / 'moe'; SH.mkdir(parents=True, exist_ok=True)
ARGS = sys.argv[1:]
WIDTHS = [int(a) for a in ARGS if a.isdigit()] or [1440, 390]
LANGS = [a for a in ARGS if a in ('en', 'zh')] or ['en', 'zh']
TILE_ONLY = 'tile' in ARGS
errs, fails, n = [], [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


Z = lambda lang, en, zh: zh if lang == 'zh' else en


def page(b, w, h, lang, dark=False, hash_='moe'):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light')
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(500)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__moe.state()")
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText:''}", sel)


def settle(pg, ms=2000):
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(220); settle(pg)


def go_part(pg, i): click(pg, f'#moe-rail [data-go="{i}"]')


def slide(pg, sel, v):
    pg.evaluate(f"""()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}));}}"""); pg.wait_for_timeout(250); settle(pg)


def mode(pg, m): click(pg, f'#moe-app .modes button[data-mode="{m}"]')


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(1000)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def next_disabled(pg): return pg.evaluate("()=>{const b=document.querySelector('#moe-rail .step.current [data-nav=\"next\"]'); return b?b.disabled:null}")


def all_text(pg):
    return pg.evaluate("""()=>{const out=[]; for(const s of ['#moe-app','#dock']){const e=document.querySelector(s); if(e&&e.offsetParent!==null) out.push(e.innerText);}
      document.querySelectorAll('#moe-app [data-tipx], #dock [data-tipx]').forEach(e=>out.push(e.getAttribute('data-tipx'))); return out.join('\\n')}""")


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/in this toy\\b|在这个玩具里|\\btoy\\b|玩具/i,'“toy”'],
   [/Locked in|已锁定/,'“Locked in”'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万 instead of K, M, B (R6.18)'],[/个权重/,'“个权重” (R6.17)'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour (R5.7)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip')||!e.getClientRects().length) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); for(const [re,name] of bad) if(re.test(t)) out.push(name+' (tooltip): '+t.slice(0,90));});
 return out;}"""
SPACING = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), bad=/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>%][\\u4e00-\\u9fff]/;
 const shown=e=>e&&!e.closest('[data-endnotes],pre,code,.mathbody,#tip')&&e.getClientRects().length;
 const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){if(!shown(n.parentElement)) continue; const t=n.textContent.replace(/\\s+/g,' '); const m=t.match(new RegExp(bad.source,'g')); if(m) out.add(m.join('|')+' in: '+t.trim().slice(0,80));}
 root.querySelectorAll('a,.def,b,.t,em').forEach(e=>{if(!shown(e)||getComputedStyle(e).display!=='inline') return;
   const t=e.textContent; if(!t) return; const prev=e.previousSibling, next=e.nextSibling;
   const pc=prev&&prev.nodeType===3?prev.textContent.slice(-1):'', nc=next&&next.nodeType===3?next.textContent.slice(0,1):'';
   const pair=(a,b)=>a&&b&&bad.test(a+b);
   if(pair(pc,t[0])) out.add(pc+t[0]+' before: '+t.slice(0,40)); if(pair(t[t.length-1],nc)) out.add(t[t.length-1]+nc+' after: '+t.slice(-40));});
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); const m=t.match(/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>][\\u4e00-\\u9fff]/g); if(m) out.add(m.join('|')+' in tooltip: '+t.slice(0,80));});
 return [...out];}"""
PUNCT = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), bad=/[\\u4e00-\\u9fff][:;,!?]|[:;,!?] ?[\\u4e00-\\u9fff]/g;
 const shown=e=>e&&!e.closest('[data-endnotes],pre,code,.mathbody,#tip')&&e.getClientRects().length;
 const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){if(!shown(n.parentElement)) continue; const t=n.textContent.replace(/\\s+/g,' '); const m=t.match(bad); if(m) out.add(m.join('|')+' in: '+t.trim().slice(0,80));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); const m=t.match(bad); if(m) out.add(m.join('|')+' in tooltip: '+t.slice(0,80));});
 return [...out];}"""
COLONS = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), texts=[];
 root.querySelectorAll('p,li,button,label,div,span,h1,h2,h3,summary').forEach(e=>{if(e.closest('[data-endnotes],pre,code,svg,.mathbody')||!e.getClientRects().length) return; if(e.querySelector('p,li,div,ul,ol,h1,h2,h3,table')) return;
   if(![...e.childNodes].some(c=>c.nodeType===3&&c.textContent.trim())) return; texts.push(e.textContent);});
 root.querySelectorAll('[data-tipx]').forEach(e=>texts.push(e.getAttribute('data-tipx')));
 for(const t of texts){const sents=t.replace(/\\s+/g,' ').split(/(?<=[。？！])|(?<=[.?!])\\s+/);
   for(const x of sents){const k=(x.match(/：/g)||[]).length; if(k>=2) out.add(x.trim().slice(0,160));}}
 return [...out];}"""
BIGNUM = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #moe-mathbody, .num-big, #moe-drill-a, .ex-res, .connect')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/moe-ex|moe-rm|moe-pw|moe-po|moe-rw|moe-ro|moe-dk|moe-cm-w|sw w\\b|seg w\\b|sw moe-sw-out|sw moe-sw-po|sw moe-sw-past/,[amber]:/moe-rk|sw kv\\b|seg kv\\b/,[red]:/status bad|badline|badc|moe-over|warn|seg over|sw over/,[violet]:/seg oh\\b|sw oh\\b/};
 const out=[]; for(const e of document.querySelectorAll('#moe-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())&&!(p==='stroke'&&s.strokeWidth==='0px')) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
HELP = """(sel)=>{const root=document.querySelector(sel); if(!root) return []; const out=[];
 for(const e of root.querySelectorAll('*')){if(!e.getClientRects().length) continue; const s=getComputedStyle(e); if(s.cursor!=='help'||s.visibility==='hidden') continue;
   if(e.closest('[data-tipx],[data-tip],.t,.def,[data-jump]')) continue; if(e.parentElement&&getComputedStyle(e.parentElement).cursor==='help'&&!e.parentElement.closest('[data-tipx],[data-tip],.t,.def,[data-jump]')) continue;
   out.push(e.tagName+'.'+(typeof e.className==='string'?e.className:(e.className&&e.className.baseVal)||'')+' '+(e.textContent||'').trim().slice(0,40));}
 return out.slice(0,10);}"""
FITS = """(sel)=>{const root=document.querySelector(sel); if(!root) return []; const out=[];
 root.querySelectorAll('.moe-badge,.status,button,.tag,.metric .v,.moe-cm-v,.moe-pv,.moe-epv,.moe-epl,.moe-cm-n,.lv-r,.moe-rlab,.pb-dot').forEach(e=>{if(!e.getClientRects().length) return;
   if(e.scrollWidth>e.clientWidth+1&&getComputedStyle(e).overflow!=='visible'||e.scrollWidth>e.clientWidth+1&&getComputedStyle(e).whiteSpace==='nowrap') out.push('HTML '+(e.className||e.tagName)+' '+e.scrollWidth+'>'+e.clientWidth+' '+(e.textContent||'').trim().slice(0,40));});
 root.querySelectorAll('svg g').forEach(g=>{const r=g.querySelector(':scope > rect'); if(!r) return; const rb=r.getBBox(); if(!rb.width) return;
   const ts=[...g.querySelectorAll(':scope > text')]; ts.forEach(x=>{const b=x.getBBox(); if(!b.width) return; if(b.x<rb.x-0.5||b.x+b.width>rb.x+rb.width+0.5) out.push('SVG '+x.textContent+' '+Math.round(b.width)+' in '+Math.round(rb.width));});
   if(ts.length>1){const a=ts[0].getBBox(), c=ts[1].getBBox(); if(a.width&&c.width&&a.x+a.width>c.x-2) out.push('SVG overlap '+ts[0].textContent+' / '+ts[1].textContent);}});
 return out.slice(0,10);}"""
rule_hits = []
EXACT = r"""(sel)=>{const root=document.querySelector(sel); if(!root) return [];
  return [...root.querySelectorAll('[data-tipx],[data-tip],[title]')].filter(e=>!e.closest('details.math,[id$="mathbody"]'))
    .map(e=>e.getAttribute('data-tipx')||e.getAttribute('data-tip')||e.getAttribute('title')||'').filter(t=>/精确值是|Exactly|shown rounded to 3|取 3 位有效数字/.test(t)).map(t=>t.slice(0,90));}"""


def rules(pg, where, lang):
    for sel in ('#moe-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
        for h in pg.evaluate(EXACT, sel): rule_hits.append(f'{where}: a hover that only gives the exact value (the owner, 23:37 UTC): {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')
    for sel in ('#moe-app', '#dock'):
        for h in pg.evaluate(HELP, sel): rule_hits.append(f'{where}: question-mark cursor without an explanation (R1.1): {h}')
        for h in pg.evaluate(FITS, sel): rule_hits.append(f'{where}: text overflows its box (R7.8): {h}')


def badges(pg): return pg.evaluate("()=>[...document.querySelectorAll('#moe-metrics .metric .d')].map(d=>d.textContent).filter(Boolean)")


def obs_on_screen(pg):
    return pg.evaluate("""()=>{const b=document.querySelector('#moe-rail [data-obs]')||document.querySelector('#moe-insight')||document.querySelector('#moe-ch-res'); if(!b||b.hidden) return null; const r=b.getBoundingClientRect(), top=document.querySelector('.topbar').getBoundingClientRect().bottom;
      return {top:Math.round(r.top), vh:innerHeight, ok:r.top>=top-2&&r.top<innerHeight-40}}""")


FIT = """(sels)=>{const s=document.querySelector('#moe-stage'), r=s.getBoundingClientRect(), tb=document.querySelector('.topbar').getBoundingClientRect().bottom;
  const q=sels.map(x=>{const e=document.querySelector(x); if(!e||!e.getClientRects().length) return null; const b=e.getBoundingClientRect(); return [Math.round(b.top),Math.round(b.bottom)];});
  return {spare:Math.round(parseFloat(getComputedStyle(s).maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight)), scroll:s.scrollHeight-s.clientHeight, top:Math.round(r.top), q,
    out:sels.filter((x,i)=>!q[i]||q[i][0]<r.top-1||q[i][1]>r.bottom+1), shown:!!q[0]&&q[0][0]>=tb-1&&q[0][1]<=innerHeight+1};}"""


def fit_check(pg, tag, part, w, h, lang, sels):
    """polish round 4 (2026-10-16, R7.12, the owner's B4; as tests/data-pipeline_flow.py part 4): at 1440 × 900, on entering parts 5, 6
    and 8, the stage needs no scrolling inside itself and could still grow by 30 px in Chinese (its max-height minus the height its content
    and borders need; in English it only needs no scrolling), the pieces named lie inside it (part 6: the GPU-memory block and the metric
    cards, which the card's “the price is memory” points at), and the first of them, the main picture, lies on the screen"""
    if w != 1440 or h != 900: return
    fit = pg.evaluate(FIT, sels)
    check(f'{tag} part {part}: the stage fits without scrolling inside' + (', 30 px to spare' if lang == 'zh' else '') + f', {", ".join(sels)} inside it, the first on the screen (R7.12, B4)',
          fit['scroll'] <= 0 and fit['spare'] >= (30 if lang == 'zh' else 0) and not fit['out'] and fit['shown'], fit)


# polish round 4 (2026-10-17, N6): on the wide stage the batch curve is drawn shorter, which brought the ring round the point at one
# request (radius 9, at the plot's left edge) onto the time axis's “10”; the plot now starts further right of the axis's labels. The
# labels' boxes (getBBox: their advance and the font's ascent and descent) and the ring with its stroke must not meet
RING1 = """()=>{const sv=document.querySelector('#moe-curve svg'), ring=sv&&sv.querySelector('.moe-cring'); if(!ring) return {n:0, hits:['no ring']};
  const rb=ring.getBBox(), pad=(parseFloat(getComputedStyle(ring).strokeWidth)||0)/2, cx=+ring.getAttribute('cx');
  const labs=[...sv.querySelectorAll('text.moe-ct')].filter(t=>t.getAttribute('text-anchor')==='end'&&+t.getAttribute('x')<cx);
  return {n:labs.length, cx, hits:labs.filter(t=>{const q=t.getBBox(); return q.x<rb.x+rb.width+pad&&rb.x-pad<q.x+q.width&&q.y<rb.y+rb.height+pad&&rb.y-pad<q.y+q.height;}).map(t=>t.textContent)};}"""

# polish round 4 (2026-10-16, R3.2, R9.7, the owner's B4; as tests/spec-decoding_flow.py): a reader's click on Next (下一部分) scrolls the
# button into view first, which element.click() does not. Entering part 7 that way at 1440 × 900, the kit's reveal put the card's top
# under the top bar, little of the rail was left under it, and the bench's end pushed the stage up under the top bar (its top at −149 px).
# This walk uses real mouse clicks, opening each gate as a reader does, and after each Next waits for the scroll to settle (keepStage
# waits for it, then pulls the stage back)
TOPS = """()=>{const s=document.querySelector('#moe-stage').getBoundingClientRect(), bar=document.querySelector('.topbar').getBoundingClientRect().bottom,
 h=document.querySelector('#moe-stage .stage-head').getBoundingClientRect(), f=document.querySelector('#moe-flow-svg svg'), q=f&&f.getClientRects().length?f.getBoundingClientRect():null,
 c=document.querySelector('#moe-rail .step.current').getBoundingClientRect();
 return {part:window.__moe.state().step+1, top:Math.round(s.top), bar:Math.round(bar), head:Math.round(h.top), card:Math.round(c.top), fig:q?[Math.round(q.top),Math.round(q.bottom)]:null, h:innerHeight};}"""


def next_into_7(b, lang):
    tag = f'{lang} 1440 Next clicked with the mouse'
    pg = b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce')
    open_drawers(pg); pg.on('pageerror', lambda e: errs.append(f'1440 {lang} mouse: {e}'))
    pg.goto(URL); pg.wait_for_timeout(250); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate("()=>{location.hash='moe'}"); pg.wait_for_timeout(700)
    miss = []

    def mouse(sel, ms):
        try: pg.locator(sel).first.click(timeout=4000)
        except Exception as e: miss.append(f'{sel}: {str(e)[:60]}')
        pg.wait_for_timeout(ms)
    gates = {2: ['[data-pred="lt40"]'], 3: ['[data-act="route"]'] * 3, 4: ['.seg-ctl[data-key="p"] button[data-v="fp8"]'], 5: ['[data-act="bplus"]'], 6: ['[data-lever="l70"]']}
    seen = []
    for part in range(1, 7):
        for g in gates.get(part, []): mouse('#moe-rail .step.current ' + g, 300)
        mouse('#moe-rail .step.current [data-nav="next"]', 1500)
        seen.append(pg.evaluate(TOPS))
    r = seen[-1]
    check(f'{tag}: entering part 7, the stage’s top stays at or below the top bar’s bottom (its head shown), the card’s top and the router picture on the screen (R3.2, R9.7, B4)',
          not miss and r['part'] == 7 and r['top'] >= r['bar'] - 1 and r['head'] >= r['bar'] - 1 and r['bar'] - 1 <= r['card'] < r['h'] - 100 and bool(r['fig']) and r['fig'][0] >= r['bar'] - 1 and r['fig'][1] <= r['h'] + 1,
          {'last': r, 'tops': [(x['part'], x['top']) for x in seen], 'miss': miss})
    pg.close()


def why_state(pg):
    """the why line under the buttons (and on the phone's part bar) against Next's tooltip"""
    return pg.evaluate("""()=>{const nb=document.querySelector('#moe-rail .step.current [data-nav="next"]'), ws=[...document.querySelectorAll('#moe-rail .moe-why')];
      const vis=ws.filter(w=>!w.classList.contains('moe-why-off')&&w.getClientRects().length&&getComputedStyle(w).visibility!=='hidden');
      return {disabled:nb?nb.disabled:null, tip:nb?(nb.dataset.tipx||''):'', shown:vis.map(w=>w.textContent), n:ws.length,
        kept:ws.filter(w=>w.classList.contains('moe-why-off')).map(w=>({h:w.getBoundingClientRect().height, d:getComputedStyle(w).display, v:getComputedStyle(w).visibility}))}}""")


def why_ok(pg, tag, phone, gated):
    w = why_state(pg)
    if gated:
        check(f'{tag}: why Next is greyed out, under the buttons and in its tooltip', w['disabled'] is True and w['tip'] and w['shown'] and all(s == w['tip'] for s in w['shown']), w)
    else:
        check(f'{tag}: no why line once Next is on', w['disabled'] is False and not w['shown'] and not w['tip'], w)
        if phone and w['kept']:
            check(f'{tag}: the phone keeps the why line\'s height', all(k['d'] != 'none' and k['v'] == 'hidden' for k in w['kept']), w)


def titles_one_line(pg):
    """option titles stay on one line (R3.17): the title's height is one line"""
    return pg.evaluate("""()=>[...document.querySelectorAll('#moe-rail .lever .lv-t b')].filter(b=>b.getClientRects().length).map(b=>{const lh=parseFloat(getComputedStyle(b).lineHeight)||16; return {t:b.textContent, h:b.getBoundingClientRect().height, lh}}).filter(x=>x.h>x.lh*1.5)""")


def status_inside(pg):
    """the status badge's right edge inside the stage (R7.8)"""
    return pg.evaluate("""()=>{const s=document.querySelector('#moe-status').getBoundingClientRect(), st=document.querySelector('#moe-stage').getBoundingClientRect(); return {r:Math.round(s.right), stage:Math.round(st.right), ok:s.right<=st.right-2}}""")


def ok_state(pg, tag):
    check(f'{tag}: no horizontal overflow', overflow(pg) <= 0, overflow(pg))


def run_guided(b, w, h, lang, dark=False):
    tag = f'{"dark_" if dark else ""}{w}_{lang}'
    pg = page(b, w, h, lang, dark)
    phone = w <= 760
    check(f'{tag} title', text(pg, '#moe-app .labtitle') == Z(lang, 'How can a 47B model run like a 13B one?', '47B 的模型怎么跑得像 13B 一样快？'), text(pg, '#moe-app .labtitle'))
    check(f'{tag} no fidelity tag on top (owner, 2026-10-11 19:28 UTC)', not any(w in text(pg, '#moe-app .labmeta') for w in ('Fidelity', '保真度')), text(pg, '#moe-app .labmeta'))
    check(f'{tag} outcome line', text(pg, '#moe-app .lab-outcome').startswith(Z(lang, 'You will be able to:', '学完你能：')))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#moe-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        pg.evaluate(f"()=>document.querySelector('#moe-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(150)
        check(f'{tag} popover {pid} opens', pg.evaluate(f"()=>{{const p=document.getElementById('{pid}'); return !!p&&p.getClientRects().length>0}}"))
        rules(pg, f'{tag} popover {pid}', lang)
        pg.evaluate(f"()=>document.querySelector('#moe-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(120)
    if phone:   # R7.3: the parts' bar with Next, and the router picture on the first screen
        fs = pg.evaluate("""()=>{const nb=[...document.querySelectorAll('#moe-rail [data-nav="next"]')].find(b=>b.getClientRects().length&&b.getBoundingClientRect().top<innerHeight), f=document.querySelector('#moe-flow').getBoundingClientRect();
          return {next:!!nb, flowTop:Math.round(f.top), vh:innerHeight}}""")
        check(f'{tag} phone first screen: Next and the router picture', fs['next'] and fs['flowTop'] < fs['vh'] - 120, fs)
    ok_state(pg, f'{tag} part 1'); rules(pg, f'{tag} part 1', lang); shot(pg, f'{tag}_p1')
    t1 = all_text(pg)
    check(f'{tag} part 1: 87.0 GiB appears nowhere', '87.0' not in t1 and '93.4' not in t1)
    st = pg.evaluate("()=>document.querySelector('#moe-controls').textContent")
    check(f'{tag} Settings: the prefill is in part 7, with a link', Z(lang, 'a prefill (part 7', 'prefill（第 7 部分') in st and pg.evaluate("()=>!!document.querySelector('#moe-controls [data-go=\"6\"]')"), st[:200])
    check(f'{tag} Settings: the 60% was measured at batch size 1', Z(lang, 'at batch size 1', 'batch 为 1 时') in st)
    # part 2: the prediction gate
    plays = state(pg)['plays']
    go_part(pg, 1)
    check(f'{tag} R3.20 part 2: the same routing picture as part 1, so its dots do not run again', state(pg)['plays'] == plays, (plays, state(pg)['plays']))
    check(f'{tag} part 2: Next disabled before a prediction', next_disabled(pg) is True)
    why_ok(pg, f'{tag} part 2', phone, True)
    check(f'{tag} part 2: memory card hidden', pg.evaluate("()=>document.querySelector('#moe-metrics [data-k=\"mem\"] .v').textContent") == '?')
    check(f'{tag} part 2: 87.0 appears nowhere', '87.0' not in all_text(pg))
    click(pg, '#moe-rail [data-pred="lt40"]')
    check(f'{tag} part 2: Next enabled after a prediction', next_disabled(pg) is False)
    why_ok(pg, f'{tag} part 2 after the pick', phone, False)
    check(f'{tag} part 2: “Saved …”', Z(lang, 'Saved. You can still change it', '已记下') in text(pg, '#moe-rail .step.current'))
    ok_state(pg, f'{tag} part 2'); rules(pg, f'{tag} part 2', lang); shot(pg, f'{tag}_p2')
    # part 3: route tokens
    go_part(pg, 2)
    check(f'{tag} part 3: Next disabled before routing', next_disabled(pg) is True)
    check(f'{tag} part 3: 87.0 appears nowhere', '87.0' not in all_text(pg))
    plays = state(pg)['plays']
    click(pg, '#moe-rail [data-act="route"]')
    check(f'{tag} R3.20 part 3: routing a token runs its dots once, and a Replay control is on the picture', state(pg)['plays'] == plays + 1 and pg.is_visible('#moe-replay'), (plays, state(pg)['plays']))
    pg.wait_for_timeout(300); click(pg, '#moe-replay')
    check(f'{tag} R3.20 part 3: Replay runs the dots again', state(pg)['plays'] == plays + 2, state(pg)['plays'])
    check(f'{tag} part 3: one token routed shows its two experts', Z(lang, 'goes to expert', '被送到专家') in text(pg, '#moe-rail [data-obs]'), text(pg, '#moe-rail [data-obs]'))
    check(f'{tag} part 3: still gated after one token', next_disabled(pg) is True)
    why_ok(pg, f'{tag} part 3 after one token', phone, True)
    check(f'{tag} part 3: the why line counts the tokens still to route', Z(lang, 'Route 2 more tokens', '再送出 2 个 token') in why_state(pg)['tip'], why_state(pg))
    par = pg.evaluate("()=>{const rows=document.querySelectorAll('#moe-prows .moe-prow'); const r=rows[1]; const segs=[...r.querySelectorAll('.moe-pbar i')].slice(1); return {lab:r.querySelector('.moe-pl').textContent, on:segs.map((s,i)=>s.classList.contains('moe-pw')?i:-1).filter(i=>i>=0)}}")
    want = sorted(pg.evaluate("()=>window.__moe.routeTok(0).top"))
    check(f'{tag} part 3: the parameter bar fills the experts of “The”, the token just routed', par['on'] == want and '“The”' in par['lab'], (par, want))
    for _ in range(2): click(pg, '#moe-rail [data-act="route"]')
    check(f'{tag} part 3: Next enabled after three tokens', next_disabled(pg) is False)
    why_ok(pg, f'{tag} part 3 after three tokens', phone, False)
    s = state(pg); check(f'{tag} part 3: three tokens', s['rt'] == 3, s)
    check(f'{tag} part 3: two experts filled for the current token', pg.evaluate("()=>document.querySelectorAll('#moe-flow .moe-ex.on').length") == 2)
    if not phone: check(f'{tag} part 3: result on screen', (obs_on_screen(pg) or {}).get('ok'), obs_on_screen(pg))
    check(f'{tag} part 3: the scope line speaks of parameters, not decode steps', Z(lang, 'config.json', 'config.json') in text(pg, '#moe-scope') and Z(lang, 'decode steps only', '只算 decode 步') not in text(pg, '#moe-scope'), text(pg, '#moe-scope'))
    ok_state(pg, f'{tag} part 3'); rules(pg, f'{tag} part 3', lang); shot(pg, f'{tag}_p3')
    # part 4: BF16 does not fit; the verdict; FP8 fits
    go_part(pg, 3)
    s = state(pg); check(f'{tag} part 4: enters in BF16', s['p'] == 'bf16', s)
    check(f'{tag} part 4: no ▲▼ on entering', not badges(pg), badges(pg))
    ob = text(pg, '#moe-rail [data-obs]')
    check(f'{tag} part 4: 87.0 GiB and the verdict with the right answer', '87.0 GiB' in ob and Z(lang, 'The right answer is “80 to 100 GiB”', '正确答案是“80 到 100 GiB”') in ob, ob)
    check(f'{tag} part 4: status says it does not fit (red)', pg.evaluate("()=>document.querySelector('#moe-status').className").endswith('bad'))
    check(f'{tag} part 4: Next disabled until FP8', next_disabled(pg) is True)
    plays = state(pg)['plays']
    click(pg, '#moe-rail .seg-ctl[data-key="p"] button[data-v="fp8"]')
    check(f'{tag} R3.20 part 4: FP8 draws a new picture, so its dots run', state(pg)['plays'] == plays + 1, (plays, state(pg)['plays']))
    check(f'{tag} part 4: FP8 43.5 GiB fits', '43.5 GiB' in text(pg, '#moe-rail [data-obs]') and next_disabled(pg) is False)
    check(f'{tag} part 4: no amber sliver in the memory bar (no KV cache is drawn there)', pg.evaluate("()=>{const k=document.querySelector('#moe-membar .seg.kv'); return !k||k.getBoundingClientRect().width<0.5}"))
    check(f'{tag} part 4: the memory legend names the margin past the dashed line', Z(lang, 'kept free', '留空') in text(pg, '#moe-memleg'), text(pg, '#moe-memleg'))
    check(f'{tag} part 4: the arithmetic names what is multiplied (the embedding table is only looked up)', Z(lang, 'only looked up', '只查表') in text(pg, '#moe-rail .step.current'), text(pg, '#moe-rail .step.current')[:300])
    check(f'{tag} part 4: ▲▼ after the learner’s change', any('▼' in x for x in badges(pg)), badges(pg))
    if not phone: check(f'{tag} part 4: result on screen', (obs_on_screen(pg) or {}).get('ok'), obs_on_screen(pg))
    ok_state(pg, f'{tag} part 4'); rules(pg, f'{tag} part 4', lang); shot(pg, f'{tag}_p4')
    click(pg, '#moe-rail .seg-ctl[data-key="p"] button[data-v="fp8"]')   # after the ▲▼ check: a press that changes nothing clears them
    check(f'{tag} R3.20 part 4: FP8 pressed again changes nothing, so nothing replays', state(pg)['plays'] == plays + 1, (plays, state(pg)['plays']))
    pg.wait_for_timeout(1000)   # the FP8 picture's dots have run to the end
    click(pg, '#moe-rail .seg-ctl[data-key="p"] button[data-v="bf16"]'); click(pg, '#moe-rail .seg-ctl[data-key="p"] button[data-v="fp8"]')
    check(f'{tag} R3.20 part 4: FP8 → BF16 → FP8 comes back to a picture already watched, so it is drawn finished with no dots', state(pg)['plays'] == plays + 1 and state(pg)['p'] == 'fp8', (plays, state(pg)['plays']))
    # part 5: the batch
    go_part(pg, 4)
    fit_check(pg, tag, 5, w, h, lang, ['#moe-flow-svg svg', '#moe-curve svg', '#moe-metrics'])
    check(f'{tag} part 5: no ▲▼ on entering', not badges(pg), badges(pg))
    check(f'{tag} part 5: Next disabled at one request', next_disabled(pg) is True)
    check(f'{tag} part 5: one request reads 2 experts, 6.48 ms', '6.48 ms' in text(pg, '#moe-metrics') and pg.evaluate("()=>document.querySelectorAll('#moe-flow .moe-ex.on').length") == 2)
    if w > 760:
        r1 = pg.evaluate(RING1)
        check(f'{tag} part 5: at one request the ring round the point touches none of the time axis’s labels (N6)', r1['n'] >= 3 and not r1['hits'], r1)
    plays = state(pg)['plays']
    slide(pg, '#moe-ic-B-in', 8)
    s = state(pg); check(f'{tag} part 5: 8 requests', s['B'] == 8 and next_disabled(pg) is False, s)
    check(f'{tag} R3.20 part 5: 8 requests draw a new picture, so its dots still run', s['plays'] == plays + 1, (plays, s['plays']))
    ob = text(pg, '#moe-rail [data-obs]')
    check(f'{tag} part 5: result in words (7.2 of 8, about 22 ms)', '7.2' in ob and '22 ms' in ob, ob)
    check(f'{tag} part 5: cards 21.99 ms and a ▲', '21.99 ms' in text(pg, '#moe-metrics') and any('▲' in x for x in badges(pg)), (text(pg, '#moe-metrics'), badges(pg)))
    stip = pg.evaluate("()=>document.querySelector('#moe-status').dataset.tipx")
    check(f'{tag} part 5: the status tooltip explains the rule in words', Z(lang, 'skips a given expert with probability 6/8', '躲过某个专家的概率是 6/8') in stip and '10%' in stip and '7.20' in stip, stip)
    tp = pg.evaluate("()=>document.querySelector('#moe-metrics [data-k=\"tpot\"]').dataset.tipx")
    check(f'{tag} part 5: TPOT tooltip divides a 3-decimal GB value (R6.6)', '44.204 GB ÷ 2.01 TB/s = 21.99 ms' in tp, tp)
    cl = pg.evaluate("()=>window.__moe.curveLines()")
    check(f'{tag} part 5: the curve shows Llama 2 70B with its KV cache, ending where one H100 is full (9 fit)', pg.evaluate("()=>!!document.querySelector('#moe-curve .moe-c70')") and Z(lang, 'with its KV cache', '含 KV cache') in text(pg, '#moe-legend') and bool(cl and cl['l70'] and cl['l70']['B1'] == 9 and cl['l70']['t1'] > cl['l70']['t0']) and Z(lang, '9 fit', '放得下 9 个') in pg.evaluate("()=>document.querySelector('#moe-curve').textContent"), cl)
    check(f'{tag} part 5: the scope line says decode steps', Z(lang, 'decode steps only', '只算 decode 步') in text(pg, '#moe-scope'), text(pg, '#moe-scope'))
    if not phone: check(f'{tag} part 5: result on screen', (obs_on_screen(pg) or {}).get('ok'), obs_on_screen(pg))
    slide(pg, '#moe-ic-B-in', 64)
    ob = text(pg, '#moe-rail [data-obs]')
    check(f'{tag} part 5 at 64: where MoE still saves (R6.26)', 'Llama 2 70B' in ob and Z(lang, 'a fifth', '五分之一') in ob, ob)
    slide(pg, '#moe-ic-B-in', 8)
    ok_state(pg, f'{tag} part 5'); rules(pg, f'{tag} part 5', lang); shot(pg, f'{tag}_p5')
    click(pg, '#moe-rail [data-nav="back"]'); click(pg, '#moe-rail [data-nav="next"]')
    check(f'{tag} part 5: Back / Next keep the part working', state(pg)['step'] == 4)
    # the language switch keeps the part, the prediction and the settings
    other = 'zh' if lang == 'en' else 'en'
    slide(pg, '#moe-ic-B-in', 16)
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    s = state(pg); check(f'{tag} language switch keeps part 5, 16 requests, the prediction', s['step'] == 4 and s['B'] == 16 and s['pred'] == 'lt40', s)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    # part 6: the advantage, against three dense models (R6.26)
    go_part(pg, 5)
    fit_check(pg, tag, 6, w, h, lang, ['#moe-cmp', '#moe-cmp-body>.moe-cm-blk:nth-child(4)', '#moe-metrics'])
    check(f'{tag} part 6: the comparison replaces the router picture', pg.evaluate("()=>!document.querySelector('#moe-cmp').hidden&&document.querySelector('#moe-flow').hidden"))
    check(f'{tag} part 6: Next disabled before the same-quality model', next_disabled(pg) is True)
    ob = text(pg, '#moe-rail [data-obs]')
    # polish round 4: the note on the paper's Table 2 (higher than Llama 2 13B on all 12 benchmarks) moved from under the bars to the
    # sources at the page end (A3), so the check reads it there
    ends = pg.evaluate("()=>{const e=document.querySelector('#moe-fid'); return e?e.textContent:''}")
    check(f'{tag} part 6: Llama 2 13B first: same arithmetic, 70.6% against 55.6%; higher on all 12 benchmarks, in the sources at the page end', '70.6%' in ob and '55.6%' in ob and Z(lang, 'on all 12 benchmarks', '全部 12 个基准') in ends, (ob, ends[:200]))
    check(f'{tag} part 6: four models on the picture, and Mistral 7B in the quality bars only', pg.evaluate("()=>new Set([...document.querySelectorAll('#moe-cmp .moe-cm-row')].map(r=>r.dataset.m)).size") == 5 and pg.evaluate("()=>document.querySelectorAll('#moe-cmp .moe-cm-row[data-m=\"m7\"]').length") == 1)
    check(f'{tag} part 6: Llama 2 13B first: part of the gap is training (Mistral 7B, 62.5%)', 'Mistral 7B' in ob and '62.5%' in ob and 'Switch' in ob, ob)
    # round-4 reader (r4-read-models.md, must-fix 2): the result no longer brings the KV cache (GQA, not MoE) into this comparison;
    # it states the price in memory, and the KV cache of each model stays in the stage's note (F2 below)
    check(f'{tag} part 6: against Llama 2 13B the price is memory, 3.6 times the weights; no GQA in the result', Z(lang, 'The price is memory', '代价是显存') in ob and '3.6' in ob and 'GQA' not in ob, ob)
    cm = text(pg, '#moe-cmp')
    check(f'{tag} part 6: the prefill times on the arithmetic rows (F1)', all(x in cm for x in ('0.21 s', '1.14 s', '0.77 s')), cm[:400])
    check(f'{tag} part 6: the read times under the weights read (F1)', all(x in cm for x in ('6.4 ms', '6.3 → 23.2 ms', '34.2 ms')), cm[:600])
    check(f'{tag} part 6: the KV cache note under the reads (F2)', all(x in cm for x in ('107 GB', '43 GB', '17 GB')) and Z(lang, 'Weights only', '只算权重') in cm, cm[-700:])
    check(f'{tag} part 6: the requests that fit (F2)', all(Z(lang, f'{x} fit', f'放得下 {x} 个') in cm for x in (37, 106, 9)), cm[-300:])
    check(f'{tag} part 6: the scope line names the paper and weights only', Z(lang, 'Mixtral paper', 'Mixtral 论文') in text(pg, '#moe-scope') and Z(lang, 'weights only', '只算权重') in text(pg, '#moe-scope'), text(pg, '#moe-scope'))
    why_ok(pg, f'{tag} part 6 before Llama 2 70B', phone, True)
    check(f'{tag} part 6: option titles on one line (R3.17)', not titles_one_line(pg), titles_one_line(pg))
    if phone: check(f'{tag} part 6: the status badge stays inside the stage', status_inside(pg)['ok'], status_inside(pg))
    click(pg, '#moe-rail [data-lever="l70"]')
    ob = text(pg, '#moe-rail [data-obs]')
    check(f'{tag} part 6: Llama 2 70B: same quality, 5.4 times the arithmetic, the advantage in words', '69.9%' in ob and '5.4' in ob and Z(lang, 'quality per unit of arithmetic', '每单位计算换来的质量') in ob and next_disabled(pg) is False, ob)
    check(f'{tag} part 6: option pressed and its result says who computes more', pg.evaluate("()=>document.querySelector('#moe-rail [data-lever=\"l70\"]').getAttribute('aria-pressed')") == 'true' and '5.4' in text(pg, '#moe-rail [data-lever="l70"] .lv-r') and 'Mixtral' in text(pg, '#moe-rail [data-lever="l70"] .lv-r'))
    check(f'{tag} part 6: Llama 2 70B in time: prefill and decode', '0.21 s' in ob and '1.14 s' in ob and '23 ms' in ob and '34 ms' in ob, ob)
    why_ok(pg, f'{tag} part 6 after Llama 2 70B', phone, False)
    if phone: check(f'{tag} part 6: the status badge stays inside the stage (70B)', status_inside(pg)['ok'], status_inside(pg))
    click(pg, '#moe-rail [data-lever="d47"]')
    ob = text(pg, '#moe-rail [data-obs]')
    check(f'{tag} part 6: dense 46.7B is marked hypothetical', Z(lang, 'hypothetical', '假想') in ob and '3.7' in ob, ob)
    check(f'{tag} part 6: dense 46.7B answers “would it score higher?” with DeepSeekMoE (F3)', 'DeepSeekMoE' in ob and 'strict upper bound' in ob and '100B' in ob, ob)
    check(f'{tag} part 6: the margin depends on the MoE design; a top-2 MoE like Mixtral’s stays clearly below (N5)', Z(lang, 'depends on the MoE design', '取决于 MoE 的设计') in ob and 'top-2' in ob and Z(lang, 'a little quality', '换一点质量') not in ob, ob)
    if phone: check(f'{tag} part 6: the status badge stays inside the stage (dense)', status_inside(pg)['ok'], status_inside(pg))
    click(pg, '#moe-cmp .seg-ctl[data-key="bench"] button[data-v="1"]')
    check(f'{tag} part 6: the benchmark switch shows GSM8K (74.4%)', '74.4%' in text(pg, '#moe-cmp') and 'GSM8K' in text(pg, '#moe-cmp'), text(pg, '#moe-cmp')[:200])
    click(pg, '#moe-rail [data-lever="l70"]')
    ob = text(pg, '#moe-rail [data-obs]')
    check(f'{tag} part 6: on GSM8K the 70B result puts the scores beside the benchmark (N2)', Z(lang, 'on GSM8K (69.6% against 74.4%) and about the same over all 12 benchmarks, yet', '低（69.6% 对 74.4%），12 个基准总体上差不多') in ob and 'though' not in ob, ob[:300])
    click(pg, '#moe-rail [data-lever="l13"]')
    ob = text(pg, '#moe-rail [data-obs]'); lv = text(pg, '#moe-rail [data-lever="l13"] .lv-r'); stt = text(pg, '#moe-status')
    check(f'{tag} part 6: the gap over Mistral 7B is not credited to MoE alone (N3)', Z(lang, 'not all of that is MoE either', '不全是 MoE 的功劳') in ob and '1.8' in ob and Z(lang, 'The rest is what the experts add', '其余的才是专家带来的') not in ob, ob)
    check(f'{tag} part 6: the option, the status and the result follow the benchmark (F11)', '40.1' in lv and 'GSM8K' in lv and 'GSM8K' in stt and '40.1' in stt and '74.4%' in ob and '34.3%' in ob and '50.0%' in ob, (lv, stt, ob[:200]))
    if phone: check(f'{tag} part 6: the status badge stays inside the stage (GSM8K)', status_inside(pg)['ok'], status_inside(pg))
    if not phone: check(f'{tag} part 6: result on screen', (obs_on_screen(pg) or {}).get('ok'), obs_on_screen(pg))
    ok_state(pg, f'{tag} part 6'); rules(pg, f'{tag} part 6', lang); shot(pg, f'{tag}_p6')
    # part 7: the skewed router
    go_part(pg, 6)
    check(f'{tag} part 7: Next disabled while balanced', next_disabled(pg) is True)
    why_ok(pg, f'{tag} part 7 while balanced', phone, True)
    click(pg, '#moe-rail [data-lever="q"]')
    ob = text(pg, '#moe-rail [data-obs]')
    check(f'{tag} part 7: a quarter → 2× as long', '2,048' in ob and next_disabled(pg) is False, ob)
    check(f'{tag} part 7: option pressed', pg.evaluate("()=>document.querySelector('#moe-rail [data-lever=\"q\"]').getAttribute('aria-pressed')") == 'true')
    click(pg, '#moe-rail [data-lever="h"]')
    check(f'{tag} part 7: half → 4× as long, 2.92 ms', '2.92 ms' in text(pg, '#moe-metrics'), text(pg, '#moe-metrics'))
    idle = pg.evaluate("()=>document.querySelector('#moe-metrics [data-k=\"idle\"] .d').textContent")
    check(f'{tag} part 7: the waiting badge in percentage points (F16)', Z(lang, '25 pts', '25 个百分点') in idle and '%' not in idle, idle)
    check(f'{tag} part 7: the scope line: one layer, 8 GPUs, a prefill', Z(lang, '8 GPUs during a 4,096-token prefill', '8 张 GPU 上 prefill 4,096 个 token') in text(pg, '#moe-scope'), text(pg, '#moe-scope'))
    check(f'{tag} part 7: option titles on one line (R3.17)', not titles_one_line(pg), titles_one_line(pg))
    check(f'{tag} part 7: the math names part 7', Z(lang, 'Part 7, one MoE layer', '第 7 部分，一个 MoE 层') in pg.evaluate("()=>document.querySelector('#moe-mathbody').textContent"))
    fitted = pg.evaluate("()=>{const g=document.querySelector('#moe-flow .moe-chip'), r=g.querySelector('rect'), x=g.querySelector('text'); return x.getComputedTextLength()<=(+r.getAttribute('width'))-8}")
    check(f'{tag} part 7: the “4,096 tokens” capsule holds its text', fitted)
    if not phone: check(f'{tag} part 7: result on screen', (obs_on_screen(pg) or {}).get('ok'), obs_on_screen(pg))
    ok_state(pg, f'{tag} part 7'); rules(pg, f'{tag} part 7', lang); shot(pg, f'{tag}_p7')
    # part 8
    go_part(pg, 7)
    fit_check(pg, tag, 8, w, h, lang, ['#moe-flow-svg svg', '#moe-curve svg', '#moe-metrics'])
    rem = text(pg, '#moe-rail .step.current')
    check(f'{tag} part 8: four misconceptions, “MoE saves nothing” among them', rem.count('✗') == 4 and Z(lang, 'MoE saves nothing', 'MoE 什么也没省') in rem, rem)
    check(f'{tag} part 8: the next-lab footer', pg.evaluate("()=>!!document.querySelector('#moe-rail .next-lab')"))
    check(f'{tag} part 8: a small picture and a part link for each point (F18, O12)', pg.evaluate("()=>[...document.querySelectorAll('#moe-rail .moe-rm li')].filter(li=>li.querySelector('.moe-rm-pic')&&li.querySelector('[data-go]')).length") == 4)
    check(f'{tag} part 8: the misconception says what a big batch removes (B-O6)', Z(lang, 'decode-time saving in weight reads', '在 decode 时读权重的节省') in rem, rem)
    ok_state(pg, f'{tag} part 8'); rules(pg, f'{tag} part 8', lang); shot(pg, f'{tag}_p8')
    # Show me on each gated part (a fresh page: nothing observed yet)
    pg2 = page(b, w, h, lang)
    for i, key in ((2, 'rt'), (3, 'p'), (4, 'B'), (5, 'cmp'), (6, 'skew')):
        pg2.evaluate(f"()=>document.querySelector('#moe-rail [data-go=\"{i}\"]').click()"); pg2.wait_for_timeout(250)
        if i >= 2 and not state(pg2)['pred']: pass
        sm = pg2.evaluate("()=>{const b=document.querySelector('#moe-rail .step.current [data-nav=\"showme\"]'); if(!b||b.hidden) return false; b.click(); return true}"); pg2.wait_for_timeout(300)
        s = state(pg2)
        check(f'{tag} part {i + 1}: Show me opens the gate', sm and next_disabled(pg2) is False, (sm, s))
    pg2.close()
    # guided → Scenarios by the last card's button (R3.12)
    click(pg, '#moe-rail [data-mode-go="sandbox"]')
    check(f'{tag} “See the scenarios” opens Scenarios', state(pg)['mode'] == 'sandbox')
    run_sandbox(pg, tag, lang, phone)
    pg.close()


def run_sandbox(pg, tag, lang, phone):
    ok_state(pg, f'{tag} scenarios'); rules(pg, f'{tag} scenarios', lang)
    for i in range(5):
        click(pg, f'#moe-rail [data-preset="{i}"]')
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#moe-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        check(f'{tag} scenario {i}: pressed alone', pressed.count('true') == 1 and pressed[i] == 'true', pressed)
        ins = text(pg, '#moe-insight')
        check(f'{tag} scenario {i}: a result', len(ins) > 40, ins)
        if not phone: check(f'{tag} scenario {i}: result on screen', (obs_on_screen(pg) or {}).get('ok'), obs_on_screen(pg))
        ok_state(pg, f'{tag} scenario {i}'); rules(pg, f'{tag} scenario {i}', lang)
        if i == 2: check(f'{tag} scenario 3: where MoE still saves (R6.26)', 'Llama 2 70B' in ins and '1.5' in ins, ins)
        if i in (2, 4): shot(pg, f'{tag}_scen{i}')
    s = state(pg); check(f'{tag} scenario 5 is BF16 and does not fit', s['p'] == 'bf16' and pg.evaluate("()=>document.querySelector('#moe-status').className").endswith('bad'), s)
    # the curve (R1.1): the pointer reads out a point, a click sets the batch there
    click(pg, '#moe-rail [data-preset="2"]')
    pg.evaluate("()=>document.querySelector('#moe-curve').scrollIntoView({block:'center'})"); pg.wait_for_timeout(300)
    bb = pg.evaluate("()=>{const r=document.querySelector('#moe-curve svg').getBoundingClientRect(); return {x:r.left, y:r.top, w:r.width, h:r.height}}")
    pg.mouse.move(bb['x'] + bb['w'] * 0.3, bb['y'] + bb['h'] * 0.5); pg.wait_for_timeout(150); pg.mouse.move(bb['x'] + bb['w'] * 0.3 + 2, bb['y'] + bb['h'] * 0.5); pg.wait_for_timeout(250)
    tipt = pg.evaluate("()=>{const t=document.getElementById('tip'); return t&&!t.hidden?t.innerText:''}")
    check(f'{tag} curve: a readout on hover', (Z(lang, 'At ', '时') in tipt) and 'ms' in tipt and ('GB' in tipt), tipt)
    mk = pg.evaluate("()=>{const m=document.querySelector('#moe-curve .moe-cmark'); const r=m.getBoundingClientRect(); return {d:getComputedStyle(m).display, h:r.height}}")
    check(f'{tag} curve: the marker is drawn where the pointer is (computed display)', mk['d'] != 'none' and mk['h'] > 20, mk)
    import re as _re
    mnum = _re.search(r'(\d+)', tipt.replace(',', ''))
    pg.mouse.click(bb['x'] + bb['w'] * 0.3 + 2, bb['y'] + bb['h'] * 0.5); pg.wait_for_timeout(400)
    check(f'{tag} curve: a click sets the batch there', mnum is not None and state(pg)['B'] == int(mnum.group(1)), (tipt, state(pg)['B']))
    ins = text(pg, '#moe-insight')
    check(f'{tag} curve: Scenarios says what was just set (R1.9)', Z(lang, 'You set', '你设的是') in ins and Z(lang, 'matches none', '都不一样') in ins, ins)
    tipt2 = pg.evaluate("()=>{const t=document.getElementById('tip'); return t&&!t.hidden?t.innerText:''}")
    check(f'{tag} curve: after the click the readout says it is set', Z(lang, 'Now set to', '现在设的就是') in tipt2, tipt2)
    mk = pg.evaluate("()=>{const c=document.querySelector('#moe-curve'); c.dispatchEvent(new MouseEvent('mouseleave',{bubbles:false})); const m=c.querySelector('.moe-cmark'); return getComputedStyle(m).display}")
    check(f'{tag} curve: the marker goes away when the pointer leaves', mk == 'none', mk)
    check(f'{tag} curve: the pointer cursor, since a click acts', pg.evaluate("()=>getComputedStyle(document.querySelector('#moe-curve')).cursor") == 'pointer')
    # N10: Llama 2 70B takes at most 4,096 tokens: no line at 16K (scenario 4) or 4K, a line at 2K
    click(pg, '#moe-rail [data-preset="3"]')
    c70 = lambda: (pg.evaluate("()=>!!document.querySelector('#moe-curve .moe-c70')"), (pg.evaluate("()=>window.__moe.curveLines()") or {}).get('l70'), text(pg, '#moe-legend'))
    has, l70d, leg = c70()
    check(f'{tag} scenario 4 (16K): no Llama 2 70B line, and the legend says why (N10)', not has and l70d is None and '4,096' in leg, (has, l70d, leg[-160:]))
    for idx, want in ((3, False), (2, True)):
        slide(pg, '#moe-ctl-c-in', idx)
        has, l70d, leg = c70()
        check(f'{tag} context {[512, 1024, 2048, 4096][idx]}: 70B line {"drawn" if want else "not drawn"} (N10)', has == want and (l70d is not None) == want and (('4,096' in leg) != want), (has, l70d, leg[-160:]))
    click(pg, '#moe-rail [data-preset="2"]')
    # N6: sweep the batch; no label sits under the current point's ring
    COLL = """()=>{const sv=document.querySelector('#moe-curve svg'); if(!sv) return ['no svg']; const ring=sv.querySelector('.moe-cring'); if(!ring) return ['no ring'];
      const rb=ring.getBBox(), hit=(a,b)=>a.x<b.x+b.width&&b.x<a.x+a.width&&a.y<b.y+b.height&&b.y<a.y+a.height;
      return [...sv.querySelectorAll('text')].filter(t=>/all 8 experts|全部 8 个专家|Llama 2 70B|ms$/.test(t.textContent.trim())&&!t.classList.contains('moe-cax')).filter(t=>hit(t.getBBox(),rb)).map(t=>t.textContent);}"""
    # N11: the value label touches no line (thick, dotted, 70B) and no other label; sampled along each line
    TOUCH = """()=>{const sv=document.querySelector('#moe-curve svg'); const lab=sv&&sv.querySelector('.moe-clab'); if(!lab) return ['no label']; const b=lab.getBBox(), m=0.5, out=[];
      const inB=(x,y)=>x>=b.x-m&&x<=b.x+b.width+m&&y>=b.y-m&&y<=b.y+b.height+m;
      sv.querySelectorAll('path.moe-cmoe,path.moe-cref,line.moe-c70').forEach(el=>{const L=el.getTotalLength(); for(let s=0;s<=L;s+=1){const p=el.getPointAtLength(s); if(inB(p.x,p.y)){out.push(el.getAttribute('class')); break;}}});
      sv.querySelectorAll('text').forEach(t=>{if(t===lab||t.classList.contains('moe-cax')) return; const q=t.getBBox(); if(q.width&&b.x<q.x+q.width&&q.x<b.x+b.width&&b.y<q.y+q.height&&q.y<b.y+b.height) out.push(t.textContent);});
      return out;}"""
    for B in (20, 28, 35, 53, 64, 71, 74, 76, 88, 100):
        slide(pg, '#moe-ctl-B-in', B)
        hits = pg.evaluate(COLL)
        check(f'{tag} curve at {B} requests: no label under the ring (N6)', hits == [], hits)
        hits = pg.evaluate(TOUCH)
        check(f'{tag} curve at {B} requests: the value label touches no line or label (N11)', hits == [], hits)
    for idx, B in ((4, 10), (5, 6), (6, 5)):   # 8K, 16K, 32K, where the dotted line used to cross the label
        slide(pg, '#moe-ctl-c-in', idx); slide(pg, '#moe-ctl-B-in', B)
        hits = pg.evaluate(TOUCH)
        check(f'{tag} curve at context index {idx}, {B} requests: the value label touches no line or label (N11)', hits == [], hits)
    click(pg, '#moe-rail [data-preset="2"]')
    # the pressed state follows the settings (R1.8): change the batch under Settings
    click(pg, '#moe-rail [data-preset="1"]')
    slide(pg, '#moe-ctl-B-in', 9)
    pressed = pg.evaluate("()=>[...document.querySelectorAll('#moe-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} no scenario pressed after a manual change', 'true' not in pressed, pressed)
    check(f'{tag} hash carries the setup', pg.evaluate("()=>location.hash") == '#moe~fp8_2048_9', pg.evaluate("()=>location.hash"))
    click(pg, '#moe-rail [data-mode-go="challenge"]')
    check(f'{tag} “Take the challenge” opens the challenge', state(pg)['mode'] == 'challenge')
    run_challenge(pg, tag, lang, phone)


def run_challenge(pg, tag, lang, phone):
    s = state(pg); check(f'{tag} challenge: FP8, 4K, 1 user', s['p'] == 'fp8' and s['c'] == 4096 and s['B'] == 1, s)
    check(f'{tag} challenge: Settings locked with a note', pg.evaluate("()=>document.querySelector('#moe-controls').classList.contains('locked')&&!document.querySelector('#moe-lock').hidden"))
    check(f'{tag} challenge: only the users slider in the card', pg.evaluate("()=>document.querySelectorAll('#moe-rail .railcard input[type=range]').length") == 1)
    check(f'{tag} challenge: the goal in full', '30 ms' in text(pg, '#moe-rail .railcard') and 'Mixtral' in text(pg, '#moe-rail .railcard'))
    check(f'{tag} challenge: the stage speaks of users (O5)', Z(lang, 'user decoding together', '个用户一起 decode') in text(pg, '#moe-st-sub'), text(pg, '#moe-st-sub'))
    check(f'{tag} challenge: no Llama 2 70B line on the curve', not pg.evaluate("()=>!!document.querySelector('#moe-curve .moe-c70')"))
    check(f'{tag} challenge: capability card and next lab', pg.evaluate("()=>!!document.querySelector('#moe-rail [data-exits]')&&!!document.querySelector('#moe-rail .next-lab')"))
    ok_state(pg, f'{tag} challenge'); rules(pg, f'{tag} challenge', lang); shot(pg, f'{tag}_ch0')
    # tier 3: not met
    slide(pg, '#moe-ic-chB-in', 30); click(pg, '#moe-rail [data-act="submit"]')
    res = text(pg, '#moe-ch-res')
    check(f'{tag} challenge: 30 users is not met, with a hint', Z(lang, 'Not yet', '还没达到') in res and Z(lang, 'Hint', '提示') in res, res)
    check(f'{tag} challenge: “Show the answer” before the answer', pg.evaluate("()=>!!document.querySelector('#moe-ch-res [data-act=\"reveal\"]')")
          and text(pg, '#moe-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), text(pg, '#moe-ch-res'))
    check(f'{tag} challenge: no chart before “Show the answer”', not pg.evaluate("()=>!!document.querySelector('#moe-ch-res .cmp')"))
    click(pg, '#moe-ch-res [data-act="reveal"]')
    cr = pg.evaluate("""()=>{const rows=[...document.querySelectorAll('#moe-ch-res > .cmp .cmp-row')], tg=r=>!!r.querySelector('.cmp-tag,.cmp-you');
      return {n:rows.length, ok:rows.filter(r=>r.classList.contains('ok')).length, me:rows.filter(r=>r.classList.contains('me')).length,
        stray:rows.filter(r=>!r.classList.contains('ok')&&!r.classList.contains('me')&&!tg(r)).length, note:(document.querySelector('#moe-ch-res .cmp-note')||{}).textContent||'',
        title:(document.querySelector('#moe-ch-res .cmp-title')||{}).textContent||''}}""")
    check(f'{tag} challenge, the answer for 30 users: the 5 counts that meet the goal (the initial setup among them) and my pick, nothing else', cr['n'] == 6 and cr['ok'] == 5 and cr['me'] == 1 and cr['stray'] == 0, cr)
    check(f'{tag} challenge: the note counts the 3 counts left out; the title no longer says “by the number of users”', cr['note'] == Z(lang, 'The other 3 user counts take over 30 ms per token and are not shown.', '另外 3 种用户数下，每个 token 都超过 30 ms，图里没有列出。')
          and not re.search(r'by the number of users|不同用户数', cr['title']), (cr['note'], cr['title']))
    # tier 2: met, but more fit
    slide(pg, '#moe-ic-chB-in', 10)
    res = text(pg, '#moe-ch-res')
    check(f'{tag} challenge: 10 users met, more fit (25)', Z(lang, 'But more users fit', '还能多服务') in res and '25' in res, res)
    # tier 1: met and the most
    slide(pg, '#moe-ic-chB-in', 25)
    res = text(pg, '#moe-ch-res')
    check(f'{tag} challenge: 25 users is the most', Z(lang, 'And that is the most', '这已经是最多') in res, res)
    check(f'{tag} challenge: says where MoE saves (Llama 2 70B cannot take 4K-token users; its weights alone take 34 ms)', 'Llama 2 70B' in res and '4,096' in res and '34 ms' in res, res[-400:])
    rows = pg.evaluate("()=>[...document.querySelectorAll('#moe-ch-res .cmp-row')].map(r=>[r.classList.contains('ok'),r.querySelector('.cmp-lab span').textContent])")
    check(f'{tag} challenge: chart, green up to 25 users', rows and all(ok == (int(lab.split()[0].replace(',', '')) <= 25) for ok, lab in rows), rows)
    cr = pg.evaluate("""()=>{const rows=[...document.querySelectorAll('#moe-ch-res > .cmp .cmp-row')], tg=r=>!!r.querySelector('.cmp-tag,.cmp-you');
      return {n:rows.length, ok:rows.filter(r=>r.classList.contains('ok')).length, me:rows.filter(r=>r.classList.contains('me')).length,
        stray:rows.filter(r=>!r.classList.contains('ok')&&!r.classList.contains('me')&&!tg(r)).length, note:(document.querySelector('#moe-ch-res .cmp-note')||{}).textContent||'',
        title:(document.querySelector('#moe-ch-res .cmp-title')||{}).textContent||''}}""")
    check(f'{tag} challenge: the brief chart holds the 5 counts that meet the goal (my pick 25 and the initial setup among them), nothing else; 3 left out', cr['n'] == 5 and cr['ok'] == 5 and cr['me'] == 1 and cr['stray'] == 0 and '3' in cr['note'], cr)
    vals = pg.evaluate("()=>[...document.querySelectorAll('#moe-ch-res .cmp-val')].map(v=>({t:v.textContent, h:Math.round(v.getBoundingClientRect().height), r:Math.round(v.getBoundingClientRect().right), card:Math.round(document.querySelector('#moe-ch-res').getBoundingClientRect().right)}))")
    check(f'{tag} challenge: the chart’s values stay on one line and inside the card (F21)', vals and all(v['h'] <= 20 and v['r'] <= v['card'] for v in vals), vals)
    ok_state(pg, f'{tag} challenge answered'); rules(pg, f'{tag} challenge answered', lang); shot(pg, f'{tag}_ch1', full=phone)
    # the language switch keeps the challenge's answer
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    s = state(pg); check(f'{tag} language switch keeps the challenge', s['mode'] == 'challenge' and s['B'] == 25 and s['chSubmitted'], s)
    check(f'{tag} language switch keeps the capability card', pg.evaluate("()=>!!document.querySelector('#moe-rail [data-exits]')"))
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)


TILE = '#tile-moe'
TILE_COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), others=['--c-kv','--c-oom','--c-overhead','--c-ok'].map(v=>toRGB(get(v))), out=[];
 for(const e of document.querySelectorAll('#tile-moe *')){if(!e.getClientRects().length) continue; const s=getComputedStyle(e), cls=(e.getAttribute('class')||'')+' '+(e.parentElement.getAttribute('class')||'');
   for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; if(p==='borderTopColor'&&s.borderTopWidth==='0px') continue; if(p==='color'&&!e.textContent.trim()) continue;
     if(v===blue&&!e.closest('.tz-moe-ex')&&!e.matches('.tz-leg .sw.w, .tz-leg .sw.tz-moe-sw-out')) out.push('blue on '+e.tagName+'.'+cls.trim()+' '+p);
     if(others.includes(v)) out.push('amber, red, violet or green on '+e.tagName+'.'+cls.trim()+' '+p);}}
 return out.slice(0,10);}"""
TILE_STATE = """()=>{const t=document.querySelector('#tile-moe'), r=t.getBoundingClientRect(), B=+t.querySelector('.seg-ctl button[aria-pressed="true"]').dataset.v;
 const lh=e=>parseFloat(getComputedStyle(e).lineHeight)||parseFloat(getComputedStyle(e).fontSize)*1.4;
 return {B, pressed:[...t.querySelectorAll('.seg-ctl button')].filter(b=>b.getAttribute('aria-pressed')==='true').map(b=>b.dataset.v),
  read:t.querySelector('.tz-read').innerText, names:[...t.querySelectorAll('.tz-moe-n')].map(x=>x.textContent), vals:[...t.querySelectorAll('.tz-moe-v')].map(x=>x.textContent),
  bars:[...t.querySelectorAll('.tz-moe-tr')].map(x=>x.querySelector('i').getBoundingClientRect().width/x.getBoundingClientRect().width*100),
  track:Math.min(...[...t.querySelectorAll('.tz-moe-tr')].map(x=>x.getBoundingClientRect().width)),
  ex:[...t.querySelectorAll('.tz-moe-ex span')].reduce((a,s)=>{const c=getComputedStyle(s), inner=s.getBoundingClientRect().width-parseFloat(c.borderLeftWidth)-parseFloat(c.borderRightWidth); return a+s.querySelector('i').getBoundingClientRect().width/inner;},0),
  visH:t.querySelector('.tz-moe-vis').getBoundingClientRect().height, tileW:r.width, tileH:r.height,
  out:[...t.querySelectorAll('*')].filter(e=>e.getClientRects().length&&e.getBoundingClientRect().right>r.right+0.5).map(e=>e.className||e.tagName).slice(0,5),
  broken:[...t.querySelectorAll('.tz-moe-n, .tz-moe-v, .tz-moe-cap, .tz-read .tz-moe-nw, .tz-leg > span')].filter(e=>e.getClientRects().length>1||e.getBoundingClientRect().height>lh(e)*1.5).map(e=>e.textContent),
  hook:window.__tzMoe.bars(B), sent:window.__tzMoe.read(B), T:window.__tzMoe.step(B).T, aria:t.querySelector('.tz-moe-vis').getAttribute('aria-label')}}"""
plain = lambda s: re.sub(r'<[^>]+>', '', s or '').replace(' ', ' ').strip()
zh_units = lambda t: len(re.findall(r'[一-鿿]', t)) + len(re.findall(r'[A-Za-z0-9%]+', t))   # CJK characters + Latin words and numbers


def zh_rules(t):
    """the Chinese writing rules on hidden text (notes, tooltips): one colon per sentence, spaces between Chinese and Latin, full-width punctuation"""
    bad = [f'two colons: {x[:60]}' for x in re.split(r'(?<=[。？！])', t) if x.count('：') >= 2]
    if re.search(r'[一-鿿][A-Za-z0-9]|[A-Za-z0-9%][一-鿿]', t): bad.append('spacing: ' + t[:60])
    if re.search(r'[一-鿿][:;,!?]|[:;,!?] ?[一-鿿]', t): bad.append('half-width punctuation: ' + t[:60])
    return bad


def tile_checks(b, w, h, lang, dark=False, narrow=False):
    """the home tile: does it answer “How can a 47B model run like a 13B one?” at every setting, readably?"""
    tag = f'{"dark_" if dark else ""}{w}_{lang}{"_tile280" if narrow else ""} tile'
    pg = page(b, w, h, lang, dark, hash_='map~models')
    if not pg.query_selector(TILE):
        check(f'{tag}: on the map', False); pg.close(); return
    if narrow:   # the narrowest tile the contract allows
        pg.evaluate("()=>{const t=document.querySelector('#tile-moe'); t.style.width='280px'; t.style.maxWidth='280px'}"); pg.wait_for_timeout(150)
    pg.evaluate("()=>document.querySelector('#tile-moe').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
    names = Z(lang, ['Dense 46.7B', 'Mixtral‑8x7B', 'Dense 12.9B'], ['稠密 46.7B', 'Mixtral‑8x7B', '稠密 12.9B'])
    prev = None; heights = []
    for B in (1, 4, 16, 64):
        pg.click(f'{TILE} .seg-ctl button[data-v="{B}"]'); pg.wait_for_timeout(450)
        st = pg.evaluate(TILE_STATE)
        rd = plain(st['read'])
        check(f'{tag} {B}: the button is pressed, the sentence is the hook’s and changed', st['pressed'] == [str(B)] and rd == plain(st['sent']) and rd != prev, (st['pressed'], rd, prev))
        prev = rd; heights.append(st['tileH'])
        n_s = 2 if B in (16, 64) else 1    # 16 and 64 requests: a second sentence on what MoE still saves (R6.26)
        if lang == 'en':
            ss = [x for x in re.split(r'(?<=\.) ', rd) if x]
            check(f'{tag} {B}: {n_s} sentence(s), each at most 30 words, at most 38 in all (R11.5)', len(ss) == n_s and all(len(x.split()) <= 30 for x in ss) and len(rd.split()) <= 38 and rd.endswith('.'), ([len(x.split()) for x in ss], rd))
        else:
            ss = [x for x in re.split(r'(?<=。)', rd) if x]
            check(f'{tag} {B}: {n_s} sentence(s), about as short as the English (each at most 45 units, at most 68 in all)', len(ss) == n_s and all(zh_units(x) <= 45 for x in ss) and zh_units(rd) <= 68 and rd.endswith('。'), ([zh_units(x) for x in ss], rd))
        if n_s == 2:
            check(f'{tag} {B}: the second sentence says MoE still saves arithmetic, not (or hardly any) memory reads, with 12.9B parameters per token',
                  Z(lang, 'It still saves arithmetic', '它省下的仍是计算') in rd and Z(lang, 'hardly any memory reads' if B == 16 else 'not memory reads', '读显存几乎没省' if B == 16 else '不是读显存') in rd and '12.9B' in ss[1], rd)
        check(f'{tag} {B}: the sentence names {"the 1 request" if B == 1 else f"the {B} requests"}, Mixtral‑8x7B (unbroken name) and a dense model', rd.startswith(str(B) + ' ') and 'Mixtral‑8x7B' in rd and Z(lang, 'dense ', '稠密 ') in rd, rd)
        check(f'{tag} {B}: three bars, top to bottom dense 46.7B, Mixtral, dense 12.9B, with the hook’s times', st['names'] == names and st['vals'] == [x['text'] for x in st['hook']] and [x['id'] for x in st['hook']] == ['d47', 'mx', 'd13'], (st['names'], st['vals']))
        check(f'{tag} {B}: the bars’ lengths on the shared scale', all(abs(a_ - x['w']) < 1.5 for a_, x in zip(st['bars'], st['hook'])), (st['bars'], [x['w'] for x in st['hook']]))
        check(f'{tag} {B}: the experts filled add up to the experts read ({st["T"]:.2f})', abs(st['ex'] - st['T']) < 0.05, (st['ex'], st['T']))
        check(f'{tag} {B}: the visual is at most 130 px tall', st['visH'] <= 130, st['visH'])
        check(f'{tag} {B}: nothing sticks out of the tile, no horizontal scroll, no name, value, caption or legend item broken over two lines', not st['out'] and overflow(pg) <= 0 and not st['broken'], (st['out'], overflow(pg), st['broken']))
        check(f'{tag} {B}: the picture is read aloud with all three times', all(x['text'] in st['aria'] for x in st['hook']), st['aria'])
    check(f'{tag}: the bars keep a readable track ({st["track"]:.0f} px)', st['track'] >= 80, st['track'])
    check(f'{tag}: the tile keeps its height at every setting (the sentence reserves the longest one’s lines, R7.9)', max(heights) - min(heights) < 0.5, heights)
    if narrow:
        pg.close(); return
    leg = pg.evaluate("()=>[...document.querySelectorAll('#tile-moe .tz-leg > span')].map(s=>s.innerText.trim())")
    check(f'{tag}: the legend names every colour: experts read and not read, Mixtral, the hypothetical dense models', leg == Z(lang, ['experts read', 'not read', 'Mixtral‑8x7B', 'hypothetical dense'], ['读到的专家', '没读的', 'Mixtral‑8x7B', '假想的稠密模型']), leg)
    cols = pg.evaluate("""()=>{const t=document.querySelector('#tile-moe'), bg=s=>getComputedStyle(t.querySelector(s)).backgroundColor;
      const tok=v=>{const d=document.createElement('i'); d.style.color=getComputedStyle(document.documentElement).getPropertyValue(v).trim(); document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
      return {mx:bg('.tz-moe-tr.tz-moe-mx i'), d47:bg('.tz-moe-tr.tz-moe-d47 i'), d13:bg('.tz-moe-tr.tz-moe-d13 i'), ex:bg('.tz-moe-ex i'), swmx:bg('.tz-moe-sw-mx'), swdn:bg('.tz-moe-sw-dn'), sww:bg('.tz-leg .sw.w'),
        out:getComputedStyle(t.querySelector('.tz-moe-sw-out')).borderTopColor, ink:tok('--ink'), ink3:tok('--ink-3'), blue:tok('--c-weights')}}""")
    check(f'{tag}: Mixtral’s bar is --ink, the dense ones --ink-3, the experts blue, each as in the legend (R5.6, R5.7)',
          cols['mx'] == cols['swmx'] == cols['ink'] and cols['d47'] == cols['d13'] == cols['swdn'] == cols['ink3'] and cols['ex'] == cols['sww'] == cols['out'] == cols['blue'] and cols['ink'] != cols['ink3'], cols)
    hits = pg.evaluate(TILE_COLORS)
    check(f'{tag}: blue only on the experts, no amber, red, violet or green (R5.6)', not hits, hits)
    notes = pg.evaluate("()=>[...document.querySelectorAll('#tile-moe .tz-note')].map(e=>e.textContent)")
    check(f'{tag}: four notes, each one sentence in the page language (B1, R11.5)', len(notes) == 4 and all(len(x) > 20 for x in notes) and all((any('一' <= ch <= '鿿' for ch in x)) == (lang == 'zh') for x in notes)
          and all(x.count('. ') == 0 and x.count('。') <= 1 for x in notes), notes)
    nb_ = lambda x: x.replace('\u00a0', ' ')
    want = Z(lang, [['decode step', 'one token', 'memory reads', '60%', 'H100 SXM', '3.35 TB/s', 'longer than its arithmetic'], ['FP8 weights', 'embedding table only one row per request', 'KV cache of 2K tokens', 'BF16', '128 KiB per token', 'all three models'],
                    ['2 of a layer’s 8 experts', 'uniform routing', '8 × (1 − (6/8) to the power N)'], ['Sources:', 'config.json', 'Mistral AI', 'H100 datasheet']],
             [['decode 步', '一个 token', '读显存', '60%', 'H100 SXM', '3.35 TB/s', '读显存总比计算久'], ['FP8 权重', 'embedding 表不整表读', '只查其中一行', '2K 个 token 的 KV cache', 'BF16', '128 KiB', '三个模型'],
              ['8 个专家里选 2 个', '均匀路由', '8 ×（1 − 6/8 的 N 次方）'], ['来源：', 'config.json', 'Mistral AI', 'H100 规格表']])
    check(f'{tag}: the notes give the step model in one sentence, what a step reads (FP8 weights, embedding rows, BF16 KV cache at 128 KiB per token, 2K tokens), uniform routing, the sources (B-08)',
          len(notes) == 4 and all(all(k_ in nb_(n_) for k_ in ks) for n_, ks in zip(notes, want)), [n_ for n_, ks in zip(notes, want) if not all(k_ in nb_(n_) for k_ in ks)])
    check(f'{tag}: no “^” in the notes, tooltips or sentence (A #8)', not any('^' in x for x in notes + pg.evaluate("()=>[...document.querySelectorAll('#tile-moe [data-tipx]')].map(e=>e.getAttribute('data-tipx'))") + [st['read']]), notes)
    tips = pg.evaluate("()=>[...document.querySelectorAll('#tile-moe .tz-moe [data-tipx]')].map(e=>[e.textContent.trim(), e.getAttribute('data-tipx')])")
    check(f'{tag}: the three names and “experts read” explain themselves on hover; the dense ones say hypothetical', len(tips) == 4 and all(len(t_) > 40 for _, t_ in tips) and sum(Z(lang, 'Hypothetical', '假想') in t_ for _, t_ in tips) == 2, tips)
    if lang == 'zh':
        bad = [x for t_ in notes + [t_ for _, t_ in tips] for x in zh_rules(t_)]
        check(f'{tag}: notes and tooltips follow the Chinese writing rules', not bad, bad)
    alltext = '\n'.join([pg.inner_text(TILE)] + notes + [t_ for _, t_ in tips] + [st['aria']])
    check(f'{tag}: no memory size on the tile (the lab’s part 2 asks for it, R3.8)', not re.search(r'\d\s*(GiB|GB)\b|87\.0|43\.5', alltext), re.findall(r'.{20}\d\s*(?:GiB|GB)\b', alltext))
    chip = pg.evaluate("()=>{const c=document.querySelector('#tile-moe .tz-kind'); return c?c.textContent:''}")
    check(f'{tag}: the chip says the numbers come from a formula', chip == Z(lang, 'Formula', '公式计算'), chip)
    for h_ in pg.evaluate(RULES, TILE): rule_hits.append(f'{tag}: {h_}')
    for h_ in pg.evaluate(BIGNUM, TILE): rule_hits.append(f'{tag}: more than 6 digits (R6.18): {h_}')
    for h_ in pg.evaluate(HELP, TILE): rule_hits.append(f'{tag}: question-mark cursor without an explanation (R1.1): {h_}')
    if lang == 'zh':
        for h_ in pg.evaluate(COLONS, TILE): rule_hits.append(f'{tag}: two colons in one sentence: {h_}')
        for h_ in pg.evaluate(SPACING, TILE): rule_hits.append(f'{tag}: Chinese spacing: {h_}')
        for h_ in pg.evaluate(PUNCT, TILE): rule_hits.append(f'{tag}: half-width punctuation in Chinese: {h_}')
    if not dark:
        pg.hover(f'{TILE} .tz-moe-n.tz-moe-d13'); pg.wait_for_timeout(350)
        tipt = pg.evaluate("()=>{const t=document.getElementById('tip'); return t&&!t.hidden?t.innerText:''}")
        check(f'{tag}: hovering “dense 12.9B” says what it is', Z(lang, 'Hypothetical', '假想') in tipt and '12.9B' in tipt, tipt)
        check(f'{tag}: no “Assumptions & sources” on the tile (owner 2026-10-13: the tile is one button, no “Assumptions & sources” on it)', pg.evaluate(f"()=>!document.querySelector('{TILE} .tz-more')&&!/假设和来源|Assumptions (&|and) sources/.test(document.querySelector('{TILE}').innerText)"))
        pg.mouse.move(2, 2)
        # the language switch keeps the setting
        pg.click(f'{TILE} .seg-ctl button[data-v="16"]'); pg.wait_for_timeout(300)
        other = 'zh' if lang == 'en' else 'en'
        pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(500)
        st2 = pg.evaluate(TILE_STATE) if pg.query_selector(TILE) else None
        check(f'{tag}: the language switch keeps 16 requests', bool(st2) and st2['pressed'] == ['16'] and '7.9' in st2['read'] and (('个' in st2['read']) == (other == 'zh')), st2 and st2['read'])
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    pg.click(f'{TILE} .seg-ctl button[data-v="4"]'); pg.wait_for_timeout(450); pg.mouse.move(2, 2); pg.wait_for_timeout(300)
    pg.locator(TILE).screenshot(path=str(SH / f'tile_{tag.replace(" tile", "")}_4.png'))
    pg.close()


def run_misc(b, w, h, lang):
    tag = f'{w}_{lang}'
    # the home tile: the answer to the question at every setting (and B1: its notes are sentences in the page language)
    tile_checks(b, w, h, lang)
    if w > 760: tile_checks(b, w, h, lang, narrow=True)
    if TILE_ONLY: return
    # the hash round trip: a shared link opens Scenarios with its setup
    pg = page(b, w, h, lang, hash_='moe~fp8_2048_8')
    s = state(pg); check(f'{tag} hash round trip', s['mode'] == 'sandbox' and s['B'] == 8 and s['c'] == 2048 and s['p'] == 'fp8', s)
    check(f'{tag} hash: the scenario “A small team” pressed', pg.evaluate("()=>document.querySelector('#moe-rail [data-preset=\"1\"]').getAttribute('aria-pressed')") == 'true')
    pg.close()
    pg = page(b, w, h, lang, hash_='moe~fp8_999_8')
    check(f'{tag} a bad hash falls back to the guided tour', state(pg)['mode'] == 'guided')
    pg.close()
    if w <= 760:   # the mini result bar at 390 once the chart scrolls away, with a short status
        pg = page(b, w, h, lang)
        go_part(pg, 1); click(pg, '#moe-rail [data-pred="80-100"]'); go_part(pg, 4); slide(pg, '#moe-ic-B-in', 8)
        pg.evaluate("()=>{const r=document.querySelector('#moe-rail .step.current .step-b'); window.scrollTo(0, r.getBoundingClientRect().top+scrollY-60)}"); pg.wait_for_timeout(600)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); if(!d||d.hidden) return null; const st=d.querySelector('.status'); return {status:st?st.textContent:'', text:d.innerText}}")
        check(f'{tag} mini result bar shows at 390', dk is not None and '21.99' in (dk or {}).get('text', ''), dk)
        check(f'{tag} mini result bar: short status', dk is not None and len((dk or {}).get('status', '')) <= 16, dk)
        rules(pg, f'{tag} dock', lang); shot(pg, f'{tag}_dock')
        pg.close()



# ----- the owner's message of 2026-10-08 (reviews/owner-b9-2026-10-08.md): G1, a lead-in that says what to pick and what to watch
# instead of “one at a time, they do not stack”; G2, no sentence leans on a detail of an earlier part, the recap states the pattern
OWNER_B9 = {'scen': ('Pick a scenario and see how many experts each decode step reads.', '选一个场景，看看每个 decode 步要读几个专家。'), 'chal': None, 'parts': [(2, [('and see which 2 experts each token goes to.', '看看每个 token 去了哪 2 个专家。')]), (5, [('A busy server reads almost every expert in every step.', '服务端一忙起来，每一步几乎要读遍所有专家。'), ('and see where MoE wins and where it loses.', '看看 MoE 赢在哪里、输在哪里。')]), (6, [('Pick how the router spreads the tokens and see how long the layer takes.', '选一种分配方式，看看这一层要算多久。')]), (7, [('MoE saves arithmetic, not memory', 'MoE 省的是计算，不是显存。'), ('That is what MoE buys; the price is memory for every expert.', '这就是 MoE 换来的好处，代价是每个专家都要占显存。'), ('So the memory follows all 46.7B parameters', '显存要按全部 46.7B 个参数算'), ('With 64 requests in a batch, every step reads every expert', 'batch 里有 64 个请求时，每一步都要读遍所有专家')])]}
OWNER_B9_NOTES = ('一次只能选', '一次选一个', '不会叠加', '每次只试', '下面的选项是单选', 'do not stack', 'One at a time', 'Pick one at a time', 'one at a time;')
OWNER_B9_BACK = re.compile(r'In part [0-9]|in part [0-9]|As in part|as in part|第 [0-9] 部分里|在第 [0-9] 部分|和第 [0-9] 部分一样|还是第 [0-9]')
owner_b9_once = []


def owner_b9(b, w, h, lang):
    if not owner_b9_once:
        owner_b9_once.append(1)
        src = (P / 'labs' / 'moe.js').read_text() + ((P / 'teasers' / 'moe.js').read_text() if (P / 'teasers' / 'moe.js').exists() else '')
        left = [s for s in OWNER_B9_NOTES if s in src]
        check('owner b9 G1: no “one at a time / do not stack” note left in the lab or its tile, EN or ZH', not left, left)
    k = 1 if lang == 'zh' else 0
    pg = page(b, w, h, lang)
    norm = "()=>{const c=[...document.querySelectorAll('.railcard')].find(x=>x.offsetParent)||document.querySelector('.step.current'); return c?c.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}"
    pg.evaluate("()=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"sandbox\"]:not([data-flow=free])')].filter(x=>x.offsetParent).pop(); if(b) b.click();}"); pg.wait_for_timeout(600)
    t = pg.evaluate(norm)
    check(f'{w} {lang} owner b9 G1: the Scenarios card leads in with what to pick and what to watch', OWNER_B9['scen'][k] in t, t[:240])
    if OWNER_B9['chal']:
        pg.evaluate("()=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"challenge\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}"); pg.wait_for_timeout(600)
        t = pg.evaluate(norm)
        check(f'{w} {lang} owner b9 G1: the challenge leads in to its options', OWNER_B9['chal'][k] in t, t[:400])
    pg.evaluate("()=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"guided\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}"); pg.wait_for_timeout(600)
    for part, wants in OWNER_B9['parts']:
        pg.evaluate(f"()=>{{const b=document.querySelector('.step-h[data-go=\"{part}\"]'); if(b) b.click();}}"); pg.wait_for_timeout(700)
        t = pg.evaluate("()=>{const c=document.querySelector('.step.current'); return c?c.textContent.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}")
        for s in wants:
            check(f'{w} {lang} owner b9: part {part + 1} has “{s[k][:40]}”', s[k] in t, t[:300])
        back = OWNER_B9_BACK.findall(t)
        check(f'{w} {lang} owner b9 G2: part {part + 1} leans on no detail of an earlier part', not back, back)
    pg.close()



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

B9V6 = {'lab': 'moe', 'math': '#moe-mathbody', 'block': '.moe-mb', 'src_gone': ['(6/8)^B', '(6/8)^${', '10^9', '10^12', '= 8 × (1 − (6/8)^', '(248/256)^8'], 'text_gone': ['(6/8)^', ')^B', '10^9', '10^12']}


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in LANGS:
        for w in WIDTHS:
            h = 900 if w > 760 else 844
            if not TILE_ONLY: run_guided(b, w, h, lang)
            run_misc(b, w, h, lang)
            owner_b9(b, w, h, lang)
            owner_b9v6(b, w, h, lang, B9V6)
    if 1440 in WIDTHS and not TILE_ONLY:
        for lang in LANGS: next_into_7(b, lang)
    if 1440 in WIDTHS and 'en' in LANGS and (not ARGS or ARGS == ['tile']):
        tile_checks(b, 1440, 900, 'en', dark=True)
    if 1440 in WIDTHS and 'en' in LANGS and not ARGS:
        pg = page(b, 1440, 900, 'en', dark=True); rules(pg, 'dark 1440 en part 1', 'en'); shot(pg, 'dark_1440_en_p1')
        go_part(pg, 1); click(pg, '#moe-rail [data-pred="80-100"]'); go_part(pg, 4); slide(pg, '#moe-ic-B-in', 8); rules(pg, 'dark 1440 en part 5', 'en'); shot(pg, 'dark_1440_en_p5'); pg.close()
    b.close()

check('no page errors', not errs, errs[:5])
for r in rule_hits[:40]: print('RULE', r)
check('writing and colour rules', not rule_hits, f'{len(rule_hits)} hits')
print(f'{n - len(fails)} / {n} checks passed')
sys.exit(1 if fails else 0)
