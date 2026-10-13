"""Flow test for the tokenization-cost lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH (light), plus one dark run at 1440 in EN:
- no page errors and no horizontal overflow at any step;
- the phone first screen (m2 template, R7.3): the parts' bar with Next, and the strips of the chart;
- the prediction gate (R3.8): Next disabled until a pick; before it, Settings are locked, and the answer (27 tokens, the Chinese
  sentence's bytes) appears nowhere in text or tooltips; “Saved …” after a pick; part 3 gives the verdict (wrong with the right answer
  in EN, right in ZH) with 36 and 27 tokens on the stage; the byte values show under the cells;
- part 4: Next disabled until 20 merges; the slider and ▸ move the merges; the strips draw one box per token (10 and 26 at 100
  merges); ▲▼ badges only after the learner's own change (R6.6);
- part 5: the training texts as single-choice options, the initial setup first and pressed; Next after another pick; the result;
- parts 6 and 7, and guided → Scenarios → Challenge by the cards' buttons (R3.12);
- Scenarios: the pressed state follows the settings (R1.8); a change in Settings that matches none says so; the hash round trip;
- the challenge: only the training text and the merges in the card, Settings locked with a note (R4.2); the three feedback tiers
  (not met with a hint and “show all options”, met but more merges than needed, met and minimal), the comparison chart (12 rows,
  4 of them green), and the capability card;
- the language switch keeps the part, the prediction, the settings and the challenge's answer;
- the mini result bar at 390 once the strips scroll away, with a short status;
- round 1 (reviews A and B): parts 1 and 2 draw the Chinese sentence as plain text over a dashed box, with no byte cells, no token
  boxes and no tooltip that names its bytes (A1, R3.8); part 2 draws the English strip at 0 merges (36 boxes) and shows “?” for the
  Chinese count, the ratio and the status; every cursor:help element has a tooltip and every capsule fits its text (R1.1, R7.8), on
  every screen; metric values are numbers only (A18); part 3 compares bytes with whole characters (A19) and the glyphs have tooltips
  (A2, A3); part 4 says why Next waits (A6) and lists the first merges (A27); part 5 shows “?” until an option is tried (A23);
  part 7's three buttons put each point's example in the chart (A8); the challenge's “why” agrees with the computed drops (B1);
- round 1b (re-review): part 5's disabled Next says why, in the card and the parts bar (N2); after the pick, part 2 says the answer
  comes in part 3 (N5); every glyph fits its byte cells (N8); part 7 lists every character that became one token (N4); part 6 gives
  Petrov et al.'s Chinese figure and says the 15 times is Shan (A30);
- the writing rules on every visible text node and tooltip (as tests/prefill-decode_flow.py): no “·” between clauses, one colon per
  Chinese sentence, a space between Chinese and Latin letters or digits, full-width punctuation in Chinese, no 起点 / starting point,
  控件 / Controls, “toy”, 亿 / 万 (outside the merges list, which shows raw tokens), no number over 6 digits outside the math and
  tooltips; blue, amber and violet nowhere, red only for a missed goal.
Screenshots: shots/tokenization-cost/.
Run: python3 tests/tokenization-cost_flow.py [en|zh] [1440|390]   (filters, for runs while fixing)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/tokenization-cost-flow.html', '--only', 'tokenization-cost,set-b'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'tokenization-cost-flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'tokenization-cost-flow.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'tokenization-cost-flow-wrapped.html')
SH = P / 'shots' / 'tokenization-cost'; SH.mkdir(parents=True, exist_ok=True)
LANGS = [a for a in sys.argv[1:] if a in ('en', 'zh')] or ['en', 'zh']
SIZES = [s for s in ((1440, 900), (390, 844)) if not [a for a in sys.argv[1:] if a.isdigit()] or str(s[0]) in sys.argv[1:]]
errs, fails, n = [], [], 0


MATHBLOCKS = """(a)=>{const [sel,pre]=a; const body=document.querySelector(sel); if(!body) return null;
  const bs=[...body.querySelectorAll('.'+pre+'-mb')];
  const words=[...body.querySelectorAll('math mi, math mtext')].map(e=>e.textContent.trim()).filter(w=>/^(left|right|frac|sqrt|mathrm|hat|cdot|times|div|approx|top|big|quad|ldots|cdots|alpha|beta|theta|phi|pi|sigma|varphi)$/.test(w));
  const ruled=bs.filter(b=>{const cs=getComputedStyle(b); return cs.borderTopStyle==='solid'&&parseFloat(cs.borderTopWidth)>=1}).length;
  const clip=[...body.querySelectorAll('[class$="math-line"]')].filter(x=>{const m=x.querySelector('math'); if(!m) return true; const r=m.getBoundingClientRect(), b=x.getBoundingClientRect(); return r.height>0&&(r.top<b.top-1||r.bottom>b.bottom+1)}).length;
  return {n:bs.length, lead:bs.filter(b=>b.querySelector('.'+pre+'-mlead')).length, fx:bs.filter(b=>b.querySelector('math')).length, ruled, words, clip, math:body.querySelectorAll('math').length}}"""

def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


Z = lambda lang, en, zh: zh if lang == 'zh' else en


def page(b, w, h, lang, dark=False, hash_='tokenization-cost'):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light')
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(500)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__tk.state()")
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText:''}", sel)


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__tk||!window.__tk.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: break
        last = y; pg.wait_for_timeout(120); t += 120
    pg.wait_for_timeout(350)   # the merge animation (0.28 s) ends


def click(pg, sel):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(200); settle(pg)


def go_part(pg, i): click(pg, f'#tk-rail [data-go="{i}"]')


def slide(pg, sel, v):
    pg.evaluate(f"""()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}));}}"""); pg.wait_for_timeout(200); settle(pg)


def mode(pg, m): click(pg, f'#tk-app .modes button[data-mode="{m}"]')


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def all_text(pg):
    return pg.evaluate("""()=>{const out=[]; for(const s of ['#tk-app','#dock']){const e=document.querySelector(s); if(e&&e.offsetParent!==null) out.push(e.innerText);}
      document.querySelectorAll('#tk-app [data-tipx], #dock [data-tipx]').forEach(e=>out.push(e.getAttribute('data-tipx'))); return out.join('\\n')}""")


def boxes(pg, l): return pg.evaluate(f"()=>document.querySelectorAll('.tk-strip[data-l=\"{l}\"] .tk-tok:not(.tk-gone)').length")


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/in this toy\\b|在这个玩具里|\\btoy\\b|玩具/i,'“toy”'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/Locked in|已锁定/,'“Locked in”'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万 (R6.18)'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],[/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part|box)/i,'dark / light named as a colour (R5.7)']];
 const skip=e=>e.closest('[data-endnotes],pre, code, #tip, .tk-mg-list, #tk-mg-first, .tk-glyphs, .labhead .tag, .pop>p:first-child>b');   // the merges show raw tokens and characters of the training text   // the fidelity label itself (Toy run) is allowed (R6.11)
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||skip(e)||!e.getClientRects().length) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{if(e.closest('.tk-mg-list, #tk-mg-first, .tk-boxes')) return; const t=e.getAttribute('data-tipx'); for(const [re,name] of bad) if(re.test(t)) out.push(name+' (tooltip): '+t.slice(0,90));});
 return out;}"""
SPACING = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), bad=/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>%][\\u4e00-\\u9fff]/;
 const shown=e=>e&&!e.closest('[data-endnotes],pre,code,svg,.mathbody,#tip,.tk-mg-list,.tk-glyphs,.tk-hex')&&e.getClientRects().length;
 const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){if(!shown(n.parentElement)) continue; const t=n.textContent.replace(/\\s+/g,' '); const m=t.match(new RegExp(bad.source,'g')); if(m) out.add(m.join('|')+' in: '+t.trim().slice(0,80));}
 root.querySelectorAll('a,.def,b,.t,em').forEach(e=>{if(!shown(e)||getComputedStyle(e).display!=='inline') return;
   const t=e.textContent; if(!t) return; const prev=e.previousSibling, next=e.nextSibling;
   const pc=prev&&prev.nodeType===3?prev.textContent.slice(-1):'', nc=next&&next.nodeType===3?next.textContent.slice(0,1):'';
   const pair=(a,b)=>a&&b&&bad.test(a+b);
   if(pair(pc,t[0])) out.add(pc+t[0]+' before: '+t.slice(0,40)); if(pair(t[t.length-1],nc)) out.add(t[t.length-1]+nc+' after: '+t.slice(-40));});
 root.querySelectorAll('[data-tipx]').forEach(e=>{if(e.closest('.tk-boxes')) return; const t=e.getAttribute('data-tipx'); const m=t.match(/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>][\\u4e00-\\u9fff]/g); if(m) out.add(m.join('|')+' in tooltip: '+t.slice(0,80));});
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
   for(const x of sents){const k=(x.match(/：|: /g)||[]).length; if(k>=2) out.add(x.trim().slice(0,160));}}
 return [...out];}"""
BIGNUM = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #tk-mathbody, .num-big, #tk-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const out=[]; for(const e of document.querySelectorAll('#tk-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p];
   for(const c of [blue,amber,violet]) if(v===c&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);
   if(v===red){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':'');
     if(!/status bad|badline|tk-over|warn|ex-res|wrong|\\bno\\b/.test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push('red on '+e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}
 return out.slice(0,12);}"""
rule_hits = []
AUDIT = """()=>{const out={help:[],fit:[],svg:[]};
 const vis=e=>e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden';
 document.querySelectorAll('#tk-app *, #dock *').forEach(e=>{if(!vis(e)) return; const s=getComputedStyle(e);
  if(s.cursor==='help'&&!e.closest('[data-tipx]')&&!e.closest('[title]')) out.help.push(e.tagName+'.'+(e.getAttribute('class')||'')+' “'+(e.textContent||'').trim().slice(0,30)+'”');
  if(e.matches('.metric .v, .metric .k, .seg-ctl button, .tk-pc, .tk-mask, .tk-chip, .status, .tk-cv, .tk-sn, .tk-cl, .tk-mg-n, .lv-r, .choice, .tag, .btn, .pb-dot, .tk-vb, .tk-g')&&e.scrollWidth>e.clientWidth+1)
    out.fit.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+e.scrollWidth+'>'+e.clientWidth+' “'+(e.textContent||'').trim().slice(0,40)+'”');});
 document.querySelectorAll('#tk-app svg text, #dock svg text').forEach(t=>{if(!vis(t)) return; const sv=t.ownerSVGElement.getBoundingClientRect(), b=t.getBoundingClientRect();
  if(b.left<sv.left-1||b.right>sv.right+1) out.svg.push((t.textContent||'').slice(0,30));});
 return out;}"""
A1 = """()=>{const z=document.querySelector('.tk-strip[data-l="zh"]'), bx=z.querySelector('.tk-boxes');
 const hexes=[...'我喜欢在早上看书。'].map(c=>[...new TextEncoder().encode(c)].map(b=>b.toString(16).padStart(2,'0')).join(' '));
 const tips=[...document.querySelectorAll('#tk-app [data-tipx], #dock [data-tipx]')].map(e=>e.getAttribute('data-tipx')).filter(t=>hexes.some(h=>t.includes(h))||/byte[s]? [0-9a-f]{2}\\b.*of “[\\u4e00-\\u9fff]/.test(t)||/的字节 [0-9a-f]{2}/.test(t));
 return {toks:z.querySelectorAll('.tk-tok:not(.tk-gone)').length, cells:z.querySelectorAll('.tk-g').length, plain:!!z.querySelector('.tk-gplain'),
   mask:(z.querySelector('.tk-mask')||{}).textContent||'', grid:getComputedStyle(bx).backgroundImage, hex:!!z.querySelector('.tk-hex:not([hidden])'), tips:tips.slice(0,3),
   sn:z.querySelector('.tk-sn').textContent}}"""
BYTES_RE = re.compile(r'(?:3|three) bytes (?:for|per|each)|of 3 bytes each|Chinese character (?:takes|took|is|stays) (?:3|three)|用 3 个字节存|每个汉字占|每个 3 个字节|汉字.{0,6}(?:3|三) ?个字节')


def audit(pg, where):
    a = pg.evaluate(AUDIT)
    check(f'{where}: every cursor:help element has a tooltip (R1.1)', not a['help'], a['help'][:4])
    check(f'{where}: every capsule fits its text (R7.8)', not a['fit'] and not a['svg'], (a['fit'][:4], a['svg'][:4]))
    mv = pg.evaluate("()=>[...document.querySelectorAll('#tk-metrics .metric .v')].map(e=>e.textContent)")
    check(f'{where}: metric values are numbers only (A18)', all(re.fullmatch(r'[\d,.?]+', v) for v in mv), mv)


def a1(pg, where, part, lang):
    x = pg.evaluate(A1)
    check(f'{where}: the Chinese strip has no token boxes, no byte cells and no byte grid before part 3 (A1)', x['toks'] == 0 and x['cells'] == 0 and x['plain'] and x['grid'] == 'none' and not x['hex'], x)
    check(f'{where}: no tooltip names the Chinese sentence’s bytes before part 3 (A1)', not x['tips'], x['tips'])
    check(f'{where}: the placeholder says why the tokens are hidden', bool(x['mask'].strip()), x['mask'])
    if part == 2:
        check(f'{where}: the Chinese count reads “?” (A1)', x['sn'].startswith('?'), x['sn'])
    t = all_text(pg)
    check(f'{where}: no sentence gives the bytes per Chinese character before part 3', not BYTES_RE.search(t), BYTES_RE.findall(t)[:3])


# the owner's clarification (2026-10-07 23:37 UTC): no hover only expands a shown number to its exact value (Show the math excepted)
EXACT_HOVERS = """()=>[...document.querySelectorAll('[data-tipx]')].filter(e=>!e.closest('#tk-mathbody')&&/精确值是|Exactly/.test(e.getAttribute('data-tipx')||''))
  .map(e=>e.textContent.trim().slice(0,30)+' → '+e.getAttribute('data-tipx').slice(0,90))"""


def rules(pg, where, lang):
    ex = pg.evaluate(EXACT_HOVERS)
    check(f'{where}: no hover that only gives a shown number’s exact value (精确值是 / Exactly)', not ex, ex)
    for sel in ('#tk-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def badges(pg): return pg.evaluate("()=>[...document.querySelectorAll('#tk-metrics .metric .d')].map(d=>d.textContent).filter(Boolean)")


def nav_next(pg): return pg.evaluate("()=>{const b=document.querySelector('#tk-rail .step.current [data-nav=\"next\"]'); return b?!b.disabled:null}")


def obs(pg): return pg.evaluate("()=>{const b=document.querySelector('#tk-rail [data-obs]'); return b&&!b.hidden?b.innerText:''}")


def run_guided(b, w, h, lang, dark=False):
    tag = f'{"dark_" if dark else ""}{w}_{lang}'
    pg = page(b, w, h, lang, dark)
    phone = w <= 760
    check(f'{tag} title', text(pg, '#tk-app .labtitle') == Z(lang, 'Chinese or English: which costs more tokens?', '同样的意思，中文和英文谁更费 token？'), text(pg, '#tk-app .labtitle'))
    check(f'{tag} no fidelity tag on top (owner, 2026-10-11 19:28 UTC)', not any(w in text(pg, '#tk-app .labmeta') for w in ('Fidelity', '保真度')), text(pg, '#tk-app .labmeta'))
    check(f'{tag} outcome line under the title', text(pg, '#tk-app .lab-outcome').startswith(Z(lang, 'You will be able to:', '学完你能：')))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#tk-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        pg.evaluate(f"()=>document.querySelector('#tk-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(150)
        check(f'{tag} popover {pid} opens', pg.evaluate(f"()=>{{const p=document.getElementById('{pid}'); return !!p&&p.getClientRects().length>0}}"))
        rules(pg, f'{tag} popover {pid}', lang)
        check(f'{tag} popover {pid}: no horizontal overflow', overflow(pg) <= 0, overflow(pg))
        pg.evaluate(f"()=>document.querySelector('#tk-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(120)
    # part 1: the mission, the answer hidden
    check(f'{tag} part 1: the stage shows 10 and 26 tokens', text(pg, '.tk-strip[data-l="en"] .tk-sn').startswith('10') and text(pg, '.tk-strip[data-l="zh"] .tk-sn').startswith('26'))
    check(f'{tag} part 1: one box per English token (10)', boxes(pg, 'en') == 10, boxes(pg, 'en'))
    a1(pg, f'{tag} part 1', 1, lang)
    audit(pg, f'{tag} part 1')
    check(f'{tag} part 1: Settings are locked with a note before the prediction', pg.evaluate("()=>!document.getElementById('tk-lock').hidden&&document.getElementById('tk-ctl-k-wrap').hidden"))
    t1 = all_text(pg)
    check(f'{tag} part 1: the answer (27) is nowhere before the prediction', not re.search(r'(?<![\d.,])27(?![\d,])', t1), re.findall(r'.{30}(?<![\d.,])27(?![\d,]).{20}', t1)[:3])
    if phone:
        fs = pg.evaluate("""()=>{const bar=document.querySelector('#tk-rail .pbar'), st=document.querySelector('#tk-strips'), nx=document.querySelector('#tk-rail .pbar [data-nav="next"]');
          const r=e=>e?e.getBoundingClientRect():null; return {bar:!!bar&&r(bar).top<innerHeight, next:!!nx&&r(nx).bottom<innerHeight, strips:!!st&&r(st).top<innerHeight-60}}""")
        check(f'{tag} phone first screen: the parts’ bar with Next, and the strips', fs['bar'] and fs['next'] and fs['strips'], fs)
    rules(pg, f'{tag} part 1', lang); check(f'{tag} part 1 no overflow', overflow(pg) <= 0, overflow(pg))
    shot(pg, f'{tag}_1_mission')
    # part 2: predict
    click(pg, '#tk-rail .step.current [data-nav="next"]')
    check(f'{tag} part 2: Next disabled before a pick', nav_next(pg) is False)
    check(f'{tag} part 2: the disabled Next says why', bool(pg.evaluate("()=>document.querySelector('#tk-rail .step.current [data-nav=\"next\"]').dataset.tipx||''")))
    a1(pg, f'{tag} part 2 before the pick', 2, lang)
    check(f'{tag} part 2: the English strip at 0 merges (36 boxes), as the card says (A1)', boxes(pg, 'en') == 36 and state(pg)['k'] == 0, (boxes(pg, 'en'), state(pg)['k']))
    p2 = pg.evaluate("()=>({st:document.getElementById('tk-status').textContent, m:[...document.querySelectorAll('#tk-metrics .metric .v')].map(e=>e.textContent), cv:document.querySelector('.tk-crow[data-l=\"zh\"] .tk-cv').textContent})")
    check(f'{tag} part 2: status “Predict first”, the Chinese count, ratio and cost “?”', p2['st'] == Z(lang, 'Predict first', '先预测') and p2['m'][1] == '?' and p2['m'][2] == '?' and p2['cv'].startswith('?'), p2)
    audit(pg, f'{tag} part 2')
    t2 = pg.evaluate("()=>{const c=document.querySelector('#tk-rail .choices'); const all=document.getElementById('tk-app').innerText; return all.replace(c?c.innerText:'','')}")
    check(f'{tag} part 2: no 27 outside the options before the pick', not re.search(r'(?<![\d.,])27(?![\d,])', t2), re.findall(r'.{30}(?<![\d.,])27(?![\d,]).{20}', t2)[:3])
    wrong = lang == 'en'
    click(pg, f'#tk-rail [data-pred="{"9" if wrong else "27"}"]')
    check(f'{tag} part 2: Saved note and Next enabled', Z(lang, 'Saved.', '已记下') in text(pg, '#tk-rail .step.current') and nav_next(pg) is True)
    check(f'{tag} part 2: the pick is pressed', pg.evaluate(f"()=>document.querySelector('#tk-rail [data-pred=\"{'9' if wrong else '27'}\"]').getAttribute('aria-pressed')") == 'true')
    a1(pg, f'{tag} part 2 after the pick', 2, lang)
    p2b = pg.evaluate("()=>({mk:(document.querySelector('.tk-mask')||{}).textContent||'', st:document.getElementById('tk-status').textContent, ctx:document.getElementById('tk-ctx').textContent})")
    check(f'{tag} part 2 after the pick: the stage says the answer comes in part 3, not “answer the question first” (N5)', Z(lang, 'part 3', '第 3 部分') in p2b['mk'] and Z(lang, 'answer the question', '先回答') not in p2b['mk']
          and p2b['st'] == Z(lang, 'Prediction saved', '预测已记下') and Z(lang, 'shown in part 3', '在第 3 部分显示') in p2b['ctx'], p2b)
    shot(pg, f'{tag}_2_predict')
    # part 3: bytes
    click(pg, '#tk-rail .step.current [data-nav="next"]')
    s = state(pg)
    check(f'{tag} part 3: no merges on the stage (36 and 27 tokens)', s['k'] == 0 and boxes(pg, 'en') == 36 and boxes(pg, 'zh') == 27, (s['k'], boxes(pg, 'en'), boxes(pg, 'zh')))
    o = obs(pg)
    check(f'{tag} part 3: result, verdict, why', ('36' in o and '27' in o) and (Z(lang, 'The right answer is “27: three per character”', '正确答案是“27 个：每个字符三个”') in o if wrong else Z(lang, 'was right', '是对的') in o), o[:300])
    vs = pg.evaluate("()=>{const v=document.getElementById('tk-vs'); return v&&!v.hidden?v.innerText:''}")
    check(f'{tag} part 3: bytes against whole characters, drawn for 我 (A19)', 'e6' in vs and '130,000' in vs and Z(lang, 'From characters', '从字符起步') in vs, vs[:160])
    gt = pg.evaluate("()=>{const g=[...document.querySelectorAll('.tk-strip[data-l=\"zh\"] .tk-g')], e=[...document.querySelectorAll('.tk-strip[data-l=\"en\"] .tk-g')]; return {zh:g.length, wo:(g[0]||{dataset:{}}).dataset.tipx||'', dot:(e.find(x=>x.textContent==='·')||{dataset:{}}).dataset.tipx||''}}")
    check(f'{tag} part 3: each glyph has a tooltip with its bytes; “·” says it is a space (A2, A3)', gt['zh'] == 9 and 'e6 88 91' in gt['wo'] and '·' in gt['dot'] and Z(lang, 'space', '空格') in gt['dot'], gt)
    check(f'{tag} part 3: the legend explains “·”', Z(lang, 'a space', '空格') in text(pg, '#tk-legend'))
    audit(pg, f'{tag} part 3')
    click(pg, '#tk-rail .seg-ctl[data-key="hex"] button[data-v="1"]')
    check(f'{tag} part 3: a caption says how to read the byte values (A41)', pg.evaluate("()=>{const c=document.getElementById('tk-hexcap'); return !!c&&!c.hidden&&/49/.test(c.textContent)}"))
    check(f'{tag} part 3: the byte values show under the cells', pg.evaluate("()=>{const h=document.querySelector('.tk-strip[data-l=\"zh\"] .tk-hex'); return !!h&&!h.hidden&&h.children.length===27}"))
    rules(pg, f'{tag} part 3', lang); check(f'{tag} part 3 no overflow', overflow(pg) <= 0, overflow(pg))
    shot(pg, f'{tag}_3_bytes')
    # part 4: merges
    click(pg, '#tk-rail .step.current [data-nav="next"]')
    check(f'{tag} part 4: Next disabled before 20 merges', nav_next(pg) is False)
    check(f'{tag} part 4: the card and the disabled Next say “at least 20 merges” (A6)', '20' in obs(pg) and '20' in pg.evaluate("()=>document.querySelector('#tk-rail .step.current [data-nav=\"next\"]').dataset.tipx||''") and '20' in text(pg, '#tk-rail .step.current'), obs(pg)[:120])
    check(f'{tag} part 4: no “0 of 0” line before the first merge', pg.evaluate("()=>document.getElementById('tk-mg-sum').hidden"))
    check(f'{tag} part 4: no ▲▼ badge on entering the part', not badges(pg), badges(pg))
    slide(pg, '#tk-ic-k-in', 100)
    s = state(pg)
    check(f'{tag} part 4: 100 merges: 10 and 26 boxes, Next enabled', s['k'] == 100 and boxes(pg, 'en') == 10 and boxes(pg, 'zh') == 26 and nav_next(pg) is True, (s['k'], boxes(pg, 'en'), boxes(pg, 'zh')))
    o = obs(pg)
    check(f'{tag} part 4: the result names the English count, the trend of the Chinese one, and why', '10' in o and '2.60' in o and Z(lang, 'barely shrinks', '几乎没变短') in o and Z(lang, 'Why', '原因') in o, o[:300])
    check(f'{tag} part 4: ▲▼ badges after the learner’s change', len(badges(pg)) >= 1, badges(pg))
    click(pg, '#tk-rail .step.current [data-act="kplus"]')
    check(f'{tag} part 4: ▸ adds one merge and the page does not move', state(pg)['k'] == 101)
    check(f'{tag} part 4: the merges list shows the last three merges', pg.evaluate("()=>document.querySelectorAll('#tk-mg-list li:not(.tk-mg-h)').length") == 3)
    fm = text(pg, '#tk-mg-first')
    check(f'{tag} part 4: the first three merges are listed, as the card cites them (A27)', '#1' in fm and '#3' in fm and '·t' in fm, fm)
    audit(pg, f'{tag} part 4')
    rules(pg, f'{tag} part 4', lang); check(f'{tag} part 4 no overflow', overflow(pg) <= 0, overflow(pg))
    shot(pg, f'{tag}_4_merges')
    # part 5: the training text
    click(pg, '#tk-rail .step.current [data-nav="next"]')
    lv = pg.evaluate("()=>[...document.querySelectorAll('#tk-rail [data-lever]')].map(b=>[b.dataset.lever,b.getAttribute('aria-pressed'),b.querySelector('.lv-r').textContent])")
    check(f'{tag} part 5: four options, the initial setup first and pressed; results only once tried (A23)', [x[0] for x in lv] == ['tile', 'en', 'bal', 'zh'] and lv[0][1] == 'true' and [x[2] for x in lv] == ['2.60×', '?', '?', '?'], lv)
    check(f'{tag} part 5: Next disabled before another pick', nav_next(pg) is False)
    nb = pg.evaluate("()=>[...document.querySelectorAll('#tk-rail [data-nav=\"next\"]')].map(b=>[b.disabled,b.dataset.tipx||''])")
    check(f'{tag} part 5: every disabled Next (card and parts bar) says why (N2)', len(nb) >= 1 and all(d and Z(lang, 'another training text', '另一种训练文本') in t for d, t in nb), nb)
    check(f'{tag} part 5: the card says what to do from the start (N2)', Z(lang, 'Pick another training text', '选另一种训练文本') in obs(pg), obs(pg)[:160])
    click(pg, '#tk-rail [data-lever="bal"]')
    o = obs(pg)
    check(f'{tag} part 5: half and half: Chinese 17 boxes, English 18; the card says it in words, with the Chinese merges (46)', state(pg)['mix'] == 'bal' and boxes(pg, 'zh') == 17 and boxes(pg, 'en') == 18 and '46' in o
          and Z(lang, 'shrinks by about a third', '短了约三分之一') in o and Z(lang, 'nearly doubles', '长到约两倍') in o and nav_next(pg) is True, o[:300])
    lv = pg.evaluate("()=>[...document.querySelectorAll('#tk-rail [data-lever] .lv-r')].map(e=>e.textContent)")
    check(f'{tag} part 5: the tried option keeps its result, the others still “?”', lv == ['2.60×', '?', '0.94×', '?'], lv)
    rb = pg.evaluate("()=>{const b=document.querySelector('#tk-rail [data-obs]').getBoundingClientRect(); return [Math.round(b.top), innerHeight]}")
    check(f'{tag} part 5: the result is on screen after the pick (R3.10)', rb[0] < rb[1] - 40, rb)
    rules(pg, f'{tag} part 5', lang); check(f'{tag} part 5 no overflow', overflow(pg) <= 0, overflow(pg)); audit(pg, f'{tag} part 5')
    shot(pg, f'{tag}_5_mix')
    # language switch keeps the part, the prediction and the settings
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    s2 = state(pg)
    check(f'{tag} language switch keeps the part, the prediction and the training text', s2['step'] == 4 and s2['pred'] == ('9' if wrong else '27') and s2['mix'] == 'bal', s2)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    # part 6: cost
    click(pg, '#tk-rail .step.current [data-nav="next"]')
    check(f'{tag} part 6: the cost rows and the context window line', Z(lang, 'context window', '上下文窗口') in text(pg, '#tk-ctx') and '13,107' in text(pg, '#tk-ctx'), text(pg, '#tk-ctx'))
    c6 = text(pg, '#tk-rail .step.current')
    check(f'{tag} part 6: Petrov et al.’s figure for Chinese (3.2 times on GPT-2’s tokenizer), and the 15 times is Shan (A30)', '3.2' in c6 and Z(lang, 'Shan', '掸语') in c6 and '15' in c6, c6[:300])
    click(pg, '#tk-rail .seg-ctl[data-key="mix"] button[data-v="zh"]')
    check(f'{tag} part 6: a training text from the card', state(pg)['mix'] == 'zh' and Z(lang, 'cheaper', '更便宜') in obs(pg), obs(pg)[:200])
    rules(pg, f'{tag} part 6', lang); check(f'{tag} part 6 no overflow', overflow(pg) <= 0, overflow(pg)); audit(pg, f'{tag} part 6')
    shot(pg, f'{tag}_6_cost')
    # part 7
    click(pg, '#tk-rail .step.current [data-nav="next"]')
    t7 = text(pg, '#tk-rail .step.current')
    check(f'{tag} part 7: three misconceptions with this lab’s examples', t7.count('✗') == 3 and '1.05B' in t7 and ('book' in t7), t7[:200])
    check(f'{tag} part 7: misconception 1 states the pattern (G2, 2026-10-08: no list of characters from an earlier part) and keeps “我” on mostly English text', Z(lang, 'enough Chinese for its bytes to be merged', '它的字节被合并起来') in t7 and '“我”' in t7, t7[:400])
    _rl = text(pg, '#tk-rail .step.current .tk-remlist') + ' ' + text(pg, '#tk-rail .step.current .obs')
    check(f'{tag} part 7 (G2): no point or misconception leans on a detail of an earlier part', not re.search(r'[Pp]art \d|第 \d 部分', _rl), _rl[:300])
    check(f'{tag} part 7: the next lab and the button to Scenarios', pg.evaluate("()=>!!document.querySelector('#tk-rail .next-lab')||!!document.querySelector('.next-lab-end .next-lab')") and pg.evaluate("()=>!!document.querySelector('#tk-rail [data-mode-go=\"sandbox\"]')"))
    check(f'{tag} part 7: three points, each with a “Show in the chart” button (A8)', pg.evaluate("()=>document.querySelectorAll('#tk-rail .tk-remlist [data-rem]').length") == 3)
    s7 = state(pg)
    check(f'{tag} part 7: it opens with the first point’s example in the chart, its button pressed (as Sampling)', s7['remView'] == 'bytes' and s7['k'] == 0 and pg.evaluate("()=>document.querySelector('#tk-rail [data-rem=bytes]').getAttribute('aria-pressed')") == 'true', s7)
    for k, (mix, kk, focus) in {'bytes': ('tile', 0, 'strips'), 'mix': ('bal', 100, 'merges'), 'cost': ('tile', 100, 'cost')}.items():
        click(pg, f'#tk-rail [data-rem="{k}"]')
        s = state(pg); pr = pg.evaluate("()=>[...document.querySelectorAll('#tk-rail [data-rem]')].map(b=>b.dataset.rem+':'+b.getAttribute('aria-pressed'))")
        check(f'{tag} part 7: “{k}” puts its example in the chart and is pressed', s['remView'] == k and s['mix'] == mix and s['k'] == kk and f'{k}:true' in pr and sum(x.endswith(':true') for x in pr) == 1
              and pg.evaluate("()=>document.getElementById('tk-graph').dataset.focus") == focus, (s['remView'], s['mix'], s['k'], pr))
        if k == 'mix':
            shot(pg, f'{tag}_7_remember_mix')
            if phone: check(f'{tag} part 7: on a phone the button brings its example into view', pg.evaluate("()=>{const r=document.getElementById('tk-merges').getBoundingClientRect(); return r.top<innerHeight&&r.bottom>0}"))
    rules(pg, f'{tag} part 7', lang); check(f'{tag} part 7 no overflow', overflow(pg) <= 0, overflow(pg)); audit(pg, f'{tag} part 7')
    shot(pg, f'{tag}_7_remember')
    # Scenarios by the card's button
    click(pg, '#tk-rail [data-mode-go="sandbox"]')
    check(f'{tag} Scenarios by the card’s button', state(pg)['mode'] == 'sandbox')
    for i, (mix, ratio) in enumerate([('en', '3.00'), ('tile', '2.00'), ('bal', '0.59'), ('zh', '0.50')]):
        click(pg, f'#tk-rail [data-preset="{i}"]')
        pr = pg.evaluate("()=>[...document.querySelectorAll('#tk-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        check(f'{tag} scenario {mix}: pressed, 300 merges, result {ratio}', pr == ['true' if j == i else 'false' for j in range(4)] and state(pg)['k'] == 300 and ratio in text(pg, '#tk-insight'), (pr, text(pg, '#tk-insight')[:120]))
    hs = pg.evaluate('()=>location.hash')
    check(f'{tag} Scenarios: the hash carries the setup', hs == '#tokenization-cost~zh_300', hs)
    rules(pg, f'{tag} scenarios', lang); check(f'{tag} scenarios no overflow', overflow(pg) <= 0, overflow(pg)); audit(pg, f'{tag} scenarios')
    shot(pg, f'{tag}_8_scenarios')
    slide(pg, '#tk-ctl-k-in', 250)
    check(f'{tag} Scenarios: a change in Settings presses no scenario and says so', pg.evaluate("()=>[...document.querySelectorAll('#tk-rail [data-preset]')].every(b=>b.getAttribute('aria-pressed')==='false')") and Z(lang, 'match none', '都不一样') in text(pg, '#tk-insight'))
    # the challenge by the card's button
    click(pg, '#tk-rail [data-mode-go="challenge"]')
    s = state(pg)
    check(f'{tag} challenge: starts at the initial setup, Settings locked with a note', s['mode'] == 'challenge' and s['mix'] == 'tile' and s['k'] == 100 and pg.evaluate("()=>!document.getElementById('tk-lock').hidden&&document.getElementById('tk-ctl-mix-wrap').hidden"), s)
    check(f'{tag} challenge: the new pair on the stage (18 and 38 tokens)', boxes(pg, 'en') == 18 and boxes(pg, 'zh') == 38, (boxes(pg, 'en'), boxes(pg, 'zh')))
    click(pg, '#tk-rail [data-act="submit"]')
    rs = text(pg, '#tk-ch-res')
    check(f'{tag} challenge tier 3: not met, a hint, “Show the answer”', Z(lang, 'Not yet', '还没达到') in rs and Z(lang, 'Hint', '提示') in rs and text(pg, '#tk-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), rs[:200])
    click(pg, '#tk-ch-res [data-act="reveal"]')
    # the owner's review of batch 8 (2026-10-07): of the 12 options the chart keeps the 4 that meet both targets and the initial setup
    # (the pick); the note counts the other 7
    rows = pg.evaluate("()=>[...document.querySelectorAll('#tk-ch-res .cmp-row')].map(r=>({ok:r.classList.contains('ok'),me:r.classList.contains('me'),init:!!r.querySelector('.cmp-tag')}))")
    check(f'{tag} challenge: the chart keeps 5 of 12 options, the 4 green ones and the initial setup (your pick); the note counts 7',
          len(rows) == 5 and sum(r['ok'] for r in rows) == 4 and all(r['ok'] or (r['me'] and r['init']) for r in rows)
          and text(pg, '#tk-ch-res .cmp-note') == Z(lang, 'The other 7 options miss the goal and are not shown.', '其余 7 种没达到目标，图里没有列出。'), rows)
    check(f'{tag} challenge: the chart title asks the question and gives the goal, without “for each training text”', text(pg, '#tk-ch-res .cmp-title') == Z(lang, 'How many tokens does the Chinese sentence take for each English token? (Goal: at most 1.5, with the English sentence at most 25 tokens.)', '英文每个 token 对应中文几个 token？（目标：最多 1.5 个，而且英文句子最多 25 个 token。）'), text(pg, '#tk-ch-res .cmp-title'))
    tr = pg.evaluate("()=>{const p=document.querySelector('#tk-ch-res .tk-trend'); const t=window.__tk.chTrend(); return {dom:p?p.dataset.trend:'', txt:p?p.textContent:'', zhMore:t.zhMore, en:[t.a.en,t.b.en], zh:[t.a.zh,t.b.zh]}}")
    check(f'{tag} challenge: the “why” names the sentence that more merges shrink most, as computed (B1)', tr['dom'] == ('zh' if tr['zhMore'] else 'en') and tr['zhMore'] == ((tr['zh'][0] - tr['zh'][1]) > (tr['en'][0] - tr['en'][1]))
          and Z(lang, 'the Chinese sentence shrinks' if tr['zhMore'] else 'the English sentence shrinks', '中文句子短了' if tr['zhMore'] else '英文句子短了') in tr['txt'], tr)
    ct = text(pg, '#tk-rail .railcard')
    check(f'{tag} challenge: the card says why English only is left out and what a merge costs (A31, A12)', Z(lang, 'English only is left out', '不含“只有英文”') in ct and '8,192' in ct, ct[:200])
    click(pg, '#tk-rail .seg-ctl[data-key="chmix"] button[data-v="bal"]'); click(pg, '#tk-rail .seg-ctl[data-key="chk"] button[data-v="400"]')
    rs = text(pg, '#tk-ch-res')
    check(f'{tag} challenge tier 2: met, but fewer merges would do', Z(lang, 'Goal met', '达成目标') in rs and Z(lang, 'smaller vocabulary', '还可以更小') in rs, rs[:200])
    click(pg, '#tk-rail .seg-ctl[data-key="chk"] button[data-v="100"]')
    rs = text(pg, '#tk-ch-res')
    check(f'{tag} challenge tier 1: met with the fewest merges', Z(lang, 'Goal met', '达成目标') in rs and Z(lang, 'the fewest', '已经是最少') in rs, rs[:200])
    check(f'{tag} challenge: the status is green', pg.evaluate("()=>document.getElementById('tk-status').classList.contains('ok')"))
    check(f'{tag} challenge: the capability card follows the task', pg.evaluate("()=>!!document.querySelector('#tk-rail [data-exits]')"))
    rules(pg, f'{tag} challenge', lang); check(f'{tag} challenge no overflow', overflow(pg) <= 0, overflow(pg)); audit(pg, f'{tag} challenge')
    shot(pg, f'{tag}_9_challenge')
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    s = state(pg)
    check(f'{tag} language switch in the challenge keeps the answer and the result', s['mode'] == 'challenge' and s['mix'] == 'bal' and s['k'] == 100 and s['chSubmitted'] and pg.evaluate("()=>!!document.querySelector('#tk-ch-res .okline')"), s)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    if phone:
        pg.evaluate("()=>{const r=document.querySelector('#tk-rail .railcard'); r.scrollIntoView({block:'start'}); window.scrollBy(0,400);}"); pg.wait_for_timeout(500)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); if(!d||d.hidden) return null; const st=d.querySelector('.status'); return {st:st?st.textContent:'', txt:d.innerText}}")
        check(f'{tag} the mini result bar shows at 390 with a short status', bool(dk) and len(dk['st']) <= 16, dk)
        if dk: rules(pg, f'{tag} dock', lang)
        shot(pg, f'{tag}_10_dock')
    pg.close()


def run_hash(b, lang):
    pg = page(b, 1440, 900, lang, hash_='tokenization-cost~bal_250')
    s = state(pg)
    check(f'hash round trip {lang}: #tokenization-cost~bal_250 opens Scenarios with that setup', s['mode'] == 'sandbox' and s['mix'] == 'bal' and s['k'] == 250, s)
    pg.close()


# the owner's review of 2026-10-09 (R6.37: a row the reader has to scroll sideways is not read): in every guided part, Scenarios and the
# challenge, with each Show the math open, no box in it scrolls sideways and it is no wider than its own box (as tests/kit_math_scroll.py)
MSCROLL = """()=>{const out=[]; for(const d of document.querySelectorAll('details.math')){if(!d.getClientRects().length) continue; const was=d.open; d.open=true;
    for(const e of d.querySelectorAll('*')){if(!/auto|scroll/.test(getComputedStyle(e).overflowX)||!e.getClientRects().length) continue;
      const by=e.scrollWidth-e.clientWidth; if(by>1) out.push(e.className+' by '+by+' px: '+(e.textContent||'').trim().replace(/\\s+/g,' ').slice(0,40));}
    if(d.scrollWidth-d.clientWidth>1) out.push('the whole Show the math by '+(d.scrollWidth-d.clientWidth)+' px'); d.open=was;}
  return out;}"""


def math_scroll(b, w, h, lang):
    pg = page(b, w, h, lang); bad = []
    n = pg.evaluate("r=>new Set([...document.querySelectorAll(r+' [data-go]')].map(e=>e.dataset.go)).size", '#tk-rail')
    for k in range(n):
        pg.evaluate("([r,k])=>{const e=document.querySelector(r+' [data-go=\"'+k+'\"]'); if(e) e.click();}", ['#tk-rail', k]); pg.wait_for_timeout(500)
        bad += [f'part {k + 1}: {x}' for x in pg.evaluate(MSCROLL)]
    for m in ('sandbox', 'challenge'):
        pg.evaluate("m=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"'+m+'\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}", m); pg.wait_for_timeout(600)
        bad += [f'{m}: {x}' for x in pg.evaluate(MSCROLL)]
    check(f'{w} {lang}: no Show the math box scrolls sideways (all {n} guided parts, Scenarios, the challenge)', n >= 5 and not bad, (n, bad[:4]))
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    for w, h, lang in ((1440, 900, 'zh'), (900, 900, 'en'), (390, 844, 'zh'), (390, 844, 'en')):
        math_scroll(b, w, h, lang)
    for w, h in SIZES:
        for lang in LANGS:
            run_guided(b, w, h, lang)
    if len(LANGS) == 2 and len(SIZES) == 2:
        run_guided(b, 1440, 900, 'en', dark=True)
    for lang in LANGS: run_hash(b, lang)
    # the owner's decisions of 2026-10-09 (H1, H2): Show the math is a few blocks, each a lead line in words, most with a typeset formula
    # (MathML); a hairline between blocks; no TeX command shown as a word; nothing clipped
    for lang in ('en', 'zh'):
        pg = page(b, 1440, 900, lang); go_part(pg, 4); pg.wait_for_timeout(400)
        pg.evaluate("()=>{document.getElementById('tk-math').open=true}"); pg.wait_for_timeout(150)
        mb = pg.evaluate(MATHBLOCKS, ['#tk-mathbody', 'tk'])
        check(f'{lang} H2: Show the math in blocks (lead line, typeset formula, hairline between), no TeX word, nothing clipped',
              bool(mb) and mb['n'] == 5 and mb['lead'] == 5 and mb['fx'] == 4 and mb['ruled'] == 4 and not mb['words'] and mb['clip'] == 0, mb)
        t = pg.evaluate("s=>{const e=document.querySelector(s).cloneNode(true); e.querySelectorAll('math').forEach(m=>m.remove()); return e.textContent}", '#tk-mathbody')   # the text outside the typeset formulas
        check(f'{lang} H2: no run-on “; ” lines in Show the math', '; ' not in t and '；' not in t, t[:400])
        # H5: the stage's three blocks each have a label, with a hairline in the line colour between them (Scenarios: no focus frame);
        # next to a guided part's dashed focus frame the hairline turns transparent
        mode(pg, 'sandbox'); pg.wait_for_timeout(400)
        sep = pg.evaluate("""()=>{const g=document.getElementById('tk-graph'); const probe=document.createElement('i'); probe.style.color='var(--line)'; g.appendChild(probe); const lc=getComputedStyle(probe).color; probe.remove();
          return {focus:g.dataset.focus||'', caps:[...g.querySelectorAll(':scope>div>.tk-cap')].filter(c=>c.textContent.trim()).length,
                  rules:['tk-merges','tk-cost'].map(id=>{const s=getComputedStyle(document.getElementById(id),'::before'); return [s.content,s.borderTopStyle,s.borderTopWidth,s.borderTopColor]}), lc}}""")
        check(f'{lang} H5: the strips, the merges and the cost each have a label and a hairline in the line colour between them',
              sep['focus'] == '' and sep['caps'] == 3 and all(r[1] == 'solid' and r[2] == '1px' and r[3] == sep['lc'] for r in sep['rules']), sep)
        pg.close()
    b.close()
check('writing and colour rules on every screen', not rule_hits, len(rule_hits))
for x in sorted(set(h.split(': ',1)[1] for h in rule_hits))[:60]: print('  RULE', x[:230])
check('no page errors', not errs, errs[:5])
# G1 and G2, the owner's site-wide decisions of 2026-10-08 (sweep): option lead-ins instead of “one at a time” notes, and no sentence
# that leans on a detail of an earlier part. Read from the lab's source, so both languages are covered.
_G_SRC = (P / 'labs' / 'tokenization-cost.js').read_text()
for _bad in ['一次只能选', '一次选一个', '不会叠加', 'do not stack', 'One at a time', 'one at a time']:
    check(f'G1: no “{_bad}” left in the lab (EN and ZH)', _bad not in _G_SRC)
for _lead in ['Pick a training text and see how the Chinese sentence’s token count compares with the English sentence’s.', '选一种训练文本，看看中文句子的 token 数是英文的几倍。', 'Pick a scenario and see how many tokens the same message takes in each language.', '选一个场景，看看同一条消息用中文和英文各要多少 token。']:
    check(f'G1: the lead-in “{_lead[:36]}…” is there', _lead in _G_SRC)
for _old, _new in [('In part 3, each Chinese character took three tokens', 'so a sentence’s token count starts from its byte count.'), ('在第 3 部分，每个汉字占三个 token', '所以一句话的 token 数从它的字节数算起。'), ('in part 5, putting half the text in Chinese', 'giving one language more of the text leaves fewer merges for the others.'), ('在第 5 部分，训练文本一半是中文时', '给一种语言多放文本，留给其他语言的合并就少了。'), ('In part 6, the same message took', 'The same message can take several times as many tokens in one language as in another'), ('在第 6 部分，同一句话的中文 token 数', '同一句话换一种语言，token 数可能差好几倍'), ('with half the text in Chinese, ${andList(cq)} did', 'It becomes one token only when the training text has enough Chinese for its bytes to be merged.'), ('训练文本一半是中文时，${andList(cq)}就是', '只有训练文本里中文够多，它的字节被合并起来，一个汉字才会变成一个 token。'), ('In part 3, a Chinese character took 3.', 'A Chinese character, by comparison, is 3 bytes.'), ('第 3 部分里，一个汉字是 3 个。', '相比之下，一个汉字是 3 个字节。'), ('Puts part 3’s strips', 'Puts the strips with no merges back in the chart'), ('在图里重新显示第 3 部分的长条', '在图里重新显示没有合并时的长条'), ('Shows part 5’s half-and-half', 'Shows the half-and-half training text with 100 merges'), ('显示第 5 部分中英各半的训练文本', '显示中英各半的训练文本学 100 次合并的结果'), ('Shows part 6’s cost rows', 'Shows the cost rows for the initial setup'), ('显示第 6 部分初始设置下的成本条', '显示初始设置下的成本条')]:
    check(f'G2: “{_old[:30]}…” is rewritten as “{_new[:30]}…”', _old not in _G_SRC and _new in _G_SRC)
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:40] if fails else 'none')
# the owner's decisions of 2026-10-09 (H1/H2): the run-on lines of Show the math are gone from the source
_H_SRC = (P / 'labs' / 'tokenization-cost.js').read_text()
for _bad in ["tx('; embedding table + LM head = 2 × '", "tx('; Chinese ','；中文 ')", "merges, applied in the order they were learned'"]:
    check(f'H2: “{_bad[:40]}” is gone from the lab', _bad not in _H_SRC)
check('H1: Show the math typesets its formulas with A.ui.math', _H_SRC.count('A.ui.math(') >= 1 and '\\\\frac{\\\\text{context window}}' in _H_SRC)
print(f'{n} checks after H1, {len(fails)} failed')
sys.exit(1 if fails else 0)
