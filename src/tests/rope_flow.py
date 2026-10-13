"""Flow test for the rope lab, v2 (CONTRACT §6; reworked after the review of preview v3).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH (light), a layout pass at 900×900 in EN and ZH, and dark
screenshots:
- no page errors and no horizontal overflow at any step; the stage fits the screen at 1440×900 and 900×900 (no hidden part);
- R1.1: every element whose computed cursor is `help` has a tooltip on itself or an ancestor (data-tipx, data-tip, data-tipfn, .t, .def),
  and hovering one element of each kind shows the tooltip;
- R7.8: no text wider than its capsule (buttons, tags, status, chips, options, metric cards, the comparison's boxes, outputs:
  scrollWidth ≤ clientWidth), every SVG text with a box fits it (getComputedTextLength), stays inside its SVG and overlaps no other text;
- the phone first screen (m2 template, R7.3): the parts' bar with Next, and part of the chart;
- the new parts: 2 draws one head (128 numbers → 64 brackets → 64 arrows in place) and "added at the input" next to "turned inside
  attention"; 3 turns pair 0 on "dog bites man" (dotted arrows before the turn, the shadow, 19.7° vs 134.3°, 9.64 vs 6.00, what turns /
  by how much / what for); 4 says RoPE stays on, asks first and hides the moved dial, the angle and the score until then (R3.8), then
  gives the verdict (EN: a wrong pick, ZH: the right one); 5 and 6 use one chart (64 points, the one-turn line, hollow marks at the
  training length, thick stems past it) and state 29, 32, 0, 16×, 2×; gating of every part's Next;
- guided → Scenarios → Challenge by the cards' buttons (R3.12); the scenarios' pressed state follows the settings (R1.8);
- the challenge: only the base in the card, Settings hidden with a note (R4.2), the three feedback tiers, "Show the answer", the
  comparison chart (brief: the passing bases, the pick and the initial setup; the owner, 2026-10-07), and the capability card; the next-lab footer in every mode (R3.11);
- R3.8, review round 1 (A1): before part 4 is answered, part 4's answer appears nowhere a learner can read: the rail's titles and
  leads, the outcome line, the mode descriptions, the cards of parts 1 to 3 and part 4 before the pick, every tooltip, the Fidelity and
  Primer popovers, Connect, the mini result bar, and the home tile on the map;
- why Next is greyed out: a line under the buttons and the same text as the button's tooltip (A13); links to other parts (A15); on a
  phone, a change in the card brings the drawing back into view (A12);
- the language switch keeps the part, the prediction, the choices and the challenge's answer; the hash round-trip in Scenarios;
- the mini result bar at 390 once the chart scrolls away, with a short status;
- the writing rules on every visible text node and tooltip (no "·" between clauses, one colon per Chinese sentence, spacing and
  full-width punctuation in Chinese, no "toy" outside the fidelity label, no 起点 / 控件 / 亿 / 万 / bare 个数, no number over 6 digits
  outside the math and tooltips), colours only for the goal.
Filters: python3 tests/rope_flow.py [1440|390|900] [en|zh]
Screenshots: shots/rope/."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/rope.html', '--only', 'rope'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-rope.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'rope.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-rope.html')
SH = P / 'shots' / 'rope'; SH.mkdir(parents=True, exist_ok=True)
ARGS = sys.argv[1:]
WIDTHS = [w for w in (1440, 900, 390) if not any(a in ('1440', '900', '390') for a in ARGS) or str(w) in ARGS]
LANGS = [l for l in ('en', 'zh') if not any(a in ('en', 'zh') for a in ARGS) or l in ARGS]
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


def page(b, w, h, lang, dark=False, hash_='rope'):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light')
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(500)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__rp.state()")
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText:''}", sel)
count = lambda pg, sel: pg.evaluate("s=>document.querySelectorAll(s).length", sel)
cards = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#rp-metrics .metric')].filter(m=>m.getClientRects().length).map(m=>m.querySelector('.v').textContent)")


def settle(pg, ms=2200):
    try: pg.wait_for_function("()=>!window.__rp||!window.__rp.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def go_part(pg, i):
    pg.evaluate(f"()=>document.querySelector('#rp-rail [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(220); settle(pg)


def slide(pg, sel, v):
    pg.evaluate(f"""()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}));}}"""); pg.wait_for_timeout(220); settle(pg)


def click(pg, sel):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(220); settle(pg)


def seg(pg, key, v):
    click(pg, f'#rp-rail .seg-ctl[data-key="{key}"] button[data-v="{v}"]')


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#rp-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(320); settle(pg)


def next_disabled(pg):
    return pg.evaluate("()=>{const b=document.querySelector('#rp-rail .step.current [data-nav=\"next\"]'); return b?b.disabled:null}")


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(60)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def bench_text(pg):
    """what the learner can read in the rail, the stage and the mini result bar, tooltips included"""
    return pg.evaluate("""()=>{const out=[]; for(const s of ['#rp-rail','#rp-stage','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length) out.push(e.innerText);}
      document.querySelectorAll('#rp-rail [data-tipx], #rp-stage [data-tipx], #dock [data-tipx]').forEach(e=>out.push(e.getAttribute('data-tipx'))); return out.join('\\n')}""")


# ---------------------------------------------------------------- R1.1 and R7.8
HELP = """()=>{const out=[], nm=e=>{const c=e.getAttribute('class')||''; return e.tagName.toLowerCase()+(e.id?'#'+e.id:'')+(c?'.'+c.trim().split(/\\s+/).join('.'):'')+' «'+(e.textContent||'').trim().slice(0,40)+'»';};
 for(const r of [document.getElementById('rp-app'),document.getElementById('dock')]){ if(!r) continue;
  for(const e of r.querySelectorAll('*')){ if(!e.getClientRects().length||getComputedStyle(e).cursor!=='help') continue;
   const t=e.closest('[data-tipx],[data-tip],[data-tipfn],.t,.def');
   if(!t||(t.hasAttribute('data-tipx')&&!t.getAttribute('data-tipx').trim()&&!t.hasAttribute('data-jump'))) out.push(nm(e));}}
 return [...new Set(out)].slice(0,8);}"""
FIT = """()=>{const out=[], nm=e=>{const c=e.getAttribute('class')||''; return e.tagName.toLowerCase()+(e.id?'#'+e.id:'')+(c?'.'+c.trim().split(/\\s+/).join('.'):'')+' «'+(e.textContent||'').trim().slice(0,40)+'»';};
 const roots=[document.getElementById('rp-app'),document.getElementById('dock')].filter(Boolean);
 const SEL='button,.tag,.status,.chip,.lv-t b,.lv-r,.metric .v,.metric .k,.metric .sub,.seg-ctl button,.preset b,.pb-dot,.rp-box,.rp-col-h,output,.lbl';
 for(const r of roots) for(const e of r.querySelectorAll(SEL)){ if(!e.getClientRects().length||!e.clientWidth) continue;
   if(e.scrollWidth>e.clientWidth+1) out.push('wider than its box: '+nm(e)+' '+e.scrollWidth+' > '+e.clientWidth);}
 for(const r of roots) for(const lv of r.querySelectorAll('.lv')){const a=lv.querySelector('.lv-t'), b=lv.querySelector('.lv-r'); if(!a||!b||!b.textContent.trim()) continue;
   const x=a.getBoundingClientRect(), y=b.getBoundingClientRect(); if(x.left<y.right-1&&y.left<x.right-1&&x.top<y.bottom-1&&y.top<x.bottom-1) out.push('option title and result overlap: '+nm(lv));}
 document.querySelectorAll('#rp-vis svg').forEach(s=>{const sr=s.getBoundingClientRect(), rs=[];
   s.querySelectorAll('text').forEach(t=>{const box=t.getAttribute('data-box');
     if(box!=null&&t.getComputedTextLength()>+box+0.5) out.push('SVG text wider than its box: '+nm(t)+' '+t.getComputedTextLength().toFixed(1)+' > '+box);
     const tr=t.getBoundingClientRect(); if(!tr.width) return; rs.push([t,tr]);
     if(tr.left<sr.left-0.5||tr.right>sr.right+0.5||tr.top<sr.top-0.5||tr.bottom>sr.bottom+0.5) out.push('SVG text outside its SVG: '+nm(t));});
   for(let i=0;i<rs.length;i++) for(let j=i+1;j<rs.length;j++){const a=rs[i][1], b=rs[j][1], ox=Math.min(a.right,b.right)-Math.max(a.left,b.left), oy=Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top);
     if(ox>1.5&&oy>3) out.push('SVG texts overlap: '+nm(rs[i][0])+' / '+nm(rs[j][0]));}});
 return out.slice(0,8);}"""
TIPS = """()=>{const seen=new Set(), out=[];
 document.querySelectorAll('[data-hv-probe]').forEach(e=>e.removeAttribute('data-hv-probe'));
 for(const r of [document.getElementById('rp-app'),document.getElementById('dock')]){ if(!r) continue;
  for(const e of r.querySelectorAll('*')){ if(!e.getClientRects().length||getComputedStyle(e).cursor!=='help') continue;
   const t=e.closest('[data-tipx],[data-tip],[data-tipfn],.t,.def'); if(!t||t.hasAttribute('data-hv-probe')) continue;
   const sig=(t.getAttribute('class')||t.tagName)+'|'+(t.parentElement&&t.parentElement.id||'')+'|'+(t.closest('svg')?'svg':'html'); if(seen.has(sig)) continue; seen.add(sig);
   t.setAttribute('data-hv-probe',String(out.length)); out.push(sig);}}
 return out;}"""
layout_hits, tip_hits = [], []


def layout(pg, where):
    for h in pg.evaluate(HELP): layout_hits.append(f'{where}: cursor:help without a tooltip (R1.1): {h}')
    for h in pg.evaluate(FIT): layout_hits.append(f'{where}: {h} (R7.8)')
    if overflow(pg) > 0: layout_hits.append(f'{where}: horizontal overflow {overflow(pg)}px')


def hover_tips(pg, where):
    """hover one element of each kind with a help cursor; the tooltip must appear (R1.1)"""
    sigs = pg.evaluate(TIPS); shown = 0
    for i, sig in enumerate(sigs):
        pt = pg.evaluate(f"""()=>{{const e=document.querySelector('[data-hv-probe="{i}"]'); if(!e) return null; e.scrollIntoView({{block:'center'}});
          const r=e.getBoundingClientRect(); const x=r.left+Math.min(r.width/2,Math.max(2,r.width-2)), y=r.top+r.height/2; const hit=document.elementFromPoint(x,y);
          return hit&&(e===hit||e.contains(hit))?[x,y]:null}}""")
        if not pt: continue
        pg.wait_for_timeout(150)                     # let the scroll events of scrollIntoView pass (they hide the tooltip)
        pg.mouse.move(pt[0], pt[1]); pg.wait_for_timeout(120)
        ok = pg.evaluate("()=>{const t=document.getElementById('tip'); return !!t&&!t.hidden&&t.textContent.trim().length>0&&getComputedStyle(t).display!=='none'}")
        if not ok: tip_hits.append(f'{where}: no tooltip on hover: {sig}')
        else: shown += 1
        pg.mouse.move(2, 2); pg.wait_for_timeout(40)
    check(f'{where}: hovering shows tooltips ({shown} kinds tried)', shown >= 3, sigs)


def stage_fits(pg, where):
    fit = pg.evaluate("()=>{const s=document.getElementById('rp-stage'); return getComputedStyle(s).position!=='sticky'||s.scrollHeight<=s.clientHeight+1}")
    if not fit: layout_hits.append(f'{where}: the stage is taller than the screen (part of it hidden)')


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/in this toy\\b|在这个玩具里/i,'“in this toy”'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/Locked in|已锁定/,'“Locked in”'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万 instead of K, M, B (R6.18)'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour (R5.7)']];
 const toy=/\\btoy\\b|玩具/i;
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip')||!e.getClientRects().length) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));
   if(toy.test(t)&&!/Toy run|玩具实跑/.test(t)) out.push('“toy” outside the fidelity label: '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); for(const [re,name] of bad) if(re.test(t)) out.push(name+' (tooltip): '+t.slice(0,90));});
 return out;}"""
SPACING = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), bad=/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>%][\\u4e00-\\u9fff]/;
 const shown=e=>e&&!e.closest('[data-endnotes],pre,code,.mathbody,#tip')&&e.getClientRects().length;
 const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){if(!shown(n.parentElement)) continue; const t=n.textContent.replace(/\\s+/g,' '); const m=t.match(new RegExp(bad.source,'g')); if(m) out.add(m.join('|')+' in: '+t.trim().slice(0,80));}
 root.querySelectorAll('a,.def,b,.t,em,code').forEach(e=>{if(!shown(e)||getComputedStyle(e).display!=='inline') return;
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
   for(const x of sents){const k=(x.match(/：|: /g)||[]).length; if(k>=2) out.add(x.trim().slice(0,160));}}
 return [...out];}"""
BIGNUM = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #rp-mathbody, .num-big, #rp-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.replace(/1,000,000/g,'').match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
# the RoPE base 1,000,000 is exempt: “1M” already means 1,048,576 tokens in the challenge (R2.8 over R6.18, see the report)
# the lab draws in grayscale; red and green only for the challenge's goal (R5.6)
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|rp-over|warn|ex-res|wrong/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#rp-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':'')+' '+(e.closest('[id]')?e.closest('[id]').id:''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
rule_hits = []


# the owner's clarification (2026-10-07 23:37 UTC): no hover only expands a shown number to its exact value (Show the math excepted)
EXACT_HOVERS = """()=>[...document.querySelectorAll('[data-tipx]')].filter(e=>!e.closest('#rp-mathbody')&&/精确值是|Exactly/.test(e.getAttribute('data-tipx')||''))
  .map(e=>e.textContent.trim().slice(0,30)+' → '+e.getAttribute('data-tipx').slice(0,90))"""


def rules(pg, where, lang):
    ex = pg.evaluate(EXACT_HOVERS)
    check(f'{where}: no hover that only gives a shown number’s exact value (精确值是 / Exactly)', not ex, ex)
    for sel in ('#rp-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')
    layout(pg, where); stage_fits(pg, where)


# part 4's answer, in the words a page could use to give it away (R3.8; review round 1, A1)
LEAK = {'en': r"only (the|their) distance|only how far apart|depends only on|only through (their |the )?(distance|m ?− ?n)|relative position|stays? (exactly )?the same|leaves the score|score stays|distance counts|changed nothing|doesn.t change the score|does not change the score|cannot change|keeps the score",
        'zh': r"只有距离|只取决于|只通过.{0,8}距离|相对位置|完全不变|得分不变|得分还是|什么也没变|保持不变|不会变"}
LEAKJS = """([re,flags])=>{const R=new RegExp(re,flags), out=[];
 const add=(where,t)=>{t=String(t||''); const i=t.search(R); if(i>=0) out.push(where+': “'+t.match(R)[0]+'” in «'+t.slice(Math.max(0,i-70),i+50).replace(/\\s+/g,' ')+'»');};
 const app=document.getElementById('rp-app'); if(app){const c=app.cloneNode(true); c.querySelectorAll('.choice').forEach(e=>e.remove()); add('page text',c.textContent);
   app.querySelectorAll('[data-tipx]').forEach(e=>{if(!e.closest('.choice')) add('tooltip',e.getAttribute('data-tipx'));});}
 const dk=document.getElementById('dock'); if(dk) add('mini bar',dk.textContent);
 for(const id of ['rope-fid','rope-pri']){const e=document.getElementById(id); if(e) add(id,e.textContent);}
 const tile=document.getElementById('tile-rope'); if(tile){add('home tile',tile.textContent); tile.querySelectorAll('[data-tipx]').forEach(e=>add('home tile tooltip',e.getAttribute('data-tipx')));}
 return out;}"""


def leaks(pg, lang):
    return pg.evaluate(LEAKJS, [LEAK[lang], 'i'])


def leak_scan(b, w, h, lang):
    """everything a learner can read before answering part 4 holds no answer to it (A1)"""
    tag = f'{w}-{lang}-leak'; pg = page(b, w, h, lang)
    hits = []
    for i, act in ((0, None), (1, lambda: (seg(pg, 'hv', 'pairs'), seg(pg, 'hv', 'arrows'))), (2, lambda: seg(pg, 'rope3', 'on')), (3, None)):
        go_part(pg, i)
        if act: act()
        hits += [f'part {i + 1}: {x}' for x in leaks(pg, lang)]
    check(f'{tag}: before part 4 is answered, its answer appears nowhere (titles, outcome, modes, cards 1–3, tooltips, popovers, Connect, mini bar)', not hits, hits[:4])
    check(f'{tag}: the scan sees the prediction’s options are there but left out', count(pg, '#rp-rail .choice') == 3 and state(pg)['pred'] is None)
    pg.evaluate("()=>{location.hash='map~models'}"); pg.wait_for_timeout(700)
    tile = leaks(pg, lang)
    check(f'{tag}: the home tile does not answer part 4 either', count(pg, '#tile-rope') == 1 and not tile, tile[:3])
    check(f'{tag}: the home tile opens at RoPE “off” (N3)', pg.evaluate("()=>{const b=[...document.querySelectorAll('#tile-rope .seg-ctl button')].find(x=>x.getAttribute('aria-pressed')==='true'); return b?b.dataset.v:null}") == 'off')
    pg.evaluate("()=>document.querySelectorAll('#tile-rope .seg-ctl button')[1].click()"); pg.wait_for_timeout(150)
    check(f'{tag}: the tile’s “on” reading names pair 0 and says the order shows (N3)', Z(lang, 'pair 0 scores', '第 0 对得分') in text(pg, '#tile-rope .tz-read') and Z(lang, 'the order shows', '看得出顺序') in text(pg, '#tile-rope .tz-read'), text(pg, '#tile-rope .tz-read'))
    pg.close()


def guided(b, w, h, lang):
    tag = f'{w}-{lang}'; pg = page(b, w, h, lang)
    check(f'{tag}: opens in guided part 1', state(pg)['mode'] == 'guided' and state(pg)['step'] == 0, state(pg))
    if w <= 760:
        fs = pg.evaluate("""()=>{const bar=document.querySelector('#rp-rail .pbar'), g=document.querySelector('#rp-graph'), nx=bar&&bar.querySelector('[data-nav="next"]');
          const r=e=>e?e.getBoundingClientRect():null; return {bar:r(bar)&&r(bar).top<innerHeight&&r(bar).height>0, next:!!nx&&r(nx).top<innerHeight, graph:r(g)&&r(g).top<innerHeight-40}}""")
        check(f'{tag}: phone first screen shows the parts bar, its Next and part of the chart (R7.3)', fs['bar'] and fs['next'] and fs['graph'], fs)
    # part 1: the problem on "dog bites man"
    check(f'{tag}: part 1 Next is enabled', next_disabled(pg) is False)
    check(f'{tag}: part 1: bites → dog is 8.12 in both orders, difference 0.00', cards(pg) == ['8.12', '8.12', '0.00'], cards(pg))
    check(f'{tag}: part 1 draws the two grids with 18 cells', count(pg, '#rp-vis rect.rp-cell') == 18 and count(pg, '#rp-vis rect.rp-hot') == 2)
    rules(pg, f'{tag} part 1', lang); shot(pg, f'{tag}-p1')
    # part 2: one head, 128 numbers → 64 pairs → 64 arrows; added versus turned
    go_part(pg, 1)
    check(f'{tag}: part 2 shows the head: 128 numbers, no brackets yet, no metric cards', state(pg)['view'] == 'head' and count(pg, '#rp-vis rect.rp-numc') == 128 and count(pg, '#rp-vis path.rp-brk') == 0 and pg.evaluate("()=>document.getElementById('rp-metrics').hidden"))
    check(f'{tag}: part 2 Next disabled until the arrows are seen', next_disabled(pg) is True)
    why = pg.evaluate("()=>{const p=[...document.querySelectorAll('#rp-rail .rp-why')].find(e=>e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden'); const n=document.querySelector('#rp-rail .step.current [data-nav=\"next\"]'); return {line:p?p.textContent:'', tip:n?n.getAttribute('data-tipx'):''}}")
    check(f'{tag}: part 2 says why Next is greyed out, under the buttons and as its tooltip (A13)', why['line'] == Z(lang, 'Click “64 arrows” to go on.', '点“64 个箭头”，才能进入下一部分。') and why['tip'] == why['line'], why)
    check(f'{tag}: part 2 “128 numbers” already has a result (A18)', Z(lang, '128 numbers', '128 个数值') in text(pg, '#rp-rail [data-obs]'), text(pg, '#rp-rail [data-obs]')[:120])
    cmp_ = pg.evaluate("()=>({cols:document.querySelectorAll('#rp-cmp .rp-col').length, thick:document.querySelectorAll('#rp-cmp .rp-posbox').length, t:document.getElementById('rp-cmp').innerText})")
    check(f'{tag}: part 2 compares “added at the input” with “turned inside attention”, a thick box where the position enters in each',
          cmp_['cols'] == 2 and cmp_['thick'] == 2 and all(x in cmp_['t'] for x in Z(lang, ('Added at the input', 'Turned inside attention: RoPE', 'GPT‑2', 'BERT', 'Llama, Mistral, Qwen', 'v is not turned'), ('加在输入上', '在注意力里转动：RoPE', 'GPT‑2', 'BERT', 'Llama、Mistral、Qwen', 'v 不转'))), cmp_)
    body = text(pg, '#rp-rail .step.current')
    check(f'{tag}: part 2 says why pairing works and that the value is never turned',
          all(x in body for x in Z(lang, ('a turn needs a plane', 'Any fixed pairing works', 'sum of the 64 pairs’ own dot products', 'Keys are split the same way', 'the value is never turned', 'KV cache'),
                                       ('转动需要一个平面', '怎么配对都行', '64 对各自点积的总和', 'key 也这样配对', 'value 从来不转', 'KV cache'))), body[:400])
    rules(pg, f'{tag} part 2 (128 numbers)', lang)
    seg(pg, 'hv', 'pairs'); rules(pg, f'{tag} part 2 (64 pairs)', lang)
    check(f'{tag}: part 2 “64 pairs” changes the result text (A18)', Z(lang, 'Each bracket joins numbers 2i and 2i + 1', '每个括号把第 2i 和第 2i + 1 个数值连成一对') in text(pg, '#rp-rail [data-obs]'), text(pg, '#rp-rail [data-obs]')[:160])
    if w <= 760:
        gv = pg.evaluate("()=>{const g=document.getElementById('rp-graph').getBoundingClientRect(); return g.top<innerHeight-100&&g.bottom>120}")
        check(f'{tag}: on a phone, a click in the card brings the drawing into view (A12)', gv)
    check(f'{tag}: part 2 “64 pairs”: 64 brackets under neighbouring numbers, pair 0 marked', count(pg, '#rp-vis path.rp-brk') == 64 and count(pg, '#rp-vis path.rp-brk.on') == 1 and count(pg, '#rp-vis .rp-mini') == 0 and next_disabled(pg) is True)
    seg(pg, 'hv', 'arrows')
    obs = text(pg, '#rp-rail [data-obs]')
    check(f'{tag}: part 2 “64 arrows”: 64 dials under the brackets, pair 0 is (1.023, −1.097); Next enabled',
          count(pg, '#rp-vis .rp-mini') == 64 and count(pg, '#rp-vis .rp-brk') == 64 and Z(lang, '(1.023, −1.097)', '（1.023，−1.097）') in obs and next_disabled(pg) is False, obs[:200])
    click(pg, '#rp-rail [data-jump="cmp"]'); pg.wait_for_timeout(500)
    vis = pg.evaluate("()=>{const r=document.getElementById('rp-cmp').getBoundingClientRect(); return r.top>=0&&r.bottom<=innerHeight+1&&r.height>0}")
    check(f'{tag}: part 2 the link “the comparison in the chart” brings it into view', vis)
    why_off = pg.evaluate("()=>[...document.querySelectorAll('#rp-rail .rp-why')].every(e=>!e.getClientRects().length||getComputedStyle(e).visibility==='hidden')")
    check(f'{tag}: part 2 the why-line goes once Next is enabled', why_off and not pg.evaluate("()=>document.querySelector('#rp-rail .step.current [data-nav=\"next\"]').hasAttribute('data-tipx')"))
    check(f'{tag}: part 2 card: body-size paragraph on the older way, with a link to part 4 and to the KV cache lab (A9, A10, A15)',
          pg.evaluate("()=>{const p=[...document.querySelectorAll('#rp-rail .step.current p')].find(x=>x.querySelector('[data-jump=\"cmp\"]')); return !!p&&!p.classList.contains('small')&&!!p.querySelector('[data-jump=\"p4\"]')&&!!p.querySelector('a[href=\"#kv-basics\"]')}"))
    rules(pg, f'{tag} part 2', lang); shot(pg, f'{tag}-p2'); hover_tips(pg, f'{tag} part 2')
    pg.evaluate("()=>document.querySelector('#rp-rail [data-jump=\"p4\"]').click()"); pg.wait_for_timeout(250); settle(pg)
    check(f'{tag}: the link “Part 4” opens part 4 (A15)', state(pg)['step'] == 3)
    go_part(pg, 1)
    # part 3: RoPE on "dog bites man"
    go_part(pg, 2)
    check(f'{tag}: part 3 Next disabled until RoPE is on; RoPE off: 8.12 in both, pair 0 77.0° apart', next_disabled(pg) is True and cards(pg) == ['8.12', '8.12', '0.00'] and '0.44' in text(pg, '#rp-metrics') and '77.0°' in pg.evaluate("()=>document.querySelector('#rp-vis svg').textContent"), (next_disabled(pg), cards(pg)))
    check(f'{tag}: part 3 RoPE off: no dotted “before the turn” arrows', count(pg, '#rp-vis line.rp-c') == 0)
    body = text(pg, '#rp-rail .step.current')
    check(f'{tag}: part 3 card says what turns and points to the arcs', all(x in body for x in ('57.3°', Z(lang, 'arcs', '弧线'))), body[:300])
    seg(pg, 'rope3', 'on')
    obs = text(pg, '#rp-rail [data-obs]')
    check(f'{tag}: part 3 RoPE on: the row of “bites” favours the token just before it (9.64, 9.37; 8.12 without); what turns, by how much, what for; the head is made up to look one token back (A2, A3)',
          all(x in obs for x in ('9.64', '9.37', '8.12')) and all(x in obs for x in Z(lang, ('What turns:', 'By how much:', 'What for:', 'made up to look for the token just before'), ('什么在转：', '转多少：', '有什么用：', '设成去找“咬”前面的那个 token'))), obs[:400])
    svgt = pg.evaluate("()=>[...document.querySelectorAll('#rp-vis svg text')].map(t=>t.textContent).join('|')")
    check(f'{tag}: part 3 draws each turn as an arc outside the dial, labelled +2 ×, +1 × and +3 × 57.3° (A4)',
          count(pg, '#rp-vis path.rp-turnq:not(.rp-ah)') == 2 and count(pg, '#rp-vis path.rp-turnk:not(.rp-ah)') == 2 and all(x in svgt for x in ('+2 × 57.3°', '+1 × 57.3°', '+3 × 57.3°')), svgt[:200])
    check(f'{tag}: part 3 bars show the row of “bites” in each sentence, the highest outlined (A2)', count(pg, '#rp-vis rect.rp-hbar') == 4 and count(pg, '#rp-vis rect.rp-hbar.rp-top') == 2)
    check(f'{tag}: part 3 cards 9.64, 6.00, 3.64; pair 0 gives 1.84 and −1.36', cards(pg) == ['9.64', '6.00', '3.64'] and '1.84' in text(pg, '#rp-metrics') and '−1.36' in text(pg, '#rp-metrics'), cards(pg))
    check(f'{tag}: part 3 draws both panels with the arrows before the turn and the shadow', count(pg, '#rp-vis .rp-big') == 2 and count(pg, '#rp-vis line.rp-c') == 4 and count(pg, '#rp-vis line.rp-shadow') == 2)
    check(f'{tag}: part 3 Next enabled', next_disabled(pg) is False)
    rules(pg, f'{tag} part 3', lang); shot(pg, f'{tag}-p3'); hover_tips(pg, f'{tag} part 3')
    # part 4: RoPE stays on; prediction first; then move both
    go_part(pg, 3)
    body = text(pg, '#rp-rail .step.current')
    check(f'{tag}: part 4 says RoPE stays on', Z(lang, 'RoPE stays on', 'RoPE 一直开着') in body, body[:200])
    check(f'{tag}: part 4 Next disabled, no slider before the prediction', next_disabled(pg) is True and count(pg, '#rp-ic-shift') == 0)
    check(f'{tag}: part 4 title does not give the answer (A1)', Z(lang, 'Move both tokens 100 positions later', '两个 token 一起往后挪 100 个位置') in pg.evaluate("()=>document.getElementById('rp-rail').textContent"))
    check(f'{tag}: part 4 says why Next is greyed out before the pick', Z(lang, 'Pick an answer to go on.', '选一个答案，才能进入下一部分。') in text(pg, '#rp-rail'))
    t = bench_text(pg)
    check(f'{tag}: part 4 hides the moved dial, its angle and its score before the prediction (R3.8)',
          count(pg, '#rp-vis .rp-mask') == 1 and count(pg, '#rp-vis .rp-big') == 1 and cards(pg)[1:] == ['?', '?'] and not any(x in t for x in ('was right', 'right answer is', '是对的', '正确答案', 'stays 9.64', '还是 9.64', '“bites” 102', '“咬” 102')), (cards(pg), t[:200]))
    pick = 'same' if lang == 'zh' else 'little'
    click(pg, f'#rp-rail [data-pred="{pick}"]')
    check(f'{tag}: part 4 “Saved …” after a pick, the slider appears, Next still waits for the move', Z(lang, 'Saved.', '已记下') in text(pg, '#rp-rail') and count(pg, '#rp-ic-shift') == 1 and next_disabled(pg) is True)
    check(f'{tag}: no “Locked in”', 'Locked in' not in text(pg, '#rp-rail') and '已锁定' not in text(pg, '#rp-rail'))
    slide(pg, '#rp-ic-shift-in', 100)
    obs = text(pg, '#rp-rail [data-obs]')
    check(f'{tag}: part 4 result: still 19.7° apart, the head still 9.64', '19.7°' in obs and '9.64' in obs and next_disabled(pg) is False, obs[:300])
    check(f'{tag}: part 4 result and stage say what an added position vector would do (A9)', Z(lang, 'added at the input', '在输入处加位置向量') in obs and Z(lang, 'added at the input', '在输入处加位置向量') in text(pg, '#rp-vis .rp-note'), obs[-300:])
    if pick == 'same': check(f'{tag}: part 4 verdict: right', Z(lang, 'was right', '是对的') in obs, obs[:300])
    else: check(f'{tag}: part 4 verdict: wrong, with the right answer', Z(lang, 'The right answer is “It stays exactly the same”', '正确答案是“完全不变”') in obs, obs[:300])
    check(f'{tag}: part 4 cards 1, 19.7°, 9.64; the score card shows no change badge', cards(pg) == ['1', '19.7°', '9.64'] and pg.evaluate("()=>document.querySelector('#rp-metrics [data-k=\"sc\"] .d').textContent") == '', cards(pg))
    check(f'{tag}: part 4 draws the score against m − n with +1 and −1 marked', count(pg, '#rp-vis path.rp-curve') == 1 and count(pg, '#rp-vis .rp-dot') == 1 and count(pg, '#rp-vis .rp-dotw') == 1)
    # the φ hover in Show the math: the φ character, and both lines of its formula fit the tooltip (re-review N1)
    pg.evaluate("()=>{document.getElementById('rp-math').open=true}"); pg.wait_for_timeout(120)
    pt = pg.evaluate("""()=>{const e=document.querySelector('#rp-mathbody .t[data-ctl="phi"]'); if(!e) return null; e.scrollIntoView({block:'center'}); const r=e.getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]}""")
    fx = None
    if pt:
        pg.wait_for_timeout(150); pg.mouse.move(pt[0], pt[1]); pg.wait_for_timeout(150)
        fx = pg.evaluate("()=>{const t=document.getElementById('tip'); const f=[...t.querySelectorAll('.fx')]; return {n:f.length, fit:f.every(x=>x.scrollWidth<=x.clientWidth+1), text:t.textContent}}")
        pg.mouse.move(2, 2); pg.wait_for_timeout(60)
    check(f'{tag}: part 4 φ hover: two formula lines that fit, with φ and (m − n), no “varphi” (N1)', bool(fx) and fx['n'] == 2 and fx['fit'] and 'varphi' not in fx['text'] and 'φ' in fx['text'], fx)
    pg.evaluate("()=>{document.getElementById('rp-math').open=false}")
    rules(pg, f'{tag} part 4', lang); shot(pg, f'{tag}-p4'); hover_tips(pg, f'{tag} part 4')
    # part 5: one chart for base, speed and length
    go_part(pg, 4)
    check(f'{tag}: part 5: the turns chart with 64 points, the one-turn line, the shaded band; Next disabled',
          state(pg)['view'] == 'turns' and count(pg, '#rp-vis .rp-pt') == 64 and count(pg, '#rp-vis line.rp-one') == 1 and count(pg, '#rp-vis rect.rp-wrapbg') == 1 and next_disabled(pg) is True)
    check(f'{tag}: part 5 start: base 500,000 over 8K, 29 pairs under one turn, the slowest 0.00320 turns', cards(pg) == ['8K', Z(lang, '29 of 64', '29 / 64'), '0.00320'], cards(pg))
    body = text(pg, '#rp-rail .step.current')
    check(f'{tag}: part 5 card starts from the dog, has no typed formula, and says what to do (A5, A13)', Z(lang, 'In “dog bites man”', '在“狗咬人”里') in body[:80] and 'base^(' not in body and Z(lang, 'Drag the text length, then switch the base.', '拖动文本长度，再换一个 base。') in body, body[:200])
    check(f'{tag}: part 5 chart: a y-axis title and the speeds at both ends (A6)', Z(lang, 'full turns over the text', '在整段文本上转几圈') in pg.evaluate("()=>document.querySelector('#rp-vis svg').textContent") and '57.3°' in pg.evaluate("()=>document.querySelector('#rp-vis svg').textContent"))
    check(f'{tag}: part 5 slider stops at 256K (the challenge’s 1M stays unseen, R3.8)', pg.evaluate("()=>+document.querySelector('#rp-ic-l5-in').max") == 6 and state(pg)['l5'] == 1)
    slide(pg, '#rp-ic-l5-in', 0)
    obs = text(pg, '#rp-rail [data-obs]')
    check(f'{tag}: part 5 at 4K: the result compares with the start, 8K → 4K, 29 → 32, and names the motion (O15, A6)', '8K → 4K' in obs and '29 → 32' in obs and Z(lang, 'moved down', '往下移') in obs and next_disabled(pg) is False, obs[:200])
    check(f'{tag}: part 5 keeps the previous dots as faint rings (A6)', count(pg, '#rp-vis circle.rp-prev') == 64)
    seg(pg, 'b5', '10000'); slide(pg, '#rp-ic-l5-in', 5)
    obs = text(pg, '#rp-rail [data-obs]')
    check(f'{tag}: part 5 base 10,000 over 128K: 29 → 0 under one turn, the slowest 2.41 turns; the line moved up',
          '29 → 0' in obs and cards(pg) == ['128K', '0 of 64' if lang == 'en' else '0 / 64', '2.41'] and Z(lang, 'moved up', '往上移') in obs, (obs[:300], cards(pg)))
    rules(pg, f'{tag} part 5', lang); shot(pg, f'{tag}-p5')
    # part 6: past the training length, and Llama‑3.1's scaling
    go_part(pg, 5)
    check(f'{tag}: part 6 at 8K: every pair within training; hollow marks for all 64 pairs; Next disabled',
          Z(lang, 'every pair stays within', '每一对都停在训练时见过的角度以内') in text(pg, '#rp-rail [data-obs]') and count(pg, '#rp-vis circle.rp-trn') == 64 and count(pg, '#rp-vis line.rp-stem') == 0 and next_disabled(pg) is True)
    slide(pg, '#rp-ic-l6-in', 4)
    obs = text(pg, '#rp-rail [data-obs]')
    check(f'{tag}: part 6 at 128K: 29 pairs past training, the slowest 16 times; 29 thick stems', '29' in obs and '16' in obs and count(pg, '#rp-vis line.rp-stem') == 29 and next_disabled(pg) is False, obs[:300])
    check(f'{tag}: part 6 reconciles the two checks: 15 of these pairs stay under one turn yet pass training (A7)', Z(lang, 'Staying under one turn is not enough', '不满一圈还不够') in obs and '15' in obs, obs[:400])
    seg(pg, 's6', '1')
    obs = text(pg, '#rp-rail [data-obs]')
    check(f'{tag}: part 6 with the scaling: still 29 past training, only 2 times as far; names it Llama‑3‑8B with Llama‑3.1’s scaling and links the scenario (A8)',
          Z(lang, 'only 2 times', '只转到训练时的 2 倍') in obs and Z(lang, 'with Llama‑3.1’s scaling, before any long-context training', '用了 Llama‑3.1 缩放、但还没做长文本训练') in obs and count(pg, '#rp-rail [data-obs] [data-jump="sc3"]') == 1
          and Z(lang, 'Llama‑3‑8B with Llama‑3.1’s scaling', '用了 Llama‑3.1 缩放的 Llama‑3‑8B') in text(pg, '#rp-st-sub'), (obs[:200], text(pg, '#rp-st-sub')))
    lk = pg.evaluate("""()=>{const a=document.querySelector('#rp-rail [data-obs] [data-jump="sc3"]'); const r=a.getClientRects(), last=r[r.length-1], nx=a.nextSibling; const rg=document.createRange(); rg.setStart(nx,0); rg.setEnd(nx,1); const d=rg.getBoundingClientRect();
      return {disp:getComputedStyle(a).display, sameLine:Math.abs(d.top-last.top)<4&&d.left>=last.right-2}}""")
    check(f'{tag}: part 6 scenario link is plain inline text and its full stop stays on its line (N2)', lk['disp'] == 'inline' and lk['sameLine'], lk)
    rules(pg, f'{tag} part 6', lang); shot(pg, f'{tag}-p6'); hover_tips(pg, f'{tag} part 6')
    slide(pg, '#rp-ic-l6-in', 3)
    check(f'{tag}: part 6 scaled at 64K: every pair within training again', Z(lang, 'with the scaling, every pair stays within', '用了缩放，每一对都停在训练时见过的角度以内') in text(pg, '#rp-rail [data-obs]') and count(pg, '#rp-vis line.rp-stem') == 0, text(pg, '#rp-rail [data-obs]')[:300])
    # the language switch keeps the part, the prediction and the choices
    other = 'en' if lang == 'zh' else 'zh'
    st0 = state(pg); pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400); st1 = state(pg)
    check(f'{tag}: the language switch keeps the part, the prediction and the choices', all(st0[k] == st1[k] for k in ('step', 'pred', 'headView', 'seen', 'rope3', 'shift', 'l5', 'base5', 'l6', 'sc6')), (st0, st1))
    check(f'{tag}: after the switch the result is still shown', text(pg, '#rp-rail [data-obs]').strip() != '')
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    # part 7: what to remember
    go_part(pg, 6)
    check(f'{tag}: part 7 with RoPE: bites → dog 9.64 and 6.00', cards(pg) == ['9.64', '6.00', '3.64'], cards(pg))
    body = text(pg, '#rp-rail .step.current')
    check(f'{tag}: part 7 misconceptions state the pattern, not a detail of an earlier part (G2, 2026-10-08)', Z(lang, 'the longer the text, the larger the base must be', '文本越长，base 就要越大') in body and Z(lang, 'wherever the sentence sits in the text', '不管这句话放在文本的哪里') in body and all(x not in body for x in ('102', '2.41', '29 slow pairs', '29 对慢的数对')), body[:300])
    check(f'{tag}: part 7 has the button to the scenarios (R3.12) and the next-lab footer (R3.11)', pg.evaluate("()=>!!document.querySelector('#rp-rail [data-mode-go=\"sandbox\"]')") and pg.evaluate("()=>!!document.querySelector('.next-lab')"))
    rules(pg, f'{tag} part 7', lang); shot(pg, f'{tag}-p7')
    click(pg, '#rp-rail [data-mode-go="sandbox"]')
    check(f'{tag}: “See the scenarios” opens Scenarios', state(pg)['mode'] == 'sandbox')
    pg.close()


def scenarios(b, w, h, lang):
    tag = f'{w}-{lang}-sb'; pg = page(b, w, h, lang)
    mode(pg, 'sandbox')
    check(f'{tag}: the initial setup is the Llama‑3.1‑8B scenario (pressed)', pg.evaluate("()=>document.querySelector('[data-preset=\"3\"]').getAttribute('aria-pressed')") == 'true')
    for i in range(5):
        click(pg, f'#rp-rail [data-preset="{i}"]')
        pressed = pg.evaluate("()=>[...document.querySelectorAll('[data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        check(f'{tag}: scenario {i} pressed alone', pressed == ['true' if k == i else 'false' for k in range(5)], pressed)
        check(f'{tag}: scenario {i} shows its result', text(pg, '#rp-insight').strip() != '' and overflow(pg) <= 0)
        if i == 2: rules(pg, f'{tag} scenario 3', lang); shot(pg, f'{tag}-scen3'); hover_tips(pg, f'{tag} scenario 3')
    click(pg, '#rp-controls .seg-ctl[data-key="sbase"] [data-v="100000"]')
    check(f'{tag}: after a manual change no scenario is pressed and the card says so (R1.8)', pg.evaluate("()=>[...document.querySelectorAll('[data-preset]')].every(b=>b.getAttribute('aria-pressed')==='false')") and Z(lang, 'match none', '都不一样') in text(pg, '#rp-insight'))
    layout(pg, f'{tag} settings changed')
    hsh = pg.evaluate("()=>location.hash"); sb0 = state(pg)['SB']
    check(f'{tag}: Scenarios writes its setup into the link', hsh.startswith('#rope~'), hsh)
    pg2 = page(b, w, h, lang, hash_=hsh[1:])
    check(f'{tag}: the hash round-trip restores Scenarios and its settings', state(pg2)['mode'] == 'sandbox' and state(pg2)['SB'] == sb0, (state(pg2), sb0))
    pg2.close()
    check(f'{tag}: next-lab footer in Scenarios', pg.evaluate("()=>!!document.querySelector('.next-lab')"))
    if w <= 760:
        pg.evaluate("()=>{const c=document.querySelector('#rp-rail .railcard'); window.scrollTo(0, c.getBoundingClientRect().top+scrollY-60)}"); pg.wait_for_timeout(700)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); if(!d||d.hidden) return null; const s=d.querySelector('.status'); return {status:s?s.textContent:'', read:d.innerText}}")
        check(f'{tag}: the mini result bar shows at 390 once the chart scrolls away, with a short status', dk is not None and len((dk or {}).get('status', '').split()) <= 4, dk)
        layout(pg, f'{tag} dock'); shot(pg, f'{tag}-dock')
    click(pg, '#rp-rail [data-mode-go="challenge"]')
    check(f'{tag}: “Take the challenge” opens the challenge', state(pg)['mode'] == 'challenge')
    pg.close()


CHROWS = """()=>[...document.querySelectorAll('#rp-ch-res .cmp-row')].map(r=>({b:parseInt(r.querySelector('.cmp-lab span').textContent.replace(/[^0-9:：]/g,'')),
  ok:r.classList.contains('ok'),me:r.classList.contains('me'),init:!!r.querySelector('.cmp-tag')}))"""


def challenge(b, w, h, lang):
    tag = f'{w}-{lang}-ch'; pg = page(b, w, h, lang)
    mode(pg, 'challenge')
    check(f'{tag}: Settings hidden with a note (R4.2)', pg.evaluate("()=>[...document.querySelectorAll('#rp-controls .lockable')].every(e=>e.hidden)&&!document.getElementById('rp-lock').hidden"))
    check(f'{tag}: the challenge starts at base 10,000, on the turns chart', state(pg)['chBase'] == 10000 and count(pg, '#rp-vis .rp-pt') == 64)
    check(f'{tag}: the capability card is there', pg.evaluate("()=>!!document.querySelector('[data-exits=\"rope\"]')"))
    rules(pg, f'{tag} start', lang); shot(pg, f'{tag}-start')
    click(pg, '#rp-rail [data-act="submit"]')
    res = text(pg, '#rp-ch-res')
    check(f'{tag}: tier 3 at base 10,000: not met, with a hint and “Show the answer”', Z(lang, 'Not yet', '还没达到') in res and text(pg, '#rp-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), res[:200])
    click(pg, '#rp-ch-res [data-act="reveal"]')
    # the owner's review of batch 8 (2026-10-07): the answer keeps the bases that pass, the pick and the initial setup (here the same row);
    # base 100,000 wraps around and is only counted in the note
    rw = pg.evaluate(CHROWS)
    check(f'{tag}: “Show the answer” shows 3 rows: 10,000 (initial setup, your pick, red), 500,000 and 1,000,000 (green); the note counts 1 left out',
          [r['b'] for r in rw] == [10000, 500000, 1000000] and [r['ok'] for r in rw] == [False, True, True] and rw[0]['me'] and rw[0]['init']
          and text(pg, '#rp-ch-res .cmp-note') == Z(lang, 'The other base wraps around within 1M tokens and is not shown.', '其余 1 种 base 在 1M 个 token 内会转过一整圈，图里没有列出。'), rw)
    check(f'{tag}: the chart title names the quantity and the goal, not “each base”', text(pg, '#rp-ch-res .cmp-title') == Z(lang, 'Tokens the slowest pair needs for one full turn, with heads of 128 numbers (goal: more than the 1M-token text)', '128 维 head 的最慢一对转满一圈需要多少个 token（目标：比 1M token 的文本更长）'), text(pg, '#rp-ch-res .cmp-title'))
    res = text(pg, '#rp-ch-res')
    check(f'{tag}: the text after the chart names base 100,000 in words, without its row', Z(lang, 'With base 100,000 it still wraps around about twice', 'base 100,000 时它仍会转过大约两圈') in res, res[-500:])
    click(pg, '#rp-ic-ch [data-lever="1000000"]')
    res = text(pg, '#rp-ch-res')
    check(f'{tag}: tier 2 at base 1,000,000: met, but a smaller base works', Z(lang, 'a smaller base works too', '更小的 base 也行') in res, res[:300])
    click(pg, '#rp-ic-ch [data-lever="500000"]')
    res = text(pg, '#rp-ch-res')
    check(f'{tag}: tier 1 at base 500,000: met and the smallest', Z(lang, 'it is the smallest base that works', '这是能达标的最小 base') in res, res[:300])
    rw = pg.evaluate(CHROWS)
    check(f'{tag}: the chart keeps 10,000 (initial setup) and the two passing bases, 500,000 (your pick) and 1,000,000; 100,000 is left out',
          [r['b'] for r in rw] == [10000, 500000, 1000000] and [r['ok'] for r in rw] == [False, True, True] and rw[1]['me'] and rw[0]['init'] and not rw[0]['me'], rw)
    check(f'{tag}: the readout and the card agree: 0.41 turns', text(pg, '#rp-ch-now') == Z(lang, '0.41 turns', '0.41 圈') and cards(pg)[1] == '0.41', (text(pg, '#rp-ch-now'), cards(pg)))
    rules(pg, f'{tag} tier 1', lang); shot(pg, f'{tag}-tier1'); hover_tips(pg, f'{tag} tier 1')
    other = 'en' if lang == 'zh' else 'zh'
    st0 = state(pg); pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400); st1 = state(pg)
    check(f'{tag}: the language switch keeps the challenge’s answer', all(st0[k] == st1[k] for k in ('mode', 'chBase', 'chSubmitted', 'chReveal')) and text(pg, '#rp-ch-res').strip() != '', (st0, st1))
    check(f'{tag}: next-lab footer in the challenge', pg.evaluate("()=>!!document.querySelector('.next-lab')"))
    check(f'{tag}: no overflow in the challenge', overflow(pg) <= 0, overflow(pg))
    pg.close()


def layout_pass(b, w, h, lang):
    """R7.8 at a middle width: every part, Scenarios and the challenge, fit checks only"""
    tag = f'{w}-{lang}'; pg = page(b, w, h, lang)
    steps = [(0, None), (1, lambda: (seg(pg, 'hv', 'pairs'), seg(pg, 'hv', 'arrows'))), (2, lambda: seg(pg, 'rope3', 'on')),
             (3, lambda: (click(pg, '#rp-rail [data-pred="same"]'), slide(pg, '#rp-ic-shift-in', 100))), (4, lambda: seg(pg, 'b5', '10000')),
             (5, lambda: (slide(pg, '#rp-ic-l6-in', 4), seg(pg, 's6', '1'))), (6, None)]
    for i, act in steps:
        go_part(pg, i); layout(pg, f'{tag} part {i + 1}'); stage_fits(pg, f'{tag} part {i + 1}')
        if act: act(); layout(pg, f'{tag} part {i + 1} after its action'); stage_fits(pg, f'{tag} part {i + 1} after its action')
        if i in (1, 3, 5): shot(pg, f'{tag}-p{i + 1}')
    mode(pg, 'sandbox'); layout(pg, f'{tag} scenarios'); shot(pg, f'{tag}-sb')
    mode(pg, 'challenge'); click(pg, '#rp-ic-ch [data-lever="100000"]'); click(pg, '#rp-rail [data-act="submit"]'); click(pg, '#rp-ch-res [data-act="reveal"]')
    layout(pg, f'{tag} challenge revealed'); shot(pg, f'{tag}-ch')
    check(f'{tag}: layout pass visited every part and mode', state(pg)['mode'] == 'challenge')
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
    n = pg.evaluate("r=>new Set([...document.querySelectorAll(r+' [data-go]')].map(e=>e.dataset.go)).size", '#rp-rail')
    for k in range(n):
        pg.evaluate("([r,k])=>{const e=document.querySelector(r+' [data-go=\"'+k+'\"]'); if(e) e.click();}", ['#rp-rail', k]); pg.wait_for_timeout(500)
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
    for w, h in [(1440, 900), (390, 844)]:
        if w not in WIDTHS: continue
        for lang in LANGS:
            guided(b, w, h, lang); scenarios(b, w, h, lang); challenge(b, w, h, lang)
    if 900 in WIDTHS:
        for lang in LANGS: layout_pass(b, 900, 900, lang)
    if 1440 in WIDTHS:
        for lang in LANGS: leak_scan(b, 1440, 900, lang)
    if 1440 in WIDTHS and 'en' in LANGS:
        pg = page(b, 1440, 900, 'en', dark=True); go_part(pg, 1); seg(pg, 'hv', 'arrows'); shot(pg, 'dark-1440-p2')
        go_part(pg, 5); slide(pg, '#rp-ic-l6-in', 4); shot(pg, 'dark-1440-p6'); layout(pg, 'dark 1440 part 6'); pg.close()
    # the owner's decisions of 2026-10-09 (H1, H2): Show the math is a few blocks, each a lead line in words and a typeset formula (MathML),
    # with a hairline between blocks; no TeX command shown as a word; nothing clipped
    for lang in ('en', 'zh'):
        pg = page(b, 1440, 900, lang)
        for i in range(7):
            go_part(pg, i); pg.wait_for_timeout(300)
            pg.evaluate("()=>{document.getElementById('rp-math').open=true}"); pg.wait_for_timeout(150)
            mb = pg.evaluate(MATHBLOCKS, ['#rp-mathbody', 'rp'])
            if not mb or mb['n'] == 0: continue
            check(f'{lang} part {i+1} H2: Show the math in blocks (lead line, typeset formula, hairline between), no TeX word, nothing clipped',
                  mb['lead'] >= mb['n'] - 1 and mb['fx'] >= mb['n'] - 1 and mb['ruled'] == mb['n'] - 1 and not mb['words'] and mb['clip'] == 0, mb)
            t = pg.evaluate("s=>{const e=document.querySelector(s).cloneNode(true); e.querySelectorAll('math').forEach(m=>m.remove()); return e.textContent}", '#rp-mathbody')   # the text outside the typeset formulas
            check(f'{lang} part {i+1} H1: no plain-text formula left in Show the math (^(, ÷ 2π, q·k, Σ as text)', not any(x in t for x in ['^(', '÷ 2π', 'q·k', 'Σ']), t[:300])
        pg.close()
    b.close()

check('no page errors', not errs, errs[:5])
for hit in sorted(set(rule_hits))[:40]: print('RULE', hit)
check('writing and colour rules', not rule_hits, len(set(rule_hits)))
for hit in sorted(set(layout_hits))[:40]: print('LAYOUT', hit)
check('R1.1 cursor:help has a tooltip; R7.8 no text wider than its capsule, SVG texts fit and do not overlap; the stage fits', not layout_hits, len(set(layout_hits)))
for hit in sorted(set(tip_hits))[:20]: print('TIP', hit)
check('R1.1 every hovered kind of help element shows its tooltip', not tip_hits, len(set(tip_hits)))
# G1 and G2, the owner's site-wide decisions of 2026-10-08 (sweep): option lead-ins instead of “one at a time” notes, and no sentence
# that leans on a detail of an earlier part. Read from the lab's source, so both languages are covered.
_G_SRC = (P / 'labs' / 'rope.js').read_text()
for _bad in ['一次只能选', '一次选一个', '不会叠加', 'do not stack', 'One at a time', 'one at a time']:
    check(f'G1: no “{_bad}” left in the lab (EN and ZH)', _bad not in _G_SRC)
for _lead in ['Pick a scenario and see how many turns each pair makes over the whole text.', '选一个场景，看看每一对在整段文本上转几圈。']:
    check(f'G1: the lead-in “{_lead[:36]}…” is there', _lead in _G_SRC)
for _old, _new in [('The arcs in ${PL(3', 'So position changes only the scores, never the content each token hands over.'), ('的弧线画出了每一次转动', '所以位置只影响得分，不改动每个 token 交出去的内容。'), ('29 slow pairs of ${M3} went past', 'To read texts longer than its training texts, a model has to slow those pairs'), ('有 29 对慢的数对超出了 8K 训练时见过的角度', '要读比训练文本更长的文本，模型就得放慢这些数对'), ('the same two tokens scored ${sc2(s4)}', 'so the same two words score the same wherever the sentence sits in the text.'), ('同样两个 token 在位置 2 和 1 上得', '不管这句话放在文本的哪里，得分都一样。'), ('base 10,000 makes even the slowest pair wrap around', 'So the longer the text, the larger the base must be'), ('base 10,000 让最慢的一对也转过一整圈', '所以文本越长，base 就要越大'), ('Moving both tokens later changed nothing', 'so a head reads only how far apart two tokens are'), ('两个 token 一起往后挪，什么也没变', '所以 head 读到的只是两者相距多远')]:
    check(f'G2: “{_old[:30]}…” is rewritten as “{_new[:30]}…”', _old not in _G_SRC and _new in _G_SRC)
print(f'{n - len(fails)} / {n} passed')
# the owner's decisions of 2026-10-09 (H1): no plain-text formula of the kinds converted left in the lab's source outside comments
_H_SRC = (P / 'labs' / 'rope.js').read_text()
_H_CODE = __import__('re').sub(r'/\*.*?\*/', '', _H_SRC, flags=__import__('re').S)
_H_CODE = '\n'.join(l for l in _H_CODE.split('\n') if not l.lstrip().startswith('#'))
for _bad in ['q<sub>', 'θ<sub>', '<sup>−2i/128</sup>', 'base^(−2i', '^(−', '2π ÷ θ', '1 ÷ base', '2π × base', '180° ÷ π', 'q·k', '√128', "'2π ≈ 6.2832", 'One full turn is 2π radians', '一整圈是 2π 弧度']:
    check(f'H1: “{_bad}” is gone from the lab', _bad not in _H_CODE)
check('H1: formulas use A.ui.math', _H_CODE.count('MX(') >= 30, _H_CODE.count('MX('))
print(f'{n} checks after H1, {len(fails)} failed')
sys.exit(1 if fails else 0)
