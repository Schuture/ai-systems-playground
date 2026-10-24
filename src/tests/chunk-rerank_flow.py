"""Flow test for the chunk-rerank lab (CONTRACT §6), at 1440×900 and 390×844, in EN and ZH.

- no page errors, no horizontal overflow, no text wider than its box, a tooltip for every help cursor, chart labels inside their picture;
- part 1: the document strip and the vector list; the cut plays on first open at 中 (R3.9), recall waiting for its end (R3.8); the
  mission answer (E-203) is cut apart (red); the card names the engine's found count, recall@1 and the ranks of the two halves; at 390
  the strip's text is drawn at full size (A21); Next enabled;
- part 2: entering replays nothing (64 tokens, no overlap was watched); Next waits for another chunking, with the kit's why line;
  picking the chunking already shown replays nothing; a new size plays its cut in about 2.4 s; going back to a watched one replays
  nothing; 重放 replays from the start; the progress scrubber pauses at the point picked (R3.20); while a cut plays, recall, the metric
  cards and the observation wait, and the ▲▼ come when it ends and stay (R3.8, P4); the observation names the engine's count of
  answers cut apart; Show me picks 64 tokens with half a chunk of overlap; on a phone, a cut started from the card brings the strip
  into view and waits for the scroll, the strip stays put when it ends, and a line under it leads back to the observation (R7.12, R3.2);
- part 3: Next waits for a prediction; no ranking is shown before it (R3.8);
- part 4: the verdict, with the engine's rank of the E-201 chunk in vector search; RRF typeset; three lists of returned chunks only;
  hybrid rows name where their shares come from and the tie that puts the E-201 chunk first is marked; a chunk clicked is marked in
  every list; Next waits for another question; the observation names the stairs question's three places; Back and Next replay
  nothing (R3.20);
- part 5: vector search, the thermostat-battery question; Next waits for rerank; before rerank is on its column holds no ranking and
  no coverage (R3.8); turning it on plays rerank in about 1.6 s at 中: the readouts wait, the candidates' coverage shows one by one,
  and the rows reach the rerank column, the right chunk first; the observation names its move and recall@1 before and after; a new
  question replays it, a watched one does not, 重放 rerank does (R3.20); the E-201 question is outside the candidates; on a phone,
  turning rerank on from the card brings the lists into view first, and a line under them leads back to the observation (the cut
  part 1 plays by itself ends without that line);
- part 6: the chart first, then the lists and the metric cards (R7.12); Next waits for another k or size;
- part 7: the bullets name the engine's ranks, the scenarios button, the next-lab footer;
- Scenarios: six; a watched one does not replay, a new one plays; each has its insight; the hash round trip; the language switch
  keeps the setup; the dock at 390;
- the challenge: starts at the initial setup with no result before Submit; a miss names what is missed, a hint and “Show the answer”;
  a new choice after a miss needs a new Submit; the best answer (64 tokens, overlap, hybrid + rerank, keep 1) meets the goal with the
  fewest tokens, and the why names each search's pool and the best setting with one choice changed; the chart of the settings that meet it, with the initial setup and the pick tagged; the capability card;
- the writing rules on every visible text and tooltip (no “toy” outside the Toy run label, Chinese spacing and punctuation, two
  colons in one sentence, no 万 or 亿), the colours (red only for a miss; blue, amber and violet not used), every formula typeset;
  and, read from the source, no “one at a time” note and no sentence that recalls a part by number.
Screenshots: shots/chunk-rerank/.
Run: python3 tests/chunk-rerank_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
(P / 'out' / 'chunk-rerank').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/chunk-rerank.html', '--only', 'chunk-rerank'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'chunk-rerank' / 'flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'chunk-rerank.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'chunk-rerank' / 'flow-wrapped.html')
SH = P / 'shots' / 'chunk-rerank'; SH.mkdir(parents=True, exist_ok=True)
ARGS = sys.argv[1:]
LANGS = [x for x in ('en', 'zh') if x in ARGS] or ['en', 'zh']
SIZES = [s for s in [(1440, 900), (390, 844)] if str(s[0]) in ARGS] or [(1440, 900), (390, 844)]
errs, fails, n = [], [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


Z = lambda lang, en, zh: zh if lang == 'zh' else en
pct = lambda v: f'{round(v * 100)}%'


def page(b, w, h, lang, hash_='chunk-rerank'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__ckr.state()")
anim = lambda pg: pg.evaluate("()=>window.__ckr.animState()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-').replace(/\\u2212/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#ckr-rail [data-nav=\"next\"]')].filter(b=>b.getClientRects().length).map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#ckr-rail [data-next-why]')].filter(p=>!p.hidden&&getComputedStyle(p).visibility!=='hidden'&&p.getClientRects().length).map(p=>p.textContent)")
speed = lambda pg: pg.evaluate("()=>{const b=document.querySelector('#ckr-transport .seg-ctl[data-key=\"speed\"] button[aria-pressed=\"true\"]'); return b?b.dataset.v:null}")
MEAS = """([lang,s])=>window.__ckr.measure(lang,s)"""
RANKS = """([lang,size,ov,q])=>window.__ckr.setup(lang,size,ov).Q[q].rk"""
meas = lambda pg, lang, s: pg.evaluate(MEAS, [lang, s])
ranks = lambda pg, lang, size, ov, q: pg.evaluate(RANKS, [lang, size, ov, q])


def settle(pg, ms=2000):
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel, wait=250):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(wait); settle(pg)


# the animation state two frames after a click: None means nothing replays
CLICK_ANIM = """async (sel)=>{document.querySelector(sel).click(); await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))); return window.__ckr.animState();}"""


# the readouts two frames after a click: the cut or rerank under way, recall, the metric cards, the observation, the rerank column
CLICK_PEND = """async (sel)=>{document.querySelector(sel).click(); await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
  const col=document.querySelector('#ckr-cols .ckr-col[data-c$="1"]'), rows=col?[...col.querySelectorAll('.ckr-li')]:[], obs=document.querySelector('#ckr-rail [data-obs]');
  return {cut:window.__ckr.animState(), rr:window.__ckr.rrState(), status:document.querySelector('#ckr-status').innerText,
    metPend:document.querySelector('#ckr-metrics').classList.contains('ckr-pend'), deltas:[...document.querySelectorAll('#ckr-metrics .d.on')].length,
    obs:obs&&!obs.hidden&&!obs.classList.contains('ckr-pend')?obs.innerText:'', obsPend:!!obs&&!obs.hidden&&obs.classList.contains('ckr-pend'),
    rows:rows.length, hidden:rows.filter(r=>getComputedStyle(r).opacity==='0').length};}"""
RR_NOW = """()=>{const col=document.querySelector('#ckr-cols .ckr-col[data-c$="1"]'), rows=col?[...col.querySelectorAll('.ckr-li')]:[], obs=document.querySelector('#ckr-rail [data-obs]');
  const cov=[...document.querySelectorAll('#ckr-cols .ckr-cov')];
  return {rr:window.__ckr.rrState(), rows:rows.length, visible:rows.filter(r=>getComputedStyle(r).opacity==='1').length, first:rows[0]?rows[0].className:'',
    cov:cov.length, covShown:cov.filter(c=>parseFloat(getComputedStyle(c).opacity)>0.5).length, metPend:document.querySelector('#ckr-metrics').classList.contains('ckr-pend'),
    status:document.querySelector('#ckr-status').innerText, obs:obs&&!obs.hidden&&!obs.classList.contains('ckr-pend')?obs.innerText:''};}"""


# the marks on the list rows (right, part of the answer, tie) and the verdict lines under the lists, with the cut's state
LIST_MARKS = """()=>({cut:!!window.__ckr.animState(), rows:document.querySelectorAll('#ckr-cols .ckr-li').length, half:document.querySelectorAll('#ckr-cols .ckr-li.half').length,
  marks:document.querySelectorAll('#ckr-cols .ckr-li.ok, #ckr-cols .ckr-li.bad, #ckr-cols .ckr-li.half, #ckr-cols .ckr-li.also, #ckr-cols .ckr-tie').length, verdicts:document.querySelectorAll('#ckr-cols .ckr-bad').length})"""
# a new question picked in the select, two frames later: the cut, recall, the metric cards, the scenario's conclusion
QCHANGE = """async (qi)=>{const s=document.querySelector('#ckr-qsel'); s.value=String(qi); s.dispatchEvent(new Event('change',{bubbles:true})); await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
  const ins=document.querySelector('#ckr-insight'); return {cut:window.__ckr.animState(), status:document.querySelector('#ckr-status').innerText, metPend:document.querySelector('#ckr-metrics').classList.contains('ckr-pend'), ins:ins?ins.innerText:''};}"""
# the numbers in front of the worked formulas under “Show the math”: they count 1, 2, 3 … in the language shown
MATH_NOS = """()=>[...document.querySelectorAll('#ckr-mathbody .ckr-mb > .ckr-mb-lead:first-child')].map(e=>(e.textContent.match(/^(\\d+)\\. /)||[])[1]).filter(Boolean).map(Number)"""

# phones (P35 of the polish review): the current card scrolled up near the top, as a reader has it when using its controls; a panel's
# top on screen; the line under a panel that leads back to the conclusion
TO_CARD = "()=>{const c=document.querySelector('#ckr-rail .step.current'); window.scrollTo(0,c.getBoundingClientRect().top+scrollY-130)}"
PAN_TOP = "s=>Math.round(document.querySelector(s).getBoundingClientRect().top)"
BACK_LINE = "()=>{const b=document.querySelector('#ckr-back'); return b&&!b.hidden&&b.getClientRects().length?{t:b.innerText, pan:(b.closest('section')||{}).id||''}:null}"


def wait_rr(pg, timeout=15000):
    pg.wait_for_function("()=>!window.__ckr.rrState()", timeout=timeout); pg.wait_for_timeout(150)


def click_anim(pg, sel):
    a = pg.evaluate(CLICK_ANIM, sel); pg.wait_for_timeout(150); settle(pg); return a


def wait_anim(pg, timeout=25000):
    pg.wait_for_function("()=>!window.__ckr.animState()", timeout=timeout); pg.wait_for_timeout(150)


def mode(pg, m):
    click(pg, f'#ckr-app .modes button[data-mode="{m}"]', 350)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · (?![0-9])/,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],
   [/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy”'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/\\bNaN\\b(?![?？])|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],
   [/吃|喂|扛|吞/,'figurative verb'],[/一次只能选|不会叠加|第 [0-9一二三四五六七八] 部分里|\\bin part [0-9]\\b|part [0-9]’s/,'recall of a part or a one-at-a-time note'],
   [/深色|浅色|浅灰|深灰|\\b(?:dark|light|pale)(?:er)? (?:bar|line|block|part)/i,'colour named by lightness (R5.7)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, .fid-tag')||!e.getClientRects().length) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); for(const [re,name] of bad) if(re.test(t)) out.push(name+' (tooltip): '+t.slice(0,90));});
 return out;}"""
SPACING = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), bad=/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>%][\\u4e00-\\u9fff]|[）】][ ][\\u4e00-\\u9fff]/;
 const shown=e=>e&&!e.closest('[data-endnotes],pre,code,svg,.mathbody,#tip,.ckr-tx,.ckr-sn,option,select')&&e.getClientRects().length;
 const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){if(!shown(n.parentElement)) continue; const t=n.textContent.replace(/\\s+/g,' '); const m=t.match(new RegExp(bad.source,'g')); if(m) out.add(m.join('|')+' in: '+t.trim().slice(0,80));}
 root.querySelectorAll('a,.def,b,.t,em,code').forEach(e=>{if(!shown(e)||getComputedStyle(e).display!=='inline') return;
   const t=e.textContent; if(!t) return; const prev=e.previousSibling, next=e.nextSibling;
   const pc=prev&&prev.nodeType===3?prev.textContent.slice(-1):'', nc=next&&next.nodeType===3?next.textContent.slice(0,1):'';
   const pair=(a,b)=>a&&b&&bad.test(a+b);
   if(pair(pc,t[0])) out.add(pc+t[0]+' before: '+t.slice(0,40)); if(pair(t[t.length-1],nc)) out.add(t[t.length-1]+nc+' after: '+t.slice(-40));});
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx').replace(/<math[\\s\\S]*?<\\/math>/g,'M'); const m=t.match(/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>][\\u4e00-\\u9fff]/g); if(m) out.add(m.join('|')+' in tooltip: '+t.slice(0,80));});
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
 root.querySelectorAll('p,li,button,label,div,span,h1,h2,h3,summary').forEach(e=>{if(e.closest('[data-endnotes],pre,code,svg,.mathbody,.ckr-tx,.ckr-sn')||!e.getClientRects().length) return; if(e.querySelector('p,li,div,ul,ol,h1,h2,h3,table')) return;
   if(![...e.childNodes].some(c=>c.nodeType===3&&c.textContent.trim())) return; const c=e.cloneNode(true); c.querySelectorAll('code').forEach(x=>x.remove()); texts.push(c.textContent);});
 root.querySelectorAll('[data-tipx]').forEach(e=>texts.push(e.getAttribute('data-tipx')));
 for(const t of texts){const sents=t.replace(/\\s+/g,' ').split(/(?<=[。？！])|(?<=[.?!])\\s+/);
   for(const x of sents){const k=(x.match(/：|: /g)||[]).length; if(k>=2) out.add(x.trim().slice(0,160));}}
 return [...out];}"""
# red only for a miss (an answer cut apart, the right chunk not kept, a limit not met, the failed verdict); blue, amber and violet are reserved
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/\\bbad\\b|ckr-cut|ckr-bad|ckr-sw-bad|badline|status bad|\\bover\\b|ex-res/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#ckr-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())&&!(p==='stroke'&&s.strokeWidth==='0px')) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#ckr-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg,.ckr-math-line')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(cs.textOverflow==='ellipsis'&&e.closest('.ckr-sn,option,select')) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#ckr-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
SVGIN = """()=>{const out=[]; for(const s of document.querySelectorAll('#ckr-stage svg')){if(!s.getClientRects().length) continue; const b=s.getBoundingClientRect();
  const ts=[...s.querySelectorAll('text')].filter(t=>t.getClientRects().length&&t.textContent.trim());
  for(const t of ts){const r=t.getBoundingClientRect(); if(r.left<b.left-1||r.right>b.right+1||r.top<b.top-1||r.bottom>b.bottom+1) out.push((s.id||'svg')+' outside: '+t.textContent.trim().slice(0,30));}}
  return out.slice(0,8);}"""
TEXWORDS = """()=>{const W=new Set(['frac','left','right','sqrt','text','sum','cdot','quad','mathrm','times']);
  const out=[]; for(const m of document.querySelectorAll('#ckr-app math, #dock math')) for(const e of m.querySelectorAll('mi,mo')){const t=e.textContent.trim(); if(W.has(t)) out.push(t);}
  const pat=[/\\\\frac|\\\\sum|\\\\text|k_1 \\(|1\\/\\(60/];
  const tw=document.createTreeWalker(document.querySelector('#ckr-app'),NodeFilter.SHOW_TEXT); let n;
  while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('math,pre,code')||!e.getClientRects().length) continue; for(const r of pat) if(r.test(n.textContent)) out.push('text: '+n.textContent.trim().slice(0,60));}
  return out;}"""
ALIGN = """()=>{const els=[...document.querySelectorAll('#ckr-transport .btn, #ckr-transport .seg-ctl')].filter(e=>e.getClientRects().length);
  const rows=[]; for(const e of els){const r=e.getBoundingClientRect(); let row=rows.find(x=>Math.abs(x.top-r.top)<8); if(!row){row={top:r.top,items:[]}; rows.push(row);}
    const b=e.matches('.seg-ctl')?e.querySelector('button'):e, rg=document.createRange(); rg.selectNodeContents(b); const t=rg.getBoundingClientRect();
    row.items.push({t:(e.textContent||'').trim().slice(0,12),top:r.top,h:r.height,base:t.bottom});}
  const bad=[]; for(const row of rows){const it=row.items; for(const x of it){if(Math.abs(x.h-it[0].h)>0.6||Math.abs(x.top-it[0].top)>0.6||Math.abs(x.base-it[0].base)>1) bad.push(x.t+' vs '+it[0].t+' h '+x.h+'/'+it[0].h+' base '+x.base.toFixed(1)+'/'+it[0].base.toFixed(1));}}
  return {n:els.length,bad};}"""
rule_hits = []


def rules(pg, where, lang):
    for sel in ('#ckr-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')
    for h in pg.evaluate(OVER): rule_hits.append(f'{where}: text wider than its box: {h}')
    for h in pg.evaluate(TEXWORDS): rule_hits.append(f'{where}: a formula not typeset or a TeX word shown: {h}')
    for h in pg.evaluate(HELP): rule_hits.append(f'{where}: help cursor without a tooltip: {h}')
    for h in pg.evaluate(SVGIN): rule_hits.append(f'{where}: chart label outside its picture: {h}')
    o = overflow(pg)
    if o > 0: rule_hits.append(f'{where}: horizontal overflow {o}px')


NEXT = '#ckr-rail .step.current [data-nav="next"]'
BACK = '#ckr-rail .step.current [data-nav="back"]'
SEG = lambda k, v: f'#ckr-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]'
INIT = {'size': 64, 'ov': 0, 'ret': 'vec', 'rr': 0, 'k': 1}


def go_next(pg):
    sel = NEXT if pg.query_selector(NEXT) else '#ckr-rail [data-nav="next"]'
    click(pg, sel, 300)


def timed_run(pg, sel, timeout=20000):
    t0 = pg.evaluate('()=>performance.now()'); a = pg.evaluate(CLICK_ANIM, sel)
    pg.wait_for_function("()=>!window.__ckr.animState()", timeout=timeout)
    return a, (pg.evaluate('()=>performance.now()') - t0) / 1000


src = (P / 'labs' / 'chunk-rerank.js').read_text()
check('source: no “one at a time” note and no sentence that recalls a part by number', not re.search(r'一次只能选|不会叠加|第 [0-9一二三四五六七八] 部分里', src), re.findall(r'.{20}(?:一次只能选|不会叠加|第 [0-9一二三四五六七八] 部分里).{20}', src)[:3])

# B4 (polish round 5, reviews/r5-fit-agents-eval.md): entering parts 1, 4 and 6 at 1440 × 900 the way the polish probe stage.py does
# (Next when it is enabled, else the part's heading), the stage needs no scrolling inside: Chinese with at least 30 px to spare,
# English with none (spare = the stage's max-height minus what its content and borders need). Its top is not under the top bar,
# and the main picture (the stage's largest svg or canvas, if it has one) is whole on the screen.
# The top bar's bottom is where it is on the screen now: on a desktop lab page the bar scrolls away (refit step 2), on a phone it
# stays on top and this is its height.
B4_FIT = """(id)=>{const s=document.getElementById(id), r=s.getBoundingClientRect(), b=document.querySelector('.topbar'), tb=b?Math.max(0,Math.round(b.getBoundingClientRect().bottom)):0;
  const cur=document.querySelector('#'+id.replace(/-stage$/,'-rail')+' .steps>.step.current');
  const f=[...s.querySelectorAll('svg,canvas')].filter(e=>e.getClientRects().length&&!e.closest('button,.btn')).map(e=>e.getBoundingClientRect()).sort((a,c)=>c.width*c.height-a.width*a.height)[0];
  return {part:cur?[...cur.parentNode.children].indexOf(cur)+1:0, spare:Math.round(parseFloat(getComputedStyle(s).maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight)),
    scroll:s.scrollHeight-s.clientHeight, top:Math.round(r.top), bar:tb, fig:f?[Math.round(f.top),Math.round(f.bottom)]:null, figOn:!f||(f.top>=tb-2&&f.bottom<=innerHeight+2)};}"""


def b4_fit(b, lang, parts=(1, 4, 6)):
    pg = page(b, 1440, 900, lang)
    for k in range(1, max(parts) + 1):
        if k > 1:
            nb = pg.locator('.pbar [data-nav="next"]:visible:not([disabled])')
            if nb.count(): nb.first.click()
            else: pg.evaluate(f"()=>document.querySelector('#ckr-rail .steps>.step:nth-child({k}) .step-h').click()")
            pg.wait_for_timeout(900)
        if k not in parts: continue
        wait_anim(pg)   # part 1's cut plays on entry; its lists get their marks and verdicts when it ends, so measure the stage then
        y = None
        for _ in range(20):   # the page's scroll has settled (the rail's reveal, then the lab's keepStage)
            y2 = pg.evaluate('()=>scrollY')
            if y2 == y: break
            y = y2; pg.wait_for_timeout(150)
        f = pg.evaluate(B4_FIT, 'ckr-stage')
        check(f'{lang}-1440 part {k}: entering it, the stage fits without scrolling inside{", 30 px to spare" if lang == "zh" else ""}; its top is below the top bar and its main picture whole on the screen (B4, R7.12)',
              f['part'] == k and f['scroll'] <= 0 and f['spare'] >= (30 if lang == 'zh' else 0) and f['top'] >= f['bar'] - 1 and f['figOn'], f)
    pg.close()


with sync_playwright() as pw:
    b = pw.chromium.launch()
    if (1440, 900) in SIZES:
        for lang in LANGS: b4_fit(b, lang)
    for lang in LANGS:
        for (w, h) in SIZES:
            phone = w <= 760; tag = f'{lang}-{w}'
            pg = page(b, w, h, lang)
            st = state(pg); a0 = anim(pg)
            check(f'{tag}: opens on part 1 of the guided tour at the initial setup', st['mode'] == 'guided' and st['step'] == 0 and all(st[k] == v for k, v in INIT.items()) and st['q'] == 5, st)
            # ---- part 1
            check(f'{tag} part 1: the cut plays on first open, at 中 (R3.9)', a0 is not None and speed(pg) == 'mid', (a0, speed(pg)))
            stp = text(pg, '#ckr-status')
            check(f'{tag} part 1: while the cut plays, recall waits for its end (R3.8, P4)', Z(lang, 'updates when the cut is done', '切完后更新') in stp and '%' not in stp, stp)
            mk = pg.evaluate(LIST_MARKS)
            check(f'{tag} part 1: while the cut plays, the list shows its rows only, no answer marks or verdicts under it (R3.8, N11)', mk['cut'] and mk['rows'] > 0 and mk['marks'] == 0 and mk['verdicts'] == 0, mk)
            wait_anim(pg)
            mk = pg.evaluate(LIST_MARKS)
            check(f'{tag} part 1: when the cut ends, the list marks the two halves and says no chunk holds the whole answer (N11)', not mk['cut'] and mk['half'] == 2 and mk['verdicts'] == 1, mk)
            if phone: check(f'{tag} part 1: the cut a part plays by itself ends without the line back to the conclusion (P35: only after the learner’s own playback)', pg.evaluate(BACK_LINE) is None, pg.evaluate(BACK_LINE))
            m0 = meas(pg, lang, INIT)
            cur = text(pg, '#ckr-rail .step.current')
            q5 = pg.evaluate("l=>{const R=window.__ckr.setup(l,64,0), q=R.Q[5], v=q.lists.vec; return {r:q.part.map(j=>v.indexOf(j)+1).filter(x=>x>0).sort((a,b)=>a-b), one:q.part.includes(v[0]), cut:!q.right.length}}", lang)
            hr = Z(lang, ' and ', '、').join(map(str, q5['r']))
            check(f'{tag} part 1: the card names {m0["hits"]} found, recall@1 {pct(m0["rec"])} and the halves at {q5["r"]}, from the engine', q5['cut'] and len(q5['r']) == 2 and Z(lang, f'{m0["hits"]} of the 20 are found now', f'现在找到 {m0["hits"]} 个') in cur and f'recall@1 {pct(m0["rec"])}' in cur.replace(' of ', ' ') and Z(lang, f'The halves rank {hr}', f'两半在舞台上的列表里排第 {hr} 名') in cur and Z(lang, 'holds only one of them' if q5['one'] else 'holds neither', '只有其中一半' if q5['one'] else '一样也没有') in cur, (q5, cur[-500:]))
            check(f'{tag} part 1: the document strip and one list are shown; Next enabled', vis(pg, '#ckr-strip') and vis(pg, '#ckr-pan-lists') and pg.evaluate("()=>document.querySelectorAll('#ckr-cols .ckr-col').length") == 1 and not any(nexts(pg)), nexts(pg))
            cutbad = pg.evaluate("()=>document.querySelectorAll('#ckr-strip .ckr-ans.bad, #ckr-strip .ckr-cut').length")
            check(f'{tag} part 1: the mission answer is cut apart, marked red (R5.11)', cutbad > 0 and Z(lang, 'The cut falls inside the answer', '一刀正好切在答案中间') in cur and Z(lang, 'no single chunk can give the whole answer', '没有哪一块能单独给出完整的答案') in text(pg, '#ckr-text'), cutbad)
            sc = pg.evaluate("()=>{const s=document.querySelector('#ckr-strip'), vb=s.viewBox.baseVal, t=s.querySelector('text.ckr-chn'); return {scale:s.clientWidth/vb.width, font:t?parseFloat(getComputedStyle(t).fontSize):0, cap:document.querySelector('#ckr-d-cap').textContent, cnt:document.querySelector('#ckr-d-cnt').textContent}}")
            check(f'{tag} part 1: the strip\'s numbers are drawn at {sc["font"] * sc["scale"]:.1f} px, its title and count in HTML (A21)', sc['font'] * sc['scale'] >= 9.5 and sc['cap'].count(':') + sc['cap'].count('：') == 0 and sc['cnt'], sc)
            al = pg.evaluate(ALIGN)
            check(f'{tag} part 1: Cut, the speed buttons and Replay share one height and one baseline', al['n'] == 3 and not al['bad'], al)
            shot(pg, f'{tag}_p1', full=phone); rules(pg, f'{tag} part 1', lang)
            # ---- part 2
            a = click_anim(pg, NEXT); st = state(pg)
            check(f'{tag} part 2: entering replays nothing, the watched cut is shown (R3.20)', a is None and st['step'] == 1, (a, st))
            check(f'{tag} part 2: Next disabled with the why line until another chunking is tried', all(nexts(pg)) and any(Z(lang, 'Try another chunk size', '换一种块大小') in x for x in why(pg)), (nexts(pg), why(pg)))
            a = click_anim(pg, SEG('size', 64))
            check(f'{tag} part 2: picking the size already shown replays nothing (R3.20)', a is None and all(nexts(pg)), a)
            t0 = pg.evaluate('()=>performance.now()'); pn = pg.evaluate(CLICK_PEND, SEG('size', 32))
            a = pn['cut']
            check(f'{tag} part 2: a new size plays its cut (R3.20)', a is not None and state(pg)['size'] == 32, a)
            check(f'{tag} part 2: while it plays, recall and the metric cards wait, with no ▲▼ yet (R3.8, P4)', Z(lang, 'updates when the cut is done', '切完后更新') in pn['status'] and '%' not in pn['status'] and pn['metPend'] and pn['deltas'] == 0 and not pn['obs'], pn)
            pg.wait_for_function("()=>!window.__ckr.animState()", timeout=20000); dt = (pg.evaluate('()=>performance.now()') - t0) / 1000
            check(f'{tag} part 2: the cut takes about 2.4 s at 中 (R3.9)', 1.9 <= dt <= 3.6, dt)
            pg.wait_for_timeout(150); dl = pg.evaluate("()=>({pend:document.querySelector('#ckr-metrics').classList.contains('ckr-pend'), d:[...document.querySelectorAll('#ckr-metrics .d.on')].map(e=>e.textContent)})")
            # P4, kit polish K2: the cards take A.fmt.delta's base:false, so no ▲▼ line names what it is compared with (the card's tip does)
            check(f'{tag} part 2: when the cut ends, the cards show their ▲▼, without “vs last” (P4)', not dl['pend'] and len(dl['d']) >= 1 and not any(('last' in x or '上次' in x or '比' in x) for x in dl['d']), dl)
            pg.wait_for_timeout(3200)
            check(f'{tag} part 2: the ▲▼ stay until the next change (P4)', pg.evaluate("()=>document.querySelectorAll('#ckr-metrics .d.on').length") == len(dl['d']))
            m32 = meas(pg, lang, {**INIT, 'size': 32})
            ob = text(pg, '#ckr-rail [data-obs]')
            check(f'{tag} part 2: the observation names {m32["splits"]} answers cut apart; Next enabled', Z(lang, f'answers cut apart: {m32["splits"]} of 20', f'被切开的答案：20 个里有 {m32["splits"]} 个') in ob and not any(nexts(pg)), ob[:300])
            a = click_anim(pg, SEG('size', 64))
            check(f'{tag} part 2: going back to a watched size replays nothing (R3.20)', a is None, a)
            a = click_anim(pg, SEG('ov', 1))
            check(f'{tag} part 2: a new overlap plays its cut (R3.20)', a is not None, a)
            wait_anim(pg)
            a = click_anim(pg, '#ckr-transport [data-act="replay"]')
            check(f'{tag} part 2: 重放 replays the cut from the start (R3.20)', a is not None and a['p'] < 0.3, a)
            wait_anim(pg)
            pg.evaluate("()=>{const i=document.querySelector('#ckr-tr-p-in'); i.value=40; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(200)
            a = anim(pg)
            check(f'{tag} part 2: the progress scrubber holds the cut at the point picked', a is not None and abs(a['p'] - 0.4) < 0.02 and a['paused'], a)
            click(pg, '#ckr-transport [data-act="play"]'); wait_anim(pg)
            click(pg, '#ckr-rail .step.current [data-nav="showme"]')
            st = state(pg)
            check(f'{tag} part 2: Show me picks 64 tokens with half a chunk of overlap; no answer is cut apart', (st['size'], st['ov']) == (64, 2) and meas(pg, lang, {**INIT, 'ov': 2})['splits'] == 0, st)
            wait_anim(pg)
            if phone:   # R7.12, R3.2; P35 of the polish review
                pg.evaluate(TO_CARD); pg.wait_for_timeout(300); off = pg.evaluate("()=>Math.round(document.querySelector('#ckr-pan-doc').getBoundingClientRect().bottom)")
                h0 = pg.evaluate(CLICK_ANIM, SEG('size', 128)); settle(pg); a1 = anim(pg); top = pg.evaluate(PAN_TOP, '#ckr-pan-doc')
                check(f'{tag} part 2: on a phone, a cut started from the card brings the strip into view first and waits for the scroll (R7.12, P35)', off < 100 and h0 is not None and h0['held'] and h0['p'] == 0 and a1 is not None and a1['p'] < 0.4 and 40 <= top <= 422, (off, h0, a1, top))
                wait_anim(pg); pg.wait_for_timeout(200); bk = pg.evaluate(BACK_LINE); top2 = pg.evaluate(PAN_TOP, '#ckr-pan-doc')
                check(f'{tag} part 2: when it ends, the strip stays put and a line under it leads back to the conclusion (R3.2, P35)', bk is not None and bk['pan'] == 'ckr-pan-doc' and Z(lang, 'the conclusion is in part 2 ↓', '结论在第 2 部分 ↓') in bk['t'] and abs(top2 - top) <= 50, (bk, top, top2))
                click(pg, '#ckr-back'); ob = pg.evaluate("()=>({top:Math.round(document.querySelector('#ckr-rail [data-obs]').getBoundingClientRect().top), vh:innerHeight, back:!document.querySelector('#ckr-back').hidden})")
                check(f'{tag} part 2: the line brings the observation into view and goes away (P35)', 40 <= ob['top'] <= ob['vh'] - 100 and not ob['back'], ob)
            shot(pg, f'{tag}_p2', full=phone); rules(pg, f'{tag} part 2', lang)
            # ---- part 3
            a = click_anim(pg, NEXT); st = state(pg)
            check(f'{tag} part 3: 64 tokens with half a chunk was watched, so entering replays nothing (R3.20)', a is None and (st['size'], st['ov'], st['q']) == (64, 2, 0), (a, st))
            check(f'{tag} part 3: Next waits for a prediction, with its why line', all(nexts(pg)) and any(Z(lang, 'Pick an answer', '选一个答案') in x for x in why(pg)), why(pg))
            cur = text(pg, '#ckr-rail .step.current')
            check(f'{tag} part 3: no ranking shown before the prediction (R3.8)', not vis(pg, '#ckr-pan-lists') and not vis(pg, '#ckr-metrics') and not re.search(Z(lang, r'ranks (?:it )?\d', r'排第 \d'), cur), cur[:300])
            click(pg, '#ckr-rail [data-pred="vec"]')
            check(f'{tag} part 3: after a pick, Next is enabled', not any(nexts(pg)))
            shot(pg, f'{tag}_p3', full=phone); rules(pg, f'{tag} part 3', lang)
            # ---- part 4
            a = click_anim(pg, NEXT); rk = ranks(pg, lang, 64, 2, 0); cur = text(pg, '#ckr-rail .step.current')
            check(f'{tag} part 4: the verdict names the E-201 chunk first in keyword search and {rk["vec0"] + 1} in vector search', a is None and Z(lang, 'Not quite:', '不对：') in cur and Z(lang, f'vector search ranks it {rk["vec0"] + 1}', f'向量检索排第 {rk["vec0"] + 1}') in cur and Z(lang, 'you predicted “Vector search puts it first; keyword search does not”, but the answer is “Keyword search puts it first; vector search does not”', '你预测的是“向量检索把它排第一，关键词检索不会”，正确答案是“关键词检索把它排第一，向量检索不会”') in cur and rk['kw0'] == 0 and rk['vec0'] >= 5, (a, rk, cur[:300]))
            check(f'{tag} part 4: RRF is typeset with the kit (R6.29)', pg.evaluate("()=>document.querySelectorAll('#ckr-rail .step.current .ckr-math-line math').length") >= 1)
            nos = pg.evaluate(MATH_NOS)
            check(f'{tag} part 4: the worked formulas are numbered 1, 2, 3 in order', nos == list(range(1, len(nos) + 1)) and len(nos) == 3, nos)
            check(f'{tag} part 4: three lists side by side or stacked', pg.evaluate("()=>document.querySelectorAll('#ckr-cols .ckr-col').length") == 3)
            zs = pg.evaluate("()=>[...document.querySelectorAll('#ckr-cols .ckr-sc')].map(e=>parseFloat(e.textContent)).filter(v=>v===0).length")
            check(f'{tag} part 4: the lists show only chunks a search returns (no score 0) (A7)', zs == 0, zs)
            hy = pg.evaluate("()=>[...document.querySelectorAll('#ckr-cols .ckr-col[data-c=\"hyb0\"] .ckr-li')].map(li=>({j:+li.dataset.j, src:(li.querySelector('.ckr-src')||{}).textContent||'', tie:!!li.querySelector('.ckr-tie'), ok:li.classList.contains('ok')}))")
            q0l = pg.evaluate("l=>{const q=window.__ckr.setup(l,64,2).Q[0]; return {kw:q.lists.kw.slice(0,5), vec:q.lists.vec.slice(0,5), rrf:q.rrf, hyb:q.lists.hyb.slice(0,5)}}", lang)
            srcok = all(((f'{Z(lang, "keyword ", "关键词第 ")}{q0l["kw"].index(x["j"]) + 1}' in x['src']) if x['j'] in q0l['kw'] else True) and ((f'{Z(lang, "vector ", "向量第 ")}{q0l["vec"].index(x["j"]) + 1}' in x['src']) if x['j'] in q0l['vec'] else True) and x['src'] for x in hy)
            tie0 = abs(q0l['rrf'][q0l['hyb'][0]] - q0l['rrf'][q0l['hyb'][1]]) < 1e-12
            check(f'{tag} part 4: each hybrid row names where its shares come from, and the tie that puts the E-201 chunk first is marked (P9, P10)', len(hy) == 5 and srcok and hy[0]['ok'] and (hy[0]['tie'] and hy[1]['tie']) == tie0, (hy, tie0))
            j1 = q0l['kw'][1]; click(pg, f'#ckr-cols .ckr-col[data-c="kw0"] .ckr-li[data-j="{j1}"] .ckr-rk', 120)
            same = pg.evaluate("j=>({same:[...document.querySelectorAll('#ckr-cols .ckr-li.same')].map(li=>+li.dataset.j), all:[...document.querySelectorAll('#ckr-cols .ckr-li[data-j=\"'+j+'\"]')].length})", j1)
            check(f'{tag} part 4: a chunk clicked is marked in every list it is on (P10)', same['same'] and all(x == j1 for x in same['same']) and len(same['same']) == same['all'] >= 2, same)
            click(pg, f'#ckr-cols .ckr-col[data-c="kw0"] .ckr-li[data-j="{j1}"] .ckr-rk', 120)
            check(f'{tag} part 4: Next waits for another question', all(nexts(pg)) and any(Z(lang, 'Pick another question', '换一个问题') in x for x in why(pg)), why(pg))
            click(pg, SEG('q', 8)); r8 = ranks(pg, lang, 64, 2, 8)
            ob = text(pg, '#ckr-rail [data-obs]')
            check(f'{tag} part 4: the stairs question: keyword returns nothing, vector 1, hybrid {r8["hyb0"] + 1}; Next enabled', Z(lang, 'keyword search: not returned (no word in common)', '关键词检索：不返回（没有共同的词）') in ob and Z(lang, f'vector search: 1\n', '向量检索：第 1 名') in ob + '\n' and Z(lang, f'hybrid search: {r8["hyb0"] + 1}', f'混合检索：第 {r8["hyb0"] + 1} 名') in ob and r8['vec0'] == 0 and r8['kw0'] == -1 and not any(nexts(pg)), ob[:300])
            kwl = text(pg, '#ckr-cols .ckr-col')
            check(f'{tag} part 4: the keyword list says it does not return the answer', Z(lang, 'keyword search does not return it', '关键词检索不会返回它') in kwl, kwl[-200:])
            shot(pg, f'{tag}_p4', full=phone); rules(pg, f'{tag} part 4', lang)
            click(pg, BACK)
            a = click_anim(pg, NEXT)
            check(f'{tag} part 4 again: Back and Next replay nothing (R3.20)', a is None and state(pg)['step'] == 3, (a, state(pg)))
            click(pg, SEG('q', 8))
            # ---- part 5
            a = click_anim(pg, NEXT); st = state(pg)
            check(f'{tag} part 5: vector search without rerank, keep 1, the thermostat-battery question; Next waits for rerank', a is None and (st['ret'], st['rr'], st['k'], st['q']) == ('vec', 0, 1, 17) and all(nexts(pg)) and any(Z(lang, 'Turn rerank on', '打开 rerank') in x for x in why(pg)), (a, st, why(pg)))
            pre = pg.evaluate("()=>({cols:document.querySelectorAll('#ckr-cols .ckr-col').length, rows:document.querySelectorAll('#ckr-cols .ckr-col[data-c$=\"1\"] .ckr-li').length, wait:(document.querySelector('#ckr-cols .ckr-col-wait')||{}).innerText||'', cov:document.querySelectorAll('#ckr-cols .ckr-cov').length})")
            check(f'{tag} part 5: before rerank is on, its column holds no ranking and no coverage (R3.8, P14)', pre['cols'] == 2 and pre['rows'] == 0 and pre['cov'] == 0 and Z(lang, 'Shown once rerank is on', '打开 rerank 后显示') in pre['wait'], pre)
            if phone: pg.evaluate(TO_CARD); pg.wait_for_timeout(300)   # a phone reader turns it on from the card, below the stage (P35)
            t0 = pg.evaluate('()=>performance.now()'); r0 = pg.evaluate(CLICK_PEND, SEG('rr', 1))
            check(f'{tag} part 5: turning rerank on plays it: its rows start hidden, recall, the cards and the observation wait (R3.8, P14)', r0['rr'] is not None and r0['rr']['p'] < 0.3 and r0['rows'] == 5 and r0['hidden'] == 5 and r0['metPend'] and Z(lang, 'updates when rerank is done', 'rerank 排完后更新') in r0['status'] and not r0['obs'], r0)
            if phone:   # the playback waits for the lists to come into view; its time counts from there
                pg.wait_for_function("()=>{const s=window.__ckr.rrState(); return !s||!s.held}", timeout=5000); t0 = pg.evaluate('()=>performance.now()') - 150
                top5 = pg.evaluate(PAN_TOP, '#ckr-pan-lists'); a5 = pg.evaluate("()=>window.__ckr.rrState()")
                check(f'{tag} part 5: on a phone, rerank turned on from the card brings the lists into view first and waits for the scroll (R7.12, P35)', r0['rr'] is not None and r0['rr']['held'] and r0['rr']['p'] == 0 and a5 is not None and a5['p'] < 0.3 and 40 <= top5 <= 422, (r0['rr'], a5, top5))
            pg.wait_for_function("()=>{const s=window.__ckr.rrState(); return !s||s.p>=0.56}", timeout=8000); mid = pg.evaluate(RR_NOW)
            check(f'{tag} part 5: mid-way the candidates’ coverage has shown one by one, the result still waits', mid['rr'] is not None and mid['cov'] == 5 and mid['covShown'] >= 3 and mid['metPend'] and not mid['obs'], mid)
            wait_rr(pg); dt = (pg.evaluate('()=>performance.now()') - t0) / 1000 - 0.15
            check(f'{tag} part 5: rerank plays in about 1.6 s at 中 (R3.9)', 1.2 <= dt <= 2.8, dt)
            if phone:
                bk = pg.evaluate(BACK_LINE)
                check(f'{tag} part 5: when rerank ends, a line under the lists leads back to the conclusion (R3.2, P35)', bk is not None and bk['pan'] == 'ckr-pan-lists' and Z(lang, 'the conclusion is in part 5 ↓', '结论在第 5 部分 ↓') in bk['t'], bk)
            r17 = ranks(pg, lang, 64, 2, 17)
            ra = meas(pg, lang, {'size': 64, 'ov': 2, 'ret': 'vec', 'rr': 0, 'k': 1}); rb = meas(pg, lang, {'size': 64, 'ov': 2, 'ret': 'vec', 'rr': 1, 'k': 1})
            end = pg.evaluate(RR_NOW)
            check(f'{tag} part 5: when it ends, the rerank column ranks the right chunk first and recall shows {pct(rb["rec"])}', end['rows'] == 5 and end['visible'] == 5 and 'ok' in end['first'].split() and not end['metPend'] and pct(rb['rec']) in end['status'], end)
            ob = text(pg, '#ckr-rail [data-obs]')
            check(f'{tag} part 5: the observation names {r17["vec0"] + 1} → {r17["vec1"] + 1} and recall@1 {pct(ra["rec"])} → {pct(rb["rec"])}; Next enabled', Z(lang, f'moves from {r17["vec0"] + 1} to {r17["vec1"] + 1}', f'从第 {r17["vec0"] + 1} 名升到第 {r17["vec1"] + 1} 名') in ob and pct(ra['rec']) in ob and pct(rb['rec']) in ob and r17['vec1'] == 0 and rb['rec'] > ra['rec'] and not any(nexts(pg)), ob[:300])
            met = text(pg, '#ckr-metrics'); ltip = pg.evaluate("()=>document.querySelector('#ckr-metrics .metric[data-k=\"lat\"]').dataset.tipx")
            check(f'{tag} part 5: the time card names the rerank time, its tip the {rb["nc"]} candidates (B2, N13)', Z(lang, f'rerank takes {round(rb["rerankMs"])} ms', f'rerank 占 {round(rb["rerankMs"])} 毫秒') in met and Z(lang, f'Rerank reads {rb["nc"]} candidates here', f'这里 rerank 读 {rb["nc"]} 块候选') in ltip and rb['nc'] <= 5, (met, ltip))
            rq = pg.evaluate(CLICK_PEND, SEG('q', 0))
            check(f'{tag} part 5: a new question replays rerank; the last observation stays greyed (R3.20, P4)', rq['rr'] is not None and rq['obsPend'] and not rq['obs'], rq)
            wait_rr(pg); ob = text(pg, '#ckr-rail [data-obs]')
            check(f'{tag} part 5: the E-201 question is outside the candidates', Z(lang, 'outside the candidates', '不在候选里') in ob, ob[:300])
            rq = pg.evaluate(CLICK_PEND, SEG('q', 17))
            check(f'{tag} part 5: going back to a question already watched replays nothing (R3.20)', rq['rr'] is None and rq['hidden'] == 0, rq)
            rq = pg.evaluate(CLICK_PEND, '#ckr-ltools [data-act="rr-replay"]')
            check(f'{tag} part 5: 重放 rerank replays it from the start', rq['rr'] is not None and rq['rr']['p'] < 0.3, rq)
            wait_rr(pg)
            shot(pg, f'{tag}_p5', full=phone); rules(pg, f'{tag} part 5', lang)
            # ---- part 6
            go_next(pg); st = state(pg)
            fv = pg.evaluate("()=>{const q=s=>Math.round(document.querySelector(s).getBoundingClientRect().top); return {chart:q('#ckr-pan-chart'), lists:q('#ckr-pan-lists'), metrics:q('#ckr-metrics')}}")
            check(f'{tag} part 6: the chart first, then the lists and the metric cards (R7.12)', fv['chart'] < fv['lists'] < fv['metrics'], fv)
            check(f'{tag} part 6: Next waits for another k or size', all(nexts(pg)) and any(Z(lang, 'Try another k', '换一个 k') in x for x in why(pg)), why(pg))
            st = state(pg)
            check(f'{tag} part 6: hybrid search without rerank, keep 1', (st['ret'], st['rr'], st['k']) == ('hyb', 0, 1), st)
            click(pg, SEG('k', 3)); m3 = meas(pg, lang, {'size': 64, 'ov': 2, 'ret': 'hyb', 'rr': 0, 'k': 3})
            ob = text(pg, '#ckr-rail [data-obs]')
            check(f'{tag} part 6: the observation names recall@3 {pct(m3["rec"])} and {m3["ctx"]} tokens; Next enabled', pct(m3['rec']) in ob and str(m3['ctx']) in ob and not any(nexts(pg)), ob[:300])
            ctitle = text(pg, '#ckr-c-title')
            check(f'{tag} part 6: the chart title is a question and the initial setup is marked (A18)', ('?' in ctitle or '？' in ctitle) and pg.evaluate("()=>document.querySelectorAll('#ckr-chart .ckr-init').length") == 1, ctitle)
            shot(pg, f'{tag}_p6', full=phone); rules(pg, f'{tag} part 6', lang)
            # ---- part 7
            go_next(pg); cur = text(pg, '#ckr-rail .step.current')
            check(f'{tag} part 7: the bullets name the engine’s rank and the scenarios button', state(pg)['step'] == 6 and Z(lang, f'vector search ranks it {rk["vec0"] + 1}', f'向量检索排第 {rk["vec0"] + 1}') in cur and Z(lang, 'keeping 1 chunk, any search returns at most part of it', '只取 1 块时，哪种检索都只能拿回一部分') in cur and vis(pg, '#ckr-rail .step.current [data-mode-go="sandbox"]'), cur[:400])
            nos = pg.evaluate(MATH_NOS)
            check(f'{tag} part 7: the worked formulas are numbered 1, 2 in order', nos == [1, 2], nos)
            check(f'{tag} part 7: the next-lab footer', pg.evaluate("()=>[...document.querySelectorAll('#ckr-app .next-lab')].some(e=>e.getBoundingClientRect().height>0)"))
            shot(pg, f'{tag}_p7', full=phone); rules(pg, f'{tag} part 7', lang)
            # ---- scenarios
            mode(pg, 'sandbox')
            check(f'{tag} scenarios: six scenarios', pg.evaluate("()=>document.querySelectorAll('#ckr-rail [data-preset]').length") == 6)
            # the strip shows the page of the question picked (the stairs question: the warranty page), so the first pick of the
            # initial setup is a new picture and may play; picking it again must not
            click(pg, '#ckr-rail [data-preset="0"]'); wait_anim(pg)
            a = click_anim(pg, '#ckr-rail [data-preset="5"]'); wait_anim(pg)
            a = click_anim(pg, '#ckr-rail [data-preset="0"]')
            check(f'{tag} scenarios: going back to a scenario already watched replays nothing (R3.20)', a is None and all(state(pg)[k] == v for k, v in INIT.items()), (a, state(pg)))
            ins = text(pg, '#ckr-insight')
            check(f'{tag} scenarios: the initial setup’s insight names {m0["hits"]} of 20 and {m0["splits"]} cut apart', Z(lang, f'{m0["hits"]} of the 20 answers reach the context', f'20 个答案里有 {m0["hits"]} 个进到上下文') in ins and Z(lang, f'{m0["splits"]} answers are cut apart', f'有 {m0["splits"]} 个答案被切开') in ins, ins[:300])
            a = click_anim(pg, '#ckr-rail [data-preset="1"]')
            check(f'{tag} scenarios: a scenario with a chunking not yet watched plays its cut', a is not None and state(pg)['size'] == 16, a)
            wait_anim(pg)
            ins = text(pg, '#ckr-insight'); m16 = meas(pg, lang, {**INIT, 'size': 16})
            check(f'{tag} scenarios: the small chunks’ insight names {m16["splits"]} answers cut apart', Z(lang, f'{m16["splits"]} of the 20 answers are cut apart', f'20 个答案里有 {m16["splits"]} 个被切开') in ins, ins[:300])
            a = click_anim(pg, '#ckr-rail [data-preset="1"]')
            check(f'{tag} scenarios: picking it again replays nothing (R3.20)', a is None, a)
            hsh = pg.evaluate("()=>location.hash")
            check(f'{tag} scenarios: the hash carries the setup', hsh.startswith('#chunk-rerank~16_0_vec_0_1_'), hsh)
            qn = pg.evaluate("()=>{const QS=window.__ckr.data().QS, q=window.__ckr.state().q; return QS.findIndex(x=>x[2]!==QS[q][2]);}"); qc = pg.evaluate(QCHANGE, qn)
            check(f'{tag} scenarios: a question on another page plays its cut, and the numbers it leaves unchanged do not wait (N12)', qc['cut'] is not None and not qc['metPend'] and pct(m16['rec']) in qc['status'] and Z(lang, 'when the cut is done', '切完后') not in qc['ins'] and Z(lang, f'{m16["splits"]} of the 20 answers are cut apart', f'20 个答案里有 {m16["splits"]} 个被切开') in qc['ins'], qc)
            wait_anim(pg)
            for i in (2, 3, 4, 5):
                click(pg, f'#ckr-rail [data-preset="{i}"]'); wait_anim(pg)
                ins = text(pg, '#ckr-insight')
                check(f'{tag} scenarios: scenario {i + 1} has its insight', '%' in ins and len(ins) > 80, ins[:200])
            shot(pg, f'{tag}_scen', full=phone); rules(pg, f'{tag} scenarios', lang)
            other = 'zh' if lang == 'en' else 'en'
            s0 = state(pg); click(pg, f'.langsw button[data-lang="{other}"]', 300); s1 = state(pg)
            check(f'{tag} scenarios: the language switch keeps the setup', all(s0[k] == s1[k] for k in ('mode', 'size', 'ov', 'ret', 'rr', 'k', 'q')), (s0, s1))
            click(pg, f'.langsw button[data-lang="{lang}"]', 300)
            if phone:
                pg.evaluate("()=>{const a=document.querySelector('#ckr-stage'); window.scrollTo(0,a.getBoundingClientRect().bottom+scrollY+400)}"); pg.wait_for_timeout(800)
                dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return d&&!d.hidden&&d.getBoundingClientRect().height>0?d.innerText:''}")
                check(f'{tag}: the dock shows once the stage scrolls away', bool(dk) and '%' in dk, dk)
                shot(pg, f'{tag}_dock'); rules(pg, f'{tag} dock', lang); pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(200)
            pg.close()
            pg = page(b, w, h, lang, hash_='chunk-rerank~128_1_kw_1_5_9'); st = state(pg)
            check(f'{tag}: the hash round trip opens the same setup in scenarios', st['mode'] == 'sandbox' and (st['size'], st['ov'], st['ret'], st['rr'], st['k'], st['q']) == (128, 1, 'kw', 1, 5, 9), st)
            # ---- challenge
            mode(pg, 'challenge'); st = state(pg)
            nores = lambda: '%' not in text(pg, '#ckr-ch-av') and not text(pg, '#ckr-ch-res').strip() and not state(pg)['chSubmitted']
            check(f'{tag} challenge: starts at the initial setup with no result before Submit', st['mode'] == 'challenge' and all(st[k] == v for k, v in INIT.items()) and nores(), (st, text(pg, '#ckr-ch-av')))
            check(f'{tag} challenge: context tokens and time shown before Submit', text(pg, '#ckr-ch-cv') == '64' and '10' in text(pg, '#ckr-ch-lv'), (text(pg, '#ckr-ch-cv'), text(pg, '#ckr-ch-lv')))
            click(pg, '#ckr-rail [data-act="submit"]')
            res = text(pg, '#ckr-ch-res')
            check(f'{tag} challenge: the initial setup misses 95%, with a hint and “{Z(lang, "Show the answer", "直接看答案")}”', Z(lang, 'Not yet', '还没达到') in res and pct(m0['rec']) in res and '95%' in res and vis(pg, '#ckr-rail [data-act="reveal"]'), res[:300])
            click(pg, SEG('ov', 2))
            check(f'{tag} challenge: a new choice after a miss needs a new Submit', not state(pg)['chSubmitted'] and not text(pg, '#ckr-ch-res').strip() and vis(pg, '#ckr-rail [data-act="submit"]'), state(pg))
            for k, v in (('ret', 'hyb'), ('rr', 1), ('k', 1)): click(pg, SEG(k, v))
            click(pg, '#ckr-rail [data-act="submit"]')
            res = text(pg, '#ckr-ch-res')
            check(f'{tag} challenge: 64 tokens, half overlap, hybrid + rerank, keep 1 meets the goal with the fewest tokens', Z(lang, 'Goal met', '达成目标') in res and Z(lang, 'No setting meets the goal with fewer context tokens', '没有哪种设置能用更少的上下文 token 达标') in res, res[:300])
            tags = pg.evaluate("()=>[...document.querySelectorAll('#ckr-ch-res .cmp-tag, #ckr-ch-res .cmp-you')].map(e=>e.textContent.trim())")
            check(f'{tag} challenge: the chart tags the initial setup and the pick', any(Z(lang, 'initial setup', '初始设置') in t for t in tags) and any(Z(lang, 'your pick', '你的选择') in t for t in tags), tags)
            check(f'{tag} challenge: the why names the pools and the best setting with one choice changed', Z(lang, 'Only hybrid search has every answer in its first 5', '只有混合检索把每一个答案都放进了前 5') in res and Z(lang, 'vector search + rerank', '向量检索 + rerank') in pg.evaluate("()=>document.querySelector('#ckr-ch-res details').textContent"), res[-600:])
            check(f'{tag} challenge: the capability card', vis(pg, '#ckr-app [data-exits]') or pg.evaluate("()=>!!document.querySelector('#ckr-rail .exits, #ckr-rail [data-exits]')"))
            shot(pg, f'{tag}_chal', full=True); rules(pg, f'{tag} challenge', lang)
            pg.close()
    b.close()

for h in rule_hits: print('RULE', h)
check('writing, colour, overflow and typesetting rules on every visited state', not rule_hits, len(rule_hits))
check('no page errors', not errs, errs[:5])
print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
