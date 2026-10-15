"""Flow test for the embeddings lab (CONTRACT §6), at 1440×900 and 390×844, in EN and ZH.

- no page errors, no horizontal overflow, no text wider than its box, a tooltip for every help cursor;
- part 1: the keyword-search picture names how many passages come before the right one; no transport; Next enabled;
- part 2: Next waits for a second held-out question, with the kit's why line; the missed right passage is ringed red (R5.11);
  Show me picks question 10; the observation names the untrained recall@1, the recall@5 card and its tooltip the untrained recall@5;
- part 3: Next waits for a prediction; no trained result is shown before it (R3.8);
- part 4: the verdict on the prediction, the loss typeset (R6.29); Next waits for 50 steps; Train at 中 plays the first 10 steps
  slowly, each step about 1.6 s with no step under 0.5 s (a mechanism, R3.9, R7.11): its own batch with the push arrows, then the
  same batch after the update; then the rest of the run fast with the table and the rows held at step 10 (R7.11); the observation waits for the
  end and the stage title keeps its height (R7.9); the main pictures come before the metric cards (R7.12); the observation names the
  recall@1 the engine gives; the mission question's rows end green once its passage ranks first; Back and Next replay nothing (R3.20);
- part 5: entering replays nothing; a new batch size plays its run; picking the one already shown, or one already watched, replays
  nothing; 重放 replays from step 0 (R3.20); the observation names the 16-start averages of B = 8 and 16 and the 800-pair range;
- part 6: the chart switches to the loss on a log scale; τ = 0.01 plays and the observation names the step its loss falls under 0.01;
- part 7: Next waits for a pair of sentences; the word-order pair gives 1.00;
- part 8: the untrained count and the gain from 32 dimensions; the scenarios button, the next-lab footer, no “part N” recall;
- Scenarios: six, each with its insight; one already watched does not replay, a new one plays at 快 in about 1.5 s (R3.9); the
  hash round trip; the language switch keeps the setup;
- the challenge (an average recall@1 of 55% over 16 starts, in the fewest steps): starts at B = 2 × 25 steps with no result before
  Submit, also after another B is chosen; Submit trains, then the verdict; a new choice after a miss needs a new Submit; a goal met
  with more steps names B = 16 × 10; B = 16 × 10 is met with the fewest steps; the chart of the choices with the challenge's default and
  the pick tagged; the capability card;
- every formula line in a part's card fits its box at 1440, 900 and 390 (no clipped formula);
- the dock at 390 once the stage scrolls away; every chart label inside its picture;
- the writing rules on every visible text and tooltip (no “toy” outside the Toy run label, Chinese spacing and punctuation, two
  colons in one sentence, no 万 or 亿), the colours (red only for a miss and a failed verdict; the reserved blue, amber and violet
  not used), every formula typeset; and, read from the source, no “one at a time” note and no sentence that recalls a part by number.
Screenshots: shots/embeddings/.
Run: python3 tests/embeddings_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
(P / 'out' / 'embeddings').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/embeddings.html', '--only', 'embeddings'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'embeddings' / 'flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'embeddings.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'embeddings' / 'flow-wrapped.html')
SH = P / 'shots' / 'embeddings'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, hash_='embeddings'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__emb.state()")
anim = lambda pg: pg.evaluate("()=>window.__emb.animState()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-').replace(/\\u2212/g,'-'):''}", sel)
svgtext = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.textContent.replace(/\\u2212/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#emb-rail [data-nav=\"next\"]')].filter(b=>b.getClientRects().length).map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#emb-rail [data-next-why]')].filter(p=>!p.hidden&&getComputedStyle(p).visibility!=='hidden'&&p.getClientRects().length).map(p=>p.textContent)")
speed = lambda pg: pg.evaluate("()=>{const b=document.querySelector('#emb-transport .seg-ctl[data-key=\"speed\"] button[aria-pressed=\"true\"]'); return b?b.dataset.v:null}")
RUN = lambda pg, lang, B, tau, d: pg.evaluate(f"()=>{{const r=window.__emb.run('{lang}',{B},{tau},{d}); return {{r1:r.r1,r5:r.r5,loss:r.loss,rank:r.rank}}}}")


def settle(pg, ms=2000):
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel, wait=250):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(wait); settle(pg)


# the animation state two frames after a click: None means nothing replays
CLICK_ANIM = """async (sel)=>{document.querySelector(sel).click(); await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))); return window.__emb.animState();}"""


def click_anim(pg, sel):
    a = pg.evaluate(CLICK_ANIM, sel); pg.wait_for_timeout(150); settle(pg); return a


def wait_anim(pg, timeout=25000):
    pg.wait_for_function("()=>!window.__emb.animState()", timeout=timeout); pg.wait_for_timeout(150)


def mode(pg, m):
    click(pg, f'#emb-app .modes button[data-mode="{m}"]', 350)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · (?![0-9])/,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],
   [/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy”'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/\\bNaN\\b(?![?？])|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],[/OpenAI|Anthropic|DeepMind|Google|Meta\\b|Microsoft|Amazon|Apple\\b/,'company name'],
   [/吃|喂|扛|吞/,'figurative verb'],[/一次只能选|不会叠加|第 [0-9一二三四五六七八] 部分里|\\bin part [0-9]\\b|part [0-9]’s/,'recall of a part or a one-at-a-time note'],
   [/深色|浅色|浅灰|深灰|\\b(?:dark|light|pale)(?:er)? (?:bar|line|block|part)/i,'colour named by lightness (R5.7)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, .fid-tag')||!e.getClientRects().length) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); for(const [re,name] of bad) if(re.test(t)) out.push(name+' (tooltip): '+t.slice(0,90));});
 return out;}"""
SPACING = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), bad=/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>%][\\u4e00-\\u9fff]|[）】][ ][\\u4e00-\\u9fff]/;
 const shown=e=>e&&!e.closest('[data-endnotes],pre,code,svg,.mathbody,#tip')&&e.getClientRects().length;
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
 root.querySelectorAll('p,li,button,label,div,span,h1,h2,h3,summary').forEach(e=>{if(e.closest('[data-endnotes],pre,code,svg,.mathbody')||!e.getClientRects().length) return; if(e.querySelector('p,li,div,ul,ol,h1,h2,h3,table')) return;
   if(![...e.childNodes].some(c=>c.nodeType===3&&c.textContent.trim())) return; const c=e.cloneNode(true); c.querySelectorAll('code').forEach(x=>x.remove()); texts.push(c.textContent);});
 root.querySelectorAll('[data-tipx]').forEach(e=>texts.push(e.getAttribute('data-tipx')));
 // literal strings in <code> (a prefix such as query: ) are not prose colons
 for(const t of texts){const sents=t.replace(/\\s+/g,' ').split(/(?<=[。？！])|(?<=[.?!])\\s+/);
   for(const x of sents){const k=(x.match(/：|: /g)||[]).length; if(k>=2) out.add(x.trim().slice(0,160));}}
 return [...out];}"""
# red only for a miss (the right passage ringed red, the miss line, the failed verdict and status); blue, amber and violet are reserved
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/emb-own bad|emb-miss bad|emb-bad|emb-sw-bad|badline|status bad|\\bbad\\b|ex-res/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#emb-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())&&!(p==='stroke'&&s.strokeWidth==='0px')) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#emb-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg,.emb-math-line')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#emb-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
SVGIN = """()=>{const out=[]; for(const s of document.querySelectorAll('#emb-stage svg')){if(!s.getClientRects().length) continue; const b=s.getBoundingClientRect();
  const ts=[...s.querySelectorAll('text')].filter(t=>t.getClientRects().length&&t.textContent.trim());
  for(const t of ts){const r=t.getBoundingClientRect(); if(r.left<b.left-1||r.right>b.right+1||r.top<b.top-1||r.bottom>b.bottom+1) out.push((s.id||'svg')+' outside: '+t.textContent.trim().slice(0,30));}}
  return out.slice(0,8);}"""
# H1: no TeX command shown as a word; no plain-text formula left in the rendered text (outside math, code and pre)
TEXWORDS = """()=>{const W=new Set(['frac','left','right','sqrt','text','sum','tau','cdot','quad','lVert','rVert','ell','mathrm','times']);
  const out=[]; for(const m of document.querySelectorAll('#emb-app math, #dock math')) for(const e of m.querySelectorAll('mi,mo')){const t=e.textContent.trim(); if(W.has(t)) out.push(t);}
  const pat=[/s_ij|s_\\{|\\\\tau|\\\\frac|e\\^\\(|−log\\(|-log\\(|softmax\\(s/,/cos\\(a, ?b\\) =/,/\\|\\|v\\|\\|/];
  const tw=document.createTreeWalker(document.querySelector('#emb-app'),NodeFilter.SHOW_TEXT); let n;
  while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('math,pre,code')||!e.getClientRects().length) continue; for(const r of pat) if(r.test(n.textContent)) out.push('text: '+n.textContent.trim().slice(0,60));}
  for(const e of document.querySelectorAll('#emb-app [data-tipx]')){const t=e.getAttribute('data-tipx').replace(/<math[\\s\\S]*?<\\/math>/g,''); for(const r of pat) if(r.test(t)) out.push('tip: '+t.slice(0,60));}
  return out;}"""
ALIGN = """()=>{const els=[...document.querySelectorAll('#emb-transport .btn, #emb-transport .seg-ctl')].filter(e=>e.getClientRects().length);
  const rows=[]; for(const e of els){const r=e.getBoundingClientRect(); let row=rows.find(x=>Math.abs(x.top-r.top)<8); if(!row){row={top:r.top,items:[]}; rows.push(row);}
    const b=e.matches('.seg-ctl')?e.querySelector('button'):e, rg=document.createRange(); rg.selectNodeContents(b); const t=rg.getBoundingClientRect();
    row.items.push({t:(e.textContent||'').trim().slice(0,12),top:r.top,h:r.height,base:t.bottom});}
  const bad=[]; for(const row of rows){const it=row.items; for(const x of it){if(Math.abs(x.h-it[0].h)>0.6||Math.abs(x.top-it[0].top)>0.6||Math.abs(x.base-it[0].base)>1) bad.push(x.t+' vs '+it[0].t+' h '+x.h+'/'+it[0].h+' base '+x.base.toFixed(1)+'/'+it[0].base.toFixed(1));}}
  return {n:els.length,bad};}"""
rule_hits = []


def rules(pg, where, lang):
    for sel in ('#emb-app', '#dock'):
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


MATHFIT = """()=>[...document.querySelectorAll('#emb-rail .step.current .emb-math-line')].filter(e=>e.getClientRects().length&&e.scrollWidth>e.clientWidth+1).map(e=>e.textContent.slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')')"""


def mathfit(pg, where):
    bad = pg.evaluate(MATHFIT)
    check(f'{where}: every formula line of the card fits its box (R6.37)', not bad, bad)


NEXT = '#emb-rail .step.current [data-nav="next"]'
BACK = '#emb-rail .step.current [data-nav="back"]'


def go_next(pg):
    sel = NEXT if pg.query_selector(NEXT) else '#emb-rail [data-nav="next"]'
    click(pg, sel, 300)


# part 4's run, sampled every 100 ms from the click to its end: the step, the slow phase, the table shown, the observation, the title
PROBE4 = """async (sel)=>{const W=ms=>new Promise(r=>setTimeout(r,ms)), E=window.__emb, obs=()=>{const o=document.querySelector('#emb-rail [data-obs]'); return !!o&&!o.hidden;};
  const snap=t0=>{const a=E.animState(), tb=E.table(); return {t:(performance.now()-t0)/1000,steps:E.state().steps,slow:!!(a&&a.slow),k:a?a.k:null,ph:a?a.ph:null,frozen:a?a.frozen:null,
    kind:tb&&tb.kind,tk:tb?tb.k:null,push:!!(tb&&tb.push),title:document.querySelector('#emb-h-title').textContent,headH:Math.round(document.querySelector('#emb-st-title').getBoundingClientRect().height)};};
  const t0=performance.now(); document.querySelector(sel).click(); await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
  const out={samples:[],obsDuring:false};
  while(E.animState()){out.samples.push(snap(t0)); if(obs()) out.obsDuring=true; await W(100); if(performance.now()-t0>45000) break;}
  out.dt=(performance.now()-t0)/1000; out.end=snap(t0); out.obsEnd=obs(); return out;}"""


def timed_run(pg, sel, end_js, timeout=20000):
    t0 = pg.evaluate('()=>performance.now()'); a = pg.evaluate(CLICK_ANIM, sel)
    pg.wait_for_function(end_js, timeout=timeout)
    return a, (pg.evaluate('()=>performance.now()') - t0) / 1000


with sync_playwright() as pw:
    b = pw.chromium.launch()
    for lang in LANGS:
        for (w, h) in SIZES:
            phone = w <= 760; tag = f'{lang}-{w}'
            pg = page(b, w, h, lang)
            D = RUN(pg, lang, 8, 0.05, 16)
            st = state(pg)
            check(f'{tag}: opens on part 1 of the guided tour', st['mode'] == 'guided' and st['step'] == 0, st)
            # ---- part 1
            kw = pg.evaluate(f"()=>window.__emb.kwRank('{lang}',0)"); ahead = sum(1 for j, v in enumerate(kw['sc']) if j and v > kw['sc'][0])
            check(f'{tag} part 1: the keyword picture is shown, no transport, Next enabled', vis(pg, '#emb-pan-pair') and not vis(pg, '#emb-transport') and not any(nexts(pg)), nexts(pg))
            pr = text(pg, '#emb-pair')
            check(f'{tag} part 1: the picture names the {ahead} passages a keyword search puts first', Z(lang, f'returns {ahead} other passages before it', f'先给出另外 {ahead} 段') in pr, pr[:300])
            shot(pg, f'{tag}_p1', full=phone); rules(pg, f'{tag} part 1', lang)
            # ---- part 2
            go_next(pg); st = state(pg)
            check(f'{tag} part 2: Next disabled with the why line until a second question is picked', st['step'] == 1 and all(nexts(pg)) and any(Z(lang, 'Pick another question', '换一个问题') in x for x in why(pg)), (st, why(pg)))
            check(f'{tag} part 2: the map and the metric cards are shown, untrained', vis(pg, '#emb-map') and vis(pg, '#emb-metrics') and st['steps'] == 0)
            rk0 = D['rank'][0][st['q']]
            ring = pg.evaluate("()=>{const r=document.querySelector('#emb-map .emb-own'); return r?r.getAttribute('class'):''}")
            check(f'{tag} part 2: the right passage ranks {rk0 + 1} and is ringed red (R5.11)', rk0 > 0 and 'bad' in ring, ring)
            click(pg, '#emb-rail .seg-ctl[data-key="q"] button[data-v="17"]')
            ob = text(pg, '#emb-rail [data-obs]')
            r5t = pg.evaluate("()=>{const e=document.querySelector('#emb-metrics [data-k=\"r5\"]'); return e?e.getAttribute('data-tipx')+' | '+e.innerText:''}")   # recall@5: its card and the card's tooltip (P6)
            check(f'{tag} part 2: after a second question Next is enabled; the observation names {pct(D["r1"][0])}, the recall@5 card and its tooltip {pct(D["r5"][0])}', not any(nexts(pg)) and pct(D['r1'][0]) in ob and r5t.count(pct(D['r5'][0])) >= 2 and state(pg)['q'] == 17, (ob[:300], r5t))
            click(pg, '#emb-rail .step.current [data-nav="showme"]')
            check(f'{tag} part 2: Show me picks question 10', state(pg)['q'] == 9, state(pg))
            mathfit(pg, f'{tag} part 2')
            shot(pg, f'{tag}_p2', full=phone); rules(pg, f'{tag} part 2', lang)
            # ---- part 3
            go_next(pg)
            check(f'{tag} part 3: Next waits for a prediction, with its why line', all(nexts(pg)) and any(Z(lang, 'Pick an answer', '选一个答案') in x for x in why(pg)), why(pg))
            cur = text(pg, '#emb-rail .step.current')
            check(f'{tag} part 3: no trained result before the prediction (R3.8)', state(pg)['steps'] == 0 and pct(D['r1'][200]) not in cur and Z(lang, 'Right.', '对了。') not in cur, cur[:200])
            check(f'{tag} part 3: the batch table is shown', vis(pg, '#emb-heat'))
            click(pg, '#emb-rail [data-pred="pos"]')
            check(f'{tag} part 3: after a pick, Next is enabled and the card says it is saved', not any(nexts(pg)) and Z(lang, 'Saved', '已记下') in text(pg, '#emb-rail .step.current'))
            shot(pg, f'{tag}_p3', full=phone); rules(pg, f'{tag} part 3', lang)
            # ---- part 4
            a = click_anim(pg, NEXT); st = state(pg)
            cur = text(pg, '#emb-rail .step.current')
            check(f'{tag} part 4: starts at step 0, nothing plays on entry, Next waits for 50 steps', a is None and st['steps'] == 0 and all(nexts(pg)) and any(Z(lang, '50 steps', '50 步') in x for x in why(pg)), (a, st, why(pg)))
            check(f'{tag} part 4: the verdict on the prediction', Z(lang, 'Not quite.', '不对。') in cur and Z(lang, 'You predicted', '你预测的是') in cur, cur[:300])
            check(f'{tag} part 4: the loss is typeset with the kit (R6.29)', vis(pg, '#emb-rail .step.current .emb-math-line math') and Z(lang, 'loss', 'loss') in pg.evaluate("()=>[...document.querySelectorAll('#emb-rail .step.current .emb-math-line mtext')].map(e=>e.textContent).join(' ')"))
            fv = pg.evaluate("()=>{const q=s=>Math.round(document.querySelector(s).getBoundingClientRect().top); return {heat:q('#emb-pan-heat'), map:q('#emb-pan-map'), chart:q('#emb-pan-chart'), metrics:q('#emb-metrics')}}")
            check(f'{tag} part 4: the pictures first, the metric cards below them (R7.12)', max(fv['heat'], fv['map']) < fv['chart'] < fv['metrics'], fv)
            check(f'{tag} part 4: the run speed defaults to 中 (R3.9)', speed(pg) == 'mid', speed(pg))
            al = pg.evaluate(ALIGN)
            check(f'{tag} part 4: Train, the speed buttons and 重放 share one height and one baseline', al['n'] == 3 and not al['bad'], al)
            SL = pg.evaluate("()=>window.__emb.slow"); NS = SL['n']
            exp4 = NS * SL['sec']['mid'] + (200 - NS) / (200 / 3)
            pr = pg.evaluate(PROBE4, '#emb-transport [data-act="play"]'); sm = pr['samples']; dt = pr['dt']
            slow = [x for x in sm if x['slow']]; fast = [x for x in sm if not x['slow']]
            check(f'{tag} part 4: Train plays the run', len(sm) > 10 and pr['end']['steps'] == 200, (len(sm), pr['end']))
            check(f'{tag} part 4: the first {NS} steps play slowly, then the rest at the result speed: about {exp4:.1f} s in all (R3.9)', 0.85 * exp4 <= dt <= 1.25 * exp4, dt)
            ch = [x['t'] for a_, x in zip(slow, slow[1:]) if x['steps'] != a_['steps']]
            gaps = [b_ - a_ for a_, b_ in zip(ch, ch[1:])]
            check(f'{tag} part 4: in the slow steps the step count turns about every {SL["sec"]["mid"]} s, never under 0.5 s (R7.11)', len(ch) >= NS - 2 and gaps and min(gaps) >= 0.5 and max(x['steps'] for x in slow) <= NS, (ch[:12], gaps[:10]))
            check(f'{tag} part 4: a slow step shows the batch it trains, first with the push arrows, then after the update without them', slow and all(x['kind'] == 'slow' and x['tk'] == x['k'] for x in slow) and any(x['push'] for x in slow) and any(not x['push'] for x in slow) and all(Z(lang, f"Step {x['k']}, training this batch", f"第 {x['k']} 步，训练这个 batch") in x['title'] for x in slow), [(x['k'], x['kind'], x['tk'], x['push'], x['title'][:30]) for x in slow[:6]])
            nr = pg.evaluate("()=>document.querySelectorAll('#emb-rows li').length")   # the mission question's rows, held with the table (N4)
            check(f'{tag} part 4: the fast steps hold the table and the {nr} rows at step {NS} and say so (R7.11)', fast and nr > 0 and all(x['frozen'] == NS and x['kind'] == 'frozen' and x['tk'] == NS for x in fast) and all(Z(lang, f'Steps {NS + 1} to 200 play fast: the table and the {nr} rows stay at step {NS}', f'快放第 {NS + 1} 到 200 步：表格和 {nr} 行停在第 {NS} 步') in x['title'] for x in fast), [(x['steps'], x['frozen'], x['kind'], x['title'][:60]) for x in fast[:4]])
            check(f'{tag} part 4: the observation waits for the end of the run, then shows (R7.9)', not pr['obsDuring'] and pr['obsEnd'], (pr['obsDuring'], pr['obsEnd']))
            hh = {x['headH'] for x in sm} | {pr['end']['headH']}
            check(f'{tag} part 4: the stage title keeps its height while the run plays (R7.9)', max(hh) - min(hh) <= 1, sorted(hh))
            check(f'{tag} part 4: at rest the table is the batch step 201 would train, with its own probability and loss on the cards', pr['end']['kind'] == 'rest' and pr['end']['tk'] is None, pr['end'])
            pg.wait_for_timeout(200)
            ob = text(pg, '#emb-rail [data-obs]')
            check(f'{tag} part 4: the observation names {pct(D["r1"][0])} to {pct(D["r1"][200])}; Next enabled', pct(D['r1'][0]) in ob and pct(D['r1'][200]) in ob and not any(nexts(pg)), ob[:300])
            res = pg.evaluate("()=>{const r=document.querySelector('#emb-r-res'); return r&&r.getClientRects().length?r.getAttribute('class'):''}")
            q = state(pg)['q']
            check(f'{tag} part 4: the mission question’s rows end green once its passage ranks first, red otherwise', ('ok' in res) == (D['rank'][200][q] == 0) and ('bad' in res) == (D['rank'][200][q] > 0), (res, D['rank'][200][q]))
            mathfit(pg, f'{tag} part 4')
            shot(pg, f'{tag}_p4', full=phone); rules(pg, f'{tag} part 4', lang)
            click(pg, BACK)
            a = click_anim(pg, NEXT)
            check(f'{tag} part 4 again: Back and Next replay nothing and show the watched run’s end (R3.20)', a is None and state(pg)['step'] == 3 and state(pg)['steps'] == 200, (a, state(pg)))
            # ---- part 5
            a = click_anim(pg, NEXT); st = state(pg)
            check(f'{tag} part 5: nothing plays on entry; Next waits for another batch size', a is None and st['steps'] == 200 and all(nexts(pg)) and any(Z(lang, 'another batch size', '换一个 batch 大小') in x for x in why(pg)), (a, st, why(pg)))
            if not phone:
                dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return !!d&&!d.hidden&&d.getBoundingClientRect().height>0}")
                check(f'{tag} part 5: no mini bar over the sticky stage on entry (P13)', not dk, dk)
            check(f'{tag} part 5: the card links back up to the chart on a phone only (P20)', vis(pg, '#emb-rail .step.current .emb-see') == phone)
            a = click_anim(pg, '#emb-rail .seg-ctl[data-key="B"] button[data-v="8"]')
            check(f'{tag} part 5: picking the batch size already shown replays nothing (R3.20)', a is None and state(pg)['steps'] == 200, (a, state(pg)))
            a = click_anim(pg, '#emb-rail .seg-ctl[data-key="B"] button[data-v="16"]')
            check(f'{tag} part 5: a new batch size plays its run from step 0 (R3.20)', a is not None and state(pg)['B'] == 16, (a, state(pg)))
            wait_anim(pg)
            pg.wait_for_function("()=>{const o=document.querySelector('#emb-rail [data-obs]'); return o&&!/Averaging 16|正在算 16/.test(o.innerText)}", timeout=60000)
            av = {B: pg.evaluate(f"()=>window.__emb.avg('{lang}',{B},0.05,16,{200 if B == 4 else 100})") for B in (4, 8, 16)}
            ob = text(pg, '#emb-rail [data-obs]'); sep = Z(lang, ': ', '：')
            check(f'{tag} part 5: the observation names the 16-start averages after 25 steps and at 800 pairs', f'B = 8{sep}{pct(av[8][25])}' in ob and f'B = 16{sep}{pct(av[16][25])}' in ob and Z(lang, f'between {pct(min(av[4][200], av[8][100], av[16][50]))} and {pct(max(av[4][200], av[8][100], av[16][50]))}', f'都在 {pct(min(av[4][200], av[8][100], av[16][50]))} 到 {pct(max(av[4][200], av[8][100], av[16][50]))} 之间') in ob and not any(nexts(pg)), ob[:400])
            a = click_anim(pg, '#emb-rail .seg-ctl[data-key="B"] button[data-v="8"]')
            check(f'{tag} part 5: going back to a watched batch size replays nothing (R3.20)', a is None and state(pg)['steps'] == 200, (a, state(pg)))
            a = click_anim(pg, '#emb-rail .seg-ctl[data-key="B"] button[data-v="16"]')
            check(f'{tag} part 5: and back to B = 16, watched too, replays nothing (R3.20)', a is None and state(pg)['steps'] == 200, (a, state(pg)))
            a = click_anim(pg, '#emb-transport [data-act="replay"]')
            check(f'{tag} part 5: 重放 replays the run from step 0 (R3.20)', a is not None and state(pg)['steps'] < 60, (a, state(pg)))
            wait_anim(pg)
            check(f'{tag} part 5: the replay ends at step 200', state(pg)['steps'] == 200, state(pg))
            shot(pg, f'{tag}_p5', full=phone); rules(pg, f'{tag} part 5', lang)
            # ---- part 6
            go_next(pg); st = state(pg)
            check(f'{tag} part 6: the chart shows the loss on a log scale', st['view'] == 'loss' and Z(lang, 'log scale', '对数刻度') in svgtext(pg, '#emb-chart'), (st, svgtext(pg, '#emb-chart')[:200]))
            a = click_anim(pg, '#emb-rail .seg-ctl[data-key="tau"] button[data-v="0.01"]')
            check(f'{tag} part 6: τ = 0.01 plays its run', a is not None, a)
            wait_anim(pg)
            T1 = RUN(pg, lang, 8, 0.01, 16); z = next(s for s in range(201) if T1['loss'][s] < 0.01)
            ob = text(pg, '#emb-rail [data-obs]')
            check(f'{tag} part 6: the observation names step {z} and {pct(T1["r1"][200])}; Next enabled', Z(lang, f'step {z}', f'第 {z} 步') in ob and pct(T1['r1'][200]) in ob and not any(nexts(pg)), ob[:300])
            shot(pg, f'{tag}_p6', full=phone); rules(pg, f'{tag} part 6', lang)
            # ---- part 7
            go_next(pg)
            check(f'{tag} part 7: Next waits for a pair of sentences', all(nexts(pg)) and any(Z(lang, 'Pick a pair', '选一对句子') in x for x in why(pg)), why(pg))
            click(pg, '#emb-rail [data-lever="order"]')
            ob = text(pg, '#emb-rail [data-obs]')
            check(f'{tag} part 7: the word-order pair gives similarity 1.00; Next enabled', '1.00' in ob and not any(nexts(pg)), ob[:200])
            click(pg, '#emb-rail [data-lever="neg"]')
            ob = text(pg, '#emb-rail [data-obs]')
            check(f'{tag} part 7: the negation pair stays close', Z(lang, 'mean the opposite', '意思相反') in ob, ob[:200])
            shot(pg, f'{tag}_p7', full=phone); rules(pg, f'{tag} part 7', lang)
            # ---- part 8
            go_next(pg); cur = text(pg, '#emb-rail .step.current')
            k0 = round(D['r1'][0] * 24); k16 = round(D['r1'][200] * 24)
            check(f'{tag} part 8: the misconceptions count {k0} untrained and the gain from 32 dimensions; the scenarios button', state(pg)['step'] == 7 and Z(lang, ('not one' if k0 == 0 else f'only {k0}') + ' of the 24', '24 个新问题里' + ('没有一个' if k0 == 0 else f'只有 {k0} 个')) in cur and re.search(Z(lang, r'finds \d+ more of the 24', r'24 个新问题里多找对 \d+ 个'), cur) and vis(pg, '#emb-rail .step.current [data-mode-go="sandbox"]'), cur[:300])
            check(f'{tag} part 8: the next-lab footer (in the rail on a wide screen, at the end of the page on a phone)', pg.evaluate("()=>[...document.querySelectorAll('#emb-app .next-lab')].some(e=>e.getBoundingClientRect().height>0)"))
            shot(pg, f'{tag}_p8', full=phone); rules(pg, f'{tag} part 8', lang)
            # ---- scenarios
            mode(pg, 'sandbox')
            lead = pg.inner_text('#emb-rail .railcard > p.small')
            check(f'{tag} scenarios: the card leads in with what to pick and what to watch', lead == Z(lang, 'Pick a scenario and see how recall@1 on the 24 held-out questions changes as the small encoder trains.', '选一个场景，看看小编码器训练时，24 个新问题的 recall@1 怎么变。'), lead)
            check(f'{tag} scenarios: six scenarios', pg.evaluate("()=>document.querySelectorAll('#emb-rail [data-preset]').length") == 6)
            a = click_anim(pg, '#emb-rail [data-preset="0"]')
            check(f'{tag} scenarios: the initial setup was watched in the tour, so it does not replay (R3.20)', a is None and state(pg)['steps'] == 200, (a, state(pg)))
            ins = text(pg, '#emb-insight')
            check(f'{tag} scenarios: the insight for the initial setup', pct(D['r1'][0]) in ins and pct(D['r1'][200]) in ins, ins[:200])
            a = click_anim(pg, '#emb-rail [data-preset="2"]')
            check(f'{tag} scenarios: B = 16 was watched in the tour, so it does not replay (R3.20)', a is None and state(pg)['steps'] == 200, (a, state(pg)))
            a, dt = timed_run(pg, '#emb-rail [data-preset="1"]', "()=>window.__emb.state().steps>=200&&!window.__emb.animState()")
            check(f'{tag} scenarios: a scenario not yet watched plays its run', a is not None, a)
            check(f'{tag} scenarios: and shows its end within about 1.5 s (R3.9)', dt <= 2.2, dt)
            pg.wait_for_timeout(150)
            ins = text(pg, '#emb-insight'); D2 = RUN(pg, lang, 2, 0.05, 16)
            check(f'{tag} scenarios: the tiny batch’s insight names its recall@1', pct(D2['r1'][200]) in ins, ins[:300])
            hsh = pg.evaluate("()=>location.hash")
            check(f'{tag} scenarios: the hash carries the setup', hsh.startswith('#embeddings~2_0.05_16_200_'), hsh)
            for i in (3, 4, 5):
                click(pg, f'#emb-rail [data-preset="{i}"]'); pg.wait_for_function("()=>window.__emb.state().steps>=200&&!window.__emb.animState()", timeout=20000)
                ins = text(pg, '#emb-insight')
                check(f'{tag} scenarios: scenario {i + 1} has its insight', '%' in ins and len(ins) > 80, ins[:200])
            shot(pg, f'{tag}_scen', full=phone); rules(pg, f'{tag} scenarios', lang)
            other = 'zh' if lang == 'en' else 'en'
            s0 = state(pg); click(pg, f'.langsw button[data-lang="{other}"]', 300); s1 = state(pg)
            check(f'{tag} scenarios: the language switch keeps the setup', all(s0[k] == s1[k] for k in ('mode', 'B', 'tau', 'd', 'steps', 'q')), (s0, s1))
            click(pg, f'.langsw button[data-lang="{lang}"]', 300)
            if phone:
                pg.evaluate("()=>{const a=document.querySelector('#emb-pan-map:not([hidden])')||document.querySelector('#emb-pan-chart'); window.scrollTo(0,a.getBoundingClientRect().bottom+scrollY+400)}"); pg.wait_for_timeout(800)
                dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return d&&!d.hidden&&d.getBoundingClientRect().height>0?d.innerText:''}")
                check(f'{tag}: the dock shows once the stage scrolls away', bool(dk) and '%' in dk, dk)
                shot(pg, f'{tag}_dock'); rules(pg, f'{tag} dock', lang); pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(200)
            pg.close()
            pg = page(b, w, h, lang, hash_='embeddings~16_0.05_16_200_9'); st = state(pg)
            check(f'{tag}: the hash round trip opens the same setup in scenarios', st['mode'] == 'sandbox' and (st['B'], st['tau'], st['d'], st['steps'], st['q']) == (16, 0.05, 16, 200, 9), st)
            # ---- challenge
            mode(pg, 'challenge'); st = state(pg)
            nores = lambda: '%' not in text(pg, '#emb-ch-av') and not text(pg, '#emb-ch-res').strip() and Z(lang, 'Not trained yet', '还没训练') in text(pg, '#emb-status') and state(pg)['steps'] == 0
            check(f'{tag} challenge: starts at B = 2, 25 steps, τ = 0.05, 16 dimensions, no result before Submit', st['mode'] == 'challenge' and (st['B'], st['chN'], st['tau'], st['d'], st['steps']) == (2, 25, 0.05, 16, 0) and nores(), (st, text(pg, '#emb-ch-av'), text(pg, '#emb-status')))
            check(f'{tag} challenge: the steps and B are shown before Submit', '25' in text(pg, '#emb-ch-gv') and 'B = 2' in text(pg, '#emb-ch-gv'), text(pg, '#emb-ch-gv'))
            click(pg, '#emb-rail .seg-ctl[data-key="B"] button[data-v="16"]')
            check(f'{tag} challenge: another B before Submit still shows no result', nores() and 'B = 16' in text(pg, '#emb-ch-gv'), (text(pg, '#emb-ch-av'), text(pg, '#emb-status'), state(pg)))
            click(pg, '#emb-rail .seg-ctl[data-key="B"] button[data-v="2"]')
            a = pg.evaluate("""async ()=>{document.querySelector('#emb-rail [data-act="submit"]').click(); await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
              return {a:window.__emb.animState(), res:document.querySelector('#emb-ch-res').innerText.trim(), st:document.querySelector('#emb-status').innerText}}""")
            check(f'{tag} challenge: Submit trains; no verdict while it runs', a['a'] is not None and not a['res'] and Z(lang, 'Training', '训练中') in a['st'], a)
            wait_anim(pg); pg.wait_for_timeout(150)
            res = text(pg, '#emb-ch-res')
            check(f'{tag} challenge: B = 2 × 25 is not met, with a hint and “{Z(lang, "Show the answer", "直接看答案")}”', Z(lang, 'Not yet', '还没达到') in res and vis(pg, '#emb-rail [data-act="reveal"]') and Z(lang, 'Average under 55%', '平均不到 55%') in text(pg, '#emb-status'), res)
            click(pg, '#emb-rail .seg-ctl[data-key="B"] button[data-v="16"]')
            check(f'{tag} challenge: a new B after a miss needs a new Submit', not state(pg)['chSubmitted'] and not text(pg, '#emb-ch-res').strip() and vis(pg, '#emb-rail [data-act="submit"]'), state(pg))
            click(pg, '#emb-rail [data-act="submit"]'); wait_anim(pg); pg.wait_for_timeout(150)
            res = text(pg, '#emb-ch-res')
            check(f'{tag} challenge: B = 16 × 25 is met, but B = 16 does it in 10 steps', Z(lang, 'Goal met', '达成目标') in res and Z(lang, 'B = 16 reaches 55% in 10 steps', 'B = 16 只要 10 步就能到 55%') in res and Z(lang, 'Average reaches 55%', '平均达到 55%') in text(pg, '#emb-status'), res[:300])
            click(pg, '#emb-rail .seg-ctl[data-key="chN"] button[data-v="10"]')
            res = text(pg, '#emb-ch-res')
            check(f'{tag} challenge: B = 16 × 10 is met in the fewest steps; the chart of the choices appears', Z(lang, 'no choice reaches 55% in fewer steps', '没有哪种选择能用更少的步数到 55%') in res and vis(pg, '#emb-ch-res .cmp'), res[:300])
            rows = pg.evaluate("()=>[...document.querySelectorAll('#emb-ch-res .cmp-row')].filter(r=>!r.closest('details')).map(r=>({ok:r.classList.contains('ok'),me:r.classList.contains('me'),tag:(r.querySelector('.cmp-tag,.cmp-you')||{}).textContent||'',lab:r.querySelector('.cmp-lab').textContent}))")
            check(f'{tag} challenge: the chart tags the initial setup and the pick; every other row reaches 55%', sum(r['me'] for r in rows) == 1 and any(Z(lang, 'the challenge’s default', '挑战的默认选择') in r['tag'] for r in rows) and all(r['ok'] or r['me'] or r['tag'] for r in rows), [r for r in rows if not (r['ok'] or r['me'] or r['tag'])] or rows)
            check(f'{tag} challenge: the capability card', vis(pg, '#emb-rail [data-exits]'))
            shot(pg, f'{tag}_challenge', full=phone); rules(pg, f'{tag} challenge', lang)
            check(f'{tag}: no horizontal overflow at the end', overflow(pg) <= 0, overflow(pg))
            pg.close()
    # ---- 900 wide: the map and the table side by side or stacked, chart text inside, no overflow
    if not [x for x in ARGS if x in ('1440', '390')]:
        for lang in LANGS:
            tag = f'{lang}-900'; pg = page(b, 900, 900, lang, hash_='embeddings~8_0.05_16_200_0')
            out_ = pg.evaluate(SVGIN); check(f'{tag} scenarios: every chart label inside its picture', not out_, out_)
            check(f'{tag}: no horizontal overflow', overflow(pg) <= 0, overflow(pg))
            shot(pg, f'{tag}_scen'); pg.close()
            pg = page(b, 900, 900, lang)
            for i in (1, 3):
                click(pg, f'#emb-rail [data-go="{i}"]', 300); mathfit(pg, f'{tag} part {i + 1}')
            pg.close()
    b.close()

for h in sorted(set(rule_hits)): print('RULE', h)
check('writing rules, colours, overflow and tooltips', not rule_hits, len(rule_hits))
check('no page errors', not errs, errs[:6])
# read from the lab's own source, so both languages are covered
_SRC = ''.join((P / d / 'embeddings.js').read_text() for d in ('labs', 'teasers'))
_G1 = re.findall(r'.{0,30}(?:一次只能选|不会叠加|do not stack|[Oo]ne at a time|一次选一|第 [0-9一二三四五六七八] 部分里|[Ii]n part [0-9]|part [0-9]’s|第 [0-9] 部分的|parts? [0-9] (?:to|and) [0-9]|第 [0-9] 到第 [0-9] 部分).{0,30}', _SRC)
check('no “one at a time” note and no sentence recalling a part by number (EN and ZH, R6.34)', not _G1, _G1[:3])
_H4 = re.findall(r'.{0,30}(?:规划中|还没上线|not built|later lab|以后的 Lab|将来|[Pp]lanned lab).{0,30}', _SRC)
check('no lab called planned or not yet live', not _H4, _H4[:3])
_TOY = re.findall(r'.{0,20}(?:\btoy\b(?! run)|玩具(?!实跑)).{0,20}', re.sub(r'Toy run|玩具实跑', '', _SRC), flags=re.I)
check('“toy” only in the Toy run label', not _TOY, _TOY[:3])
print(f'{n - len(fails)} / {n} passed')
sys.exit(1 if fails else 0)
