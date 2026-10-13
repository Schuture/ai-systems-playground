"""Flow test for the lora lab (CONTRACT §6). Analytical lab: no toy and no animation, so R3.9 and R3.20 do not apply (checked: no
player, no replay control). Every mode and every guided part at 1440×900 and 390×844, in EN and ZH:
- no page errors, no horizontal overflow, no text wider than its box (R7.8), a tooltip for every help cursor (R1.1), the writing rules
  on every visible text node and tooltip (R6.14, Chinese spacing and punctuation, no 亿/万), colours (R5.6);
- part 1: only the idea (review codex-b9v3 应修 2): no gate, the two-path picture (frozen W, A then B, the sum), no shapes, no rank, no matrix
  picker, no estimation notes in Settings; the answer (0.52%, 41.9M) appears nowhere (R3.8);
- part 2: shapes, rank and the 7 matrices; the formula; Next waits for another matrix; Show me picks down (0.50%); B and A drawn;
- part 3: the prediction gate with the kit's reason; nothing of the answer before the pick; the verdict with the right answer; Skip;
- part 4: Next waits for LoRA; 168.7 GB → 40.8 GB; the red overflow on the full fine-tuning column (R5.11); the estimation notes appear,
  closed, without the QLoRA note;
- part 5: Next waits for a change of rank or matrices; r = 64: 2.7 GB of LoRA parameters; 应修 6: q and v / attention / all 7 switch
  B and A on and off for k and down (the same check runs in Settings and in the challenge);
- part 6: 16,384 tokens without recomputation 113.2 GB in red; with it 29.4 GB; Next waits for both;
- part 7: QLoRA 50.3 GB; QLoRA without recomputation 153.6 GB in red; the QLoRA note joins the estimation notes; part 8: misconceptions;
- Scenarios: four, single choice, the first pressed on entry, the URL carries the setup and a fresh page restores it;
- the challenge: Settings locked, the three feedback tiers; 应修 5: the summary table first (three blocks, each with the setting that sets it),
  the next block to shrink, the plan with the least memory beside yours once revealed or met, “See all plans” closed by default and
  9 rows with 3 passing once opened, tags; the capability card;
  a language switch keeps the plan and the result;
- the dock at 390 once the chart scrolls away.
Screenshots: shots/lora/.
Run: python3 tests/lora_flow.py [en|zh] [1440|390] [guided|modes]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/lora.html', '--only', 'lora'], check=True, cwd=P, stdout=subprocess.DEVNULL)
OUT = P / 'out' / 'lora'; OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'wrapped-flow.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'lora.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-flow.html')
SH = P / 'shots' / 'lora'; SH.mkdir(parents=True, exist_ok=True)
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
RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy” (the fidelity label “Toy run” is the exception, R6.11)'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/Locked in|已锁定/,'“Locked in”'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万 instead of K, M, B (R6.18)'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour (R5.7)'],
   [/一次只能选|一次选一个|不会叠加|do not stack|one at a time/i,'a mechanics note above options instead of a lead-in (G1, the owner 2026-10-08)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, .fid-tag')||!e.getClientRects().length) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); for(const [re,name] of bad) if(re.test(t)) out.push(name+' (tooltip): '+t.slice(0,90));});
 root.querySelectorAll('[aria-label]').forEach(e=>{const t=e.getAttribute('aria-label'); if(/一次只能选|一次选一个|不会叠加|do not stack|one at a time/i.test(t)) out.push('a mechanics note in an aria-label (G1): '+t.slice(0,90));});
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
   if(![...e.childNodes].some(c=>c.nodeType===3&&c.textContent.trim())) return; texts.push(e.textContent);});
 root.querySelectorAll('[data-tipx]').forEach(e=>texts.push(e.getAttribute('data-tipx')));
 for(const t of texts){const sents=t.replace(/\\s+/g,' ').split(/(?<=[。？！])|(?<=[.?!])\\s+/);
   for(const x of sents){const k=(x.match(/：|: /g)||[]).length; if(k>=2) out.add(x.trim().slice(0,160));}}
 return [...out];}"""
BIGNUM = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #lora-mathbody, .num-big, #lora-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/lora-b-w|lora-b-fw|lora-b-fb|lora-w(?![-\\w])|lora-sw-w|lora-sw-fw|lora-sw-fb|lora-sw-q4|lora-pqln|lora-dk|tz-lora|sw w/,[amber]:/^$/,[red]:/status bad|badline|badc|lora-over|lora-oomt|lora-tot|lora-miss|lora-poln|lora-sw-over|\\bbad\\b|warn|wrong|ex-res|sw over/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#lora-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#lora-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#lora-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
LABELS = """()=>{const svg=document.querySelector('#lora-mem svg'); if(!svg) return []; const host=document.getElementById('lora-mem').getBoundingClientRect();
 const hit=(a,b,t)=>a.left<b.right-t&&b.left<a.right-t&&a.top<b.bottom-t&&b.top<a.bottom-t;
 const ts=[...svg.querySelectorAll('text')].map(t=>({t:t.textContent, r:t.getBoundingClientRect()})).filter(o=>o.r.width>0), out=[];
 for(let i=0;i<ts.length;i++){const a=ts[i]; if(a.r.left<host.left-0.5||a.r.right>host.right+0.5) out.push('cut at the edge: '+a.t);
   for(let j=i+1;j<ts.length;j++) if(hit(a.r,ts[j].r,2)) out.push('labels overlap: '+a.t+' / '+ts[j].t);}
 const cols=[...svg.querySelectorAll('.lora-colbox')].map(c=>c.getBoundingClientRect());
 svg.querySelectorAll('text[data-col]').forEach(t=>{const r=t.getBoundingClientRect(); cols.forEach(c=>{if(hit(r,c,0.5)) out.push('a narrow total over a column: '+t.textContent);});});
 return out;}"""
rule_hits, fit_hits = [], []
# the owner's review of 2026-10-08. G2: no sentence leans on a detail from an earlier part (“you saw in part 2”, “part 4’s 16.1 GB”);
# a pointer forward (“part 3 reveals it”) or to the lab where an idea is taught is fine.
BACK = re.compile(r'你在|看到过|讲过|第 ?\d+ ?部分(?:里|中|看到|的两)|（第 ?\d+ ?部分）|\byou saw\b|\bseen in\b|\b[Ii]n part \d|\(part \d\)|\bfrom part \d')
backs = lambda t: [m.group(0) for m in BACK.finditer(t)]
G1 = re.compile(r'一次只能选|一次选一个|不会叠加|do not stack|one at a time', re.I)
# item 1: the matrix picture labels W, A, B and B·A with their shapes (rows × columns); no label over another label or a block; all inside
MATGEOM = """()=>{const svg=document.querySelector('#lora-mat svg.lora-matsvg'); if(!svg) return null; const sb=svg.getBoundingClientRect();
  const ts=[...svg.querySelectorAll('text')].map(t=>({k:t.dataset.lbl||'', t:t.textContent.replace(/\\u00a0/g,' '), r:t.getBoundingClientRect()}));
  const blocks=[...svg.querySelectorAll('rect.lora-w,rect.lora-ba')].map(e=>e.getBoundingClientRect());
  const hit=(a,b,m)=>a.left<b.right-m&&b.left<a.right-m&&a.top<b.bottom-m&&b.top<a.bottom-m, bad=[];
  ts.forEach((a,i)=>{if(a.r.left<sb.left-0.5||a.r.right>sb.right+0.5||a.r.top<sb.top-0.5||a.r.bottom>sb.bottom+0.5) bad.push('outside the picture: '+a.t);
    ts.slice(i+1).forEach(b=>{if(hit(a.r,b.r,0.5)) bad.push('labels overlap: '+a.t+' / '+b.t);});
    blocks.forEach(r=>{if(hit(a.r,r,0.5)) bad.push('a label over a block: '+a.t);});});
  const lbl={}; ts.forEach(x=>{if(x.k) lbl[x.k]=x.t;});
  return {lbl, bad, aria:svg.getAttribute('aria-label'), tips:[...svg.querySelectorAll('[data-tipx]')].map(e=>e.getAttribute('data-tipx')).join(' | ')};}"""
SHAPES = {'q': {'W': 'W 4,096 × 4,096', 'A': 'A 16 × 4,096', 'B': 'B 4,096 × 16', 'BA': 'B·A 4,096 × 4,096'},
          'k': {'W': 'W 1,024 × 4,096', 'A': 'A 16 × 4,096', 'B': 'B 1,024 × 16', 'BA': 'B·A 1,024 × 4,096'},
          'down': {'W': 'W 4,096 × 14,336', 'A': 'A 16 × 14,336', 'B': 'B 4,096 × 16', 'BA': 'B·A 4,096 × 14,336'}}


def page(b, w, h, lang, hash_='lora'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__lora.state()")
shown = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return !!e&&e.checkVisibility()}", sel)
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#lora-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#lora-rail [data-next-why]')].filter(e=>!e.hidden&&e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden').map(e=>e.textContent)")
obs = lambda pg: pg.evaluate("()=>{const o=document.querySelector('#lora-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' '):''}")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__lora||!window.__lora.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel, wait=250):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(wait); settle(pg)


def slide(pg, sel, v):
    pg.evaluate(f"""()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}));}}"""); pg.wait_for_timeout(250); settle(pg)


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#lora-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


# the owner's clarification (2026-10-07 23:37 UTC): no hover only expands a shown number to its exact value; Show the math may write one out
EXACT_HOVER = """r=>{const root=document.querySelector(r), out=[]; if(!root) return out;
  for(const e of root.querySelectorAll('[data-tipx],[data-tip]')){if(e.closest('.mathbody')) continue; let t=e.getAttribute('data-tipx')||''; const k=e.getAttribute('data-tip');
    if(k&&window.ASP) t+=' '+(window.ASP.t(k)||''); if(/精确值是|Exactly/.test(t)) out.push(t.slice(0,100));}
  for(const e of root.querySelectorAll('.num-big')) if(!e.closest('.mathbody')&&(e.hasAttribute('data-tipx')||e.closest('.def'))) out.push('hover on '+e.textContent);
  return out;}"""


# H1 (the owner, 2026-10-09): no formula written as plain text (text, hovers, the tip on screen), and no TeX command left as a word in MathML
PLAINRE = r'd_in|d_out|α / r|α / √r|/ √r|scaling = lora_alpha|x\(AB\)|\(xA\)B|B·A x|r × d|d × r'   # PEFT’s scaling line is code (<code>), never prose
PLAINF = '''([r,re])=>{const root=document.querySelector(r), out=[]; if(!root) return out; const rx=new RegExp(re); let n;
  const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
  while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('math,pre,code,script,style')) continue; if(rx.test(n.textContent)) out.push('text: '+n.textContent.trim().slice(0,60));}
  for(const e of root.querySelectorAll('[data-tipx]')){const t=e.getAttribute('data-tipx'); if(rx.test(t)) out.push('hover: '+t.slice(0,60));}
  const tip=document.getElementById('tip'); if(tip&&!tip.hidden){const w=document.createTreeWalker(tip,NodeFilter.SHOW_TEXT); while((n=w.nextNode())){if(n.parentElement.closest('math')) continue; if(rx.test(n.textContent)) out.push('tip: '+n.textContent.trim().slice(0,60));}}
  for(const m of document.querySelectorAll('math')) for(const i of m.querySelectorAll('mi')) if(/^(frac|sqrt|left|right|hat|text|cdot|times|div|approx|sum|mathrm|operatorname|quad|beta|pi|sigma|rho|theta|bar|tilde|overline|langle|rangle|mid|le|ge|big|Big)$/.test(i.textContent)) out.push('TeX word: '+i.textContent+' in '+m.textContent.slice(0,40));
  return out;}'''
# H2: each named formula's lines (MathML text), its operators, whether the operators line up; H5: the hairline above each block heading
EQJS = '''(sel)=>[...document.querySelectorAll(sel+' .lora-eq')].map(e=>{const os=[...e.querySelectorAll('.lora-eq-o')];
  const xs=os.filter(o=>o.querySelector('math')).map(o=>Math.round(o.querySelector('math').getBoundingClientRect().left));
  return {name:(e.querySelector('.lora-eq-n')||{}).textContent||'', lines:[...e.querySelectorAll('.lora-fx')].map(m=>m.textContent.replace(/\\s+/g,'')), ops:os.map(o=>o.textContent.trim()), aligned:new Set(xs).size<=1};})'''
MHJS = '''(sel)=>[...document.querySelectorAll(sel+' .lora-mh')].map(h=>{const cs=getComputedStyle(h); return {t:h.textContent.slice(0,30), rule:cs.borderTopWidth+' '+cs.borderTopStyle};})'''


def rules(pg, where, lang):
    pf = pg.evaluate(PLAINF, ['#lora-app', PLAINRE]); check(f'{where}: H1: no formula written as plain text; every formula rendered (no TeX word)', not pf, pf[:3])
    hx = pg.evaluate(EXACT_HOVER, '#lora-app'); check(f'{where}: no exact-value hover outside Show the math', not hx, hx[:3])
    for sel in ('#lora-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def fit(pg, where):
    for h in pg.evaluate(OVER): fit_hits.append(f'{where}: text wider than its box (R7.8): {h}')
    for h in pg.evaluate(LABELS): fit_hits.append(f'{where}: memory chart labels (R7.8, review S21): {h}')
    for h in pg.evaluate(HELP): fit_hits.append(f'{where}: help cursor without a tooltip (R1.1): {h}')


# the owner's review of 2026-10-09 (R6.37, R6.33): with Show the math open, no box in it scrolls sideways (a typeset formula never wraps
# by itself, so a row wider than its box hid its end until the reader scrolled), and Show the math is no wider than its own box
MSCROLL = r"""(r)=>{const v=document.querySelector(r); if(!v) return []; const out=[], ds=[...v.querySelectorAll('details.math')], was=ds.map(d=>d.open);
  ds.forEach(d=>{d.open=true;});
  for(const d of ds){if(!d.getClientRects().length) continue;
    for(const e of d.querySelectorAll('*')){if(!e.getClientRects().length||!/auto|scroll/.test(getComputedStyle(e).overflowX)) continue;
      const by=e.scrollWidth-e.clientWidth; if(by>1) out.push(by+' px: '+(e.textContent||'').trim().replace(/\s+/g,' ').slice(0,40));}
    const by=d.scrollWidth-d.clientWidth; if(by>1) out.push('the whole Show the math, '+by+' px');}
  ds.forEach((d,i)=>{d.open=was[i];}); return out;}"""


def math_scroll(pg, where):
    ms = pg.evaluate(MSCROLL, '#lora-app')
    check(f'{where}: no row of Show the math scrolls sideways at {pg.viewport_size["width"]} px (the owner, 2026-10-09; R6.37)', not ms, ms[:3])


def every(pg, where, lang, full_shot=None):
    math_scroll(pg, where)
    check(f'{where}: no overflow', overflow(pg) <= 0, overflow(pg))
    rules(pg, where, lang); fit(pg, where)
    if full_shot: shot(pg, full_shot, full=pg.viewport_size['width'] <= 760)


def before_answer(pg):
    return pg.evaluate("""()=>{const t=[]; for(const s of ['#lora-app .labhead','#lora-rail','#lora-stage','#lora-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#lora-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")


def nav(pg, what):
    click(pg, f'#lora-rail [data-nav="{what}"]', 300)


def wait_end(pg, timeout=15000):
    try: pg.wait_for_function("()=>window.__lora.state().playT>=1&&!window.__lora.state().playing", timeout=timeout); return True
    except Exception: return False


def scrub_end(pg):
    slide(pg, '#lora-player input[data-key="scrub"]', 1000)




def gb_of(v):
    return f"{v/1e9:,.1f} GB"


def run_guided(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    phone = w <= 760
    check(f'{tag} title', text(pg, '#lora-app .labtitle') == Z(lang, 'How does LoRA fine-tune an 8B model on one GPU?', 'LoRA 怎么在一张卡上微调 8B 模型？'), text(pg, '#lora-app .labtitle'))
    check(f'{tag} no animation: no player and no replay control (R3.9 / R3.20 do not apply)', pg.evaluate("()=>!document.querySelector('#lora-app [data-act=\"play\"], #lora-app .player')"))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#lora-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        pg.evaluate(f"()=>document.querySelector('#lora-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(200)
        check(f'{tag} popover {pid} opens', pg.evaluate(f"()=>{{const p=document.getElementById('{pid}'); return !!p&&p.getClientRects().length>0}}"))
        rules(pg, f'{tag} popover {pid}', lang)
        pg.evaluate(f"()=>document.querySelector('#lora-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(150)
    # ---- part 1: only the idea (review codex-b9v3 应修 2)
    st = state(pg)
    check(f'{tag} part 1: guided, step 0, no gate', st['mode'] == 'guided' and st['step'] == 0 and nexts(pg) and not any(nexts(pg)), (st, nexts(pg)))
    ba = before_answer(pg)
    check(f'{tag} part 1: the answer appears nowhere (R3.8)', '0.52%' not in ba and '41.9M' not in ba and '41,943,040' not in ba, [l for l in ba.split('\n') if '0.52' in l or '41.9' in l][:3])
    check(f'{tag} part 1: the memory chart shows full fine-tuning only, 168.7 GB in red', pg.evaluate("()=>document.querySelectorAll('#lora-mem .lora-colbox').length") == 1 and vis(pg, '#lora-mem .lora-over') and '168.7 GB' in text(pg, '#lora-stage'))
    fig = pg.evaluate("""()=>{const s=document.querySelector('#lora-mat .lora-ideasvg'), g=document.getElementById('lora-mat-segw');
      return {idea:!!s, mat:!!document.querySelector('#lora-mat .lora-matsvg'), w:!!(s&&s.querySelector('.lora-w')), ba:s?s.querySelectorAll('.lora-ba').length:0,
        plus:!!(s&&s.querySelector('.lora-plus')), labels:s?[...s.querySelectorAll('text')].map(t=>t.textContent).join('|'):'', seg:!!g&&!g.hidden&&g.getClientRects().length>0}}""")
    check(f'{tag} part 1: the picture is the idea: a frozen W, A then B beside it, the two results added; no shapes, no matrix picker (应修 2)',
          fig['idea'] and not fig['mat'] and fig['w'] and fig['ba'] == 2 and fig['plus'] and not fig['seg'] and Z(lang, 'frozen', '冻结') in fig['labels'], fig)
    p1 = text(pg, '#lora-rail .step.current') or text(pg, '#lora-rail')
    check(f'{tag} part 1: SFT linked and spelled out, the two paths in words; no shapes, rank or 7 matrices yet (应修 2)',
          pg.evaluate("()=>!!document.querySelector('#lora-rail a[href=\"#sft\"]')") and Z(lang, 'supervised fine-tuning', '监督微调') in p1
          and Z(lang, 'two paths', '两条路') in p1 and 'd_in' not in p1 and Z(lang, 'rank', '秩') not in p1 and 'gate' not in p1, p1[:500])
    notes = lambda: pg.evaluate("""()=>{const r={}; for(const id of ['lora-ctl-note','lora-asm-q','lora-asm-lr']){const e=document.getElementById(id); r[id]=!!e&&!e.hidden&&!e.closest('[hidden]');}
      const d=document.getElementById('lora-asm'); r.open=!!d&&d.open; r.summary=d?d.querySelector('summary').textContent:''; return r}""")
    check(f'{tag} part 1: the 18 bytes stated in full with a pointer to the lab that teaches them, no “you saw” (G2)',
          Z(lang, '(see Training memory)', '（见训练显存）') in p1 and not backs(p1), backs(p1) or p1[:400])
    nt = notes()
    check(f'{tag} part 1: Settings shows no estimation notes yet (应修 2)', not nt['lora-ctl-note'], nt)
    every(pg, f'{tag} part 1', lang, f'{tag}_p1')
    # ---- part 2: shapes, rank, the seven matrices
    nav(pg, 'next')
    p2 = text(pg, '#lora-rail .step.current') or text(pg, '#lora-rail')
    check(f'{tag} part 2: d_in and d_out, the rank and the 7 matrices defined here (应修 2)', Z(lang, '-dimensional input', ' 维的输入') in p2 and pg.evaluate("()=>[...document.querySelectorAll('#lora-rail .step.current math')].some(m=>m.querySelector('msub')&&m.textContent==='din')") and Z(lang, 'gate, up and down', 'gate、up、down') in p2 and Z(lang, 'rank', '秩') in p2, p2[:500])
    check(f'{tag} part 2: the forward formula with A.ui.math', pg.evaluate("()=>!!document.querySelector('#lora-rail .lora-fx math')"))
    check(f'{tag} part 2: the matrix picture draws W, B and A', pg.evaluate("()=>!!document.querySelector('#lora-mat .lora-matsvg')&&document.querySelectorAll('#lora-mat .lora-ba').length===2"))
    check(f'{tag} part 2: the two paths said again in words before the formula, no “part 1” (G2)',
          Z(lang, 'goes through the frozen W and, beside it, through A and then B', '一路乘冻结的 W，一路先乘 A、再乘 B') in p2 and not backs(p2), backs(p2) or p2[:500])
    mg = pg.evaluate(MATGEOM)
    check(f'{tag} part 2, q: W, A, B and B·A each labelled with its shape, B’s too; no label over another or over a block (the owner, item 1)',
          mg and mg['lbl'] == SHAPES['q'] and not mg['bad'], mg)
    check(f'{tag} part 2, q: the tooltips of B and B·A and the picture’s accessible name give the shapes (item 1)',
          mg and Z(lang, 'B: 4,096 × 16 (outputs × r)', 'B：4,096 × 16（输出 × r）') in mg['tips'] and Z(lang, 'B·A: 4,096 × 4,096, the same shape as W', 'B·A：4,096 × 4,096，和 W 的形状一样') in mg['tips']
          and Z(lang, 'B 4,096 × 16', 'B 是 4,096 × 16') in mg['aria'], mg)
    mb = pg.evaluate("()=>{const e=document.getElementById('lora-mathbody'); return e?e.textContent.replace(/\\u00a0/g,' '):''}")
    eqs = pg.evaluate(EQJS, '#lora-mathbody')
    check(f'{tag} part 2: Show the math names B’s and A’s shapes in the count (item 1), typeset: d_out × r + r × d_in, then 4,096 × 16 + 16 × 4,096, then 131,072 (H1, H2)',
          Z(lang, 'B has', 'B 有') in mb and any(e['lines'][:3] == ['dout×r+r×din', '4,096×16+16×4,096', '131,072'] and e['aligned'] for e in eqs), eqs[:2])
    check(f'{tag} part 2: Next waits for another matrix', all(nexts(pg)) and any(Z(lang, 'another matrix', '换一个矩阵') in x for x in why(pg)), why(pg))
    ba = before_answer(pg)
    check(f'{tag} part 2: the answer still appears nowhere (R3.8); no estimation notes yet', '0.52%' not in ba and '41.9M' not in ba and not notes()['lora-ctl-note'], notes())
    click(pg, '#lora-rail [data-nav="showme"]', 350)
    st = state(pg); o = obs(pg)
    check(f'{tag} part 2: Show me picks down: 0.50% of W, Next enabled', st['mat'] == 'down' and '0.50%' in o and not any(nexts(pg)), (st['mat'], o[:300]))
    mg = pg.evaluate(MATGEOM)
    check(f'{tag} part 2, down (W too wide for labels at its side): the four shapes under W, none over another (item 1)', mg and mg['lbl'] == SHAPES['down'] and not mg['bad'], mg)
    click(pg, '#lora-rail .seg-ctl[data-key="mat"] button[data-v="k"]')
    o = obs(pg)
    check(f'{tag} part 2: k: 2.0% of W', '2.0%' in o, o[:300])
    mg = pg.evaluate(MATGEOM)
    check(f'{tag} part 2, k: W 1,024 × 4,096 = B 1,024 × 16 times A 16 × 4,096 (item 1)', mg and mg['lbl'] == SHAPES['k'] and not mg['bad'], mg)
    every(pg, f'{tag} part 2', lang, f'{tag}_p2')
    # ---- part 3: the prediction, with everything it needs already defined (R3.21)
    nav(pg, 'next')
    check(f'{tag} part 3: Next disabled with its reason', nexts(pg) and all(nexts(pg)) and any(Z(lang, 'Pick an answer', '选一个答案') in x for x in why(pg)), (nexts(pg), why(pg)))
    ba = before_answer(pg)
    check(f'{tag} part 3: the answer appears nowhere before the pick (R3.8)', '0.52%' not in ba and '41.9M' not in ba and '41,943,040' not in ba, [l for l in ba.split('\n') if '0.52' in l or '41.9' in l][:3])
    every(pg, f'{tag} part 3 question', lang)
    click(pg, '#lora-rail [data-pred="10-30"]')
    rt = text(pg, '#lora-rail')
    check(f'{tag} part 3: the verdict, wrong pick in red, the right answer, 41.9M, 0.52%, Next enabled', vis(pg, '#lora-rail .lora-verdict.badline') and Z(lang, 'Under 2%', '不到 2%') in rt and '41.9M' in rt and '0.52%' in rt and not any(nexts(pg)), rt[:400])
    check(f'{tag} part 3: the small share stated again, without “you saw in part 2” (G2)',
          Z(lang, 'On every matrix, B and A have only a small share of W’s parameters.', '每个矩阵旁边的 B 和 A，参数都只占 W 的一小部分。') in rt and not backs(rt), backs(rt) or rt[:500])
    check(f'{tag} part 3: the trained-parameter card shows the LoRA and says so', '41.9M' in text(pg, '#lora-metrics') and Z(lang, 'With a rank-16 LoRA', '如果加秩 16 的 LoRA') in text(pg, '#lora-metrics'), text(pg, '#lora-metrics'))
    every(pg, f'{tag} part 3', lang, f'{tag}_p3')
    # ---- part 4
    nav(pg, 'next')
    check(f'{tag} part 4: Next waits for LoRA', all(nexts(pg)) and any(Z(lang, 'Pick LoRA', '选 LoRA') in x for x in why(pg)), why(pg))
    p4 = text(pg, '#lora-rail')
    l3a = pg.evaluate("()=>{const e=document.querySelector('#lora-ic-l3 [aria-label]'); return e?e.getAttribute('aria-label'):''}")
    check(f'{tag} part 4: a lead-in above the methods gives the setup and what the chart shows; the group is named the same way (G1)',
          Z(lang, 'with one sequence of 4,096 tokens and no recomputation. Pick a fine-tuning method and see how much memory one step needs.', '一条 4,096 个 token 的序列，不重算。选一种微调方法，看看一步要多少显存。') in p4
          and l3a == Z(lang, 'Pick a fine-tuning method', '选一种微调方法') and not G1.search(p4 + l3a), (l3a, p4[:600]))
    check(f'{tag} part 4: before the pick, no LoRA column and no 40.8 GB (R3.8)', pg.evaluate("()=>document.querySelectorAll('#lora-mem .lora-colbox').length") == 1 and '40.8' not in before_answer(pg))
    check(f'{tag} part 4: the full fine-tuning column in red (R5.11)', vis(pg, '#lora-mem .lora-over') and vis(pg, '#lora-stage .status.bad'))
    click(pg, '#lora-rail [data-lever="l3:lora"]')
    st = state(pg); o = obs(pg)
    check(f'{tag} part 4: LoRA 168.7 GB → 40.8 GB, fits, Next enabled', st['meth'] == 'lora' and '168.7 GB' in o and '40.8 GB' in o and not any(nexts(pg)) and vis(pg, '#lora-stage .status.ok'), o[:300])
    check(f'{tag} part 4: two columns in the chart', pg.evaluate("()=>document.querySelectorAll('#lora-mem .lora-colbox').length") == 2)
    check(f'{tag} part 4: the compute card says the frozen weights get no gradients (review S11)', Z(lang, 'the frozen weights get no gradients', '冻结的权重不求梯度') in text(pg, '#lora-metrics'), text(pg, '#lora-metrics'))
    nt = notes()
    check(f'{tag} part 4: the estimation notes appear in Settings, under “{nt["summary"]}”, closed, without the QLoRA note (应修 2)', nt['lora-ctl-note'] and nt['lora-asm-lr'] and not nt['open'] and not nt['lora-asm-q'] and nt['summary'] == Z(lang, 'Estimation assumptions', '估算假设'), nt)
    every(pg, f'{tag} part 4', lang, f'{tag}_p4')
    if phone:
        pg.evaluate("()=>{const t=document.getElementById('lora-vis'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+120)}"); pg.wait_for_timeout(700)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!!d&&!d.hidden, status:(document.getElementById('dkStatus')||{}).textContent, read:(document.getElementById('dkRead')||{}).textContent, bar:!!document.querySelector('#dock .lora-dk')}}")
        check(f'{tag} part 4: the dock with a short status and the memory', dk['shown'] and dk['status'] == Z(lang, 'Fits', '放得下') and '40.8 GB' in (dk['read'] or '') and dk['bar'], dk)
        shot(pg, f'{tag}_p4_dock')
        rules(pg, f'{tag} part 4 dock', lang)
        pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(200)
    # ---- part 5
    nav(pg, 'next')
    check(f'{tag} part 5: Next waits for a change', all(nexts(pg)), why(pg))
    slide(pg, '#lora-ic-r-in', 4)
    st = state(pg); o = obs(pg)
    check(f'{tag} part 5: r = 64: 4 times the initial setup, 2.7 GB, Next enabled', st['r'] == 64 and Z(lang, '4 times as many', '初始设置的 4 倍') in o and '2.7 GB' in o and not any(nexts(pg)), (st['r'], o[:300]))
    click(pg, '#lora-rail .seg-ctl[data-key="tg"] button[data-v="qv"]')
    o = obs(pg)
    check(f'{tag} part 5: r = 64 on q and v: 65% of the initial setup', Z(lang, 'only 65% as many', '只有初始设置的 65%') in o, o[:300])
    tg_mat_check(pg, f'{tag} part 5', lang, '#lora-rail .seg-ctl[data-key="tg"]')
    every(pg, f'{tag} part 5', lang, f'{tag}_p5')
    # ---- part 6
    nav(pg, 'next')
    check(f'{tag} part 6: Next waits', all(nexts(pg)), why(pg))
    slide(pg, '#lora-ic-seq-in', 4)
    o = obs(pg)
    check(f'{tag} part 6: 16,384 tokens without recomputation: 113.2 GB in red, 88.0 GB of activations', '113.2 GB' in o and '88.0 GB' in o and vis(pg, '#lora-rail [data-obs] .badline') and vis(pg, '#lora-mem .lora-over') and all(nexts(pg)), o[:300])
    every(pg, f'{tag} part 6 oom', lang, f'{tag}_p6_oom')
    click(pg, '#lora-rail .seg-ctl[data-key="rc"] button[data-v="1"]')
    o = obs(pg)
    check(f'{tag} part 6: with recomputation 29.4 GB, fits, Next enabled', '29.4 GB' in o and not any(nexts(pg)), o[:300])
    check(f'{tag} part 6: the compute card adds the 2N of recomputation (review S11)', Z(lang, '4N, plus 2N to recompute', '4N，加上重算的 2N') in text(pg, '#lora-metrics'), text(pg, '#lora-metrics'))
    check(f'{tag} part 6: LoRA with recomputation about 6N per token, 4N without; full fine-tuning with recomputation 8N, why, and where (the owner, item 2)',
          Z(lang, 'LoRA with recomputation costs about 6N operations per token', 'LoRA 加上重算，每个 token 约要 6N 次运算') in o and Z(lang, 'against about 4N without it', '不重算时约 4N') in o
          and Z(lang, 'Full fine-tuning with recomputation costs 8N', '全量微调加上重算是 8N') in o and Z(lang, '2N more than LoRA', '比 LoRA 多 2N') in o
          and pg.evaluate("()=>!!document.querySelector('#lora-rail [data-obs] a[href=\"#training-memory\"]')"), o[:700])
    mb = pg.evaluate("()=>{const e=document.getElementById('lora-mathbody'); return e?e.textContent.replace(/\\u00a0/g,' '):''}")
    check(f'{tag} part 6: Show the math writes the compute per token term by term: 2N + 2N + 2N ≈ 6N, and names 8N for full fine-tuning (item 2)',
          Z(lang, 'operations per token', '每个 token 的运算') in mb and '8N' in mb
          and any(e['lines'][1:] == ['2N+2N+2N', '6N'] and e['ops'][-1] == '≈' and e['aligned'] for e in pg.evaluate(EQJS, '#lora-mathbody')), mb[-700:])
    p6 = text(pg, '#lora-rail')
    check(f'{tag} part 6: recomputation points to the lab that teaches it, no “taught earlier” (G2)', Z(lang, '(see Training memory)', '（见训练显存）') in p6 and not backs(p6), backs(p6) or p6[:500])
    every(pg, f'{tag} part 6', lang, f'{tag}_p6')
    # ---- part 7
    nav(pg, 'next')
    check(f'{tag} part 7: Next waits for QLoRA; 70B LoRA in red', all(nexts(pg)) and vis(pg, '#lora-stage .status.bad'), why(pg))
    p7 = text(pg, '#lora-rail')
    l6a = pg.evaluate("()=>{const e=document.querySelector('#lora-ic-l6 [aria-label]'); return e?e.getAttribute('aria-label'):''}")
    check(f'{tag} part 7: a lead-in above the choices gives the setup and what the chart shows; the group is named the same way (G1)',
          Z(lang, 'and every layer is recomputed. Pick a way to store the frozen weights and see how much memory one step of the 70B needs.', '每一层都重算。选一种存冻结权重的办法，看看 70B 一步要多少显存。') in p7
          and l6a == Z(lang, 'Pick how to store the frozen weights', '选一种存冻结权重的办法') and not G1.search(p7 + l6a), (l6a, p7[:700]))
    check(f'{tag} part 7: before the pick, no QLoRA column and no 50.3 GB (R3.8)', pg.evaluate("()=>document.querySelectorAll('#lora-mem .lora-colbox').length") == 2 and '50.3' not in before_answer(pg))
    click(pg, '#lora-rail [data-lever="l6:qlora"]')
    o = obs(pg)
    check(f'{tag} part 7: QLoRA 151.9 GB → 50.3 GB, Next enabled', '151.9 GB' in o and '50.3 GB' in o and not any(nexts(pg)), o[:300])
    check(f'{tag} part 7: three columns', pg.evaluate("()=>document.querySelectorAll('#lora-mem .lora-colbox').length") == 3)
    nt = notes()
    check(f'{tag} part 7: the QLoRA note joins the estimation notes (应修 2)', nt['lora-asm-q'], nt)
    every(pg, f'{tag} part 7', lang, f'{tag}_p7')
    click(pg, '#lora-rail [data-lever="l6:qnorc"]')
    o = obs(pg)
    check(f'{tag} part 7: QLoRA without recomputation 153.6 GB in red', '153.6 GB' in o and '108.7 GB' in o and vis(pg, '#lora-rail [data-obs] .badline'), o[:300])
    pressed = pg.evaluate("()=>[...document.querySelectorAll('#lora-rail [data-lever]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} part 7: single choice', pressed == ['false', 'false', 'true'], pressed)
    # ---- part 8
    nav(pg, 'next')
    rt = text(pg, '#lora-rail')
    check(f'{tag} part 8: five misconceptions (four with a chart button), See the scenarios', pg.evaluate("()=>document.querySelectorAll('#lora-rail .lora-rem').length") == 5 and pg.evaluate("()=>document.querySelectorAll('#lora-rail .lora-remb').length") == 4 and pg.evaluate("()=>!!document.querySelector('#lora-rail [data-mode-go=\"sandbox\"]')"), rt[-300:])
    check(f'{tag} part 8: the quality item quotes both READMEs with their scope', all(x in rt for x in ('GLUE', 'RoBERTa', 'DeBERTa', '1.5B', 'LLaMA', 'T5')), rt[-700:])
    check(f'{tag} part 8: the recap keeps each rule and says what follows, no number to recall from an earlier part (G2)',
          all(x in rt for x in (Z(lang, 'every layer still runs, so a GPU must first hold every frozen weight', '每一层都还要跑，所以一张卡先得放下全部冻结的权重'),
                                Z(lang, 'Then what helps is recomputation; a smaller rank does not.', '这时该开的是重算，把秩调小没有用。'),
                                Z(lang, 'enough to fit a much larger model on one GPU, while the compute per token stays the same', '一张卡能放下大得多的模型，每个 token 的计算量却不变')))
          and not backs(rt) and not any(v in rt for v in ('16.1 GB', '113.2 GB', '50.3 GB')), backs(rt) or rt[:700])
    names = {'small': Z(lang, 'Show the frozen weights', '在图里看冻结的权重'), 'acts': Z(lang, 'Show a long sequence', '在图里看长序列'),
             'rank': Z(lang, 'Show rank 64', '在图里看秩 64'), 'q4': Z(lang, 'Show the 70B with QLoRA', '在图里看 70B 用 QLoRA')}
    labels = pg.evaluate("()=>Object.fromEntries([...document.querySelectorAll('#lora-rail [data-rem]')].map(b=>[b.dataset.rem,b.textContent]))")
    check(f'{tag} part 8: each chart button names the case it shows (G2)', labels == names, labels)
    want = {'small': (Z(lang, 'frozen', '冻结'), '16.1 GB'), 'acts': ('16,384', '113.2 GB'), 'rank': ('r = 64', '2.7 GB'), 'q4': ('70B', '50.3 GB')}
    for k in ('small', 'acts', 'rank', 'q4'):
        click(pg, f'#lora-rail [data-rem="{k}"]')
        st = state(pg); stg = text(pg, '#lora-stage') + ' ' + pg.evaluate("()=>[...document.querySelectorAll('#lora-mem [data-tipx]')].map(e=>e.dataset.tipx).join(' ')")
        lab = text(pg, f'#lora-rail [data-rem="{k}"]')
        check(f'{tag} part 8: “{names[k]}” puts its case in the chart ({want[k][1]}) and the button then says so', st['remView'] == k and pg.evaluate(f"()=>document.querySelector('#lora-rail [data-rem=\"{k}\"]').getAttribute('aria-pressed')") == 'true'
              and want[k][1] in stg and lab == Z(lang, 'In the chart now', '图里正显示'), (st['remView'], lab, stg[:500]))
    every(pg, f'{tag} part 8', lang, f'{tag}_p8')
    pg.close()
    pg = page(b, w, h, lang)
    nav(pg, 'next'); click(pg, '#lora-rail .seg-ctl[data-key="mat"] button[data-v="k"]'); nav(pg, 'next')
    nav(pg, 'skip')
    st = state(pg)
    check(f'{tag} part 3: Skip goes on without a prediction and without a verdict', st['step'] == 3 and st['pred'] is None and not pg.evaluate("()=>!!document.querySelector('#lora-rail .lora-verdict')"), st)
    click(pg, '#lora-rail [data-lever="l3:lora"]')
    check(f'{tag} part 4 after Skip: the LoRA count shows once LoRA is picked', '41.9M' in text(pg, '#lora-metrics'), text(pg, '#lora-metrics'))
    pg.close()


def tg_mat_check(pg, where, lang, tg_sel):
    """review codex-b9v3 应修 6: switch the matrices that get a LoRA; k and an MLP matrix draw B and A only when they are targets."""
    want = {'qv': {'k': False, 'down': False, 'v': True}, 'attn': {'k': True, 'down': False}, 'all': {'k': True, 'down': True}}
    for tg, mats in want.items():
        click(pg, f'{tg_sel} button[data-v="{tg}"]')
        for m, on in mats.items():
            click(pg, f'#lora-mat-segw button[data-v="{m}"]')
            r = pg.evaluate("""()=>{const w=document.querySelector('#lora-mat .lora-w'); return {tg:window.__lora.state().tg, ba:document.querySelectorAll('#lora-mat .lora-ba').length,
              note:document.getElementById('lora-mat-note').innerText, cap:document.getElementById('lora-mat-cap').innerText, wtip:w?w.getAttribute('data-tipx'):'',
              tips:[...document.querySelectorAll('#lora-mat [data-tipx]')].map(e=>e.getAttribute('data-tipx')).join(' | '), lbl:[...document.querySelectorAll('#lora-mat text[data-lbl]')].map(t=>t.dataset.lbl).join(',')}}""")
            no = Z(lang, f'no LoRA on {m}', f'没有给 {m} 加 LoRA')
            if on:
                ok = r['tg'] == tg and r['ba'] == 2 and no not in r['note'] and Z(lang, 'trained', '要训练') in r['tips'] and r['lbl'] == 'W,A,B,BA'
            else:
                ok = (r['tg'] == tg and r['ba'] == 0 and no in r['note'] and no in r['wtip'] and Z(lang, 'frozen', '冻结') in r['wtip'] and r['lbl'] == 'W'
                      and Z(lang, 'trained', '要训练') not in r['tips'] and Z(lang, f'{m} has no LoRA', f'没给 {m} 加 LoRA') in r['cap'])
            check(f'{where}: LoRA on {tg}, matrix {m}: ' + ('B and A drawn, trained' if on else 'only the frozen W, and it says so (应修 6)'), ok, r)


def run_modes(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    mode(pg, 'sandbox')
    rt = text(pg, '#lora-rail')
    check(f'{tag} scenarios: a lead-in names what the chart shows; the group named for screen readers (G1)',
          Z(lang, 'Pick a scenario and see where one step’s memory goes.', '选一个场景，看看一步的显存都花在哪里。') in rt and not G1.search(rt)
          and pg.evaluate("()=>document.querySelector('#lora-rail .presets').getAttribute('aria-label')") == Z(lang, 'Pick a scenario', '选一个应用场景'), rt[:400])
    check(f'{tag} scenarios: four, the first pressed on entry', pg.evaluate("()=>document.querySelectorAll('#lora-rail [data-preset]').length") == 4
          and pg.evaluate("()=>document.querySelector('#lora-rail [data-preset=\"0\"]').getAttribute('aria-pressed')") == 'true', state(pg))
    want = {0: ['40.8 GB', '22.0 GB', '30.3'], 1: ['9.5 GB', '19.9 GB', Z(lang, 'about 10 GB', '约 10 GB'), '7.4'], 2: ['50.3 GB', '141.1 GB'], 3: ['1,270.0 GB', '16']}
    for i in range(4):
        click(pg, f'#lora-rail [data-preset="{i}"]')
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#lora-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        ins = text(pg, '#lora-insight')
        check(f'{tag} scenario {i}: pressed alone, with its result', pressed.count('true') == 1 and pressed[i] == 'true' and all(x in ins for x in want[i]), (pressed, ins[:300]))
        dl = pg.evaluate("()=>[...document.querySelectorAll('#lora-metrics .d')].map(d=>d.textContent).join('')")
        check(f'{tag} scenario {i}: no ▲▼ when a scenario sets the state (R6.6, review S20)', dl == '', dl)
        every(pg, f'{tag} scenario {i}', lang, f'{tag}_scenario_{i}' if i in (1, 3) else None)
    check(f'{tag} scenario 3: red, out of memory (R5.11)', vis(pg, '#lora-stage .status.bad') and vis(pg, '#lora-mem .lora-over'))
    click(pg, '#lora-rail [data-preset="2"]')
    hsh = pg.evaluate("()=>location.hash")
    check(f'{tag} scenarios: the URL carries the setup', hsh.startswith('#lora~'), hsh)
    pg2 = page(b, w, h, lang, hash_=hsh[1:])
    st = state(pg2)
    check(f'{tag} hash round-trip: Scenarios, the 70B QLoRA pressed', st['mode'] == 'sandbox' and st['m'] == '70b' and st['meth'] == 'qlora'
          and pg2.evaluate("()=>document.querySelector('#lora-rail [data-preset=\"2\"]').getAttribute('aria-pressed')") == 'true', st)
    pg2.close()
    # Settings in Scenarios
    click(pg, '#lora-controls .seg-ctl[data-key="gpu"] button[data-v="g24"]')
    st = state(pg)
    check(f'{tag} Settings: the 24 GB card, no scenario pressed', st['gpu'] == 'g24' and pg.evaluate("()=>[...document.querySelectorAll('#lora-rail [data-preset]')].every(b=>b.getAttribute('aria-pressed')==='false')"), st)
    click(pg, '#lora-controls .seg-ctl[data-key="meth"] button[data-v="lora"]')
    tg_mat_check(pg, f'{tag} Settings', lang, '#lora-controls .seg-ctl[data-key="tg"]')
    every(pg, f'{tag} settings', lang, f'{tag}_settings' if w > 760 else None)
    click(pg, '#lora-controls .seg-ctl[data-key="meth"] button[data-v="full"]'); click(pg, '#lora-controls .seg-ctl[data-key="rc"] button[data-v="1"]')
    mt = text(pg, '#lora-metrics')
    check(f'{tag} Settings, full fine-tuning with recomputation: 8N, 6N plus 2N (the owner, item 2)', '8N' in mt and Z(lang, 'full fine-tuning’s 6N, plus 2N to recompute', '全量微调 6N，加上重算的 2N') in mt, mt)
    click(pg, '#lora-controls .seg-ctl[data-key="rc"] button[data-v="0"]')
    mt = text(pg, '#lora-metrics')
    check(f'{tag} Settings, full fine-tuning without recomputation: 6N', '6N' in mt and '8N' not in mt, mt)
    # ---- the challenge
    mode(pg, 'challenge')
    st = state(pg)
    check(f'{tag} challenge: starts at LoRA r = 8 on q and v, no recomputation, 70B on 48 GB', (st['m'], st['gpu'], st['seq'], st['meth'], st['r'], st['tg'], st['rc']) == ('70b', 'g48', 2048, 'lora', 8, 'qv', 0), st)
    check(f'{tag} challenge: Settings locked', pg.evaluate("()=>document.getElementById('lora-controls').classList.contains('locked')") and vis(pg, '#lora-lock'))
    check(f'{tag} challenge: plain statement', Z(lang, 'Your team has one GPU with 48 GB', '你们只有一张 48 GB 的卡') in text(pg, '#lora-rail'))
    tg_mat_check(pg, f'{tag} challenge', lang, '#lora-rail .seg-ctl[data-key="cht"]')
    click(pg, '#lora-rail .seg-ctl[data-key="cht"] button[data-v="qv"]'); click(pg, '#lora-mat-segw button[data-v="k"]')
    check(f'{tag} challenge: back at the start, k on q and v shows only its frozen W (the review’s case, 应修 6)', state(pg)['tg'] == 'qv' and pg.evaluate("()=>document.querySelectorAll('#lora-mat .lora-ba').length") == 0)
    every(pg, f'{tag} challenge start', lang, f'{tag}_challenge_start')
    click(pg, '#lora-rail [data-act="submit"]')
    r = text(pg, '#lora-ch-res')
    SUM = """()=>{const res=document.getElementById('lora-ch-res'), t=res.querySelector('.lora-sum'), d=document.getElementById('lora-allp');
      return {order:[...res.children].map(e=>e.tagName.toLowerCase()+(e.className?'.'+e.className.split(' ').join('.'):'')).slice(0,3), cols:t?t.querySelectorAll('thead th').length:0,
        rows:t?[...t.querySelectorAll('tbody tr')].map(r=>r.innerText.replace(/\\s+/g,' ')):[], diff:t?[...t.querySelectorAll('tbody tr.lora-diff th')].map(e=>e.innerText.split('\\n')[0]):[],
        all:!!d, open:!!d&&d.open, summary:d?d.querySelector('summary').textContent:''}}"""
    sm = pg.evaluate(SUM)
    check(f'{tag} challenge tier 3: not yet, then the summary first: three blocks with the setting that sets each, your plan alone (应修 5)',
          Z(lang, 'Not yet', '还没达到') in r and sm['order'][:2] == ['p.badline.lora-mt', 'table.lora-sum'] and sm['cols'] == 2 and len(sm['rows']) == 5
          and all(x in ' '.join(sm['rows']) for x in ('141.1 GB', '0.3 GB', '55.4 GB', '196.8 GB', '16.4M', Z(lang, 'the method sets it', '由方法决定'), Z(lang, 'recomputation sets it', '由重算决定'), Z(lang, 'the rank and the matrices set it', '由秩和矩阵决定'))), sm)
    check(f'{tag} challenge tier 3: the next block to shrink and its setting, a hint, no answer yet, no list of plans', Z(lang, 'The block to shrink next is the weights, 141.1 GB. The method sets it.', '接下来要压的是权重这一块，有 141.1 GB。它由方法决定。') in r
          and Z(lang, 'Hint', '提示') in r and vis(pg, '#lora-ch-res [data-act="reveal"]') and '44.9 GB' not in r and not sm['all'], r[:600])
    rv = text(pg, '#lora-ch-res [data-act="reveal"]').strip()
    check(f'{tag} challenge tier 3: the reveal button says “{Z(lang, "Show the answer", "直接看答案")}” (the owner, batch 8)', rv == Z(lang, 'Show the answer', '直接看答案'), rv)
    every(pg, f'{tag} challenge tier 3', lang, f'{tag}_challenge_tier3')
    click(pg, '#lora-rail .seg-ctl[data-key="chm"] button[data-v="qlora"]')
    r = text(pg, '#lora-ch-res')
    check(f'{tag} challenge tier 3, QLoRA without recomputation: the activations are next, recomputation sets them', Z(lang, 'The block to shrink next is the activations and logits', '接下来要压的是激活和 logits 这一块') in r and Z(lang, 'Recomputation sets it.', '它由重算决定。') in r, r[:600])
    click(pg, '#lora-rail .seg-ctl[data-key="chm"] button[data-v="qlora"]')
    click(pg, '#lora-rail .seg-ctl[data-key="chc"] button[data-v="1"]')
    click(pg, '#lora-rail .seg-ctl[data-key="cht"] button[data-v="all"]')
    slide(pg, '#lora-ic-chr-in', 2)
    r = text(pg, '#lora-ch-res')
    check(f'{tag} challenge tier 2 (QLoRA, recompute, all 7, r = 16): goal met, but a plan with less memory', Z(lang, 'Goal met', '达成目标') in r and Z(lang, 'a plan with less memory', '还有更省的方案') in r and '46.6 GB' in r and '44.9 GB' in r, r[:300])
    sm = pg.evaluate(SUM)
    check(f'{tag} challenge tier 2: your plan next to the least-memory plan, the LoRA row marked, the gap in words, the full list closed (应修 5)',
          sm['cols'] == 3 and sm['diff'] == [Z(lang, 'LoRA parameters, with gradients and optimizer state', 'LoRA 参数（含梯度和优化器状态）')] and '3.3 GB' in sm['rows'][1] and '1.7 GB' in sm['rows'][1]
          and Z(lang, 'The gap is the LoRA parameters. Your plan trains 2 times as many', '差距在 LoRA 参数这一块。你的方案要训练的参数是最省方案的 2 倍') in r
          and sm['all'] and not sm['open'] and sm['summary'] == Z(lang, 'See all settings that pass (6)', '看全部可行的设置（6 种）') and not shown(pg, '#lora-ch-res .cmp'), (sm, r[:500]))
    every(pg, f'{tag} challenge tier 2', lang, f'{tag}_challenge_tier2')
    pg.evaluate("()=>{const d=document.getElementById('lora-allp'); d.open=true}"); pg.wait_for_timeout(250)
    check(f'{tag} challenge: “See all plans” opens the chart', shown(pg, '#lora-ch-res .cmp'))
    slide(pg, '#lora-ic-chr-in', 1)
    r = text(pg, '#lora-ch-res')
    check(f'{tag} challenge tier 1 (r = 8): goal met, and the least memory', Z(lang, 'no plan that meets the goal uses less memory', '没有比它更省显存的') in r and '44.9 GB' in r and '104M' in r, r[:300])
    sm = pg.evaluate(SUM)
    check(f'{tag} challenge tier 1: the summary alone (it is the least-memory plan), the list stays as the reader left it', sm['cols'] == 2 and sm['open'] and '39.5 GB' in sm['rows'][0] and '1.7 GB' in sm['rows'][1] and '3.7 GB' in sm['rows'][2], sm)
    rows = pg.evaluate("()=>{const c=document.querySelector('#lora-allp .cmp'); if(!c) return null; const rs=[...c.querySelectorAll('.cmp-row')]; return {all:rs.length, ok:rs.filter(r=>r.classList.contains('ok')).length, me:rs.filter(r=>r.classList.contains('me')).map(r=>r.querySelector('.cmp-lab').textContent), title:c.querySelector('.cmp-title').textContent}}")
    check(f'{tag} challenge: the full list holds only the 6 plans that pass, your pick (the least memory) among them (the owner, batch 8)',
          rows and rows['all'] == 6 and rows['ok'] == 6 and len(rows['me']) == 1 and Z(lang, 'your pick', '你的选择') in rows['me'][0] and Z(lang, 'least memory', '最省') in rows['me'][0], rows)
    check(f'{tag} challenge: the list’s title names the plans that meet the goal, not every plan', rows and Z(lang, 'plans that meet the goal', '达标的方案') in rows['title'] and not any(x in rows['title'] for x in ('each', 'every', '每种')), rows and rows['title'])
    tags = text(pg, '#lora-allp')
    check(f'{tag} challenge: no initial setup row in the list; the why names the plans that do not fit, their memory on hover',
          Z(lang, 'initial setup', '初始设置') not in text(pg, '#lora-allp .cmp') and Z(lang, 'does not fit', '放不下') in tags
          and all(x in pg.evaluate("()=>[...document.querySelectorAll('#lora-allp [data-tipx]')].map(e=>e.dataset.tipx).join(' ')") for x in ('146.5 GB', '96.6 GB')), tags[:400])
    check(f'{tag} challenge: the capability card', vis(pg, '#lora-rail [data-exits]'))
    pg.evaluate("()=>document.getElementById('lora-ch-res').scrollIntoView({block:'start'})"); pg.wait_for_timeout(300)
    every(pg, f'{tag} challenge solved', lang, f'{tag}_challenge_solved')
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} challenge: a language switch keeps the plan, the summary and the open list', (st['meth'], st['rc'], st['r'], st['tg']) == ('qlora', 1, 8, 'all') and st['chSubmitted'] and shown(pg, '#lora-ch-res .lora-sum') and shown(pg, '#lora-ch-res .cmp'), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    mode(pg, 'guided')
    check(f'{tag} back to Guided', state(pg)['mode'] == 'guided')
    every(pg, f'{tag} guided again', lang)
    pg.close()


SRC = (P / 'labs' / 'lora.js').read_text()
check('G1: no mechanics note above options (一次只能选 / 一次选一个 / 不会叠加 / do not stack / one at a time) anywhere in the lab, hidden text included', not G1.search(SRC), G1.findall(SRC))
PARTS = [x for x in ('guided', 'modes') if x in ARGS] or ['guided', 'modes']
with sync_playwright() as p:
    b = p.chromium.launch()
    for (w, h) in SIZES:
        for lang in LANGS:
            if 'guided' in PARTS: run_guided(b, w, h, lang)
            if 'modes' in PARTS: run_modes(b, w, h, lang)
    b.close()
for hit in rule_hits[:60]:
    check('writing rules: ' + hit, False)
for hit in fit_hits[:60]:
    check(hit, False)
check('no page errors', not errs, errs[:5])
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:60] if fails else 'none')
sys.exit(1 if fails else 0)
