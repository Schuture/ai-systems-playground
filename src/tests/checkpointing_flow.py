"""Flow test for the checkpointing lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH:
- no page errors, no horizontal overflow, no text wider than its box (R7.8), a tooltip for every help cursor (R1.1), the writing rules
  on every visible text node and tooltip (R6.14, Chinese spacing and punctuation, no 亿/万), colours (R5.6: red only for the failure);
- part 1: the prediction gate with the kit's reason under a greyed-out Next; before the pick the resumed run is not drawn and the
  answer (the losses at step 600) appears nowhere, in text, tooltips or the header's popovers (R3.8);
- part 2: the whole run plays on entering at Fast (R3.9) and ends within about 10 s; then the losses at step 600, the verdict, the red
  crash and the red “worse” area (R5.11); Next waits for all four parts (or Show me), and all four give the exact run;
- R3.20: a new set of parts replays only the resumed segment; going back to a set already watched does not replay; the replay control
  (Replay, Pause, scrubber) works; with the coordinator's addition, a run watched to the end shows its end at once, not a replay, when a
  later part, Scenarios, the challenge or a return to the part shows it again;
- part 3: the files, the red half-written file, 500 then 100 steps redone; part 4: Next waits for the interval and asynchronous saving,
  the numbers of the 70B; part 5: the spike plays once, then each fix; part 6: three points with “show in the chart”, three
  misconceptions, See the scenarios;
- Scenarios: four, single choice, the first pressed on entry, the URL carries the setup and a fresh page restores it;
- the challenge: Settings locked, the feedback tiers; the result as the train-capstone lab's (R1): your plan next to the least-waste
  plan, the differing settings marked, what fails in red, the passing plans only behind a closed disclosure; the capability card; a
  language switch keeps the plan and the result;
- owner b8: part 4 shows the formula's three terms with their meaning and draws each one; switching synchronous and asynchronous
  changes the strip, the saving part of the bar and the total; the restart is grey hatching, not red. Scenario 3 says the data position
  is not saved and the data strip shows the reread and the unread batches. The four Settings toggles keep a readable label on hover,
  focus and press, in light and dark;
- the dock at 390 once the chart scrolls away.
Screenshots: shots/checkpointing/.
Run: python3 tests/checkpointing_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/checkpointing.html', '--only', 'checkpointing'], check=True, cwd=P, stdout=subprocess.DEVNULL)
OUT = P / 'out' / 'checkpointing'; OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'wrapped-flow.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'checkpointing.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-flow.html')
SH = P / 'shots' / 'checkpointing'; SH.mkdir(parents=True, exist_ok=True)
ARGS = sys.argv[1:]
LANGS = [x for x in ('en', 'zh') if x in ARGS] or ['en', 'zh']
SIZES = [s for s in [(1440, 900), (390, 844)] if str(s[0]) in ARGS] or [(1440, 900), (390, 844)]
ONLY = [x for x in ('guided', 'modes') if x in ARGS] or ['guided', 'modes']
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
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour (R5.7)']];
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
   if(![...e.childNodes].some(c=>c.nodeType===3&&c.textContent.trim())) return; texts.push(e.textContent);});
 root.querySelectorAll('[data-tipx]').forEach(e=>texts.push(e.getAttribute('data-tipx')));
 for(const t of texts){const sents=t.replace(/\\s+/g,' ').split(/(?<=[。？！])|(?<=[.?!])\\s+/);
   for(const x of sents){const k=(x.match(/：|: /g)||[]).length; if(k>=2) out.add(x.trim().slice(0,160));}}
 return [...out];}"""
BIGNUM = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #ckp-mathbody, .num-big, #ckp-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|ckp-miss|ckp-worse|ckp-crashl|ckp-x|ckp-bad|ckp-badzone|ckp-lost|ckp-rst|ckp-hl|ckp-fail|ckp-file bad|ckp-verdict bad|\\bbad\\b|warn|wrong|ex-res|ckp-sw-worse|ckp-sw-lost|ckp-sw-rst|ckp-sw-fail|ckp-twice|ckp-never|ckp-nl|ckp-dsl/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#ckp-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#ckp-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#ckp-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
SVGTEXT = """()=>{const ts=[...document.querySelectorAll('#ckp-chart svg text')].filter(t=>t.textContent.trim()&&t.getClientRects().length), out=[];
  const rs=ts.map(t=>t.getBoundingClientRect()), host=document.getElementById('ckp-chart').getBoundingClientRect();
  for(let i=0;i<ts.length;i++){const a=rs[i]; if(a.left<host.left-1||a.right>host.right+1) out.push('outside the chart: '+ts[i].textContent);
    for(let j=i+1;j<ts.length;j++){const b=rs[j]; const ox=Math.min(a.right,b.right)-Math.max(a.left,b.left), oy=Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top);
      if(ox>1&&oy>2) out.push(ts[i].textContent+' / '+ts[j].textContent);}}
  return out;}"""
LEASTX = """()=>{const t=[...document.querySelectorAll('#ckp-chart svg text')].find(t=>/^(least|最少)/.test(t.textContent)); if(!t||!t.getClientRects().length) return [];
  const r=t.getBoundingClientRect(), out=[];
  for(const p of document.querySelectorAll('#ckp-chart svg polyline')){const L=p.getTotalLength(), m=p.getScreenCTM(); if(!m) continue;
    for(let s=0;s<=L;s+=1){const q=p.getPointAtLength(s), X=m.a*q.x+m.c*q.y+m.e, Y=m.b*q.x+m.d*q.y+m.f;
      if(X>r.left+1&&X<r.right-1&&Y>r.top+2&&Y<r.bottom-2){out.push(t.textContent+' crossed by '+(p.getAttribute('class')||'a curve')); break;}}}
  return out;}"""
CONTRAST = """(sel)=>{const e=document.querySelector(sel); if(!e) return null;
 const rgb=c=>{const m=c.match(/rgba?\\(([^)]+)\\)/); if(!m) return null; const v=m[1].split(',').map(x=>parseFloat(x)); return {r:v[0],g:v[1],b:v[2],a:v.length>3?v[3]:1};};
 let bg=null; for(let x=e;x;x=x.parentElement){const c=rgb(getComputedStyle(x).backgroundColor); if(c&&c.a>0.5){bg=c; break;}}
 if(!bg) bg={r:255,g:255,b:255};
 const fg=rgb(getComputedStyle(e).color), lum=c=>{const f=v=>{v/=255; return v<=0.03928?v/12.92:Math.pow((v+0.055)/1.055,2.4);}; return 0.2126*f(c.r)+0.7152*f(c.g)+0.0722*f(c.b);};
 const L1=lum(fg), L2=lum(bg); return {ratio:(Math.max(L1,L2)+0.05)/(Math.min(L1,L2)+0.05), text:e.innerText.trim(), fg:getComputedStyle(e).color, bg:'rgb('+bg.r+','+bg.g+','+bg.b+')'};}"""
rule_hits, fit_hits = [], []


def page(b, w, h, lang, hash_='checkpointing'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__ckp.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#ckp-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#ckp-rail [data-next-why]')].filter(e=>!e.hidden&&e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden').map(e=>e.textContent)")
obs = lambda pg: pg.evaluate("()=>{const o=document.querySelector('#ckp-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' '):''}")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__ckp||!window.__ckp.state().stageBusy", timeout=4000)
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
    pg.evaluate(f"()=>document.querySelector('#ckp-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def rules(pg, where, lang):
    for sel in ('#ckp-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def fit(pg, where):
    for h in pg.evaluate(OVER): fit_hits.append(f'{where}: text wider than its box (R7.8): {h}')
    for h in pg.evaluate(SVGTEXT): fit_hits.append(f'{where}: chart labels overlap (R7.8): {h}')
    for h in pg.evaluate(LEASTX): fit_hits.append(f'{where}: a curve runs through the label of the least waste (R7.8): {h}')
    for h in pg.evaluate(HELP): fit_hits.append(f'{where}: help cursor without a tooltip (R1.1): {h}')


def exact_hovers(pg):
    # the owner's clarification (2026-10-07 23:37 UTC): no hover only expands a shown number to its exact value; Show the math is exempt
    return pg.evaluate(r"""()=>{const out=[]; document.querySelectorAll('[data-tipx]').forEach(e=>{if(e.closest('#ckp-math')) return; const t=e.getAttribute('data-tipx')||'';
        if(/精确值是|Exactly|有效数字|significant digits/.test(t)) out.push(t.slice(0,120));});
      document.querySelectorAll('.num-big').forEach(e=>{if(e.closest('#ckp-math')) return; if(e.hasAttribute('data-tipx')||e.hasAttribute('tabindex')||e.classList.contains('def')) out.push('num-big with a hover: '+e.textContent);});
      return out}""")


# H1 (the owner, 2026-10-09: “全站帮我找到这些旧的，没有latex渲染的公式，并修好”): every formula is typeset with A.ui.math; no TeX command
# shows as a word inside the MathML, and no plain-text formula (x_i, x^y, √, <sup>/<sub>) is left in the lab's visible text or Show the math
FORMULA_H1 = r"""(sel)=>{const root=document.querySelector(sel); if(!root) return {n:0,bad:[],txt:[]};
  const CMD=/^(frac|left|right|text|sqrt|times|cdot|hat|approx|leq|geq|quad|phantom|mathbf|mathrm|operatorname|big|Big|alpha|beta|gamma|delta|eta|theta|lambda|mu|pi|rho|sigma|tau|phi|omega|epsilon|infty|ldots|cdots|leftarrow|rightarrow|div|neq)$/;
  const bad=[]; let n=0;
  root.querySelectorAll('math').forEach(m=>{n++; m.querySelectorAll('mi').forEach(x=>{const t=x.textContent.trim(); if(CMD.test(t)) bad.push(t);}); if(m.textContent.includes('\\')) bad.push(m.textContent.slice(0,40));});
  const PAT=/√|\b[A-Za-z]{1,2}_\{?[A-Za-z0-9]{1,3}\b|\b[A-Za-z]\^[A-Za-z0-9({α-ω]/;
  const txt=[]; const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let nd;
  while((nd=tw.nextNode())){const e=nd.parentElement; if(!e||e.closest('math,pre,code,svg,script,style')) continue; if(PAT.test(nd.nodeValue)) txt.push(nd.nodeValue.trim().slice(0,80));}
  root.querySelectorAll('sup,sub').forEach(e=>{if(!e.closest('math,pre,code')) txt.push('<'+e.tagName.toLowerCase()+'>'+e.textContent);});
  return {n,bad:bad.slice(0,5),txt:txt.slice(0,5)};}"""


def formulas_h1(pg, where):
    fm = pg.evaluate(FORMULA_H1, '#ckp-app')
    check(f'{where}: formulas typeset as MathML, no TeX word shown, no plain-text formula left (H1, R6.29)', not fm['bad'] and not fm['txt'], fm)


def every(pg, where, lang, full_shot=None):
    formulas_h1(pg, where)
    check(f'{where}: no overflow', overflow(pg) <= 0, overflow(pg))
    eh = exact_hovers(pg)
    check(f'{where}: no exact-value hover outside Show the math (the owner, 23:37 UTC)', not eh, eh[:3])
    if pg.viewport_size['width'] > 760:
        gap = pg.evaluate("()=>{const r=document.getElementById('ckp-rail'), s=document.getElementById('ckp-stage'); return r&&s&&r.getClientRects().length?s.getBoundingClientRect().left-r.getBoundingClientRect().right:99}")
        check(f'{where}: the rail does not run into the stage', gap >= 0, gap)
    rules(pg, where, lang); fit(pg, where)
    if full_shot: shot(pg, full_shot, full=pg.viewport_size['width'] <= 760)


def before_answer(pg):
    return pg.evaluate("""()=>{const t=[]; for(const s of ['#ckp-app .labhead','#ckp-rail','#ckp-stage','#ckp-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#ckp-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")


def nav(pg, what):
    click(pg, f'#ckp-rail [data-nav="{what}"]', 300)


def wait_end(pg, timeout=15000):
    try: pg.wait_for_function("()=>window.__ckp.state().playT>=1&&!window.__ckp.state().playing", timeout=timeout); return True
    except Exception: return False


def scrub_end(pg):
    slide(pg, '#ckp-player input[data-key="scrub"]', 1000)



hook = lambda pg, expr: pg.evaluate(f"()=>{{const h=window.__ckp; return {expr}}}")
lf = lambda v: f'{v:.2f}' if v >= 1 else f'{v:.3f}'


def wait_end(pg, timeout=15000):
    try: pg.wait_for_function("()=>window.__ckp.state().playT>=1&&!window.__ckp.state().playing", timeout=timeout); return True
    except Exception: return False


def lever(pg, v):
    click(pg, f'#ckp-rail [data-lever="{v}"]')


def run_guided(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    phone = w <= 760
    check(f'{tag} title', text(pg, '#ckp-app .labtitle') == Z(lang, 'Your run crashed at step 90,000: how do you resume it?', '训练到第 90,000 步崩了，怎么接着训？'), text(pg, '#ckp-app .labtitle'))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#ckp-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        pg.evaluate(f"()=>document.querySelector('#ckp-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(200)
        check(f'{tag} popover {pid} opens', pg.evaluate(f"()=>{{const p=document.getElementById('{pid}'); return !!p&&p.getClientRects().length>0}}"))
        rules(pg, f'{tag} popover {pid}', lang)
        pg.evaluate(f"()=>document.querySelector('#ckp-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(150)
    w0 = hook(pg, "(()=>{const r=h.resumed({}); return {final:r.final, final0:r.final0}})()")
    fw, f0 = lf(w0['final']), lf(w0['final0'])
    # ---- part 1
    st = state(pg)
    check(f'{tag} part 1: guided, step 0, nothing playing', st['mode'] == 'guided' and st['step'] == 0 and not st['playing'], st)
    check(f'{tag} part 1: Next disabled with its reason', nexts(pg) and all(nexts(pg)) and any(Z(lang, 'Pick an answer', '选一个答案') in x for x in why(pg)), (nexts(pg), why(pg)))
    ba = before_answer(pg)
    check(f'{tag} part 1: the answer appears nowhere before the pick (R3.8)', fw not in ba and f0 not in ba, [l for l in ba.split('\n') if fw in l or f0 in l][:3])
    check(f'{tag} part 1: the resumed run is not drawn', pg.evaluate("()=>document.querySelectorAll('#ckp-chart .ckp-res, #ckp-chart .ckp-worse').length") == 0)
    check(f'{tag} part 1: the crash is drawn in red (R5.11)', vis(pg, '#ckp-chart .ckp-x') and pg.evaluate("()=>document.querySelectorAll('#ckp-chart .ckp-crashl').length") == 1)
    every(pg, f'{tag} part 1', lang, f'{tag}_p1')
    click(pg, '#ckp-rail [data-pred="bump"]')
    check(f'{tag} part 1: Saved, Next enabled', Z(lang, 'Saved', '已记下') in text(pg, '#ckp-rail') and not any(nexts(pg)))
    # ---- part 2: the whole run plays at Fast
    nav(pg, 'next')
    st = state(pg)
    check(f'{tag} part 2: the whole run plays on entering, at Fast (R3.9)', st['step'] == 1 and st['playing'] and st['spd'] == 'fast' and st['from0'] and st['playT'] < 0.5, st)
    check(f'{tag} part 2: Next waits for all four parts', all(nexts(pg)) and any(Z(lang, 'Add all four', '把四样都加回去') in x for x in why(pg)), why(pg))
    if phone:
        scrub_end(pg)
    else:
        check(f'{tag} part 2: Fast ends within about 10 s', wait_end(pg, 12500), state(pg))
    pg.wait_for_timeout(300)
    st = state(pg); o = obs(pg)
    check(f'{tag} part 2: done and watched', st['playT'] >= 1 and not st['playing'] and 'ls:0000' in st['watched'], st)
    check(f'{tag} part 2: the losses at step 600 and the verdict', fw in o and f0 in o and Z(lang, 'The right answer is', '正确答案是') in o, o[:300])
    check(f'{tag} part 2: the red area where the resumed run is worse (R5.11)', pg.evaluate("()=>document.querySelectorAll('#ckp-chart .ckp-worse').length") > 5)
    every(pg, f'{tag} part 2', lang, f'{tag}_p2')
    if phone:
        pg.evaluate("()=>{const t=document.getElementById('ckp-graph'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+80)}"); pg.wait_for_timeout(700)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!!d&&!d.hidden, status:(document.getElementById('dkStatus')||{}).textContent, read:(document.getElementById('dkRead')||{}).textContent}}")
        check(f'{tag} part 2: the dock with a short status', dk['shown'] and dk['status'] == Z(lang, 'Not the same run', '不是同一次训练'), dk)
        shot(pg, f'{tag}_p2_dock')
        rules(pg, f'{tag} part 2 dock', lang)
    # R3.20: a new set of parts replays only the resumed segment
    click(pg, '#ckp-rail [data-part="o"]')
    st = state(pg)
    check(f'{tag} R3.20: a new set of parts replays, only the resumed part', st['o'] == 1 and st['playing'] and not st['from0'], st)
    scrub_end(pg)
    check(f'{tag} R3.20: the scrubber reaches the end and marks it watched', state(pg)['playT'] >= 1 and 'ls:1000' in state(pg)['watched'], state(pg))
    o = obs(pg)
    check(f'{tag} part 2: the optimizer state alone, the learning rate still warms up again', Z(lang, 'Still missing', '还缺') in o, o[:300])
    click(pg, '#ckp-rail [data-part="o"]')
    st = state(pg)
    check(f'{tag} R3.20: back to weights only (already watched): no replay', st['o'] == 0 and not st['playing'] and st['playT'] >= 1, st)
    # the replay control works
    click(pg, '#ckp-player [data-act="play"]', 400)
    st = state(pg)
    check(f'{tag} R3.20: “Replay” plays again', st['playing'] and st['playT'] < 0.5, st)
    check(f'{tag} R3.20: the control is called Replay before the press', True)
    click(pg, '#ckp-player [data-act="play"]', 200)
    st = state(pg)
    check(f'{tag} R3.20: pressing again pauses', not st['playing'] and 0 < st['playT'] < 1, st)
    scrub_end(pg)
    lbl = text(pg, '#ckp-player [data-act="play"]')
    check(f'{tag} R3.20: at the end the button reads Replay', Z(lang, 'Replay', '重放') in lbl, lbl)
    # Show me: all four
    click(pg, '#ckp-rail [data-nav="showme"]', 400)
    scrub_end(pg)
    st = state(pg); o = obs(pg)
    check(f'{tag} part 2: all four: the exact run, Next enabled', all(st[k] == 1 for k in 'osdr') and Z(lang, 'exactly on the gray one', '完全重合') in o and not any(nexts(pg)) and hook(pg, "h.resumed({o:1,s:1,d:1,r:1}).same"), (st, o[:200]))
    # coordinator's addition: re-entering part 2 shows the watched run at once
    nav(pg, 'back'); nav(pg, 'next')
    st = state(pg)
    check(f'{tag} R3.20+: re-entering part 2 shows the watched run at once, no replay', st['step'] == 1 and not st['playing'] and st['playT'] >= 1, st)
    # ---- part 3
    nav(pg, 'next')
    st = state(pg)
    check(f'{tag} part 3: the files, no player, Next waits', st['view'] == 'files' and not vis(pg, '#ckp-player') and all(nexts(pg)), st)
    o = obs(pg)
    check(f'{tag} part 3: overwrite: the red half-written file, 500 steps redone', vis(pg, '#ckp-chart .ckp-file.bad') and '500' in o and vis(pg, '#ckp-chart .ckp-verdict.bad'), o[:200])
    lever(pg, 'atom:atomic')
    o = obs(pg)
    check(f'{tag} part 3: atomic: resumes from 400, 100 steps redone, Next enabled', '100' in o and vis(pg, '#ckp-chart .ckp-verdict.ok') and not any(nexts(pg)), o[:200])
    every(pg, f'{tag} part 3', lang, f'{tag}_p3')
    # ---- part 4
    nav(pg, 'next')
    st = state(pg)
    check(f'{tag} part 4: the interval picture, 3 h synchronous, Next waits', st['view'] == 'waste' and st['T'] == 180 and st['save'] == 'sync' and all(nexts(pg)), st)
    t = text(pg, '#ckp-rail')
    fit(pg, f'{tag} part 4 (synchronous, every 3 h)')
    check(f'{tag} part 4: what a checkpoint holds, 0.99 TB, defined before the options (R2.10)', '0.99 TB' in t and Z(lang, '14 bytes per parameter', '14 字节') in t and Z(lang, 'Synchronous save', '同步存档') in t, t[:300])
    o = obs(pg)
    nm = pg.evaluate("()=>[...document.querySelectorAll('#ckp-rail .ckp-tt tr')].map(r=>[r.querySelector('math')?1:0, r.querySelector('b').textContent, r.querySelector('.ckp-tt-v').textContent])")
    check(f'{tag} part 4 (owner b8): the formula’s three terms, each with its meaning, its value and the picture’s swatch, on its own row', len(nm) == 3 and all(x[0] == 1 for x in nm)
          and [x[1] for x in nm] == [Z(lang, 'Saving', '存档'), Z(lang, 'Redone work', '重算'), Z(lang, 'Restarts', '重启')] and pg.evaluate("()=>document.querySelectorAll('#ckp-rail .ckp-tt .sw').length") == 3, nm)
    ter = lambda: pg.evaluate("""()=>{const t=document.getElementById('ckp-terms'); if(!t||t.hidden) return null; const seg=k=>{const e=t.querySelector('[data-seg="'+k+'"]'); return e?e.getBoundingClientRect().width:-1};
      return {save:seg('save'), lost:seg('lost'), restart:seg('restart'), sum:t.querySelector('.ckp-tsum').textContent, items:t.querySelectorAll('[data-term][data-tipx]').length,
        ticks:document.querySelectorAll('#ckp-chart .ckp-save').length, aticks:document.querySelectorAll('#ckp-chart .ckp-save.ckp-savea').length,
        rst:document.querySelector('#ckp-chart .ckp-rst')?getComputedStyle(document.querySelector('#ckp-chart .ckp-rst')).stroke:'', red:getComputedStyle(document.documentElement).getPropertyValue('--c-oom').trim()}}""")
    s0 = ter()
    check(f'{tag} part 4 (owner b8): under the picture, the three terms as a bar with a hover each; the strip draws the saves, the restart in grey hatching, not red', s0 and s0['items'] == 3 and s0['save'] > 0 and s0['lost'] > 0 and s0['restart'] > 0 and s0['ticks'] >= 3 and s0['aticks'] == 0 and s0['rst'] and 'rgb(207, 47, 42)' not in s0['rst'], s0)
    slide(pg, '#ckp-ic-T-in', 2)
    s1 = ter()
    click(pg, '#ckp-rail .seg-ctl[data-key="mode"] button[data-v="async"]')
    s2 = ter()
    check(f'{tag} part 4 (owner b8): at 15 min, asynchronous saving shrinks the saving part of the bar (at least 5 times), turns the strip’s blocks into thin ticks and changes the total',
          s1 and s2 and s2['save'] * 5 <= s1['save'] and s1['save'] >= 20 and abs(s2['lost'] - s1['lost']) < 0.6 and s2['aticks'] == s2['ticks'] > 0 and s1['aticks'] == 0 and s1['sum'] != s2['sum'], (s1, s2))
    slide(pg, '#ckp-ic-T-in', 6)
    st = state(pg); o = obs(pg)
    check(f'{tag} part 4: async every 1 h; least about every 15 min; Next enabled', st['T'] == 60 and st['save'] == 'async' and not any(nexts(pg)) and Z(lang, 'every 15 min', '每 15 分钟') in o, (st, o[:300]))
    every(pg, f'{tag} part 4', lang, f'{tag}_p4')
    # ---- part 5
    nav(pg, 'next')
    st = state(pg)
    check(f'{tag} part 5: the spike run plays once, whole', st['view'] == 'spike' and st['playing'] and st['from0'], st)
    chip = lambda v: pg.evaluate("v=>{const b=document.querySelector('#ckp-rail [data-lever=\"fix:'+v+'\"] .lv-r'); return b?b.textContent:null}", v)
    check(f'{tag} part 5: while the run plays, the option shows no loss at step 600 yet (R3.8)', chip('latest') == '', chip('latest'))
    scrub_end(pg)
    check(f'{tag} part 5: once watched, the option shows its loss at step 600', Z(lang, 'loss at step 600', '第 600 步 loss') in (chip('latest') or ''), chip('latest'))
    check(f'{tag} part 5: Next waits for the last option', all(nexts(pg)), why(pg))
    lever(pg, 'fix:back')
    st = state(pg)
    check(f'{tag} R3.20: rolling back replays only the resumed part', st['playing'] and not st['from0'], st)
    check(f'{tag} part 5: the new option shows its loss only after its run (R3.8)', chip('back') == '', chip('back'))
    scrub_end(pg)
    lever(pg, 'fix:skip')
    scrub_end(pg)
    o = obs(pg); r = hook(pg, "({skip:h.remedy('skip').final, latest:h.remedy('latest').final})")
    check(f'{tag} part 5: skip: no spike, the two losses, Next enabled', lf(r['skip']) in o and lf(r['latest']) in o and not any(nexts(pg)), o[:300])
    lever(pg, 'fix:latest')
    st = state(pg)
    check(f'{tag} R3.20: back to a watched fix: no replay', not st['playing'] and st['playT'] >= 1, st)
    every(pg, f'{tag} part 5', lang, f'{tag}_p5')
    # ---- part 6
    nav(pg, 'next')
    rt = text(pg, '#ckp-rail')
    check(f'{tag} part 6: three points, three misconceptions, See the scenarios', pg.evaluate("()=>document.querySelectorAll('#ckp-rail .ckp-remlist li').length") == 3
          and rt.count('✗') == 3 and pg.evaluate("()=>!!document.querySelector('#ckp-rail [data-mode-go=\"sandbox\"]')"), rt[-300:])
    for k in ('atomic', 'interval', 'weights'):
        click(pg, f'#ckp-rail [data-rem="{k}"]')
        st = state(pg)
        check(f'{tag} part 6: “show in the chart” {k}', st['remView'] == k and pg.evaluate(f"()=>document.querySelector('#ckp-rail [data-rem=\"{k}\"]').getAttribute('aria-pressed')") == 'true', st)
    st = state(pg)
    check(f'{tag} R3.20+: part 6 shows the watched weights-only run at once, no replay', st['view'] == 'loss' and not st['playing'] and st['playT'] >= 1, st)
    every(pg, f'{tag} part 6', lang, f'{tag}_p6')
    # the challenge and a return do not replay what was watched
    mode(pg, 'challenge'); mode(pg, 'guided')
    st = state(pg)
    check(f'{tag} R3.20+: back from the challenge to part 6, no replay', st['step'] == 5 and not st['playing'] and st['playT'] >= 1, st)
    for i in (4, 1):
        click(pg, f'#ckp-rail [data-go="{i}"]', 300)
        st = state(pg)
        check(f'{tag} R3.20+: returning to part {i + 1} shows the watched run at once', st['step'] == i and not st['playing'] and st['playT'] >= 1, st)
    pg.close()
    pg = page(b, w, h, lang)
    nav(pg, 'skip')
    check(f'{tag} part 1: Skip goes on without a prediction', state(pg)['step'] == 1 and state(pg)['pred'] is None)
    pg.close()


def run_modes(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    mode(pg, 'sandbox')
    lead = pg.inner_text('#ckp-rail .railcard > p.small')
    check(f'{tag} G1: the scenarios card leads in with what to pick and what to watch, not “one at a time”', lead == Z(lang, 'Pick a scenario and see what goes wrong and what the checkpoint does about it.', '选一个场景，看看会出什么问题、存档怎么应对。'), lead)
    check(f'{tag} scenarios: four, the first pressed on entry', pg.evaluate("()=>document.querySelectorAll('#ckp-rail [data-preset]').length") == 4
          and pg.evaluate("()=>document.querySelector('#ckp-rail [data-preset=\"0\"]').getAttribute('aria-pressed')") == 'true', state(pg))
    r = hook(pg, "({d:h.resumed({o:1,s:1,d:0,r:1}), skip:h.remedy('skip').final, latest:h.remedy('latest').final})")
    want = {0: ['15', '%'], 1: ['6', '%'], 2: [f"{r['d']['twice']:,}", f"{r['d']['never']:,}"], 3: [lf(r['skip']), lf(r['latest'])]}
    for i in range(4):
        click(pg, f'#ckp-rail [data-preset="{i}"]')
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#ckp-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        ins = text(pg, '#ckp-insight')
        if i in (2, 3):
            check(f'{tag} R3.20: scenario {i} shows a run not yet watched, so it plays', state(pg)['playing'], state(pg))
            scrub_end(pg); ins = text(pg, '#ckp-insight')
        check(f'{tag} scenario {i}: pressed alone, with its result', pressed.count('true') == 1 and pressed[i] == 'true' and all(x in ins for x in want[i]), (pressed, ins[:300]))
        every(pg, f'{tag} scenario {i}', lang, f'{tag}_scenario_{i}' if i in (0, 2, 3) else None)
    # owner b8, item 2: scenario 3 says what its checkpoint holds; the box and the data strip show it
    click(pg, '#ckp-rail [data-preset="2"]'); scrub_end(pg)
    pt = text(pg, '#ckp-rail [data-preset="2"]'); ins = text(pg, '#ckp-insight')
    box = ' | '.join(pg.evaluate("()=>[...document.querySelectorAll('#ckp-box .ckp-cell')].map(c=>c.textContent.replace(/\\s+/g,' ').trim())"))
    check(f'{tag} scenario 3 (owner b8): the scenario and its result say the data position is not saved, and what resuming does', Z(lang, 'lacks the data position', '没存数据读到哪里') in pt
          and Z(lang, 'but not where the data loader is', '唯独没存数据读到哪里') in ins and Z(lang, 'starts epoch 7 again from its first batch', '把第 7 个 epoch 从第一个 batch 重新读起') in ins
          and Z(lang, 'Where the data loader is (not saved)', '数据读到哪里（没存）') in box, (pt, ins[:300], box))
    geo = pg.evaluate("""()=>{const tw=document.querySelector('#ckp-chart .ckp-twice'), nv=document.querySelector('#ckp-chart .ckp-never'), cells=[...document.querySelectorAll('#ckp-chart [data-ep]')];
      const x=s=>{const sv=document.querySelector('#ckp-chart svg'); return null}; if(!tw||!nv) return {tw:!!tw, nv:!!nv};
      const r=tw.getBoundingClientRect(), n=nv.getBoundingClientRect(), e7=cells.filter(c=>c.dataset.ep==='7').map(c=>[c.getBoundingClientRect().left,c.getBoundingClientRect().width]);
      const ax=[...document.querySelectorAll('#ckp-chart svg text.ckp-mid')].filter(t=>/^[0-9]+$/.test(t.textContent)).map(t=>[+t.textContent,(t.getBoundingClientRect().left+t.getBoundingClientRect().right)/2]);
      return {tw:true,nv:true,twL:r.left,twR:r.right,nvL:n.left,e7,ax}}""")
    ok3 = geo.get('tw') and geo.get('nv')
    if ok3:
        X = lambda s_: next(x for v, x in geo['ax'] if v == 0) + (next(x for v, x in geo['ax'] if v == 600) - next(x for v, x in geo['ax'] if v == 0)) * s_ / 600
        ok3 = abs(geo['twL'] - X(400)) < 2 and abs(geo['twR'] - X(440)) < 2 and abs(geo['nvL'] - X(600)) < 2 and len(geo['e7']) == 2
    check(f'{tag} scenario 3 (owner b8): the data strip: epoch 7 twice, its first 40 steps (400–440) red as read twice, the 40 unread steps hatched past step 600', ok3, geo)
    shot(pg, f'{tag}_scenario_3_data')
    # owner b8, item 3: the four Settings toggles keep a readable label on hover, focus and press, in light and dark
    for theme in ('light', 'dark'):
        pg.evaluate(f"()=>{{document.documentElement.dataset.theme='{theme}'}}"); pg.wait_for_timeout(120)
        open_drawers(pg); pg.wait_for_timeout(150)
        for k in 'osdr':
            sel = f'#ckp-controls [data-part="{k}"]'
            pg.evaluate(f"()=>document.querySelector('{sel}').scrollIntoView({{block:'center'}})"); pg.wait_for_timeout(120)
            pressed = pg.evaluate(f"()=>document.querySelector('{sel}').getAttribute('aria-pressed')")
            pg.hover(sel); pg.wait_for_timeout(120); hv = pg.evaluate(CONTRAST, sel)
            pg.mouse.down(); pg.wait_for_timeout(60); dn = pg.evaluate(CONTRAST, sel); pg.mouse.move(2, 2); pg.mouse.up(); pg.wait_for_timeout(80)
            if pg.evaluate(f"()=>document.querySelector('{sel}').getAttribute('aria-pressed')") != pressed: pg.click(sel); pg.wait_for_timeout(150)
            pg.evaluate(f"()=>document.querySelector('{sel}').focus()"); pg.keyboard.press('Shift'); pg.wait_for_timeout(80); fc = pg.evaluate(CONTRAST, sel)
            good = lambda c: c and c['text'] and c['ratio'] >= 4.5
            check(f'{tag} Settings toggle {k} ({"pressed" if pressed == "true" else "not pressed"}, {theme}) (owner b8): readable on hover, press and focus', good(hv) and good(dn) and good(fc) and hv['text'] == fc['text'], (hv, dn, fc))
        if w > 760 and theme == 'light':
            pg.hover('#ckp-controls [data-part="o"]'); pg.wait_for_timeout(120); shot_el = pg.query_selector('#ckp-controls')
            if shot_el: shot_el.screenshot(path=str(SH / f'{tag}_settings_hover.png'))
    pg.evaluate("()=>{document.documentElement.dataset.theme='light'}"); pg.wait_for_timeout(100)
    click(pg, '#ckp-rail [data-preset="0"]'); click(pg, '#ckp-rail [data-preset="1"]')
    dlt = pg.evaluate("()=>{const d=document.querySelector('#ckp-metrics [data-k=\"w\"] .d'); return d?d.textContent:null}")
    check(f'{tag} scenarios: no ▲▼ when a scenario sets another run (R6.6)', dlt == '', dlt)
    click(pg, '#ckp-rail [data-preset="2"]')
    st = state(pg)
    check(f'{tag} R3.20+: a scenario watched before shows its end at once', not st['playing'] and st['playT'] >= 1, st)
    hsh = pg.evaluate("()=>location.hash")
    check(f'{tag} scenarios: the URL carries the setup', hsh.startswith('#checkpointing~'), hsh)
    pg2 = page(b, w, h, lang, hash_=hsh[1:])
    st = state(pg2)
    check(f'{tag} hash round-trip: Scenarios, the data-loader scenario pressed', st['mode'] == 'sandbox' and (st['o'], st['s'], st['d'], st['r']) == (1, 1, 0, 1)
          and pg2.evaluate("()=>document.querySelector('#ckp-rail [data-preset=\"2\"]').getAttribute('aria-pressed')") == 'true', st)
    pg2.close()
    # ---- the challenge
    mode(pg, 'challenge')
    st = state(pg)
    check(f'{tag} challenge: starts synchronous, every hour, keep 2; nothing plays', (st['chS']['mode'], st['chS']['T'], st['chS']['k']) == ('sync', 60, 2) and not st['playing'], st)
    check(f'{tag} challenge: Settings locked', pg.evaluate("()=>document.getElementById('ckp-controls').classList.contains('locked')") and vis(pg, '#ckp-lock'))
    check(f'{tag} challenge: statement with the size and the limits', '5.68 TB' in text(pg, '#ckp-rail') and '30 TB' in text(pg, '#ckp-rail'), text(pg, '#ckp-rail')[:300])
    every(pg, f'{tag} challenge start', lang, f'{tag}_challenge_start')
    click(pg, '#ckp-rail [data-act="submit"]')
    r = text(pg, '#ckp-ch-res')
    check(f'{tag} challenge tier 3: not yet, a hint, Show the plan that wastes least', Z(lang, 'Not yet', '还没达到') in r and Z(lang, 'Hint', '提示') in r and vis(pg, '#ckp-ch-res [data-act="reveal"]'), r[:300])
    check(f'{tag} challenge (R1): before the reveal no list of plans', not vis(pg, '#ckp-ch-res .ckp-sum') and not vis(pg, '#ckp-ch-res .cmp'))
    ct = pg.evaluate("()=>[...document.querySelectorAll('#ckp-chart svg text')].map(t=>t.textContent).join(' ')")
    check(f'{tag} challenge (R15): before the reveal the label of the lowest point says it looks at time alone, and no pale range or feasible mark gives the answer away', Z(lang, 'by waste alone:', '只看时间浪费：') in ct and Z(lang, 'least within the limits', '本题可行最省') not in ct and not pg.query_selector('#ckp-chart rect.ckp-nofit'), ct)
    click(pg, '#ckp-ch-res [data-act="reveal"]')
    sm = pg.evaluate("""()=>{const s=document.querySelector('#ckp-ch-res .ckp-sum'); if(!s) return null; const rows=[...s.querySelectorAll('.ckp-plan')];
      return {roles:rows.map(r=>r.dataset.role), on:[...s.querySelectorAll('.ckp-plan[data-role="me"] .ckp-chip.on')].length, bad:[...s.querySelectorAll('.ckp-plan[data-role="me"] .ckp-plan-m .bad')].map(b=>b.textContent),
        meRed:s.querySelector('.ckp-plan[data-role="me"]').classList.contains('bad'), open:s.querySelector('details.ckp-all').open, sum:s.querySelector('details.ckp-all summary').textContent, cmp:!!document.querySelector('#ckp-ch-res .cmp'), gap:!!s.querySelector('.ckp-gap'), next:!!s.querySelector('.ckp-next')}}""")
    npass = hook(pg, "h.chAll().filter(e=>e.ok).length")
    check(f'{tag} challenge (R1): the start next to the least-waste plan; all 3 settings differ and are marked; its waste in red; one gap sentence, one change sentence; the passing plans ({npass}) behind a closed disclosure, no chart of all plans',
          sm and sm['roles'] == ['me', 'best'] and sm['on'] == 3 and sm['meRed'] and len(sm['bad']) == 1 and '25.4%' in sm['bad'][0] and sm['gap'] and sm['next'] and not sm['open']
          and sm['sum'] == Z(lang, f'See all settings that pass ({npass})', f'看全部可行的设置（{npass} 种）') and not sm['cmp'] and npass == 3, sm)
    click(pg, '#ckp-rail .seg-ctl[data-key="chm"] button[data-v="async"]')
    click(pg, '#ckp-rail .seg-ctl[data-key="chT"] button[data-v="20"]')
    click(pg, '#ckp-rail .seg-ctl[data-key="chk"] button[data-v="4"]')
    r = text(pg, '#ckp-ch-res')
    check(f'{tag} challenge tier 2 (async, 20 min, keep 4): goal met, but a better plan', Z(lang, 'Goal met', '达成目标') in r and Z(lang, 'wastes less', '浪费更少') in r, r[:300])
    click(pg, '#ckp-rail .seg-ctl[data-key="chT"] button[data-v="15"]')
    click(pg, '#ckp-rail .seg-ctl[data-key="chk"] button[data-v="5"]')
    r = text(pg, '#ckp-ch-res')
    check(f'{tag} challenge tier 1 (async, 15 min, keep 5): goal met, the least waste', Z(lang, 'no plan within the limits wastes less', '这已经是浪费最少的方案') in r and '10.8%' in r, r[:300])
    cg = pg.evaluate("""()=>{const s=document.querySelector('#ckp-chart svg'), r=s.querySelector('rect.ckp-nofit'), l=s.querySelector('line.ckp-best-l'), b=s.querySelector('circle.ckp-best');
      return {t:[...s.querySelectorAll('text')].map(t=>t.textContent).join(' '), pale:r?+r.getAttribute('x')+(+r.getAttribute('width')):null, line:l?+l.getAttribute('x1'):null, dot:b?+b.getAttribute('cx'):null,
        leg:[...document.querySelectorAll('#ckp-leg span')].map(x=>x.textContent).join(' | '), why:(document.querySelector('#ckp-ch-res .ckp-why')||{}).textContent||''};}""")
    lm, bb = hook(pg, "h.chLimit()"), hook(pg, "h.chBest()")
    check(f'{tag} challenge (R15): the curve says 7.7 min by waste alone and 15 min the least within the limits; the pale range ends at the 15 min line; the legend and the explanation name both',
          lm['kf'] == 5 and lm['tmin'] == 15 and bb['T'] == 15 and bb['mode'] == 'async'
          and Z(lang, 'by waste alone: 7.7 min', '只看时间浪费：7.7 分钟') in cg['t'] and Z(lang, 'least within the limits: 15 min', '本题可行最省：15 分钟') in cg['t']
          and cg['pale'] is not None and cg['line'] is not None and abs(cg['pale'] - cg['line']) < 0.6 and abs(cg['dot'] - cg['line']) < 0.6
          and Z(lang, 'pale: breaks the storage or rollback limit', '淡色：达不到存储或回退的要求') in cg['leg'] and Z(lang, 'green dot: least within the limits', '绿点：本题可行最省') in cg['leg']
          and Z(lang, 'by waste alone: 7.7 min', '只看时间浪费：7.7 分钟') in cg['why'] and Z(lang, 'the green dot marks 15 min', '绿点标的 15 分钟') in cg['why'], cg)
    sm = pg.evaluate("()=>{const s=document.querySelector('#ckp-ch-res .ckp-sum'); return s?{roles:[...s.querySelectorAll('.ckp-plan')].map(r=>r.dataset.role), gap:!!s.querySelector('.ckp-gap'), next:!!s.querySelector('.ckp-next')}:null}")
    check(f'{tag} challenge (R1): the best plan is one row, your plan, with no gap to explain', sm and sm['roles'] == ['both'] and not sm['gap'] and not sm['next'], sm)
    pg.evaluate("()=>{const d=document.querySelector('#ckp-ch-res details.ckp-all'); d.open=true; d.dispatchEvent(new Event('toggle'))}"); pg.wait_for_timeout(200)
    rows = pg.evaluate("()=>[...document.querySelectorAll('#ckp-ch-res details.ckp-all li')].map(l=>({ok:l.dataset.ok==='1', me:l.classList.contains('me'), t:l.textContent}))")
    want = hook(pg, "h.chAll().filter(e=>e.ok).sort((a,b)=>a.w-b.w||a.k-b.k||a.T-b.T).map(e=>[e.T,e.k])")
    check(f'{tag} challenge (R1): the open list holds only the passing plans, least waste first, yours marked', len(rows) == len(want) == 3 and all(r['ok'] for r in rows)
          and all(Z(lang, f'every {T} min, keep {k}', f'每 {T} 分钟一次，留 {k} 份') in r['t'] for r, (T, k) in zip(rows, want)) and rows[0]['me'] and sum(r['me'] for r in rows) == 1, (rows, want))
    check(f'{tag} challenge: the capability card', vis(pg, '#ckp-rail [data-exits]'))
    pg.evaluate("()=>document.getElementById('ckp-ch-res').scrollIntoView({block:'start'})"); pg.wait_for_timeout(300)
    every(pg, f'{tag} challenge solved', lang, f'{tag}_challenge_solved')
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} challenge: a language switch keeps the plan, the result and the open list', (st['chS']['mode'], st['chS']['T'], st['chS']['k']) == ('async', 15, 5) and st['chSubmitted'] and vis(pg, '#ckp-ch-res .ckp-sum') and vis(pg, '#ckp-ch-res details.ckp-all li'), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    mode(pg, 'sandbox')
    st = state(pg)
    check(f'{tag} R3.20+: back from the challenge, the watched scenario shows its end at once', st['mode'] == 'sandbox' and not st['playing'] and st['playT'] >= 1, st)
    mode(pg, 'guided')
    check(f'{tag} back to Guided', state(pg)['mode'] == 'guided')
    every(pg, f'{tag} guided again', lang)
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    for (w, h) in SIZES:
        for lang in LANGS:
            if 'guided' in ONLY: run_guided(b, w, h, lang)
            if 'modes' in ONLY: run_modes(b, w, h, lang)
    b.close()
for hit in rule_hits[:60]:
    check('writing rules: ' + hit, False)
for hit in fit_hits[:60]:
    check(hit, False)
check('no page errors', not errs, errs[:5])
# ---- G1/G2 (the owner's message of 2026-10-08): option lead-ins instead of “one at a time”, and no sentence that leans on a detail
# of an earlier part. Read from the lab's own source, so both languages are covered.
import re as _re
_SRC = ''.join((P / d / 'checkpointing.js').read_text() for d in ('labs', 'teasers') if (P / d / 'checkpointing.js').exists())
_G1 = _re.findall(r'.{0,30}(?:一次只能选|不会叠加|do not stack|[Oo]ne at a time|一次选一).{0,30}', _SRC)
check('G1: no “one at a time / do not stack” note left in the lab or its tile (EN and ZH)', not _G1, _G1[:3])
_NEW = ["A checkpoint that resumes exactly holds four more things. Add them back, several at once if you like, and watch the dashed line.", "一份能原样接上的存档，除了权重还有四样。把它们加回存档，可以同时加几样，看看虚线怎么变。", "written, at 60%. Pick a way to save and see how many steps are redone after the crash. The second way relies on", "存档写到 60% 的时候。选一种存法，看看崩了以后要重算多少步。第二种存法靠的是“", "and you notice the spike at step 520. Pick what to do and see the loss at step 600.", "你在第 520 步才发现突刺。选一种办法，看看第 600 步的 loss 是多少。", "Shows the run resumed with the weights only.", "在图里显示只用权重接着训的结果。", "Shows the atomic save after the crash.", "在图里显示原子存档在故障之后的样子。", "Shows the 70B’s wasted time against the save interval.", "在图里显示 70B 浪费的时间随存档间隔怎么变。", "Save the whole training state, not just the weights.", "With the weights only, AdamW starts again from empty m and v and the learning rate warms up again, so the resumed run ends worse. With the optimizer state, the scheduler step, the data position and the RNG state as well, the resumed run matches step for step.", "Write the new checkpoint completely before you mark it as the latest.", "Overwrite the only checkpoint, and a crash halfway through the save leaves nothing usable. Write a temporary file, rename it, and update “latest” last: a complete checkpoint is then always on disk.", "Pick the interval from the save time and the failure rate.", "The longer each save stops training, the less often to save. The more often failures come, the more often to save. An asynchronous save stops training only briefly, so it can save far more often.", "without them the resumed run is a different run, and often a worse one.", "✗ “The more often you save, the safer.” Every synchronous save stops training. Save too often, and the stops add up to more than the failures cost.", "存整个训练状态，不只是权重。", "只存权重时，AdamW 要从空的 m 和 v 重新开始，学习率也会重新预热，接着训的 loss 就变差了。把优化器状态、学习率走到的步数、数据位置和随机数状态也存上，接着训的每一步都和没崩时一样。", "新存档完整写好，再把它标成“最新”。", "直接覆盖唯一的存档，存到一半崩了，就一份能用的存档都不剩。先写临时文件，再改名，最后才更新 latest，磁盘上就总有一份完整的存档。", "按存档要停多久、多久出一次故障来定间隔。", "每次存档停得越久，就该存得越少。故障来得越勤，就该存得越勤。异步存档只停很短的时间，所以可以存得勤得多。", "少了它们，接着训就成了另一次训练，往往还更差。", "✗ “存得越勤越安全。”每次同步存档都要停下训练。存得太勤，停下来的时间加起来，比故障造成的浪费还多。", "A 70B run whose synchronous saves stop training for", "s and that fails every 12 hours on average wastes least by saving about every", "一次 70B 的训练，每次同步存档停", "秒，平均每 12 小时出一次故障，大约每"]
check(f'G1/G2: the new lead-ins and the rewritten sentences are in the lab ({len(_NEW)})', all(x in _SRC for x in _NEW), [x for x in _NEW if x not in _SRC][:3])
_OLD = ["A checkpoint that resumes exactly holds four more things. Add them back one at a time and watch the dashed line. They stack", "一份能原样接上的存档，除了权重还有四样。一样一样加回去，看虚线怎么变（可以叠加）：", "written, at 60%. Two ways to save; one at a time, they do not stack. The second relies on", "存档写到 60% 的时候。下面是两种存法，一次只能选一个，不会叠加。第二种靠的是“", "and you notice the spike at step 520. One at a time; they do not stack", "你在第 520 步才发现突刺。下面一次只能选一个，不会叠加：", "Shows part 2’s resume with the weights only.", "在图里显示第 2 部分只用权重接着训的结果。", "Shows part 3’s atomic save after the crash.", "在图里显示第 3 部分原子存档在故障之后的样子。", "Shows part 4’s wasted time against the interval.", "在图里显示第 4 部分浪费的时间随间隔怎么变。", "In part 2, resuming with the weights only ended at a loss of", "In part 3, a crash during the save cost", "In part 4, the 70B on 1,024 GPUs wastes least by saving about every", "without them the resumed run is a different run, and in part 2 a worse one.", "✗ “The more often you save, the safer.” Every synchronous save stops training; in part 4, saving every 5 minutes wastes more than saving every 3 hours.", "第 2 部分里，只用权重接着训，第 600 步的 loss 是", "第 3 部分里，存档时崩了：直接覆盖的要重算", "第 4 部分里，1,024 张卡训练 70B：每次存档停", "少了它们，接着训就成了另一次训练，第 2 部分里还是更差的一次。", "✗ “存得越勤越安全。”每次同步存档都要停下训练；第 4 部分里，每 5 分钟存一次，比每 3 小时存一次浪费得还多。", "In part 4, the 70B whose synchronous saves stop training for", "第 4 部分里，70B 每次同步存档停"]
check(f'G2: none of the {len(_OLD)} sentences that leaned on an earlier part (or said “one at a time”) is left', not any(x in _SRC for x in _OLD), [x for x in _OLD if x in _SRC][:3])
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:60] if fails else 'none')
sys.exit(1 if fails else 0)
