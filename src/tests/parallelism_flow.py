"""Flow test for the parallelism lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH:
- no page errors, no horizontal overflow, no text wider than its box (R7.8), a tooltip for every help cursor (R1.1);
- part 1: the prediction gate with the reason under a greyed-out Next; the columns, the timeline and the memory stay hidden before
  the pick (R3.8), “Saved” after it;
- part 2: the step auto-plays on entering at Medium (R3.9: a mechanism replay), Next waits for its end, the result says 148.4 GB with
  the verdict (numbers as R6.18 writes them; the owner's clarification of 2026-10-07 23:37 UTC withdrew R2); part 3: ZeRO-1 replays (a new timeline); part 4: ZeRO-2 does not replay (R3.20: the same timeline as ZeRO-1), the
  replay control replays it on demand; part 5: fetch-when-needed shows red waiting gaps, prefetch replays and removes them;
  part 6: the levers, Next after two; part 7: three points, three misconceptions, See the scenarios;
- R3.20: going back to a part whose timeline was already played shows it finished, not replaying; a language switch does not replay;
- R3.8: the challenge's answer (8,192 tokens per GPU) appears nowhere before the challenge;
- Scenarios: four, single choice, the first pressed on entry, the URL carries the setup and a fresh page restores it;
- the challenge (the owner's review of 2026-10-07, R1): Settings locked, the three feedback tiers; “Show the best setting” shows your
  setting beside the best one (the differing settings highlighted, what fails in red, the main gap and the one change that helps most),
  never all 24 combinations; the 2 settings that pass sit behind a disclosure closed by default; the capability card, a language
  switch keeps the plan; the dock's mini bar at 390;
- the owner's review of 2026-10-07: (1) the caption above the legend keeps its sentence for a whole pass while only its numbers change
  (the sentence node is not replaced within a phase, the box height never changes, the transfer line never blinks off), and every
  caption sentence fits its fixed box; (2) every formula has a name on its left, and the memory formulas say activations are not in
  them; (3) the four metric cards sit in one row at 1440, 1280 and 900, in two rows of two at 720 and 390; (5) the header names the
  lab “ZeRO 与 FSDP”, ZeRO first; the owner's clarification of 23:37 UTC: no hover outside Show the math only gives a number's exact value
  (no “精确值是” / “Exactly”, no hover on the challenge result's numbers or on the kit's big numbers);
- the writing rules on every visible text node and tooltip, colours (blue = weights, red = does not fit / waiting).
Screenshots: shots/parallelism/.
Run: python3 tests/parallelism_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, subprocess, sys, time
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/parallelism.html', '--only', 'parallelism'], check=True, cwd=P, stdout=subprocess.DEVNULL)
OUT = P / 'out' / 'parallelism'; OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'wrapped-flow.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'parallelism.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-flow.html')
SH = P / 'shots' / 'parallelism'; SH.mkdir(parents=True, exist_ok=True)
ARGS = sys.argv[1:]
LANGS = [x for x in ('en', 'zh') if x in ARGS] or ['en', 'zh']
SIZES = [s for s in [(1440, 900), (390, 844)] if str(s[0]) in ARGS] or [(1440, 900), (390, 844)]
FULL = not ARGS
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
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #zr-mathbody, .num-big, #zr-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/zr-b-w|zr-sw-w|zr-e-AG|zr-sw-AG|zr-b-inf|zr-sw-inf/,[amber]:/^$/,[red]:/status bad|badline|badc|zr-ovr|zr-wait|zr-over|zr-bad|zr-clip|zr-col over|zr-sw-ovr|zr-sw-wait|zr-now mono bad|zr-dk|tz-zr-over|warn|wrong|ex-res/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#zr-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#zr-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#zr-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
EXACT = """()=>{const out=[]; for(const e of document.querySelectorAll('#zr-app [data-tipx], #zr-app [data-tip], #zr-app [title], #dock [data-tipx], #dock [data-tip], #dock [title], #tip')){
  if(e.closest('#zr-mathbody')) continue; const h=[e.getAttribute('data-tipx'),e.getAttribute('data-tip'),e.getAttribute('title'),e.id==='tip'?e.textContent:''].join(' ');
  if(/精确值是|\\bExactly\\b/.test(h)) out.push((e.id||e.className||e.tagName)+': '+h.trim().slice(0,60));}
  for(const e of document.querySelectorAll('#zr-app .zr-sv, #zr-app .num-big, #dock .num-big')){if(e.matches('.def,[tabindex],[data-tipx],[data-tip]')||e.closest('.def,[data-tipx]')) out.push('hover on a number: '+e.textContent.trim());}
  return out.slice(0,8);}"""
CAPS = """async(ms)=>{const now=document.getElementById('zr-now'); const out=[]; const t0=performance.now(); let node=null, nid=0, nnode=null, nnid=0;
  while(performance.now()-t0<ms){await new Promise(r=>requestAnimationFrame(r)); const c=now.querySelector('.zr-nl-c'), m=now.querySelector('.zr-nl-n');
    if(c!==node){node=c; nid++;} if(m!==nnode){nnode=m; nnid++;}
    const w=now.querySelector('.zr-waitn [data-n="w"]');
    out.push({ph:now.dataset.phase, c:nid, n:nnid, h:Math.round(now.getBoundingClientRect().height*10)/10, over:now.scrollHeight-now.clientHeight,
      t:(now.querySelector('[data-n="t"]')||{}).textContent, m:m?m.textContent.trim():'', w:w?parseFloat(w.textContent):null,
      wred:w?getComputedStyle(w.parentElement).color:''});}
  return out;}"""
RED = "()=>{const d=document.createElement('i'); d.style.color=getComputedStyle(document.documentElement).getPropertyValue('--c-oom').trim(); document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;}"


def caption_checks(pg, where, samples, want):
    """the owner's review of 2026-10-07, item 1: the sentence stays for a whole phase; only the numbers change"""
    phases = [x['ph'] for x in samples]
    per = {}
    for x in samples: per.setdefault(x['ph'], set()).add((x['c'], x['n']))
    check(f'{where}: the caption sentence is not replaced while one phase lasts', all(len(v) == 1 for v in per.values()), per)
    check(f'{where}: the caption covers the phases {want}', all(ph in phases for ph in want), sorted(set(phases)))
    check(f'{where}: the caption box keeps one height during playback (R7.9)', len({x['h'] for x in samples}) == 1, sorted({x['h'] for x in samples}))
    check(f'{where}: nothing in the caption is cut (R7.9)', max(x['over'] for x in samples) <= 1, max(x['over'] for x in samples))
    check(f'{where}: the numbers in it move', len({x['t'] for x in samples}) >= 8, len({x['t'] for x in samples}))
    check(f'{where}: the transfer line never blinks off', all(x['m'] for x in samples if x['ph'] in ('F', 'B')), [x for x in samples if not x['m']][:2])


rule_hits, fit_hits = [], []


def page(b, w, h, lang, dark=False, hash_='parallelism'):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light')
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    # the home tile lives in the training world of the tree: open it and run the writing rules on it and its help text (R5.7)
    pg.evaluate("()=>{const b=document.querySelector('button[data-world=\"training\"]'); if(b) b.click()}"); pg.wait_for_timeout(300)
    check(f'{w} {lang}: home tile rendered', pg.evaluate("()=>!!document.querySelector('.tz-zr svg')"))
    for hit in pg.evaluate(RULES, '.tz-zr'): rule_hits.append(f'{w} {lang} home tile: {hit}')
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__zr.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#zr-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#zr-rail [data-next-why]')].filter(e=>!e.hidden&&e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden').map(e=>e.textContent)")
obs = lambda pg: pg.evaluate("()=>{const o=document.querySelector('#zr-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' '):''}")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__zr||!window.__zr.state().stageBusy", timeout=4000)
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
    pg.evaluate(f"()=>document.querySelector('#zr-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def rules(pg, where, lang):
    for sel in ('#zr-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def fit(pg, where):
    for h in pg.evaluate(OVER): fit_hits.append(f'{where}: text wider than its box (R7.8): {h}')
    for h in pg.evaluate(HELP): fit_hits.append(f'{where}: help cursor without a tooltip (R1.1): {h}')
    for h in pg.evaluate(EXACT): fit_hits.append(f'{where}: a hover that only gives a number’s exact value (the owner, 2026-10-07 23:37 UTC): {h}')


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
    fm = pg.evaluate(FORMULA_H1, '#zr-app')
    check(f'{where}: formulas typeset as MathML, no TeX word shown, no plain-text formula left (H1, R6.29)', not fm['bad'] and not fm['txt'], fm)


def every(pg, where, lang, full_shot=None):
    formulas_h1(pg, where)
    check(f'{where}: no overflow', overflow(pg) <= 0, overflow(pg))
    nf = pg.evaluate("()=>window.__zr.nowFit()")
    if nf is not None: check(f'{where}: every caption sentence fits its fixed box', not nf, nf)
    rules(pg, where, lang); fit(pg, where)
    if full_shot: shot(pg, full_shot, full=pg.viewport_size['width'] <= 760)


def before_answer(pg):
    return pg.evaluate("""()=>{const t=[]; for(const s of ['#zr-app .labhead','#zr-rail','#zr-stage','#zr-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#zr-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")


def nav(pg, what):
    click(pg, f'#zr-rail [data-nav="{what}"]', 300)




def wait_play(pg, timeout=12000):
    try: pg.wait_for_function("()=>!window.__zr.state().playing", timeout=timeout)
    except Exception: pass
    pg.wait_for_timeout(150)


def fast_finish(pg):
    click(pg, '#zr-spd button[data-v="fast"]', 100); wait_play(pg)


def rail_text(pg):
    return pg.evaluate("()=>{const r=document.querySelector('#zr-rail'); const t=[r.innerText]; r.querySelectorAll('[data-tipx]').forEach(e=>t.push(e.getAttribute('data-tipx'))); return t.join('\\n')}")


def run_guided(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    check(f'{tag} title', text(pg, '#zr-app .labtitle') == Z(lang, 'How do you train a model that does not fit on one GPU?', '一张卡放不下的模型，怎么训练？'), text(pg, '#zr-app .labtitle'))
    check(f'{tag} no fidelity tag on top (owner, 2026-10-11 19:28 UTC)', not any(w in text(pg, '#zr-app .labmeta') for w in ('Fidelity', '保真度')), text(pg, '#zr-app .labmeta'))
    hd = text(pg, '#zr-app .labhead')
    check(f'{tag} R4: the lab is named ZeRO first, then FSDP', Z(lang, 'ZeRO and FSDP', 'ZeRO 与 FSDP') in hd and Z(lang, 'FSDP and ZeRO', 'FSDP 与 ZeRO') not in hd, hd[:300])
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#zr-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        pg.evaluate(f"()=>document.querySelector('#zr-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(200)
        check(f'{tag} popover {pid} opens', pg.evaluate(f"()=>{{const p=document.getElementById('{pid}'); return !!p&&p.getClientRects().length>0}}"))
        rules(pg, f'{tag} popover {pid}', lang)
        pg.evaluate(f"()=>document.querySelector('#zr-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(150)
    seen_rail = []
    # ---- part 1
    st = state(pg)
    check(f'{tag} p1 starts guided at part 1', st['mode'] == 'guided' and st['step'] == 0, st)
    check(f'{tag} p1 Next disabled before the pick', nexts(pg) and all(nexts(pg)))
    check(f'{tag} p1 reason under Next', any(Z(lang, 'Pick an answer', '选一个答案') in x for x in why(pg)), why(pg))
    check(f'{tag} p1 timeline hidden, memory hidden', not vis(pg, '#zr-tlbox') and pg.evaluate("()=>document.querySelectorAll('#zr-cols .zr-blk').length") == 0)
    ba = before_answer(pg)
    check(f'{tag} p1 the answer is not shown before the pick', '146.3' not in ba and '144.5' not in ba, '')
    every(pg, f'{tag} p1', lang, f'{tag}_p1')
    seen_rail.append(rail_text(pg))
    click(pg, '#zr-rail [data-pred="q18"]')
    check(f'{tag} p1 saved', Z(lang, 'Saved', '已记下') in text(pg, '#zr-rail'))
    check(f'{tag} p1 Next enabled after the pick', not any(nexts(pg)))
    nav(pg, 'next')
    # ---- part 2: auto-plays at Medium
    st = state(pg)
    check(f'{tag} p2 auto-plays on entering (R3.9)', st['playing'] and st['spd'] == 'mid', st)
    check(f'{tag} p2 Next waits for the end', all(nexts(pg)))
    pg.wait_for_timeout(600)
    shot(pg, f'{tag}_p2_playing')
    fast_finish(pg)
    st = state(pg)
    check(f'{tag} p2 finished, watched', not st['playing'] and len(st['watched']) == 1, st)
    o = obs(pg)
    check(f'{tag} p2 result 148.4 GB', '148.4 GB' in o and '144.5 GB' in o, o[:200])
    tops = pg.evaluate("()=>[...document.querySelectorAll('#zr-metrics .metric')].map(e=>Math.round(e.getBoundingClientRect().top))")
    check(f'{tag} item 3: the four cards, {"one row" if w > 760 else "two rows of two"}', len(tops) == 4 and (len(set(tops)) == 1 if w > 760 else (tops[0] == tops[1] and tops[2] == tops[3] and tops[0] < tops[2])), tops)
    check(f'{tag} item 3: nothing cut in the cards', pg.evaluate("()=>[...document.querySelectorAll('#zr-metrics .metric, #zr-metrics .metric .v')].every(e=>e.scrollWidth<=e.clientWidth+1)"))
    if w > 760:
        for vw in (1280, 900, 720):
            pg.set_viewport_size({'width': vw, 'height': h}); pg.wait_for_timeout(250)
            tops = pg.evaluate("()=>[...document.querySelectorAll('#zr-metrics .metric')].map(e=>Math.round(e.getBoundingClientRect().top))")
            check(f'{tag} item 3 at {vw}: the four cards, {"one row" if vw > 760 else "two rows of two"}', len(tops) == 4 and (len(set(tops)) == 1 if vw > 760 else (tops[0] == tops[1] and tops[2] == tops[3] and tops[0] < tops[2])), tops)
            check(f'{tag} item 3 at {vw}: nothing cut in the cards, no overflow', overflow(pg) <= 0 and pg.evaluate("()=>[...document.querySelectorAll('#zr-metrics .metric, #zr-metrics .metric .v')].every(e=>e.scrollWidth<=e.clientWidth+1)"))
            if vw == 900: shot(pg, f'{tag}_p2_cards_900')
        pg.set_viewport_size({'width': w, 'height': h}); pg.wait_for_timeout(250)
    check(f'{tag} p2 verdict with the right answer', Z(lang, 'The right answer is', '正确答案是') in o, o[:300])
    check(f'{tag} p2 replay control', Z(lang, 'Replay', '重放') in text(pg, '#zr-player'), text(pg, '#zr-player'))
    check(f'{tag} p2 the over-80 GB part is red', pg.evaluate("()=>document.querySelectorAll('#zr-cols .zr-ovr').length") == 8)
    every(pg, f'{tag} p2', lang, f'{tag}_p2')
    seen_rail.append(rail_text(pg))
    nav(pg, 'next')
    # ---- part 3: ZeRO-1 replays (new timeline)
    st = state(pg)
    check(f'{tag} p3 entering with the watched DP timeline does not replay (R3.20)', not st['playing'], st)
    check(f'{tag} p3 Next waits', all(nexts(pg)))
    click(pg, '#zr-spd button[data-v="mid"]', 50)
    click(pg, '#zr-rail .seg-ctl[data-key="st13"] button[data-v="z1"]', 0)
    caps = pg.evaluate(f"({CAPS})(3600)")
    caption_checks(pg, f'{tag} p3 ZeRO-1 at Medium', caps, ['F', 'B'])
    check(f'{tag} p3 the transfer line names the transfer of its pass', all('all-gather' in x['m'].lower() for x in caps if x['ph'] == 'F') and all('reduce-scatter' in x['m'].lower() for x in caps if x['ph'] == 'B'), [x['m'] for x in caps[::40]])
    st = state(pg)
    check(f'{tag} p3 ZeRO-1 replays: a new timeline', st['playing'] or st['pT'] < st['step_s'], st)
    o = obs(pg)
    check(f'{tag} p3 result 64.1 GB, formula on its own line', '64.1 GB' in o and vis(pg, '#zr-rail .zr-fx .katex, #zr-rail .zr-fx math, #zr-rail .zr-fx'), o[:200])
    fx = pg.evaluate("()=>[...document.querySelectorAll('#zr-rail .zr-fx')].map(e=>e.textContent)")
    check(f'{tag} p3 item 2: the formula has its name on the left', fx and all(Z(lang, 'bytes of model states per GPU', '每张卡存模型状态的字节数') in f.split('=')[0] for f in fx), fx)
    # H1 (the owner's message of 2026-10-09): 12N/d is now a typeset fraction (MathML), so its text no longer carries the slash
    frac12 = pg.evaluate("()=>[...document.querySelectorAll('#zr-rail [data-obs] p math')].some(m=>m.querySelector('mfrac')&&m.textContent.replace(/\\s/g,'')==='12Nd')")
    check(f'{tag} p3 item 2: what each term is, activations not in it', Z(lang, 'Activations are not in it', '激活不算在里面') in o and Z(lang, 'is the optimizer state', '是优化器状态') in o and frac12, o[:600])
    fast_finish(pg)
    every(pg, f'{tag} p3', lang, f'{tag}_p3')
    seen_rail.append(rail_text(pg))
    nav(pg, 'next')
    # ---- part 4: ZeRO-2 does not replay
    click(pg, '#zr-rail .seg-ctl[data-key="st24"] button[data-v="z2"]', 200)
    st = state(pg)
    check(f'{tag} p4 ZeRO-2 does not replay: the same timeline as ZeRO-1 (R3.20)', not st['playing'] and st['pT'] >= st['step_s'] - 1e-9, st)
    o = obs(pg)
    check(f'{tag} p4 result 36.0 GB', '36.0 GB' in o, o[:200])
    fx = pg.evaluate("()=>[...document.querySelectorAll('#zr-rail .zr-fx')].map(e=>e.textContent)")
    check(f'{tag} p4 item 2: the formula has its name on the left', fx and all(Z(lang, 'bytes of model states per GPU', '每张卡存模型状态的字节数') in f.split('=')[0] for f in fx) and Z(lang, 'Activations are not in it', '激活不算在里面') in o, fx)
    click(pg, '#zr-player [data-act="play"]', 300)
    check(f'{tag} p4 the replay control replays on demand', state(pg)['playing'])
    fast_finish(pg)
    every(pg, f'{tag} p4', lang, f'{tag}_p4')
    seen_rail.append(rail_text(pg))
    nav(pg, 'next')
    # ---- part 5: fetch when needed, then prefetch
    st = state(pg)
    check(f'{tag} p5 replays the new ZeRO-3 timeline', st['playing'] or st['pT'] < st['step_s'], st)
    caps = pg.evaluate(f"({CAPS})(1800)")
    caption_checks(pg, f'{tag} p5 fetch when needed', caps, ['F'])
    ws = [x['w'] for x in caps if x['w'] is not None]
    check(f'{tag} p5 the time waited so far is a red number that only grows (R5.11)', len(ws) == len(caps) and ws == sorted(ws) and ws[-1] > ws[0] and all(x['wred'] == pg.evaluate(RED) for x in caps), (ws[:3], ws[-3:], caps[0]['wred']))
    fast_finish(pg)
    gaps = pg.evaluate("()=>document.querySelectorAll('#zr-tl .zr-wait').length")
    check(f'{tag} p5 waiting gaps red without prefetch (R5.11)', gaps >= 32, gaps)
    o = obs(pg)
    check(f'{tag} p5 waits 15%', '15%' in o, o[:300])
    shot(pg, f'{tag}_p5_noov')
    click(pg, '#zr-rail [data-lever="ov"]', 150)
    check(f'{tag} p5 prefetch replays', state(pg)['playing'] or state(pg)['pT'] < state(pg)['step_s'])
    fast_finish(pg)
    o = obs(pg)
    check(f'{tag} p5 prefetch result', Z(lang, 'drops from 15%', '从这一步的 15% 降到') in o and '1.5' in o, o[:300])
    fx = pg.evaluate("()=>[...document.querySelectorAll('#zr-rail .zr-fx')].map(e=>e.textContent)")
    check(f'{tag} p5 item 2: the formula has its name on the left', fx and all(Z(lang, 'bytes of model states per GPU', '每张卡存模型状态的字节数') in f.split('=')[0] for f in fx) and Z(lang, 'Not in it: the activations', '不算在里面的有激活') in o, (fx, o[-200:]))
    check(f'{tag} p5 Next enabled', not any(nexts(pg)))
    every(pg, f'{tag} p5', lang, f'{tag}_p5')
    seen_rail.append(rail_text(pg))
    nav(pg, 'next')
    # ---- part 6
    fast_finish(pg)
    o = obs(pg)
    check(f'{tag} p6 ZeRO-3 over 64 GPUs, one NIC per machine, waits 45%', '45%' in o, o[:200])
    check(f'{tag} p6 Next waits for a second way', all(nexts(pg)))
    fx = pg.evaluate("()=>[...document.querySelectorAll('#zr-rail .zr-fx')].map(e=>e.textContent)")
    check(f'{tag} p6 item 2: the formula has its name on the left', fx and all(Z(lang, 'time of one all-gather', '一次 all-gather 的时间') in f.split('≈')[0] for f in fx), fx)
    click(pg, '#zr-rail [data-lever="tok"]'); fast_finish(pg)
    check(f'{tag} p6 more tokens', '16,384' in obs(pg))
    click(pg, '#zr-rail [data-lever="net"]'); fast_finish(pg)
    o = obs(pg)
    check(f'{tag} p6 a NIC per GPU result', Z(lang, 'One NIC per GPU', '每张卡配一块网卡') in o and '400 GB/s' in o, o[:300])
    check(f'{tag} p6 Next enabled', not any(nexts(pg)))
    check(f'{tag} p6 machines note', vis(pg, '#zr-cols .zr-more'))
    every(pg, f'{tag} p6', lang, f'{tag}_p6')
    seen_rail.append(rail_text(pg))
    # R3.20: back to part 5 shows its timeline finished, not replaying
    nav(pg, 'back')
    st = state(pg)
    check(f'{tag} back to part 5: no replay of a watched timeline', not st['playing'], st)
    nav(pg, 'next'); nav(pg, 'next')
    # ---- part 7
    t = text(pg, '#zr-rail')
    check(f'{tag} p7 three points and three misconceptions', pg.evaluate("()=>document.querySelectorAll('#zr-rail .zr-remlist li').length") == 3 and t.count('✗') == 3, t[:200])
    click(pg, '#zr-rail [data-rem="z3"]')
    st = state(pg)
    check(f'{tag} p7 show in the chart: the run watched in part 5 shows its end at once (R3.20)', st['st'] == 'z3' and not st['playing'] and st['pT'] >= st['step_s'] - 1e-9, st)
    click(pg, '#zr-rail [data-rem="net"]'); st = state(pg)
    check(f'{tag} p7 the run watched in part 6 shows its end at once (R3.20)', st['N'] == 64 and not st['playing'] and st['pT'] >= st['step_s'] - 1e-9, st)
    check(f'{tag} p7 the replay control is called Replay', Z(lang, '▶ Replay', '▶ 重放') in text(pg, '#zr-player'), text(pg, '#zr-player'))
    pg.evaluate("()=>{document.querySelector('#zr-math').open=true}"); pg.wait_for_timeout(150)
    fx = pg.evaluate("()=>[...document.querySelectorAll('#zr-mathbody .zr-fx')].map(e=>e.textContent)")
    import re as _re
    check(f'{tag} p7 item 2: every formula in “Show the math” has a name on its left', len(fx) >= 6 and all(len(_re.findall(r'[A-Za-z\u4e00-\u9fff]', _re.split('[=≈]', f)[0])) >= 4 for f in fx), fx)
    pg.evaluate("()=>{document.querySelector('#zr-math').open=false}"); pg.wait_for_timeout(100)
    every(pg, f'{tag} p7', lang, f'{tag}_p7')
    seen_rail.append(rail_text(pg))
    check(f'{tag} R3.8: the challenge answer (8,192 tokens) appears nowhere before the challenge', not any('8,192' in x for x in seen_rail))
    # a language switch keeps the part and does not replay
    other = 'en' if lang == 'zh' else 'zh'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} language switch keeps the part, no replay', st['step'] == 6 and not st['playing'], st)
    pg.close()


def run_modes(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    mode(pg, 'sandbox')
    pressed = pg.evaluate("()=>[...document.querySelectorAll('#zr-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    lead = pg.inner_text('#zr-rail .railcard > p.small')
    check(f'{tag} G1: the scenarios card leads in with what to pick and what to watch, not “one at a time”', lead == Z(lang, 'Pick a scenario and see how much memory each GPU needs and how much of the time it waits for data.', '选一个场景，看看每张卡要放多少显存、有多少时间在等数据。'), lead)
    check(f'{tag} scenarios: four, the first pressed', pressed == ['true', 'false', 'false', 'false'], pressed)
    for i in range(4):
        click(pg, f'#zr-rail [data-preset="{i}"]', 350); fast_finish(pg)
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#zr-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        check(f'{tag} scenario {i + 1} single choice', pressed.count('true') == 1 and pressed[i] == 'true', pressed)
        check(f'{tag} scenario {i + 1} insight', len(text(pg, '#zr-insight')) > 40)
        every(pg, f'{tag} scenario {i + 1}', lang, f'{tag}_s{i + 1}')
    check(f'{tag} scenario 4 Ethernet waits 79%', '79%' in text(pg, '#zr-insight'))
    hash_ = pg.evaluate("()=>location.hash")
    pg2 = b.new_page(viewport={'width': w, 'height': h}); open_drawers(pg2)
    pg2.goto(URL + hash_); pg2.wait_for_timeout(700)
    st2 = pg2.evaluate("()=>window.__zr&&window.__zr.state()")
    check(f'{tag} the URL restores the setup', st2 and st2['mode'] == 'sandbox' and st2['N'] == 32 and st2['net'] == 'eth', (hash_, st2))
    pg2.close()
    click(pg, '#zr-rail [data-preset="0"]', 250); st = state(pg)
    check(f'{tag} back to scenario 1: no replay of a watched run (R3.20)', not st['playing'] and st['pT'] >= st['step_s'] - 1e-9, st)
    # ---- challenge
    mode(pg, 'challenge')
    check(f'{tag} challenge: Settings locked', pg.evaluate("()=>document.querySelector('#zr-controls').classList.contains('locked')"))
    st = state(pg)
    check(f'{tag} challenge starts at ZeRO-2, 4,096', st['m'] == '70b' and st['N'] == 64 and st['st'] == 'z2' and st['T'] == 4096, st)
    check(f'{tag} challenge: its new timeline plays', st['playing'] or st['pT'] < st['step_s'], st)
    fast_finish(pg)
    mode(pg, 'sandbox'); mode(pg, 'challenge'); st = state(pg)
    check(f'{tag} back in the challenge: the watched run shows its end at once (R3.20)', not st['playing'] and st['pT'] >= st['step_s'] - 1e-9, st)
    every(pg, f'{tag} challenge', lang, f'{tag}_ch')
    click(pg, '#zr-rail [data-act="submit"]')
    r = text(pg, '#zr-ch-res')
    check(f'{tag} tier 3: not yet, does not fit, about 168 GB (R1: rounded)', Z(lang, 'Not yet', '还没达到') in r and '168 GB' in r and '167.6' not in r, r[:200])
    click(pg, '#zr-rail [data-act="reveal"]')
    r = text(pg, '#zr-ch-res')
    n_rows = pg.evaluate("()=>document.querySelectorAll('#zr-ch-res .zr-c2r').length")
    check(f'{tag} R1: your setting beside the best one, never all 24', n_rows == 2 and pg.evaluate("()=>document.querySelectorAll('#zr-ch-res .cmp-row, #zr-ch-res [data-row]').length") == 0, n_rows)
    check(f'{tag} R1: the settings both share are written once, the differing ones highlighted', Z(lang, 'Both train', '两种设置都') in r and pg.evaluate("()=>document.querySelectorAll('#zr-ch-res .zr-c2r.me .zr-chip.on').length") == 2, r[:400])
    check(f'{tag} R1: what fails is red', pg.evaluate("()=>[...document.querySelectorAll('#zr-ch-res .zr-c2r.me .zr-sv.zr-bad')].length") == 1)
    check(f'{tag} R1: the main gap and the one change that helps most', Z(lang, 'ZeRO-2 still keeps the full BF16 weights', 'ZeRO-2 每张卡还存着一整份 BF16 权重') in r and Z(lang, 'No single change meets the goal', '只改一项还达不到目标') in r, r[-500:])
    dd = pg.evaluate("()=>{const d=document.querySelector('#zr-ch-res details.zr-all'); return d?[d.open,d.querySelector('summary').textContent]:null}")
    check(f'{tag} R1: the settings that pass behind a closed disclosure', dd and dd[0] is False and Z(lang, 'See all settings that pass (2)', '看全部可行的设置（2 种）') in dd[1], dd)
    click(pg, '#zr-ch-res details.zr-all summary', 150)
    check(f'{tag} R1: it lists the 2 settings that pass, and only those', pg.evaluate("()=>document.querySelectorAll('#zr-ch-res .zr-pass li').length") == 2 and '8,192' in text(pg, '#zr-ch-res .zr-pass') and '16,384' in text(pg, '#zr-ch-res .zr-pass'))
    shot(pg, f'{tag}_ch_reveal')
    click(pg, '#zr-rail .seg-ctl[data-key="chs"] button[data-v="z3"]'); click(pg, '#zr-rail .seg-ctl[data-key="cht"] button[data-v="16384"]')
    r = text(pg, '#zr-ch-res')
    check(f'{tag} tier 2: met, a shorter step exists', Z(lang, 'Goal met', '达标了') in r and '8,192' in r, r[:300])
    check(f'{tag} R1 tier 2: the one change is lowering the tokens to 8,192', Z(lang, 'lowering the tokens per GPU to 8,192', '把每张卡的 token 减到 8,192') in r, r[-300:])
    click(pg, '#zr-rail .seg-ctl[data-key="cht"] button[data-v="8192"]')
    r = text(pg, '#zr-ch-res')
    check(f'{tag} tier 1: the shortest step', Z(lang, 'no passing option has a shorter step', '这一步最短') in r, r[:300])
    check(f'{tag} R1: the best setting needs no comparison rows', pg.evaluate("()=>document.querySelectorAll('#zr-ch-res .zr-c2r').length") == 0 and pg.evaluate("()=>!!document.querySelector('#zr-ch-res details.zr-all')"))
    check(f'{tag} capability card', vis(pg, '#zr-rail [data-exits]'))
    every(pg, f'{tag} challenge result', lang, f'{tag}_ch_res')
    other = 'en' if lang == 'zh' else 'zh'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(450)
    st = state(pg)
    check(f'{tag} language switch keeps the plan and the result', st['st'] == 'z3' and st['T'] == 8192 and st['chSubmitted'], st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(450)
    if w <= 760:
        pg.evaluate("()=>window.scrollTo(0, document.querySelector('#zr-stage').getBoundingClientRect().bottom+scrollY+100)"); pg.wait_for_timeout(500)   # the chart out of view (the flow list moved the rail, 2026-10-11)
        check(f'{tag} the dock shows the mini bar', pg.evaluate("()=>{const d=document.getElementById('dock'); return !!d&&!d.hidden&&!!d.querySelector('.zr-dk')}"))
        shot(pg, f'{tag}_dock')
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    for (w, h) in SIZES:
        for lang in LANGS:
            run_guided(b, w, h, lang)
            run_modes(b, w, h, lang)
    b.close()
for hit in rule_hits[:60]:
    check('writing rules: ' + hit, False)
for hit in fit_hits[:60]:
    check(hit, False)
check('no page errors', not errs, errs[:5])
# ---- G1/G2 (the owner's message of 2026-10-08): option lead-ins instead of “one at a time”, and no sentence that leans on a detail
# of an earlier part. Read from the lab's own source, so both languages are covered.
import re as _re
_SRC = ''.join((P / d / 'parallelism.js').read_text() for d in ('labs', 'teasers') if (P / d / 'parallelism.js').exists())
_G1 = _re.findall(r'.{0,30}(?:一次只能选|不会叠加|do not stack|[Oo]ne at a time|一次选一).{0,30}', _SRC)
check('G1: no “one at a time / do not stack” note left in the lab or its tile (EN and ZH)', not _G1, _G1[:3])
_NEW = ["When should a layer’s weights be fetched? Pick one and see how much of the time the GPU waits for data.", "什么时候去取这一层的权重？选一种取法，看看 GPU 有多少时间在等数据。", "one 400 Gb/s NIC each, 50 GB/s. Pick one of the three ways and see how much of the time the GPU waits for data. The first is the setup now; the other two are ways to improve it.", "也就是 50 GB/s。选一种做法，看看 GPU 有多少时间在等数据。第一种就是现在这样，另外两种是改进的办法。", "Shows data parallelism on 8 GPUs.", "在图里显示 8 张卡的数据并行。", "Shows ZeRO-3 with prefetching.", "在图里显示带预取的 ZeRO-3。", "Shows ZeRO-3 over 64 GPUs with one NIC per machine and 4,096 tokens per GPU.", "在图里显示 64 张卡上的 ZeRO-3：每台机器一块网卡，每张卡 4,096 个 token。", "Data parallelism splits the data, not the model.", "Each GPU holds a full copy of the model states, so more GPUs process more data while each GPU needs just as much memory.", "Shard what is identical, and fetch it when needed.", "ZeRO-3 splits the weights, the gradients and the optimizer state (PyTorch’s FSDP does this by default), so with d GPUs each holds ${A.ui.math('\\\\frac{1}{d}')} of the model states.", "The extra transfers cost time only when the computation cannot cover them.", "When it cannot, give each GPU more tokens per step, or give the machines a faster network. Either way the waiting drops.", "✗ “Data parallelism makes a model fit that does not fit on one GPU.” Every GPU still keeps everything: its model states alone are as large as on a single GPU.", "✗ “ZeRO-3 saves memory for free.” It moves half again as much data. Without prefetching, the GPU waits for each layer’s weights before it can compute that layer.", "数据并行分的是数据，不是模型。", "每张卡都存着一整份模型状态。所以卡越多，能同时算的数据越多，每张卡要的显存却一点不少。", "一样的东西分开存，用的时候再取。", "ZeRO-3 把权重、梯度和优化器状态都分开存，PyTorch 的 FSDP 默认就这样做。d 张卡时，每张卡只存 ${A.ui.math('\\\\frac{1}{d}')} 的模型状态。", "多传的数据，只有计算盖不住时才花时间。", "盖不住时，可以让每张卡每步多算一些 token，也可以给机器配更快的网络。这两种办法都能让等数据的时间降下来。", "✗ “数据并行能让一张卡放不下的模型放下。”每张卡还是什么都存，光模型状态就和只用一张卡时一样多。", "✗ “ZeRO-3 省显存，不花代价。”它要多传一半的数据。不预取的话，GPU 每算一层，都要先等这一层的权重传过来。", "每张卡每步 4,096 个 token，每台机器 50 GB/s", "4,096 tokens per GPU per step; 50 GB/s per machine"]
check(f'G1/G2: the new lead-ins and the rewritten sentences are in the lab ({len(_NEW)})', all(x in _SRC for x in _NEW), [x for x in _NEW if x not in _SRC][:3])
_OLD = ["When should a layer’s weights be fetched? Two ways below; pick one to see.", "one 400 Gb/s NIC each, 50 GB/s. Three ways below, one at a time. The first is the setup now; the other two are ways to improve it.", "也就是 50 GB/s。下面有三种做法，一次选一种。第一种就是现在这样，另外两种是改进的办法。", "Shows part 2: data parallel on 8 GPUs.", "在图里显示第 2 部分：8 张卡的数据并行。", "Shows part 5: ZeRO-3 with prefetching.", "在图里显示第 5 部分：带预取的 ZeRO-3。", "Shows part 6: ZeRO-3 over 64 GPUs with one NIC per machine, 4,096 tokens per GPU.", "在图里显示第 6 部分：每台机器一块网卡、每张卡 4,096 个 token 的 ZeRO-3。", "In part 2 each of the 8 GPUs still needed", "In part 6, with one NIC per machine, ZeRO-3 across 8 machines waited", "✗ “Data parallelism makes a model fit that does not fit on one GPU.” Every GPU still keeps everything. In part 2, each GPU held", "✗ “ZeRO-3 saves memory for free.” It moves half again as much data. In part 5, without prefetching the GPU waited", "第 2 部分里，8 张卡每张还是要", "第 6 部分里，每台机器只有一块网卡，ZeRO-3 跨 8 台机器时有", "✗ “数据并行能让一张卡放不下的模型放下。”每张卡还是什么都存。第 2 部分里，每张卡光模型状态就有", "✗ “ZeRO-3 省显存，不花代价。”它要多传一半的数据。第 5 部分里，不预取时，这一步有", "和前面一样，每张卡每步 4,096 个 token", "4,096 tokens per GPU per step, as before"]
check(f'G2: none of the {len(_OLD)} sentences that leaned on an earlier part (or said “one at a time”) is left', not any(x in _SRC for x in _OLD), [x for x in _OLD if x in _SRC][:3])
print(f'{n - len(fails)}/{n} passed')
print('FAILED:', fails[:60] if fails else 'none')
sys.exit(1 if fails else 0)
