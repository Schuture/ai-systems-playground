"""Flow test for the model-parallel lab (CONTRACT §6).

Every guided part and every mode at 1440×900 and 390×844, in EN and ZH (plus one dark run when run in full):
- no page errors, no horizontal overflow, no text wider than its box (R7.8), a tooltip for every help cursor (R1.1), the writing rules
  on every visible text node and tooltip (R6.14, R6.7, Chinese spacing and punctuation, no 亿/万), colours (R5.6);
- part 1: the prediction gate with its reason under a greyed-out Next; before the pick no cell of the schedule is drawn and the answer
  (88%) appears nowhere, in text, tooltips or the header's popovers (R3.8);
- part 2: the step auto-plays on entering at Medium (R3.9; owner b8: Medium is the old Fast, 8 s) with the idle time outlined red (R5.11);
  Pause stops it; Fast finishes in about 4.5 s; at m 32 and 64 the blocks stay apart (owner b8 item 2: no block thinner than 4.5 px, a
  gap between neighbours, thin outlines on thin blocks, the “one block = g micro-batches” note); the symbols p and m with their meaning
  and the worked examples of the bubble formula (owner b8 item 5); Next waits for 32 micro-batches; a new picture plays, a picture already watched does not (R3.20), Replay plays it again;
- part 3: entered right after a change in part 2 without ▲▼ badges (R6.6), GPipe then 1F1B, both results on their options (16 and 8),
  the bubble unchanged (30%); part 4: the layer view, TP 8: 5.31 ms, 1.04 ms, 16%, the zoomed row; part 5: across machines 9 times as
  long, nearly two thirds, 9.40 ms and 64% on the cards, four red all-reduces, the mini result bar at 390; part 6: the cluster, 37.2 GB,
  2.3%, 16%, and the schedule drawn as separate blocks of 1F1B, not one dark bar (owner b8 item 4); part 7: three points, three misconceptions; Back to part 3 does not replay;
- Scenarios: four, single choice, the first pressed on entry, 169 GB / 67.0 GB / 32% / 179 GB with their status; the URL carries the
  setup and a fresh page restores it; a change in Settings releases the scenario buttons;
- the challenge: Settings locked, the start (165 GB), the three tiers (does not fit; TP 8 × PP 4 × 256 fits but is slower, 32.9 s; TP 4 × PP 4
  × 128 meets the goal, 30.2 s), the R1 summary (verdict first, your layout next to the fastest, the differing settings, the main gap
  and the best single change, the layouts that pass behind a closed disclosure), the capability card, a language switch keeps the plan;
- G7 (Codex on b9 preview v5): the task card's own m change is the one the result names (TP 4 from TP 8 × PP 4 × 256 sets m = 128);
  PP 2 × TP 8 × 64 names m = 128 with “33.4 → 33.2”, and doing it shows 33.2 s; PP 2 × TP 8 × 128 (the Codex pick) says no single change
  helps and names PP 4 and TP 4, and every single change the card allows, actually made, either does not fit or does not show a
  shorter step; a tensor-parallel option the pipeline does not allow is disabled;
- G1 (the owner, 2026-10-08): no “一次只能选 / 不会叠加 / do not stack / one at a time” anywhere (every stop, every tooltip, the source),
  and the lead-ins of parts 3 and 5 and of the scenarios; G2: the recap and the capability card lean on no earlier part's detail.
Screenshots: shots/model-parallel/.
Run: python3 tests/model-parallel_flow.py [en|zh] [1440|390] [guided|modes]  (filters for a partial run)"""
import pathlib, re, subprocess, sys, time
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/model-parallel.html', '--only', 'model-parallel'], check=True, cwd=P, stdout=subprocess.DEVNULL)
OUT = P / 'out' / 'model-parallel'; OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'wrapped-flow.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'model-parallel.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-flow.html')
SH = P / 'shots' / 'model-parallel'; SH.mkdir(parents=True, exist_ok=True)
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
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour (R5.7)'],
   [/一次只能选|一次选一个|不会叠加|do not stack|one at a time/i,'a mechanics note instead of a lead-in (G1)']];
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
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #mp-mathbody, .num-big, #mp-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/mp-st|mp-mw|mp-sw-w/,[amber]:/^$/,[red]:/status bad|badline|badc|mp-over|warn|wrong|ex-res|red|over|bad/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#mp-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#mp-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#mp-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
BLOCKS = """()=>{const out={minW:1e9,minGap:1e9,thick:0,rows:[],mix:document.querySelectorAll('#mp-gantt .mp-cm').length,note:(document.querySelector('#mp-gleg .mp-gn')||{}).textContent||''};
 document.querySelectorAll('#mp-gantt .mp-gtrk').forEach(t=>{const bs=[...t.querySelectorAll('.mp-cf,.mp-cb,.mp-cm')].map(e=>{const r=e.getBoundingClientRect(), cs=getComputedStyle(e); return {x:r.left,w:r.width,bw:parseFloat(cs.borderLeftWidth)||0};}).sort((a,b)=>a.x-b.x);
   out.rows.push(bs.length);
   bs.forEach((b,i)=>{if(i>0){const g=b.x-(bs[i-1].x+bs[i-1].w); out.minGap=Math.min(out.minGap,g);} if(b.bw>1.01&&b.w<12) out.thick++;});
   bs.slice(1,-1).forEach(b=>{out.minW=Math.min(out.minW,b.w);});});
 return out;}"""
rule_hits, fit_hits = [], []


# the owner's clarification (2026-10-07 23:37 UTC): no hover that only expands a shown number to its exact value; Show the math is the exception
EXACT = """(sel)=>{const root=document.querySelector(sel); if(!root) return []; const out=[];
 root.querySelectorAll('[data-tipx],[data-tip],[title]').forEach(e=>{if(e.closest('#mp-math,.mathbody')) return;
  for(const a of ['data-tipx','data-tip','title']){const t=e.getAttribute(a)||''; if(/精确值是|Exactly/.test(t)) out.push(a+': '+t.slice(0,90));}});
 return out;}"""


def page(b, w, h, lang, dark=False, hash_='model-parallel'):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light')
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__mp.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#mp-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#mp-rail [data-next-why]')].filter(e=>!e.hidden&&e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden').map(e=>e.textContent)")
badges = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#mp-metrics .d.on')].map(e=>e.textContent)")
obs = lambda pg: pg.evaluate("()=>{const o=document.querySelector('#mp-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__mp||!window.__mp.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel, wait=250):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(wait); settle(pg)


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#mp-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def rules(pg, where, lang):
    for sel in ('#mp-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
        for h in pg.evaluate(EXACT, sel): rule_hits.append(f'{where}: a hover that only gives the exact value: {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def fit(pg, where):
    for h in pg.evaluate(OVER): fit_hits.append(f'{where}: text wider than its box (R7.8): {h}')
    for h in pg.evaluate(HELP): fit_hits.append(f'{where}: help cursor without a tooltip (R1.1): {h}')


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
    fm = pg.evaluate(FORMULA_H1, '#mp-app')
    check(f'{where}: formulas typeset as MathML, no TeX word shown, no plain-text formula left (H1, R6.29)', not fm['bad'] and not fm['txt'], fm)


def every(pg, where, lang, full_shot=None):
    formulas_h1(pg, where)
    check(f'{where}: no overflow', overflow(pg) <= 0, overflow(pg))
    rules(pg, where, lang); fit(pg, where)
    if full_shot: shot(pg, full_shot, full=pg.viewport_size['width'] <= 760)


def before_answer(pg):
    return pg.evaluate("""()=>{const t=[]; for(const s of ['#mp-app .labhead','#mp-rail','#mp-stage','#mp-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#mp-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")


def nav(pg, what):
    click(pg, f'#mp-rail [data-nav="{what}"]', 300)


def wait_done(pg, ms=12000):
    try: pg.wait_for_function("()=>window.__mp.state().ph>=1&&!window.__mp.state().playing", timeout=ms); return True
    except Exception: return False


def fast(pg):
    pg.evaluate("()=>{const b=document.querySelector('#mp-player .seg-ctl[data-key=\"spd\"] button[data-v=\"fast\"]'); if(b) b.click();}"); pg.wait_for_timeout(60)


def run_guided(b, w, h, lang, dark=False):
    tag = f'{"dark_" if dark else ""}{w}_{lang}'
    pg = page(b, w, h, lang, dark)
    phone = w <= 760
    check(f'{tag} title', text(pg, '#mp-app .labtitle') == Z(lang, 'How do pipeline and tensor parallelism cut one model apart?', '流水线并行和张量并行怎么切开一个模型？'), text(pg, '#mp-app .labtitle'))
    check(f'{tag} R4: the outcome names pipeline before tensor parallelism', Z(lang, 'pipeline', '安排流水线、张量和数据并行') in text(pg, '#mp-app .labhead'), text(pg, '#mp-app .labhead')[:300])
    check(f'{tag} no fidelity tag on top (owner, 2026-10-11 19:28 UTC)', not any(w in text(pg, '#mp-app .labmeta') for w in ('Fidelity', '保真度')), text(pg, '#mp-app .labmeta'))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#mp-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        pg.evaluate(f"()=>document.querySelector('#mp-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(200)
        check(f'{tag} popover {pid} opens', pg.evaluate(f"()=>{{const p=document.getElementById('{pid}'); return !!p&&p.getClientRects().length>0}}"))
        rules(pg, f'{tag} popover {pid}', lang)
        pg.evaluate(f"()=>document.querySelector('#mp-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(150)
    # ---- part 1: the prediction gate; the schedule stays hidden and no answer shows (R3.8)
    st = state(pg)
    check(f'{tag} part 1: gate, Next disabled with its reason', st['step'] == 0 and all(nexts(pg)) and any(Z(lang, 'Pick an answer', '选一个答案') in x for x in why(pg)), (nexts(pg), why(pg)))
    check(f'{tag} part 1: the schedule hidden (no cells, no player)', pg.evaluate("()=>document.querySelectorAll('#mp-gantt .mp-cf,#mp-gantt .mp-cb').length") == 0 and not vis(pg, '#mp-player'))
    ba = before_answer(pg)
    check(f'{tag} part 1: the answer appears nowhere before the pick (R3.8)', '88%' not in ba and '87.5' not in ba and Z(lang, 'Only 1 works', '只有 1 张在干活') not in ba, [x for x in ('88%', '87.5') if x in ba])
    every(pg, f'{tag} part 1', lang, f'{tag}_p1')
    check(f'{tag} part 1: p defined where the stage is (owner b8 item 5)', 'p = 8' in text(pg, '#mp-rail .step.current'), text(pg, '#mp-rail .step.current')[-300:])
    click(pg, '#mp-rail [data-pred="one"]')
    st = state(pg)
    check(f'{tag} part 1: the pick is saved, Next enabled', st['pred'] == 'one' and not any(nexts(pg)) and Z(lang, 'Saved', '已记下') in text(pg, '#mp-rail'), st['pred'])
    # ---- part 2: GPipe; auto-plays on entering (R3.9, medium), replays only a new picture (R3.20)
    nav(pg, 'next')
    st = state(pg)
    check(f'{tag} part 2: auto-plays at Medium on entering', st['step'] == 1 and st['playing'] and st['spd'] == 'mid' and st['m'] == 1, st)
    check(f'{tag} part 2: the red outline marks the idle time (R5.11)', pg.evaluate("()=>document.querySelectorAll('#mp-gantt .mp-ci.red').length") > 0)
    click(pg, '#mp-player [data-act="play"]', 300)
    ph1 = state(pg)['ph']; pg.wait_for_timeout(500)
    check(f'{tag} part 2: Pause stops the run', not state(pg)['playing'] and state(pg)['ph'] == ph1, state(pg))
    fast(pg); click(pg, '#mp-player [data-act="play"]', 100)
    t0 = time.time(); done = wait_done(pg); dt = time.time() - t0
    check(f'{tag} part 2: Fast finishes in about 4.5 s (owner b8 item 1)', done and dt < 7, dt)
    check(f'{tag} part 2: Next waits for 32 micro-batches, with its reason', all(nexts(pg)) and any(Z(lang, '32 or more', '32 或更多') in x for x in why(pg)), why(pg))
    click(pg, '#mp-ic-m button[data-v="32"]', 300)
    st = state(pg)
    check(f'{tag} part 2: a new picture (m 32) plays', st['m'] == 32 and st['playing'], st)
    wait_done(pg)
    o = obs(pg)
    check(f'{tag} part 2: the result names 18% and 88% and the bubble formula', '18%' in o and '88%' in o and vis(pg, '#mp-rail [data-obs] .mp-fx'), o[:200])
    # H1 (2026-10-09): the table's formula and its worked examples are typeset (MathML fractions), so read numerator / denominator
    wx = pg.evaluate("()=>[...document.querySelectorAll('#mp-rail [data-obs] .mp-wx tbody tr')].map(r=>({cur:r.classList.contains('cur'),m:r.cells[0].textContent.replace(/\\s+/g,''),fr:[...r.querySelectorAll('mfrac')].map(f=>[...f.children].map(c=>c.textContent.replace(/\\s+/g,'')).join('/'))}))")
    wh = pg.evaluate("()=>{const t=document.querySelector('#mp-rail [data-obs] .mp-wx th:last-child math mfrac'); return t?[...t.children].map(c=>c.textContent.replace(/\\s+/g,'')).join('/'):''}")
    check(f'{tag} part 2: worked examples of the formula for the m tried, the current one bold (owner b8 item 5), typeset (H1)', wh == 'p−1/m+p−1' and any(x['m'] == 'm=1' and '7/8' in x['fr'] for x in wx) and any(x['m'] == 'm=32' and x['cur'] and '7/39' in x['fr'] for x in wx), (wh, wx))
    # H1 (2026-10-09): the option's worked example is typeset (kit escMath keeps A.ui.math in a hover), so read its fractions
    check(f'{tag} part 2: each m option carries its worked example, typeset; the label, caption and result name m and p', '7/39' in (pg.evaluate("()=>{const t=document.createElement('template'); t.innerHTML=document.querySelector('#mp-ic-m button[data-v=\"32\"]').getAttribute('data-tipx')||''; return [...t.content.querySelectorAll('mfrac')].map(f=>[...f.children].map(c=>c.textContent.replace(/\\s+/g,'')).join('/'))}") or [])
          and 'm' in text(pg, '#mp-ic-m .lbl') and 'p = 8' in text(pg, '#mp-g-cap') and 'm = 32' in text(pg, '#mp-g-cap') and 'p = 8' in o and 'm = 32' in o, (text(pg, '#mp-ic-m .lbl'), text(pg, '#mp-g-cap')))
    bl = pg.evaluate(BLOCKS)
    check(f'{tag} part 2, m 32: blocks stay apart (owner b8 item 2)', bl['minW'] >= 4.5 and bl['minGap'] >= 0.5 and bl['thick'] == 0, bl)
    click(pg, '#mp-ic-m button[data-v="64"]', 300); fast(pg); wait_done(pg)
    bl = pg.evaluate(BLOCKS)
    check(f'{tag} part 2, m 64: blocks stay apart, grouped with the note (owner b8 item 2)', bl['minW'] >= 4.5 and bl['minGap'] >= 0.5 and bl['thick'] == 0 and Z(lang, 'micro-batches', 'micro-batch') in bl['note'] and all(r >= 8 for r in bl['rows']), bl)
    shot(pg, f'{tag}_p2_m64')
    click(pg, '#mp-ic-m button[data-v="32"]', 300); fast(pg); wait_done(pg)
    check(f'{tag} part 2: Next enabled', not any(nexts(pg)))
    click(pg, '#mp-ic-m button[data-v="1"]', 400)
    st = state(pg)
    check(f'{tag} part 2: a picture already watched does not replay (R3.20)', st['m'] == 1 and not st['playing'] and st['ph'] >= 1, st)
    click(pg, '#mp-player [data-act="play"]', 200)
    check(f'{tag} part 2: the Replay control plays it again', state(pg)['playing'], state(pg))
    click(pg, '#mp-ic-m button[data-v="32"]', 300); fast(pg); wait_done(pg)
    every(pg, f'{tag} part 2', lang, f'{tag}_p2')
    click(pg, '#mp-ic-m button[data-v="16"]', 150); click(pg, '#mp-ic-m button[data-v="32"]', 150)
    # ---- part 3: 1F1B
    nav(pg, 'next')
    check(f'{tag} part 3: entered right after a change, no ▲▼ badge (B1, R6.6)', badges(pg) == [], badges(pg))
    st = state(pg)
    check(f'{tag} part 3: GPipe at 16 micro-batches plays (a new picture)', st['step'] == 2 and st['sched'] == 'gpipe' and st['m'] == 16 and st['playing'], st)
    check(f'{tag} part 3: Next waits for 1F1B', all(nexts(pg)) and any('1F1B' in x for x in why(pg)), why(pg))
    check(f'{tag} part 3: the options have a lead-in naming what to watch (G1)', Z(lang, 'pick a schedule and see how many micro-batches of activations GPU 1 holds at once', '选一种排法，看看第 1 张卡最多要同时存几份激活') in text(pg, '#mp-rail .step.current'), text(pg, '#mp-rail .step.current')[-200:])
    fast(pg); wait_done(pg)
    click(pg, '#mp-ic-lv3 [data-lever="1f1b"]', 300)
    st = state(pg)
    check(f'{tag} part 3: 1F1B plays', st['sched'] == '1f1b' and st['playing'], st)
    fast(pg); wait_done(pg)
    o = obs(pg)
    res = pg.evaluate("()=>[...document.querySelectorAll('#mp-rail [data-lever] .lv-r')].map(e=>e.textContent)")
    check(f'{tag} part 3: both results on their options, m = 16 and p = 8', len(res) == 2 and 'm = 16' in res[0] and 'p = 8' in res[1], res)
    check(f'{tag} part 3: the result: 8 instead of 16, the bubble unchanged (30%)', '8' in o and '16' in o and '30%' in o and not any(nexts(pg)), o[:200])
    m1 = pg.evaluate("()=>document.querySelector('#mp-metrics [data-k=\"act\"] .v').textContent")
    check(f'{tag} part 3: the card shows 8 micro-batches held', m1 == '8', m1)
    every(pg, f'{tag} part 3', lang, f'{tag}_p3')
    # ---- part 4: tensor parallelism
    nav(pg, 'next')
    st = state(pg)
    check(f'{tag} part 4: the layer view, no Gantt, nothing playing', st['step'] == 3 and vis(pg, '#mp-tp') and not vis(pg, '#mp-gantt') and not st['playing'], st)
    check(f'{tag} part 4: Next waits for TP 8', all(nexts(pg)))
    click(pg, '#mp-ic-t button[data-v="8"]', 300)
    o = obs(pg)
    check(f'{tag} part 4: TP 8: 5.31 ms, 1.04 ms, 16%', '5.31 ms' in o and '1.04 ms' in o and '16%' in o and not any(nexts(pg)), o[:200])
    zw = pg.evaluate("()=>[...document.querySelectorAll('#mp-lt .mp-ltz .mp-ar')].map(e=>e.getBoundingClientRect().width)")
    check(f'{tag} part 4: the zoomed row shows the 4 all-reduces at least 10 px wide (S13, R5.10)', len(zw) == 4 and min(zw) >= 10, zw)
    check(f'{tag} part 4: 8 blue slices in each matrix', pg.evaluate("()=>[...document.querySelectorAll('#mp-tp .mp-mat')].map(m=>m.querySelectorAll('.mp-st').length)") == [8, 8])
    every(pg, f'{tag} part 4', lang, f'{tag}_p4')
    # ---- part 5: inside one machine
    nav(pg, 'next')
    check(f'{tag} part 5: Next waits for the other option', all(nexts(pg)))
    check(f'{tag} part 5: the options have a lead-in naming what to watch (G1)', Z(lang, 'Pick where the same 8 GPUs sit and see how much of the layer’s time goes to waiting for the all-reduces', '同样 8 张卡，选一种放法，看看这一层的时间里有多少花在等 all-reduce 上') in text(pg, '#mp-rail .step.current'), text(pg, '#mp-rail .step.current')[-200:])
    click(pg, '#mp-ic-lv5 [data-lever="out"]', 300)
    o = obs(pg)
    mt = text(pg, '#mp-metrics')
    check(f'{tag} part 5: across machines 9 times as long, nearly two thirds; 9.40 ms, 64% on the cards; red (R5.11)', Z(lang, '9 times', '9 倍') in o and Z(lang, 'nearly two thirds', '将近三分之二') in o and '9.40 ms' in mt and '64%' in mt
          and pg.evaluate("()=>document.querySelectorAll('#mp-lt .mp-ltbar:not(.z) .mp-ar.red').length") == 4
          and 'bad' in pg.evaluate("()=>document.getElementById('mp-status').className") and not any(nexts(pg)), o[:200])
    if phone:
        pg.evaluate("()=>{const t=document.getElementById('mp-graph'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+80)}"); pg.wait_for_timeout(700)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, status:document.getElementById('dkStatus').textContent, read:document.getElementById('dkRead').textContent}}")
        check(f'{tag} part 5: the mini result bar with a short status and the result', dk['shown'] and '64%' in dk['status'] + dk['read'], dk)
        shot(pg, f'{tag}_p5_dock')
        rules(pg, f'{tag} part 5 dock', lang)
    every(pg, f'{tag} part 5', lang, f'{tag}_p5')
    # ---- part 6: stacking them
    nav(pg, 'next')
    st = state(pg)
    check(f'{tag} part 6: the cluster, 8 cells, the memory bar, nothing playing', st['step'] == 5 and pg.evaluate("()=>document.querySelectorAll('#mp-cl .mp-cell').length") == 8 and vis(pg, '#mp-mem .mp-mbar') and not st['playing'], st)
    mt = text(pg, '#mp-metrics')
    check(f'{tag} part 6: 37.2 GB, 2.3%, 16% on the cards', '37.2 GB' in mt and '2.3%' in mt and '16%' in mt and not any(nexts(pg)), mt)
    bl = pg.evaluate(BLOCKS)
    check(f'{tag} part 6: the 1F1B schedule as separate blocks, not one dark bar (owner b8 item 4)', bl['mix'] > 0 and bl['minGap'] >= 0.5 and bl['thick'] == 0 and all(r >= 8 for r in bl['rows']) and Z(lang, 'micro-batches', 'micro-batch') in bl['note'], bl)
    check(f'{tag} part 6: context and expert parallelism named', Z(lang, 'context parallelism', '上下文并行') in text(pg, '#mp-rail').lower() and Z(lang, 'expert parallelism', '专家并行') in text(pg, '#mp-rail').lower())
    every(pg, f'{tag} part 6', lang, f'{tag}_p6')
    # ---- part 7: what to remember
    nav(pg, 'next')
    rt = text(pg, '#mp-rail')
    check(f'{tag} part 7: three points, three misconceptions, See the scenarios', pg.evaluate("()=>document.querySelectorAll('#mp-rail .mp-remlist > li').length") == 3 and rt.count('✗') == 3
          and pg.evaluate("()=>!!document.querySelector('#mp-rail [data-mode-go=\"sandbox\"]')"), rt[-400:])
    check(f'{tag} part 7: the recap states the patterns and what follows, not an earlier part\'s numbers (G2)', not re.search(r'[Ii]n part \d|第 \d 部分里|第 \d 部分的', rt)
          and all(x in rt for x in Z(lang, ['make the number of micro-batches m much larger than the number of stages p', 'without keeping more activations', 'every all-reduce takes several times as long'],
                                         ['micro-batch 数 m 要比段数 p 大得多', '激活也不会跟着变多', '每次 all-reduce 都要多等好几倍的时间'])), rt[-700:])
    every(pg, f'{tag} part 7', lang, f'{tag}_p7')
    # ---- the language switch keeps the place; Back returns to part 6
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    check(f'{tag} language switch keeps the part', state(pg)['step'] == 6)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    nav(pg, 'back'); nav(pg, 'back'); nav(pg, 'back'); nav(pg, 'back')
    st = state(pg)
    check(f'{tag} back in part 3 (1F1B, watched): no replay (R3.20)', st['step'] == 2 and not st['playing'] and st['ph'] >= 1, st)
    nav(pg, 'back')
    st = state(pg)
    check(f'{tag} back in part 2 (m 1, watched to the end): its end shown at once, no replay (R3.20)', st['step'] == 1 and st['m'] == 1 and not st['playing'] and st['ph'] >= 1, st)
    click(pg, '#mp-player [data-act="play"]', 200)
    check(f'{tag} back in part 2: the Replay control still replays on demand', state(pg)['playing'], state(pg))
    pg.close()


def run_modes(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    mode(pg, 'sandbox')
    st = state(pg)
    check(f'{tag} scenarios: four, the first pressed on entry, nothing playing', pg.evaluate("()=>document.querySelectorAll('#mp-rail [data-preset]').length") == 4
          and pg.evaluate("()=>document.querySelector('#mp-rail [data-preset=\"0\"]').getAttribute('aria-pressed')") == 'true' and not st['playing'], st)
    check(f'{tag} scenarios: a lead-in instead of “one at a time” (G1)', Z(lang, 'Pick a scenario and see whether each GPU has enough memory', '选一个场景，看看每张卡的显存够不够') in text(pg, '#mp-rail'), text(pg, '#mp-rail')[:300])
    want = {0: ('169 GB', 'bad'), 1: ('67.0 GB', 'ok'), 2: ('32%', 'ok'), 3: ('179 GB', 'bad')}
    for i in range(4):
        click(pg, f'#mp-rail [data-preset="{i}"]', 400)
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#mp-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        ins = text(pg, '#mp-insight'); mt = text(pg, '#mp-metrics'); cls = pg.evaluate("()=>document.getElementById('mp-status').className")
        check(f'{tag} scenario {i + 1}: no ▲▼ badge for a scenario (S8, R6.6)', badges(pg) == [], badges(pg))
        check(f'{tag} scenario {i + 1}: pressed alone, {want[i][0]}, status {want[i][1]}', pressed == ['true' if j == i else 'false' for j in range(4)] and want[i][0] in mt + ins and want[i][1] in cls, (pressed, mt, cls, ins[:120]))
        every(pg, f'{tag} scenario {i + 1}', lang, f'{tag}_scenario{i + 1}')
    url = pg.url
    click(pg, '#mp-rail [data-preset="2"]', 300)
    url = pg.url; st = state(pg)
    pg2 = b.new_page(viewport={'width': w, 'height': h}); open_drawers(pg2); pg2.goto(URL); pg2.wait_for_timeout(250)
    pg2.click(f'.langsw button[data-lang="{lang}"]'); pg2.wait_for_timeout(80)
    pg2.evaluate(f"()=>{{location.hash={url.split('#', 1)[1]!r}}}"); pg2.wait_for_timeout(500)
    st2 = pg2.evaluate("()=>window.__mp.state()")
    check(f'{tag} scenarios: the URL carries the setup, a fresh page restores it', all(st2[k] == st[k] for k in ('model', 't', 'p', 'd', 'm', 'seq', 'sched', 'place', 'zero')) and st2['mode'] == 'sandbox', (url, st2))
    pg2.close()
    # Settings: change one value, the scenario buttons follow
    click(pg, '#mp-ctl-m button[data-v="64"]', 300)
    pressed = pg.evaluate("()=>[...document.querySelectorAll('#mp-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: after a change in Settings no scenario is pressed', pressed == ['false'] * 4 and state(pg)['m'] == 64, (pressed, state(pg)['m']))
    check(f'{tag} Settings: the learner\'s own change shows ▲▼ badges (R6.6)', len(badges(pg)) >= 1, badges(pg))
    every(pg, f'{tag} sandbox changed', lang)
    click(pg, '#mp-ctl-m button[data-v="32"]', 100)
    # ---- the challenge
    mode(pg, 'challenge')
    check(f'{tag} challenge: entered right after a change, no ▲▼ badge (B1, R6.6)', badges(pg) == [], badges(pg))
    st = state(pg)
    check(f'{tag} challenge: starts at TP 8 × PP 1, 8 micro-batches; Settings locked', (st['t'], st['p'], st['m'], st['d']) == (8, 1, 8, 4) and not vis(pg, '#mp-ctl-t-wrap'), st)
    check(f'{tag} challenge: the goal lines show the start (165 GB)', '165 GB' in text(pg, '#mp-ch-mem'), text(pg, '#mp-ch-mem'))
    every(pg, f'{tag} challenge start', lang, f'{tag}_challenge_start')
    click(pg, '#mp-rail [data-act="submit"]', 400)
    r = text(pg, '#mp-ch-res')
    check(f'{tag} challenge tier 1 (does not fit): red, a hint, no summary yet, a button to show the fastest layout', Z(lang, 'Not yet', '还没达到') in r and pg.evaluate("()=>!!document.querySelector('#mp-ch-res .badline')") and not vis(pg, '#mp-ch-res .mp-chsum') and vis(pg, '#mp-ch-res [data-act="reveal"]'), r[:200])
    click(pg, '#mp-ch-res [data-act="reveal"]', 300)
    r = text(pg, '#mp-ch-res')
    check(f'{tag} challenge tier 1 revealed: your layout in red (does not fit) next to the fastest; no single change fits', vis(pg, '#mp-ch-res .mp-chsum') and pg.evaluate("()=>document.querySelectorAll('#mp-ch-res .mp-c3r').length") == 2
          and vis(pg, '#mp-ch-res .mp-c3r.me .mp-c3x.bad') and Z(lang, 'No single change makes it fit', '只改一项还放不下') in r, r[-400:])
    nx = text(pg, '#mp-ch-res .mp-next')
    check(f'{tag} challenge tier 1: names the three settings the fastest layout changes together (G7)', all(x in nx for x in Z(lang, ['a pipeline of 4 stages instead of none', 'tensor parallelism 4 ways instead of 8', 'm = 128 micro-batches instead of 8'], ['改用 4 段流水线', '把张量并行改成 4 份', '把 micro-batch 数改成 m = 128'])), nx)
    click(pg, '#mp-ic-chp button[data-v="4"]', 300)
    click(pg, '#mp-ic-chm button[data-v="256"]', 400)
    st = state(pg); r = text(pg, '#mp-ch-res')
    check(f'{tag} challenge tier 2 (fits, not the fastest): TP 8 × PP 4 × 256, 32.9 s; the verdict first, rounded', (st['t'], st['p'], st['m']) == (8, 4, 256) and '32.9 s' in text(pg, '#mp-ch-step') and Z(lang, 'not the fastest', '不是最快的') in r.split('\n')[0] and Z(lang, 'about 33 s', '约 33 秒') in r.split('\n')[0], (st, r[:200]))
    check(f'{tag} challenge tier 2: R1 summary: two rows, shared settings once, your differing settings marked, the gap, the best single change (TP 4)', pg.evaluate("()=>document.querySelectorAll('#mp-ch-res .mp-c3r').length") == 2
          and pg.evaluate("()=>document.querySelectorAll('#mp-ch-res .mp-c3r.me .mp-chip.on').length") >= 2 and Z(lang, 'p = 4', 'p = 4') in text(pg, '#mp-ch-res .mp-same')
          and Z(lang, 'tensor-parallel all-reduces', '张量并行的 all-reduce') in text(pg, '#mp-ch-res .mp-gap') and Z(lang, 'tensor parallelism 4 ways instead of 8', '从 8 份改成 4 份') in text(pg, '#mp-ch-res .mp-next')
          and not vis(pg, '#mp-ch-res .cmp'), r[-500:])
    check(f'{tag} challenge tier 2: the layouts that pass sit behind a closed disclosure (2)', pg.evaluate("()=>{const d=document.querySelector('#mp-ch-res details.mp-all'); return !!d&&!d.open}") and Z(lang, '(2)', '（2 种）') in text(pg, '#mp-ch-res details.mp-all summary'), text(pg, '#mp-ch-res details.mp-all summary'))
    nx = text(pg, '#mp-ch-res .mp-next')
    check(f'{tag} challenge tier 2: the change with the step before and after, and the card\'s m change (G7)', Z(lang, 'a step goes from 32.9 s to 30.2 s, which meets the goal', '每步从 32.9 秒缩短到 30.2 秒，改完就达标了') in nx and Z(lang, 'the card then sets m = 128', '任务卡会把 m 随之改成 128') in nx, nx)
    click(pg, '#mp-ic-cht button[data-v="4"]', 300)
    st = state(pg)
    check(f'{tag} challenge: doing it, the task card sets m = 128 as the result said, and shows 30.2 s', (st['t'], st['p'], st['m']) == (4, 4, 128) and '30.2 s' in text(pg, '#mp-ch-step'), (st, text(pg, '#mp-ch-step')))
    click(pg, '#mp-ic-chm button[data-v="128"]', 400)
    st = state(pg); r = text(pg, '#mp-ch-res')
    check(f'{tag} challenge tier 3 (goal met): TP 4 × PP 4 × 128, 30.2 s', (st['t'], st['p'], st['m'], st['d']) == (4, 4, 128, 2) and Z(lang, 'Goal met', '达成目标') in r and '30.2 s' in text(pg, '#mp-ch-step') and Z(lang, 'about 30 s', '约 30 秒') in r, (st, r[:200]))
    check(f'{tag} challenge tier 3: one row, yours is the fastest; the disclosure lists the 2 layouts that pass when opened', pg.evaluate("()=>[...document.querySelectorAll('#mp-ch-res .mp-c3r')].map(e=>e.dataset.role).join()") == 'mebest'
          and pg.evaluate("()=>{const d=document.querySelector('#mp-ch-res details.mp-all'); if(!d||d.open) return false; d.open=true; return d.querySelectorAll('.mp-pass li').length===2}"), r[-300:])
    check(f'{tag} challenge result: no hover in the lab gives an exact value (the owner, 23:37 UTC)', not pg.evaluate(EXACT, '#mp-app') and not pg.evaluate(EXACT, '#dock'), pg.evaluate(EXACT, '#mp-app'))
    check(f'{tag} challenge: the capability card', vis(pg, '#mp-rail [data-exits]'))
    pg.evaluate("()=>document.getElementById('mp-ch-res').scrollIntoView({block:'start'})"); pg.wait_for_timeout(300)
    every(pg, f'{tag} challenge solved', lang, f'{tag}_challenge_solved')
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} challenge: a language switch keeps the plan and the result', (st['t'], st['p'], st['m']) == (4, 4, 128) and st['chSubmitted'] and vis(pg, '#mp-ch-res .mp-chsum'), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    # ---- G7 (Codex on b9 preview v5): a single change is named only when the step the card shows gets shorter, and doing it shows that
    card = lambda: (text(pg, '#mp-ch-step'), text(pg, '#mp-ch-mem'))
    secs = lambda x: float(x.split()[0].replace(',', ''))
    gbs = lambda x: float(x.split()[0].replace(',', ''))
    click(pg, '#mp-ic-chp button[data-v="2"]', 250); click(pg, '#mp-ic-cht button[data-v="8"]', 250); click(pg, '#mp-ic-chm button[data-v="64"]', 300)
    st = state(pg); nx = text(pg, '#mp-ch-res .mp-next')
    check(f'{tag} G7: PP 2 × TP 8 × 64 (33.4 s) names m = 128 with the step before and after as the card shows them', (st['t'], st['p'], st['m']) == (8, 2, 64) and card()[0] == '33.4 s'
          and Z(lang, 'm = 128 micro-batches instead of 64: a step goes from 33.4 s to 33.2 s', '把 micro-batch 数从 m = 64 改成 m = 128 最划算。每步从 33.4 秒缩短到 33.2 秒') in nx, (st, card(), nx))
    click(pg, '#mp-ic-chm button[data-v="128"]', 300)
    st = state(pg); r = text(pg, '#mp-ch-res'); nx = text(pg, '#mp-ch-res .mp-next')
    check(f'{tag} G7: doing it shows 33.2 s, shorter than 33.4 s', (st['t'], st['p'], st['m']) == (8, 2, 128) and card()[0] == '33.2 s', (st, card()))
    check(f'{tag} G7 (Codex): PP 2 × TP 8 × 128 says no single change helps and names PP 4 and TP 4 together; no “about 0 s”', nx.startswith(Z(lang, 'No single change makes a step shorter.', '不管只改哪一项，每步都不会变短。'))
          and all(x in nx for x in Z(lang, ['4 pipeline stages instead of 2', 'tensor parallelism 4 ways instead of 8'], ['把流水线改成 4 段', '把张量并行改成 4 份']))
          and not any(x in r for x in ('about 0 s', '约 0 秒', '少约', '最划算', 'best single change')), nx)
    pg.evaluate("()=>document.getElementById('mp-ch-res').scrollIntoView({block:'start'})"); pg.wait_for_timeout(250)
    every(pg, f'{tag} challenge, the Codex pick', lang, f'{tag}_challenge_codex')
    # make every single change the card allows, as a reader would, and read the card: none fits with a shorter step than 33.2 s
    tried, better = [], []
    def back():
        click(pg, '#mp-ic-chp button[data-v="2"]', 150); click(pg, '#mp-ic-cht button[data-v="8"]', 150); click(pg, '#mp-ic-chm button[data-v="128"]', 200)
    for key, vals in (('cht', ('1', '2', '4')), ('chp', ('1', '4', '8')), ('chm', ('8', '16', '32', '64', '256'))):
        for v in vals:
            dis = pg.evaluate(f"()=>document.querySelector('#mp-ic-{key} button[data-v=\"{v}\"]').disabled")
            if dis:
                tried.append(f'{key} {v}: disabled'); continue
            click(pg, f'#mp-ic-{key} button[data-v="{v}"]', 200)
            stp, mem = card(); s2 = state(pg); tried.append(f"{key} {v}: p{s2['p']} t{s2['t']} m{s2['m']} {stp} {mem}")
            if gbs(mem) <= 80 and secs(stp) < 33.2: better.append(tried[-1])
            back()
    check(f'{tag} G7: made one by one in the card, no single change fits and shows a shorter step than 33.2 s', not better and len([x for x in tried if 'disabled' not in x]) == 9, (better, tried))
    st = state(pg)
    check(f'{tag} G7: back at the Codex pick', (st['t'], st['p'], st['m']) == (8, 2, 128), st)
    click(pg, '#mp-ic-cht button[data-v="4"]', 200); click(pg, '#mp-ic-chp button[data-v="8"]', 250)
    check(f'{tag} challenge: with 8 pipeline stages, tensor parallelism 8 is disabled (it would leave no whole data-parallel copy)', pg.evaluate("()=>document.querySelector('#mp-ic-cht button[data-v=\"8\"]').disabled") and state(pg)['p'] == 8, state(pg))
    ex = pg.evaluate("()=>window.__mp.exits().text")
    check(f'{tag} G2: the capability card names no earlier part', not re.search(r'[Ii]n part \d|[Pp]art \d’s|第 \d 部分', ex), [l for l in ex.split('\n') if re.search(r'part \d|部分', l)][:3])
    # the Connect drill
    pg.evaluate("()=>{const b=document.querySelector('#mp-conn [data-act=\"drill-reveal\"], [data-act=\"drill-reveal\"]'); if(b) b.click();}"); pg.wait_for_timeout(250)
    check(f'{tag} connect: the drill answer opens', vis(pg, '#mp-drill-a'))
    every(pg, f'{tag} connect', lang)
    mode(pg, 'guided')
    check(f'{tag} back to Guided', state(pg)['mode'] == 'guided')
    every(pg, f'{tag} guided again', lang)
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    for (w, h) in SIZES:
        for lang in LANGS:
            if 'modes' not in ARGS: run_guided(b, w, h, lang)
            if 'guided' not in ARGS: run_modes(b, w, h, lang)
    if FULL:
        run_guided(b, 1440, 900, 'zh', dark=True)
    b.close()
SRC = (P / 'labs' / 'model-parallel.js').read_text() + (P / 'teasers' / 'model-parallel.js').read_text()
check('G1: no “一次只能选 / 一次选一个 / 不会叠加 / do not stack / one at a time” anywhere in the lab source (aria-labels included)', not re.search(r'一次只能选|一次选一个|不会叠加|do not stack|[Oo]ne at a time', SRC), re.findall(r'.{30}(?:一次只能选|一次选一个|不会叠加|do not stack|[Oo]ne at a time).{20}', SRC)[:4])
for hit in rule_hits[:60]:
    check('writing rules: ' + hit, False)
for hit in fit_hits[:60]:
    check(hit, False)
check('no page errors', not errs, errs[:5])
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:60] if fails else 'none')
sys.exit(1 if fails else 0)
