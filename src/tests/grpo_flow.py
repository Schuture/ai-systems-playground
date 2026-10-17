"""Flow test for the grpo lab (CONTRACT §6), at 1440×900 and 390×844, in EN and ZH.

- no page errors, no horizontal overflow, no text wider than its box, a tooltip for every help cursor;
- part 1: the memory picture (PPO's four models against GRPO), Next enabled, no transport;
- part 2: Next disabled with the kit's why line until the 8 answers are generated; the group replays at 中 (R3.9); while it samples,
  neither the stage status nor the observation gives the result (review S15); the observation names 2 of 8 right and the 0.25 baseline;
- part 3: Next waits for a prediction; no result shown before it (R3.8);
- part 4: the verdict on the prediction, the formula, +1.62 and −0.54 on the chart; going Back and Next again replays nothing (R3.20);
- part 5: Next waits for 10 steps; the run plays at 快 (R3.9) to 30 steps in about 10 s; the observation names 44% → 84%; the main
  picture comes before the metric cards (R7.12); Back and Next again does not replay the watched run (R3.20);
- part 5: no “blown up” red label before part 7 explains it (review S17); every chart label stays inside its picture (review S16);
- part 6: Next waits for a second G; G = 2 gives 50%; G = 64 names the 98% question and its 31% of all-right groups (review S8);
- part 7: the wrong answer's large advantage is red (R5.11); switching the division off replays the advantages only (R3.20) and gives
  −0.88; switching it back replays nothing; 重放 replays from the start;
- part 8: 144.5 GB, the RLVR lab linked and never called planned (review B3), the scenarios button and the next-lab footer;
- Scenarios: five, the insight for each; a scenario already watched does not replay, a new one plays; the hash round trip; the
  language switch keeps the setup;
- the scenario without the division names the 2.4 times and the under-1-point gap at the matched step (review B1);
- the challenge: starts at G = 2 with no accuracy and no verdict before Submit, also after another G is chosen (review S19); Submit
  trains the 30 steps, then the verdict; a new G after a miss needs a new Submit; the three feedback tiers, the chart of all options
  with the pick, the capability card;
- the dock at 390 once the stage scrolls away;
- the writing rules on every visible text and tooltip (no “toy” outside the Toy run label, Chinese spacing and punctuation, two colons
  in one sentence), and the
  colours: red only for the failure moment and the challenge verdict, the weights colour only in the memory picture.
- the owner's review of 2026-10-09: the replay and speed buttons share one height and one baseline (item 1, H7); the link to the
  advantage bars frames the bars only, a focused bar gets its own ring (item 2, H7); part 5 says why the clip cannot act with one
  update per batch (item 3, H3); part 7 shows the formula with its denominator marked (item 4, R6.34); part 2 says where a reward
  comes from and what a reward model costs, part 8 and the memory note show it (item 5, R6.35, H9); the per-question chart colours
  all right green, some right blue, all wrong red in bars, legend and words, and names every mark (item 6, H6); every formula is
  typeset and no TeX command shows as a word (H1); Show the math is four blocks with aligned chains (H2); the status names the replayed
  phase of a step (H8); no planned-lab wording (H4); every chart label inside its picture and panel at 1440, 900 and 390 (H7).
Screenshots: shots/grpo/.
Run: python3 tests/grpo_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
(P / 'out' / 'grpo').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/grpo.html', '--only', 'grpo'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'grpo' / 'flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'grpo.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'grpo' / 'flow-wrapped.html')
SH = P / 'shots' / 'grpo'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, hash_='grpo'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__grpo.state()")
anim = lambda pg: pg.evaluate("()=>window.__grpo.animState()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-').replace(/\\u2212/g,'-'):''}", sel)
svgtext = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.textContent.replace(/\\u2212/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#grpo-rail [data-nav=\"next\"]')].filter(b=>b.getClientRects().length).map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#grpo-rail [data-next-why]')].filter(p=>!p.hidden&&getComputedStyle(p).visibility!=='hidden'&&p.getClientRects().length).map(p=>p.textContent)")
speed = lambda pg: pg.evaluate("()=>{const b=document.querySelector('#grpo-transport .seg-ctl[data-key=\"speed\"] button[aria-pressed=\"true\"]'); return b?b.dataset.v:null}")


def settle(pg, ms=2000):
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel, wait=250):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(wait); settle(pg)


# the animation state two frames after a click: None means nothing replays
CLICK_ANIM = """async (sel)=>{document.querySelector(sel).click(); await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))); return window.__grpo.animState();}"""


def click_anim(pg, sel):
    a = pg.evaluate(CLICK_ANIM, sel); pg.wait_for_timeout(150); settle(pg); return a


def wait_anim(pg, timeout=25000):
    pg.wait_for_function("()=>!window.__grpo.animState()", timeout=timeout); pg.wait_for_timeout(120)


def mode(pg, m):
    click(pg, f'#grpo-app .modes button[data-mode="{m}"]', 350)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def all_text(pg):
    return pg.evaluate("""()=>{const out=[]; for(const s of ['#grpo-app','#dock']){const e=document.querySelector(s); if(e&&e.offsetParent!==null) out.push(e.innerText);}
      document.querySelectorAll('#grpo-app [data-tipx], #dock [data-tipx]').forEach(e=>out.push(e.getAttribute('data-tipx'))); return out.join('\\n').replace(/\\u2212/g,'-').replace(/\\u2011/g,'-')}""")


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · (?![0-9])/,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],
   [/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy”'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],[/OpenAI|Anthropic|DeepMind|Google|Meta\\b|Microsoft/,'company name'],[/批评家|评论家/,'critic translated'],
   [/深色|浅色|浅灰|深灰|\\b(?:dark|light|pale)(?:er)? (?:bar|line|block|part|cell)/i,'colour named by lightness (R5.7)']];
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
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/grpo-mb|grpo-mbar|grpo-mseg|grpo-sw-w|\\bsw w\\b|mem|k-mix|grpo-kt mix|grpo-sw-k mix/,[amber]:/^$/,[red]:/grpo-adv|grpo-big|grpo-sw-big|status bad|badline|bad\\b|warn|wrong|ex-res/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#grpo-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#grpo-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg,.grpo-math-line')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#grpo-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
SVGIN = """()=>{const out=[]; for(const s of document.querySelectorAll('#grpo-stage svg')){if(!s.getClientRects().length) continue; const b=s.getBoundingClientRect();
  const ts=[...s.querySelectorAll('text')].filter(t=>t.getClientRects().length&&t.textContent.trim());
  for(const t of ts){const r=t.getBoundingClientRect(); if(r.left<b.left-1||r.right>b.right+1||r.top<b.top-1||r.bottom>b.bottom+1) out.push('outside: '+t.textContent.trim().slice(0,30));}
  for(const t of ts) if(t.classList.contains('grpo-big-t')){const r=t.getBoundingClientRect(); for(const u of ts){if(u===t) continue; const q=u.getBoundingClientRect();
    if(r.left<q.right-1&&q.left<r.right-1&&r.top<q.bottom-1&&q.top<r.bottom-1) out.push('overlap: '+t.textContent.trim()+' / '+u.textContent.trim().slice(0,30));}}}
  return out.slice(0,8);}"""
rule_hits = []


# the owner's clarification (2026-10-07 23:37 UTC): no hover only expands a shown number to its exact value; Show the math may write one out
EXACT_HOVER = """r=>{const root=document.querySelector(r), out=[]; if(!root) return out;
  for(const e of root.querySelectorAll('[data-tipx],[data-tip]')){if(e.closest('.mathbody')) continue; let t=e.getAttribute('data-tipx')||''; const k=e.getAttribute('data-tip');
    if(k&&window.ASP) t+=' '+(window.ASP.t(k)||''); if(/精确值是|Exactly/.test(t)) out.push(t.slice(0,100));}
  for(const e of root.querySelectorAll('.num-big')) if(!e.closest('.mathbody')&&(e.hasAttribute('data-tipx')||e.closest('.def'))) out.push('hover on '+e.textContent);
  return out;}"""


def rules(pg, where, lang):
    hx = pg.evaluate(EXACT_HOVER, '#grpo-app'); check(f'{where}: no exact-value hover outside Show the math', not hx, hx[:3])
    for sel in ('#grpo-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')
    for h in pg.evaluate(OVER): rule_hits.append(f'{where}: text wider than its box: {h}')
    for h in pg.evaluate(TEXWORDS): rule_hits.append(f'{where}: H1, a formula not typeset or a TeX word shown: {h}')
    for h in pg.evaluate(HELP): rule_hits.append(f'{where}: help cursor without a tooltip: {h}')
    o = overflow(pg)
    if o > 0: rule_hits.append(f'{where}: horizontal overflow {o}px')


# item 1 (H7): the buttons of one transport row share one height and one text baseline
ALIGN = """()=>{const els=[...document.querySelectorAll('#grpo-transport .btn, #grpo-transport .seg-ctl')].filter(e=>e.getClientRects().length);
  const rows=[]; for(const e of els){const r=e.getBoundingClientRect(); let row=rows.find(x=>Math.abs(x.top-r.top)<8); if(!row){row={top:r.top,items:[]}; rows.push(row);}
    const b=e.matches('.seg-ctl')?e.querySelector('button'):e, rg=document.createRange(); rg.selectNodeContents(b); const t=rg.getBoundingClientRect();
    row.items.push({t:(e.textContent||'').trim().slice(0,12),top:r.top,h:r.height,base:t.bottom});}
  const bad=[]; for(const row of rows){const it=row.items; for(const x of it){if(Math.abs(x.h-it[0].h)>0.6||Math.abs(x.top-it[0].top)>0.6||Math.abs(x.base-it[0].base)>1) bad.push(x.t+' vs '+it[0].t+' h '+x.h+'/'+it[0].h+' base '+x.base.toFixed(1)+'/'+it[0].base.toFixed(1));}}
  return {n:els.length,bad};}"""
# H1: no TeX command shown as a word; no plain-text formula left in the rendered text (outside math, code and pre)
# An unknown TeX command, or a bare word in the TeX source, comes out of the kit as <mi>word</mi>; a word written with \text{} is an
# <mtext> label on purpose (A with “right” under it in Show the math), so only <mi> and <mo> are checked.
TEXWORDS = """()=>{const W=new Set(['frac','left','right','sqrt','text','big','mathrm','hat','sum','cdot','times','div','epsilon','theta','rho','beta','pi']);
  const out=[]; for(const m of document.querySelectorAll('#grpo-app math, #dock math')) for(const e of m.querySelectorAll('mi,mo')){const t=e.textContent.trim(); if(W.has(t)) out.push(t);}
  const pat=[/÷ \\(/,/√\\(/,/Σ\\(r/,/π<sub>|πref|π_ref/,/e\\^/,/\\[1 − ε/,/\\b8 × G\\b/,/\\bn − 1\\b/,/A = \\(r/,/0\\.9 to the 8th|0\\.9 的 8 次方/];
  const tw=document.createTreeWalker(document.querySelector('#grpo-app'),NodeFilter.SHOW_TEXT); let n;
  while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('math,pre,code')||!e.getClientRects().length) continue; for(const r of pat) if(r.test(n.textContent)) out.push('text: '+n.textContent.trim().slice(0,60));}
  for(const e of document.querySelectorAll('#grpo-app [data-tipx]')){const t=e.getAttribute('data-tipx'); for(const r of pat) if(r.test(t)) out.push('tip: '+t.slice(0,60));}
  return out;}"""
NEXT = '#grpo-rail .step.current [data-nav="next"]'
BACK = '#grpo-rail .step.current [data-nav="back"]'


def go_next(pg):
    sel = NEXT if pg.query_selector(NEXT) else '#grpo-rail [data-nav="next"]'
    click(pg, sel, 300)


# B4 (polish r4, reviews/r4-fit-post-b.md): entering parts 5 and 6 at 1440 × 900 as the polish probe stage.py does (Next when it is
# enabled, else the part's heading), the stage needs no scrolling inside: Chinese with at least 30 px to spare, English with none.
# Spare = the stage's max-height minus what its content and borders need.
FIT = """()=>{const s=document.getElementById('grpo-stage'); return {spare:Math.round(parseFloat(getComputedStyle(s).maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight)), scroll:s.scrollHeight-s.clientHeight}}"""


def fit_entry(b, lang, parts=(5, 6)):
    pg = page(b, 1440, 900, lang)
    for k in range(2, max(parts) + 1):
        nb = pg.locator('.pbar [data-nav="next"]:visible:not([disabled])')
        if nb.count(): nb.first.click()
        else: pg.evaluate(f"()=>document.querySelector('#grpo-rail .steps>.step:nth-child({k}) .step-h').click()")
        pg.wait_for_timeout(900)
        if k not in parts: continue
        f = pg.evaluate(FIT); f['step'] = state(pg)['step'] + 1
        check(f'{lang}-1440 part {k}: entering it, the stage fits without scrolling inside{", 30 px to spare" if lang == "zh" else ""} (B4, R7.12)',
              f['step'] == k and f['scroll'] <= 0 and f['spare'] >= (30 if lang == 'zh' else 0), f)
    pg.close()


with sync_playwright() as pw:
    b = pw.chromium.launch()
    if (1440, 900) in SIZES:
        for lang in LANGS: fit_entry(b, lang)
    for lang in LANGS:
        for (w, h) in SIZES:
            phone = w <= 760; tag = f'{lang}-{w}'
            pg = page(b, w, h, lang)
            st = state(pg)
            check(f'{tag}: opens on part 1 of the guided tour', st['mode'] == 'guided' and st['step'] == 0, st)
            # ---- part 1
            check(f'{tag} part 1: the memory picture is shown, the transport is hidden, Next is enabled', vis(pg, '#grpo-pan-mem') and not vis(pg, '#grpo-transport') and not any(nexts(pg)), nexts(pg))
            mem = text(pg, '#grpo-pan-mem')
            check(f'{tag} part 1: 321.2 GB for PPO and 176.7 GB for GRPO in the memory picture', '321.2 GB' in mem and '176.7 GB' in mem, mem[:300])
            shot(pg, f'{tag}_p1', full=phone); rules(pg, f'{tag} part 1', lang)
            # ---- part 2
            go_next(pg); st = state(pg)
            check(f'{tag} part 2: Next disabled with the why line until the answers are generated', st['step'] == 1 and all(nexts(pg)) and any(Z(lang, 'Generate the 8 answers', '先生成 8 个回答') in x for x in why(pg)), (st, why(pg)))
            check(f'{tag} part 2: no answers on the chart before Generate (R3.8)', '0.25' not in svgtext(pg, '#grpo-gsvg'), svgtext(pg, '#grpo-gsvg')[:200])
            a = click_anim(pg, '#grpo-rail [data-act="gen"]')
            check(f'{tag} part 2: Generate replays the group from sampling at 中 (R3.9)', a and a['kind'] == 'group' and a['ph'] == 0 and speed(pg) == 'mid', (a, speed(pg)))
            st2 = text(pg, '#grpo-status'); obh = pg.evaluate("()=>{const e=document.querySelector('#grpo-rail [data-obs]'); return !e||e.hidden||!e.getClientRects().length}")
            check(f'{tag} part 2: while it samples, the status says so and no observation gives the result (S15)', Z(lang, 'Sampling', '抽样中') in st2 and obh and '0.25' not in svgtext(pg, '#grpo-gsvg'), (st2, obh))
            wait_anim(pg)
            ob = text(pg, '#grpo-rail [data-obs]')
            check(f'{tag} part 2: the observation names 2 of 8 right and the 0.25 baseline; Next enabled', ('2' in ob and '0.25' in ob) and not any(nexts(pg)), ob)
            al = pg.evaluate(ALIGN)
            check(f'{tag} part 2 (owner item 1, H7): 重放 and the speed buttons share one height and one baseline', al['n'] == 2 and not al['bad'], al)
            cur = text(pg, '#grpo-rail .step.current')
            check(f'{tag} part 2 (owner item 5): the card says a reward comes from a checker or a reward model, and what the reward model costs',
                  all(x in cur for x in Z(lang, ('checker', 'reward model', 'reads and scores each of the 8', 'its own GPU memory', 'this lab uses addition questions and a checker'),
                                               ('判分程序', '奖励模型', '读 8 遍、打 8 次分', '还要占自己的一份显存', '这个 Lab 为了简单，用加法题和判分程序')))
                  and pg.query_selector('#grpo-rail .step.current a[href="#reward-model"]') is not None, cur[:600])
            shot(pg, f'{tag}_p2', full=phone); rules(pg, f'{tag} part 2', lang)
            # ---- part 3
            go_next(pg)
            check(f'{tag} part 3: Next waits for a prediction, with its why line', all(nexts(pg)) and any(Z(lang, 'Pick an answer', '选一个答案') in x for x in why(pg)), why(pg))
            check(f'{tag} part 3: no advantage on the chart before the prediction (R3.8)', '1.62' not in svgtext(pg, '#grpo-gsvg') and '1.62' not in text(pg, '#grpo-rail .step.current'))
            click(pg, '#grpo-rail [data-pred="same"]')
            check(f'{tag} part 3: after a pick, Next is enabled and the card says it is saved', not any(nexts(pg)) and Z(lang, 'Saved', '已记下') in text(pg, '#grpo-rail .step.current'))
            shot(pg, f'{tag}_p3', full=phone); rules(pg, f'{tag} part 3', lang)
            # ---- part 4
            a = click_anim(pg, NEXT)
            check(f'{tag} part 4: entering replays only the new advantage phase (R3.20)', a and a['kind'] == 'group' and a['ph'] == 3, a)
            wait_anim(pg)
            cur = text(pg, '#grpo-rail .step.current'); g = svgtext(pg, '#grpo-gsvg')
            check(f'{tag} part 4: the verdict on the prediction names the right answer', Z(lang, 'Not quite', '不对') in cur and Z(lang, 'You predicted', '你预测的是') in cur, cur[:300])
            check(f'{tag} part 4: the formula is set with the kit (A.ui.math)', vis(pg, '#grpo-rail .step.current .grpo-math-line math'))
            check(f'{tag} part 4: +1.62 and -0.54 on the chart', '+1.62' in g and '-0.54' in g, g[:300])
            check(f'{tag} part 4 (H1): the formula has its name in words on the left', Z(lang, 'advantage', '优势') in pg.evaluate("()=>[...document.querySelectorAll('#grpo-rail .step.current .grpo-math-line mtext')].map(e=>e.textContent).join(' ')"))
            pg.evaluate("()=>document.querySelector('#grpo-rail .step.current [data-jump=\"bars\"]').click()"); pg.wait_for_timeout(450)
            hl = pg.evaluate("""()=>{const f=document.querySelector('#grpo-gsvg .grpo-bars-hl'); if(!f) return null; const b=f.getBoundingClientRect(), s=document.querySelector('#grpo-gsvg').getBoundingClientRect();
              const bars=[...document.querySelectorAll('#grpo-gsvg .grpo-adv')].map(e=>e.getBoundingClientRect());
              return {inside:bars.every(r=>r.left>=b.left-1&&r.right<=b.right+1&&r.top>=b.top-1&&r.bottom<=b.bottom+1), small:b.height<s.height*0.6,
                flash:[...document.querySelectorAll('.flash')].map(e=>e.id||e.className), leg:!!document.querySelector('#grpo-g-leg .grpo-leg-hl')}}""")
            check(f'{tag} part 4 (owner item 2, H7): the link frames the advantage bars and their legend entry, and nothing else flashes', hl and hl['inside'] and hl['small'] and hl['leg'] and not hl['flash'], hl)
            shot(pg, f'{tag}_p4_bars')
            pg.wait_for_timeout(1700)
            check(f'{tag} part 4 (owner item 2): the frame goes away again', pg.evaluate("()=>!document.querySelector('#grpo-gsvg .grpo-bars-hl')&&!document.querySelector('.grpo-leg-hl')"))
            fr = pg.evaluate("""()=>{const b=[...document.querySelectorAll('#grpo-gsvg .grpo-adv')]; const t=b.filter(e=>e.getAttribute('data-tipx')&&e.getAttribute('tabindex')==='0');
              if(!t.length) return {n:0}; t[0].focus(); const sw=getComputedStyle(t[0]).strokeWidth, pan=getComputedStyle(document.querySelector('#grpo-pan-group')).outlineStyle, st=getComputedStyle(document.querySelector('#grpo-stage')).outlineStyle;
              t[0].blur(); return {n:t.length,sw,pan,st};}""")
            check(f'{tag} part 4 (owner item 2): every bar explains itself on hover or focus, and focus rings the bar, not the panel', fr['n'] == 8 and fr['sw'] == '2.5px' and fr['pan'] == 'none' and fr['st'] == 'none', fr)
            shot(pg, f'{tag}_p4', full=phone); rules(pg, f'{tag} part 4', lang)
            click(pg, BACK)
            a = click_anim(pg, NEXT)
            check(f'{tag} part 4 again: Back and Next replay nothing (R3.20)', a is None and state(pg)['step'] == 3, a)
            # ---- part 5
            a = click_anim(pg, NEXT); st = state(pg)
            check(f'{tag} part 5: starts at step 0, nothing plays on entry, Next waits for 10 steps', a is None and st['steps'] == 0 and all(nexts(pg)) and any(Z(lang, '10 steps', '10 步') in x for x in why(pg)), (a, st, why(pg)))
            fv = pg.evaluate("()=>{const q=s=>document.querySelector(s).getBoundingClientRect(); return {grp:Math.round(q('#grpo-pan-group').top), chart:Math.round(q('#grpo-pan-chart').top), metrics:Math.round(q('#grpo-metrics').top)}}")
            check(f'{tag} part 5: the main picture first, the metric cards below it (R7.12)', fv['grp'] < fv['chart'] < fv['metrics'], fv)
            check(f'{tag} part 5: the run speed defaults to 快 (R3.9)', speed(pg) == 'fast', speed(pg))
            t0 = pg.evaluate('()=>performance.now()')
            a = click_anim(pg, '#grpo-transport [data-act="play"]')
            check(f'{tag} part 5: Train plays the run', a and a['kind'] == 'run', a)
            pg.wait_for_function("()=>window.__grpo.state().steps>=30&&!window.__grpo.animState()", timeout=20000)
            dt = (pg.evaluate('()=>performance.now()') - t0) / 1000
            check(f'{tag} part 5: the whole run ends in about 10 s (R3.9)', 8.5 <= dt <= 12.5, dt)
            ob = text(pg, '#grpo-rail [data-obs]')
            check(f'{tag} part 5: the observation names 44% to 84%; Next enabled', '44%' in ob and '84%' in ob and not any(nexts(pg)), ob)
            check(f'{tag} part 5: no red “blown up” label before part 7 explains it (S17)', pg.evaluate("()=>document.querySelectorAll('#grpo-gsvg .grpo-big-t, #grpo-gsvg .grpo-adv.big').length") == 0)
            out_ = pg.evaluate(SVGIN)
            check(f'{tag} part 5: every chart label stays inside its picture (S16)', not out_, out_)
            al = pg.evaluate(ALIGN)
            check(f'{tag} part 5 (owner item 1, H7): the run buttons and the speed buttons share one height and one baseline in each row', al['n'] == 5 and not al['bad'], al)
            cur = text(pg, '#grpo-rail .step.current')
            # ρ is typeset (MathML), so its text comes out as the math italic 𝜌 with line breaks around it: compare without white space
            curn = ''.join(cur.split()).replace('\U0001d70c', 'ρ')
            check(f'{tag} part 5 (owner item 3, H3): the card says why the clip cannot act with one update per batch',
                  ''.join(Z(lang, 'still exactly the model that wrote the answers, so every ρ is 1', '正在训练的模型和写出这批回答的模型还是同一个，所以每个回答的 ρ 都等于 1').split()) in curn
                  and Z(lang, 'learned a second time', '同一批回答学第二遍时') in cur and Z(lang, 'it limits how far an answer’s probability has already moved', '截断管的不是优势有多大') in cur
                  and pg.evaluate("()=>!!document.querySelector('#grpo-rail .step.current p math')"), cur[:900])   # ρ typeset (since the reader round, inline in the sentence that defines it)
            kinds = pg.evaluate("""()=>{const cs=getComputedStyle(document.documentElement), col=v=>{const d=document.createElement('i'); d.style.color=cs.getPropertyValue(v).trim(); document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
              const C={ok:col('--c-ok'),mix:col('--c-weights'),wrong:col('--c-oom')};
              const rows=[...document.querySelectorAll('#grpo-probs .grpo-prow')].map(r=>{const k=['ok','mix','wrong'].find(x=>r.classList.contains('k-'+x))||''; const f=getComputedStyle(r.querySelector('.grpo-pfill')).backgroundColor;
                const tg=r.querySelector('.grpo-kt'); return {k,fill:f===C[k],tag:tg?getComputedStyle(tg).color===C[k]:null}});
              const leg=['ok','mix','wrong'].map(k=>{const s=document.querySelector('#grpo-p-leg .grpo-sw-k.'+k); return !!s&&getComputedStyle(s).backgroundColor===C[k];});
              return {rows,leg,ticks:document.querySelectorAll('#grpo-probs .grpo-pref').length,tick:!!document.querySelector('#grpo-p-leg .grpo-sw-tick'),legtxt:document.querySelector('#grpo-p-leg').innerText,note:document.querySelector('#grpo-p-note').innerText};}""")
            tr = pg.evaluate("()=>window.__grpo.train('mix',8,'on')")
            last = tr['o'][-1]; want = ['ok' if all(x == 4 for x in o) else ('wrong' if all(x != 4 for x in o) else 'mix') for o in last]
            check(f'{tag} part 5 (owner item 6, H6): after 30 steps each question is coloured by its group (all right green, some right blue, all wrong red), the same in bar, word and legend',
                  [r_['k'] for r_ in kinds['rows']] == want and all(r_['fill'] for r_ in kinds['rows']) and (phone or all(r_['tag'] for r_ in kinds['rows'])) and all(kinds['leg']) and {'ok', 'mix', 'wrong'} <= set(want), (kinds, want))
            check(f'{tag} part 5 (owner item 6, H6): every mark is named next to the chart: the line before training, the bar length, the framed row; no “hollow box” left',
                  kinds['ticks'] == 8 and kinds['tick'] and Z(lang, 'line: before training', '竖线：训练前') in kinds['legtxt'] and Z(lang, 'Bar length', '条的长度') in kinds['note'] and Z(lang, 'framed row', '框起来的一行') in kinds['note']
                  and not any(x in kinds['legtxt'] for x in ('空心', 'outlined', 'filled', '实心', '灰色')), kinds)
            shot(pg, f'{tag}_p5', full=phone); rules(pg, f'{tag} part 5', lang)
            click(pg, BACK); wait_anim(pg)
            a = click_anim(pg, NEXT)
            check(f'{tag} part 5 again: the watched run does not replay and stays at 30 steps (R3.20)', a is None and state(pg)['steps'] == 30, (a, state(pg)))
            a = click_anim(pg, '#grpo-transport [data-act="replay"]')
            check(f'{tag} part 5: 重放 replays the run from step 0', a and a['kind'] == 'run' and state(pg)['steps'] < 5 and Z(lang, 'Replay', '重放') in text(pg, '#grpo-transport [data-act="replay"]'), (a, state(pg)))
            pg.wait_for_function("()=>window.__grpo.state().steps>=30&&!window.__grpo.animState()", timeout=20000)
            click(pg, '#grpo-transport [data-act="reset0"]')
            a = click_anim(pg, '#grpo-transport [data-act="step1"]'); st2 = text(pg, '#grpo-status')
            check(f'{tag} part 5 (H8): while +1 step replays its group, the status says it is sampling, not training', a and a['kind'] == 'group' and a['ph'] == 0 and st2 == Z(lang, 'Sampling', '抽样中'), (a, st2))
            wait_anim(pg)
            check(f'{tag} part 5 (H8): after the replay the status gives the accuracy again', Z(lang, 'Accuracy', '答对率') in text(pg, '#grpo-status'), text(pg, '#grpo-status'))
            # ---- part 6
            go_next(pg)
            check(f'{tag} part 6: Next waits for a second G', all(nexts(pg)) and any(Z(lang, 'another G', '换一个 G') in x for x in why(pg)), why(pg))
            click(pg, '#grpo-rail .seg-ctl[data-key="peq"] button[data-v="2"]')
            ob = text(pg, '#grpo-rail [data-obs]')
            check(f'{tag} part 6: G = 2 gives 50% of the groups all the same; Next enabled', '50%' in ob and not any(nexts(pg)), ob)
            click(pg, '#grpo-rail .seg-ctl[data-key="peq"] button[data-v="64"]')
            ob = text(pg, '#grpo-rail [data-obs]')
            check(f'{tag} part 6: G = 64 names the 8 times the answers, the 98% question and its 31% all-right groups (S8)', Z(lang, '8 times', '8 倍') in ob and '98%' in ob and '31%' in ob, ob)
            shot(pg, f'{tag}_p6', full=phone); rules(pg, f'{tag} part 6', lang)
            # ---- part 7
            a = click_anim(pg, NEXT)
            check(f'{tag} part 7: a new group replays from sampling', a and a['kind'] == 'group' and a['ph'] == 0, a)
            wait_anim(pg)
            nb = pg.evaluate("()=>document.querySelectorAll('#grpo-gsvg .grpo-adv.big').length")
            check(f'{tag} part 7: the wrong answer’s large advantage is red, with its label (R5.11)', nb == 1 and '-2.47' in svgtext(pg, '#grpo-gsvg') and Z(lang, 'blown up by small std', '被小标准差放大') in svgtext(pg, '#grpo-gsvg'), (nb, svgtext(pg, '#grpo-gsvg')[:300]))
            out_ = pg.evaluate(SVGIN)
            check(f'{tag} part 7: the red label and the numbers stay inside the picture, without overlap (S16)', not out_, out_)
            check(f'{tag} part 7: Next waits for the division to be switched off', all(nexts(pg)), why(pg))
            mh = pg.evaluate("()=>{const e=document.querySelector('#grpo-rail .step.current .grpo-math-line .grpo-mhl'); if(!e) return null; const r=e.getBoundingClientRect(); return {txt:e.textContent, w:r.width, h:r.height, frac:!!e.closest('mfrac')}}")
            check(f'{tag} part 7 (owner item 4, R6.34): the advantage formula is shown where the std is divided by, its denominator marked', mh and 'std' in mh['txt'] and mh['frac'] and mh['w'] > 20
                  and Z(lang, 'the denominator of the formula', '也就是公式的分母') in text(pg, '#grpo-rail .step.current'), mh)
            shot(pg, f'{tag}_p7', full=phone); rules(pg, f'{tag} part 7', lang)
            pg.evaluate("()=>{document.querySelector('#grpo-math').open=true}"); pg.wait_for_timeout(250)
            mb = pg.evaluate("()=>{const m=document.querySelector('#grpo-mathbody'); return {eqs:m.querySelectorAll('.grpo-eqs').length, leads:[...m.querySelectorAll('.grpo-mb-lead b')].map(b=>b.textContent), txt:m.innerText}}")
            check(f'{tag} Show the math (H2): four blocks, each with a lead line and aligned chains with this group’s numbers (7 of 8 right: 0.875, 0.354, -2.47)', len(mb['leads']) == 4 and mb['eqs'] >= 6
                  and all(x in ''.join(mb['txt'].split()).replace('\u2212', '-') for x in ('0.875', '0.354', '-2.47', '0.272')), mb['txt'][-400:])
            shot(pg, f'{tag}_math', full=phone)
            pg.evaluate("()=>{document.querySelector('#grpo-math').open=false}")
            a = click_anim(pg, '#grpo-rail .seg-ctl[data-key="std"] button[data-v="off"]')
            check(f'{tag} part 7: switching the division off replays the advantages only (R3.20)', a and a['kind'] == 'group' and a['ph'] == 3, a)
            wait_anim(pg)
            ob = text(pg, '#grpo-rail [data-obs]')
            check(f'{tag} part 7: without the division, -0.88 and no red bar; Next enabled', '-0.88' in ob and pg.evaluate("()=>document.querySelectorAll('#grpo-gsvg .grpo-adv.big').length") == 0 and not any(nexts(pg)), ob)
            a = click_anim(pg, '#grpo-rail .seg-ctl[data-key="std"] button[data-v="on"]')
            check(f'{tag} part 7: switching it back replays nothing (R3.20)', a is None, a)
            a = click_anim(pg, '#grpo-transport [data-act="replay"]')
            check(f'{tag} part 7: 重放 replays the group from sampling', a and a['kind'] == 'group' and a['ph'] == 0, a)
            wait_anim(pg)
            # ---- part 8
            go_next(pg)
            cur = text(pg, '#grpo-rail .step.current')
            check(f'{tag} part 8: 144.5 GB, RLVR, the scenarios button', '144.5 GB' in cur and 'RLVR' in cur and pg.query_selector('#grpo-rail .step.current [data-mode-go="sandbox"], #grpo-rail .step.current [data-modego="sandbox"], #grpo-rail .step.current .toscen, #grpo-rail .step.current button') is not None, cur[:300])
            check(f'{tag} part 8: the RLVR lab is a working link, never called planned (B3)', pg.query_selector('#grpo-rail .step.current a[href="#rlvr"]') is not None and '规划中' not in all_text(pg) and 'planned' not in all_text(pg).lower(), cur[-300:])
            check(f'{tag} part 8: the next-lab footer', pg.evaluate("()=>[...document.querySelectorAll('.next-lab')].some(e=>e.getBoundingClientRect().height>0)"))
            check(f'{tag} part 8: no transport', not vis(pg, '#grpo-transport'))
            check(f'{tag} part 8 (owner item 5, H9): the comparison names what the reward model adds: 16.1 GB, 160.6 GB with a checker, G times as many answers to score',
                  all(x in cur for x in ('16.1 GB', '176.7 GB', '160.6 GB', Z(lang, 'it also scores G times as many answers', '它要打分的回答也是 G 倍'))), cur[:500])
            mn = text(pg, '#grpo-m-note') + ' ' + text(pg, '#grpo-m-title')
            check(f'{tag} part 8 (owner item 5, H9): the memory note keeps the reward model in view and the title says all GPUs together', all(x in mn for x in Z(lang, ('the reward model stays', '16.1 GB', 'all GPUs together'), ('但奖励模型还在', '16.1 GB', '所有 GPU 加起来'))), mn)
            shot(pg, f'{tag}_p8', full=phone); rules(pg, f'{tag} part 8', lang)
            # ---- scenarios
            mode(pg, 'sandbox')
            lead = pg.inner_text('#grpo-rail .railcard > p.small')
            check(f'{tag} G1: the scenarios card leads in with what to pick and what to watch, not “one at a time”', lead == Z(lang, 'Pick a scenario and see how the average accuracy changes as GRPO trains.', '选一个场景，看看 GRPO 训练时平均答对率怎么变。'), lead)
            check(f'{tag} scenarios: five scenarios', pg.evaluate("()=>document.querySelectorAll('#grpo-rail [data-preset]').length") == 5)
            a = click_anim(pg, '#grpo-rail [data-preset="0"]')
            check(f'{tag} scenarios: the mixed set was watched in part 5, so it does not replay (R3.20)', a is None and state(pg)['steps'] == 30, (a, state(pg)))
            ins = text(pg, '#grpo-insight')
            check(f'{tag} scenarios: the insight for the mixed set', '44%' in ins and '84%' in ins, ins)
            a = click_anim(pg, '#grpo-rail [data-preset="1"]')
            check(f'{tag} scenarios: a scenario not yet watched plays its run at 快', a and a['kind'] == 'run' and speed(pg) == 'fast', (a, speed(pg)))
            pg.wait_for_function("()=>window.__grpo.state().steps>=30&&!window.__grpo.animState()", timeout=20000)
            ins = text(pg, '#grpo-insight')
            check(f'{tag} scenarios: the easy set’s insight says most groups were all right', Z(lang, 'Most groups were all right', '大多数组全对') in ins, ins)
            hsh = pg.evaluate("()=>location.hash")
            check(f'{tag} scenarios: the hash carries the setup', hsh == '#grpo~easy_8_on_30_0', hsh)
            for i, key in ((2, ('At first most groups were all wrong', '一开始大多数组全错')), (3, ('With only 2 answers', '只生成 2 个回答')), (4, ('Without the division', '不除以标准差时'))):
                click(pg, f'#grpo-rail [data-preset="{i}"]'); pg.wait_for_function("()=>window.__grpo.state().steps>=30&&!window.__grpo.animState()", timeout=20000)
                ins = text(pg, '#grpo-insight')
                check(f'{tag} scenarios: scenario {i + 1} has its insight', Z(lang, *key) in ins, ins)
                if i == 4: check(f'{tag} scenarios: without the division, 2.4 times, 71% against 84%, under 1 point at the matched step (B1)', '2.4' in ins and '71%' in ins and '84%' in ins and Z(lang, 'less than 1 percentage point', '不到 1 个百分点') in ins, ins)
            shot(pg, f'{tag}_scen', full=phone); rules(pg, f'{tag} scenarios', lang)
            other = 'zh' if lang == 'en' else 'en'
            s0 = state(pg); click(pg, f'.langsw button[data-lang="{other}"]', 300); s1 = state(pg)
            check(f'{tag} scenarios: the language switch keeps the setup', all(s0[k] == s1[k] for k in ('mode', 'set', 'G', 'std', 'steps')), (s0, s1))
            click(pg, f'.langsw button[data-lang="{lang}"]', 300)
            if phone:
                pg.evaluate("()=>{const a=document.querySelector('#grpo-pan-group:not([hidden])')||document.querySelector('#grpo-pan-chart'); window.scrollTo(0,a.getBoundingClientRect().bottom+scrollY+120)}"); pg.wait_for_timeout(800)
                dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return d&&!d.hidden&&d.getBoundingClientRect().height>0?d.innerText:''}")
                check(f'{tag}: the dock shows once the stage scrolls away', bool(dk) and '%' in dk, dk)
                shot(pg, f'{tag}_dock'); pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(200)
            pg.close()
            pg = page(b, w, h, lang, hash_='grpo~easy_8_on_30_0'); st = state(pg)
            check(f'{tag}: the hash round trip opens the same setup in scenarios', st['mode'] == 'sandbox' and (st['set'], st['G'], st['std'], st['steps']) == ('easy', 8, 'on', 30), st)
            # ---- challenge
            mode(pg, 'challenge'); st = state(pg)
            check(f'{tag} challenge: starts at G = 2 on the challenge set, no result before Submit', st['mode'] == 'challenge' and st['G'] == 2 and st['set'] == 'ch' and st['steps'] == 0 and not text(pg, '#grpo-ch-res').strip(), st)
            nores = lambda: '%' not in text(pg, '#grpo-ch-av') and not text(pg, '#grpo-ch-res').strip() and Z(lang, 'Not trained yet', '还没训练') in text(pg, '#grpo-status') and state(pg)['steps'] == 0
            check(f'{tag} challenge: no accuracy, no verdict and no trained curve before Submit (S19)', nores(), (text(pg, '#grpo-ch-av'), text(pg, '#grpo-status'), state(pg)))
            rail = text(pg, '#grpo-rail')
            check(f'{tag} challenge: the statement names the 12% start', '12%' in rail, rail[:400])
            click(pg, '#grpo-rail .seg-ctl[data-key="G"] button[data-v="16"]')
            check(f'{tag} challenge: another G before Submit still shows no result (S19)', nores(), (text(pg, '#grpo-ch-av'), text(pg, '#grpo-status'), state(pg)))
            click(pg, '#grpo-rail .seg-ctl[data-key="G"] button[data-v="2"]')
            a = click_anim(pg, '#grpo-rail [data-act="submit"]')
            check(f'{tag} challenge: Submit trains the 30 steps; no verdict while they run', a and a['kind'] == 'run' and not text(pg, '#grpo-ch-res').strip(), a)
            pg.wait_for_function("()=>window.__grpo.state().steps>=30&&!window.__grpo.animState()", timeout=20000); pg.wait_for_timeout(250)
            res = text(pg, '#grpo-ch-res')
            check(f'{tag} challenge: G = 2 is not met, with a hint and “{Z(lang, "Show the answer", "直接看答案")}”', Z(lang, 'Not yet', '还没达到') in res and vis(pg, '#grpo-rail [data-act="reveal"]') and Z(lang, 'Under 80%', '不到 80%') in text(pg, '#grpo-status')
                  and text(pg, '#grpo-rail [data-act="reveal"]').strip() == Z(lang, 'Show the answer', '直接看答案'), res)
            click(pg, '#grpo-rail .seg-ctl[data-key="G"] button[data-v="16"]')
            check(f'{tag} challenge: a new G after a miss needs a new Submit', not state(pg)['chSubmitted'] and not text(pg, '#grpo-ch-res').strip() and vis(pg, '#grpo-rail [data-act="submit"]'), state(pg))
            click(pg, '#grpo-rail [data-act="submit"]')
            pg.wait_for_function("()=>window.__grpo.state().steps>=30&&!window.__grpo.animState()", timeout=20000); pg.wait_for_timeout(250)
            res = text(pg, '#grpo-ch-res')
            check(f'{tag} challenge: G = 16 is met, but G = 8 does it with fewer answers', Z(lang, 'Goal met', '达成目标') in res and 'G = 8' in res and '1,920' in res, res)
            click(pg, '#grpo-rail .seg-ctl[data-key="G"] button[data-v="8"]')
            res = text(pg, '#grpo-ch-res')
            check(f'{tag} challenge: G = 8 is met with the fewest answers; the chart of all options appears', Z(lang, 'no smaller group', '更小的组都到不了') in res and vis(pg, '#grpo-ch-res .cmp'), res)
            rows = pg.evaluate("()=>[...document.querySelectorAll('#grpo-ch-res .cmp-row')].map(r=>({ok:r.classList.contains('ok'),me:r.classList.contains('me'),tag:!!r.querySelector('.cmp-tag,.cmp-you'),lab:r.querySelector('.cmp-lab').textContent}))")
            note = pg.evaluate("()=>(document.querySelector('#grpo-ch-res .cmp-note')||{}).textContent||''")
            title = pg.evaluate("()=>(document.querySelector('#grpo-ch-res .cmp-title')||{}).textContent||''")
            check(f'{tag} challenge (the owner, batch 8): 5 rows, the baseline before training, the initial G = 2, and G = 8 (your pick), 16, 32, which pass; G = 4 left out',
                  len(rows) == 5 and [r['ok'] for r in rows] == [False, False, True, True, True] and rows[2]['me'] and all(r['ok'] or r['me'] or r['tag'] for r in rows)
                  and Z(lang, 'baseline', '基准') in rows[0]['lab'] and Z(lang, 'initial setup', '初始设置') in rows[1]['lab'] and not any('G = 4' in r['lab'] for r in rows), rows)
            check(f'{tag} challenge: the note counts the one G left out; the title no longer says “for each G”', Z(lang, 'The other G value stays under 80%', '其余 1 个 G 到不了 80%') in note
                  and title and not any(x in title for x in ('each', 'every', '按 G 分', '各个')), (note, title))
            check(f'{tag} challenge: the capability card', vis(pg, '#grpo-rail [data-exits]'))
            shot(pg, f'{tag}_challenge', full=phone); rules(pg, f'{tag} challenge', lang)
            check(f'{tag}: no horizontal overflow at the end', overflow(pg) <= 0, overflow(pg))
            pg.close()
    # ---- 900 wide (H7: the owner's review of 2026-10-09 names 1440, 900 and 390): the control rows line up, chart text stays inside
    if not [x for x in ARGS if x in ('1440', '390')]:
        for lang in LANGS:
            tag = f'{lang}-900'; pg = page(b, 900, 900, lang)
            go_next(pg); click(pg, '#grpo-rail [data-act="gen"]'); wait_anim(pg)
            al = pg.evaluate(ALIGN); check(f'{tag} part 2 (owner item 1, H7): replay and speed buttons aligned', al['n'] == 2 and not al['bad'], al)
            shot(pg, f'{tag}_p2')
            go_next(pg); click(pg, '#grpo-rail [data-pred="far"]')
            go_next(pg); wait_anim(pg); go_next(pg)
            check(f'{tag}: reached part 5', state(pg)['step'] == 4, state(pg))
            click(pg, '#grpo-rail [data-nav="showme"]'); pg.wait_for_function("()=>window.__grpo.state().steps>=30&&!window.__grpo.animState()", timeout=20000)
            al = pg.evaluate(ALIGN); check(f'{tag} part 5 (owner item 1, H7): run buttons aligned', al['n'] == 5 and not al['bad'], al)
            out_ = pg.evaluate(SVGIN); check(f'{tag} part 5 (H7): every chart label inside its picture', not out_, out_)
            check(f'{tag} part 5: no horizontal overflow', overflow(pg) <= 0, overflow(pg))
            shot(pg, f'{tag}_p5'); pg.close()
    b.close()

for h in sorted(set(rule_hits)): print('RULE', h)
check('writing rules, colours, overflow and tooltips', not rule_hits, len(rule_hits))
check('no page errors', not errs, errs[:6])
# ---- G1/G2 (the owner's message of 2026-10-08): option lead-ins instead of “one at a time”, and no sentence that leans on a detail
# of an earlier part. Read from the lab's own source, so both languages are covered.
import re as _re
_SRC = ''.join((P / d / 'grpo.js').read_text() for d in ('labs', 'teasers') if (P / d / 'grpo.js').exists())
_G1 = _re.findall(r'.{0,30}(?:一次只能选|不会叠加|do not stack|[Oo]ne at a time|一次选一).{0,30}', _SRC)
check('G1: no “one at a time / do not stack” note left in the lab or its tile (EN and ZH)', not _G1, _G1[:3])
_NEW = ["In the group in the chart, 2 of the 8 answers are right", "优势等于奖励减去基准。图里这一组 8 个回答里有 2 个对，", "divided by such a small number, its advantage grows to about", ". The more alike a group’s scores, the smaller its std, and the more the odd answer’s advantage is blown up.", "除以这么小的数，它的优势被放大到约", "。一组的分数越一致，标准差越小，和大家不一样的那个回答，优势就被放得越大。", "GRPO needs G answers for every question in every step, so each step generates G times as many answers as PPO.", "GRPO 每一步每道题都要生成 G 个回答，所以每一步生成的回答是 PPO 的 G 倍。", "When a question gets 8 answers and 2 are right, each right answer’s advantage is +1.62.", "一道题生成 8 个回答、2 个对时，每个答对的回答优势是 +1.62。", "About +1.5, close to the 2-of-8 group", "约 +1.5，和 8 个里 2 个对时差不多", "About +3, twice the 2-of-8 group", "约 +3，是 8 个里 2 个对时的 2 倍", "About +0.75, half the 2-of-8 group", "约 +0.75，是 8 个里 2 个对时的一半", "A question answered right half the time almost never has a whole group of 8 score the same.", "答对概率 50% 的题，8 个回答几乎不会整组得分一样。"]
check(f'G1/G2: the new lead-ins and the rewritten sentences are in the lab ({len(_NEW)})', all(x in _SRC for x in _NEW), [x for x in _NEW if x not in _SRC][:3])
_OLD = ["In the group you just generated, 2 of the 8 answers are right", "优势等于奖励减去基准。刚才那一组 8 个回答里有 2 个对，", "further from 0 than any advantage in part 4.", "，比第 4 部分里任何一个优势离 0 都远。", "GRPO needs G answers for every question in every step: in part 5 the policy generated", "GRPO 每一步每道题都要生成 G 个回答。第 5 部分的 30 步里，策略模型一共生成了", "In part 4, 2 of 8 answers were right and each right answer’s advantage was +1.62.", "第 4 部分里，8 个回答 2 个对，每个答对的回答优势是 +1.62。", "About +1.5, close to part 4", "约 +1.5，和第 4 部分差不多", "About +3, twice part 4", "约 +3，是第 4 部分的 2 倍", "About +0.75, half of part 4", "约 +0.75，是第 4 部分的一半", "In part 6, a question answered right half the time almost never had a whole group of 8 score the same.", "第 6 部分里，答对概率 50% 的题，8 个回答几乎不会整组得分一样。"]
check(f'G2: none of the {len(_OLD)} sentences that leaned on an earlier part (or said “one at a time”) is left', not any(x in _SRC for x in _OLD), [x for x in _OLD if x in _SRC][:3])
# ---- H1 and H4 (the owner's message of 2026-10-09), read from the source so both languages are covered
_H1 = _re.findall(r'.{0,30}(?:A = \(r − mean\) ÷|β × \(e<sup>|log π<sub>|√\(\$\{|Σ\(r − mean\)²|\[1 − ε, 1 \+ ε\]|8 × G|n − 1|π<sub>old|u<sub>i|A<sub>|0\.9 to the 8th|0\.9 的 8 次方|空心框|outlined: reference).{0,30}', _SRC)
check('H1: none of the converted plain-text formulas (or the old hollow-box legend) is left in the lab or its tile', not _H1, _H1[:3])
_H4 = _re.findall(r'.{0,30}(?:规划中|还没上线|not built|later lab|以后的 Lab|将来|[Pp]lanned lab).{0,30}', _SRC)
check('H4: no lab called planned or not yet live', not _H4, _H4[:3])
print(f'{n - len(fails)} / {n} passed')
sys.exit(1 if fails else 0)
