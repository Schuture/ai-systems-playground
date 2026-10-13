"""Flow test for the reward-model lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH:
- no page errors, no horizontal overflow, no text wider than its box (R7.8), a tooltip for every help cursor (R1.1);
- the main picture first, the metric cards below it and compact (R7.12); on the phone the parts' bar comes first (m2);
- part 1: no scores and no run result before training (R3.8); the card's own text (not a hover) says what a reward model is, why it is
  needed and how it is used once trained (the owner's review of 2026-10-08, item 2, G3);
- no PPO, KL penalty or “policy” anywhere in the lab, its tooltips or its home tile (item 1); no “one at a time / 一次只能选一个” note,
  a lead-in instead (G1); no sentence that recalls a part by its number (G2); no formula box clips its formula (G5);
- part 2: Next disabled with the why line until a prediction; then σ on the stage, σ(2) = 88% and the verdict with the right answer;
- part 3 (R3.9, R3.20, G6): the run plays on entry at the default speed (Medium), on screen, and ends in about 3 s (about 1.5 s at Fast);
  Back and Next again: no replay; ↺ 重放 replays from step 0; a speed change does not replay; Next waits for the run;
- part 4: Next waits for the slider; Show me adds 2 to every score and the gap stays;
- part 5: the other data set plays a new run (the picture differs; on a phone, where the stage is above the card, its end state at once,
  G6); going back to the data already watched does not replay;
- part 6: Next waits for n = 64; the curves reach only the n the reader has set (R3.8, review S11); the turn is red on the chart and in the
  badge (R5.11), and only there; the “best n” card appears at n = 64;
- Show the math (review B1): no TeX word left as a variable (“big”, “left”, “right”, “pick”); every formula's symbols defined;
- before part 5 the stage names neither the data's length nor balanced pairs; before part 7 no balanced pairs (review S7);
- part 2: after the pick the verdict is on screen (R3.10, review S12);
- part 7: a new number of balanced pairs plays its run with a replay control (on a phone its end state at once, G6); an option already
  watched does not replay;
- part 8: three misconceptions, “See the scenarios”, the next lab;
- Scenarios: four, single choice, an insight each, the end state at once (G6); the URL carries the setup and a fresh page restores it; a change under Settings
  does not move the page (R3.2);
- the challenge: Settings locked, two choices in the card; no result before Submit (R3.8); the three feedback tiers (R4.4); “Show the answer”;
  a brief chart (the owner's review of batch 8, 2026-10-07): your pick, the initial setup and the smallest change, a note that counts the
  11 options that miss, and the 8 passing options behind a closed “See all settings that pass (8)”; the capability card; the language
  switch keeps the answer;
- the mini result bar at 390 once the chart scrolls away;
- the writing rules on every visible text and tooltip (Chinese spacing and punctuation, two colons, no company names, no English
  “reward model / policy” inside Chinese), and red only for what goes wrong, green only for the goal (R5.6).
R6.11: the model is “小奖励模型 / the small reward model”; “玩具 / toy” only in the fidelity label “玩具实跑 / Toy run” (the older labs' check).
Screenshots: shots/reward-model/.
Run: python3 tests/reward-model_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys, time
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
(P / 'out' / 'reward-model').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/reward-model.html', '--only', 'reward-model'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'reward-model' / 'flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'reward-model.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'reward-model' / 'flow-wrapped.html')
SH = P / 'shots' / 'reward-model'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, hash_='reward-model'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__rm.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#rm-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#rm-rail [data-next-why]')].filter(p=>!p.hidden&&getComputedStyle(p).visibility!=='hidden'&&p.getClientRects().length).map(p=>p.textContent)")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__rm||!window.__rm.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel, ms=250):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(ms); settle(pg)


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#rm-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def to_end(pg):
    """drag the run's scrubber to its last step (finishes a run without waiting)"""
    pg.evaluate("()=>{const s=document.querySelector('#rm-player input[data-key=\"scrub\"]'); s.value=s.max; s.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(200)


def playing_after(pg, sel):
    """click, then report whether a run is playing from (near) step 0 two frames later"""
    return pg.evaluate("""async (sel)=>{document.querySelector(sel).click(); await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
      const s=window.__rm.state(); return {playing:s.playing, t:s.tShow};}""", sel)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


FIRST = """()=>{const q=s=>document.querySelector(s).getBoundingClientRect(), st=q('#rm-stage'), pr=q('#rm-pair'), m=q('#rm-metrics');
  return {stage:Math.round(st.top), pair:Math.round(pr.top), pairEnd:Math.round(pr.bottom), metrics:Math.round(m.top), metricsH:Math.round(m.height), ih:innerHeight};}"""
RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · (?![0-9])/,'“·” joining parts'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy”'],[/亿|万/,'亿 or 万'],[/OpenAI|Anthropic|DeepMind|Google|Meta\\b|Microsoft/,'company name'],
   [/[\\u4e00-\\u9fff][^。]*\\b(?:reward model|value model|policy)\\b/i,'English term inside Chinese (S2)'],
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour (R5.7)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, .fid-tag, a[href*="github.com/Schuture"]')||!e.getClientRects().length) continue; const t=n.textContent;
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
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|rm-bad|rm-over|warn|wrong|ex-res/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#rm-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#rm-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#rm-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
rule_hits = []

# ---- the owner's review of 2026-10-08 (reviews/owner-b9-2026-10-08.md): item 1 (no PPO), G1 (lead-ins), G2 (no part numbers), G5 (formulas)
SRC_JS = (P / 'labs' / 'reward-model.js').read_text(encoding='utf-8') + (P / 'teasers' / 'reward-model.js').read_text(encoding='utf-8')
CODE_JS = re.sub(r'(?<=\s)//[^\n]*', '', re.sub(r'/\*.*?\*/', '', SRC_JS, flags=re.S))
NO_PPO_RE = r'\bPPO\b|\bKL\b|策略模型|\b[Pp]olic(?:y|ies)\b|强化学习|[Rr]einforcement learning|参考模型|\b[Rr]eference model'   # case-sensitive (“opposite”)
G1_RE = r'一次只能选|一次选一个|不会叠加|do not stack|one at a time'
G2_RE = r'第 ?\d+ ?部分|第 ?\$\{[^}]*\} ?部分|\b[Pp]arts? (?:\d|\$\{)|分步引导用的数据|比分步引导里|than in the guided parts|data of the guided parts'
check('item 1: no PPO, KL penalty, reference model or “policy” in the lab’s or the tile’s strings', not re.findall(NO_PPO_RE, CODE_JS), re.findall(NO_PPO_RE, CODE_JS)[:8])
check('G1: no “one at a time / 一次只能选一个 / 不会叠加” note in the source', not re.findall(G1_RE, CODE_JS, re.I), re.findall(G1_RE, CODE_JS, re.I)[:5])
check('G2: no string recalls a part by its number', not re.findall(G2_RE, CODE_JS), [CODE_JS[max(0, m.start() - 40):m.end() + 20] for m in re.finditer(G2_RE, CODE_JS)][:5])
NOPPO = """([sel,src])=>{const root=document.querySelector(sel); if(!root) return []; const c=root.cloneNode(true); c.querySelectorAll('.next-lab').forEach(e=>e.remove());
  const re=new RegExp(src), out=[]; const m=c.textContent.match(new RegExp(src,'g')); if(m) out.push('text: '+[...new Set(m)].join(', '));
  c.querySelectorAll('[data-tipx],[aria-label]').forEach(e=>{const x=(e.getAttribute('data-tipx')||'')+' '+(e.getAttribute('aria-label')||''); if(re.test(x)) out.push('attribute: '+x.trim().slice(0,80));});
  return out;}"""
FXCLIP = """()=>[...document.querySelectorAll('#rm-app .fx')].filter(e=>e.getClientRects().length).map(e=>{const cs=getComputedStyle(e);
  return {h:e.scrollHeight-e.clientHeight, pt:parseFloat(cs.paddingTop), pb:parseFloat(cs.paddingBottom), t:e.textContent.slice(0,30)};}).filter(x=>x.h>0||x.pt<4||x.pb<4)"""


def owner_b9(pg, where):
    """item 1 and G1 on everything the lab has rendered (rail, stage, Settings, Connect, the header's popovers; not the kit's next-lab
    footer, which names the next lab), and G5 on every formula box on screen"""
    for sel in ('#rm-app', '#dock'):
        hit = pg.evaluate(NOPPO, [sel, NO_PPO_RE]); check(f'{where}: no PPO, KL penalty or policy in {sel} (item 1)', not hit, hit[:3])
    hit = pg.evaluate(NOPPO, ['#rm-app', '(?:' + G1_RE + ')|[Oo]ne at a time']); check(f'{where}: no “one at a time” note (G1)', not hit, hit[:3])
    fx = pg.evaluate(FXCLIP); check(f'{where}: every formula box keeps its whole formula, with room above and below (G5)', not fx, fx[:3])


# the owner's clarification (2026-10-07 23:37 UTC): no hover only expands a shown number to its exact value; Show the math may write one out
EXACT_HOVER = """r=>{const root=document.querySelector(r), out=[]; if(!root) return out;
  for(const e of root.querySelectorAll('[data-tipx],[data-tip]')){if(e.closest('.mathbody')) continue; let t=e.getAttribute('data-tipx')||''; const k=e.getAttribute('data-tip');
    if(k&&window.ASP) t+=' '+(window.ASP.t(k)||''); if(/精确值是|Exactly/.test(t)) out.push(t.slice(0,100));}
  for(const e of root.querySelectorAll('.num-big')) if(!e.closest('.mathbody')&&(e.hasAttribute('data-tipx')||e.closest('.def'))) out.push('hover on '+e.textContent);
  return out;}"""


# H1 (the owner, 2026-10-09): no formula written as plain text (text, hovers, the tip on screen), no <sub>/<sup> outside code, and no TeX command left as a word
PLAINRE = r'σ\(\(|σ\(r|σ\([0-9]\)|−log σ|ln\(|w · φ|φ = \[|∂loss|gap = |分差 = |÷ 150|P\(person|P\(人选'
PLAINF = '''([r,re])=>{const root=document.querySelector(r), out=[]; if(!root) return out; const rx=new RegExp(re); let n;
  const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
  while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('math,pre,code,script,style')) continue; if(rx.test(n.textContent)) out.push('text: '+n.textContent.trim().slice(0,60));}
  for(const e of root.querySelectorAll('sub,sup')) if(!e.closest('[data-endnotes],pre,code')) out.push('sub/sup: '+e.parentElement.textContent.trim().slice(0,40));
  for(const e of root.querySelectorAll('[data-tipx]')){const t=e.getAttribute('data-tipx'); if(rx.test(t)) out.push('hover: '+t.slice(0,60));}
  const tip=document.getElementById('tip'); if(tip&&!tip.hidden){const w=document.createTreeWalker(tip,NodeFilter.SHOW_TEXT); while((n=w.nextNode())){if(n.parentElement.closest('math')) continue; if(rx.test(n.textContent)) out.push('tip: '+n.textContent.trim().slice(0,60));}}
  for(const m of document.querySelectorAll('math')) for(const i of m.querySelectorAll('mi')) if(/^(frac|sqrt|left|right|hat|text|cdot|times|div|approx|sum|mathrm|operatorname|quad|beta|pi|sigma|rho|theta|bar|tilde|overline|langle|rangle|mid|le|ge|big|Big|partial)$/.test(i.textContent)) out.push('TeX word: '+i.textContent+' in '+m.textContent.slice(0,40));
  return out;}'''
EQJS = '''(sel)=>{const b=document.querySelector(sel); const eqs=[...b.querySelectorAll('.rm-eq')].map(e=>{const xs=[...e.querySelectorAll('.rm-eq-o math')].map(m=>Math.round(m.getBoundingClientRect().left));
    return {name:((e.querySelector('.rm-eq-n')||{}).textContent||'').trim().slice(0,24), lines:[...e.querySelectorAll('.fx')].map(m=>m.textContent.replace(/\\s+/g,'')), aligned:new Set(xs).size<=1};});
  const hs=[...b.querySelectorAll('.rm-mh')].slice(1); return {eqs, rules:hs.length, ruleOk:hs.every(h=>{const cs=getComputedStyle(h); return cs.borderTopWidth==='1px'&&cs.borderTopStyle==='solid';})};}'''


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
    ms = pg.evaluate(MSCROLL, '#rm-app')
    check(f'{where}: no row of Show the math scrolls sideways at {pg.viewport_size["width"]} px (the owner, 2026-10-09; R6.37)', not ms, ms[:3])


def rules(pg, where, lang):
    math_scroll(pg, where)
    pf = pg.evaluate(PLAINF, ['#rm-app', PLAINRE]); check(f'{where}: H1: no formula written as plain text; every formula rendered (no TeX word)', not pf, pf[:3])
    hx = pg.evaluate(EXACT_HOVER, '#rm-app'); check(f'{where}: no exact-value hover outside Show the math', not hx, hx[:3])
    for sel in ('#rm-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')
    for h in pg.evaluate(OVER): rule_hits.append(f'{where}: text wider than its box: {h}')
    for h in pg.evaluate(HELP): rule_hits.append(f'{where}: help cursor without a tooltip: {h}')
    o = overflow(pg)
    if o > 0: rule_hits.append(f'{where}: horizontal overflow {o}px')


with sync_playwright() as pw:
    b = pw.chromium.launch()
    first = True
    for lang in LANGS:
        for (w, h) in SIZES:
            phone = w <= 760; tag = f'{lang}-{w}'
            pg = page(b, w, h, lang)
            st = state(pg)
            check(f'{tag}: opens on part 1 of the guided tour', st['mode'] == 'guided' and st['step'] == 0, st)
            fv = pg.evaluate(FIRST)
            check(f'{tag}: the main picture leads the stage; the metric cards sit below it and are compact (R7.12)',
                  fv['metrics'] >= fv['pairEnd'] and fv['metricsH'] <= 150 and (phone or fv['pairEnd'] <= fv['ih']), fv)
            if phone:
                fs = pg.evaluate("()=>{const p=document.querySelector('#rm-rail .pbar'), g=document.querySelector('#rm-pair'); return {bar:!!p&&p.getBoundingClientRect().top<innerHeight, chart:!!g&&g.getBoundingClientRect().top<innerHeight*1.6}}")
                check(f'{tag}: the phone first screen has the parts bar, the picture follows it', fs['bar'] and fs['chart'], fs)
            # ---- part 1: nothing trained yet
            t1 = text(pg, '#rm-stage')
            check(f'{tag} part 1: no scores and no run result before training (R3.8)', not vis(pg, '#rm-pair .rm-sc') and '%' not in text(pg, '#rm-status') and '%' not in text(pg, '#rm-metrics'), (text(pg, '#rm-status'), text(pg, '#rm-metrics')))
            check(f'{tag} part 1: the small reward model is named with its size (R6.11)', Z(lang, 'small reward model with 3 parameters', '只有 3 个参数的小奖励模型') in text(pg, '#rm-rail .step.current'))
            names = text(pg, '#rm-st-sub') + ' ' + text(pg, '#rm-metrics')
            check(f'{tag} part 1: the stage names no setting the page has not explained yet (review S7)', not any(x in names for x in ('balanced', 'follows quality', '长短平衡', '跟着质量')), names)
            c1 = text(pg, '#rm-rail .step.current')
            need = Z(lang, ['an LLM', 'the chat model after SFT', 'sometimes a smaller one', 'outputs one score', 'scores any new response', 'the best of several responses', 'later training'],
                     ['也是一个 LLM', 'SFT 之后的聊天模型', '有时用一个小一些的', '只输出一个分数', '自动打分', '挑出最好的', '后续训练'])
            check(f'{tag} part 1: the card’s own text says what a reward model is, why it is needed and how it is used (item 2, G3)', all(x in c1 for x in need), [x for x in need if x not in c1])
            lead1 = pg.evaluate("()=>[...document.querySelectorAll('#rm-rail .step.current, #rm-rail .pbar')].map(e=>e.innerText).join(' ')")
            check(f'{tag} part 1: the opening says a model learns a preference from labelled pairs, and names no PPO (item 1)', 'PPO' not in lead1 and Z(lang, 'learns people’s preference from these pairs', '从这些偏好对里学到人的偏好') in c1, lead1[:200])
            owner_b9(pg, f'{tag} part 1')
            shot(pg, f'{tag}_p1', full=phone); rules(pg, f'{tag} part 1', lang)
            # ---- part 2: the prediction
            click(pg, '#rm-rail [data-nav="next"]')
            check(f'{tag} part 2: Next disabled until a prediction, with the why line', all(nexts(pg)) and any(Z(lang, 'Pick an answer', '选一个答案') in x for x in why(pg)), (nexts(pg), why(pg)))
            check(f'{tag} part 2: the formula on its own line; no σ chart, no answer before the pick (R3.8)', vis(pg, '#rm-rail .step.current .fx') and not vis(pg, '#rm-sig') and '88%' not in text(pg, '#rm-app'))
            click(pg, '#rm-rail [data-pred="b"]')
            ob = text(pg, '#rm-rail [data-obs]')
            check(f'{tag} part 2: after the pick, σ(2) = 88%, the verdict with the right answer, the σ chart', '88%' in ob and Z(lang, 'You predicted', '你预测的是') in ob and Z(lang, '80% to 95%', '80% 到 95%') in ob and vis(pg, '#rm-sig') and not any(nexts(pg)), ob[:200])
            check(f'{tag} part 2: still no trained scores on the stage', not vis(pg, '#rm-pair .rm-sc'))
            vy = pg.evaluate("()=>{const p=[...document.querySelectorAll('#rm-rail [data-obs] p')].find(x=>/You predicted|你预测的是/.test(x.textContent)); if(!p) return null; const r=p.getBoundingClientRect(); return {top:Math.round(r.top), bottom:Math.round(r.bottom), ih:innerHeight}}")
            check(f'{tag} part 2: after the pick the verdict is on screen (R3.10, review S12)', vy and vy['top'] >= 0 and vy['bottom'] <= vy['ih'], vy)
            owner_b9(pg, f'{tag} part 2')
            shot(pg, f'{tag}_p2', full=phone); rules(pg, f'{tag} part 2', lang)
            # ---- part 3: the run
            p3 = playing_after(pg, '#rm-rail [data-nav="next"]'); t0 = time.time()
            st = state(pg)
            check(f'{tag} part 3: the run plays on entry, from step 0, at the default speed, Medium (R3.9, G6)', p3['playing'] and p3['t'] < 20 and st['spd'] == 'mid', (p3, st['spd']))
            settle(pg, 1500)   # the kit brings the parts bar under the top bar within about 0.7 s of Next; the run is still playing then
            onv = pg.evaluate("()=>{const r=document.querySelector('#rm-tr-charts').getBoundingClientRect(), p=document.querySelector('#rm-player').getBoundingClientRect(); return {top:Math.round(r.top), bottom:Math.round(p.bottom), ih:innerHeight, playing:window.__rm.state().playing}}")
            check(f'{tag} part 3: while it plays, the run (its charts and its player) is on screen (G6)', onv['playing'] and onv['top'] >= 0 and onv['bottom'] <= onv['ih'], onv)
            check(f'{tag} part 3: Next waits for the run, with its why line', all(nexts(pg)) and any(Z(lang, 'Play the training', '把训练播放完') in x for x in why(pg)), why(pg))
            if first:
                pg.wait_for_function("()=>!window.__rm.state().playing", timeout=20000); dt = time.time() - t0
                check(f'{tag} part 3: the whole run of 200 steps ends in about 3 s at the default speed (G6)', 2.0 <= dt <= 4.5, dt)
            else:
                to_end(pg)
            pg.wait_for_timeout(300); settle(pg)
            ob = text(pg, '#rm-rail [data-obs]')
            check(f'{tag} part 3: the result: loss 0.69 → 0.52, accuracy 75%, ceiling 81%, ±4 points; Next enabled', '0.69' in ob and '0.52' in ob and '75%' in ob and '81%' in ob and '±4' in ob and not any(nexts(pg)), ob[:300])
            check(f'{tag} part 3: the replay control is named 重放 / Replay (R3.20)', Z(lang, 'Replay', '重放') in text(pg, '#rm-player'))
            owner_b9(pg, f'{tag} part 3')
            shot(pg, f'{tag}_p3', full=phone); rules(pg, f'{tag} part 3', lang)
            pg.evaluate("()=>{document.getElementById('rm-math').open=true}"); pg.wait_for_timeout(150)
            mi = pg.evaluate("()=>[...document.querySelectorAll('#rm-mathbody math mi')].map(e=>e.textContent).filter(t=>t.length>1)")
            nfx = pg.evaluate("()=>[...document.querySelectorAll('#rm-mathbody .fx')].filter(e=>e.querySelector('math')).length")
            ndef = pg.evaluate("()=>document.querySelectorAll('#rm-mathbody .rm-defs li').length")
            # H2 (the owner, 2026-10-09): the two scores, Bradley–Terry and the loss are named on the left with the numbers under the formula
            # (4 named chains, 12 formula lines with the gradient step and best-of-n); r_chosen and r_rejected are now defined in the sentence under the loss
            eq = pg.evaluate(EQJS, '#rm-mathbody')
            check(f'{tag} Show the math: twelve formula lines, no TeX word left as a variable (big, left, right, pick), symbols defined beside them (review B1, H2)',
                  nfx == 12 and not [x for x in mi if x not in ('log', 'exp', 'max', 'min', 'mean', 'ln')] and ndef >= 7, (nfx, mi, ndef))
            check(f'{tag} Show the math: H2: the scores of A and B, P(person picks A) and the loss each named on the left, their “=” lined up, a hairline above each block (H5)',
                  len(eq['eqs']) == 4 and all(e['name'] and e['aligned'] for e in eq['eqs']) and eq['eqs'][2]['lines'][0] == 'σ(rA−rB)' and eq['rules'] >= 4 and eq['ruleOk'], eq)
            fx = pg.evaluate(FXCLIP); check(f'{tag} Show the math: every formula box keeps its whole formula, the fraction included (G5)', not fx, fx[:3])
            if not phone: shot(pg, f'{tag}_math')
            pg.evaluate("()=>{document.getElementById('rm-math').open=false}"); pg.wait_for_timeout(100)
            click(pg, '#rm-rail [data-nav="back"]')
            p3b = playing_after(pg, '#rm-rail [data-nav="next"]')
            check(f'{tag} part 3: back and forth, the run already watched is not replayed (R3.20)', not p3b['playing'] and p3b['t'] == 200, p3b)
            click(pg, '#rm-player .seg-ctl[data-key="spd"] button[data-v="fast"]')
            check(f'{tag} part 3: a speed change does not replay', not state(pg)['playing'] and state(pg)['tShow'] == 200)
            t0 = time.time(); rp = playing_after(pg, '#rm-player [data-act="play"]')
            check(f'{tag} part 3: 重放 plays the run again from step 0', rp['playing'] and rp['t'] < 20, rp)
            if first:
                pg.wait_for_function("()=>!window.__rm.state().playing", timeout=20000); dt = time.time() - t0
                check(f'{tag} part 3: at Fast the run ends in about 1.5 s (G6)', 0.9 <= dt <= 2.6, dt)
                first = False
            click(pg, '#rm-player .seg-ctl[data-key="spd"] button[data-v="mid"]')
            to_end(pg); settle(pg)
            # ---- part 4
            click(pg, '#rm-rail [data-nav="next"]')
            check(f'{tag} part 4: Next waits for the slider', all(nexts(pg)) and any(Z(lang, 'Move the slider', '拖动滑块') in x for x in why(pg)), why(pg))
            click(pg, '#rm-rail [data-nav="showme"]')
            ob = text(pg, '#rm-rail [data-obs]')
            check(f'{tag} part 4: Show me adds 2: the gap and the chance stay; TRL’s centring term', '2.15' in ob and '90%' in ob and 'center_rewards_coefficient' in ob and not any(nexts(pg)), ob[:300])
            if phone:   # the integrator's note (00:12 UTC): the phone's copy of the slider keeps its number with its value, also while it has focus (a drag)
                ps = pg.evaluate("""()=>{const e=document.querySelector('#rm-ph-shift-in'); if(!e) return null; const o=()=>e.closest('.inline-ctl,.ctl').querySelector('output').textContent.replace('−','-');
                  const a={v:e.value, out:o()}; e.focus(); e.value=-1; e.dispatchEvent(new Event('input',{bubbles:true})); const b={v:e.value, out:o(), fill:e.style.getPropertyValue('--fill')};
                  e.value=2; e.dispatchEvent(new Event('input',{bubbles:true})); e.dispatchEvent(new Event('change',{bubbles:true})); e.blur(); return {a,b,c:{v:e.value,out:o()}}}""")
                check(f'{tag} part 4: the phone’s slider shows +2 after Show me, and its number follows it while it moves', bool(ps) and ps['a'] == {'v': '2', 'out': '+2'}
                      and ps['b']['out'] == '-1' and ps['c']['out'] == '+2', ps)
            shot(pg, f'{tag}_p4', full=phone); rules(pg, f'{tag} part 4', lang)
            # ---- part 5
            click(pg, '#rm-rail [data-nav="next"]')
            check(f'{tag} part 5: Next waits for the other data', all(nexts(pg)), why(pg))
            p5 = playing_after(pg, '#rm-rail [data-lever="none"]')
            if phone:   # G6: the stage is above the card on a phone, so the new run shows its end state at once
                check(f'{tag} part 5: other data, the new run shows its end state at once (G6, phone)', not p5['playing'] and p5['t'] == 200 and vis(pg, '#rm-player [data-act="play"]'), p5)
            else:
                check(f'{tag} part 5: other data, a new run plays (the picture differs)', p5['playing'] and p5['t'] < 20, p5)
            to_end(pg); settle(pg)
            ob = text(pg, '#rm-rail [data-obs]')
            check(f'{tag} part 5: the length weight falls to almost nothing; AlpacaEval’s 64%', '0.51' in ob and '64%' in ob and not any(nexts(pg)), ob[:300])
            p5b = playing_after(pg, '#rm-rail [data-lever="tied"]')
            check(f'{tag} part 5: back to the data already watched: no replay (R3.20)', not p5b['playing'] and p5b['t'] == 200, p5b)
            owner_b9(pg, f'{tag} part 5')
            shot(pg, f'{tag}_p5', full=phone); rules(pg, f'{tag} part 5', lang)
            # ---- part 6
            click(pg, '#rm-rail [data-nav="next"]')
            check(f'{tag} part 6: Next waits for n = 64, nothing red yet', all(nexts(pg)) and not pg.evaluate("()=>document.querySelector('#rm-status').classList.contains('bad')"))
            p6 = pg.evaluate("()=>({pts:[...document.querySelectorAll('#rm-bon path.rm-line')].map(e=>(e.getAttribute('d').match(/[ML]/g)||[]).length), red:document.querySelectorAll('#rm-bon .rm-bad').length, best:!!document.querySelector('#rm-metrics [data-k=\"bestn\"]')})")
            check(f'{tag} part 6: at n = 1 the curves are not drawn ahead, no red, no “best n” card (R3.8, review S11)', all(x <= 1 for x in p6['pts']) and p6['red'] == 0 and not p6['best'], p6)
            # 可选 9 (Codex review b9v3): the pair card names who chose, and the reward model's verdict is a separate line under it
            pc = pg.evaluate("""()=>{const a=document.querySelector('#rm-pair .rm-anss'), v=document.querySelector('#rm-pair .rm-verdict');
              return {tags:[...document.querySelectorAll('#rm-pair .rm-tag')].map(e=>e.textContent.trim()), tips:[...document.querySelectorAll('#rm-pair .rm-tag')].map(e=>e.getAttribute('data-tipx')||''),
                vd:v?v.textContent.trim():'', v:v?v.dataset.verdict:'', sc:[...document.querySelectorAll('#rm-pair .rm-sc')].map(e=>e.textContent.trim()),
                sep:!!(a&&v&&v.getBoundingClientRect().top>=a.getBoundingClientRect().bottom-1)}}""")
            check(f'{tag} part 6: the pair says the person chose A and not B; a line of its own says the reward model scores B higher (可选 9)',
                  pc['tags'] == [Z(lang, 'the person chose A', '标注的人选了 A'), Z(lang, 'the person did not choose B', '标注的人没选 B')] and pc['v'] == 'disagree'
                  and Z(lang, 'scores B higher', '却给 B 打了更高的分') in pc['vd'] and pc['sep'] and len(pc['sc']) == 2
                  and float(pc['sc'][1].split('=')[1].replace('−', '-')) > float(pc['sc'][0].split('=')[1].replace('−', '-')), pc)
            check(f'{tag} part 6: the tags name the pair with the S2 words (chosen / rejected response)', all(Z(lang, 'the chosen response', '选中的回答') in t and Z(lang, 'the rejected one', '落选的回答') in t for t in pc['tips']), pc['tips'])
            shot(pg, f'{tag}_p6_n1', full=phone)
            pg.evaluate("()=>{const s=document.querySelector('#rm-ic-n-in'); s.value=6; s.dispatchEvent(new Event('input',{bubbles:true})); s.dispatchEvent(new Event('change',{bubbles:true}));}"); pg.wait_for_timeout(300); settle(pg)
            ob = text(pg, '#rm-rail [data-obs]')
            red = pg.evaluate("()=>({seg:document.querySelectorAll('#rm-bon path.rm-bad').length, dot:document.querySelectorAll('#rm-bon circle.rm-bad').length, badge:document.querySelector('#rm-status').classList.contains('bad')})")
            check(f'{tag} part 6: n = 64: the score still rises, the quality fell from 6.95 to 6.68; Next enabled', '6.95' in ob and '6.68' in ob and 'reward hacking' in ob and not any(nexts(pg)), ob[:300])
            check(f'{tag} part 6: the turn is red on the chart, the dot and the badge (R5.11)', red['seg'] == 1 and red['dot'] == 1 and red['badge'], red)
            check(f'{tag} part 6: at n = 64 the “best n” card shows 16', vis(pg, '#rm-metrics [data-k="bestn"]') and '16' in text(pg, '#rm-metrics [data-k="bestn"]'), text(pg, '#rm-metrics'))
            owner_b9(pg, f'{tag} part 6')
            shot(pg, f'{tag}_p6', full=phone); rules(pg, f'{tag} part 6', lang)
            # ---- part 7
            click(pg, '#rm-rail [data-nav="next"]')
            c7 = text(pg, '#rm-rail .step.current')
            check(f'{tag} part 7: a lead-in names what to pick and what to watch (G1); the first option no longer cites part 3 (G2)',
                  Z(lang, 'Pick an amount and see how the true quality of the pick at n = 256 changes.', '选一个数量，看看 n = 256 时挑出的回答，真实质量怎么变。') in c7
                  and Z(lang, 'only the original 1,000 pairs', '只用原来的 1,000 对') in c7, c7[:400])
            p7 = playing_after(pg, '#rm-rail [data-lever="50"]')
            check(f'{tag} part 7: 50 balanced pairs, a new run {"shows its end state at once (G6, phone)" if phone else "plays"} with its replay control',
                  (not p7['playing'] and p7['t'] == 200 if phone else p7['playing']) and vis(pg, '#rm-player [data-act="play"]'), p7)
            to_end(pg); settle(pg)
            ob = text(pg, '#rm-rail [data-obs]')
            check(f'{tag} part 7: the length weight falls to 0.18; 8.72 against 6.40 at n = 256; no fix that needs PPO (item 1)', '0.18' in ob and '8.72' in ob and '6.40' in ob and not re.search(NO_PPO_RE, ob) and not vis(pg, '#rm-bon path.rm-bad'), ob[:400])
            p7b = playing_after(pg, '#rm-rail [data-lever="0"]')
            check(f'{tag} part 7: back to the initial setup already watched: no replay', not p7b['playing'], p7b)
            click(pg, '#rm-rail [data-lever="200"]'); to_end(pg); settle(pg)
            ob = text(pg, '#rm-rail [data-obs]')
            check(f'{tag} part 7: 200 pairs: the length weight below 0, accuracy 75% → 69%', '75%' in ob and '69%' in ob and '−0.21' in ob, ob[:300])
            owner_b9(pg, f'{tag} part 7')
            shot(pg, f'{tag}_p7', full=phone); rules(pg, f'{tag} part 7', lang)
            # ---- part 8
            click(pg, '#rm-rail [data-nav="next"]')
            cur = text(pg, '#rm-rail .step.current')
            check(f'{tag} part 8: three misconceptions, See the scenarios, the next lab', cur.count('✗') == 3 and vis(pg, '#rm-rail [data-mode-go="sandbox"]') and pg.evaluate("()=>!!document.querySelector('#rm-rail .next-lab')"), cur[:200])
            rec = Z(lang, ['so a score on its own says nothing', 'add pairs that break them', 'Check the picks themselves'], ['一个分数单独拿出来说明不了好坏', '补上能打破它的偏好对', '直接检查挑出来的回答'])
            check(f'{tag} part 8: the recap keeps each rule and says what follows, with no detail of an earlier part (G2)', all(x in cur for x in rec) and not re.search(G2_RE, cur) and not re.search(r'[Ii]n part|第 \d 部分|示例 1|example 1', cur), cur[:300])
            owner_b9(pg, f'{tag} part 8')
            shot(pg, f'{tag}_p8', full=phone); rules(pg, f'{tag} part 8', lang)
            # ---- Scenarios
            mode(pg, 'sandbox')
            check(f'{tag} scenarios: four', pg.evaluate("()=>document.querySelectorAll('#rm-rail [data-preset]').length") == 4)
            check(f'{tag} scenarios: a lead-in says what to pick and what to watch (G1)', Z(lang, 'Pick a scenario and see how much weight the reward model gives length', '选一个场景，看看奖励模型给长度多大的权重') in text(pg, '#rm-rail .railcard'), text(pg, '#rm-rail .railcard')[:200])
            for k, sc in enumerate(['def', 'flat', 'noisy', 'small']):
                click(pg, f'#rm-rail [data-preset="{k}"]', 300)
                sk = state(pg)
                check(f'{tag} scenario {sc}: the end state at once (G6)', not sk['playing'] and sk['tShow'] == 200, (sk['playing'], sk['tShow']))
                to_end(pg) if sk['playing'] else None
                settle(pg)
                pr = pg.evaluate("()=>[...document.querySelectorAll('#rm-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
                check(f'{tag} scenario {sc}: single choice, its insight shown', pr.count('true') == 1 and pr[k] == 'true' and vis(pg, '#rm-insight'), pr)
                shot(pg, f'{tag}_sb_{sc}', full=phone); rules(pg, f'{tag} scenario {sc}', lang)
            ins = text(pg, '#rm-insight')
            owner_b9(pg, f'{tag} scenarios')
            check(f'{tag} scenario small: the weights follow chance, best-of-n below a random pick, accuracy 75%', Z(lang, 'chance in the sample', '偶然') in ins and Z(lang, 'worse responses than a random pick', '比随便拿一个还差') in ins and '75%' in ins, ins[:300])
            p_sb = playing_after(pg, '#rm-rail [data-preset="3"]')
            check(f'{tag} scenarios: the same scenario again: no replay (R3.20)', not p_sb['playing'], p_sb)
            if not phone:
                pg.evaluate("()=>document.querySelector('#rm-controls').scrollIntoView({block:'center'})"); pg.wait_for_timeout(300)
                y0 = pg.evaluate("()=>document.querySelector('#rm-ctl-bal').getBoundingClientRect().top")
                pg.evaluate("()=>document.querySelector('#rm-ctl-bal button[data-v=\"25\"]').click()"); pg.wait_for_timeout(300)
                y1 = pg.evaluate("()=>document.querySelector('#rm-ctl-bal').getBoundingClientRect().top")
                check(f'{tag} scenarios: a change under Settings does not move the page (R3.2)', abs(y1 - y0) < 1.5 and state(pg)['bal'] == 25, (y0, y1))
            hs = pg.evaluate('()=>location.hash')
            check(f'{tag} scenarios: the URL carries the setup', hs.startswith('#reward-model~tied_1_50_'), hs)
            pg2 = page(b, w, h, lang, hash_=hs[1:])
            s2 = state(pg2)
            check(f'{tag} scenarios: a fresh page restores the setup', s2['mode'] == 'sandbox' and (s2['rel'], s2['tau'], s2['N'], s2['bal']) == (state(pg)['rel'], state(pg)['tau'], state(pg)['N'], state(pg)['bal']), s2)
            pg2.close()
            # ---- the challenge
            mode(pg, 'challenge')
            check(f'{tag} challenge: Settings locked; two choices in the card', pg.evaluate("()=>document.querySelector('#rm-controls').classList.contains('locked')") and pg.evaluate("()=>document.querySelectorAll('#rm-rail .seg-ctl[data-key^=\"ch\"]').length") == 2)
            check(f'{tag} challenge: before Submit no result shows: no accuracy, no true quality, no best-of-n chart (R3.8)',
                  Z(lang, 'after you submit', '提交后显示') in text(pg, '#rm-ch-g1') and Z(lang, 'after you submit', '提交后显示') in text(pg, '#rm-ch-g2') and '%' not in text(pg, '#rm-status') and '%' not in text(pg, '#rm-metrics') and not vis(pg, '#rm-bon') and not vis(pg, '#rm-tr'),
                  (text(pg, '#rm-ch-g1'), text(pg, '#rm-status'), text(pg, '#rm-metrics')))
            check(f'{tag} challenge: the length weight is there to reason with', vis(pg, '#rm-w') and '0.61' in text(pg, '#rm-metrics'), text(pg, '#rm-metrics'))
            click(pg, '#rm-rail [data-act="submit"]')
            check(f'{tag} challenge: after Submit the goal row shows the true quality, the accuracy row is marked as not a goal', '6.49' in text(pg, '#rm-ch-g1') and '7.6' in text(pg, '#rm-ch-g1') and '74.8%' in text(pg, '#rm-ch-g2') and Z(lang, 'not a goal', '不算条件') in text(pg, '#rm-ch-g2') and vis(pg, '#rm-bon'), (text(pg, '#rm-ch-g1'), text(pg, '#rm-ch-g2')))
            r = text(pg, '#rm-ch-res')
            check(f'{tag} challenge: not met, with a hint and “{Z(lang, "Show the answer", "直接看答案")}” (R4.4)', vis(pg, '#rm-ch-res .badline') and Z(lang, 'Hint', '提示') in r and vis(pg, '#rm-ch-res [data-act="reveal"]')
                  and text(pg, '#rm-ch-res [data-act="reveal"]').strip() == Z(lang, 'Show the answer', '直接看答案'), r[:200])
            check(f'{tag} challenge (R10B): not met, so no “still judged wrong” mark and no note about the average', not pg.evaluate("()=>!!document.querySelector('#rm-pair .rm-wrong, #rm-pair .rm-still-n, #rm-ch-res .rm-avg')"))
            click(pg, '#rm-ch-res [data-act="reveal"]')
            BRIEF = """()=>{const res=document.querySelector('#rm-ch-res'), c=res.querySelector('.cmp:not(details .cmp)'), d=res.querySelector('details.cmp-more');
              const rs=c?[...c.querySelectorAll('.cmp-row')]:[], tg=r=>!!r.querySelector('.cmp-tag,.cmp-you');
              return {rows:rs.map(r=>({ok:r.classList.contains('ok'),me:r.classList.contains('me'),tag:tg(r),lab:r.querySelector('.cmp-lab').textContent})), title:c?c.querySelector('.cmp-title').textContent:'',
                note:(res.querySelector('.cmp-note')||{}).textContent||'', more:d?d.querySelector('summary').textContent:'', open:!!d&&d.open,
                pass:d?[...d.querySelectorAll('.cmp-row')].map(r=>r.classList.contains('ok')):[]};}"""
            def brief(name, n, labs):
                b = pg.evaluate(BRIEF)
                check(f'{tag} challenge ({name}): the chart holds {n} rows, only your pick, the initial setup and the smallest change, no other failing option',
                      len(b['rows']) == n and all(x['ok'] or x['me'] or x['tag'] for x in b['rows']) and sum(x['me'] for x in b['rows']) == 1 and all(any(l in x['lab'] for x in b['rows']) for l in labs), b['rows'])
                check(f'{tag} challenge ({name}): the note counts the 11 options that miss; the 8 that pass wait behind a closed list',
                      Z(lang, 'The other 11 options miss the goal', '其余 11 种没达到目标') in b['note'] and b['more'] == Z(lang, 'See all settings that pass (8)', '看全部可行的设置（8 种）') and not b['open'] and b['pass'] == [True] * 8, b)
                check(f'{tag} challenge ({name}): the title no longer claims every option', b['title'] and not any(x in b['title'] for x in ('every', 'each', '每种', '各')), b['title'])
            brief('as given, revealed', 2, [Z(lang, 'initial setup', '初始设置'), Z(lang, '100 balanced pairs, 200 steps', '加 100 对，训练 200 步')])
            click(pg, '#rm-rail .seg-ctl[data-key="chB"] button[data-v="200"]'); click(pg, '#rm-rail .seg-ctl[data-key="chS"] button[data-v="50"]')
            r = text(pg, '#rm-ch-res')
            check(f'{tag} challenge: met, but a smaller change exists (100 pairs, 200 steps)', vis(pg, '#rm-ch-res .okline') and '100' in r, r[:240])
            brief('200 pairs, 50 steps', 3, [Z(lang, 'initial setup', '初始设置'), Z(lang, '100 balanced pairs, 200 steps', '加 100 对，训练 200 步'), Z(lang, '200 balanced pairs, 50 steps', '加 200 对，训练 50 步')])
            click(pg, '#rm-rail .seg-ctl[data-key="chB"] button[data-v="100"]'); click(pg, '#rm-rail .seg-ctl[data-key="chS"] button[data-v="200"]')
            r = text(pg, '#rm-ch-res')
            check(f'{tag} challenge: met with the smallest change', vis(pg, '#rm-ch-res .okline') and Z(lang, 'smallest change', '改动已经是最少的') in r and vis(pg, '#rm-status.ok'), r[:300])
            brief('the smallest change', 2, [Z(lang, 'initial setup', '初始设置'), Z(lang, '100 balanced pairs, 200 steps', '加 100 对，训练 200 步')])
            # Codex R10B (round 3): met on the average; the fixed example 2 still scores the padded B higher, and says so (not changed to agree)
            ex2 = pg.evaluate("""()=>{const v=document.querySelector('#rm-pair .rm-verdict'), m=v&&v.querySelector('.rm-wrong'), n=document.querySelector('#rm-pair .rm-still-n');
              return {v:v?v.dataset.verdict:'', mark:m?m.textContent.trim():'', note:n?n.textContent:'', cap:document.querySelector('#rm-pair .rm-cap').textContent,
                sc:[...document.querySelectorAll('#rm-pair .rm-sc')].map(e=>parseFloat(e.textContent.split('=')[1].replace('−','-')))}}""")
            check(f'{tag} challenge (R10B): the pass state says the goal is an average and a single example can still be wrong, naming example 2',
                  Z(lang, 'average true quality', '平均的真实质量') in r and Z(lang, 'can still be judged wrong', '仍可能判错') in r and Z(lang, 'Example 2', '示例 2') in r, r[-400:])
            check(f'{tag} challenge (R10B): the fixed example 2 still scores B higher and is marked as still judged wrong, with a note on the stage',
                  Z(lang, 'Example 2', '示例 2') in ex2['cap'] and ex2['v'] == 'disagree' and ex2['sc'][1] > ex2['sc'][0] and ex2['mark'] == Z(lang, 'still judged wrong', '此例仍判错')
                  and Z(lang, 'average', '平均') in ex2['note'], ex2)
            check(f'{tag} challenge: the capability card follows', vis(pg, '#rm-rail [data-exits]'))
            owner_b9(pg, f'{tag} challenge')
            exs = pg.evaluate("()=>window.__rm.exits()"); exw = ' '.join([q['q'] + ' ' + q['why'] for q in exs['transfer'] + exs['explain']])
            check(f'{tag} challenge: the “new case” and “explain it” questions and answers name no part (G2) and no PPO (item 1)', not re.search(G2_RE, exw) and not re.search(r'[Ii]n part|第 \d 部分', exw) and not re.search(NO_PPO_RE, exw), exw[:300])
            tile = pg.evaluate("()=>[0,1,2,3,4,5,6].map(i=>window.__tzRm.read(i)).concat(window.__tzRm.notes()).join(' ')")
            check(f'{tag} the home tile names no PPO (item 1)', tile and not re.search(NO_PPO_RE, tile), tile[:200])
            if phone:
                pg.evaluate("()=>{const t=document.getElementById('rm-graph'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+40)}"); pg.wait_for_timeout(700)
                dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, status:document.getElementById('dkStatus').textContent, read:document.getElementById('dkRead').textContent}}")
                check(f'{tag} challenge: the mini result bar with a short status and both numbers', dk['shown'] and len(dk['status']) <= 12 and '7.77' in dk['read'], dk)
            shot(pg, f'{tag}_ch', full=True); rules(pg, f'{tag} challenge', lang)
            other = 'en' if lang == 'zh' else 'zh'
            pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
            st = state(pg)
            check(f'{tag} challenge: the language switch keeps the answer', st['mode'] == 'challenge' and st['chSt'] == {'bal': 100, 'steps': 200} and st['chSubmitted'] and vis(pg, '#rm-ch-res .okline'), st)
            pg.close()
    b.close()
check('no page errors', not errs, errs[:5])
for x in rule_hits[:40]: print('RULE', x)
check('writing, colour and layout rules', not rule_hits, len(rule_hits))
print(f'{n - len(fails)}/{n} checks passed')
sys.exit(1 if fails else 0)
