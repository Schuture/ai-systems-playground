"""Flow test for the rlvr lab (CONTRACT §6, batch-9 brief).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH, and parts 3–5 again at 900×900 (the owner's item 1):
- no page errors, no horizontal overflow, no text wider than its box (R7.8), a tooltip for every help cursor (R1.1);
- part 1: Next disabled with the kit's why line until a prediction; no score, advantage or result before it (R3.8);
- part 2 (codex-b9v3 #1): the stage shows the picked answer read by both checkers side by side; each of the three answers changes the
  problem line, the answer, the numbers each checker reads and both scores; only a wrong answer the loose checker pays is red (R5.11);
  the card explains the picture without a second problem; no group of 8 and no metric cards; Next waits for the candidate-numbers answer;
- part 3: the group of 8 plays (its scores show for the first time); the group formula; “Next problem” plays a new group; the
  advantages sum to about 0;
- Settings (codex-b9v3 #2): the three tables of assumed values sit in a closed block named for what is in it, and keep their content;
- part 4: the run plays (fast by default), Next waits for its end; the verdict on the prediction and the right answer;
- part 5: Next waits for a loose run; the candidates take over in red (R5.11); part 6: the format reward from the base model;
- part 7: the pass@k chart (before / after); part 8: three misconceptions, “See the scenarios”;
- R3.20: a replay when something changed (a new setup, a new group), none when nothing changed (the same lever again), none when
  coming back to a run or a group already watched (Back / Next), and the replay controls work (run and group);
- Scenarios: four, the insight for each; the URL carries the setup and a fresh page restores it;
- the challenge: Settings locked (R4.2), live progress against the goal (R4.3), the three feedback tiers (R4.4), “Show the answer”, and a
  brief chart (the owner's review of batch 8, 2026-10-07): the 2 passing options, the initial setup and the pick, with a note that counts
  the rest of the 8; the capability card; the language switch keeps the answer;
- the mini result bar at 390 once the chart scrolls away;
- the writing rules on every visible text and tooltip (Chinese spacing and punctuation, two colons, no 控件 / Controls, no sentence
  about the page itself, no “toy” / 玩具 outside the fidelity label: corrected S1, R6.11), no company names, and red only where the
  checker is gamed (R5.6).
The owner's review of 2026-10-09 (one fix round, no review afterwards):
- item 1 / H7: no text or mark of the stage crosses its framed block (the group of answers, the training run, pass@k, part 2's
  columns) or the stage, nor any SVG <text> its <svg>, at 1440, 900 and 390 in both languages (the “优势” column ran past the group's
  right edge at 761–930 px); the replay button and the speed buttons share one height and one baseline; a click on a row highlights
  that row only and keeps the focus on it;
- item 1 / H8: while one problem's 8 answers play, the stage says it is generating a rollout and does not say training; while the run
  plays it says training; the group's block names its step;
- item 2 / H6: the shares chart names each band by the full name of its way of writing (no “候选数”), with a y axis and a caption that
  use the same words; every mark of the line charts and pass@k has a legend item drawn in its own style;
- H1 / H2: no plain-text formula left in the source or on the page, every formula renders as MathML with no TeX command shown as a word,
  and Show the math is a few blocks, each with a heading and a typeset formula; H5: each block of the stage is framed and has a heading;
  H10: plain words; R7.9: the stage head, the group's player and the selected answer keep their size while the group or the run plays.
Screenshots: shots/rlvr/.
Run: python3 tests/rlvr_flow.py [en|zh] [1440|900|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
(P / 'out' / 'rlvr').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/rlvr.html', '--only', 'rlvr'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'rlvr' / 'flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'rlvr.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'rlvr' / 'flow-wrapped.html')
SH = P / 'shots' / 'rlvr'; SH.mkdir(parents=True, exist_ok=True)
ARGS = sys.argv[1:]
LANGS = [x for x in ('en', 'zh') if x in ARGS] or ['en', 'zh']
_W = [a for a in ARGS if a.isdigit()]
SIZES = [s for s in [(1440, 900), (390, 844)] if not _W or str(s[0]) in _W]
RUN900 = not _W or '900' in _W
errs, fails, n = [], [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


Z = lambda lang, en, zh: zh if lang == 'zh' else en


def page(b, w, h, lang, hash_='rlvr'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__rlvr.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#rlvr-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#rlvr-rail [data-next-why]')].filter(p=>!p.hidden&&getComputedStyle(p).visibility!=='hidden'&&p.getClientRects().length).map(p=>p.textContent)")


def in_stage(pg, sel):
    """S1 (b9v1 review): the chart the card talks about ends inside the stage's visible box (its own scroll box at 1440 × 900)"""
    return pg.evaluate("s=>{const st=document.getElementById('rlvr-stage').getBoundingClientRect(), e=document.querySelector(s); if(!e||!e.getClientRects().length) return {ok:false}; const b=e.getBoundingClientRect().bottom; return {ok:b<=Math.min(st.bottom,innerHeight)+1, bottom:Math.round(b), visible:Math.round(Math.min(st.bottom,innerHeight))}}", sel)


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__rlvr||!window.__rlvr.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def idle(pg, ms=16000):
    """wait for both players to stop (the run plays about 10 s at the default speed)"""
    try: pg.wait_for_function("()=>{const s=window.__rlvr.state(); return !s.playing&&!s.grpPlaying}", timeout=ms)
    except Exception: pass
    pg.wait_for_timeout(150)


def click(pg, sel):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(250); settle(pg)


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#rlvr-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def playing(pg):
    """right after a click: is a player running (two frames later)?"""
    s = state(pg); return s['playing'] or s['grpPlaying']


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def all_text(pg):
    return pg.evaluate("""()=>{const out=[]; for(const s of ['#rlvr-app','#dock']){const e=document.querySelector(s); if(e&&e.offsetParent!==null) out.push(e.innerText);}
      document.querySelectorAll('#rlvr-app [data-tipx], #dock [data-tipx]').forEach(e=>out.push(e.getAttribute('data-tipx'))); return out.join('\\n')}""")


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · (?![0-9])/,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy”'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万'],[/OpenAI|Anthropic|DeepMind|Google|Meta\\b|Microsoft/,'company name'],
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour (R5.7)'],
   [/<\\/?(?:p|div|span|math|mi|mrow|b|em|sub|sup|br)\\b[^>]*>/,'markup shown as text (the kit escapes data-tipx)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, .fid-tag, a[href*="github.com/Schuture"]')||!e.getClientRects().length) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 // a hover's typeset formula and the simple tags around it are kept as markup by the kit (escMath, 2026-10-09): test what would show as text
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx').replace(/<math class="tex"[\\s\\S]*?<\\/math>/g,' x ').replace(/<\\/?(b|i|em|strong|br|sub|sup|small|code|div|span|p)(\\s+class="[\\w\\s-]*")?\\s*\\/?>/g,' '); for(const [re,name] of bad) if(re.test(t)) out.push(name+' (tooltip): '+t.slice(0,90));});
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
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx').replace(/<math[\\s\\S]*?<\\/math>/g,' x ').replace(/<[^>]+>/g,''); const m=t.match(/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>][\\u4e00-\\u9fff]/g); if(m) out.add(m.join('|')+' in tooltip: '+t.slice(0,80));});
 return [...out];}"""
PUNCT = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), bad=/[\\u4e00-\\u9fff][:;,!?]|[:;,!?] ?[\\u4e00-\\u9fff]/g;
 const shown=e=>e&&!e.closest('[data-endnotes],pre,code,.mathbody,#tip')&&e.getClientRects().length;
 const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){if(!shown(n.parentElement)) continue; const t=n.textContent.replace(/\\s+/g,' '); const m=t.match(bad); if(m) out.add(m.join('|')+' in: '+t.trim().slice(0,80));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx').replace(/<math[\\s\\S]*?<\\/math>/g,' x ').replace(/<[^>]+>/g,''); const m=t.match(bad); if(m) out.add(m.join('|')+' in tooltip: '+t.slice(0,80));});
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
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|\\bred\\b|warn|wrong|ex-res/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#rlvr-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#rlvr-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#rlvr-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
rule_hits = []


# the owner's clarification (2026-10-07 23:37 UTC): no hover only expands a shown number to its exact value; Show the math may write one out
EXACT_HOVER = """r=>{const root=document.querySelector(r), out=[]; if(!root) return out;
  for(const e of root.querySelectorAll('[data-tipx],[data-tip]')){if(e.closest('.mathbody')) continue; let t=e.getAttribute('data-tipx')||''; const k=e.getAttribute('data-tip');
    if(k&&window.ASP) t+=' '+(window.ASP.t(k)||''); if(/精确值是|Exactly/.test(t)) out.push(t.slice(0,100));}
  for(const e of root.querySelectorAll('.num-big')) if(!e.closest('.mathbody')&&(e.hasAttribute('data-tipx')||e.closest('.def'))) out.push('hover on '+e.textContent);
  return out;}"""


def rules(pg, where, lang):
    hx = pg.evaluate(EXACT_HOVER, '#rlvr-app'); check(f'{where}: no exact-value hover outside Show the math', not hx, hx[:3])
    for sel in ('#rlvr-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')
    for h in pg.evaluate(OVER): rule_hits.append(f'{where}: text wider than its box: {h}')
    for h in pg.evaluate("()=>[...document.querySelectorAll('#rlvr-metrics .v')].filter(e=>e.getClientRects().length&&e.scrollWidth>e.clientWidth+1).map(e=>e.textContent+' ('+e.scrollWidth+' > '+e.clientWidth+')')"): rule_hits.append(f'{where}: metric value wider than its card: {h}')
    for h in pg.evaluate("()=>{const out=[]; document.querySelectorAll('#rlvr-p1 .rlvr-endl, #rlvr-len .rlvr-endl').forEach(box=>{const r=[...box.children].map(e=>e.getBoundingClientRect()); for(let i=0;i<r.length;i++) for(let j=i+1;j<r.length;j++){const a=r[i],c=r[j]; if(a.left<c.right&&c.left<a.right&&a.top<c.bottom-1&&c.top<a.bottom-1) out.push(box.children[i].textContent+' / '+box.children[j].textContent);}}); return out}"): rule_hits.append(f'{where}: end labels overlap: {h}')
    for h in pg.evaluate(HELP): rule_hits.append(f'{where}: help cursor without a tooltip: {h}')
    o = overflow(pg)
    if o > 0: rule_hits.append(f'{where}: horizontal overflow {o}px')


# item 1 / H7 (the owner's review of 2026-10-09): every text of the stage inside its framed block and the stage; SVG text inside its svg
BOUNDS = """()=>{const st=document.getElementById('rlvr-stage'); if(!st) return ['no stage']; const out=[];
  const shown=e=>{for(let x=e;x&&x!==document.body;x=x.parentElement){const cs=getComputedStyle(x); if(cs.display==='none'||cs.visibility==='hidden') return false;} return e.getClientRects().length>0;};
  const boxes=[...st.querySelectorAll('.rlvr-blk,.rlvr-chk,.rlvr-col')].filter(shown); boxes.unshift(st);
  const inner=e=>boxes.filter(b=>b!==e&&b.contains(e)).pop();
  const own=e=>[...e.childNodes].some(c=>c.nodeType===3&&c.textContent.trim());
  for(const e of st.querySelectorAll('span,b,em,p,div,button,h3,h4,li,label,i')){ if(!shown(e)||(!own(e)&&!e.matches('i,button'))) continue;
    if(e.closest('details.math:not([open]),.mathbody,svg')) continue;
    const r=e.getBoundingClientRect(); if(r.width<1||r.height<1) continue; const b=inner(e); if(!b) continue; const q=b.getBoundingClientRect();
    const by=Math.max(r.right-q.right,q.left-r.left); if(by>1) out.push((b.id||b.className.split(' ')[0])+' > '+(e.className&&typeof e.className==='string'?e.className.split(' ')[0]:e.tagName)+' “'+(e.textContent||'').trim().slice(0,24)+'” by '+Math.round(by)+' px');}
  for(const t of st.querySelectorAll('svg text')){ if(!shown(t)||!t.textContent.trim()) continue; const r=t.getBoundingClientRect(), sv=t.ownerSVGElement.getBoundingClientRect(), b=inner(t), q=b.getBoundingClientRect();
    const by=Math.max(r.right-sv.right,sv.left-r.left,r.bottom-sv.bottom,sv.top-r.top,r.right-q.right,q.left-r.left); if(by>1) out.push('svg text “'+t.textContent.trim().slice(0,24)+'” by '+Math.round(by)+' px');}
  return out.slice(0,8);}"""
# H7: the replay button and the speed buttons of a player share one height, one top and one text baseline
PLAYERS = """()=>[...document.querySelectorAll('#rlvr-stage .rlvr-pl')].filter(e=>e.getClientRects().length).map(row=>{
  const b=row.querySelector('.btn'), s=row.querySelector('.seg-ctl'); if(!b||!s) return {id:row.id, ok:false};
  const rb=b.getBoundingClientRect(), rs=s.getBoundingClientRect();
  const base=e=>{const t=[...e.childNodes].find(n=>n.nodeType===3&&n.textContent.trim()); if(!t) return NaN; const r=document.createRange(); r.selectNodeContents(t); return r.getBoundingClientRect().bottom;};
  const d={id:row.id, h:[+rb.height.toFixed(1), +rs.height.toFixed(1)], top:[+rb.top.toFixed(1), +rs.top.toFixed(1)], base:[+base(b).toFixed(1), +base(s.querySelector('button')).toFixed(1)]};
  d.ok=Math.abs(d.h[0]-d.h[1])<=0.5&&Math.abs(d.top[0]-d.top[1])<=0.5&&Math.abs(d.base[0]-d.base[1])<=1; return d;})"""
# R7.9: sizes that must not change while a player runs, sampled every frame in the page
SAMPLE = """(what)=>{window.__rlvrH=[]; const sz=()=>{const q=s=>document.querySelector(s), h=s=>q(s)?Math.round(q(s).getBoundingClientRect().height):-1;
    return [h('#rlvr-st-title'), h('#rlvr-st-sub'), what==='grp'?h('#rlvr-sel'):h('#rlvr-run'), what==='grp'?Math.round(q('#rlvr-gplay').getBoundingClientRect().top-q('#rlvr-grp').getBoundingClientRect().top):h('#rlvr-grp')];};
  const f=()=>{const s=window.__rlvr.state(); if(!(what==='grp'?s.grpPlaying:s.playing)) return; window.__rlvrH.push(sz()); requestAnimationFrame(f);}; f();}"""
SAMPLED = "()=>{const a=window.__rlvrH||[]; const out=[]; for(let i=0;i<4;i++){const v=a.map(x=>x[i]); out.push(v.length?Math.max(...v)-Math.min(...v):-1);} return {n:a.length, spread:out}}"
# H1: every formula is MathML and no TeX command shows as a word
MATHOK = r"""()=>{const bad=[]; const words=/^(frac|sqrt|left|right|text|mathrm|operatorname|cdot|hat|big|Big|quad|rho|theta|beta|pi|sigma|sum)$/;
  document.querySelectorAll('#rlvr-app math').forEach(m=>{m.querySelectorAll('mi').forEach(x=>{if(words.test(x.textContent.trim())) bad.push(x.textContent);}); if(/\\/.test(m.textContent)) bad.push('backslash');});
  for(const e of document.querySelectorAll('#rlvr-app [data-tipx]')){const t=e.getAttribute('data-tipx'); if(/\\frac|\\text|\\sqrt/.test(t)) bad.push('raw TeX in a tooltip');}
  return {n:document.querySelectorAll('#rlvr-app math').length, bad:bad.slice(0,5)};}"""
PLAIN_FX = [r'ratio = new', r'Advantage = reward', r'优势 = 奖励', r'\) ÷ \(', r'÷ \(\d', r'\(1 − \d', r'std = \d', r'标准差 = \d']
# H6: the marks drawn in each line chart, and the legend items that name them
MARKS = """(sel)=>{const box=document.querySelector(sel); if(!box||!box.getClientRects().length) return null; const plot=box.querySelector('.rlvr-plot'); const drawn=new Set(), named=new Set();
  plot.querySelectorAll('svg polyline').forEach(l=>{const c=l.classList; drawn.add(c.contains('dash')?'dash':c.contains('red')?'red':c.contains('other')?'other':'solid');});
  plot.querySelectorAll('svg line.rlvr-base').forEach(l=>drawn.add(l.getAttribute('x1')===l.getAttribute('x2')?'vdot':'dot'));
  if(plot.querySelector('.rlvr-mk')) drawn.add('mk');
  box.querySelectorAll('.rlvr-legs .rlvr-li:not(.off)').forEach(li=>{const l=li.querySelector('line'); if(li.querySelector('.rlvr-ldot')) named.add('mk'); else if(l){const c=l.classList; named.add(c.contains('rlvr-base')?(l.getAttribute('x1')===l.getAttribute('x2')?'vdot':'dot'):c.contains('dash')?'dash':c.contains('red')?'red':c.contains('other')?'other':'solid');}});
  return {drawn:[...drawn].sort(), named:[...named].sort()};}"""


def bounds(pg, where):
    b = pg.evaluate(BOUNDS); check(f'{where}: no text crosses its block or the stage (item 1, H7)', not b, b)


def r320(pg, sel, tag, what, expect):
    """R3.20: click, then look two frames later whether a player started"""
    pg.evaluate(f"()=>document.querySelector('{sel}').click()")
    pg.evaluate("()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))")
    p = playing(pg)
    check(f'{tag} {what}: {"a replay" if expect else "no replay"} (R3.20)', p == expect, state(pg))
    pg.wait_for_timeout(200); settle(pg)
    return p


with sync_playwright() as pw:
    b = pw.chromium.launch()
    for lang in LANGS:
        for (w, h) in SIZES:
            phone = w <= 760; tag = f'{lang}-{w}'
            pg = page(b, w, h, lang)
            st = state(pg)
            check(f'{tag}: opens on part 1 of the guided tour, run speed fast and group speed medium by default (R3.9)',
                  st['mode'] == 'guided' and st['step'] == 0 and st['speed'] == 'f' and st['gspeed'] == 'm', st)
            fv = pg.evaluate("()=>{const q=s=>document.querySelector(s).getBoundingClientRect(); return {grp:Math.round(q('#rlvr-grp').top), metrics:Math.round(q('#rlvr-metrics').top), grpEnd:Math.round(q('#rlvr-grp').bottom), ih:innerHeight}}")
            check(f'{tag}: the group of 8 leads the stage, the metric cards below it (R7.12)', fv['metrics'] >= fv['grpEnd'] and (phone or fv['grp'] < fv['ih']), fv)
            # codex-b9v3 #2: the three tables of assumed values wait in a closed block in Settings, named for what is in it (R2.6)
            asm = pg.evaluate("""()=>{const d=document.getElementById('rlvr-asm-d'), s=d&&d.querySelector('summary'), r=s?s.getBoundingClientRect():null, c=document.getElementById('rlvr-controls').getBoundingClientRect();
              return {exists:!!d, open:!!(d&&d.open), tables:[...document.querySelectorAll('#rlvr-controls table.rlvr-at')].filter(t=>t.checkVisibility()).length,
                      summary:s?s.textContent:'', top:r?Math.round(r.top):-1, ctlEnd:Math.round(c.bottom), ih:innerHeight}}""")
            check(f'{tag}: the assumed values are closed by default, no table shows, and the block is named for what is in it (codex-b9v3 #2, R2.6)',
                  asm['exists'] and not asm['open'] and asm['tables'] == 0 and all(k in asm['summary'] for k in Z(lang, ('lengths', 'right', 'problems'), ('长度', '答对的概率', '各类题'))), asm)
            # codex-b9v3 #2's first-screen check went with the right-hand Settings column: since 2026-10-11 Settings show only in 自由调参, all unlocked
            pg.evaluate("()=>{document.getElementById('rlvr-asm-d').open=true}"); pg.wait_for_timeout(150)
            tb = pg.evaluate("()=>[...document.querySelectorAll('#rlvr-asm-d table.rlvr-at')].filter(t=>t.checkVisibility()).map(t=>t.rows.length)")
            check(f'{tag}: opened, the block holds the three tables as before (4 ways; 4 ways and the loose row; 2 sets of problems)', tb == [5, 6, 3], tb)
            pg.evaluate("()=>{document.getElementById('rlvr-asm-d').open=false}"); pg.wait_for_timeout(100)
            # ---- part 1
            check(f'{tag} part 1: Next disabled until a prediction, with the kit’s why line', all(nexts(pg)) and any(Z(lang, 'Pick an answer', '选一个答案') in x for x in why(pg)), (nexts(pg), why(pg)))
            idle(pg, 9000)
            rows = text(pg, '#rlvr-rows'); mt = text(pg, '#rlvr-metrics')
            check(f'{tag} part 1: no score, no advantage and no result before the prediction (R3.8)',
                  Z(lang, 'adv.', '优势') not in rows and '✓' not in rows and '✗' not in rows and '31%' not in mt and not vis(pg, '#rlvr-run') and not vis(pg, '#rlvr-pk'), (rows[:200], mt))
            p1t = text(pg, '#rlvr-rail .step.current')
            check(f'{tag} part 1: the card gives the numbers to reason with (R3.21): 20 / 55 / 75% and 68 / 25 / 5%', all(x in p1t for x in ('20%', '55%', '75%', '68%', '25%', '5%')), p1t[:300])
            check(f'{tag} part 1: GRPO is linked to its lab (S6)', pg.evaluate("()=>!!document.querySelector('#rlvr-rail .step.current a[href=\"#grpo\"]')"))
            shot(pg, f'{tag}_p1', full=phone); rules(pg, f'{tag} part 1', lang)
            click(pg, '#rlvr-rail [data-pred="S"]')
            check(f'{tag} part 1: after a pick, Next is enabled and the card says it is saved', not any(nexts(pg)) and Z(lang, 'Saved', '已记下') in text(pg, '#rlvr-rail .step.current'))
            # ---- part 2 (codex-b9v3 #1): the stage shows the picked answer, read by both checkers side by side; the group of 8 starts in part 3
            r320(pg, '#rlvr-rail [data-nav="next"]', tag, 'part 2, a still picture of one answer (nothing to play)', False)
            # codex-b9v3 check: part 2 no longer plays the group, so coming back to part 1 must not replay the group watched there, and that
            # visit must not use up the group's first play in part 3 (checked on entering part 3 below)
            r320(pg, '#rlvr-rail [data-nav="back"]', tag, 'part 2 Back to part 1 (its group already watched there)', False)
            r320(pg, '#rlvr-rail [data-nav="next"]', tag, 'part 1 Next to part 2 again (a still picture)', False)
            check(f'{tag} part 2: Next waits for the candidate-numbers answer, with its why line', all(nexts(pg)) and any(Z(lang, 'candidate', '候选') in x for x in why(pg)), why(pg))
            check(f'{tag} part 2: the stage shows one answer and the two checkers; the group of 8 and the metric cards are not on it',
                  vis(pg, '#rlvr-chk') and not vis(pg, '#rlvr-grp') and not vis(pg, '#rlvr-metrics') and state(pg)['ex2'] == 1, state(pg))
            if not phone:
                v = in_stage(pg, '#rlvr-chk'); check(f'{tag} part 2: the whole picture is inside the stage’s visible box (R7.12)', v['ok'], v)
            # the 验收: each option changes the problem line, the answer, the numbers each checker reads and both scores; only a wrong answer
            # the loose checker pays is red. Expected values written out here from the three answers' texts (the numbers test checks the rule).
            EXP = {'1': dict(way=Z(lang, 'Step by step', '一步步算'), s=['20'], l=['3', '6', '18', '18', '2', '20', '20'], sp=1, lp=1, red=False),
                   '0': dict(way=Z(lang, 'Answer only', '直接写答案'), s=['18'], l=['18'], sp=0, lp=0, red=False),
                   '3': dict(way=Z(lang, 'List candidate numbers', '写好几个候选数字'), s=['18'], l=['24', '20', '18', '18'], sp=0, lp=1, red=True)}
            PIC = """()=>{const q=s=>document.querySelector(s), nums=s=>[...document.querySelectorAll(s)].map(e=>e.textContent.trim());
              const pay=k=>{const t=(q(`#rlvr-chk [data-chk="${k}"] .rlvr-cv`)||{}).textContent||''; return {ok:t.trim().startsWith('✓'), n:+(t.match(/[01]/)||['-1'])[0]};};
              const reds=[...document.querySelectorAll('#rlvr-chk .red')].filter(e=>e.getClientRects().length);
              return {q:(q('#rlvr-chk .rlvr-q')||{}).innerText||'', way:(q('#rlvr-chk .rlvr-cwho')||{}).innerText||'', s:nums('#rlvr-chk [data-chk="strict"] .rlvr-num'), l:nums('#rlvr-chk [data-chk="loose"] .rlvr-num'),
                sHit:nums('#rlvr-chk [data-chk="strict"] .rlvr-num.hit'), lHit:nums('#rlvr-chk [data-chk="loose"] .rlvr-num.hit'), sp:pay('strict'), lp:pay('loose'),
                reds:reds.length, sRed:!!q('#rlvr-chk [data-chk="strict"].red, #rlvr-chk [data-chk="strict"] .red'), lCol:!!q('#rlvr-chk [data-chk="loose"].red'),
                redNum:nums('#rlvr-chk .rlvr-num.red'), status:(q('#rlvr-status')||{}).className||'', head:(q('#rlvr-st-sub')||{}).textContent||''};}"""
            for lv in ('1', '0', '3'):
                r320(pg, f'#rlvr-rail [data-lever="{lv}"]', tag, f'part 2, answer {lv} picked (a still picture: no animation)', False)
                e, pic, ob = EXP[lv], pg.evaluate(PIC), text(pg, '#rlvr-rail [data-obs]')
                check(f'{tag} part 2, answer {lv}: the picture shows the pencil problem with its reference answer 20, and the picked answer',
                      Z(lang, 'pencils', '铅笔') in pic['q'] and '20' in pic['q'] and e['way'] in pic['way'] and state(pg)['ex2'] == int(lv), (pic['q'][:80], pic['way']))
                check(f'{tag} part 2, answer {lv}: the strict checker reads only the “####” number, the loose checker every number',
                      pic['s'] == e['s'] and pic['l'] == e['l'] and pic['sHit'] == [x for x in e['s'] if x == '20'] and pic['lHit'] == [x for x in e['l'] if x == '20'], pic)
                check(f'{tag} part 2, answer {lv}: both scores, side by side, and the head line says them',
                      pic['sp'] == {'ok': bool(e['sp']), 'n': e['sp']} and pic['lp'] == {'ok': bool(e['lp']), 'n': e['lp']} and str(e['lp']) in pic['head'] and str(e['sp']) in pic['head'], pic)
                if e['red']:
                    check(f'{tag} part 2, answer {lv}: the wrong answer the loose checker pays is red: the column, the matched 20, the score and a note (R5.11)',
                          pic['lCol'] and pic['redNum'] == ['20'] and pic['reds'] >= 4 and not pic['sRed'] and 'bad' in pic['status'], pic)
                else:
                    check(f'{tag} part 2, answer {lv}: nothing red when the two checkers agree (R5.6)', pic['reds'] == 0 and 'bad' not in pic['status'], pic)
                check(f'{tag} part 2, answer {lv}: the card explains the picture without a second problem (no problem text, no group of 8)',
                      len(ob) > 20 and Z(lang, 'pencils', '铅笔') not in ob and Z(lang, '240 pages', '240 页') not in text(pg, '#rlvr-stage') and not vis(pg, '#rlvr-grp'), ob[:200])
                if lv == '3':
                    check(f'{tag} part 2: after the candidates answer, Next is enabled', not any(nexts(pg)))
                    shot(pg, f'{tag}_p2', full=phone); rules(pg, f'{tag} part 2', lang)
            if phone:
                dk = pg.evaluate("()=>{window.scrollTo(0, document.querySelector('#rlvr-rail [data-obs]').getBoundingClientRect().top+scrollY-300); return 0}"); pg.wait_for_timeout(700)
                dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!!d&&!d.hidden, cls:(document.getElementById('dkStatus')||{}).className||'', read:(document.getElementById('dkRead')||{}).textContent||''}}")
                check(f'{tag} part 2: with the card on screen, the mini result bar gives both scores of the picked answer', dk['shown'] and 'bad' in dk['cls'] and '0' in dk['read'] and '1' in dk['read'], dk)
                pg.evaluate('()=>window.scrollTo(0,0)'); pg.wait_for_timeout(200)
            # ---- part 3: the group of 8 shows its scores for the first time, so it plays (moved here from part 2)
            r320(pg, '#rlvr-rail [data-nav="next"]', tag, 'part 3, the group of 8 with its scores for the first time', True)
            check(f'{tag} part 3: the group of 8 is on the stage and the two-checker picture is gone', vis(pg, '#rlvr-grp') and not vis(pg, '#rlvr-chk') and vis(pg, '#rlvr-metrics'))
            idle(pg)
            check(f'{tag} part 3: the advantage formula and Next enabled', vis(pg, '#rlvr-rail .step.current .fx') and not any(nexts(pg)))
            check(f'{tag} part 3: the advantages shown with the group’s mean and std, GRPO linked (S2, S6)', Z(lang, 'adv.', '优势') in text(pg, '#rlvr-rows') and vis(pg, '#rlvr-gstat') and Z(lang, 'standard deviation', '标准差') in text(pg, '#rlvr-gstat') and pg.evaluate("()=>!!document.querySelector('#rlvr-rail .step.current a[href=\"#grpo\"]')"), text(pg, '#rlvr-gstat'))
            advs = pg.evaluate("()=>{const s=window.__rlvr.state(), r=window.__rlvr.run({}), g=r[s.tShow].groups[s.grpI]; return g.resp.map(x=>x[5])}")
            check(f'{tag} part 3: the group’s advantages sum to about 0', abs(sum(advs)) < 1e-9, advs)
            r320(pg, '#rlvr-rail [data-act="grp-next"]', tag, 'part 3, Next problem', True)
            idle(pg)
            check(f'{tag} part 3: Next problem moved to group 2 and the observation explains it', state(pg)['grpI'] == 1 and len(text(pg, '#rlvr-rail [data-obs]')) > 40)
            r320(pg, '#rlvr-rail [data-nav="back"]', tag, 'part 3 Back to part 2 (a still picture)', False)
            r320(pg, '#rlvr-rail [data-nav="next"]', tag, 'part 2 Next to part 3 again', False)
            r320(pg, '#rlvr-gplay', tag, 'part 3, the group’s Replay control', True)
            pg.evaluate(SAMPLE, 'grp')
            if not pg.evaluate('()=>matchMedia("(prefers-reduced-motion: reduce)").matches'):
                check(f'{tag} part 3: during the replay no advantage shows before all 8 rewards are in (S2)', Z(lang, 'adv.', '优势') not in text(pg, '#rlvr-rows') and not vis(pg, '#rlvr-gstat'), text(pg, '#rlvr-rows')[:200])
                hd = (text(pg, '#rlvr-st-title'), text(pg, '#rlvr-st-sub'), text(pg, '#rlvr-ghd'))
                check(f'{tag} part 3: while the 8 answers play, the stage says it is generating a rollout, not training (item 1, H8)',
                      Z(lang, 'Generating', '正在生成') in hd[1] and 'rollout' in hd[1] and Z(lang, 'train', '训练') not in hd[1].lower() and 'rollout' in hd[0].lower() and 'rollout' in hd[2].lower(), hd)
                bounds(pg, f'{tag} part 3 mid-replay')
            shot(pg, f'{tag}_p3', full=phone); rules(pg, f'{tag} part 3', lang)
            idle(pg)
            sm = pg.evaluate(SAMPLED)
            check(f'{tag} part 3: while the group plays, the stage head, the selected answer and the group’s player keep their size and place (R7.9)', sm['n'] > 3 and max(sm['spread']) <= 1, sm)
            hd = text(pg, '#rlvr-st-sub')
            check(f'{tag} part 3: once the 8 answers are in, the stage still does not call it training (H8)', Z(lang, 'train', '训练') not in hd.lower() and Z(lang, 'does not change the model', '不会改变模型') in hd, hd)
            pl = pg.evaluate(PLAYERS)
            check(f'{tag} part 3: the group’s replay and speed buttons share one height and one baseline (H7)', pl and all(x['ok'] for x in pl), pl)
            pg.evaluate("()=>{const r=document.querySelectorAll('#rlvr-rows .rlvr-row')[2]; r.click();}"); pg.wait_for_timeout(200)
            hl = pg.evaluate("""()=>{const g=getComputedStyle(document.getElementById('rlvr-grp')), s=getComputedStyle(document.getElementById('rlvr-stage'));
              return {on:[...document.querySelectorAll('#rlvr-rows .rlvr-row.on')].map(e=>e.dataset.row), focus:(document.activeElement||{}).dataset?document.activeElement.dataset.row:null,
                      ring:[g.outlineStyle, g.boxShadow, s.outlineStyle, s.boxShadow].filter(v=>v!=='none').length}}""")
            check(f'{tag} part 3: a click on a row highlights that row only and keeps the focus on it; no ring around the panel (H7)', hl['on'] == ['2'] and hl['focus'] == '2' and hl['ring'] == 0, hl)
            bounds(pg, f'{tag} part 3')
            # ---- part 4
            r320(pg, '#rlvr-rail [data-nav="next"]', tag, 'part 4, a new run', True)
            pg.evaluate(SAMPLE, 'run')
            check(f'{tag} part 4: Next waits for the run, with its why line', all(nexts(pg)) and any(Z(lang, 'run finish', '训练播完') in x for x in why(pg)), why(pg))
            hd = (text(pg, '#rlvr-st-title'), text(pg, '#rlvr-st-sub'), text(pg, '#rlvr-ghd'), text(pg, '#rlvr-rhd'))
            check(f'{tag} part 4: while the run plays, the stage says training, and the group’s block names the step its answers come from (H8)',
                  Z(lang, 'Training: step', '正在训练') in hd[1] and Z(lang, 'update', '更新') in hd[0] and 'rollout' in hd[2] and Z(lang, 'step 0', '第 0 步') in hd[2] and hd[3], hd)
            t0 = pg.evaluate('()=>performance.now()'); idle(pg); dt = (pg.evaluate('()=>performance.now()') - t0) / 1000
            check(f'{tag} part 4: the whole run plays in about 10 s at the default speed (R3.9)', 6 <= dt <= 14, dt)
            sm = pg.evaluate(SAMPLED)
            check(f'{tag} part 4: while the run plays, the stage head, the run’s block and the group’s block keep their size (R7.9)', sm['n'] > 3 and max(sm['spread']) <= 1, sm)
            hd = text(pg, '#rlvr-st-sub')
            check(f'{tag} part 4: at the end of the run the stage says the training is done (H8)', Z(lang, 'Trained', '训练完成') in hd, hd)
            pl = pg.evaluate(PLAYERS)
            check(f'{tag} part 4: both players: replay and speed buttons share one height and one baseline (H7)', len(pl) == 2 and all(x['ok'] for x in pl), pl)
            for sel in ('#rlvr-p1', '#rlvr-len'):
                mk = pg.evaluate(MARKS, sel); check(f'{tag} part 4: every mark of {sel} has a legend item in its own style (H6)', mk and mk['drawn'] == mk['named'], mk)
            bounds(pg, f'{tag} part 4')
            ob = text(pg, '#rlvr-rail [data-obs]')
            check(f'{tag} part 4: the result 5% → 93%, 65 → 230 tokens, the verdict on the prediction with the right answer, Next enabled',
                  '5%' in ob and '93%' in ob and '65' in ob and '230' in ob and Z(lang, 'You predicted', '你预测的是') in ob and not any(nexts(pg)), ob[:400])
            if phone:
                pg.evaluate("()=>{const t=document.getElementById('rlvr-mix'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+500)}"); pg.wait_for_timeout(700)
                dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!!d&&!d.hidden, status:(document.getElementById('dkStatus')||{}).textContent||'', read:(document.getElementById('dkRead')||{}).textContent||''}}")
                check(f'{tag} part 4: the mini result bar with a short status and the result', dk['shown'] and len(dk['status']) <= 12 and '73%' in dk['read'], dk)
                pg.evaluate('()=>window.scrollTo(0,0)'); pg.wait_for_timeout(200)
            if not phone:
                v = in_stage(pg, '#rlvr-p1'); check(f'{tag} part 4: the pass@1 and length charts end inside the stage’s visible box (S1)', v['ok'], v)
            shot(pg, f'{tag}_p4', full=phone); rules(pg, f'{tag} part 4', lang)
            r320(pg, '#rlvr-rplay', tag, 'part 4, the run’s Replay control', True)
            pg.wait_for_timeout(400)
            click(pg, '#rlvr-rplay'); st = state(pg)
            check(f'{tag} part 4: Pause stops the run part-way', not st['playing'] and 0 < st['tShow'] < 60, st)
            click(pg, '#rlvr-rplay'); idle(pg)
            check(f'{tag} part 4: Play resumes to the end', state(pg)['tShow'] == 60)
            # ---- part 5
            r320(pg, '#rlvr-rail [data-nav="next"]', tag, 'part 5, the strict run already watched', False)
            check(f'{tag} part 5: Next waits for a loose run', all(nexts(pg)) and any(Z(lang, 'loose', '宽松') in x for x in why(pg)), why(pg))
            r320(pg, '#rlvr-rail [data-lever="strict"]', tag, 'part 5, strict again (nothing changed)', False)
            r320(pg, '#rlvr-rail [data-lever="loose"]', tag, 'part 5, the loose checker (a new run)', True)
            idle(pg)
            ob = text(pg, '#rlvr-rail [data-obs]')
            reds = pg.evaluate("()=>[...document.querySelectorAll('#rlvr-graph .red')].filter(e=>e.getClientRects().length).length")
            check(f'{tag} part 5: candidates take over (93%), right in one try falls to 19%, in red (R5.11); Next enabled',
                  '93%' in ob and '19%' in ob and reds >= 2 and state(pg)['chk'] == 'loose' and not any(nexts(pg)) and Z(lang, 'Gamed', '被钻空子') in text(pg, '#rlvr-status'), (ob[:300], reds))
            if not phone:
                v = in_stage(pg, '#rlvr-p1'); check(f'{tag} part 5: the red fall in the pass@1 chart is inside the stage’s visible box (S1)', v['ok'], v)
            lab = pg.evaluate("()=>[...document.querySelectorAll('#rlvr-mix .rlvr-endl > span')].map(e=>e.innerText.replace(/\\s+/g,' ').trim())")
            ways = Z(lang, ['Answer only', 'Step by step', 'Solve, then check', 'List candidate numbers'], ['直接写答案', '一步步算', '算完再检查', '写好几个候选数字'])
            mixt = text(pg, '#rlvr-mix')
            check(f'{tag} part 5: the shares chart names each band by its way of writing in full, no “候选数”; caption and axis use the same words (item 2, H6)',
                  sorted(x.rsplit(' ', 1)[0] for x in lab) == sorted(ways) and not re.search('候选数(?!字)', all_text(pg)) and Z(lang, 'answers written each way', '每种写法的回答') in mixt
                  and Z(lang, 'list candidate numbers', '写好几个候选数字') in mixt and '100%' in mixt and Z(lang, 'Red:', '红色') in mixt, (lab, mixt[:300]))
            for sel in ('#rlvr-p1', '#rlvr-len'):
                mk = pg.evaluate(MARKS, sel); check(f'{tag} part 5: every mark of {sel} has a legend item in its own style, red included (H6)', mk and mk['drawn'] == mk['named'], mk)
            bounds(pg, f'{tag} part 5')
            shot(pg, f'{tag}_p5', full=phone); rules(pg, f'{tag} part 5', lang)
            r320(pg, '#rlvr-rail [data-nav="back"]', tag, 'part 5 Back to part 4 (run watched)', False)
            r320(pg, '#rlvr-rail [data-nav="next"]', tag, 'part 4 Next to part 5 (loose run watched)', False)
            # ---- part 6
            r320(pg, '#rlvr-rail [data-nav="next"]', tag, 'part 6, the base model (a new run)', True)
            idle(pg)
            check(f'{tag} part 6: Next waits for the format reward', all(nexts(pg)))
            r320(pg, '#rlvr-rail [data-lever="1"]', tag, 'part 6, format reward on (a new run)', True)
            idle(pg)
            ob = text(pg, '#rlvr-rail [data-obs]')
            check(f'{tag} part 6: step 10: 16% without, 45% with; TinyZero’s 0.1; Next enabled', '16%' in ob and '45%' in ob and '0.1' in ob and not any(nexts(pg)), ob[:400])
            mk = pg.evaluate("()=>[...document.querySelectorAll('#rlvr-p1 .rlvr-mk em, #rlvr-len .rlvr-mk em')].map(e=>e.textContent)")
            check(f'{tag} part 6: the “####” chart, the other run in grey and both values at step 10 marked (S5)', '####' in text(pg, '#rlvr-len') and pg.evaluate("()=>document.querySelectorAll('#rlvr-p1 .rlvr-plot .rlvr-ln.other').length") == 1 and all(x in mk for x in ('45%', '16%', '91%', '41%')), mk)
            if not phone:
                v = in_stage(pg, '#rlvr-len'); check(f'{tag} part 6: both charts end inside the stage’s visible box (S1)', v['ok'], v)
            for sel in ('#rlvr-p1', '#rlvr-len'):
                mk = pg.evaluate(MARKS, sel); check(f'{tag} part 6: every mark of {sel} (the grey run, step 10’s line and dots) has a legend item (H6)', mk and mk['drawn'] == mk['named'], mk)
            bounds(pg, f'{tag} part 6')
            shot(pg, f'{tag}_p6', full=phone); rules(pg, f'{tag} part 6', lang)
            # ---- part 7
            r320(pg, '#rlvr-rail [data-nav="next"]', tag, 'part 7, the strict run already watched', False)
            pk = text(pg, '#rlvr-pk'); ob = text(pg, '#rlvr-rail [data-obs]')
            check(f'{tag} part 7: pass@k before and after, the ceiling 90%', vis(pg, '#rlvr-pk') and Z(lang, 'before', '训练前') in pk and '90%' in pk and '90%' in ob, (pk[:200], ob[:300]))
            if not phone:
                v = in_stage(pg, '#rlvr-pk'); check(f'{tag} part 7: the pass@k chart is inside the stage’s visible box on entry (S1)', v['ok'], v)
            mk = pg.evaluate(MARKS, '#rlvr-pk'); check(f'{tag} part 7: both pass@k lines have a legend item in their own style (H6)', mk and mk['drawn'] == mk['named'] == ['dash', 'solid'], mk)
            bounds(pg, f'{tag} part 7')
            pg.evaluate("()=>{document.getElementById('rlvr-math').open=true}"); pg.wait_for_timeout(200)
            mb = pg.evaluate("""()=>[...document.querySelectorAll('#rlvr-mathbody .rlvr-mb')].map(b=>({h:(b.querySelector('.rlvr-mh')||{}).textContent||'', math:b.querySelectorAll('math').length, lead:!!b.querySelector('p')}))""")
            check(f'{tag} Show the math: a few blocks, each with a heading, a lead in words and typeset formulas (H2)', 4 <= len(mb) <= 6 and all(x['h'] and x['math'] >= 1 and x['lead'] for x in mb), mb)
            mo = pg.evaluate(MATHOK); check(f'{tag} Show the math: every formula is MathML, no TeX command shown as a word (H1)', mo['n'] >= 10 and not mo['bad'], mo)
            mt = text(pg, '#rlvr-mathbody')
            check(f'{tag} Show the math: no plain-text formula left (H1)', not [x for x in PLAIN_FX if re.search(x, mt)], [x for x in PLAIN_FX if re.search(x, mt)])
            pg.evaluate("()=>{document.getElementById('rlvr-math').open=false}")
            shot(pg, f'{tag}_p7', full=phone); rules(pg, f'{tag} part 7', lang)
            # ---- part 8
            click(pg, '#rlvr-rail [data-nav="next"]')
            t8 = text(pg, '#rlvr-rail .step.current')
            check(f'{tag} part 8: three points, three misconceptions, a way to the scenarios', t8.count('\n') > 5 and vis(pg, '#rlvr-rail [data-mode-go="sandbox"]'), t8[:300])
            shot(pg, f'{tag}_p8', full=phone); rules(pg, f'{tag} part 8', lang)
            # ---- Scenarios
            click(pg, '#rlvr-rail [data-mode-go="sandbox"]'); idle(pg)
            lead = pg.inner_text('#rlvr-rail .railcard > p.small')
            check(f'{tag} G1: the scenarios card leads in with what to pick and what to watch, not “one at a time”', lead == Z(lang, 'Pick a scenario and see how often the model is right in one try as it trains.', '选一个场景，看看训练时一次答对的比例怎么变。'), lead)
            check(f'{tag} scenarios: four, single choice', pg.evaluate("()=>document.querySelectorAll('#rlvr-rail .preset').length") == 4)
            for i in range(4):
                click(pg, f'#rlvr-rail [data-preset="{i}"]'); idle(pg)
                ins = text(pg, '#rlvr-insight')
                check(f'{tag} scenario {i + 1}: pressed, with its insight', pg.evaluate(f"()=>document.querySelector('#rlvr-rail [data-preset=\"{i}\"]').getAttribute('aria-pressed')") == 'true' and len(ins) > (30 if lang == 'zh' else 60), ins[:200])
            bounds(pg, f'{tag} scenarios')
            shot(pg, f'{tag}_sandbox', full=phone); rules(pg, f'{tag} scenarios', lang)
            hs = pg.evaluate('()=>location.hash'); st = state(pg)
            pg2 = page(b, w, h, lang, hash_=hs[1:]); s2 = state(pg2)
            check(f'{tag} scenarios: the URL carries the setup and a fresh page restores it', s2['mode'] == 'sandbox' and all(s2[k] == st[k] for k in ('chk', 'fmt', 'dyn', 'start', 'set')), (hs, s2))
            pg2.close()
            pg.evaluate("()=>document.querySelector('#rlvr-ctl-chk-wrap').scrollIntoView({block:'center'})"); pg.wait_for_timeout(300)
            y0 = pg.evaluate("()=>document.querySelector('#rlvr-ctl-chk-wrap').getBoundingClientRect().top")
            pg.evaluate("()=>document.querySelector('#rlvr-controls .seg-ctl[data-key=\"chk\"] button[data-v=\"loose\"]').click()"); pg.wait_for_timeout(300)
            y1 = pg.evaluate("()=>document.querySelector('#rlvr-ctl-chk-wrap').getBoundingClientRect().top")
            check(f'{tag} scenarios: a change under Settings does not move the page (R3.2)', abs(y1 - y0) < 1.5 and state(pg)['chk'] == 'loose', (y0, y1))
            idle(pg)
            # ---- the challenge
            mode(pg, 'challenge'); idle(pg)
            st = state(pg)
            check(f'{tag} challenge: starts from the loose checker on the new problems, 20 steps; Settings locked; three choices in the card',
                  st['mode'] == 'challenge' and st['chk'] == 'loose' and st['set'] == 'ch' and st['T'] == 20 and pg.evaluate("()=>document.querySelector('#rlvr-controls').classList.contains('locked')")
                  and pg.evaluate("()=>document.querySelectorAll('#rlvr-rail .seg-ctl[data-key^=\"ch\"]').length") == 3, st)
            check(f'{tag} challenge: live progress against the goal before Submit (R4.3)', vis(pg, '#rlvr-ch-g1') and '31%' in text(pg, '#rlvr-ch-g1') and '10%' in text(pg, '#rlvr-ch-g2'), (text(pg, '#rlvr-ch-g1'), text(pg, '#rlvr-ch-g2')))
            click(pg, '#rlvr-rail .seg-ctl[data-key="chC"] button[data-v="strict"]'); idle(pg)
            click(pg, '#rlvr-rail [data-act="submit"]')
            res = text(pg, '#rlvr-ch-res')
            check(f'{tag} challenge tier 1: not yet, with a hint and no answer', Z(lang, 'Not yet', '还没达标') in res and Z(lang, 'Hint', '提示') in res and not vis(pg, '#rlvr-ch-res .cmp'), res[:300])
            shot(pg, f'{tag}_ch1', full=phone)
            BRIEF = """()=>{const res=document.querySelector('#rlvr-ch-res'), c=res.querySelector('.cmp'); const rs=c?[...c.querySelectorAll('.cmp-row')]:[], tg=r=>!!r.querySelector('.cmp-tag,.cmp-you');
              return {n:rs.length, ok:rs.filter(r=>r.classList.contains('ok')).length, me:rs.filter(r=>r.classList.contains('me')).length, stray:rs.filter(r=>!r.classList.contains('ok')&&!r.classList.contains('me')&&!tg(r)).length,
                title:c?c.querySelector('.cmp-title').textContent:'', note:(res.querySelector('.cmp-note')||{}).textContent||'', reveal:(res.querySelector('[data-act="reveal"]')||{}).textContent||''};}"""
            bf = pg.evaluate(BRIEF)
            check(f'{tag} challenge tier 1: the reveal button says “{Z(lang, "Show the answer", "直接看答案")}” (the owner, batch 8)', bf['reveal'].strip() == Z(lang, 'Show the answer', '直接看答案'), bf)
            click(pg, '#rlvr-ch-res [data-act="reveal"]'); idle(pg)
            bf = pg.evaluate(BRIEF)
            check(f'{tag} challenge tier 1 revealed: the 2 passing options, the initial setup and your pick (strict checker only), no other; the note counts the other 4',
                  (bf['n'], bf['ok'], bf['me'], bf['stray']) == (4, 2, 1, 0) and Z(lang, 'The other 4 combinations', '其余 4 种组合') in bf['note'], bf)
            check(f'{tag} challenge: the chart title no longer claims every option', bf['title'] and not any(x in bf['title'] for x in ('each', 'every', '每种', '各')), bf['title'])
            click(pg, '#rlvr-rail .seg-ctl[data-key="chD"] button[data-v="1"]'); idle(pg)
            click(pg, '#rlvr-rail .seg-ctl[data-key="chF"] button[data-v="1"]'); idle(pg)
            res = text(pg, '#rlvr-ch-res')
            check(f'{tag} challenge tier 2: goal met, a smaller change also works', Z(lang, 'Goal met', '达成目标') in res and Z(lang, 'smaller change', '更少') in res, res[:300])
            click(pg, '#rlvr-rail .seg-ctl[data-key="chF"] button[data-v="0"]'); idle(pg)
            res = text(pg, '#rlvr-ch-res')
            check(f'{tag} challenge tier 3: goal met with the fewest changes, charted with the initial setup and the pick', Z(lang, 'already the smallest', '已经是最少') in res and vis(pg, '#rlvr-ch-res .cmp')
                  and Z(lang, 'initial setup', '初始设置') in res and Z(lang, 'your pick', '你的选择') in res, res[:400])
            bf = pg.evaluate(BRIEF)
            check(f'{tag} challenge tier 3: the chart holds the 2 passing options (your pick among them) and the initial setup; the note counts the other 5',
                  (bf['n'], bf['ok'], bf['me'], bf['stray']) == (3, 2, 1, 0) and Z(lang, 'The other 5 combinations', '其余 5 种组合') in bf['note'], bf)
            check(f'{tag} challenge: the capability card', vis(pg, '#rlvr-rail [data-exits]'))
            bounds(pg, f'{tag} challenge')
            ct = text(pg, '#rlvr-ch-res .cmp')
            check(f'{tag} challenge: the result chart says what it counts in plain words, no “候选数” (item 2, H6)', '候选数 ' not in ct and Z(lang, 'answers listing candidate numbers', '写候选数字的回答') in ct, ct[:300])
            pg.evaluate("()=>document.querySelector('#rlvr-app [data-act=\"drill-reveal\"]').click()"); pg.wait_for_timeout(200)
            dr = pg.evaluate("()=>({math:document.querySelectorAll('#rlvr-drill-a math').length, t:document.getElementById('rlvr-drill-a').innerText})")
            check(f'{tag} drill: the worked answer typesets its four steps, no “(1 − m) ÷ (std + 0.0001)” text (H1)', dr['math'] >= 4 and not [x for x in PLAIN_FX if re.search(x, dr['t'])], dr)
            mo = pg.evaluate(MATHOK); check(f'{tag}: every formula on the page renders as MathML, no TeX command shown as a word (H1)', not mo['bad'], mo)
            shot(pg, f'{tag}_ch3', full=phone); rules(pg, f'{tag} challenge', lang)
            other = 'zh' if lang == 'en' else 'en'
            pg.evaluate("()=>{document.getElementById('rlvr-asm-d').open=true}")
            pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(500); st = state(pg)
            check(f'{tag} challenge: the language switch keeps the answer', st['mode'] == 'challenge' and st['chk'] == 'strict' and st['dyn'] == 1 and st['chSubmitted'], st)
            check(f'{tag}: the language switch keeps the assumed values open once the reader opened them', pg.evaluate("()=>document.getElementById('rlvr-asm-d').open"))
            pg.evaluate("()=>{document.getElementById('rlvr-asm-d').open=false}")
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(300)
            check(f'{tag}: no page errors so far', not errs, errs[:3])
            pg.close()
    # item 1 / H7 at 900 px: the “优势” column ran past the group's right edge between 761 and 930 px; parts 3, 4 and 5, during play and after
    for lang in (LANGS if RUN900 else []):
        tag = f'{lang}-900'; pg = page(b, 900, 900, lang)
        click(pg, '#rlvr-rail [data-pred="S"]')
        pg.evaluate("()=>document.querySelector('#rlvr-rail [data-go=\"2\"]').click()"); pg.wait_for_timeout(700)
        check(f'{tag} part 3: the group plays on entry', state(pg)['grpPlaying'] or pg.evaluate('()=>matchMedia("(prefers-reduced-motion: reduce)").matches'), state(pg))
        bounds(pg, f'{tag} part 3 mid-play'); idle(pg); bounds(pg, f'{tag} part 3')
        pl = pg.evaluate(PLAYERS); check(f'{tag} part 3: the group’s replay and speed buttons share one height and one baseline (H7)', pl and all(x['ok'] for x in pl), pl)
        shot(pg, f'{tag}_p3'); rules(pg, f'{tag} part 3', lang)
        pg.evaluate("()=>document.querySelector('#rlvr-rail [data-go=\"3\"]').click()"); pg.wait_for_timeout(1500)
        bounds(pg, f'{tag} part 4 mid-run'); idle(pg); bounds(pg, f'{tag} part 4')
        shot(pg, f'{tag}_p4'); rules(pg, f'{tag} part 4', lang)
        pg.evaluate("()=>document.querySelector('#rlvr-rail [data-go=\"4\"]').click()"); pg.wait_for_timeout(300); settle(pg)
        click(pg, '#rlvr-rail [data-lever="loose"]'); idle(pg); bounds(pg, f'{tag} part 5')
        mk = pg.evaluate(MARKS, '#rlvr-p1'); check(f'{tag} part 5: every mark of the pass@1 chart has a legend item, red included (H6)', mk and mk['drawn'] == mk['named'], mk)
        shot(pg, f'{tag}_p5'); rules(pg, f'{tag} part 5', lang)
        check(f'{tag}: no page errors so far', not errs, errs[:3])
        pg.close()
    b.close()

for hit in sorted(set(rule_hits)): print('RULE', hit)
check('writing, colour and box rules on every visible text', not rule_hits, len(rule_hits))
check('no page errors', not errs, errs[:5])
# ---- G1/G2 (the owner's message of 2026-10-08): option lead-ins instead of “one at a time”, and no sentence that leans on a detail
# of an earlier part. Read from the lab's own source, so both languages are covered.
import re as _re
_SRC = ''.join((P / d / 'rlvr.js').read_text() for d in ('labs', 'teasers') if (P / d / 'rlvr.js').exists())
_G1 = _re.findall(r'.{0,30}(?:一次只能选|不会叠加|do not stack|[Oo]ne at a time|一次选一).{0,30}', _SRC)
check('G1: no “one at a time / do not stack” note left in the lab or its tile (EN and ZH)', not _G1, _G1[:3])
_NEW = ["The chart compares the model before training with the one after", "steps of training with the strict checker.", "图里比较训练前的模型，和用严格的判分程序训练", "So when a group has both right and wrong answers, the right ones are pushed up, and an all-wrong group teaches nothing. As training goes on, the way of writing that is right most often takes over.", "The checker is the reward model now; test it before training.", "If the checker has a loophole, the model finds it: when listing candidate numbers earns a reward, the model soon learns to guess. Before you train, try such answers on your checker.", "Watch pass@k, not only pass@1.", "RL often only makes the model use the ways of writing it already had more often: pass@1 rises, while the problems it solves with many tries do not grow.", "the problems no way could solve stayed unsolved. A study of real models found something similar.", "Tülu 3 的 RLVR 用的就是 PPO。所以一组回答有对有错时，答对的回答被加强。全错的一组，什么也学不到。训练下去，最常答对的那种写法就成了主流。", "判分程序现在就是奖励模型，训练前先检查它。", "判分程序有漏洞，模型就会去钻。只要写一串候选数字也能拿分，模型很快就学会了猜。训练之前，先拿这类回答去试一试你的判分程序。", "不只看 pass@1，也看 pass@k。", "RL 常常只是让模型更多地用它本来就会的写法。所以 pass@1 涨了，给很多次机会时能解出的题却没有变多。", "哪种写法都做不对的题，训练后还是做不对。一项对真实模型的研究，也有类似的发现。", "With nothing but 1 or 0 on the final answer, training took “solve, then check” from", "of the answers.", "只按最后的答案给 1 分或 0 分，训练下来，“算完再检查”这种写法就从", "without making sure the tests were still the original ones: a checker that is too loose.", "没确认测试还是原来那些：这是一个太松的判分程序。", "of the time on average. Real reasoning models", "A checker that is too loose is soon gamed.", "判分程序太松，模型很快就会钻它的空子。"]
check(f'G1/G2: the new lead-ins and the rewritten sentences are in the lab ({len(_NEW)})', all(x in _SRC for x in _NEW), [x for x in _NEW if x not in _SRC][:3])
_OLD = ["steps of part 4.", "图里比较训练前的模型和第 4 部分训练", "In part 3, a group with some right answers pushed the right ones up, and a group of all wrong answers taught nothing. Over", "In part 5 the loose checker paid for listing candidate numbers, and the model learned to guess. Feed your checker such answers before you train.", "In part 7, pass@1 rose from", "the problems no way could solve stayed unsolved. The study quoted in part 7 found something similar for real models.", "✗ “A verifiable reward cannot be gamed.” In part 5 a loose checker was gamed in", "Tülu 3 的 RLVR 用的就是 PPO。第 3 部分里，有对有错的一组，答对的回答被加强。全错的一组什么也没教。训练", "第 5 部分里，宽松的判分程序给写候选数字的回答发分，模型就学会了猜。训练之前，先拿这类回答去试一试你的判分程序。", "第 7 部分里，pass@1 从", "哪种写法都做不对的题，训练后还是做不对。第 7 部分引的那项研究，在真实模型上也有类似的发现。", "✗ “可验证的奖励不会被钻空子。”第 5 部分里，宽松的判分程序", "In part 4, “solve, then check” grew from", "第 4 部分只按最后的答案给分，“算完再检查”从", "without making sure the tests were still the original ones: like the loose checker of part 5.", "没确认测试还是原来那些：和第 5 部分的宽松判分程序一样。", "of the time on average: the numbers of part 1. Real reasoning models", "，也就是第 1 部分给的数。"]
check(f'G2: none of the {len(_OLD)} sentences that leaned on an earlier part (or said “one at a time”) is left', not any(x in _SRC for x in _OLD), [x for x in _OLD if x in _SRC][:3])
# the owner's review of 2026-10-09, read from the source with its comments taken out (both languages)
_CODE = _re.sub(r'(?m)^\s*//.*$', '', _re.sub(r'/\*.*?\*/', '', _SRC, flags=_re.S))
_FX = [x for x in ('ratio = new ÷ old', 'Advantage = reward −', '优势 = 奖励 −', 'std + 0.0001)', '(1 − m) ÷') if x in _CODE]
check('H1: no plain-text formula left in the source (ratio, advantage, the drill’s steps)', not _FX, _FX)
_H4 = _re.findall(r'.{0,20}(?:规划中|即将推出|[Pp]lanned|[Cc]oming soon).{0,20}', _CODE)
check('H4: no “planned” lab named as a destination', not _H4, _H4[:3])
_H10 = [x for x in ('Feed your checker', '往哪里钻') if x in _CODE]
check('H10: the vague phrases are gone (“Feed your checker”, “往哪里钻”)', not _H10, _H10)
_I2 = _re.findall(r'.{0,20}候选数(?!字).{0,20}', _CODE)
check('item 2: no “候选数” label left in the lab or its tile', not _I2, _I2[:3])
print(f'flow: {n - len(fails)} passed, {len(fails)} failed')
sys.exit(1 if fails else 0)
