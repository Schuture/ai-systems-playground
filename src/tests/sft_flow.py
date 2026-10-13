"""Flow test for the sft lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH:
- no page errors, no horizontal page scroll, no text wider than its box (R7.8), a tooltip for every help cursor (R1.1);
- part 1: the base model's generation plays at Medium (a mechanism replay, R3.9); the metric cards wait for part 3;
- R3.20: a picture watched to the end shows its end at once when it comes back (Back, the next part, a lever moved back, a language
  switch, a mode switch and back); nothing replays when nothing changed (the pressed lever again); a new picture replays from the
  first token that differs; “Replay” replays from the first token; Pause pauses; the scrubber jumps to the end;
- part 3: Next waits for the prediction with the kit's reason; before the pick the answer (6.25%) appears nowhere (R3.8); the pick
  shows it with the verdict;
- parts 4 and 5: Next waits for the other lever and the end of the run; Show me does it; the observation appears; part 5 marks the
  moment it should have stopped in red (R5.11), and the end token that lost its label;
- part 6: the bridge table (5 answer tokens) and the link to DPO; part 7: three points, three misconceptions, the way to Scenarios;
- Scenarios: four, single choice, the first pressed on entry, the URL carries the setup and a fresh page restores it;
- the challenge: Settings locked, feedback (not yet: does not stop; met with the fewest changes), “Show the answer”, and a brief chart
  (the owner's review of batch 8, 2026-10-07): only the one passing combination, your pick and the initial setup, with a note that
  counts the other combinations of the 16; the capability card; at 390 the dock once the chart scrolls away;
- the writing rules on every visible text node and tooltip (banned words, Chinese spacing and punctuation, two colons, R6.18),
  colours (red only for the failure moment and missed goals, R5.6/R5.11); “toy” / 玩具 only in the fidelity label (R6.11, S1 as corrected).
The owner's review of batch 9 (2026-10-08) and its site-wide rules:
- item 1 / G5: every formula (the cards, Show the math, the small model's hover) sits inside the box that clips it with room to spare:
  each MathML box at least 3 px from the edge, a fraction's numerator text inside it (the top of 120 over 1,919 was cut off);
- item 2 / G1: no “one at a time, they do not stack” note anywhere (source and page, aria labels included); the options of parts 4
  and 5 and the scenarios have a lead-in right above them that says what to pick and what to watch;
- item 3 / G2: no sentence leans on an earlier part by number; part 7 states each rule and what follows; the new cases restate their
  reference in full;
- item 4 / G4: part 5 says why a batch is padded (one matrix, rows of one length) before the end token as pad; the batch picture:
  3 rows of 21 positions, the one in the strip padded by 3, every real end token red when the pad token is the end token;
- G6: a run the reader starts in a guided part plays where it can be seen (390, and a 1280 × 720 window); a guided run that starts
  out of view waits at its first token until it is scrolled into view (390, parts 1 and 4); in Scenarios and Settings a new run ends within
  about 1.5 s.
Screenshots: shots/sft/.
Run: python3 tests/sft_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, subprocess, sys, re
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/sft.html', '--only', 'sft'], check=True, cwd=P, stdout=subprocess.DEVNULL)
URL = 'file://' + str(P / 'out' / 'sft.html')
SH = P / 'shots' / 'sft'; SH.mkdir(parents=True, exist_ok=True)
ARGS = sys.argv[1:]
LANGS = [x for x in ('en', 'zh') if x in ARGS] or ['en', 'zh']
SIZES = [s for s in [(1440, 900), (390, 844)] if str(s[0]) in ARGS] or [(1440, 900), (390, 844)]
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


# the owner's review of batch 9 (2026-10-08), G1 and G2, in the source: no single-choice note, no earlier part named by number
SRC = (P / 'labs' / 'sft.js').read_text()
G1RE = r'一次只能选|一次选一个|不会叠加|do not stack|one at a time|One at a time'
check('G1: no “one at a time, they do not stack” note in the lab source', not re.findall(G1RE, SRC), re.findall(G1RE, SRC))
G2RE = r'In part \d|part \d’s|第 \d 部分(?:里|的|中)'
check('G2: no sentence names an earlier part by number', not re.findall(G2RE, SRC), re.findall(G2RE, SRC))


RULES = """(sel,zh)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],
   [/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy” (only the fidelity label “Toy run” / 玩具实跑, R6.11)'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/Locked in|已锁定/,'“Locked in”'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万 instead of K, M, B (R6.18)'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour (R5.7)'],
   ...(zh?[[/\\bpolicy\\b|reward model|value model/,'English term inside Chinese (S2)']]:[])];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, .fid-tag, a')||!e.getClientRects().length) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{if(e.matches('a[href^="#"]')) return; const t=e.getAttribute('data-tipx'); for(const [re,name] of bad) if(re.test(t)) out.push(name+' (tooltip): '+t.slice(0,90));});
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
BIGNUM = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #sft-mathbody, .num-big, #sft-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|sft-bad|sft-red|sft-redt|sft-sw-bad|sft-lb|warn|wrong|ex-res|cmp|goal|progress/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#sft-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#sft-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg,.sft-math-line,.sft-dp-tw')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#sft-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""


# G5: each formula against the nearest box that clips it (overflow not visible): every MathML box at least 3 px inside; a fraction's
# numerator box at least 3 px from the top, and the text of the numerator inside the box
MATHROOM = """(sel)=>{const out=[]; for(const m of document.querySelectorAll(sel+' math')){if(!m.getClientRects().length) continue;
  let p=m.parentElement; while(p&&p!==document.body){const cs=getComputedStyle(p); if(cs.overflowX!=='visible'||cs.overflowY!=='visible') break; p=p.parentElement;}
  if(!p||p===document.body) continue; const r=p.getBoundingClientRect(), top=r.top+p.clientTop, bot=top+p.clientHeight;
  let t=1e9, b=1e9, nb=1e9, nt=1e9;
  for(const e of [m,...m.querySelectorAll('*')]){const q=e.getBoundingClientRect(); if(!q.width&&!q.height) continue; t=Math.min(t,q.top-top); b=Math.min(b,bot-q.bottom);}
  for(const nu of m.querySelectorAll('mfrac>:first-child')){nb=Math.min(nb,nu.getBoundingClientRect().top-top);
    const tw=document.createTreeWalker(nu,NodeFilter.SHOW_TEXT); let x; while((x=tw.nextNode())){const rg=document.createRange(); rg.selectNodeContents(x); for(const q of rg.getClientRects()) nt=Math.min(nt,q.top-top);}}
  out.push({f:m.textContent.replace(/\\s+/g,'').slice(0,28), box:String(p.className||p.id||p.tagName).slice(0,20), t:+t.toFixed(1), b:+b.toFixed(1), num:nb<1e9?+nb.toFixed(1):null, numText:nt<1e9?+nt.toFixed(1):null});}
 return out;}"""


def math_room(pg, sel, tag, what, want):
    rs = pg.evaluate("s=>(" + MATHROOM + ")(s)", sel)
    bad = [r for r in rs if r['t'] < 3 or r['b'] < 3 or (r['num'] is not None and (r['num'] < 3 or r['numText'] < 0.5))]
    check(f'{tag}: {what}: {len(rs)} formulas, each whole inside its box with room to spare (item 1, G5)', len(rs) >= want and not bad, (len(rs), bad[:3]))
    return rs


# G1 on the page: visible text, tooltips and aria labels
G1JS = """(re)=>{const root=document.querySelector('#sft-app'); const rx=new RegExp(re); const out=[]; if(!root) return out;
  if(rx.test(root.innerText)) out.push('text: '+(root.innerText.match(new RegExp('.{0,30}('+re+').{0,10}'))||[''])[0]);
  root.querySelectorAll('[aria-label],[data-tipx]').forEach(e=>{const t=(e.getAttribute('aria-label')||'')+' '+(e.getAttribute('data-tipx')||''); if(rx.test(t)) out.push('attr: '+t.slice(0,60));});
  return out;}"""


def page(b, w, h, lang, hash_='sft'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    pg.set_default_timeout(10000)
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    # the page loading can take more than 10 s when up to eleven agents share the machine: wait up to 40 s for it
    pg.goto(URL, timeout=40000); pg.wait_for_function('()=>!!(window.ASP&&window.__sft)', timeout=40000); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(600)
    return pg, errs


def st(pg):
    return pg.evaluate('()=>window.__sft.state()')


def click(pg, sel):
    return pg.evaluate(f"""()=>{{const e=[...document.querySelectorAll('{sel}')].find(x=>x.offsetParent!==null)||document.querySelector('{sel}'); if(!e) return 0; e.click(); return 1;}}""")


def next_off(pg):
    return pg.evaluate("()=>{const b=document.querySelector('#sft-rail [data-nav=\"next\"]'); return !!b&&(b.disabled||b.getAttribute('aria-disabled')==='true')}")


def nxt(pg):
    click(pg, '#sft-rail [data-nav="next"]'); pg.wait_for_timeout(350)


def mode(pg, m):
    pg.evaluate(f"()=>{{const b=[...document.querySelectorAll('.modes button[data-mode=\"{m}\"]')].find(e=>e.offsetParent!==null); b.click();}}"); pg.wait_for_timeout(400)


def to_end(pg):
    pg.evaluate("()=>{const i=document.querySelector('#sft-player input[data-key=\"scrub\"]'); if(!i) return; i.value=99; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(250)


def done(s):
    return s['n'] > 0 and s['i'] >= s['n'] and not s['playing']


def text(pg, sel):
    return pg.evaluate(f"()=>{{const e=document.querySelector('{sel}'); return e?e.innerText:''}}")


# the owner's clarification (2026-10-07 23:37 UTC): no hover only expands a shown number to its exact value; Show the math may write one out
EXACT_HOVER = """r=>{const root=document.querySelector(r), out=[]; if(!root) return out;
  for(const e of root.querySelectorAll('[data-tipx],[data-tip]')){if(e.closest('.mathbody')) continue; let t=e.getAttribute('data-tipx')||''; const k=e.getAttribute('data-tip');
    if(k&&window.ASP) t+=' '+(window.ASP.t(k)||''); if(/精确值是|Exactly/.test(t)) out.push(t.slice(0,100));}
  for(const e of root.querySelectorAll('.num-big')) if(!e.closest('.mathbody')&&(e.hasAttribute('data-tipx')||e.closest('.def'))) out.push('hover on '+e.textContent);
  return out;}"""


# H1 (the owner, 2026-10-09): no formula written as plain text (text, hovers, the tip on screen), and no TeX command left as a word in MathML
PLAINRE = r'−log p|log π\(|π_?ref|x_t|W_j|Σ_|÷ 全部|tokens ÷'
PLAINF = """([r,re])=>{const root=document.querySelector(r), out=[]; if(!root) return out; const rx=new RegExp(re); let n;
  const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
  while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('math,pre,code,script,style')) continue; if(rx.test(n.textContent)) out.push('text: '+n.textContent.trim().slice(0,60));}
  for(const e of root.querySelectorAll('[data-tipx]')){const t=e.getAttribute('data-tipx'); if(rx.test(t)) out.push('hover: '+t.slice(0,60));}
  const tip=document.getElementById('tip'); if(tip&&!tip.hidden){const w=document.createTreeWalker(tip,NodeFilter.SHOW_TEXT); while((n=w.nextNode())){if(n.parentElement.closest('math')) continue; if(rx.test(n.textContent)) out.push('tip: '+n.textContent.trim().slice(0,60));}}
  for(const m of document.querySelectorAll('math')) for(const i of m.querySelectorAll('mi')) if(/^(frac|sqrt|left|right|hat|text|cdot|times|div|approx|sum|mathrm|operatorname|quad|beta|pi|sigma|rho|theta|bar|tilde|overline|langle|rangle|mid|le|ge|big|Big)$/.test(i.textContent)) out.push('TeX word: '+i.textContent+' in '+m.textContent.slice(0,40));
  return out;}"""
EQJS = """(sel)=>{const b=document.querySelector(sel); const eqs=[...b.querySelectorAll('.sft-eq')];
  const names=eqs.map(e=>(e.querySelector('.sft-eq-n')||{}).textContent||''); let aligned=true;
  for(const e of eqs){const os=[...e.querySelectorAll('.sft-eq-o math')].map(m=>Math.round(m.getBoundingClientRect().left)); if(new Set(os).size>1) aligned=false;}
  const hs=[...b.querySelectorAll('.sft-mh')]; const ruleOk=hs.every(h=>{const cs=getComputedStyle(h); return cs.borderTopWidth==='1px'&&cs.borderTopStyle==='solid';});
  return {n:eqs.length, names:names.map(x=>x.trim().slice(0,20)), aligned, rules:hs.length, ruleOk};}"""


def rules(pg, tag, lang):
    pf = pg.evaluate(PLAINF, ['#sft-app', PLAINRE]); check(f'{tag}: H1: no formula written as plain text; every formula rendered (no TeX word)', not pf, pf[:3])
    hx = pg.evaluate(EXACT_HOVER, '#sft-app'); check(f'{tag}: no exact-value hover outside Show the math', not hx, hx[:3])
    for name, js in (('rules', RULES), ('spacing', SPACING), ('punct', PUNCT), ('colons', COLONS), ('bignum', BIGNUM)):
        if name in ('spacing', 'punct', 'colons') and lang != 'zh': continue
        hits = pg.evaluate("([s,z])=>(" + js + ")(s,z)", ['#sft-app', lang == 'zh'])
        check(f'{tag}: writing rules ({name})', not hits, hits[:3])
    check(f'{tag}: no horizontal page scroll', pg.evaluate('()=>document.documentElement.scrollWidth<=innerWidth+1'), pg.evaluate('()=>[document.documentElement.scrollWidth,innerWidth]'))
    over = pg.evaluate(OVER); check(f'{tag}: no text wider than its box', not over, over)
    hp = pg.evaluate(HELP); check(f'{tag}: every help cursor has a tooltip', not hp, hp)
    g1 = pg.evaluate("r=>(" + G1JS + ")(r)", G1RE); check(f'{tag}: G1: no “one at a time, they do not stack” note on the page', not g1, g1)
    dup = pg.evaluate("()=>{const c={}; document.querySelectorAll('[id]').forEach(e=>{c[e.id]=(c[e.id]||0)+1}); return Object.entries(c).filter(([k,v])=>v>1)}")
    check(f'{tag}: no duplicate ids (the kit owns sft-fid, sft-pri, sft-share)', not dup, dup)


# the challenge's chart after Submit: its rows (passing, your pick, tagged, stray = none of these), its title and the note under it
CHART = """()=>{const res=document.querySelector('#sft-ch-res'), cmp=res&&res.querySelector('.cmp'); if(!cmp) return {rows:0,ok:0,me:0,tag:0,stray:0,title:'',note:'',labels:[]};
  const rows=[...cmp.querySelectorAll('.cmp-row')], tg=r=>!!r.querySelector('.cmp-tag,.cmp-you');
  return {rows:rows.length, ok:rows.filter(r=>r.classList.contains('ok')).length, me:rows.filter(r=>r.classList.contains('me')).length, tag:rows.filter(tg).length,
    stray:rows.filter(r=>!r.classList.contains('ok')&&!r.classList.contains('me')&&!tg(r)).length, title:(cmp.querySelector('.cmp-title')||{}).textContent||'',
    note:(res.querySelector('.cmp-note')||{}).textContent||'', labels:rows.map(r=>r.querySelector('.cmp-lab').textContent)};}"""


# item 4: the batch picture of part 5
BATCH = """()=>{const b=document.querySelector('#sft-batch'); if(!b||b.hidden||!b.getClientRects().length) return null; const box=document.querySelector('#sft-stripbox').getBoundingClientRect();
  const rows=[...b.querySelectorAll('.sft-bcells')]; return {rows:rows.map(r=>r.children.length), pad:rows.map(r=>r.querySelectorAll('.sft-bc-x').length), red:b.querySelectorAll('.sft-bc-e.sft-bad').length,
  inside:rows.every(r=>r.getBoundingClientRect().right<=box.right+0.5), strip:document.querySelectorAll('#sft-strip .sft-tpad').length, cap:(b.querySelector('.sft-cap')||{}).textContent||''};}"""
# G6: the box where the run plays is on screen (inside the sticky stage's visible part on wide screens), caption, player and first tokens
GENVIS = """()=>{const g=document.querySelector('#sft-genbox'), r=g.getBoundingClientRect(), st=document.querySelector('#sft-stage'), bar=document.querySelector('.topbar').getBoundingClientRect().bottom;
  const dk=document.getElementById('dock'), dh=dk&&!dk.hidden?dk.getBoundingClientRect().height:0; let v0=bar+dh, v1=innerHeight;
  if(getComputedStyle(st).position==='sticky'){const sr=st.getBoundingClientRect(); v0=Math.max(v0,sr.top); v1=Math.min(v1,sr.bottom);}
  const need=Math.min(r.height,170); return {ok:r.top>=v0-2&&r.top+need<=v1+2, top:Math.round(r.top), v0:Math.round(v0), v1:Math.round(v1), playing:window.__sft.state().playing};}"""
LEAD = "s=>{const l=document.querySelector(s); const p=l&&l.previousElementSibling; return p?p.textContent.trim():''}"
# G6: the row of generated tokens can be seen (the rule the guided run waits on)
GENIN = """()=>{const r=document.querySelector('#sft-gen').getBoundingClientRect(), st=document.querySelector('#sft-stage'); let v0=document.querySelector('.topbar').getBoundingClientRect().bottom, v1=innerHeight;
  if(getComputedStyle(st).position==='sticky'){const sr=st.getBoundingClientRect(); v0=Math.max(v0,sr.top); v1=Math.min(v1,sr.bottom);} return r.top<v1-16&&r.bottom>v0+16;}"""


def shot(pg, name):
    pg.evaluate("()=>{const s=document.querySelector('#sft-stage'); if(s) s.scrollIntoView({block:'start'});}"); pg.wait_for_timeout(200)
    pg.screenshot(path=str(SH / f'{name}.png'))


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in LANGS:
        for (w, h) in SIZES:
            tag = f'{lang} {w}'
            phone = w < 700
            pg, errs = page(b, w, h, lang)
            Z = lambda en, zh: zh if lang == 'zh' else en
            # ---------- part 1: the base model's generation, Medium (R3.9)
            s = st(pg)
            check(f'{tag} part 1: the base model’s generation plays at Medium', s['step'] == 0 and s['playing'] and s['spd'] == 'med' and s['n'] == 16, s)
            check(f'{tag} part 1: the metric cards wait for part 3', pg.evaluate("()=>document.querySelector('#sft-metrics').hidden"))
            pg.wait_for_timeout(900); s2 = st(pg)
            if not pg.evaluate(GENIN):
                # G6: below the fold on a phone, the guided run waits at its first token until its row is scrolled into view
                check(f'{tag} part 1: off screen, the run waits at its first token (G6)', s2['playing'] and s2['i'] == 0, (s['i'], s2['i']))
                pg.evaluate("()=>document.querySelector('#sft-genbox').scrollIntoView({block:'center'})"); pg.wait_for_timeout(300)
                s = st(pg); pg.wait_for_timeout(900); s2 = st(pg)
            check(f'{tag} part 1: the tokens appear one by one', s2['i'] > s['i'] and s2['i'] < s2['n'], (s['i'], s2['i']))
            to_end(pg); s = st(pg)
            check(f'{tag} part 1: the scrubber to the end finishes the run, remembered', done(s) and s['sig'] in s['watched'], s)
            check(f'{tag} part 1: the button now says Replay', pg.evaluate("()=>document.querySelector('#sft-player [data-act=\"play\"]').textContent") == Z('↺ Replay', '↺ 重放'))
            check(f'{tag} part 1: it never stops (16 tokens, no end token)', pg.evaluate("()=>document.querySelectorAll('#sft-gen .sft-stop').length") == 0 and pg.evaluate("()=>document.querySelectorAll('#sft-gen .sft-red').length") == 1)
            pg.focus('#sft-rail .t[data-ctl="toy"]'); pg.wait_for_timeout(250)
            tp = pg.evaluate("()=>{const t=document.getElementById('tip'); return t&&!t.hidden?{math:t.querySelectorAll('math').length, txt:t.innerText}:null}")
            check(f'{tag} part 1: “small model” hover: its size in words first, then the two formulas typeset (R6.29, review S5)', bool(tp) and tp['math'] == 4 and '5,434' in tp['txt'] and '\\sum' not in tp['txt'], tp)
            math_room(pg, '#tip', f'{tag} part 1', 'the “small model” hover', 2)
            pg.evaluate("()=>{document.activeElement&&document.activeElement.blur(); window.ASP.tip.hide()}"); pg.wait_for_timeout(100)
            rules(pg, f'{tag} part 1', lang); shot(pg, f'{lang}-{w}-1-mission')
            # ---------- part 2: a new picture replays; Back shows the watched run at once
            nxt(pg); s = st(pg)
            check(f'{tag} part 2: the template run is a new picture: it plays from the first token', s['step'] == 1 and s['playing'] and s['i'] < 1, s)
            check(f'{tag} part 2: the strip shows the special tokens in the Llama 3.1 format', pg.evaluate("()=>[...document.querySelectorAll('#sft-strip .sft-sp')].map(e=>e.textContent).join(' ')").count('<|start_header_id|>') == 2)
            to_end(pg)
            click(pg, '#sft-rail [data-nav="back"]'); pg.wait_for_timeout(300); s = st(pg)
            check(f'{tag} R3.20: Back to part 1 shows its end at once, no replay', s['step'] == 0 and done(s), s)
            nxt(pg); s = st(pg)
            check(f'{tag} R3.20: part 2 again, already watched: no replay', s['step'] == 1 and done(s), s)
            rules(pg, f'{tag} part 2', lang); shot(pg, f'{lang}-{w}-2-template')
            # ---------- part 3: predict (R3.8, R3.21)
            nxt(pg); s = st(pg)
            check(f'{tag} part 3: Next waits for the prediction', s['step'] == 2 and next_off(pg), s['step'])
            why = text(pg, '#sft-rail')
            check(f'{tag} part 3: the kit’s reason under the greyed-out Next', Z('Pick an answer', '选一个答案') in why)
            lab_txt = pg.evaluate("()=>{const a=document.querySelector('#sft-app'); let t=a.innerText; a.querySelectorAll('[data-tipx]').forEach(e=>t+=' '+e.getAttribute('data-tipx')); return t}")
            check(f'{tag} part 3: the answer appears nowhere before the guess', '6.25' not in lab_txt and '1,919' not in lab_txt, re.findall(r'.{20}6\.25.{20}', lab_txt)[:2])
            check(f'{tag} part 3: the question gives the counts', '1,800' in why and '120' in why)
            hid = text(pg, '#sft-sharebar')
            check(f'{tag} part 3: the hidden bar says when it shows, without naming a part (G2)', hid.strip() == Z('Shown once you pick an answer.', '选好答案后再显示。'), hid)
            shot(pg, f'{lang}-{w}-3-predict')
            click(pg, '#sft-rail [data-pred="10to50"]'); pg.wait_for_timeout(300)
            obs = text(pg, '#sft-rail [data-obs]')
            check(f'{tag} part 3: the share, the verdict and the factor', '6.25%' in obs and Z('the right answer is', '正确答案是') in obs and '16' in obs, obs[:200])
            check(f'{tag} part 3: Next open after the pick', not next_off(pg))
            sub = pg.evaluate("()=>document.querySelector('#sft-metrics [data-k=\"sh\"] .sub').textContent")
            check(f'{tag} part 3: the share card says it was computed with every token counting (review B1)', sub == Z('every token counts', '每个 token 都算'), sub)
            check(f'{tag} part 3: the bar of counted tokens is drawn', pg.evaluate("()=>{const r=document.querySelector('#sft-sharebar .sft-bar'); return !!r&&r.getBoundingClientRect().height>8}"))
            pg.evaluate("()=>{document.querySelector('#sft-math').open=true}"); pg.wait_for_timeout(150)
            rs = math_room(pg, '#sft-app', f'{tag} part 3', 'the loss in the card and Show the math', 6)
            fr = [r for r in rs if r['f'].startswith('1201,799')]   # H2 (2026-10-09): the share's fraction with the reader's numbers, 120 over 1,799 + 120
            check(f'{tag} part 3: Show the math: 120 over 1,919 with its numerator whole (the owner’s item 1)', len(fr) == 1 and fr[0]['num'] >= 3 and fr[0]['numText'] >= 0.5, fr)
            pg.evaluate("()=>{const d=document.querySelector('#sft-math'); d.open=true; d.scrollIntoView({block:'nearest'}); [...d.querySelectorAll('.sft-math-line')].find(e=>/799/.test(e.textContent)).scrollIntoView({block:'center'})}"); pg.wait_for_timeout(200)
            pg.screenshot(path=str(SH / f'{lang}-{w}-3-math.png'))
            pg.evaluate("()=>{document.querySelector('#sft-math').open=false}"); pg.wait_for_timeout(100)
            rules(pg, f'{tag} part 3', lang); shot(pg, f'{lang}-{w}-3-done')
            # ---------- part 4: every token vs the answer only; R3.20 in full
            nxt(pg); s = st(pg)
            check(f'{tag} part 4: enters with every token counting, a new picture plays', s['step'] == 3 and s['count'] == 'all' and s['playing'], s)
            check(f'{tag} part 4: Next waits for the other lever', next_off(pg))
            ld = pg.evaluate(LEAD, '#sft-rail .step.current .levers')
            check(f'{tag} part 4: G1 lead-in right above the options', ld.endswith(Z('Pick which tokens count, and see how the trained small model answers.', '选一种算 loss 的方式，看看训练出来的小模型怎么回答。')), ld[-90:])
            check(f'{tag} part 4: the batch picture waits for part 5', pg.evaluate(BATCH) is None)
            if phone:   # G6: after Next the card is in view and the written tokens are below the fold: the run waits at its first token
                pg.wait_for_timeout(400); s0 = st(pg)   # the page has settled after Next (it may pass over the tokens while it scrolls)
                off = pg.evaluate("()=>{const g=document.querySelector('#sft-gen').getBoundingClientRect(); return g.top>innerHeight-16||g.bottom<70}")
                pg.wait_for_timeout(700); s1 = st(pg)
                pg.evaluate("()=>document.querySelector('#sft-genbox').scrollIntoView({block:'center'})"); pg.wait_for_timeout(900); s2 = st(pg)
                check(f'{tag} part 4 (G6): out of view, the new run waits near its first token; scrolled into view, it plays',
                      off and s1['playing'] and abs(s1['i'] - s0['i']) < 1e-9 and s1['i'] < 1 and s2['i'] > s1['i'] + 0.5, (off, s0['i'], s1['i'], s2['i']))
            to_end(pg)
            click(pg, '#sft-rail [data-lever="all"]'); pg.wait_for_timeout(250); s = st(pg)
            check(f'{tag} R3.20: the pressed lever again changes nothing: no replay', done(s), s)
            click(pg, '#sft-rail [data-lever="resp"]'); pg.wait_for_timeout(150); s = st(pg)
            check(f'{tag} R3.20: the other lever is a new picture: it replays from the first token that differs', s['count'] == 'resp' and s['playing'] and s['i'] < 1, s)
            click(pg, '#sft-player [data-act="play"]'); pg.wait_for_timeout(150); s1 = st(pg); pg.wait_for_timeout(400); s2 = st(pg)
            check(f'{tag} R3.20: Pause pauses', not s1['playing'] and abs(s1['i'] - s2['i']) < 1e-9, (s1['i'], s2['i']))
            check(f'{tag} part 4: Next still waits for the end of the run', next_off(pg))
            to_end(pg)
            obs = text(pg, '#sft-rail [data-obs]')
            check(f'{tag} part 4: the observation and Next', 'KV cache' in obs and not next_off(pg), obs[:160])
            check(f'{tag} part 4: the model stops (green end token)', pg.evaluate("()=>document.querySelectorAll('#sft-gen .sft-stop').length") == 1)
            sub = pg.evaluate("()=>document.querySelector('#sft-metrics [data-k=\"sh\"] .sub').textContent")
            check(f'{tag} part 4: the share card follows the lever (only the answer counts)', sub == Z('only the answer counts', '只算回答部分'), sub)
            pg.evaluate("()=>{document.querySelector('#sft-math').open=true}"); pg.wait_for_timeout(150)
            mb = pg.evaluate("()=>{const m=document.querySelector('#sft-mathbody'); return {math:m.querySelectorAll('math').length, txt:m.innerText}}")
            check(f'{tag} Show the math: the small model’s formulas typeset, no TeX source (R6.29, review S5)', mb['math'] >= 5 and '_{' not in mb['txt'] and 'Σ_' not in mb['txt'], (mb['math'], mb['txt'][-200:]))
            # H2 (the owner, 2026-10-09): short blocks with a hairline between them; each formula named in words on the left, its chain on lines that line up
            eq = pg.evaluate(EQJS, '#sft-mathbody')
            check(f'{tag} Show the math: H2: the loss, the share, the weight, the logits, the probabilities and the update each named on the left', eq['n'] >= 6 and all(eq['names']) and eq['aligned'], eq)
            check(f'{tag} Show the math: H5: a 1px hairline in the line colour above each block heading', eq['rules'] >= 2 and eq['ruleOk'], eq)
            pg.evaluate("()=>{document.querySelector('#sft-math').open=false}"); pg.wait_for_timeout(100)
            click(pg, '#sft-rail [data-lever="all"]'); pg.wait_for_timeout(250); s = st(pg)
            check(f'{tag} R3.20: back to a run already watched: its end at once', s['count'] == 'all' and done(s), s)
            click(pg, '#sft-rail [data-lever="resp"]'); pg.wait_for_timeout(250); s = st(pg)
            check(f'{tag} R3.20: and back again: no replay', s['count'] == 'resp' and done(s), s)
            click(pg, '#sft-player [data-act="play"]'); pg.wait_for_timeout(150); s = st(pg)
            check(f'{tag} R3.20: “Replay” replays from the first token', s['playing'] and s['i'] < 1, s)
            to_end(pg)
            other = 'zh' if lang == 'en' else 'en'
            pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400); s = st(pg)
            check(f'{tag} R3.20: a language switch keeps the part and the setting, no replay', s['step'] == 3 and s['count'] == 'resp' and done(s), s)
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
            mode(pg, 'challenge'); mode(pg, 'guided'); s = st(pg)
            check(f'{tag} R3.20: the challenge and back: same part, no replay', s['step'] == 3 and done(s), s)
            if phone:
                pg.evaluate("()=>{const t=document.getElementById('sft-graph'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+60)}"); pg.wait_for_timeout(700)
                dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, status:(document.getElementById('dkStatus')||{}).textContent||'', read:(document.getElementById('dkRead')||{}).textContent||''}}")
                check(f'{tag} part 4: the dock with a short status and the share', dk['shown'] and 0 < len(dk['status']) <= 14 and '%' in dk['read'], dk)
            rules(pg, f'{tag} part 4', lang); shot(pg, f'{lang}-{w}-4-resp')
            # ---------- part 5: the pad token is the end token
            nxt(pg); s = st(pg)
            check(f'{tag} part 5: enters on a run already watched: no replay; Next waits', s['step'] == 4 and done(s) and next_off(pg), s)
            card = text(pg, '#sft-rail .step.current')
            i_why, i_eot = card.find(Z('stacked into one matrix', '叠成一个矩阵')), card.find(Z('pad with the end token instead', '拿结束 token 来补'))
            check(f'{tag} part 5: why a batch is padded comes before padding with the end token (the owner’s item 4, G4)', 0 <= i_why < i_eot, (i_why, i_eot))
            ld = pg.evaluate(LEAD, '#sft-rail .step.current .levers')
            check(f'{tag} part 5: G1 lead-in right above the options', ld.endswith(Z('Pick a pad token, and see whether the trained small model stops after its answer.', '选一种 pad token，看看训练出来的小模型答完停不停。')), ld[-90:])
            bt = pg.evaluate(BATCH)
            check(f'{tag} part 5: the batch picture: 3 rows of 21 positions, padded by 0, 3 (the strip’s 3) and 7, inside its box, no red end token with a separate pad token',
                  bool(bt) and bt['rows'] == [21, 21, 21] and bt['pad'] == [0, 3, 7] and bt['strip'] == 3 and bt['inside'] and bt['red'] == 0 and Z('one matrix', '一个矩阵') in bt['cap'], bt)
            click(pg, '#sft-rail [data-nav="showme"]'); pg.wait_for_timeout(150); s = st(pg)
            check(f'{tag} part 5: Show me picks “pad = end token” and plays', s['pad'] == 'eot' and s['playing'], s)
            pg.wait_for_timeout(750); gv = pg.evaluate(GENVIS)
            check(f'{tag} part 5 (G6): the run the reader started plays where it can be seen', gv['ok'] and gv['playing'], gv)
            to_end(pg)
            bt = pg.evaluate(BATCH)
            check(f'{tag} part 5: the pad token is the end token: every row’s real end token red in the batch picture', bool(bt) and bt['red'] == 3, bt)
            check(f'{tag} part 5: red where it should have stopped (R5.11), red end token in the strip', pg.evaluate("()=>document.querySelectorAll('#sft-gen .sft-red').length") == 1 and pg.evaluate("()=>document.querySelectorAll('#sft-strip .sft-bad').length") == 1)
            obs = text(pg, '#sft-rail [data-obs]')
            check(f'{tag} part 5: the observation names the fixes; Next open', '<|finetune_right_pad_id|>' in obs and not next_off(pg), obs[:160])
            rules(pg, f'{tag} part 5', lang); shot(pg, f'{lang}-{w}-5-eot')
            # ---------- part 6: the bridge to DPO
            nxt(pg)
            # Codex R11 (round 3): the table and the derivation sit on the stage, step by step; the rail keeps one short paragraph
            rl = pg.evaluate("()=>{const b=document.querySelector('#sft-rail .step.current .step-b')||document.querySelector('#sft-rail .step.current'); return {p:b.querySelectorAll('p').length, tbl:b.querySelectorAll('table').length, math:b.querySelectorAll('math').length}}")
            check(f'{tag} part 6 (R11): the rail keeps one short paragraph, no table and no formula; it names DPO', rl == {'p': 1, 'tbl': 0, 'math': 0} and 'DPO' in text(pg, '#sft-rail .step.current'), rl)
            check(f'{tag} part 6 (R11): the stage shows the table of the 5 answer tokens; the strip, the generation and the metric cards make way', pg.evaluate("()=>document.querySelectorAll('#sft-dpobox:not([hidden]) .sft-tbl tbody tr').length") == 5
                  and pg.evaluate("()=>['#sft-stripbox','#sft-genbox','#sft-sharebox','#sft-metrics'].every(s=>document.querySelector(s).hidden)"))
            DP = """()=>{const b=document.querySelector('#sft-dpobox'), t=b.querySelector('.sft-tbl'), w=b.querySelector('.sft-dp-tw'), cur=[...t.querySelectorAll('tbody tr.sft-dp-cur')].map(r=>r.rowIndex-1);
              const hs=[...t.querySelectorAll('tbody tr')].map(r=>r.getBoundingClientRect().height), th=[...t.querySelectorAll('th,td')].filter(c=>c.getClientRects().length);
              const lines=th.filter(c=>{const r=document.createRange(); r.selectNodeContents(c); const rs=[...r.getClientRects()].filter(x=>x.width>0&&x.height>0); if(!rs.length) return false; const b0=Math.min(...rs.map(x=>x.bottom)); return rs.some(x=>x.top>=b0-1);}).map(c=>c.textContent.trim());
              return {cols:t.querySelectorAll('thead th').length, foot:!!t.querySelector('tfoot'), cur, eq:b.querySelectorAll('.sft-eq').length, math:b.querySelectorAll('math').length,
                k:document.querySelector('#sft-dp-k [aria-pressed="true"]').dataset.v, now:b.querySelector('.sft-dp-now').innerText, wrapped:lines, hmax:Math.max(...hs), hmin:Math.min(...hs),
                scroll:w.scrollWidth-w.clientWidth, status:document.querySelector('#sft-status').textContent}}"""
            d0 = pg.evaluate(DP)
            check(f'{tag} part 6 (R11): step 1 shows token, logit and probability, the first token’s row highlighted; the status names the step', d0['cols'] == 3 and not d0['foot'] and d0['cur'] == [0] and d0['k'] == '0' and Z('Step 1 of 4', '第 1 步，共 4 步') == d0['status'], d0)
            check(f'{tag} part 6 (R11): no table cell wraps onto a second line; every row one line high', not d0['wrapped'] and d0['hmax'] - d0['hmin'] < 2, d0)
            if w >= 900:
                check(f'{tag} part 6 (R11): at desktop width the table needs no sideways scroll', d0['scroll'] <= 1, d0['scroll'])
            click(pg, '#sft-dpobox [data-brow="2"]'); pg.wait_for_timeout(150)
            d1 = pg.evaluate(DP)
            check(f'{tag} part 6 (R11): clicking a token moves the highlight to its row, and the text follows it', d1['cur'] == [2] and '75.96%' in d1['now'], d1)
            seen = []
            for k in range(1, 4):
                click(pg, '#sft-dpobox [data-act="bk-next"]'); pg.wait_for_timeout(150); seen.append(pg.evaluate(DP))
            check(f'{tag} part 6 (R11): each Next step adds one thing: the log p column, then the sum, then the reference column with the log-ratio',
                  [x['cols'] for x in seen] == [4, 4, 5] and [x['foot'] for x in seen] == [False, True, True] and [x['eq'] for x in seen] == [0, 1, 1]
                  and seen[0]['cur'] == [2] and seen[1]['cur'] == [] and seen[2]['cur'] == [2] and [x['k'] for x in seen] == ['1', '2', '3'], [(x['cols'], x['foot'], x['eq'], x['cur'], x['k']) for x in seen])
            check(f'{tag} part 6 (R11): π and π_ref are explained next to the formula', 'π' in seen[2]['now'] and Z('reference model', '参考模型') in seen[2]['now'] and Z('before SFT', 'SFT 之前') in seen[2]['now'], seen[2]['now'][:300])
            check(f'{tag} part 6 (R11): the last step still has no wrapped cell', not seen[2]['wrapped'], seen[2]['wrapped'])
            if w >= 900:
                check(f'{tag} part 6 (R11): at desktop width the full table needs no sideways scroll', seen[2]['scroll'] <= 1, seen[2]['scroll'])
            check(f'{tag} part 6: the formulas in MathML on the stage', seen[2]['math'] >= 1)
            math_room(pg, '#sft-dpobox', f'{tag} part 6', 'the bridge formulas', 3)
            check(f'{tag} part 6: the batch picture only in part 5', pg.evaluate(BATCH) is None)
            rules(pg, f'{tag} part 6', lang); shot(pg, f'{lang}-{w}-6-dpo')
            # ---------- part 7
            nxt(pg)
            check(f'{tag} part 7: three points, three misconceptions, the way to Scenarios', pg.evaluate("()=>document.querySelectorAll('#sft-rail .sft-rem li').length") == 3
                  and pg.evaluate("()=>[...document.querySelectorAll('#sft-rail .obs p')].filter(p=>p.textContent.startsWith('✗')).length") == 3
                  and pg.evaluate("()=>!!document.querySelector('#sft-rail [data-mode-go=\"sandbox\"]')"))
            rem = text(pg, '#sft-rail .sft-rem')
            check(f'{tag} part 7 (the owner’s item 3, G2): each point states the rule and what follows, no detail of an earlier part',
                  not re.search(r'[Pp]art \d|第 \d 部分|6\.25', rem)
                  and Z('the answer gets only a small share of the loss', '回答在 loss 里就只占一小份') in rem
                  and Z('the model never learns to stop', '模型就学不会停') in rem
                  and Z('it cannot tell where its answer starts or where to stop', '它认不出该从哪里开始答、在哪里停') in rem, rem)
            rules(pg, f'{tag} part 7', lang); shot(pg, f'{lang}-{w}-7-remember')
            # ---------- Scenarios
            mode(pg, 'sandbox')
            pr = pg.evaluate("()=>[...document.querySelectorAll('#sft-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
            check(f'{tag} scenarios: four, the first pressed on entry', pr == ['true', 'false', 'false', 'false'], pr)
            check(f'{tag} scenarios: Settings open', not pg.evaluate("()=>document.querySelector('#sft-controls').classList.contains('locked')"))
            ld = pg.evaluate(LEAD, '#sft-rail .presets')
            check(f'{tag} scenarios: G1 lead-in right above the scenarios', ld == Z('Pick a scenario and see what share of the counted tokens the answer gets.', '选一个场景，看看回答部分在算 loss 的 token 里占多少。'), ld)
            click(pg, '#sft-controls .seg-ctl[data-key="steps"] button[data-v="20"]'); pg.wait_for_timeout(100); s = st(pg)
            check(f'{tag} settings (G6): a new run starts, longer than 1.5 s at Medium', s['playing'] and s['spd'] == 'med' and s['n'] * 0.45 > 1.6, s)
            pg.wait_for_timeout(1600); s = st(pg)
            check(f'{tag} settings (G6): it shows its end within about 1.5 s', done(s), s)
            click(pg, '#sft-controls .seg-ctl[data-key="steps"] button[data-v="60"]'); pg.wait_for_timeout(300)
            for i in range(4):
                click(pg, f'#sft-rail [data-preset="{i}"]'); pg.wait_for_timeout(300)
                to_end(pg)
                pr = pg.evaluate("()=>[...document.querySelectorAll('#sft-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
                ins = text(pg, '#sft-insight')
                check(f'{tag} scenario {i + 1}: single choice, insight with the share and the small model', pr.count('true') == 1 and pr[i] == 'true' and '%' in ins and Z('small model', '小模型') in ins, (pr, ins[:120]))
                rules(pg, f'{tag} scenario {i + 1}', lang); shot(pg, f'{lang}-{w}-sb{i + 1}')
            hsh = pg.evaluate('()=>location.hash')
            pg2, e2 = page(b, w, h, lang, hsh.lstrip('#'))
            s2 = st(pg2)
            check(f'{tag} scenarios: a fresh page restores the setup from the URL', s2['mode'] == 'sandbox' and s2['pad'] == 'eot' and s2['count'] == 'resp' and s2['ex'] == 'toy', (hsh, s2))
            pg2.close()
            # ---------- the challenge
            mode(pg, 'challenge'); s = st(pg)
            check(f'{tag} challenge: starts from the colleague’s setup, Settings locked', (s['count'], s['eotl'], s['pad'], s['tmpl'], s['steps']) == ('all', 'yes', 'eot', 'same', 150)
                  and pg.evaluate("()=>document.querySelector('#sft-controls').classList.contains('locked')"), s)
            click(pg, '#sft-rail [data-act="submit"]'); pg.wait_for_timeout(250)
            check(f'{tag} challenge: as given, not yet (does not stop)', pg.evaluate("()=>!!document.querySelector('#sft-ch-res .badline')") and Z('does not stop', '不停') in text(pg, '#sft-ch-res'))
            rv = text(pg, '#sft-ch-res [data-act="reveal"]')
            check(f'{tag} challenge: the reveal button says “{Z("Show the answer", "直接看答案")}”', rv.strip() == Z('Show the answer', '直接看答案'), rv)
            click(pg, '#sft-ch-res [data-act="reveal"]'); pg.wait_for_timeout(250)
            c = pg.evaluate(CHART)
            check(f'{tag} challenge (as given, revealed): the chart holds only the passing combination and the initial setup, which is your pick',
                  (c['rows'], c['ok'], c['me'], c['stray']) == (2, 1, 1, 0) and Z('initial setup', '初始设置') in ' '.join(c['labels']), c)
            check(f'{tag} challenge (as given, revealed): the note counts the other 14 combinations', Z('The other 14 combinations', '其余 14 种组合') in c['note'], c['note'])
            check(f'{tag} challenge: the chart title no longer claims every combination', c['title'] and not any(x in c['title'] for x in ('every', 'each', '每种', '各')), c['title'])
            click(pg, '#sft-rail .seg-ctl[data-key="count"] button[data-v="resp"]'); pg.wait_for_timeout(250)
            check(f'{tag} challenge: one fix is not enough', pg.evaluate("()=>!!document.querySelector('#sft-ch-res .badline')"))
            c = pg.evaluate(CHART)
            check(f'{tag} challenge (one fix): the chart holds the passing combination, your pick and the initial setup; the note counts the other 13',
                  (c['rows'], c['ok'], c['me'], c['stray']) == (3, 1, 1, 0) and Z('The other 13 combinations', '其余 13 种组合') in c['note'], c)
            click(pg, '#sft-rail .seg-ctl[data-key="pad"] button[data-v="sep"]'); pg.wait_for_timeout(250)
            res = text(pg, '#sft-ch-res')
            check(f'{tag} challenge: the two fixes meet the goal with the fewest changes', pg.evaluate("()=>!!document.querySelector('#sft-ch-res .okline')") and '100%' in res and '2' in res, res[:200])
            c = pg.evaluate(CHART)
            check(f'{tag} challenge (met): of the 16 combinations the chart holds the one passing (your pick) and the initial setup; the note counts the other 14',
                  (c['rows'], c['ok'], c['me'], c['stray']) == (2, 1, 1, 0) and Z('The other 14 combinations', '其余 14 种组合') in c['note'], c)
            check(f'{tag} challenge: the capability card', pg.evaluate("()=>!!document.querySelector('#sft-rail [data-exits]')"))
            ex = pg.evaluate('()=>window.__sft.exits().text')
            check(f'{tag} explain it (G2): the answer states the pattern, not the article example’s number', '6.25' not in ex[0] and Z('the longer it is compared with the answer', '提问比回答长得越多') in ex[0], ex[0][-200:])
            check(f'{tag} new cases (G2): no earlier part named; the reference restated in full', not any(re.search(r'[Pp]art \d|第 \d 部分', t) for t in ex)
                  and all(Z('For example, with a 1,800-token prompt part and a 120-token response part', '比如提问部分有 1,800 个 token、回答部分有 120 个') in t for t in ex[2:4]), [t[:100] for t in ex[2:4]])
            rules(pg, f'{tag} challenge', lang); shot(pg, f'{lang}-{w}-ch')
            cols = pg.evaluate(COLORS); check(f'{tag}: colours', not cols, cols)
            check(f'{tag}: no page errors', not errs, errs[:2])
            pg.close()
    # G6 in a short window (1280 × 720, a laptop): part 5's run, below the batch picture, still plays where it can be seen
    if 'zh' in LANGS and (1440, 900) in SIZES:
        pg, errs = page(b, 1280, 720, 'zh')
        nxt(pg); nxt(pg); click(pg, '#sft-rail [data-pred="10to50"]'); pg.wait_for_timeout(250); nxt(pg)
        click(pg, '#sft-rail [data-nav="showme"]'); pg.wait_for_timeout(150); to_end(pg); nxt(pg)
        s = st(pg); gv0 = pg.evaluate(GENVIS)
        click(pg, '#sft-rail [data-nav="showme"]'); pg.wait_for_timeout(900); gv = pg.evaluate(GENVIS); s2 = st(pg)
        check('zh 1280×720 part 5 (G6): the run the reader started plays where it can be seen', s['step'] == 4 and s2['pad'] == 'eot' and gv['ok'] and gv['playing'], (gv0, gv))
        check('zh 1280×720: no page errors', not errs, errs[:2])
        pg.close()
    b.close()
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:40] if fails else 'none')
sys.exit(1 if fails else 0)
