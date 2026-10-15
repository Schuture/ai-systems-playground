"""Flow test for the eval-capstone lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH:
- no page errors, no horizontal overflow, no text wider than its box (R7.8), a tooltip for every help cursor (R1.1);
- the main picture first (R7.12): the forest plot ends within one screen of the stage's top (on the first screen at 1440), the metric
  cards under it and compact; on the phone the parts' bar with Next comes first, the chart right after it;
- part 1: the reader's task and its question in the first paragraphs (R6.35); Next disabled with the kit's why line until a
  prediction; only the total is shown, no category's difference, no true difference (R3.8);
- part 2: Next waits for “each category”; the five rows, writing −4.0 in red; the verdict on the prediction with the right answer;
- part 3: no result before a size is picked (R3.8); Next waits for fewer items (100; 300 is not offered here, so the challenge's 300
  for safety stays the reader's own inference); 100 items leave categories unsure; the −3 margin line; the card says why (4 times
  the items and the cost for half the width) and that the lab's measured difference is the truth; Show the math gives the
  real-evaluation chance (64%); 1,000 and 3,000 say how the interval narrows (58% at 3 times); other sizes keep the 1,000-item
  intervals as dashed boxes, and the cost card compares with 1,000 items each (P7, P8);
- part 4: the same-family premise before it is used; a guide line over the three ways; Next waits for the judge in both orders;
  the true differences appear (hollow diamonds); judged once, writing is read wrong, its lean drawn under its row (self-preference,
  then position and length); in both orders still wrong, position and length marked removed (P11); each tried way shows its full
  verdict name on its title row (R3.6, R3.17, R6.4); on a phone a pick brings the stage up under the top bar (P22); in part 5 too,
  where the strip over the chart points to the pick in the card (N3);
- part 5: the judge judged once; Next waits for 1,000 people's labels; 100 and 300 cannot tell; 1,000 catches the loss at $2,030
  and the card says why (disagreement), with no second word on the real-evaluation chance (P14); the judge's own result drawn
  hollow beside the corrected one (P15); the PPI formula typeset in Show the math;
- part 6: the online view; 1 and 3 days no alarm, 7 days the writing guardrail's alarm in red; the mini result bar at 390;
- part 7: three misconceptions, “See the scenarios”, the next-lab footer;
- R3.20: the one animation is the writing row's build-up in parts 4 and 5 (P11, P15): only the build-up's own keyframes, within
  1.1 s, once per new result; the total's row, the stage's badge and the verdict card appear with the writing row's verdict, on one
  clock (N10); a part change, a repeated click, a scenario or a submit redraws the chart, the badge and the cards at once;
- P22 and todo 6 (English layout): on a phone the stage fits under the top bar in every guided part, in English as in Chinese; at
  1440×900 the three answers of part 1 are on the first screen in both languages; the English verdict card says “No verdict yet” on one
  line;
- Scenarios: four, single choice, the insight for each; a lead-in on what to pick and watch, no “one at a time” note; the URL carries
  the setup and a fresh page restores it; a change under Settings does not move the page (R3.2);
- the challenge: the plan's controls in the card, Settings hidden with a note (R4.2); before Submit only cost and days, no verdict,
  no true difference (R3.8); three feedback tiers (R4.4: miss with a hint and “Show the answer”, met but a cheaper plan exists,
  met at the lowest cost among the settings the reader can pick, with the free-count note); the hint names the cause (disagreement)
  and gives no method; the real-evaluation chance after a pass; the brief chart (R4.6/R5.4): only the passing setting, the initial
  setup and the pick, tagged, and a note saying the 7 left out are combinations of judging and calibration count; “Show the answer” does not change the learner's plan; the language switch keeps the answer; the
  capability card after a pass;
- the writing rules on every visible text and tooltip (Chinese spacing and punctuation, two colons, no “toy”, no 控件 / Controls,
  no sentence about the page itself, no 吃 / 喂 / 扛 / 吞 / 撑, no “一次只能选 / 不会叠加 / 第 N 部分里”), no company names, and red
  only for a loss, an alarm or a wrong verdict, green only for the challenge's goal (R5.6);
- every formula typeset (H1), Show the math in blocks with hairlines and one “=” column (H2, R6.33).
Screenshots: shots/eval-capstone/.
Run: python3 tests/eval-capstone_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
(P / 'out' / 'eval-capstone').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/eval-capstone.html', '--only', 'eval-capstone'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'eval-capstone' / 'flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'eval-capstone.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'eval-capstone' / 'flow-wrapped.html')
SH = P / 'shots' / 'eval-capstone'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, hash_='eval-capstone'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__ecap.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
nvis = lambda pg, sel: pg.evaluate("s=>[...document.querySelectorAll(s)].filter(e=>{const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}).length", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
brief = lambda pg: pg.evaluate("""()=>{const res=document.getElementById('ecap-ch-res'), c=res&&res.querySelector(':scope .cmp:not(details .cmp)'); if(!c) return null;
  const rows=[...c.querySelectorAll('.cmp-row')], has=(r,k)=>r.classList.contains(k), init=r=>!!r.querySelector('.cmp-tag:not(.cmp-you)')&&!has(r,'me');
  return {rows:rows.length, ok:rows.filter(r=>has(r,'ok')).length, me:rows.filter(r=>has(r,'me')).length, init:rows.filter(init).length,
    stray:rows.filter(r=>!has(r,'ok')&&!has(r,'me')&&!init(r)).length, note:(res.querySelector('.cmp-note')||{}).textContent||'',
    title:(c.querySelector('.cmp-title')||{}).textContent||''};}""")
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#ecap-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#ecap-rail [data-next-why]')].filter(p=>!p.hidden&&getComputedStyle(p).visibility!=='hidden'&&p.getClientRects().length).map(p=>p.textContent)")
fprows = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#ecap-fp .ecap-row:not(.ecap-axis)')].map(r=>({cls:r.className, name:(r.querySelector('.ecap-rn b')||{}).textContent||'', vd:(r.querySelector('.ecap-vd')||{}).textContent||'', red:!!r.querySelector('.ecap-bar.red'), tru:!!r.querySelector('.ecap-tru')}))")


def settle(pg, ms=2000):
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(220); settle(pg)


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#ecap-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


# R3.20: the animations found two frames after a click (other than the kit's cue outline): its name, when it ends, its row; in the chart,
# and beside it on the stage's badge and the verdict card. Only parts 4 and 5 build the writing row up (P11, P15), once per new result,
# and what follows from it (the total's row, the badge, the verdict card) appears on the same clock (N10); anything else is a replay
STILL = """async (sel)=>{document.querySelector(sel).click(); await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
  const of=(e,nm)=>e?e.getAnimations({subtree:true}).filter(a=>a.animationName!=='cue').map(a=>{const t=a.effect&&a.effect.getComputedTiming?a.effect.getComputedTiming():{}, g=a.effect&&a.effect.target, r=g&&g.closest?g.closest('[data-row]'):null;
    return [a.animationName||a.transitionProperty||'?', Math.round(t.endTime||0), nm||(r?r.dataset.row:'')];}):[];
  return {g:of(document.querySelector('#ecap-graph')), side:of(document.querySelector('#ecap-status'),'badge').concat(of(document.querySelector('#ecap-metrics [data-k="dec"]'),'verdict card'))};}"""
BUILD = ('ecap-grow-r', 'ecap-grow-l', 'ecap-in')


def still(pg, sel, tag, what, build=False):
    r = pg.evaluate(STILL, sel); an, side = r['g'], r['side']
    if build:
        bad = [a for a in an if a[0] not in BUILD or a[1] > 1100 or a[2] not in ('write', 'total')]
        check(f'{tag} {what}: only the writing row and the total it moves build up, once, within 1.1 s (R3.20, P11, P15)', an and not bad and any(a[2] == 'write' for a in an), an)
        end = max([a[1] for a in an if a[2] == 'write'] or [0])
        tot = max([a[1] for a in an if a[2] == 'total'] or [0])
        check(f'{tag} {what}: the total, the badge and the verdict card appear with the writing row’s verdict, on one clock (N10)',
              abs(tot - end) <= 25 and len(side) >= 2 and all(a[0] == 'ecap-in' and abs(a[1] - end) <= 25 for a in side), (end, tot, side))
        pg.wait_for_timeout(1000)
    else:
        check(f'{tag} {what}: the chart is redrawn at once, nothing replays (R3.20)', not an and not side, (an, side))
    pg.wait_for_timeout(220); settle(pg)


# P22: on a phone the whole stage fits under the top bar, in Chinese and (todo 6) in English
FITS = """()=>{const st=document.getElementById('ecap-stage').getBoundingClientRect(), bar=document.querySelector('.topbar');
  return {h:Math.round(st.height), room:innerHeight-(bar?bar.offsetHeight:0), top:Math.round(st.top), bar:bar?bar.offsetHeight:0, bottom:Math.round(st.bottom), ih:innerHeight};}"""


def fits(pg, tag, what, lang, phone):
    if not phone: return
    f = pg.evaluate(FITS)
    check(f'{tag} {what}: the stage fits under the top bar on a phone (P22)', f['h'] <= f['room'], f)


FIRST = """()=>{const q=s=>document.querySelector(s).getBoundingClientRect(), st=q('#ecap-stage'), fp=q('#ecap-fp'), m=q('#ecap-metrics');
  return {stage:Math.round(st.top), fpEnd:Math.round(fp.bottom), metrics:Math.round(m.top), metricsH:Math.round(m.height), ih:innerHeight};}"""


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def all_text(pg):
    return pg.evaluate("""()=>{const out=[]; for(const s of ['#ecap-app','#dock']){const e=document.querySelector(s); if(e&&e.offsetParent!==null){const en=e.querySelector('[data-endnotes]'); out.push(en?e.innerText.replace(en.innerText,''):e.innerText);}}
      document.querySelectorAll('#ecap-app [data-tipx]:not([data-endnotes] *), #dock [data-tipx]').forEach(e=>out.push(e.getAttribute('data-tipx'))); return out.join('\\n')}""")


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · (?![0-9])/,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/\\btoy\\b|玩具/i,'“toy”'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万'],[/OpenAI|Anthropic|DeepMind|Google|Meta\\b|Microsoft|LMSYS|EleutherAI/,'company name'],[/[吃喂扛吞撑]/,'吃 / 喂 / 扛 / 吞 / 撑'],
   [/一次只能选|不会叠加|第 [0-9一二三四五六七八] 部分里/,'one-at-a-time note or a pointer to an earlier part'],
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour (R5.7)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, .fid-tag, a[href*="github.com"]')||!e.getClientRects().length) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); for(const [re,name] of bad) if(re.test(t)) out.push(name+' (tooltip): '+t.slice(0,90));
   if(!e.closest('.mathbody, details.math')&&/精确值是|Exactly/.test(t)) out.push('a hover that only expands a number to its exact value: '+t.slice(0,90));});
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
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx').replace(/<\\/?(?:p|b|i|em|strong|br|sub|sup|small|code|div|span)(?:\\s[^>]*)?>/g,'\\n'); const m=t.match(/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>][\\u4e00-\\u9fff]/g); if(m) out.add(m.join('|')+' in tooltip: '+t.slice(0,80));});
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
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|ecap-over|ecap-wr|ecap-vd|ecap-bar red|ecap-pt red|ecap-sw red|wrong|red|ex-res/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#ecap-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#ecap-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#ecap-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
# the forest plot's text stays inside its column: the name, the verdict, the axis labels (R7.8)
FPFIT = """()=>{const out=[]; document.querySelectorAll('#ecap-fp .ecap-row').forEach(r=>{const rr=r.getBoundingClientRect();
  r.querySelectorAll('.ecap-rn b, .ecap-rn span, .ecap-vd, .ecap-axis span').forEach(e=>{const b=e.getBoundingClientRect(); if(b.width&&(b.right>rr.right+1||b.left<rr.left-1)) out.push(e.textContent.slice(0,30));});});
  document.querySelectorAll('#ecap-fp .ecap-vd, #ecap-fp .ecap-rn').forEach(e=>{if(e.scrollWidth>e.clientWidth+1) out.push('wider than its column: '+e.textContent.slice(0,30));}); return out;}"""
rule_hits = []


def rules(pg, where, lang):
    for sel in ('#ecap-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')
    for h in pg.evaluate(OVER): rule_hits.append(f'{where}: text wider than its box: {h}')
    for h in pg.evaluate(FPFIT): rule_hits.append(f'{where}: forest plot text outside its row: {h}')
    for h in pg.evaluate(HELP): rule_hits.append(f'{where}: help cursor without a tooltip: {h}')
    o = overflow(pg)
    if o > 0: rule_hits.append(f'{where}: horizontal overflow {o}px')


def seg(key, v, root='#ecap-rail'):
    return f'{root} .seg-ctl[data-key="{key}"] button[data-v="{v}"]'


with sync_playwright() as pw:
    b = pw.chromium.launch()
    for lang in LANGS:
        for (w, h) in SIZES:
            phone = w <= 760; tag = f'{lang}-{w}'
            pg = page(b, w, h, lang)
            st = state(pg)
            check(f'{tag}: opens on part 1 of the guided tour', st['mode'] == 'guided' and st['step'] == 0, st)
            fv = pg.evaluate(FIRST)
            check(f'{tag}: the forest plot leads the stage, the metric cards below it and compact (R7.12)',
                  fv['metrics'] >= fv['fpEnd'] and fv['metricsH'] <= 150 and fv['fpEnd'] - fv['stage'] <= fv['ih'] and (phone or fv['fpEnd'] <= fv['ih']), fv)
            if phone:
                fs = pg.evaluate("()=>{const p=document.querySelector('#ecap-rail .pbar'), g=document.querySelector('#ecap-fp'); return {bar:!!p&&p.getBoundingClientRect().top<innerHeight, chart:!!g&&g.getBoundingClientRect().top<innerHeight*1.6}}")
                check(f'{tag}: the phone first screen has the parts bar, the chart follows it', fs['bar'] and fs['chart'], fs)
            # ---- part 1
            cur = text(pg, '#ecap-rail .step.current')
            core = Z(lang, 'You own the evaluation of an LLM assistant', '你负责一个 LLM 助手的评测')
            q1 = Z(lang, 'Can the candidate replace the current model?', '候选能替换当前模型吗')
            check(f'{tag} part 1: the reader’s task opens the card, and the question follows (R6.35)', core in cur and cur.find(core) < 40 and q1 in cur, cur[:300])
            if not phone:
                ab = pg.evaluate("()=>({y:scrollY, b:Math.max(...[...document.querySelectorAll('#ecap-rail [data-pred]')].map(e=>Math.round(e.getBoundingClientRect().bottom))), ih:innerHeight})")
                check(f'{tag} part 1: the three answers are on the first screen (todo 6)', ab['y'] == 0 and ab['b'] <= ab['ih'], ab)
            check(f'{tag} part 1: Next disabled until a prediction, with the kit’s why line', all(nexts(pg)) and any(Z(lang, 'Pick an answer', '选一个答案') in x for x in why(pg)), (nexts(pg), why(pg)))
            rows = fprows(pg); t1 = all_text(pg)
            check(f'{tag} part 1: only the total row, no category, no true difference (R3.8)', len(rows) == 1 and not any(r['tru'] for r in rows) and not vis(pg, '#ecap-fp .ecap-tru') and '−4.0' not in t1 and '-4.0' not in t1, rows)
            check(f'{tag} part 1: the total, +2.4 with +1.2 to +3.6, on the card and the chart', '2.4' in cur and '+1.2' in cur and '+3.6' in cur and '+1.2' in text(pg, '#ecap-fp') and '+3.6' in text(pg, '#ecap-fp'), text(pg, '#ecap-fp'))
            shot(pg, f'{tag}_p1', full=phone); rules(pg, f'{tag} part 1', lang); fits(pg, tag, 'part 1', lang, phone)
            click(pg, '#ecap-rail [data-pred="ship"]')
            check(f'{tag} part 1: after a pick, Next is enabled', not any(nexts(pg)))
            # ---- part 2
            still(pg, '#ecap-rail [data-nav="next"]', tag, 'part 2, Next')
            check(f'{tag} part 2: Next waits for “each category”, with its why line', all(nexts(pg)) and any(Z(lang, 'each category', '按类别分开看') in x for x in why(pg)), why(pg))
            still(pg, seg('split', '1'), tag, 'part 2, each category')
            still(pg, seg('split', '1'), tag, 'part 2, each category again (nothing changed)')
            rows = fprows(pg); ob = text(pg, '#ecap-rail [data-obs]')
            wr = [r for r in rows if r['name'] == Z(lang, 'Chinese writing', '中文写作')]
            check(f'{tag} part 2: the total and five categories; writing −4.0 in red, worse', len(rows) == 6 and wr and wr[0]['red'] and Z(lang, 'worse', '变差了') in wr[0]['vd'], rows)
            check(f'{tag} part 2: the verdict on the prediction names the right answer', '−4.0' in ob and Z(lang, 'You predicted', '你预测的是') in ob and Z(lang, 'The right answer is', '正确答案是') in ob and not any(nexts(pg)), ob[:300])
            shot(pg, f'{tag}_p2', full=phone); rules(pg, f'{tag} part 2', lang); fits(pg, tag, 'part 2', lang, phone)
            # ---- part 3
            still(pg, '#ecap-rail [data-nav="next"]', tag, 'part 3, Next')
            check(f'{tag} part 3: Next waits for fewer items; the −3 margin line is drawn', all(nexts(pg)) and any(Z(lang, 'fewer items', '更小的题数') in x for x in why(pg)) and nvis(pg, '#ecap-fp .ecap-mg') >= 5, why(pg))
            cur = text(pg, '#ecap-rail .step.current')
            check(f'{tag} part 3: the sizes offered are 100, 1,000 and 3,000', pg.evaluate("()=>[...document.querySelectorAll('#ecap-rail .seg-ctl[data-key=\"gsize\"] button')].map(b=>b.dataset.v)") == ['100', '1000', '3000'])
            check(f'{tag} part 3: no result before a size is picked (R3.8)', not vis(pg, '#ecap-rail [data-obs]'))
            check(f'{tag} part 3: the card says why: half the width takes 4 times the items and the cost (P5)', Z(lang, 'halving it takes 4 times the items, and 4 times what people cost', '区间要窄一半，题数和请人判的花费都得变成 4 倍') in cur, cur[:300])
            mt = pg.evaluate("()=>{const d=document.getElementById('ecap-math'); d.open=true; const t=document.getElementById('ecap-mathbody').textContent; d.open=false; return t}")
            check(f'{tag} part 3: the card says the measured difference is the truth here; Show the math gives the real-evaluation chance (S2)', Z(lang, 'In this lab the measured difference equals the true difference', '在这个 Lab 里，量出来的差就等于真实的差') in cur and '64%' in mt, (cur[-300:], mt[-300:]))
            still(pg, seg('gsize', '1000'), tag, 'part 3, 1,000 items (nothing changed)')
            ob = text(pg, '#ecap-rail [data-obs]')
            check(f'{tag} part 3: 1,000 items: safety not worse by over 3, and why a narrower interval would be needed', Z(lang, 'to be sure it got better too', '要确定它也变好了') in ob and not nvis(pg, '#ecap-fp .ecap-ghost'), ob[:300])
            still(pg, seg('gsize', '100'), tag, 'part 3, 100 items')
            ob = text(pg, '#ecap-rail [data-obs]')
            check(f'{tag} part 3: 100 items: some categories cannot tell yet; Next enabled', Z(lang, 'tell yet', '还看不出') in ob and not any(nexts(pg)), ob[:200])
            vd = pg.evaluate("()=>{const v=document.querySelector('#ecap-metrics [data-k=\"dec\"] .v'), c=v.parentElement.getBoundingClientRect(), g=document.createRange(); g.selectNodeContents(v); const t=g.getBoundingClientRect();"
                             " return {t:v.textContent, h:Math.round(v.getBoundingClientRect().height), lh:parseFloat(getComputedStyle(v).lineHeight), inside:t.left>=c.left&&t.right<=c.right}}")
            check(f'{tag} part 3: 100 items: the verdict card says “{vd["t"]}” within its card, in English on one line (todo 6; Chinese may break it between words on a phone, “还不能 / 下结论”)',
                  vd['t'] == Z(lang, 'No verdict yet', '还不能下结论') and (lang == 'zh' or vd['h'] < vd['lh'] * 1.5) and vd['inside'], vd)
            check(f'{tag} part 3: 100 items: each row keeps its 1,000-item interval as a dashed box, named in the legend (P7)', nvis(pg, '#ecap-fp .ecap-ghost') == 6 and Z(lang, 'dashed box', '虚线框') in text(pg, '#ecap-leg'), (nvis(pg, '#ecap-fp .ecap-ghost'), text(pg, '#ecap-leg')))
            still(pg, seg('gsize', '3000'), tag, 'part 3, 3,000 items')
            ob = text(pg, '#ecap-rail [data-obs]'); dl = text(pg, '#ecap-metrics')
            check(f'{tag} part 3: 3,000 items: safety really better; the interval narrows only to 58%', Z(lang, 'it really got better', '它确实变好了') in ob and '58%' in ob, ob[:300])
            check(f'{tag} part 3: 3,000 items: the cost card compares with 1,000 items each, as the card does (P8); on two lines when one is too short (N5; in English “vs” starts or ends the second line, todo 6)',
                  bool(re.search(r'▲ 3 times\s+(vs\s+)?(the 1,000-item plan|1,000 items)', dl.replace('\u00a0', ' '))) if lang == 'en' else bool(re.search(r'是每类 1,000 题时的 3 倍|▲ 3 倍\s*比 ?(每类 )?1,000 题时', dl.replace('\u00a0', ' '))), dl)
            shot(pg, f'{tag}_p3', full=phone); rules(pg, f'{tag} part 3', lang); fits(pg, tag, 'part 3', lang, phone)
            # ---- part 4
            still(pg, '#ecap-rail [data-nav="next"]', tag, 'part 4, Next')
            check(f'{tag} part 4: Next waits for the judge in both orders; the true differences appear', all(nexts(pg)) and any(Z(lang, 'both orders', '两种顺序') in x for x in why(pg)) and nvis(pg, '#ecap-fp .ecap-tru') >= 5, why(pg))
            cur = text(pg, '#ecap-rail .step.current')
            fam = Z(lang, 'This judge comes from the candidate’s family', '这个裁判和候选模型来自同一家族')
            check(f'{tag} part 4: the same-family premise and the candidate-first setting come before any result (S5, S6)', fam in cur and Z(lang, 'always shown first', '总排在前面') in cur and cur.find(fam) < cur.find(Z(lang, 'Pick a way to grade writing', '选一种判写作的办法')), cur[:200])
            check(f'{tag} part 4: a guide line over the three ways (S18)', vis(pg, '#ecap-rail .ecap-lvh'))
            still(pg, '#ecap-rail [data-lever="once"]', tag, 'part 4, judged once', build=True)
            if phone:
                f = pg.evaluate(FITS)
                check(f'{tag} part 4: on a phone a pick brings the stage up under the top bar (P22)', f['bar'] - 2 <= f['top'] <= f['bar'] + 14 and f['bottom'] <= f['ih'], f)
            ob = text(pg, '#ecap-rail [data-obs]'); rows = fprows(pg)
            wr = [r for r in rows if r['name'] == Z(lang, 'Chinese writing', '中文写作')]
            check(f'{tag} part 4: judged once, writing reads better and is marked wrong', Z(lang, 'The verdict is wrong', '结论错了') in ob and wr and 'wrong' in wr[0]['cls'] and Z(lang, 'wrong', '看错了') in wr[0]['vd'], (ob[:200], wr))
            an = text(pg, '#ecap-fp [data-row="write"] .ecap-ann')
            check(f'{tag} part 4: judged once, the lean is drawn under writing’s row: self-preference, then position and length (P11)', nvis(pg, '#ecap-fp [data-row="write"] .ecap-bs.s') == 1 and nvis(pg, '#ecap-fp [data-row="write"] .ecap-bs.p') == 1 and nvis(pg, '#ecap-fp .ecap-bs') == 2
                  and Z(lang, 'self-preference', '自我偏好') in an and Z(lang, 'position and length', '位置和长度') in an, an)
            still(pg, '#ecap-rail [data-lever="once"]', tag, 'part 4, judged once again (nothing changed)')
            still(pg, '#ecap-rail [data-lever="both"]', tag, 'part 4, both orders', build=True)
            ob = text(pg, '#ecap-rail [data-obs]'); an = text(pg, '#ecap-fp [data-row="write"] .ecap-ann')
            check(f'{tag} part 4: in both orders, still wrong: self-preference; Next enabled', Z(lang, 'Still wrong', '还是错的') in ob and not any(nexts(pg)), ob[:300])
            check(f'{tag} part 4: in both orders, position and length drawn as removed, self-preference still there (P11)', nvis(pg, '#ecap-fp [data-row="write"] .ecap-bs.x') == 1 and nvis(pg, '#ecap-fp [data-row="write"] .ecap-bs.s') == 1 and Z(lang, 'removed', '已去掉') in an and Z(lang, 'still there', '还在') in an, an)
            res = pg.evaluate("()=>[...document.querySelectorAll('#ecap-rail .lever .lv-r')].map(e=>e.textContent)")
            check(f'{tag} part 4: each tried way shows its full verdict name on its title row (R3.6, R6.4)', len(res) == 3 and all(res) and Z(lang, 'not worse by over 3', '没有变差超过 3 分') in res[2].replace('\u00a0', ' '), res)
            lv = pg.evaluate("()=>[...document.querySelectorAll('#ecap-rail .lever')].map(b=>{const t=b.querySelector('.lv-t b'), r=b.getBoundingClientRect(); return t.getBoundingClientRect().width/r.width})")
            check(f'{tag} part 4: no title squeezed by its result (R3.17)', all(x >= 0.45 for x in lv), lv)
            shot(pg, f'{tag}_p4', full=phone); rules(pg, f'{tag} part 4', lang); fits(pg, tag, 'part 4', lang, phone)
            # ---- part 5
            still(pg, '#ecap-rail [data-nav="next"]', tag, 'part 5, Next')
            check(f'{tag} part 5: Next waits for people’s labels; no formula in the card (P14)', all(nexts(pg)) and not vis(pg, '#ecap-rail .step.current .fx math'), why(pg))
            if phone:
                check(f'{tag} part 5: on a phone the strip over the chart points to the pick in the card (N3)', vis(pg, '#ecap-phone [data-act="to-cal"]') and Z(lang, 'Pick below', '在下面选') in text(pg, '#ecap-phone'), text(pg, '#ecap-phone'))
            still(pg, seg('gcal', '100'), tag, 'part 5, 100 labels', build=True)
            if phone:
                f = pg.evaluate(FITS)
                check(f'{tag} part 5: on a phone a pick in the card brings the stage up under the top bar (N3, P22)', f['bar'] - 2 <= f['top'] <= f['bar'] + 14 and f['bottom'] <= f['ih'], f)
            ob = text(pg, '#ecap-rail [data-obs]')
            check(f'{tag} part 5: 100 labels cannot tell yet; Next still waits', Z(lang, 'cannot tell yet', '还看不出') in ob and all(nexts(pg)), ob[:200])
            mt = pg.evaluate("()=>{const d=document.getElementById('ecap-math'); d.open=true; const b=document.getElementById('ecap-mathbody'), o={t:b.textContent, n:b.querySelectorAll('math').length}; d.open=false; return o}")
            check(f'{tag} part 5: Show the math gives the PPI correction, typeset (P14)', Z(lang, 'corrected', '修正后的差') in mt['t'] and mt['n'] >= 1, mt['t'][:200])
            still(pg, seg('gcal', '100'), tag, 'part 5, 100 labels again (nothing changed)')
            check(f'{tag} part 5: the judge judged once; 20% disagreement named (S9)', state(pg)['S']['ctl'] == 0 and '20%' in text(pg, '#ecap-rail .step.current'))
            still(pg, seg('gcal', '300'), tag, 'part 5, 300 labels', build=True)
            ob = text(pg, '#ecap-rail [data-obs]')
            check(f'{tag} part 5: 300 labels still cannot tell, because judge and people disagree; Next still waits', Z(lang, 'cannot tell yet', '还看不出') in ob and Z(lang, 'disagree on 20%', '20% 的题裁判和人意见不一致') in ob and all(nexts(pg)), ob[:300])
            still(pg, seg('gcal', '1000'), tag, 'part 5, 1,000 labels', build=True)
            ob = text(pg, '#ecap-rail [data-obs]'); rows = fprows(pg)
            check(f'{tag} part 5: 1,000 labels catch the loss at $2,030, no wrong verdict; why it costs that much; Next enabled', Z(lang, 'the loss is caught', '抓到了退步') in ob and '$2,030' in ob and Z(lang, 'disagrees with people too often', '意见不一致的题太多') in ob and '58%' not in ob and not any('wrong' in r['cls'] for r in rows) and not any(nexts(pg)), ob[:400])
            check(f'{tag} part 5: the judge’s own result drawn hollow, the bias taken off it toward the corrected result (P15)', nvis(pg, '#ecap-fp [data-row="write"] .ecap-rawp') == 1 and nvis(pg, '#ecap-fp [data-row="write"] .ecap-bs.cor') == 1 and nvis(pg, '#ecap-fp .ecap-rawp') == 1, text(pg, '#ecap-fp [data-row="write"]'))
            shot(pg, f'{tag}_p5', full=phone); rules(pg, f'{tag} part 5', lang); fits(pg, tag, 'part 5', lang, phone)
            # ---- part 6
            still(pg, '#ecap-rail [data-nav="next"]', tag, 'part 6, Next')
            check(f'{tag} part 6: the online view, Next waits for 7 days', Z(lang, 'After release', '上线以后') in text(pg, '#ecap-st-title') and all(nexts(pg)), text(pg, '#ecap-st-title'))
            still(pg, seg('abd', '3'), tag, 'part 6, 3 days')
            ob = text(pg, '#ecap-rail [data-obs]')
            check(f'{tag} part 6: 3 days: no alarm yet, no red', Z(lang, 'no alarm yet', '还没报警') in ob and not nvis(pg, '#ecap-fp .ecap-bar.red'), ob[:200])
            still(pg, seg('abd', '7'), tag, 'part 6, 7 days')
            ob = text(pg, '#ecap-rail [data-obs]')
            check(f'{tag} part 6: 7 days: the writing guardrail sounds the alarm, in red; Next enabled', Z(lang, 'sounds the alarm', '护栏报警') in ob and nvis(pg, '#ecap-fp .ecap-bar.red') == 1 and not any(nexts(pg)), ob[:200])
            if phone:
                pg.evaluate("()=>{const t=document.getElementById('ecap-fp'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+60)}"); pg.wait_for_timeout(700)
                dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, status:document.getElementById('dkStatus').textContent, read:document.getElementById('dkRead').textContent}}")
                check(f'{tag} part 6: the mini result bar with a short status and the writing guardrail', dk['shown'] and len(dk['status']) <= 14 and Z(lang, 'Writing conversations', '写作类对话') in dk['read'], dk)
            shot(pg, f'{tag}_p6', full=phone); rules(pg, f'{tag} part 6', lang); fits(pg, tag, 'part 6', lang, phone)
            # ---- part 7
            still(pg, '#ecap-rail [data-nav="back"]', tag, 'part 6, Back to part 5')
            still(pg, '#ecap-rail [data-nav="next"]', tag, 'part 5, Next to part 6 again')
            still(pg, '#ecap-rail [data-nav="next"]', tag, 'part 7, Next')
            cur = text(pg, '#ecap-rail .step.current')
            check(f'{tag} part 7: three misconceptions, See the scenarios, the next lab', cur.count('✗') == 3 and vis(pg, '#ecap-rail [data-mode-go="sandbox"]') and pg.evaluate("()=>!!document.querySelector('#ecap-rail .next-lab')"), cur[:200])
            shot(pg, f'{tag}_p7', full=phone); rules(pg, f'{tag} part 7', lang); fits(pg, tag, 'part 7', lang, phone)
            # ---- Scenarios
            mode(pg, 'sandbox')
            rt = text(pg, '#ecap-rail')
            lead = Z(lang, 'Pick a scenario and see which verdicts it gets right', '选一个场景，看看它哪些结论对')
            check(f'{tag} scenarios: the card leads in with what to pick and watch; no “one at a time” note', lead in rt and not any(x in rt for x in ('一次只能选', '不会叠加', 'one at a time', 'One at a time')), rt[:300])
            check(f'{tag} scenarios: four', pg.evaluate("()=>document.querySelectorAll('#ecap-rail [data-preset]').length") == 4)
            for k, sc in enumerate(['people', 'cheap', 'ctl', 'small']):
                still(pg, f'#ecap-rail [data-preset="{k}"]', tag, f'scenario {sc}')
                pr = pg.evaluate("()=>[...document.querySelectorAll('#ecap-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
                check(f'{tag} scenario {sc}: single choice, its insight shown', pr.count('true') == 1 and pr[k] == 'true' and vis(pg, '#ecap-insight') and len(text(pg, '#ecap-insight')) > 40, pr)
                shot(pg, f'{tag}_sb_{sc}', full=phone); rules(pg, f'{tag} scenario {sc}', lang)
            if not phone:
                pg.evaluate("()=>document.querySelector('#ecap-ctl-m-safe').scrollIntoView({block:'center'})"); pg.wait_for_timeout(300)
                y0 = pg.evaluate("()=>document.querySelector('#ecap-ctl-m-safe').getBoundingClientRect().top")
                pg.evaluate("()=>document.querySelector('#ecap-ctl-m-safe button[data-v=\"human\"]').click()"); pg.wait_for_timeout(300)
                y1 = pg.evaluate("()=>document.querySelector('#ecap-ctl-m-safe').getBoundingClientRect().top")
                check(f'{tag} scenarios: a change under Settings does not move the page (R3.2)', abs(y1 - y0) < 1.5 and state(pg)['S']['m']['safe'] == 'human', (y0, y1))
            hs = pg.evaluate('()=>location.hash')
            check(f'{tag} scenarios: the URL carries the setup', hs.startswith('#eval-capstone~'), hs)
            pg2 = page(b, w, h, lang, hash_=hs[1:])
            s2 = state(pg2)
            check(f'{tag} scenarios: a fresh page restores the setup', s2['mode'] == 'sandbox' and s2['S'] == state(pg)['S'], (s2['S'], state(pg)['S']))
            pg2.close()
            # ---- the challenge
            mode(pg, 'challenge')
            st = state(pg); C = pg.evaluate("()=>window.__ecap.consts")
            check(f'{tag} challenge: starts from the initial setup', st['S'] == C['CH_START'], st['S'])
            check(f'{tag} challenge: Settings hidden with a note; the plan’s 12 choices in the card (R4.2)', not vis(pg, '#ecap-ctl-wrap') and vis(pg, '#ecap-lock') and pg.evaluate("()=>document.querySelectorAll('#ecap-rail .seg-ctl[data-key]').length") == 12, (vis(pg, '#ecap-ctl-wrap'), vis(pg, '#ecap-lock'), pg.evaluate("()=>document.querySelectorAll('#ecap-rail .seg-ctl[data-key]').length")))
            check(f'{tag} challenge: before Submit only cost and days: no verdict, no true difference (R3.8)', not vis(pg, '#ecap-fp .ecap-tru') and not vis(pg, '#ecap-fp .ecap-vd') and '$6' in text(pg, '#ecap-ch-cost') and not any(x in text(pg, '#ecap-metrics') + text(pg, '#ecap-status') for x in (Z(lang, 'worse', '变差'), Z(lang, 'better', '变好'), Z(lang, 'wrong', '看错'))), (text(pg, '#ecap-status'), text(pg, '#ecap-metrics')))
            shot(pg, f'{tag}_ch0', full=phone); rules(pg, f'{tag} challenge before Submit', lang)
            still(pg, '#ecap-rail [data-act="submit"]', tag, 'challenge, Submit')
            r = text(pg, '#ecap-ch-res')
            check(f'{tag} challenge: the hint names the cause (disagreement 20% against 7%) and gives no method (S9, S11)', '20%' in r and '7%' in r and not any(x in r for x in ('fit the budget', '放得进预算', 'A judge in both orders, checked', '两种顺序都判、再用几百题')), r[:400])
            check(f'{tag} challenge: not met, writing read wrong, with a hint and “Show the answer” (R4.4)', vis(pg, '#ecap-ch-res .badline') and Z(lang, 'Hint', '提示') in r and Z(lang, 'got a wrong verdict', '的结论错了') in r
                  and text(pg, '#ecap-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案') and vis(pg, '#ecap-fp .ecap-tru'), r[:240])
            # R12: the plan Codex describes (300 items each; the judge in both orders, length-corrected; people check 300): writing cannot tell,
            # and the hint is about writing's items, with what that costs, not advice the plan already follows (checked before the answer is
            # shown: from then on the answer stands in for the hints)
            for sel in [seg('ctl', '1'), seg('cal', '300')]:
                click(pg, sel)
            r = text(pg, '#ecap-ch-res')
            check(f'{tag} challenge: writing cannot tell at 300 judged and 300 checked: the cause, the one change with its cost, no already-met advice (R12)',
                  vis(pg, '#ecap-ch-res .badline') and Z(lang, 'none of the judge’s grades is used', '裁判的分数一个也没用上') in r and Z(lang, 'Raise Chinese writing from 300 to 3,000 items', '把中文写作的题数从 300 题加到 3,000 题') in r
                  and '$54' in r and Z(lang, 'that meets the goal', '这样就达到目标了') in r and not any(x in r for x in Z(lang, ('without people to check it', 'Judged once, the judge', 'People alone on 1,000'), ('不拿人校准', '只判一次的裁判', '全请人判 1,000 题'))), r[:600])
            s_before = state(pg)['S']
            click(pg, '#ecap-ch-res [data-act="reveal"]')
            bf = brief(pg); r = text(pg, '#ecap-ch-res')
            check(f'{tag} challenge: Show the answer gives the answer and the brief chart (1 passing setting, initial setup, pick; 7 left out), and keeps the plan',
                  bf and bf['rows'] == 3 and bf['ok'] == 1 and bf['me'] == 1 and bf['init'] == 1 and bf['stray'] == 0 and Z(lang, 'The other 7 combinations of judging and calibration count', '另外 7 种判法和校准题数的组合') in bf['note'] and Z(lang, 'cannot meet the goal', '都不达标') in bf['note']
                  and '$1,260' in r and Z(lang, 'among the settings you can pick', '在能选的设置里') in r and state(pg)['S'] == s_before, (bf, r[:300]))
            check(f'{tag} challenge: once the answer is shown, no hint repeats it', vis(pg, '#ecap-ch-res .badline') and Z(lang, 'Hint', '提示') not in r, r[:400])
            # a plan that meets the goal but not at the lowest cost: the judge with people's labels on safety too ($1,266)
            for sel in [seg('m-write', 'judge'), seg('n-write', '3000'), seg('ctl', '1'), seg('cal', '300'), seg('n-safe', '300')]:
                click(pg, sel)
            r = text(pg, '#ecap-ch-res')
            check(f'{tag} challenge: met at $1,266, but a cheaper plan exists, said in the first line', vis(pg, '#ecap-ch-res .okline') and '$1,266' in r and Z(lang, 'But among the settings you can pick, a cheaper plan also meets it', '不过在能选的设置里，有更便宜的计划也能达标') in text(pg, '#ecap-ch-res > p:first-child'), r[:300])
            bf = brief(pg)
            check(f'{tag} challenge: the brief chart holds the passing setting (the pick shares it) and the initial setup', bf and bf['ok'] >= 1 and bf['me'] == 1 and bf['init'] == 1 and bf['stray'] == 0, bf)
            click(pg, seg('m-safe', 'human'))
            r = text(pg, '#ecap-ch-res')
            check(f'{tag} challenge: met at the lowest cost among the settings one can pick, $1,260 in 2 days; any count would be cheaper (S10)', vis(pg, '#ecap-ch-res .okline') and Z(lang, 'Goal met, at the lowest cost among the settings you can pick', '达成目标，而且是这道题能选的设置里最便宜的') in r and '$1,260' in r and '130' in r and vis(pg, '#ecap-status.ok'), r[:300])
            check(f'{tag} challenge: after a pass, the real-evaluation chance (58%) is said (S2)', Z(lang, 'In a real evaluation the measured difference wanders', '真实评测里，量出来的差会晃动') in r and '58%' in r, r[-500:])
            check(f'{tag} challenge: the capability card follows', vis(pg, '#ecap-rail [data-exits]'))
            shot(pg, f'{tag}_ch', full=True); rules(pg, f'{tag} challenge', lang)
            other = 'en' if lang == 'zh' else 'zh'
            sv = state(pg)['S']
            pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
            st = state(pg)
            check(f'{tag} challenge: the language switch keeps the answer', st['mode'] == 'challenge' and st['S'] == sv and st['chSubmitted'] and vis(pg, '#ecap-ch-res .okline'), st)
            pg.close()
    b.close()

# H1 / H2: every formula typeset as MathML, Show the math in blocks with a hairline between them, each chain's “=” in one column
_PFX = 'ecap'
_PLAIN = ['√', 'sqrt(', '/ √', 'h = 1.96']
_PROBE = """([pfx,plain])=>{const app=document.querySelector('#view-dyn:not([hidden])')||document; const mb=document.getElementById(pfx+'-mathbody');
  const out={nmath:0,words:[],plain:[],blocks:0,sep:true,clip:[],page:[]};
  const TEXW=/^(frac|sqrt|left|right|text|hat|cdot|times|sum|quad|ldots|approx|mathrm|big|Big)$/;
  const scanText=(root,arr)=>{const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n; while((n=tw.nextNode())){const e=n.parentElement;
    if(!e||e.closest('math,code,pre,svg,script,style,#tip')||!e.getClientRects().length) continue; for(const p of plain) if(n.textContent.includes(p)) arr.push(p+' in “'+n.textContent.trim().slice(0,70)+'”');}};
  if(mb){const d=mb.closest('details'); if(d) d.open=true;
    out.nmath=mb.querySelectorAll('math').length;
    scanText(mb,out.plain);
    const bl=[...mb.querySelectorAll('.'+pfx+'-mb')]; out.blocks=bl.length;
    bl.slice(1).forEach(b=>{const cs=getComputedStyle(b); if(!(parseFloat(cs.borderTopWidth)>=1&&cs.borderTopStyle==='solid')) out.sep=false;});
    mb.querySelectorAll('.'+pfx+'-eqs').forEach(q=>{if(q.scrollHeight>q.clientHeight+1) out.clip.push(q.textContent.slice(0,40));
      const k=[...q.children], cols=getComputedStyle(q).gridTemplateColumns.split(' ').length; if(cols<3) return; for(let i=1;i<k.length;i+=3){const a=k[i].getBoundingClientRect(), b=k[i+3]&&k[i+3].getBoundingClientRect(); if(b&&Math.abs(a.left-b.left)>1) out.clip.push('“=” not in one column: '+q.textContent.slice(0,40));}});}
  scanText(app,out.page);
  app.querySelectorAll('mi').forEach(x=>{if(TEXW.test(x.textContent.trim())) out.words.push(x.textContent);});
  return out;}"""


def _states(b, w, h, lang):
    pg = page(b, w, h, lang)
    got = []
    for k in range(7):
        pg.evaluate("k=>{const e=document.querySelector('#ecap-rail [data-go=\"'+k+'\"]'); if(e) e.click();}", k); pg.wait_for_timeout(400)
        got.append((f'part {k + 1}', pg.evaluate(_PROBE, [_PFX, _PLAIN])))
    for m in ('sandbox', 'challenge'):
        mode(pg, m)
        got.append((m, pg.evaluate(_PROBE, [_PFX, _PLAIN])))
    click(pg, '#ecap-rail [data-act="submit"]')
    got.append(('challenge submitted', pg.evaluate(_PROBE, [_PFX, _PLAIN])))
    dr = pg.evaluate("()=>{const b=document.querySelector('#view-dyn [data-act=\"drill-reveal\"]'); if(!b) return null; b.click(); return true}"); pg.wait_for_timeout(300)
    drill = pg.evaluate("()=>{const a=document.getElementById('ecap-drill-a'); if(!a||a.hidden) return null; return {n:a.querySelectorAll('math').length}}") if dr else None
    pg.close()
    return got, drill


with sync_playwright() as _p:
    _b = _p.chromium.launch()
    for (_w, _h, _lang) in [(1440, 900, 'en'), (390, 844, 'zh')]:
        if str(_w) not in str(SIZES) or _lang not in LANGS: continue
        _got, _drill = _states(_b, _w, _h, _lang)
        check(f'H1 {_w} {_lang}: Show the math has typeset formulas in every state but the unsubmitted challenge (R3.8)', all(o['nmath'] for s, o in _got if s != 'challenge') and not dict(_got)['challenge']['nmath'], [(s, o['nmath']) for s, o in _got])
        for s, o in _got:
            check(f'H1 {_w} {_lang} {s}: no TeX command shown as a word', not o['words'], o['words'][:5])
            check(f'H1 {_w} {_lang} {s}: no plain-text formula in Show the math or on the page', not o['plain'] and not o['page'], (o['plain'][:3], o['page'][:3]))
            check(f'H2 {_w} {_lang} {s}: a hairline between the blocks of Show the math ({o["blocks"]} blocks)', o['sep'] and (o['blocks'] >= 1 or s == 'challenge'), o)
            check(f'H2/R6.33 {_w} {_lang} {s}: no formula box clipped, every “=” of a chain in one column', not o['clip'], o['clip'][:3])
        if _drill is not None:
            check(f'H1 {_w} {_lang}: the drill answer typesets its working', _drill['n'] >= 1, _drill)
    _b.close()

src = (P / 'labs' / 'eval-capstone.js').read_text() + (P / 'teasers' / 'eval-capstone.js').read_text()
for x in ('一次只能选', '不会叠加', '一次选一个', 'do not stack', 'one at a time'):
    check(f'no “{x}” in the lab or its tile', x not in src)
check('no 吃 / 喂 / 扛 / 吞 / 撑 in the lab or its tile', not re.search('[吃喂扛吞撑]', src), re.findall('.{8}[吃喂扛吞撑].{4}', src)[:3])
check('no “第 N 部分里” in the lab or its tile', not re.search('第 [0-9一二三四五六七八] 部分里', src))
check('no <prefix>-fid / -pri / -share ids', not re.search(r'ecap-(fid|pri|share)\b', src))
check('no page errors', not errs, errs[:5])
for x in rule_hits[:40]: print('RULE', x)
check('writing, colour and layout rules', not rule_hits, len(rule_hits))
print(f'{n - len(fails)}/{n} checks passed')
sys.exit(1 if fails else 0)
