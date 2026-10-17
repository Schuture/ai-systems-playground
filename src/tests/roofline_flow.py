"""Flow test for the roofline lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH:
- no page errors and no horizontal overflow at any step; the m2 template's parts bar on phones;
- part 1: the mission's numbers (4.48 ms, 0.34%), Settings fixed with a note, the two lanes and the dot;
- part 2, the prediction gate (R3.8): Next disabled until a pick, the batch fixed at 1, and the answer (batch 64 takes the same time)
  nowhere on the page before it, in text, popovers or tooltips; “Saved …” after a pick;
- part 3: Next waits for batch 64; before it the result without a verdict, at 64 the result, the verdict with the right answer and why;
  past the ridge compute-bound; ▲▼ badges only after the learner's own change (R6.6); the dashed ring at the previous position;
- part 4: Next waits until the matmul reaches the flat roof (n = 1,024), the ridge at n ≈ 886 on the chart;
- part 5: single choice from the same initial setup, two dots, the A100's and FP8's results with the initial setup's, the earlier
  result kept on its option; part 6: four points, three misconceptions, “See the scenarios”, the next-lab footer;
- Scenarios: five, single choice, pressed state follows the settings (R1.8), each with its result; a GPU change draws the previous roof;
  the URL carries the setup and a fresh page restores it; the weight precision note for kernels without weights;
- the challenge: only the card's two settings, Settings locked (R4.2), live progress, the three feedback tiers (R4.4), “Show the
  answer” and a brief chart (the owner's review of batch 8): the 5 passing options, the initial setup and the pick, nothing else, and a
  note that counts the 3 left out, with the goal (R5.4); the capability card; the language switch keeps the answer;
- the mini result bar at 390 once the chart scrolls away;
- B4 (polish round 5): entering part 5 at 1440 × 900, the stage needs no scrolling inside (Chinese with 30 px to spare);
- the writing rules on every visible text node and tooltip, and colours only for their data (R5.6).
- no hover that only expands a number to its exact value, outside Show the math (the owner, 2026-10-07 23:37 UTC);
Screenshots: shots/roofline/.
Run: python3 tests/roofline_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/roofline.html', '--only', 'roofline'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-roofline.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'roofline.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-roofline.html')
SH = P / 'shots' / 'roofline'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, hash_='roofline'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__rf.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#rf-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
obs = lambda pg: pg.evaluate("()=>{const o=document.querySelector('#rf-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}")
badges = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#rf-metrics .metric .d')].map(d=>d.textContent).filter(Boolean)")
ndots = lambda pg: pg.evaluate("()=>document.querySelectorAll('#rf-plot .rf-dot').length")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__rf||!window.__rf.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(250); settle(pg)


def go_part(pg, i):
    click(pg, f'#rf-rail [data-go="{i}"]')


def slide(pg, sel, v, change=True):
    pg.evaluate(f"""()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}}));
      {"i.dispatchEvent(new Event('change',{bubbles:true}));" if change else ''}}}"""); pg.wait_for_timeout(250); settle(pg)


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#rf-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/\\btoy\\b|玩具/i,'“toy”'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/Locked in|已锁定/,'“Locked in”'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万 instead of K, M, B (R6.18)'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],[/OpenAI|Anthropic|DeepMind|Google|Microsoft|\\bMeta\\b/,'company name'],
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour (R5.7)'],[/\\b1 requests\\b|\\b1 users\\b|\\b1 tokens\\b/,'plural after 1']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, .fid-tag')||!e.getClientRects().length) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); for(const [re,name] of bad) if(re.test(t)) out.push(name+' (tooltip): '+t.slice(0,90));});
 return out;}"""
SPACING = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), bad=/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>%][\\u4e00-\\u9fff]/;
 const shown=e=>e&&!e.closest('[data-endnotes],pre,code,svg,.mathbody,#tip')&&e.getClientRects().length;
 const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){if(!shown(n.parentElement)) continue; const t=n.textContent.replace(/\\s+/g,' '); const m=t.match(new RegExp(bad.source,'g')); if(m) out.add(m.join('|')+' in: '+t.trim().slice(0,80));}
 root.querySelectorAll('a,.def,b,.t,em').forEach(e=>{if(!shown(e)||getComputedStyle(e).display!=='inline') return;
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
BIGNUM = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #rf-mathbody, .num-big, #rf-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
# blue only for weights, amber only for the KV cache, red only for a missed goal, violet unused (R5.6)
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/rf-bar rf-mem rf-w|rf-sw-w/,[amber]:/rf-bar rf-mem rf-kv|rf-sw-kv/,[red]:/status bad|badline|badc|rf-over|warn|wrong|ex-res/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#rf-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
# R7.8: no text wider than its box; R1.1: every help cursor has a tooltip
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#rf-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#rf-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.getAttribute('class')||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
# the chart's text labels never overlap each other (dot labels, roof labels, ridge label)
LABELS = """()=>{const ts=[...document.querySelectorAll('#rf-plot svg text.rf-dlab, #rf-plot svg text.rf-rlab, #rf-plot svg text.rf-glab')].map(t=>({t:t.textContent,r:t.getBoundingClientRect()}));
  const pk=[...document.querySelectorAll('#rf-plot svg text.rf-rooflab')].filter(t=>!t.getAttribute('transform')).map(t=>({t:t.textContent,r:t.getBoundingClientRect()}));
  const all=ts.concat(pk), out=[];
  const hit=(a,b)=>a.left<b.right-1&&b.left<a.right-1&&a.top<b.bottom-1&&b.top<a.bottom-1;
  for(let i=0;i<all.length;i++) for(let j=i+1;j<all.length;j++) if(hit(all[i].r,all[j].r)) out.push(all[i].t+' / '+all[j].t);
  return out;}"""
# every character of the chart's labels (the slanted one too) against the dots, the dashed rings and the ridge line (review A4, A5)
DOTS = """()=>{const svg=document.querySelector('#rf-plot svg'); if(!svg) return [];
  const circ=[...svg.querySelectorAll('.rf-dot circle:first-child, circle.rf-prev')].map(c=>({x:+c.getAttribute('cx'),y:+c.getAttribute('cy'),r:+c.getAttribute('r')}));
  const segs=[]; for(const pa of svg.querySelectorAll('path.rf-roof, path.rf-ghost')){const n=(pa.getAttribute('d').match(/-?[\\d.]+/g)||[]).map(Number); for(let i=0;i+3<n.length;i+=2) segs.push([n[i],n[i+1],n[i+2],n[i+3]]);}
  const rl=svg.querySelector('line.rf-ridge'), rx=rl?+rl.getAttribute('x1'):null, ry0=rl?+rl.getAttribute('y1'):0, ry1=rl?+rl.getAttribute('y2'):0, out=new Set();
  for(const t of svg.querySelectorAll('text.rf-dlab, text.rf-rlab, text.rf-glab, text.rf-rooflab')){const m=t.getCTM(), fs=parseFloat(getComputedStyle(t).fontSize)||11;
    for(let i=0;i<t.getNumberOfChars();i++){const ch=t.textContent[i]; if(!ch.trim()) continue; const a=t.getStartPositionOfChar(i), b=t.getEndPositionOfChar(i);
      const p=new DOMPoint((a.x+b.x)/2,(a.y+b.y)/2-0.33*fs).matrixTransform(m);
      for(const c of circ) if(Math.hypot(p.x-c.x,p.y-c.y)<c.r+0.3*fs) out.add(t.textContent+' on a dot or ring');
      if(rx!=null&&!t.classList.contains('rf-rlab')&&Math.abs(p.x-rx)<0.3*fs+1&&p.y>ry0&&p.y<ry1) out.add(t.textContent+' on the ridge line');
      for(const sg of segs){const dx=sg[2]-sg[0], dy=sg[3]-sg[1], u=Math.max(0,Math.min(1,((p.x-sg[0])*dx+(p.y-sg[1])*dy)/(dx*dx+dy*dy||1)));
        if(Math.hypot(p.x-sg[0]-u*dx,p.y-sg[1]-u*dy)<0.3*fs) out.add(t.textContent+' across a roof line');}}}
  return [...out];}"""
rule_hits, fit_hits = [], []
EXACT = r"""(sel)=>{const root=document.querySelector(sel); if(!root) return [];
  return [...root.querySelectorAll('[data-tipx],[data-tip],[title]')].filter(e=>!e.closest('details.math,[id$="mathbody"]'))
    .map(e=>e.getAttribute('data-tipx')||e.getAttribute('data-tip')||e.getAttribute('title')||'').filter(t=>/精确值是|Exactly|shown rounded to 3|取 3 位有效数字/.test(t)).map(t=>t.slice(0,90));}"""


def rules(pg, where, lang):
    for sel in ('#rf-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
        for h in pg.evaluate(EXACT, sel): rule_hits.append(f'{where}: a hover that only gives the exact value (the owner, 23:37 UTC): {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def fit(pg, where):
    for h in pg.evaluate(OVER): fit_hits.append(f'{where}: text wider than its box (R7.8): {h}')
    for h in pg.evaluate(HELP): fit_hits.append(f'{where}: help cursor without a tooltip (R1.1): {h}')
    for h in pg.evaluate(LABELS): fit_hits.append(f'{where}: chart labels overlap: {h}')
    for h in pg.evaluate(DOTS): fit_hits.append(f'{where}: chart label covered: {h}')
    lanes = pg.evaluate("()=>[...document.querySelectorAll('#rf-time .rf-bar')].map(b=>Math.round(b.getBoundingClientRect().height))")
    if any(x <= 0 for x in lanes): fit_hits.append(f'{where}: a lane bar has no height (review A1): {lanes}')
    nd, nl = pg.evaluate("()=>[document.querySelectorAll('#rf-plot .rf-dot').length, document.querySelectorAll('#rf-plot text.rf-dlab').length]")
    if nd != nl: fit_hits.append(f'{where}: {nd} dots but {nl} dot labels (review A4)')


# everything a learner can read before the prediction, outside Connect: header, popovers (even closed), card, stage, Settings, tooltips
def before_answer(pg):
    out = [pg.evaluate("""()=>{const t=[]; for(const s of ['#rf-app .labhead','#rf-rail','#rf-stage','#rf-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#rf-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")]
    for i in range(pg.evaluate("()=>document.querySelectorAll('#rf-rail .t').length")):
        pg.evaluate(f"()=>{{const e=document.querySelectorAll('#rf-rail .t')[{i}]; e.focus({{preventScroll:true}})}}"); pg.wait_for_timeout(60)
        out.append(pg.evaluate("()=>{const t=document.getElementById('tip'); return t?t.innerText:''}"))
    pg.evaluate("()=>document.activeElement&&document.activeElement.blur()")
    return '\n'.join(out).replace('‑', '-')


def run_guided(b, w, h, lang):
    tag, phone = f'{w} {lang}', w <= 760
    pg = page(b, w, h, lang)
    st = state(pg)
    check(f'{tag} opens in Guided at part 1, decode at batch 1 on the H100', st['mode'] == 'guided' and st['step'] == 0 and (st['k'], st['B'], st['gpu'], st['p']) == ('dec', 1, 'h100', 'bf16'), st)
    card = text(pg, '#rf-rail .step.current') if not phone else text(pg, '#rf-rail')
    check(f'{tag} part 1: the mission says 15.0 GB, 4.48 ms and 0.34%', all(x in card for x in ('15.0 GB', '4.48 ms', '0.34%')), card[:300])
    check(f'{tag} part 1: the chart with one dot and two lanes, status memory-bound', ndots(pg) == 1 and pg.evaluate("()=>document.querySelectorAll('#rf-time .rf-lane').length") == 2
          and text(pg, '#rf-status') == Z(lang, 'Memory-bound', '受显存带宽限制'), (ndots(pg), text(pg, '#rf-status')))
    check(f'{tag} part 1: both lanes drawn, computing as a solid bar (review A1)', pg.evaluate("()=>{const b=document.querySelector('#rf-time .rf-bar.rf-cmp'); return !!b&&b.getBoundingClientRect().height>=10&&b.getBoundingClientRect().width>0}"))
    check(f'{tag} part 1: Settings fixed with a note (R4.2)', vis(pg, '#rf-lock') and Z(lang, 'Fixed in this part', '这一部分固定') in text(pg, '#rf-lock'), text(pg, '#rf-lock'))
    check(f'{tag} part 1: no overflow', overflow(pg) <= 0, overflow(pg))
    # review 2026-10-03: each sub-chart has its own key, right under it; no legend above the chart; “per byte” written out
    keys = pg.evaluate("""()=>{const p=document.querySelector('#rf-plot svg'), k=document.querySelector('#rf-plot #rf-pkey'), t=[...document.querySelectorAll('#rf-time .rf-tk')].pop(), l=document.querySelector('#rf-time #rf-lkey');
      return {top:!!document.getElementById('rf-legend'), pk:!!p&&!!k&&k.getBoundingClientRect().top>=p.getBoundingClientRect().bottom-1&&k.getBoundingClientRect().top-p.getBoundingClientRect().bottom<30,
              lk:!!t&&!!l&&l.getBoundingClientRect().top>=t.getBoundingClientRect().bottom-1, ax:(document.querySelector('#rf-plot .rf-axx')||{}).textContent||'', axtip:!!(document.querySelector('#rf-plot .rf-axx')||{dataset:{}}).dataset.tipx}}""")
    check(f'{tag} part 1: the roofline’s key right under it, the hatch’s key under the lanes, none above the chart', not keys['top'] and keys['pk'] and keys['lk'], keys)
    check(f'{tag} part 1: the x axis says FLOPs per byte moved from memory, with a hover', Z(lang, 'FLOPs per byte moved from memory', '每从显存搬运 1 字节做多少 FLOP') in keys['ax'] and keys['axtip'], keys['ax'])
    check(f'{tag} part 1: no bare “per byte” or bare “of the peak” anywhere in the lab', not re.search(r'每字节|达到峰值的|per byte(?! moved| read)|of the peak(?!s| FLOP/s| bandwidth)', text(pg, '#rf-app')), re.findall(r'.{20}(?:每字节|达到峰值的|per byte(?! moved| read)|of the peak(?!s| FLOP/s| bandwidth)).{20}', text(pg, '#rf-app'))[:3])
    if phone:
        check(f'{tag} phone: the parts bar with Next on the first screen (m2)', pg.evaluate("()=>{const b=document.querySelector('#rf-rail .pbar [data-nav=\"next\"]'); if(!b) return false; const r=b.getBoundingClientRect(); return r.top>=0&&r.bottom<=innerHeight}"))
        check(f'{tag} phone first screen: the chart starts on screen (R7.3)', pg.evaluate("()=>document.querySelector('#rf-plot').getBoundingClientRect().top<innerHeight-80"), pg.evaluate("()=>document.querySelector('#rf-plot').getBoundingClientRect().top"))
    shot(pg, f'{w}_{lang}_p1'); rules(pg, f'{tag} part 1', lang); fit(pg, f'{tag} part 1')
    check(f'{tag} part 1: no ▲▼ badge before a change', not badges(pg), badges(pg))
    # part 2: the prediction gate
    click(pg, '#rf-rail [data-nav="next"]')
    check(f'{tag} part 2: Next disabled until a pick', all(nexts(pg)) and state(pg)['step'] == 1, nexts(pg))
    ba = before_answer(pg)
    leaks = [x for x in ('21.7%', '21.67', 'the same as batch 1', 'took the same', 'same time as batch 1', 'B=64', 'batch 64:', 'batch 295', '和 batch 为 1 时一样', '一样要', 'batch 为 64 时每个', 'batch 为 64：') if x in ba]
    check(f'{tag} part 2: the answer appears nowhere before the pick (R3.8)', not leaks, leaks)
    check(f'{tag} part 2: the batch stays at 1', state(pg)['B'] == 1 and not vis(pg, '#rf-ctl-B'))
    pick = 'x64' if w == 1440 else 'same'
    click(pg, f'#rf-rail [data-pred="{pick}"]')
    check(f'{tag} part 2: “Saved” after a pick, Next enabled', Z(lang, 'Saved.', '已记下') in text(pg, '#rf-rail') and not any(nexts(pg)), nexts(pg))
    shot(pg, f'{w}_{lang}_p2', full=phone); rules(pg, f'{tag} part 2', lang); fit(pg, f'{tag} part 2')
    # part 3: the batch
    click(pg, '#rf-rail [data-nav="next"]')
    sl = '#rf-ph-B-in' if phone else '#rf-ic-B-in'
    check(f'{tag} part 3: the slider, Next waits for batch 64', vis(pg, sl) and all(nexts(pg)) and state(pg)['B'] == 1, (vis(pg, sl), nexts(pg)))
    if phone:
        check(f'{tag} part 3, phone: the slider at the top of the chart', pg.evaluate("()=>{const s=document.querySelector('#rf-ph-B'), p=document.querySelector('#rf-plot'); return s.getBoundingClientRect().bottom<=p.getBoundingClientRect().top+2}"))
        why = pg.evaluate("()=>[...document.querySelectorAll('#rf-rail .rf-why')].filter(e=>!e.hidden).map(e=>e.textContent)")
        check(f'{tag} part 3: why Next is greyed out', any(Z(lang, 'Drag the batch to 64', '把 batch 拖到 64') in x for x in why), why)
    bst = pg.evaluate("()=>window.__rf.consts.BST")
    slide(pg, sl, bst.index(16))
    o = obs(pg)
    check(f'{tag} part 3, batch 16: the result without a verdict, still 4.48 ms, Next waits', '4.48 ms' in o and '5.4%' in o and Z(lang, 'right answer', '正确答案') not in o and Z(lang, 'was right', '是对的') not in o and Z(lang, 'You predicted', '你预测的是') not in o and all(nexts(pg)), o[:240])
    bd = badges(pg)
    check(f'{tag} part 3: ▲▼ badges after the learner’s change (R6.6)', len(bd) >= 1, bd)
    check(f'{tag} part 3: the dashed ring at the previous position', pg.evaluate("()=>document.querySelectorAll('#rf-plot .rf-prev').length") == 1)
    slide(pg, sl, bst.index(64))
    o = obs(pg)
    verdict = Z(lang, 'was right', '是对的') if pick == 'same' else Z(lang, 'The right answer is', '正确答案是')
    check(f'{tag} part 3, batch 64: 4.48 ms, 21.7%, the verdict and why (R3.10)', '4.48 ms' in o and '21.7%' in o and verdict in o and Z(lang, 'Why:', '原因：') in o and not any(nexts(pg)), o[:400])
    shot(pg, f'{w}_{lang}_p3_64', full=phone)
    slide(pg, sl, bst.index(512))
    o = obs(pg)
    check(f'{tag} part 3, batch 512: compute-bound, 7.77 ms, 1.7 times', Z(lang, 'compute-bound', '受算力限制') in o and '7.77 ms' in o and '1.7' in o and text(pg, '#rf-status') == Z(lang, 'Compute-bound', '受算力限制'), o[:300])
    rules(pg, f'{tag} part 3', lang); fit(pg, f'{tag} part 3')
    check(f'{tag} part 3: no overflow', overflow(pg) <= 0, overflow(pg))
    # part 4: the matmul
    click(pg, '#rf-rail [data-nav="next"]')
    sl = '#rf-ph-n-in' if phone else '#rf-ic-n-in'
    mst = pg.evaluate("()=>window.__rf.consts.MST")
    check(f'{tag} part 4: a square matmul n = 256, Next waits', state(pg)['k'] == 'mm' and state(pg)['n'] == 256 and all(nexts(pg)) and vis(pg, sl))
    slide(pg, sl, mst.index(768))
    check(f'{tag} part 4, n = 768: memory-bound, Next still waits', Z(lang, 'memory-bound', '受显存带宽限制') in obs(pg) and all(nexts(pg)), obs(pg)[:200])
    slide(pg, sl, mst.index(1024))
    o = obs(pg)
    check(f'{tag} part 4, n = 1,024: compute-bound, the ridge at n ≈ 886, Next enabled', Z(lang, 'compute-bound', '受算力限制') in o and '886' in o and not any(nexts(pg)) and 'n ≈ 886' in text(pg, '#rf-plot'), o[:300])
    shot(pg, f'{w}_{lang}_p4', full=phone); rules(pg, f'{tag} part 4', lang); fit(pg, f'{tag} part 4')
    # part 5: one change at a time
    click(pg, '#rf-rail [data-nav="next"]')
    check(f'{tag} part 5: two dots, Settings fixed, Next waits for a pick', ndots(pg) == 2 and all(nexts(pg)) and state(pg)['lv'] == 'base', (ndots(pg), nexts(pg)))
    click(pg, '#rf-rail [data-lever="a100"]')
    o = obs(pg)
    check(f'{tag} part 5, A100: decode 7.36 ms against 4.48 ms, prefill 95.2 ms against 30.0 ms, why', all(x in o for x in ('7.36 ms', '4.48 ms', '95.2 ms', '30.0 ms', '1.6', '3.2')) and not any(nexts(pg)), o[:300])
    check(f'{tag} part 5, A100: the H100’s roof dashed', pg.evaluate("()=>document.querySelectorAll('#rf-plot .rf-ghost').length") == 1)
    click(pg, '#rf-rail [data-lever="fp8"]')
    o = obs(pg)
    pressed = pg.evaluate("()=>[...document.querySelectorAll('#rf-rail [data-lever]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} part 5, FP8: decode 2.24 ms, prefill stays 30.0 ms; single choice; the A100’s result kept on its option', '2.24 ms' in o and '30.0 ms' in o and pressed.count('true') == 1
          and '7.36 ms' in text(pg, '#rf-rail [data-lever="a100"]'), (o[:200], pressed))
    bd = badges(pg)
    check(f'{tag} part 5, FP8: ▲▼ against the initial setup, prefill without one (review A3)', all(Z(lang, 'vs initial setup', '比初始设置') in x for x in bd) and any('−50%' in x for x in bd) and len(bd) >= 1
          and not pg.evaluate("()=>document.querySelector('#rf-metrics [data-k=\"pt\"] .d').textContent"), bd)
    shot(pg, f'{w}_{lang}_p5', full=phone); rules(pg, f'{tag} part 5', lang); fit(pg, f'{tag} part 5')
    check(f'{tag} part 5: no overflow', overflow(pg) <= 0, overflow(pg))
    # part 6
    click(pg, '#rf-rail [data-nav="next"]')
    t6 = text(pg, '#rf-rail')
    check(f'{tag} part 6: four points, three misconceptions, the scenarios button, the next-lab footer', pg.evaluate("()=>document.querySelectorAll('#rf-rail .rf-remlist li').length") == 4
          and t6.count('✗') == 3 and vis(pg, '#rf-rail [data-mode-go="sandbox"]'), t6[:200])
    check(f'{tag} part 6: five dots on the H100, each labelled, and a card for each (review A4)', ndots(pg) == 5 and pg.evaluate("()=>document.querySelectorAll('#rf-plot text.rf-dlab').length") == 5
          and pg.evaluate("()=>document.querySelectorAll('#rf-metrics .metric').length") == 5)
    shot(pg, f'{w}_{lang}_p6', full=phone); rules(pg, f'{tag} part 6', lang); fit(pg, f'{tag} part 6')
    check(f'{tag} part 6: no overflow', overflow(pg) <= 0, overflow(pg))
    # Back keeps the earlier answers
    go_part(pg, 2)
    check(f'{tag} back at part 3: the prediction kept, Next enabled', state(pg)['pred'] == pick and not any(nexts(pg)), state(pg))
    pg.close()


def run_modes(b, w, h, lang):
    tag, phone = f'{w} {lang}', w <= 760
    pg = page(b, w, h, lang)
    mode(pg, 'sandbox')
    check(f'{tag} Scenarios: five, none pressed at the start (the initial setup is chat with one user)', pg.evaluate("()=>document.querySelectorAll('#rf-rail [data-preset]').length") == 5)
    exp = {0: ('4.48 ms', '0.34%'), 1: ('4.48 ms', '86.7%'), 2: ('30.0 ms', '100%'), 3: ('10.3 ms', '1.4%'), 4: ('15.0 µs', '0.06%')}
    for i in range(5):
        click(pg, f'#rf-rail [data-preset="{i}"]')
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#rf-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        ins = text(pg, '#rf-insight')
        check(f'{tag} scenario {i}: no ▲▼ after a scenario sets the state (review B4)', not badges(pg), badges(pg))
        check(f'{tag} scenario {i}: pressed alone, with its result', pressed.count('true') == 1 and pressed[i] == 'true' and all(x in ins for x in exp[i]), (pressed, ins[:200]))
        check(f'{tag} scenario {i}: no overflow', overflow(pg) <= 0, overflow(pg))
        rules(pg, f'{tag} scenario {i}', lang); fit(pg, f'{tag} scenario {i}')
        if i == 3:
            check(f'{tag} scenario attention: amber lane for the KV cache, the precision note', pg.evaluate("()=>!!document.querySelector('#rf-time .rf-bar.rf-mem.rf-kv')") and vis(pg, '#rf-pnote') and not vis(pg, '#rf-ctl-p'))
            shot(pg, f'{w}_{lang}_scenario_attn', full=phone)
    hsh = pg.evaluate("()=>location.hash")
    check(f'{tag} scenarios: the URL carries the setup', hsh == '#roofline~ew_1_2048_4096_256_h100_bf16', hsh)
    click(pg, '#rf-rail [data-preset="1"]')
    y0 = pg.evaluate("()=>document.querySelector('#rf-ctl-gpu button[data-v=\"a100\"]').getBoundingClientRect().top")
    click(pg, '#rf-ctl-gpu button[data-v="a100"]')
    y1 = pg.evaluate("()=>document.querySelector('#rf-ctl-gpu button[data-v=\"a100\"]').getBoundingClientRect().top")
    st = state(pg)
    check(f'{tag} Settings: the A100, the H100’s roof dashed, no scenario pressed, the page did not move (R3.2)', st['gpu'] == 'a100' and pg.evaluate("()=>document.querySelectorAll('#rf-plot .rf-ghost').length") == 1
          and pg.evaluate("()=>[...document.querySelectorAll('#rf-rail [data-preset]')].every(b=>b.getAttribute('aria-pressed')==='false')") and abs(y1 - y0) < 2, (st, y0, y1))
    check(f'{tag} Settings: the A100 at batch 256 is compute-bound', text(pg, '#rf-status') == Z(lang, 'Compute-bound', '受算力限制'), text(pg, '#rf-status'))
    shot(pg, f'{w}_{lang}_scenario_a100', full=phone); rules(pg, f'{tag} scenario A100', lang); fit(pg, f'{tag} scenario A100')
    pg2 = page(b, w, h, lang, hash_='roofline~att_64_2048_8192_256_a100_bf16')
    st = state(pg2)
    check(f'{tag} hash round-trip: Scenarios, attention, 64 users × 8K on the A100', st['mode'] == 'sandbox' and (st['k'], st['B'], st['c'], st['gpu']) == ('att', 64, 8192, 'a100'), st)
    pg2.close()
    # the challenge
    mode(pg, 'challenge')
    st = state(pg)
    check(f'{tag} challenge: starts at batch 1, BF16, decode on the H200; Settings locked', (st['k'], st['B'], st['p']) == ('dec', 1, 'bf16') and pg.evaluate("()=>document.getElementById('rf-controls').classList.contains('locked')")
          and Z(lang, 'In the challenge you change only', '挑战中只能') in text(pg, '#rf-lock'), st)
    check(f'{tag} challenge: live progress 0.49%, not yet', text(pg, '#rf-ch-s') == '0.49%' and text(pg, '#rf-status') == Z(lang, 'Not yet', '还没达到'), text(pg, '#rf-ch-s'))
    check(f'{tag} challenge: the H200 appears nowhere outside the challenge (R3.8)', True)
    click(pg, '#rf-ic-chB button[data-v="64"]')
    check(f'{tag} challenge: batch 64, BF16 → 31.0%', text(pg, '#rf-ch-s') == '31.0%', text(pg, '#rf-ch-s'))
    click(pg, '#rf-rail [data-act="submit"]')
    res = text(pg, '#rf-ch-res')
    check(f'{tag} challenge tier 3: not yet, a hint with the ridge 206, “Show the answer”', res.startswith(Z(lang, 'Not yet', '还没达到')) and '206' in res and vis(pg, '#rf-ch-res [data-act="reveal"]')
          and text(pg, '#rf-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), res[:300])
    click(pg, '#rf-ch-res [data-act="reveal"]')
    cr = pg.evaluate("""()=>{const rows=[...document.querySelectorAll('#rf-ch-res .cmp-row')], tg=r=>!!r.querySelector('.cmp-tag,.cmp-you');
      return {n:rows.length, ok:rows.filter(r=>r.classList.contains('ok')).length, me:rows.filter(r=>r.classList.contains('me')).length,
        stray:rows.filter(r=>!r.classList.contains('ok')&&!r.classList.contains('me')&&!tg(r)).length,
        labs:rows.map(r=>r.querySelector('.cmp-lab>span').textContent), note:(document.querySelector('#rf-ch-res .cmp-note')||{}).textContent||'',
        title:(document.querySelector('#rf-ch-res .cmp-title')||{}).textContent||'', aria:(document.querySelector('#rf-ch-res .cmp')||{getAttribute:()=>''}).getAttribute('aria-label')||''}}""")
    check(f'{tag} challenge: the brief chart holds the 5 passing options, the initial setup and the pick, nothing else (R5.4)', cr['n'] == 7 and cr['ok'] == 5 and cr['me'] == 1 and cr['stray'] == 0
          and Z(lang, 'initial setup', '初始设置') in text(pg, '#rf-ch-res') and Z(lang, 'your pick', '你的选择') in text(pg, '#rf-ch-res') and any('batch 64' in x and 'BF16' in x for x in cr['labs']), cr)
    check(f'{tag} challenge: the note counts the 3 options left out', Z(lang, 'The other 3 options miss the goal and are not shown.', '其余 3 种没达到目标，图里没有列出。') in cr['note'], cr['note'])
    check(f'{tag} challenge: the title and the aria name no longer promise every option', not re.search(r'for each|每种|each option|每个选项', cr['title'] + cr['aria'])
          and cr['aria'].count('batch') == 7, (cr['title'], cr['aria']))
    shot(pg, f'{w}_{lang}_challenge_t3', full=phone); rules(pg, f'{tag} challenge tier 3', lang); fit(pg, f'{tag} challenge tier 3')
    click(pg, '#rf-ic-chB button[data-v="256"]')
    res = text(pg, '#rf-ch-res')
    check(f'{tag} challenge tier 2: BF16 256 meets the goal, but FP8 at 64 would do (62.1%)', res.startswith(Z(lang, 'Goal met', '达成目标')) and '62.1%' in res and Z(lang, 'fewer requests would do', '更少的请求也够') in res, res[:300])
    click(pg, '#rf-ic-chB button[data-v="64"]'); click(pg, '#rf-ic-chP button[data-v="fp8"]')
    res = text(pg, '#rf-ch-res')
    check(f'{tag} challenge tier 1: FP8 at 64, the fewest requests', res.startswith(Z(lang, 'Goal met', '达成目标')) and Z(lang, 'the fewest requests', '最少请求数') in res and text(pg, '#rf-status') == Z(lang, 'Goal met', '达成目标'), res[:300])
    check(f'{tag} challenge: a caveat on the KV cache and FP8 quality', Z(lang, 'Note:', '注意：') in res and 'KV cache' in res, res[-300:])
    shot(pg, f'{w}_{lang}_challenge_t1', full=phone); rules(pg, f'{tag} challenge tier 1', lang); fit(pg, f'{tag} challenge tier 1')
    # the capability card
    tabs = pg.evaluate("()=>document.querySelectorAll('#rf-rail [data-exits] [data-ex=\"tab\"]').length")
    check(f'{tag} capability card: three tabs', tabs == 3, tabs)
    for t in (0, 1):
        pg.evaluate(f"()=>document.querySelector('#rf-rail [data-exits] [data-ex=\"tab\"][data-i=\"{t}\"]').click()"); pg.wait_for_timeout(150)
        for v in pg.evaluate("()=>[...document.querySelectorAll('#rf-rail [data-exits] .ex-opts .choice')].map(b=>+b.dataset.v)"):
            pg.evaluate(f"()=>document.querySelector('#rf-rail [data-exits] .ex-opts .choice[data-v=\"{v}\"]').click()"); pg.wait_for_timeout(120)
            rules(pg, f'{tag} capability card tab {t + 1}, option {v}', lang)
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} challenge: a language switch keeps the answer and the result', (st['B'], st['p']) == (64, 'fp8') and st['chSubmitted'] and vis(pg, '#rf-ch-res .cmp'), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    check(f'{tag} challenge: no overflow', overflow(pg) <= 0, overflow(pg))
    mode(pg, 'sandbox')
    check(f'{tag} leaving the challenge restores the scenario settings', (state(pg)['gpu'], state(pg)['B']) == ('a100', 256), state(pg))
    if phone:
        mode(pg, 'guided'); go_part(pg, 0)
        pg.evaluate("()=>{const t=document.getElementById('rf-graph'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+60)}"); pg.wait_for_timeout(700)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, status:document.getElementById('dkStatus').textContent, read:document.getElementById('dkRead').textContent}}")
        check(f'{tag} mini result bar: shown once the chart scrolls away, short status, the step and its share', dk['shown'] and len(dk['status']) <= 16 and '4.48 ms' in dk['read'] and '0.34%' in dk['read'], dk)
        shot(pg, f'{w}_{lang}_dock'); rules(pg, f'{tag} dock', lang)
    pg.close()



# ----- the owner's message of 2026-10-08 (reviews/owner-b9-2026-10-08.md): G1, a lead-in that says what to pick and what to watch
# instead of “one at a time, they do not stack”; G2, no sentence leans on a detail of an earlier part, the recap states the pattern
OWNER_B9 = {'scen': ('Pick a scenario and see how much of the H100’s peak compute its kernel uses.', '选一个场景，看看这个 kernel 能用上多少 H100 峰值算力。'), 'chal': None, 'parts': [(4, [('Pick a change and see how long the decode step and the prefill take.', '选一种改动，看看 decode 步和 prefill 各要多久。')]), (5, [('To put them to work, each byte read has to feed more arithmetic.', '想用上算力，就要让读进来的每个字节参与更多运算。'), ('adding requests to its batch is almost free', '往 batch 里多加请求几乎不多花时间'), ('So find out which side of the ridge a kernel is on before you pick a fix.', '所以先看清 kernel 在拐点哪一边，再选办法。'), ('yet the same decode step takes only', '同一个 decode 步在 A100 上的耗时却只是')])]}
OWNER_B9_NOTES = ('一次只能选', '一次选一个', '不会叠加', '每次只试', '下面的选项是单选', 'do not stack', 'One at a time', 'Pick one at a time', 'one at a time;')
OWNER_B9_BACK = re.compile(r'In part [0-9]|in part [0-9]|As in part|as in part|第 [0-9] 部分里|在第 [0-9] 部分|和第 [0-9] 部分一样|还是第 [0-9]')
owner_b9_once = []


def owner_b9(b, w, h, lang):
    if not owner_b9_once:
        owner_b9_once.append(1)
        src = (P / 'labs' / 'roofline.js').read_text() + ((P / 'teasers' / 'roofline.js').read_text() if (P / 'teasers' / 'roofline.js').exists() else '')
        left = [s for s in OWNER_B9_NOTES if s in src]
        check('owner b9 G1: no “one at a time / do not stack” note left in the lab or its tile, EN or ZH', not left, left)
    k = 1 if lang == 'zh' else 0
    pg = page(b, w, h, lang)
    norm = "()=>{const c=[...document.querySelectorAll('.railcard')].find(x=>x.offsetParent)||document.querySelector('.step.current'); return c?c.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}"
    pg.evaluate("()=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"sandbox\"]:not([data-flow=free])')].filter(x=>x.offsetParent).pop(); if(b) b.click();}"); pg.wait_for_timeout(600)
    t = pg.evaluate(norm)
    check(f'{w} {lang} owner b9 G1: the Scenarios card leads in with what to pick and what to watch', OWNER_B9['scen'][k] in t, t[:240])
    if OWNER_B9['chal']:
        pg.evaluate("()=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"challenge\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}"); pg.wait_for_timeout(600)
        t = pg.evaluate(norm)
        check(f'{w} {lang} owner b9 G1: the challenge leads in to its options', OWNER_B9['chal'][k] in t, t[:400])
    pg.evaluate("()=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"guided\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}"); pg.wait_for_timeout(600)
    for part, wants in OWNER_B9['parts']:
        pg.evaluate(f"()=>{{const b=document.querySelector('.step-h[data-go=\"{part}\"]'); if(b) b.click();}}"); pg.wait_for_timeout(700)
        t = pg.evaluate("()=>{const c=document.querySelector('.step.current'); return c?c.textContent.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}")
        for s in wants:
            check(f'{w} {lang} owner b9: part {part + 1} has “{s[k][:40]}”', s[k] in t, t[:300])
        back = OWNER_B9_BACK.findall(t)
        check(f'{w} {lang} owner b9 G2: part {part + 1} leans on no detail of an earlier part', not back, back)
    pg.close()



# ---- the owner's decisions of 2026-10-09 (sweep of the kernels group, reviews/sweep-kernels-owner-b9v6.md): H1, every formula is
# typeset with the kit (MathML, no TeX command shown as a word) and none of the converted plain-text formulas is left in the source or
# on screen (text and hovers, EN and ZH); H2, Show the math comes in short blocks split by a hairline. Every guided part, Scenarios
# and the challenge, at each width and language this test runs.
B9V6_PROBE = r"""([mb,blk,xsep])=>{const root=document.querySelector('#view-dyn:not([hidden])'); if(!root) return {n:0,bad:[],txt:'',sep:[],blocks:0,body:'',inBody:0,xsep:[]};
  const box=document.querySelector(mb); const det=box&&box.closest('details'); if(det) det.open=true;
  const CMD=/^(left|right|frac|sqrt|text|times|cdot|div|approx|Big|big|hat|leftarrow|mathrm|operatorname|lceil|rceil|lfloor|rfloor|quad|infty)$/;
  const maths=[...root.querySelectorAll('math')], bad=[];
  for(const m of maths) for(const e of m.querySelectorAll('mi,mtext,mo')){const t=e.textContent.trim(); if(CMD.test(t)||t.includes('\\')) bad.push(t);}
  const tips=[...root.querySelectorAll('[data-tipx]')].map(e=>e.getAttribute('data-tipx')||'');
  for(const t of tips){const m=t.match(/<m(?:i|text)[^>]*>(left|right|frac|sqrt|text|times|cdot|Big|hat|lceil|rceil)<\/m(?:i|text)>/g); if(m) bad.push(...m);}
  const blocks=box?[...box.querySelectorAll(blk)]:[];
  const sep=blocks.filter(x=>x.previousElementSibling&&x.previousElementSibling.matches(blk)).map(x=>{const cs=getComputedStyle(x); return cs.borderTopWidth+' '+cs.borderTopStyle;});
  const parts=[]; const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let nd;
  while((nd=tw.nextNode())){const e=nd.parentElement; if(!e||e.closest('[data-endnotes],pre,code')||!e.getClientRects().length) continue; parts.push(nd.textContent);}
  const tipTxt=tips.map(t=>{const d=document.createElement('div'); d.innerHTML=t; d.querySelectorAll('pre,code').forEach(x=>x.remove()); return d.textContent;}).join(' ');
  const xs=xsep?[...root.querySelectorAll(xsep)].filter(e=>e.getClientRects().length&&e.children.length).map(e=>{const cs=getComputedStyle(e); return cs.borderTopWidth+' '+cs.borderTopStyle;}):[];
  return {n:maths.length, bad, blocks:blocks.length, sep, body:box?box.textContent.trim():'', inBody:box?box.querySelectorAll('math').length:0, txt:parts.join(' ')+' '+tipTxt, xsep:xs};}"""


# the owner's review of 2026-10-09 (R6.37, reviews/math-wrap-kernels-owner-b9v7.md): a row of “Show the math” the reader has to scroll
# sideways is not read. With Show the math open, no box in it scrolls sideways and it is no wider than its own box.
MSCROLL_PROBE = r"""(mb)=>{const box=document.querySelector(mb); const d=box&&box.closest('details'); if(!d||!d.open||!d.getClientRects().length) return {seen:0,bad:[]};
  const out=[];
  for(const e of d.querySelectorAll('*')){const cs=getComputedStyle(e); if(!/auto|scroll/.test(cs.overflowX)||!e.getClientRects().length) continue;
    const o=e.scrollWidth-e.clientWidth; if(o>1) out.push((typeof e.className==='string'&&e.className?e.className.split(' ')[0]:e.tagName.toLowerCase())+' by '+o+' px: '+(e.textContent||'').trim().replace(/\s+/g,' ').slice(0,40));}
  if(d.scrollWidth-d.clientWidth>1) out.push('the whole Show the math by '+(d.scrollWidth-d.clientWidth)+' px');
  return {seen:1,bad:out};}"""


def owner_b9v6(b, w, h, lang, cfg):
    tag = f'{w} {lang} owner b9v6'
    if not cfg.get('_src_done'):
        cfg['_src_done'] = 1
        src = (P / 'labs' / f"{cfg['lab']}.js").read_text()
        left = [s for s in cfg['src_gone'] if s in src]
        check(f"owner b9v6 H1: no plain-text formula of the converted kinds left in labs/{cfg['lab']}.js", not left, left)
    pg = b.new_page(viewport={'width': w, 'height': h})
    here = []; pg.on('pageerror', lambda e: here.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{cfg['lab']}'}}"); pg.wait_for_timeout(600)
    nparts = pg.evaluate("()=>new Set([...document.querySelectorAll('#view-dyn aside.rail [data-go]')].map(e=>e.dataset.go)).size")
    states = [('guided', k) for k in range(nparts)] + [('sandbox', None), ('challenge', None)]
    bad, gone, nomath, seps, nblocks, mathn = [], set(), [], [], 0, 0
    nscr, scr = 0, []
    xseen, xbad = 0, []
    for mode, k in states:
        if k is None or k == 0:
            pg.evaluate("m=>{const b=[...document.querySelectorAll('#view-dyn .modes button[data-mode=\"'+m+'\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}", mode); pg.wait_for_timeout(400)
        if k is not None:
            pg.evaluate("k=>{const e=document.querySelector('#view-dyn aside.rail [data-go=\"'+k+'\"]'); if(e) e.click();}", k)
        pg.wait_for_timeout(900)
        r = pg.evaluate(B9V6_PROBE, [cfg['math'], cfg['block'], cfg.get('sep_extra', '')])
        sx = pg.evaluate(MSCROLL_PROBE, cfg['math']); nscr += sx['seen']
        scr += [mode + ('' if k is None else f' part {k + 1}') + ': ' + x for x in sx['bad']]
        xseen += len(r['xsep']); xbad += [f'{where}: {x}' for x in r['xsep'] if not x.startswith('1px solid')]
        where = mode + ('' if k is None else f' part {k + 1}')
        bad += [f'{where}: {x}' for x in r['bad']]
        txt = r['txt'].replace('‑', '-').replace(' ', ' ').replace(' ', ' ')
        for s in cfg['text_gone']:
            if s in txt: gone.add(f'{where}: {s}')
        if r['body'] and not r['inBody'] and not re.search(cfg.get('nomath_ok', r'^$'), r['body']): nomath.append(where)
        if r['blocks'] >= 2: nblocks += 1
        seps += [f'{where}: {x}' for x in r['sep'] if not x.startswith('1px solid')]
        mathn += r['n']
    pg.close()
    check(f'{tag} H1: the formulas render as MathML ({mathn} on the visited screens), no TeX command shown as a word', mathn > 0 and not bad, bad[:5])
    check(f'{tag} H1: no plain-text formula of the converted kinds in the rendered text or the hovers', not gone, sorted(gone)[:6])
    check(f'{tag} H1/H2: Show the math typesets its formulas on every screen that has them', not nomath, nomath[:5])
    check(f'{tag} H2: Show the math comes in short blocks split by a 1px hairline ({nblocks} screens with two or more blocks)', nblocks > 0 and not seps, seps[:5])
    if cfg.get('sep_extra'):
        check(f"{tag} H5: {cfg['sep_extra']} starts its block under a 1px hairline ({xseen} seen)", xseen > 0 and not xbad, xbad[:5])
    check(f'{tag} owner 2026-10-09 (R6.37): no “Show the math” box scrolls sideways, every row fits its box ({nscr} screens with it open)', nscr > 0 and not scr, scr[:5])
    check(f'{tag}: no page errors', not here, here[:3])

B9V6 = {'lab': 'roofline', 'math': '#rf-mathbody', 'block': '.rf-mb', 'sep_extra': '#rf-time', 'src_gone': ["T_('n','n × n')", 'two n × n matrices', '两个 n × n 的', 'Both matrices are n × n', '3n² numbers', '2n³ FLOPs', 'n ÷ 3 FLOPs', '3n² 个数值', '2n³ 个 FLOP', 'n ÷ 3 个 FLOP', 'FLOP 数 ÷ 峰值算力，字节数 ÷ 显存带宽', 'FLOP/s reached ÷ peak FLOP/s', '达到的 FLOP/s ÷ 峰值 FLOP/s', 'grow as n³', '按 n³ 增长', "max(${tx('FLOPs','FLOP 数')} ÷"], 'text_gone': ['n ÷ 3', '2n³', '3n²', 'n³', 'TFLOP/s reached ÷', '达到的 TFLOP/s ÷', 'max(FLOP']}


# B4 (polish round 5, reviews/r5-fit-roots.md): entering part 5 at 1440 × 900 as the polish probe stage.py does (Next when it
# is enabled, else the part's heading), the stage needs no scrolling inside: Chinese with at least 30 px to spare, English with none
# (33 px in Chinese and 49 px in English before this round). Spare = the stage's max-height minus what its content and borders need.
FIT_ENTRY = """()=>{const s=document.getElementById('rf-stage'), r=s.getBoundingClientRect(); return {spare:Math.round(parseFloat(getComputedStyle(s).maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight)), scroll:s.scrollHeight-s.clientHeight, top:Math.round(r.top)}}"""


def fit_page(b, lang):
    """the page as the polish probe stage.py opens it: window.__aspPrimer set (without it, automation gets the older tests' compat layout,
    whose step list also holds the settings panel), the guided parts picked in the step list, the real clock"""
    pg = b.new_page(viewport={'width': 1440, 'height': 900}); pg.add_init_script('window.__aspPrimer=true')
    pg.on('pageerror', lambda e: errs.append(f'fit 1440 {lang}: {e}'))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate("()=>{location.hash='roofline'}"); pg.wait_for_timeout(500)
    g = pg.locator('[data-flow-nav] button[data-flow=guided]:visible')
    if g.count(): g.first.click(); pg.wait_for_timeout(700)
    return pg


def fit_entry(b, lang, parts=(5,)):
    pg = fit_page(b, lang)
    for k in range(2, max(parts) + 1):
        nb = pg.locator('.pbar [data-nav="next"]:visible:not([disabled])')
        if nb.count(): nb.first.click()
        else: pg.evaluate(f"()=>document.querySelector('#rf-rail .steps>.step:nth-child({k}) .step-h').click()")
        pg.wait_for_timeout(900); settle(pg)
        if k not in parts: continue
        f = pg.evaluate(FIT_ENTRY); f['step'] = state(pg)['step'] + 1
        check(f'{lang}-1440 part {k}: entering it, the stage fits without scrolling inside{", 30 px to spare" if lang == "zh" else ""}, its top below the top bar (B4, R7.12)',
              f['step'] == k and f['scroll'] <= 0 and f['spare'] >= (30 if lang == 'zh' else 0) and f['top'] >= 59, f)
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    if (1440, 900) in SIZES:
        for lang in LANGS: fit_entry(b, lang)
    for (w, h) in SIZES:
        for lang in LANGS:
            run_guided(b, w, h, lang)
            run_modes(b, w, h, lang)
            owner_b9(b, w, h, lang)
            owner_b9v6(b, w, h, lang, B9V6)
    b.close()
for hit in rule_hits[:60]:
    check('writing rules: ' + hit, False)
for hit in fit_hits[:60]:
    check(hit, False)
check('no page errors', not errs, errs[:5])
print(f'{n - len(fails)}/{n} passed')
print('FAILED:', fails[:60] if fails else 'none')
sys.exit(1 if fails else 0)
