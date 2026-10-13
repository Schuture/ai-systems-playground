"""Flow test for the flash-attention lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH:
- no page errors and no horizontal overflow at any step; the phone first screen (m2): the parts' bar with Next and the chart;
- part 1: the mission's numbers (128 MiB per head, 4 GiB per layer); part 2: the prediction gate (R3.8): Next disabled until a pick, and
  the answer (moving S and P, about 160 µs) appears nowhere before it, in text or tooltips; “Saved …” after a pick;
- part 3: the player (R3.9): ▸ steps, Play advances on its own at the chosen speed, Pause stops it, the scrubber jumps; Next waits for the
  last step; the result (768 numbers, 576 of them S and P), the verdict with the right answer, and why (160 µs against 17 µs);
- part 4: Show me runs to the last tile; 384 numbers; the outputs match (largest difference shown); the row table shows the rescale;
- part 5: Next waits for 128K; the curve; both grow; 32 GiB for one head's S; ▲▼ badges only after the learner's own change (R6.6);
- part 6: single-choice tile sizes; 32 moves more than standard, 256 does not fit (red gauge, red status); part 7: three misconceptions,
  “See the scenarios”, the next-lab footer;
- Scenarios: four, single choice, pressed state follows the settings (R1.8), the URL carries the setup and a fresh page restores it, a
  change under Settings does not move the page (R3.2), the 12-token example in the sandbox;
- the challenge: Settings locked (R4.2), live progress, the three feedback tiers (R4.4), “Show the answer”, a brief chart (the owner's
  review of batch 8): the 2 passing pairs, the initial setup and the pick, nothing else, and a note that counts the 13 left out (R5.4),
  the capability card; the language switch keeps the answer;
- the mini result bar at 390 once the chart scrolls away;
- the writing rules on every visible text node and tooltip, Chinese spacing and punctuation, more than 6 digits, colours only for their
- no hover that only expands a number to its exact value, outside Show the math (the owner, 2026-10-07 23:37 UTC);
  data (R5.6), no text wider than its box (R7.8), a tooltip for every help cursor (R1.1);
- the home tile: the slider changes the sentence; no overflow.
Screenshots: shots/flash-attention/.
Run: python3 tests/flash-attention_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/flash-attention-flow.html', '--only', 'flash-attention'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'flash-attention-flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'flash-attention-flow.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'flash-attention-flow-wrapped.html')
SH = P / 'shots' / 'flash-attention'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, hash_='flash-attention'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    if hash_: pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__fa.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#fa-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
obs = lambda pg: pg.evaluate("()=>{const o=document.querySelector('#fa-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}")
badges = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#fa-metrics .metric .d')].map(d=>d.textContent).filter(Boolean)")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__fa||!window.__fa.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def slide(pg, sel, v, change=True):
    pg.evaluate(f"""()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}}));
      {"i.dispatchEvent(new Event('change',{bubbles:true}));" if change else ''}}}"""); pg.wait_for_timeout(250); settle(pg)


def click(pg, sel):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(250); settle(pg)


def nxt(pg): click(pg, '#fa-rail .step.current [data-nav="next"]')


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#fa-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy” (the fidelity label “Toy run” is the exception, R6.11)'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/Locked in|已锁定/,'“Locked in”'],[/\\bNaN\\b(?![?？])|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万 instead of K, M, B (R6.18)'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],[/OpenAI|Anthropic|DeepMind|Google|Meta\\b|Microsoft|xAI/,'a company named'],
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part|tile)/i,'dark / light named as a colour (R5.7)']];
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
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #fa-mathbody, #fa-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
# blue (weights), amber (KV cache) and violet (runtime overhead) mean nothing in this lab; red only for "does not fit" and a missed
# goal, green only for "fits" and a met goal (R5.6)
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|fa-over|fa-nofit|over|warn|wrong|ex-res/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#fa-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')) out.push(p+' '+c+' on '+(e.id||e.className||e.tagName));}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#fa-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg,.fa-row')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#fa-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
rule_hits, fit_hits = [], []
EXACT = r"""(sel)=>{const root=document.querySelector(sel); if(!root) return [];
  return [...root.querySelectorAll('[data-tipx],[data-tip],[title]')].filter(e=>!e.closest('details.math,[id$="mathbody"]'))
    .map(e=>e.getAttribute('data-tipx')||e.getAttribute('data-tip')||e.getAttribute('title')||'').filter(t=>/精确值是|Exactly|shown rounded to 3|取 3 位有效数字/.test(t)).map(t=>t.slice(0,90));}"""


def rules(pg, where, lang):
    for sel in ('#fa-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
        for h in pg.evaluate(EXACT, sel): rule_hits.append(f'{where}: a hover that only gives the exact value (the owner, 23:37 UTC): {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')
    for h in pg.evaluate(OVER): fit_hits.append(f'{where}: text wider than its box (R7.8): {h}')
    for h in pg.evaluate(HELP): fit_hits.append(f'{where}: help cursor without a tooltip (R1.1): {h}')


def every(pg, where, lang, w):
    check(f'{where}: no horizontal overflow', overflow(pg) <= 0, overflow(pg))
    rules(pg, where, lang)


def before_answer(pg):
    """everything a learner can read before the prediction: header and its popovers, the card, the stage, Settings, tooltips, term tips"""
    out = [pg.evaluate("""()=>{const t=[]; for(const s of ['#fa-app .labhead','#fa-rail','#fa-stage','#fa-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#fa-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")]
    for i in range(pg.evaluate("()=>document.querySelectorAll('#fa-rail .t').length")):
        pg.evaluate(f"()=>{{const e=document.querySelectorAll('#fa-rail .t')[{i}]; e.focus({{preventScroll:true}})}}"); pg.wait_for_timeout(60)
        out.append(pg.evaluate("()=>{const t=document.getElementById('tip'); return t?t.innerText:''}"))
    pg.evaluate("()=>document.activeElement&&document.activeElement.blur()")
    return '\n'.join(out)


def run_guided(b, w, h, lang):
    tag = f'{w} {lang}'
    pg = page(b, w, h, lang)
    s = state(pg)
    check(f'{tag}: opens in the guided parts, part 1', s['mode'] == 'guided' and s['step'] == 0, s)
    check(f'{tag}: catalog question in the header', Z(lang, 'How does FlashAttention compute exact attention', 'FlashAttention 怎么在结果不变') in text(pg, '#fa-app .labhead'), text(pg, '#fa-app .labhead')[:200])
    if w <= 760:
        r = pg.evaluate("()=>{const b=document.querySelector('#fa-bars').getBoundingClientRect(), nb=document.querySelector('#fa-rail .pbar [data-nav=\"next\"]'); const n=nb?nb.getBoundingClientRect():null; return {bars:b.top, n:n?n.bottom:null, h:innerHeight}}")
        check(f'{tag}: phone first screen has the parts’ bar with Next, and the bars (R7.3)', r['n'] is not None and r['n'] < r['h'] and r['bars'] < r['h'] - 40, r)
    o = text(pg, '#fa-rail .step.current')
    check(f'{tag}: part 1 says 128 MiB per head; thread blocks wait for part 6 (v6 S-2)', '128 MiB' in o and 'thread block' not in o, o[:300])
    check(f'{tag}: part 1 bars show standard only', vis(pg, '#fa-bars .fa-brow[data-k="std"]') and 'ghost' in pg.evaluate("()=>document.querySelector('#fa-bars .fa-brow[data-k=\"fa\"]').className"))
    every(pg, f'{tag} part 1', lang, w); shot(pg, f'{w}-{lang}-1-mission')
    nxt(pg)
    # part 2: the prediction gate
    check(f'{tag}: part 2 Next disabled before a pick', all(nexts(pg)), nexts(pg))
    ba = before_answer(pg)
    for bad in ('160 µs', 'memory-bound', '显存带宽受限', '受显存带宽限制'):
        check(f'{tag}: before the prediction, no “{bad}”', bad not in ba, [l for l in ba.split('\n') if bad in l][:2])
    every(pg, f'{tag} part 2', lang, w); shot(pg, f'{w}-{lang}-2-predict')
    click(pg, '#fa-rail [data-pred="qk"]')
    check(f'{tag}: part 2 saved after a pick', Z(lang, 'Saved', '已记下') in text(pg, '#fa-rail .step.current') and not any(nexts(pg)), nexts(pg))
    nxt(pg)
    # part 3: standard attention, the player
    s = state(pg)
    check(f'{tag}: part 3 is the 12-token example, standard path, step 0', s['tstep'] == 0 and vis(pg, '#fa-toy') and not vis(pg, '#fa-real'), s)
    check(f'{tag}: part 3 Next disabled before the last step', all(nexts(pg)), nexts(pg))
    click(pg, '#fa-rail [data-act="fwd"]')
    check(f'{tag}: ▸ moves one step', state(pg)['tstep'] == 1, state(pg)['tstep'])
    check(f'{tag}: step 1 reads Q and K (96 numbers)', '96' in text(pg, '#fa-xfer'), text(pg, '#fa-xfer'))
    check(f'{tag}: the partial result shows after the first step', Z(lang, 'Step 1 of 6', '第 1 步') in obs(pg), obs(pg))
    click(pg, '#fa-rail .seg-ctl[data-key="speed"] button[data-v="fast"]')
    pg.evaluate("()=>document.querySelector('#fa-rail [data-act=\"play\"]').click()"); pg.wait_for_timeout(2900)
    s = state(pg)
    check(f'{tag}: Play advances on its own (fast: the whole run in 8 s)', s['tstep'] >= 2 and s['speed'] == 'fast', s['tstep'])
    pg.evaluate("()=>document.querySelector('#fa-rail [data-act=\"play\"]').click()"); pg.wait_for_timeout(1500)
    t1 = state(pg)['tstep']; pg.wait_for_timeout(1400)
    check(f'{tag}: Pause stops it', state(pg)['tstep'] == t1 and not state(pg)['playing'], state(pg)['tstep'])
    every(pg, f'{tag} part 3 mid', lang, w); shot(pg, f'{w}-{lang}-3-std-mid')
    slide(pg, '#fa-rail input[data-key="tstep"]', 6)
    o = obs(pg)
    check(f'{tag}: part 3 result: 768 numbers, 576 S and P', '768' in o and '576' in o, o[:300])
    check(f'{tag}: part 3 verdict names the right answer', Z(lang, 'The right answer is “Moving S and P between HBM and the chip”', '正确答案是“在 HBM 和芯片之间搬运 S 和 P”') in o, o[:400])
    check(f'{tag}: part 3 why: 160 µs against 17 µs', '160 µs' in o and '17 µs' in o, o[:500])
    check(f'{tag}: part 3 Next enabled at the last step', not any(nexts(pg)), nexts(pg))
    check(f'{tag}: part 3 counter at 768', '768' in text(pg, '#fa-cnt'), text(pg, '#fa-cnt'))
    every(pg, f'{tag} part 3 end', lang, w); shot(pg, f'{w}-{lang}-3-std-end')
    nxt(pg)
    # part 4: FlashAttention, tile by tile
    check(f'{tag}: part 4 Next disabled before the last tile', all(nexts(pg)), nexts(pg))
    click(pg, '#fa-rail [data-act="fwd"]'); click(pg, '#fa-rail [data-act="fwd"]'); click(pg, '#fa-rail [data-act="fwd"]')
    check(f'{tag}: part 4 tile 3: the row table shows a rescale', pg.evaluate("()=>document.querySelectorAll('#fa-row tr.fa-up').length") >= 1, text(pg, '#fa-row'))
    check(f'{tag}: part 4: S, P never written', vis(pg, '.fa-never'))
    every(pg, f'{tag} part 4 mid', lang, w); shot(pg, f'{w}-{lang}-4-fa-mid')
    click(pg, '#fa-rail [data-nav="showme"]')
    o = obs(pg)
    check(f'{tag}: part 4 Show me runs to the end: 384 instead of 768', state(pg)['doneFa'] and '384' in o and '768' in o, o[:300])
    check(f'{tag}: part 4 outputs match (1.7 × 10^-16)', '1.7 × 10' in o and Z(lang, 'Same output', '输出相同') in text(pg, '#fa-status'), (o[:300], text(pg, '#fa-status')))
    check(f'{tag}: part 4 rescale factor 0.19 named', '0.19' in o, o[:600])
    every(pg, f'{tag} part 4 end', lang, w); shot(pg, f'{w}-{lang}-4-fa-end')
    # R3.20 (product owner, 2026-10-04): back to part 3 and on to part 4 again: both runs were watched to the end, so each shows its end
    # at once (no replay) and Next stays open
    click(pg, '#fa-rail .step.current [data-nav="back"]'); s = state(pg)
    check(f'{tag}: R3.20: back in part 3 the standard run shows its end, no replay', s['step'] == 2 and s['tstep'] == 6 and not s['playing'] and not any(nexts(pg)), (s['step'], s['tstep'], nexts(pg)))
    nxt(pg); s = state(pg)
    check(f'{tag}: R3.20: on to part 4 again, its run shows its end, no replay', s['step'] == 3 and s['doneFa'] and s['tstep'] > 0 and not s['playing'] and not any(nexts(pg)), (s['step'], s['tstep'], nexts(pg)))
    nxt(pg)
    # part 5: longer context
    check(f'{tag}: part 5 Next disabled before 128K', all(nexts(pg)), nexts(pg))
    check(f'{tag}: part 5 no ▲▼ on entry', badges(pg) == [], badges(pg))
    check(f'{tag}: part 5 curve shown', vis(pg, '#fa-curve svg'))
    slide(pg, '#fa-ic-N input', 4)
    check(f'{tag}: part 5 ▲ after the learner’s own change', any('▲' in x for x in badges(pg)), badges(pg))
    slide(pg, '#fa-ic-N input', 7)
    o = obs(pg)
    check(f'{tag}: part 5 at 128K: 128.1 GiB against 64.1 GiB, in two short paragraphs, no L2 caveat (v6 S-4)', '128.1 GiB' in o and '64.1 GiB' in o and 'L2' not in o and pg.evaluate("()=>document.querySelectorAll('#fa-rail [data-obs] p').length") <= 2, o[:500])
    check(f'{tag}: part 5 Next enabled at 128K', not any(nexts(pg)), nexts(pg))
    # R5.9: each curve is named at its right end, and the top legend no longer keys the lines; no name meets a dot, the marker line,
    # a tick label or the other line, at 128K (dots at the right end) and at 8K; the bar segments' key sits right under the bars
    lg = text(pg, '#fa-legend')
    check(f'{tag}: part 5 the top legend names no line (R5.9)', not any(x in lg for x in ('solid line', 'dashed line', '实线', '虚线')), lg)
    geo = "()=>{const sv=document.querySelector('#fa-curve svg'), B=e=>{const r=e.getBoundingClientRect(); return [r.left,r.top,r.right,r.bottom]};" \
          "const lab=[...sv.querySelectorAll('text.fa-cv-lab')].map(B), oth=[...sv.querySelectorAll('circle, line, text:not(.fa-cv-lab)')].map(B), pl=[...sv.querySelectorAll('polyline')].map(B);" \
          "const hit=(a,b)=>a[0]<b[2]&&b[0]<a[2]&&a[1]<b[3]&&b[1]<a[3]; return {n:lab.length,txt:[...sv.querySelectorAll('text.fa-cv-lab')].map(t=>t.textContent)," \
          "bad:lab.length===2&&!hit(lab[0],lab[1])&&lab.every(a=>!oth.some(b=>hit(a,b))&&!pl.some(b=>a[0]<b[2]))?0:1}}"
    for k in (7, 3):
        slide(pg, '#fa-ic-N input', k)
        g_ = pg.evaluate(geo)
        check(f'{tag}: part 5 at step {k}: the two lines named at their right ends, overlapping nothing (R5.9)', g_['n'] == 2 and g_['bad'] == 0 and g_['txt'] == [Z(lang, 'Standard', '标准'), 'FlashAttention'], g_)
    below = pg.evaluate("()=>{const r=[...document.querySelectorAll('#fa-bars .fa-brow')].pop().getBoundingClientRect(), k=document.querySelector('#fa-bars .fa-bkey').getBoundingClientRect(); return k.top-r.bottom}")
    check(f'{tag}: part 5 the bar key sits right under the bars (R5.9)', 0 <= below <= 12, below)
    slide(pg, '#fa-ic-N input', 7)
    every(pg, f'{tag} part 5', lang, w); shot(pg, f'{w}-{lang}-5-len')
    nxt(pg)
    # part 6: tile size
    check(f'{tag}: part 6 Next disabled before another size', all(nexts(pg)), nexts(pg))
    check(f'{tag}: part 6 gauge shown, fits at 128', vis(pg, '#fa-gauge .fa-gtrk') and not pg.evaluate("()=>document.querySelector('#fa-gauge .fa-gtrk').classList.contains('fa-over')"))
    click(pg, '#fa-rail [data-lever="32"]')
    o = obs(pg)
    check(f'{tag}: part 6 tiles of 32 move 1,028 MiB, more than standard', '1,028 MiB' in o and Z(lang, 'more than', '还多') in o, o[:300])
    check(f'{tag}: part 6 single choice', pg.evaluate("()=>document.querySelectorAll('#fa-rail [data-lever][aria-pressed=\"true\"]').length") == 1)
    click(pg, '#fa-rail [data-lever="256"]')
    o = obs(pg)
    check(f'{tag}: part 6 tiles of 256 do not fit (578 KiB)', '578 KiB' in o and pg.evaluate("()=>document.querySelector('#fa-gauge .fa-gtrk').classList.contains('fa-over')") and 'bad' in pg.evaluate("()=>document.querySelector('#fa-status').className"), o[:300])
    check(f'{tag}: part 6 Next enabled', not any(nexts(pg)), nexts(pg))
    every(pg, f'{tag} part 6', lang, w); shot(pg, f'{w}-{lang}-6-tile')
    nxt(pg)
    o = text(pg, '#fa-rail .step.current')
    check(f'{tag}: part 7 three misconceptions', o.count('✗') == 3, o.count('✗'))
    check(f'{tag}: part 7 keeps the storage point moved from part 5: 32.0 GiB for one head’s S at 128K (v6 S-4)', '32.0 GiB' in o, o[:600])
    check(f'{tag}: part 7 offers the scenarios and the next lab', vis(pg, '#fa-rail [data-mode-go="sandbox"]') and pg.evaluate("()=>!!document.querySelector('#fa-rail .nextlab, #fa-rail [data-next-lab], #fa-rail .next-lab')"), o[-300:])
    every(pg, f'{tag} part 7', lang, w); shot(pg, f'{w}-{lang}-7-remember')
    pg.close()


def run_modes(b, w, h, lang):
    tag = f'{w} {lang}'
    pg = page(b, w, h, lang)
    mode(pg, 'sandbox')
    check(f'{tag}: scenarios: four, none pressed at first', pg.evaluate("()=>document.querySelectorAll('#fa-rail [data-preset]').length") == 4)
    exp = {0: '8.50 MiB', 1: '520 MiB', 2: '128.1 GiB', 3: '4.06 MiB'}
    for i in range(4):
        click(pg, f'#fa-rail [data-preset="{i}"]')
        pr = pg.evaluate("()=>[...document.querySelectorAll('#fa-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        check(f'{tag}: scenario {i} pressed alone', pr == ['true' if k == i else 'false' for k in range(4)], pr)
        ins = text(pg, '#fa-insight')
        check(f'{tag}: scenario {i} insight', exp[i] in ins, ins[:300])
        every(pg, f'{tag} scenario {i}', lang, w)
        if i in (0, 3): shot(pg, f'{w}-{lang}-s{i}')
    tok = pg.evaluate("()=>location.hash")
    check(f'{tag}: the URL carries the setup', '128_3_128_128_1' in tok, tok)
    # a change under Settings does not move the page (R3.2)
    pg.evaluate("()=>document.querySelector('#fa-ctl-Br').scrollIntoView({block:'center'})"); pg.wait_for_timeout(300)
    y0 = pg.evaluate("()=>document.querySelector('#fa-ctl-Br').getBoundingClientRect().top")
    pg.evaluate("()=>document.querySelector('#fa-ctl-Br button[data-v=\"64\"]').click()"); pg.wait_for_timeout(300)
    y1 = pg.evaluate("()=>document.querySelector('#fa-ctl-Br').getBoundingClientRect().top")
    check(f'{tag}: Settings change keeps its place', abs(y1 - y0) < 2, (y0, y1))
    check(f'{tag}: no scenario pressed after a change', pg.evaluate("()=>[...document.querySelectorAll('#fa-rail [data-preset]')].every(b=>b.getAttribute('aria-pressed')==='false')"))
    pg.evaluate("()=>window.scrollTo(0,0)")
    # the 12-token example in the sandbox
    click(pg, '#fa-view button[data-view="toy"]'); click(pg, '#fa-pathsw button[data-path="fa"]')
    check(f'{tag}: sandbox shows the example, FlashAttention path', vis(pg, '#fa-toy') and state(pg)['path'] == 'fa')
    every(pg, f'{tag} sandbox toy', lang, w); shot(pg, f'{w}-{lang}-s-toy')
    # R3.20 (product owner, 2026-10-04): the path already shown changes nothing; a run watched to the end shows its end when its path
    # comes back; the play button then says it plays the run again, and does
    PL = '#fa-st-play [data-act="play"]'
    s = state(pg); n_ = pg.evaluate("()=>+document.querySelector('#fa-st-play input[data-key=\"tstep\"]').max")
    check(f'{tag}: R3.20: a path not watched yet starts at its first step', s['path'] == 'fa' and s['tstep'] == 0 and not s['playing'], s['tstep'])
    pg.evaluate(f"()=>{{const i=document.querySelector('#fa-st-play input[data-key=\"tstep\"]'); i.value={n_}; i.dispatchEvent(new Event('input',{{bubbles:true}}));}}"); pg.wait_for_timeout(150)
    click(pg, '#fa-pathsw button[data-path="fa"]'); s = state(pg)
    check(f'{tag}: R3.20: the path already shown changes nothing', s['tstep'] == n_ and s['doneFa'] and not s['playing'], s['tstep'])
    click(pg, '#fa-pathsw button[data-path="std"]'); s = state(pg)
    check(f'{tag}: R3.20: the other path, not watched yet, starts at its first step', s['path'] == 'std' and s['tstep'] == 0, s['tstep'])
    click(pg, '#fa-pathsw button[data-path="fa"]'); s = state(pg)
    tip_ = pg.evaluate("s=>document.querySelector(s).dataset.tipx||''", PL)
    check(f'{tag}: R3.20: back on the path watched to the end: its end at once, no replay', s['tstep'] == n_ and not s['playing'] and Z(lang, 'Replay', '重放') in text(pg, PL), (s['tstep'], text(pg, PL)))
    check(f'{tag}: R3.20: the play button’s tooltip says it plays the run again from the start', Z(lang, 'again from the start', '从头再放') in tip_, tip_)
    pg.evaluate("s=>document.querySelector(s).click()", PL); pg.wait_for_timeout(150); s = state(pg)
    check(f'{tag}: R3.20: Replay plays it again from the start', s['playing'] and s['tstep'] < n_, s['tstep'])
    pg.evaluate("s=>document.querySelector(s).click()", PL); pg.wait_for_timeout(100)
    tok = pg.evaluate("()=>location.hash")
    pg2 = page(b, w, h, lang, hash_=tok.lstrip('#'))
    s2 = state(pg2)
    check(f'{tag}: a fresh page restores the setup', s2['mode'] == 'sandbox' and s2['Br'] == 64 and s2['dec'] and s2['view'] == 'toy' and s2['path'] == 'fa', (tok, s2))
    pg2.close()
    # the challenge
    mode(pg, 'challenge')
    s = state(pg)
    check(f'{tag}: challenge starts at 64 × 64', s['Br'] == 64 and s['Bc'] == 64, s)
    check(f'{tag}: challenge: Settings locked with a note', not vis(pg, '#fa-ctl-wrap') and vis(pg, '#fa-lock'))
    g = text(pg, '#fa-ch-g2')
    check(f'{tag}: challenge progress shows 4.01 GiB', '4.01 GiB' in g, g)
    every(pg, f'{tag} challenge start', lang, w); shot(pg, f'{w}-{lang}-c0')
    click(pg, '#fa-rail [data-act="submit"]')
    r = text(pg, '#fa-ch-res')
    check(f'{tag}: tier 1: not yet, with a hint', Z(lang, 'Not yet', '还没达到') in r and Z(lang, 'Hint', '提示') in r and vis(pg, '#fa-ch-res [data-act="reveal"]'), r[:300])
    check(f'{tag}: the reveal is “Show the answer”', text(pg, '#fa-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), text(pg, '#fa-ch-res [data-act="reveal"]'))
    click(pg, '#fa-ch-res [data-act="reveal"]')
    cr = pg.evaluate("""()=>{const rows=[...document.querySelectorAll('#fa-ch-res > .cmp .cmp-row')], tg=r=>!!r.querySelector('.cmp-tag,.cmp-you');
      return {n:rows.length, ok:rows.filter(r=>r.classList.contains('ok')).length, me:rows.filter(r=>r.classList.contains('me')).length,
        stray:rows.filter(r=>!r.classList.contains('ok')&&!r.classList.contains('me')&&!tg(r)).length, note:(document.querySelector('#fa-ch-res .cmp-note')||{}).textContent||'',
        title:(document.querySelector('#fa-ch-res .cmp-title')||{}).textContent||'', aria:(document.querySelector('#fa-ch-res > .cmp')||{getAttribute:()=>''}).getAttribute('aria-label')||''}}""")
    check(f'{tag}: the answer, from the initial setup: the 2 passing pairs and the initial setup (my pick), nothing else', cr['n'] == 3 and cr['ok'] == 2 and cr['me'] == 1 and cr['stray'] == 0, cr)
    check(f'{tag}: the note counts the 13 pairs left out', Z(lang, 'The other 13 tile pairs miss the goal or do not fit, and are not shown.', '其余 13 对 tile 没达标或放不下，图里没有列出。') in cr['note'], cr['note'])
    check(f'{tag}: the title and the aria name only what the chart shows', not re.search(r'for each|each tile pair|按 Q tile|每对', cr['title'] + cr['aria']) and cr['aria'].count('×') == 0 and cr['aria'].count('Q tile') == 3, (cr['title'], cr['aria']))
    click(pg, '#fa-ic-chBr button[data-v="256"]'); click(pg, '#fa-ic-chBc button[data-v="128"]')
    r = text(pg, '#fa-ch-res')
    check(f'{tag}: 256 × 128 does not fit (258 KiB)', '258 KiB' in r and Z(lang, 'Not yet', '还没达到') in r, r[:300])
    click(pg, '#fa-ic-chBc button[data-v="32"]')
    r = text(pg, '#fa-ch-res')
    check(f'{tag}: tier 2: met, but more changed than needed', Z(lang, 'Goal met', '达成目标') in r and Z(lang, 'changed more than needed', '改得比需要的多') in r, r[:300])
    click(pg, '#fa-ic-chBc button[data-v="64"]')
    r = text(pg, '#fa-ch-res')
    check(f'{tag}: tier 3: met with the smallest change, 1.01 GiB', Z(lang, 'smallest change', '改动最少') in r and '1.01 GiB' in r, r[:300])
    cr = pg.evaluate("""()=>{const rows=[...document.querySelectorAll('#fa-ch-res > .cmp .cmp-row')], tg=r=>!!r.querySelector('.cmp-tag,.cmp-you');
      return {n:rows.length, ok:rows.filter(r=>r.classList.contains('ok')).length, me:rows.filter(r=>r.classList.contains('me')).length,
        stray:rows.filter(r=>!r.classList.contains('ok')&&!r.classList.contains('me')&&!tg(r)).length, note:(document.querySelector('#fa-ch-res .cmp-note')||{}).textContent||'',
        title:(document.querySelector('#fa-ch-res .cmp-title')||{}).textContent||'', aria:(document.querySelector('#fa-ch-res > .cmp')||{getAttribute:()=>''}).getAttribute('aria-label')||''}}""")
    check(f'{tag}: brief chart: the 2 passing pairs (one is mine) and the initial setup, nothing else', cr['n'] == 3 and cr['ok'] == 2 and cr['me'] == 1 and cr['stray'] == 0, cr)
    check(f'{tag}: the note still counts the 13 pairs left out', ('13' in cr['note']) and Z(lang, 'not shown', '没有列出') in cr['note'], cr['note'])
    check(f'{tag}: chart tags the initial setup', Z(lang, 'initial setup', '初始设置') in text(pg, '#fa-ch-res'))
    check(f'{tag}: status green', 'ok' in pg.evaluate("()=>document.querySelector('#fa-status').className"))
    check(f'{tag}: capability card shown', vis(pg, '#fa-rail [data-exits]'))
    every(pg, f'{tag} challenge done', lang, w); shot(pg, f'{w}-{lang}-c-done')
    other = 'zh' if lang == 'en' else 'en'
    click(pg, f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(300)
    s = state(pg)
    check(f'{tag}: language switch keeps the answer', s['mode'] == 'challenge' and s['Br'] == 256 and s['Bc'] == 64 and s['chSubmitted'], s)
    click(pg, f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(300)
    if w <= 760:
        pg.evaluate("()=>{const r=document.querySelector('#fa-rail [data-exits]'); r.scrollIntoView({block:'start'}); window.scrollBy(0,-80)}"); pg.wait_for_timeout(700)
        d = pg.evaluate("()=>{const d=document.getElementById('dock'); return d&&!d.hidden&&d.getBoundingClientRect().height>0?d.innerText:''}")
        check(f'{tag}: the mini result bar shows once the chart scrolls away', bool(d.strip()), d)
        every(pg, f'{tag} dock', lang, w); shot(pg, f'{w}-{lang}-dock')
    pg.close()


def run_tile(b, w, h, lang):
    tag = f'{w} {lang}'
    pg = page(b, w, h, lang, hash_='')
    pg.wait_for_timeout(300)
    pg.evaluate("()=>{const g=document.querySelector('.world[data-world=\"models\"]'); if(g) g.dispatchEvent(new MouseEvent('click',{bubbles:true}))}"); pg.wait_for_timeout(400)
    t = text(pg, '#tile-flash-attention')
    check(f'{tag}: home tile rendered', bool(t), t[:100])
    if t:
        r0 = text(pg, '.tz-fa .tz-read')
        slide(pg, '#tz-fa-n', 7)
        r1 = text(pg, '.tz-fa .tz-read')
        check(f'{tag}: tile sentence changes', r0 != r1 and '128.1 GiB' in r1 and '520 MiB' in r0, (r0, r1))
        check(f'{tag}: tile sentence at most 25 words', len(r1.split()) <= 25 if lang == 'en' else len(re.findall(r'[\u4e00-\u9fff]|[A-Za-z0-9.\-]+', r1)) <= 30, r1)
        check(f'{tag}: home: no horizontal overflow', overflow(pg) <= 0, overflow(pg))
        pg.evaluate("()=>document.querySelector('.tz-fa').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
        shot(pg, f'{w}-{lang}-tile')
    pg.close()



# ----- the owner's message of 2026-10-08 (reviews/owner-b9-2026-10-08.md): G1, a lead-in that says what to pick and what to watch
# instead of “one at a time, they do not stack”; G2, no sentence leans on a detail of an earlier part, the recap states the pattern
OWNER_B9 = {'scen': ('Pick a scenario and see how much one head reads and writes in HBM.', '选一个场景，看看一个 head 要在 HBM 里读写多少数据。'), 'chal': None, 'parts': [(5, [('Pick a tile size and see how much it moves through HBM and whether it fits in SRAM.', '选一种 tile 大小，看看它要经 HBM 搬多少数据，SRAM 又放不放得下。')]), (6, [('so they grow with the square of the context', '所以它们随上下文长度的平方增长'), ('the row’s running sum and the output built so far are rescaled once', '这一行的累计和与已经算出的输出就整体缩放一次'), ('pick the largest tile that still fits in the SRAM of one thread block', '所以要选一个 thread block 的 SRAM 放得下的最大 tile'), ('so its output differs from standard attention’s only by rounding', '所以输出和标准 attention 只差舍入')])]}
OWNER_B9_NOTES = ('一次只能选', '一次选一个', '不会叠加', '每次只试', '下面的选项是单选', 'do not stack', 'One at a time', 'Pick one at a time', 'one at a time;')
OWNER_B9_BACK = re.compile(r'In part [0-9]|in part [0-9]|As in part|as in part|第 [0-9] 部分里|在第 [0-9] 部分|和第 [0-9] 部分一样|还是第 [0-9]')
owner_b9_once = []


def owner_b9(b, w, h, lang):
    if not owner_b9_once:
        owner_b9_once.append(1)
        src = (P / 'labs' / 'flash-attention.js').read_text() + ((P / 'teasers' / 'flash-attention.js').read_text() if (P / 'teasers' / 'flash-attention.js').exists() else '')
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
  const xs=xsep?[...root.querySelectorAll(xsep)].filter(e=>e.getClientRects().length).map(e=>{const cs=getComputedStyle(e); return cs.borderTopWidth+' '+cs.borderTopStyle;}):[];
  return {n:maths.length, bad, blocks:blocks.length, sep, body:box?box.textContent.trim():'', inBody:box?box.querySelectorAll('math').length:0, txt:parts.join(' ')+' '+tipTxt, xsep:xs};}"""

# H7 (the owner on RLVR and GRPO): the buttons in one control row share one height and one top (the player's ◂, play, ▸ and the
# speed buttons 慢/中/快 used to be 32 px and 22 px tall, 5 px apart); a segmented group counts by its outer box
CTL_ROWS = r"""(sel)=>{const out={n:0,bad:[]}; for(const row of document.querySelectorAll('#view-dyn:not([hidden]) '+sel)){ if(!row.getClientRects().length) continue;
  const bs=[...new Set([...row.querySelectorAll('button')].filter(x=>x.getClientRects().length).map(x=>x.closest('.seg-ctl')||x))].map(x=>{const r=x.getBoundingClientRect(); return {t:r.top,h:r.height,x:(x.textContent||'').trim().slice(0,8)};});
  const lines=[]; for(const q of bs){const l=lines.find(L=>Math.abs(L[0].t+L[0].h/2-(q.t+q.h/2))<12); if(l) l.push(q); else lines.push([q]);}
  for(const L of lines){ if(L.length<2) continue; out.n++; const dt=Math.max(...L.map(q=>q.t))-Math.min(...L.map(q=>q.t)), dh=Math.max(...L.map(q=>q.h))-Math.min(...L.map(q=>q.h));
    if(dt>1||dh>1) out.bad.push(L.map(q=>q.x+':'+q.t.toFixed(0)+'/'+q.h.toFixed(0)).join(' '));}}
  return out;}"""


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
        tz = P / 'teasers' / f"{cfg['lab']}.js"
        src = (P / 'labs' / f"{cfg['lab']}.js").read_text() + (tz.read_text() if tz.exists() else '')
        left = [s for s in cfg['src_gone'] if s in src]
        check(f"owner b9v6 H1: no plain-text formula of the converted kinds left in labs/{cfg['lab']}.js or its tile", not left, left)
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
    cseen, cbad = 0, []
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
        if cfg.get('ctl_rows'):
            cr = pg.evaluate(CTL_ROWS, cfg['ctl_rows'])
            cseen += cr['n']; cbad += [f'{where}: {x}' for x in cr['bad']]
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
    if cfg.get('ctl_rows'):
        check(f"{tag} H7: the buttons in each player row share one height and one top ({cseen} rows seen)", cseen > 0 and not cbad, cbad[:4])
    check(f'{tag} owner 2026-10-09 (R6.37): no “Show the math” box scrolls sideways, every row fits its box ({nscr} screens with it open)', nscr > 0 and not scr, scr[:5])
    check(f'{tag}: no page errors', not here, here[:3])

B9V6 = {'lab': 'flash-attention', 'math': '#fa-mathbody', 'block': '.fa-mb', 'sep_extra': '.fa-cvcap', 'ctl_rows': '.fa-pbtns', 'src_gone': ['N × N numbers each', '各 N × N 个数值', 'four moves of N × N', '四次 N × N', 'why its N × N scores', '时 N × N 的得分', 'four N × d matrices', 'two N × d matrices', '四个 N × d', '两个 N × d', 'grows with Br × Bc', '大小是 Br × Bc', 'it has N × N of them', 'N × N in all', 'stores no N × N matrix', '不存 N × N 的矩阵', 'moves its N × N matrices', 'QKᵀ 只要约', 'and QKᵀ and P·V about', 'QKᵀ 和 P·V 各约', 'q·k ÷ √head_dim', 'exp(score − m)', 'exp(old m − new m)', 'q<sub>i</sub>', 'o ÷ ℓ is the row', 'S = QKᵀ ÷ √d written', 'S = QKᵀ ÷ √d 写到', 'Θ(Nd + N²)', 'computes S = QKᵀ ÷ √4', 'q·k ÷ √4', '最后用 o 除以 ℓ', 'Σ_j e^{'], 'text_gone': ['q·k ÷ √', 'exp(score', 'QKᵀ ÷ √', 'o ÷ ℓ', 'Θ(Nd', 'exp(old m', 'e^{']}


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in LANGS:
        for w, h in SIZES:
            run_guided(b, w, h, lang)
            run_modes(b, w, h, lang)
            owner_b9(b, w, h, lang)
            owner_b9v6(b, w, h, lang, B9V6)
            run_tile(b, w, h, lang)
    b.close()

for e in errs[:20]: print('PAGE ERROR', e)
for hit in rule_hits[:60]: print('RULE', hit)
for hit in fit_hits[:60]: print('FIT', hit)
total = n + 3
passed = n - len(fails) + (0 if errs else 1) + (0 if rule_hits else 1) + (0 if fit_hits else 1)
print(f'{passed}/{total} checks passed')
sys.exit(0 if passed == total else 1)
