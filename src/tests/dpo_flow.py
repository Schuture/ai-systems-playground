"""Flow test for the DPO lab: every mode and part at 1440x900 and 390x844 in EN and ZH, in light and dark,
plus 1280/900/720 layout checks, reduced motion, keyboard jumps, the dock, the hash round-trip, the
language switch mid-challenge, the round-5 rules (reveal on navigation, stage scroll reset, state-gated
sentences, grayscale roles, complete curve key, labels inside the plot), the copy-pass rules (no "·" joining
clauses, no 起点 / starting point, no 控件 / Controls, no sentences about the mini result bar or hovering,
training locked until the prediction), the playback speed control and the home-tile teaser.
Screenshots go to shots/dpo/.

Run: python3 tests/dpo_flow.py            (everything)
     python3 tests/dpo_flow.py timing     (only the timing-sensitive play, pause and speed checks, to re-run them alone)
"""
import json, pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/dpo.html', '--only', 'dpo'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-dpo.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'dpo.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-dpo.html')
SH = P / 'shots' / 'dpo'; SH.mkdir(parents=True, exist_ok=True)
ONLY = sys.argv[1] if len(sys.argv) > 1 else ''
errs, notes, fails = [], [], []

def check(name, ok, info=''):
    notes.append(('OK   ' if ok else 'FAIL ') + name + (f'  [{info}]' if info else ''))
    if not ok: fails.append(name)

def page(b, w, h, lang, hash_='dpo', dark=False, reduced=False):
    pg = open_drawers(b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light', reduced_motion='reduce' if reduced else 'no-preference'))
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_TUNNEL' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(350)
    return pg

overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
disabled = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return !!e&&e.disabled}", sel)
txt = lambda pg, sel: pg.inner_text(sel).replace('\n', ' | ')
state = lambda pg: pg.evaluate("()=>window.__dpo.state()")
calc = lambda pg, s: pg.evaluate("(s)=>{const r=window.__dpo.compute(s); return {pi:r.pi,kl:r.kl,w:r.wAB,loss:r.loss}}", s)
Z = lambda lang, en, zh: zh if lang == 'zh' else en
pct = lambda p: f'{p * 100:.0f}%'
units = lambda t: len(re.findall(r'[\u4e00-\u9fff]', t)) + len(re.findall(r'[A-Za-z0-9%]+', t))   # CJK characters + Latin words and numbers (as in continuous-batching_flow.py)
TOPBAR = 40
IDS = ['mission', 'predict', 'steps', 'loss', 'fade', 'beta', 'shared', 'remember']
PART = {k: i for i, k in enumerate(IDS)}

def go_step(pg, i):
    pg.evaluate(f"()=>document.querySelector('#dpo-rail [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(180)

def set_steps(pg, v, sel='dpo-ic-steps-in'):
    pg.evaluate(f"()=>{{const i=document.getElementById('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}}))}}"); pg.wait_for_timeout(150)

def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(120)   # no stray hover tooltip in the picture
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)

def in_view(pg, sel):
    return pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.height>0&&r.top>=58&&r.bottom<=innerHeight+1}", sel)

def settle(pg, ms=2500):
    """Wait until a smooth scroll (A.ui.reveal) has finished."""
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120

def top_shown(pg, sel):
    """The element's top edge after scrolling settles; A.ui.reveal puts it between the top bar and 160 px above the bottom."""
    settle(pg)
    return pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return null; return Math.round(e.getBoundingClientRect().top)}", sel)

def hover_tip(pg, sel):
    loc = pg.locator(sel).first; loc.evaluate("e=>e.scrollIntoView({block:'center',inline:'nearest',behavior:'instant'})"); pg.wait_for_timeout(150)   # centered: clear of the sticky top bar
    bb = loc.bounding_box(); pg.mouse.move(bb['x'] + 3, bb['y'] + 3); pg.mouse.move(bb['x'] + min(8, bb['width'] / 2), bb['y'] + min(6, bb['height'] / 2)); pg.wait_for_timeout(180)
    return pg.inner_text('#tip') if vis(pg, '#tip') else ''

# no blue, amber or violet (weights, KV cache, runtime overhead elsewhere on the site) in light or dark (R5.6)
SCAN = """(sel)=>{const bad=['rgb(40, 85, 197)','rgb(211, 135, 0)','rgb(113, 83, 230)','rgb(90, 138, 255)','rgb(239, 174, 54)','rgb(155, 133, 255)'];
 const out=[]; for(const e of document.querySelectorAll(sel)){const cs=getComputedStyle(e); if(cs.display==='none') continue;
  for(const p of ['color','backgroundColor','borderTopColor','borderLeftColor','outlineColor','fill','stroke','backgroundImage','boxShadow']){const v=cs[p]; if(v&&bad.some(b=>v.includes(b))) out.push((e.getAttribute('class')||e.tagName)+' '+p+' '+v);}}
 return out;}"""

# the copy-pass rules, checked on every visible text node (R6.14, R6.17, R2.1, R3.4)
RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|mini version|迷你结果条|迷你版/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],
   [/updates as you change|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/最多能离|may drift from|may let the model drift|能让模型离/,'β described as a hard cap']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||!e.offsetParent&&getComputedStyle(e).position!=='fixed') continue;
   if(e.closest('svg, pre, code, #tip')) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,80));}
 return out;}"""
COLONS = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const A=window.ASP, out=new Set(), texts=[];
 // leaf blocks (a paragraph with an inline jump link still counts as one), the popovers and the math panel (also when closed), every tooltip string
 const blocks=root.querySelectorAll('p,li,button,label,div,span.lbl,span.tag,.status,.dk-read,.cmp-lab,h1,h2,h3,.tz-leg > span,.legend-row > span');
 blocks.forEach(e=>{if(e.closest('[data-endnotes],pre,code,svg')) return; if(e.matches('.tz-leg,.presets,.choices,.legend-row,.chips,.seg-ctl,.modes,.levers')||e.querySelector('p,li,div,ul,ol,h1,h2,h3,table')) return; texts.push(e.textContent);});
 root.querySelectorAll('[data-tipx]').forEach(e=>texts.push(e.getAttribute('data-tipx').replace(/<[^>]+>/g,'')));
 root.querySelectorAll('[data-tip]').forEach(e=>{const k=e.getAttribute('data-tip'); const v=A&&A.t?A.t(k):''; if(v&&v!==k) texts.push(String(v).replace(/<[^>]+>/g,''));});
 for(const t of texts){const sents=t.replace(/\\s+/g,' ').split(/(?<=[。？！])|(?<=[.?!])\\s+/);
   for(const x of sents){const n=(x.match(/：|: /g)||[]).length; if(n>=2) out.add(x.trim().slice(0,160));}}
 return [...out];}"""
# ---- the owner's review of 2026-10-09: formulas typeset (H1), Show the math as short blocks (H2), labeled stage blocks (H5), svg text in bounds (H7)
MATHS = r"""(sel)=>[...document.querySelectorAll(sel+' math')].map(m=>m.textContent.replace(/\s+/g,''))"""
maths = lambda pg, sel: pg.evaluate(MATHS, sel)
# H1 rendered: no converted pattern left as plain text (page text and every hover string it uses), and no TeX command shown as a word
PLAIN = r"""(sel)=>{const pat=/σ\(|−log|log ?\(π|π_ref|r̂|e\^|β·log|Σ|\/\(1 ?\+|<sub>|<sup>|\(η ÷ β\)/, out=[], A=window.ASP;
 const scan=(root,src)=>{if(!root) return; const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('math,code,pre,script,style')) continue; const t=n.textContent; if(pat.test(t)) out.push(src+': '+t.trim().slice(0,90));}};
 scan(document.querySelector(sel),'page');
 for(const k of new Set([...document.querySelectorAll(sel+' [data-tip]')].map(e=>e.getAttribute('data-tip')))){const v=A&&A.t?A.t(k):''; if(v&&v!==k) scan(new DOMParser().parseFromString('<div>'+v+'</div>','text/html').body,'tip '+k);}
 for(const e of document.querySelectorAll(sel+' [data-tipx]')){const t=e.getAttribute('data-tipx'); if(pat.test(t)) out.push('tipx: '+t.slice(0,90));}
 return out;}"""
TEXWORD = r"""(sel)=>{const out=[], bad=/^(left|right|frac|sqrt|hat|text|mathrm|operatorname|cdot|times|sigma|beta|pi|eta|Delta|partial|nabla|sum|quad|big|sim|succ|mathbb|infty|approx)$/, A=window.ASP, roots=[document.querySelector(sel)];
 for(const k of new Set([...document.querySelectorAll(sel+' [data-tip]')].map(e=>e.getAttribute('data-tip')))){const v=A&&A.t?A.t(k):''; if(v&&v!==k) roots.push(new DOMParser().parseFromString('<div>'+v+'</div>','text/html').body);}
 for(const r of roots){if(!r) continue; for(const e of r.querySelectorAll('math mi, math mtext, math mo')){const t=e.textContent.trim(); if(bad.test(t)||t.includes('\\')) out.push(t);}} return out;}"""
h1_hits = []
WIRE = r"""()=>{const w=document.getElementById('dpo-wire'); if(!w) return {shown:false}; const q=w.getBoundingClientRect();
 return {shown:!w.hidden&&w.getClientRects().length>0, label:((w.querySelector('.dpo-blk-h')||{}).textContent||''), lines:[...w.querySelectorAll('svg line')].map(l=>+l.getAttribute('stroke-width')),
   cells:[...w.querySelectorAll('.dpo-wd')].map(e=>e.textContent), cap:((w.querySelector('.dpo-wcap')||{}).textContent||''), svgText:w.querySelectorAll('svg text').length,
   inside:[...w.querySelectorAll('*')].every(e=>{const r=e.getBoundingClientRect(); return r.width===0||(r.left>=q.left-0.5&&r.right<=q.right+0.5);})};}"""
# polish round 5: on a stage at least 736 px wide the curve sits beside the answers (dpo.css); its block is then set off by the rule
# between the two columns (the chart's ::before) instead of a rule above it, so either is accepted (R7.13)
BLOCKS_H5 = r"""()=>{const g=document.getElementById('dpo-graph'), bars=document.getElementById('dpo-bars');
 return [...document.querySelectorAll('#dpo-stage .dpo-blk-h, #dpo-stage .dpo-curve-head')].filter(e=>e.getClientRects().length).map(e=>{
  if(e.classList.contains('dpo-curve-head')&&e.getBoundingClientRect().left>=bars.getBoundingClientRect().right){const b=getComputedStyle(g,'::before'); return b.borderLeftWidth+' '+b.borderLeftStyle+' (between the columns)';}
  return getComputedStyle(e).borderTopWidth+' '+getComputedStyle(e).borderTopStyle;});}"""
h7_hits = []
def h7(pg, where):
    for x in pg.evaluate(SVGTEXT): h7_hits.append(f'{where}: {x}')
def h1(pg, where):
    for x in pg.evaluate(PLAIN, '#dpo-app'): h1_hits.append(f'{where}: plain formula: {x}')
    for x in pg.evaluate(TEXWORD, '#dpo-app'): h1_hits.append(f'{where}: TeX word in a formula: {x}')
# H2: Show the math is a few labeled blocks of aligned rows, no run-on "= … = … =" row, no loose paragraph, nothing that needs a scrollbar
MATHPANEL = r"""()=>{const mb=document.getElementById('dpo-mathbody'), bl=[...mb.querySelectorAll(':scope > .dpo-mb')];
 const runon=[]; for(const eq of mb.querySelectorAll('.dpo-eq')){const rows={}; for(const c of eq.children){(rows[c.dataset.r]=rows[c.dataset.r]||[]).push(c.textContent.trim());} for(const cs of Object.values(rows)) if(cs.filter(x=>x==='=').length>2) runon.push(cs.join(' '));}
 const over=[...mb.querySelectorAll('.dpo-eq,.dpo-fx')].filter(e=>e.getClientRects().length&&e.scrollWidth>e.clientWidth+1).map(e=>e.textContent.slice(0,40));
 return {ks:bl.map(b=>b.dataset.k), lead:bl.every(b=>!!b.querySelector(':scope > .dpo-ml')&&!!b.querySelector('math')), rule:bl.every(b=>getComputedStyle(b).borderTopWidth==='1px'),
   runon, over, loose:mb.querySelectorAll(':scope > p').length, h:mb.textContent.includes('h = ')};}"""
# H7: every svg <text> inside its svg and its panel
SVGTEXT = r"""()=>{const out=[]; for(const s of document.querySelectorAll('#dpo-app svg')){if(!s.getClientRects().length) continue; const sb=s.getBoundingClientRect(), pn=s.closest('.panel'), pb=pn?pn.getBoundingClientRect():null;
  for(const t of s.querySelectorAll('text')){const b=t.getBoundingClientRect(); if(!b.width) continue;
    if(b.left<sb.left-1||b.right>sb.right+1||b.top<sb.top-1||b.bottom>sb.bottom+1) out.push('svg: '+t.textContent+' '+[b.left,b.right,sb.left,sb.right].map(Math.round));
    if(pb&&(b.left<pb.left-0.5||b.right>pb.right+0.5)) out.push('panel: '+t.textContent);}}
 return out;}"""
rule_hits = []
badges = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#dpo-metrics .metric .d')].map(d=>d.textContent)")
no_logit = lambda pg: pg.evaluate("()=>{const t=['#dpo-rail','#dpo-stage'].map(s=>document.querySelector(s).innerText).join(' '); return !/logit|softmax/i.test(t)}")
# the owner's clarification (2026-10-07 23:37 UTC): no hover only expands a shown number to its exact value; Show the math may write one out
EXACT_HOVER = """r=>{const root=document.querySelector(r), out=[]; if(!root) return out;
  for(const e of root.querySelectorAll('[data-tipx],[data-tip]')){if(e.closest('.mathbody')) continue; let t=e.getAttribute('data-tipx')||''; const k=e.getAttribute('data-tip');
    if(k&&window.ASP) t+=' '+(window.ASP.t(k)||''); if(/精确值是|Exactly/.test(t)) out.push(t.slice(0,100));}
  for(const e of root.querySelectorAll('.num-big')) if(!e.closest('.mathbody')&&(e.hasAttribute('data-tipx')||e.closest('.def'))) out.push('hover on '+e.textContent);
  return out;}"""


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
    ms = pg.evaluate(MSCROLL, '#dpo-app')
    check(f'{where}: no row of Show the math scrolls sideways at {pg.viewport_size["width"]} px (the owner, 2026-10-09; R6.37)', not ms, ms[:3])


def rules(pg, where):
    math_scroll(pg, where)
    hx = pg.evaluate(EXACT_HOVER, '#dpo-app'); check(f'{where}: no exact-value hover outside Show the math', not hx, hx[:3])
    for sel in ('#dpo-app', '#tile-dpo'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: sentence with two colons: {h}')
    for h in pg.evaluate(COLONS, '#dock'): rule_hits.append(f'{where}: sentence with two colons in the mini result bar: {h}')

def run_flow(b, w, h, lang, dark=False):
    tag = f'{"dark_" if dark else ""}{w}_{lang}'; pg = page(b, w, h, lang, dark=dark)
    title = pg.inner_text('#dpo-app .labtitle')
    check(f'{tag} title', title == Z(lang, 'How does DPO pull a model toward preferences without a reward model?', '不训练奖励模型，DPO 怎么把模型拉向偏好？'), title)
    check(f'{tag} mode description', Z(lang, 'Eight short parts', '一共 8 个部分') in pg.evaluate("()=>{const b=document.querySelector('#dpo-app [data-flow-nav] button[aria-pressed=true]'); return b?(b.dataset.tipx||''):''}"))
    check(f'{tag} starts guided, part 1 of 8', Z(lang, 'part 1 of 8', '第 1 / 8 部分') in pg.inner_text('#dpo-rail .rail-top').lower())
    stt = pg.inner_text('#dpo-status')
    check(f'{tag} status at start says the model most often gives B (red)', pg.evaluate("()=>document.getElementById('dpo-status').className.includes('bad')") and stt == Z(lang, 'The model most often gives B (52%)', '模型最常给出 B（52%）'), stt)
    if w < 760:
        order = pg.evaluate("()=>['#dpo-rail','#dpo-stage','#dpo-controls'].map(s=>Math.round(document.querySelector(s).getBoundingClientRect().top))")
        check(f'{tag} phone layout: the part card comes first, then the chart, then the settings', order[0] < order[1] < order[2] and order[0] < h, str(order))
    tip = hover_tip(pg, '#dpo-st-title .dpo-bt')
    check(f'{tag} stage title: the β tooltip says what β is and marks its effect as this lab’s', Z(lang, 'weight of the KL penalty in the RLHF objective', 'KL 惩罚的系数') in tip and Z(lang, 'In this lab, the larger β, the less the model changes in the same number of steps', '在这个 Lab 里，β 越大，同样的训练步数下模型改动得越少') in tip, tip[:200])
    m1 = pg.inner_text('#dpo-rail .step.current')
    check(f'{tag} mission: defines π_ref (typeset), names DPO’s one idea, says what β is before its effect in this lab; no chips, no “examples” line, no “at most”', 'πref' in maths(pg, '#dpo-rail .step.current') and Z(lang, 'no separate reward model', '不用另训一个奖励模型') in m1 and Z(lang, 'In this lab, the larger β', '在这个 Lab 里，β 越大') in m1
          and 0 <= m1.find(Z(lang, 'β is the weight of the KL penalty in the RLHF objective', 'β 是 RLHF 目标里 KL 惩罚的系数')) < m1.find(Z(lang, 'In this lab, the larger β', '在这个 Lab 里，β 越大'))
          and not vis(pg, '#dpo-rail .step.current .chips') and Z(lang, 'examples written for this lab', '为本 Lab 写的示例') not in m1 and Z(lang, 'at most', '最多') not in m1, m1[:300])
    check(f'{tag} mission: at most 3 paragraphs', pg.evaluate("()=>document.querySelectorAll('#dpo-rail .step.current .step-b > p, #dpo-rail .step.current .step-body > p').length") <= 3)
    check(f'{tag} no “logit” or “softmax” on the card or the stage before part 3 defines them (setup line without it)', no_logit(pg) and Z(lang, 'logit', 'logit') not in pg.inner_text('#dpo-st-sub'), pg.inner_text('#dpo-st-sub'))
    check(f'{tag} before the prediction the stage cannot train (R3.8)', not vis(pg, '#dpo-transport [data-act="play"]') and Z(lang, 'Training starts after you answer part 2.', '在第 2 部分作答后开始训练。') == pg.inner_text('#dpo-tnote') and not vis(pg, '#dpo-ctl-steps'), pg.inner_text('#dpo-tnote'))
    rules(pg, f'{tag} mission')
    pg.evaluate("()=>window.scrollTo(0,0)"); shot(pg, f'{tag}_1_mission')
    check(f'{tag} overflow, mission', overflow(pg) <= 0, str(overflow(pg)))
    # part 2: predict gates Next
    pg.click('#dpo-rail [data-nav="next"]'); pg.wait_for_timeout(250)
    t2 = top_shown(pg, '#dpo-rail .pbar' if w <= 760 else '#dpo-rail .step.current')   # phones (m2): the part bar
    check(f'{tag} Next reveals the new {"part bar" if w <= 760 else "card"} (top on screen, below the top bar)', t2 is not None and TOPBAR <= t2 <= h - 160, str(t2))
    check(f'{tag} predict: Next disabled until a choice, “pick one” said', disabled(pg, '#dpo-rail [data-nav="next"]') and Z(lang, 'Pick one.', '选一个。') in pg.inner_text('#dpo-rail .step.current'))
    opts = pg.evaluate("()=>[...document.querySelectorAll('#dpo-rail [data-pred]')].map(b=>b.textContent)")
    check(f'{tag} predict: three options, each “claim: reason”, no mechanism word; the group is “Your prediction”', len(opts) == 3 and all(Z(lang, ': ', '：') in o_ for o_ in opts) and not any('logit' in o_ or Z(lang, 'although', '尽管') in o_ for o_ in opts)
          and opts[2] == Z(lang, 'They change: the four probabilities add up to 100%, so they move when A and B do', '会变：四个概率加起来总是 100%，A 和 B 一变，它们也跟着变') and pg.get_attribute('#dpo-rail .choices', 'aria-label') == Z(lang, 'Your prediction', '你的预测'), str(opts))
    pg.click('#dpo-rail [data-pred="down"]'); pg.wait_for_timeout(150)
    body = pg.inner_text('#dpo-rail .step.current')
    check(f'{tag} predict: choice pressed + “saved, can still change” + Next enabled', pg.get_attribute('#dpo-rail [data-pred="down"]', 'aria-pressed') == 'true' and not disabled(pg, '#dpo-rail [data-nav="next"]')
          and Z(lang, 'Saved. You can still change it before the next part shows the result.', '已记下。下一部分给出结果之前，还可以改。') in body and Z(lang, 'Locked', '锁定') not in body)
    check(f'{tag} predict: after answering, training opens', vis(pg, '#dpo-transport [data-act="play"]') and vis(pg, '#dpo-ctl-steps') and state(pg)['steps'] == 0)
    pg.click('#dpo-transport [data-act="step1"]'); pg.wait_for_timeout(120)
    check(f'{tag} predict: a step from the stage shows no “logit” before part 3', no_logit(pg) and state(pg)['steps'] == 1, str(pg.evaluate("()=>[...document.querySelectorAll('#dpo-bars .dpo-foot')].map(e=>e.textContent)")))
    pg.click('#dpo-transport [data-act="reset0"]'); pg.wait_for_timeout(120)
    rules(pg, f'{tag} predict')
    shot(pg, f'{tag}_2_predict')
    # part 3: take steps
    pg.click('#dpo-rail [data-nav="next"]'); pg.wait_for_timeout(200)
    check(f'{tag} steps: Next disabled, observation hidden, speed control in the card', disabled(pg, '#dpo-rail [data-nav="next"]') and not vis(pg, '#dpo-rail [data-obs]') and vis(pg, '#dpo-rail .dpo-speed')
          and pg.evaluate("()=>document.querySelector('#dpo-rail .dpo-speed [data-v=\"med\"]').getAttribute('aria-pressed')") == 'true')
    c3 = pg.inner_text('#dpo-rail .step.current .step-b')
    check(f'{tag} steps: the card defines logit and softmax once', Z(lang, 'scores each answer with a logit, and a softmax turns the four scores into probabilities that sum to 100%', '叫 logit。softmax 把四个 logit 换算成四个概率，加起来是 100%') in c3, c3[:200])
    # item 1 (the owner's review of 2026-10-09): part 3 says what DPO's loss does (A's logit up, B's down, by how far A has pulled ahead) before it names the loss; the formula waits for part 4
    i_push = c3.find(Z(lang, 'Each DPO step pushes A’s logit up and B’s down,', 'DPO 每训练一步，都把选中的回答 A 的 logit 往上推，把落选的回答 B 的 logit 往下压。'))
    i_how = c3.find(Z(lang, 'more gently the more A has gained on B since training began', '拉开得越多，推得越轻')); i_loss = c3.find(Z(lang, 'DPO loss', 'DPO 的 loss'))
    check(f'{tag} item 1: part 3 says what the loss does (A up, B down, how hard) before naming it, and points to part 4 for the formula', 0 <= i_push < i_how < i_loss and Z(lang, 'its formula is in part 4', 'loss 的公式在第 4 部分') in c3 and not maths(pg, '#dpo-rail .step.current'), c3[:400])
    check(f'{tag} steps: entering the part shows no ▲▼ badge', not any(badges(pg)), str(badges(pg)))
    h5 = pg.evaluate(BLOCKS_H5)
    check(f'{tag} H5: the stage groups the answers and the curve into labeled blocks, each set off by a 1px rule', len(h5) >= 2 and all(x.startswith('1px solid') for x in h5) and pg.inner_text('#dpo-bars-h') == Z(lang, 'How often the model gives each answer', '模型给出每个回答的概率'), str(h5))
    h1(pg, f'{tag} steps')
    pg.click('#dpo-rail [data-act="step1"]'); pg.wait_for_timeout(100)
    check(f'{tag} steps: 1 step is not enough', disabled(pg, '#dpo-rail [data-nav="next"]') and state(pg)['steps'] == 1)
    a1 = pg.inner_text('#dpo-metrics [data-k="pA"] .v')
    pg.click('#dpo-rail [data-act="step10"]'); pg.wait_for_timeout(200)
    o = txt(pg, '#dpo-rail [data-obs]'); a2 = pg.inner_text('#dpo-metrics [data-k="pA"] .v'); dpts = int(a2.rstrip('%')) - int(a1.rstrip('%'))
    check(f'{tag} steps: the learner’s own step shows ▲▼ badges computed from the shown values', badges(pg)[0].startswith('▲') and re.findall(r'\d+', badges(pg)[0])[:1] == [str(dpts)], f'{a1} → {a2}: {badges(pg)}')
    # R6.20: A's change is the result's one number; B's, C's and D's values are on the bars and the metric cards, the text says the trend
    check(f'{tag} steps: 11 steps unlock Next; the result first, with π(A) explained and B in words (its 33% on its card)', not disabled(pg, '#dpo-rail [data-nav="next"]') and Z(lang, 'After 11 steps, π(A) (how often the model gives A) rises from 18% to 34%, and π(B) falls.', '训练 11 步后，π(A)（模型给出 A 的概率）从 18% 升到 34%，π(B) 明显下降。') in o
          and pg.inner_text('#dpo-metrics [data-k="pB"] .v') == '33%' and '33%' not in o and Z(lang, 'The model now gives A most often', '现在模型最常给出 A') not in o, o[:160])
    paras = pg.evaluate("()=>[...document.querySelectorAll('#dpo-rail [data-obs] > p')].map(p=>p.textContent)")
    cd_bars = pg.evaluate("()=>[3,4].map(i=>document.querySelector(`#dpo-bars .dpo-row:nth-child(${i}) .dpo-val`).textContent)")
    check(f'{tag} steps: result → verdict → reason; the verdict says in words what C and D did (their values on the bars), the reason is two sentences and toy-limited', len(paras) == 3 and Z(lang, 'C and D moved too: both rose a little.', 'C 和 D 也变了：两个都略有上升。') in paras[1] and Z(lang, 'So far they rose instead', '反而上升了') in paras[1]
          and not re.search(r'\d', paras[1].split(Z(lang, 'You predicted', '你预测的是'))[0]) and cd_bars == [Z(lang, '20% (was 18%)', '20%（训练前 18%）'), Z(lang, '13% (was 12%)', '13%（训练前 12%）')]
          and paras[2].startswith(Z(lang, 'In this lab each step changes only A’s and B’s logits', '在这个 Lab 里，每一步只改 A 和 B 的 logit')) and len(re.findall(r'[.](\s|$)', paras[2]) if lang == 'en' else re.findall(r'。', paras[2])) == 2, json.dumps(paras, ensure_ascii=False)[:500] + str(cd_bars))
    check(f'{tag} steps: settings slider mirrors the card’s', pg.evaluate("()=>document.getElementById('dpo-ctl-steps-in').value") == '11')
    check(f'{tag} steps: C row says its logit is unchanged, values say “before training”', Z(lang, 'logit unchanged', 'logit 没变') == pg.inner_text('#dpo-bars .dpo-row:nth-child(3) .dpo-foot')
          and pg.inner_text('#dpo-bars .dpo-row:nth-child(1) .dpo-val') == Z(lang, '34% (was 18%)', '34%（训练前 18%）'), pg.inner_text('#dpo-bars .dpo-row:nth-child(1) .dpo-val'))
    if w >= 1000: check(f'{tag} steps: card control and bars on one screen', in_view(pg, '#dpo-ic-steps') and in_view(pg, '#dpo-bars .dpo-row:first-child'))
    roles = pg.evaluate("""()=>[...document.querySelectorAll('#dpo-bars .dpo-row')].map(r=>{const f=getComputedStyle(r.querySelector('.dpo-fill')); return [r.className.split(' ').pop(), f.backgroundColor, f.backgroundImage.slice(0,25)]})""")
    check(f'{tag} bars: chosen solid ink, rejected hatched, not-labeled gray (no data colors)', roles[0][0] == 'dpo-c-w' and roles[1][0] == 'dpo-c-l' and 'repeating-linear-gradient' in roles[1][2] and roles[2][0] == 'dpo-c-o' and roles[2][1] != roles[0][1], json.dumps(roles))
    rules(pg, f'{tag} steps')
    shot(pg, f'{tag}_3_steps')
    # part 4: why no reward model
    pg.click('#dpo-rail [data-nav="next"]'); pg.wait_for_timeout(250)
    b4 = pg.inner_text('#dpo-rail .step.current'); o4 = pg.inner_text('#dpo-rail [data-obs]')
    check(f'{tag} loss: Next shows no ▲▼ badge (the part sets 40 steps itself)', not any(badges(pg)), str(badges(pg)))
    check(f'{tag} loss: the card says what the implicit reward replaces, in three paragraphs', Z(lang, 'The model scores answers itself', '打分由模型自己给') in b4 and pg.evaluate("()=>document.querySelectorAll('#dpo-rail .step.current .step-b > p').length") == 3, b4[:200])
    m4 = maths(pg, '#dpo-rail .step.current')
    check(f'{tag} loss: card defines the implicit reward, ln, σ and the reward margin, the three formulas typeset on their own lines with their names', 'r^=βlogππref' in m4 and 'L=−logσ(r^A−r^B)' in m4 and 'σ(x)=11+e−x' in m4 and Z(lang, 'natural log', '自然对数') in b4 and Z(lang, 'is the reward margin', '叫奖励差') in b4
          and pg.evaluate("()=>[...document.querySelectorAll('#dpo-rail .step.current .dpo-fx')].map(e=>(e.querySelector('.dpo-fxn')||{}).textContent)") == [Z(lang, 'Implicit reward', '隐式奖励'), Z(lang, 'DPO loss', 'DPO 的 loss'), Z(lang, 'where', '其中')], str(m4)[:300])
    check(f'{tag} loss: no “Where these numbers come from” box; the numbers stay in Show the math and on the metric cards', not vis(pg, '#dpo-rail [data-obs]') and Z(lang, 'Where these numbers come from', '这些数怎么来的') not in pg.inner_text('#dpo-rail'), o4[:200])
    pg.locator('#dpo-rail .step.current [data-jump="math"]').first.click(); pg.wait_for_timeout(500)
    mb = pg.inner_text('#dpo-mathbody')
    check(f'{tag} loss: the link opens Show the math, which defines π, Z and Bradley–Terry', pg.evaluate("()=>document.getElementById('dpo-math').open") and Z(lang, 'how often the model gives answer y', '模型给出回答 y 的概率') in mb and 'Z' in mb and 'Bradley–Terry' in mb and Z(lang, 'log-ratio gap h', 'log 比值差 h') not in mb, mb[:200])
    mp = pg.evaluate(MATHPANEL)
    check(f'{tag} item 4 / H2: Show the math is short labeled blocks (log-ratio, implicit reward, reward margin, loss, push weight, next step, KL, derivation), each with a lead line and a typeset formula, rows aligned at “=”, no run-on line, no loose paragraph, nothing to scroll', mp['ks'] == ['lr', 'r', 'm', 'loss', 'pw', 'next', 'kl', 'derive'] and mp['lead'] and mp['rule'] and not mp['runon'] and mp['loose'] == 0 and not mp['over'] and not mp['h'], json.dumps(mp, ensure_ascii=False)[:400])
    h1(pg, f'{tag} loss with Show the math open')
    tip = hover_tip(pg, '#dpo-mathbody .t[data-ctl="beta"]')
    check(f'{tag} loss: term tooltip says what it is and where a click goes', Z(lang, 'In this lab, the larger β, the less the model changes in the same number of steps', '在这个 Lab 里，β 越大，同样的训练步数下模型改动得越少') in tip and Z(lang, 'Click to jump', '点击跳到') in tip, tip.replace('\n', ' | '))
    pg.locator('#dpo-mathbody .t[data-ctl="beta"]').first.click(); pg.wait_for_timeout(700)
    check(f'{tag} loss: term click jumps to the β control', pg.evaluate("()=>!!document.activeElement.closest('#dpo-ctl-beta')") and pg.evaluate("()=>document.getElementById('dpo-ctl-beta').classList.contains('flash')"))
    pg.focus('#dpo-mathbody .t[data-ctl="steps"]'); pg.keyboard.press('Enter'); pg.wait_for_timeout(600)
    check(f'{tag} loss: Enter on a term jumps (keyboard)', pg.evaluate("()=>document.activeElement.id") == 'dpo-ctl-steps-in')
    rules(pg, f'{tag} loss')
    pg.evaluate("()=>document.querySelector('#dpo-rail .step.current').scrollIntoView({block:'start'})"); pg.wait_for_timeout(200)
    shot(pg, f'{tag}_4_loss')
    # Back to part 3 at 0 steps: the observation is gated
    pg.click('#dpo-rail [data-nav="back"]'); pg.wait_for_timeout(250)
    o = txt(pg, '#dpo-rail [data-obs]')
    check(f'{tag} Back to part 3 at 0 steps: no contradictory “changed anyway”', Z(lang, 'Step 0: nothing trained yet.', '第 0 步：还没训练。') in o and Z(lang, 'changed anyway', '还是变了') not in o, o[:200])
    tb = top_shown(pg, '#dpo-rail .pbar' if w <= 760 else '#dpo-rail .step.current')   # phones (m2): the part bar
    check(f'{tag} Back reveals the {"part bar" if w <= 760 else "card"}', tb is not None and TOPBAR <= tb <= h - 160, str(tb))
    go_step(pg, PART['loss'])
    pg.evaluate("()=>{document.getElementById('dpo-math').open=true; const s=document.getElementById('dpo-stage'); s.scrollTop=s.scrollHeight}"); pg.wait_for_timeout(100)
    # part 5: the push fades
    pg.click('#dpo-rail [data-nav="next"]'); pg.wait_for_timeout(300)
    check(f'{tag} fade: stage scroll reset when the part starts', pg.evaluate("()=>document.getElementById('dpo-stage').scrollTop") == 0)
    check(f'{tag} fade: β 0.1, push-weight view, Next disabled, no ▲▼ badge on entering', state(pg)['beta'] == 0.1 and state(pg)['view'] == 'wt' and disabled(pg, '#dpo-rail [data-nav="next"]') and not any(badges(pg)), str(badges(pg)))
    check(f'{tag} fade: the card says what each step pushes; the chart title asks how far', Z(lang, 'Each step pushes A’s logit up and B’s down.', '每一步都把 A 的 logit 往上推、把 B 的往下推。') in pg.inner_text('#dpo-rail .step.current') and pg.inner_text('#dpo-curve-title') == Z(lang, 'How far does each step still push A and B? (push weight)', '每一步还把 A、B 推多少？（推力权重）'), pg.inner_text('#dpo-curve-title'))
    set_steps(pg, 30)
    w30 = calc(pg, {'steps': 30})['w']
    live = pg.inner_text('#dpo-pw-live')
    check(f'{tag} fade: push weight shown live in the card before the goal is reached', live == Z(lang, f'At step 30 the push weight is {w30:.3f} (0.500 at step 0).', f'第 30 步的推力权重是 {w30:.3f}（第 0 步是 0.500）。'), live)
    set_steps(pg, 70)
    w70 = calc(pg, {'steps': 70})['w']; rel = round(float(f'{w70:.3f}') / 0.5 * 100)
    o = txt(pg, '#dpo-rail [data-obs]')
    check(f'{tag} fade: observation gives the share of the first step’s push, without repeating the live weight above it', Z(lang, f'By step 70, each step moves the logits only {rel}% as far as the first step did.', f'到第 70 步，每一步对 logit 的改动只有第一步的 {rel}%。') in o and f'{w70:.3f}' not in o and not disabled(pg, '#dpo-rail [data-nav="next"]'), o[:200])
    curve_checks(pg, tag, 'push-weight view')
    h7(pg, f'{tag} fade'); h1(pg, f'{tag} fade')
    rules(pg, f'{tag} fade')
    shot(pg, f'{tag}_5_fade')
    if w < 700:
        pg.evaluate("()=>document.querySelector('#dpo-rail .step.current').scrollIntoView({block:'start'})"); pg.evaluate("()=>window.scrollBy(0,-60)"); pg.wait_for_timeout(500)   # phones (m2): the card is under the stage
        dk = pg.evaluate("()=>({shown:!document.getElementById('dock').hidden, text:document.getElementById('dock').innerText})")
        check(f'{tag} fade: the mini result bar shows the push weight in a sentence', dk['shown'] and Z(lang, f'At step 70 the push weight is {w70:.3f}', f'第 70 步的推力权重是 {w70:.3f}') in dk['text'], json.dumps(dk, ensure_ascii=False))
        shot(pg, f'{tag}_5b_fade_dock')
    # part 6: β
    pg.click('#dpo-rail [data-nav="next"]'); pg.wait_for_timeout(250)
    check(f'{tag} beta: entered at 40 steps, KL view, Next disabled', state(pg)['steps'] == 40 and disabled(pg, '#dpo-rail [data-nav="next"]') and state(pg)['view'] == 'kl')
    b6 = pg.inner_text('#dpo-rail .step.current')
    BL = Z(lang, 'β (the weight of the KL penalty in the RLHF objective)', 'β（RLHF 目标里 KL 惩罚的系数）')
    labs_ = pg.evaluate("()=>['#dpo-ic-beta','#dpo-ctl-beta'].map(s=>{const g=document.querySelector(s); const c=g&&g.closest('.ctl,.inline-ctl'); const l=c&&c.querySelector('label,.lbl,.ctl-l,.k'); return l?l.textContent.trim():(c?c.textContent.trim().slice(0,70):'')})")
    check(f'{tag} beta: the β control in the card and under Settings says what β is', all(x.startswith(BL) for x in labs_) and Z(lang, 'pull back', '拉回') not in pg.inner_text('#dpo-app'), str(labs_))
    check(f'{tag} beta: the card states the experiment, KL in nats, and this lab’s equal first step, without η/β', Z(lang, 'Try a few values of β. Each one trains from the same SFT model for the same number of steps.', '换几个 β 试试。每个 β 都从同一个 SFT 模型开始，训练同样的步数。') in b6
          and Z(lang, 'In this lab the first step moves the logits by the same amount for every β.', '在这个 Lab 里，不管 β 取多少，第一步对 logit 的改动都一样大。') in b6 and Z(lang, 'in nats', '单位 nats') in b6 and 'η' not in b6, b6[:400])
    pg.click('#dpo-ic-beta button[data-v="1"]'); pg.wait_for_timeout(200)
    o = txt(pg, '#dpo-rail [data-obs]')
    # R6.20: the KL against the initial setup's is the one key number (a pair, 0.31 on the KL card); π(A) and the push weights in words
    check(f'{tag} beta: observation against β = 0.1', Z(lang, 'β = 1, same 40 steps: the model changes less than with β = 0.1, the initial setup. Its KL from the SFT model is 0.31 nats instead of 0.85, and π(A) rises less.', 'β = 1，同样训练 40 步：模型改动得比初始设置 β = 0.1 时更少。它离 SFT 模型的 KL 是 0.31 nats，初始设置时是 0.85 nats；π(A) 也涨得更少。') in o
          and pg.inner_text('#dpo-metrics [data-k="kl"] .v') == '0.31' and '49%' not in o and not disabled(pg, '#dpo-rail [data-nav="next"]') and Z(lang, 'The β values you tried', '已试过的 β') not in o, o[:260])
    rules(pg, f'{tag} beta, first β tried')
    # the push weights at step 40 left the sentence (R6.20): the reason names the push-weight curve, and Show the math gives this β's weight
    mt = pg.evaluate("()=>document.getElementById('dpo-mathbody').textContent")
    check(f'{tag} beta: the reason points to the push-weight curve instead of listing push weights; Show the math has β = 1’s push weight at step 40 (0.122)', Z(lang, 'and the push weight small (compare the push-weight curve).', '推力权重就掉下来（对比推力权重曲线）。') in o and '0.122' not in o and '0.410' not in o
          and pg.query_selector('#dpo-rail [data-obs] [data-jump="curve-wt"]') is not None and pg.evaluate("()=>{const b=document.querySelector('#dpo-mathbody [data-k=\"pw\"]'); const c=b?[...b.querySelectorAll('.dpo-eq > [data-r=\"2\"]')].map(e=>e.textContent.trim()).filter(Boolean):[]; return c.join(' ')}") == '= 0.122', o[200:700])
    check(f'{tag} beta: the dotted line of the previous β is named in the key', Z(lang, 'the previous setting (β = 0.1)', '上一个设置（β = 0.1）') in pg.inner_text('#dpo-curve-key'), pg.inner_text('#dpo-curve-key'))
    curve_checks(pg, tag, 'KL view with previous run')
    pg.click('#dpo-ic-beta button[data-v="2"]'); pg.wait_for_timeout(200)
    o2 = txt(pg, '#dpo-rail [data-obs]')
    check(f'{tag} beta: the first line compares with the initial setup once (0.15 on the KL card)', o2.count('0.85') == 1 and Z(lang, 'β = 2, same 40 steps: the model changes less than with β = 0.1, the initial setup. Its KL from the SFT model is 0.15 nats instead of 0.85, and π(A) rises less.', 'β = 2，同样训练 40 步：模型改动得比初始设置 β = 0.1 时更少。它离 SFT 模型的 KL 是 0.15 nats，初始设置时是 0.85 nats；π(A) 也涨得更少。') in o2
          and pg.inner_text('#dpo-metrics [data-k="kl"] .v') == '0.15', o2[:260])
    # the tried values as a trend in words (R6.20), which the engine's numbers must bear out: KL and π(A) both fall as β grows
    tq = [calc(pg, {'steps': 40, 'beta': bv}) for bv in (0.1, 1, 2)]
    check(f'{tag} beta: each click changes the first sentence; from the third β the tried values as a trend, true of the engine’s numbers', o2 != o and Z(lang, 'The β values you tried, each after 40 steps: the larger β, the lower the KL and the lower π(A).', '已试过的 β（都训练 40 步）：β 越大，KL 越小，π(A) 也越低。') in o2
          and tq[0]['kl'] > tq[1]['kl'] > tq[2]['kl'] and tq[0]['pi'][0] > tq[1]['pi'][0] > tq[2]['pi'][0], o2[-300:])
    check(f'{tag} beta: the “larger β” sentence reads in order and is marked as this lab’s', Z(lang, 'in this lab, a larger β makes the push fade earlier, so in the same number of steps the model changes less and stays closer to the SFT model', '在这个 Lab 里，β 越大，推力衰减得越早，同样的步数下模型改动得越少，离 SFT 模型越近') in o2, o2[-400:])
    check(f'{tag} beta: settings β mirrors the card’s', pg.get_attribute('#dpo-ctl-beta button[data-v="2"]', 'aria-pressed') == 'true')
    set_steps(pg, 0, 'dpo-ctl-steps-in')
    o0 = txt(pg, '#dpo-rail [data-obs]')
    check(f'{tag} beta: at 0 steps the observation says nothing is trained yet', Z(lang, 'Step 0: nothing trained yet', '第 0 步：还没训练') in o0 and 'KL 0.00' not in o0, o0[:160])
    set_steps(pg, 40, 'dpo-ctl-steps-in')
    rules(pg, f'{tag} beta')
    shot(pg, f'{tag}_6_beta')
    # part 7: shared weight
    pg.click('#dpo-rail [data-nav="next"]'); pg.wait_for_timeout(250)
    check(f'{tag} shared: Next disabled until switched; the title asks instead of telling; the detail level is explained', disabled(pg, '#dpo-rail [data-nav="next"]') and pg.inner_text('#dpo-rail .step.current .step-h') .endswith(Z(lang, 'With shared weights, does the chosen answer always rise?', '共享权重时，选中的回答一定会上升吗？'))
          and Z(lang, 'Each answer has a detail level, how much technical detail it gives', '每个回答有一个细节程度，表示它讲了多少技术细节') in pg.inner_text('#dpo-rail .step.current'), pg.inner_text('#dpo-rail .step.current .step-h'))
    c7 = pg.inner_text('#dpo-rail .step.current'); wr = pg.evaluate(WIRE)
    check(f'{tag} item 2: part 7 says the lab so far assumed independent logits, that a real model’s answers are linked, and shows the four logits as unconnected boxes', Z(lang, 'So far this lab has assumed that each answer’s logit is independent', '到目前为止，这个 Lab 假设每个回答的 logit 都是独立的') in c7
          and Z(lang, 'so the answers are linked', '回答之间相互关联') in c7 and Z(lang, 'share words or structure with it', '和它用词或结构相似的回答也会跟着变') in c7 and 'logit=' in ''.join(maths(pg, '#dpo-rail .step.current'))
          and wr['shown'] and wr['label'] == Z(lang, 'Where the logits come from', 'logit 从哪来') and not wr['lines'] and wr['inside'] and Z(lang, 'the four logits are not connected', '四个 logit 互不相连') in wr['cap'], json.dumps(wr, ensure_ascii=False)[:300])
    pg.click('#dpo-ic-mode button[data-v="shared"]'); pg.wait_for_timeout(200)
    wr = pg.evaluate(WIRE)
    check(f'{tag} item 2 / H6: with the shared weight the picture draws one w feeding all four logits, each line as thick as the answer’s detail level (D thickest, B thinnest), the caption says what the thickness means, and the svg holds no text', wr['shown'] and len(wr['lines']) == 4
          and sorted(range(4), key=lambda i: wr['lines'][i]) == [1, 2, 0, 3] and Z(lang, 'the thicker the line, the more detail', '线越粗，细节越多') in wr['cap'] and wr['svgText'] == 0 and wr['inside']
          and Z(lang, '+2.290', '+2.290') in wr['cells'][0] and '+4.580' in wr['cells'][3], json.dumps(wr, ensure_ascii=False)[:400])
    h7(pg, f'{tag} shared'); h1(pg, f'{tag} shared')
    o = txt(pg, '#dpo-rail [data-obs]')
    # R6.20: A's change is the one number; the peak, D's rise and the loss in words, their values on the stage (bars, loss card)
    feet = pg.evaluate("()=>[...document.querySelectorAll('#dpo-bars .dpo-row .dpo-foot')].map(e=>e.textContent)")
    check(f'{tag} shared: chosen answer falls, full reason in words, loss still falls, real runs named without the paper (it is in Fidelity); the numbers on the stage', Z(lang, 'π(A) falls from 18% to 12%, below where it started', 'π(A) 从 18% 降到 12%，比训练前还低') in o and Z(lang, 'The loss still fell: it sees only the reward margin between A and B', 'loss 照样下降：它只看 A 和 B 之间的奖励差') in o
          and Z(lang, 'A rose at first, then turned and fell.', 'A 先升后降。') in o and Z(lang, 'and that lifts every logit by its detail level: D’s logit rises twice as fast as A’s, so D took most of the probability', '而这会按细节程度抬高所有 logit：D 的 logit 涨幅是 A 的两倍，所以 D 拿走了大部分概率') in o
          and not any(x in o for x in ('0.606', '21.4%', '2.290', '4.580', '79%')) and pg.inner_text('#dpo-metrics [data-k="loss"] .v') == '0.606'
          and Z(lang, 'logit up 2.290', 'logit 升了 2.290') in feet[0] and Z(lang, 'logit up 4.580', 'logit 升了 4.580') in feet[3]
          and Z(lang, 'Real DPO runs show it too', '真实的 DPO 训练里也有这种现象') in o and 'Razin' not in o, o[:300] + str(feet))
    check(f'{tag} shared: the comparison with separate logits is in words, and the Chinese one has no colon', Z(lang, '(with separate logits it ends up far higher)', '（各自的 logit 时要高得多）') in o and '88%' not in o, o[:160])
    check(f'{tag} shared: status turns red, most often D', pg.inner_text('#dpo-status') == Z(lang, 'The model most often gives D (79%)', '模型最常给出 D（79%）') and 'bad' in pg.get_attribute('#dpo-status', 'class'), pg.inner_text('#dpo-status'))
    rules(pg, f'{tag} shared')
    shot(pg, f'{tag}_7_shared')
    set_steps(pg, 0, 'dpo-ctl-steps-in')
    o = txt(pg, '#dpo-rail [data-obs]')
    check(f'{tag} shared at 0 steps: gated, no “still rising / loss still fell”', Z(lang, 'Step 0: nothing trained yet.', '第 0 步：还没训练。') in o and '0.693' not in o, o[:160])
    # r12: the stage shows the rises (w × detail level, from the w of Show the math) and the observation says the same in words ("twice"),
    # and 1 step never says "stays at"
    set_steps(pg, 10, 'dpo-ctl-steps-in')
    o = txt(pg, '#dpo-rail [data-obs]'); footA = pg.inner_text('#dpo-bars .dpo-row[data-i="0"] .dpo-foot'); foot = pg.inner_text('#dpo-bars .dpo-row[data-i="3"] .dpo-foot')
    check(f'{tag} shared at 10 steps: the stage gives A’s and D’s rises from the w of Show the math (0.397 × 2 = 0.794); the observation says “twice” without the numbers', Z(lang, 'logit up 0.397', 'logit 升了 0.397') in footA and Z(lang, 'logit up 0.794', 'logit 升了 0.794') in foot
          and Z(lang, 'D’s logit rises twice as fast as A’s', 'D 的 logit 涨幅是 A 的两倍') in o and '0.794' not in o and '0.397' not in o, (o[:260], footA, foot))
    set_steps(pg, 1, 'dpo-ctl-steps-in')
    o = txt(pg, '#dpo-rail [data-obs]')
    check(f'{tag} shared at 1 step: π(A) rises with one decimal, never “stays at” next to “still rising”', Z(lang, 'π(A) rises from 18.0% to 18.3%', 'π(A) 从 18.0% 升到 18.3%') in o and Z(lang, 'A is still rising', 'A 还在上升') in o and Z(lang, 'stays at', '保持') not in o, o[:220])
    check(f'{tag} shared at 1 step: before A turns, D “gains faster” (not “took”), and at most one rise/rose per EN sentence', Z(lang, 'so D gains faster than A and holds A’s gain down', '所以 D 涨得比 A 快，把 A 的涨幅压了下去') in o
          and (lang != 'en' or all(len(re.findall(r'\b(?:rose|rises|rise)\b', s_)) <= 1 for s_ in re.split(r'(?<=[.!?])\s+', o))), o[:400])
    set_steps(pg, 60, 'dpo-ctl-steps-in')
    # part 8: remember, into the challenge
    pg.click('#dpo-rail [data-nav="next"]'); pg.wait_for_timeout(250)
    body = pg.inner_text('#dpo-rail .step.current')
    nl = pg.inner_text('#dpo-rail .next-lab') if pg.query_selector('#dpo-rail .next-lab') else ''
    # R6.20: the summary keeps the mechanism and its one number ("2 for every 1", review 7); the push weights after 60 steps are in words,
    # which the engine's weights must bear out (0.373 is most of the first 0.5, 0.043 almost nothing), and π(A)'s fall is in words
    w60, w60b2, sh60 = calc(pg, {'steps': 60})['w'], calc(pg, {'steps': 60, 'beta': 2})['w'], calc(pg, {'steps': 60, 'mode': 'shared'})
    check(f'{tag} remember: misconceptions, lab-limited claim, the mechanism with its number, the rest in words, next lab below the card', Z(lang, 'β is the learning rate', 'β 就是学习率') in body and Z(lang, 'π(A), the chosen answer’s probability, even ends up below where it started', '选中的回答 A，概率反而降到了训练前以下') in body and sh60['pi'][0] < 0.18
          and Z(lang, 'after 60 steps the push still has most of its strength with β = 0.1 but has almost faded with β = 2', '训练 60 步后，β = 0.1 的推力还剩大半，β = 2 的推力几乎消失了') in body and round(w60, 3) / 0.5 >= 0.5 and round(w60b2, 3) / 0.5 < 0.2
          and not any(x in body for x in ('0.373', '0.043', '4.58', '2.29', '12%'))
          and Z(lang, 'In this lab, with separate logits,', '在这个 Lab 里，用各自的 logit 时') in body.replace('\n','') and bool(nl)
          and Z(lang, 'common misconceptions', '常见误解') in body.lower() and Z(lang, 'In this lab, β sets how early the push fades, not how big the first step is', '在这个 Lab 里，β 决定推力多早衰减，而不是第一步推多少') in body
          and Z(lang, 'its loss has no SFT cross-entropy term', 'loss 里也没有 SFT 的交叉熵') in body and Z(lang, 'yet its logit rises the most: every step raises w, and w lifts D’s logit 2 for every 1 it lifts A’s', 'logit 却涨得最多：每一步都在调大 w，而 w 每让 A 的 logit 升 1，就让 D 的升 2') in body, body[:300])
    # the two mechanisms are defined on hover (review 7): D is not trained, softmax is not a second loss
    tips = pg.evaluate("()=>[...document.querySelectorAll('#dpo-rail .step.current .def')].map(e=>e.dataset.tip?ASP.t(e.dataset.tip).replace(/<math[\\s\\S]*?<\\/math>/g,' ').replace(/<[^>]+>/g,''):e.dataset.tipx)")
    check(f'{tag} remember: separate logits and shared weight carry hover definitions', len(tips) == 2 and Z(lang, 'only the logits of the two answers in the pair', '只改偏好对里 A 和 B 这两个回答的 logit') in tips[0]
          and Z(lang, 'D never enters the loss', 'D 从没进过 loss') in tips[1] and Z(lang, 'reward margin', '奖励差') in tips[1], str(tips)[:300])
    h1(pg, f'{tag} remember')
    rules(pg, f'{tag} remember')
    shot(pg, f'{tag}_8_remember')
    # R3.12: guided ends in Scenarios, whose card ends with the way to the challenge
    pg.click('#dpo-rail [data-mode-go="sandbox"]'); pg.wait_for_timeout(450)
    check(f'{tag} guided ends in Scenarios, and the scenarios card ends with “Take the challenge”', pg.get_attribute('#view-dyn .modes button[data-mode="sandbox"]:not([data-flow=free])', 'aria-pressed') == 'true' and pg.query_selector('#dpo-rail [data-mode-go="challenge"]') is not None)
    pg.click('#dpo-rail [data-mode-go="challenge"]'); pg.wait_for_timeout(450)
    tc = top_shown(pg, '#dpo-stage' if w <= 760 else '#dpo-rail .railcard')   # phones (m2): the stage first, the card right under it
    check(f'{tag} “Take the challenge” reveals the challenge {"stage" if w <= 760 else "card (goal and β in view)"}', tc is not None and TOPBAR <= tc <= h - 160 and (w <= 760 or in_view(pg, '#dpo-rail .rh')), str(tc))
    # challenge
    check(f'{tag} challenge: mode pressed and goal written in full', pg.get_attribute('#dpo-app .modes button[data-mode="challenge"]', 'aria-pressed') == 'true' and pg.inner_text('#dpo-rail .rh') == Z(lang, 'After 60 DPO steps, make the model give A at least half the time, with its KL from the SFT model at most 0.8 nats', '训练 60 步 DPO 后，让模型至少一半的时候给出 A，且离 SFT 模型的 KL 不超过 0.8 nats'), pg.inner_text('#dpo-rail .rh'))
    check(f'{tag} challenge: the KL note has no second KL for β = 0.1; no ▲▼ badge on entering', Z(lang, 'KL measures how far the model has moved from the SFT model, in nats. The 50% and 0.8-nat thresholds are set for this challenge.', 'KL 衡量模型离 SFT 模型有多远，单位 nats。50% 和 0.8 nats 是这道题定的门槛。') in pg.inner_text('#dpo-rail .railcard')
          and Z(lang, 'For scale', '作为参照') not in pg.inner_text('#dpo-rail .railcard') and not any(badges(pg)), str(badges(pg)))
    check(f'{tag} challenge: locked controls hidden, note says what is fixed, β kept', not vis(pg, '#dpo-ctl-ds') and not vis(pg, '#dpo-ctl-steps') and not vis(pg, '#dpo-adv') and vis(pg, '#dpo-lock') and vis(pg, '#dpo-ctl-beta') and vis(pg, '#dpo-ic-beta')
          and Z(lang, 'In the challenge you can change only β.', '挑战中只能改 β。') in pg.inner_text('#dpo-lock'))
    lc = pg.evaluate("()=>{const g=document.querySelector('#dpo-ic-beta'); const c=g&&g.closest('.ctl,.inline-ctl'); const l=c&&c.querySelector('label,.lbl,.ctl-l,.k'); return l?l.textContent.trim():(c?c.textContent.trim().slice(0,70):'')}")
    check(f'{tag} challenge: the β control says what β is', lc.startswith(Z(lang, 'β (the weight of the KL penalty in the RLHF objective)', 'β（RLHF 目标里 KL 惩罚的系数）')), lc)
    check(f'{tag} challenge: no grayed-out controls', pg.evaluate("()=>[...document.querySelectorAll('#dpo-app .seg-ctl button')].filter(b=>b.offsetParent).every(b=>!b.disabled)"))
    check(f'{tag} challenge: transport hidden, no sentence about the curve or the run (R3.4)', not vis(pg, '#dpo-transport [data-act="step1"]') and not vis(pg, '#dpo-transport .dpo-speed') and pg.inner_text('#dpo-tnote') == '', pg.inner_text('#dpo-tnote'))
    check(f'{tag} challenge: live progress at start (β 0.1)', pg.inner_text('#dpo-ch-av') == '88%' and pg.inner_text('#dpo-ch-kv') == '1.23' and 'dpo-over' in pg.get_attribute('#dpo-ch-k', 'class'))
    stt = pg.inner_text('#dpo-status')
    check(f'{tag} challenge: status badge names the failing condition in words', stt == Z(lang, 'KL is above 0.8 nats', 'KL 超过 0.8 nats'), stt)
    if w < 700:
        pg.evaluate("()=>{document.querySelector('#dpo-rail .railcard').scrollIntoView({block:'start'}); window.scrollBy(0,-120)}"); pg.wait_for_timeout(500)
        dk = pg.evaluate("()=>({shown:!document.getElementById('dock').hidden, text:document.getElementById('dock').innerText.replace(/\\s+/g,' ').trim(), fits:[...document.querySelectorAll('#dock .dock-in > *')].filter(e=>getComputedStyle(e).display!=='none').every(e=>e.getBoundingClientRect().right<=innerWidth+0.5)})")
        check(f'{tag} challenge: the mini result bar says “not yet” and why, in a sentence that fits', dk['shown'] and dk['fits'] and Z(lang, 'Not yet', '未达标') in dk['text'] and Z(lang, 'The KL from the SFT model is 1.23 nats, over the 0.8-nat limit', '离 SFT 模型的 KL 是 1.23 nats，超过上限 0.8 nats') in dk['text'], json.dumps(dk, ensure_ascii=False))
        shot(pg, f'{tag}_9b_challenge_dock')
    pg.click('#dpo-rail [data-act="submit"]'); pg.wait_for_timeout(200)
    r = txt(pg, '#dpo-ch-res')
    check(f'{tag} challenge tier 3 (KL over): hint, no chart yet, no sentence about the page updating', Z(lang, 'Not yet: the KL is 1.23 nats, over the 0.8-nat limit', '还没达到：KL 是 1.23 nats，超过了 0.8 nats 的上限') in r and not vis(pg, '#dpo-ch-res .cmp') and vis(pg, '#dpo-ch-res [data-act="reveal"]') and Z(lang, 'updates', '随之更新') not in r, r[:160])
    pg.click('#dpo-ic-beta button[data-v="2"]'); pg.wait_for_timeout(200)
    r = txt(pg, '#dpo-ch-res')
    check(f'{tag} challenge tier 3 (A short): live after submit', Z(lang, 'Not yet: the model gives A only 42% of the time', '还没达到：模型给出 A 的概率只有 42%') in r, r[:160])
    rv = pg.inner_text('#dpo-ch-res [data-act="reveal"]').strip()
    check(f'{tag} challenge: the reveal button says “{Z(lang, "Show the answer", "直接看答案")}” (the owner, batch 8)', rv == Z(lang, 'Show the answer', '直接看答案'), rv)
    pg.click('#dpo-ch-res [data-act="reveal"]'); pg.wait_for_timeout(200)
    BRIEF = """()=>{const res=document.querySelector('#dpo-ch-res'), c=res.querySelector('.cmp'); const rs=c?[...c.querySelectorAll('.cmp-row')]:[], tg=r=>!!r.querySelector('.cmp-tag,.cmp-you');
      return {n:rs.length, stray:rs.filter(r=>!r.classList.contains('ok')&&!r.classList.contains('me')&&!tg(r)).length, labs:rs.map(r=>r.querySelector('.cmp-lab').textContent),
        title:c?c.querySelector('.cmp-title').textContent:'', note:(res.querySelector('.cmp-note')||{}).textContent||''};}"""
    bf = pg.evaluate(BRIEF)
    check(f'{tag} challenge: Show the answer → the baseline, the initial β, the two passing β values and your pick (β = 2): 5 rows, none stray, no note',
          bf['n'] == 5 and bf['stray'] == 0 and not bf['note'] and vis(pg, '#dpo-ch-res .cmp-you') and vis(pg, '#dpo-ch-res .cmp-tag')
          and Z(lang, 'baseline', '基准') in bf['labs'][0] and Z(lang, 'initial setup', '初始设置') in bf['labs'][1], bf)
    check(f'{tag} challenge: the chart title no longer says “for each β”', bf['title'] and not any(x in bf['title'] for x in ('each', 'every', '各个', '每个')), bf['title'])
    leg = pg.inner_text('#dpo-ch-res .cmp')
    check(f'{tag} challenge: chart legend says what the green bars mean', Z(lang, 'meets both conditions', '两个条件都满足') in leg and Z(lang, 'misses at least one', '至少一个不满足') in leg, leg[-160:].replace('\n', ' | '))
    okrows = pg.evaluate("()=>[...document.querySelectorAll('#dpo-ch-res .cmp-row')].map(r=>r.classList.contains('ok'))")
    check(f'{tag} challenge: the rows that meet both conditions are marked (β 0.5 and 1)', okrows == [False, False, True, True, False], str(okrows))
    pg.click('#dpo-ic-beta button[data-v="0.5"]'); pg.wait_for_timeout(200)
    r = txt(pg, '#dpo-ch-res')
    # R4.4: the closer option is named; its numbers are in the chart right under the verdict (R6.20)
    check(f'{tag} challenge tier 2 (met, closer option exists)', Z(lang, 'Goal met: the model gives A 71% of the time, with KL 0.76 nats.', '达成目标：模型给出 A 的概率是 71%，KL 0.76 nats。') in r and Z(lang, 'You could stay closer to the SFT model: β = 1 also meets both conditions, with a smaller KL.', '其实还可以离 SFT 模型更近：β = 1 也满足两个条件，而且 KL 更小。') in r
          and vis(pg, '#dpo-ch-res .cmp') and Z(lang, 'β = 1, KL 0.44 nats', 'β = 1，KL 0.44 nats') in pg.inner_text('#dpo-ch-res .cmp'), r[:240])
    bf = pg.evaluate(BRIEF)
    check(f'{tag} challenge tier 2 (β = 0.5): the chart drops β = 2, which misses; the note counts it', bf['n'] == 4 and bf['stray'] == 0 and not any('β = 2' in x for x in bf['labs'])
          and Z(lang, 'The other β value misses at least one condition', '其余 1 个 β 至少有一个条件不满足') in bf['note'], bf)
    pg.click('#dpo-ic-beta button[data-v="1"]'); pg.wait_for_timeout(200)
    r = txt(pg, '#dpo-ch-res')
    check(f'{tag} challenge tier 1 (met and closest) + reason and caveat naming the planned lab; no list repeating the chart', Z(lang, 'Goal met: after 60 steps the model gives A 56% of the time, with KL 0.44 nats.', '达成目标：训练 60 步后，模型给出 A 的概率是 56%，KL 0.44 nats。') in r and Z(lang, 'closest to the SFT model', '离 SFT 模型最近') in r
          and Z(lang, 'After 60 steps, β = 0.1 gives', '训练 60 步后：β = 0.1 时') not in r and Z(lang, 'held-out preference pairs', '留出的偏好对') in r and 'rewards/margins' not in r and Z(lang, 'planned', '规划中') not in r and pg.query_selector('#dpo-ch-res a[href="#eval-ci"]') is not None and Z(lang, 'error bars', '误差范围') in r, r[:240])
    h1(pg, f'{tag} challenge'); h7(pg, f'{tag} challenge')
    check(f'{tag} challenge: status green, badge says both conditions are met', 'ok' in pg.get_attribute('#dpo-status', 'class') and pg.inner_text('#dpo-status') == Z(lang, 'Both conditions are met', '两个条件都满足'), pg.inner_text('#dpo-status'))
    check(f'{tag} challenge: the “Why” sentence reads in order and states this lab’s equal first step in plain words', Z(lang, 'A larger β makes the push fade earlier and the model change less', 'β 越大，推力衰减得越早，模型改动得越少') in r
          and Z(lang, 'Why: in this lab the first step moves the logits by the same amount for every β', '原因：在这个 Lab 里，不管 β 取多少，第一步对 logit 的改动都一样大') in r and 'η' not in r, r[-500:])
    tip = hover_tip(pg, '#dpo-app .legend-row [data-tip="dpo_lgW"]')
    check(f'{tag} challenge: locked legend jump says the fixed value', Z(lang, 'Fixed during the challenge', '挑战中固定不变') in tip, tip.replace('\n', ' | '))
    rules(pg, f'{tag} challenge')
    pg.locator('#dpo-rail .railcard:not(.exits)').screenshot(path=str(SH / f'{tag}_9_challenge_card.png'))
    shot(pg, f'{tag}_9_challenge')
    check(f'{tag} overflow, challenge', overflow(pg) <= 0, str(overflow(pg)))
    # language switch keeps mode, β and the submitted result
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(250)
    s = state(pg); r = txt(pg, '#dpo-ch-res')
    check(f'{tag} language switch keeps challenge state', s['mode'] == 'challenge' and s['beta'] == 1 and Z(other, 'Goal met: after 60 steps the model gives A 56%', '达成目标：训练 60 步后，模型给出 A 的概率是 56%') in r and vis(pg, '#dpo-ch-res .cmp'), r[:80])
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(250)
    # scenarios: presets follow the settings
    pg.click('#dpo-app .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(350)
    tr = top_shown(pg, '#dpo-stage' if w <= 760 else '#dpo-rail')   # phones (m2): the stage
    check(f'{tag} mode switch reveals the {"stage" if w <= 760 else "rail"}', tr is not None and tr >= TOPBAR - 1 and tr <= h - 160, str(tr))
    check(f'{tag} scenarios card is titled “Pick a scenario”; its only intro line is the lead-in (G1: what to pick and what to watch, not “one at a time”)', pg.inner_text('#dpo-rail .rh') == Z(lang, 'Pick a scenario', '选一个应用场景') and pg.inner_text('#dpo-rail .railcard > p.small') == Z(lang, 'Pick a scenario and see how the four answers’ probabilities change as DPO trains.', '选一个场景，看看 DPO 训练时四个回答的概率怎么变。')
          and pg.get_attribute('#dpo-rail .presets', 'aria-label') == Z(lang, 'Scenarios, pick one', '应用场景，选一个'), (pg.inner_text('#dpo-rail .rh'), pg.inner_text('#dpo-rail .railcard > p.small')))
    names = pg.evaluate("()=>[...document.querySelectorAll('#dpo-rail [data-preset] b')].map(b=>b.textContent)")
    descs = pg.evaluate("()=>[...document.querySelectorAll('#dpo-rail [data-preset] span')].map(b=>b.textContent)")
    check(f'{tag} scenarios have plain names; situations for the two setting-like ones; one-line descriptions without steps or β = 0.1', names[0] == Z(lang, 'Initial setup: one preference pair', '初始设置：一对偏好') and names[4] == Z(lang, 'Closer to a real LLM: shared weights', '更像真实 LLM：回答共享权重') and names[5] == Z(lang, 'Worried about drift: large β', '怕模型跑偏：用大 β')
          and not any(Z(lang, 'steps', '步') in d_ or 'β = 0.1' in d_ for d_ in descs), str(names) + str(descs))
    ins0 = txt(pg, '#dpo-insight')
    check(f'{tag} scenarios: entered at 60 steps, nothing is selected and the box says so (R1.8)', pg.evaluate("()=>[...document.querySelectorAll('#dpo-rail [data-preset]')].every(b=>b.getAttribute('aria-pressed')==='false')")
          and ins0.startswith(Z(lang, 'Your own setup', '你自己的设置')) and Z(lang, 'The current settings match none of these scenarios.', '现在的设置和这几个场景都不一样。') in ins0, ins0[:200])
    on_screen = []
    for i_ in range(6):
        btn = pg.locator(f'#dpo-rail [data-preset="{i_}"]'); btn.scroll_into_view_if_needed(); btn.click(); pg.wait_for_timeout(900)
        on_screen.append(pg.evaluate("()=>{const r=document.getElementById('dpo-insight').getBoundingClientRect(), d=document.getElementById('dock'), bar=document.querySelector('.topbar'); const top=Math.max(bar?bar.getBoundingClientRect().bottom:0, d&&!d.hidden?d.getBoundingClientRect().bottom:0); return [Math.round(r.top),Math.round(r.bottom),Math.round(top),innerHeight]}"))
    check(f'{tag} scenarios: after each click the whole result is on screen, below the top bar and the mini result bar', all(t_ >= tb - 1 and b_ <= ih for t_, b_, tb, ih in on_screen), str(on_screen))
    more = pg.inner_text('#dpo-rail .railcard > p.dpo-sc-more')
    check(f'{tag} scenarios: the card ends with the Settings line, as a jump link', more == Z(lang, 'Change anything else under Settings.', '想改别的，在“自由调参”里改。') and vis(pg, '#dpo-rail .railcard > p.dpo-sc-more [data-jump="controls"]'), more)
    md_ = pg.evaluate("()=>{const b=document.querySelector('#dpo-app [data-flow-nav] button[aria-pressed=true]'); return b?(b.dataset.tipx||''):''}")
    check(f'{tag} scenarios: the mode line does not repeat the Settings line', md_ == Z(lang, 'Pick a kind of preference data and SFT model and watch DPO train on it.', '选一种偏好数据和 SFT 模型，看 DPO 在上面怎么训练。'), md_)
    pg.locator('#dpo-rail .railcard > p.dpo-sc-more [data-jump="controls"]').click(); pg.wait_for_timeout(900)
    lk = pg.evaluate("()=>{const c=document.getElementById('dpo-controls'), r=c.getBoundingClientRect(), d=document.getElementById('dock'), bar=document.querySelector('.topbar'); const top=Math.max(bar?bar.getBoundingClientRect().bottom:0, d&&!d.hidden?d.getBoundingClientRect().bottom:0); return [Math.round(r.top), Math.round(top), innerHeight, c.classList.contains('flash')]}")
    check(f'{tag} scenarios: the Settings link lands with the panel’s name in view and flashes it', lk[0] >= lk[1] - 1 and lk[0] <= lk[2] - 80 and lk[3], str(lk))
    pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(200)
    pg.click('#dpo-rail [data-preset="2"]'); pg.wait_for_timeout(200)
    ins = txt(pg, '#dpo-insight')
    check(f'{tag} scenarios: preset pressed, result names it and the log 2 balance point, with the margin as shown on its card', pg.get_attribute('#dpo-rail [data-preset="2"]', 'aria-pressed') == 'true' and ins.startswith(Z(lang, 'Annotators disagree', '标注员意见不一')) and Z(lang, 'After 100 steps', '训练 100 步后') in ins
          and Z(lang, 'the loss is lowest when, under the implicit reward, A beats B two times out of three: at a reward margin of log 2 ≈ 0.693', '按隐式奖励算，A 三次里有两次胜过 B 时 loss 最小，也就是奖励差 = log 2 ≈ 0.693 时') in ins
          and Z(lang, 'The reward margin is now close to that, so more steps change almost nothing.', '现在的奖励差已经很接近，再训练几乎不变。') in ins and '0.687' not in ins
          and pg.inner_text('#dpo-metrics [data-k="margin"] .v') == '+0.687', ins[:300])
    check(f'{tag} scenarios: a scenario click is the learner’s change, so the cards may show ▲▼ badges', any(badges(pg)), str(badges(pg)))
    pg.click('#dpo-rail [data-preset="3"]'); pg.wait_for_timeout(200)
    ins = txt(pg, '#dpo-insight')
    check(f'{tag} scenarios: stubborn SFT model compares with the initial setup', Z(lang, 'It just takes more steps for A to pass B: step 31 here, against step 11 in the initial setup.', '只是要训练更多步，A 才会超过 B：这里是第 31 步，初始设置是第 11 步。') in ins, ins[:300])
    pg.click('#dpo-rail [data-preset="0"]'); pg.wait_for_timeout(200)
    ins = txt(pg, '#dpo-insight')
    check(f'{tag} scenarios: initial setup says why A keeps rising for any β (the paper is in Fidelity), no stray push weight', Z(lang, 'This pair has one label and no vote against it, so the loss is lowest only when B’s probability reaches 0: A keeps rising for any β, and a larger β only makes the push fade sooner.', '这一对只有一个标注，没有反对票，所以 B 的概率要降到 0，loss 才最小：不管 β 多大，A 都会一直上升，β 越大只是推力衰减得越早。') in ins
          and 'Azar' not in ins and Z(lang, 'push weight is now', '现在的推力权重') not in ins, ins[:300])
    pg.click('#dpo-rail [data-preset="2"]'); pg.wait_for_timeout(200)
    pg.click('#dpo-ctl-beta button[data-v="0.5"]'); pg.wait_for_timeout(200)
    ins = txt(pg, '#dpo-insight')
    check(f'{tag} scenarios: a manual change unpresses every scenario and says so', pg.evaluate("()=>[...document.querySelectorAll('#dpo-rail [data-preset]')].every(b=>b.getAttribute('aria-pressed')==='false')") and Z(lang, 'The current settings match none of these scenarios.', '现在的设置和这几个场景都不一样。') in ins, ins[:200])
    curve_checks(pg, tag, 'probability view at 100 steps')
    rules(pg, f'{tag} scenarios')
    shot(pg, f'{tag}_10_scenarios')
    hsh = pg.evaluate("()=>location.hash")
    check(f'{tag} hash written', hsh == '#dpo~mixed_b_sep_0.5_0.1_100', hsh)
    if w < 700:
        # the mini result bar appears when the chart scrolls away while working on the lab
        pg.evaluate("()=>document.getElementById('dpo-controls').scrollIntoView({block:'center'})"); pg.wait_for_timeout(400)
        dk = pg.evaluate("()=>({shown:!document.getElementById('dock').hidden, text:document.getElementById('dock').innerText, fits:[...document.querySelectorAll('#dock .dock-in > *')].every(e=>{const r=e.getBoundingClientRect(); return r.right<=innerWidth+0.5})})")
        check(f'{tag} dock visible at {w} with a full reading that fits', dk['shown'] and dk['fits'] and Z(lang, 'After 100 steps the model gives A 37% of the time', '训练 100 步后，模型给出 A 的概率是 37%') in dk['text'] and Z(lang, 'The model most often gives', '模型最常给出') in dk['text'], json.dumps(dk, ensure_ascii=False))
        shot(pg, f'{tag}_11_dock')
        pg.click('#dock'); pg.wait_for_timeout(700)
        check(f'{tag} dock click returns to the chart', pg.evaluate("()=>{const r=document.getElementById('dpo-graph').getBoundingClientRect(); return r.top<innerHeight&&r.bottom>0}"))
    # settings panel: named Settings / 自由调参, no hint line, notes as sentences
    ctl = pg.inner_text('#dpo-controls')
    check(f'{tag} settings panel is “Settings”, with the SFT model named for before training', ctl.split('\n')[0].lower() == Z(lang, 'settings', '自由调参') and Z(lang, 'SFT model (before training)', 'SFT 模型（训练前）') in ctl, ctl[:160])
    sc = pg.evaluate(SCAN, '#dpo-app *')
    check(f'{tag} no blue, amber or violet anywhere in the lab (R5.6)', not sc, str(sc[:4]))
    # guided mode again: current part and prediction are kept
    pg.evaluate("()=>window.scrollTo(0,0)"); pg.click('#dpo-app .modes button[data-mode="guided"]'); pg.wait_for_timeout(250)
    check(f'{tag} back to guided keeps the part', Z(lang, 'part 8 of 8', '第 8 / 8 部分') in pg.inner_text('#dpo-rail .rail-top').lower())
    pg.close()
    # hash round trip: a fresh page opens the same setup in scenarios
    pg = page(b, w, h, lang, hash_=hsh[1:], dark=dark)
    s = state(pg)
    check(f'{tag} hash round-trip restores the setup in scenarios', s['mode'] == 'sandbox' and (s['ds'], s['ref'], s['logits'], s['beta'], s['steps']) == ('mixed', 'b', 'sep', 0.5, 100) and pg.get_attribute('#dpo-ctl-beta button[data-v="0.5"]', 'aria-pressed') == 'true', json.dumps(s))
    pg.close()

def curve_checks(pg, tag, where):
    """Every line style drawn is in the key, and the end labels stay inside the plot without touching the axis text."""
    r = pg.evaluate("""()=>{const svg=document.getElementById('dpo-curve'), sb=svg.getBoundingClientRect();
      const box=e=>{const b=e.getBoundingClientRect(); return [b.left,b.top,b.right,b.bottom]};
      const lbl=[...svg.querySelectorAll('.dpo-lbl')].map(box), ax=[...svg.querySelectorAll('.dpo-ax')].map(box);
      const hit=(a,b)=>a[0]<b[2]-0.5&&b[0]<a[2]-0.5&&a[1]<b[3]-0.5&&b[1]<a[3]-0.5;
      const lines=new Set([...svg.querySelectorAll('path.dpo-ln')].map(p=>p.getAttribute('class')));
      const keyed=new Set([...document.querySelectorAll('#dpo-curve-key svg line')].map(l=>l.getAttribute('class')));
      if(svg.querySelector('.dpo-ref')) lines.add('dpo-ref'); lines.add('dpo-now');
      return {overlap:lbl.some(a=>ax.some(b=>hit(a,b))), outside:lbl.some(a=>a[0]<sb.left-1||a[2]>sb.right+1||a[1]<sb.top-1||a[3]>sb.bottom+1),
              missing:[...lines].filter(c=>!keyed.has(c)), n:lbl.length}}""")
    check(f'{tag} {where}: curve key names every line style; end labels inside the plot, clear of the axis text', not r['overlap'] and not r['outside'] and not r['missing'], json.dumps(r))

def tile_checks(b, w, lang, dark=False):
    tag = f'teaser {"dark " if dark else ""}{w} {lang}'
    pg = page(b, w, 844, lang, hash_='map', dark=dark)
    pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="post"], #domStrip [data-world="post"]')].find(e=>e.offsetParent);
      if(g) g.click(); else {const m=document.querySelector('#mapSvg .world[data-world="post"]'); m.dispatchEvent(new MouseEvent('click',{bubbles:true}));}}"""); pg.wait_for_timeout(350)
    tile = pg.locator('#tile-dpo'); tile.scroll_into_view_if_needed(); pg.wait_for_timeout(150)
    reads, words = [], []
    for v in (0, 1, 20, 30, 40):
        pg.evaluate(f"()=>{{const i=document.querySelector('#tile-dpo input[type=range]'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}}))}}"); pg.wait_for_timeout(80)
        rd = pg.inner_text('#tile-dpo .tz-read'); reads.append(rd); words.append(len(rd.split()) if lang == 'en' else units(rd))
    r30 = calc(pg, {'steps': 30})
    # R6.20: A's change is the one number (on the bar); B's fall in words, which the engine must bear out (more than 25 points at 30 steps)
    check(f'{tag}: the sentences change with the slider; no constants without referents in them; one key number, B in words', len(set(reads)) == 5 and Z(lang, f'Annotators prefer answer A to B. After 30 DPO steps, the model gives A {pct(r30["pi"][0])} of the time instead of 18%, and B far less often.', f'标注员更喜欢回答 A 而不是 B。训练 30 步 DPO 后，模型给出 A 的概率从 18% 升到 {pct(r30["pi"][0])}，给出 B 的概率则大幅下降。') == reads[3]
          and r30['pi'][1] - 0.52 <= -0.25, reads[3])
    lab_ = pg.inner_text('#tile-dpo .tz-lab span')
    check(f'{tag}: the slider label names the steps without notation', lab_ == Z(lang, 'DPO training steps', 'DPO 训练步数'), lab_)
    check(f'{tag}: one or two sentences, at most 35 words (Chinese: at most 60 units, as in the continuous-batching test)', all(n <= (35 if lang == 'en' else 60) for n in words) and all(len(re.findall(r'[.。](\s|$)', r)) <= 2 for r in reads), str(words))
    cap = pg.inner_text('#tile-dpo .dpo-tz-cap')
    check(f'{tag}: a caption names the bars', cap == Z(lang, 'Four answers to one prompt; each bar is how often the model gives that answer.', '一个提问的四个回答；每条是模型给出这个回答的概率。'), cap)
    leg = pg.evaluate("()=>[...document.querySelectorAll('#tile-dpo .tz-leg > span')].map(s=>s.textContent)")
    check(f'{tag}: legend names chosen, rejected, not labeled and the SFT outline', leg == [Z(lang, 'A: preferred', 'A：标注员更喜欢'), Z(lang, 'B: rejected', 'B：落选'), Z(lang, 'C, D: not labeled', 'C、D：未标注'), Z(lang, 'SFT model before training', '训练前的 SFT 模型')], str(leg))
    notes_ = pg.evaluate("()=>[...document.querySelectorAll('#tile-dpo .tz-note')].map(n=>[n.textContent, getComputedStyle(n).display])")
    check(f'{tag}: three notes, hidden on the tile: what β is and does in this lab, what the step size means (no formula), toy run and source', len(notes_) == 3 and all(n[1] == 'none' for n in notes_) and Z(lang, 'β is the weight of the KL penalty in the RLHF objective', 'β 是 RLHF 目标里 KL 惩罚的系数') in notes_[0][0] and Z(lang, 'In this lab, the larger β, the less the model changes in the same number of steps', '在这个 Lab 里，β 越大，同样的训练步数下模型改动得越少') in notes_[0][0]
          and Z(lang, 'Step size 0.1 sets how much each training step changes the model.', '步长 0.1 决定每训练一步模型改动多少。') in notes_[1][0] and 'logit' not in notes_[1][0] and '÷' not in notes_[1][0] and Z(lang, 'toy run', '玩具实跑') in notes_[2][0] and Z(lang, 'Rafailov et al., 2023', 'Rafailov 等，2023') in notes_[2][0], json.dumps(notes_, ensure_ascii=False)[:300])
    check(f'{tag}: no “Assumptions & sources” on the tile (owner 2026-10-13: the tile is one button, no “Assumptions & sources” on it)', pg.evaluate("()=>!document.querySelector('#tile-dpo .tz-more')&&!/假设和来源|Assumptions (&|and) sources/.test(document.querySelector('#tile-dpo').innerText)"))
    chip = pg.inner_text('#tile-dpo .tz-kind')
    check(f'{tag}: the tile chip says Toy run and the note agrees', chip == Z(lang, 'Toy run', '玩具实跑'), chip)
    hgt = pg.evaluate("()=>document.querySelector('#tile-dpo .dpo-tz').getBoundingClientRect().height")
    check(f'{tag}: visual at most 130 px tall', hgt <= 130, str(hgt))
    sc = pg.evaluate(SCAN, '#tile-dpo *')
    check(f'{tag}: grayscale (no blue, amber or violet)', not sc, str(sc[:3]))
    check(f'{tag}: overflow', overflow(pg) <= 0)
    for h_ in pg.evaluate(RULES, '#tile-dpo') + pg.evaluate(COLONS, '#tile-dpo'): rule_hits.append(f'{tag}: {h_}')
    pg.evaluate("()=>{const i=document.querySelector('#tile-dpo input[type=range]'); i.value=20; i.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(80)
    pg.mouse.move(1, 1); pg.wait_for_timeout(150)
    tile.screenshot(path=str(SH / f'teaser_{"dark_" if dark else ""}{w}_{lang}.png')); pg.close()

def timing_checks(b):
    # animated play advances at the chosen speed (steps per second, by elapsed time) and can be paused (R3.9)
    pg = page(b, 1440, 900, 'en'); go_step(pg, PART['steps'])
    sp = pg.evaluate("()=>window.__dpo.consts.SPEEDS")
    check('speed control: slow / medium / fast, medium by default', state(pg)['speed'] == 'med' and sp == {'slow': 2, 'med': 5, 'fast': 20}, json.dumps(sp))
    pg.click('#dpo-transport [data-act="play"]'); pg.wait_for_timeout(1000)
    s1 = state(pg)['steps']; pg.click('#dpo-transport [data-act="play"]'); pg.wait_for_timeout(300); s2 = state(pg)['steps']; pg.wait_for_timeout(500)
    check('play advances about 5 steps per second at medium, pause stops', 3 <= s1 <= 8 and state(pg)['steps'] == s2, f'{s1} {s2}')
    pg.click('#dpo-transport [data-act="reset0"]'); pg.wait_for_timeout(100)
    pg.click('#dpo-transport .dpo-speed [data-v="slow"]'); pg.click('#dpo-transport [data-act="play"]'); pg.wait_for_timeout(2000)
    slow = state(pg)['steps']
    pg.click('#dpo-transport .dpo-speed [data-v="fast"]'); pg.wait_for_timeout(1000)
    fast = state(pg)['steps'] - slow
    pg.click('#dpo-transport [data-act="play"]')
    pressed = pg.evaluate("()=>[...document.querySelectorAll('.dpo-speed [data-v=\"fast\"]')].every(b=>b.getAttribute('aria-pressed')==='true')")
    check('speed control: slow ≈ 2 steps per second, switching to fast mid-play ≈ 20 per second, both controls pressed', 2 <= slow <= 6 and 12 <= fast <= 26 and pressed, f'slow 2 s: {slow} steps; fast 1 s: {fast} steps; pressed {pressed}')
    tip = hover_tip(pg, '#dpo-transport .dpo-speed')
    check('speed control tooltip states the rates in a sentence with a subject', 'Slow trains 2 steps per second, Medium 5 and Fast 20.' in tip, tip)
    pg.close()

with sync_playwright() as p:
    b = p.chromium.launch()
    if ONLY != 'timing':
        for dark in (False, True):
            for (w, h) in [(1440, 900), (390, 844)]:
                for lang in ['en', 'zh']:
                    run_flow(b, w, h, lang, dark)
        # the other review widths: layout, overflow, cause and effect on one screen
        for w, h in [(1280, 800), (900, 800), (720, 800)]:
            for lang in ['zh', 'en']:
                pg = page(b, w, h, lang); go_step(pg, 1); pg.click('#dpo-rail [data-pred="drift"]'); go_step(pg, PART['steps'])
                cols = pg.evaluate("w=>[w<=760?'#dpo-rail .pbar':'#dpo-rail','#dpo-stage',w<=760?'#dpo-rail .steps':'#dpo-controls','#dpo-controls'].map(s=>{const r=document.querySelector(s).getBoundingClientRect(); return [Math.round(r.left),Math.round(r.top)]})", w)
                if w >= 761: check(f'{w}_{lang} rail+settings left, stage right', cols[0][0] == cols[2][0] and cols[1][0] > cols[0][0], str(cols))
                else: check(f'{w}_{lang} one column: part bar, chart, card, settings (m2)', cols[0][1] < cols[1][1] < cols[2][1] < cols[3][1], str(cols))
                check(f'{w}_{lang} overflow', overflow(pg) <= 0, str(overflow(pg)))
                for vw_ in ('prob', 'wt'):
                    pg.evaluate(f"()=>document.querySelector('#dpo-view button[data-v=\"{vw_}\"]').click()"); pg.wait_for_timeout(120); h7(pg, f'{w}_{lang} {vw_} view')
                pg.evaluate("()=>document.querySelector('#dpo-view button[data-v=\"prob\"]').click()"); pg.wait_for_timeout(80)
                if w >= 761: pg.evaluate("()=>document.getElementById('dpo-ic-steps').scrollIntoView({block:'nearest'})"); pg.wait_for_timeout(300)   # the flow list above the card can push its control below the fold (owner's layout, 2026-10-11)
                if w >= 761: check(f'{w}_{lang} card control and bars on one screen', in_view(pg, '#dpo-ic-steps') and pg.evaluate("()=>{const r=document.getElementById('dpo-bars').getBoundingClientRect(); return r.top<innerHeight&&r.bottom>58}"))
                shot(pg, f'{w}_{lang}_3_steps')
                # hide the bars: one column scrolls the page; wider layouts scroll the sticky stage to its math
                pg.evaluate("()=>{const r=document.getElementById('dpo-bars').getBoundingClientRect(); window.scrollBy(0, r.bottom+20); document.getElementById('dpo-math').open=true; const st=document.getElementById('dpo-stage'); st.scrollTop=st.scrollHeight; st.dispatchEvent(new Event('scroll')); window.dispatchEvent(new Event('scroll'))}"); pg.wait_for_timeout(500)
                dk = pg.evaluate("()=>({shown:!document.getElementById('dock').hidden, fits:[...document.querySelectorAll('#dock .dock-in > *')].filter(e=>getComputedStyle(e).display!=='none').every(e=>e.getBoundingClientRect().right<=innerWidth+0.5), text:document.getElementById('dock').innerText})")
                check(f'{w}_{lang} mini result bar: shown when the bars scroll away, one line that fits (Back to the chart visible)', dk['shown'] and dk['fits'], json.dumps(dk, ensure_ascii=False))
                pg.close()
        # reduced motion: Play jumps straight to the end state
        pg = page(b, 1440, 900, 'en', reduced=True); go_step(pg, PART['steps'])
        pg.click('#dpo-transport [data-act="play"]'); pg.wait_for_timeout(120)
        check('reduced motion: Play jumps to step 100 at once', state(pg)['steps'] == 100)
        pg.close()
        # fidelity popover and its jump link; metric, legend and bar tooltips; the part-7 link in "In production"
        pg = page(b, 1440, 900, 'en'); go_step(pg, PART['steps'])
        pg.locator('#dpo-fid').scroll_into_view_if_needed(); pg.wait_for_timeout(120)
        fp = pg.inner_text('#dpo-fid')
        check('notes at the page end (owner, 2026-10-11 19:28 UTC): simplified, sources (Llama 3 and its reward model, Razin ICLR 2025, DPO-Positive, Azar), η/β, fixed learning rate and Adam, β in the RLHF objective',
              'Real compute' not in fp and 'a larger β would make the first steps larger' in fp and 'β weights the KL penalty' in fp and 'simplified' in fp.lower() and 'sources' in fp.lower() and 'illustrative' in fp and 'Last verified' in fp and 'Llama Team, 2024' in fp and 'reward model' in fp
              and 'Razin et al., ICLR 2025' in fp and 'DPO-Positive' in fp and 'Azar et al., 2023' in fp and 'Adam' in fp and '·' not in fp.replace('β·log', '').replace('π·log', ''))
        pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(100)
        tip = hover_tip(pg, '#dpo-metrics [data-k="kl"]')
        check('metric tooltip: KL in nats, the meaning before the formula', tip.startswith('KL measures how far the model has moved from the SFT model') and any(x.startswith('KL(π‖πref)=Σyπ(y)log') for x in maths(pg, '#tip')) and 'nats' in tip, tip[:160])
        tip = hover_tip(pg, '#dpo-metrics [data-k="margin"]')
        check('metric tooltip: the reward margin, named as the subject, explained through the implicit reward; no TRL log name', tip.startswith('The reward margin is A’s implicit reward minus B’s.') and 'r^=βlogππref' in maths(pg, '#tip') and 'rewards/margins' not in tip, tip[:200])
        tip = hover_tip(pg, '#dpo-app .legend-row [data-tip="dpo_lgRef"]')
        check('legend tooltip with jump to Settings', 'The dashed outline is' in tip and 'πref' in maths(pg, '#tip') and 'under Settings' in tip, tip[:200])
        tip = hover_tip(pg, '#dpo-app .legend-row [data-tip="dpo_lgL"]')
        check('legend tooltip describes the grayscale style (hatched)', 'A hatched bar marks' in tip, tip[:120])
        tip = hover_tip(pg, '#dpo-bars .dpo-row:nth-child(3)')
        check('bar tooltip: a full sentence with the answer named, and why the loss ignores it', tip.startswith('Answer C:') and 'before training' in tip.lower() and 'the loss never uses it' in tip, tip[:240])
        conn = pg.inner_text('#dpo-app .connect')
        check('Connect: code intro, loss_type list, logs, Llama 3, papers without “·”, no English quote', 'four log-probabilities per pair' in conn and 'loss_type=["sigmoid"]' in conn and 'rewards/margins' in conn and 'Llama Team, 2024' in conn and 'DPO-Positive' in conn
              and 'Razin et al., ICLR 2025' in conn and ' · ' not in conn and 'Higher β means' not in conn and 'the model keeps moving toward the chosen answer whatever β is' in conn)
        pg.locator('#dpo-app .connect [data-jump="step7"]').first.click(); pg.wait_for_timeout(500)
        s = state(pg)
        check('the “part 7” link in In production opens guided part 7 and reveals it', s['mode'] == 'guided' and s['step'] == PART['shared'] and (top_shown(pg, '#dpo-rail .step.current') or -1) >= TOPBAR, json.dumps(s))
        pg.close()
        # home tile teaser, light and dark
        for w, lang, dark in [(1440, 'en', False), (390, 'zh', False), (390, 'en', False), (1440, 'zh', True), (390, 'en', True)]:
            tile_checks(b, w, lang, dark)
        # narrowest tile width 280 px
        pg = b.new_page(viewport={'width': 312, 'height': 800}); pg.goto(URL + '#map'); pg.wait_for_timeout(300)
        pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="post"], #domStrip [data-world="post"]')].find(e=>e.offsetParent);
          if(g) g.click(); else {const m=document.querySelector('#mapSvg .world[data-world="post"]'); m.dispatchEvent(new MouseEvent('click',{bubbles:true}));}}"""); pg.wait_for_timeout(300)
        tz = pg.evaluate("()=>{const t=document.getElementById('tile-dpo'); return {w:t.getBoundingClientRect().width, inner:[...t.querySelectorAll('*')].every(e=>{const r=e.getBoundingClientRect(), q=t.getBoundingClientRect(); return r.width===0||(r.left>=q.left-0.5&&r.right<=q.right+0.5)})}}")
        pg.locator('#tile-dpo').screenshot(path=str(SH / 'teaser_280.png'))
        check('teaser at a 280 px tile: nothing sticks out of the tile', tz['inner'] and tz['w'] <= 282, json.dumps(tz)); pg.close()
        check(f'H1: no plain-text formula and no TeX word in any visited state, its hovers included ({len(h1_hits)} hits)', not h1_hits, '; '.join(h1_hits[:6]))
        check(f'H7: every svg <text> stays inside its svg and its panel at 1440, 1280, 900, 720 and 390 ({len(h7_hits)} hits)', not h7_hits, '; '.join(h7_hits[:6]))
        check('copy rules in every visited state (no “·” joins, no 起点 / starting point, no 控件 / Controls, no mini-bar, hover or page-updating sentences, no sentence with two colons)', not rule_hits, '; '.join(rule_hits[:6]))
    timing_checks(b)
    b.close()

# polish round 5 (2026-10-17, R3.2, R9.7, R7.9, R7.12, the owner's B4; as tests/moe_flow.py and tests/grpo_flow.py): the reader's walk on
# the whole site (built into out/ and wrapped with a doctype; the one-lab page lacks the “Read first” line, which moves every card 28 px),
# in the lab's real layout (window.__aspPrimer, as tests/kit_flow.py), entering each part as the polish probe stage.py does: Next when it
# is on, else the part's title. Before, at 1440 × 900 the stage scrolled inside itself at every part (40 px at parts 1–2, 53 at 3–6 and 8,
# 182 at part 7), the curve started 722 px down under the answers, and entering part 7 (and 8) pushed the stage up to −213 px (−35 px; in
# English part 6 to −271 and part 7 to −55). Now, at every part in both languages: the stage's top at its sticky line (70 px) or lower and
# its head below the top bar; no scrolling inside, at least 30 px to spare in Chinese (none needed in English); the curve (the main figure)
# beside the answers at the top of the chart and whole on the screen. Spare = the stage's max-height minus what its content and borders
# need. Part 7 is the exception: the picture “Where the logits come from” keeps the room of its shared-weight form (R7.9, dpo.js
# holdWire), so the stage scrolls inside a little (rule 7 of the round's brief): at most 50 px in Chinese and 70 in English, with the
# whole chart (answers, picture, curve) in the stage's visible part and on the screen. R7.9: in parts 3 and 7, after training,
# switching the curve's view and playing leave the stage's height alone; in part 6 so do Show me, another β and the curve's views with
# the key naming the previous setting; in part 7 the shared weight on and off, step 0, playing and Show me; in Settings the logits
# switched both ways and a run from step 0; every scenario card.
if ONLY != 'timing':
    subprocess.run([sys.executable, str(P / 'build.py'), 'out/site-dpo.html'], check=True, cwd=P, stdout=subprocess.DEVNULL)
    (P / 'out' / 'wrapped-site-dpo.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'site-dpo.html').read_text() + '</body></html>')
    SITE = 'file://' + str(P / 'out' / 'wrapped-site-dpo.html')
    WALK = """()=>{const s=document.querySelector('#dpo-stage'), r=s.getBoundingClientRect(), bar=document.querySelector('.topbar').getBoundingClientRect().bottom,
      cur=document.querySelector('#dpo-rail .steps>.step.current'), h=s.querySelector('.stage-head').getBoundingClientRect(), f=document.getElementById('dpo-curve').getBoundingClientRect(),
      bs=document.getElementById('dpo-bars').getBoundingClientRect(), ch=s.querySelector('.dpo-curve-head').getBoundingClientRect();
      return {part:cur?[...cur.parentNode.children].indexOf(cur)+1:0, top:Math.round(r.top), head:Math.round(h.top), bar:Math.round(bar), scroll:s.scrollHeight-s.clientHeight,
        spare:Math.round(parseFloat(getComputedStyle(s).maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight)), fig:[Math.round(f.top),Math.round(f.bottom)], vh:innerHeight,
        beside:Math.abs(ch.top-bs.top)<=2&&ch.left>=bs.right, graph:(q=>[Math.round(q.top),Math.round(q.bottom)])(document.getElementById('dpo-graph').getBoundingClientRect()),
        vis:[Math.round(r.top+s.clientTop),Math.round(r.top+s.clientTop+s.clientHeight)]};}"""
    HT = "()=>{const s=document.getElementById('dpo-stage'); return s.scrollHeight+s.offsetHeight-s.clientHeight}"
    with sync_playwright() as _p:
        _b = _p.chromium.launch()
        for _lang in ('en', 'zh'):
            _pg = _b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce'); _pg.add_init_script('window.__aspPrimer=true')
            _pg.on('pageerror', lambda e: errs.append(f'1440 walk: {e}'))
            _pg.goto(SITE); _pg.wait_for_timeout(600); _pg.click(f'.langsw button[data-lang="{_lang}"]'); _pg.wait_for_timeout(80)
            _pg.evaluate("()=>{location.hash='dpo'}"); _pg.wait_for_timeout(500)
            _g = _pg.locator('[data-flow-nav] button[data-flow=guided]:visible')
            if _g.count(): _g.first.click(); _pg.wait_for_timeout(700)
            _n = _pg.evaluate("()=>document.querySelectorAll('#dpo-rail .steps>.step').length"); _rows = [_pg.evaluate(WALK)]; _held = {}; _held6, _keys6 = None, ''
            for _i in range(1, _n):
                _nb = _pg.locator('.pbar [data-nav=next]:visible:not([disabled])')
                if _nb.count(): _nb.first.click()
                else: _pg.evaluate(f"()=>document.querySelector('#dpo-rail .steps>.step:nth-child({_i + 1}) .step-h').click()")
                _pg.wait_for_timeout(1500); _rows.append(_pg.evaluate(WALK))
                if _i + 1 in (3, 7):
                    # R7.9: after 10 training steps, each view of the curve and a few frames of playback leave the stage's height alone
                    _pg.evaluate("()=>{const b=document.querySelector('#dpo-stage [data-act=step10]'); if(b) b.click();}"); _pg.wait_for_timeout(300)
                    _hs = [_pg.evaluate(HT)]
                    for _v in ('kl', 'loss', 'wt', 'prob'):
                        _pg.evaluate(f"()=>{{const b=document.querySelector('#dpo-view button[data-v={_v}]'); if(b) b.click();}}"); _pg.wait_for_timeout(250); _hs.append(_pg.evaluate(HT))
                    _pg.evaluate("()=>{const b=document.querySelector('#dpo-stage [data-act=play]'); if(b) b.click();}")
                    for _k in range(4): _pg.wait_for_timeout(250); _hs.append(_pg.evaluate(HT))
                    _pg.evaluate("()=>{const b=document.querySelector('#dpo-stage [data-act=play]'); if(b&&b.getAttribute('aria-pressed')==='true') b.click();}"); _pg.wait_for_timeout(200)
                    _held[_i + 1] = _hs
                    _pg.evaluate("()=>{const b=document.querySelector('#dpo-stage [data-act=reset0]'); if(b) b.click();}"); _pg.wait_for_timeout(200)
                if _i + 1 == 6:
                    # R7.9 where β changes: the key can also name the previous setting, four lines in English on the half-width column
                    # (dpo.css keeps room for them in part 6): Show me, another β and each view of the curve leave the stage's height alone
                    _hs = [_pg.evaluate(HT)]
                    _pg.evaluate("()=>{const b=[...document.querySelectorAll('#dpo-rail .step.current [data-nav=showme]')].find(e=>e.getClientRects().length); if(b) b.click();}"); _pg.wait_for_timeout(1500); _hs.append(_pg.evaluate(HT))
                    _pg.evaluate("()=>{const b=[...document.querySelectorAll('#dpo-ic-beta button')].filter(e=>e.getClientRects().length&&e.getAttribute('aria-pressed')!=='true').pop(); if(b) b.click();}"); _pg.wait_for_timeout(600); _hs.append(_pg.evaluate(HT))
                    for _v in ('prob', 'kl', 'loss', 'wt', 'prob'):
                        _pg.evaluate(f"()=>{{const b=document.querySelector('#dpo-view button[data-v={_v}]'); if(b) b.click();}}"); _pg.wait_for_timeout(250); _hs.append(_pg.evaluate(HT))
                    _keys6 = _pg.evaluate("()=>document.getElementById('dpo-curve-key').textContent")
                    _held6 = _hs
            # R7.9 where the shared weight comes and goes (dpo.js holdWire): part 7 again (its title), Settings, then every scenario card
            _clk = lambda sel, ms: (_pg.evaluate(f"()=>{{const b=[...document.querySelectorAll('{sel}')].find(e=>e.getClientRects().length); if(b) b.click();}}"), _pg.wait_for_timeout(ms))
            _pg.evaluate("()=>document.querySelector('#dpo-rail .steps>.step:nth-child(7) .step-h').click()"); _pg.wait_for_timeout(1500)
            _h7 = [_pg.evaluate(HT)]
            for _sel, _ms in (('#dpo-ic-mode button[data-v=shared]', 900), ('#dpo-stage [data-act=reset0]', 600), ('#dpo-stage [data-act=play]', 300)):
                _clk(_sel, _ms); _h7.append(_pg.evaluate(HT))
            for _k in range(6): _pg.wait_for_timeout(200); _h7.append(_pg.evaluate(HT))
            _pg.evaluate("()=>{const b=document.querySelector('#dpo-stage [data-act=play]'); if(b&&b.getAttribute('aria-pressed')==='true') b.click();}"); _pg.wait_for_timeout(200)
            for _sel, _ms in (('#dpo-ic-mode button[data-v=sep]', 900), ('#dpo-rail .step.current [data-nav=showme]', 1500)):
                _clk(_sel, _ms); _h7.append(_pg.evaluate(HT))
            _pg.locator('[data-flow-nav] button[data-flow=free]:visible').first.click(); _pg.wait_for_timeout(1500)
            _hset = [_pg.evaluate(HT)]
            for _sel, _ms in (('#dpo-ctl-mode button[data-v=shared]', 900), ('#dpo-ctl-mode button[data-v=sep]', 900), ('#dpo-stage [data-act=reset0]', 600), ('#dpo-stage [data-act=play]', 300)):
                _clk(_sel, _ms); _hset.append(_pg.evaluate(HT))
            for _k in range(6): _pg.wait_for_timeout(200); _hset.append(_pg.evaluate(HT))
            _pg.evaluate("()=>{const b=document.querySelector('#dpo-stage [data-act=play]'); if(b&&b.getAttribute('aria-pressed')==='true') b.click();}"); _pg.wait_for_timeout(200)
            _pg.locator('[data-flow-nav] button[data-flow=scen]:visible').first.click(); _pg.wait_for_timeout(1500)
            _hscen = []
            for _k in range(_pg.evaluate("()=>document.querySelectorAll('#dpo-rail [data-preset]').length")):
                _pg.evaluate(f"()=>document.querySelectorAll('#dpo-rail [data-preset]')[{_k}].click()"); _pg.wait_for_timeout(1200); _hscen.append(_pg.evaluate(HT))
            check(f'1440 {_lang} part 7: the shared weight on, step 0, playing, the separate logits again and Show me leave the stage’s height alone (R7.9)', len(_h7) == 12 and max(_h7) - min(_h7) <= 1, _h7)
            check(f'1440 {_lang} Settings: the logits switched to the shared weight and back, step 0 and playing leave the stage’s height alone (R7.9)', len(_hset) == 11 and max(_hset) - min(_hset) <= 1, _hset)
            check(f'1440 {_lang} Scenarios: every card, the shared-weight one among them, leaves the stage’s height alone (R7.9)', len(_hscen) == 6 and max(_hscen) - min(_hscen) <= 1, _hscen)
            check(f'1440 {_lang} part 6: Show me, another β and each view of the curve (the key naming the previous setting) leave the stage’s height alone (R7.9)',
                  bool(_held6) and max(_held6) - min(_held6) <= 1 and ('previous' in _keys6 or '上一个' in _keys6), (_held6, _keys6))
            check(f'1440 {_lang} walk: entering every part, the stage’s top at its sticky line (70 px) or lower and its head below the top bar (R3.2, R9.7, B4)',
                  len(_rows) == _n == 8 and all(r['part'] == i + 1 and r['top'] >= max(68, r['bar'] - 1) and r['head'] >= r['bar'] - 1 for i, r in enumerate(_rows)), [(r['part'], r['top']) for r in _rows])
            for _r in _rows:
                if _r['part'] == 7:   # the picture's room kept for its shared-weight form (R7.9): rule 7, a short scroll inside, the whole chart on show
                    _cap = 50 if _lang == 'zh' else 70
                    check(f'1440 {_lang} walk part 7: the stage scrolls inside at most {_cap} px (the room of the shared-weight picture is kept, R7.9), the whole chart (answers, picture, curve) in its visible part and on the screen, the curve beside the answers (rule 7, R7.12)',
                          _r['scroll'] <= _cap and _r['beside'] and _r['graph'][0] >= max(_r['vis'][0], _r['bar']) - 1 and _r['graph'][1] <= min(_r['vis'][1], _r['vh']) + 1
                          and _r['fig'][0] >= _r['bar'] - 1 and _r['fig'][1] <= _r['vh'] + 1, _r)
                    continue
                check(f'1440 {_lang} walk part {_r["part"]}: the stage needs no scrolling inside' + (', 30 px to spare' if _lang == 'zh' else '') + ', the curve beside the answers and whole on the screen (B4, R7.12)',
                      _r['scroll'] <= 0 and _r['spare'] >= (30 if _lang == 'zh' else 0) and _r['beside'] and _r['fig'][0] >= _r['bar'] - 1 and _r['fig'][1] <= _r['vh'] + 1, _r)
            for _k, _hs in _held.items():
                check(f'1440 {_lang} part {_k}: switching the curve’s view and playing leave the stage’s height alone (R7.9)', max(_hs) - min(_hs) <= 1, _hs)
            _was = {('zh', 7): '−213', ('zh', 8): '−35', ('en', 6): '−271', ('en', 7): '−55', ('en', 8): '70'}   # stage.py, before
            for _k in {'zh': (7, 8), 'en': (6, 7, 8)}[_lang]:   # the parts whose stage the bench's end pushed up before (or that sit there): at 70 px
                _r = _rows[_k - 1] if len(_rows) >= _k else {}
                check(f'1440 {_lang} walk part {_k}: the stage’s top at its sticky line, 70 px (it was at {_was[(_lang, _k)]} px)', bool(_r) and abs(_r['top'] - 70) <= 2, _r)
            _pg.close()
        _b.close()

check('no page errors', not errs, '; '.join(errs[:5]))
# ---- G1/G2 (the owner's message of 2026-10-08): option lead-ins instead of “one at a time”, and no sentence that leans on a detail
# of an earlier part. Read from the lab's own source, so both languages are covered.
import re as _re
_SRC = ''.join((P / d / 'dpo.js').read_text() for d in ('labs', 'teasers') if (P / d / 'dpo.js').exists())
_G1 = _re.findall(r'.{0,30}(?:一次只能选|不会叠加|do not stack|[Oo]ne at a time|一次选一).{0,30}', _SRC)
check('G1: no “one at a time / do not stack” note left in the lab or its tile (EN and ZH)', not _G1, _G1[:3])
_NEW = ["logps/chosen", "means the chosen answers are getting less likely too, which happens when the answers share weights", "往下走，说明选中的回答也变得不太可能了。回答之间共享权重时，就会这样（见"]
_SRCJS = _re.sub(r'/\*.*?\*/', '', _SRC, flags=_re.S); _SRCJS = _re.sub(r'(?m)(^|[;,{}()\]])\s*//[^\n]*', r'\1', _SRCJS)
_PLAIN_OLD = ['β·log', 'log(π/π_ref)', '−log σ(', 'σ(x) = 1/(1 + e', 'e<sup>', '<sub>', 'π_ref', 'r̂', '(η ÷ β)', 'η ÷ β', 'Σ π', 'e^(', 'e^−x', '−log σ(奖励差)']
check('H1 (source): none of the old plain-text formula strings is left in the lab’s copy', not [x for x in _PLAIN_OLD if x in _SRCJS], [x for x in _PLAIN_OLD if x in _SRCJS])
_H4 = _re.findall(r'.{0,30}(?:规划中|planned|还没上线|not built|coming|later lab|以后的 Lab|将来).{0,30}', _SRCJS)
check('H4 (source): no “planned / 规划中” lab; eval-ci is linked with labLink', not _H4 and "A.shell.labLink('eval-ci')" in _SRCJS, _H4[:3])
_H10 = _re.findall(r'.{0,20}[吃喂扛吞].{0,20}', _SRCJS)
check('H10 (source): no figurative 吃 / 喂 / 扛 / 吞', not _H10, _H10[:3])
check(f'G1/G2: the new lead-ins and the rewritten sentences are in the lab ({len(_NEW)})', all(x in _SRC for x in _NEW), [x for x in _NEW if x not in _SRC][:3])
_OLD = []
check(f'G2: none of the {len(_OLD)} sentences that leaned on an earlier part (or said “one at a time”) is left', not any(x in _SRC for x in _OLD), [x for x in _OLD if x in _SRC][:3])
print('\n'.join(notes))
print('FAILED:', fails if fails else 'none')
sys.exit(1 if fails else 0)
