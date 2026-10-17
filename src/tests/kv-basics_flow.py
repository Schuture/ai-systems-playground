"""Flow test for the kv-basics lab (Transformer series ③).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH, light, plus one dark run; the layout at 1280, 900 and 720;
the prediction gate (Next, the small model's lock, the hidden 100-token count, R3.8); each part's action gate (4 words, mask off,
cache on); the challenge's three feedback tiers, “show all” and the chart; the language switch keeping state mid-guided and
mid-challenge; the hash round-trip; the mini result bar at 390; card heights after each action at 1440×1000 and 1440×900 with the
result on screen (R3.10); ▲▼ badges only after the learner's own change; keyboard jumps from Show the math; the home tile; and the
writing rules on every visible text node and tooltip: no “·” between clauses (R6.14), never two colons in one Chinese sentence (R6.7),
no “in this toy” / “在这个玩具里”, no “toy” / “玩具” as the model's name except the label “Toy run” / “玩具实跑” (r12), no 起点 /
starting point, no 控件 / Controls, no sentence about the page itself (R3.4), colours only for their data (R5.6).
Review round 12: the stage's title, badge and legend below the top bar after Next and Back at 1440×900, 1280×800 and 1440×1000
(N4); the line under the word the latest step wrote, in real pixels at 1x, words across and upright (N1); the phone challenge
slider staying put once the chart is shown (N7); the Fidelity popover's “Show the math” link (N9); the KV head definition (#8); the
Fidelity and primer sentences (N8, N13, N14, the source line with one colon).
Review round 14 (review 8): guided → Scenarios → Challenge by the cards' buttons (R3.12); a change under Settings never moves the page
(R3.2); part 6 says the GPU reads memory and computes at the same time, what both steps read and which term decides each, and explains
“reads the weights” and GiB on hover; every term of Show the math has its own tooltip, the pairs' derivation sits in its term and no
gray note is left (R1.4), in EN and ZH; the primer's words, part 1's “layer” and part 5's “KV cache” carry the kit's definitions
(CONTRACT §11); no number over 6 digits outside the math and tooltips (R6.18), no 亿 / 万 / 个权重 (R6.17); the Connect code without
sideways scrolling at 1440 and 1280. Polish round 4: part 7's formula in the card (KV cache per token = 2 × layers × …) is cut before
its operators and wraps instead of scrolling sideways, at 1440 (EN, ZH), 900 (EN) and 390 (EN, ZH) (R6.37). Screenshots: shots/kv-basics/.

Run: python3 tests/kv-basics_flow.py
"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/kv-basics.html', '--only', 'kv-basics'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-kv-basics.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'kv-basics.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-kv-basics.html')
SH = P / 'shots' / 'kv-basics'; SH.mkdir(parents=True, exist_ok=True)
errs, notes, fails, heights = [], [], [], []

def check(name, ok, info=''):
    notes.append(('OK   ' if ok else 'FAIL ') + name + (f'  [{str(info)[:300]}]' if info and not ok else ''))
    if not ok: fails.append(name); print('FAIL', name, '::', str(info)[:400])

Z = lambda lang, en, zh: zh if lang == 'zh' else en

def page(b, w, h, lang, dark=False, reduced=False):
    pg = open_drawers(b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light', reduced_motion='reduce' if reduced else 'no-preference'))
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate("()=>{location.hash='kv-basics'}"); pg.wait_for_timeout(350)
    return pg

overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
# the next lab: below the rail card on wide screens; on phones (≤ 760 px) the kit shows a copy after the bench instead (CONTRACT §9)
next_lab = lambda pg: pg.evaluate("""()=>{const shown=x=>!!x&&getComputedStyle(x).display!=='none'&&x.getBoundingClientRect().height>0;
  const r=document.querySelector('#kvb-rail .next-lab'), e=document.querySelector('#view-dyn .next-lab-end .next-lab');
  return innerWidth<=760 ? (shown(e)&&!shown(r)) : shown(r)}""")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
disabled = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return !!e&&e.disabled}", sel)
state = lambda pg: pg.evaluate("()=>window.__kvb.state()")
txt = lambda pg, sel: pg.inner_text(sel) if vis(pg, sel) else ''
nb = lambda s: s.replace('\u2011', '-').replace('\u00a0', ' ')

def settle(pg, ms=2500):
    # first the lab's own correction that keeps the sticky stage below the top bar (r12 N4), then the scroll itself
    try: pg.wait_for_function("()=>!window.__kvb||!window.__kvb.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120

# r12 N4: on wide screens the stage's title, badge and legend sit below the top bar after Next, Back or a pick
STAGE_HEAD = """()=>{const tb=document.querySelector('.topbar').getBoundingClientRect().bottom, st=document.getElementById('kvb-stage');
 if(innerWidth<=760||getComputedStyle(st).position!=='sticky') return {ok:true};
 const g=s=>{const e=document.querySelector(s); return e&&e.getClientRects().length?Math.round(e.getBoundingClientRect().top):null};
 const t=g('#kvb-st-title'), b=g('#kvb-status'), l=g('#kvb-legend');
 return {ok:[t,b,l].every(v=>v==null||v>=tb-1), title:t, badge:b, legend:l, bar:Math.round(tb)}}"""
# r12 N1: the word the latest step wrote carries a 2 px line along the bottom of its column head, in real pixels at 1x (words across
# and upright); a neighbouring column has none
def marker_px(pg):
    from PIL import Image
    import io
    loc = pg.locator('#kvb-grid .kvb-tok.kvb-new')
    if loc.count() != 1: return {'ok': False, 'why': 'no marked column'}
    loc.evaluate("e=>e.scrollIntoView({block:'center'})"); pg.wait_for_timeout(150)
    ink = pg.evaluate("()=>{const d=document.createElement('i'); d.style.color='var(--ink)'; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c}")
    rgb = [int(x) for x in ink[ink.find('(') + 1:ink.find(')')].split(',')[:3]]
    def rows(sel):
        bb = pg.locator(sel).bounding_box()
        im = Image.open(io.BytesIO(pg.screenshot(clip={'x': bb['x'], 'y': bb['y'], 'width': bb['width'], 'height': bb['height']}))).convert('RGB')
        w, h = im.size
        return [sum(1 for x in range(w) if sum(abs(im.getpixel((x, y))[i] - rgb[i]) for i in range(3)) < 60) / w for y in range(max(0, h - 3), h)]
    j = pg.evaluate("()=>+document.querySelector('#kvb-grid .kvb-tok.kvb-new').dataset.j")
    new, other = rows('#kvb-grid .kvb-tok.kvb-new'), rows(f'#kvb-grid .kvb-tok[data-j="{j - 1}"]')
    return {'ok': max(new) >= 0.8 and max(other) < 0.2, 'new': [round(x, 2) for x in new], 'neighbour': [round(x, 2) for x in other],
            'dpr': pg.evaluate('()=>devicePixelRatio'), 'upright': pg.evaluate("()=>getComputedStyle(document.querySelector('#kvb-grid .kvb-tok.kvb-new b')).writingMode!=='horizontal-tb'")}

def go_part(pg, i):
    pg.evaluate(f"()=>document.querySelector('#kvb-rail [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(200); settle(pg)

def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(100)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)

def hover_tip(pg, sel):
    loc = pg.locator(sel).first; loc.evaluate("e=>e.scrollIntoView({block:'center',inline:'nearest',behavior:'instant'})"); pg.wait_for_timeout(150)
    # the first line box: a term that wraps has a bounding box whose corner can sit on other text
    bb = loc.evaluate("el=>{const r=el.getClientRects()[0]; return {x:r.x,y:r.y,width:r.width,height:r.height}}")
    pg.mouse.move(bb['x'] + 3, bb['y'] + 3); pg.mouse.move(bb['x'] + min(8, bb['width'] / 2), bb['y'] + min(6, bb['height'] / 2)); pg.wait_for_timeout(200)
    return pg.inner_text('#tip') if vis(pg, '#tip') else ''

def badges(pg): return pg.evaluate("()=>[...document.querySelectorAll('#kvb-metrics .metric .d')].map(d=>d.textContent).filter(Boolean)")

# the writing rules, on every visible text node, every tooltip string and the hidden popovers (R6.14, R6.17, R2.1, R3.4, R6.11)
RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/in this toy\\b|在这个玩具里/i,'“in this toy”'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/Locked in|已锁定/,'“Locked in”'],[/\\bNaN\\b|undefined|\\[object/,'broken value'],
   [/\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy” as the model’s name (r12: “the small model”; only the label “Toy run” keeps it)'],
   [/亿|万/,'亿 or 万 instead of K, M, B (R6.18)'],[/个权重/,'“个权重” where parameters are counted (R6.17: 参数)'],
   [/个数(?!值)|每个数(?!值)|存的数(?!值)|里的数(?!值)|出的数(?!值)/,'“数” for numbers a cache or a comparison holds (r15 N3, R6.17: 数值)'],
   [/读写显存(?!的流量)/,'“读写显存” for the memory term (r15 N4: 读显存; only the activations’ traffic is read and written)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip')) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); for(const [re,name] of bad) if(re.test(t)) out.push(name+' (tooltip): '+t.slice(0,90));});
 return out;}"""
# Chinese typesetting on this site: a space between Chinese characters and Latin letters or digits (“8 个 KV head 上”)
SPACING = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(); root.querySelectorAll('p,li,h1,h3,.status,.tz-read,.kvb-spv,.kvb-lcap,.kvb-ctitle,.metric .k,.metric .sub,.dk-read,.meter-sub,.meter-title,.lv-d,.preset span,.cmp-title,.cmp-lab span').forEach(e=>{if(e.closest('[data-endnotes],[data-flow-nav],pre,code,svg,.mathbody')||e.matches('.step')) return; if(e.querySelector('p,li,div,ul,ol,h1,h2,h3')) return;
   const t=e.textContent.replace(/\\s+/g,' '); const m=t.match(/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>][\\u4e00-\\u9fff]/g); if(m) out.add(m.join('|')+' in: '+t.trim().slice(0,80));});
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); const m=t.match(/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>][\\u4e00-\\u9fff]/g); if(m) out.add(m.join('|')+' in tooltip: '+t.slice(0,80));});
 return [...out];}"""
COLONS = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), texts=[];
 root.querySelectorAll('p,li,button,label,div,span,h1,h2,h3,summary').forEach(e=>{if(e.closest('[data-endnotes],pre,code,svg')) return; if(e.querySelector('p,li,div,ul,ol,h1,h2,h3,table')) return; texts.push(e.textContent);});
 root.querySelectorAll('[data-tipx]').forEach(e=>texts.push(e.getAttribute('data-tipx')));
 for(const t of texts){const sents=t.replace(/\\s+/g,' ').split(/(?<=[。？！])|(?<=[.?!])\\s+/);
   for(const x of sents){const n=(x.match(/：|: /g)||[]).length; if(n>=2) out.add(x.trim().slice(0,160));}}   // r15 N13: English too
 return [...out];}"""
# blue (weights) and amber (KV cache) only on the elements that mean them; red only for a stale value or a missed goal (R5.6)
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom'));
 const ok={[blue]:/kvb-mw|sw w\\b/,[amber]:/kvb-read|kvb-mk|sw kv\\b|kvb-sw-read|kvb-tzr|kvb-dkg/,[red]:/kvb-bad|kvb-sw-stale|status bad|badline|badc|kvb-dkg/};
 const out=[]; for(const e of document.querySelectorAll('#kvb-app *, #tile-kv-basics *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
# R6.18 (r14): outside Show the math, the drill's worked lines and tooltips, no number has more than 6 digits
BIGNUM = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #kvb-mathbody, .num-big')) continue;
   const dp=e.closest('#kvb-drill-a p'); if(dp&&dp.previousElementSibling) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
rule_hits = []
def rules(pg, where):
    for sel in ('#kvb-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
        if '_zh' in where or ' zh' in where:
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: no space between Chinese and Latin: {h}')

def run_guided(b, w, h, lang, dark=False):
    tag = f'{"dark_" if dark else ""}{w}_{lang}'
    pg = page(b, w, h, lang, dark)
    title = pg.inner_text('#kvb-app .labtitle')
    check(f'{tag} title', title == Z(lang, 'Why does generation keep every token’s keys and values?', '生成时为什么要把 K 和 V 存起来？'), title)
    check(f'{tag} no fidelity tag on top (owner, 2026-10-11 19:28 UTC)', not any(w in pg.inner_text('#kvb-app .labmeta') for w in ('Fidelity', '保真度')), pg.inner_text('#kvb-app .labmeta'))
    check(f'{tag} no primer tag on top (owner, 2026-10-11 19:28 UTC)', not any(w in pg.inner_text('#kvb-app .labmeta') for w in ('Primer', '前置知识')), pg.inner_text('#kvb-app .labmeta'))
    fid = nb(pg.evaluate("()=>document.getElementById('kv-basics-fid').textContent"))
    # the “Modeled” list and the “Toy run” lead (the small model’s shape, the “Show the math” link, the step-time item) went with the top Fidelity card (owner, 2026-10-11 19:28 UTC)
    check(f'{tag} notes at the page end: the H100 source with one colon (r12 site #12 note)',
          Z(lang, 'H100 SXM, from NVIDIA’s product page: BF16', 'H100 SXM：NVIDIA 产品页') in fid and 'H100 SXM: NVIDIA’s product page:' not in fid, fid[-900:])
    # the 前置知识 popover and its term checks went with the header buttons (owner, 2026-10-11 19:28 UTC); the 知识梳理 step replaces it
    kit = pg.evaluate("()=>{const T=window.ASP.terms; return {layer:T.layer(),attention:T.attention(),head:T.head(),aweight:T.aweight(),mha:T.mha(),kvcache:T.kvcache(),gib:T.gib(),mask:T.mask(),kib:T.kib()}}")
    check(f'{tag} the model is “the small model”: chart switch and stage title (r12 site #12)',
          pg.inner_text('#kvb-vseg button[data-v="toy"]') == Z(lang, 'Small model', '小模型') and pg.inner_text('#kvb-st-title').startswith(Z(lang, 'Small model: 2 layers', '小模型：2 层'))
          and Z(lang, 'Here a small model takes in', '这里有一个小模型，先读入') in pg.inner_text('#kvb-rail .step.current'), pg.inner_text('#kvb-st-title'))
    check(f'{tag} mode description: seven parts', Z(lang, 'Seven short parts', '一共 7 个部分') in pg.evaluate("()=>{const b=document.querySelector('#kvb-app [data-flow-nav] button[aria-pressed=true]'); return b?(b.dataset.tipx||''):''}"))
    check(f'{tag} part 1 of 7', Z(lang, 'part 1 of 7', '第 1 / 7 部分') in pg.inner_text('#kvb-rail .rail-top').lower().replace('guided', '').strip() or Z(lang, 'part 1 of 7', '第 1 / 7 部分') in pg.inner_text('#kvb-rail .rail-top'))
    if w < 760:
        order = pg.evaluate("()=>['#kvb-rail','#kvb-stage','#kvb-controls'].map(s=>Math.round(document.querySelector(s).getBoundingClientRect().top))")
        check(f'{tag} phone layout: card, then chart, then settings', order[0] < order[1] < order[2] and order[0] < h, order)
    m1 = pg.inner_text('#kvb-rail .step.current')
    check(f'{tag} mission says one word is one token and poses the question', Z(lang, 'in this lab one word is one token', '在这个 Lab 里，一个词或一个标点就是一个 token') in m1 and Z(lang, 'Compute them again, or keep them?', '是重新算一遍，还是存下来？') in m1, m1[:300])
    # CONTRACT §11 (r14): part 1 defines “layer” with the kit's text, then this lab's two models
    d1 = pg.evaluate("()=>[...document.querySelectorAll('#kvb-rail .step.current .def')].map(e=>[e.textContent,e.getAttribute('data-tipx')])")
    lay = [nb(t_) for w_, t_ in d1 if w_ == Z(lang, 'layer', '层')]
    check(f'{tag} part 1: “layer” carries the kit’s definition, then the two models’ layer counts (A.terms, r14)',
          lay == [nb(kit['layer'] + Z(lang, ' The small model here has 2 layers; Llama-3.1-8B has 32.', '这里的小模型有 2 层，Llama-3.1-8B 有 32 层。'))], d1)
    check(f'{tag} mission: at most 3 paragraphs', pg.evaluate("()=>document.querySelectorAll('#kvb-rail .step.current .step-b > p').length") <= 3)
    check(f'{tag} before the prediction the 100-token count is hidden (R3.8)', pg.evaluate("()=>document.querySelector('#kvb-metrics [data-k=\"p100\"] .v').textContent") == '?'
          and '5,050' not in pg.evaluate("()=>document.querySelector('#kvb-metrics [data-k=\"p100\"]').dataset.tipx"))
    check(f'{tag} before the prediction the toy cannot write (Settings says why)', not vis(pg, '#kvb-ctl-n') and vis(pg, '#kvb-nlock'))
    check(f'{tag} no badge on entering', not badges(pg), badges(pg))
    rules(pg, f'{tag} part 1')
    pg.evaluate("()=>window.scrollTo(0,0)"); shot(pg, f'{tag}_1_mission')
    check(f'{tag} overflow, part 1', overflow(pg) <= 0, overflow(pg))
    def stage_head(label):
        sh = pg.evaluate(STAGE_HEAD)
        check(f'{tag} {label}: the stage’s title, badge and legend are below the top bar (r12 N4)', sh['ok'], sh)
    # part 2: the prediction gates Next
    pg.click('#kvb-rail [data-nav="next"]'); pg.wait_for_timeout(250); settle(pg)
    stage_head('part 2 after Next')
    check(f'{tag} part 2: the card top (phones: the part bar) is on screen after Next (R3.7)', pg.evaluate("w=>{const r=document.querySelector(w<=760?'#kvb-rail .pbar':'#kvb-rail .step.current').getBoundingClientRect(); return r.top>=50&&r.top<innerHeight-120}", w))
    check(f'{tag} part 2: Next disabled until a prediction', disabled(pg, '#kvb-rail [data-nav="next"]'))
    opts = pg.evaluate("()=>[...document.querySelectorAll('#kvb-rail [data-pred]')].map(b=>b.textContent)")
    check(f'{tag} part 2: four single-choice options that cover the range', opts == Z(lang, ['Under 200', '200 to 2,000', '2,000 to 7,500', 'More than 7,500'], ['不到 200 次', '200 到 2,000 次', '2,000 到 7,500 次', '超过 7,500 次']), opts)
    pick = '200' if lang == 'en' else '2000'
    pg.click(f'#kvb-rail [data-pred="{pick}"]'); pg.wait_for_timeout(200)
    c2 = pg.inner_text('#kvb-rail .step.current')
    check(f'{tag} part 2: saved note, no “Locked in”', Z(lang, 'Saved. You can still change it', '已记下。') in c2 and 'Locked' not in c2, c2[-200:])
    check(f'{tag} part 2: Next enabled after the prediction', not disabled(pg, '#kvb-rail [data-nav="next"]'))
    check(f'{tag} part 2: the answer is still hidden', pg.evaluate("()=>document.querySelector('#kvb-metrics [data-k=\"p100\"] .v').textContent") == '?')
    rules(pg, f'{tag} part 2'); shot(pg, f'{tag}_2_predict')
    # part 3: write words; the observation appears from 4 steps
    pg.click('#kvb-rail [data-nav="next"]'); pg.wait_for_timeout(250); settle(pg)
    stage_head('part 3 after Next')
    check(f'{tag} part 3: Next disabled before writing', disabled(pg, '#kvb-rail [data-nav="next"]'))
    for k in range(3):
        pg.click('#kvb-rail [data-act="gen"]'); pg.wait_for_timeout(120)
    check(f'{tag} part 3: after 3 steps the result and the 100-token count are still hidden', not vis(pg, '#kvb-rail [data-obs]') and pg.evaluate("()=>document.querySelector('#kvb-metrics [data-k=\"p100\"] .v').textContent") == '?')
    check(f'{tag} part 3: token-passes 1, 3, 6 so far', pg.evaluate("()=>document.querySelector('#kvb-metrics [data-k=\"passes\"] .v').textContent") == '6')
    b3 = badges(pg)
    check(f'{tag} part 3: a badge after the learner’s click, from the shown values', any(x.startswith('▲') for x in b3), b3)
    pg.click('#kvb-rail [data-act="gen"]'); pg.wait_for_timeout(250); settle(pg)
    o3 = pg.inner_text('#kvb-rail [data-obs]')
    right = Z(lang, '2,000 to 7,500', '2,000 到 7,500 次')
    verdict_ok = (Z(lang, 'was right', '是对的') in o3) if pick == '2000' else (Z(lang, f'the right answer is “{right}”', f'正确答案是“{right}”') in o3)
    check(f'{tag} part 3: result first, then the verdict quoting both options, then the reason', o3.find(Z(lang, 'After 4 decode steps: 10 token-passes', '4 个 decode 步之后，一共做了 10 次 token 前向')) >= 0
          and '5,050' in o3 and verdict_ok and o3.find(Z(lang, 'Why:', '原因：')) > o3.find('5,050'), o3)
    check(f'{tag} part 3: the reason says the trend in words; the sums are in the metric card’s tooltip and Show the math (R6.20)',
          Z(lang, 'so the total grows with the square of n', '所以总次数随 n 的平方增长') in o3 and '1 + 2 +' not in o3 and 'n(n + 1)' not in o3, o3)
    check(f'{tag} part 3: the 100-token count shows after the result', pg.evaluate("()=>document.querySelector('#kvb-metrics [data-k=\"p100\"] .v').textContent") == '5,050')
    check(f'{tag} part 3: Next enabled', not disabled(pg, '#kvb-rail [data-nav="next"]'))
    check(f'{tag} part 3: the grid shows the triangle (10 computed cells per layer)', pg.evaluate("()=>[...document.querySelectorAll('#kvb-grid .kvb-layer')].map(l=>l.querySelectorAll('.kvb-comp').length)") == [10, 10])
    check(f'{tag} part 3: the word step 4 wrote (token 5) is underlined, and the legend says so', pg.evaluate("()=>[...document.querySelectorAll('#kvb-grid .kvb-tok.kvb-new')].map(t=>+t.dataset.j)") == [4]
          and Z(lang, 'underlined: the word the latest step wrote', '下划线：最近一步生成的词') in pg.inner_text('#kvb-legend'))
    rules(pg, f'{tag} part 3'); shot(pg, f'{tag}_3_steps')
    if w >= 1000:
        check(f'{tag} part 3: the button and the grid on one screen (R3.2)', pg.evaluate("()=>{const a=document.querySelector('#kvb-ic-gen').getBoundingClientRect(), g=document.querySelector('#kvb-grid .kvb-layer').getBoundingClientRect(); return a.top>=50&&a.bottom<=innerHeight&&g.top>=50&&g.top<innerHeight-60}"))
    mk = marker_px(pg)
    check(f'{tag} part 3: the line under the word step 4 wrote shows in real pixels at 1x, words {"upright" if mk.get("upright") else "across"} (r12 N1)', mk['ok'] and mk['dpr'] == 1, mk)
    for k in range(5):
        pg.click('#kvb-rail [data-act="gen"]'); pg.wait_for_timeout(80)
    check(f'{tag} part 3: at 9 steps the button stops and says why', disabled(pg, '#kvb-rail [data-act="gen"]') and Z(lang, 'The sentence ends at “it”.', '这句话到“它”就结束了。') in pg.inner_text('#kvb-rail .step.current'))
    # part 4: the recompute check
    pg.click('#kvb-rail [data-nav="next"]'); pg.wait_for_timeout(250); settle(pg)
    stage_head('part 4 after Next')
    check(f'{tag} part 4: no badge on entering a part', not badges(pg), badges(pg))
    o4 = pg.inner_text('#kvb-rail [data-obs]')
    check(f'{tag} part 4: with the mask no earlier key or value changed (the “change” rows below show 0 in both layers)', Z(lang, 'No earlier key or value changed.', '之前的 key 和 value 一个都没变。') in o4, o4)
    check(f'{tag} part 4: Next disabled until the mask is switched off', disabled(pg, '#kvb-rail [data-nav="next"]'))
    chk = pg.evaluate("()=>[...document.querySelectorAll('#kvb-grid .kvb-chk')].map(r=>r.hidden?null:[...r.querySelectorAll('.kvb-d')].map(d=>d.textContent))")
    check(f'{tag} part 4: the “change” rows show 0 under every earlier token', chk == [['0'] * 8 + ['—'], ['0'] * 8 + ['—']], chk)
    pg.click('#kvb-ic-mask button[data-v="0"]'); pg.wait_for_timeout(250); settle(pg)
    o4 = pg.inner_text('#kvb-rail [data-obs]')
    word = Z(lang, 'The', '小猫')
    check(f'{tag} part 4, mask off: layer 2 differs by 1.16 at token 2, layer 1 by 0; the size of layer 2’s numbers in words, their average on hover (R6.20)', Z(lang, f'layer 1’s did not. In layer 2 the largest change is 1.16 (at “{word}”), larger than the average size of the numbers in its keys and values.', f'第 1 层的没变。第 2 层变化最大的是 1.16（在“{word}”上），比这一层 key 和 value 里数值的平均大小还大。') in o4
          and Z(lang, 'At step 9, the numbers in layer 2’s keys and values average 0.9, ignoring sign.', '第 9 步时，第 2 层 key 和 value 里的数值，不计正负号，平均是 0.9。') in [nb(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#kvb-rail [data-obs] .def')].map(e=>e.getAttribute('data-tipx'))")], o4)
    d4 = pg.evaluate("()=>[...document.querySelectorAll('#kvb-rail .step.current .def')].map(e=>[e.textContent,e.getAttribute('data-tipx')])")
    mk4 = [nb(t_) for w_, t_ in d4 if w_ == Z(lang, 'causal mask', '因果 mask')]
    check(f'{tag} part 4: “causal mask” is the kit’s definition, then this lab’s sentence that a later token cannot change what an earlier token computes (r15 ARCH #16)',
          mk4 == [nb(kit['mask'] + Z(lang, ' So a later token cannot change what an earlier token computes, in any layer.', '所以后面来的 token 改变不了前面的 token 在任何一层算出的东西。'))], d4)
    chk = pg.evaluate("()=>[...document.querySelectorAll('#kvb-grid .kvb-chk')].map(r=>[...r.querySelectorAll('.kvb-d')].map(d=>d.textContent+(d.classList.contains('kvb-bad')?'!':'')))")
    check(f'{tag} part 4, mask off: layer-2 differences in red, layer 1 still 0', chk[0][:8] == ['0'] * 8 and all(x.endswith('!') for x in chk[1][:8]), chk)
    check(f'{tag} part 4, mask off: the legend explains red', Z(lang, 'changed, so a kept copy would be stale', '变了，存下来的就过期了') in pg.inner_text('#kvb-legend'))
    check(f'{tag} part 4: Next enabled', not disabled(pg, '#kvb-rail [data-nav="next"]'))
    rules(pg, f'{tag} part 4'); shot(pg, f'{tag}_4_check')
    # R3.2, CONTRACT §11 (r14): a change under Settings never scrolls the page; the control stays under the pointer
    pg.evaluate("()=>document.getElementById('kvb-ctl-mask').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200); settle(pg)
    top_of = "()=>Math.round(document.getElementById('kvb-ctl-mask').getBoundingClientRect().top)"
    t0 = pg.evaluate(top_of)
    pg.click('#kvb-ctl-mask button[data-v="1"]'); pg.wait_for_timeout(500); settle(pg); t1 = pg.evaluate(top_of)
    pg.click('#kvb-ctl-mask button[data-v="0"]'); pg.wait_for_timeout(500); settle(pg); t2 = pg.evaluate(top_of)
    check(f'{tag} part 4: switching the causal mask under Settings leaves the page where it is (R3.2)', abs(t1 - t0) <= 2 and abs(t2 - t0) <= 2 and not state(pg)['mask'], (t0, t1, t2))
    # part 5: the cache (from part 4's long card with the mask off: the card gets shorter, r12 N4)
    pg.click('#kvb-rail [data-nav="next"]'); pg.wait_for_timeout(250); settle(pg)
    stage_head('part 5 after Next')
    check(f'{tag} part 5: entering sets the mask back on and the cache off, and says so', state(pg)['mask'] and not state(pg)['cache']
          and Z(lang, '(The causal mask is back on.)', '（因果 mask 已重新打开。）') in pg.inner_text('#kvb-rail [data-obs]'))
    check(f'{tag} part 5: Next disabled until the cache is on', disabled(pg, '#kvb-rail [data-nav="next"]'))
    d5 = pg.evaluate("()=>[...document.querySelectorAll('#kvb-rail .step.current .def')].map(e=>[e.textContent,e.getAttribute('data-tipx')])")
    kvd = [nb(t_) for w_, t_ in d5 if w_ == 'KV cache']
    check(f'{tag} part 5: “KV cache” carries the kit’s definition, then how it grows (A.terms, r14)',
          kvd == [nb(kit['kvcache'] + Z(lang, ' It grows by one token’s worth with every decode step.', '每个 decode 步它都多存一个 token 的量。'))], d5)
    pg.click('#kvb-ic-cache button[data-v="1"]'); pg.wait_for_timeout(250); settle(pg)
    o5 = pg.inner_text('#kvb-rail [data-obs]')
    check(f'{tag} part 5: 1 token-pass instead of 9; the work grows in step with the length (100 instead of 5,050); 32 numbers per token kept (R6.20)', all(x in o5 for x in ['5,050', Z(lang, '32 numbers here', '一共 32 个数值')])
          and Z(lang, 'step 9 ran 1 token-pass instead of 9', '第 9 步只做 1 次 token 前向，而不是 9 次') in o5 and Z(lang, 'the work grows in step with the length, not its square', '工作量随长度线性增长，而不是平方增长') in o5, o5)
    cells = pg.evaluate("()=>[...document.querySelectorAll('#kvb-grid .kvb-layer')].map(l=>[l.querySelectorAll('.kvb-comp').length,l.querySelectorAll('.kvb-read').length])")
    check(f'{tag} part 5: the diagonal computed, the rest read from the cache', cells == [[9, 36], [9, 36]], cells)
    st = pg.inner_text('#kvb-status')
    check(f'{tag} part 5: the status says the cache matches a recompute (green)', 'ok' in pg.evaluate("()=>document.getElementById('kvb-status').className") and st == Z(lang, 'Cache matches a recompute', 'cache 和重算的一致'), st)
    b5 = badges(pg)
    check(f'{tag} part 5: badges after switching the cache on', any('−80%' in x for x in b5), b5)
    rules(pg, f'{tag} part 5'); shot(pg, f'{tag}_5_keep')
    # part 6: the real model
    pg.click('#kvb-rail [data-nav="next"]'); pg.wait_for_timeout(250); settle(pg)
    stage_head('part 6 after Next')
    check(f'{tag} part 6: the stage shows Llama-3.1-8B', vis(pg, '#kvb-sp') and not vis(pg, '#kvb-grid'))
    tip = nb(hover_tip(pg, '#kvb-rail .step.current .def'))
    check(f'{tag} part 6: the KV head definition is the kit’s, then Llama-3.1-8B’s “32 heads”, never “query heads” (r12 site #8)',
          Z(lang, 'A KV head is one key vector and one value vector that each token stores in every layer', 'KV head 是每个 token 在每一层存的一个 key 向量和一个 value 向量') in tip
          and Z(lang, 'fewer KV heads mean a smaller KV cache in total. Llama-3.1-8B’s 32 heads per layer share 8 KV heads, one for every 4 heads, so each token keeps 8 keys and 8 values per layer',
                'KV head 越少，KV cache 的总量越小。Llama-3.1-8B 每层 32 个 head 共用 8 个 KV head，每 4 个共用一个，所以每个 token 每层存 8 个 key 和 8 个 value') in tip and 'query head' not in tip
          and Z(lang, '“GQA and MLA” (GQA is grouped-query attention, MLA multi-head latent attention)', '“GQA 和 MLA”（GQA 即 grouped-query attention，MLA 即 multi-head latent attention）') in tip, tip)
    pg.mouse.move(w - 2, h - 2); pg.wait_for_timeout(150)
    o6 = pg.inner_text('#kvb-rail [data-obs]').replace(' ', ' ')
    check(f'{tag} part 6 at 8K: about 8.0 ms with the cache and about 270 ms without (rounded, R6.20), the cache 1 GiB', all(x in o6 for x in ['1 GiB', Z(lang, 'about 8.0 ms with the KV cache and about 270 ms without it', '用 KV cache 约 8.0 ms，不用约 270 ms')]), o6)
    # review 8 points 2, 4, 5 (r14): the time model said plainly where it first appears; what both steps read; which term decides each
    body6 = nb(pg.inner_text('#kvb-rail .step.current .step-b'))
    check(f'{tag} part 6: the card says the GPU reads memory and computes at the same time, so a step takes the slower of the two (R2.7)',
          Z(lang, 'The GPU reads memory and computes at the same time, so a decode step takes about as long as the slower of the two.', 'GPU 读显存和做计算是同时进行的，所以一个 decode 步的耗时约等于两者中较慢的那个。') in body6, body6[:500])
    check(f'{tag} part 6 at 8K: the reason says both steps read the weights once, and memory decides with the cache, arithmetic without it (R2.7), in words (R6.20)',
          Z(lang, 'every step reads the weights once, however many tokens it computes. With the cache it computes one token, so memory decides and its time barely grows. Without it, the step recomputes the whole context, so arithmetic decides and the time grows with the context.',
                  '不管算多少个 token，每一步都要读一遍权重。用 cache 时只算一个 token，所以读显存决定耗时，几乎不随上下文变长。不用 cache 时要把整段上下文重算一遍，所以计算决定耗时，而且随上下文增长。') in o6, o6)
    tips6 = pg.evaluate("()=>[...document.querySelectorAll('#kvb-rail .step.current .def')].map(e=>[e.textContent,e.getAttribute('data-tipx')])")
    ov = [nb(t_) for w_, t_ in tips6 if 'slower of the two' in w_ or '较慢的那个' in w_]
    rw = [nb(t_) for w_, t_ in tips6 if w_ in ('reads the weights', '读一遍权重')]
    gibs = [t_ for w_, t_ in tips6 if w_ == 'GiB']
    check(f'{tag} part 6: “the slower of the two” explains the overlap, “reads the weights” the mechanism (one read per step for any number of tokens, also without the cache), GiB the kit’s definition',
          len(ov) == 1 and Z(lang, 'at the same time', '同时进行') in ov[0] and len(rw) == 1 and Z(lang, 'so it costs the same for 1 token or 100', '算 1 个 token 和算 100 个 token 读的一样多') in rw[0]
          and Z(lang, 'still reads the weights only once', '权重也只读一次') in rw[0] and '14.0 GiB' in rw[0] and gibs == [kit['gib']] and Z(lang, '1 GiB (1.07 GB)', '1 GiB（1.07 GB）') in o6, (ov, rw, gibs))
    kibs = [nb(t_) for w_, t_ in tips6 if w_ == 'KiB']
    check(f'{tag} part 6: KiB carries the kit’s definition (r15 ARCH #16)', kibs == [nb(kit['kib'])], kibs)
    lgw = nb(pg.evaluate("()=>document.querySelector('#kvb-legend [data-tipx]').getAttribute('data-tipx')"))
    check(f'{tag} part 6: the legend’s “weights” tip has one colon and says parameters (r15 N13)',
          lgw == Z(lang, 'The weights are the model’s 8.03B parameters, 2 bytes each in BF16: 8,030,261,248 × 2 bytes = 16.1 GB = 15.0 GiB. They stay in GPU memory and are shared by every conversation.',
                   '权重就是模型的参数，一共 8.03B 个，BF16 下每个 2 字节：8,030,261,248 × 2 字节 = 16.1 GB = 15.0 GiB。它们一直放在显存里，所有对话共用。'), lgw)
    pg.evaluate("()=>{const i=document.getElementById('kvb-ic-ctx-in'); i.value=11; i.dispatchEvent(new Event('input',{bubbles:true})); i.dispatchEvent(new Event('change',{bubbles:true}))}"); pg.wait_for_timeout(250); settle(pg)
    o6 = pg.inner_text('#kvb-rail [data-obs]').replace(' ', ' ')
    check(f'{tag} part 6 at 128K: 16 GiB, more than the weights, about 16 ms with the cache and about 13 s without, over half of the step reading the cache', all(x in o6 for x in ['16 GiB', Z(lang, 'about 16 ms', '约 16 ms'), Z(lang, 'about 13 s', '约 13 s'), Z(lang, 'more than all the weights', '比全部权重还大'), Z(lang, 'over half of the step', '一半以上的时间')]), o6)
    pg.evaluate("()=>{const i=document.getElementById('kvb-ic-ctx-in'); i.value=0; i.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(250)
    o6 = pg.inner_text('#kvb-rail [data-obs]').replace(' ', ' ')
    check(f'{tag} part 6 at 256: the same 7.5 ms with or without (the sentence follows the numbers, R6.10)', Z(lang, 'takes about 7.5 ms with or without the KV cache', '用不用 KV cache 都约 7.5 ms') in o6 and Z(lang, 'so the cache saves nothing yet', '所以 cache 暂时省不了什么') in o6, o6)
    rules(pg, f'{tag} part 6'); shot(pg, f'{tag}_6_real')
    # part 7
    pg.click('#kvb-rail [data-nav="next"]'); pg.wait_for_timeout(250); settle(pg)
    stage_head('part 7 after Next')
    c7 = pg.inner_text('#kvb-rail .step.current')
    check(f'{tag} part 7: three misconceptions, with one number each at most (R6.20); the encoder one states what follows, not the mask-off change of part 4 (G2, the owner 2026-10-08)', c7.count('✗') == 3 and Z(lang, 'over 30 times as long', '30 多倍') in c7 and '1.16' not in c7
          and Z(lang, 'and a stored copy would be out of date.', '存下来的那份就不能再用了。') in c7
          and Z(lang, 'work grows with the length squared; with it, linearly.', '工作量随长度的平方增长；存下来，只随长度线性增长。') in c7, c7[-500:])
    check(f'{tag} part 7: the speed misconception’s ratio is the stage’s at 8K (the badge says 33 times as fast)', Z(lang, 'so at 8K tokens it takes', '到 8K 个 token 时') in nb(c7) and '33' in pg.inner_text('#kvb-status'), (c7[-400:], pg.inner_text('#kvb-status')))
    check(f'{tag} part 7: the next lab below the card (on phones, after the settings)', next_lab(pg))
    rules(pg, f'{tag} part 7'); shot(pg, f'{tag}_7_remember')
    check(f'{tag} overflow, guided', overflow(pg) <= 0, overflow(pg))
    return pg

def run_modes(pg, w, h, lang, tag):
    # scenarios, from the last part's button (R3.12: guided → Scenarios → Challenge → next lab)
    go_part(pg, 6)
    c7btn = pg.evaluate("()=>[...document.querySelectorAll('#kvb-rail .step.current [data-mode-go]')].map(b=>[b.dataset.modeGo,b.textContent])")
    check(f'{tag} part 7 ends with the button to the scenarios, not the challenge (R3.12)', c7btn == [['sandbox', Z(lang, 'See the scenarios', '去看应用场景')]], c7btn)
    pg.click('#kvb-rail .step.current [data-mode-go="sandbox"]'); pg.wait_for_timeout(300); settle(pg)
    check(f'{tag} the button opens Scenarios', state(pg)['mode'] == 'sandbox', state(pg)['mode'])
    check(f'{tag} scenarios: no badge on a mode switch', not badges(pg), badges(pg))
    sh = pg.evaluate(STAGE_HEAD)
    check(f'{tag} scenarios after the switch from part 7: the stage’s title, badge and legend are below the top bar (r12 N4)', sh['ok'], sh)
    check(f'{tag} scenarios: the stage shows Llama-3.1-8B and the card asks for a pick', state(pg)['view'] == 'real' and Z(lang, 'Pick one of the four scenarios above.', '从上面四个场景里选一个。') in pg.inner_text('#kvb-insight'), pg.inner_text('#kvb-insight'))
    card = pg.inner_text('#kvb-rail .railcard')
    check(f'{tag} scenarios card: name, the lead-in that says what to pick and what to watch (G1, the owner 2026-10-08), four presets, Settings link', Z(lang, 'Pick a scenario', '选一个应用场景') in card and Z(lang, 'Pick a scenario and see how long the whole answer takes with and without the KV cache.', '选一个场景，看看用和不用 KV cache，整个回答各要多久。') in card
          and Z(lang, 'do not stack', '不会叠加') not in card
          and pg.evaluate("()=>document.querySelectorAll('#kvb-rail [data-preset]').length") == 4 and vis(pg, '#kvb-rail [data-jump="controls"]'), card[:200])
    check(f'{tag} scenarios: the next lab below the card (on phones, after the settings)', next_lab(pg))
    tots = []
    for i in range(4):
        pg.click(f'#kvb-rail [data-preset="{i}"]'); pg.wait_for_timeout(300); settle(pg)
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#kvb-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        t = pg.inner_text('#kvb-insight')
        box = pg.evaluate("()=>{const r=document.getElementById('kvb-insight').getBoundingClientRect(); return [Math.round(r.top),Math.round(r.bottom),innerHeight]}")
        check(f'{tag} scenario {i + 1}: pressed, result on screen after the click (R3.10)', pressed.count('true') == 1 and pressed[i] == 'true' and 50 <= box[0] < box[2] - 60, (pressed, box))
        if w >= 1000:
            btn = pg.evaluate(f"()=>{{const r=document.querySelector('#kvb-rail [data-preset=\"{i}\"]').getBoundingClientRect(); return [Math.round(r.top),Math.round(r.bottom)]}}")
            fits = box[1] - btn[0] <= box[2] - 56 - 8        # the picked button and the whole result fit on one screen
            check(f'{tag} scenario {i + 1}: scrolled just enough: the whole result is on screen, and so is the picked scenario when both fit (R3.10)',
                  box[0] >= 56 and box[1] <= box[2] and (btn[0] >= 56 or not fits), (btn, box, fits))
        tots.append(t)
        if i == 0: shot(pg, f'{tag}_sb_{i + 1}')
        if i == 1:
            gd = pg.evaluate("()=>[...document.querySelectorAll('#kvb-insight .def')].map(e=>[e.textContent,e.getAttribute('data-tipx')])")
            check(f'{tag} scenario 2: the cache at the end in GiB with GB beside it, the unit defined (r15 N10, R6.12)',
                  Z(lang, 'by the last one it holds 4.12 GiB (4.43 GB).', '它占 4.12 GiB（4.43 GB）。') in nb(t) and ['GiB', pg.evaluate('()=>window.ASP.terms.gib()')] in gd, (nb(t)[-260:], gd))
    check(f'{tag} scenarios: each result names whole-answer times with and without the cache',
          all(Z(lang, 'with the KV cache the whole answer takes about', '用 KV cache 整个回答大约要') in t for t in tots) and '25 min' in nb(tots[1]) and '35 min' in nb(tots[3]), [t[:80] for t in tots])
    pg.evaluate("()=>document.getElementById('kvb-ctl-ctx').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200); settle(pg)
    top_of = "()=>Math.round(document.getElementById('kvb-ctl-ctx-in').getBoundingClientRect().top)"
    t0 = pg.evaluate(top_of)
    pg.evaluate("()=>{const i=document.getElementById('kvb-ctl-ctx-in'); i.value=7; i.dispatchEvent(new Event('input',{bubbles:true})); i.dispatchEvent(new Event('change',{bubbles:true}))}"); pg.wait_for_timeout(500); settle(pg)
    t1 = pg.evaluate(top_of)
    check(f'{tag} scenarios: moving the context slider under Settings leaves the page where it is (R3.2)', abs(t1 - t0) <= 2, (t0, t1))
    check(f'{tag} scenarios: a change under Settings leaves no scenario pressed and says so', pg.evaluate("()=>[...document.querySelectorAll('#kvb-rail [data-preset]')].every(b=>b.getAttribute('aria-pressed')==='false')")
          and Z(lang, 'match none of these scenarios', '都不一样') in pg.inner_text('#kvb-insight'))
    rules(pg, f'{tag} scenarios')
    # hash round-trip: the setup link restores the state
    tok = pg.evaluate("()=>location.hash")
    check(f'{tag} hash carries the state', tok.startswith('#kv-basics~r_'), tok)
    # challenge, from the scenarios card's button (R3.12)
    scbtn = pg.evaluate("()=>[...document.querySelectorAll('#kvb-rail .railcard [data-mode-go]')].map(b=>[b.dataset.modeGo,b.textContent])")
    check(f'{tag} the scenarios card ends with the button to the challenge (R3.12)', scbtn == [['challenge', Z(lang, 'Take the challenge', '去做挑战')]], scbtn)
    pg.click('#kvb-rail .railcard [data-mode-go="challenge"]'); pg.wait_for_timeout(300); settle(pg)
    check(f'{tag} the button opens the challenge', state(pg)['mode'] == 'challenge', state(pg)['mode'])
    sh = pg.evaluate(STAGE_HEAD)
    check(f'{tag} challenge after the switch: the stage’s title, badge and legend are below the top bar (r12 N4)', sh['ok'], sh)
    card = nb(pg.inner_text('#kvb-rail .railcard'))
    check(f'{tag} challenge: the goal names model, hardware, condition and target (R6.8)', all(x in card for x in ['Llama-3.1-8B', 'H100', '20', '50 ms']) and Z(lang, 'KV cache off', '关掉 KV cache') in card, card[:300])
    check(f'{tag} challenge: only the context control, the rest hidden with a note (R4.2)', vis(pg, '#kvb-ic-chctx') and vis(pg, '#kvb-lock') and not vis(pg, '#kvb-ctl-cache') and not vis(pg, '#kvb-ctl-ctx'))
    check(f'{tag} challenge: starts at 4K, not met, live readout 8.0 / 20', pg.inner_text('#kvb-ch-now') == '8.0' and 'bad' in pg.evaluate("()=>document.getElementById('kvb-status').className"))
    check(f'{tag} challenge: no chart before submitting (R3.8)', not vis(pg, '#kvb-ch-res .cmp'))
    rules(pg, f'{tag} challenge'); shot(pg, f'{tag}_ch_1')
    def set_stop(i):
        pg.evaluate(f"()=>{{const e=document.getElementById('kvb-ic-chctx-in'); e.value={i}; e.dispatchEvent(new Event('input',{{bubbles:true}})); e.dispatchEvent(new Event('change',{{bubbles:true}}))}}"); pg.wait_for_timeout(200)
    set_stop(4)   # 2,048: missed
    pg.click('#kvb-rail [data-act="submit"]'); pg.wait_for_timeout(300); settle(pg)
    r = pg.inner_text('#kvb-ch-res')
    check(f'{tag} challenge tier 3 (missed): the hint, no answer, a “Show the answer” link', r.startswith(Z(lang, 'Not yet', '还没达到')) and Z(lang, 'Hint', '提示') in r and '1,536' not in r and vis(pg, '#kvb-ch-res [data-act="reveal"]')
          and pg.inner_text('#kvb-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), r)
    ch_top = pg.evaluate("()=>{const r=document.getElementById('kvb-ch-res').getBoundingClientRect(); return [Math.round(r.top),innerHeight]}")
    check(f'{tag} challenge: the verdict is on screen after submitting', 50 <= ch_top[0] < ch_top[1] - 40, ch_top)
    set_stop(2)   # 1,024: met, shorter than possible
    r = pg.inner_text('#kvb-ch-res')
    check(f'{tag} challenge tier 2 (met, longer possible): names 1,536', r.startswith(Z(lang, 'Goal met', '达成目标')) and '1,536' in r and vis(pg, '#kvb-ch-res .cmp'), r[:300])
    set_stop(3)   # 1,536: exact
    r = pg.inner_text('#kvb-ch-res')
    # the brief chart leaves 2,048 out (it misses the goal): “the next one is already too slow” carries its 60.0 ms on hover
    tip = pg.evaluate("()=>[...document.querySelectorAll('#kvb-ch-res .okline .def')].map(d=>d.dataset.tipx||'').join(' | ')")
    check(f'{tag} challenge tier 1 (met, longest)', r.startswith(Z(lang, 'Goal met', '达成目标')) and Z(lang, 'longest context here that keeps up', '能达标的最长上下文') in r and '44.6' in r and '60.0' in tip and '2,048' in tip, (r[:300], tip))
    rows = pg.evaluate("()=>[...document.querySelectorAll('#kvb-ch-res .cmp-row')].map(r=>[r.classList.contains('ok'),r.classList.contains('me'),[...r.querySelectorAll('.cmp-tag')].map(t=>t.textContent).join()])")
    # the owner's review of batch 8: the chart keeps the four stops that pass, the pick and the initial setup; the note counts the rest
    check(f'{tag} challenge chart: only the four passing stops and the initial setup (4K), your pick marked', len(rows) == 5 and [x[0] for x in rows] == [True] * 4 + [False] and rows[3][1]
          and rows[4][2] == Z(lang, 'initial setup', '初始设置') and all(x[0] or x[1] or x[2] for x in rows), rows)
    note = pg.inner_text('#kvb-ch-res .cmp-note') if vis(pg, '#kvb-ch-res .cmp-note') else ''
    check(f'{tag} challenge chart: the note counts the 5 stops left out', note == Z(lang, 'The other 5 context lengths give fewer than 20 tokens per second and are not shown.', '其余 5 个上下文长度每秒不到 20 个 token，图里没有列出。'), note)
    ttl = pg.inner_text('#kvb-ch-res .cmp-title')
    check(f'{tag} challenge chart: the title names the quantity and the conditions, not every context', 'H100' in ttl and 'Llama-3.1-8B' in ttl.replace('\u2011', '-') and Z(lang, 'by context', '各上下文') not in ttl, ttl)
    xt = pg.evaluate("()=>[...document.querySelectorAll('#view-dyn [data-tipx]')].filter(e=>!e.closest('[id$=\"-mathbody\"],[id$=\"-math\"]')).map(e=>e.getAttribute('data-tipx')).filter(t=>t.includes('精确值是')||t.includes('Exactly'))")
    check(f'{tag} no hover outside Show the math gives a number’s exact value (the owner’s clarification of 2026-10-07)', not xt, xt[:2])
    check(f'{tag} challenge chart: the legend and the caveat', vis(pg, '#kvb-ch-res .cmp-leg') and Z(lang, 'Note:', '注意：') in pg.inner_text('#kvb-ch-res'))
    if w < 760:
        # r12 N7: with the chart shown the result is taller than a phone screen; while its top is on screen, moving the slider
        # leaves the page where it is, so the slider stays under the finger
        pg.evaluate("()=>{const s=document.getElementById('kvb-ic-chctx'); window.scrollBy(0, s.getBoundingClientRect().top-200)}"); pg.wait_for_timeout(300); settle(pg)
        ys = []
        for i in (5, 2, 3):
            y0 = pg.evaluate('()=>scrollY'); set_stop(i); pg.wait_for_timeout(300); settle(pg)
            ys.append((y0, pg.evaluate('()=>scrollY'), pg.evaluate("()=>Math.round(document.getElementById('kvb-ic-chctx-in').getBoundingClientRect().top)")))
        check(f'{tag} challenge on a phone: with the chart shown, moving the slider keeps the page and the slider in place (r12 N7)', all(a_ == b_ and 60 < t < h - 40 for a_, b_, t in ys), ys)
        shot(pg, f'{tag}_ch_slider_after_chart')
    rules(pg, f'{tag} challenge after submit'); shot(pg, f'{tag}_ch_2', full=(w < 760))
    # language switch mid-challenge keeps the state
    other = 'zh' if lang == 'en' else 'en'
    s0 = state(pg)
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(300)
    s1 = state(pg)
    check(f'{tag} language switch keeps the challenge state and the submitted result', s1['mode'] == 'challenge' and s1['ctx'] == s0['ctx'] and s1['chSubmitted'] and vis(pg, '#kvb-ch-res .cmp'), (s0, s1))
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(300)
    check(f'{tag} overflow, modes', overflow(pg) <= 0, overflow(pg))
    # back to guided: the state from before the challenge comes back
    pg.click('#kvb-app .modes button[data-mode="guided"]'); pg.wait_for_timeout(300)
    check(f'{tag} leaving the challenge restores the setup', state(pg)['mode'] == 'guided' and state(pg)['ctx'] != 1536 or state(pg)['step'] == 6)

def card_heights(b, lang, vh=1000):
    """R3.10: after each part's action at 1440×1000, the result (and a verdict) is on screen; record the card's height.
    At 1440×900 (the height the coordinator reports), the whole result box is on screen after the action."""
    pg = page(b, 1440, vh, lang)
    acts = {0: None, 1: ('click', '#kvb-rail [data-pred="2000"]'), 2: ('gen', 4), 3: ('click', '#kvb-ic-mask button[data-v="0"]'), 4: ('click', '#kvb-ic-cache button[data-v="1"]'),
            5: ('range', 11), 6: None}
    for i in range(7):
        go_part(pg, i); pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(100)
        pg.evaluate("()=>{const s=document.querySelector('#kvb-rail .step.current'); s.scrollIntoView({block:'start'}); window.scrollBy(0,-80)}"); pg.wait_for_timeout(150)
        a = acts[i]
        if a and a[0] == 'click': pg.click(a[1])
        elif a and a[0] == 'gen':
            for k in range(a[1]): pg.click('#kvb-rail [data-act="gen"]'); pg.wait_for_timeout(60)
        elif a and a[0] == 'range':
            pg.evaluate(f"()=>{{const e=document.getElementById('kvb-ic-ctx-in'); e.value={a[1]}; e.dispatchEvent(new Event('input',{{bubbles:true}})); e.dispatchEvent(new Event('change',{{bubbles:true}}))}}")
        pg.wait_for_timeout(250); settle(pg)
        m = pg.evaluate("""()=>{const c=document.querySelector('#kvb-rail .step.current').getBoundingClientRect(), o=document.querySelector('#kvb-rail [data-obs]');
          const r=o&&!o.hidden?o.getBoundingClientRect():null; return {h:Math.round(c.height), top:Math.round(c.top), bottom:Math.round(c.bottom), obsTop:r?Math.round(r.top):null, obsBottom:r?Math.round(r.bottom):null, vh:innerHeight}}""")
        heights.append((vh, lang, i + 1, m['h']))
        if m['obsTop'] is not None:
            check(f'1440x{vh} {lang} part {i + 1}: the result box starts on screen after the action (R3.10)', 50 <= m['obsTop'] < m['vh'] - 80, m)
            if vh < 1000: check(f'1440x{vh} {lang} part {i + 1}: the whole result box is on screen after the action', 50 <= m['obsTop'] and m['obsBottom'] <= m['vh'], m)
        if vh >= 1000: check(f'1440x1000 {lang} part {i + 1}: the card fits one screen ({m["h"]} px)', m['h'] <= m['vh'] - 70, m)
    pg.close()

def stage_after_part4(b, w, h, lang):
    """r12 N4 at the reviewer's other sizes: part 4 with the mask off (the long card), then Next and Back"""
    pg = page(b, w, h, lang)
    go_part(pg, 1); pg.click('#kvb-rail [data-pred="2000"]'); pg.wait_for_timeout(150)
    go_part(pg, 3); settle(pg)
    pg.click('#kvb-ic-mask button[data-v="0"]'); pg.wait_for_timeout(250); settle(pg)
    for label, nav in (('part 5 after Next', 'next'), ('part 4 after Back', 'back')):
        pg.click(f'#kvb-rail [data-nav="{nav}"]'); pg.wait_for_timeout(250); settle(pg)
        sh = pg.evaluate(STAGE_HEAD)
        top = pg.evaluate("()=>Math.round(document.querySelector('#kvb-rail .step.current').getBoundingClientRect().top)")
        check(f'{w}x{h} {lang} {label}: the stage’s title, badge and legend are below the top bar, and the card’s top is on screen (r12 N4)', sh['ok'] and 50 <= top <= h - 160, (sh, top))
        shot(pg, f'n4_{w}x{h}_{lang}_{nav}')
    pg.close()

# R3.2 (r15 N2): a change under Settings keeps the control just used where it was, also when it switches the chart (real clicks and
# keys; the review measured 49 px at 390 px before holdPlace)
def settings_hold(b, w, h, lang):
    pg = page(b, w, h, lang)
    TOP = "s=>document.querySelector(s).getBoundingClientRect().top"
    moves = []
    def act(label, sel, do):
        pg.evaluate("s=>document.querySelector(s).scrollIntoView({block:'center'})", sel); pg.wait_for_timeout(200); settle(pg)
        t0, v0 = pg.evaluate(TOP, sel), state(pg)['view']
        do(); pg.wait_for_timeout(450); settle(pg)
        moves.append((label, round(pg.evaluate(TOP, sel) - t0, 1), v0 + '->' + state(pg)['view']))
    key = lambda sel, k: (pg.focus(sel), pg.keyboard.press(k))
    pg.click('#kvb-app .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(300); settle(pg)
    act('scenarios: KV cache on', '#kvb-ctl-cache button[data-v="1"]', lambda: pg.click('#kvb-ctl-cache button[data-v="1"]'))
    act('scenarios: context +1', '#kvb-ctl-ctx-in', lambda: key('#kvb-ctl-ctx-in', 'ArrowRight'))
    act('scenarios: KV cache off', '#kvb-ctl-cache button[data-v="0"]', lambda: pg.click('#kvb-ctl-cache button[data-v="0"]'))
    act('scenarios: Reset', '#kvb-controls [data-act="reset"]', lambda: pg.click('#kvb-controls [data-act="reset"]'))
    pg.click('#kvb-app .modes button[data-mode="guided"]'); pg.wait_for_timeout(300); settle(pg)
    go_part(pg, 1); pg.click('#kvb-rail [data-pred="2000"]'); pg.wait_for_timeout(150); go_part(pg, 3)
    act('guided part 4: context +1', '#kvb-ctl-ctx-in', lambda: key('#kvb-ctl-ctx-in', 'ArrowRight'))
    act('guided part 4: KV cache off', '#kvb-ctl-cache button[data-v="0"]', lambda: pg.click('#kvb-ctl-cache button[data-v="0"]'))
    act('guided part 4: context +1 again', '#kvb-ctl-ctx-in', lambda: key('#kvb-ctl-ctx-in', 'ArrowRight'))
    act('guided part 4: causal mask off', '#kvb-ctl-mask button[data-v="0"]', lambda: pg.click('#kvb-ctl-mask button[data-v="0"]'))
    check(f'{w}x{h} {lang}: a change under Settings keeps the control within 1 px, also when it switches the chart (r15 N2, R3.2)',
          all(abs(m[1]) <= 1 for m in moves) and sum(m[2] in ('real->toy', 'toy->real') for m in moves) >= 5, moves)
    pg.close()

# G1 and G2, the owner's message of 2026-10-08 (reviews/owner-b9-2026-10-08.md): a lead-in above each group of options instead of
# “one at a time; they do not stack”, and no sentence that leans on a detail of an earlier part. Checked on the lab's source, both languages.
_G_SRC = (P / 'labs' / 'kv-basics.js').read_text()
for _b in ["一次只能选", "一次选一个", "一次选一种", "一次只选", "不会叠加", "do not stack", "One at a time", "pick one at a time", "Pick one at a time", "Try one change at a time", "一次试一种"]:
    check(f'G1: no “{_b}” left in the lab (EN and ZH)', _b not in _G_SRC)
for _s in ["Pick a scenario and see how long the whole answer takes with and without the KV cache.", "选一个场景，看看用和不用 KV cache，整个回答各要多久。"]:
    check(f'G1: the lead-in “{_s[:48]}” is in the lab', _s in _G_SRC)
for _s in ["and a stored copy would be out of date.", "存下来的那份就不能再用了。"]:
    check(f'G2: the rewritten sentence “{_s[:48]}” is in the lab', _s in _G_SRC)
for _s in ["(by up to ${d2} in this lab)", "（在这个 Lab 里最多差 ${d2}）"]:
    check(f'G2: the sentence that leaned on an earlier part, “{_s[:48]}”, is gone', _s not in _G_SRC)

with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ['en', 'zh']:
        for (w, h) in [(1440, 900), (390, 844)]:
            pg = run_guided(b, w, h, lang)
            tag = f'{w}_{lang}'
            if w == 390:
                # the mini result bar: after an action, while the chart is out of view
                go_part(pg, 2); pg.wait_for_timeout(150)
                pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(150)
                pg.locator('#kvb-rail [data-act="gen"]').scroll_into_view_if_needed()
                if disabled(pg, '#kvb-rail [data-act="gen"]'): pg.click('#kvb-rail [data-act="restart"]'); pg.wait_for_timeout(150)
                pg.click('#kvb-rail [data-act="gen"]')
                pg.wait_for_timeout(400)
                dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, t:d.innerText}}")
                check(f'{tag} the mini result bar shows the toy’s count after an action', dk['shown'] and re.search(Z(lang, r'Step \d of 9 without the KV cache: \d+ token-pass(es)? so far', r'第 \d / 9 步，不用 KV cache：目前 \d+ 次 token 前向'), dk['t']) is not None, dk)
                shot(pg, f'{tag}_dock_toy')
                go_part(pg, 3); pg.click('#kvb-ic-mask button[data-v="0"]'); pg.wait_for_timeout(400)
                pg.evaluate("()=>window.scrollBy(0,-40)"); pg.wait_for_timeout(300)
                dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, t:d.innerText}}")
                check(f'{tag} part 4 on a phone: the mini result bar shows the check, 1.16 in layer 2', dk['shown'] and '1.16' in dk['t'], dk)
                shot(pg, f'{tag}_dock_check')
            run_modes(pg, w, h, lang, tag)
            rules(pg, f'{tag} end')
            pg.close()
        card_heights(b, lang)
        card_heights(b, lang, 900)
        for (w, h) in [(1280, 800), (1440, 1000)]:
            stage_after_part4(b, w, h, lang)
        for (w, h) in [(390, 844), (1440, 900)]:
            settings_hold(b, w, h, lang)
    # dark mode, once
    pg = run_guided(b, 1440, 900, 'en', dark=True)
    bad = pg.evaluate(COLORS)
    check('dark: colours only for their data (blue weights, amber KV cache, red stale or missed)', not bad, bad)
    go_part(pg, 3); pg.click('#kvb-ic-mask button[data-v="0"]'); pg.wait_for_timeout(250); shot(pg, 'dark_1440_en_4_check')
    go_part(pg, 5); pg.wait_for_timeout(200); shot(pg, 'dark_1440_en_6_real')
    pg.close()
    pg = page(b, 1440, 900, 'en')
    bad = pg.evaluate(COLORS)
    check('light: colours only for their data', not bad, bad)
    # widths: no horizontal scroll, three layouts
    code_fits = "()=>{const c=document.getElementById('kvb-code'); return [c.scrollWidth,c.clientWidth]}"
    cw = pg.evaluate(code_fits)
    check('1440: the Connect code reads without sideways scrolling in the two-column layout (R7.6, r14)', cw[0] <= cw[1], cw)
    for w in (1280, 900, 720, 390):
        pg.set_viewport_size({'width': w, 'height': 900}); pg.wait_for_timeout(250)
        for part in (0, 3, 5):
            go_part(pg, part); pg.wait_for_timeout(150)
            check(f'{w}: no horizontal overflow, part {part + 1}', overflow(pg) <= 0, overflow(pg))
        if w == 1280:
            cw = pg.evaluate(code_fits)
            check('1280: the Connect code reads without sideways scrolling', cw[0] <= cw[1], cw)
        shot(pg, f'w{w}_en_part6')
    pg.set_viewport_size({'width': 1440, 'height': 900}); pg.wait_for_timeout(200)
    # hash round-trip: a setup link opens Scenarios with that state
    pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(150)
    pg.evaluate("()=>{location.hash='kv-basics~t_7_1_0_2048'}"); pg.wait_for_timeout(400)
    s = state(pg)
    check('hash round-trip: #kv-basics~t_7_1_0_2048 opens Scenarios with 7 steps, cache on, mask off, 2K', s['mode'] == 'sandbox' and s['n'] == 7 and s['cache'] and not s['mask'] and s['ctx'] == 2048 and s['view'] == 'toy', s)
    check('hash round-trip: the page writes the same token back', pg.evaluate("()=>location.hash") == '#kv-basics~t_7_1_0_2048')
    pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(150)
    pg.evaluate("()=>{location.hash='kv-basics~bogus'}"); pg.wait_for_timeout(400)
    check('a broken token falls back without errors', pg.evaluate("()=>!!document.getElementById('kvb-app')"))
    # keyboard: a formula term jumps to its control; a fixed term explains itself
    pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(150)
    pg.evaluate("()=>{location.hash='kv-basics~r_9_1_1_8192'}"); pg.wait_for_timeout(400)
    pg.evaluate("()=>{document.getElementById('kvb-math').open=true}"); pg.wait_for_timeout(100)
    tip = hover_tip(pg, '#kvb-mathbody .t[data-ctl="ctx"]')
    check('math term tooltip: what the number is, then the click action', Z('en', 'The tokens one decode step works on', '') in tip and 'Click to jump' in tip, tip)
    pg.focus('#kvb-mathbody .t[data-ctl="ctx"]'); pg.keyboard.press('Enter'); pg.wait_for_timeout(700)
    check('Enter on a formula term jumps to the context slider', pg.evaluate("()=>!!document.activeElement.closest('#kvb-ctl-ctx')"))
    tip = hover_tip(pg, '#kvb-mathbody .t[data-ctl="wr"]')
    check('a fixed formula term says it is fixed', 'Fixed in this lab' in tip and '15,009,849,344' in tip, tip)
    def math_terms(want_keys, label, lang='en'):
        pg.evaluate("()=>{document.getElementById('kvb-math').open=true}"); pg.wait_for_timeout(150)
        keys = pg.evaluate("()=>[...document.querySelectorAll('#kvb-mathbody .t')].map(e=>e.dataset.ctl)")
        got = {}
        for i, k in enumerate(keys):
            loc = pg.locator('#kvb-mathbody .t').nth(i); loc.evaluate("e=>e.scrollIntoView({block:'center'})"); pg.wait_for_timeout(120)
            bb = loc.evaluate("el=>{const r=el.getClientRects()[0]; return {x:r.x,y:r.y,w:r.width,h:r.height}}")
            pg.mouse.move(bb['x'] + 3, bb['y'] + 3); pg.mouse.move(bb['x'] + min(8, bb['w'] / 2), bb['y'] + min(6, bb['h'] / 2)); pg.wait_for_timeout(220)
            got.setdefault(k, nb(pg.inner_text('#tip')) if vis(pg, '#tip') else '')
            pg.mouse.move(2, 2); pg.wait_for_timeout(60)
        bodies = {k: v.split('\n')[0] for k, v in got.items()}
        check(f'{label}: every term has a tooltip that says what it is, then what a click does (R1.4)', set(keys) == set(want_keys) and all(v and (Z(lang, 'Click to', '点击') in v or Z(lang, 'Fixed', '固定不变') in v) for v in got.values()), got)
        check(f'{label}: no two terms share a tooltip (R6.15)', len(set(bodies.values())) == len(bodies), bodies)
        return got
    got = math_terms(['kv2', 'layers', 'kvheads', 'hdim', 'bytes2', 'ctx', 'wl', 'tok1', 'wh', 'attnpair', 'pairs1', 'effc', 'wr', 'kvread', 'effm', 'tokall', 'pairsall', 'kvnone'], 'Show the math at 8K')
    check('Show the math at 8K: the pairs’ derivation is in the term it explains, and no gray note is left under the formulas (review 8 point 3)',
          '1 + 2 + … + 8,192 = 8,192 × 8,193 ÷ 2 query–key pairs' in got.get('pairsall', '') and 'Click to jump to “Context”' in got.get('pairsall', '')
          and pg.evaluate("()=>document.querySelectorAll('#kvb-mathbody .small, #kvb-mathbody .muted').length") == 0, got.get('pairsall'))
    # H2 (2026-10-09): Show the math is small tables (name and typeset formula, the numbers under it, the result); read as one line
    mt = ' '.join(nb(pg.inner_text('#kvb-mathbody')).split())
    check('Show the math at 8K: the two memory lines differ by the cache read, and each step names the term that decides it (review 8 point 4)',
          '(15,009,849,344 + 8,191 × 131,072) ÷ 2.01 TB/s = 16.083 GB ÷ 2.01 TB/s = 8.00 ms' in mt and '(15,009,849,344 + 0 × 131,072) ÷ 2.01 TB/s = 15.010 GB' in mt
          and '= 8.0 ms, set by memory' in mt and '= 266.7 ms, set by arithmetic' in mt, mt)
    ovd = pg.evaluate("()=>[...document.querySelectorAll('#kvb-mathbody .def')].filter(e=>e.textContent.startsWith('Step time')).map(e=>e.getAttribute('data-tipx'))")
    check('Show the math: “Step time” explains on hover why it is the longer of the two (the GPU reads and computes at the same time)', len(ovd) == 2 and all('at the same time' in x for x in ovd), ovd)
    # H1/H2 (the owner, 2026-10-09): every formula typeset as MathML, no TeX command shown as a word, a hairline between the blocks
    fx = pg.evaluate(r"""()=>{const b=document.getElementById('kvb-mathbody'), m=[...b.querySelectorAll('math')].map(x=>x.textContent);
      const blocks=[...b.querySelectorAll('.kvb-mb')], ruled=blocks.slice(1).filter(x=>getComputedStyle(x).borderTopStyle==='solid'&&parseFloat(getComputedStyle(x).borderTopWidth)>=1).length;
      return {n:m.length, leak:m.filter(x=>/\\|frac|left|right|text\{/.test(x)), blocks:blocks.length, ruled, names:[...b.querySelectorAll('tr.kvb-mname th')].map(x=>x.textContent.trim())}}""")
    check('H1/H2: Show the math at 8K: the formulas are MathML, no TeX leaks; 4 blocks with a hairline between them; each formula has its name in words',
          fx['n'] >= 20 and not fx['leak'] and fx['blocks'] == 4 and fx['ruled'] == 3 and fx['names'][:4] == ['KV cache per token', 'KV cache for the context', 'Compute time', 'Memory time'], fx)
    pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(150)
    pg.evaluate("()=>{location.hash='kv-basics~t_9_1_0_8192'}"); pg.wait_for_timeout(400)
    math_terms(['skv2', 'slayers', 'sheads', 'shd', 'n', 'sumn', 'gauss', 'pass1', 'reads', 'mask'], 'Show the math, small model')
    # the same in Chinese: every term its own tip, 维 for vector lengths (R1.4, R6.15, R6.17)
    pg.click('.langsw button[data-lang="zh"]'); pg.wait_for_timeout(120)
    pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(150)
    pg.evaluate("()=>{location.hash='kv-basics~r_9_1_1_8192'}"); pg.wait_for_timeout(400)
    gz = math_terms(['kv2', 'layers', 'kvheads', 'hdim', 'bytes2', 'ctx', 'wl', 'tok1', 'wh', 'attnpair', 'pairs1', 'effc', 'wr', 'kvread', 'effm', 'tokall', 'pairsall', 'kvnone'], 'zh: Show the math at 8K', 'zh')
    mz = ' '.join(nb(pg.inner_text('#kvb-mathbody')).split())
    check('zh: Show the math at 8K: the pairs’ derivation in its term, head_dim named in the formula, the deciding term named on each line',
          '一共 1 + 2 + … + 8,192 = 8,192 × 8,193 ÷ 2 对 query 和 key' in gz.get('pairsall', '') and 'head_dim' in mz and '= 2 × 32 × 8 × 128 × 2 = 131,072 字节' in mz
          and '= 8.0 ms，由读显存决定' in mz and '(15,009,849,344 + 0 × 131,072) ÷ 2.01 TB/s' in mz and '读写显存' not in mz and '= 266.7 ms，由计算决定' in mz, mz[-400:])
    pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(150)
    pg.evaluate("()=>{location.hash='kv-basics~t_9_1_0_8192'}"); pg.wait_for_timeout(400)
    math_terms(['skv2', 'slayers', 'sheads', 'shd', 'n', 'sumn', 'gauss', 'pass1', 'reads', 'mask'], 'zh: Show the math, small model', 'zh')
    pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(120)
    # r12 N9 (the Fidelity popover’s “Show the math” link) went with the top Fidelity card (owner, 2026-10-11 19:28 UTC)
    # terms defined on hover (R2.8)
    pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(150)
    pg.evaluate("()=>{location.hash='kv-basics'}"); pg.wait_for_timeout(300)
    pg.click('#kvb-app .modes button[data-mode="guided"]'); pg.wait_for_timeout(150)
    for i, term in [(0, 'decode step'), (0, 'start token <s>'), (1, 'token-passes'), (3, 'causal mask'), (3, 'an encoder such as BERT'), (4, 'KV cache'), (5, 'KV heads')]:
        go_part(pg, i)
        sel = f'#kvb-rail .step.current .def'
        texts = pg.evaluate(f"()=>[...document.querySelectorAll('{sel}')].map(e=>e.textContent)")
        check(f'part {i + 1} defines “{term}” where it first appears', term in texts, texts)
    # the home tile
    pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(250)
    pg.evaluate("()=>{const g=[...document.querySelectorAll('#domStrip [data-world=\"models\"]')].find(e=>e.offsetParent); if(g) g.click();}"); pg.wait_for_timeout(300)
    t = pg.inner_text('#tile-kv-basics .tz-read')
    # the tile opens on 6 tokens, where every square shows (product owner, 2026-10-02: at 64 the squares merged into a gray band)
    check('tile: opens on 6 tokens, 21 vs 6, at most 25 words', '21' in t and '6' in t and '2,080' not in t and len(t.split()) <= 25, t)
    check('tile: at 6 tokens every square is drawn on its own', pg.evaluate("()=>document.querySelector('#tile-kv-basics svg').classList.contains('kvb-tzcells')"))
    pg.locator('#tile-kv-basics').screenshot(path=str(SH / 'tile_en.png'))
    notes_ = pg.evaluate("()=>[...document.querySelectorAll('#tile-kv-basics .tz-note')].map(n=>n.textContent)")
    check('tile: 2–4 notes, the second says the counts are a formula', 2 <= len(notes_) <= 4 and 'formula' in notes_[1], notes_)
    rules(pg, 'tile'); [rule_hits.append('tile: ' + h) for h in pg.evaluate(RULES, '#tile-kv-basics') + pg.evaluate(COLONS, '#tile-kv-basics')]
    pg.click('.langsw button[data-lang="zh"]'); pg.wait_for_timeout(300)
    pg.locator('#tile-kv-basics').scroll_into_view_if_needed(); pg.locator('#tile-kv-basics').screenshot(path=str(SH / 'tile_zh.png'))
    [rule_hits.append('tile zh: ' + h) for h in pg.evaluate(RULES, '#tile-kv-basics') + pg.evaluate(COLONS, '#tile-kv-basics')]
    pg.set_viewport_size({'width': 390, 'height': 844}); pg.wait_for_timeout(300)
    pg.locator('#tile-kv-basics').scroll_into_view_if_needed(); pg.locator('#tile-kv-basics').screenshot(path=str(SH / 'tile_zh_390.png'))
    check('tile at 390: no overflow', overflow(pg) <= 0, overflow(pg))
    pg.close()
    b.close()

# The owner's review of 2026-10-09 (R6.37, R6.33): no row of “Show the math” scrolls sideways at the sizes this test visits, in every
# guided part, Scenarios and the challenge. On a box too narrow for the one-line table, the lab sets the name on its own line and breaks
# a long formula before an operator, the numbers still under the terms they replace.
MSCROLL = r"""r=>{const out=[]; for(const d of document.querySelectorAll(r+' details.math')){if(!d.offsetParent) continue;
  for(const e of d.querySelectorAll('*')){const cs=getComputedStyle(e); if(!/auto|scroll/.test(cs.overflowX)||!e.offsetParent) continue;
    const o=e.scrollWidth-e.clientWidth; if(o>1) out.push(o+' px: '+(e.textContent||'').trim().replace(/\s+/g,' ').slice(0,40));}}
  const pg=document.documentElement.scrollWidth-innerWidth; if(pg>1) out.push('the page scrolls sideways by '+pg+' px'); return out;}"""
# a formula on a line of its own in a card (part 7's “KV cache per token = 2 × layers × …”): cut before its operators into pieces that
# wrap (.kv-fx, as the kit's KV cache lab writes it), never a box scrolled sideways (R6.37, kit_math_scroll's second probe; polish round 4)
FXSCROLL = r"""()=>{const fs=[...document.querySelectorAll('#view-dyn aside.rail .step.current .kvb-fx')].filter(f=>f.offsetParent);
  return {bad:fs.filter(f=>f.scrollWidth-f.clientWidth>1).map(f=>(f.scrollWidth-f.clientWidth)+' px: '+f.textContent.trim().replace(/\s+/g,' ').slice(0,40)),
    split:fs.map(f=>f.classList.contains('kv-fx')?f.querySelectorAll(':scope>math').length:0)};}"""
with sync_playwright() as _p:
    _b = _p.chromium.launch()
    for (_w, _h, _lang) in [(1440, 900, 'en'), (1440, 900, 'zh'), (900, 900, 'en'), (390, 844, 'en'), (390, 844, 'zh')]:
        _pg = _b.new_page(viewport={'width': _w, 'height': _h}); _pg.goto(URL); _pg.wait_for_timeout(250)
        _pg.click(f'.langsw button[data-lang="{_lang}"]'); _pg.wait_for_timeout(60)
        _pg.evaluate("()=>{location.hash='kv-basics'}"); _pg.wait_for_timeout(500)
        _n = _pg.evaluate("()=>new Set([...document.querySelectorAll('#view-dyn aside.rail [data-go]')].map(e=>e.dataset.go)).size")
        _bad, _fx, _fx7 = [], [], None
        for _mode, _k in [('guided', k) for k in range(_n)] + [('sandbox', None), ('challenge', None)]:
            if _k is None or _k == 0:
                _pg.evaluate("m=>{const b=[...document.querySelectorAll('#view-dyn .modes button[data-mode=\"'+m+'\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}", _mode); _pg.wait_for_timeout(300)
            if _k is not None:
                _pg.evaluate("k=>{const e=document.querySelector('#view-dyn aside.rail [data-go=\"'+k+'\"]'); if(e) e.click();}", _k)
            _pg.wait_for_timeout(700)
            _pg.evaluate("()=>document.querySelectorAll('#view-dyn details.math').forEach(d=>{d.open=true})"); _pg.wait_for_timeout(400)
            _bad += [f"{_mode}{'' if _k is None else ' part ' + str(_k + 1)}: {x}" for x in _pg.evaluate(MSCROLL, '#view-dyn')]
            if _mode == 'guided':
                _f = _pg.evaluate(FXSCROLL); _fx += [f"part {_k + 1}: {x}" for x in _f['bad']]
                if _k == 6: _fx7 = _f['split']
        check(f'{_w} {_lang}: no row of Show the math scrolls sideways, in any guided part, Scenarios or the challenge (the owner, 2026-10-09; R6.37)', not _bad, _bad[:6])
        check(f'{_w} {_lang}: part 7’s formula in the card (KV cache per token = 2 × layers × …) is cut before its operators and wraps, no card formula scrolls sideways (R6.37, polish round 4)',
              not _fx and _fx7 == [6], (_fx[:4], _fx7))
        _pg.close()
    _b.close()


check('writing rules: no “·” between clauses, no two colons in one sentence (Chinese, and English from r15), no “in this toy”, no 起点 / 控件, no page-behaviour sentence', not rule_hits, sorted(set(rule_hits))[:12])
check('no page errors', not errs, errs[:5])
print('card heights after the action (viewport height, language, part, px):', heights)
for n in notes:
    if n.startswith('FAIL'): print(n)
print(f'{len(notes)} checks, {len(fails)} failed')
sys.exit(1 if fails else 0)
