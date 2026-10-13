"""Flow test for the Transformer ① lab (`transformer`): every mode and guided part at 1440×900 and 390×844 in EN and ZH, dark mode
once, plus 1280 / 900 / 720 layout checks, card heights after each action at 1440×1000 (R3.10: ≤ 941 px, the height under the top bar) and
at 1440×900 (reported; the result must be on screen), the prediction gating and masking
(R3.8), the challenge tiers (R4.4), the language switch keeping state (R8.3), the hash round-trip, the mini result bar at 390, the
jump links and definitions, the home tile, the mode order guided → Scenarios → Challenge (R3.12), Settings that never scroll the page
(R3.2), counts over 6 digits as 8.03B with no exact-value hover (R6.18; the owner, 2026-10-07), 参数 and 维 in Chinese (R6.17), the Connect code without
sideways scrolling (R7.6), and the copy rules on every visible string and tooltip: no "·" joining clauses (R6.14),
never two colons in one Chinese sentence (R6.7), no "in this toy / 在这个玩具里", no 起点 / starting point, no 控件 / Controls, no
sentences about the page itself (R3.4). Screenshots go to shots/transformer/.

Run: python3 tests/transformer_flow.py
"""
import json, pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/transformer.html', '--only', 'transformer'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-transformer.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'transformer.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-transformer.html')
SH = P / 'shots' / 'transformer'; SH.mkdir(parents=True, exist_ok=True)
errs, notes, fails = [], [], []
heights = []


def check(name, ok, info=''):
    notes.append(('OK   ' if ok else 'FAIL ') + name + (f'  [{info}]' if info and not ok else ''))
    if not ok:
        fails.append(name)
        print('FAIL', name, '::', str(info)[:300])


def page(b, w, h, lang, hash_='transformer', dark=False, reduced=False):
    pg = open_drawers(b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light', reduced_motion='reduce' if reduced else 'no-preference'))
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(400)
    return pg


Zf = lambda lang: (lambda en, zh: zh if lang == 'zh' else en)
nth = lambda s, i: ([l for l in s.split('\n') if l.strip()] + ['', '', ''])[i]      # the i-th non-empty line
norm = lambda s: s.replace('‑', '-').replace(' ', ' ')
overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
txt = lambda pg, sel: norm(pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText:''}", sel))
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0&&getComputedStyle(e).visibility!=='hidden'}", sel)
disabled = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return !!e&&e.disabled}", sel)
state = lambda pg: pg.evaluate("()=>window.__tf.state()")


def go(pg, i):
    pg.evaluate(f"()=>document.querySelector('#tf-rail [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(200)


def settle(pg, ms=2500):
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last:
            return
        last = y; pg.wait_for_timeout(120); t += 120


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(120)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def top_of(pg, sel):
    settle(pg)
    return pg.evaluate("s=>{const e=document.querySelector(s); return e?Math.round(e.getBoundingClientRect().top):null}", sel)


def hover_tip(pg, sel):
    loc = pg.locator(sel).first
    loc.evaluate("e=>e.scrollIntoView({block:'center',inline:'nearest',behavior:'instant'})"); pg.wait_for_timeout(150)
    bb = loc.bounding_box(); pg.mouse.move(bb['x'] + 3, bb['y'] + 3); pg.mouse.move(bb['x'] + min(8, bb['width'] / 2), bb['y'] + min(6, bb['height'] / 2)); pg.wait_for_timeout(200)
    return norm(pg.inner_text('#tip')) if vis(pg, '#tip') else ''


# the copy rules, on every visible text node of the lab, the tile and the mini result bar (R6.14, R6.17, R2.1, R3.4, R6.11)
RULES = r"""(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\bControls\b/,'Controls'],
   [/in this toy|在这个玩具/i,'in this toy'],[/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\bhover\b|悬停/i,'hover instruction'],
   [/updates as you|随之更新|会立刻更新|updates at once|chart updates|图会/i,'sentence about the page updating'],[/\bscore(s|d)?\b|得分/i,'“score” (lab ② uses it for q·k)'],[/\bsteps?\b/i,'“step” (reserved for the engine)'],
   [/splits (each token|the hidden size)|heads of \d|切成几|把隐藏维度切成|个数的 head/i,'a head as a slice of the token (lab ② lists it as a misconception)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||!e.offsetParent&&getComputedStyle(e).position!=='fixed') continue;
   if(e.closest('svg, pre, code, #tip')) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 return out;}"""
COLONS = r"""(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const A=window.ASP, out=new Set(), texts=[];
 const blocks=root.querySelectorAll('p,li,button,label,div,span.lbl,span.tag,.status,.dk-read,.cmp-lab,h1,h2,h3,.tz-leg > span,.legend-row > span,.tf-lab,.tf-io');
 blocks.forEach(e=>{if(e.closest('[data-endnotes],pre,code,svg')) return; if(e.matches('.tz-leg,.presets,.choices,.legend-row,.seg-ctl,.modes,.levers,.tf-cfg')||e.querySelector('p,li,div,ul,ol,h1,h2,h3,table')) return; texts.push(e.textContent);});
 root.querySelectorAll('[data-tipx]').forEach(e=>texts.push(e.getAttribute('data-tipx').replace(/<[^>]+>/g,'')));
 for(const t of texts){const sents=t.replace(/\s+/g,' ').split(/(?<=[。？！])|(?<=[.?!])\s+/);
   for(const x0 of sents){const x=x0.replace(/\([A-Z]{2,4}: [^)]*\)/g,'');   // an abbreviation's full name, "(LM: language model)", is not a sentence colon (common r15 (h))
     if(!/[一-鿿]/.test(x)&&!/: .*: /.test(x)) continue; const n=(x.match(/：|: /g)||[]).length; if(n>=2) out.add(x0.trim().slice(0,160));}}
 return [...out];}"""
# R6.18, R6.17, R2.8 (review 8): no count over 6 digits written out outside Show the math and the drill's worked answer, no 万 / 亿,
# no 个数 for parameters or vector lengths, and the site's structural words, in visible text and in tooltips
WORDS = r"""(sel)=>{const root=document.querySelector(sel); if(!root) return []; const out=[], zh=document.documentElement.lang==='zh'||!!document.querySelector('.langsw button[data-lang="zh"][aria-pressed="true"]');
 const skip=e=>!!e.closest('#tf-mathbody, #tf-drill-a, pre, code, .labhead, .crumbs, .tile-q, #tip');   // the lab's title comes from the site's catalog
 const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||skip(e)||(!e.offsetParent&&getComputedStyle(e).position!=='fixed')) continue; const t=n.textContent;
   if(/\d{1,3}(?:,\d{3}){2,}/.test(t)) out.push('number over 6 digits written out: '+t.trim().slice(0,80));
   if(/[万亿]/.test(t)) out.push('万 / 亿 instead of K / M / B: '+t.trim().slice(0,80));}
 const texts=[...root.querySelectorAll('*')].filter(e=>!skip(e)&&(e.offsetParent||getComputedStyle(e).position==='fixed')).map(e=>{let x=''; for(const c of e.childNodes) if(c.nodeType===3) x+=c.textContent; return x;});
 root.querySelectorAll('[data-tipx]').forEach(e=>texts.push(e.getAttribute('data-tipx')));
 for(const t of texts){
   if(/个数(?![的值])/.test(t)&&!/每个数/.test(t)) out.push('个数 (R6.17): '+t.trim().slice(0,80));
   for(const w of ['残差流','输出层','MLP 宽度','residual stream','MLP width','token id ','共享权重','tied weights','复用 embedding 表，','（共用）','LM head 共用','共用的表','共用：','一组权重','set of weights']) if(t.includes(w)) out.push('old word “'+w+'”: '+t.trim().slice(0,80));
   if(/共用/.test(t)&&/embedding|LM head/.test(t)&&!/KV head/.test(t)) out.push('共用 for a tied table (ARCH #15): '+t.trim().slice(0,80));}
 return out;}"""
rule_hits = []


# the owner's clarification (2026-10-07 23:37 UTC): no hover only expands a shown number to its exact value (Show the math excepted)
EXACT_HOVERS = """()=>[...document.querySelectorAll('[data-tipx]')].filter(e=>!e.closest('#tf-mathbody')&&/精确值是|Exactly|exactly [0-9]/.test(e.getAttribute('data-tipx')||''))
  .map(e=>e.textContent.trim().slice(0,30)+' → '+e.getAttribute('data-tipx').slice(0,90))"""


def rules(pg, where):
    ex = pg.evaluate(EXACT_HOVERS)
    check(f'{where}: no hover that only gives a shown number’s exact value (精确值是 / Exactly)', not ex, ex)
    for sel in ('#tf-app', '#dock'):
        for h in pg.evaluate(RULES, sel):
            rule_hits.append(f'{where}: {h}')
        for h in pg.evaluate(COLONS, sel):
            rule_hits.append(f'{where}: two colons in one sentence: {h}')
        for h in pg.evaluate(WORDS, sel):
            rule_hits.append(f'{where}: {h}')


def next_lab(pg, w):
    """R3.11: below the rail card on wide screens; on phones (≤ 760 px) the kit shows a copy after the settings panel instead"""
    return pg.evaluate("""w=>{const r=document.querySelector('#tf-rail .next-lab'), e=document.querySelector('#view-dyn .next-lab-end'), on=x=>!!x&&getComputedStyle(x).display!=='none'&&x.getBoundingClientRect().height>0;
      return w<=760?(on(e)&&!!e.querySelector('.next-lab')&&!on(r)):(on(r)&&!on(e));}""", w)


def masked_ok(pg, Z):
    """R3.8: before part 4 no share anywhere (share bar, metric cards, layer split, Show the math, mini result bar)"""
    m = pg.evaluate("""()=>({share:!!document.querySelector('#tf-share .tf-share-mask'), bar:!!document.querySelector('#tf-share .tf-sbar'),
      mets:[...document.querySelectorAll('#tf-metrics .metric .v')].slice(2).map(v=>v.textContent), split:document.querySelectorAll('#tf-stack .tf-pat-attn').length,
      math:document.getElementById('tf-mathbody').innerText, dock:document.getElementById('dock').innerText})""")
    return m['share'] and not m['bar'] and m['mets'] == ['?', '?', '?'] and m['split'] == 0 and '41,943,040' not in m['math'] and '176,160,768' not in m['math'] and '%' not in m['dock'], m


def run_flow(b, w, h, lang, dark=False):
    Z = Zf(lang); tag = f'{"dark_" if dark else ""}{w}_{lang}'
    pg = page(b, w, h, lang, dark=dark)
    check(f'{tag} title', txt(pg, '#tf-app .labtitle') == Z('Where do an 8B model’s parameters live?', '8B 模型的参数都在哪？'))
    check(f'{tag} starts guided, part 1 of 7', Z('part 1 of 7', '第 1 / 7 部分') in txt(pg, '#tf-rail .rail-top').lower())
    check(f'{tag} header: no Fidelity or Primer button (owner, 2026-10-11 19:28 UTC)', not any(x in txt(pg, '#tf-app .labmeta') for x in ('Fidelity', '保真度', 'Primer', '前置知识')), txt(pg, '#tf-app .labmeta'))
    if w < 760:
        order = pg.evaluate("()=>['#tf-rail','#tf-stage','#tf-controls'].map(s=>Math.round(document.querySelector(s).getBoundingClientRect().top))")
        check(f'{tag} phone layout: card, then chart, then settings', order[0] < order[1] < order[2] and order[0] < h, order)
    c1 = txt(pg, '#tf-rail .step.current')
    # R6.20: the mission keeps the count and names the memory in words; the weights card on the stage shows 15.0 GiB and 16.1 GB
    mem1 = txt(pg, '#tf-metrics .metric[data-k="mem"]')
    check(f'{tag} part 1: count (8.03B), memory in words (the card shows 15.0 GiB, 16.1 GB), seven config numbers as definitions', '8.03B' in c1 and Z('each one takes up GPU memory', '每一个都要占显存') in c1 and 'GiB' not in c1
          and '15.0 GiB' in mem1 and '16.1 GB' in mem1 and pg.evaluate("()=>document.querySelectorAll('#tf-rail .step.current .tf-cfg .def').length") == 7, (c1[:200], mem1))
    check(f'{tag} part 1: at most 3 paragraphs', pg.evaluate("()=>document.querySelectorAll('#tf-rail .step.current .step-b > p').length") <= 3)
    ok, info = masked_ok(pg, Z); check(f'{tag} part 1: every share hidden (R3.8)', ok, info)
    check(f'{tag} part 1: no next-lab footer yet (R3.11)', not pg.evaluate("()=>[...document.querySelectorAll('#tf-rail .next-lab, #view-dyn .next-lab-end')].some(x=>getComputedStyle(x).display!=='none'&&x.getBoundingClientRect().height>0)"))
    cap = txt(pg, '#tf-cap')
    check(f'{tag} stage caption: bars on one scale, full length named (R6.10), 1.05B (R6.18)', '1.05B' in cap and Z('one scale', '同一把尺子') in cap, cap)
    rules(pg, f'{tag} part 1'); pg.evaluate("()=>window.scrollTo(0,0)"); shot(pg, f'{tag}_1_mission')
    check(f'{tag} overflow part 1', overflow(pg) <= 0, overflow(pg))
    # part 2: the prediction gates Next and is not locked
    pg.click('#tf-rail [data-nav="next"]'); pg.wait_for_timeout(250)
    t2 = top_of(pg, '#tf-rail .pbar' if w <= 760 else '#tf-rail .step.current')   # phones (m2): the part bar comes back to the top, the picture under it
    check(f'{tag} Next reveals the new {"part bar" if w <= 760 else "card"} (top on screen)', t2 is not None and 40 <= t2 <= h - 160, t2)
    check(f'{tag} part 2: Next disabled until a choice', disabled(pg, '#tf-rail [data-nav="next"]'))
    opts = pg.evaluate("()=>[...document.querySelectorAll('#tf-rail [data-pred]')].map(b=>b.textContent)")
    check(f'{tag} part 2: four options covering the range', opts == [Z('Under 1/4', '不到 1/4'), Z('1/4 to 1/2', '1/4 到 1/2'), Z('1/2 to 3/4', '1/2 到 3/4'), Z('Over 3/4', '3/4 以上')], opts)
    ok, info = masked_ok(pg, Z); check(f'{tag} part 2: every share hidden before the answer', ok, info)
    tip = hover_tip(pg, '#tf-metrics .metric[data-k="attn"]')
    check(f'{tag} part 2: the hidden card says why, with a subject (R6.15)', Z('This share is hidden until you answer part 2', '这个占比在第 2 部分作答前先隐藏') in tip, tip)
    pg.click('#tf-rail [data-pred="q3"]'); pg.wait_for_timeout(150)
    body = txt(pg, '#tf-rail .step.current')
    check(f'{tag} part 2: pressed, “saved, can still change”, Next enabled', pg.get_attribute('#tf-rail [data-pred="q3"]', 'aria-pressed') == 'true' and not disabled(pg, '#tf-rail [data-nav="next"]')
          and Z('Saved. You can still change it before part 4 shows the answer.', '已记下。第 4 部分揭晓答案之前，还可以改。') in body and Z('Locked', '锁定') not in body)
    pg.click('#tf-rail [data-pred="q2"]'); pg.wait_for_timeout(150)
    check(f'{tag} part 2: the answer can still change', state(pg)['pred'] == 'q2')
    rules(pg, f'{tag} part 2'); shot(pg, f'{tag}_2_predict')
    # part 3: one token down the stack
    pg.click('#tf-rail [data-nav="next"]'); pg.wait_for_timeout(250)
    ok, info = masked_ok(pg, Z); check(f'{tag} part 3: still hidden (shown in part 4)', ok, info)
    check(f'{tag} part 3: Next disabled until the token reaches the LM head', disabled(pg, '#tf-rail [data-nav="next"]'))
    seen = []
    for i in range(6):
        pg.click('#tf-rail [data-walk="down"]'); pg.wait_for_timeout(120)
        seen.append(nth(txt(pg, '#tf-rail [data-obs]'), 1))
        tk = pg.evaluate("()=>{const t=document.querySelector('#tf-stack .tf-tok'), on=document.querySelector('#tf-stack .tf-on'); return {tok:!!t&&!t.hidden, on:!!on, fl:[...document.querySelectorAll('#tf-stack .tf-fl.tf-on')].map(e=>e.textContent)}}")
        check(f'{tag} part 3 stop {i + 2}: the token marker and the highlighted block are drawn', tk['tok'] and tk['on'], tk)
        if i in (1, 2):   # stops 3 and 4: the order inside a layer shows which half the token is in (R1.9)
            want = ['RMSNorm', Z('attention block', 'attention 块'), '⊕'] if i == 1 else ['RMSNorm', Z('MLP block', 'MLP 块'), '⊕']   # ⊕ as in the lab Architecture overview (ARCH #21)
            check(f'{tag} part 3 stop {i + 2}: the layer strip outlines {want[1]}', tk['fl'] == want, tk['fl'])
        if i == 1 and w >= 1000:
            shot(pg, f'{tag}_3_walk_attention')
        if i == 1 and w < 760:   # R3.2: on phones the mini result bar is "3 / 7" and one sentence; the walk squares give it their width
            pg.evaluate("()=>document.getElementById('tf-controls').scrollIntoView({block:'center'})"); pg.wait_for_timeout(500)
            dk = pg.evaluate("()=>{const d=document.getElementById('dock'), v=document.getElementById('dkVis'), r=document.getElementById('dkRead'); return {shown:!d.hidden, h:Math.round(d.getBoundingClientRect().height), vis:getComputedStyle(v).display, st:document.getElementById('dkStatus').innerText.trim(), rd:r.innerText, lines:Math.round(r.getBoundingClientRect().height/(parseFloat(getComputedStyle(r).lineHeight)||1.3*parseFloat(getComputedStyle(r).fontSize)))}}")
            check(f'{tag} part 3 dock: “3 / 7”, no walk squares under the sentence, at most two lines, at most 56 px tall', dk['shown'] and dk['st'] == '3 / 7' and dk['vis'] == 'none' and dk['lines'] <= 2 and dk['h'] <= 56 and Z('layer 1, attention block', '第 1 层的 attention 块') in dk['rd'], dk)
            shot(pg, f'{tag}_3_dock')
            pg.evaluate("()=>document.getElementById('tf-rail').scrollIntoView({block:'start'})"); pg.wait_for_timeout(300)
        if i == 1:
            lt = norm(pg.evaluate("()=>{const s=[...document.querySelectorAll('#tf-legend > span')].find(e=>e.querySelector('.tf-sw-on')); return s?(s.getAttribute('data-tipx')||''):''}"))
            check(f'{tag} part 3: the outline legend says “thick”, no colour word (R5.7)', Z('A thick outline marks', '粗框标出') in lt and not re.search(r'dark|light|深色|浅色', lt), lt)
    check(f'{tag} part 3: every stop says something new', len(set(seen)) == 6, seen)
    obs = txt(pg, '#tf-rail [data-obs]')
    out3 = txt(pg, '#tf-stack .tf-out')   # R6.20: the shapes are on the stage's labels, the stop says it in words
    check(f'{tag} part 3 end: the LM head gives one logit per vocabulary token (128,256 on the stage); “ Paris” is token 12366', 'logit' in obs and '12366' in obs and Z('the size of the embedding table', '和 embedding 表一样大') in obs
          and '128,256' in out3 and '12366' in out3, (obs, out3))
    check(f'{tag} part 3 end: Next enabled, Down disabled', not disabled(pg, '#tf-rail [data-nav="next"]') and disabled(pg, '#tf-rail [data-walk="down"]'))
    pg.click('#tf-rail [data-walk="up"]'); pg.wait_for_timeout(120)
    check(f'{tag} part 3: going back up keeps Next enabled', not disabled(pg, '#tf-rail [data-nav="next"]'))
    rules(pg, f'{tag} part 3'); shot(pg, f'{tag}_3_walk_end')
    check(f'{tag} overflow part 3', overflow(pg) <= 0, overflow(pg))
    # part 4: the reveal, verdict with both options
    pg.click('#tf-rail [data-nav="next"]'); pg.wait_for_timeout(300)
    ob = txt(pg, '#tf-rail [data-obs]')
    check(f'{tag} part 4: result, then the verdict, then why', ob.find('16.7%') < ob.find(Z('You predicted', '你预测的是')) < ob.find(Z('Why', '原因')) and Z('the right answer is “Under 1/4”', '正确答案是“不到 1/4”') in ob, ob)
    rev = pg.evaluate("()=>({bar:!!document.querySelector('#tf-share .tf-sbar'), split:document.querySelectorAll('#tf-stack .tf-pat-attn').length, mets:[...document.querySelectorAll('#tf-metrics .metric .v')].map(v=>v.textContent), badges:[...document.querySelectorAll('#tf-metrics .d')].map(d=>d.textContent).filter(Boolean)})")
    check(f'{tag} part 4: shares and split shown, no ▲▼ on entering a part (R6.6)', rev['bar'] and rev['split'] >= 5 and rev['mets'][2:] == ['70.2%', '16.7%', '13.08%'] and not rev['badges'], rev)
    rules(pg, f'{tag} part 4'); shot(pg, f'{tag}_4_split')
    if w < 760:   # R3.2: on phones the mini result bar is a short status and one sentence (the share bar and the unit repeat the sentence)
        pg.evaluate("()=>document.getElementById('tf-controls').scrollIntoView({block:'center'})"); pg.wait_for_timeout(500)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'), v=document.getElementById('dkVis'); return {shown:!d.hidden, h:Math.round(d.getBoundingClientRect().height), vis:getComputedStyle(v).display, st:document.getElementById('dkStatus').innerText.replace(/\\u00a0/g,' ').trim(), rd:document.getElementById('dkRead').innerText}}")
        check(f'{tag} part 4 dock: at most 56 px tall, no share bar, status without its unit', dk['shown'] and dk['h'] <= 56 and dk['vis'] == 'none' and dk['st'] == '8.03B' and '70.2%' in dk['rd'] and '8.03B' in dk['rd'], dk)
        rules(pg, f'{tag} part 4 dock'); shot(pg, f'{tag}_4_dock')
    # part 5: the vocabulary
    pg.click('#tf-rail [data-nav="next"]'); pg.wait_for_timeout(250)
    check(f'{tag} part 5: Next disabled until Mistral is tried', disabled(pg, '#tf-rail [data-nav="next"]'))
    pg.click('#tf-ic-model button[data-v="mistral-7b"]'); pg.wait_for_timeout(350)
    ob = txt(pg, '#tf-rail [data-obs]')
    check(f'{tag} part 5: 789M fewer, all in the vocabulary tables (7.24B)', '789M' in ob and '7.24B' in ob, ob)
    bd = pg.evaluate("()=>[...document.querySelectorAll('#tf-metrics .d')].map(d=>d.textContent)")
    check(f'{tag} part 5: ▲▼ after the learner’s own change, against the initial setup like the card (R6.1)', any(Z('vs the initial setup', '比初始设置') in x for x in bd) and not any(Z('vs last', '比上次') in x for x in bd), bd)
    check(f'{tag} part 5: stage title names Mistral', 'Mistral-7B-v0.1' in txt(pg, '#tf-st-title'))
    pg.click('#tf-ic-model button[data-v="llama31-8b"]'); pg.wait_for_timeout(250)
    ob2 = txt(pg, '#tf-rail [data-obs]')
    check(f'{tag} part 5: back to Llama, the first sentence changes (R1.9)', nth(ob2, 1) != nth(ob, 1) and Z('initial setup', '初始设置') in nth(ob2, 1), ob2[:200])
    rules(pg, f'{tag} part 5'); shot(pg, f'{tag}_5_vocab')
    # part 6: levers
    pg.click('#tf-rail [data-nav="next"]'); pg.wait_for_timeout(250)
    b6 = txt(pg, '#tf-rail .step.current .step-b')
    check(f'{tag} part 6: the intro asks the question and does not describe the result box (R3.4)', Z('Which grows more', '哪个长得多') in b6 and Z('The result shows', '结果里有') not in b6, b6[:200])
    check(f'{tag} part 6: starts on the initial setup, Next waits for one change', pg.get_attribute('#tf-rail [data-lever="0"]', 'aria-pressed') == 'true' and disabled(pg, '#tf-rail [data-nav="next"]'))
    pg.click('#tf-rail [data-lever="2"]'); pg.wait_for_timeout(400)
    r = pg.evaluate("()=>[...document.querySelectorAll('#tf-rail [data-lever]')].map(b=>({p:b.getAttribute('aria-pressed'),r:b.querySelector('.lv-r').textContent}))")
    check(f'{tag} part 6: the pick is pressed and tried options keep their result (R3.6)', r[2]['p'] == 'true' and r[0]['r'] and r[2]['r'] and not r[1]['r'], r)
    check(f'{tag} part 6: Next enabled after a change; stage names the option', not disabled(pg, '#tf-rail [data-nav="next"]') and Z('twice as deep', '深一倍') in txt(pg, '#tf-st-title'), txt(pg, '#tf-st-title'))
    pg.click('#tf-rail [data-lever="1"]'); pg.wait_for_timeout(400)
    if w >= 1000:
        box = pg.evaluate("()=>{const r=document.querySelector('#tf-rail [data-obs]').getBoundingClientRect(); return [Math.round(r.top),Math.round(r.bottom),innerHeight]}")
        check(f'{tag} part 6: after the click the result box is on screen (R3.10)', box[0] >= 50 and box[0] < box[2] - 100, box)
    rules(pg, f'{tag} part 6'); shot(pg, f'{tag}_6_config')
    # part 7
    pg.click('#tf-rail [data-nav="next"]'); pg.wait_for_timeout(250)
    c7 = txt(pg, '#tf-rail .step.current')
    check(f'{tag} part 7: three misconceptions with numbers, “See the scenarios” (not the challenge, R3.12), the next-lab footer', c7.count('✗') == 3 and vis(pg, '#tf-rail .step.current .stepnav [data-mode-go="sandbox"]')
          and not vis(pg, '#tf-rail [data-mode-go="challenge"]') and Z('See the scenarios', '去看应用场景') in c7 and next_lab(pg, w), (c7[-300:], next_lab(pg, w)))
    sl = pg.get_attribute('#tf-rail .step.current [data-mode-go="sandbox"]', 'data-tipx') or ''
    check(f'{tag} part 7: the Scenarios link says where it goes (R1.1)', sl == Z('Click to switch to Scenarios, where the five models are.', '点击切换到“应用场景”，五个模型都在那里。'), sl)
    rules(pg, f'{tag} part 7'); shot(pg, f'{tag}_7_remember')
    check(f'{tag} overflow part 7', overflow(pg) <= 0, overflow(pg))
    # scenarios, reached from the end of the guided tour (R3.12)
    pg.click('#tf-rail .step.current .stepnav [data-mode-go="sandbox"]'); pg.wait_for_timeout(300)
    check(f'{tag} “See the scenarios” opens Scenarios (R3.12)', state(pg)['mode'] == 'sandbox')
    t8 = top_of(pg, '#view-dyn .bench>.stage' if w <= 760 else '#tf-rail')   # phones (m2): the stage first, the card right under it
    check(f'{tag} the Scenarios {"stage" if w <= 760 else "card"} comes into view', t8 is not None and 40 <= t8 <= h - 160, t8)
    sb = txt(pg, '#tf-rail .railcard')
    check(f'{tag} scenarios: “Pick a scenario”, the lead-in (G1, 2026-10-08), the Settings link, and it ends with “Take the challenge” (R3.12)', Z('Pick a scenario', '选一个应用场景') in sb and Z('Pick a scenario and see how many parameters the model has and how much memory its weights take.', '选一个场景，看看模型有多少参数、权重占多少显存。') in sb and vis(pg, '#tf-rail [data-jump="controls"]')
          and vis(pg, '#tf-rail .railcard .stepnav [data-mode-go="challenge"]') and sb.rstrip().endswith(Z('Take the challenge', '去做挑战')), sb[-120:])
    check(f'{tag} scenarios: the next-lab footer (R3.11)', next_lab(pg, w), next_lab(pg, w))
    firsts = []
    for i in range(5):
        pg.click(f'#tf-rail [data-preset="{i}"]'); pg.wait_for_timeout(300)
        firsts.append(nth(txt(pg, '#tf-insight'), 1))
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#tf-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        check(f'{tag} scenario {i + 1}: only it is pressed', pressed == ['true' if k == i else 'false' for k in range(5)], pressed)
        if w >= 1000:
            bx = pg.evaluate("()=>{const r=document.getElementById('tf-insight').getBoundingClientRect(); return [Math.round(r.top),innerHeight]}")
            settle(pg)
            bx = pg.evaluate("()=>{const r=document.getElementById('tf-insight').getBoundingClientRect(); return [Math.round(r.top),innerHeight]}")
            check(f'{tag} scenario {i + 1}: the result box top is on screen after the click (R3.10)', 50 <= bx[0] <= bx[1] - 80, bx)
            pt = pg.evaluate(f"()=>Math.round(document.querySelector('#tf-rail [data-preset=\"{i}\"]').getBoundingClientRect().top)")
            check(f'{tag} scenario {i + 1}: the option just clicked stays on screen (R3.10)', 56 <= pt <= h - 40, pt)
        if i == 0:   # Llama-3.2-1B ties its LM head to the embedding table: 共享参数 / tied everywhere (ARCH #15), the kit's LM head words first (common r15 (h))
            td = pg.evaluate("""()=>{const lg=[...document.querySelectorAll('#tf-legend > span')].find(e=>e.querySelector('.tf-sw-tied')), m=document.querySelector('#tf-mathbody .t[data-ctl="tied"]'),
              o=document.querySelector('#tf-ctl-tied button[data-v="1"]'); return {lg:lg?lg.textContent:'', tip:lg?lg.getAttribute('data-tipx'):'', math:m?m.textContent:'', opt:o?o.textContent:'', kit:window.ASP.terms.lmhead()}}""")
            check(f'{tag} scenario 1 (tied): legend, its hover, the math term and the Settings option say tied / 共享参数',
                  td['lg'] == Z('LM head = embedding table (tied)', 'LM head = embedding 表（共享参数）') and td['tip'].startswith(td['kit'])
                  and Z('Here the two are tied, so the LM head is drawn as a dashed outline', '这里两者共享参数，所以 LM head 画成虚线框') in td['tip']
                  and td['math'] == Z('0 (LM head tied)', '0（LM head 共享参数）') and td['opt'] == Z('Tied', '共享参数'), td)
        rules(pg, f'{tag} scenario {i + 1}')
    check(f'{tag} scenarios: every pick changes the first sentence', len(set(firsts)) == 5, firsts)
    pg.click('#tf-ctl-p button[data-v="fp8"]'); pg.wait_for_timeout(200)
    check(f'{tag} scenarios: after a change in Settings none is pressed and the card says so', pg.evaluate("()=>[...document.querySelectorAll('#tf-rail [data-preset]')].every(b=>b.getAttribute('aria-pressed')==='false')") and Z('match none', '都不一样') in txt(pg, '#tf-insight'))
    shot(pg, f'{tag}_8_scenarios')
    # challenge, reached from the end of the Scenarios card (R3.12; the tiers themselves are tested in challenge_tiers)
    pg.click('#tf-rail .railcard .stepnav [data-mode-go="challenge"]'); pg.wait_for_timeout(300); settle(pg)   # the reveal is a smooth scroll
    check(f'{tag} “Take the challenge” opens the challenge', state(pg)['mode'] == 'challenge')
    t8 = top_of(pg, '#view-dyn .bench>.stage' if w <= 760 else '#tf-rail')   # phones (m2): the stage, the card right under it
    check(f'{tag} the button reveals the challenge {"stage" if w <= 760 else "card"}', t8 is not None and 40 <= t8 <= h - 160, t8)
    ch = txt(pg, '#tf-rail .railcard')
    check(f'{tag} challenge: goal with hardware, precision, budget; only the layer slider', 'RTX 4090' in ch and '20 GiB' in ch and 'BF16' in ch and pg.evaluate("()=>document.querySelectorAll('#tf-rail input,#tf-rail .seg-ctl:not(.ex-tabs)').length") == 1)
    locked = pg.evaluate("()=>[...document.querySelectorAll('#tf-controls .lockable')].map(e=>getComputedStyle(e).display)")
    check(f'{tag} challenge: other settings hidden, not grayed (R4.2), with a note', all(d == 'none' for d in locked) and vis(pg, '#tf-lock') and vis(pg, '#tf-ctl-L'), locked)
    check(f'{tag} challenge: the next-lab footer (R3.11)', next_lab(pg, w), next_lab(pg, w))
    rules(pg, f'{tag} challenge'); shot(pg, f'{tag}_9_challenge')
    check(f'{tag} overflow challenge', overflow(pg) <= 0, overflow(pg))
    if w < 760:   # mini result bar at 390: acted in the lab, chart off screen
        pg.evaluate("()=>{const i=document.getElementById('tf-ic-L-in'); i.value=40; i.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(200)
        pg.evaluate("()=>document.getElementById('tf-controls').scrollIntoView({block:'center'})"); pg.wait_for_timeout(400)
        dk = pg.evaluate("()=>({shown:!document.getElementById('dock').hidden, text:document.getElementById('dock').innerText.replace(/\\s+/g,' ').trim(), fits:[...document.querySelectorAll('#dock .dock-in > *')].filter(e=>getComputedStyle(e).display!=='none').every(e=>e.getBoundingClientRect().right<=innerWidth+0.5)})")
        check(f'{tag} dock at {w}: shown, fits, names the layers and GiB', dk['shown'] and dk['fits'] and '40' in dk['text'] and 'GiB' in dk['text'], dk)
        rules(pg, f'{tag} dock'); shot(pg, f'{tag}_10_dock')
        pg.click('#dock'); pg.wait_for_timeout(700)
        check(f'{tag} dock click returns to the chart', pg.evaluate("()=>{const r=document.getElementById('tf-graph').getBoundingClientRect(); return r.top<innerHeight&&r.bottom>0}"))
    pg.close()


# the challenge chart's rows: layers, passing (green), the pick, the initial-setup tag
ROWS = """()=>[...document.querySelectorAll('#tf-ch-res .cmp-row')].map(r=>({l:parseInt(r.querySelector('.cmp-lab span').textContent),
  ok:r.classList.contains('ok'),me:r.classList.contains('me'),init:!!r.querySelector('.cmp-tag')}))"""


def challenge_tiers(b, lang):
    Z = Zf(lang); pg = page(b, 1440, 1000, lang)
    pg.click('#tf-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(300)
    setL = lambda L: (pg.evaluate(f"()=>{{const i=document.getElementById('tf-ic-L-in'); i.value={L}; i.dispatchEvent(new Event('input',{{bubbles:true}}))}}"), pg.wait_for_timeout(150))
    setL(48)
    st = txt(pg, '#tf-status')
    check(f'{lang} challenge 48 layers: live progress red, status over by 1.5 GiB', pg.evaluate("()=>document.getElementById('tf-ch-prog').classList.contains('tf-over')") and Z('Over 20 GiB by 1.5 GiB', '超出 20 GiB，多了 1.5 GiB') == st, st)
    pg.click('#tf-rail [data-act="submit"]'); pg.wait_for_timeout(300)
    res = txt(pg, '#tf-ch-res')
    check(f'{lang} tier 3 (over): not yet, a hint with the per-layer size, no answer, a “Show the answer” link', Z('Not yet', '还没达到') in res and '0.406 GiB' in res and '44' not in res and vis(pg, '#tf-ch-res [data-act="reveal"]')
          and txt(pg, '#tf-ch-res [data-act="reveal"]') == Z('Show the answer', '直接看答案') and not vis(pg, '#tf-ch-res .cmp'), res)
    shot(pg, f'1440_{lang}_ch_tier3')
    # the owner's review of batch 8 (2026-10-07): the answer shows only the layer counts that fit, the pick and the initial setup, and a
    # note under the chart counts the rest (52 layers here; 48 is the pick)
    pg.click('#tf-ch-res [data-act="reveal"]'); pg.wait_for_timeout(300)
    rows = pg.evaluate(ROWS)
    check(f'{lang} “Show the answer” at 48: 5 rows, green up to 44 layers, “your pick” on 48, the note counts the 1 left out',
          [x['l'] for x in rows] == [32, 36, 40, 44, 48] and [x['ok'] for x in rows] == [True, True, True, True, False] and rows[4]['me'] and rows[0]['init']
          and txt(pg, '#tf-ch-res .cmp-note') == Z('The other layer count does not fit and is not shown.', '其余 1 种层数放不下，图里没有列出。') and not vis(pg, '#tf-ch-res [data-act="reveal"]'), rows)
    check(f'{lang} chart title names the quantity and the budget, not every layer count', txt(pg, '#tf-ch-res .cmp-title') == Z('BF16 weights of a Llama-3.1-8B-shaped model (budget: 20 GiB on one RTX 4090)', 'Llama-3.1-8B 结构下 BF16 权重的大小（一张 RTX 4090 上的预算是 20 GiB）')
          and '52' not in pg.evaluate("()=>document.querySelector('#tf-ch-res .cmp').getAttribute('aria-label')"), txt(pg, '#tf-ch-res .cmp-title'))
    setL(40)
    res = txt(pg, '#tf-ch-res')
    check(f'{lang} tier 2 (fits, not the most): says 4 more fit, chart shown', Z('It fits', '放得下') in res and Z('4 more layers', '再加 4 层') in res and vis(pg, '#tf-ch-res .cmp'), res[:300])
    setL(44)
    res = txt(pg, '#tf-ch-res')
    # R6.20: the result and the reason in words; the division (20 − 1.957) ÷ 0.406 = 44.4 is in Show the math, one click away
    check(f'{lang} tier 1 (44): goal met at 19.8 GiB, one more layer does not fit, the reason in words, the caveat', Z('Goal met', '达成目标') in res and '19.8 GiB' in res
          and Z('one more would go over', '再加一层就超了') in res and Z('the same memory at any depth', '不管多少层都占同样多的显存') in res and '44.4' not in res
          and Z('counts only the weights', '只算权重') in res and vis(pg, '#tf-ch-res [data-jump="math"]'), res[:500])
    pg.click('#tf-ch-res [data-jump="math"]'); pg.wait_for_timeout(500)
    ml = txt(pg, '#tf-mathbody')
    check(f'{lang} tier 1: the Show-the-math link opens it, with the division (20 − 1.957) ÷ 0.406 = 44.4', pg.evaluate("()=>document.getElementById('tf-math').open") and '(20 − 1.957) ÷ 0.406 = 44.4' in ml, ml[-300:])
    rows = pg.evaluate(ROWS)
    check(f'{lang} chart: only the 4 layer counts that fit, “your pick” on 44, “initial setup” on 32, the note counts the 2 left out (48, 52)',
          [x['l'] for x in rows] == [32, 36, 40, 44] and all(x['ok'] for x in rows) and rows[3]['me'] and rows[0]['init']
          and txt(pg, '#tf-ch-res .cmp-note') == Z('The other 2 layer counts do not fit and are not shown.', '其余 2 种层数放不下，图里没有列出。'), rows)
    check(f'{lang} chart legend names green, gray and the dashed budget', Z('fits in 20 GiB', '20 GiB 以内，放得下') in txt(pg, '#tf-ch-res .cmp-leg') and Z('budget', '预算') in txt(pg, '#tf-ch-res .cmp-leg'))
    setL(41)
    rows = pg.evaluate(ROWS)
    check(f'{lang} chart adds an off-list pick (41) as its own row, among the counts that fit', [x['l'] for x in rows] == [32, 36, 40, 41, 44] and all(x['ok'] for x in rows) and rows[3]['me'], rows)
    rules(pg, f'{lang} challenge tiers'); shot(pg, f'1440_{lang}_ch_tier2_chart')
    # language switch keeps the challenge state
    other = 'en' if lang == 'zh' else 'zh'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(300)
    s = state(pg)
    check(f'{lang}→{other}: challenge, 41 layers and the submitted result kept', s['mode'] == 'challenge' and s['L'] == 41 and s['chSubmitted'] and vis(pg, '#tf-ch-res .cmp'), s)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(250)
    # leaving the challenge restores the setup from before it
    pg.click('#tf-app .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(250)
    check(f'{lang}: leaving the challenge restores the earlier setup', state(pg)['L'] == 32 and state(pg)['mode'] == 'sandbox')
    pg.close()


def submit_scroll(b, lang, w, h):
    """R3.10 (review r13 M4): after Submit the verdict's first paragraph is on screen and the layer slider stays below the top bar and
    the mini result bar, since the verdict may ask for more layers. Real mouse: wheel the button into view (as is, or at the bottom
    edge of the screen), then press it"""
    Z = Zf(lang)
    for L in (40, 48, 44):
        for where in ('as is', 'bottom edge'):
            pg = page(b, w, h, lang)
            pg.click('#tf-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(300); settle(pg)
            if w <= 760:   # phones (m2): the mode switch shows the stage; the learner scrolls down to the task card under it
                pg.evaluate("()=>document.querySelector('#tf-rail [data-act=\"submit\"]').scrollIntoView({block:'center'})"); pg.wait_for_timeout(300); settle(pg)
            pg.evaluate(f"()=>{{const i=document.getElementById('tf-ic-L-in'); i.value={L}; i.dispatchEvent(new Event('input',{{bubbles:true}}))}}"); pg.wait_for_timeout(150)
            if where == 'bottom edge':
                pg.mouse.move(w / 2, h / 2)
                for _ in range(12):   # wheel so that the button's bottom sits about 30 px above the screen's bottom edge (as far as the page allows)
                    bb = pg.locator('#tf-rail [data-act="submit"]').bounding_box()
                    d = (bb['y'] + bb['height']) - (h - 30)
                    if abs(d) <= 10:
                        break
                    y0 = pg.evaluate('()=>scrollY'); pg.mouse.wheel(0, d); pg.wait_for_timeout(250); settle(pg)
                    if pg.evaluate('()=>scrollY') == y0:
                        break
                settle(pg)
            bb = pg.locator('#tf-rail [data-act="submit"]').bounding_box()
            pg.mouse.click(bb['x'] + bb['width'] / 2, bb['y'] + bb['height'] / 2); pg.wait_for_timeout(700); settle(pg)
            m = pg.evaluate("""()=>{const s=document.getElementById('tf-ic-L').getBoundingClientRect(), vv=document.querySelector('#tf-ch-res > *'), v=vv?vv.getBoundingClientRect():{top:-1,bottom:-1},
              bar=document.querySelector('.topbar').getBoundingClientRect(), dk=document.getElementById('dock'), d=dk.hidden?null:dk.getBoundingClientRect();
              return {sTop:Math.round(s.top), sBot:Math.round(s.bottom), vTop:Math.round(v.top), vBot:Math.round(v.bottom), cover:Math.round(Math.max(bar.bottom, d?d.bottom:0)), vh:innerHeight}}""")
            verdict = txt(pg, '#tf-ch-res > p')
            want = {40: Z('It fits', '放得下'), 48: Z('Not yet', '还没达到'), 44: Z('Goal met', '达成目标')}[L]
            check(f'{lang} {w}×{h} submit {L} layers ({where}): the verdict “{want}” is on screen and the slider stays under the bars, not behind them',
                  want in verdict and m['vTop'] >= m['cover'] and m['vBot'] <= m['vh'] and m['sTop'] >= m['cover'] and m['sBot'] <= m['vh'], m)
            if L == 40 and where == 'bottom edge':
                shot(pg, f'{w}_{lang}_ch_submit_40')
            pg.close()


def settings_no_scroll(b, lang, w, h):
    """R3.2 (review 8): a change under Settings updates the stage in place and never scrolls the page, in the guided tour and in Scenarios"""
    for where in ('guided', 'sandbox'):
        pg = page(b, w, h, lang)
        if where == 'guided':
            go(pg, 1); pg.click('#tf-rail [data-pred="q2"]'); go(pg, 3)
        else:
            pg.click('#tf-app .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(300); pg.click('#tf-rail [data-preset="1"]'); pg.wait_for_timeout(300)
        pg.evaluate("()=>document.getElementById('tf-ctl-d').scrollIntoView({block:'center'})"); pg.wait_for_timeout(400); settle(pg)
        moved = []
        # (what, the element brought into view first by the test itself, the change)
        acts = [('model 70B', '#tf-model', lambda: pg.select_option('#tf-model', 'llama31-70b')), ('model 8B', '#tf-model', lambda: pg.select_option('#tf-model', 'llama31-8b'))]
        for key, v in (('L', 40), ('d', 5), ('f', 3), ('V', 2)):
            acts.append((f'slider {key}', f'#tf-ctl-{key}', lambda key=key, v=v: pg.evaluate(f"()=>{{const i=document.getElementById('tf-ctl-{key}-in'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}}))}}")))
        for sel_ in ('#tf-ctl-tied button[data-v="1"]', '#tf-ctl-H button[data-v="16"]', '#tf-ctl-KV button[data-v="4"]', '#tf-ctl-p button[data-v="fp8"]', '#tf-controls [data-act="reset"]'):
            acts.append((sel_, sel_, lambda sel_=sel_: pg.click(sel_)))
        pos = lambda t: pg.evaluate("s=>Math.round(document.querySelector(s).getBoundingClientRect().top)", t)
        for name, target, act in acts:
            pg.locator(target).first.scroll_into_view_if_needed(); pg.wait_for_timeout(150); settle(pg)
            y0, p0 = pg.evaluate('()=>scrollY'), pos(target); act(); pg.wait_for_timeout(350); settle(pg); y1, p1 = pg.evaluate('()=>scrollY'), pos(target)
            # since 2026-10-11 Settings sit under the card when every part is shown (tests), so a card that changes height moves scrollY; the control must still stay put
            if abs(p1 - p0) > 1:
                moved.append((name, 'scrollY', y0, y1, 'control top', p0, p1))
        check(f'{lang} {w} {where}: no change under Settings moves the page under the learner (R3.2): the control stays where it was', not moved, moved)
        pg.close()


def lang_and_hash(b, lang):
    Z = Zf(lang); pg = page(b, 1440, 900, lang)
    go(pg, 1); pg.click('#tf-rail [data-pred="q1"]'); go(pg, 2)
    for _ in range(3):
        pg.click('#tf-rail [data-walk="down"]'); pg.wait_for_timeout(80)
    other = 'en' if lang == 'zh' else 'zh'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(300)
    s = state(pg)
    check(f'{lang}→{other}: part 3, the prediction and the token’s position kept', s['step'] == 2 and s['pred'] == 'q1' and s['walk'] == 3 and s['mode'] == 'guided', s)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(250)
    pg.click('#tf-app .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(250)
    pg.select_option('#tf-model', 'llama31-70b'); pg.wait_for_timeout(200)
    pg.click('#tf-ctl-p button[data-v="fp8"]'); pg.click('#tf-ctl-tied button[data-v="1"]'); pg.wait_for_timeout(200)
    pg.evaluate("()=>{const i=document.getElementById('tf-ctl-L-in'); i.value=40; i.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(200)
    hsh = pg.evaluate("()=>location.hash")
    check(f'{lang} hash written', hsh == '#transformer~128256_8192_40_64_8_28672_1_fp8', hsh)
    check(f'{lang} custom config: the model select says so', pg.evaluate("()=>document.getElementById('tf-model').value") == 'custom' and Z('Custom config', '自定义配置') in txt(pg, '#tf-st-title'))
    pg.close()
    pg = page(b, 1440, 900, lang, hash_=hsh[1:])
    s = state(pg)
    check(f'{lang} hash round-trip: opens that config in scenarios', s['mode'] == 'sandbox' and (s['V'], s['d'], s['L'], s['H'], s['KV'], s['f'], s['tied'], s['p']) == (128256, 8192, 40, 64, 8, 28672, True, 'fp8'), s)
    check(f'{lang} hash round-trip: controls follow', pg.evaluate("()=>document.getElementById('tf-ctl-L-in').value") == '40' and pg.get_attribute('#tf-ctl-tied button[data-v="1"]', 'aria-pressed') == 'true')
    pg.close()
    pg = page(b, 1440, 900, lang, hash_='transformer~bad_token')
    check(f'{lang} a bad token opens the guided tour', state(pg)['mode'] == 'guided')
    pg.close()


def card_heights(b, lang, H=1000):
    """R3.10: after each part's action the result is on screen; at 1440 × 1000 the card also fits under the top bar (941 px).
    At 1440 × 900 the heights are only recorded for the delivery note (the layout depends on the width, so they are the same)."""
    Z = Zf(lang); pg = page(b, 1440, H, lang)
    acts = {1: lambda: pg.click('#tf-rail [data-pred="q2"]'), 2: lambda: [pg.click('#tf-rail [data-walk="down"]') for _ in range(6)],
            4: lambda: pg.click('#tf-ic-model button[data-v="mistral-7b"]'), 5: lambda: pg.click('#tf-rail [data-lever="1"]')}
    clicked = {1: '#tf-rail [data-pred="q2"]', 4: '#tf-ic-model button[data-v="mistral-7b"]', 5: '#tf-rail [data-lever="1"]'}
    for i in range(7):
        go(pg, i); pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(100)
        pg.evaluate("()=>{const c=document.querySelector('#tf-rail .step.current'); if(c) c.scrollIntoView({block:'start'}); window.scrollBy(0,-70)}"); pg.wait_for_timeout(100)
        if i in acts:
            acts[i](); pg.wait_for_timeout(450); settle(pg)
            if i in clicked:   # R3.10: the option just clicked is not scrolled away
                ct = pg.evaluate(f"()=>Math.round(document.querySelector('{clicked[i]}').getBoundingClientRect().top)")
                check(f'{lang} 1440×{H} part {i + 1}: the option just clicked stays on screen', 56 <= ct <= H - 40, ct)
        m = pg.evaluate("""()=>{const c=document.querySelector('#tf-rail .step.current').getBoundingClientRect(), o=document.querySelector('#tf-rail [data-obs]'), ob=o&&!o.hidden?o.getBoundingClientRect():null, bar=document.querySelector('.topbar');
          return {h:Math.round(c.height), top:Math.round(c.top), obsTop:ob?Math.round(ob.top):null, obsBottom:ob?Math.round(ob.bottom):null, vh:innerHeight, bar:bar?bar.offsetHeight:0}}""")
        heights.append((lang, H, i + 1, m['h']))
        if H == 1000:
            check(f'{lang} part {i + 1}: card height {m["h"]} px ≤ {m["vh"] - m["bar"]} px, 1440 × 1000 under the top bar (R3.10)', m['h'] <= m['vh'] - m['bar'], m)
        if m['obsTop'] is not None and i in acts:
            check(f'{lang} 1440×{H} part {i + 1}: the result is on screen after the action', 50 <= m['obsTop'] < m['vh'] - 60, m)
    shot(pg, f'1440x{H}_{lang}_heights_last')
    pg.close()


def layouts(b):
    for w in (1280, 900, 720):
        for lang in ('en', 'zh'):
            pg = page(b, w, 900, lang)
            go(pg, 3)
            check(f'{w} {lang}: no horizontal overflow', overflow(pg) <= 0, overflow(pg))
            if w <= 1180 and w > 760:
                pos = pg.evaluate("()=>['#tf-rail','#tf-stage','#tf-controls'].map(s=>{const r=document.querySelector(s).getBoundingClientRect(); return [Math.round(r.left),Math.round(r.top)]})")
                check(f'{w} {lang}: cards left, chart right (R3.2)', pos[0][0] < pos[1][0] and pos[2][0] < pos[1][0], pos)
            lab = pg.evaluate("()=>[...document.querySelectorAll('#tf-stack .tf-lab, #tf-share .tf-skey span')].filter(e=>e.scrollWidth>e.clientWidth+1).length")
            check(f'{w} {lang}: stack labels do not overflow', lab == 0, lab)
            shot(pg, f'{w}_{lang}_part4')
            pg.close()


TILE_MODELS = ('llama32-1b', 'llama31-8b', 'llama31-70b')
TILE_L = {'llama32-1b': 16, 'llama31-8b': 32, 'llama31-70b': 80}
# the tile's picture, measured: its height, cells, blocks, text that does not fit, what is animating and when it ends
TILE_GEO = """()=>{const t=document.getElementById('tile-transformer'), q=s=>t.querySelector(s), R=e=>e.getBoundingClientRect();
  const v=R(q('.tf-tz-vis')), tr=R(t), cs=[...t.querySelectorAll('.tf-tz-grid:not(.tf-tz-old) .tf-tz-cell')].map(R), an=t.getAnimations({subtree:true});
  const cut=[...t.querySelectorAll('.tf-tz-cap, .tz-leg > span, .tz-read, .seg-ctl button')].filter(e=>e.scrollWidth>e.clientWidth+1).map(e=>e.textContent.slice(0,40));
  return {vh:v.height, vin:v.left>=tr.left-0.5&&v.right<=tr.right+0.5, cells:cs.length, cw:cs.length?Math.min(...cs.map(r=>r.width)):0, ch:cs.length?Math.min(...cs.map(r=>r.height)):0,
    emb:R(q('.tf-tz-emb')).width, head:R(q('.tf-tz-head')).width, cut, run:an.filter(a=>a.playState==='running').length,
    end:Math.max(0,...an.map(a=>{const c=a.effect&&a.effect.getComputedTiming(); return c?c.endTime:0;})),
    hidden:[...t.querySelectorAll('.tf-tz-grid:not(.tf-tz-old) .tf-tz-cell')].filter(c=>getComputedStyle(c).opacity!=='1').length,
    pend:q('.tf-tz-frame').classList.contains('tf-tz-pend'), grids:t.querySelectorAll('.tf-tz-grid').length}}"""
FOCUS = "()=>{const e=document.activeElement, f=e&&e.closest('.tf-tz-frame'); return {inside:!!f, cls:e?e.className:'', tip:e&&e.dataset?(e.dataset.tipx||''):'', stops:f?[...f.querySelectorAll('[tabindex=\"0\"]')].length:-1}}"


def tile_page(b, w, h, lang, dark=False, reduced=False, tilew=None):
    pg = page(b, w, h, lang, hash_='map', dark=dark, reduced=reduced)
    if tilew:   # a tile of exactly this width (tiles are 280–440 px wide)
        pg.add_style_tag(content=f'#tiles{{grid-template-columns:{tilew}px!important}}')
    pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="models"], #domStrip [data-world="models"]')].find(e=>e.offsetParent); if(g) g.click(); else {const m=document.querySelector('#mapSvg .world[data-world="models"]'); if(m) m.dispatchEvent(new MouseEvent('click',{bubbles:true}));}}"""); pg.wait_for_timeout(300)
    pg.evaluate("()=>document.getElementById('tile-transformer').scrollIntoView({block:'center'})")
    return pg


def tile(b):
    for lang in ('en', 'zh'):
        Z = Zf(lang)
        for w in (1440, 390):
            pg = page(b, w, 900, lang, hash_='map')
            pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="models"], #domStrip [data-world="models"]')].find(e=>e.offsetParent); if(g) g.click(); else {const m=document.querySelector('#mapSvg .world[data-world="models"]'); if(m) m.dispatchEvent(new MouseEvent('click',{bubbles:true}));}}"""); pg.wait_for_timeout(300)
            pg.evaluate("()=>document.getElementById('tile-transformer').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
            reads = []
            for k in TILE_MODELS:
                pg.click(f'#tile-transformer .seg-ctl button[data-v="{k}"]'); pg.wait_for_timeout(120)
                rd = txt(pg, '#tile-transformer .tz-read'); reads.append(rd)
                words = len(re.findall(r'[A-Za-z0-9.,%\-]+', rd)) if lang == 'en' else len(re.findall(r'[一-鿿]', rd)) + len(re.findall(r'[A-Za-z0-9%]+', rd))
                check(f'tile {lang} {w} {k}: one sentence of at most 25 words (CONTRACT §5, R11.5)', rd.count('. ') == 0 and rd.count('。') <= 1 and words <= (25 if lang == 'en' else 50), (words, rd))
                pg.wait_for_timeout(1000)
                g = pg.evaluate(TILE_GEO)
                check(f'tile {lang} {w} {k}: the picture is at most 130 px tall, inside the tile, {TILE_L[k]} cells all drawn, no text cut off',
                      g['vh'] <= 130 and g['vin'] and g['cells'] == TILE_L[k] and g['hidden'] == 0 and not g['cut'] and g['run'] == 0, g)
            check(f'tile {lang} {w}: the sentence changes with the model', len(set(reads)) == 3, reads)
            leg = pg.evaluate("()=>[...document.querySelectorAll('#tile-transformer .tz-leg span')].map(s=>s.textContent)")
            check(f'tile {lang} {w}: legend names every pattern', len(leg) >= 3, leg)
            check(f'tile {lang} {w}: chip says Formula; no “Assumptions & sources” on the tile (owner 2026-10-13: the tile is one button, no “Assumptions & sources” on it)', Z('Formula', '公式计算') in txt(pg, '#tile-transformer .tile-meta')
                  and pg.evaluate("()=>!document.querySelector('#tile-transformer .tz-more')&&!/假设和来源|Assumptions (&|and) sources/.test(document.querySelector('#tile-transformer').innerText)"))
            # the legend: each item says what its part of the model does on hover; the attention block in the home hero's words (A #20, B-19)
            for key_, want_ in (('emb', Z('The embedding table turns each token ID into a vector', 'embedding 表把每个 token ID 变成一个向量')),
                                ('lay', Z('an attention block, where each token takes information from itself and the tokens before it, and an MLP block', 'attention 块（每个 token 在这里从它自己和前面的 token 取信息）和一个 MLP 块')),
                                ('head', Z('The LM head (LM: language model) turns the final vector into one score for every token in the vocabulary.', 'LM head（LM 即语言模型）把最后的向量变成词表里每个 token 的一个得分。'))):
                tip = hover_tip(pg, f'#tile-transformer .tz-leg [data-k="{key_}"]')
                check(f'tile {lang} {w}: hovering “{key_}” in the legend says what it is', want_ in tip and not re.search(r'\d+(\.\d+)?%|\d+(\.\d+)?[MB]\b', tip), tip)
            pg.mouse.move(2, 2); pg.wait_for_timeout(200)
            # hover: a layer cell and the two tables say what they hold (R1.1, R6.15)
            tip = hover_tip(pg, '#tile-transformer .tf-tz-cell:nth-child(12)')
            check(f'tile {lang} {w}: hovering cell 12 names its layer and count', Z('Layer 12 of 80 holds 856M parameters, 1.21% of the model’s 70.6B.', '第 12 层（共 80 层）有 856M 个参数，占模型 70.6B 个参数的 1.21%。') in tip, tip)
            tip = hover_tip(pg, '#tile-transformer .tf-tz-head')
            check(f'tile {lang} {w}: hovering the LM head names it, its shape and count', Z('The LM head, an 8,192 × 128,256 matrix', 'LM head 是一个 8,192 × 128,256 的矩阵') in tip and '1.05B' in tip, tip)
            OPA = "()=>{const f=document.querySelector('#tile-transformer .tf-tz-frame'), o=x=>getComputedStyle(f.querySelector(x)).opacity; return [f.dataset.hl||null,o('.tf-tz-emb'),o('.tf-tz-lay'),o('.tf-tz-head')]}"
            lb = pg.locator('#tile-transformer .tz-leg [data-k="lay"]').bounding_box(); pg.mouse.move(lb['x'] + 4, lb['y'] + 4); pg.mouse.move(lb['x'] + 6, lb['y'] + 5); pg.wait_for_timeout(300)
            on_ = pg.evaluate(OPA); pg.mouse.move(2, 2); pg.wait_for_timeout(300); off_ = pg.evaluate(OPA)
            check(f'tile {lang} {w}: hovering “layers” in the legend lightens the two tables, then restores them', on_[0] == 'lay' and on_[2] == '1' and float(on_[1]) < 0.3 and float(on_[3]) < 0.3
                  and off_ == [None, '1', '1', '1'], (on_, off_))
            check(f'tile {lang} {w}: no overflow', overflow(pg) <= 0)
            for h_ in pg.evaluate(RULES, '#tile-transformer'):
                rule_hits.append(f'tile {lang}: {h_}')
            for h_ in pg.evaluate(COLONS, '#tile-transformer'):
                rule_hits.append(f'tile {lang}: two colons: {h_}')
            for h_ in pg.evaluate(WORDS, '#tile-transformer'):
                rule_hits.append(f'tile {lang}: {h_}')
            shot(pg, f'tile_{w}_{lang}')
            pg.close()
    # keyboard (R1.6): each legend item is a tab stop after the picture; focusing it shows its tooltip and lightens the rest of the picture
    for lang in ('en', 'zh'):
        Z = Zf(lang)
        pg = tile_page(b, 1440, 900, lang); pg.wait_for_timeout(1300)
        pg.focus('#tile-transformer .tz-leg [data-k="emb"]'); pg.wait_for_timeout(150)
        seen = []
        for _ in range(3):
            info = pg.evaluate("()=>{const e=document.activeElement, f=document.querySelector('#tile-transformer .tf-tz-frame'); return {k:e&&e.dataset?e.dataset.k:null, inLeg:!!(e&&e.closest('#tile-transformer .tz-leg')), hl:f.dataset.hl||null}}")
            tp = norm(pg.inner_text('#tip')) if vis(pg, '#tip') else ''
            seen.append((info['k'], info['hl'], bool(tp)))
            pg.keyboard.press('Tab'); pg.wait_for_timeout(150)
        check(f'tile {lang} keyboard: Tab walks the legend (embedding table, layers, LM head), each with its tooltip, lightening the rest of the picture',
              seen == [('emb', 'emb', True), ('lay', 'lay', True), ('head', 'head', True)], seen)
        pg.close()
    # keyboard (R1.6): the picture is one tab stop after the controls; the arrow keys move between its blocks and each shows its tooltip
    for lang in ('en', 'zh'):
        Z = Zf(lang)
        pg = tile_page(b, 1440, 900, lang); pg.wait_for_timeout(1300)
        pg.click('#tile-transformer .seg-ctl button[data-v="llama31-70b"]'); pg.wait_for_timeout(1100)
        # one Tab: the caption's 70.6B is no tab stop any more (the owner's clarification of 2026-10-07: no exact-value hover)
        pg.focus('#tile-transformer .seg-ctl button[data-v="llama31-70b"]'); pg.keyboard.press('Tab'); pg.wait_for_timeout(150)
        f0 = pg.evaluate(FOCUS); tip0 = norm(pg.inner_text('#tip')) if vis(pg, '#tip') else ''
        check(f'tile {lang} keyboard: Tab reaches the picture, one tab stop in it, on the embedding table, with its tooltip',
              f0['inside'] and 'tf-tz-emb' in f0['cls'] and f0['stops'] == 1 and Z('The embedding table', 'embedding 表是') in tip0, (f0, tip0))
        seq = [('ArrowRight', 'Layer 1 of 80', '第 1 层'), ('ArrowDown', 'Layer 2 of 80', '第 2 层'), ('ArrowRight', 'Layer 7 of 80', '第 7 层'), ('End', 'The LM head', 'LM head 是'),
               ('ArrowLeft', 'Layer 76 of 80', '第 76 层'), ('ArrowUp', 'Layer 75 of 80', '第 75 层'), ('Home', 'The embedding table', 'embedding 表是')]
        for key, en, zh in seq:
            pg.keyboard.press(key); pg.wait_for_timeout(120)
            f = pg.evaluate(FOCUS); tp = norm(pg.inner_text('#tip')) if vis(pg, '#tip') else ''
            check(f'tile {lang} keyboard: {key} moves to “{Z(en, zh)}…” and its tooltip shows', f['inside'] and f['tip'].startswith(Z(en, zh)) and tp.startswith(Z(en, zh)) and f['stops'] == 1, (f, tp[:80]))
        pg.keyboard.press('Tab'); pg.wait_for_timeout(100)
        check(f'tile {lang} keyboard: Tab leaves the picture', not pg.evaluate(FOCUS)['inside'])
        pg.close()
    # motion: the build-up plays once when the picture comes into view; a switch re-tiles in under 1 s (time-based); then nothing moves
    pg = tile_page(b, 1440, 900, 'en'); pg.wait_for_timeout(250)
    g = pg.evaluate(TILE_GEO)
    check('tile motion: the build-up plays when the picture comes into view and ends within 1 s', g['run'] > 0 and 0 < g['end'] <= 1000 and not g['pend'], g)
    pg.wait_for_timeout(1100); g = pg.evaluate(TILE_GEO)
    check('tile motion: after the build-up every cell is drawn and nothing runs', g['run'] == 0 and g['hidden'] == 0 and g['cells'] == 32, g)
    pg.click('#tile-transformer .seg-ctl button[data-v="llama31-70b"]'); pg.wait_for_timeout(60)
    g = pg.evaluate(TILE_GEO)
    check('tile motion: switching to 70B re-tiles the cells one after another, all of it within 1 s', g['run'] > 0 and g['hidden'] > 0 and g['cells'] == 80 and 0 < g['end'] <= 1000, g)
    pg.wait_for_timeout(1100); g = pg.evaluate(TILE_GEO)
    check('tile motion: after the switch, one grid of 80 drawn cells and nothing runs', g['run'] == 0 and g['hidden'] == 0 and g['cells'] == 80 and g['grids'] == 1, g)
    pg.close()
    # reduced motion: the end state at once, no animation at any time
    pg = tile_page(b, 1440, 900, 'en', reduced=True); pg.wait_for_timeout(150)
    g = pg.evaluate(TILE_GEO)
    check('tile reduced motion: no build-up, the picture is drawn at once', g['run'] == 0 and g['hidden'] == 0 and g['cells'] == 32 and not g['pend'], g)
    pg.click('#tile-transformer .seg-ctl button[data-v="llama32-1b"]'); pg.wait_for_timeout(40)
    g = pg.evaluate(TILE_GEO)
    check('tile reduced motion: switching draws the end state at once (16 cells, one shared table)', g['run'] == 0 and g['hidden'] == 0 and g['cells'] == 16 and g['grids'] == 1 and g['head'] < 0.5, g)
    pg.close()
    # 280 and 440 px tiles, light and dark: everything fits, the thinnest blocks stay visible (70B: tables 1.49% of the width)
    for tw in (280, 440):
        for lang in ('en', 'zh'):
            for dark in (False, True):
                pg = tile_page(b, 1440, 900, lang, dark=dark, tilew=tw); pg.wait_for_timeout(1300)
                for k in TILE_MODELS:
                    pg.click(f'#tile-transformer .seg-ctl button[data-v="{k}"]'); pg.wait_for_timeout(1100)
                    g = pg.evaluate(TILE_GEO)
                    check(f'tile {tw} px {lang}{" dark" if dark else ""} {k}: fits, nothing cut off, cells at least 8 × 8 px, tables at least 3 px wide',
                          g['vh'] <= 130 and g['vin'] and not g['cut'] and g['cw'] >= 8 and g['ch'] >= 8 and g['emb'] >= 3 and (k == 'llama32-1b' or g['head'] >= 3) and overflow(pg) <= 0, g)
                if dark:
                    col = pg.evaluate("()=>[getComputedStyle(document.querySelector('#tile-transformer .tf-tz-cell')).boxShadow, getComputedStyle(document.querySelector('#tile-transformer .tf-tz-frame')).backgroundColor]")
                    check(f'tile {tw} px {lang} dark: the cells stay weights blue on a dark surface (R5.6)', 'rgb(90, 138, 255)' in col[0] and col[1] != 'rgb(255, 255, 255)', col)
                shot(pg, f'tile_w{tw}_{lang}{"_dark" if dark else ""}')
                pg.close()


def keyboard_and_jumps(b, lang):
    Z = Zf(lang); pg = page(b, 1440, 900, lang)
    go(pg, 3)
    pg.evaluate("()=>{document.getElementById('tf-math').open=true}"); pg.wait_for_timeout(100)
    # the owner's decisions of 2026-10-09 (H1, H2): Show the math is a few blocks, each a lead line in words, the formula typeset as MathML,
    # then the numbers; blocks separated by a hairline; no TeX command shown as a word; nothing clipped
    mb = pg.evaluate("""()=>{const bs=[...document.querySelectorAll('#tf-mathbody .tf-mb')];
      const words=[...document.querySelectorAll('#tf-mathbody math mi')].map(e=>e.textContent).filter(w=>w.length>1);
      const rule=bs.slice(1).every(b=>{const cs=getComputedStyle(b); return cs.borderTopStyle==='solid'&&parseFloat(cs.borderTopWidth)>=1});
      const clip=[...document.querySelectorAll('#tf-mathbody .tf-math-line')].filter(x=>{const m=x.querySelector('math'); if(!m) return true; const r=m.getBoundingClientRect(), b=x.getBoundingClientRect(); return r.top<b.top-1||r.bottom>b.bottom+1});
      return {n:bs.length, lead:bs.every(b=>b.querySelector('.tf-mlead')&&b.querySelector('.tf-math-line math')), words, rule, clip:clip.length,
              chain:[...document.querySelectorAll('#tf-mathbody .tf-chain')].length}}""")
    check(f'{lang} Show the math (H2): blocks with a lead line and a typeset formula, separated by a hairline, chains on aligned lines, no TeX word, nothing clipped',
          mb['n'] >= 6 and mb['lead'] and mb['rule'] and not mb['words'] and mb['clip'] == 0 and mb['chain'] >= 2, mb)
    tip = hover_tip(pg, '#tf-mathbody .t[data-ctl="f"]')
    check(f'{lang} math term: says what the number is and where the click goes', 'intermediate_size' in tip and Z('Click to jump to “Intermediate size”.', '点击跳到“中间维度”。') in tip, tip)
    pg.focus('#tf-mathbody .t[data-ctl="f"]'); pg.keyboard.press('Enter'); pg.wait_for_timeout(800)
    check(f'{lang} math term: Enter jumps to the control and focuses it', pg.evaluate("()=>!!document.activeElement.closest('#tf-ctl-f')"))
    tip = hover_tip(pg, '#tf-mathbody .t[data-ctl="tied"]')
    check(f'{lang} math term for the LM head: 4,096 × 128,256, a hidden size × vocabulary matrix (N14)', pg.evaluate("s=>document.querySelector(s).textContent", '#tf-mathbody .t[data-ctl="tied"]') == '4,096 × 128,256'
          and Z('tie_word_embeddings is false, so the LM head is its own hidden size × vocabulary matrix.', 'tie_word_embeddings 是 false，所以 LM head 是单独的一个“隐藏维度 × 词表大小”矩阵。') in tip, tip)
    lg = pg.evaluate("()=>{const i=document.querySelector('#tf-legend .tf-sw-rs'), a=i&&getComputedStyle(i); return a?[a.width,a.height,a.borderRightWidth,a.borderTopLeftRadius,a.borderLeftWidth]:null}")
    check(f'{lang} legend: the residual connection has the glyph of the lab Architecture overview (ARCH #21: 12 × 11, open on the right, 6 px corners)', lg[:4] == ['12px', '11px', '0px', '6px'] and lg[4] in ('1px', '1.5px'), lg)
    tip = hover_tip(pg, '#tf-mathbody .t[data-ctl="H"]')
    check(f'{lang} math term for heads: pieces of head_dim numbers cut from what Wq, Wk and Wv compute (r13 M6)',
          Z('Each head works with pieces of head_dim numbers cut from what Wq, Wk and Wv compute from the whole token:',
            '每个 head 分到 Wq、Wk、Wv 从整个 token 算出的向量里的 head_dim 维，') in tip
          and pg.evaluate("()=>{const m=document.querySelector('#tip math'); return !!m&&/head_dim/.test(m.textContent)&&/128/.test(m.textContent)}"), tip)   # 2026-10-09 (H1): the formula typeset
    note = norm(pg.evaluate("()=>{const c=document.getElementById('tf-ctl-H'); const n=c&&c.querySelector('.ctl-note'); return n?n.innerText:''}"))
    check(f'{lang} Settings “Heads” note: head_dim said once, as a formula after the pieces (r13 M6)',
          Z('each head works with pieces of head_dim numbers cut from what Wq, Wk and Wv compute, with', '每个 head 分到 Wq、Wk、Wv 算出的向量里的 head_dim 维，') in note
          and pg.evaluate("()=>{const c=document.getElementById('tf-ctl-H'); const m=c&&c.querySelector('.ctl-note math'); return !!m&&/head_dim/.test(m.textContent)}"), note)   # 2026-10-09 (H1): the formula typeset
    pg.click('#tf-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(300)
    pg.evaluate("()=>{document.getElementById('tf-math').open=true}"); pg.wait_for_timeout(100)
    tip = hover_tip(pg, '#tf-mathbody .t[data-ctl="d"]')
    check(f'{lang} challenge: a fixed term says its value instead of jumping', Z('Fixed during the challenge: “Hidden size” is 4,096.', '挑战中固定不变：“隐藏维度”是 4,096。') in tip, tip)
    tip = hover_tip(pg, '#tf-mathbody .t[data-ctl="tied"]')
    check(f'{lang} challenge: the fixed LM head term names the two separate tables', Z('Fixed during the challenge: “Embedding table and LM head” are two separate tables.', '挑战中固定不变：“embedding 表和 LM head”是分开的两张表。') in tip, tip)
    pg.click('#tf-app .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(300)
    pg.click('#tf-rail [data-jump="controls"]'); pg.wait_for_timeout(800)
    check(f'{lang} “Settings” link brings the panel into view', pg.evaluate("()=>{const r=document.getElementById('tf-controls').getBoundingClientRect(); return r.top<innerHeight&&r.bottom>0}"))
    pg.close()


def connect_code(b):
    """R7.6 (review 8): Connect is two columns above 760 px; at 1440 and 1280 the code reads without sideways scrolling"""
    for w in (1440, 1280):
        pg = page(b, w, 900, 'en')
        m = pg.evaluate("()=>{const c=document.getElementById('tf-code'), g=c.closest('.connect-grid'); return {sw:c.scrollWidth, cw:c.clientWidth, cols:getComputedStyle(g).gridTemplateColumns.split(' ').length}}")
        check(f'{w}: the Connect code block needs no sideways scrolling (two columns)', m['sw'] <= m['cw'] and m['cols'] == 2, m)
        pg.close()


def dark_and_reduced(b):
    pg = page(b, 1440, 900, 'en', dark=True, reduced=True)
    go(pg, 3)
    col = pg.evaluate("()=>{const m=getComputedStyle(document.querySelector('#tf-stack .tf-pat-mlp')), s=getComputedStyle(document.querySelector('#tf-stage')); return [m.backgroundColor, s.backgroundColor]}")
    check('dark: weights stay blue on a dark surface', col[0] == 'rgb(90, 138, 255)' and col[1] != 'rgb(255, 255, 255)', col)
    shot(pg, 'dark_1440_en_part4')
    pg.close()


# the owner's review of 2026-10-09 (R6.37: a row the reader has to scroll sideways is not read): in every guided part, Scenarios and the
# challenge, with each Show the math open, no box in it scrolls sideways and it is no wider than its own box (as tests/kit_math_scroll.py)
MSCROLL = """()=>{const out=[]; for(const d of document.querySelectorAll('details.math')){if(!d.getClientRects().length) continue; const was=d.open; d.open=true;
    for(const e of d.querySelectorAll('*')){if(!/auto|scroll/.test(getComputedStyle(e).overflowX)||!e.getClientRects().length) continue;
      const by=e.scrollWidth-e.clientWidth; if(by>1) out.push(e.className+' by '+by+' px: '+(e.textContent||'').trim().replace(/\\s+/g,' ').slice(0,40));}
    if(d.scrollWidth-d.clientWidth>1) out.push('the whole Show the math by '+(d.scrollWidth-d.clientWidth)+' px'); d.open=was;}
  return out;}"""


def math_scroll(b, w, h, lang):
    pg = page(b, w, h, lang); bad = []
    n = pg.evaluate("r=>new Set([...document.querySelectorAll(r+' [data-go]')].map(e=>e.dataset.go)).size", '#tf-rail')
    for k in range(n):
        pg.evaluate("([r,k])=>{const e=document.querySelector(r+' [data-go=\"'+k+'\"]'); if(e) e.click();}", ['#tf-rail', k]); pg.wait_for_timeout(500)
        bad += [f'part {k + 1}: {x}' for x in pg.evaluate(MSCROLL)]
    for m in ('sandbox', 'challenge'):
        pg.evaluate("m=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"'+m+'\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}", m); pg.wait_for_timeout(600)
        bad += [f'{m}: {x}' for x in pg.evaluate(MSCROLL)]
    check(f'{w} {lang}: no Show the math box scrolls sideways (all {n} guided parts, Scenarios, the challenge)', n >= 5 and not bad, (n, bad[:4]))
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    for w, h, lang in ((1440, 900, 'zh'), (900, 900, 'en'), (390, 844, 'zh'), (390, 844, 'en')):
        math_scroll(b, w, h, lang)
    for lang in ('en', 'zh'):
        run_flow(b, 1440, 900, lang)
        run_flow(b, 390, 844, lang)
        challenge_tiers(b, lang)
        submit_scroll(b, lang, 1440, 900)
        submit_scroll(b, lang, 390, 844)
        settings_no_scroll(b, lang, 1440, 900)
        settings_no_scroll(b, lang, 900, 900)
        settings_no_scroll(b, lang, 390, 844)
        lang_and_hash(b, lang)
        card_heights(b, lang, 1000)
        card_heights(b, lang, 900)
        keyboard_and_jumps(b, lang)
    run_flow(b, 1440, 900, 'zh', dark=True)
    dark_and_reduced(b)
    connect_code(b)
    layouts(b)
    tile(b)
    b.close()

# the owner's decisions of 2026-10-09: H1, no plain-text formula of the kinds converted (head_dim = hidden size ÷ heads, the weights'
# product, 2^30 / 10^9) left in the lab's source outside comments and the numbers lines of Show the math; H4, the quantization lab
# named as planned only through its status (quantRef), no static “planned lab” sentence
SRC = (P / 'labs' / 'transformer.js').read_text()
CODE = re.sub(r'/\*.*?\*/', '', SRC, flags=re.S)
left = [x for x in ['head_dim = hidden size ÷ heads', 'head_dim = 隐藏维度 ÷ head 数', '2^30', '10^9', 'parameters × bytes per parameter', '参数量 × 每个参数的字节数', 'Weight memory = parameters', '权重的显存 = 参数量'] if x in CODE]
check('H1: no plain-text formula of the converted kinds left in the source', not left, left)
check('H1: the formulas use A.ui.math (MX)', CODE.count('MX(') >= 20 and 'const MX=t=>A.ui.math(t)' in CODE, CODE.count('MX('))
check('H4: the quantization lab is named as planned only through its status (A.catalog.LAB, labStatus), and becomes a link when live',
      "const quantRef=()=>" in CODE and "c.labStatus(l)==='live'" in CODE and "A.shell.labLink('quantization')" in CODE and CODE.count('规划中') == 1 and CODE.count('planned lab') == 1, [CODE.count('规划中'), CODE.count('planned lab')])
check('copy rules: no “·” between clauses, no two colons in a Chinese sentence, no 起点 / 控件 / “in this toy”, no page-behavior sentences', not rule_hits, rule_hits[:12])
check('no page errors', not errs, errs[:5])
print('\n'.join(x for x in notes if x.startswith('FAIL')))
for H in (1000, 900):
    print(f'card heights after the action at 1440 × {H} (lang, part, px):', [(l, p_, h_) for l, H_, p_, h_ in heights if H_ == H])
# G1 and G2, the owner's site-wide decisions of 2026-10-08 (sweep): option lead-ins instead of “one at a time” notes, and no sentence
# that leans on a detail of an earlier part. Read from the lab's source, so both languages are covered.
_G_SRC = (P / 'labs' / 'transformer.js').read_text()
for _bad in ['一次只能选', '一次选一个', '不会叠加', 'do not stack', 'One at a time', 'one at a time']:
    check(f'G1: no “{_bad}” left in the lab (EN and ZH)', _bad not in _G_SRC)
for _lead in ['Pick a change and see what the parameter count becomes.', '选一种改法，看看参数总数变成多少。', 'Pick a scenario and see how many parameters the model has and how much memory its weights take.', '选一个场景，看看模型有多少参数、权重占多少显存。']:
    check(f'G1: the lead-in “{_lead[:36]}…” is there', _lead in _G_SRC)
for _old, _new in [('免得提前透露第 2 部分的答案', '两者各占多少，第 4 部分揭晓。'), ('so it does not give away the answer to part 2', 'How they split is shown in part 4.')]:
    check(f'G2: “{_old[:30]}…” is rewritten as “{_new[:30]}…”', _old not in _G_SRC and _new in _G_SRC)
print(f'{len(notes)} checks, {len(fails)} failed')
sys.exit(1 if fails else 0)
