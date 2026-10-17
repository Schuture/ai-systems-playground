"""Flow test for the prefix-caching lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH, light, plus one dark run:
- no page errors and no horizontal overflow at any step;
- the phone first screen (m2 template, R7.3): the parts' bar with Next, and the block strip of the chart;
- the prediction gate (R3.8): Next disabled until a prediction; request 2's answer (8.13 ms, 10,000 cached) appears nowhere, in text
  or tooltips, until part 3 sends it; “Saved …” after a pick (R2.5); part 3 gives the result, the verdict with the right answer, why;
- each part's action gate and result: part 3's send, part 4's levers (results kept on the title row, R3.6), part 6's memory levers;
- part 5 (Batch 4), the experiment: part 5 of 7; both strips and the three positions (on a phone's first screen too); the prediction
  gate with no answer anywhere before it; the sweep (mid-sweep: the result waits) and its end state for the front, the middle and the end
  (counters, first-miss line and arrow, “=” and “≠” keys, the label, result → verdict → why, the title-row results); a quick burst of
  clicks ends in the last one; a language switch or Back during a sweep; on a phone a pick in the chart never scrolls it away; reduced
  motion (no sweep, “Block by block” in five stops with the counters following), at 1440 EN and ZH and 390 ZH;
  review r1: on a phone the question sits under the positions in the chart, and after a real tap on a choice (or on a position in the
  card) both strips and the three counters stay on screen while the lookup plays; the line under the chart leads to the result; no
  tooltip covers request 2's strip or the key badges after a real click on a position; the counters and the choices count the system
  prompt only; an edge position counts the two equivalent choices as right; part 4 no longer gives part 5's answer; part 6's first
  frame shows request 1; the parts' dots stay on one line at 360, 390 and 720 px, in EN and ZH;
  ▲▼ badges only after the learner's own change (R6.6); the result box on screen after the action at 1440×1000 (R3.10);
- guided → Scenarios → Challenge by the cards' buttons (R3.12); the scenarios' pressed state follows the settings (R1.8); the rows
  select a request; a change under Settings does not move the page (R3.2);
- the challenge: only the parts' order in the card, Settings locked with a note (R4.2), live progress, the three feedback tiers,
  “show all options” and the comparison chart (R4.4, R5.4), the capability card;
- the language switch keeps the part, the prediction, the lever and the challenge's order and answer; the hash round-trip in
  Scenarios; the mini result bar at 390 once the chart scrolls away, with a short status;
- the writing rules on every visible text node and tooltip (as tests/prefill-decode_flow.py checks them).
Screenshots: shots/prefix-caching/.
Run: python3 tests/prefix-caching_flow.py"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/prefix-caching.html', '--only', 'prefix-caching'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-prefix-caching.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'prefix-caching.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-prefix-caching.html')
SH = P / 'shots' / 'prefix-caching'; SH.mkdir(parents=True, exist_ok=True)
errs, fails, n = [], [], 0


reduced_motion = lambda pg: pg.evaluate("()=>ASP.reduceMotion?ASP.reduceMotion():matchMedia('(prefers-reduced-motion: reduce)').matches")


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


Z = lambda lang, en, zh: zh if lang == 'zh' else en
NB = lambda s: s.replace(' ', ' ').replace(' ', ' ').replace('‑', '-')


def page(b, w, h, lang, dark=False, drawers=True, hash_='prefix-caching'):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light')
    if drawers: open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(500)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__px.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: NB(pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText:''}", sel))


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__px||!window.__px.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel, wait=300):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(wait); settle(pg)


def go_part(pg, i): click(pg, f'#px-rail [data-go="{i}"]', 300)
def mode(pg, m): click(pg, f'#px-app .modes button[data-mode="{m}"]', 400)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def all_text(pg):
    return NB(pg.evaluate("""()=>{const out=[]; for(const s of ['#px-app','#dock']){const e=document.querySelector(s); if(e&&e.offsetParent!==null) out.push(e.innerText);}
      document.querySelectorAll('#px-app [data-tipx], #dock [data-tipx]').forEach(e=>out.push(e.getAttribute('data-tipx'))); return out.join('\\n')}"""))


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/in this toy\\b|在这个玩具里|\\btoy\\b|玩具/i,'“toy”'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/Locked in|已锁定/,'“Locked in”'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万 instead of K, M, B (R6.18)'],[/个权重/,'“个权重” (R6.17)'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],[/\\ba 8/,'“a 8…” (an)'],
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour (R5.7)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, .fid-toy')||!e.getClientRects().length) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)&&!(name==='“toy”'&&/Toy run|玩具实跑/.test(t))) out.push(name+': '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); for(const [re,name] of bad) if(re.test(t)&&!(name==='“toy”'&&/Toy run|玩具实跑/.test(t))) out.push(name+' (tooltip): '+t.slice(0,90));});
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
 root.querySelectorAll('p,li,button,label,div,span,b,strong,h1,h2,h3,summary').forEach(e=>{if(e.closest('[data-endnotes],pre,code,svg,.mathbody')||!e.getClientRects().length) return; if(e.querySelector('p,li,div,ul,ol,h1,h2,h3,table')) return;
   if(![...e.childNodes].some(c=>c.nodeType===3&&c.textContent.trim())) return; texts.push(e.textContent);});
 root.querySelectorAll('[data-tipx]').forEach(e=>texts.push(e.getAttribute('data-tipx')));
 for(const t of texts){const sents=t.replace(/\\s+/g,' ').split(/(?<=[。？！])|(?<=[.?!])\\s+/);
   for(const x of sents){const k=(x.match(/：|: /g)||[]).length; if(k>=2) out.add(x.trim().slice(0,160));}}
 return [...out];}"""
BIGNUM = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #px-mathbody, .num-big, #px-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
# amber only for the KV cache (cached blocks, and in part 5 the bar of new blocks in the cache), red only for a missed goal or a wrong pick;
# no blue or violet on this page (R5.6)
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/px-s0|sw kv\\b|px-dk-kv|px-rc|px-cbar|px-al-held|px-sw-held/,[red]:/status bad|badline|badc|warn|wrong|ex-res no/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#px-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
rule_hits = []


def rules(pg, where, lang):
    for sel in ('#px-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def badges(pg): return pg.evaluate("()=>[...document.querySelectorAll('#px-metrics .metric .d')].map(d=>d.textContent).filter(Boolean)")


# part 5 (Batch 4): what the stage and the card show, in one read
OT_STATE = """()=>{const q=s=>document.querySelector(s), vis=e=>!!e&&e.getClientRects().length>0&&getComputedStyle(e).visibility!=='hidden';
  const c=k=>{const e=q(`#px-ctrs .px-ctr[data-k="${k}"]`); return e?[e.querySelector('.v').textContent,e.querySelector('.sub').textContent].map(s=>s.replace(/\\u00a0/g,' ')):null;};
  const ob=q('#px-rail [data-obs]');
  return {cells2:document.querySelectorAll('#px-pc2 .px-cell').length, amber:document.querySelectorAll('#px-pc2 .px-s0').length, hatched:document.querySelectorAll('#px-pc2 .px-s2').length,
    cover:vis(q('#px-pcover')), miss:vis(q('#px-pmiss')), arrow:vis(q('#px-larrow')), arrowEnd:q('#px-larrow').classList.contains('end'), tick:vis(q('#px-ptick')),
    kp:[...document.querySelectorAll('#px-lane .px-kp')].filter(vis).map(e=>({eq:e.classList.contains('eq'),a:e.children[0].textContent,b:e.children[2].textContent})),
    same:vis(q('#px-same'))?q('#px-same').textContent:'', rec:c('rec'), t:c('t'), nw:c('nw'), obs:ob&&!ob.hidden&&vis(ob)?ob.innerText.replace(/\\u00a0/g,' '):'',
    next:[...document.querySelectorAll('#px-rail [data-nav="next"]')].map(b=>b.disabled), status:q('#px-status').textContent.replace(/\\u00a0/g,' '), step:q('#px-otstep').textContent}}"""
OT_ANS = ['5,208', '8.41 ms', '0.63 GiB', '1.24 GiB', '0.03 GiB', '9,984', '188.9 ms', '4,992', '5,008', '0.61 GiB', 'about half cached', '命中约一半']
# each position's end state (review r1 #3: the system prompt only): its tokens recomputed, first token, its blocks stored twice, their memory,
# and a phrase of the result
OT_END = {'middle': ('5,008', '188.9 ms', '313', '0.61 GiB', ('190 ms', '190 ms')), 'front': ('10,000', '342.9 ms', '625', '1.22 GiB', ('340 ms', '340 ms')),
          'end': ('16', '8.41 ms', '1', ('< 0.01 GiB', '多占不到 0.01 GiB 显存'), ('almost as soon as with no change', '和完全没有改动时差不多'))}
# each position's result on its title row: how much of the system prompt request 2 finds in the cache (R3.17)
OT_LV = {'middle': ('about half cached', '命中约一半'), 'front': ('none of it cached', '一点也没命中'), 'end': ('almost all cached', '命中几乎全部')}
# the predictions that give the right result at each position (the front: “before” and “none”; the end: “block” and “before”)
OT_OK = {'front': {'before', 'none'}, 'middle': {'before'}, 'end': {'block', 'before'}}
OT_MID_MS = {'middle': 900, 'front': 900, 'end': 1560}     # when the sweep has just passed the first miss (it stops there briefly)
OT_TOK = {'front': '10', 'middle': '5,000', 'end': '9,990'}


# on screen: below the top bar (and the mini result bar when it shows) and above the bottom of the window
INVIEW = """(sels)=>{const bar=document.querySelector('.topbar').getBoundingClientRect().bottom, dk=document.getElementById('dock');
  const lim=Math.max(bar,dk&&!dk.hidden&&dk.getClientRects().length?dk.getBoundingClientRect().bottom:0);
  return sels.map(s=>{const e=document.querySelector(s); if(!e) return {s,ok:false}; const r=e.getBoundingClientRect();
    return {s,top:Math.round(r.top),bottom:Math.round(r.bottom),lim:Math.round(lim),ok:r.height>0&&r.top>=lim-1&&r.bottom<=innerHeight+1};});}"""
# the tooltip after a real click on a position: none over request 2's strip or the key badges (review r1 #5)
TIP_OVER = """()=>{const t=document.getElementById('tip'), p=document.querySelector('#px-stage .px-pbtn[aria-pressed="true"]'), r=e=>e.getBoundingClientRect();
  const tipx=!!p&&p.hasAttribute('data-tipx'); if(!t||t.hidden||!t.getClientRects().length) return {over:false,tipx};
  const a=r(t), hit=b=>!(a.right<=b.left||a.left>=b.right||a.bottom<=b.top||a.top>=b.bottom);
  return {over:[document.getElementById('px-pt2'),...[...document.querySelectorAll('#px-lane .px-kp')].filter(e=>!e.hidden)].some(e=>hit(r(e))),tipx,txt:t.textContent.slice(0,90)};}"""
# the parts' dots on one line (review r1 #10): the bar shows on narrow screens only
DOTS = """()=>{const d=[...document.querySelectorAll('#px-rail .pbar .pb-dot')].filter(e=>e.getClientRects().length); if(!d.length) return {shown:false,one:true};
  const tops=d.map(e=>Math.round(e.getBoundingClientRect().top)); return {shown:true,n:d.length,one:Math.max(...tops)-Math.min(...tops)<=1,tops};}"""


def in_view(pg, sels): return pg.evaluate(INVIEW, sels)


def sweep_done(pg, t=8000):
    pg.wait_for_function("()=>window.__px.state().sweep===null", timeout=t); pg.wait_for_timeout(150)


def ot_end_checks(pg, tag, lang, v, pick):
    o = pg.evaluate(OT_STATE); rec, t, nw, g, say = OT_END[v]
    if isinstance(g, tuple): g = Z(lang, *g)
    check(f'{tag} part 5 {v}: the end state: request 2’s blocks all looked up, the thick first-miss line, the arrow to the end',
          o['cells2'] > 0 and not o['cover'] and o['miss'] and o['arrow'] and o['arrowEnd'] and o['tick'] and (o['amber'] > 0) == (v != 'front') and (o['hatched'] > 0) == (v != 'end'), o)
    check(f'{tag} part 5 {v}: the counters: {rec} recomputed, {t}, {nw} new blocks ({g})', o['rec'][0] == rec and o['t'][0] == t and o['nw'][0] == nw and g in o['nw'][1], (o['rec'], o['t'], o['nw']))
    kp = o['kp']; ne = [k for k in kp if not k['eq']]; eq = [k for k in kp if k['eq']]
    check(f'{tag} part 5 {v}: the keys: “=” with one hash before the change (none at the front), “≠” with two different hashes after it',
          len(ne) == 1 and ne[0]['a'] != ne[0]['b'] and len(eq) == (0 if v == 'front' else 1) and all(k['a'] == k['b'] for k in eq), kp)
    lab = Z(lang, 'Same tokens, different key: a block’s key includes every token before it.', 'token 相同，key 却不同：block 的 key 包含它之前的全部 token。') if v != 'end' else Z(lang, 'One token differs, so this block’s key and every key after it differ.', '有一个 token 不同，所以这个 block 和之后的 key 都不同。')
    check(f'{tag} part 5 {v}: the label under the strips (“same tokens, different key” where the tokens are the same)', lab in o['same'], o['same'])
    ob = o['obs']
    # the verdict (review r1 #11): the right choice, a choice that gives the same result at this edge position, or wrong with the right answer
    right = Z(lang, 'was right', '是对的') if pick == 'before' else (Z(lang, 'gives the right result here', '在这里给出的结果是对的') if pick in OT_OK[v] else Z(lang, 'but the right answer is', '但正确答案是'))
    i_res, i_ver, i_why = ob.find(Z(lang, say[0], say[1])), ob.find(right), ob.find(Z(lang, 'Why:', '原因：'))
    # at an edge position two choices give the same result: the right choice names the other one, the other one says why it is right
    # here and gives the answer in general; a wrong choice gets neither (review r1 #11)
    same, general = Z(lang, 'gives the same result here', '在这里结果一样'), Z(lang, 'In general the answer is', '一般情况下，答案是')
    want = None if v == 'middle' or pick not in OT_OK[v] else (same if pick == 'before' else general)
    check(f'{tag} part 5 {v}: the result first (one rounded number), then the verdict, then why (R3.10, R6.20); at an edge position why two choices give the same result',
          -1 < i_res < i_ver < i_why and (want is None or -1 < ob.find(want) < i_why) and all(x not in ob for x in (same, general) if x != want), ob[:420])
    if v != 'middle':
        check(f'{tag} part 5 {v}: the verdict links to the middle, which tells the choices apart (review r1 #11)',
              pg.evaluate("()=>{const b=document.querySelector('#px-rail [data-obs] [data-otpos=\"middle\"]'); return !!b&&b.getClientRects().length>0&&!!b.dataset.tipx}"))
    check(f'{tag} part 5 {v}: the principle: from the changed block on no key matches, not even for the same tokens; the parts that change belong at the end',
          Z(lang, 'the parts that change belong at the end', '会变的内容应该放在最后') in ob and (Z(lang, 'here that is only the question', '这里它后面只有问题') in ob if v == 'end' else Z(lang, 'none of request 2’s keys is in the cache, not even for the', '请求 2 的 key 没有一个在缓存里，连装着同样 token 的') in ob), ob[-400:])
    check(f'{tag} part 5 {v}: one verdict line, no “You predicted” line above the result box (review r1 #8)', text(pg, '#px-rail .step.current').count(Z(lang, 'You predicted', '你预测的是')) <= 1 and not pg.evaluate("()=>[...document.querySelectorAll('#px-rail .step.current [data-otpred]')].some(b=>b.getClientRects().length)"))
    lvr = text(pg, f'#px-rail [data-lever="{v}"] .lv-r')
    check(f'{tag} part 5 {v}: its result on its title row, how much of the system prompt is cached (R3.6, R3.17)', lvr == Z(lang, *OT_LV[v]), lvr)
    # R7.8: after the lookup the options' titles stay horizontal (at most two lines), their results never overlap the titles or the
    # descriptions and stay inside their option
    lt = pg.evaluate("""()=>[...document.querySelectorAll('#px-rail .px-otlv .lever')].filter(b=>b.getClientRects().length).map(b=>{const t=b.querySelector('.lv-t b'), d=b.querySelector('.lv-d'), r=b.querySelector('.lv-r');
      const R=e=>e.getBoundingClientRect(), hit=(a,c)=>!(a.right<=c.left||a.left>=c.right||a.bottom<=c.top||a.top>=c.bottom), bb=R(b), tr=R(t), rr=R(r), lh=parseFloat(getComputedStyle(t).lineHeight)||20;
      return {v:b.dataset.lever, lines:Math.round(tr.height/lh), over:rr.width>0&&(hit(rr,tr)||(!!d&&hit(rr,R(d)))), inside:rr.width===0||(rr.left>=bb.left-0.5&&rr.right<=bb.right+0.5&&rr.bottom<=bb.bottom+0.5)};})""")
    check(f'{tag} part 5 {v}: the positions’ titles stay on at most two lines; their results never overlap the titles and stay inside the option (R3.17, R7.8)',
          len(lt) == 3 and all(x['lines'] <= 2 and not x['over'] and x['inside'] for x in lt), lt)
    return o


def part5(pg, tag, lang, w, dark):
    phone = w <= 760
    click(pg, '#px-rail .step.current [data-nav="next"]', 400)
    parts = pg.evaluate("()=>({n:document.querySelectorAll('#px-rail .pbar .pb-dot').length, on:(document.querySelector('#px-rail .pbar .pb-dot.on')||{}).textContent, top:(document.querySelector('#px-rail .rail-top')||{}).textContent, t:(document.querySelector('#px-rail .pbar .pb-t b')||{}).textContent})")
    check(f'{tag} part 5: “Experiment: change one token” is part 5 of 7', state(pg)['step'] == 4 and parts['n'] == 7 and parts['on'] == '5' and Z(lang, 'part 5 of 7', '第 5 / 7 部分') in parts['top'] and parts['t'] == Z(lang, 'Experiment: change one token', '实验：只改一个 token'), parts)
    o = pg.evaluate(OT_STATE)
    check(f'{tag} part 5: Next waits for the prediction', all(o['next']), o['next'])
    check(f'{tag} part 5: both strips and the three positions; request 2 not sent yet; the counters empty; no badge',
          vis(pg, '#px-pt1') and vis(pg, '#px-pt2') and pg.evaluate("()=>[...document.querySelectorAll('#px-stage .px-pbtn')].filter(b=>b.getClientRects().length).length") == 3 and o['cells2'] == 0 and o['rec'][0] == '?' and not o['tick'] and not badges(pg), o)
    check(f'{tag} part 5: Settings locked with a note, no grayed controls (R4.2)', vis(pg, '#px-lock') and not vis(pg, '#px-ctl-P'))
    check(f'{tag} part 5: no overflow', overflow(pg) <= 0, overflow(pg))
    if phone:
        fs = pg.evaluate("""()=>{const top=document.querySelector('.topbar').getBoundingClientRect().bottom, r=s=>document.querySelector(s).getBoundingClientRect(), inV=b=>b.top>=top-1&&b.bottom<=innerHeight;
          return {s1:inV(r('#px-pt1')), s2:inV(r('#px-pt2')), btn:[...document.querySelectorAll('#px-stage .px-pbtn')].every(b=>inV(b.getBoundingClientRect())), bar:inV(r('#px-rail .pbar')), y:Math.round(r('#px-ppos').bottom), vh:innerHeight}}""")
        check(f'{tag} part 5: phone first screen: the parts’ bar, both strips and the three positions (m2)', fs['s1'] and fs['s2'] and fs['btn'] and fs['bar'], fs)
        dt = pg.evaluate(DOTS)
        check(f'{tag} part 5: the 7 parts’ dots on one line (review r1 #10, R7.4)', dt['shown'] and dt['n'] == 7 and dt['one'], dt)
        shot(pg, f'{tag}_p5_first_screen')
    shot(pg, f'{tag}_p5_before', full=phone)
    rules(pg, f'{tag} part 5 before a position', lang)
    # a position from the chart: the changed token's tick and the question that names it; the answer stays hidden (R3.8)
    loc = pg.locator('#px-stage [data-otpos="middle"]'); loc.scroll_into_view_if_needed(); y0 = loc.bounding_box()['y']
    loc.click(); pg.wait_for_timeout(350)
    o = pg.evaluate(OT_STATE); y1 = loc.bounding_box()['y']
    qsel = '#px-otqs .px-otq' if phone else '#px-rail .step.current .px-otq'
    q = text(pg, qsel)
    nch = pg.evaluate("()=>[...document.querySelectorAll('[data-otpred]')].filter(b=>b.getClientRects().length).length")
    check(f'{tag} part 5: after a position, the question names it with three choices; request 2 still hidden; “Predict first”',
          vis(pg, qsel) and Z(lang, 'token 5,000', '第 5,000 个') in q and nch == 3 and o['tick'] and o['cells2'] == 0 and o['status'] == Z(lang, 'Predict first', '先预测') and all(o['next']), (q, nch, o))
    check(f'{tag} part 5: the question says that the choices and the counters leave the question’s tokens out (review r1 #3)',
          Z(lang, 'the choices and the counters leave it out', '所以选项和计数都不算它') in text(pg, '#px-otqs' if phone else '#px-rail .step.current'))
    if phone:
        check(f'{tag} part 5: picking a position in the chart does not move it (m2)', abs(y1 - y0) < 3, (y0, y1))
        iv = in_view(pg, ['#px-otqs .px-otq', '#px-otqs [data-otpred]'])
        check(f'{tag} part 5: on a phone the question and its first choice sit on screen right under the positions (review r1 #2)', all(x['ok'] for x in iv) and not vis(pg, '#px-rail .step.current .px-otq'), iv)
    else:
        check(f'{tag} part 5: on a wide screen the question is in the card, not in the chart', not vis(pg, '#px-otqs .px-otq'))
    at = all_text(pg)
    check(f'{tag} part 5: no answer anywhere before the prediction (text and tooltips)', not any(a in at for a in OT_ANS), [a for a in OT_ANS if a in at])
    shot(pg, f'{tag}_p5_predict', full=phone)
    rules(pg, f'{tag} part 5 before the prediction', lang)
    # the prediction, then the sweep over request 2's blocks: the middle
    pick = 'before' if dark else {'en': 'none', 'zh': 'block'}[lang]
    rm = reduced_motion(pg)
    ch = pg.locator(f'#px-stage [data-otpred="{pick}"]' if phone else f'#px-rail [data-otpred="{pick}"]'); ch.scroll_into_view_if_needed(); ch.click()
    WATCH = ['#px-pairbox', '#px-pt1', '#px-pt2', '#px-ctrs']       # the whole chart box: it must not slide under the top bar
    if not rm:
        pg.wait_for_timeout(OT_MID_MS['middle'])
        o = pg.evaluate(OT_STATE)
        check(f'{tag} part 5 middle: the sweep runs: part of request 2 looked up, the counters on their way, no result yet (R3.8)', state(pg)['sweep'] is not None and o['cover'] and not o['obs'] and o['rec'][0] not in ('5,208', '5,008') and o['status'] == Z(lang, 'Request 2: looking up its blocks', '请求 2：逐个查找 block'), o)
        if phone:
            iv = in_view(pg, WATCH)
            check(f'{tag} part 5 middle: after a real tap on a choice, the whole chart box (both strips, the three counters) is on screen while the lookup plays (review r1 #1)', all(x['ok'] for x in iv), iv)
        shot(pg, f'{tag}_p5_mid_middle')
    sweep_done(pg); settle(pg)
    if phone:
        iv = in_view(pg, WATCH)
        check(f'{tag} part 5 middle: after the lookup the strips and the counters are still on screen (the page does not jump away)', all(x['ok'] for x in iv), iv)
        hint = pg.locator('#px-prog [data-act="ot-res"]')
        check(f'{tag} part 5 middle: a line under the chart leads to the result in the card (review r1 #1, R3.2)', hint.count() == 1 and hint.is_visible() and in_view(pg, ['#px-prog'])[0]['ok']
              and Z(lang, 'The result is in part 5 ↓', '结论在第 5 部分 ↓') in hint.inner_text() and bool(hint.get_attribute('data-tipx')), hint.count())
        shot(pg, f'{tag}_p5_end_middle_chart')
        hint.click(); pg.wait_for_timeout(200); settle(pg)
        oc = obs_on_screen(pg)
        check(f'{tag} part 5 middle: the line under the chart brings the result on screen', oc and oc['ok'], oc)
    ot_end_checks(pg, tag, lang, 'middle', pick)
    check(f'{tag} part 5: Next enabled once the result is shown', not any(pg.evaluate(OT_STATE)['next']))
    if w == 1440 and not dark:
        oc = obs_on_screen(pg)
        check(f'{tag} part 5: after the sweep the result is on screen (R3.10)', oc and oc['ok'], oc)
    check(f'{tag} part 5 middle: no overflow', overflow(pg) <= 0, overflow(pg))
    shot(pg, f'{tag}_p5_end_middle', full=phone)
    rules(pg, f'{tag} part 5 middle', lang)
    # the other positions, from the chart: the page stays where it is on a phone
    for v in ['front', 'end']:
        loc = pg.locator(f'#px-stage [data-otpos="{v}"]'); loc.scroll_into_view_if_needed(); y0 = loc.bounding_box()['y']
        loc.click()
        if not rm:
            pg.wait_for_timeout(OT_MID_MS[v] - 150)
            ob = pg.evaluate(OT_STATE)['obs']
            check(f'{tag} part 5 {v}: the sweep starts again for the new position; the result box keeps its place but names only the position (R3.8)',
                  state(pg)['sweep'] is not None and Z(lang, 'Why:', '原因：') not in ob and Z(lang, 'prediction', '预测') not in ob and Z(lang, f'token {OT_TOK[v]}', f'第 {OT_TOK[v]} 个 token') in ob, ob[:200])
            shot(pg, f'{tag}_p5_mid_{v}')
        sweep_done(pg); pg.wait_for_timeout(250)
        if phone:
            y1 = loc.bounding_box()['y']
            check(f'{tag} part 5 {v}: picking a position does not scroll it away (m2)', abs(y1 - y0) < 3 and 0 < y1 < pg.viewport_size['height'] - 30, (y0, y1))
        else:
            bb = loc.bounding_box(); pg.mouse.move(bb['x'] + bb['width'] / 2 + 1, bb['y'] + bb['height'] / 2); pg.wait_for_timeout(200)
            tp = pg.evaluate(TIP_OVER)
            check(f'{tag} part 5 {v}: after a real click on a position, the pressed position has no tooltip and none covers request 2’s strip or the key badges (review r1 #5)', not tp['over'] and not tp['tipx'], tp)
        ot_end_checks(pg, tag, lang, v, pick)
        shot(pg, f'{tag}_p5_end_{v}')
        rules(pg, f'{tag} part 5 {v}', lang)
    # R3.20 (watched runs, 2026-10-04): the front's lookup was watched to the end; picked again after the end (A → B → A), its end state shows at once
    s = pg.evaluate("()=>{document.querySelector('#px-stage [data-otpos=\"front\"]').click(); return window.__px.state()}"); pg.wait_for_timeout(300)
    o = pg.evaluate(OT_STATE)
    check(f'{tag} part 5: the front again after the end (A → B → A): its lookup was watched to the end, so its end state shows at once, no replay (R3.20)',
          s['otPos'] == 'front' and s['sweep'] is None and state(pg)['sweep'] is None and o['rec'][0] == '10,000' and o['obs'] != '', (s['sweep'], state(pg)['sweep'], o['rec']))
    if phone and not rm:
        lv = pg.locator('#px-rail [data-lever="middle"]'); lv.scroll_into_view_if_needed(); lv.click(); pg.wait_for_timeout(500)
        iv = in_view(pg, ['#px-pt1', '#px-pt2', '#px-ctrs', '#px-stage [data-otpos="middle"]'])
        check(f'{tag} part 5: on a phone a position picked in the card brings the chart on screen (review r1 #1); its lookup, watched before, shows its end state there at once (R3.20)', state(pg)['sweep'] is None and all(x['ok'] for x in iv), iv)
        sweep_done(pg); settle(pg)
    # a quick burst of clicks ends in the last one; no old sweep draws later
    for v in ['end', 'front', 'middle', 'end', 'front']:
        pg.evaluate(f"()=>document.querySelector('#px-stage [data-otpos=\"{v}\"]').click()"); pg.wait_for_timeout(35)
    sweep_done(pg)
    o1 = pg.evaluate(OT_STATE); pg.wait_for_timeout(2200); o2 = pg.evaluate(OT_STATE)
    check(f'{tag} part 5: a quick burst of position clicks ends in the last one clicked (the front), and nothing older draws after it',
          state(pg)['otPos'] == 'front' and o1['rec'][0] == o2['rec'][0] == '10,000' and o2['t'][0] == '342.9 ms' and o2['amber'] == 0 and o1['obs'] == o2['obs'] != ''
          and pg.evaluate("()=>document.querySelector('#px-stage [data-otpos=\"front\"]').getAttribute('aria-pressed')") == 'true', (o1['rec'], o2['rec'], o2['t']))
    # a language switch during a sweep: the end state in the other language, the part, the position and the prediction kept
    pg.evaluate("()=>document.querySelector('#px-stage [data-otpos=\"middle\"]').click()"); pg.wait_for_timeout(60)
    pg.evaluate("()=>document.getElementById('px-otreplay').click()"); pg.wait_for_timeout(390)   # R3.20: the middle was watched, so ↻ Replay starts its sweep
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(2300)
    st, o = state(pg), pg.evaluate(OT_STATE)
    check(f'{tag} part 5: a language switch during the sweep keeps the part, the position and the prediction, and shows the end state',
          st['step'] == 4 and st['otPos'] == 'middle' and st['otPred'] == pick and st['sweep'] is None and o['rec'][0] == '5,008' and o['obs'] != '', (st, o['rec']))
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    # leaving the part during a sweep stops it; coming back shows the end state
    pg.evaluate("()=>document.querySelector('#px-stage [data-otpos=\"end\"]').click()"); pg.wait_for_timeout(60)
    pg.evaluate("()=>document.getElementById('px-otreplay').click()"); pg.wait_for_timeout(290)   # R3.20: the end was watched, so ↻ Replay starts its sweep
    click(pg, '#px-rail .step.current [data-nav="back"]', 300); pg.wait_for_timeout(1900)
    check(f'{tag} part 5: Back during the sweep stops it: part 4 with its own strip', state(pg)['step'] == 3 and state(pg)['sweep'] is None and not vis(pg, '#px-pairbox') and vis(pg, '#px-stripbox'), state(pg))
    click(pg, '#px-rail .step.current [data-nav="next"]', 400)
    o = pg.evaluate(OT_STATE)
    check(f'{tag} part 5: back in part 5, the end position in its end state', state(pg)['otPos'] == 'end' and state(pg)['sweep'] is None and o['rec'][0] == '16' and o['t'][0] == '8.41 ms', (state(pg), o['rec']))
    # R3.20 (owner, 2026-10-04): the position already on screen is not looked up again; ↻ Replay is the visible replay control
    pg.evaluate("()=>document.querySelector('#px-stage [data-otpos=\"end\"]').click()"); pg.wait_for_timeout(60)
    s1 = state(pg); pg.wait_for_timeout(300); s2 = state(pg); o = pg.evaluate(OT_STATE)
    check(f'{tag} part 5: clicking the position already shown does not replay its lookup (R3.20)', s1['sweep'] is None and s2['sweep'] is None and o['rec'][0] == '16' and o['obs'] != '', (s1['sweep'], s2['sweep'], o['rec']))
    rb = pg.evaluate("()=>{const b=document.getElementById('px-otreplay'); return b&&{shown:b.getClientRects().length>0, text:b.textContent, tip:b.getAttribute('data-tipx')||''}}")
    check(f'{tag} part 5: a visible ↻ Replay button with a tooltip (R3.20)', bool(rb) and rb['shown'] and rb['text'] == Z(lang, '↻ Replay', '↻ 重放') and rb['tip'] == Z(lang, 'Plays request 2’s lookup again, from its first block.', '从第一个 block 起，把请求 2 的查找再播一遍。'), rb)
    pg.evaluate("()=>document.getElementById('px-otreplay').click()"); pg.wait_for_timeout(120)
    s3 = state(pg)
    check(f'{tag} part 5: ↻ Replay looks request 2’s blocks up again (reduced motion: the end state at once)', (s3['sweep'] is None) if reduced_motion(pg) else (s3['sweep'] is not None), s3['sweep'])
    sweep_done(pg); settle(pg)
    o = pg.evaluate(OT_STATE)
    check(f'{tag} part 5: after the replay, the same end state', state(pg)['sweep'] is None and o['rec'][0] == '16' and o['obs'] != '', o['rec'])


def run_rm(b, w, h, lang):
    """reduced motion: no sweep, the end state at once; “Block by block” goes through request 2’s blocks one cell per click, the counters follow"""
    tag = f'rm_{w}_{lang}'
    pg = b.new_page(viewport={'width': w, 'height': h}, reduced_motion='reduce')
    open_drawers(pg); pg.on('pageerror', lambda e: errs.append(f'{tag}: {e}'))
    pg.goto(URL); pg.wait_for_timeout(250); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate("()=>{location.hash='prefix-caching'}"); pg.wait_for_timeout(500)
    check(f'{tag}: reduced motion is on', reduced_motion(pg))
    go_part(pg, 4)
    pg.evaluate("()=>document.querySelector('#px-stage [data-otpos=\"middle\"]').click()"); pg.wait_for_timeout(150)
    if w <= 760:
        ch = pg.locator('#px-stage [data-otpred="before"]'); ch.scroll_into_view_if_needed(); ch.click(); pg.wait_for_timeout(150)
        iv = in_view(pg, ['#px-pt1', '#px-pt2', '#px-ctrs'])
        check(f'{tag} part 5: reduced motion on a phone: after a real tap on a choice the end state is on screen, strips and counters (review r1 #1)', all(x['ok'] for x in iv), iv)
    else:
        pg.evaluate("()=>document.querySelector('#px-rail [data-otpred=\"before\"]').click()"); pg.wait_for_timeout(120)
    st, o = state(pg), pg.evaluate(OT_STATE)
    check(f'{tag} part 5: no sweep: the end state and the result at once', st['sweep'] is None and o['rec'][0] == '5,008' and o['obs'] != '' and not o['cover'], (st['sweep'], o['rec'], o['obs'][:80]))
    check(f'{tag} part 5: the “Block by block” button', o['step'] == Z(lang, '▸ Block by block', '▸ 逐个 block 看'), o['step'])
    pg.evaluate("()=>document.getElementById('px-otstep').click()"); pg.wait_for_timeout(80)
    st, o = state(pg), pg.evaluate(OT_STATE)
    check(f'{tag} part 5 block by block: back to block 0, the counters at 0, the button names the first stop (review r1 #6)', st['swStep'] and st['sweep'] == 0 and o['rec'][0] == '0' and o['nw'][0] == '0' and o['cover'] and o['step'] == Z(lang, 'Show blocks 1 to 312 ▸', '看第 1 到第 312 个 block ▸'), (st, o['rec'], o['step']))
    ks, recs, labels, progs, n = [], [], [], [], 0
    while state(pg)['swStep'] and n < 200:
        pg.evaluate("()=>document.getElementById('px-otstep').click()"); pg.wait_for_timeout(15); n += 1
        s = state(pg)
        if s['sweep'] is not None:
            o = pg.evaluate(OT_STATE); ks.append(s['sweep']); recs.append(o['rec'][0]); labels.append(o['step']); progs.append(text(pg, '#px-prog'))
            if s['sweep'] == 313:
                pg.evaluate("()=>document.getElementById('px-pairbox').scrollIntoView({block:'center'})"); pg.wait_for_timeout(100)
                shot(pg, f'{tag}_p5_block_by_block')
    check(f'{tag} part 5 block by block: stopping at the last hit (312) and at the changed block (313), in order', 312 in ks and 313 in ks and ks == sorted(ks), ks[:50])
    # review r1 #6: five clicks instead of one per cell: all the hits at once, the changed block, the block after it, the rest of the
    # system prompt, then the end state; each button names what the next click shows
    check(f'{tag} part 5 block by block: the stops 312, 313, 314, 625, then the end (review r1 #6)', ks == [312, 313, 314, 625] and n == 5, (ks, n))
    check(f'{tag} part 5 block by block: each button names the next stop',
          labels == [Z(lang, x, y) for x, y in [('Show the changed block ▸', '看改动所在的 block ▸'), ('Show the block after it ▸', '看它后面的一个 block ▸'),
                                                 ('Show the rest of the system prompt ▸', '看 system prompt 剩下的 block ▸'), ('Show the question’s blocks ▸', '看问题的 block ▸')]], labels)
    check(f'{tag} part 5 block by block: each stop says what it showed',
          len(progs) == 4 and Z(lang, 'All of them hit', '它们全部命中') in progs[0] and Z(lang, 'the changed one, misses', '个 block 未命中') in progs[1]
          and Z(lang, 'misses too, although its tokens are the same', '也未命中，尽管它的 token 和请求 1 的一样') in progs[2] and Z(lang, 'Every block of the system prompt after the change missed', 'system prompt 的每个 block 都未命中') in progs[3], progs)
    i312, i313 = ks.index(312), ks.index(313)
    check(f'{tag} part 5 block by block: the counters follow: 0 recomputed through the hits, 16 at the changed block, more after', recs[i312] == '0' and recs[i313] == '16' and recs[-1] != '16', (recs[i312], recs[i313], recs[-1]))
    check(f'{tag} part 5 block by block: the counters at each stop: 0, 16, 32, 5,008', recs == ['0', '16', '32', '5,008'], recs)
    o = pg.evaluate(OT_STATE)
    check(f'{tag} part 5 block by block: after the last block, the end state again', state(pg)['sweep'] is None and not state(pg)['swStep'] and o['rec'][0] == '5,008' and o['step'] == Z(lang, '▸ Block by block', '▸ 逐个 block 看'), (state(pg), o['rec']))
    pg.evaluate("()=>document.querySelector('#px-stage [data-otpos=\"end\"]').click()"); pg.wait_for_timeout(100)
    o = pg.evaluate(OT_STATE)
    check(f'{tag} part 5: another position, at once', state(pg)['sweep'] is None and o['rec'][0] == '16' and o['obs'] != '', o['rec'])
    pg.evaluate("()=>document.getElementById('px-otstep').click()"); pg.wait_for_timeout(60)
    ks = []
    for _ in range(6):
        pg.evaluate("()=>document.getElementById('px-otstep').click()"); pg.wait_for_timeout(15)
        if state(pg)['sweep'] is None: break
        ks.append(state(pg)['sweep'])
    check(f'{tag} part 5 block by block at the end: the stops 624, 625, then the end state', ks == [624, 625] and state(pg)['sweep'] is None, ks)
    rules(pg, f'{tag} part 5', lang)
    pg.close()


def obs_on_screen(pg):
    return pg.evaluate("""()=>{const b=document.querySelector('#px-rail [data-obs]'); if(!b||b.hidden) return null; const r=b.getBoundingClientRect(), top=document.querySelector('.topbar').getBoundingClientRect().bottom;
      return {top:Math.round(r.top), vh:innerHeight, ok:r.top>=top-2&&r.top<innerHeight-40}}""")


def run_guided(b, w, h, lang, dark=False):
    tag = f'{"dark_" if dark else ""}{w}_{lang}'
    pg = page(b, w, h, lang, dark)
    phone = w <= 760
    title = text(pg, '#px-app .labtitle')
    check(f'{tag} title', title == Z(lang, 'How can a 10,000-token system prompt be nearly free?', '10,000 个 token 的 system prompt 怎么做到几乎免费？'), title)
    check(f'{tag} no fidelity tag on top (owner, 2026-10-11 19:28 UTC)', not any(w in text(pg, '#px-app .labmeta') for w in ('Fidelity', '保真度')), text(pg, '#px-app .labmeta'))
    check(f'{tag} outcome line under the title', text(pg, '#px-app .lab-outcome').startswith(Z(lang, 'You will be able to:', '学完你能：')))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#px-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        pg.evaluate(f"()=>document.querySelector('#px-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(200)
        check(f'{tag} popover {pid} opens', pg.evaluate(f"()=>{{const p=document.getElementById('{pid}'); return !!p&&p.getClientRects().length>0}}"))
        rules(pg, f'{tag} popover {pid}', lang)
        pg.evaluate(f"()=>document.querySelector('#px-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(150)
    # part 1: request 1, nothing cached
    check(f'{tag} part 1: the timeline shows request 1 at 342.9 ms', '342.9 ms' in text(pg, '#px-tl'), text(pg, '#px-tl'))
    check(f'{tag} part 1: request 2 waits, unsent', vis(pg, '#px-rows .px-row-wait'))
    check(f'{tag} part 1: no overflow', overflow(pg) <= 0, overflow(pg))
    if phone:
        fs = pg.evaluate("""()=>{const pb=document.querySelector('#px-rail .pbar'), nx=pb&&pb.querySelector('[data-nav="next"]'), st=document.getElementById('px-cells');
          const r=e=>e?e.getBoundingClientRect():null, a=r(pb), c=r(nx), d=r(st);
          return {bar:a&&a.top<innerHeight&&a.height>0, next:!!c&&c.bottom<=innerHeight&&c.height>0, cells:d?Math.round(d.top):null, vh:innerHeight}}""")
        check(f'{tag} phone first screen: the parts’ bar with Next, and the block strip of the chart (R7.3)', fs['bar'] and fs['next'] and fs['cells'] is not None and fs['cells'] < fs['vh'] - 20, fs)
        shot(pg, f'{tag}_first_screen')
    shot(pg, f'{tag}_p1', full=phone)
    rules(pg, f'{tag} part 1', lang)
    ans_absent = lambda: ('8.13' not in all_text(pg)) and ('10,000 tokens cached' not in all_text(pg)) and ('命中 10,000' not in all_text(pg))
    check(f'{tag} part 1: request 2’s answer (8.13 ms) appears nowhere yet', ans_absent())
    # part 2: the gate
    click(pg, '#px-rail [data-nav="next"]')
    st = state(pg)
    check(f'{tag} part 2 reached', st['step'] == 1, st)
    nd = pg.evaluate("()=>[...document.querySelectorAll('#px-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
    check(f'{tag} part 2: every Next is disabled before a prediction', nd and all(nd), nd)
    check(f'{tag} part 2: Settings wait for the prediction (note shown, controls hidden)', vis(pg, '#px-prelock') and not vis(pg, '#px-ctl-P'))
    check(f'{tag} part 2: the answer appears nowhere (text and tooltips)', ans_absent())
    click(pg, '#px-rail [data-pred="half"]', 200)
    check(f'{tag} part 2: “Saved … you can still change it”', Z(lang, 'Saved. You can still change it before part 3 shows the result.', '已记下。第 3 部分给出结果之前，还可以改。') in text(pg, '#px-rail .step.current'))
    check(f'{tag} part 2: the pick shows as pressed with a radio dot', pg.evaluate("()=>{const b=document.querySelector('#px-rail [data-pred=\"half\"]'); return b.getAttribute('aria-pressed')==='true'&&getComputedStyle(b,'::before').content!=='none'}"))
    nd = pg.evaluate("()=>[...document.querySelectorAll('#px-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
    check(f'{tag} part 2: Next enabled after a prediction', nd and not any(nd), nd)
    check(f'{tag} part 2: still no answer after the pick', ans_absent())
    shot(pg, f'{tag}_p2')
    rules(pg, f'{tag} part 2', lang)
    # part 3: send request 2
    click(pg, '#px-rail [data-nav="next"]')
    check(f'{tag} part 3: Next waits for the send; the answer still hidden', pg.evaluate("()=>document.querySelector('#px-rail .step.current [data-nav=\"next\"]').disabled") and ans_absent())
    check(f'{tag} part 3: no ▲▼ badge on entering a part', not badges(pg), badges(pg))
    if w == 1440 and not dark:
        pg.set_viewport_size({'width': 1440, 'height': 1000}); pg.wait_for_timeout(200)
    click(pg, '#px-rail [data-act="send"]', 900)
    o = text(pg, '#px-rail [data-obs]')
    check(f'{tag} part 3: the result first, then the verdict with the right answer, then why (R3.8, R3.10)',
          o.find('8 ms') > -1 and o.find(Z(lang, 'the right answer is', '正确答案是')) > o.find('8 ms') and o.find(Z(lang, 'Why:', '原因：')) > o.find(Z(lang, 'the right answer is', '正确答案是'))
          and Z(lang, 'after about 8 ms instead of about 340 ms', '从约 340 ms 降到约 8 ms') in o, o[:400])
    check(f'{tag} part 3: the strip lights up: request 2 selected, 10,000 tokens cached', state(pg)['sel'] == 1 and text(pg, '#px-metrics [data-k="hit"] .v') == '10,000' and text(pg, '#px-metrics [data-k="w"] .v') == '8.13 ms', text(pg, '#px-metrics'))
    check(f'{tag} part 3: a badge after the learner’s own send', any(('▼' in x or '▲' in x) for x in badges(pg)), badges(pg))
    if w == 1440 and not dark:
        oc = obs_on_screen(pg)
        check(f'{tag} part 3: after the send the result is on screen at 1440×1000 (R3.10)', oc and oc['ok'], oc)
        pg.set_viewport_size({'width': w, 'height': h}); pg.wait_for_timeout(200)
    shot(pg, f'{tag}_p3', full=phone)
    rules(pg, f'{tag} part 3', lang)
    # the language switch keeps the part, the prediction and the send
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(350)
    st2 = state(pg)
    check(f'{tag} part 3: a language switch keeps the part, the prediction and the sent request', st2['step'] == 2 and st2['pred'] == 'half' and st2['sent'] and vis(pg, '#px-rail [data-obs]'), st2)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(350)
    # part 4: the levers
    click(pg, '#px-rail .step.current [data-nav="next"]')
    check(f'{tag} part 4: back to the initial setup, request 2 selected, no badge on entering', state(pg)['change'] == 'none' and state(pg)['sel'] == 1 and not badges(pg), (state(pg), badges(pg)))
    if w == 1440 and not dark:
        pg.set_viewport_size({'width': 1440, 'height': 1000}); pg.wait_for_timeout(200)
    SAYS = {'dateStart': Z(lang, 'not one of request 2’s tokens', '请求 2 一个 token 也没'), 'edit': Z(lang, 'about half of the system prompt', 'system prompt 的约一半'),
            'dateAfter': Z(lang, 'still finds its whole system prompt', '仍在缓存里找到整段 system prompt'), 'nameStart': Z(lang, 'not one of request 2’s tokens', '请求 2 一个 token 也没')}
    for v, hit, t in [('dateStart', '0', '343.4 ms'), ('edit', '4,992', '188.9 ms'), ('dateAfter', '10,000', '8.25 ms'), ('nameStart', '0', '343.2 ms')]:
        click(pg, f'#px-rail [data-lever="{v}"]', 350)
        o = text(pg, '#px-rail [data-obs]')
        pressed = pg.evaluate(f"()=>document.querySelector('#px-rail [data-lever=\"{v}\"]').getAttribute('aria-pressed')")
        res = text(pg, f'#px-rail [data-lever="{v}"] .lv-r')
        check(f'{tag} part 4 {v}: pressed, its result {t} on its title row, {hit} cached on the card, and the result in words (R6.20)', pressed == 'true' and res == t and text(pg, '#px-metrics [data-k="hit"] .v') == hit and SAYS[v] in o, (pressed, res, text(pg, '#px-metrics'), o[:240]))
        if v == 'edit':
            lead = pg.evaluate("()=>{const e=document.querySelector('#px-rail .pbar .pb-t'); return e?e.textContent:''}")
            check(f'{tag} part 4 one word: the result says what happens but not why, and leads to part 5 (review r1 #4, R3.8)',
                  not any(x in o + lead for x in [Z(lang, 'every block after it', '之后的每个 block'), Z(lang, 'every block misses', '每个 block 都未命中'), Z(lang, 'From the block that holds it on', '从装着它的 block 起')])
                  and Z(lang, 'Where does that miss start', '这段未命中从哪里开始') in o
                  and pg.evaluate("()=>{const b=document.querySelector('#px-rail [data-obs] [data-go=\"4\"]'); return !!b&&(b.dataset.tipx||'').length>10}"), (o[:300], lead))
        if v == 'dateStart':
            check(f'{tag} part 4 date first: hatched blocks in the strip (same tokens after a change)', pg.evaluate("()=>document.querySelectorAll('#px-cells .px-s2').length>0"))
            if w == 1440 and not dark:
                oc = obs_on_screen(pg)
                check(f'{tag} part 4: after a lever the result is on screen at 1440×1000 (R3.10)', oc and oc['ok'], oc)
            shot(pg, f'{tag}_p4_date_first', full=phone)
            rules(pg, f'{tag} part 4 date first', lang)
    tried = pg.evaluate("()=>[...document.querySelectorAll('#px-rail [data-lever] .lv-r')].map(x=>x.textContent)")
    check(f'{tag} part 4: every tried lever keeps its result (R3.6)', all(x for x in tried), tried)
    if w == 1440 and not dark:
        pg.set_viewport_size({'width': w, 'height': h}); pg.wait_for_timeout(200)
    shot(pg, f'{tag}_p4', full=phone)
    rules(pg, f'{tag} part 4', lang)
    # part 5 (Batch 4): the experiment, one changed token; then part 6, the memory (one part later than before Batch 4)
    part5(pg, tag, lang, w, dark)
    FRAME = """(sel)=>{const f=()=>{const s=window.__px.state(), cap=document.getElementById('px-strip-cap').textContent, st=document.getElementById('px-status').textContent;
        return {seq:s.seq, sel:s.sel, cap, st, wait:document.querySelectorAll('#px-rows .px-row-wait').length};};
      if(sel) document.querySelector(sel).click(); return f();}"""
    def frames(sel):
        out = [pg.evaluate(FRAME, sel)]
        for ms_ in (120, 200):
            pg.wait_for_timeout(ms_); out.append(pg.evaluate(FRAME, None))
        return out
    if not reduced_motion(pg):
        fr = frames('#px-rail .step.current [data-nav="next"]')
        ok = all(f['seq'] is not None and f['sel'] < f['seq'] and Z(lang, 'Request 8', '请求 8') not in f['cap'] + f['st'] for f in fr) and fr[0]['seq'] == 1 and fr[0]['sel'] == 0 and fr[0]['wait'] == 7
        check(f'{tag} part 6: from its first frame the chart shows request 1 while the others wait, never request 8’s result (review r1 #9)', ok, fr)
        pg.wait_for_function("()=>window.__px.state().seq===null", timeout=15000); pg.wait_for_timeout(250)
    else:
        click(pg, '#px-rail .step.current [data-nav="next"]')
    rows = pg.evaluate("()=>document.querySelectorAll('#px-rows .px-row').length")
    check(f'{tag} part 6: “The cost: GPU memory” is part 6 of 7', state(pg)['step'] == 5 and Z(lang, 'part 6 of 7', '第 6 / 7 部分') in pg.evaluate("()=>document.querySelector('#px-rail .rail-top').textContent"))
    check(f'{tag} part 6: 8 requests to apps A, B, C in the chart, the requests first', rows == 8 and pg.evaluate("()=>document.getElementById('px-graph').dataset.lead") == 'rows', rows)
    check(f'{tag} part 6: Next waits for a memory pick', pg.evaluate("()=>document.querySelector('#px-rail .step.current [data-nav=\"next\"]').disabled"))
    # R3.20 (owner, 2026-10-04): a pick replays only the rows that change: the rows before the first changed request stay, and the arrival
    # starts at that request. Apps A, A, B, A, C, A, B, A with LRU: room for 2 differs from the whole pool from request 5 on; room for 1
    # differs from room for 2, and the whole pool from room for 1, from request 3 on
    FIRST = {'2': 4, '1': 2, 'pool': 2}
    for v, hits in [('2', 4), ('1', 1), ('pool', 5)]:
        seen_ = v == 'pool'   # R3.20 (watched runs, 2026-10-04): the whole pool's arrival was watched to the end on entering part 6
        if not reduced_motion(pg):
            fr = frames(f'#px-rail [data-lever="{v}"]')
            if seen_:
                check(f'{tag} part 6 memory pool again after room for 1 (A → B → A): its arrival was watched to the end on entering part 6, so all 8 rows show at once, no replay (R3.20)',
                      all(f['seq'] is None and f['wait'] == 0 for f in fr), fr)
            else:
                f0 = FIRST[v]
                check(f'{tag} part 6 memory {v}: only the rows that change arrive again: the first frame shows request {f0 + 1} with the rows before it, never request 8’s result while rows wait (review r1 #9, R3.20)',
                      all(f['seq'] is not None and f['sel'] < f['seq'] and Z(lang, 'Request 8', '请求 8') not in f['cap'] + f['st'] for f in fr) and fr[0]['seq'] == f0 + 1 and fr[0]['sel'] == f0 and fr[0]['wait'] == 8 - (f0 + 1), fr)
            pg.wait_for_timeout(100); settle(pg)
        else:
            click(pg, f'#px-rail [data-lever="{v}"]', 350)
        if not reduced_motion(pg) and not seen_:
            # the 8 requests arrive one at a time (Batch 3 review): later rows wait, the line under the rows says what the latest one did, the result waits
            mid = pg.evaluate("()=>({seq:window.__px.state().seq, wait:document.querySelectorAll('#px-rows .px-row-wait').length, sum:document.getElementById('px-sum').textContent, obs:!document.querySelector('#px-rail [data-obs]')||document.querySelector('#px-rail [data-obs]').hidden})")
            check(f'{tag} part 6 memory {v}: the requests arrive one at a time; the result waits for the last', mid['seq'] is not None and mid['wait'] > 0 and mid['obs'], mid)
            pg.wait_for_function("()=>window.__px.state().seq===null", timeout=15000); pg.wait_for_timeout(250)
        chips = pg.evaluate("()=>document.querySelectorAll('#px-rows .px-rc i').length")
        check(f'{tag} part 6 memory {v}: every row shows which apps’ system prompts the cache holds after it (A, B, C)', chips == 24, chips)
        o = text(pg, '#px-rail [data-obs]')
        cached = pg.evaluate("()=>[...document.querySelectorAll('#px-rows .px-row')].filter(r=>r.querySelector('.px-s0')).length")
        check(f'{tag} part 6 memory {v}: {hits} of 8 hit, in the card and on the rows', cached == hits and Z(lang, f'{hits} of the 8 requests hit', f'{hits} 个命中') in o, (cached, o[:200]))
    if not reduced_motion(pg):
        # R3.20: the option already shown, and coming back to part 6, show the finished rows at once; “Send them again” replays them
        fr = frames('#px-rail [data-lever="pool"]')
        o = text(pg, '#px-rail [data-obs]')
        check(f'{tag} part 6: picking the option already shown replays nothing: all 8 rows and the result stay (R3.20)', all(f['seq'] is None and f['wait'] == 0 for f in fr) and Z(lang, '5 of the 8 requests hit', '5 个命中') in o, (fr, o[:120]))
        click(pg, '#px-rail .step.current [data-nav="back"]', 300)
        fr = frames('#px-rail .step.current [data-nav="next"]')
        o = text(pg, '#px-rail [data-obs]')
        check(f'{tag} part 6: back from part 5, the run already watched is not replayed: all 8 rows and the result at once (R3.20)', state(pg)['step'] == 5 and all(f['seq'] is None and f['wait'] == 0 for f in fr) and Z(lang, '5 of the 8 requests hit', '5 个命中') in o, (fr, o[:120]))
        rp = pg.locator('#px-graph .px-replay')
        check(f'{tag} part 6: “Send them again” is there to replay the arrival (R3.20)', rp.count() == 1 and rp.is_visible(), rp.count())
        fr = frames('#px-graph .px-replay')
        check(f'{tag} part 6: “Send them again” replays all 8 from request 1 (R3.20)', fr[0]['seq'] == 1 and fr[0]['sel'] == 0 and fr[0]['wait'] == 7, fr)
        pg.wait_for_function("()=>window.__px.state().seq===null", timeout=15000); pg.wait_for_timeout(250); settle(pg)
    shot(pg, f'{tag}_p6', full=phone)
    rules(pg, f'{tag} part 6', lang)
    # a row selects its request
    click(pg, '#px-rows [data-sel="2"]', 250)
    check(f'{tag} part 6: a row selects its request (request 3, app B)', state(pg)['sel'] == 2 and Z(lang, 'Request 3', '请求 3') in text(pg, '#px-strip-cap'), text(pg, '#px-strip-cap'))
    # part 7 (part 6 before Batch 4)
    click(pg, '#px-rail .step.current [data-nav="next"]')
    t6 = text(pg, '#px-rail .step.current')
    check(f'{tag} part 7: “What to remember” is part 7 of 7', state(pg)['step'] == 6 and Z(lang, 'part 7 of 7', '第 7 / 7 部分') in pg.evaluate("()=>document.querySelector('#px-rail .rail-top').textContent"))
    check(f'{tag} part 7: three misconceptions, and the button to the scenarios (R3.12)', t6.count('✗') == 3 and vis(pg, '#px-rail [data-mode-go="sandbox"]'), t6[:200])
    nl = pg.evaluate("""()=>{const shown=x=>!!x&&getComputedStyle(x).display!=='none'&&x.getBoundingClientRect().height>0; const r=document.querySelector('#px-rail .next-lab'), e=document.querySelector('#view-dyn .next-lab-end .next-lab');
      return innerWidth<=760 ? shown(e) : shown(r)}""")
    check(f'{tag} part 7: the next-lab footer (R3.11)', nl)
    shot(pg, f'{tag}_p7', full=phone)
    rules(pg, f'{tag} part 7', lang)
    click(pg, '#px-rail [data-mode-go="sandbox"]', 400)
    check(f'{tag} “See the scenarios” opens Scenarios', state(pg)['mode'] == 'sandbox')
    check(f'{tag} no overflow at the end of the tour', overflow(pg) <= 0, overflow(pg))
    pg.close()


def order_to(pg, target):
    """move the challenge's parts with the card's ▲ buttons until they are in the target order"""
    for i, p_ in enumerate(target):
        for _ in range(6):
            cur = state(pg)['chOrder']
            if cur.index(p_) <= i: break
            pg.evaluate(f"()=>document.querySelector('#px-order [data-part=\"{p_}\"][data-act=\"up\"]').click()"); pg.wait_for_timeout(60)
    pg.wait_for_timeout(200)


def run_modes(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    phone = w <= 760
    mode(pg, 'sandbox')
    presets = pg.evaluate("()=>[...document.querySelectorAll('#px-rail [data-preset]')].length")
    check(f'{tag} scenarios: four, with the lead-in that says what to pick and what to watch (G1, the owner 2026-10-08)', presets == 4 and Z(lang, 'Pick a scenario and see how much sooner the first token comes once the cache hits.', '选一个场景，看看命中缓存后，首 token 能快多少。') in text(pg, '#px-rail .railcard') and Z(lang, 'do not stack', '不会叠加') not in text(pg, '#px-rail .railcard'))
    for i in range(presets):
        click(pg, f'#px-rail [data-preset="{i}"]', 350)
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#px-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        ins = text(pg, '#px-insight')
        check(f'{tag} scenario {i}: pressed alone, with its result', pressed.count('true') == 1 and pressed[i] == 'true' and len(ins) > 80, (pressed, ins[:120]))
        check(f'{tag} scenario {i}: no overflow', overflow(pg) <= 0, overflow(pg))
        rules(pg, f'{tag} scenario {i}', lang)
        if i == 2:
            check(f'{tag} scenario: the long document in words: the first question about 8.1 s, the last about 27 ms (R6.20)', '8.1 s' in ins and '27 ms' in ins, ins[:240])
            shot(pg, f'{tag}_scenario_doc', full=phone)
        if i == 1:
            check(f'{tag} scenario: multi-turn chat: each turn computes only its new message (R6.20)', Z(lang, 'computes only its new message', '只需要算新消息') in ins, ins[:240])
            shot(pg, f'{tag}_scenario_chat', full=phone)
    click(pg, '#px-rail [data-preset="0"]', 350)
    hsh = pg.evaluate("()=>location.hash")
    check(f'{tag} scenarios: the URL carries the setup', hsh == '#prefix-caching~10000_200_5_same_none_1_pool', hsh)
    shot(pg, f'{tag}_scenario_support', full=phone)
    pg2 = page(b, w, h, lang, hash_='prefix-caching~2000_100_5_chat_none_1_pool')
    st = state(pg2)
    check(f'{tag} hash round-trip: Scenarios with the chat setup, “Multi-turn chat” pressed', st['mode'] == 'sandbox' and (st['P'], st['O'], st['n'], st['kind']) == (2000, 100, 5, 'chat')
          and pg2.evaluate("()=>document.querySelector('#px-rail [data-preset=\"1\"]').getAttribute('aria-pressed')") == 'true', st)
    pg2.close()
    # a change under Settings: no scenario matches any more, and the page does not move (R1.8, R3.2)
    pg.evaluate("()=>document.querySelector('#px-app [data-flow-nav] button[data-flow=free]').click()"); pg.wait_for_timeout(300)   # settings are changed in 自由调参 (owner, 2026-10-11)
    y0 = pg.evaluate("()=>scrollY")
    pg.evaluate("()=>{const i=document.querySelector('#px-ctl-O-in'); i.value=4; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(300)
    if not phone:
        check(f'{tag} a change under Settings does not scroll the page', abs(pg.evaluate("()=>scrollY") - y0) < 2)
    pg.evaluate("()=>document.querySelector('#px-app [data-flow-nav] button[data-flow=scen]').click()"); pg.wait_for_timeout(300)
    pressed = pg.evaluate("()=>[...document.querySelectorAll('#px-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: after a change under Settings no scenario is pressed, and the card says so', 'true' not in pressed and Z(lang, 'match none', '都不一样') in text(pg, '#px-insight'), pressed)
    # Settings: the chat kind, the apps and the memory buttons work
    click(pg, '#px-ctl-apps [data-v="3"]', 250); click(pg, '#px-ctl-mem [data-v="1"]', 250)
    st = state(pg)
    check(f'{tag} Settings: 3 apps and room for 1 request', st['apps'] == 3 and st['mem'] == 1, st)
    click(pg, '#px-ctl-kind [data-v="chat"]', 250)
    check(f'{tag} Settings: chat turns turn the differing prefixes off', state(pg)['kind'] == 'chat' and state(pg)['apps'] == 1 and state(pg)['change'] == 'none')
    rules(pg, f'{tag} settings', lang)
    # the challenge, by the card's button (R3.12)
    click(pg, '#px-rail [data-preset="0"]', 300)
    click(pg, '#px-rail [data-mode-go="challenge"]', 400)
    st = state(pg)
    check(f'{tag} challenge: the initial order, nothing submitted', st['mode'] == 'challenge' and st['chOrder'] == ['date', 'name', 'sys', 'shots', 'hist', 'msg'] and not st['chSubmitted'], st)
    check(f'{tag} challenge: Settings locked with a note, no grayed controls (R4.2)', vis(pg, '#px-lock') and not vis(pg, '#px-ctl-P') and not vis(pg, '#px-ctl-mem'))
    card = NB(text(pg, '#px-rail .railcard'))
    check(f'{tag} challenge: the goal in full: model, GPU, the requests, the target (R6.8)', all(x in card for x in ['Llama-3.1-8B', 'H100 SXM', '4,400', '8'] + ([] if lang == 'zh' else ['Day 1: Alice, Bob, Carol'])), card[:300])
    check(f'{tag} challenge: six parts with move buttons', pg.evaluate("()=>document.querySelectorAll('#px-order .px-part').length") == 6)
    check(f'{tag} challenge: the capability card after the task', vis(pg, '#px-rail [data-exits]'))
    check(f'{tag} challenge: no overflow', overflow(pg) <= 0, overflow(pg))
    check(f'{tag} challenge: live progress 0 of 4,400', text(pg, '#px-ch-now') == '0' and not pg.evaluate("()=>document.getElementById('px-ch-goal').classList.contains('met')"))
    shot(pg, f'{tag}_challenge', full=phone)
    rules(pg, f'{tag} challenge', lang)
    # tier 3: not met, with a hint and “show all options”
    click(pg, '#px-rail [data-act="submit"]', 350)
    res = text(pg, '#px-ch-res')
    check(f'{tag} challenge tier 3: not met, with a hint and “Show the answer” (R4.4)', res.startswith(Z(lang, 'Not yet', '还没达到')) and Z(lang, 'Hint', '提示') in res and vis(pg, '#px-ch-res [data-act="reveal"]') and not vis(pg, '#px-ch-res .cmp')
          and text(pg, '#px-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), res[:200])
    click(pg, '#px-ch-res [data-act="reveal"]', 300)
    rows = pg.evaluate("()=>[...document.querySelectorAll('#px-ch-res .cmp-row')].map(r=>({ok:r.classList.contains('ok'),me:r.classList.contains('me'),t:r.querySelector('.cmp-lab span').textContent}))")
    # the owner's review of batch 8: a brief chart, the two orders that pass and the initial setup (= the pick); the two single moves are left out and counted
    check(f'{tag} challenge chart: three orders, two green, the initial setup is your pick', len(rows) == 3 and sum(r['ok'] for r in rows) == 2 and sum(r['me'] for r in rows) == 1 and Z(lang, 'initial setup', '初始设置') in text(pg, '#px-ch-res .cmp'), rows)
    check(f'{tag} challenge chart: the note counts the 2 orders left out', text(pg, '#px-ch-res .cmp-note') == Z(lang, 'The other 2 orders compared here move only the date or only the name; they miss the goal and are not shown.', '其余 2 种顺序只移了日期或名字，没达到目标，图里没有列出。'), text(pg, '#px-ch-res .cmp-note'))
    ttl = text(pg, '#px-ch-res .cmp-title')
    check(f'{tag} challenge chart: the title names the quantity and the conditions, not “by prompt order”', 'H100' in ttl and '4,400' in ttl and Z(lang, 'by prompt order', '不同 prompt 顺序') not in ttl, ttl)
    tips1 = pg.evaluate("()=>[...document.querySelectorAll('#px-ch-res > p .def')].map(d=>d.dataset.tipx||'')")
    check(f'{tag} challenge: the single moves are named in words, each with its average on hover', len([t_ for t_ in tips1 if Z(lang, 'under 4,400', '不到 4,400') in t_]) == 2, tips1)
    check(f'{tag} challenge: the side-by-side is closed by default (the owner’s review of batch 8)', pg.evaluate("()=>{const d=document.getElementById('px-alx'); return !!d&&!d.open}"))
    xt = pg.evaluate("()=>[...document.querySelectorAll('#view-dyn [data-tipx]')].filter(e=>!e.closest('[id$=\"-mathbody\"],[id$=\"-math\"]')).map(e=>e.getAttribute('data-tipx')).filter(t=>t.includes('精确值是')||t.includes('Exactly'))")
    check(f'{tag} no hover outside Show the math gives a number’s exact value (the owner’s clarification of 2026-10-07)', not xt, xt[:2])
    shot(pg, f'{tag}_challenge_wrong', full=phone)
    rules(pg, f'{tag} challenge wrong', lang)
    # tier 2: met, but more parts moved than needed
    order_to(pg, ['shots', 'sys', 'name', 'hist', 'date', 'msg'])
    res = text(pg, '#px-ch-res')
    check(f'{tag} challenge tier 2: met, but a smaller change would do', res.startswith(Z(lang, 'Goal met', '达成目标')) and Z(lang, 'A smaller change would do', '其实改得更少也行') in res and '4,500' in res, res[:260])
    check(f'{tag} challenge: live progress turns green (R4.3)', pg.evaluate("()=>document.getElementById('px-ch-goal').classList.contains('met')"))
    # tier 1: met with the fewest moves
    order_to(pg, ['sys', 'shots', 'name', 'hist', 'msg', 'date'])
    res = text(pg, '#px-ch-res')
    check(f'{tag} challenge tier 1: met with 2 moved parts, 4,594 cached tokens per request', res.startswith(Z(lang, 'Goal met', '达成目标')) and '4,594' in res and Z(lang, 'the fewest that meet the goal', '达标所需的最少') in res, res[:260])
    check(f'{tag} challenge: baseline → effect → reason, and a specific caveat (R5.5, R6.5)', all(x in res for x in [Z(lang, 'Initial setup', '初始设置'), Z(lang, 'Why:', '原因：'), Z(lang, 'Note:', '注意：'), 'vllm:prefix_cache_hits']), res[-500:])
    # GPT-6 review of v9, #2: the answer names the whole best order, and day 1 against day 2 shows where each order stops matching
    check(f'{tag} challenge: the answer states the full best order', Z(lang, 'One best order: System instructions, few-shot examples, the user’s name, the chat history, the new message, today’s date (4,594 cached per request)',
          '最好的顺序之一：系统指令、few-shot 示例、用户名、对话历史、新消息、今天的日期（平均每个请求缓存 4,594）') in res, res[-900:])
    check(f'{tag} side by side: still closed after the order changes', pg.evaluate("()=>!document.getElementById('px-alx').open"))
    pg.evaluate("()=>{document.getElementById('px-alx').open=true}"); pg.wait_for_timeout(150)   # opened, so the layout checks below measure it
    al = pg.evaluate("""()=>[...document.querySelectorAll('#px-al .px-al-fig')].map(f=>({o:f.dataset.o, me:f.classList.contains('me'),
      a:[...f.querySelectorAll('.px-al-p[data-side="1"]')].map(e=>e.dataset.st), b:[...f.querySelectorAll('.px-al-p[data-side="2"]')].map(e=>e.dataset.st),
      cutRow:getComputedStyle(f.querySelector('.px-al-cut')).gridRowStart, br:!!f.querySelector('.px-al-br'), foot:f.querySelector('.px-al-foot').textContent.replace(/\\u00a0/g,' '),
      ovf:f.scrollWidth-f.clientWidth, trunc:[...f.querySelectorAll('.px-al-p b, .px-al-ch')].filter(e=>e.scrollWidth>e.clientWidth+0.5).map(e=>e.textContent)}))""")
    want = [('sys,shots,name,hist,msg,date', 5, ['same', 'new', 'new'], '472', '5,648'), ('sys,shots,name,hist,date,msg', 4, ['same', 'same', 'new', 'new'], '616', '5,504'),
            ('sys,shots,hist,name,msg,date', 3, ['same', 'same', 'same', 'new', 'new'], '632', '5,488')]
    check(f'{tag} side by side: A, B, C with the cut after 5, 4 and 3 matching parts, day 2 hatched then gray after it, the counts under each',
          len(al) == 3 and all(f['o'] == o and f['a'][:m] == ['hit'] * m and set(f['a'][m:]) == {'held'} and f['b'][:m] == ['hit'] * m and f['b'][m:] == tail and f['cutRow'] == str(2 + m)
                                 and comp in f['foot'] and hit in f['foot'] and f['br'] for f, (o, m, tail, comp, hit) in zip(al, want)), al)
    check(f'{tag} side by side: your pick marked on A, nothing cut off or overflowing', [f['me'] for f in al] == [True, False, False] and all(f['ovf'] <= 0 and not f['trunc'] for f in al), al)
    # review of v10: the footer says the hit is the matching tokens rounded down to whole blocks (S1); no tooltip compares a part with itself (S2)
    check(f'{tag} side by side: each footer gives the matching tokens and the whole-block hit', all(m_ in f['foot'] for f, m_ in zip(al, ['5,658', '5,508', '5,500'])), [f['foot'] for f in al])
    same_tips = pg.evaluate("""()=>[...document.querySelectorAll('#px-al .px-al-p[data-st="same"]')].map(e=>e.dataset.tipx)""")
    selfcmp = [t_ for t_ in same_tips if re.search(r'after (.+?) and here after \1\.|前面是(.+?)，这里前面是\2。', t_)]
    check(f'{tag} side by side: hatched tooltips never say “after X and here after X”', same_tips and not selfcmp, selfcmp[:2])
    lasthit = pg.evaluate("""()=>[...document.querySelectorAll('#px-al .px-al-fig')].map(f=>{const h=[...f.querySelectorAll('.px-al-p[data-side="2"][data-st="hit"]')]; return h.length?h[h.length-1].dataset.tipx:''})""")
    check(f'{tag} side by side: the last amber part’s tooltip says its tail is computed again (not a whole block)', all(Z(lang, 'do not fill a whole 16-token block', '凑不满一个 16 个 token 的 block') in t_ for t_ in lasthit), lasthit)
    alt = text(pg, '#px-al')
    check(f'{tag} side by side: the question answered, and the premises stated with what changes when they change',
          all(x in alt for x in Z(lang, ['Why not put the date or the name right before the new message?', 'This depends on the setup', 'B would match A', 'answer ① would come from the cache too'],
                                         ['为什么日期和用户名不放在新消息前面？', '这个结论依赖题设', 'B 和 A 一样好', '回答①也能从缓存读'])), alt[-600:])
    # review of v10 (S6, first step): a link under the verdict leads to the side-by-side
    pg.evaluate("()=>{document.getElementById('px-alx').open=false; window.scrollTo(0,0)}"); pg.wait_for_timeout(100)
    pg.evaluate("()=>document.querySelector('#px-ch-res [data-jump=\"alq\"]').click()"); pg.wait_for_timeout(900)
    top_al = pg.evaluate("()=>{const r=document.getElementById('px-al').getBoundingClientRect(); return [r.top, innerHeight]}")
    check(f'{tag} side by side: the question under the verdict opens it and leads to it', 40 <= top_al[0] < top_al[1] - 100 and pg.evaluate("()=>document.getElementById('px-alx').open"), top_al)
    if not phone: pg.evaluate("()=>document.getElementById('px-al').scrollIntoView({block:'start'})"); pg.wait_for_timeout(150)
    shot(pg, f'{tag}_challenge_sbs', full=phone)
    shot(pg, f'{tag}_challenge_met', full=phone)
    rules(pg, f'{tag} challenge met', lang)
    for t in (0, 1):
        pg.evaluate(f"()=>document.querySelector('#px-rail [data-exits] [data-ex=\"tab\"][data-i=\"{t}\"]').click()"); pg.wait_for_timeout(150)
        for v in pg.evaluate("()=>[...document.querySelectorAll('#px-rail [data-exits] .ex-opts .choice')].map(b=>+b.dataset.v)"):
            pg.evaluate(f"()=>document.querySelector('#px-rail [data-exits] .ex-opts .choice[data-v=\"{v}\"]').click()"); pg.wait_for_timeout(150)
            rules(pg, f'{tag} capability card tab {t + 1}, option {v}', lang)
    # the language switch keeps the order and the answer
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} challenge: a language switch keeps the order and the submitted result', st['chOrder'] == ['sys', 'shots', 'name', 'hist', 'msg', 'date'] and st['chSubmitted'] and vis(pg, '#px-ch-res .cmp'), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    # the mini result bar on phones: once the chart has scrolled away, a short status and a full sentence
    if phone:
        mode(pg, 'guided'); go_part(pg, 3)
        pg.evaluate("()=>{const t=document.getElementById('px-rail').querySelector('.step.current .levers'); window.scrollTo(0, t.getBoundingClientRect().top+scrollY-60)}"); pg.wait_for_timeout(500)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, status:document.getElementById('dkStatus').textContent, read:document.getElementById('dkRead').textContent.replace(/\\u00a0/g,' ')}}")
        check(f'{tag} mini result bar: shown once the chart scrolls away, with a short status and the times', dk['shown'] and len(dk['status']) <= 14 and '8.13 ms' in dk['read'] and '342.9 ms' in dk['read'], dk)
        shot(pg, f'{tag}_dock')
        rules(pg, f'{tag} dock', lang)
    pg.close()


# G1 and G2, the owner's message of 2026-10-08 (reviews/owner-b9-2026-10-08.md): a lead-in above each group of options instead of
# “one at a time; they do not stack”, and no sentence that leans on a detail of an earlier part. Checked on the lab's source, both languages.
_G_SRC = (P / 'labs' / 'prefix-caching.js').read_text()
for _b in ["一次只能选", "一次选一个", "一次选一种", "一次只选", "不会叠加", "do not stack", "One at a time", "pick one at a time", "Pick one at a time", "Try one change at a time", "一次试一种"]:
    check(f'G1: no “{_b}” left in the lab (EN and ZH)', _b not in _G_SRC)
for _s in ["Pick a change and see how much of the cache request 2 still hits.", "选一种改动，看看请求 2 还能命中多少缓存。", "How much GPU memory may the cache use? Pick one and see how many of the 8 requests hit.", "缓存最多能用多少显存？选一种，看看 8 个请求里有几个命中。", "Pick a scenario and see how much sooner the first token comes once the cache hits.", "选一个场景，看看命中缓存后，首 token 能快多少。"]:
    check(f'G1: the lead-in “{_s[:48]}” is in the lab', _s in _G_SRC)
for _s in ["halfway through the system prompt.", "正好在 system prompt 的中间。", "so a later request with the same prefix reuses its cached keys and values", "所以后来的请求只要前缀相同，就能复用缓存里这段 key 和 value", "A request that hits still computes its own question", "命中的请求仍要算它自己的问题", "A few different tokens at the very start, such as today’s date", "最前面差几个 token（比如今天的日期）", "A request that hits still waits a few milliseconds", "命中的请求，首 token 仍要等几毫秒"]:
    check(f'G2: the rewritten sentence “{_s[:48]}” is in the lab', _s in _G_SRC)
for _s in ["It is the word that part", "部分改的那个词", "so request 2 reuses the cached system prompt", "所以请求 2 复用缓存里的 system prompt", "with today’s date put first, not one of request 2’s tokens hits", "把今天的日期放到最前面，请求 2 一个 token 也没命中"]:
    check(f'G2: the sentence that leaned on an earlier part, “{_s[:48]}”, is gone', _s not in _G_SRC)

# polish r5 (B4, R7.12): entering the parts this round fitted, the way a reader does (Next, or the part's title while Next waits for an
# answer), at 1440 x 900 the stage fits without scrolling inside, with 30 px to spare in Chinese. Sampled for 6.6 s after entering, so a
# part that plays on entry (part 6 of prefix-caching sends its 8 requests in 5.3 s) fits all the way through; measured in standard mode (the page wrapped in a doctype, as it is published)
FIT = """p=>{const s=document.getElementById(p+'-stage'), c=document.querySelector('#'+p+'-rail .steps>.step.current');
  return {step:c?[...c.parentNode.children].indexOf(c)+1:0, spare:Math.round(parseFloat(getComputedStyle(s).maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight)),
          scroll:s.scrollHeight-s.clientHeight, top:Math.round(s.getBoundingClientRect().top)}}"""


def fit_next(pg, k):
    """into part k as a reader goes: Next when it is open, otherwise the part's title"""
    nb = pg.locator('#px-rail .step.current [data-nav="next"]:not([disabled])')
    if nb.count(): nb.first.click()
    else: pg.evaluate(f"()=>document.querySelector('#px-rail .steps>.step:nth-child({k}) .step-h').click()")


def fit_entry(b, lang, parts=(6,)):
    pg = page(b, 1440, 900, lang)
    for k in range(1, max(parts) + 1):
        if k > 1: fit_next(pg, k)
        if k not in parts:
            pg.wait_for_timeout(900); continue
        fs = []
        for _ in range(44):
            pg.wait_for_timeout(150); fs.append(pg.evaluate(FIT, 'px'))
        f = {'step': fs[-1]['step'], 'spare': min(x['spare'] for x in fs), 'scroll': max(x['scroll'] for x in fs), 'top': fs[-1]['top']}
        check(f'{lang} 1440 part {k}: entering it, the stage fits without scrolling inside{", 30 px to spare" if lang == "zh" else ""} (B4, R7.12; polish r5)',
              f['step'] == k and f['scroll'] <= 0 and f['spare'] >= (30 if lang == 'zh' else 0), f)
    pg.close()

with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ('en', 'zh'):
        fit_entry(b, lang)
    for (w, h) in [(1440, 900), (390, 844)]:
        for lang in ('en', 'zh'):
            run_guided(b, w, h, lang)
            run_modes(b, w, h, lang)
    run_guided(b, 1440, 900, 'en', dark=True)
    for (w, h, lang) in [(1440, 900, 'en'), (1440, 900, 'zh'), (390, 844, 'zh')]:
        run_rm(b, w, h, lang)
    for (w, h) in [(360, 740), (720, 900)]:
        for lang in ('en', 'zh'):
            pg = page(b, w, h, lang); go_part(pg, 4)
            dt = pg.evaluate(DOTS)
            check(f'{w} {lang} part 5: the 7 parts’ dots on one line (review r1 #10, R7.4)', dt['shown'] and dt['n'] == 7 and dt['one'], dt)
            check(f'{w} {lang} part 5: no overflow', overflow(pg) <= 0, overflow(pg))
            pg.close()
    for (w, h) in [(1280, 800), (900, 900), (720, 900)]:
        pg = page(b, w, h, 'en')
        for m in ('guided', 'sandbox', 'challenge'):
            mode(pg, m)
            check(f'{w} {m}: no overflow', overflow(pg) <= 0, overflow(pg))
        mode(pg, 'guided'); go_part(pg, 4)
        pg.evaluate("()=>document.querySelector('#px-stage [data-otpos=\"middle\"]').click()"); pg.evaluate("()=>document.querySelector('#px-rail [data-otpred=\"before\"]').click()"); sweep_done(pg)
        ov = pg.evaluate("()=>{const bs=[...document.querySelectorAll('#px-stage .px-pbtn')].map(b=>b.getBoundingClientRect()); return bs.every((r,i)=>i===0||r.left>=bs[i-1].right-0.5)}")
        check(f'{w} part 5: no overflow, the three positions side by side', overflow(pg) <= 0 and ov, overflow(pg))
        shot(pg, f'{w}_en_challenge')
        pg.close()
    b.close()
# The owner's review of 2026-10-09 (R6.37, R6.33): no row of “Show the math” scrolls sideways at the sizes this test visits, in every
# guided part, Scenarios and the challenge. On a box too narrow for the one-line table, the lab sets the name on its own line and breaks
# a long formula before an operator, the numbers still under the terms they replace.
MSCROLL = r"""r=>{const out=[]; for(const d of document.querySelectorAll(r+' details.math')){if(!d.offsetParent) continue;
  for(const e of d.querySelectorAll('*')){const cs=getComputedStyle(e); if(!/auto|scroll/.test(cs.overflowX)||!e.offsetParent) continue;
    const o=e.scrollWidth-e.clientWidth; if(o>1) out.push(o+' px: '+(e.textContent||'').trim().replace(/\s+/g,' ').slice(0,40));}}
  const pg=document.documentElement.scrollWidth-innerWidth; if(pg>1) out.push('the page scrolls sideways by '+pg+' px'); return out;}"""
with sync_playwright() as _p:
    _b = _p.chromium.launch()
    for (_w, _h, _lang) in [(1440, 900, 'en'), (1440, 900, 'zh'), (900, 900, 'en'), (390, 844, 'en'), (390, 844, 'zh')]:
        _pg = _b.new_page(viewport={'width': _w, 'height': _h}); _pg.goto(URL); _pg.wait_for_timeout(250)
        _pg.click(f'.langsw button[data-lang="{_lang}"]'); _pg.wait_for_timeout(60)
        _pg.evaluate("()=>{location.hash='prefix-caching'}"); _pg.wait_for_timeout(500)
        _n = _pg.evaluate("()=>new Set([...document.querySelectorAll('#view-dyn aside.rail [data-go]')].map(e=>e.dataset.go)).size")
        _bad = []
        for _mode, _k in [('guided', k) for k in range(_n)] + [('sandbox', None), ('challenge', None)]:
            if _k is None or _k == 0:
                _pg.evaluate("m=>{const b=[...document.querySelectorAll('#view-dyn .modes button[data-mode=\"'+m+'\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}", _mode); _pg.wait_for_timeout(300)
            if _k is not None:
                _pg.evaluate("k=>{const e=document.querySelector('#view-dyn aside.rail [data-go=\"'+k+'\"]'); if(e) e.click();}", _k)
            _pg.wait_for_timeout(700)
            _pg.evaluate("()=>document.querySelectorAll('#view-dyn details.math').forEach(d=>{d.open=true})"); _pg.wait_for_timeout(400)
            _bad += [f"{_mode}{'' if _k is None else ' part ' + str(_k + 1)}: {x}" for x in _pg.evaluate(MSCROLL, '#view-dyn')]
        check(f'{_w} {_lang}: no row of Show the math scrolls sideways, in any guided part, Scenarios or the challenge (the owner, 2026-10-09; R6.37)', not _bad, _bad[:6])
        _pg.close()
    _b.close()


print(len(rule_hits), 'writing-rule hits')
for hit in sorted(set(rule_hits))[:120]:
    check('writing rules: ' + hit, False)
check('no page errors', not errs, errs[:5])
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:60] if fails else 'none')
sys.exit(1 if fails else 0)
