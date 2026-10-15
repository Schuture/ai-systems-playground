"""Flow test for the quantization lab (CONTRACT §6), at 1440×900 and 390×844, in EN and ZH.

- no page errors, no horizontal overflow, no text wider than its box, a tooltip for every help cursor, no exact-value hovers;
- part 1: the step-time picture, Next enabled, no block transport; the main picture before the metric cards (R7.12);
- part 2: Next disabled with the kit's why line until the rounding of the clean block has played; it plays at 中 (R3.9) through its
  three phases; the rounding's example is a value of 1 or more, clear of the half-way line, on a backing of its own (N3); the
  phase-1 note says the weights stretch; Show me shows the end at once and the observation (N4); Back and Next again replays nothing
  (R3.20);
- part 3: Next waits for a prediction; before it the block shows no rounding and the metric cards no result (R3.8); after
  “Round this block” the rounding plays, its phase-1 note says the other 31 crowd near 0 (N5), 31 of the 31 small weights become 0
  and the outlier is red (R5.11);
- part 4: Next waits for blocks of 16 or NVFP4; a scheme whose picture was watched shows the end at once (INT4 per 128 draws the
  picture of blocks of 32: R3.20, N6), a new picture plays the 1.5-s run (R3.9), a watched scheme does not replay; 重放 / Replay plays
  from the first phase at the chosen speed; the chart of every scheme with the pick;
- part 5: Next waits for both 4-bit weights and FP8; part 6 waits for the 405B and shows memory first; part 7 links the labs it
  names (never “planned”) and opens the scenarios;
- Scenarios: six, an insight for each; the hash carries the setup and opens it again; the language switch keeps the setup;
- the challenge: no verdict before Submit; a setup that does not fit, one that fits but loses the pick, and the one answer (NVFP4
  weights with an FP8 KV cache) with the result chart;
- the dock at 390 once the stage scrolls away;
- the writing rules on every visible text and tooltip (no “toy” outside the fidelity label, Chinese spacing and punctuation, two
  colons in one sentence, no “one at a time” notes, no “in part N” leaning), and the colours: red only for the outlier that wipes out
  its block and for verdicts; the kit's header ids are not reused.
Screenshots: shots/quantization/.
Run: python3 tests/quantization_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys, time
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
(P / 'out' / 'quantization').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/quantization.html', '--only', 'quantization'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'quantization' / 'flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'quantization.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'quantization' / 'flow-wrapped.html')
SH = P / 'shots' / 'quantization'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, hash_='quantization'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__qz.state()")
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-').replace(/\\u2212/g,'-'):''}", sel)
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
NEXT = '#qz-rail .step.current [data-nav="next"]'
BACK = '#qz-rail .step.current [data-nav="back"]'
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#qz-rail [data-nav=\"next\"]')].filter(b=>b.getClientRects().length).map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#qz-rail [data-next-why]')].filter(p=>!p.hidden&&getComputedStyle(p).visibility!=='hidden'&&p.getClientRects().length).map(p=>p.textContent)")
speed = lambda pg: pg.evaluate("()=>{const b=document.querySelector('#qz-tp .seg-ctl[data-key=\"speed\"] button[aria-pressed=\"true\"]'); return b?b.dataset.v:null}")


def settle(pg, ms=2000):
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel, wait=250):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(wait); settle(pg)


# the animation state two frames after a click
CLICK_ANIM = """async (sel)=>{document.querySelector(sel).click(); await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))); return window.__qz.state().anim;}"""


def click_anim(pg, sel):
    a = pg.evaluate(CLICK_ANIM, sel); pg.wait_for_timeout(150); settle(pg); return a


def wait_anim(pg, timeout=40000):
    pg.wait_for_function("()=>!window.__qz.state().anim.on", timeout=timeout); pg.wait_for_timeout(150)


def wait_done(pg, timeout=40000):
    # a replay waiting for its picture (a phone) counts as not done
    pg.wait_for_function("()=>{const a=window.__qz.state().anim; return !a.on&&!a.pend}", timeout=timeout); pg.wait_for_timeout(150)


# the callout of the rounding phase: its text, and whether its backing holds the whole text
CALL = """()=>{const g=document.querySelector('#qz-svg .qz-callg'); if(!g) return null; const t=g.querySelector('text').getBBox(), r=g.querySelector('rect.qz-callbg');
  const b=r?{x:+r.getAttribute('x'),y:+r.getAttribute('y'),w:+r.getAttribute('width'),h:+r.getAttribute('height')}:null;
  return {s:g.textContent, back:!!b&&b.x<=t.x+0.5&&b.x+b.w>=t.x+t.width-0.5&&b.y<=t.y+0.5&&b.y+b.h>=t.y+t.height-0.5};}"""
NOTE = "i=>{const d=document.querySelector(`#qz-b-note .qz-nt[data-i=\"${i}\"]`); return d?d.textContent:''}"


def call_num(s):
    m = re.match(r'\s*([−-]?[0-9.]+)\s*→\s*([−-]?[0-9.]+)', (s or '').replace('−', '-'))
    return (float(m.group(1)), float(m.group(2))) if m else None


def mode(pg, m):
    click(pg, f'#qz-app .modes button[data-mode="{m}"]', 350)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def all_text(pg):
    return pg.evaluate("""()=>{const out=[]; for(const s of ['#qz-app','#dock']){const e=document.querySelector(s); if(e&&e.offsetParent!==null) out.push(e.innerText);}
      document.querySelectorAll('#qz-app [data-tipx], #dock [data-tipx]').forEach(e=>out.push(e.getAttribute('data-tipx'))); return out.join('\\n').replace(/\\u2212/g,'-').replace(/\\u2011/g,'-')}""")


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · (?![0-9])/,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],
   [/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy”'],[/一次只能选|不会叠加|第 ?[0-9一二三四五六七八] ?部分里|[Oo]ne at a time/,'one-at-a-time or in-part-N wording'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],[/规划中|[Pp]lanned|not built yet|还没上线/,'planned'],
   [/深色|浅色|浅灰|深灰|\\b(?:dark|light|pale)(?:er)? (?:bar|line|block|part|cell)/i,'colour named by lightness (R5.7)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, .fid-tag, .nl-last')||!e.getClientRects().length) continue; const t=n.textContent;
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
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx').replace(/<math[\\s\\S]*?<\\/math>/g,'M'); const m=t.match(/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>][\\u4e00-\\u9fff]/g); if(m) out.add(m.join('|')+' in tooltip: '+t.slice(0,80));});
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
# red (--c-oom) only on the outlier that wipes out its block, the challenge's memory over 70 GiB and the verdicts
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const red=toRGB(get('--c-oom')), amber=toRGB(get('--c-kv'));
 const ok={[red]:/qz-red|qz-redt|qz-sw-r|qz-over|status bad|badline|bad\\b|warn|wrong|ex-res|cmp/,[amber]:/qz-kv|qz-sw-kv/};
 const out=[]; for(const e of document.querySelectorAll('#qz-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [red,amber]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())&&!(p==='stroke'&&s.strokeWidth==='0px')) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#qz-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg,.qz-eqs')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.classList.contains('qz-skt')) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#qz-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
SVGIN = """()=>{const out=[]; for(const s of document.querySelectorAll('#qz-stage svg')){if(!s.getClientRects().length) continue; const b=s.getBoundingClientRect();
  for(const t of s.querySelectorAll('text')){if(!t.getClientRects().length||!t.textContent.trim()) continue; const r=t.getBoundingClientRect(); if(r.left<b.left-1||r.right>b.right+1||r.top<b.top-1||r.bottom>b.bottom+1) out.push('outside: '+t.textContent.trim().slice(0,30));}}
  const st=document.querySelector('#qz-stage'); if(st){const b=st.getBoundingClientRect(); for(const e of st.querySelectorAll('.qz-gls span, .qz-cl, .qz-skv, .qz-mv2')){if(!e.getClientRects().length) continue; const r=e.getBoundingClientRect(); if(r.left<b.left-1||r.right>b.right+1) out.push('outside the panel: '+e.textContent.trim().slice(0,30));}}
  return out.slice(0,8);}"""
EXACT_HOVER = """r=>{const root=document.querySelector(r), out=[]; if(!root) return out;
  for(const e of root.querySelectorAll('[data-tipx],[data-tip]')){if(e.closest('.mathbody')) continue; let t=e.getAttribute('data-tipx')||''; const k=e.getAttribute('data-tip');
    if(k&&window.ASP) t+=' '+(window.ASP.t(k)||''); if(/精确值是|Exactly/.test(t)) out.push(t.slice(0,100));}
  for(const e of root.querySelectorAll('.num-big')) if(!e.closest('.mathbody')&&(e.hasAttribute('data-tipx')||e.closest('.def'))) out.push('hover on '+e.textContent);
  return out;}"""
TEXWORDS = """()=>{const W=new Set(['frac','left','right','sqrt','text','mathrm','hat','sum','cdot','times','div','round','max','lfloor','rfloor','log']);
  const out=[]; for(const m of document.querySelectorAll('#qz-app math, #dock math')) for(const e of m.querySelectorAll('mi')){const t=e.textContent.trim(); if(W.has(t)&&t!=='max'&&t!=='log'&&t!=='round') out.push(t);}
  return out;}"""
rule_hits = []


def rules(pg, where, lang):
    hx = pg.evaluate(EXACT_HOVER, '#qz-app'); check(f'{where}: no exact-value hover outside Show the math', not hx, hx[:3])
    for sel in ('#qz-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')
    for h in pg.evaluate(OVER): rule_hits.append(f'{where}: text wider than its box: {h}')
    for h in pg.evaluate(HELP): rule_hits.append(f'{where}: help cursor without a tooltip: {h}')
    for h in pg.evaluate(SVGIN): rule_hits.append(f'{where}: chart label outside its picture: {h}')
    for h in pg.evaluate(TEXWORDS): rule_hits.append(f'{where}: a TeX word shown: {h}')
    o = overflow(pg)
    if o > 0: rule_hits.append(f'{where}: horizontal overflow {o}px')


def go_next(pg):
    click(pg, NEXT if pg.query_selector(NEXT) else '#qz-rail [data-nav="next"]', 300)


def top(pg, sel):
    return pg.evaluate("s=>{const e=document.querySelector(s); return e&&e.getClientRects().length?e.getBoundingClientRect().top:null}", sel)


with sync_playwright() as pw:
    b = pw.chromium.launch()
    for lang in LANGS:
        for (w, h) in SIZES:
            tag = f'{lang}-{w}'
            pg = page(b, w, h, lang)
            st = state(pg)
            check(f'{tag}: opens in guided part 1 with the initial setup', st['mode'] == 'guided' and st['step'] == 0 and st['key'] == 'bf16', st)
            check(f'{tag}: the kit header ids are not reused', pg.evaluate("()=>!document.querySelector('#qz-fid,#qz-pri,#qz-share')"))
            # ---- part 1
            check(f'{tag} part 1: Next enabled', (nexts(pg) and not any(nexts(pg))), nexts(pg))
            check(f'{tag} part 1: the step-time picture, no block transport', vis(pg, '#qz-steps') and not vis(pg, '#qz-blockv'))
            check(f'{tag} part 1: the main picture first, the metric cards below it (R7.12)', (top(pg, '#qz-graph') or 0) < (top(pg, '#qz-metrics') or 0))
            r1 = text(pg, '#qz-rail .step.current')
            check(f'{tag} part 1: the observation names 7.60 ms', '7.60' in r1, r1[-300:])
            # round 1 polish P1: part 1 changes the weights only, and every option says so; the prefill comes in part 5
            labs1 = pg.evaluate("()=>[...document.querySelectorAll('#qz-ic-p1s button')].map(b=>b.innerText)")
            check(f'{tag} part 1: three options, each naming weights and none activations (P1)', len(labs1) == 3 and all(Z(lang, 'weights', '权重') in t and Z(lang, 'activation', '激活') not in t for t in labs1) and Z(lang, 'weights only', '只量化权重') in text(pg, '#qz-ic-p1'), (labs1, text(pg, '#qz-ic-p1')))
            click(pg, '#qz-ic-p1s button[data-v="int8"]', 300)
            r1 = text(pg, '#qz-rail .step.current')
            check(f'{tag} part 1: 8-bit weights give 3.87 ms', '3.87' in r1 and '3.87' in text(pg, '#qz-steps'), r1[-300:])
            check(f'{tag} part 1: the chart shows the one-request decode step only (P2)', 'prefill' not in text(pg, '#qz-steps').lower() and Z(lang, '64 requests', '64 个请求') not in text(pg, '#qz-steps'), text(pg, '#qz-steps'))
            check(f'{tag} part 1: the small-layer cards wait for part 2 (S5)', '%' not in text(pg, '#qz-m-oerr') and '%' not in text(pg, '#qz-m-same'), text(pg, '#qz-metrics'))
            check(f'{tag} part 1: the stage title names step time only, no memory chart (S9)', Z(lang, 'memory', '显存') not in text(pg, '#qz-st-title') and Z(lang, 'Decode (1 request)', 'decode 一步（1 个请求）') in text(pg, '#qz-status'), (text(pg, '#qz-st-title'), text(pg, '#qz-status')))
            rules(pg, f'{tag} p1', lang); shot(pg, f'{tag}_p1')
            # ---- part 2; B1: a change, then Next at once: no badge carries into the new part (R6.6)
            b1 = pg.evaluate("""n=>{document.querySelector('#qz-ic-p1s button[data-v="int4-128"]').click(); const d=document.querySelector('#qz-m-d1 .d').textContent; document.querySelector(n).click(); return d;}""", NEXT)
            pg.wait_for_timeout(300)
            check(f'{tag} part 1 → 2: the change showed a badge, and Next at once clears every badge (B1)', bool(b1.strip()) and pg.evaluate("()=>[...document.querySelectorAll('#qz-metrics .d')].every(d=>!d.textContent.trim()&&!d.classList.contains('on'))"), (b1, text(pg, '#qz-metrics')))
            check(f'{tag} part 2: the badge names the whole layer’s output error (S3)', Z(lang, 'layer output error', '整层输出误差') in text(pg, '#qz-status'), text(pg, '#qz-status'))
            check(f'{tag} part 2: the third phase is called “What is lost” / “丢了什么” (S18)', Z(lang, 'what is lost', '丢了什么') in text(pg, '#qz-tp').lower(), text(pg, '#qz-tp'))
            check(f'{tag} part 2: the phases act out the card’s three steps: divide by s, round, multiply back (P6)', all(x in text(pg, '#qz-tp').lower() for x in Z(lang, ('divide by s', 'round', 'multiply back'), ('除以 s', '舍入', '乘回 s'))), text(pg, '#qz-tp'))
            # P36: on a phone a replay starts once its picture is on screen, which the move to the new part does
            pg.wait_for_function("()=>{const a=window.__qz.state().anim; return a.on||!a.pend}", timeout=4000)
            a = state(pg)['anim']
            check(f'{tag} part 2: the rounding of the clean block plays on arrival', a['on'] and a['key'] == 'clean|int4-32', a)
            check(f'{tag} part 2: Next disabled with a why line while it plays', (nexts(pg) and all(nexts(pg))) and len(why(pg)) >= 1, (nexts(pg), why(pg)))
            check(f'{tag} part 2: the mechanism replay runs at 中 / medium (R3.9)', speed(pg) == 'med' and state(pg)['speed'] == 'med', speed(pg))
            check(f'{tag} part 2: the transport has the three phases, play, Replay and the speeds', pg.evaluate("()=>document.querySelectorAll('#qz-tp .qz-ph').length") == 3 and vis(pg, '#qz-replay') and Z(lang, 'Replay', '重放') in text(pg, '#qz-replay'), text(pg, '#qz-tp'))
            # P9: the replay draws the grid (−7…7 while divided by s) and the band that rounds to 0
            pg.wait_for_function("()=>{const a=window.__qz.state().anim; return a.ph===2&&a.t>0.5}", timeout=30000)
            g2 = pg.evaluate("()=>({grid:document.querySelectorAll('#qz-svg .qz-grid').length, band:document.querySelectorAll('#qz-svg .qz-band').length, ticks:[...document.querySelectorAll('#qz-svg .qz-yl')].map(t=>t.textContent)})")
            check(f'{tag} part 2: while rounding, the grid lines of −7…7 and the band around 0 are drawn (P6, P9)', g2['grid'] >= 14 and g2['band'] >= 1 and '7' in g2['ticks'] and '−7' in g2['ticks'], g2)
            c2 = pg.evaluate(CALL); u2 = call_num(c2 and c2['s'])
            check(f'{tag} part 2: the rounding example is a value of 1 or more, clear of the half-way line, rounded to the nearest whole number (N3)', bool(u2) and abs(u2[0]) >= 1 and u2[1] == round(u2[0]) and abs(abs(u2[0]) % 1 - 0.5) >= 0.15, c2)
            check(f'{tag} part 2: the callout sits on a backing that holds all of it (N3)', bool(c2) and c2['back'], c2)
            n1 = pg.evaluate(NOTE, 1)
            check(f'{tag} part 2: the phase-1 note says the others stretch with the largest weight', Z(lang, 'stretch', '按比例拉开') in n1, n1)
            check(f'{tag} part 2: the why line names Show me as the way to the result at once (N4)', any(Z(lang, 'Show me', '演示给我看') in x for x in why(pg)), why(pg))
            click(pg, '#qz-rail .step.current [data-nav="showme"]', 300)
            a = state(pg)['anim']
            check(f'{tag} part 2: Show me shows the end of the replay at once (N4)', not a['on'] and a['ph'] == 3 and 'clean|int4-32' in state(pg)['watched'], a)
            check(f'{tag} part 2: Next enabled once it has played', (nexts(pg) and not any(nexts(pg))), nexts(pg))
            r2 = text(pg, '#qz-rail .step.current')
            check(f'{tag} part 2: the observation shows', '[data-obs]' and pg.evaluate("()=>{const o=document.querySelector('#qz-rail .step.current [data-obs]'); return !!o&&!o.hidden&&o.getClientRects().length>0}"), r2[-200:])
            rules(pg, f'{tag} p2', lang); shot(pg, f'{tag}_p2')
            plays = state(pg)['plays']
            click(pg, BACK, 300); a = click_anim(pg, NEXT)
            check(f'{tag} part 2: Back and Next again replays nothing (R3.20)', not a['on'] and state(pg)['plays'] == plays, (a, state(pg)['plays'], plays))
            # ---- part 3
            go_next(pg)
            check(f'{tag} part 3: Next waits for the prediction', (nexts(pg) and all(nexts(pg))) and len(why(pg)) >= 1, (nexts(pg), why(pg)))
            check(f'{tag} part 3: no rounding and no result before the prediction (R3.8)', pg.evaluate("()=>document.querySelectorAll('#qz-svg .qz-q, #qz-svg .qz-x').length") == 0 and '%' not in text(pg, '#qz-metrics') and not state(pg)['anim']['on'], text(pg, '#qz-metrics'))
            r3 = text(pg, '#qz-rail .step.current')
            check(f'{tag} part 3: the question gives the 50 times and the 4 bits (R3.21)', '50' in r3 and '7' in r3, r3[:300])
            shot(pg, f'{tag}_p3_q')
            click(pg, '#qz-rail [data-pred="zero"]', 200)
            check(f'{tag} part 3: still waits until the block is rounded', (nexts(pg) and all(nexts(pg))))
            a = click_anim(pg, '#qz-rail [data-act="pred-go"]')
            check(f'{tag} part 3: the outlier block plays', a['on'] and a['key'] == 'outlier|int4-32', a)
            n1 = pg.evaluate(NOTE, 1)
            check(f'{tag} part 3: the phase-1 note says the other 31 crowd near 0, not that they stretch (N5)', Z(lang, 'crowded near 0', '挤在 0 附近') in n1 and '31' in n1 and Z(lang, 'stretch', '拉开') not in n1, n1)
            pg.wait_for_function("()=>{const a=window.__qz.state().anim; return a.ph===2&&a.t>0.5}", timeout=30000)
            c3 = pg.evaluate(CALL); u3 = call_num(c3 and c3['s'])
            check(f'{tag} part 3: the rounding example becomes 0 well inside the band (N3)', bool(u3) and u3[1] == 0 and abs(u3[0]) <= 0.47 and bool(c3['back']), c3)
            wait_anim(pg)
            note = text(pg, '#qz-b-note')
            check(f'{tag} part 3: 31 of the 31 small weights become 0', '31' in note and (' 0' in note or '0。' in note), note)
            red = pg.evaluate("()=>{const e=document.querySelector('#qz-svg .qz-orig.qz-red'); if(!e) return null; const c=getComputedStyle(e).stroke, d=document.createElement('i'); d.style.color=getComputedStyle(document.documentElement).getPropertyValue('--c-oom'); document.body.appendChild(d); const r=getComputedStyle(d).color; d.remove(); return c===r}")
            check(f'{tag} part 3: the outlier that wipes out its block is red (R5.11)', red is True, red)
            check(f'{tag} part 3: Next enabled, the verdict on the prediction', (nexts(pg) and not any(nexts(pg))) and pg.evaluate("()=>{const o=document.querySelector('#qz-rail .step.current [data-obs]'); return !!o&&!o.hidden}"))
            rules(pg, f'{tag} p3', lang); shot(pg, f'{tag}_p3')
            # ---- part 4
            go_next(pg)
            check(f'{tag} part 4: Next waits for blocks of 16 or NVFP4', (nexts(pg) and all(nexts(pg))) and len(why(pg)) >= 1, (nexts(pg), why(pg)))
            check(f'{tag} part 4: the picture shows the watched block at once (R3.20)', not state(pg)['anim']['on'], state(pg)['anim'])
            check(f'{tag} part 4: the chart of every scheme with the pick', pg.evaluate("()=>document.querySelectorAll('#qz-chart .qz-crow').length") == 9 and pg.evaluate("()=>document.querySelectorAll('#qz-chart .qz-crow.me').length") == 1)
            a = click_anim(pg, '#qz-ic-p4 [data-lever="int4-128"]')
            check(f'{tag} part 4: INT4 per 128 draws the watched picture of blocks of 32, so it shows the end at once (R3.20, N6)', not a['on'] and not a['pend'] and a['ph'] == 3 and a['key'] == 'outlier|int4-128', a)
            t0 = time.time(); a = click_anim(pg, '#qz-ic-p4 [data-lever="mx"]')
            check(f'{tag} part 4: a new picture plays (N6)', (a['on'] or a['pend'] == 'outlier|mx') and a['key'] == 'outlier|mx', a)
            wait_done(pg, 8000); dt = time.time() - t0
            check(f'{tag} part 4: it plays the 1.5-s run, not the slow mechanism replay (R3.9, N6)', state(pg)['anim']['quick'] and dt < 5, (state(pg)['anim'], dt))
            a = click_anim(pg, '#qz-ic-p4 [data-lever="int4-32"]')
            check(f'{tag} part 4: a watched scheme does not replay (R3.20)', not a['on'] and not a['pend'], a)
            plays = state(pg)['plays']
            a = click_anim(pg, '#qz-replay')
            check(f'{tag} part 4: Replay plays from the first phase at the chosen speed', a['on'] and a['ph'] == 1 and not a['quick'] and state(pg)['plays'] == plays + 1, a)
            wait_anim(pg)
            a = click_anim(pg, '#qz-ic-p4 [data-lever="int4-16"]'); wait_done(pg, 8000)
            check(f'{tag} part 4: Next enabled after blocks of 16', (nexts(pg) and not any(nexts(pg))), nexts(pg))
            note = text(pg, '#qz-b-note')
            check(f'{tag} part 4: blocks of 16 save the half without the outlier (18 become 0)', '18' in note, note)
            rules(pg, f'{tag} p4', lang); shot(pg, f'{tag}_p4')
            # ---- part 5
            go_next(pg)
            check(f'{tag} part 5: Next waits for both choices', (nexts(pg) and all(nexts(pg))), nexts(pg))
            click(pg, '#qz-ic-p5 [data-lever="w4"]', 250)
            check(f'{tag} part 5: still waits after one', (nexts(pg) and all(nexts(pg))))
            click(pg, '#qz-ic-p5 [data-lever="fp8"]', 250)
            check(f'{tag} part 5: Next enabled after both', (nexts(pg) and not any(nexts(pg))), nexts(pg))
            r5 = text(pg, '#qz-steps')
            check(f'{tag} part 5: the prefill with FP8 weights and activations is faster than BF16', '60.0' in r5, r5[:400])
            rules(pg, f'{tag} p5', lang); shot(pg, f'{tag}_p5')
            # ---- part 6
            go_next(pg)
            check(f'{tag} part 6: Next waits for the 405B', (nexts(pg) and all(nexts(pg))), nexts(pg))
            check(f'{tag} part 6: memory first', (top(pg, '#qz-mem') or 0) < (top(pg, '#qz-metrics') or 1e9) and not vis(pg, '#qz-steps'))
            check(f'{tag} part 6: no 8B metric cards and no step badge while models are compared (O11, S9)', not vis(pg, '#qz-metrics') and not vis(pg, '#qz-status') and Z(lang, 'how many', '要几张') in text(pg, '#qz-st-title'), (text(pg, '#qz-st-title'), text(pg, '#qz-status')))
            click(pg, '#qz-ic-p6s button[data-v="405b"]', 300)
            check(f'{tag} part 6: Next enabled after the 405B', (nexts(pg) and not any(nexts(pg))), nexts(pg))
            r6 = text(pg, '#qz-mem')
            check(f'{tag} part 6: the 405B rows name the H100s', Z(lang, 'GPUs', '张卡') in r6, r6[:300])
            rules(pg, f'{tag} p6', lang); shot(pg, f'{tag}_p6')
            # ---- part 7
            go_next(pg)
            r7 = text(pg, '#qz-rail .step.current')
            check(f'{tag} part 7: 37.0 GiB, the scenarios button', '37.0' in r7 and pg.query_selector('#qz-rail .step.current [data-mode-go="sandbox"], #qz-rail .step.current [data-go="sandbox"], #qz-rail .step.current .toscen, #qz-rail .step.current button') is not None, r7[-300:])
            # the number-formats link lived in the 前置知识 popover, which went with the header buttons (owner, 2026-10-11 19:28 UTC)
            check(f'{tag} part 7: no lab called planned', 'planned' not in all_text(pg).lower() and '规划中' not in all_text(pg))
            rules(pg, f'{tag} p7', lang); shot(pg, f'{tag}_p7', full=(w == 1440))
            # ---- scenarios
            mode(pg, 'sandbox')
            ps = pg.query_selector_all('#qz-rail [data-preset]')
            check(f'{tag} scenarios: six', len(ps) == 6, len(ps))
            for i in range(6):
                click(pg, f'#qz-rail [data-preset="{i}"]', 300)
                ins = text(pg, '#qz-insight')
                check(f'{tag} scenario {i}: an insight', len(ins) > 40 and pg.evaluate(f"()=>document.querySelector('#qz-rail [data-preset=\"{i}\"]').getAttribute('aria-pressed')") == 'true', ins[:200])
                if state(pg)['anim']['on']: wait_anim(pg)
                check(f'{tag} scenario {i}: a scenario the lab applies shows no badge (R6.6)', pg.evaluate("()=>[...document.querySelectorAll('#qz-metrics .d')].every(d=>!d.textContent.trim()&&!d.classList.contains('on'))"), text(pg, '#qz-metrics'))
                if i == 3: rules(pg, f'{tag} scen{i}', lang); shot(pg, f'{tag}_scen_nvfp4')
            rules(pg, f'{tag} scen', lang)
            hsh = pg.evaluate("()=>location.hash")
            check(f'{tag} scenarios: the hash carries the setup', hsh == '#quantization~fp8_32_fp8_405b_serve', hsh)
            other = 'zh' if lang == 'en' else 'en'
            s0 = state(pg); click(pg, f'.langsw button[data-lang="{other}"]', 300); s1 = state(pg)
            check(f'{tag}: the language switch keeps the setup', (s0['key'], s0['kv'], s0['model'], s0['mode']) == (s1['key'], s1['kv'], s1['model'], s1['mode']), (s0, s1))
            click(pg, f'.langsw button[data-lang="{lang}"]', 300)
            # ---- challenge; B1: a change in Scenarios, then the Challenge tab at once
            pg.evaluate("""()=>{const b=document.querySelector('#qz-controls .seg-ctl[data-key="kv"] button[data-v="bf16"]'); if(b) b.click();}""")
            mode(pg, 'challenge')
            check(f'{tag} challenge: no badge carried over from Scenarios (B1)', pg.evaluate("()=>[...document.querySelectorAll('#qz-metrics .d')].every(d=>!d.textContent.trim()&&!d.classList.contains('on'))"), text(pg, '#qz-metrics'))
            check(f'{tag} challenge: no verdict before Submit', text(pg, '#qz-ch-res').strip() == '' and Z(lang, 'Not submitted', '未提交') in text(pg, '#qz-status'), text(pg, '#qz-ch-res'))
            check(f'{tag} challenge: the metric cards hold back the pick until Submit', pg.evaluate("()=>['#qz-m-same','#qz-m-oerr'].map(s=>document.querySelector(s+' .qz-mval').textContent.trim())") == ['—', '—'], text(pg, '#qz-metrics'))
            click(pg, '#qz-rail [data-act="submit"]', 300)
            res = text(pg, '#qz-ch-res')
            check(f'{tag} challenge: BF16 does not fit', Z(lang, 'Not met', '没达标') in res and '171.4' in res, res[:300])
            click(pg, '#qz-ic-chw [data-lever="int4-128"]', 200); click(pg, '#qz-ic-chkv button[data-v="fp8"]', 200)
            check(f'{tag} challenge: a new pick needs a new Submit', text(pg, '#qz-ch-res').strip() == '' or Z(lang, 'Not met', '没达标') not in text(pg, '#qz-ch-res') or not state(pg)['chSubmitted'], state(pg)['chSubmitted'])
            click(pg, '#qz-rail [data-act="submit"]', 300)
            res = text(pg, '#qz-ch-res')
            check(f'{tag} challenge: INT4 per 128 fits but loses the pick', Z(lang, 'Not met', '没达标') in res and '80.4' in res, res[:300])
            click(pg, '#qz-ic-chw [data-lever="nv"]', 200); click(pg, '#qz-rail [data-act="submit"]', 400)
            res = text(pg, '#qz-ch-res')
            check(f'{tag} challenge: NVFP4 with an FP8 KV cache meets the goal', Z(lang, 'Goal met', '达标') in res and '93.3' in res and '57.0' in res, res[:400])
            check(f'{tag} challenge: after Submit the cards show the pick on the other layer', '93.3' in text(pg, '#qz-m-same'), text(pg, '#qz-m-same'))
            check(f'{tag} challenge: the result chart with every setup and the pick', pg.evaluate("()=>document.querySelectorAll('#qz-ch-res .cmp-row, #qz-ch-res .cmp .row, #qz-ch-res [data-row]').length") >= 14 or pg.query_selector('#qz-ch-res .cmp') is not None)
            rules(pg, f'{tag} challenge', lang); shot(pg, f'{tag}_challenge', full=(w == 1440))
            if w < 500:
                pg.evaluate("()=>window.scrollTo(0, document.querySelector('#qz-ch-res').getBoundingClientRect().top+scrollY+300)"); pg.wait_for_timeout(500)
                dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return d&&!d.hidden&&d.getBoundingClientRect().height>0?d.innerText:''}")
                check(f'{tag}: the dock shows once the stage scrolls away', bool(dk), dk)
                if dk: rules(pg, f'{tag} dock', lang); shot(pg, f'{tag}_dock')
            pg.close()
            # ---- the hash round trip
            pg = page(b, w, h, lang, hash_='quantization~nv_32_fp8_70b_block'); st = state(pg)
            check(f'{tag}: the hash round trip opens the same setup in scenarios', st['mode'] == 'sandbox' and (st['key'], st['kv'], st['model'], st['view']) == ('nv', 'fp8', '70b', 'block'), st)
            if st['anim']['on']: wait_anim(pg)
            rules(pg, f'{tag} hash', lang); shot(pg, f'{tag}_sandbox_block')
            pg.close()
    b.close()

for hh in sorted(set(rule_hits)): print('RULE', hh)
check('writing rules, colours, overflow and tooltips', not rule_hits, len(rule_hits))
check('no page errors', not errs, errs[:6])
_SRC = (P / 'labs' / 'quantization.js').read_text()
_G = re.findall(r'.{0,30}(?:一次只能选|不会叠加|第 ?[0-9一二三四五六七八] ?部分里|do not stack|[Oo]ne at a time|规划中|还没上线|[Pp]lanned lab).{0,30}', _SRC)
check('no “one at a time”, “in part N” or “planned” wording in the lab source (EN and ZH)', not _G, _G[:3])
_T = re.findall(r'.{0,30}(?:玩具|\btoy\b).{0,30}', _SRC)
check('“toy” / 玩具 never in the lab’s own copy (the kit’s fidelity label only)', not _T, _T[:3])
print(f'{n - len(fails)} / {n} passed')
sys.exit(1 if fails else 0)
