"""Flow test for the product design capstone lab: every guided part, the scenarios and the challenge at 1440x900 and 390x844 in EN and
ZH; the gates and their reasons, the prediction (no leak before the pick, the verdict after), the levers, the red failure moments
(R5.11), the three challenge tiers and the brief comparison chart, the capability layer, the language switch mid-challenge, the hash
round trip, the dock on phones, no horizontal overflow, the copy rules shared with the other labs (no option-group disclaimers, Chinese
spacing and punctuation, no formula as plain text), and the replay rules of R3.20 for the message player (no replay when nothing
changed, none when returning to a watched message, a replay when something changed, the 重放 / Replay control, default speed 中,
pause, step, scrub). Screenshots go to shots/product-design/.

Run: python3 tests/product-design_flow.py [en|zh] [1440|390]   (filters; without them, everything)
"""
import json, pathlib, subprocess, sys, time
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/product-design.html', '--only', 'product-design'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'pdz').mkdir(parents=True, exist_ok=True)
(P / 'out' / 'pdz' / 'wrapped-flow.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'product-design.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'pdz' / 'wrapped-flow.html')
SH = P / 'shots' / 'product-design'; SH.mkdir(parents=True, exist_ok=True)
ARGS = sys.argv[1:]
LANGS = [a for a in ARGS if a in ('en', 'zh')] or ['en', 'zh']
WIDTHS = [int(a) for a in ARGS if a in ('1440', '390')] or [1440, 390]
errs, notes, fails, rule_hits = [], [], [], []


def check(name, ok, info=''):
    notes.append(('OK   ' if ok else 'FAIL ') + name + (f'  [{info}]' if info else ''))
    if not ok: fails.append(name)


def page(b, w, lang, hash_='product-design', reduced=True):
    pg = open_drawers(b.new_page(viewport={'width': w, 'height': 900 if w > 500 else 844}, reduced_motion='reduce' if reduced else 'no-preference'))
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(900)
    return pg


Z = lambda lang, en, zh: zh if lang == 'zh' else en
st = lambda pg: pg.evaluate("()=>window.__pdz.state()")
cp = lambda pg, s: pg.evaluate("s=>window.__pdz.compute(s)", s)
overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
count = lambda pg, sel: pg.evaluate("s=>[...document.querySelectorAll(s)].filter(e=>e.getClientRects().length).length", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\s+/g,' ').trim():''}", sel)
next_off = lambda pg: pg.evaluate("()=>{const b=document.querySelector('#pdz-rail .step.current [data-nav=\"next\"]'); return !b||b.disabled||b.getAttribute('aria-disabled')==='true'}")
click = lambda pg, sel, ms=250: (pg.evaluate("s=>document.querySelector(s).click()", sel), pg.wait_for_timeout(ms))
lever = lambda pg, k, v, ms=300: click(pg, f'#pdz-rail [data-lkey="{k}"][data-lever="{v}"]', ms)
nxt = lambda pg, ms=400: click(pg, '#pdz-rail .step.current [data-nav="next"]', ms)
shot = lambda pg, n: pg.screenshot(path=str(SH / f'{n}.png'))
sT = lambda v: f'{v:.2f} s'
usd = lambda v: f'${round(v):,}'
red = lambda pg: count(pg, '#pdz-wf .pdz-bar.red')

# the copy rules the other labs' flow tests check, on this lab's root. Company names are allowed here: the lab names what OpenAI's and
# Anthropic's APIs, Meta's Llama Guard and others publish (brief R6.31); it never names a company as the source of a problem.
RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · (?![0-9])/,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],
   [/in this toy\\b|在这个玩具里|\\btoy\\b|玩具/i,'“toy”'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],
   [/一次只能选|一次选一个|不会叠加|do not stack|don't stack|one at a time/i,'option-group disclaimer (G1)'],[/第 ?[0-9一二三四五六七八] ?部分里/,'“第 N 部分里”'],
   [/深色|浅色|浅灰|深灰|\\b(?:dark|light|pale)(?:er)? (?:bar|line|block|part|cell|square)/i,'colour named by lightness (R5.7)'],
   [/÷|\\bceil\\(|\\bmax\\(|λ ?=|\\bN − 1\\b/,'formula as plain text (H1)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, math, #tip, .fid-tag')||!e.getClientRects().length) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx').replace(/<math[\\s\\S]*?<\\/math>/g,'M'); for(const [re,name] of bad) if(re.test(t)) out.push(name+' (tooltip): '+t.slice(0,90));});
 for(const e of root.querySelectorAll('sub,sup')) if(!e.closest('[data-endnotes],pre,code,math')) out.push('sub/sup outside math: '+e.parentElement.textContent.trim().slice(0,40));
 return out;}"""
SPACING = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), bad=/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>%][\\u4e00-\\u9fff]|[）】][ ][\\u4e00-\\u9fff]/;
 const shown=e=>e&&!e.closest('[data-endnotes],pre,code,svg,.mathbody,#tip,math')&&e.getClientRects().length;
 const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){if(!shown(n.parentElement)) continue; const t=n.textContent.replace(/\\s+/g,' '); const m=t.match(new RegExp(bad.source,'g')); if(m) out.add(m.join('|')+' in: '+t.trim().slice(0,80));}
 root.querySelectorAll('a,.def,b,.t,em,code').forEach(e=>{if(!shown(e)||getComputedStyle(e).display!=='inline') return;
   const t=e.textContent; if(!t) return; const prev=e.previousSibling, next=e.nextSibling;
   const pc=prev&&prev.nodeType===3?prev.textContent.slice(-1):'', nc=next&&next.nodeType===3?next.textContent.slice(0,1):'';
   const pair=(a,b)=>a&&b&&bad.test(a+b);
   if(pair(pc,t[0])) out.add(pc+t[0]+' before: '+t.slice(0,40)); if(pair(t[t.length-1],nc)) out.add(t[t.length-1]+nc+' after: '+t.slice(-40));});
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx').replace(/<math[\\s\\S]*?<\\/math>/g,'M').replace(/<[^>]+>/g,''); const m=t.match(/[\\u4e00-\\u9fff][A-Za-z0-9]|[A-Za-z0-9][\\u4e00-\\u9fff]/g); if(m) out.add(m.join('|')+' in tooltip: '+t.slice(0,80));});
 return [...out];}"""
PUNCT = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), bad=/[\\u4e00-\\u9fff][:;,!?]|[:;,!?] ?[\\u4e00-\\u9fff]/g;
 const shown=e=>e&&!e.closest('[data-endnotes],pre,code,.mathbody,#tip,math')&&e.getClientRects().length;
 const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){if(!shown(n.parentElement)) continue; const t=n.textContent.replace(/\\s+/g,' '); const m=t.match(bad); if(m) out.add(m.join('|')+' in: '+t.trim().slice(0,80));}
 return [...out];}"""
COLONS = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), texts=[];
 root.querySelectorAll('p,li,button,label,div,span,h1,h2,h3,summary').forEach(e=>{if(e.closest('[data-endnotes],pre,code,svg,.mathbody')||!e.getClientRects().length) return; if(e.querySelector('p,li,div,ul,ol,h1,h2,h3,table')) return;
   if(![...e.childNodes].some(c=>c.nodeType===3&&c.textContent.trim())) return; texts.push(e.textContent);});
 for(const t of texts){const sents=t.replace(/\\s+/g,' ').split(/(?<=[。？！])/);
   for(const x of sents){const k=(x.match(/：/g)||[]).length; if(k>=2) out.add(x.trim().slice(0,160));}}
 return [...out];}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#pdz-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg,.pdz-sr,.pdz-pbar')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
SVGIN = """()=>{const out=[]; for(const s of document.querySelectorAll('#pdz-stage svg')){if(!s.getClientRects().length) continue; const b=s.getBoundingClientRect();
  for(const t of s.querySelectorAll('text')){if(!t.getClientRects().length||!t.textContent.trim()) continue; const r=t.getBoundingClientRect();
    if(r.left<b.left-2||r.right>b.right+2||r.top<b.top-2||r.bottom>b.bottom+2) out.push('outside: '+t.textContent.trim().slice(0,30));}}
  return out;}"""


def rules(pg, where, lang):
    for sel in ('#pdz-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
    for h in pg.evaluate(OVER): rule_hits.append(f'{where}: text wider than its box: {h}')
    for h in pg.evaluate(SVGIN): rule_hits.append(f'{where}: drawing text: {h}')
    o = overflow(pg)
    if o > 0: rule_hits.append(f'{where}: horizontal overflow {o}px')


BASE = dict(dau=1e6, msgs=10, peak=3, stream=1, cache=0, ret=0, share=0, filt=8, price=3)


def walk(b, lang, w):
    tag = f'{lang} {w}'
    sTl = lambda v: sT(v) if lang == 'en' else f'{v:.2f} 秒'
    pg = page(b, w, lang)
    s = st(pg); r0 = cp(pg, {})
    check(f'{tag} opens on part 1 with the plain design (1M users, streaming, no cache, all large, chunks of 8)', s['mode'] == 'guided' and s['step'] == 0 and all(s[k] == v for k, v in BASE.items()), json.dumps({k: s[k] for k in ('mode', 'step', 'cache', 'share', 'filt')}))
    rail = text(pg, '#pdz-rail')
    check(f'{tag} part 1: the message has been watched (reduced motion) and the first word is at {sTl(r0["vis"])}, within 1.8 s', sTl(r0['vis']) in rail and s['done'] and len(s['watched']) == 1 and not next_off(pg), rail[:160])
    check(f'{tag} part 1: the waterfall has one row per component, a dashed 1.8 s budget line and no red', count(pg, '#pdz-wf .pdz-row') >= 8 and count(pg, '#pdz-wf .pdz-goal') == 1 and red(pg) == 0, count(pg, '#pdz-wf .pdz-row'))
    check(f'{tag} part 1: the player has play, back, forward, 重放 / Replay, speeds 慢 中 快 with 中 pressed, a progress slider',
          count(pg, '#pdz-play') == 1 and count(pg, '#pdz-player [data-act="back"]') == 1 and count(pg, '#pdz-player [data-act="fwd"]') == 1
          and text(pg, '#pdz-player [data-act="replay"]') == Z(lang, 'Replay', '重放') and text(pg, '#pdz-player .seg-ctl[data-key="spd"]').replace(' ', '') == Z(lang, 'SlowMediumFast', '慢中快')
          and text(pg, '#pdz-player .seg-ctl[data-key="spd"] [aria-pressed="true"]') == Z(lang, 'Medium', '中') and count(pg, '#pdz-pos') == 1)
    check(f'{tag} part 1: four metric cards under the picture', count(pg, '#pdz-metrics .metric') == 4 and pg.evaluate("()=>document.querySelector('#pdz-metrics').getBoundingClientRect().top>document.querySelector('#pdz-trace').getBoundingClientRect().top"))
    rules(pg, f'{tag} part 1', lang); shot(pg, f'{lang}_{w}_1_mission')
    nxt(pg)
    # part 2: streaming
    check(f'{tag} part 2 gated until both options are tried; the reason is shown', next_off(pg) and Z(lang, 'Try both options', '两个选项都试过') in text(pg, '#pdz-rail'))
    off = cp(pg, {'stream': 0})
    lever(pg, 'stream', '0')
    check(f'{tag} part 2 (R5.11): sending when done takes {sTl(off["vis"])}, red in the waterfall and in the text', sTl(off['vis']) in text(pg, '#pdz-rail .obs') and red(pg) >= 1 and count(pg, '#pdz-rail .obs .pdz-bad') >= 1 and count(pg, '#pdz-metrics .pdz-mbad') == 1, text(pg, '#pdz-rail .obs')[:160])
    lever(pg, 'stream', '1')
    check(f'{tag} part 2: both tried, next opens; the GPUs are the same either way', not next_off(pg) and cp(pg, {'stream': 0})['total'] == cp(pg, {'stream': 1})['total'] and red(pg) == 0)
    rules(pg, f'{tag} part 2', lang); shot(pg, f'{lang}_{w}_2_stream')
    nxt(pg)
    # part 3: context and caching
    a, c = cp(pg, {}), cp(pg, {'cache': 1})
    check(f'{tag} part 3 gated; the prompt bar and the fleet are shown', next_off(pg) and count(pg, '#pdz-prompt .pdz-pbar') == 1 and count(pg, '#pdz-fleet .pdz-sq') > 0)
    lever(pg, 'cache', '1'); lever(pg, 'cache', '0'); lever(pg, 'cache', '1')
    obs = text(pg, '#pdz-rail .obs')
    check(f'{tag} part 3: caching cuts the GPUs {a["total"]} → {c["total"]}, {usd(a["costDay"])} → {usd(c["costDay"])}; the cached part is drawn as a hit', not next_off(pg) and str(a['total']) in obs and str(c['total']) in obs and usd(c['costDay']) in obs and count(pg, '#pdz-prompt .pdz-ps.sys.hit') == 1, obs[:200])
    rules(pg, f'{tag} part 3', lang); shot(pg, f'{lang}_{w}_3_context')
    nxt(pg)
    # part 4: the prediction
    r7 = cp(pg, {'cache': 1, 'share': 0.7}); pct = f"{r7['costDay'] / c['costDay'] * 100:.0f}%"
    rail = text(pg, '#pdz-rail'); stg = text(pg, '#pdz-stage')
    check(f'{tag} part 4: gated until a pick; no answer shown before it ({pct}, {r7["total"]} GPUs nowhere)', next_off(pg) and pct not in rail and pct not in stg and str(r7['total']) not in stg and usd(r7['costDay']) not in stg + rail
          and Z(lang, 'Pick an answer to continue', '选一个答案才能继续') in rail, rail[:120])
    check(f'{tag} part 4 (R3.21): the question gives the share, the price ratio and the base of 100%', all(x in rail for x in ('70%', '30%', '100%')) and Z(lang, 'a tenth', '十分之一') in rail)
    click(pg, '#pdz-rail [data-pred="p37"]')
    check(f'{tag} part 4: a pick opens next and is only recorded (the verdict waits for part 5)', not next_off(pg) and pct not in text(pg, '#pdz-rail'))
    rules(pg, f'{tag} part 4', lang); shot(pg, f'{lang}_{w}_4_predict')
    nxt(pg)
    # part 5: routing
    check(f'{tag} part 5: gated until 70% and 90% are tried', next_off(pg) and Z(lang, 'Try 70% and 90%', '试过 70% 和 90%') in text(pg, '#pdz-rail'))
    lever(pg, 'share', '0.7'); lv = text(pg, '#pdz-rail [data-lkey="share"][data-lever="0.7"]')
    check(f'{tag} part 5: 70% costs {usd(r7["costDay"])} a day (on the option); the observation waits for 90%', usd(r7['costDay']) in lv and count(pg, '#pdz-rail .obs') == 0, lv[:200])
    r9 = cp(pg, {'cache': 1, 'share': 0.9})
    lever(pg, 'share', '0.9'); obs = text(pg, '#pdz-rail .obs')
    check(f'{tag} part 5 (R5.11): 90% is cheaper ({usd(r9["costDay"])}) but answers badly, in red; next opens', usd(r9['costDay']) in obs and count(pg, '#pdz-rail .obs .pdz-bad') >= 1 and not next_off(pg), obs[:200])
    check(f'{tag} part 5: the verdict says the pick was right: 70% leaves {pct} of the cost', pct in obs and '✓' in obs, obs[-200:])
    check(f'{tag} part 5: the fleet shows both pools and the checks', count(pg, '#pdz-fleet .pdz-sq.l') > 0 and count(pg, '#pdz-fleet .pdz-sq.s') > 0 and count(pg, '#pdz-fleet .pdz-sq.f') > 0)
    rules(pg, f'{tag} part 5', lang); shot(pg, f'{lang}_{w}_5_route')
    nxt(pg)
    # part 6: safety
    check(f'{tag} part 6: gated; the design so far is cache + 70%', next_off(pg) and st(pg)['cache'] == 1 and st(pg)['share'] == 0.7)
    lever(pg, 'filt', 'after'); obs = text(pg, '#pdz-rail .obs')
    check(f'{tag} part 6 (R5.11): checking after sending shows words before they are checked, in red', count(pg, '#pdz-rail .obs .pdz-bad') >= 1 and Z(lang, 'already seen', '已经看到') in obs, obs[:200])
    f16 = cp(pg, {'cache': 1, 'share': 0.7, 'filt': 16})
    lever(pg, 'filt', '16'); obs = text(pg, '#pdz-rail .obs')
    check(f'{tag} part 6 (R5.11): chunks of 16 put the first word at {sTl(f16["vis"])}, over 1.8 s: the hold is red in the waterfall; next opens', sTl(f16['vis']) in obs and red(pg) >= 1 and count(pg, '#pdz-metrics .pdz-mbad') == 1 and not next_off(pg), obs[:200])
    shot(pg, f'{lang}_{w}_6_safety_red')
    f4 = cp(pg, {'cache': 1, 'share': 0.7, 'filt': 4})
    lever(pg, 'filt', '4'); obs = text(pg, '#pdz-rail .obs')
    check(f'{tag} part 6: chunks of 4 meet the budget ({sTl(f4["vis"])}) with {f4["fG"]} GPUs of checks', sTl(f4['vis']) in obs and str(f4['fG']) in obs and red(pg) == 0, obs[:200])
    rules(pg, f'{tag} part 6', lang)
    nxt(pg)
    # part 7: sizing
    check(f'{tag} part 7: gated; the sizing sheet is shown', next_off(pg) and count(pg, '#pdz-sheet table') == 1)
    hi = cp(pg, {'cache': 1, 'share': 0.7, 'dau': 1e7})
    lever(pg, 'dau', '10000000'); obs = text(pg, '#pdz-rail .obs')
    check(f'{tag} part 7: 10M users need {hi["total"]} GPUs, {usd(hi["costDay"])} a day; next opens', str(f"{hi['total']:,}") in text(pg, '#pdz-stage') + obs and usd(hi['costDay']) in obs and not next_off(pg), obs[:200])
    rules(pg, f'{tag} part 7', lang); shot(pg, f'{lang}_{w}_7_size')
    nxt(pg)
    rail = text(pg, '#pdz-rail')
    check(f'{tag} part 8: misconceptions, production systems and the way to the scenarios', count(pg, '#pdz-rail [data-mode-go="sandbox"]') >= 1 and 'vLLM' in rail and 'Llama Guard' in rail and '✗' in rail)
    pg.evaluate("()=>{const m=document.querySelector('#pdz-math'); if(m) m.open=true}"); pg.wait_for_timeout(150)
    check(f'{tag} Show the math: named formulas, typeset', count(pg, '#pdz-mathbody .pdz-mb') >= 4 and count(pg, '#pdz-mathbody math') >= 8, count(pg, '#pdz-mathbody math'))
    rules(pg, f'{tag} part 8', lang); shot(pg, f'{lang}_{w}_8_remember')
    # scenarios
    click(pg, '#pdz-rail [data-mode-go="sandbox"]', 500)
    presets = pg.evaluate("()=>window.__pdz.consts.PRESETS")
    for i, p_ in enumerate(presets):
        click(pg, f'#pdz-rail [data-preset="{i}"]', 450)
        r = cp(pg, p_); ins = text(pg, '#pdz-insight')
        check(f'{tag} scenario {i + 1}: {r["total"]} GPUs, {usd(r["costDay"])}, first word {sTl(r["vis"])}', usd(r['costDay']) in ins and sTl(r['vis']) in ins and usd(r['costDay']) in text(pg, '#pdz-metrics'), ins[:140])
        if i == 3: check(f'{tag} scenario 4 (no streaming): the first word is red', count(pg, '#pdz-insight .pdz-bad') == 1 and red(pg) >= 1)
        rules(pg, f'{tag} scenario {i + 1}', lang)
        if i in (2, 3): shot(pg, f'{lang}_{w}_sc{i + 1}')
    s = st(pg); h = pg.evaluate("()=>location.hash")
    check(f'{tag} the hash carries the setting', h.endswith('4_10_3_1_0_0_0_8_30'), h)
    click(pg, '#pdz-controls .seg-ctl[data-key="ret"] button[data-v="1"]', 500)
    h = pg.evaluate("()=>location.hash"); s = st(pg)
    check(f'{tag} settings: retrieval on changes the hash and adds the retrieval row', s['ret'] == 1 and h.endswith('4_10_3_1_0_1_0_8_30') and count(pg, '#pdz-wf .pdz-row[data-k="ret"]') == 1, h)
    pg2 = page(b, w, lang, hash_=h.lstrip('#')); s2 = st(pg2)
    check(f'{tag} the hash round trip restores the setting', s2['mode'] == 'sandbox' and all(s2[k] == s[k] for k in BASE), json.dumps({k: s2[k] for k in ('mode', 'ret', 'cache')}))
    pg2.close()
    rules(pg, f'{tag} settings', lang)
    # challenge
    click(pg, '#pdz-app .modes [data-mode="challenge"]', 500); s = st(pg)
    c0 = cp(pg, dict(dau=2e6, msgs=12, peak=3, stream=1, cache=0, share=0, filt=16))
    check(f'{tag} challenge opens on streaming, no cache, all large, chunks of 16: {sTl(c0["vis"])} / {usd(c0["costDay"])}', s['mode'] == 'challenge' and (s['stream'], s['cache'], s['share'], s['filt']) == (1, 0, 0, 16) and sTl(c0['vis']) in text(pg, '#pdz-ch-g') and usd(c0['costDay']) in text(pg, '#pdz-ch-g'), text(pg, '#pdz-ch-g'))
    check(f'{tag} challenge: no feedback before submitting', text(pg, '#pdz-ch-res') == '')
    fl = text(pg, '#pdz-ic-filt')
    check(f'{tag} challenge: the output check says what its options mean in the card itself (R13)', all(x in fl for x in Z(lang, ('every how many tokens', 'No check', 'After all is sent', 'Check all, then send', 'Every 32 tokens', 'Every 4 tokens'), ('每多少个 token 检查一次', '不查', '全部发完再查', '整段先查再发', '每 32 个 token', '每 4 个 token'))), fl)
    click(pg, '#pdz-rail [data-act="submit"]', 400); r = text(pg, '#pdz-ch-res')
    check(f'{tag} challenge tier 1 (not met): red line, a hint, Show the answer, no chart', count(pg, '#pdz-ch-res .badline') == 1 and Z(lang, 'Hint', '提示') in r and count(pg, '#pdz-ch-res [data-act="reveal"]') == 1 and count(pg, '#pdz-ch-res .cmp, #pdz-ch-res svg') == 0, r[:160])
    for k, v in (('cache', '1'), ('filt', '4'), ('share', '0.5')): click(pg, f'#pdz-ic-{k} button[data-v="{v}"]', 350)
    r = text(pg, '#pdz-ch-res')
    check(f'{tag} challenge tier 2 (met, not best): 50% gives $30,816 and names the cheaper design at $21,888', count(pg, '#pdz-ch-res .okline') == 1 and '$30,816' in r and '$21,888' in r, r[:200])
    click(pg, '#pdz-ic-share button[data-v="0.7"]', 400); r = text(pg, '#pdz-ch-res')
    check(f'{tag} challenge tier 3 (best): $21,888, 1.29 s, the brief chart lists the passing designs only, the why', count(pg, '#pdz-ch-res .okline') == 1 and '$21,888' in r and sTl(cp(pg, dict(dau=2e6, msgs=12, peak=3, cache=1, share=0.7, filt=4))['vis']) in r and count(pg, '#pdz-ch-res svg, #pdz-ch-res .cmp') >= 1 and Z(lang, 'is the cheapest design that meets every goal', '是最便宜的达标设计') in r, r[:200])
    rules(pg, f'{tag} challenge', lang); shot(pg, f'{lang}_{w}_ch_best')
    other = 'en' if lang == 'zh' else 'zh'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400); s = st(pg)
    check(f'{tag} the language switch keeps the challenge, the choice and the result', s['mode'] == 'challenge' and s['share'] == 0.7 and s['filt'] == 4 and s['chSubmitted'] and '$21,888' in text(pg, '#pdz-ch-res'))
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    check(f'{tag} the capability layer follows the challenge', count(pg, '#pdz-rail [data-exits]') == 1)
    rules(pg, f'{tag} exits', lang)
    if w == 390:
        pg.evaluate("()=>window.scrollTo(0, document.querySelector('#pdz-stage').getBoundingClientRect().top+scrollY+300)"); pg.wait_for_timeout(400)
        d = pg.evaluate("()=>{const d=document.getElementById('dock'); return d?{shown:d.getClientRects().length>0&&!d.hidden, t:d.innerText.replace(/\\s+/g,' ').trim().slice(0,120)}:null}")
        check(f'{tag} the dock shows the result on phones', bool(d) and d['shown'] and d['t'] != '', json.dumps(d, ensure_ascii=False))
    o = overflow(pg); check(f'{tag} no horizontal overflow', o <= 0, o)
    pg.close()


def wait_end(pg, limit=20):
    t0 = time.time()
    while time.time() - t0 < limit:
        if not st(pg)['playing']: return time.time() - t0
        pg.wait_for_timeout(100)
    return time.time() - t0


def replay_rules(b, lang):
    tag = f'{lang} 1440 R3.20'
    pg = page(b, 1440, lang, reduced=False)
    s = st(pg)
    check(f'{tag}: part 1 sends the message by itself (not yet watched) at 中', s['playing'] and s['spd'] == 'med' and s['plays'] == 1 and next_off(pg), json.dumps({k: s[k] for k in ('playing', 'spd', 'plays', 't')}))
    pg.wait_for_timeout(1500); t1 = st(pg)['t']
    shot(pg, f'{lang}_1440_r_playing')
    click(pg, '#pdz-play', 150); s = st(pg); ta = s['t']
    check(f'{tag}: the play button pauses', not s['playing'])
    click(pg, '#pdz-player [data-act="fwd"]', 120); tb = st(pg)['t']
    click(pg, '#pdz-player [data-act="back"]', 120); tc = st(pg)['t']
    check(f'{tag}: forward and back move one component', tb > ta and tc < tb, f'{ta:.3f} {tb:.3f} {tc:.3f}')
    pg.evaluate("()=>{const r=document.querySelector('#pdz-pos'); r.value=r.max; r.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(200); s = st(pg)
    check(f'{tag}: scrubbing to the end counts the message as watched and opens next', s['done'] and len(s['watched']) == 1 and not next_off(pg), json.dumps({k: s[k] for k in ('done', 'watched')}))
    n = s['plays']
    nxt(pg, 300); s = st(pg)
    check(f'{tag}: part 2 opens on the same, watched message without a replay', s['plays'] == n and not s['playing'] and s['done'])
    click(pg, '#pdz-rail .step.current [data-nav="back"]', 300); s = st(pg)
    check(f'{tag}: no replay when returning to the watched message in part 1', s['plays'] == n and not s['playing'] and s['done'], json.dumps({k: s[k] for k in ('plays', 'playing', 'done')}))
    click(pg, '#pdz-player .seg-ctl[data-key="spd"] button[data-v="fast"]', 100)
    t0 = time.time(); click(pg, '#pdz-player [data-act="replay"]', 80); s = st(pg)
    check(f'{tag}: the 重放 / Replay control replays', s['plays'] == n + 1 and s['playing'])
    dt = wait_end(pg)
    vis = cp(pg, {})['vis']
    check(f'{tag}: 快 plays twice as fast as 中 (the whole axis in 3.5 s against 7 s)', 1.6 <= (vis / dt) / (t1 / 1.5) <= 2.4 and 1.2 <= dt <= 3.6, f'{dt:.1f} s, {t1:.3f}')
    click(pg, '#pdz-player .seg-ctl[data-key="spd"] button[data-v="med"]', 100)
    nxt(pg, 300); n = st(pg)['plays']
    lever(pg, 'stream', '0', 200); s = st(pg)
    check(f'{tag}: a replay when something changed (sending when done is a new message), shown quickly', s['plays'] == n + 1 and s['playing'])
    pg.wait_for_timeout(1700); s = st(pg)
    check(f'{tag}: the changed message reaches its end within 1.5 s', not s['playing'] and s['done'])
    lever(pg, 'stream', '1', 200); s = st(pg)
    check(f'{tag}: streaming again shows its end at once (watched)', s['plays'] == n + 1 and not s['playing'] and s['done'])
    lever(pg, 'stream', '1', 200); s = st(pg)
    check(f'{tag}: no replay when nothing changed', s['plays'] == n + 1 and not s['playing'])
    lever(pg, 'stream', '0', 200); s = st(pg)
    check(f'{tag}: the watched no-streaming message shows its end at once', s['plays'] == n + 1 and not s['playing'])
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in LANGS:
        for w in WIDTHS:
            walk(b, lang, w)
        if 1440 in WIDTHS: replay_rules(b, lang)
    b.close()
check('copy rules (RULES, spacing, punctuation, colons, boxes, drawing text)', not rule_hits, '; '.join(rule_hits[:10]))
check('no page errors', not errs, '; '.join(errs[:4]))
print('\n'.join(notes))
print(f'\n{len(notes) - len(fails)}/{len(notes)} passed')
sys.exit(1 if fails else 0)
