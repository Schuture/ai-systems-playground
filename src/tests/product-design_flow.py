"""Flow test for the product design capstone lab: every guided part, the scenarios and the challenge at 1440x900 and 390x844 in EN and
ZH; the gates and their reasons, the prediction (no leak before the pick, the verdict after), the levers, the red failure moments
(R5.11), the three challenge tiers and the brief comparison chart, the capability layer, the language switch mid-challenge, the hash
round trip, the dock on phones, no horizontal overflow, the copy rules shared with the other labs (no option-group disclaimers, Chinese
spacing and punctuation, no formula as plain text), and the replay rules of R3.20 for the message player (no replay when nothing
changed, none when returning to a watched message, a replay when something changed, the 重放 / Replay control, default speed 中,
pause, step, scrub). Polish round 1: the GPU squares use one scale everywhere, so fewer GPUs never draw more squares (P1); the hard
messages on the small model are drawn red and named on the badge (P10); a replay after a choice in the guided parts plays at the
reader's speed, each component before the first word holds at least the dwell, and a longer wait plays longer (P3, P5); paused, the
title, badge and button all say paused (P6); the picture is drawn at its own width (P2); the slider follows the time axis (P22); at
1440 the metric cards are in view inside the stage (P15). After the check of round 1: part 8 gives the path and the hold in ms (N1);
part 7 opens with its method (N2); part 6 draws every option on one time axis, so a larger N moves the first word right, and the whole
answer checked first says where its first word is (N3); Next keeps the sticky stage whole under the top bar at 1440 (N4); on a phone the
picture does not move while the title changes (N5); the wording of N6–N12 and P27; the dock never sits on the stage's title row or
the new card (P15: at 1440 it stays away while the sticky stage is on screen; on a phone it waits until the title row is gone).
Screenshots go to shots/product-design/.

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
# the drawn area of the GPU squares (P1): with one scale, it follows the number of GPUs
sq_area = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#pdz-fleet .pdz-sqs .pdz-sq')].reduce((a,e)=>{const r=e.getBoundingClientRect(); return a+r.width*r.height;},0)")
plan = lambda pg, q=False: pg.evaluate("q=>window.__pdz.plan(q)", q)
# P15: the metric cards fit in the stage's visible box without scrolling inside it (the stage is capped at 1440)
cards_in_view = lambda pg: pg.evaluate("()=>{const s=document.querySelector('#pdz-stage'), m=document.querySelector('#pdz-metrics'); if(!m||!m.getClientRects().length) return true; return m.getBoundingClientRect().bottom-s.getBoundingClientRect().top+s.scrollTop<=s.clientHeight+1;}")
red_after_goal = lambda pg: pg.evaluate("()=>{const g=document.querySelector('#pdz-wf .pdz-goal'); if(!g) return false; const gx=g.getBoundingClientRect().left; return [...document.querySelectorAll('#pdz-wf .pdz-bar.red')].every(e=>e.getBoundingClientRect().left>=gx-1.5);}")
svg_scale = lambda pg: pg.evaluate("()=>{const s=document.querySelector('#pdz-wf svg'); return s?s.getBoundingClientRect().width/s.viewBox.baseVal.width:1}")
msT = lambda v: (f'{v * 1000:,.0f}' if v * 1000 >= 100 else f'{v * 1000:.1f}') + ' ms'
vis_x = lambda pg: pg.evaluate("()=>{const l=document.querySelector('#pdz-wf .pdz-vis'); return l?+l.getAttribute('x1'):null}")
legend = lambda pg: pg.evaluate("()=>document.querySelector('#pdz-leg1').textContent")
# N7, N9: the decode ticks are drawn one per token, one per chunk of N (too dense for one per token) or not at all (too dense even for that);
# the legend says which
def tick_mode(pg):
    n, full = count(pg, '#pdz-wf .pdz-tok'), count(pg, '#pdz-wf .pdz-tok.full')
    return 'none' if n == 0 else 'chunk' if n == full else 'tok'
LEG_TICK = {'tok': ('one tick per token, the first at the end of prefill', '一道细线一个 token，第 1 个在 prefill 末端'),
            'chunk': ('one tick per chunk of 8 tokens', '一道线是一段（8 个 token）'), 'none': ('decode: the rest of the answer', 'decode：写回答的其余部分')}
leg_ok = lambda pg, lang: Z(lang, *LEG_TICK[tick_mode(pg)]) in legend(pg)
# N4: on a desktop the sticky stage stays whole under the top bar, and the new part's card starts on screen
STAGE_POS = """()=>{const st=document.querySelector('#pdz-stage'), c=document.querySelector('#pdz-rail .step.current'), bar=document.querySelector('.topbar');
  return {stage:Math.round(st.getBoundingClientRect().top), card:Math.round(c.getBoundingClientRect().top), bar:Math.round(bar.getBoundingClientRect().bottom), vh:innerHeight, y:Math.round(scrollY)};}"""
stage_ok = lambda g: g['stage'] >= g['bar'] - 1 and g['bar'] <= g['card'] <= g['vh'] - 150
# P15: the dock never covers the stage's title row or the top of the current card
DOCK = """()=>{const dk=document.getElementById('dock'), hd=document.querySelector('#pdz-stage .stage-head'), c=document.querySelector('#pdz-rail .step.current')||document.querySelector('#pdz-rail');
  const on=!!dk&&!dk.hidden&&dk.getClientRects().length>0, d=on?dk.getBoundingClientRect():null, h=hd.getBoundingClientRect(), ct=c.getBoundingClientRect().top;
  return {dock:on, dockB:d?Math.round(d.bottom):0, head:[Math.round(h.top),Math.round(h.bottom)], card:Math.round(ct), covers:on&&h.top<d.bottom&&h.bottom>d.top, cardUnder:on&&ct<d.bottom&&ct>d.top-1};}"""
dock_ok = lambda g: not g['covers'] and not g['cardUnder']

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
    if pg.viewport_size['width'] > 500 and not cards_in_view(pg): rule_hits.append(f'{where}: the metric cards are below the stage’s visible box (P15)')
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
    check(f'{tag} part 1 (P2): the waterfall is drawn at the width it is shown, not stretched', abs(svg_scale(pg) - 1) < 0.01, svg_scale(pg))
    if w == 390:
        tops = pg.evaluate("()=>[...document.querySelectorAll('#pdz-metrics .metric')].map(e=>Math.round(e.getBoundingClientRect().top))")
        check(f'{tag} part 1 (P35): the metric cards sit two to a row on a phone', len(tops) == 4 and tops[0] == tops[1] and tops[2] == tops[3] and tops[2] > tops[0], tops)
    check(f'{tag} part 1 (N6): the first requirement names the time from Send to the first word, then its name', Z(lang, 'the time from pressing Send to the first word on screen (the time to the first visible word', '从按下发送到第一个字出现不超过 1.8 秒（这段时间叫“首字可见时间”') in text(pg, '#pdz-rail .step.current'), text(pg, '#pdz-rail .step.current')[:200])
    check(f'{tag} part 1 (N7): the legend says how the decode ticks are drawn; one per token at 1440, the first at the end of prefill', leg_ok(pg, lang) and (w != 1440 or tick_mode(pg) == 'tok'), tick_mode(pg) + ' ' + legend(pg)[:200])
    rules(pg, f'{tag} part 1', lang); shot(pg, f'{lang}_{w}_1_mission')
    nxt(pg)
    # part 2: streaming
    check(f'{tag} part 2 gated until both options are tried; the reason is shown', next_off(pg) and Z(lang, 'Try both options', '两个选项都试过') in text(pg, '#pdz-rail'))
    check(f'{tag} part 2 (P31): the picture already shows the result, and the play button says play, not send', text(pg, '#pdz-play') == Z(lang, '▶ Play', '▶ 播放'), text(pg, '#pdz-play'))
    off = cp(pg, {'stream': 0})
    lever(pg, 'stream', '0')
    check(f'{tag} part 2 (R5.11): sending when done takes {sTl(off["vis"])}, red in the waterfall and in the text', sTl(off['vis']) in text(pg, '#pdz-rail .obs') and red(pg) >= 1 and count(pg, '#pdz-rail .obs .pdz-bad') >= 1 and count(pg, '#pdz-metrics .pdz-mbad') == 1, text(pg, '#pdz-rail .obs')[:160])
    lever(pg, 'stream', '1')
    check(f'{tag} part 2: both tried, next opens; the GPUs are the same either way', not next_off(pg) and cp(pg, {'stream': 0})['total'] == cp(pg, {'stream': 1})['total'] and red(pg) == 0)
    obs = text(pg, '#pdz-rail .obs')
    check(f'{tag} part 2 (N8): the same GPUs because the output is checked the same way; part 2 does not name N yet', Z(lang, 'the output is checked the same way', '输出检查的做法也一样') in obs and Z(lang, 'every 8 tokens', '每 8 个 token') not in obs, obs[-200:])
    check(f'{tag} part 2 (N9): the legend says how the decode ticks are drawn; on the 20 s axis at 1440 one tick is a chunk of 8 tokens', leg_ok(pg, lang) and (w != 1440 or tick_mode(pg) == 'chunk'), tick_mode(pg) + ' ' + legend(pg)[:200])
    rules(pg, f'{tag} part 2', lang); shot(pg, f'{lang}_{w}_2_stream')
    nxt(pg)
    # part 3: context and caching
    a, c = cp(pg, {}), cp(pg, {'cache': 1})
    check(f'{tag} part 3 gated; the prompt bar and the fleet are shown', next_off(pg) and count(pg, '#pdz-prompt .pdz-pbar') == 1 and count(pg, '#pdz-fleet .pdz-sq') > 0)
    lever(pg, 'cache', '1'); a_c = sq_area(pg); lever(pg, 'cache', '0'); a_n = sq_area(pg); lever(pg, 'cache', '1')
    key = text(pg, '#pdz-fleet .pdz-fkey')
    check(f'{tag} part 3 (P1): one square is 4 GPUs (the key says so); with the cache the squares cover less ({c["total"]} against {a["total"]} GPUs, the same ratio)',
          Z(lang, '= 4 GPUs', '= 4 张卡') in key and 0 < a_c < a_n and abs(a_c / a_n - c['total'] / a['total']) < 0.02, f'{a_c:.0f} {a_n:.0f} {key}')
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
    check(f'{tag} part 5 (P10): at 70% no square is red and the badge is green', count(pg, '#pdz-fleet .pdz-sq.hard') == 0 and pg.evaluate("()=>document.querySelector('#pdz-status').classList.contains('ok')"), text(pg, '#pdz-status'))
    r9 = cp(pg, {'cache': 1, 'share': 0.9})
    lever(pg, 'share', '0.9'); obs = text(pg, '#pdz-rail .obs')
    check(f'{tag} part 5 (R5.11): 90% is cheaper ({usd(r9["costDay"])}) but answers badly, in red; next opens', usd(r9['costDay']) in obs and count(pg, '#pdz-rail .obs .pdz-bad') >= 1 and not next_off(pg), obs[:200])
    check(f'{tag} part 5 (P10): at 90% the small model’s squares for hard messages are red, and the badge says so', count(pg, '#pdz-fleet .pdz-sq.hard') > 0 and text(pg, '#pdz-status') == Z(lang, 'Hard messages on the small model', '难题给了小模型')
          and pg.evaluate("()=>document.querySelector('#pdz-status').classList.contains('bad')"), text(pg, '#pdz-status'))
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
    check(f'{tag} part 6 (P19): only what lies past the budget line is red; the hold keeps its hatching before it', red_after_goal(pg) and count(pg, '#pdz-wf .pdz-row[data-k="hold"] .pdz-bar.hold') == 1)
    check(f'{tag} part 6 (P12): the tokens, the full chunks and every check of the answer are drawn', count(pg, '#pdz-wf .pdz-tok.full') >= 2 and count(pg, '#pdz-wf .pdz-row[data-k="hold"] .pdz-bar.chk') >= 2, (count(pg, '#pdz-wf .pdz-tok.full'), count(pg, '#pdz-wf .pdz-bar.chk')))
    pl = plan(pg); dw = pg.evaluate("()=>window.__pdz.dwell.med"); pre = [x for x in pl['P'] if x['w1'] <= pl['fw'] + 1e-9]
    check(f'{tag} part 6 (P3, P5): a replay at 中 gives every component before the first word at least {dw} s, then holds the first word',
          len(pre) >= 8 and all(x['w1'] - x['w0'] >= dw - 1e-9 for x in pre) and any(x['t0'] == x['t1'] and x['w0'] >= pl['fw'] - 1e-9 and x['w1'] > x['w0'] for x in pl['P']) and pl['total'] >= 6, json.dumps({k: pl[k] for k in ('X', 'total', 'fw')}))
    shot(pg, f'{lang}_{w}_6_safety_red')
    f4 = cp(pg, {'cache': 1, 'share': 0.7, 'filt': 4})
    lever(pg, 'filt', '4'); obs = text(pg, '#pdz-rail .obs')
    check(f'{tag} part 6: chunks of 4 meet the budget ({sTl(f4["vis"])}) with {f4["fG"]} GPUs of checks', sTl(f4['vis']) in obs and str(f4['fG']) in obs and red(pg) == 0, obs[:200])
    # N3: one time axis for every way of checking, so a larger N moves the first word right; the whole answer checked first is past it
    xs, vx, back = {}, {}, ''
    for v in ('after', 'whole', '16', '8', '4'):
        lever(pg, 'filt', v); xs[v] = st(pg)['X']; vx[v] = vis_x(pg)
        if v == 'whole': back = pg.evaluate("()=>{const g=document.querySelector('#pdz-wf .pdz-row[data-k=\"back\"]'); return g?g.textContent:''}"); shot(pg, f'{lang}_{w}_6_whole')
    fw = cp(pg, {'cache': 1, 'share': 0.7, 'filt': 'whole'})['vis']
    check(f'{tag} part 6 (N3): every option on one {xs["8"]} s axis; the first word moves right from N = 4 to 8 to 16',
          len(set(xs.values())) == 1 and None not in (vx['4'], vx['8'], vx['16']) and vx['4'] < vx['8'] < vx['16'], json.dumps({'X': xs, 'vis': vx}))
    check(f'{tag} part 6 (N3): the whole answer checked first has its first word past the axis, said in the last row of the path', vx['whole'] is None and Z(lang, 'first word', '首字') in back and Z(lang, f'{sT(fw)} →', f'{fw:.2f} 秒 →') in back, back)
    rules(pg, f'{tag} part 6', lang)
    nxt(pg)
    if w == 1440:
        g = pg.evaluate(STAGE_POS)
        check(f'{tag} part 7 (N4): after Next the stage stays whole under the top bar and the card starts on screen', stage_ok(g), json.dumps(g))
    # part 7: sizing
    check(f'{tag} part 7: gated; the sizing sheet is shown; the card says the design built so far is kept, right above the options (P14, N2)', next_off(pg) and count(pg, '#pdz-sheet table') == 1
          and Z(lang, 'all with the design built so far', '都用前面搭好的设计') in pg.evaluate("()=>{const l=document.querySelector('#pdz-rail .step.current .levers'), p=l&&l.previousElementSibling; return p?p.textContent:''}")
          and text(pg, '#pdz-rail .step.current .step-b p').startswith(Z(lang, 'Work out the messages per second', '先算最忙那个小时每秒有多少条消息')))
    hi = cp(pg, {'cache': 1, 'share': 0.7, 'dau': 1e7}); lo = cp(pg, {'cache': 1, 'share': 0.7, 'dau': 1e5}); mid = cp(pg, {'cache': 1, 'share': 0.7})
    a6 = sq_area(pg); lever(pg, 'dau', '100000'); a5 = sq_area(pg)
    lever(pg, 'dau', '10000000'); a7 = sq_area(pg); obs = text(pg, '#pdz-rail .obs')
    check(f'{tag} part 7 (P1): {lo["total"]} < {mid["total"]} < {hi["total"]} GPUs draw growing squares, in the same ratios', 0 < a5 < a6 < a7 and abs(a5 / a6 - lo['total'] / mid['total']) < 0.02 and abs(a7 / a6 - hi['total'] / mid['total']) < 0.02 * hi['total'] / mid['total'], f'{a5:.0f} {a6:.0f} {a7:.0f}')
    check(f'{tag} part 7: 10M users need {hi["total"]} GPUs, {usd(hi["costDay"])} a day; next opens', str(f"{hi['total']:,}") in text(pg, '#pdz-stage') + obs and usd(hi['costDay']) in obs and not next_off(pg), obs[:200])
    sh = text(pg, '#pdz-sheet')
    check(f'{tag} part 7 (N10, P27): the sheet says each check takes 1 ms of a GPU, and why the storage row is not in the cost', Z(lang, '1 ms of one GPU per check', '每次占卡 1 ms') in sh and Z(lang, 'costs far less than the GPUs', '比卡便宜得多') in sh, sh[-260:])
    rules(pg, f'{tag} part 7', lang); shot(pg, f'{lang}_{w}_7_size')
    if w == 1440:   # N4: the case the check found, Next from far down the page
        pg.evaluate("()=>window.scrollTo(0, 1000)"); pg.wait_for_timeout(200)
    nxt(pg)
    if w == 1440:
        g = pg.evaluate(STAGE_POS)
        check(f'{tag} part 8 (N4): after Next the stage stays whole under the top bar and the card starts on screen', stage_ok(g), json.dumps(g))
        g = pg.evaluate(DOCK)
        check(f'{tag} part 8 (P15): no dock over the sticky stage, so its title row and the card are clear', not g['dock'] and dock_ok(g), json.dumps(g))
    else:   # P15 on a phone: the dock waits until the stage's title row has left the screen
        pg.evaluate("()=>{const s=document.querySelector('#pdz-stage'); window.scrollTo(0, s.getBoundingClientRect().top+scrollY-120)}"); pg.wait_for_timeout(300); g1 = pg.evaluate(DOCK)
        pg.evaluate("()=>{const c=document.querySelector('#pdz-rail .step.current'); window.scrollTo(0, c.getBoundingClientRect().top+scrollY-200)}"); pg.wait_for_timeout(300); g2 = pg.evaluate(DOCK)
        check(f'{tag} part 8 (P15): no dock while the stage title row is on screen; the dock shows the result at the card', not g1['dock'] and g2['dock'] and dock_ok(g2), json.dumps([g1, g2]))
    rail = text(pg, '#pdz-rail')
    fin = cp(pg, {'cache': 1, 'share': 0.7})
    check(f'{tag} part 8 (P16): the design meets all three requirements ({sTl(fin["vis"])}, {usd(fin["costDay"])}), each point answers a misconception (✗), the way to the scenarios',
          count(pg, '#pdz-rail [data-mode-go="sandbox"]') >= 1 and Z(lang, 'meets all three requirements', '三条要求都满足了') in rail and sTl(fin['vis']) in rail and usd(fin['costDay']) in rail and rail.count('✗') == 3, rail[:200])
    path = fin['vis'] - fin['tModel'] - fin['hold'] - fin['gen']
    check(f'{tag} part 8 (N1): besides the first token, the path takes {msT(path)} and the hold {msT(fin["hold"])}, which a faster first token does not shorten',
          msT(path) in rail and msT(fin['hold']) in rail and Z(lang, 'a faster first token shortens neither', '首 token 再快也省不掉') in rail, rail[:400])
    check(f'{tag} part 8 (N11): the routing point answers “the more to the small model, the better”', Z(lang, '“The more goes to the small model, the better”', '“交给小模型的越多越好”') in rail)
    if w == 1440:
        cw = pg.evaluate("()=>{const p=document.querySelector('#pdz-code'); return p?[p.scrollWidth,p.clientWidth]:null}")
        check(f'{tag} (N12): every line of the minimal code fits its box at 1440', bool(cw) and cw[0] <= cw[1] + 1, cw)
    con = text(pg, '#pdz-app .connect')
    check(f'{tag} the production systems (vLLM, Llama Guard, Prompt Guard 2, rate limits) are named once, under Take it further (P7)', all(x in con for x in ('vLLM', 'Llama Guard', 'Prompt Guard 2', 'RPM')) and 'vLLM' not in rail, con[:120])
    pg.evaluate("()=>{const m=document.querySelector('#pdz-math'); if(m) m.open=true}"); pg.wait_for_timeout(150)
    check(f'{tag} Show the math: named formulas, typeset', count(pg, '#pdz-mathbody .pdz-mb') >= 4 and count(pg, '#pdz-mathbody math') >= 8, count(pg, '#pdz-mathbody math'))
    rules(pg, f'{tag} part 8', lang); shot(pg, f'{lang}_{w}_8_remember')
    # scenarios
    click(pg, '#pdz-rail [data-mode-go="sandbox"]', 500)
    presets = pg.evaluate("()=>window.__pdz.consts.PRESETS"); docks = []
    for i, p_ in enumerate(presets):
        click(pg, f'#pdz-rail [data-preset="{i}"]', 450)
        if w == 1440: docks.append(pg.evaluate(DOCK))
        r = cp(pg, p_); ins = text(pg, '#pdz-insight')
        check(f'{tag} scenario {i + 1}: {r["total"]} GPUs, {usd(r["costDay"])}, first word {sTl(r["vis"])}', usd(r['costDay']) in ins and sTl(r['vis']) in ins and usd(r['costDay']) in text(pg, '#pdz-metrics'), ins[:140])
        if i == 3: check(f'{tag} scenario 4 (no streaming): the first word is red, only past the budget line', count(pg, '#pdz-insight .pdz-bad') == 1 and red(pg) >= 1 and red_after_goal(pg))
        if i == 2: check(f'{tag} scenario 3 (P29): its line gives the messages per user and the share to the small model', Z(lang, '5 messages per user a day, half to the small model', '每人每天 5 条消息，一半交给小模型') in text(pg, '#pdz-rail [data-preset="2"]'), text(pg, '#pdz-rail [data-preset="2"]'))
        if i >= 1:
            mt = text(pg, '#pdz-metrics')
            check(f'{tag} scenario {i + 1} (P30): the change is measured against the scenario before, named', '“' in mt and Z(lang, 'vs last', '上次') not in mt, mt[:160])
        rules(pg, f'{tag} scenario {i + 1}', lang)
        if i in (2, 3): shot(pg, f'{lang}_{w}_sc{i + 1}')
    if w == 1440: check(f'{tag} scenarios (P15): no dock over the sticky stage in any of the {len(docks)}', all(not g['dock'] and dock_ok(g) for g in docks), json.dumps(docks[:2]))
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
    check(f'{tag} challenge (P32): the readout is one row per goal, and the bar says how many are met', count(pg, '#pdz-ch-g .pdz-chm') == 4 and text(pg, '#pdz-ch-n') == Z(lang, '2 of 4 goals met', '达标 2 / 4 项'), text(pg, '#pdz-ch-g') + ' | ' + text(pg, '#pdz-ch-n'))
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
    check(f'{tag}: part 1 sends the message by itself (not yet watched) at 中, at the reader’s speed', s['playing'] and s['spd'] == 'med' and s['plays'] == 1 and next_off(pg) and not s['quick'], json.dumps({k: s[k] for k in ('playing', 'spd', 'plays', 't', 'quick')}))
    check(f'{tag} (P2): while it plays, the waterfall is drawn at the width it is shown', abs(svg_scale(pg) - 1) < 0.01, svg_scale(pg))
    pm = plan(pg); dw = pg.evaluate("()=>window.__pdz.dwell.med"); pre = [x for x in pm['P'] if x['w1'] <= pm['fw'] + 1e-9]
    check(f'{tag} (P5): at 中 each of the {len(pre)} components before the first word plays for at least {dw} s; the first word at {pm["fw"]:.1f} s, the whole message {pm["total"]:.1f} s',
          len(pre) >= 8 and all(x['w1'] - x['w0'] >= dw - 1e-9 for x in pre) and 5 <= pm['total'] <= 15, json.dumps({k: pm[k] for k in ('X', 'total', 'fw')}))
    w_play = pg.evaluate("()=>document.querySelector('#pdz-play').getBoundingClientRect().width")
    pg.wait_for_timeout(1500)
    shot(pg, f'{lang}_1440_r_playing')
    click(pg, '#pdz-play', 150); s = st(pg); ta = s['t']
    check(f'{tag}: the play button pauses; the title, the badge and the button all say paused (P6)', not s['playing'] and Z(lang, 'Paused', '停在') in text(pg, '#pdz-st-title')
          and text(pg, '#pdz-status') == Z(lang, 'Paused', '已暂停') and text(pg, '#pdz-play') == Z(lang, '▶ Continue', '▶ 继续'), text(pg, '#pdz-st-title') + ' | ' + text(pg, '#pdz-status') + ' | ' + text(pg, '#pdz-play'))
    pos = pg.evaluate("()=>{const r=document.querySelector('#pdz-pos'); return +r.value/+r.max}")
    check(f'{tag} (P22): the slider stands where the cursor stands on the time axis', abs(pos - ta / s['X']) < 0.01, f'{pos:.3f} {ta / s["X"]:.3f}')
    w_pause = pg.evaluate("()=>document.querySelector('#pdz-play').getBoundingClientRect().width")
    click(pg, '#pdz-player [data-act="fwd"]', 120); tb = st(pg)['t']
    click(pg, '#pdz-player [data-act="back"]', 120); tc = st(pg)['t']
    check(f'{tag}: forward and back move one component', tb > ta and tc < tb, f'{ta:.3f} {tb:.3f} {tc:.3f}')
    pg.evaluate("()=>{const r=document.querySelector('#pdz-pos'); r.value=r.max; r.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(200); s = st(pg)
    check(f'{tag}: scrubbing to the end counts the message as watched and opens next', s['done'] and len(s['watched']) == 1 and not next_off(pg), json.dumps({k: s[k] for k in ('done', 'watched')}))
    w_done = pg.evaluate("()=>document.querySelector('#pdz-play').getBoundingClientRect().width")
    check(f'{tag} (P20): the play button keeps its width playing, paused and at the end', max(w_play, w_pause, w_done) - min(w_play, w_pause, w_done) < 0.5, (w_play, w_pause, w_done))
    n = s['plays']
    nxt(pg, 300); s = st(pg)
    check(f'{tag}: part 2 opens on the same, watched message without a replay', s['plays'] == n and not s['playing'] and s['done'])
    click(pg, '#pdz-rail .step.current [data-nav="back"]', 300); s = st(pg)
    check(f'{tag}: no replay when returning to the watched message in part 1', s['plays'] == n and not s['playing'] and s['done'], json.dumps({k: s[k] for k in ('plays', 'playing', 'done')}))
    click(pg, '#pdz-player .seg-ctl[data-key="spd"] button[data-v="fast"]', 100); pf = plan(pg)
    t0 = time.time(); click(pg, '#pdz-player [data-act="replay"]', 80); s = st(pg)
    check(f'{tag}: the 重放 / Replay control replays', s['plays'] == n + 1 and s['playing'])
    wait_end(pg); dt = time.time() - t0
    check(f'{tag}: 快 plays the message in about half the time of 中 ({pf["total"]:.1f} s against {pm["total"]:.1f} s), and the replay takes that time', 1.6 <= pm['total'] / pf['total'] <= 2.4 and pf['total'] - 0.5 <= dt <= pf['total'] * 1.2 + 0.8, f'{dt:.1f} s')
    click(pg, '#pdz-player .seg-ctl[data-key="spd"] button[data-v="med"]', 100)
    nxt(pg, 300); n = st(pg)['plays']; p_on = plan(pg)
    t0 = time.time(); lever(pg, 'stream', '0', 200); s = st(pg); p_off = plan(pg)
    check(f'{tag}: a replay when something changed (sending when done is a new message), at the reader’s speed, not the quick one (P3)', s['plays'] == n + 1 and s['playing'] and not s['quick'] and abs((s['total'] or 0) - p_off['total']) < 1e-6, json.dumps({k: s[k] for k in ('plays', 'playing', 'quick', 'total')}))
    check(f'{tag} (P3): both ways share one time axis ({p_off["X"]:.0f} s), and sending when done reaches its first word much later on screen ({p_off["fw"]:.1f} s against {p_on["fw"]:.1f} s)', p_off['X'] == p_on['X'] and p_off['fw'] > 2 * p_on['fw'], (p_off['fw'], p_on['fw']))
    wait_end(pg, 25); dt = time.time() - t0; s = st(pg)
    check(f'{tag}: the changed message plays for its planned {p_off["total"]:.1f} s', p_off['total'] - 0.5 <= dt <= p_off['total'] * 1.15 + 1.0 and s['done'] and not s['playing'], f'{dt:.1f} s')
    lever(pg, 'stream', '1', 200); s = st(pg)
    check(f'{tag}: streaming again shows its end at once (watched)', s['plays'] == n + 1 and not s['playing'] and s['done'])
    lever(pg, 'stream', '1', 200); s = st(pg)
    check(f'{tag}: no replay when nothing changed', s['plays'] == n + 1 and not s['playing'])
    lever(pg, 'stream', '0', 200); s = st(pg)
    check(f'{tag}: the watched no-streaming message shows its end at once', s['plays'] == n + 1 and not s['playing'])
    # the scenarios keep the quick replay for switching back and forth (R3.9)
    click(pg, '#pdz-app .modes [data-mode="sandbox"]', 500); n = st(pg)['plays']; q = pg.evaluate("()=>window.__pdz.quick")
    t0 = time.time(); click(pg, '#pdz-rail [data-preset="1"]', 60); s = st(pg)
    check(f'{tag}: in the scenarios a new setting plays quickly, the whole axis in {q} s', s['plays'] == n + 1 and s['playing'] and s['quick'], json.dumps({k: s[k] for k in ('plays', 'playing', 'quick')}))
    wait_end(pg); dt = time.time() - t0
    check(f'{tag}: the quick replay ends in about {q} s', dt <= q * 1.5 + 0.6, f'{dt:.1f} s')
    pg.close()


# N5: on a phone the stage title changes while the message plays; the picture under it must not move
TRACE_TOPS = """()=>{window.__tops={}; const f=()=>{const st=document.querySelector('#pdz-stage'), tr=document.querySelector('#pdz-trace');
  if(st&&tr&&tr.getClientRects().length){const k=Math.round(tr.getBoundingClientRect().top-st.getBoundingClientRect().top); __tops[k]=(__tops[k]||0)+1;} requestAnimationFrame(f);}; f();}"""


def phone_title(b, lang):
    tag = f'{lang} 390 N5'
    pg = page(b, 390, lang, reduced=False)
    click(pg, '#pdz-player .seg-ctl[data-key="spd"] button[data-v="fast"]', 50)
    pg.evaluate(TRACE_TOPS)
    wait_end(pg, 20)
    click(pg, '#pdz-player [data-act="replay"]', 900); click(pg, '#pdz-play', 400); click(pg, '#pdz-play', 100); wait_end(pg, 20)
    nxt(pg, 300); lever(pg, 'stream', '0', 300); wait_end(pg, 25)
    tops = pg.evaluate("()=>window.__tops"); titles = text(pg, '#pdz-st-title')
    check(f'{tag}: playing, pausing, continuing and the first word change the title, but the picture keeps its place in the stage', len(tops) == 1, json.dumps(tops) + ' ' + titles)
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in LANGS:
        for w in WIDTHS:
            walk(b, lang, w)
        if 1440 in WIDTHS: replay_rules(b, lang)
        if 390 in WIDTHS: phone_title(b, lang)
    b.close()
check('copy rules (RULES, spacing, punctuation, colons, boxes, drawing text)', not rule_hits, '; '.join(rule_hits[:10]))
check('no page errors', not errs, '; '.join(errs[:4]))
print('\n'.join(notes))
print(f'\n{len(notes) - len(fails)}/{len(notes)} passed')
sys.exit(1 if fails else 0)
