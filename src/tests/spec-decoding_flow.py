"""Flow test for the speculative decoding lab: every guided part, the scenarios and the challenge at 1440x900 and 390x844 in EN and ZH;
the gates and their reasons, the prediction (no leak before the pick, the verdict after), the levers, the red failure moments (R5.11),
the three challenge tiers and the brief comparison chart, the capability layer, the language switch mid-challenge, the hash round trip,
the dock on phones, no horizontal overflow, the copy rules shared with the other labs (no “toy”, no option-group disclaimers, Chinese
spacing and punctuation, no formula as plain text), and the replay rules of R3.20 for the round player (no replay when nothing changed,
none when returning to a watched round, a replay when something changed, the 重放 / Replay control, default speed 中, pause, step, scrub).
Screenshots go to shots/spec-decoding/.

Run: python3 tests/spec-decoding_flow.py [en|zh] [1440|390]   (filters; without them, everything)
"""
import json, pathlib, subprocess, sys, time
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/spec-decoding.html', '--only', 'spec-decoding'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'spd').mkdir(parents=True, exist_ok=True)
(P / 'out' / 'spd' / 'wrapped-flow.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'spec-decoding.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'spd' / 'wrapped-flow.html')
SH = P / 'shots' / 'spec-decoding'; SH.mkdir(parents=True, exist_ok=True)
ARGS = sys.argv[1:]
LANGS = [a for a in ARGS if a in ('en', 'zh')] or ['en', 'zh']
WIDTHS = [int(a) for a in ARGS if a in ('1440', '390')] or [1440, 390]
errs, notes, fails, rule_hits = [], [], [], []


def check(name, ok, info=''):
    notes.append(('OK   ' if ok else 'FAIL ') + name + (f'  [{info}]' if info else ''))
    if not ok: fails.append(name)


def page(b, w, lang, hash_='spec-decoding', reduced=True):
    pg = open_drawers(b.new_page(viewport={'width': w, 'height': 900 if w > 500 else 844}, reduced_motion='reduce' if reduced else 'no-preference'))
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


Z = lambda lang, en, zh: zh if lang == 'zh' else en
st = lambda pg: pg.evaluate("()=>window.__spd.state()")
overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
count = lambda pg, sel: pg.evaluate("s=>[...document.querySelectorAll(s)].filter(e=>e.getClientRects().length).length", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\s+/g,' ').trim():''}", sel)
next_off = lambda pg: pg.evaluate("()=>{const b=document.querySelector('#spd-rail .step.current [data-nav=\"next\"]'); return !b||b.disabled||b.getAttribute('aria-disabled')==='true'}")
click = lambda pg, sel, ms=250: (pg.evaluate("s=>document.querySelector(s).click()", sel), pg.wait_for_timeout(ms))
lever = lambda pg, v, ms=250: click(pg, f'#spd-rail [data-lkey][data-lever="{v}"]', ms)
go = lambda pg, i, ms=350: click(pg, f'#spd-rail [data-go="{i}"]', ms)
nxt = lambda pg, ms=350: click(pg, '#spd-rail .step.current [data-nav="next"]', ms)
shot = lambda pg, n: pg.screenshot(path=str(SH / f'{n}.png'))

# the copy rules the other labs' flow tests check (tests/grpo_flow.py, tests/ppo_flow.py), on this lab's root
RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · (?![0-9])/,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],
   [/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy”'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],[/OpenAI|Anthropic|DeepMind|Google|Meta\\b|Microsoft/,'company name'],
   [/一次只能选|一次选一个|不会叠加|do not stack|don't stack|one at a time/i,'option-group disclaimer (G1)'],[/第 ?[0-9一二三四五六七八] ?部分里/,'“第 N 部分里”'],
   [/深色|浅色|浅灰|深灰|\\b(?:dark|light|pale)(?:er)? (?:bar|line|block|part|cell)/i,'colour named by lightness (R5.7)'],
   [/÷|\\bk \\+ 1\\b|\\bj \\+ 1\\b|1 − α|α\\^|min\\(1|max\\(0|\\bp\\/q\\b|p − q/,'formula as plain text (H1)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, math, #tip, .fid-tag')||!e.getClientRects().length) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx').replace(/<math[\\s\\S]*?<\\/math>/g,'M'); for(const [re,name] of bad) if(re.test(t)) out.push(name+' (tooltip): '+t.slice(0,90));});
 for(const e of root.querySelectorAll('sub,sup')) if(!e.closest('[data-endnotes],pre,code')) out.push('sub/sup outside math: '+e.parentElement.textContent.trim().slice(0,40));
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
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#spd-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg,.spd-fx,.spd-sr')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.classList.contains('spd-chip')) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
# every svg text inside its drawing; the token chips: their word fits (a chip clips at its edge, so a word wider than its chip would be cut)
SVGIN = """()=>{const out=[]; for(const s of document.querySelectorAll('#spd-stage svg')){if(!s.getClientRects().length) continue; const b=s.getBoundingClientRect();
  for(const t of s.querySelectorAll('text')){if(!t.getClientRects().length||!t.textContent.trim()) continue; const r=t.getBoundingClientRect();
    if(r.left<b.left-2||r.right>b.right+2||r.top<b.top-2||r.bottom>b.bottom+2) out.push('outside: '+t.textContent.trim().slice(0,30));}}
  for(const c of document.querySelectorAll('#spd-stage .spd-chip')){if(!c.getClientRects().length||!c.textContent.trim()) continue; if(c.scrollWidth>c.clientWidth+1) out.push('chip cut: '+c.textContent);}
  return out;}"""


def rules(pg, where, lang):
    for sel in ('#spd-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
    for h in pg.evaluate(OVER): rule_hits.append(f'{where}: text wider than its box: {h}')
    for h in pg.evaluate(SVGIN): rule_hits.append(f'{where}: drawing text: {h}')
    o = overflow(pg)
    if o > 0: rule_hits.append(f'{where}: horizontal overflow {o}px')


def walk(b, lang, w):
    tag = f'{lang} {w}'
    pg = page(b, w, lang)
    s = st(pg)
    check(f'{tag} opens on part 1, 8B + 1B, k = 4, batch 1', s['mode'] == 'guided' and s['step'] == 0 and s['tgt'] == 't8' and s['dr'] == 'd1' and s['k'] == 4 and s['B'] == 1, json.dumps({k: s[k] for k in ('mode', 'step', 'dr', 'k', 'B')}))
    check(f'{tag} part 1 shows the one-token step: 7.53 ms', '7.53 ms' in text(pg, '#spd-cost'), text(pg, '#spd-cost')[:80])
    rules(pg, f'{tag} part 1', lang); shot(pg, f'{lang}_{w}_1_mission')
    nxt(pg)
    # part 2: the step cost
    check(f'{tag} part 2 gated until 5 and 640 tokens are tried; the reason is shown', next_off(pg) and Z(lang, 'Try 5 tokens and 640 tokens', '试过 5 个和 640 个 token') in text(pg, '#spd-rail'))
    check(f'{tag} part 2: the rows not tried yet (and the plain-decode baseline of 128 requests) show no step time', '20.1' not in text(pg, '#spd-cost') and '8.54' not in text(pg, '#spd-cost') and '16.0' not in text(pg, '#spd-cost'), text(pg, '#spd-cost')[:120])
    lever(pg, '5'); c5 = text(pg, '#spd-cost')
    check(f'{tag} part 2: 5 tokens take the same 7.53 ms as 1', c5.count('7.53 ms') == 2, c5[:120])
    lever(pg, '640'); s = st(pg)
    check(f'{tag} part 2: 640 tokens take 20.1 ms against 16.0 ms for plain decode of the same 128 requests; compute-bound; next opens', '20.1 ms' in text(pg, '#spd-cost') and '16.0 ms' in text(pg, '#spd-cost') and not next_off(pg) and Z(lang, 'compute-bound', '受算力限制') in text(pg, '#spd-rail') and '16.0 ms' in text(pg, '#spd-rail .obs'))
    rules(pg, f'{tag} part 2', lang); shot(pg, f'{lang}_{w}_2_cost')
    nxt(pg)
    # part 3: one round (reduced motion: the round shows its end at once and counts as watched)
    s = st(pg); rr = text(pg, '#spd-round')
    check(f'{tag} part 3: the example round with k = 4 keeps 2 guesses + the target’s own token; red rejected, green accepted', s['rpK'] == 4 and s['rpAcc'] == 2 and '4-2' in s['watched']
          and count(pg, '#spd-round .spd-ok') >= 4 and count(pg, '#spd-round .spd-bad') == 1 and count(pg, '#spd-round .spd-own') == 2, json.dumps({k: s[k] for k in ('rpK', 'rpAcc', 'watched')}))
    check(f'{tag} part 3 (R6.35): the core concept is named in the body text', Z(lang, 'speculative decoding', '投机解码') in text(pg, '#spd-rail .step.current'))
    check(f'{tag} part 3: the player has pause/play, step back/forward, 重放 / Replay, speeds 慢 中 快 with 中 pressed, a progress slider',
          count(pg, '#spd-rp-play') == 1 and count(pg, '#spd-round [data-act="rp-back"]') == 1 and count(pg, '#spd-round [data-act="rp-fwd"]') == 1
          and text(pg, '#spd-round [data-act="rp-replay"]') == Z(lang, 'Replay', '重放') and text(pg, '#spd-round .seg-ctl[data-key="rpSpd"]').replace(' ', '') == Z(lang, 'SlowMediumFast', '慢中快')
          and text(pg, '#spd-round .seg-ctl[data-key="rpSpd"] [aria-pressed="true"]') == Z(lang, 'Medium', '中') and count(pg, '#spd-rp-pos') == 1, text(pg, '#spd-round .seg-ctl[data-key="rpSpd"]'))
    check(f'{tag} part 3: next is open once the round has been seen to its end', not next_off(pg))
    rules(pg, f'{tag} part 3', lang); shot(pg, f'{lang}_{w}_3_round')
    nxt(pg)
    # part 4: the prediction
    rail = text(pg, '#spd-rail'); stg = text(pg, '#spd-stage')
    check(f'{tag} part 4: gated until a pick; no answer shown before it (3.36 nowhere, no metric card)', next_off(pg) and '3.36' not in rail and '3.36' not in stg and count(pg, '#spd-metrics .metric') == 0
          and Z(lang, 'Pick an answer to continue', '选一个答案才能继续') in rail)
    check(f'{tag} part 4 (R3.21): the question gives k, α, the drop rule and the bonus token', all(x in rail for x in (Z(lang, '4 tokens', '4 个 token'), '0.8', Z(lang, 'thrown away', '丢掉'), Z(lang, 'own', '自己的 1 个'))), rail[:200])
    click(pg, '#spd-rail [data-pred="a34"]')
    check(f'{tag} part 4: a pick opens next and is only recorded (the verdict waits for part 5)', not next_off(pg) and '3.36' not in text(pg, '#spd-rail'))
    rules(pg, f'{tag} part 4', lang); shot(pg, f'{lang}_{w}_4_predict')
    nxt(pg)
    # part 5: the formula, k = 2 / 4 / 8
    check(f'{tag} part 5: gated until k = 2 and 8 are tried; the formula is typeset; one metric card (3.36 at k = 4)', next_off(pg) and count(pg, '#spd-rail .spd-eq math') >= 2 and count(pg, '#spd-metrics .metric') == 1 and '3.36' in text(pg, '#spd-metrics'))
    lever(pg, '2'); lever(pg, '8', 400)
    obs = text(pg, '#spd-rail .obs')
    check(f'{tag} part 5: the verdict says the pick was right; k = 8 gives 4.33 and its simulated mean', not next_off(pg) and Z(lang, 'right', '是对的') in obs and '4.33' in obs and '4.37' in obs and '4.33' in text(pg, '#spd-metrics'), obs[:200])
    rules(pg, f'{tag} part 5', lang); shot(pg, f'{lang}_{w}_5_expect')
    nxt(pg)
    # part 6: the acceptance rule
    check(f'{tag} part 6: gated; the sampled rule matches the target within 0.6 percentage points', next_off(pg) and Z(lang, '0.6 percentage points', '0.6 个百分点') in text(pg, '#spd-rail [data-lkey][data-lever="sample"] .lv-r'))
    lever(pg, 'greedy')
    check(f'{tag} part 6: greedy keeps exactly the target’s output', Z(lang, 'exactly', '完全一样') in text(pg, '#spd-rail [data-lkey][data-lever="greedy"] .lv-r'))
    lever(pg, 'nocheck')
    check(f'{tag} part 6 (R5.11): without the check the output changes, drawn in red, the status says so; next opens', count(pg, '#spd-acc .spd-badb') >= 1 and Z(lang, 'changed', '输出变了') in text(pg, '#spd-stage') and not next_off(pg) and '55.4%' in text(pg, '#spd-rail .obs'))
    rules(pg, f'{tag} part 6', lang); shot(pg, f'{lang}_{w}_6_same')
    nxt(pg)
    # part 7: the batch
    check(f'{tag} part 7: gated; four metric cards; batch 1 gives 1.67 times', next_off(pg) and count(pg, '#spd-metrics .metric') == 4 and '1.67' in text(pg, '#spd-metrics'))
    lever(pg, '128')
    obs = text(pg, '#spd-rail .obs')
    check(f'{tag} part 7 (R5.11): at batch 128 the 3B is slower than plain decode (0.86, red points); next opens', not next_off(pg) and '0.86' in obs and count(pg, '#spd-bcurve .spd-baddot') >= 1 and st(pg)['B'] == 128, obs[:160])
    check(f'{tag} part 7: the timeline shows 24 tokens with plain decode and with guessing', count(pg, '#spd-time svg') == 1 and '24' in text(pg, '#spd-time'))
    rules(pg, f'{tag} part 7', lang); shot(pg, f'{lang}_{w}_7_batch')
    nxt(pg)
    check(f'{tag} part 8: misconceptions, production systems and the way to the scenarios', count(pg, '#spd-rail [data-mode-go="sandbox"]') >= 1 and 'vLLM' in text(pg, '#spd-rail') and '✗' in text(pg, '#spd-rail'))
    pg.evaluate("()=>{document.querySelector('#spd-math').open=true}"); pg.wait_for_timeout(150)
    check(f'{tag} Show the math: four named formulas, typeset', count(pg, '#spd-mathbody .spd-eq') >= 4 and count(pg, '#spd-mathbody math') >= 8)
    rules(pg, f'{tag} part 8', lang); shot(pg, f'{lang}_{w}_8_remember')
    # scenarios
    click(pg, '#spd-rail [data-mode-go="sandbox"]', 400)
    want = ['1.67', '1.32', '1.23', '2.57', '2.30']
    for i, x in enumerate(want):
        click(pg, f'#spd-rail [data-preset="{i}"]', 300)
        ins = text(pg, '#spd-insight')
        check(f'{tag} scenario {i + 1}: {x} times plain decode', x in ins and x in text(pg, '#spd-metrics'), ins[:120])
        rules(pg, f'{tag} scenario {i + 1}', lang)
        if i in (1, 4): shot(pg, f'{lang}_{w}_sc{i + 1}')
    s = st(pg); h = pg.evaluate("()=>location.hash")
    check(f'{tag} scenario 5 adds batch 8 to the batch choices; the hash carries the setting', s['B'] == 8 and count(pg, '#spd-controls .seg-ctl[data-key="B"] button[data-v="8"]') == 1 and h.endswith('t8_ngram_4_8_60'), h)
    pg2 = page(b, w, lang, hash_=h.lstrip('#')); s2 = st(pg2)
    check(f'{tag} the hash round trip restores the scenario', s2['mode'] == 'sandbox' and all(s2[k] == s[k] for k in ('tgt', 'dr', 'k', 'B', 'a')), json.dumps({k: s2[k] for k in ('mode', 'dr', 'B', 'a')}))
    pg2.close()
    click(pg, '#spd-controls .seg-ctl[data-key="tgt"] button[data-v="t70"]', 300)
    check(f'{tag} settings: the 70B is reachable and the hash changes with it', st(pg)['tgt'] == 't70' and 't70_' in pg.evaluate("()=>location.hash"))
    rules(pg, f'{tag} settings', lang)
    # challenge
    click(pg, '#spd-app .modes [data-mode="challenge"]', 400); s = st(pg)
    check(f'{tag} challenge opens on the 70B with 4 GPUs, batch 128, 3B, k = 4, at 1.22 times', s['mode'] == 'challenge' and s['tgt'] == 't70' and s['dr'] == 'd3' and s['k'] == 4 and s['B'] == 128 and '1.22' in text(pg, '#spd-ch-g'))
    check(f'{tag} challenge: no feedback before submitting', text(pg, '#spd-ch-res') == '')
    click(pg, '#spd-rail [data-act="submit"]', 300); r = text(pg, '#spd-ch-res')
    check(f'{tag} challenge tier 1 (not met): red line, a hint about the check of k + 1 tokens, Show the answer', count(pg, '#spd-ch-res .badline') == 1 and Z(lang, 'Hint', '提示') in r and count(pg, '#spd-ch-res [data-act="reveal"]') == 1, r[:160])
    click(pg, '#spd-ic-k button[data-v="2"]', 300); r = text(pg, '#spd-ch-res')
    check(f'{tag} challenge tier 2 (met, not best): 3B k = 2 gives 1.59 and names 1B k = 2 at 1.61', count(pg, '#spd-ch-res .okline') == 1 and '1.59' in r and '1.61' in r, r[:160])
    click(pg, '#spd-ic-dr button[data-v="d1"]', 300); r = text(pg, '#spd-ch-res')
    check(f'{tag} challenge tier 3 (best): 1B k = 2, 1.61, nothing better; the brief chart and the reason are shown', '1.61' in r and Z(lang, 'no other choice does better', '没有别的选择更快') in r and count(pg, '#spd-ch-res .cmp') + count(pg, '#spd-ch-res .cmpchart') + count(pg, '#spd-ch-res svg') >= 1 and Z(lang, 'compute-bound', '受算力限制') in r, r[:200])
    rules(pg, f'{tag} challenge', lang); shot(pg, f'{lang}_{w}_ch_best')
    other = 'en' if lang == 'zh' else 'zh'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(300); s = st(pg)
    check(f'{tag} the language switch keeps the challenge, the choice and the result', s['mode'] == 'challenge' and s['dr'] == 'd1' and s['k'] == 2 and s['chSubmitted'] and '1.61' in text(pg, '#spd-ch-res'))
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(300)
    ex = count(pg, '#spd-rail [data-exits]')
    check(f'{tag} the capability layer follows the challenge', ex == 1)
    if w == 390:
        pg.evaluate("()=>window.scrollTo(0, document.querySelector('#spd-stage').getBoundingClientRect().top+scrollY+300)"); pg.wait_for_timeout(300)
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
    nxt(pg); lever(pg, '5'); lever(pg, '640'); nxt(pg, 300)
    s = st(pg)
    check(f'{tag}: entering part 3 plays the round (not yet watched) at 中', s['playing'] and s['rpSpd'] == 'med' and s['plays'] == 1 and s['rpF'] < 3, json.dumps({k: s[k] for k in ('playing', 'rpSpd', 'plays', 'rpF')}))
    pg.wait_for_timeout(2200); f1 = st(pg)['rpF']
    check(f'{tag}: about one frame a second at 中', 1 <= f1 <= 3, f1)
    shot(pg, f'{lang}_1440_r_playing')
    click(pg, '#spd-rp-play', 150); s = st(pg)
    check(f'{tag}: the play button pauses', not s['playing'] and s['rpF'] == f1 or not s['playing'])
    click(pg, '#spd-round [data-act="rp-fwd"]', 100); f2 = st(pg)['rpF']
    click(pg, '#spd-round [data-act="rp-back"]', 100); f3 = st(pg)['rpF']
    check(f'{tag}: step forward and back move one frame', f2 == f3 + 1, f'{f2} {f3}')
    pg.evaluate("()=>{const r=document.querySelector('#spd-rp-pos'); r.value=r.max; r.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(150); s = st(pg)
    check(f'{tag}: scrubbing to the end counts the round as watched and opens next', '4-2' in s['watched'] and not next_off(pg))
    n = s['plays']
    click(pg, '#spd-rail .step.current [data-nav="back"]', 300); nxt(pg, 300); s = st(pg)
    check(f'{tag}: no replay when returning to a watched round', s['plays'] == n and not s['playing'] and s['rpF'] == 9, json.dumps({k: s[k] for k in ('plays', 'playing', 'rpF')}))
    click(pg, '#spd-round .seg-ctl[data-key="rpSpd"] button[data-v="fast"]', 100)
    t0 = time.time(); click(pg, '#spd-round [data-act="rp-replay"]', 80); s = st(pg)
    check(f'{tag}: the 重放 / Replay control replays', s['plays'] == n + 1 and s['playing'])
    dt = wait_end(pg) + (time.time() - t0 - 0.08) * 0
    check(f'{tag}: 9 frames at 快 take about 4.5 s', 3.5 <= dt <= 6.0, f'{dt:.1f} s')
    click(pg, '#spd-round .seg-ctl[data-key="rpSpd"] button[data-v="med"]', 100)
    nxt(pg, 300); click(pg, '#spd-rail [data-pred="a2"]', 200); n = st(pg)['plays']
    nxt(pg, 300); s = st(pg)
    check(f'{tag}: part 5 opens on the watched k = 4 round without a replay', s['plays'] == n and not s['playing'] and s['rpK'] == 4)
    lever(pg, '2', 200); s = st(pg)
    check(f'{tag}: a replay when something changed (k = 2 is a new round)', s['plays'] == n + 1 and s['playing'] and s['rpK'] == 2)
    lever(pg, '4', 200); s = st(pg)
    check(f'{tag}: k = 4 again shows its end at once (watched)', s['plays'] == n + 1 and not s['playing'] and s['rpK'] == 4)
    lever(pg, '4', 200); s = st(pg)
    check(f'{tag}: no replay when nothing changed', s['plays'] == n + 1 and not s['playing'])
    lever(pg, '2', 200); s = st(pg)
    check(f'{tag}: k = 2, cut short before its end, plays again from the start', s['plays'] == n + 2 and s['playing'])
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
