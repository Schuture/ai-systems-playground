"""Flow test for the speculative decoding lab: every guided part, the scenarios and the challenge at 1440x900 and 390x844 in EN and ZH;
the gates and their reasons, the prediction (no leak before the pick, the verdict after), the levers, the red failure moments (R5.11),
the three challenge tiers and the brief comparison chart, the capability layer, the language switch mid-challenge, the hash round trip,
the dock on phones, no horizontal overflow, the copy rules shared with the other labs (no “toy”, no option-group disclaimers, Chinese
spacing and punctuation, no formula as plain text), and the replay rules of R3.20 for the round player (no replay when nothing changed,
none when returning to a watched round, a replay when something changed, the 重放 / Replay control, default speed 中, pause, step, scrub).
The 知识梳理 (primer, outline approved 2026-10-14): its four sections in order, the link to the prefill-decode primer, the two inert pictures,
no result of the guided parts and no sentence copied from them, the copy rules, and its button to the guided parts.
Screenshots go to shots/spec-decoding/.

Run: python3 tests/spec-decoding_flow.py [en|zh] [1440|390]   (filters; without them, everything)
"""
import json, pathlib, re, subprocess, sys, time
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


def folds(pg, where, n, lang):
    """owner review 2026-10-14 (point 4): the card says the point in plain words; formulas and detail wait in n folds, closed at first.
    Opens them (they stay open for the checks that follow) and runs the copy rules on what they hold."""
    vis_math = "()=>[...document.querySelectorAll('#spd-rail .step.current math')].filter(m=>m.getClientRects().length&&!m.closest('details:not([open])')).length"
    k = count(pg, '#spd-rail .step.current details.spd-more'); shut = count(pg, '#spd-rail .step.current details.spd-more:not([open])')
    check(f'{where} (owner 2026-10-14): {n} fold(s), closed at first; no formula in the card text in view', k == n and shut == n and pg.evaluate(vis_math) == 0, f'{k} folds, {shut} closed, {pg.evaluate(vis_math)} formulas in view')
    pg.evaluate("()=>document.querySelectorAll('#spd-rail .step.current details.spd-more').forEach(d=>d.open=true)"); pg.wait_for_timeout(150)
    check(f'{where}: the folds open and hold the detail', count(pg, '#spd-rail .step.current details.spd-more[open]') == n and len(text(pg, '#spd-rail .step.current .spd-more-b')) > 40)
    rules(pg, f'{where} (folds open)', lang)


def walk(b, lang, w):
    tag = f'{lang} {w}'
    pg = page(b, w, lang)
    s = st(pg)
    check(f'{tag} opens on part 1, 8B + 1B, k = 4, batch 1', s['mode'] == 'guided' and s['step'] == 0 and s['tgt'] == 't8' and s['dr'] == 'd1' and s['k'] == 4 and s['B'] == 1, json.dumps({k: s[k] for k in ('mode', 'step', 'dr', 'k', 'B')}))
    check(f'{tag} part 1 shows the one-token step: 7.53 ms', '7.53 ms' in text(pg, '#spd-cost'), text(pg, '#spd-cost')[:80])
    rules(pg, f'{tag} part 1', lang); shot(pg, f'{lang}_{w}_1_mission')
    nxt(pg)
    # part 2: the step cost
    check(f'{tag} part 2 gated until 1 × 5 and 128 × 5 tokens are tried; the reason is shown', next_off(pg) and Z(lang, 'Try 1 request × 5 tokens and 128 requests × 5 tokens', '试过“1 个请求 × 5 个 token”和“128 个请求 × 5 个 token”') in text(pg, '#spd-rail'))
    hit = pg.evaluate("""()=>{const t=[...document.querySelectorAll('#spd-cost .spd-stepin svg text')].filter(e=>e.textContent.trim()).map(e=>e.getBoundingClientRect()), o=[];
      for(let i=0;i<t.length;i++) for(let j=i+1;j<t.length;j++){const a=t[i], b=t[j]; if(a.left<b.right-1&&b.left<a.right-1&&a.top<b.bottom-1&&b.top<a.bottom-1) o.push(i+'/'+j);} return o;}""")
    check(f'{tag} part 2 (reader 2026-10-14): the one-token step drawn first has no labels running into each other; the card sends the reader to 1 × 5 for the causal mask',
          not hit and Z(lang, 'Pick “1 request × 5 tokens” below', '选下面的“1 个请求 × 5 个 token”') in text(pg, '#spd-rail .step.current'), hit)
    check(f'{tag} part 2: the rows not tried yet (and the plain-decode baseline of 128 requests) show no step time', '20.1' not in text(pg, '#spd-cost') and '8.54' not in text(pg, '#spd-cost') and '16.0' not in text(pg, '#spd-cost'), text(pg, '#spd-cost')[:120])
    lever(pg, '5'); c5 = text(pg, '#spd-cost')
    check(f'{tag} part 2: 5 tokens take the same 7.53 ms as 1', c5.count('7.53 ms') == 2, c5[:120])
    check(f'{tag} part 2 (owner 2026-10-14): the step drawn holds 5 positions of one request; every row of the time chart is labelled requests × tokens',
          count(pg, '#spd-cost .spd-cxc') == 5 and all(Z(lang, e, z) in c5 for e, z in (('1 request × 1 token', '1 个请求 × 1 个 token'), ('1 request × 5 tokens', '1 个请求 × 5 个 token'), ('16 requests × 5 tokens', '16 个请求 × 5 个 token'),
          ('128 requests × 1 token', '128 个请求 × 1 个 token'), ('128 requests × 5 tokens', '128 个请求 × 5 个 token'))), c5[:300])
    lever(pg, '640'); s = st(pg)
    check(f'{tag} part 2: at 128 × 5 the step drawn is one request’s card of 5 positions, times 128 requests', count(pg, '#spd-cost .spd-cxc') == 5 and Z(lang, '× 128 requests', '× 128 个请求') in text(pg, '#spd-cost'), text(pg, '#spd-cost')[:200])
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
    rc = text(pg, '#spd-rcost')
    check(f'{tag} part 3 (owner 2026-10-14): the time strip compares the round with plain decode for the same 3 tokens; it computes 1.9 times more and reads the weights once',
          Z(lang, 'Plain decode', '普通 decode') in rc and Z(lang, '3 tokens', '3 个 token') in rc and '1.9' in rc and Z(lang, 'once instead of 3', '1 次，而不是 3 次') in rc, rc[:300])
    def stages_at(f):
        pg.evaluate(f"()=>{{const r=document.querySelector('#spd-rp-pos'); r.value={f}; r.dispatchEvent(new Event('input',{{bubbles:true}}))}}"); pg.wait_for_timeout(120)
        return pg.evaluate("()=>[...document.querySelectorAll('#spd-round .spd-stl')].map(e=>e.classList.contains('on')?'on':e.classList.contains('done')?'done':'-')")
    seq = [stages_at(f) for f in (2, 5, 6, 9)]
    grow = [(stages_at(f), pg.evaluate("()=>[...document.querySelectorAll('#spd-rcost svg text')].map(t=>t.textContent).join(' | ')"))[1] for f in (5, 7, 8)]
    check(f'{tag} part 3 (reader 2026-10-14): plain decode’s row grows with the tokens kept in ③ (1, 2, then 3 steps)',
          all(Z(lang, x, y) in g.split(' | ')[-1] for g, x, y in zip(grow, ('1 token', '2 tokens', '3 tokens'), ('1 个 token', '2 个 token', '3 个 token'))), grow)
    stages_at(9)
    check(f'{tag} part 3 (owner 2026-10-14): the stages ① guess, ② check, ③ keep are named on the figure and light up in order as the round plays',
          all(Z(lang, e, z) in text(pg, '#spd-round') for e, z in (('The draft model guesses', 'draft 模型猜'), ('The target model checks, in one step', '目标模型一次检查'), ('Keep', '留下')))
          and seq == [['on', '-', '-'], ['done', 'on', '-'], ['done', 'done', 'on'], ['done', 'done', 'done']], json.dumps(seq))
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
    vis_eq = "()=>[...document.querySelectorAll('#spd-rail .spd-eq math')].filter(m=>m.getClientRects().length&&!m.closest('details:not([open])')).length"
    shut5 = count(pg, '#spd-rail .step.current details.spd-more:not([open])'); eq0 = pg.evaluate(vis_eq)
    pg.evaluate("()=>document.querySelectorAll('#spd-rail .step.current details.spd-more').forEach(d=>d.open=true)"); pg.wait_for_timeout(150)
    check(f'{tag} part 5: gated until k = 2 and 8 are tried; the general formula waits in a closed fold and is typeset when opened (reader 2026-10-14); one metric card (3.36 at k = 4)',
          next_off(pg) and shut5 == 1 and eq0 == 0 and pg.evaluate(vis_eq) >= 2 and count(pg, '#spd-metrics .metric') == 1 and '3.36' in text(pg, '#spd-metrics')
          and Z(lang, 'each extra guess adds less', '多猜的收益越来越小') in text(pg, '#spd-rail .step.current'), f'{shut5} closed, {eq0} formulas before opening')
    lever(pg, '2'); lever(pg, '8', 400)
    obs = text(pg, '#spd-rail .obs')
    check(f'{tag} part 5: the verdict says the pick was right; k = 8 gives 4.33 and guess 8 survives about 17% of the time; the chart’s dots show the simulated rounds', not next_off(pg) and Z(lang, 'right', '是对的') in obs and '4.33' in obs and '17%' in obs and '4.33' in text(pg, '#spd-metrics')
          and count(pg, '#spd-kcurve .spd-dot') == 32, obs[:200])
    check(f'{tag} part 5 (reader 2026-10-14): the options say what the average counts', Z(lang, '2.44 tokens per target step on average', '跑一次平均 2.44 个 token') in text(pg, '#spd-rail [data-lkey][data-lever="2"] .lv-r'), text(pg, '#spd-rail [data-lkey][data-lever="2"] .lv-r'))
    rules(pg, f'{tag} part 5', lang); shot(pg, f'{lang}_{w}_5_expect')
    nxt(pg)
    # part 6: the acceptance rule
    check(f'{tag} part 6: gated; the sampled rule matches the target within 0.6 percentage points', next_off(pg) and Z(lang, '0.6 percentage points', '0.6 个百分点') in text(pg, '#spd-rail [data-lkey][data-lever="sample"] .lv-r'))
    lever(pg, 'greedy')
    check(f'{tag} part 6: greedy keeps exactly the target’s output; its chart runs to 100% (reader 2026-10-14)', Z(lang, 'exactly', '完全一样') in text(pg, '#spd-rail [data-lkey][data-lever="greedy"] .lv-r')
          and pg.evaluate("()=>[...document.querySelectorAll('#spd-acc svg text.spd-tk')].some(t=>t.textContent==='100%')"))
    lever(pg, 'nocheck')
    check(f'{tag} part 6 (R5.11): without the check the output changes, drawn in red, the status says so; next opens', count(pg, '#spd-acc .spd-badb') >= 1 and Z(lang, 'changed', '输出变了') in text(pg, '#spd-stage') and not next_off(pg) and '55.4%' in text(pg, '#spd-rail .obs')
          and Z(lang, '15.4 percentage points', '15.4 个百分点') in text(pg, '#spd-rail [data-lkey][data-lever="nocheck"] .lv-r'))
    folds(pg, f'{tag} part 6', 1, lang)
    rules(pg, f'{tag} part 6', lang); shot(pg, f'{lang}_{w}_6_same')
    nxt(pg)
    # part 7: the batch
    check(f'{tag} part 7: gated; four metric cards; batch 1 gives 1.67 times', next_off(pg) and count(pg, '#spd-metrics .metric') == 4 and '1.67' in text(pg, '#spd-metrics'))
    lever(pg, '128')
    obs = text(pg, '#spd-rail .obs')
    check(f'{tag} part 7 (R5.11): at batch 128 the 3B is slower than plain decode (0.86, red points); next opens', not next_off(pg) and '0.86' in obs and count(pg, '#spd-bcurve .spd-baddot') >= 1 and st(pg)['B'] == 128, obs[:160])
    check(f'{tag} part 7: the timeline shows 24 tokens with plain decode and with guessing; its caption names the guesser and the acceptance rate (reader 2026-10-14)', count(pg, '#spd-time svg') == 1 and '24' in text(pg, '#spd-time')
          and Z(lang, 'close to the average', '接近平均水平') in text(pg, '#spd-time .spd-tnote') and Z(lang, 'not drawn at random', '不是随机抽的') not in text(pg, '#spd-time'), text(pg, '#spd-time .spd-tnote'))
    folds(pg, f'{tag} part 7', 1, lang)
    rules(pg, f'{tag} part 7', lang); shot(pg, f'{lang}_{w}_7_batch')
    nxt(pg)
    r8 = text(pg, '#spd-rail .step.current')
    check(f'{tag} part 8 (owner 2026-10-14): three short take-aways with their numbers (2.77 tokens, 1.67 times, 0.86 times at batch 128)',
          count(pg, '#spd-rail .step.current ul > li') == 3 and all(x in r8 for x in ('2.77', '1.67', '0.86')) and 'vLLM' not in r8, r8[:300])
    folds(pg, f'{tag} part 8', 3, lang)
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


def primer_walk(b, lang, w):
    """The 知识梳理 (outline approved by the owner on 2026-10-14): four sections in the approved order, the link to the prefill-decode
    primer (which owns memory-bound decode), the lab's own two drawings as an inert display, no result the guided parts let the reader
    find, no sentence copied from the guided parts, the copy rules; its last button opens the guided parts on part 1."""
    tag = f'{lang} {w} primer'
    pg = b.new_page(viewport={'width': w, 'height': 900 if w > 500 else 844}, reduced_motion='reduce'); pg.add_init_script('window.__aspPrimer=true')
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang} primer: {e}'))
    pg.goto(URL); pg.wait_for_timeout(250); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate("()=>{location.hash='spec-decoding'}"); pg.wait_for_timeout(700)
    PR = '#spd-app .primer[data-primer="spec-decoding"]'
    r = pg.evaluate("""s=>{const b=[...document.querySelectorAll('.bench.flowed')].find(x=>x.offsetParent), p=document.querySelector(s);
      return {flow:b&&b.dataset.flow, h:p?[...p.querySelectorAll('.pr-h')].map(e=>e.textContent):[], k:p?[...p.querySelectorAll('.pr-k')].map(e=>e.textContent):[],
        figs:p?p.querySelectorAll('.pr-fig').length:0, inert:!!p&&[...p.querySelectorAll('.pr-frames')].every(f=>f.inert),
        a1:p?[...p.querySelectorAll('.pr-sec[data-psec="1"] a')].map(a=>a.getAttribute('href')):[]}}""", PR)
    want = [Z(lang, 'memory-bound', '受显存带宽限制'), Z(lang, 'one forward pass', '一次前向'), Z(lang, 'Speculative decoding', '投机解码'), Z(lang, 'output does not change', '输出为什么不变')]
    check(f'{tag}: opens on the 知识梳理; four sections in the approved order (memory-bound decode, one forward pass, speculative decoding, why the output stays)',
          r['flow'] == 'primer' and len(r['h']) == 4 and r['k'][-1] == '4/4' and all(x in h for x, h in zip(want, r['h'])), json.dumps(r['h'], ensure_ascii=False))
    # this page is built with this lab only, so the name is plain text here; in the site it is the link that opens that lab on its primer
    check(f'{tag}: section 1 points to the prefill-decode primer instead of teaching memory-bound decode again',
          Z(lang, 'the basics of Prefill and decode', '“Prefill 和 decode”的知识梳理') in text(pg, PR + ' .pr-sec[data-psec="1"]') and all(h == '#prefill-decode' for h in r['a1']), r['a1'])
    check(f'{tag}: two pictures, the lab’s own drawings shown as a display (inert): what one forward pass computes, and the example round with ① ② ③',
          r['figs'] == 2 and r['inert'] and count(pg, PR + ' .pr-fig svg.spd-svg') == 1 and count(pg, PR + ' .pr-fig .spd-stl') == 3, r)
    t = text(pg, PR)
    leak = [x for x in ('7.53', '2.44', '3.36', '4.33', '2.77', '1.67', '0.86', '12.5', '22.6', '1.9', '15.0', '15 GB', '0.8') if x in t]
    check(f'{tag}: no result the guided parts let the reader find (step times, tokens per step, the speedups)', not leak, leak)
    check(f'{tag}: the formulas are typeset; the acceptance rule adds up to p on a line of its own', count(pg, PR + ' .spd-prfx math') >= 4 and count(pg, PR + ' .pr-text p math') >= 20)
    rules(pg, tag, lang)
    out = pg.evaluate("""s=>{const o=[]; for(const v of document.querySelectorAll(s+' svg')){if(!v.getClientRects().length) continue; const b=v.getBoundingClientRect();
      for(const x of v.querySelectorAll('text')){if(!x.textContent.trim()) continue; const q=x.getBoundingClientRect(); if(q.left<b.left-2||q.right>b.right+2||q.top<b.top-2||q.bottom>b.bottom+2) o.push(x.textContent.trim());}}
      for(const c of document.querySelectorAll(s+' .spd-chip')) if(c.textContent.trim()&&c.scrollWidth>c.clientWidth+1) o.push('chip cut: '+c.textContent); return o;}""", PR)
    check(f'{tag}: every drawing text inside its picture, every chip word whole', not out, out[:4])
    pg.screenshot(path=str(SH / f'{lang}_{w}_0_primer.png'), full_page=True)
    click(pg, PR + ' .pr-end [data-flow-go="guided"]', 400)
    check(f'{tag}: “Next: the guided parts” opens part 1', pg.evaluate("()=>[...document.querySelectorAll('.bench.flowed')].find(x=>x.offsetParent).dataset.flow") == 'guided' and st(pg)['step'] == 0)
    # readers flagged primers that repeat the guided parts word for word: no sentence of 16 characters or more appears in a guided card (folds open)
    cards = []
    for i in range(8):
        click(pg, f'#spd-rail [data-go="{i}"]', 250)
        pg.evaluate("()=>document.querySelectorAll('#spd-rail .step.current details.spd-more').forEach(d=>d.open=true)"); pg.wait_for_timeout(60)
        cards.append(text(pg, '#spd-rail .step.current'))
    G = ' '.join(cards)
    sents = [x.strip() for x in re.split(r'(?<=[。？！.?!])\s+|(?<=[。？！])', t) if len(x.strip()) >= 16]
    same = [x for x in sents if x in G]
    check(f'{tag}: no sentence copied from the guided parts ({len(sents)} sentences compared)', len(sents) >= 15 and not same, same[:3])
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in LANGS:
        for w in WIDTHS:
            walk(b, lang, w)
            primer_walk(b, lang, w)
        if 1440 in WIDTHS: replay_rules(b, lang)
    b.close()
check('copy rules (RULES, spacing, punctuation, colons, boxes, drawing text)', not rule_hits, '; '.join(rule_hits[:10]))
check('no page errors', not errs, '; '.join(errs[:4]))
print('\n'.join(notes))
print(f'\n{len(notes) - len(fails)}/{len(notes)} passed')
sys.exit(1 if fails else 0)
