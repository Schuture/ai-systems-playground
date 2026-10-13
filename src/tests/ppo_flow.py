"""Flow test for the PPO lab: every guided part, the scenarios and the challenge at 1440x900 and 390x844 in EN and ZH; the gates,
the prediction, the levers, the red failure moments (R5.11), the memory strip, the three challenge tiers and the comparison chart,
the capability layer, the language switch mid-challenge, the hash round-trip, the dock on phones, no horizontal overflow, and the
replay rules of R3.20 (no replay when nothing changed, none when returning to a watched run, a replay when something changed, the
重放 / Replay control, default speeds 中 for the whole run, the one-step replay and part 6's passes, about 3 s for 200 steps at 中
and 1.5 s at 快). The owner's message of 2026-10-08 (reviews/owner-b9-2026-10-08.md), PPO items 1-4 and rules G1/G2, each has its
own checks, named “owner b9 item …”, “G1” and “G2”. Screenshots go to shots/ppo/.

Run: python3 tests/ppo_flow.py [en|zh] [1440|390]   (filters; without them, everything)
"""
import json, pathlib, re, subprocess, sys, time
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/ppo.html', '--only', 'ppo'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'ppo').mkdir(parents=True, exist_ok=True)
(P / 'out' / 'ppo' / 'wrapped-ppo.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'ppo.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'ppo' / 'wrapped-ppo.html')
SH = P / 'shots' / 'ppo'; SH.mkdir(parents=True, exist_ok=True)
ARGS = sys.argv[1:]
LANGS = [a for a in ARGS if a in ('en', 'zh')] or ['en', 'zh']
WIDTHS = [int(a) for a in ARGS if a in ('1440', '390')] or [1440, 390]
errs, notes, fails = [], [], []

def check(name, ok, info=''):
    notes.append(('OK   ' if ok else 'FAIL ') + name + (f'  [{info}]' if info else ''))
    if not ok: fails.append(name)

def page(b, w, lang, hash_='ppo', reduced=True):
    pg = open_drawers(b.new_page(viewport={'width': w, 'height': 900 if w > 500 else 844}, reduced_motion='reduce' if reduced else 'no-preference'))
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_TUNNEL' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(400)
    return pg

Z = lambda lang, en, zh: zh if lang == 'zh' else en
st = lambda pg: pg.evaluate("()=>window.__ppo.state()")
overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
count = lambda pg, sel: pg.evaluate("s=>document.querySelectorAll(s).length", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\s+/g,' ').trim():''}", sel)
next_off = lambda pg: pg.evaluate("()=>{const b=document.querySelector('#ppo-rail [data-nav=\"next\"]'); return !!b&&(b.disabled||b.getAttribute('aria-disabled')==='true')}")
click = lambda pg, sel, ms=250: (pg.evaluate("s=>document.querySelector(s).click()", sel), pg.wait_for_timeout(ms))

TOY = r"""()=>{const re=/in this toy\b|在这个玩具里|\btoy\b(?! run)|玩具(?!实跑)/i, out=[], root=document.querySelector('#ppo-app'); if(!root) return out;
 const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, script, style, .fid-tag')) continue; if(re.test(n.textContent)) out.push(n.textContent.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx],[data-tip]').forEach(e=>{const t=e.getAttribute('data-tipx')||e.getAttribute('data-tip')||''; if(re.test(t)) out.push('(tooltip) '+t.slice(0,90));});
 const d=document.getElementById('dock'); if(d&&re.test(d.innerText)) out.push('(dock) '+d.innerText.slice(0,90));
 return out;}"""
toy_hits = []
# the owner's clarification (2026-10-07 23:37 UTC): no hover only expands a shown number to its exact value; Show the math may write one out
EXACT_HOVER = """r=>{const root=document.querySelector(r), out=[]; if(!root) return out;
  for(const e of root.querySelectorAll('[data-tipx],[data-tip]')){if(e.closest('.mathbody')) continue; let t=e.getAttribute('data-tipx')||''; const k=e.getAttribute('data-tip');
    if(k&&window.ASP) t+=' '+(window.ASP.t(k)||''); if(/精确值是|Exactly/.test(t)) out.push(t.slice(0,100));}
  for(const e of root.querySelectorAll('.num-big')) if(!e.closest('.mathbody')&&(e.hasAttribute('data-tipx')||e.closest('.def'))) out.push('hover on '+e.textContent);
  return out;}"""


# H1 (the owner, 2026-10-09): no formula written as plain text (text, hovers, the tip on screen), no <sub>/<sup> outside code, no TeX command left as a word
PLAINRE = r'β × log|log\(π|π_ref|π_old|KL\(π|Σ π|1 \+ ε|1 − ε|ρA|clip\(ρ|min\(ρ|A = R|R = r|÷'
PLAINF = '''([r,re])=>{const root=document.querySelector(r), out=[]; if(!root) return out; const rx=new RegExp(re); let n;
  const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
  while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('math,pre,code,script,style')) continue; if(rx.test(n.textContent)) out.push('text: '+n.textContent.trim().slice(0,60));}
  for(const e of root.querySelectorAll('sub,sup')) if(!e.closest('[data-endnotes],pre,code')) out.push('sub/sup: '+e.parentElement.textContent.trim().slice(0,40));
  for(const e of root.querySelectorAll('[data-tipx]')){const t=e.getAttribute('data-tipx'); if(rx.test(t)) out.push('hover: '+t.slice(0,60));}
  const tip=document.getElementById('tip'); if(tip&&!tip.hidden){const w=document.createTreeWalker(tip,NodeFilter.SHOW_TEXT); while((n=w.nextNode())){if(n.parentElement.closest('math')) continue; if(rx.test(n.textContent)) out.push('tip: '+n.textContent.trim().slice(0,60));}}
  for(const m of document.querySelectorAll('math')) for(const i of m.querySelectorAll('mi')) if(/^(frac|sqrt|left|right|hat|text|cdot|times|div|approx|sum|mathrm|operatorname|quad|beta|pi|sigma|rho|theta|varepsilon|bar|tilde|overline|langle|rangle|mid|le|ge|big|Big|partial|nabla)$/.test(i.textContent)) out.push('TeX word: '+i.textContent+' in '+m.textContent.slice(0,40));
  return out;}'''
# H2: each named formula in Show the math: its name, its lines, whether its “=” line up; H5: a hairline above every block heading but the first
EQJS = '''(sel)=>{const b=document.querySelector(sel); if(!b) return null; const eqs=[...b.querySelectorAll('.ppo-eq')].map(e=>{const xs=[...e.querySelectorAll('.ppo-eq-o math')].map(m=>Math.round(m.getBoundingClientRect().left));
    return {name:((e.querySelector('.ppo-eq-n')||{}).textContent||'').trim().slice(0,24), lines:[...e.querySelectorAll('.ppo-fx math')].map(m=>m.textContent.replace(/\\s+/g,'')), aligned:new Set(xs).size<=1};});
  const hs=[...b.querySelectorAll('.ppo-mh')].slice(1); return {eqs, rules:hs.length, ruleOk:hs.every(h=>{const cs=getComputedStyle(h); return cs.borderTopWidth==='1px'&&cs.borderTopStyle==='solid';})};}'''


def toy(pg, where):
    hx = pg.evaluate(EXACT_HOVER, '#ppo-app'); check(f'{where}: no exact-value hover outside Show the math', not hx, hx[:3])
    pf = pg.evaluate(PLAINF, ['#ppo-app', PLAINRE]); check(f'{where}: H1: no formula written as plain text; every formula rendered (no TeX word)', not pf, pf[:3])
    eq = pg.evaluate(EQJS, '#ppo-mathbody')
    if eq and eq['eqs']:
        check(f'{where}: Show the math: H2: every formula named on the left, its “=” lined up, the reward and the advantage with this response’s numbers; H5: hairlines between blocks',
              all(e['name'] and e['aligned'] for e in eq['eqs']) and any(len(e['lines']) >= 2 and 'Â' in e['name'] or len(e['lines']) >= 2 and 'A' in e['name'] for e in eq['eqs']) and eq['rules'] >= 2 and eq['ruleOk'], eq)
    for h in pg.evaluate(TOY): toy_hits.append(f'{where}: {h}')

def go(pg, i, ms=350): click(pg, f'#ppo-rail [data-go="{i}"]', ms)

# rule G1 of the owner's message of 2026-10-08: option groups get a lead-in, never “pick one at a time / they do not stack”
G1_BAN = ['一次只能选', '一次选一个', '不会叠加', 'do not stack', "don't stack", 'one at a time']
G1_SRC = [x for x in G1_BAN if x in (P / 'labs' / 'ppo.js').read_text()]
G1_TXT = """()=>{const root=document.querySelector('#ppo-app'); if(!root) return ''; let t=root.innerText;
  root.querySelectorAll('[aria-label],[data-tipx],[title]').forEach(e=>{t+=' '+(e.getAttribute('aria-label')||'')+' '+(e.getAttribute('data-tipx')||'')+' '+(e.getAttribute('title')||'');});
  return t;}"""
g1_hits = []
def g1(pg, where):
    t = pg.evaluate(G1_TXT)
    for x in G1_BAN:
        if x in t: g1_hits.append(f'{where}: {x}')
LEADS = {3: ('Pick an update rule and see how true quality changes for the 6 seeds.', '选一种更新规则，看看 6 个种子的真实质量怎么变。'),
         4: ('Pick a KL coefficient β and see whether true quality falls over 200 steps.', '选一个 KL 系数 β，看看 200 步里真实质量会不会跌下去。'),
         5: ('Pick how each batch is used and watch, in the top chart, which responses of step 1’s batch are still pushed at each pass.', '选一种学法，在最上面的图里看第 1 步这批回答每学一遍，哪些还在被推。'),
         6: ('Pick a method and see which models it keeps and how much memory they take.', '选一种方法，看看它要放哪几个模型、一共占多少显存。')}
# part 6's picture (owner b9, item 3): step 1's batch, frame by frame (pass 1 … K, then “after”); only the current frame's title,
# legend and caption are visible (the others are stacked under it, R7.9)
CP = """()=>{const h=document.querySelector('#ppo-clip'), n=s=>h.querySelectorAll(s).length, t=s=>{const e=h.querySelector(s); return e?e.innerText.replace(/\\s+/g,' ').trim():''};
  return {show:!h.hidden&&!h.closest('[hidden]'), bad:n('.ppo-cp-svg .ppo-cd-bad'), flat:n('.ppo-cp-svg .ppo-cd-flat'), on:n('.ppo-cp-svg .ppo-cd-on'),
    ex:[...h.querySelectorAll('.ppo-cp-svg .ppo-cg')].map(g=>g.textContent), panels:n('.ppo-cp'), pass:window.__ppo.state().clipPass,
    title:t('.ppo-clip-t.on'), sub:t('.ppo-clip-s'), cap:t('.ppo-cp-cap.on'), key:t('.ppo-cp-key.on'), pressed:t('.seg-ctl[data-key="pass"] [aria-pressed="true"]'),
    play:t('#ppo-pass-play'), txt:h.innerText.replace(/\\s+/g,' ')};}"""
# the lead-in is the paragraph right before the option group
LEAD_BEFORE = "()=>{const g=document.querySelector('#ppo-rail .step.current [data-lkey]'); const box=g&&g.closest('.levers,[role=\"radiogroup\"],[role=\"group\"]')||g&&g.parentElement; const p=box&&box.previousElementSibling; return p?p.innerText.replace(/\\s+/g,' ').trim():''}"

# review codex-b9v3, item 2: each part shows only the knobs and pictures it has introduced
KNOBS = ['#ppo-steps-wrap', '#ppo-ctl-base-wrap', '#ppo-ctl-beta-wrap', '#ppo-ctl-K-wrap', '#ppo-ctl-clip-wrap', '#ppo-ctl-eta-wrap']
shown = lambda pg, sels: pg.evaluate("ss=>ss.map(s=>{const e=document.querySelector(s); return !!e&&!e.closest('[hidden]')})", sels)
def knobs(pg): return dict(zip(['steps', 'base', 'beta', 'K', 'clip', 'eta'], shown(pg, KNOBS)))
def cards(pg): return pg.evaluate("()=>[...document.querySelectorAll('#ppo-metrics .metric')].filter(e=>!e.closest('[hidden]')).map(e=>e.dataset.k)")
CARD_FITS = """()=>{const c=document.querySelector('#ppo-rail .step.current'); if(!c) return {ok:true};
  const r=document.querySelector('#ppo-rail').getBoundingClientRect(), b=c.getBoundingClientRect(), phone=r.width===0;
  const right=phone?innerWidth-12:r.right+0.5, left=phone?12:r.left-0.5; return {ok:b.right<=right&&b.left>=left, card:[Math.round(b.left),Math.round(b.right)], rail:[Math.round(r.left),Math.round(r.right)]};}"""
card_bad = []
def card_fits(pg, where):
    m = pg.evaluate(CARD_FITS)
    if not m['ok']: card_bad.append(f'{where}: {m}')
def lever(pg, v, ms=350): click(pg, f'#ppo-rail [data-lkey][data-lever="{v}"]', ms)
def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(100)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)
def wait_run(pg, limit=16000):
    t0 = time.time()
    while pg.evaluate("()=>window.__ppo.state().running") and (time.time() - t0) * 1000 < limit: pg.wait_for_timeout(100)
    return time.time() - t0

def walk(b, lang, w):
    tag = f'{lang}_{w}'
    pg = page(b, w, lang)
    s = st(pg)
    check(f'{tag} opens in guided part 1', s['mode'] == 'guided' and s['step'] == 0)
    check(f'{tag} title is the catalog question', text(pg, '#ppo-app h1') == Z(lang, 'Why does PPO need a value model and a KL penalty?', 'PPO 为什么需要价值模型和 KL 惩罚？'), text(pg, '#ppo-app h1'))
    check(f'{tag} owner b9 item 4: default speeds Medium for the whole run, the one-step replay and part 6’s passes; a whole run takes 6 / 3 / 1.5 s, Scenarios and Settings at most 1.5 s (R3.9)',
          (s['runSpd'], s['loopSpd'], s['passSpd']) == ('med', 'med', 'med') and s['runSecs'] == {'slow': 6, 'med': 3, 'fast': 1.5} and s['quickSecs'] == 1.5, json.dumps({k: s[k] for k in ('runSpd', 'loopSpd', 'passSpd', 'runSecs', 'quickSecs')}))
    card1, app1 = text(pg, '#ppo-rail .step.current'), text(pg, '#ppo-app')
    rl_def = Z(lang, 'reinforcement learning (RL)', '强化学习（reinforcement learning，简称 RL）')
    m_rl = re.search(r'\bRL\b(?!HF)', app1)
    check(f'{tag} owner b9 item 1 (G3): the mission says what RL is (try, a reward per try, high-scoring behaviour more likely) and that PPO is one RL algorithm, the one InstructGPT used; no “RL” before that',
          rl_def in card1 and m_rl is not None and m_rl.start() >= app1.find(rl_def) and all(x in card1 for x in Z(lang, ['letting it try', 'reward', 'high-scoring behaviour more likely', 'PPO is one RL algorithm', 'InstructGPT was trained with it'],
          ['让它自己去试', '叫作奖励', '让得分高的做法更常出现', 'PPO 是一种 RL 算法', 'InstructGPT 就是用它训练的'])), card1[:400])
    check(f'{tag} G1: no “pick one at a time / they do not stack” in the lab’s source', not G1_SRC, str(G1_SRC))
    shot(pg, f'{tag}_1_mission')
    k, v = knobs(pg), shown(pg, ['#ppo-mem', '#ppo-cvw', '#ppo-math', '#ppo-clip', '#ppo-later', '#ppo-reset-wrap']); lat = text(pg, '#ppo-later')
    check(f'{tag} review item 2: part 1 shows no knob, no memory strip, no curve, no formulas; the later list names the 6 knobs by part',
          not any(k.values()) and v == [False, False, False, False, True, False] and all(x in lat for x in Z(lang, ['Coming in later parts', 'Part 4', 'Part 5', 'Part 6', 'value model', 'KL coefficient', 'clipping', 'step size'], ['后面会讲', '第 4 部分', '第 5 部分', '第 6 部分', '价值模型', 'KL 系数', '截断', '步长'])), f'{k} {v} {lat[:120]}')
    go(pg, 1)
    check(f'{tag} part 2 gated until step 1 is played', next_off(pg))
    click(pg, '#ppo-rail [data-nav="showme"]', 400)
    s = st(pg)
    check(f'{tag} part 2 show me plays step 1 and opens the gate', s['steps'] == 1 and not next_off(pg))
    obs = text(pg, '#ppo-rail [data-obs]')
    check(f'{tag} part 2 observation names the most-written response', Z(lang, 'was written 3 times', '被写了 3 次') in obs, obs[:160])
    check(f'{tag} part 2 loop panel shows 12 responses with scores and advantages', count(pg, '#ppo-rows .ppo-chip:not(.ppo-hide)') == 12 and count(pg, '#ppo-rows .ppo-ca.up') >= 1)
    shot(pg, f'{tag}_2_loop')
    k, v = knobs(pg), shown(pg, ['#ppo-loop', '#ppo-mem', '#ppo-cvw', '#ppo-clip'])
    check(f'{tag} review item 2: part 2 stage shows one step only (generate, score, update): no memory strip, no curve, no knobs; cards are score and quality',
          not any(k.values()) and v == [True, False, False, False] and cards(pg) == ['rm', 'q'], f'{k} {v} {cards(pg)}')
    go(pg, 2)
    check(f'{tag} part 3 gated until a prediction', next_off(pg))
    q = text(pg, '#ppo-rail .step.current')
    check(f'{tag} part 3 question gives the score range (R3.21)', '0.80' in q and '0.94' in q, q[:120])
    click(pg, '#ppo-rail [data-pred="best"]')
    check(f'{tag} part 3 prediction opens the gate', not next_off(pg) and st(pg)['pred'] == 'best')
    shot(pg, f'{tag}_3_predict')
    go(pg, 3, 500)
    s = st(pg)
    check(f'{tag} part 4 shows the 6-seed fan without a baseline', s['view'] == 'fan' and s['base'] == 0 and count(pg, '#ppo-curve .ppo-l-fan') == 5 and next_off(pg))
    k = knobs(pg); lat = text(pg, '#ppo-later')
    check(f'{tag} review item 2: part 4 adds the training steps and the value model; β and the part 6 knobs still wait', k == dict(steps=True, base=True, beta=False, K=False, clip=False, eta=False)
          and Z(lang, 'value model', '价值模型') not in lat and Z(lang, 'Part 5', '第 5 部分') in lat and shown(pg, ['#ppo-mem']) == [False], f'{k} {lat[:100]}')
    lever(pg, '1', 500)
    obs = text(pg, '#ppo-rail [data-obs]')
    check(f'{tag} part 4 both options tried: gate open, prediction answered', not next_off(pg) and Z(lang, 'You predicted', '你预测的是') in obs and Z(lang, 'every one above', '全都高于') in obs, obs[:200])
    shot(pg, f'{tag}_4_value')
    go(pg, 4, 500)
    s = st(pg)
    check(f'{tag} part 5 starts with no KL penalty, 200 steps', s['beta'] == 0 and s['steps'] == 200 and next_off(pg))
    c5 = text(pg, '#ppo-rail .step.current')
    check(f'{tag} G2: part 5 says what its chart shows without recalling part 4’s seeds', Z(lang, 'The chart shows true quality for 6 random seeds, 200 steps each.', '图里画的是 6 个随机种子各训练 200 步的真实质量。') in c5
          and not re.search(Z(lang, r'as part 4|[Ii]n part 4', r'第 4 部分'), c5), c5[:240])
    k = knobs(pg)
    check(f'{tag} review item 2: part 5 adds β and the KL card; passes, clipping and step size still wait', k == dict(steps=True, base=True, beta=True, K=False, clip=False, eta=False) and 'kl' in cards(pg) and 'rho' not in cards(pg), f'{k} {cards(pg)}')
    check(f'{tag} part 5 failure marked red: hack segment, E red on all 3 prompts, status bad (R5.11)',
          count(pg, '#ppo-curve .ppo-l-hack') == 1 and count(pg, '#ppo-bars .ppo-hot') == 3 and 'bad' in pg.get_attribute('#ppo-status', 'class'))
    shot(pg, f'{tag}_5a_nokl')
    lever(pg, '0.05', 500)
    check(f'{tag} part 5 beta 0.05: no red, status ok, gate open', count(pg, '#ppo-curve .ppo-l-hack') == 0 and count(pg, '#ppo-bars .ppo-hot') == 0 and 'ok' in pg.get_attribute('#ppo-status', 'class') and not next_off(pg))
    obs = text(pg, '#ppo-rail [data-obs]')
    check(f'{tag} part 5 observation compares E with and without the penalty', '11%' in obs and '99%' in obs and '3%' in obs, obs[:200])
    shot(pg, f'{tag}_5b_kl')
    go(pg, 5, 500)
    s = st(pg)
    check(f'{tag} part 6 starts with one pass and the large step; its 200-step run is there at once', s['view'] == 'rho' and s['eta'] == 'l' and s['K'] == 1 and s['steps'] == 200 and not s['running'] and next_off(pg),
          json.dumps({k: s[k] for k in ('view', 'eta', 'K', 'steps', 'running')}))
    k = knobs(pg); card6 = text(pg, '#ppo-rail .step.current')
    check(f'{tag} owner b9 item 3a/3c: part 6’s picture is step 1’s batch, first in the stage; no 200-step curve, run transport or ratio card (they taught nothing here); passes, clipping and step size in the settings',
          all(k.values()) and shown(pg, ['#ppo-later', '#ppo-cvw', '#ppo-run', '#ppo-clip']) == [False, False, False, True] and 'rho' not in cards(pg)
          and pg.evaluate("()=>document.querySelector('#ppo-stage .stage-head').nextElementSibling.id") == 'ppo-clip', f'{k} {cards(pg)}')
    check(f'{tag} review item 7: part 6 gives the two examples (better and worse than expected, ratios 1.3 and 0.7) before the formula on its own line',
          all(x in card6 for x in Z(lang, ['better than expected', 'worse than expected', '1.3', '0.7', 'still pushed up', 'still pushed down'], ['比预期好', '比预期差', '1.3', '0.7', '还会被往上推', '还会被往下压']))
          and card6.index(Z(lang, 'better than expected', '比预期好')) < card6.index('min') and pg.evaluate("()=>document.querySelectorAll('#ppo-rail .step.current math').length") >= 2, card6[:300])
    check(f'{tag} owner b9 item 3d: the card says in bold that clipping does not keep ρ inside [0.8, 1.2]; a response out of it the way its advantage pushes gets gradient 0, the others keep learning',
          all(x in card6 for x in Z(lang, ['does not keep ρ inside [0.8, 1.2].', 'its gradient is 0', 'The other responses keep learning'], ['不会把 ρ 留在 [0.8, 1.2] 里。', '它的梯度是 0', '其余的回答照样学']))
          and Z(lang, 'does not keep ρ inside', '不会把 ρ 留在') in pg.evaluate("()=>[...document.querySelectorAll('#ppo-rail .step.current b')].map(b=>b.innerText).join(' ')"), card6[:400])
    check(f'{tag} owner b9 item 3b: the card ties steps and passes (one batch per step, learned several times, then a new batch)',
          all(x in card6 for x in Z(lang, ['each PPO training step has two halves', 'learns from that same batch several times, one update per pass', 'The next step writes a new batch'], ['PPO 的每一步训练分两段', '再用这同一批回答学几遍，每学一遍更新一次参数', '下一步再重新写一批'])), card6[:300])
    note = text(pg, '#ppo-ctl-clip-wrap')
    check(f'{tag} review item 7: the clipping note in the settings names the advantage’s sign both ways', all(x in note for x in Z(lang, ['positive advantage', 'negative advantage', 'above 1.2', 'below 0.8', 'pushed back'], ['优势为正', '优势为负', '超过 1.2', '低于 0.8', '推回来'])), note[:200])
    cp = pg.evaluate(CP)
    check(f'{tag} owner b9 item 3b: one pass: the picture says “step 1’s batch after its 1 pass” and that each step writes a batch and learns from it once, 200 steps in all',
          cp['show'] and cp['pass'] == 0 and cp['title'] == Z(lang, 'Step 1’s batch after its one pass: where did the ratios end?', '第 1 步的这批回答学完 1 遍：比值停在哪里？')
          and all(x in cp['sub'] for x in Z(lang, ['Each training step writes one batch', 'learns from that same batch once', '200 steps in all'], ['每一步训练先写一批回答', '再用这同一批学 1 遍', '一共训练 200 步'])) and cp['pressed'] == Z(lang, 'After', '学完'), json.dumps(cp, ensure_ascii=False)[:300])
    lever(pg, '4-0', 500)
    cp = pg.evaluate(CP); lr = text(pg, '#ppo-rail [data-lkey][data-lever="4-0"] .lv-r')
    check(f'{tag} review item 7: without clipping, after the 4 passes, the ratios out of the band are red and the caption says every pass pushed again (24 times)',
          cp['pass'] == 0 and cp['bad'] >= 1 and cp['flat'] == 0 and Z(lang, 'Every pass pushed again', '每一遍都照样推') in cp['cap'] and Z(lang, '24 times', '24 倍') in cp['cap'] and Z(lang, '24 times', '24 倍') in lr, json.dumps(cp, ensure_ascii=False)[:300] + ' | ' + lr)
    click(pg, '#ppo-pass button[data-v="2"]', 300)
    cp = pg.evaluate(CP)
    check(f'{tag} review item 7: without clipping, pass 2 marks in red the 5 responses already out of the band the way their advantage pushes, still pushed further',
          cp['pass'] == 2 and cp['bad'] == 5 and cp['flat'] == 0 and Z(lang, '5 responses that have already left the band', '有 5 个回答已经朝优势的方向走出区间，还在被往外推') in cp['cap']
          and cp['title'] == Z(lang, 'Step 1’s batch, pass 2 of 4: which responses are still pushed?', '第 1 步的这批回答，第 2 遍（共 4 遍）：哪些还在被推？'), json.dumps(cp, ensure_ascii=False)[:300])
    shot(pg, f'{tag}_6a_noclip')
    lever(pg, '4-1', 500)
    cp = pg.evaluate(CP); lr = text(pg, '#ppo-rail [data-lkey][data-lever="4-1"] .lv-r')
    check(f'{tag} part 6 clipped: gate open, the end view shows hollow (stopped) and filled (pushed to the last pass) dots, the lever reads 1.57', not next_off(pg) and cp['pass'] == 0 and cp['flat'] == 6 and cp['on'] == 2 and cp['bad'] == 0 and '1.57' in lr, json.dumps(cp, ensure_ascii=False)[:200] + ' | ' + lr)
    check(f'{tag} owner b9 item 3d: the end caption says clipping did not pull the 6 hollow responses back into the band, it only stopped pushing them; 2 were pushed to the last pass',
          all(x in cp['cap'] for x in Z(lang, ['6 responses end outside [0.8, 1.2] (hollow)', 'Clipping did not pull them back into the band', 'stopped pushing them', 'The other 2 were pushed to the last pass', '1.57'],
              ['有 6 个回答停在 [0.8, 1.2] 外（空心）', '截断没有把它们拉回区间', '就不再推', '另外 2 个一直学到最后一遍', '1.57'])), cp['cap'])
    if w == 1440:
        pg.hover('#ppo-clip .ppo-cp-svg .ppo-cd-flat'); pg.wait_for_timeout(250)
        tp = text(pg, '#tip')
        check(f'{tag} review item 7: a dot’s hover lists the response, its advantage, its ratio and what the passes did', all(x in tp for x in Z(lang, ['Advantage', 'ρ', 'no longer pushed'], ['优势', 'ρ', '不再'])), tp[:200])
        pg.mouse.move(2, 2)
    click(pg, '#ppo-pass button[data-v="1"]', 300)
    p1 = pg.evaluate(CP)
    check(f'{tag} review item 7: the pass picker moves the dots (pass 1: all pushed)', p1['pass'] == 1 and p1['flat'] == 0 and p1['on'] >= 1 and p1['pressed'] == Z(lang, 'Pass 1', '第 1 遍'), json.dumps(p1, ensure_ascii=False)[:200])
    click(pg, '#ppo-pass button[data-v="2"]', 300)
    cp = pg.evaluate(CP)
    check(f'{tag} owner b9 item 3d: clipped, pass 2: 5 hollow dots (gradient 0), 3 filled ones still learning; the caption says so and that clipping does not pull ρ back; the dashed examples at 0.7 and 1.3 and all four zones named',
          cp['panels'] == 2 and cp['flat'] == 5 and cp['on'] == 3 and cp['bad'] == 0 and cp['ex'] == Z(lang, ['e.g. 0.7', 'e.g. 1.3'], ['例 0.7', '例 1.3'])
          and all(x in cp['cap'] for x in Z(lang, ['Pass 2: 5 responses have already left [0.8, 1.2]', 'gradient 0', 'The other 3 keep learning', 'Clipping does not pull ρ back into the band'], ['第 2 遍：有 5 个回答的 ρ 已经朝优势的方向走出 [0.8, 1.2]', '梯度为 0', '其余 3 个照样学', '截断不会把 ρ 拉回区间']))
          and all(x in cp['txt'] for x in Z(lang, ['still pushed up', 'still pushed down', 'plateau', 'no more push'], ['仍往上推', '仍往下压', '平台', '不再往上推', '不再往下压'])), json.dumps(cp, ensure_ascii=False)[:400])
    obs = text(pg, '#ppo-rail [data-obs]')
    check(f'{tag} owner b9 item 3e: the clipped observation makes one point (clipping did not keep ρ in the band; out the way of the advantage: no longer pushed; the others learn) after the 200-step result',
          all(x in obs for x in Z(lang, ['4 passes, clipped', 'the same as one pass', 'Clipping did not keep ρ inside the band', 'is no longer pushed (hollow)', 'the others keep learning (filled)'], ['学 4 遍、截断', '和学 1 遍时一样', '截断没有把 ρ 留在区间里', '不再被推（空心）', '其余的照样学（实心）']))
          and not any(x in obs for x in Z(lang, ['shared', 'does not lock', '1.55'], ['共用', '锁住', '1.55'])), obs[:400])
    shot(pg, f'{tag}_6b_clip')
    go(pg, 6, 400)
    mem = text(pg, '#ppo-mem')
    check(f'{tag} part 7 memory strip first in the stage: 321.2 GB, 5 H100s', pg.evaluate("()=>document.querySelector('#ppo-stage .stage-head').nextElementSibling.id") == 'ppo-mem' and '321.2' in mem and Z(lang, 'at least 5 H100s', '至少 5 张') in mem, mem[:140])
    lever(pg, 'grpo'); g = text(pg, '#ppo-mem'); lever(pg, 'dpo'); d = text(pg, '#ppo-mem')
    check(f'{tag} part 7 GRPO 176.7 GB, DPO 160.6 GB, 3 H100s each', '176.7' in g and '160.6' in d and Z(lang, 'at least 3 H100s', '至少 3 张') in d)
    shot(pg, f'{tag}_7_cost')
    lever(pg, 'ppo')
    go(pg, 7, 400)
    r = text(pg, '#ppo-rail .step.current')
    check(f'{tag} part 8 lists the three misconceptions', r.count('✗') == 3)
    check(f'{tag} G2: the recap says what each piece does and why, without recalling what parts 4, 5 and 6 showed',
          not re.search(Z(lang, r'[Ii]n part \d|as in part \d|as part \d', r'第 \d 部分'), r) and all(x in r for x in Z(lang,
          ['Without it, where training goes depends mostly on which responses happen to be written. With it, another seed gives about the same result.',
           'Without it the reward-model score keeps rising while true quality can fall',
           'Clipping lets one batch be used several times without the passes adding up.',
           'So later methods drop some of them', '“Clipping keeps ρ inside [0.8, 1.2].” It holds ρ nowhere.'],
          ['没有它，训练往哪走，主要看碰巧写出了哪些回答。有了它，换个种子，结果也差不多。',
           '没有它，奖励模型的分数会一路涨，真实质量却可能跌下去，因为策略模型会去钻奖励模型的漏洞，比如注水。',
           '截断让同一批回答可以多学几遍，又不会越推越远。朝优势方向走出区间的回答不再被推，其余的照样学。',
           '所以后面的方法都想办法省掉一些。', '“截断会把 ρ 限制在 [0.8, 1.2] 里。”它不管 ρ 停在哪里'])), r[:300])
    lead_bad = []
    for i in range(8):
        go(pg, i, 150); toy(pg, f'{tag} part {i + 1}'); card_fits(pg, f'{tag} part {i + 1}'); g1(pg, f'{tag} part {i + 1}')
        if i in LEADS:
            lb = pg.evaluate(LEAD_BEFORE)
            if lb != Z(lang, *LEADS[i]): lead_bad.append(f'part {i + 1}: {lb[:90]}')
    check(f'{tag} G1: parts 4-7 open their options with a lead-in sentence right above them', not lead_bad, '; '.join(lead_bad))
    check(f'{tag} every part’s card stays inside the rail, with a gutter on phones (B1)', not card_bad, '; '.join(card_bad[:3]))
    check(f'{tag} no horizontal overflow in guided', overflow(pg) <= 0, str(overflow(pg)))
    shot(pg, f'{tag}_8_remember')
    click(pg, '#ppo-rail [data-mode-go="sandbox"]', 500)
    check(f'{tag} into the scenarios', st(pg)['mode'] == 'sandbox')
    sl = pg.evaluate("()=>{const g=document.querySelector('#ppo-rail .presets'); const p=g&&g.previousElementSibling; return [p?p.innerText.trim():'', g?g.getAttribute('aria-label'):'']}")
    check(f'{tag} G1: the scenarios open with a lead-in right above them', sl == [Z(lang, 'Pick a scenario and watch the reward-model score and true quality over 200 steps.', '选一个场景，看看训练 200 步里，奖励模型的分数和真实质量怎么变。'), Z(lang, 'Scenarios: pick one', '应用场景：选一个')], str(sl))
    check(f'{tag} review item 2: the scenarios show every knob, the memory strip and no later list', all(knobs(pg).values()) and shown(pg, ['#ppo-mem', '#ppo-later']) == [True, False], str(knobs(pg)))
    for i in range(4):
        click(pg, f'#ppo-rail [data-preset="{i}"]', 450)
        on = pg.get_attribute(f'#ppo-rail [data-preset="{i}"]', 'aria-pressed') == 'true'
        ins = text(pg, '#ppo-insight')
        good = on and ins and (i != 1 or Z(lang, 'locks onto', '锁定在') in ins) and (i != 2 or (Z(lang, 'found E', '找到了 E') in ins and Z(lang, 'Over 40 seeds', '试 40 个种子') in ins)) and (i != 3 or Z(lang, 'At step 1', '第 1 步') in ins)
        check(f'{tag} scenario {i + 1} selected with its insight', good, ins[:160])
        g1(pg, f'{tag} scenario {i + 1}')
        shot(pg, f'{tag}_9_scenario{i + 1}'); toy(pg, f'{tag} scenario {i + 1}')
    hsh = pg.evaluate("()=>location.hash")
    check(f'{tag} hash written for the last scenario', hsh == '#ppo~1_0.05_4_0_l_200', hsh)
    if w == 390:
        pg.evaluate("()=>{const e=document.querySelector('#ppo-metrics'); e.scrollIntoView({block:'start'}); window.scrollBy(0,600)}"); pg.wait_for_timeout(500)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!!d&&!d.hidden, text:d?d.innerText.replace(/\\s+/g,' '):'', fits:[...document.querySelectorAll('#dock .dock-in > *')].filter(e=>getComputedStyle(e).display!=='none').every(e=>e.getBoundingClientRect().right<=innerWidth+0.5)}}")
        check(f'{tag} dock shows the reading when the bars are off screen', dk['shown'] and Z(lang, 'true quality', '真实质量') in dk['text'] and dk['fits'], json.dumps(dk, ensure_ascii=False)[:200])
        shot(pg, f'{tag}_9b_dock')
    click(pg, '#ppo-app .modes [data-mode="challenge"]', 500)
    s = st(pg)
    check(f'{tag} challenge starts from no KL, 4 passes, no clipping, large step', s['mode'] == 'challenge' and (s['beta'], s['K'], s['clip'], s['eta'], s['steps']) == (0, 4, 0, 'l', 200))
    check(f'{tag} challenge settings panel locked', 'locked' in pg.get_attribute('#ppo-controls', 'class'))
    click(pg, '#ppo-rail [data-act="submit"]', 400)
    check(f'{tag} tier 1: not yet, with a hint', count(pg, '#ppo-ch-res .badline') == 1 and count(pg, '#ppo-rail [data-act="reveal"]') == 1)
    click(pg, '#ppo-ic-beta button[data-v="0.2"]', 400)
    res = text(pg, '#ppo-ch-res')
    check(f'{tag} tier 1b: beta 0.2 alone fixes quality but not the KL', count(pg, '#ppo-ch-res .badline') == 1 and Z(lang, 'over 0.5', '超过了 0.5') in res, res[:160])
    click(pg, '#ppo-ic-beta button[data-v="0.05"]', 400)
    res = text(pg, '#ppo-ch-res')
    check(f'{tag} tier 1c: beta 0.05 alone: quality still low, the hint points at clipping', count(pg, '#ppo-ch-res .badline') == 1 and Z(lang, 'pushed far enough', '被推得够远') in res, res[:200])
    rv = text(pg, '#ppo-ch-res [data-act="reveal"]').strip()
    check(f'{tag} tier 1c: the reveal button says “{Z(lang, "Show the answer", "直接看答案")}” (the owner, batch 8)', rv == Z(lang, 'Show the answer', '直接看答案'), rv)
    click(pg, '#ppo-ch-res [data-act="reveal"]', 400)
    BRIEF = """()=>{const res=document.querySelector('#ppo-ch-res'), c=res.querySelector('.cmp'); const rs=c?[...c.querySelectorAll('.cmp-row')]:[], tg=r=>!!r.querySelector('.cmp-tag,.cmp-you');
      return {n:rs.length, ok:rs.filter(r=>r.classList.contains('ok')).length, me:rs.filter(r=>r.classList.contains('me')).length, stray:rs.filter(r=>!r.classList.contains('ok')&&!r.classList.contains('me')&&!tg(r)).length,
        labs:rs.map(r=>r.querySelector('.cmp-lab').textContent), title:c?c.querySelector('.cmp-title').textContent:'', note:(res.querySelector('.cmp-note')||{}).textContent||''};}"""
    bf = pg.evaluate(BRIEF)
    check(f'{tag} tier 1c revealed: the chart holds the 6 passing combinations, your pick and the initial setup, and no other failing one; the note counts the other 4',
          (bf['n'], bf['ok'], bf['me'], bf['stray']) == (8, 6, 1, 0) and any(Z(lang, 'initial setup', '初始设置') in x for x in bf['labs']) and Z(lang, 'The other 4 combinations', '其余 4 种组合') in bf['note'], bf)
    check(f'{tag} challenge: the chart title no longer claims every combination', bf['title'] and not any(x in bf['title'] for x in ('each', 'every', '各种', '每种')), bf['title'])
    click(pg, '#ppo-ic-clip button[data-v="1"]', 500)
    res = text(pg, '#ppo-ch-res')
    bf = pg.evaluate(BRIEF)
    check(f'{tag} tier 3: two changes meet the goal, the fewest possible; the chart holds the 6 passing combinations (your pick among them) and the initial setup, the note counts the other 5',
          count(pg, '#ppo-ch-res .okline') == 1 and Z(lang, '2 changes is the fewest', '2 处改动已经是最少的') in res and (bf['n'], bf['ok'], bf['me'], bf['stray']) == (7, 6, 1, 0) and Z(lang, 'The other 5 combinations', '其余 5 种组合') in bf['note'], (res[:160], bf))
    check(f'{tag} capability layer shown after submitting', count(pg, '#ppo-rail [data-exits]') >= 1)
    toy(pg, f'{tag} challenge'); g1(pg, f'{tag} challenge')
    lead_s = text(pg, '#ppo-rail')
    check(f'{tag} G1: the challenge says its three settings can be changed together', Z(lang, 'You can change these three together; change as few as you can.', '下面三项可以一起改，改得越少越好。') in lead_s, lead_s[:300])
    check(f'{tag} G1: no “pick one at a time / they do not stack” in any part, the scenarios or the challenge (text, labels, tooltips)', not g1_hits, '; '.join(g1_hits[:4]))
    n0 = len(toy_hits); [toy_hits.append(f'{tag} teaser: {x}') for x in pg.evaluate("""()=>{const re=/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i, el=document.createElement('div'); document.body.appendChild(el);
      window.__ppo.teaser(el); const out=[...el.querySelectorAll('*')].map(e=>e.childNodes).flatMap(c=>[...c]).filter(n=>n.nodeType===3&&re.test(n.textContent)).map(n=>n.textContent.slice(0,90)); el.remove(); return out;}""")]
    check(f'{tag} no “toy” / 玩具 outside the fidelity label (R6.11), guided, scenarios, challenge, dock, tooltips', not toy_hits, '; '.join(toy_hits[:4]))
    tz = pg.evaluate("""()=>{const el=document.createElement('div'); el.style.width='280px'; document.body.appendChild(el); window.__ppo.teaser(el); const out=[];
      for(const v of ['0','1','2']){el.querySelector(`.seg-ctl[data-key="tzset"] button[data-v="${v}"]`).click(); out.push([el.querySelector('.tz-read').textContent, el.querySelectorAll('path.ppo-tz-q').length, el.querySelectorAll('path.ppo-tz-bad').length]);}
      const fits=el.scrollWidth<=el.clientWidth+1; el.remove(); return {out, fits};}""")
    reads = [o[0] for o in tz['out']]
    check(f'{tag} tile: three settings answer the title (score only / + value model / + KL penalty), 6 seeds each, red only where quality falls', len(set(reads)) == 3 and all(o[1] == 6 for o in tz['out']) and tz['out'][1][2] == 6 and tz['out'][2][2] == 0 and '0.69' in reads[2] and tz['fits'], json.dumps(tz, ensure_ascii=False)[:300])
    check(f'{tag} no horizontal overflow in the challenge', overflow(pg) <= 0, str(overflow(pg)))
    shot(pg, f'{tag}_10_challenge', full=True)
    other = 'en' if lang == 'zh' else 'zh'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(500)
    s = st(pg)
    check(f'{tag} language switch keeps the challenge state', s['mode'] == 'challenge' and (s['beta'], s['clip']) == (0.05, 1))
    pg.close()
    pg = page(b, w, lang, hash_='ppo~1_0_4_1_n_200')
    s = st(pg)
    check(f'{tag} hash round-trip opens the setup in scenarios', s['mode'] == 'sandbox' and (s['base'], s['beta'], s['K'], s['clip'], s['eta'], s['steps']) == (1, 0, 4, 1, 'n', 200) and not s['running'])
    pg.close()

def wait_pass(pg, limit=16000):
    t0 = time.time()
    while pg.evaluate("()=>window.__ppo.state().passing") and (time.time() - t0) * 1000 < limit: pg.wait_for_timeout(50)
    return time.time() - t0

def replay_rules(b, lang):
    """R3.20 with normal motion at 1440, the speeds of the owner's item 4 and the passes of part 6 (item 3)."""
    tag = f'{lang}_1440_R3.20'
    pg = page(b, 1440, lang, reduced=False)
    go(pg, 1); click(pg, '#ppo-rail [data-act="loop-play"]', 200)
    s = st(pg); check(f'{tag} one-step replay plays', s['looping'] and s['loopStarts'] == 1)
    pg.wait_for_timeout(8600); s = st(pg)
    check(f'{tag} one-step replay (3 phases) ends at Medium in about 6 s', not s['looping'] and s['steps'] == 1, json.dumps(s))
    go(pg, 0); go(pg, 1); s = st(pg)
    check(f'{tag} returning to a watched step: no replay, the end shown at once', s['loopStarts'] == 1 and not s['looping'] and s['steps'] == 1)
    go(pg, 2); click(pg, '#ppo-rail [data-pred="noise"]'); go(pg, 3, 200); wait_run(pg); lever(pg, '1', 200); wait_run(pg); go(pg, 4, 150)
    s0 = st(pg)
    check(f'{tag} entering part 5 plays its run', s0['running'])
    dt = wait_run(pg) + 0.15
    check(f'{tag} owner b9 item 4: 200 steps at Medium (the default) take about 3 s', 2.4 <= dt <= 4.0, f'{dt:.1f} s')
    n = st(pg)['runStarts']
    lever(pg, '0', 300); s = st(pg)
    check(f'{tag} no replay when nothing changed', s['runStarts'] == n and not s['running'])
    go(pg, 5, 300); go(pg, 4, 300); s = st(pg)
    check(f'{tag} no replay when returning to a watched run (part 6 in between plays passes, not a run)', s['runStarts'] == n and not s['running'] and s['steps'] == 200, json.dumps({k: s[k] for k in ('runStarts', 'running', 'steps')}))
    lever(pg, '0.05', 300); s = st(pg)
    check(f'{tag} a replay when something changed', s['runStarts'] == n + 1 and s['running'])
    lr = text(pg, '#ppo-rail [data-lkey][data-lever="0.05"] .lv-r')
    check(f'{tag} no result before its run ends: the lever reads “{Z(lang, "training…", "训练中…")}” (S2)', lr == Z(lang, 'training…', '训练中…'), lr)
    wait_run(pg)
    lr = text(pg, '#ppo-rail [data-lkey][data-lever="0.05"] .lv-r')
    check(f'{tag} the lever shows its result once the run has ended', '0.69' in lr, lr)
    rb = text(pg, '#ppo-run-replay')
    check(f'{tag} the replay control is named {Z(lang, "Replay", "重放")}', rb == Z(lang, 'Replay', '重放'), rb)
    click(pg, '#ppo-run .seg-ctl[data-key="runSpd"] button[data-v="fast"]', 100)
    t0 = time.time(); click(pg, '#ppo-run-replay', 60); s = st(pg)
    check(f'{tag} the replay control replays', s['runStarts'] == n + 2 and s['running'])
    shot(pg, f'{lang}_1440_11_replaying')
    wait_run(pg); dt = time.time() - t0
    check(f'{tag} owner b9 item 4: 200 steps at Fast take about 1.5 s', 1.1 <= dt <= 2.3, f'{dt:.1f} s')
    click(pg, '#ppo-run .seg-ctl[data-key="runSpd"] button[data-v="med"]', 100)
    # part 6 (owner b9, item 3): step 1's batch plays pass by pass, on screen, at the top of the stage
    ps0 = st(pg)['passStarts']   # part 6 was entered (and left at once) above
    go(pg, 5, 150); s = st(pg); cp = pg.evaluate(CP)
    vis = pg.evaluate("()=>{const r=document.querySelector('#ppo-clip .ppo-cps').getBoundingClientRect(), h=document.querySelector('#ppo-clip').getBoundingClientRect(); return {top:Math.round(h.top), bottom:Math.round(r.bottom), vh:innerHeight}}")
    check(f'{tag} owner b9 item 3a: entering part 6 plays step 1’s batch at once, in view at the top of the stage; the picker, the title and the button follow the pass; no 200-step run plays',
          s['passing'] and s['passStarts'] == ps0 + 1 and s['clipPass'] == 1 and not s['running'] and 0 <= vis['top'] and vis['bottom'] <= vis['vh'] and cp['pressed'] == Z(lang, 'Pass 1', '第 1 遍')
          and cp['title'] == Z(lang, 'Step 1’s batch, pass 1 of 1: which responses are still pushed?', '第 1 步的这批回答，第 1 遍（共 1 遍）：哪些还在被推？') and cp['play'] == Z(lang, '❚❚ Pause', '❚❚ 暂停'), json.dumps({**vis, 'cp': {k: cp[k] for k in ('pressed', 'title', 'play')}}, ensure_ascii=False))
    shot(pg, f'{lang}_1440_12_p6_enter')
    wait_pass(pg)
    t0 = time.time(); lever(pg, '4-1', 100); s = st(pg); cp = pg.evaluate(CP); lr = text(pg, '#ppo-rail [data-lkey][data-lever="4-1"] .lv-r')
    check(f'{tag} owner b9 item 3b: a new option plays its 4 passes (“pass 1 of 4”); its result waits until they end (“{Z(lang, "playing…", "播放中…")}”)',
          s['passing'] and s['passStarts'] == ps0 + 2 and s['clipPass'] == 1 and cp['title'] == Z(lang, 'Step 1’s batch, pass 1 of 4: which responses are still pushed?', '第 1 步的这批回答，第 1 遍（共 4 遍）：哪些还在被推？') and lr == Z(lang, 'playing…', '播放中…'), f'{cp["title"]} | {lr}')
    pg.wait_for_timeout(1300); s = st(pg); cp = pg.evaluate(CP)
    check(f'{tag} owner b9 item 3d: while it plays, pass 2’s caption says clipping no longer pushes the responses out of the band the way of their advantage and does not pull ρ back',
          s['clipPass'] == 2 and cp['pressed'] == Z(lang, 'Pass 2', '第 2 遍') and Z(lang, 'Clipping does not pull ρ back into the band', '截断不会把 ρ 拉回区间') in cp['cap'], json.dumps({k: cp[k] for k in ('pressed', 'cap')}, ensure_ascii=False))
    shot(pg, f'{lang}_1440_13_p6_pass2')
    wait_pass(pg); dt = time.time() - t0; s = st(pg); cp = pg.evaluate(CP); lr = text(pg, '#ppo-rail [data-lkey][data-lever="4-1"] .lv-r')
    check(f'{tag} owner b9 item 4 (G6): 4 passes and the end view take about 5 s at Medium (1 s a frame)', 4.4 <= dt <= 6.3, f'{dt:.1f} s')
    check(f'{tag} the passes end on the “{Z(lang, "After", "学完")}” view; the button reads “{Z(lang, "Replay the 4 passes", "重放这 4 遍")}”, the lever its result',
          s['clipPass'] == 0 and cp['pressed'] == Z(lang, 'After', '学完') and cp['play'] == Z(lang, 'Replay the 4 passes', '重放这 4 遍') and '1.57' in lr, f'{cp["pressed"]} | {cp["play"]} | {lr}')
    n = s['passStarts']
    lever(pg, '4-1', 300); s = st(pg)
    check(f'{tag} R3.20, part 6: no replay of the passes when nothing changed', s['passStarts'] == n and not s['passing'])
    go(pg, 6, 300); go(pg, 5, 300); s = st(pg)
    check(f'{tag} R3.20, part 6: no replay when returning to watched passes', s['passStarts'] == n and not s['passing'] and s['clipPass'] == 0, json.dumps({k: s[k] for k in ('passStarts', 'passing', 'clipPass', 'K', 'clip')}))
    click(pg, '#ppo-pass-play', 150); s = st(pg)
    check(f'{tag} R3.20, part 6: the replay button replays the passes', s['passStarts'] == n + 1 and s['passing'])
    click(pg, '#ppo-pass-play', 150); s = st(pg)
    check(f'{tag} part 6: the same button pauses', not s['passing'])
    # owner b9, item 4: the Scenarios and the Settings show their end at once or within about 1.5 s
    t0 = time.time(); click(pg, '#ppo-app .modes [data-mode="sandbox"]:not([data-flow=free])', 60); wait_run(pg); dt = time.time() - t0
    check(f'{tag} owner b9 item 4: the scenarios open on their end within about 1.5 s', st(pg)['mode'] == 'sandbox' and dt <= 2.0, f'{dt:.1f} s')
    t0 = time.time(); click(pg, '#ppo-rail [data-preset="3"]', 60); r0 = st(pg)['running']; wait_run(pg); dt = time.time() - t0
    check(f'{tag} owner b9 item 4: a scenario plays (visible) and ends within about 1.5 s', r0 and 1.0 <= dt <= 2.0, f'{dt:.1f} s')
    y0 = pg.evaluate("()=>scrollY")
    t0 = time.time(); click(pg, '#ppo-ctl-beta-wrap button[data-v="0.2"]', 60); r0 = st(pg)['running']; wait_run(pg); dt = time.time() - t0
    check(f'{tag} owner b9 item 4: a Settings change plays and ends within about 1.5 s, and does not scroll the page (R3.2)', r0 and 1.0 <= dt <= 2.0 and pg.evaluate("()=>scrollY") == y0, f'{dt:.1f} s')
    pg.close()

with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in LANGS:
        for w in WIDTHS:
            walk(b, lang, w)
        if 1440 in WIDTHS: replay_rules(b, lang)
    b.close()
check('no page errors', not errs, '; '.join(errs[:4]))
print('\n'.join(notes))
print(f'\n{len(notes)} checks, {len(fails)} failed')
sys.exit(1 if fails else 0)
