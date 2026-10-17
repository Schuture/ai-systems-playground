"""Flow test for the chunked prefill lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH, light, plus one dark run:
- no page errors and no horizontal overflow at any step;
- the header: the catalog's question, “Fidelity: Toy run”, the outcome line; the fidelity and primer popovers follow the writing rules;
- the phone first screen (m2 template, R7.3): the parts' bar with Next, and the users' row of the timeline;
- the prediction gate (R3.8): Next disabled until a prediction; before part 4 the chunk size stays “none” (control hidden, note shown) and
  the answer (127.2 ms, 93.90 ms and 1,502 ms with 2K-token chunks) appears nowhere, in text or tooltips; “Saved …” after a pick (R2.5);
- part 3: the stall (1,494 ms, set by arithmetic) with its replay: play and pause, the speed control (time-based, Medium by default, the
  card's and the stage's controls in step), “Next step”, the scrubber, the first-token label hidden until the replay reaches it (R3.9);
- part 4: the learner's own chunk size, then the verdict with the right answer, then why (R3.8, R3.10); ▲▼ only after the learner's own
  change (R6.6); the result on screen after the click at 1440×1000 (R3.10); a language switch keeps the part, the chunk and the prediction;
- part 5: the result shows from the start (2K: too big, try smaller); 256-token chunks (16.63 ms, 1,714 ms, 128 steps, the first chunk
  rides free, then “make it smaller, down to 32”); Next waits for a chunk too small to help; 32-token chunks (product owner, 29 Sep): the
  users' wait stays close to a usual step, the first token at about 13 s against 1.5 s, 1,024 steps each reading the weights again;
  the timeline stops at its right edge and says there when the first token comes; no bracket, no thick line and their legend entries
  gone; “why is a chunk that just rides free the best” opens and stays open across chunk changes; the deep dive opens and closes (✓);
- part 6: three misconceptions, the button to the scenarios (R3.12) and the next-lab footer (R3.11);
- Scenarios: four, single choice, pressed state follows the settings (R1.8), the URL carries the setup and a fresh page opens it
  (hash round-trip), a change under Settings does not move the page (R3.2);
- the challenge by the card's button: only the chunk size in the card, Settings locked with a note (R4.2), the goal in full (R6.8), live
  progress red over 50 ms (R4.3), the three tiers (1K: not met, hint, “show all options”; 256: met, but 512 gives an earlier first token;
  512: met and the earliest, 1K would be 64.45 ms), the comparison chart with eight sizes, green rows for every chunk from 1K down (R4.4, R5.4), baseline →
  effect → reason and a caveat (R5.5, R6.5), the capability card (R3.16); a language switch keeps the answer;
- the mini result bar at 390 once the timeline scrolls away, with a short status;
- the writing rules on every visible text node and tooltip: no “·” between clauses (R6.14), at most one colon per Chinese sentence (R6.7),
  a space between Chinese and Latin letters or digits, full-width punctuation in Chinese, no “in this toy” / 玩具 (the fidelity name
  “Toy run” / “玩具实跑” is the kit's), 起点 / starting point, 控件 / Controls, 亿 / 万, no sentence about the page itself (R3.4), no
  number over 6 digits outside the math, the drill's answer and tooltips (R6.18), colours only for their data (R5.6).
Screenshots: shots/chunked-prefill/.
Run: python3 tests/chunked-prefill_flow.py"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/chunked-prefill.html', '--only', 'chunked-prefill'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-chunked-prefill.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'chunked-prefill.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-chunked-prefill.html')
SH = P / 'shots' / 'chunked-prefill'; SH.mkdir(parents=True, exist_ok=True)
errs, fails, n = [], [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


Z = lambda lang, en, zh: zh if lang == 'zh' else en
NB = lambda s: s.replace(' ', ' ').replace('‑', '-').replace('\xa0', ' ')


def page(b, w, h, lang, dark=False, drawers=True, hash_='chunked-prefill', reduced=False):
    ctx = b.new_context(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light', reduced_motion='reduce' if reduced else 'no-preference')
    pg = ctx.new_page()
    if drawers: open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__ck.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0&&getComputedStyle(e).visibility!=='hidden'}", sel)
text = lambda pg, sel: NB(pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText:''}", sel))
click = lambda pg, sel: pg.evaluate("s=>document.querySelector(s).click()", sel)


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__ck||!window.__ck.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def go_part(pg, i):
    click(pg, f'#ck-rail [data-go="{i}"]'); pg.wait_for_timeout(250); settle(pg)


def nxt(pg):
    click(pg, '#ck-rail .step.current [data-nav="next"]'); pg.wait_for_timeout(250); settle(pg)


def chunk(pg, where, c):
    click(pg, f'{where} .seg-ctl[data-key="C"] button[data-v="{c}"]'); pg.wait_for_timeout(250); settle(pg)


def slide(pg, sel, v, change=True):
    pg.evaluate(f"""()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}}));
      {"i.dispatchEvent(new Event('change',{bubbles:true}));" if change else ''}}}"""); pg.wait_for_timeout(250); settle(pg)


def mode(pg, m):
    click(pg, f'#ck-app .modes button[data-mode="{m}"]'); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def all_text(pg):
    """every visible text of the lab and the mini result bar, plus every tooltip string (they can be read on hover)"""
    return NB(pg.evaluate("""()=>{const out=[]; for(const s of ['#ck-app','#dock']){const e=document.querySelector(s); if(e&&e.offsetParent!==null) out.push(e.innerText);}
      document.querySelectorAll('#ck-app [data-tipx], #dock [data-tipx]').forEach(e=>out.push(e.getAttribute('data-tipx'))); return out.join('\\n')}"""))


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy”'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/Locked in|已锁定/,'“Locked in”'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万 instead of K, M, B (R6.18)'],[/个权重/,'“个权重” (R6.17)'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],[/\\ba 8/,'“a 8…” (an)'],
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour (R5.7)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip')||!e.getClientRects().length) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); for(const [re,name] of bad) if(re.test(t)) out.push(name+' (tooltip): '+t.slice(0,90));});
 return out;}"""
SPACING = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), bad=/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>%][\\u4e00-\\u9fff]|[）】][ ][\\u4e00-\\u9fff]/;
 const shown=e=>e&&!e.closest('[data-endnotes],pre,code,svg,.mathbody,#tip')&&e.getClientRects().length;
 const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){if(!shown(n.parentElement)) continue; const t=n.textContent.replace(/\\s+/g,' '); const m=t.match(new RegExp(bad.source,'g')); if(m) out.add(m.join('|')+' in: '+t.trim().slice(0,80));}
 root.querySelectorAll('a,.def,b,.t,em,code,button.linkbtn').forEach(e=>{if(!shown(e)||getComputedStyle(e).display!=='inline'&&!e.matches('button.linkbtn')) return;
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
 root.querySelectorAll('p,li,button,label,div,span,h1,h2,h3,summary').forEach(e=>{if(e.closest('[data-endnotes],pre,code,svg,.mathbody')||!e.getClientRects().length) return; if(e.querySelector('p,li,div,ul,ol,h1,h2,h3,table')) return;
   if(![...e.childNodes].some(c=>c.nodeType===3&&c.textContent.trim())) return; texts.push(e.textContent);});
 root.querySelectorAll('[data-tipx]').forEach(e=>texts.push(e.getAttribute('data-tipx')));
 for(const t of texts){const sents=t.replace(/\\s+/g,' ').split(/(?<=[。？！])|(?<=[.?!])\\s+/);
   for(const x of sents){const k=(x.match(/：|: /g)||[]).length; if(k>=2) out.add(x.trim().slice(0,160));}}
 return [...out];}"""
BIGNUM = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #ck-mathbody, .num-big, #ck-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
# blue (weights) and amber (KV cache) only on the elements that mean them; red only for a missed goal (R5.6)
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom'));
 const ok={[blue]:/ck-bw|sw w\\b/,[amber]:/ck-bk|sw kv\\b/,[red]:/status bad|badline|badc|ck-over|ck-bad|warn|ex-res no\\b/};
 const out=[]; for(const e of document.querySelectorAll('#ck-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','borderLeftColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='borderLeftColor'&&s.borderLeftWidth==='0px')&&!(p==='color'&&!e.textContent.trim())&&!(p==='stroke'&&s.stroke==='none')) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
rule_hits = []


def rules(pg, where, lang):
    for sel in ('#ck-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':   # R6.7 is a rule for Chinese sentences
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
        if lang == 'zh':
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def badges(pg): return pg.evaluate("()=>[...document.querySelectorAll('#ck-metrics .metric .d')].map(d=>d.textContent).filter(Boolean)")
def card(pg, k): return NB(pg.evaluate(f"()=>document.querySelector('#ck-metrics [data-k=\"{k}\"] .v').textContent"))


def obs_on_screen(pg, sel='#ck-rail [data-obs]'):
    return pg.evaluate("""(s)=>{const b=document.querySelector(s); if(!b||b.hidden) return null; const r=b.getBoundingClientRect(), top=document.querySelector('.topbar').getBoundingClientRect().bottom;
      return {top:Math.round(r.top), bottom:Math.round(r.bottom), vh:innerHeight, ok:r.top>=top-2&&r.top<innerHeight-40}}""", sel)


ANSWER = ('127.2', '1,502', '93.90')   # the prediction's case, 2K-token chunks: longest wait, first token, TPOT during the prefill


def run_guided(b, w, h, lang, dark=False):
    tag = f'{"dark_" if dark else ""}{w}_{lang}'
    pg = page(b, w, h, lang, dark)
    phone = w <= 760
    title = text(pg, '#ck-app .labtitle')
    check(f'{tag} title (the catalog question)', title == Z(lang, 'Why does one huge prompt slow everyone down?', '一个超长 prompt 为什么会拖慢所有人？'), title)
    check(f'{tag} no fidelity tag on top (owner, 2026-10-11 19:28 UTC)', not any(w in text(pg, '#ck-app .labmeta') for w in ('Fidelity', '保真度')), text(pg, '#ck-app .labmeta'))
    check(f'{tag} outcome line under the title', text(pg, '#ck-app .lab-outcome').startswith(Z(lang, 'You will be able to:', '学完你能：')))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#ck-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        click(pg, f'#ck-app .labhead [data-pop="{pid}"]'); pg.wait_for_timeout(200)
        check(f'{tag} popover {pid} opens', pg.evaluate(f"()=>{{const p=document.getElementById('{pid}'); return !!p&&p.getClientRects().length>0}}"))
        rules(pg, f'{tag} popover {pid}', lang)
        click(pg, f'#ck-app .labhead [data-pop="{pid}"]'); pg.wait_for_timeout(150)
    # part 1: the mission, the stall on the stage
    st = state(pg)
    check(f'{tag} part 1: the mission, 16 users at 4K, a 32K-token prompt, no chunking', (st['U'], st['ctx'], st['N'], st['C'], st['step']) == (16, 4096, 32768, 0, 0), st)
    lg, ft = text(pg, '#ck-lg em'), text(pg, '#ck-arr .ck-arr-f')
    check(f'{tag} part 1: the timeline marks the longest wait and the first token (1,494 ms)', '1,494 ms' in lg and '1,494 ms' in ft, (lg, ft))
    check(f'{tag} part 1: the usual step, 11.74 ms, on the wait card; the mission says the stall in rounded seconds (R6.20)', '11.74 ms' in text(pg, '#ck-metrics') and '1.5 s' in text(pg, '#ck-rail'), text(pg, '#ck-rail')[:300])
    an = text(pg, '#ck-anat')
    check(f'{tag} part 1: above the timeline, what the one step holds: the users’ tokens and the whole prompt, one pass (Batch 3 review)', Z(lang, 'The one step', '这一步') in an and Z(lang, '16 users × 1 token', '16 个用户各 1 个 token') in an and '32K' in an, an)
    check(f'{tag} part 1: the cards: 1,494 ms wait, 1,494 ms first token, 1 step', card(pg, 'wait') == '1,494 ms' and card(pg, 'ttft') == '1,494 ms' and card(pg, 'steps') == '1', [card(pg, k) for k in ('wait', 'tpot', 'ttft', 'steps')])
    check(f'{tag} part 1: the scope line above the cards (R6.13)', text(pg, '#ck-scope').startswith(Z(lang, 'Estimates for one GPU', '单卡估算')))
    check(f'{tag} part 1: no overflow', overflow(pg) <= 0, overflow(pg))
    if phone:
        fs = pg.evaluate("""()=>{const pb=document.querySelector('#ck-rail .pbar'), nx=pb&&pb.querySelector('[data-nav="next"]'), tr=document.getElementById('ck-tu'), st=document.getElementById('ck-stage');
          const r=e=>e?e.getBoundingClientRect():null, a=r(pb), b=r(nx), c=r(tr), d=r(st);
          return {bar:!!a&&a.top<innerHeight&&a.height>0, next:!!b&&b.bottom<=innerHeight&&b.height>0, track:c?Math.round(c.bottom):null, stage:d?Math.round(d.top):null, vh:innerHeight}}""")
        check(f'{tag} phone first screen: the parts’ bar with Next, and the users’ row of the timeline (R7.3)', fs['bar'] and fs['next'] and fs['track'] is not None and fs['track'] <= fs['vh'] and fs['stage'] < fs['vh'] - 200, fs)
        shot(pg, f'{tag}_first_screen')
    shot(pg, f'{tag}_p1', full=phone)
    rules(pg, f'{tag} part 1', lang)
    ans_absent = lambda: not any(a in all_text(pg) for a in ANSWER)
    check(f'{tag} part 1: the prediction’s answer (127.2 ms, 1,502 ms, 93.90 ms) appears nowhere yet', ans_absent(), [a for a in ANSWER if a in all_text(pg)])
    # part 2: the gate
    nxt(pg)
    st = state(pg)
    check(f'{tag} part 2 reached', st['step'] == 1, st)
    nd = pg.evaluate("()=>[...document.querySelectorAll('#ck-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
    check(f'{tag} part 2: every Next is disabled before a prediction', nd and all(nd), nd)
    check(f'{tag} part 2: the chunk size is fixed at “none” (note shown, control hidden)', vis(pg, '#ck-Clock') and not vis(pg, '#ck-ctl-C-wrap') and st['C'] == 0)
    check(f'{tag} part 2: the answer appears nowhere (text and tooltips)', ans_absent())
    click(pg, '#ck-rail [data-pred="c"]'); pg.wait_for_timeout(200)
    check(f'{tag} part 2: “Saved … you can still change it”', Z(lang, 'Saved. You can still change it before part 4 shows the result.', '已记下。第 4 部分给出结果之前，还可以改。') in text(pg, '#ck-rail .step.current'))
    check(f'{tag} part 2: the pick shows as pressed with a radio dot', pg.evaluate("()=>{const b=document.querySelector('#ck-rail [data-pred=\"c\"]'); return b.getAttribute('aria-pressed')==='true'&&getComputedStyle(b,'::before').content!=='none'}"))
    nd = pg.evaluate("()=>[...document.querySelectorAll('#ck-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
    check(f'{tag} part 2: Next enabled after a prediction', nd and not any(nd), nd)
    shot(pg, f'{tag}_p2', full=phone)
    rules(pg, f'{tag} part 2', lang)
    # part 3: the stall, with its replay
    nxt(pg)
    o = text(pg, '#ck-rail [data-obs]')
    check(f'{tag} part 3: the stall in words: one step of about 1.5 s, over 100 times a usual step; the exact 1,494 ms on the card (R6.20)', '1.5 s' in o and Z(lang, 'over 100 times', '一百多倍') in o and card(pg, 'wait') == '1,494 ms', o[:240])
    check(f'{tag} part 3: the step’s two limits: arithmetic sets the time (tag on the arithmetic row)', pg.evaluate("()=>{const r=document.querySelector('.ck-ph[data-ph=\"s\"] .ck-row[data-k=\"c\"]'); return r.classList.contains('ck-set')&&!r.querySelector('.ck-settag').hidden}"))
    check(f'{tag} part 3: still no answer anywhere', ans_absent())
    check(f'{tag} part 3: no ▲▼ badge on entering a part', not badges(pg), badges(pg))
    click(pg, '#ck-rail #ck-ic-play [data-act="play"]'); pg.wait_for_timeout(700)
    pl = state(pg)['play']
    check(f'{tag} part 3: Replay runs (a playhead and a clock line)', pl['playing'] and pl['t'] is not None and vis(pg, '#ck-tu .ck-head') and text(pg, '#ck-rail #ck-ic-play [data-act="play"]').startswith(Z(lang, '❚❚ Pause', '❚❚ 暂停')), pl)
    if w == 1440 and lang == 'en' and not dark: shot(pg, f'{tag}_p3_playing')
    click(pg, '#ck-rail #ck-ic-play [data-act="play"]'); pg.wait_for_timeout(150)
    t1 = state(pg)['play']; pg.wait_for_timeout(400); t2 = state(pg)['play']
    check(f'{tag} part 3: Pause stops the replay where it is', not t1['playing'] and t1['t'] is not None and t1['t'] == t2['t'], (t1, t2))
    shot(pg, f'{tag}_p3', full=phone)
    rules(pg, f'{tag} part 3', lang)
    slide(pg, '#ck-scrub', 1000)
    # part 4: the learner's own chunk size, the reveal
    nxt(pg)
    st = state(pg)
    check(f'{tag} part 4 reached, still no chunking, Next waits for the action', st['step'] == 3 and st['C'] == 0 and pg.evaluate("()=>document.querySelector('#ck-rail .step.current [data-nav=\"next\"]').disabled"), st)
    check(f'{tag} part 4: no ▲▼ badge on entering a part', not badges(pg), badges(pg))
    chunk(pg, '#ck-ic-C', 2048)
    o = text(pg, '#ck-rail [data-obs]')
    ra = Z(lang, 'the right answer is', '正确答案是')
    check(f'{tag} part 4: the result first, then the verdict with the right answer, then why (R3.8, R3.10)',
          o.find('130 ms') > -1 and o.find(ra) > o.find('130 ms') and o.find(Z(lang, 'Why:', '原因：')) > o.find(ra)
          and Z(lang, 'from about 1.5 s to about 130 ms', '从约 1.5 s 降到约 130 ms') in o, o[:400])
    check(f'{tag} part 4: the first token barely later, and why in words: each extra step computes the users’ tokens once more', Z(lang, 'barely later', '几乎没有变晚') in o and Z(lang, 'computes the users’ tokens once more', '再算一遍用户的 token') in o, o[-300:])
    check(f'{tag} part 4: the dashed box spans both rows at the longest step, and the legend names it', vis(pg, '#ck-band') and Z(lang, 'dashed box: one step', '虚线框：一步') in text(pg, '#ck-legend'), text(pg, '#ck-legend'))
    check(f'{tag} part 4: above the timeline, one step = the users’ tokens + one 2K chunk, one pass', Z(lang, 'one 2K chunk', '一个 2K 的 chunk') in text(pg, '#ck-anat') and Z(lang, 'one pass', '模型一起算一遍') in text(pg, '#ck-anat'), text(pg, '#ck-anat'))
    check(f'{tag} part 4: the cards: 127.2 ms, 93.90 ms, 1,502 ms, 16 steps', [card(pg, k) for k in ('wait', 'tpot', 'ttft', 'steps')] == ['127.2 ms', '93.90 ms', '1,502 ms', '16'], [card(pg, k) for k in ('wait', 'tpot', 'ttft', 'steps')])
    check(f'{tag} part 4: a ▼ badge after the learner’s own change', any('▼' in x for x in badges(pg)), badges(pg))
    check(f'{tag} part 4: the timeline shows 16 chunks', pg.evaluate("()=>document.querySelectorAll('#ck-tp .ck-seg').length") == 16)
    if w == 1440 and not dark:
        pg.set_viewport_size({'width': 1440, 'height': 1000}); pg.wait_for_timeout(200)
        pg.evaluate("()=>window.scrollTo(0,0)"); chunk(pg, '#ck-ic-C', 1024); oc = obs_on_screen(pg)
        check(f'{tag} part 4: after the click the result is on screen at 1440×1000 (R3.10)', oc and oc['ok'], oc)
        chunk(pg, '#ck-ic-C', 2048)
        pg.set_viewport_size({'width': w, 'height': h}); pg.wait_for_timeout(200)
    shot(pg, f'{tag}_p4', full=phone)
    rules(pg, f'{tag} part 4', lang)
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(350)
    st2 = state(pg)
    check(f'{tag} part 4: a language switch keeps the part, the chunk size and the prediction', st2['step'] == 3 and st2['C'] == 2048 and st2['pred'] == 'c' and vis(pg, '#ck-rail [data-obs]'), st2)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(350)
    # part 5: how small
    nxt(pg)
    st = state(pg)
    check(f'{tag} part 5: starts at 2K-token chunks; Next waits for a chunk that is too small', st['step'] == 4 and st['C'] == 2048 and pg.evaluate("()=>document.querySelector('#ck-rail .step.current [data-nav=\"next\"]').disabled"), st)
    check(f'{tag} part 5: the second box shows the first chunk’s step', Z(lang, 'The first chunk’s step', '第一个 chunk 的那一步') in text(pg, '.ck-ph[data-ph="s"] .ck-ph-h'))
    o2k = text(pg, '#ck-rail [data-obs]')
    check(f'{tag} part 5: the result shows from the start: 2K-token chunks are too big to ride free, try a smaller one', Z(lang, 'too big to ride free', '搭不了便车') in o2k and Z(lang, 'Try a smaller chunk', '试试更小的 chunk') in o2k, o2k[:300])
    chunk(pg, '#ck-ic-C', 256)
    o = text(pg, '#ck-rail [data-obs]')
    check(f'{tag} part 5: 256 is not too small yet: Next still waits; the card asks to go down to 32', pg.evaluate("()=>document.querySelector('#ck-rail .step.current [data-nav=\"next\"]').disabled") and Z(lang, 'down to 32 tokens', '一直切到 32 个 token') in o, o[-200:])
    check(f'{tag} part 5: 256-token chunks in words: about 17 ms, close to a usual step, the first token noticeably later; 16.63 ms, 1,714 ms and 128 steps on the cards',
          '17 ms' in o and Z(lang, 'close to a usual step', '和平时一步差不多') in o and Z(lang, 'noticeably later', '晚了一些') in o and Z(lang, 'barely gets longer', '几乎没变长') in o
          and [card(pg, k) for k in ('wait', 'ttft', 'steps')] == ['16.63 ms', '1,714 ms', '128'], o[:400])
    check(f'{tag} part 5: the cost of small chunks: the number of steps, each reading all the weights', Z(lang, 'The price of small chunks is the number of steps', 'chunk 太小的代价是步数') in o, o[-260:])
    # the far end: 32-token chunks (or “Show me”)
    if dark:
        click(pg, '#ck-rail [data-nav="showme"]'); pg.wait_for_timeout(300); settle(pg)
        check(f'{tag} part 5: “Show me” sets 32-token chunks', state(pg)['C'] == 32)
    else:
        chunk(pg, '#ck-ic-C', 32)
    o = text(pg, '#ck-rail [data-obs]')
    check(f'{tag} part 5: 32-token chunks in words: the wait close to a usual step, the first token at about 13 s against 1.5 s, 1,024 steps each reading the weights again',
          Z(lang, 'still wait about as long as in a usual step', '用户的等待仍和平时一步差不多') in o and '13 s' in o and '1.5 s' in o and '1,024' in o and Z(lang, 'reads all the weights again', '把权重完整读一遍') in o
          and [card(pg, k) for k in ('wait', 'ttft', 'steps')] == ['13.88 ms', '13.12 s', '1,024'], (o[:500], [card(pg, k) for k in ('wait', 'ttft', 'steps')]))
    check(f'{tag} part 5: a chunk too small to help opens Next', not pg.evaluate("()=>document.querySelector('#ck-rail .step.current [data-nav=\"next\"]').disabled"))
    ft = text(pg, '#ck-arr .ck-arr-f')
    check(f'{tag} part 5: the timeline ends at its right edge and says there when the first token comes', ft == Z(lang, 'first token at 13.12 s, off the chart ▸', '13.12 s 才出首 token，在图外 ▸') and pg.evaluate("()=>document.getElementById('ck-tp').classList.contains('ck-offr')")
          and pg.evaluate("()=>{const f=document.querySelector('#ck-arr .ck-arr-f').getBoundingClientRect(), t=document.getElementById('ck-tp').getBoundingClientRect(); return f.right<=t.right+1&&f.left>=t.left}"), ft)
    marks = pg.evaluate("()=>({first:!!document.querySelector('#ck-mkp .ck-first'), lg:document.getElementById('ck-lg').hidden, over:[...document.querySelectorAll('#ck-mkp .ck-seg, #ck-mku .ck-tick')].filter(e=>parseFloat(e.style.left)>=99.999).length, leg:[...document.querySelectorAll('#ck-legend > span')].map(s=>s.innerText)})")
    check(f'{tag} part 5: past the edge nothing is drawn; no thick line, no bracket, and the legend drops them',
          not marks['first'] and marks['lg'] and marks['over'] == 0 and not any(Z(lang, 'thick line', '粗线') in x or Z(lang, 'bracket', '括号') in x for x in marks['leg']), marks)
    check(f'{tag} part 5: the clock says the first token comes past the right edge', Z(lang, 'past the chart’s right edge', '在图的右边之外') in text(pg, '#ck-clock'), text(pg, '#ck-clock'))
    check(f'{tag} part 5: the “why” for the trade-off is closed at first', vis(pg, '#ck-rail details.ck-why summary') and not pg.evaluate("()=>document.querySelector('#ck-rail details.ck-why').open"))
    click(pg, '#ck-rail details.ck-why summary'); pg.wait_for_timeout(200)
    shot(pg, f'{tag}_p5_small', full=phone)
    rules(pg, f'{tag} part 5 at 32', lang)
    chunk(pg, '#ck-ic-C', 256)
    wy = text(pg, '#ck-rail details.ck-why')
    check(f'{tag} part 5: the “why” stays open across chunk changes and names both ends and the middle', pg.evaluate("()=>document.querySelector('#ck-rail details.ck-why').open") and Z(lang, 'Neither a large nor a tiny chunk works well', 'chunk 太大、太小都不好') in wy and '吃亏' not in wy and Z(lang, 'like 256 tokens here', '这里是 256 个 token') in wy, wy[:300])
    click(pg, '#ck-rail [data-act="deep"]'); pg.wait_for_timeout(300)
    check(f'{tag} part 5: the deep dive opens on the stage', vis(pg, '#ck-deep svg') and not vis(pg, '#ck-lim') and Z(lang, 'Deep dive', '深入') in text(pg, '#ck-deep'))
    dp = text(pg, '#ck-deep')
    check(f'{tag} part 5: the deep dive: the size up to which a chunk rides free is marked on the chart; the note says why in words (R6.20)', Z(lang, 'free: up to 394 tokens', '白搭车：不超过 394 个 token') in pg.evaluate("()=>document.querySelector('#ck-deep svg').textContent") and '394' not in dp.split(Z(lang, 'thin line', '细线'))[-1] and Z(lang, 'shaded', '阴影') in dp, dp[-300:])
    shot(pg, f'{tag}_p5_deep', full=phone)
    rules(pg, f'{tag} part 5 deep dive', lang)
    click(pg, '#ck-deep [data-act="deep-close"]'); pg.wait_for_timeout(250)
    check(f'{tag} part 5: back to the two limits, the deep dive marked as seen', vis(pg, '#ck-lim') and not vis(pg, '#ck-deep') and '✓' in text(pg, '#ck-rail [data-act="deep"]'))
    shot(pg, f'{tag}_p5', full=phone)
    rules(pg, f'{tag} part 5', lang)
    # part 6
    nxt(pg)
    t6 = text(pg, '#ck-rail .step.current')
    check(f'{tag} part 6: three misconceptions checked, and the button to the scenarios (R3.12)', t6.count('✗') == 3 and vis(pg, '#ck-rail [data-mode-go="sandbox"]'), t6[:200])
    nl = pg.evaluate("""()=>{const shown=x=>!!x&&getComputedStyle(x).display!=='none'&&x.getBoundingClientRect().height>0; const r=document.querySelector('#ck-rail .next-lab'), e=document.querySelector('#view-dyn .next-lab-end .next-lab');
      return innerWidth<=760 ? (shown(e)||shown(r)) : shown(r)}""")
    check(f'{tag} part 6: the next-lab footer (R3.11)', nl)
    shot(pg, f'{tag}_p6', full=phone)
    rules(pg, f'{tag} part 6', lang)
    click(pg, '#ck-rail [data-mode-go="sandbox"]'); pg.wait_for_timeout(350); settle(pg)
    check(f'{tag} “See the scenarios” opens Scenarios', state(pg)['mode'] == 'sandbox')
    check(f'{tag} no overflow at the end of the tour', overflow(pg) <= 0, overflow(pg))
    pg.context.close()


def run_modes(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    phone = w <= 760
    mode(pg, 'sandbox')
    presets = pg.evaluate("()=>[...document.querySelectorAll('#ck-rail [data-preset]')].length")
    check(f'{tag} scenarios: four, with the lead-in that says what to pick and what to watch (G1, the owner 2026-10-08)', presets == 4 and Z(lang, 'Pick a scenario and see how long the users wait at most once the prompt is cut into chunks.', '选一个场景，看看把 prompt 切成 chunk 后，用户最长要等多久。') in text(pg, '#ck-rail .railcard') and Z(lang, 'do not stack', '不会叠加') not in text(pg, '#ck-rail .railcard'))
    chunk(pg, '#ck-sb-C', 2048)
    for i in range(presets):
        click(pg, f'#ck-rail [data-preset="{i}"]'); pg.wait_for_timeout(300); settle(pg)
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#ck-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        ins = text(pg, '#ck-insight')
        check(f'{tag} scenario {i}: pressed alone, with its result', pressed.count('true') == 1 and pressed[i] == 'true' and len(ins) > 80, (pressed, ins[:120]))
        check(f'{tag} scenario {i}: no overflow', overflow(pg) <= 0, overflow(pg))
        rules(pg, f'{tag} scenario {i}', lang)
        if i == 1:
            check(f'{tag} scenario upload: 32 users, 120K tokens; attention grows with the square of the prompt', Z(lang, 'grows with the square of the prompt', '随 prompt 长度的平方增长') in ins, ins[:300])
            shot(pg, f'{tag}_scenario_upload', full=phone)
        if i == 2:
            check(f'{tag} scenario chat: a short prompt costs little either way (one chunk = no chunking)', Z(lang, 'A short prompt costs little either way', '短 prompt 切不切都不贵') in ins, ins[:300])
    hsh = pg.evaluate("()=>location.hash")
    check(f'{tag} scenarios: the URL carries the setup', hsh == '#chunked-prefill~8_8192_8192_2048', hsh)
    shot(pg, f'{tag}_scenario_code', full=phone)
    pg2 = page(b, w, h, lang, hash_='chunked-prefill~32_4096_122880_2048')
    st = state(pg2)
    check(f'{tag} hash round-trip: Scenarios with 32 users at 4K, a 120K-token prompt, 2K chunks, “A long upload” pressed', st['mode'] == 'sandbox' and (st['U'], st['ctx'], st['N'], st['C']) == (32, 4096, 122880, 2048)
          and pg2.evaluate("()=>document.querySelector('#ck-rail [data-preset=\"1\"]').getAttribute('aria-pressed')") == 'true', st)
    pg2.context.close()
    # a change under Settings: no scenario matches any more, and the page does not move (R1.8, R3.2)
    pg.evaluate("()=>document.querySelector('#ck-app [data-flow-nav] button[data-flow=free]').click()"); pg.wait_for_timeout(300)   # settings are changed in 自由调参 (owner, 2026-10-11)
    y0 = pg.evaluate("()=>scrollY")
    slide(pg, '#ck-ctl-U-in', 2, change=False)
    pressed = pg.evaluate("()=>[...document.querySelectorAll('#ck-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: after a change under Settings no scenario is pressed, and the card says so', 'true' not in pressed and Z(lang, 'match none', '都不一样') in text(pg, '#ck-insight'), pressed)
    if not phone:
        check(f'{tag} a change under Settings does not scroll the page', abs(pg.evaluate("()=>scrollY") - y0) < 2, (y0, pg.evaluate("()=>scrollY")))
    pg.evaluate("()=>document.querySelector('#ck-app [data-flow-nav] button[data-flow=scen]').click()"); pg.wait_for_timeout(300)
    # the challenge, by the card's button (R3.12)
    click(pg, '#ck-rail [data-mode-go="challenge"]'); pg.wait_for_timeout(350); settle(pg)
    st = state(pg)
    check(f'{tag} challenge: 32 users at 2K, a 16K-token document, no chunking to start (a setting the guided parts never show, R3.8)', st['mode'] == 'challenge' and (st['U'], st['ctx'], st['N'], st['C']) == (32, 2048, 16384, 0), st)
    check(f'{tag} challenge: Settings locked with a note, no grayed controls (R4.2)', vis(pg, '#ck-lock') and not vis(pg, '#ck-ctl-U') and not vis(pg, '#ck-ctl-N') and not vis(pg, '#ck-ctl-C-wrap'))
    cd = text(pg, '#ck-rail .railcard')
    check(f'{tag} challenge: the goal in full: model, GPU, users, context, document, target (R6.8)', all(x in cd for x in ['Llama-3.1-8B', 'H100 SXM', '32', '2K', '16K', '50 ms']), cd[:300])
    check(f'{tag} challenge: the capability card after the task', vis(pg, '#ck-rail [data-exits]'))
    check(f'{tag} challenge: no chart, no answer before a submit', not vis(pg, '#ck-ch-res .cmp') and '47.16' not in text(pg, '#ck-rail'))
    check(f'{tag} challenge: no overflow', overflow(pg) <= 0, overflow(pg))
    shot(pg, f'{tag}_challenge', full=phone)
    rules(pg, f'{tag} challenge', lang)
    # tier 3: 2K chunks break the goal
    chunk(pg, '#ck-ic-chC', 2048)
    check(f'{tag} challenge: live progress turns red over 50 ms (R4.3)', pg.evaluate("()=>document.getElementById('ck-ch-goal').classList.contains('ck-over')") and '92.16 ms' in text(pg, '#ck-ch-now'))
    click(pg, '#ck-rail [data-act="submit"]'); pg.wait_for_timeout(300); settle(pg)
    res = text(pg, '#ck-ch-res')
    check(f'{tag} challenge tier 3: not met, with a hint and “Show the answer” (R4.4)', res.startswith(Z(lang, 'Not yet', '还没达到')) and '92.16 ms' in res and Z(lang, 'Hint', '提示') in res and vis(pg, '#ck-ch-res [data-act="reveal"]') and not vis(pg, '#ck-ch-res .cmp')
          and text(pg, '#ck-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), res[:240])
    check(f'{tag} challenge: no done line in the capability card (R3.16, Batch 3 review)', not pg.evaluate("()=>!!document.querySelector('#ck-rail [data-exits] .ex-done')"))
    click(pg, '#ck-ch-res [data-act="reveal"]'); pg.wait_for_timeout(300)
    rows = pg.evaluate("()=>[...document.querySelectorAll('#ck-ch-res .cmp-row')].map(r=>({ok:r.classList.contains('ok'),me:r.classList.contains('me'),off:r.classList.contains('ck-off'),t:r.querySelector('.cmp-lab span').textContent}))")
    oks = [r['t'] for r in rows if r['ok']]
    check(f'{tag} challenge chart: eight sizes, green rows for 1K and every smaller chunk; the pick and the initial setup tagged; the bar past the scale marked',
          len(rows) == 8 and len(oks) == 6 and all(any(k in t for k in ('1K', '512', '256', '128', '64', '32')) for t in oks) and not any('2K' in t for t in oks) and sum(r['me'] for r in rows) == 1 and Z(lang, 'initial setup', '初始设置') in text(pg, '#ck-ch-res .cmp')
          and sum(r['off'] for r in rows) == 1, rows)
    # the owner's review of batch 8: a brief chart; with 2K picked, every row is a passing size, the pick or the initial setup, so none is left out
    check(f'{tag} challenge chart (brief, 2K picked): no row left out, no note', not vis(pg, '#ck-ch-res .cmp-note'), text(pg, '#ck-ch-res .cmp-note'))
    ttl = text(pg, '#ck-ch-res .cmp-title')
    check(f'{tag} challenge chart: the title names the quantity and the conditions, not “for each chunk size”', 'H100' in ttl and '16K' in ttl and Z(lang, 'for each chunk size', '每种 chunk') not in ttl, ttl)
    xt = pg.evaluate("()=>[...document.querySelectorAll('#view-dyn [data-tipx]')].filter(e=>!e.closest('[id$=\"-mathbody\"],[id$=\"-math\"]')).map(e=>e.getAttribute('data-tipx')).filter(t=>t.includes('精确值是')||t.includes('Exactly'))")
    check(f'{tag} no hover outside Show the math gives a number’s exact value (the owner’s clarification of 2026-10-07)', not xt, xt[:2])
    shot(pg, f'{tag}_challenge_reveal', full=phone)
    rules(pg, f'{tag} challenge reveal', lang)
    # tier 2: 512 meets the goal, but a larger chunk starts the answer sooner
    chunk(pg, '#ck-ic-chC', 512)
    res = text(pg, '#ck-ch-res')
    check(f'{tag} challenge tier 2: met, but a larger chunk starts the answer sooner', res.startswith(Z(lang, 'Goal met', '达成目标')) and '24.24 ms' in res and Z(lang, 'a larger chunk also stays within 50 ms', '更大的 chunk 也不超过 50 ms') in res, res[:300])
    # tier 1: 1K
    chunk(pg, '#ck-ic-chC', 1024)
    res = text(pg, '#ck-ch-res')
    check(f'{tag} challenge tier 1: 1K-token chunks, 47.16 ms, and the earliest first token', res.startswith(Z(lang, 'Goal met: with 1K-token chunks', '达成目标：切成 1K 个 token 的 chunk 时')) and '47.16 ms' in res and Z(lang, 'earliest first token', '最早的首 token') in res, res[:300])
    # with 1K picked the brief chart keeps the 6 passing sizes and the initial setup; 2K is left out, counted, and named in words with its wait on hover
    rows = pg.evaluate("()=>[...document.querySelectorAll('#ck-ch-res .cmp-row')].map(r=>({ok:r.classList.contains('ok'),me:r.classList.contains('me'),tag:!!r.querySelector('.cmp-tag'),t:r.querySelector('.cmp-lab span').textContent}))")
    tip2k = pg.evaluate("()=>[...document.querySelectorAll('#ck-ch-res .def')].map(d=>d.dataset.tipx||'').find(t=>t.includes('2K'))||''")
    check(f'{tag} challenge chart (brief, 1K picked): 7 rows, only passing sizes and the initial setup; the note counts the 1 left out; 2K’s 92.16 ms on hover',
          len(rows) == 7 and all(r['ok'] or r['me'] or r['tag'] for r in rows) and not any('2K' in r['t'] for r in rows)
          and text(pg, '#ck-ch-res .cmp-note') == Z(lang, 'The remaining chunk size goes over 50 ms and is not shown.', '其余 1 种 chunk 大小超过了 50 ms，图里没有列出。') and '92.16 ms' in tip2k.replace('\xa0', ' '), (rows, tip2k))
    check(f'{tag} challenge: baseline → effect → reason (a shorter document lets a larger chunk fit, said as the pattern, not against the guided parts’ document: G2, the owner 2026-10-08), and a specific caveat (R5.5, R6.5)', all(x in res for x in [Z(lang, 'Initial setup', '初始设置'), Z(lang, 'Why:', '原因：'), Z(lang, 'The shorter the document, the less its last chunk has to look back at', '文档越短，最后一个 chunk 要回看的前文就越少'), Z(lang, 'Note:', '注意：'), '1,056', 'max_num_batched_tokens'])
          and Z(lang, 'guided parts', '分步引导') not in res, res[-500:])
    check(f'{tag} challenge: the goal bracket on the timeline turns green', pg.evaluate("()=>document.getElementById('ck-lg').classList.contains('ck-good')"))
    shot(pg, f'{tag}_challenge_met', full=phone)
    rules(pg, f'{tag} challenge met', lang)
    for t in (0, 1):
        click(pg, f'#ck-rail [data-exits] [data-ex="tab"][data-i="{t}"]'); pg.wait_for_timeout(150)
        for v in pg.evaluate("()=>[...document.querySelectorAll('#ck-rail [data-exits] .ex-opts .choice')].map(b=>+b.dataset.v)"):
            click(pg, f'#ck-rail [data-exits] .ex-opts .choice[data-v="{v}"]'); pg.wait_for_timeout(150)
            rules(pg, f'{tag} capability card tab {t + 1}, option {v}', lang)
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} challenge: a language switch keeps 1K-token chunks and the submitted result', st['C'] == 1024 and st['chSubmitted'] and vis(pg, '#ck-ch-res .cmp'), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    if phone:
        mode(pg, 'guided'); go_part(pg, 0)
        pg.evaluate("()=>{const t=document.getElementById('ck-lim'); window.scrollTo(0, t.getBoundingClientRect().top+scrollY+80)}"); pg.wait_for_timeout(500)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, status:document.getElementById('dkStatus').textContent, read:document.getElementById('dkRead').textContent}}")
        check(f'{tag} mini result bar: shown once the timeline scrolls away, with a short status and the numbers', dk['shown'] and len(dk['status']) <= 14 and '1,494' in dk['read'], dk)
        shot(pg, f'{tag}_dock')
        rules(pg, f'{tag} dock', lang)
    pg.context.close()


def run_replay(b):
    """R3.9 in detail, at 1440 EN: time-based speeds, the two speed controls in step, Next step, the scrubber, the first-token label"""
    pg = page(b, 1440, 900, 'en')
    go_part(pg, 2)
    segs = pg.evaluate("()=>[...document.querySelectorAll('.seg-ctl[data-key=speed]')].map(g=>[...g.querySelectorAll('button')].map(x=>x.textContent+(x.getAttribute('aria-pressed')==='true'?'*':'')).join('/'))")
    check('replay: a Slow / Medium / Fast control in part 3 and on the stage, Medium pressed', segs == ['Slow/Medium*/Fast', 'Slow/Medium*/Fast'], segs)
    sp = pg.evaluate("()=>window.__ck.span({U:16,ctx:4096,N:32768})")
    W = sp['x1'] - sp['x0']

    def rate(v):
        click(pg, f'#ck-graph .seg-ctl[data-key=speed] button[data-v={v}]')
        slide(pg, '#ck-scrub', 0); click(pg, '#ck-graph [data-act=play]'); pg.wait_for_timeout(300)
        a = pg.evaluate("()=>[window.__ck.state().play.t, performance.now()]"); pg.wait_for_timeout(1500); z = pg.evaluate("()=>[window.__ck.state().play.t, performance.now()]")
        click(pg, '#ck-graph [data-act=play]'); pg.wait_for_timeout(100)
        return (z[0] - a[0]) / W / ((z[1] - a[1]) / 1000)
    rs = {v: rate(v) for v in ('slow', 'med', 'fast')}
    check('replay: the whole timeline takes about 48 / 24 / 8 s (time-based)', all(abs(rs[v] * s - 1) < 0.3 for v, s in (('slow', 48), ('med', 24), ('fast', 8))), {k: round(1 / v, 1) for k, v in rs.items()})
    same = pg.evaluate("()=>[...document.querySelectorAll('.seg-ctl[data-key=speed] button[aria-pressed=true]')].map(x=>x.dataset.v)")
    check('replay: the card’s and the stage’s speed controls stay in step', same == ['fast', 'fast'], same)
    tip = pg.evaluate("()=>document.querySelector('#ck-graph .ck-speed').dataset.tipx")
    check('replay: the speed tooltip is a sentence with the three durations', tip == 'Slow plays the whole timeline in 48 s, Medium in 24 s and Fast in 8 s.', tip)
    # Next step: the arrival, then the end of the prefill step, then the whole run
    slide(pg, '#ck-scrub', 1000)
    click(pg, '#ck-graph [data-act=stepfwd]'); pg.wait_for_timeout(150)
    c0 = text(pg, '#ck-clock')
    check('replay: Next step goes to the arrival: step 1 of 1, the users have waited 0 ms', state(pg)['play']['t'] == 0 and c0 == 'At 0 ms: step 1 of 1. The users have waited 0 ms since their last token.', c0)
    check('replay: the first-token label waits until the replay reaches it', not vis(pg, '#ck-arr .ck-arr-f'))
    click(pg, '#ck-graph [data-act=stepfwd]'); pg.wait_for_timeout(150)
    c1 = text(pg, '#ck-clock')
    check('replay: the next step ends the prefill: the first token at 1,494 ms, and the prompt decodes with the users', c1.startswith('At 1,494 ms: the prompt got its first token at 1,494 ms') and vis(pg, '#ck-arr .ck-arr-f'), c1)
    click(pg, '#ck-graph [data-act=stepfwd]'); pg.wait_for_timeout(150)
    check('replay: one more step shows the whole run again', state(pg)['play']['t'] is None and text(pg, '#ck-clock').startswith('The whole run'), text(pg, '#ck-clock'))
    # the scrubber: the wait so far, shaded, and the step shown in the second box
    slide(pg, '#ck-scrub', 500)
    t = state(pg)['play']['t']; c2 = text(pg, '#ck-clock')
    check('replay: the scrubber moves the moment shown; the clock says how long the users have waited', t is not None and abs(t - (sp['x0'] + 0.5 * W)) < W / 900 and 'The users have waited' in c2 and vis(pg, '#ck-wt') and 'Step 1 of 1, now' in text(pg, '.ck-ph[data-ph="s"] .ck-ph-h'), (t, c2))
    shot(pg, 'replay_scrub_1440_en')
    # with 2K-token chunks a Next step walks the 16 chunks
    go_part(pg, 3); chunk(pg, '#ck-ic-C', 2048)
    ts = []
    for _ in range(4):
        click(pg, '#ck-graph [data-act=stepfwd]'); pg.wait_for_timeout(120); ts.append(state(pg)['play']['t'])
    r2 = pg.evaluate("()=>window.__ck.run({U:16,ctx:4096,N:32768,C:2048})")
    check('replay: with 2K-token chunks, Next step goes from the arrival to the end of each chunk’s step', ts[0] == 0 and abs(ts[1] - r2['steps'][0]['end']) < 1e-9 and abs(ts[3] - r2['steps'][2]['end']) < 1e-9, ts)
    check('replay: while a chunk’s step runs, the second box shows that step', 'Step 3 of 16, now' in text(pg, '.ck-ph[data-ph="s"] .ck-ph-h') or 'Step 4 of 16, now' in text(pg, '.ck-ph[data-ph="s"] .ck-ph-h'), text(pg, '.ck-ph[data-ph="s"] .ck-ph-h'))
    shot(pg, 'replay_chunks_1440_en')
    pg.context.close()
    # reduced motion: Replay jumps to the whole run
    pg = page(b, 1440, 900, 'en', reduced=True)
    go_part(pg, 2); click(pg, '#ck-rail #ck-ic-play [data-act="play"]'); pg.wait_for_timeout(200)
    check('replay: with reduced motion, Replay shows the whole run at once', not state(pg)['play']['playing'] and state(pg)['play']['t'] is None)
    pg.context.close()


# G1 and G2, the owner's message of 2026-10-08 (reviews/owner-b9-2026-10-08.md): a lead-in above each group of options instead of
# “one at a time; they do not stack”, and no sentence that leans on a detail of an earlier part. Checked on the lab's source, both languages.
_G_SRC = (P / 'labs' / 'chunked-prefill.js').read_text()
for _b in ["一次只能选", "一次选一个", "一次选一种", "一次只选", "不会叠加", "do not stack", "One at a time", "pick one at a time", "Pick one at a time", "Try one change at a time", "一次试一种"]:
    check(f'G1: no “{_b}” left in the lab (EN and ZH)', _b not in _G_SRC)
for _s in ["Pick a scenario and see how long the users wait at most once the prompt is cut into chunks.", "选一个场景，看看把 prompt 切成 chunk 后，用户最长要等多久。"]:
    check(f'G1: the lead-in “{_s[:48]}” is in the lab', _s in _G_SRC)
for _s in ["The shorter the document, the less its last chunk has to look back at", "文档越短，最后一个 chunk 要回看的前文就越少", "Now 64 users decode instead, 4 times as many", "现在换成 64 个用户，是 16 个的 4 倍", "With a 32K-token prompt, the users’ longest wait between two tokens is about", "prompt 有 32K 个 token 时，用户两个 token 之间最长约等"]:
    check(f'G2: the rewritten sentence “{_s[:48]}” is in the lab', _s in _G_SRC)
for _s in ["half as long as the one in the guided parts", "分步引导里那份的一半", "twice as long as the 32K document in the guided parts", "分步引导里那份 32K 文档", "是原来的 4 倍，每人 4K 个 token 的上下文"]:
    check(f'G2: the sentence that leaned on an earlier part, “{_s[:48]}”, is gone', _s not in _G_SRC)

# polish r5 (B4, R7.12): entering the parts this round fitted, the way a reader does (Next, or the part's title while Next waits for an
# answer), at 1440 x 900 the stage fits without scrolling inside, with 30 px to spare in Chinese. Sampled for 6.6 s after entering, so a
# part that plays on entry (part 6 of prefix-caching sends its 8 requests in 5.3 s) fits all the way through; measured in standard mode (the page wrapped in a doctype, as it is published)
FIT = """p=>{const s=document.getElementById(p+'-stage'), c=document.querySelector('#'+p+'-rail .steps>.step.current');
  return {step:c?[...c.parentNode.children].indexOf(c)+1:0, spare:Math.round(parseFloat(getComputedStyle(s).maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight)),
          scroll:s.scrollHeight-s.clientHeight, top:Math.round(s.getBoundingClientRect().top)}}"""


def fit_next(pg, k):
    """into part k as a reader goes: Next when it is open, otherwise the part's title"""
    nb = pg.locator('#ck-rail .step.current [data-nav="next"]:not([disabled])')
    if nb.count(): nb.first.click()
    else: pg.evaluate(f"()=>document.querySelector('#ck-rail .steps>.step:nth-child({k}) .step-h').click()")


def fit_entry(b, lang, parts=(1, 2, 3, 4, 5, 6)):
    pg = page(b, 1440, 900, lang)
    for k in range(1, max(parts) + 1):
        if k > 1: fit_next(pg, k)
        if k not in parts:
            pg.wait_for_timeout(900); continue
        fs = []
        for _ in range(44):
            pg.wait_for_timeout(150); fs.append(pg.evaluate(FIT, 'ck'))
        f = {'step': fs[-1]['step'], 'spare': min(x['spare'] for x in fs), 'scroll': max(x['scroll'] for x in fs), 'top': fs[-1]['top']}
        check(f'{lang} 1440 part {k}: entering it, the stage fits without scrolling inside{", 30 px to spare" if lang == "zh" else ""} (B4, R7.12; polish r5)',
              f['step'] == k and f['scroll'] <= 0 and f['spare'] >= (30 if lang == 'zh' else 0), f)
    pg.context.close()


# polish r5 (R7.9): on a wide stage Show me, the replay and the Scenarios cards keep the stage at one height: the subtitle runs across the
# whole head in one line, and the legend keeps room for the second line the replay adds to it (“now”, “waiting”)
HOLD = """p=>{const s=document.getElementById(p+'-stage'); return s.scrollHeight+s.offsetHeight-s.clientHeight;}"""


def hold_ck(b, lang):
    pg = page(b, 1440, 900, lang)
    for k in (2, 3, 4):
        fit_next(pg, k); pg.wait_for_timeout(900)
    hs = [pg.evaluate(HOLD, 'ck')]
    pg.evaluate("()=>{const e=[...document.querySelectorAll('#ck-rail .step.current [data-nav=showme]')].find(e=>e.getClientRects().length&&!e.disabled); if(e) e.click();}")
    for _ in range(8):
        pg.wait_for_timeout(150); hs.append(pg.evaluate(HOLD, 'ck'))
    played = pg.evaluate("()=>{const e=[...document.querySelectorAll('#ck-stage [data-act=play]')].find(e=>e.getClientRects().length&&!e.disabled); if(e){e.click(); return true;} return false;}")
    for _ in range(20):
        pg.wait_for_timeout(150); hs.append(pg.evaluate(HOLD, 'ck'))
    check(f'{lang} 1440 part 4: Show me, then the replay, keep the stage at one height (R7.9; polish r5)', played and max(hs) - min(hs) <= 1, hs)
    pg.evaluate("()=>{const b=[...document.querySelectorAll('[data-flow-nav] button[data-flow=scen]')].find(x=>x.offsetParent); if(b) b.click();}"); pg.wait_for_timeout(800)
    n = pg.evaluate("()=>[...document.querySelectorAll('#ck-rail [data-preset]')].filter(e=>e.getClientRects().length).length")
    hs = [pg.evaluate(HOLD, 'ck')]
    for i in range(n):
        pg.evaluate("i=>[...document.querySelectorAll('#ck-rail [data-preset]')].filter(e=>e.getClientRects().length)[i].click()", i); pg.wait_for_timeout(700)
        hs.append(pg.evaluate(HOLD, 'ck'))
    check(f'{lang} 1440 Scenarios: picking each card keeps the stage at one height (R7.9; polish r5)', n >= 4 and max(hs) - min(hs) <= 1, hs)
    pg.context.close()

with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ('en', 'zh'):
        fit_entry(b, lang)
        hold_ck(b, lang)
    run_replay(b)
    for (w, h) in [(1440, 900), (390, 844)]:
        for lang in ('en', 'zh'):
            run_guided(b, w, h, lang)
            run_modes(b, w, h, lang)
    run_guided(b, 1440, 900, 'en', dark=True)
    for (w, h) in [(1280, 800), (900, 900), (720, 900)]:
        pg = page(b, w, h, 'en')
        for m in ('guided', 'sandbox', 'challenge'):
            mode(pg, m)
            check(f'{w} {m}: no overflow', overflow(pg) <= 0, overflow(pg))
        shot(pg, f'{w}_en_challenge')
        pg.context.close()
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
        _pg.evaluate("()=>{location.hash='chunked-prefill'}"); _pg.wait_for_timeout(500)
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


for hit in rule_hits[:80]:
    check('writing rules: ' + hit, False)
check('no page errors', not errs, errs[:5])
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:80] if fails else 'none')
sys.exit(1 if fails else 0)
