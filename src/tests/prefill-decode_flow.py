"""Flow test for the prefill-decode lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH, light, plus one dark run:
- no page errors and no horizontal overflow at any step;
- the phone first screen (m2 template, R7.3): the parts' bar with Next, and the timeline of the chart;
- the prediction gate (R3.8): Next disabled until a prediction; before it, the prompt length is fixed and the answer (266.7 ms and
  8.00 ms at 8K) appears nowhere, in text or tooltips; “Saved …” after a pick (R2.5); in part 3 the decode half stays hidden (still no
  8.00 ms) while the time to first token follows the prompt; part 4 gives the verdict with the right answer and the learner’s own run;
  the results say the trend in words with at most one rounded number, the exact times being read from the metric cards (R6.20);
- each part's action gate and result; ▲▼ badges only after the learner's own change (R6.6); the roofline deep dive opens and closes;
- guided → Scenarios → Challenge by the cards' buttons (R3.12); the scenarios' pressed state follows the settings (R1.8);
- the challenge: only the batch size in the card, Settings locked with a note (R4.2), the three feedback tiers, “show all options”,
  the comparison chart with green rows up to 46 users (R4.4, R5.4), and the capability card;
- the language switch keeps the part, the settings, the prediction and the challenge's answer; the hash round-trip in Scenarios;
- the mini result bar at 390 once the timeline scrolls away, with a short status;
- the result box on screen after each action at 1440×1000 (R3.10);
- the writing rules on every visible text node and tooltip: no “·” between clauses (R6.14), at most one colon per sentence (R6.7), a
  space between Chinese and Latin letters or digits, no “in this toy” / 玩具, 起点 / starting point, 控件 / Controls, 亿 / 万, no
  sentence about the page itself (R3.4), no number over 6 digits outside the math and tooltips (R6.18), colours only for their data;
- part 6, the experiment (twice the compute, or twice the bandwidth), at 1440×900 and 390×844 in EN and ZH: the prediction gate (Next
  disabled, the two doubled GPUs masked and their numbers nowhere until a pick), the GPU options and the load switch after it, the
  verdict (right in EN, wrong with the right answer in ZH), the result in words (R6.20), one time scale (the original's timeline does
  not move when another GPU is picked; compute × 2's prefill is half as long, bandwidth × 2's answer about half), the bars against the
  original's outlines, the cards' change against the original, a timeline row picks its GPU, the phone's first screen shows the three
  timelines and the rows that pick a GPU, and a pick never scrolls the control away; rapid clicks end in the last one; the language
  switch keeps the GPU, the load and the prediction; leaving the part resets them (Scenarios keep the real H100); reduced motion.
  Review r1: four options (“neither changes much” too); before the pick, the phone's first screen has a way in (a button, and a click on
  a masked row, bring the question on screen); the chat's first token zoomed in (compute × 2's prefill visibly half); the original's
  bars hatched, not dashed; one unit per load (the long document in s); the cards' change as “▼ −50% vs the original”; the phone's
  mini result bar names the load and gives the original's value; at 1440×1000 the card's Next stays on screen after an option.
The guided tour has 7 parts since part 6 was added (the old part 6, “What to remember”, is now part 7).
Screenshots: shots/prefill-decode/.
Run: python3 tests/prefill-decode_flow.py"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/prefill-decode.html', '--only', 'prefill-decode'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-prefill-decode.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'prefill-decode.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-prefill-decode.html')
SH = P / 'shots' / 'prefill-decode'; SH.mkdir(parents=True, exist_ok=True)
errs, fails, n = [], [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


Z = lambda lang, en, zh: zh if lang == 'zh' else en


def page(b, w, h, lang, dark=False, drawers=True, hash_='prefill-decode'):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light')
    if drawers: open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__pd.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText:''}", sel)


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__pd||!window.__pd.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def go_part(pg, i):
    pg.evaluate(f"()=>document.querySelector('#pd-rail [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(250); settle(pg)


def slide(pg, sel, v, change=True):
    pg.evaluate(f"""()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}}));
      {"i.dispatchEvent(new Event('change',{bubbles:true}));" if change else ''}}}"""); pg.wait_for_timeout(250); settle(pg)


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#pd-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def all_text(pg):
    """every visible text of the lab and the mini result bar, plus every tooltip string (they can be read on hover)"""
    return pg.evaluate("""()=>{const out=[]; for(const s of ['#pd-app','#dock']){const e=document.querySelector(s); if(e&&e.offsetParent!==null) out.push(e.innerText);}
      document.querySelectorAll('#pd-app [data-tipx], #dock [data-tipx]').forEach(e=>out.push(e.getAttribute('data-tipx'))); return out.join('\\n')}""")


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/in this toy\\b|在这个玩具里|\\btoy\\b|玩具/i,'“toy”'],
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
 // every text node on its own
 const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){if(!shown(n.parentElement)) continue; const t=n.textContent.replace(/\\s+/g,' '); const m=t.match(new RegExp(bad.source,'g')); if(m) out.add(m.join('|')+' in: '+t.trim().slice(0,80));}
 // and the edges of an inline element inside a sentence (a link, a defined term, bold): the characters on both sides of its boundary
 root.querySelectorAll('a,.def,b,.t,em,code').forEach(e=>{if(!shown(e)||getComputedStyle(e).display!=='inline') return;
   const t=e.textContent; if(!t) return; const prev=e.previousSibling, next=e.nextSibling;
   const pc=prev&&prev.nodeType===3?prev.textContent.slice(-1):'', nc=next&&next.nodeType===3?next.textContent.slice(0,1):'';
   const pair=(a,b)=>a&&b&&bad.test(a+b);
   if(pair(pc,t[0])) out.add(pc+t[0]+' before: '+t.slice(0,40)); if(pair(t[t.length-1],nc)) out.add(t[t.length-1]+nc+' after: '+t.slice(-40));});
 // H1 (2026-10-09): a hover may hold a typeset formula. A formula on its own line (div.fx) is a line break here and an inline one counts as a
 // Latin word; the text of each formula's pieces (mi, mn, mo, mtext) is checked on its own, so “token数” inside a formula is still caught.
 const tipParts=t=>{if(!t.includes('<math')) return [t]; const d=document.createElement('div'); d.innerHTML=t; const parts=[];
   d.querySelectorAll('math').forEach(m=>{m.querySelectorAll('mi,mn,mo,mtext').forEach(x=>parts.push(x.textContent)); const f=m.parentElement;
     if(f&&f!==d&&f.classList.contains('fx')&&f.children.length===1&&!f.textContent.replace(m.textContent,'').trim()) f.replaceWith('\\n'); else m.replaceWith('X');});
   return [d.innerHTML, ...parts];};
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t0=e.getAttribute('data-tipx'); tipParts(t0).forEach(t=>{const m=t.match(/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>][\\u4e00-\\u9fff]/g); if(m) out.add(m.join('|')+' in tooltip: '+t0.slice(0,80));});});
 return [...out];}"""
# Chinese text takes full-width punctuation: no ASCII colon, semicolon, comma, question or exclamation mark next to a Chinese character
PUNCT = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), bad=/[\\u4e00-\\u9fff][:;,!?]|[:;,!?] ?[\\u4e00-\\u9fff]/g;
 const shown=e=>e&&!e.closest('[data-endnotes],pre,code,.mathbody,#tip')&&e.getClientRects().length;
 const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){if(!shown(n.parentElement)) continue; const t=n.textContent.replace(/\\s+/g,' '); const m=t.match(bad); if(m) out.add(m.join('|')+' in: '+t.trim().slice(0,80));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); const m=t.match(bad); if(m) out.add(m.join('|')+' in tooltip: '+t.slice(0,80));});
 return [...out];}"""
COLONS = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), texts=[];
 // an element that holds its own words (a direct text node): a legend or a paper's title and description are separate items, not one sentence
 root.querySelectorAll('p,li,button,label,div,span,h1,h2,h3,summary').forEach(e=>{if(e.closest('[data-endnotes],pre,code,svg,.mathbody')||!e.getClientRects().length) return; if(e.querySelector('p,li,div,ul,ol,h1,h2,h3,table')) return;
   if(![...e.childNodes].some(c=>c.nodeType===3&&c.textContent.trim())) return; texts.push(e.textContent);});
 root.querySelectorAll('[data-tipx]').forEach(e=>texts.push(e.getAttribute('data-tipx')));
 for(const t of texts){const sents=t.replace(/\\s+/g,' ').split(/(?<=[。？！])|(?<=[.?!])\\s+/);
   for(const x of sents){const k=(x.match(/：|: /g)||[]).length; if(k>=2) out.add(x.trim().slice(0,160));}}
 return [...out];}"""
BIGNUM = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #pd-mathbody, .num-big, #pd-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
# blue (weights), amber (KV cache) and violet (runtime overhead) only on the elements that mean them; red only for a missed goal (R5.6)
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/pd-bw|sw w\\b|seg w\\b/,[amber]:/pd-bk|sw kv\\b|seg kv\\b/,[red]:/status bad|badline|badc|pd-over|warn/,[violet]:/seg oh\\b|sw oh\\b/};
 const out=[]; for(const e of document.querySelectorAll('#pd-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
rule_hits = []


def rules(pg, where, lang):
    for sel in ('#pd-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':   # R6.7 is a rule for Chinese sentences (the kit's English LM head definition, “(LM: language model) … matrix: …”, is fine)
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
        if lang == 'zh':
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def badges(pg): return pg.evaluate("()=>[...document.querySelectorAll('#pd-metrics .metric .d')].map(d=>d.textContent).filter(Boolean)")


def obs_on_screen(pg):
    return pg.evaluate("""()=>{const b=document.querySelector('#pd-rail [data-obs]'); if(!b||b.hidden) return null; const r=b.getBoundingClientRect(), top=document.querySelector('.topbar').getBoundingClientRect().bottom;
      return {top:Math.round(r.top), vh:innerHeight, ok:r.top>=top-2&&r.top<innerHeight-40}}""")


def run_guided(b, w, h, lang, dark=False):
    tag = f'{"dark_" if dark else ""}{w}_{lang}'
    pg = page(b, w, h, lang, dark)
    phone = w <= 760
    title = text(pg, '#pd-app .labtitle')
    check(f'{tag} title', title == Z(lang, 'Why is the first token slow and the rest fast?', '为什么首 token 要等很久，之后却一个接一个很快？'), title)
    check(f'{tag} no fidelity tag on top (owner, 2026-10-11 19:28 UTC)', not any(w in text(pg, '#pd-app .labmeta') for w in ('Fidelity', '保真度')), text(pg, '#pd-app .labmeta'))
    check(f'{tag} outcome line under the title', text(pg, '#pd-app .lab-outcome').startswith(Z(lang, 'You will be able to:', '学完你能：')))
    # the fidelity and primer popovers, opened, follow the writing rules too
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#pd-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        pg.evaluate(f"()=>document.querySelector('#pd-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(200)
        check(f'{tag} popover {pid} opens', pg.evaluate(f"()=>{{const p=document.getElementById('{pid}'); return !!p&&p.getClientRects().length>0}}"))
        rules(pg, f'{tag} popover {pid}', lang)
        pg.evaluate(f"()=>document.querySelector('#pd-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(150)
    # part 1
    check(f'{tag} part 1: the stage shows the mission (60.0 ms, then 7.60 ms)', '60.0' in text(pg, '.pd-tl-labs') and '7.60' in text(pg, '.pd-tl-labs'), text(pg, '.pd-tl-labs'))
    check(f'{tag} part 1: no overflow', overflow(pg) <= 0, overflow(pg))
    if phone:
        fs = pg.evaluate("""()=>{const pb=document.querySelector('#pd-rail .pbar'), nx=pb&&pb.querySelector('[data-nav="next"]'), tr=document.getElementById('pd-tl-track'), st=document.querySelector('#pd-app .bench>.stage');
          const r=e=>e?e.getBoundingClientRect():null, a=r(pb), b=r(nx), c=r(tr), d=r(st);
          return {bar:a&&a.top<innerHeight&&a.height>0, next:!!b&&b.bottom<=innerHeight&&b.height>0, track:c?Math.round(c.bottom):null, stage:d?Math.round(d.top):null, vh:innerHeight}}""")
        check(f'{tag} phone first screen: the parts’ bar with Next, and the timeline of the chart (R7.3)', fs['bar'] and fs['next'] and fs['track'] is not None and fs['track'] <= fs['vh'] and fs['stage'] < fs['vh'] - 200, fs)
        shot(pg, f'{tag}_first_screen')
    shot(pg, f'{tag}_p1', full=phone)
    rules(pg, f'{tag} part 1', lang)
    ans_absent = lambda: ('266.7' not in all_text(pg)) and ('8.00' not in all_text(pg))
    check(f'{tag} part 1: the prediction’s answer (266.7 ms, 8.00 ms) appears nowhere yet', ans_absent())
    # part 2: the gate
    pg.evaluate("()=>document.querySelector('#pd-rail [data-nav=\"next\"]').click()"); pg.wait_for_timeout(250); settle(pg)
    st = state(pg)
    check(f'{tag} part 2 reached', st['step'] == 1, st)
    nd = pg.evaluate("()=>[...document.querySelectorAll('#pd-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
    check(f'{tag} part 2: every Next is disabled before a prediction', nd and all(nd), nd)
    check(f'{tag} part 2: the prompt length is fixed before the prediction (note shown, control hidden)', vis(pg, '#pd-Nlock') and not vis(pg, '#pd-ctl-N'))
    check(f'{tag} part 2: the answer appears nowhere (text and tooltips)', ans_absent())
    pg.evaluate("()=>document.querySelector('#pd-rail [data-pred=\"later\"]').click()"); pg.wait_for_timeout(200)
    check(f'{tag} part 2: “Saved … you can still change it”', Z(lang, 'Saved. You can still change it before part 4 shows the result.', '已记下。第 4 部分给出结果之前，还可以改。') in text(pg, '#pd-rail .step.current'))
    check(f'{tag} part 2: the pick shows as pressed with a radio dot', pg.evaluate("()=>{const b=document.querySelector('#pd-rail [data-pred=\"later\"]'); return b.getAttribute('aria-pressed')==='true'&&getComputedStyle(b,'::before').content!=='none'}"))
    nd = pg.evaluate("()=>[...document.querySelectorAll('#pd-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
    check(f'{tag} part 2: Next enabled after a prediction', nd and not any(nd), nd)
    shot(pg, f'{tag}_p2')
    rules(pg, f'{tag} part 2', lang)
    # part 3: prefill; decode stays hidden
    pg.evaluate("()=>document.querySelector('#pd-rail [data-nav=\"next\"]').click()"); pg.wait_for_timeout(250); settle(pg)
    check(f'{tag} part 3: the decode half is hidden (TPOT card “?”, the decode bars masked)', text(pg, '#pd-metrics [data-k="tpot"] .v') == '?' and vis(pg, '.pd-ph[data-ph="dec"] .pd-mask'))
    check(f'{tag} part 3: no ▲▼ badge on entering a part', not badges(pg), badges(pg))
    check(f'{tag} part 3: Next waits for the action', pg.evaluate("()=>document.querySelector('#pd-rail .step.current [data-nav=\"next\"]').disabled"))
    slide(pg, '#pd-ic-N-in', 4)
    o = text(pg, '#pd-rail [data-obs]')
    # R6.20: the result says the trend in words with one rounded pair; the exact time is on the TTFT card
    ttft8 = text(pg, '#pd-metrics [data-k="ttft"] .v').replace('\xa0', ' ')
    check(f'{tag} part 3: dragging to 8K gives the first token at 266.7 ms (card), about 270 ms instead of about 60 ms in words, a little faster than the prompt',
          ttft8 == '266.7 ms' and Z(lang, 'about 270 ms instead of about 60 ms at 2K, so the time grows a little faster than the prompt', '约 270 ms 后才出现第一个 token，2K 时约 60 ms，时间比 prompt 涨得还快一点') in o.replace('\xa0', ' '), (ttft8, o[:200]))
    check(f'{tag} part 3: still no decode time at 8K anywhere (8.00 ms)', '8.00' not in all_text(pg))
    check(f'{tag} part 3: a ▲ badge after the learner’s own change', any('▲' in x for x in badges(pg)), badges(pg))
    if w == 1440 and not dark:
        pg.set_viewport_size({'width': 1440, 'height': 1000}); pg.wait_for_timeout(200)
        pg.evaluate("()=>window.scrollTo(0,0)"); slide(pg, '#pd-ic-N-in', 6); oc = obs_on_screen(pg)
        check(f'{tag} part 3: after the drag the result is on screen at 1440×1000 (R3.10)', oc and oc['ok'], oc)
        slide(pg, '#pd-ic-N-in', 4)
        pg.set_viewport_size({'width': w, 'height': h}); pg.wait_for_timeout(200)
    shot(pg, f'{tag}_p3', full=phone)
    rules(pg, f'{tag} part 3', lang)
    # part 4: the reveal, with the learner's own run (8K from part 3)
    pg.evaluate("()=>document.querySelector('#pd-rail .step.current [data-nav=\"next\"]').click()"); pg.wait_for_timeout(250); settle(pg)
    o = text(pg, '#pd-rail [data-obs]')
    # R6.20: the result in words with one rounded pair (the exact 266.7 ms and 8.00 ms are on the cards), then the verdict about 2K → 8K, then why
    o4 = o.replace('\xa0', ' '); i_res = o4.find(Z(lang, 'each later token takes about 8.0 ms instead of about 7.6 ms at 2K, barely more', '之后每个 token 约 8.0 ms，2K 时约 7.6 ms，几乎没变'))
    i_ans, i_why = o4.find(Z(lang, 'the right answer is', '正确答案是')), o4.find(Z(lang, 'Why:', '原因：'))
    check(f'{tag} part 4: the result first, then the verdict with the right answer, then why (R3.8, R3.10)',
          -1 < i_res < i_ans < i_why and Z(lang, 'From 2K to 8K tokens', '从 2K 到 8K 个 token') in o4
          and text(pg, '#pd-metrics [data-k="ttft"] .v').replace('\xa0', ' ') == '266.7 ms', o[:300])
    check(f'{tag} part 4: the TPOT card shows 8.00 ms, no badge on entering', text(pg, '#pd-metrics [data-k="tpot"] .v').replace(' ', ' ') == '8.00 ms' and not badges(pg), (text(pg, '#pd-metrics [data-k="tpot"] .v'), badges(pg)))
    shot(pg, f'{tag}_p4', full=phone)
    rules(pg, f'{tag} part 4', lang)
    # the language switch keeps the part, the prompt and the prediction
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(350)
    st2 = state(pg)
    check(f'{tag} part 4: a language switch keeps the part, the prompt length and the prediction', st2['step'] == 3 and st2['N'] == 8192 and st2['pred'] == 'later' and vis(pg, '#pd-rail [data-obs]'), st2)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(350)
    # part 5: batch the decode
    go_part(pg, 4)
    check(f'{tag} part 5: back to the 2K mission with one request', state(pg)['N'] == 2048 and state(pg)['B'] == 1)
    slide(pg, '#pd-ic-B-in', 64)
    o = text(pg, '#pd-rail [data-obs]')
    # R6.20: the exact 16.01 ms and 3,996 tokens per second are on the cards; the result says about 30 times, and about twice as long a wait
    cd5 = (text(pg, '#pd-metrics [data-k="tpot"] .v').replace('\xa0', ' '), text(pg, '#pd-metrics [data-k="thr"] .v'))
    check(f'{tag} part 5: 64 requests: 16.01 ms per step and 3,996 tokens per second (cards), about 30 times one request, about twice the wait',
          cd5 == ('16.01 ms', '3,996') and Z(lang, 'about 30 times as many tokens per second', '约是一个请求时的 30 倍') in o and Z(lang, 'about twice as long', '约是单独跑时的两倍') in o, (cd5, o[:240]))
    check(f'{tag} part 5: the GPU memory bar shows the KV caches of the batch', vis(pg, '#pd-membar .seg.kv'))
    b64 = badges(pg)
    check(f'{tag} part 5: badges after the change say what they compare with', b64 and all(('vs last' in x or 'last value' in x) if lang == 'en' else ('上次' in x) for x in b64), b64)
    pg.evaluate("()=>document.querySelector('#pd-ic-B [data-act=\"bplus\"]').click()"); pg.wait_for_timeout(200)
    check(f'{tag} part 5: the + button adds one request', state(pg)['B'] == 65, state(pg))
    # the roofline deep dive
    pg.evaluate("()=>document.querySelector('#pd-rail [data-act=\"roof\"]').click()"); pg.wait_for_timeout(300)
    check(f'{tag} part 5: the roofline deep dive opens on the stage', vis(pg, '#pd-roof svg') and not vis(pg, '#pd-lim') and Z(lang, 'Deep dive', '深入') in text(pg, '#pd-roof'))
    shot(pg, f'{tag}_p5_roofline', full=phone)
    rules(pg, f'{tag} part 5 roofline', lang)
    pg.evaluate("()=>document.querySelector('#pd-roof [data-act=\"roof-close\"]').click()"); pg.wait_for_timeout(250)
    check(f'{tag} part 5: back to the two limits, the deep dive marked as seen', vis(pg, '#pd-lim') and not vis(pg, '#pd-roof') and '✓' in text(pg, '#pd-rail [data-act="roof"]'))
    shot(pg, f'{tag}_p5', full=phone)
    rules(pg, f'{tag} part 5', lang)
    # part 6: the experiment
    run_part6(pg, tag, w, h, lang, phone, dark)
    # part 7 (was part 6 before the experiment was added)
    go_part(pg, 6)
    t6 = text(pg, '#pd-rail .step.current')
    check(f'{tag} part 7: three misconceptions checked, and the button to the scenarios (R3.12)', t6.count('✗') == 3 and vis(pg, '#pd-rail [data-mode-go="sandbox"]'), t6[:200])
    nl = pg.evaluate("""()=>{const shown=x=>!!x&&getComputedStyle(x).display!=='none'&&x.getBoundingClientRect().height>0; const r=document.querySelector('#pd-rail .next-lab'), e=document.querySelector('#view-dyn .next-lab-end .next-lab');
      return innerWidth<=760 ? shown(e) : shown(r)}""")
    check(f'{tag} part 7: the next-lab footer (R3.11)', nl)
    shot(pg, f'{tag}_p7', full=phone)
    rules(pg, f'{tag} part 7', lang)
    pg.evaluate("()=>document.querySelector('#pd-rail [data-mode-go=\"sandbox\"]').click()"); pg.wait_for_timeout(350); settle(pg)
    check(f'{tag} “See the scenarios” opens Scenarios', state(pg)['mode'] == 'sandbox')
    check(f'{tag} no overflow at the end of the tour', overflow(pg) <= 0, overflow(pg))
    pg.close()


XS = lambda pg: pg.evaluate("()=>{const s=window.__pd.state(); return [s.xv,s.xl,s.xpred]}")
CARDS = "()=>Object.fromEntries([...document.querySelectorAll('#pd-metrics .metric')].map(m=>[m.dataset.k,m.querySelector('.v').textContent.replace(/\\u00a0/g,' ')]))"
CMPS = "()=>Object.fromEntries([...document.querySelectorAll('#pd-metrics .metric')].map(m=>[m.dataset.k,m.querySelector('.d').textContent]))"
# the original's timeline, measured: the prefill block's width and the last tick's position, in px and in % of the track
ORIG_GEOM = """()=>{const r=document.querySelector('.pd-xrow[data-xv="orig"]'), t=r.querySelector('.pd-xtrack').getBoundingClientRect(), p=r.querySelector('.pd-xpre').getBoundingClientRect(), k=[...r.querySelectorAll('.pd-xticks i')].pop().getBoundingClientRect();
  return [Math.round((p.right-t.left)*10)/10, Math.round((k.left-t.left)*10)/10, Math.round(t.width*10)/10]}"""
ROW_PCT = """(v)=>{const r=document.querySelector('.pd-xrow[data-xv="'+v+'"]'); const pre=parseFloat(r.querySelector('.pd-xpre').style.width), b=r.querySelector('.pd-xband');
  return [pre, parseFloat(b.style.left)+parseFloat(b.style.width)]}"""
BARS = """()=>[...document.querySelectorAll('#pd-lim .pd-ph')].map(p=>[...p.querySelectorAll('.pd-row')].map(rw=>{const bar=rw.querySelector('.pd-bar').getBoundingClientRect(), g=rw.querySelector('.pd-ghost');
  const fill=[...rw.querySelectorAll('.pd-bar i')].reduce((m,i)=>Math.max(m,i.getBoundingClientRect().right),bar.left);
  return {fill:Math.round((fill-bar.left)/bar.width*1000)/10, ghost:g&&!g.hidden?Math.round(g.getBoundingClientRect().width/bar.width*1000)/10:null, set:rw.classList.contains('pd-set')}}))"""


def lever(pg, v):
    pg.evaluate(f"()=>document.querySelector('#pd-rail [data-lever=\"{v}\"]').click()"); pg.wait_for_timeout(250); settle(pg)


def load(pg, l):
    pg.evaluate(f"()=>document.querySelector('#pd-rail .seg-ctl[data-key=\"xl\"] button[data-v=\"{l}\"]').click()"); pg.wait_for_timeout(250); settle(pg)


def run_part6(pg, tag, w, h, lang, phone, dark):
    """part 6, the experiment: prediction gate, three GPUs on one time scale, the load switch, phone first screen, rapid clicks"""
    go_part(pg, 5)
    st = state(pg)
    check(f'{tag} part 6 reached, the chat load and the original H100', st['step'] == 5 and (st['N'], st['M'], st['B']) == (2048, 300, 1) and XS(pg)[:2] == ['orig', 'chat'], st)
    t6 = text(pg, '#pd-rail .step.current')
    check(f'{tag} part 6: its title and question', Z(lang, 'Experiment: twice the compute, or twice the bandwidth?', '实验：算力翻倍，还是带宽翻倍？') in (t6 if not phone else text(pg, '#pd-rail .pbar'))
          and Z(lang, 'Twice the arithmetic per second, same memory bandwidth. What gets faster?', '每秒的计算量翻倍，显存带宽不变。什么会变快？') in t6, t6[:200])
    if phone:
        check(f'{tag} part 6: the part bar says the lead', Z(lang, 'Doubling a GPU’s arithmetic or its bandwidth: which part of the answer gets faster?', '把 GPU 的算力或带宽翻倍：回答的哪一段会变快？') in text(pg, '#pd-rail .pbar'))
    nd = pg.evaluate("()=>[...document.querySelectorAll('#pd-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
    check(f'{tag} part 6: every Next is disabled before a prediction', nd and all(nd), nd)
    check(f'{tag} part 6: four choices, “neither changes much” among them (R3.8), no GPU options or Show me yet', pg.evaluate("()=>document.querySelectorAll('#pd-rail [data-xpred]').length") == 4
          and Z(lang, 'Neither changes much', '两个都变化不大') in text(pg, '#pd-rail [data-xpred="neither"]')
          and not pg.query_selector('#pd-rail [data-lever]') and not vis(pg, '#pd-rail [data-nav="showme"]') and not vis(pg, '#pd-rail [data-obs]'))
    masked = pg.evaluate("()=>['c2','b2'].map(v=>{const r=document.querySelector('.pd-xrow[data-xv=\"'+v+'\"]'); return r.classList.contains('pd-xhid')&&r.querySelector('.pd-xmask').getClientRects().length>0&&!r.querySelector('.pd-xrow-e').textContent})")
    at = all_text(pg).replace('\xa0', ' ')
    check(f'{tag} part 6: before the pick the two doubled GPUs are masked and their numbers are nowhere (30.0 ms, 3.80 ms, 2.31 s, 1.20 s), the status says predict first',
          masked == [True, True] and not any(x in at for x in ['30.0 ms', '3.80 ms', '2.31 s', '1.20 s']) and text(pg, '#pd-status') == Z(lang, 'Predict first', '先预测'), (masked, [x for x in ['30.0 ms', '3.80 ms', '2.31 s', '1.20 s'] if x in at]))
    check(f'{tag} part 6: Settings locked with a note (R4.2)', vis(pg, '#pd-lock') and not vis(pg, '#pd-ctl-N') and not vis(pg, '#pd-ctl-B') and Z(lang, 'In this experiment', '这个实验里') in text(pg, '#pd-lock'))
    check(f'{tag} part 6: the idealised label on the stage', text(pg, '.pd-xa-ideal') == Z(lang, 'Idealised: one number doubled, everything else the same; not a real GPU.', '理想化实验：只把一项指标翻倍，其余不变；不是某张真实的 GPU。'), text(pg, '.pd-xa-ideal'))
    if phone:
        fs = pg.evaluate("""()=>{const r=e=>e.getBoundingClientRect(), rows=[...document.querySelectorAll('.pd-xrow')].map(e=>Math.round(r(e).bottom)), pb=r(document.querySelector('#pd-rail .pbar')), tb=document.querySelector('.topbar').getBoundingClientRect().bottom;
          return {rows, pbTop:Math.round(pb.top), topbar:Math.round(tb), vh:innerHeight}}""")
        check(f'{tag} part 6 phone first screen: the part bar, then the three timelines, each a row that picks its GPU (R7.3)', fs['pbTop'] >= fs['topbar'] - 2 and len(fs['rows']) == 3 and max(fs['rows']) <= fs['vh'], fs)
        go = pg.evaluate("()=>{const g=document.querySelector('.pd-xgo'); const r=g.getBoundingClientRect(); return {shown:r.height>0, bottom:Math.round(r.bottom), vh:innerHeight, t:g.textContent}}")
        check(f'{tag} part 6 phone first screen: a button to the question (R7.3, review A1)', go['shown'] and go['bottom'] <= go['vh'] and Z(lang, 'Answer the question', '先回答问题') in go['t'], go)
        shot(pg, f'{tag}_p6_first_screen')
        pg.click('.pd-xgo'); pg.wait_for_timeout(900); settle(pg)
        q = pg.evaluate("()=>{const r=document.querySelector('#pd-rail .step.current .choices').getBoundingClientRect(); return [Math.round(r.top),Math.round(r.bottom),innerHeight]}")
        check(f'{tag} part 6 phone: the button brings the question’s choices on screen', 0 <= q[0] and q[0] < q[2] - 120, q)
        pg.evaluate("()=>window.scrollTo(0,0)"); go_part(pg, 5)
    else:
        pg.evaluate("()=>window.scrollTo(0,document.body.scrollHeight)"); pg.wait_for_timeout(200)
        pg.click('.pd-xrow[data-xv="c2"]'); pg.wait_for_timeout(700); settle(pg)
        q = pg.evaluate("()=>{const r=document.querySelector('#pd-rail .step.current .choices').getBoundingClientRect(); return [Math.round(r.top),Math.round(r.bottom),innerHeight]}")
        check(f'{tag} part 6: a click on a masked timeline brings the question on screen and picks nothing (review A1)', 0 <= q[0] < q[2] - 120 and XS(pg)[0] == 'orig', (q, XS(pg)))
    shot(pg, f'{tag}_p6_before', full=phone)
    rules(pg, f'{tag} part 6 before the prediction', lang)
    # the pick: right in EN, wrong in ZH (the verdict then gives the right answer)
    pick = 'first' if lang == 'en' else 'both'
    pg.evaluate(f"()=>document.querySelector('#pd-rail [data-xpred=\"{pick}\"]').click()"); pg.wait_for_timeout(300); settle(pg)
    check(f'{tag} part 6: after the pick, the prediction is kept, the three GPU options (the original first, pressed) and the load switch appear',
          XS(pg)[2] == pick and pg.evaluate("()=>[...document.querySelectorAll('#pd-rail [data-lever]')].map(b=>b.dataset.lever+':'+b.getAttribute('aria-pressed')).join()") == 'orig:true,c2:false,b2:false'
          and vis(pg, '#pd-ic-xl') and not pg.query_selector('#pd-rail [data-xpred]'))
    res = pg.evaluate("()=>[...document.querySelectorAll('#pd-rail [data-lever] .lv-r')].map(e=>e.textContent.replace(/\\u00a0/g,' '))")
    check(f'{tag} part 6: each option shows its whole answer on its title row (2.34 s, 2.31 s, 1.20 s)', res == ['2.34 s', '2.31 s', '1.20 s'], res)
    lv = pg.evaluate("()=>[...document.querySelectorAll('#pd-rail [data-lever] .lv-d')].map(e=>e.textContent)")
    check(f'{tag} part 6: below each, what the doubling did, in words, each by its own ratio (R6.10b)', lv[1] == Z(lang, 'first token 2× faster, whole answer about the same', '首 token 快一倍，整个回答几乎不变') and lv[2] == Z(lang, 'later tokens 2× faster, whole answer about 2× faster', '之后的 token 快一倍，整个回答快了约一倍'), lv)
    nd = pg.evaluate("()=>[...document.querySelectorAll('#pd-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
    check(f'{tag} part 6: Next still waits for a GPU to be picked', nd and all(nd), nd)
    check(f'{tag} part 6: the doubled GPUs are unmasked, the original drawn in full and the others faded', pg.evaluate("()=>[...document.querySelectorAll('.pd-xrow')].map(r=>(r.getAttribute('aria-disabled')||'-')+(r.classList.contains('pd-xoff')?'f':'s')).join()") == '-s,-f,-f')
    g0 = pg.evaluate(ORIG_GEOM)
    # compute × 2, from the card
    if phone:
        pg.evaluate("()=>document.querySelector('#pd-rail [data-lever=\"c2\"]').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
        y0, top0 = pg.evaluate("()=>scrollY"), pg.evaluate("()=>Math.round(document.querySelector('#pd-rail [data-lever=\"c2\"]').getBoundingClientRect().top)")
        pg.click('#pd-rail [data-lever="c2"]'); pg.wait_for_timeout(350); settle(pg)
        top1 = pg.evaluate("()=>Math.round(document.querySelector('#pd-rail [data-lever=\"c2\"]').getBoundingClientRect().top)")
        check(f'{tag} part 6 phone: picking an option in the card does not scroll it away (it stays under the finger)', abs(top1 - top0) <= 2, (top0, top1, y0, pg.evaluate("()=>scrollY")))
        load(pg, 'doc'); lt0 = pg.evaluate("()=>Math.round(document.querySelector('#pd-rail .seg-ctl[data-key=\"xl\"]').getBoundingClientRect().top)")
        pg.click('#pd-rail .seg-ctl[data-key="xl"] button[data-v="chat"]'); pg.wait_for_timeout(350); settle(pg)
        lt1 = pg.evaluate("()=>Math.round(document.querySelector('#pd-rail .seg-ctl[data-key=\"xl\"]').getBoundingClientRect().top)")
        check(f'{tag} part 6 phone: switching the load does not scroll the switch away', abs(lt1 - lt0) <= 2 and XS(pg)[:2] == ['c2', 'chat'], (lt0, lt1, XS(pg)))
    else:
        lever(pg, 'c2')
        top1 = pg.evaluate("()=>{const b=document.querySelector('#pd-rail [data-lever=\"c2\"]').getBoundingClientRect(), t=document.querySelector('.topbar').getBoundingClientRect().bottom, d=document.getElementById('dock'); return [Math.round(b.top), Math.round(t+(d&&!d.hidden?d.offsetHeight:0)), innerHeight]}")
        check(f'{tag} part 6: after picking an option, it stays on screen below the bars (R3.2, R3.10)', top1[1] - 2 <= top1[0] < top1[2], top1)
    check(f'{tag} part 6: compute × 2 picked, and Next enabled', XS(pg)[0] == 'c2' and not any(pg.evaluate("()=>[...document.querySelectorAll('#pd-rail [data-nav=\"next\"]')].map(b=>b.disabled)")))
    g1 = pg.evaluate(ORIG_GEOM)
    check(f'{tag} part 6 same scale: the original’s timeline does not move when compute × 2 is picked', all(abs(a - b) <= 0.6 for a, b in zip(g0, g1)), (g0, g1))
    po, pc = pg.evaluate(ROW_PCT, 'orig'), pg.evaluate(ROW_PCT, 'c2')
    check(f'{tag} part 6 same scale: compute × 2’s prefill block is half the original’s, its answer ends about where the original’s does', abs(pc[0] / po[0] - 0.5) < 0.01 and 0.97 < pc[1] / po[1] < 1.0, (po, pc))
    cd, cm = pg.evaluate(CARDS), pg.evaluate(CMPS)
    check(f'{tag} part 6 cards, compute × 2: 30.0 ms, 7.60 ms, 2.31 s, each against the original', cd == {'ttft': '30.0 ms', 'tpot': '7.60 ms', 'e2e': '2.31 s'}
          and cm == {'ttft': Z(lang, '▼ −50% vs the original', '▼ −50%（比原版）'), 'tpot': Z(lang, 'same as the original', '和原版一样'), 'e2e': Z(lang, '▼ −1% vs the original', '▼ −1%（比原版）')}, (cd, cm))
    bars = pg.evaluate(BARS)
    check(f'{tag} part 6 bars, compute × 2: prefill’s arithmetic halved inside the original’s outline and still sets the time; the decode step’s memory bar fills its outline',
          abs(bars[0][0]['fill'] - 50) <= 1 and abs(bars[0][0]['ghost'] - 100) <= 1 and bars[0][0]['set'] and abs(bars[1][1]['fill'] - 100) <= 1 and abs(bars[1][1]['ghost'] - 100) <= 1 and bars[1][1]['set'], bars)
    o = text(pg, '#pd-rail [data-obs]').replace('\xa0', ' ')
    i_res = o.find(Z(lang, 'With twice the arithmetic, the first token comes twice as fast, but the whole answer still takes about 2.3 s.', '算力翻倍，首 token 快了一倍，整个回答却仍要约 2.3 s。'))
    i_ver = o.find(Z(lang, 'Your prediction, “Mainly the first token; later tokens barely change”, was right.', '正确答案是“主要是首 token 变快，之后的 token 几乎不变”'))
    i_why = o.find(Z(lang, 'Why:', '原因：'))
    check(f'{tag} part 6 compute × 2: the result in words with one rounded number, then the verdict, then why (R3.10, R6.20)', -1 < i_res < i_ver < i_why and Z(lang, 'so memory traffic sets its time, which stays the same', '读写显存决定它的耗时，这个耗时不变') in o, o[:400])
    zm = pg.evaluate("""()=>{const z=document.getElementById('pd-xzoom'); if(!z||!z.getClientRects().length) return null; const w=v=>Math.round(z.querySelector('.pd-xzrow[data-xv="'+v+'"] .pd-xzpre').getBoundingClientRect().width*10)/10;
      return {o:w('orig'), c:w('c2'), b:w('b2'), cap:z.querySelector('.pd-xzoom-cap').textContent}}""")
    check(f'{tag} part 6 chat: the first token zoomed in, compute × 2’s prefill visibly half the original’s (review A2)', zm and abs(zm['c'] / zm['o'] - 0.5) < 0.02 and zm['o'] - zm['c'] >= (60 if not phone else 40) and abs(zm['b'] - zm['o']) < 0.6, zm)
    gh = pg.evaluate("()=>{const g=document.querySelector('.pd-ph[data-ph=\"pre\"] .pd-row[data-k=\"c\"] .pd-ghost'), s=getComputedStyle(g); return [g.hidden, s.borderTopStyle, s.backgroundImage.slice(0,25)]}")
    check(f'{tag} part 6: the original’s bar is hatched, not a dashed outline (one meaning for dashed boxes, review A3)', gh[0] is False and gh[1] != 'dashed' and 'repeating-linear' in gh[2], gh)
    shot(pg, f'{tag}_p6_compute', full=phone)
    rules(pg, f'{tag} part 6 compute × 2', lang)
    # bandwidth × 2, from its timeline row (the row is the control on the stage)
    if phone:
        pg.evaluate("()=>window.scrollTo(0,0)"); go_part(pg, 5)   # back to the part's first screen: its bar at the top, the three timelines below
        check(f'{tag} part 6 phone: back on the part, its first screen shows the three timelines, now enabled, and the GPU is reset to the original', XS(pg)[:2] == ['orig', 'chat']
              and pg.evaluate("()=>[...document.querySelectorAll('.pd-xrow')].every(r=>!r.hasAttribute('aria-disabled')&&!r.classList.contains('pd-xhid')&&r.getBoundingClientRect().bottom<=innerHeight)"))
        y0 = pg.evaluate("()=>scrollY"); rt0 = pg.evaluate("()=>Math.round(document.querySelector('.pd-xrow[data-xv=\"b2\"]').getBoundingClientRect().top)")
        pg.click('.pd-xrow[data-xv="b2"]'); pg.wait_for_timeout(350); settle(pg)
        rt1 = pg.evaluate("()=>Math.round(document.querySelector('.pd-xrow[data-xv=\"b2\"]').getBoundingClientRect().top)")
        check(f'{tag} part 6 phone: tapping a timeline picks its GPU and does not move the page', XS(pg)[0] == 'b2' and abs(pg.evaluate("()=>scrollY") - y0) <= 2 and abs(rt1 - rt0) <= 2, (y0, rt0, rt1))
        shot(pg, f'{tag}_p6_first_screen_bandwidth')
    else:
        pg.click('.pd-xrow[data-xv="b2"]'); pg.wait_for_timeout(350); settle(pg)
        check(f'{tag} part 6: a click on a timeline row picks its GPU', XS(pg)[0] == 'b2')
    check(f'{tag} part 6: the card’s options follow the row picked', pg.evaluate("()=>document.querySelector('#pd-rail [data-lever=\"b2\"]').getAttribute('aria-pressed')") == 'true')
    g2 = pg.evaluate(ORIG_GEOM); pb = pg.evaluate(ROW_PCT, 'b2')
    check(f'{tag} part 6 same scale: with bandwidth × 2 picked the original’s timeline still does not move; bandwidth × 2’s prefill is the original’s and its answer ends about halfway',
          all(abs(a - b) <= 0.6 for a, b in zip(g0, g2)) and abs(pb[0] - po[0]) < 1e-6 and 0.5 < pb[1] / po[1] < 0.56, (g0, g2, po, pb))
    cd, cm = pg.evaluate(CARDS), pg.evaluate(CMPS)
    check(f'{tag} part 6 cards, bandwidth × 2: 60.0 ms, 3.80 ms, 1.20 s', cd == {'ttft': '60.0 ms', 'tpot': '3.80 ms', 'e2e': '1.20 s'}
          and cm == {'ttft': Z(lang, 'same as the original', '和原版一样'), 'tpot': Z(lang, '▼ −50% vs the original', '▼ −50%（比原版）'), 'e2e': Z(lang, '▼ −49% vs the original', '▼ −49%（比原版）')}, (cd, cm))
    o = text(pg, '#pd-rail [data-obs]').replace('\xa0', ' ')
    check(f'{tag} part 6 bandwidth × 2: every later token twice as fast, the whole answer from about 2.3 s to about 1.2 s, the first token does not move',
          Z(lang, 'every later token comes twice as fast, and the whole answer drops from about 2.3 s to about 1.2 s.', '之后每个 token 都快了一倍，整个回答从约 2.3 s 降到约 1.2 s。') in o
          and Z(lang, 'The first token does not move.', '首 token 没有变。') in o and Z(lang, 'Your prediction', '你的预测') not in o, o[:300])
    shot(pg, f'{tag}_p6_bandwidth', full=phone)
    rules(pg, f'{tag} part 6 bandwidth × 2', lang)
    # the long document: the twist
    load(pg, 'doc')
    check(f'{tag} part 6: the long document load: 32K prompt, 20-token answer', (state(pg)['N'], state(pg)['M']) == (32768, 20) and XS(pg)[1] == 'doc')
    lever(pg, 'c2')
    cd = pg.evaluate(CARDS)
    o = text(pg, '#pd-rail [data-obs]').replace('\xa0', ' ')
    check(f'{tag} part 6 long document, compute × 2: 0.747 s, 9.60 ms, 0.929 s (one unit per load, review A4); the whole answer from about 1.7 s to about 0.93 s, and the twist',
          cd == {'ttft': '0.747 s', 'tpot': '9.60 ms', 'e2e': '0.929 s'} and Z(lang, 'from about 1.7 s to about 0.93 s', '从约 1.7 s 降到约 0.93 s') in o
          and Z(lang, 'Which doubling helps the whole answer more depends on whether a request is mostly prompt or mostly answer.', '哪种翻倍对整个回答更有用，要看请求主要是 prompt 还是回答。') in o, (cd, o[:300]))
    rows_doc = pg.evaluate("()=>[...document.querySelectorAll('.pd-xrow')].map(r=>r.querySelector('.pd-xrow-f').textContent+' | '+r.querySelector('.pd-xrow-e').textContent).map(t=>t.replace(/\\u00a0/g,' '))")
    check(f'{tag} part 6 long document: one unit per column on the timelines (s, 3 significant digits), no zoom needed', rows_doc == [Z(lang, 'first token 1.49 s | whole answer 1.68 s', '首 token 1.49 s | 整个回答 1.68 s'), Z(lang, 'first token 0.747 s | whole answer 0.929 s', '首 token 0.747 s | 整个回答 0.929 s'),
          Z(lang, 'first token 1.49 s | whole answer 1.58 s', '首 token 1.49 s | 整个回答 1.58 s')] and not vis(pg, '#pd-xzoom'), rows_doc)
    if phone:
        # the mini result bar, from the card (the timelines are off screen): the load as its status, the original's value in the sentence (review A6)
        pg.evaluate("()=>document.querySelector('#pd-rail [data-lever=\"c2\"]').scrollIntoView({block:'center'})"); pg.wait_for_timeout(400)
        pg.click('#pd-rail [data-lever="c2"]'); pg.wait_for_timeout(500)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, status:document.getElementById('dkStatus').textContent, read:document.getElementById('dkRead').innerText.replace(/\\u00a0/g,' ')}}")
        check(f'{tag} part 6 phone mini result bar: the load as its status, the GPU with the load and its whole answer next to the original’s', dk['shown'] and dk['status'] == Z(lang, 'Long document', '长文档')
              and dk['read'] == Z(lang, 'Compute × 2, 32K prompt, 20-token answer: whole answer 0.929 s (original 1.68 s)', '算力 × 2，prompt 32K 个 token、回答 20 个 token：整个回答 0.929 s（原版 1.68 s）'), dk)
    pd, pc = pg.evaluate(ROW_PCT, 'orig'), pg.evaluate(ROW_PCT, 'c2')
    check(f'{tag} part 6 long document: on its own scale, compute × 2’s whole answer is visibly shorter (about 0.55 of the original’s)', 0.52 < pc[1] / pd[1] < 0.58, (pd, pc))
    shot(pg, f'{tag}_p6_doc_compute', full=phone)
    rules(pg, f'{tag} part 6 long document', lang)
    # rapid clicks end in the last one
    pg.evaluate("""()=>{const q=v=>document.querySelector('#pd-rail [data-lever="'+v+'"]'); ['b2','orig','c2','b2'].forEach(v=>q(v).click());
      document.querySelector('.pd-xrow[data-xv="orig"]').click(); document.querySelector('#pd-rail .seg-ctl[data-key="xl"] button[data-v="chat"]').click(); q('c2').click();}""")
    pg.wait_for_timeout(600); settle(pg)
    cd = pg.evaluate(CARDS)
    check(f'{tag} part 6: a quick run of clicks ends in the last clicked state (compute × 2, chat)', XS(pg)[:2] == ['c2', 'chat'] and cd == {'ttft': '30.0 ms', 'tpot': '7.60 ms', 'e2e': '2.31 s'}
          and pg.evaluate("()=>document.querySelector('.pd-xrow[data-xv=\"c2\"]').getAttribute('aria-pressed')") == 'true', (XS(pg), cd))
    check(f'{tag} part 6: no overflow', overflow(pg) <= 0, overflow(pg))
    # the language switch keeps the GPU, the load and the prediction
    load(pg, 'doc'); lever(pg, 'b2')
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(350)
    check(f'{tag} part 6: a language switch keeps the GPU, the load and the prediction', XS(pg) == ['b2', 'doc', pick] and state(pg)['step'] == 5
          and pg.evaluate("()=>document.querySelector('#pd-rail [data-lever=\"b2\"]').getAttribute('aria-pressed')") == 'true', XS(pg))
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(350)
    # leaving the part resets it; Scenarios keep the real H100
    go_part(pg, 6); go_part(pg, 5)
    check(f'{tag} part 6: leaving the part and coming back resets the GPU and the load, the prediction stays', XS(pg) == ['orig', 'chat', pick], XS(pg))
    lever(pg, 'c2')


def run_card_height(b):
    """R3.10 (review A8): at 1440×1000 in EN, after the longest case (a wrong pick, then compute × 2 on the chat), the clicked option
    and the card's Next are both on screen"""
    pg = page(b, 1440, 1000, 'en'); go_part(pg, 5)
    pg.evaluate("()=>document.querySelector('#pd-rail [data-xpred=\"both\"]').click()"); pg.wait_for_timeout(300); settle(pg)
    lever(pg, 'c2'); pg.wait_for_timeout(300); settle(pg)
    r = pg.evaluate("""()=>{const n=document.querySelector('#pd-rail .step.current .stepnav [data-nav="next"]').getBoundingClientRect(), l=document.querySelector('#pd-rail [data-lever="c2"]').getBoundingClientRect(),
      c=document.querySelector('#pd-rail .step.current').getBoundingClientRect(); return {next:[Math.round(n.top),Math.round(n.bottom)], lever:Math.round(l.top), card:Math.round(c.height), vh:innerHeight}}""")
    check('1440×1000 EN part 6: after a wrong pick and compute × 2, the card’s Next and the option clicked are on screen (R3.10)', r['next'][1] <= r['vh'] and 56 <= r['lever'] < r['vh'], r)
    pg.close()


def run_reduced(b):
    """reduced motion: no transition on the timelines, the bars or the rows; the state after a click is final at once"""
    pg = b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce')
    pg.on('pageerror', lambda e: errs.append(f'reduced: {e}'))
    pg.goto(URL); pg.wait_for_timeout(250); pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(80)
    pg.evaluate("()=>{location.hash='prefill-decode'}"); pg.wait_for_timeout(450)
    go_part(pg, 5)
    pg.evaluate("()=>document.querySelector('#pd-rail [data-xpred=\"later\"]').click()"); pg.wait_for_timeout(200)
    tr = pg.evaluate("()=>['.pd-xpre','.pd-xband','.pd-xfirst','.pd-xrow','.pd-bar i','.pd-ghost'].map(s=>getComputedStyle(document.querySelector(s)).transitionDuration)")
    check('reduced motion: no transition on the timelines, the rows, the bars or the outlines', all(x.split(',')[0].strip() in ('0s', '') for x in tr), tr)
    pg.evaluate("()=>document.querySelector('#pd-rail [data-lever=\"c2\"]').click()")
    now = pg.evaluate("()=>{const p=document.querySelector('.pd-xrow[data-xv=\"c2\"] .pd-xpre'), t=p.parentNode; return [Math.round(p.getBoundingClientRect().width/t.getBoundingClientRect().width*1e4)/100, parseFloat(p.style.width)]}")
    check('reduced motion: right after a click the timeline is already in its final state', abs(now[0] - now[1]) < 0.05, now)
    pg.close()
    # without reduced motion the change is short
    pg = page(b, 1440, 900, 'en'); go_part(pg, 5)
    tr = pg.evaluate("()=>['.pd-xpre','.pd-ghost'].map(s=>parseFloat(getComputedStyle(document.querySelector(s)).transitionDuration))")
    check('motion: the bars’ and timelines’ change is short (at most 0.3 s)', all(0 < x <= 0.3 for x in tr), tr)
    pg.close()


def run_modes(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    phone = w <= 760
    mode(pg, 'sandbox')
    presets = pg.evaluate("()=>[...document.querySelectorAll('#pd-rail [data-preset]')].length")
    check(f'{tag} scenarios: four, with the lead-in that says what to pick and what to watch (G1, the owner 2026-10-08)', presets == 4 and Z(lang, 'Pick a scenario and see how much of the wait for the whole answer is prefill.', '选一个场景，看看整个回答要等的时间里，prefill 占多少。') in text(pg, '#pd-rail .railcard') and Z(lang, 'do not stack', '不会叠加') not in text(pg, '#pd-rail .railcard'))
    names = []
    for i in range(presets):
        pg.evaluate(f"()=>document.querySelector('#pd-rail [data-preset=\"{i}\"]').click()"); pg.wait_for_timeout(300); settle(pg)
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#pd-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        ins = text(pg, '#pd-insight'); names.append(ins.split('\n')[0])
        check(f'{tag} scenario {i}: pressed alone, with its result', pressed.count('true') == 1 and pressed[i] == 'true' and len(ins) > 80, (pressed, ins[:120]))
        check(f'{tag} scenario {i}: no overflow', overflow(pg) <= 0, overflow(pg))
        rules(pg, f'{tag} scenario {i}', lang)
        if i == 2:
            check(f'{tag} scenario: code completion, prefill is almost half of the wait', Z(lang, 'prefill is 46% of the wait', 'prefill 占 46%') in ins, ins[:200])
            shot(pg, f'{tag}_scenario_code', full=phone)
    hsh = pg.evaluate("()=>location.hash")
    check(f'{tag} scenarios: the URL carries the setup', hsh == '#prefill-decode~2048_300_64', hsh)
    shot(pg, f'{tag}_scenario_busy', full=phone)
    # the hash round-trip: a fresh page on that URL opens Scenarios with the setup
    pg2 = page(b, w, h, lang, hash_='prefill-decode~2048_300_64')
    st = state(pg2)
    check(f'{tag} hash round-trip: Scenarios with 2K, 300 tokens, 64 requests, “Busy chat server” pressed', st['mode'] == 'sandbox' and (st['N'], st['M'], st['B']) == (2048, 300, 64)
          and pg2.evaluate("()=>document.querySelector('#pd-rail [data-preset=\"3\"]').getAttribute('aria-pressed')") == 'true', st)
    pg2.close()
    # a change under Settings: no scenario matches any more, and the page does not move (R1.8, R3.2)
    pg.evaluate("()=>document.querySelector('#pd-app [data-flow-nav] button[data-flow=free]').click()"); pg.wait_for_timeout(300)   # settings are changed in 自由调参 (owner, 2026-10-11)
    y0 = pg.evaluate("()=>scrollY")
    slide(pg, '#pd-ctl-M-in', 2, change=False)
    if not phone:
        check(f'{tag} a change under Settings does not scroll the page', abs(pg.evaluate("()=>scrollY") - y0) < 2)
    pg.evaluate("()=>document.querySelector('#pd-app [data-flow-nav] button[data-flow=scen]').click()"); pg.wait_for_timeout(300)
    pressed = pg.evaluate("()=>[...document.querySelectorAll('#pd-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: after a change under Settings no scenario is pressed, and the card says so', 'true' not in pressed and Z(lang, 'match none', '都不一样') in text(pg, '#pd-insight'), pressed)
    # the challenge, by the card's button (R3.12)
    pg.evaluate("()=>document.querySelector('#pd-rail [data-mode-go=\"challenge\"]').click()"); pg.wait_for_timeout(350); settle(pg)
    st = state(pg)
    check(f'{tag} challenge: 4K tokens of context, one user to start', st['mode'] == 'challenge' and st['N'] == 4096 and st['B'] == 1, st)
    check(f'{tag} challenge: Settings locked with a note, no grayed controls (R4.2)', vis(pg, '#pd-lock') and not vis(pg, '#pd-ctl-N') and not vis(pg, '#pd-ctl-B'))
    card = text(pg, '#pd-rail .railcard')
    check(f'{tag} challenge: the goal in full: model, GPU, context, target (R6.8)', all(x in card.replace('‑', '-').replace(' ', ' ') for x in ['Llama-3.1-8B', 'H100 SXM', '4K', '50', '20 ms']), card[:300])
    check(f'{tag} challenge: the capability card after the task', vis(pg, '#pd-rail [data-exits]'))
    check(f'{tag} challenge: no overflow', overflow(pg) <= 0, overflow(pg))
    shot(pg, f'{tag}_challenge', full=phone)
    rules(pg, f'{tag} challenge', lang)
    # tier 3: not met, with a hint and “show all options”
    slide(pg, '#pd-ic-chB-in', 64)
    check(f'{tag} challenge: live progress turns red over 20 ms (R4.3)', pg.evaluate("()=>document.getElementById('pd-ch-goal').classList.contains('pd-over')"))
    pg.evaluate("()=>document.querySelector('#pd-rail [data-act=\"submit\"]').click()"); pg.wait_for_timeout(300); settle(pg)
    res = text(pg, '#pd-ch-res')
    check(f'{tag} challenge tier 3: not met, with a hint and “Show the answer” (R4.4)', res.startswith(Z(lang, 'Not yet', '还没达到')) and Z(lang, 'Hint', '提示') in res and vis(pg, '#pd-ch-res [data-act="reveal"]') and not vis(pg, '#pd-ch-res .cmp')
          and text(pg, '#pd-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), res[:200])
    pg.evaluate("()=>document.querySelector('#pd-ch-res [data-act=\"reveal\"]').click()"); pg.wait_for_timeout(300)
    rows = pg.evaluate("()=>[...document.querySelectorAll('#pd-ch-res .cmp-row')].map(r=>({ok:r.classList.contains('ok'),me:r.classList.contains('me'),tag:!!r.querySelector('.cmp-tag'),t:r.querySelector('.cmp-lab span').textContent}))")
    oks = [r for r in rows if r['ok']]
    check(f'{tag} challenge chart: rows up to 46 users are green, 47 and more gray; the pick and the initial setup tagged', rows and all(int(re.match(r'\d+', r['t']).group()) <= 46 for r in oks) and len(oks) == 5
          and sum(r['me'] for r in rows) == 1 and Z(lang, 'initial setup', '初始设置') in text(pg, '#pd-ch-res .cmp'), rows)
    # the owner's review of batch 8: the chart keeps the 5 passing user counts and the pick (64); 47 and 48 are left out and counted
    check(f'{tag} challenge chart (brief): only the passing rows and the pick, 6 rows; the note counts the 2 left out', len(rows) == 6 and all(r['ok'] or r['me'] or r['tag'] for r in rows)
          and [int(re.match(r'\d+', r['t']).group()) for r in rows] == [1, 16, 32, 44, 46, 64]
          and text(pg, '#pd-ch-res .cmp-note') == Z(lang, 'The other 2 user counts give each user fewer than 50 tokens per second and are not shown.', '其余 2 档用户数下，每人每秒不到 50 个 token，图里没有列出。'), rows)
    ttl = text(pg, '#pd-ch-res .cmp-title')
    check(f'{tag} challenge chart: the title names the quantity and the conditions, not “by the number of users”', 'H100' in ttl and '4K' in ttl and Z(lang, 'by the number of users', '不同用户数下') not in ttl, ttl)
    xt = pg.evaluate("()=>[...document.querySelectorAll('#view-dyn [data-tipx]')].filter(e=>!e.closest('[id$=\"-mathbody\"],[id$=\"-math\"]')).map(e=>e.getAttribute('data-tipx')).filter(t=>t.includes('精确值是')||t.includes('Exactly'))")
    check(f'{tag} no hover outside Show the math gives a number’s exact value (the owner’s clarification of 2026-10-07)', not xt, xt[:2])
    # tier 2: met, but more users fit
    slide(pg, '#pd-ic-chB-in', 20)
    res = text(pg, '#pd-ch-res')
    check(f'{tag} challenge tier 2: met, but up to 46 users fit', res.startswith(Z(lang, 'Goal met', '达成目标')) and Z(lang, 'But more users fit: up to 46', '最多 46 个用户') in res, res[:220])
    # tier 1: met and the most
    slide(pg, '#pd-ic-chB-in', 46)
    res = text(pg, '#pd-ch-res')
    # the brief chart leaves 47 users out (it misses the goal); the sentence names it in words, its time on hover
    tip47 = pg.evaluate("()=>[...document.querySelectorAll('#pd-ch-res .def')].map(d=>d.dataset.tipx||'').find(t=>t.includes('47'))||''")
    check(f'{tag} challenge tier 1: 46 users, and that is the most (47 users take 20.02 ms, on hover; not on the chart)', res.startswith(Z(lang, 'Goal met: 46 users', '达成目标：46 个用户')) and Z(lang, 'that is the most', '已经是最多') in res
          and '20.02' in tip47.replace('\xa0', ' ') and Z(lang, '47 users: 20.02 ms per token', '47 个用户：每个 token 20.02 ms') not in res.replace('\xa0', ' ')
          and text(pg, '#pd-ch-res .cmp-note') == Z(lang, 'The other 3 user counts give each user fewer than 50 tokens per second and are not shown.', '其余 3 档用户数下，每人每秒不到 50 个 token，图里没有列出。'), (res[:220], tip47))
    check(f'{tag} challenge: baseline → effect → reason, and a specific caveat (R5.5, R6.5)', all(x in res for x in [Z(lang, 'Initial setup', '初始设置'), Z(lang, 'Why:', '原因：'), Z(lang, 'Note:', '注意：'), '59']), res[-400:])
    shot(pg, f'{tag}_challenge_met', full=phone)
    rules(pg, f'{tag} challenge met', lang)
    # the capability card's verdicts follow the writing rules too (a wrong pick and the right one, in both tabs with questions)
    for t in (0, 1):
        pg.evaluate(f"()=>document.querySelector('#pd-rail [data-exits] [data-ex=\"tab\"][data-i=\"{t}\"]').click()"); pg.wait_for_timeout(150)
        for v in pg.evaluate("()=>[...document.querySelectorAll('#pd-rail [data-exits] .ex-opts .choice')].map(b=>+b.dataset.v)"):
            pg.evaluate(f"()=>document.querySelector('#pd-rail [data-exits] .ex-opts .choice[data-v=\"{v}\"]').click()"); pg.wait_for_timeout(150)
            rules(pg, f'{tag} capability card tab {t + 1}, option {v}', lang)
    # the language switch keeps the answer
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} challenge: a language switch keeps 46 users and the submitted result', st['B'] == 46 and st['chSubmitted'] and vis(pg, '#pd-ch-res .cmp'), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    # the mini result bar on phones: once the timeline has scrolled away, a short status and a full sentence
    if phone:
        mode(pg, 'guided'); go_part(pg, 0)
        pg.evaluate("()=>{const t=document.getElementById('pd-lim'); window.scrollTo(0, t.getBoundingClientRect().top+scrollY+80)}"); pg.wait_for_timeout(500)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, status:document.getElementById('dkStatus').textContent, read:document.getElementById('dkRead').textContent}}")
        check(f'{tag} mini result bar: shown once the timeline scrolls away, with a short status and the times', dk['shown'] and len(dk['status']) <= 14 and '60.0' in dk['read'] and '7.60' in dk['read'], dk)
        shot(pg, f'{tag}_dock')
        rules(pg, f'{tag} dock', lang)
    pg.close()


# G1 and G2, the owner's message of 2026-10-08 (reviews/owner-b9-2026-10-08.md): a lead-in above each group of options instead of
# “one at a time; they do not stack”, and no sentence that leans on a detail of an earlier part. Checked on the lab's source, both languages.
_G_SRC = (P / 'labs' / 'prefill-decode.js').read_text()
for _b in ["一次只能选", "一次选一个", "一次选一种", "一次只选", "不会叠加", "do not stack", "One at a time", "pick one at a time", "Pick one at a time", "Try one change at a time", "一次试一种"]:
    check(f'G1: no “{_b}” left in the lab (EN and ZH)', _b not in _G_SRC)
for _s in ["Pick a GPU and see which gets faster: the first token or the later tokens.", "选一张 GPU，看看首 token 和之后的 token 哪个变快了。", "Pick a scenario and see how much of the wait for the whole answer is prefill.", "选一个场景，看看整个回答要等的时间里，prefill 占多少。"]:
    check(f'G1: the lead-in “{_s[:48]}” is in the lab', _s in _G_SRC)
for _s in ["The request is one user’s chat:", "请求是一个用户的一次聊天，prompt 有", "<b>Batching decode shares the weight read.</b> So the more requests decode together", "<b>把 decode 攒成 batch，大家共用读权重的那一遍。</b>所以一起 decode 的请求越多", "Twice the FLOP/s speeds up only the prefill", "FLOP/s 翻倍只让 prefill 变快"]:
    check(f'G2: the rewritten sentence “{_s[:48]}” is in the lab', _s in _G_SRC)
for _s in ["The request is the chat of part 1.", "请求还是第 1 部分的那次聊天。", "in part 6, twice the H100", "在第 6 部分，H100 的 FLOP/s 翻倍后", "with 64 requests at 2K tokens each, the GPU writes about", "64 个请求、每个 2K 个 token 时，GPU 每秒写出的 token 约是一个请求时的"]:
    check(f'G2: the sentence that leaned on an earlier part, “{_s[:48]}”, is gone', _s not in _G_SRC)

with sync_playwright() as p:
    b = p.chromium.launch()
    for (w, h) in [(1440, 900), (390, 844)]:
        for lang in ('en', 'zh'):
            run_guided(b, w, h, lang)
            run_modes(b, w, h, lang)
    run_guided(b, 1440, 900, 'en', dark=True)
    run_reduced(b)
    run_card_height(b)
    # other widths: no overflow in each mode (R7.1)
    for (w, h) in [(1280, 800), (900, 900), (720, 900)]:
        pg = page(b, w, h, 'en')
        for m in ('guided', 'sandbox', 'challenge'):
            mode(pg, m)
            check(f'{w} {m}: no overflow', overflow(pg) <= 0, overflow(pg))
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
        _pg.evaluate("()=>{location.hash='prefill-decode'}"); _pg.wait_for_timeout(500)
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


for hit in rule_hits[:60]:
    check('writing rules: ' + hit, False)
check('no page errors', not errs, errs[:5])
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:60] if fails else 'none')
sys.exit(1 if fails else 0)
