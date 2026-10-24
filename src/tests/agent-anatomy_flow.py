"""Flow test for the agent-anatomy lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH (plus one dark run when run in full):
- no page errors, no horizontal overflow, no text wider than its box (R7.8; the conversation's one-line previews are cut on purpose,
  their full text is in the row's tooltip), a tooltip for every help cursor (R1.1);
- part 1: the prediction gate with the kit's reason under a greyed-out Next, “Saved” after a pick; before the pick the total
  (73,011) appears nowhere, in text, tooltips or the header's popovers (R3.8);
- part 2: the replay auto-plays on entering (R3.9), pause stops it, Fast finishes within 5 s, Next waits for the end; the boxes of the
  stage keep their height during the replay (R7.9); then the result: 8 replies, 7 tool calls run by your code;
- part 3: Next waits for call 8; call 4 shows 9,441 sent, 5,650 already sent; at call 8 the total 73,011, 6.1 times, the verdict
  and the formula on its own line; the resending happens between calls, inside one call the input is computed once (the KV cache),
  the bill (73,011 input, 242 output; the owner, 2026-10-14), and Show the math sums the tokens written; the dock at 390;
- part 4: three ways, single choice, results on the options; stop on error ends after call 5 (red status, the stop row); “exit
  code 1” fixes at call 10 with 18,109 more tokens; Next waits for both;
- part 5: the speeds can be changed; three tools at once: 6 calls, 20.2 s against 22.8 s; the time line;
- part 6: four points with “show in the chart”, three misconceptions, See the scenarios; Skip on part 1;
- Scenarios: four, single choice, pressed state follows the settings, the URL carries the setup and a fresh page restores it;
- the challenge: Settings locked, the plain statement, a 1px line between the rows of an option group that wraps (the owner,
  2026-10-14), the three feedback tiers, tier 2 saying in its first line that a tighter
  setting exists, “Show the answer”; the brief chart (the owner's review of batch 8): only the 6 settings that let the normal run
  finish, the initial setup and the pick, both tagged, no other row, a note in the lab's words that counts the settings left out
  (10, or 9 when the pick stops the normal run too early), a title without “each setting”; the capability card; a language switch
  keeps the plan and the result;
- the writing rules on every visible text node and tooltip (R6.14, R6.7, Chinese spacing and punctuation, no 亿/万), colours (R5.6).
Screenshots: shots/agent-anatomy/.
Run: python3 tests/agent-anatomy_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, subprocess, sys, time
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/agent-anatomy.html', '--only', 'agent-anatomy'], check=True, cwd=P, stdout=subprocess.DEVNULL)
OUT = P / 'out' / 'agent-anatomy'; OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'wrapped-flow.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'agent-anatomy.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-flow.html')
SH = P / 'shots' / 'agent-anatomy'; SH.mkdir(parents=True, exist_ok=True)
ARGS = sys.argv[1:]
LANGS = [x for x in ('en', 'zh') if x in ARGS] or ['en', 'zh']
SIZES = [s for s in [(1440, 900), (390, 844)] if str(s[0]) in ARGS] or [(1440, 900), (390, 844)]
FULL = not ARGS
errs, fails, n = [], [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


Z = lambda lang, en, zh: zh if lang == 'zh' else en
RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy” (the fidelity label “Toy run” is the exception, R6.11)'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/Locked in|已锁定/,'“Locked in”'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万 instead of K, M, B (R6.18)'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour (R5.7)'],
   [/不是哪家|不是实测|不是测出来|不是对真实|假设，不是|自己的假设|不是真实录制|不是真实 agent|不是真的分词器|not a vendor|not a quote|, not a measurement|its own assumption|not a recording|not a real tokenizer/i,'negative disclaimer (the owner, 2026-10-14: state the assumption positively)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, .fid-tag')||!e.getClientRects().length) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); for(const [re,name] of bad) if(re.test(t)) out.push(name+' (tooltip): '+t.slice(0,90));
   if(!e.closest('.mathbody, details.math')&&/精确值是|Exactly/.test(t)) out.push('a hover that only expands a number to its exact value (the owner, 2026-10-07 23:37): '+t.slice(0,90));});
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
 root.querySelectorAll('p,li,button,label,div,span,h1,h2,h3,summary').forEach(e=>{if(e.closest('[data-endnotes],pre,code,svg,.mathbody')||!e.getClientRects().length) return; if(e.querySelector('p,li,div,ul,ol,h1,h2,h3,table')) return;
   if(![...e.childNodes].some(c=>c.nodeType===3&&c.textContent.trim())) return; texts.push(e.textContent);});
 root.querySelectorAll('[data-tipx]').forEach(e=>texts.push(e.getAttribute('data-tipx')));
 for(const t of texts){const sents=t.replace(/\\s+/g,' ').split(/(?<=[。？！])|(?<=[.?!])\\s+/);
   for(const x of sents){const k=(x.match(/：|: /g)||[]).length; if(k>=2) out.add(x.trim().slice(0,160));}}
 return [...out];}"""
BIGNUM = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #ana-mathbody, .num-big, #ana-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|ana-over|ana-row|ana-sw-err|nd|warn|wrong|ex-res/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#ana-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#ana-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg,.ana-pv')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#ana-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
rule_hits, fit_hits = [], []


def page(b, w, h, lang, dark=False, hash_='agent-anatomy'):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light')
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__ana.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
# the challenge's brief chart (A.ui.cmpChart({brief:true})): its rows by kind and the note under it
brief = lambda pg: pg.evaluate("""()=>{const res=document.getElementById('ana-ch-res'), c=res&&res.querySelector(':scope .cmp:not(details .cmp)'); if(!c) return null;
  const rows=[...c.querySelectorAll('.cmp-row')], has=(r,k)=>r.classList.contains(k), init=r=>!!r.querySelector('.cmp-tag');
  return {rows:rows.length, ok:rows.filter(r=>has(r,'ok')).length, me:rows.filter(r=>has(r,'me')).length, init:rows.filter(init).length,
    stray:rows.filter(r=>!has(r,'ok')&&!has(r,'me')&&!init(r)).length, note:(res.querySelector('.cmp-note')||{}).textContent||'',
    title:(c.querySelector('.cmp-title')||{}).textContent||'', more:res.querySelectorAll('details').length};}""")
def brief_ok(lang, r, rows, hidden):
    return (r is not None and r['rows'] == rows and r['ok'] == 6 and r['me'] == 1 and r['init'] == 1 and r['stray'] == 0 and r['more'] == 0
            and Z(lang, f'The other {hidden} settings stop the normal run too early', f'其余 {hidden} 种设置会把正常的运行拦得太早') in r['note']
            and Z(lang, 'each setting', '每种设置') not in r['title'])
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#ana-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#ana-rail [data-next-why]')].filter(e=>!e.hidden&&e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden'&&e.textContent.trim()).map(e=>e.textContent)")
obs = lambda pg: pg.evaluate("()=>{const o=document.querySelector('#ana-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' '):''}")
HEIGHTS = "()=>['#ana-stage .stage-head','#ana-loopbox','#ana-player','#ana-convbox','#ana-barsbox','#ana-metrics','#ana-graph'].map(s=>{const e=document.querySelector(s); return e?Math.round(e.getBoundingClientRect().height*10)/10:-1})"


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__ana||!window.__ana.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel, wait=250):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(wait); settle(pg)


def slide(pg, sel, v):
    pg.evaluate(f"""()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}));}}"""); pg.wait_for_timeout(250); settle(pg)


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#ana-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def rules(pg, where, lang):
    for sel in ('#ana-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def fit(pg, where):
    for h in pg.evaluate(OVER): fit_hits.append(f'{where}: text wider than its box (R7.8): {h}')
    for h in pg.evaluate(HELP): fit_hits.append(f'{where}: help cursor without a tooltip (R1.1): {h}')


def every(pg, where, lang, full_shot=None):
    check(f'{where}: no overflow', overflow(pg) <= 0, overflow(pg))
    rules(pg, where, lang); fit(pg, where)
    if full_shot: shot(pg, full_shot, full=pg.viewport_size['width'] <= 760)


def before_answer(pg):
    return pg.evaluate("""()=>{const t=[]; for(const s of ['#ana-app .labhead','#ana-rail','#ana-stage','#ana-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#ana-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")


def nav(pg, what):
    click(pg, f'#ana-rail [data-nav="{what}"]', 300)


def run_guided(b, w, h, lang, dark=False):
    tag = f'{"dark_" if dark else ""}{w}_{lang}'
    pg = page(b, w, h, lang, dark)
    phone = w <= 760
    check(f'{tag} title', text(pg, '#ana-app .labtitle') == Z(lang, 'What actually happens inside one agent task?', 'Agent 一次任务里到底发生了什么？'), text(pg, '#ana-app .labtitle'))
    check(f'{tag} no fidelity tag on top (owner, 2026-10-11 19:28 UTC)', not any(w in text(pg, '#ana-app .labmeta') for w in ('Fidelity', '保真度')), text(pg, '#ana-app .labmeta'))
    # the owner (2026-10-14): the end notes state what the lab assumes, positively, and how a provider bills an agent task
    ends = pg.evaluate("()=>{const v=document.querySelector('#view-dyn:not([hidden])')||document.querySelector('#view-lab:not([hidden])'); const e=v&&v.querySelector('[data-endnotes]'); return e?e.textContent:''}")
    check(f'{tag} end notes: the billing note (input plus output per call, cached input cheaper), no negative disclaimer',
          Z(lang, 'a provider bills each call for its input tokens plus its output tokens', '服务商对每次调用收它的输入 token 加输出 token') in ends
          and not any(x in ends for x in ('not a recording', 'not counted here', 'is left out', '不是真实', '这里不算', '这里没有算')), ends[:300])
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#ana-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        pg.evaluate(f"()=>document.querySelector('#ana-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(200)
        check(f'{tag} popover {pid} opens', pg.evaluate(f"()=>{{const p=document.getElementById('{pid}'); return !!p&&p.getClientRects().length>0}}"))
        rules(pg, f'{tag} popover {pid}', lang)
        pg.evaluate(f"()=>document.querySelector('#ana-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(150)
    # ---- part 1: mission and prediction
    st = state(pg)
    check(f'{tag} part 1: the bug fix, guided part 1, nothing played', (st['run'], st['mode'], st['step'], st['pos']) == ('base', 'guided', 0, 0), st)
    check(f'{tag} part 1: Next disabled, with the kit’s reason under it', nexts(pg) and all(nexts(pg)) and set(why(pg)) == {Z(lang, 'Pick an answer to continue, or Skip.', '选一个答案才能继续，也可以跳过。')}, (nexts(pg), why(pg)))
    ba = before_answer(pg)
    check(f'{tag} part 1: the total (73,011) appears nowhere before the prediction (R3.8)', '73,011' not in ba and '73K' not in ba and '6.1 ' not in ba, [l for l in ba.split('\n') if '73' in l][:3])
    check(f'{tag} part 1: the question names 8 calls and about 12,000 tokens in the last', '12,000' in text(pg, '#ana-rail') and '8' in text(pg, '#ana-rail'))
    check(f'{tag} part 1: three options covering the range', pg.evaluate("()=>document.querySelectorAll('#ana-rail [data-pred]').length") == 3)
    if phone:
        fs = pg.evaluate("""()=>{const pb=document.querySelector('#ana-rail .pbar'), nx=pb&&pb.querySelector('[data-nav="next"]'), g=document.getElementById('ana-graph');
          const r=e=>e?e.getBoundingClientRect():null, a=r(pb), b=r(nx), c=r(g); return {bar:!!a&&a.top<innerHeight&&a.height>0, next:!!b&&b.bottom<=innerHeight&&b.height>0, g:c?Math.round(c.top):null, vh:innerHeight}}""")
        check(f'{tag} phone first screen: the parts’ bar with Next, and the picture starts on screen (R7.3)', fs['bar'] and fs['next'] and fs['g'] is not None and fs['g'] < fs['vh'] - 60, fs)
        shot(pg, f'{tag}_first_screen')
    every(pg, f'{tag} part 1', lang, f'{tag}_p1')
    click(pg, '#ana-rail [data-pred="small"]')
    check(f'{tag} part 1: Saved after a pick, Next enabled, the reason gone', Z(lang, 'Saved.', '已记下。') in text(pg, '#ana-rail') and not any(nexts(pg)) and why(pg) == [], (nexts(pg), why(pg)))
    click(pg, '#ana-rail [data-pred="mid"]')
    check(f'{tag} part 1: the prediction can still change', state(pg)['pred'] == 'mid')
    # ---- part 2: the replay auto-plays
    nav(pg, 'next')
    st = state(pg)
    check(f'{tag} part 2: the replay starts on its own (R3.9)', st['step'] == 1 and st['playing'], st)
    check(f'{tag} part 2: Next waits for the replay, with the reason', all(nexts(pg)) and set(why(pg)) == {Z(lang, 'Let the replay finish to continue (“Fast” takes a few seconds).', '等回放播完才能继续（选“快”只要几秒）。')}, why(pg))
    hs = set()
    for _ in range(8):
        hs.add(tuple(pg.evaluate(HEIGHTS))); pg.wait_for_timeout(260)
    check(f'{tag} part 2: the stage’s boxes keep their height during the replay (R7.9)', len(hs) == 1, hs)
    click(pg, '#ana-player [data-act="play"]', 100)
    p1 = state(pg)['pos']; pg.wait_for_timeout(1200); st = state(pg)
    check(f'{tag} part 2: Pause stops the replay', not st['playing'] and st['pos'] == p1 and 0 < p1 < 23, (p1, st['pos']))
    check(f'{tag} part 2: the total stays hidden during the replay', '73,011' not in text(pg, '#ana-stage'), text(pg, '#ana-bars-cap'))
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400); st = state(pg)
    check(f'{tag} part 2: a language switch keeps the paused replay paused (R3.20)', not st['playing'] and st['pos'] == p1 and st['step'] == 1, (st['playing'], st['pos'], p1))
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    click(pg, '#ana-player .seg-ctl[data-key="spd"] button[data-v="fast"]', 50)
    click(pg, '#ana-player [data-act="play"]', 50)
    t0 = time.time()
    for _ in range(6):
        hs.add(tuple(pg.evaluate(HEIGHTS))); pg.wait_for_timeout(150)
    try: pg.wait_for_function("()=>window.__ana.state().replayDone", timeout=9000)
    except Exception: pass
    dt = time.time() - t0
    check(f'{tag} part 2: Fast finishes the rest within 5 s', state(pg)['replayDone'] and dt < 5, dt)
    check(f'{tag} part 2: the heights stay the same at Fast too (R7.9)', len(hs) == 1, hs)
    pg.wait_for_timeout(400); settle(pg)
    o = obs(pg)
    check(f'{tag} part 2: the result: 8 replies, 7 tool calls run by your code, end_turn', Z(lang, 'In 8 calls the model replied 8 times', '8 次调用，模型回复了 8 次') in o and Z(lang, 'All 7 tool calls', '这 7 次工具调用全是你的代码执行的') in o and 'end_turn' in o, o[:300])
    check(f'{tag} part 2: the conversation shows every block, the last one the answer', pg.evaluate("()=>document.querySelectorAll('#ana-conv .ana-row').length") == 18, pg.evaluate("()=>document.querySelectorAll('#ana-conv .ana-row').length"))
    check(f'{tag} part 2: Next enabled', not any(nexts(pg)))
    every(pg, f'{tag} part 2', lang, f'{tag}_p2')
    # ---- part 3: the conversation is the memory
    nav(pg, 'next')
    check(f'{tag} part 3: Next waits for call 8, with the reason', all(nexts(pg)) and set(why(pg)) == {Z(lang, 'Step on to call 8 to continue.', '看到第 8 次调用，才能继续。')}, why(pg))
    slide(pg, '#ana-ic-call-in', 4)
    o = obs(pg)
    check(f'{tag} part 3: call 4 sends 9,441, of which 9,071 were already sent', '9,441' in o and '9,071' in o and '370' in o and all(nexts(pg)), o[:200])
    check(f'{tag} part 3: the slider’s output follows', Z(lang, 'call 4', '第 4 次') in text(pg, '#ana-ic-call'), text(pg, '#ana-ic-call'))
    check(f'{tag} part 3: stepping through the calls shows no ▲▼ (R6.6)', pg.evaluate("()=>[...document.querySelectorAll('#ana-metrics .d')].every(d=>!d.textContent.trim())"),
          pg.evaluate("()=>[...document.querySelectorAll('#ana-metrics .d')].map(d=>d.textContent)"))
    check(f'{tag} part 3: the conversation’s caption says what a row is', Z(lang, '(one row per block)', '（一行是一块）') in text(pg, '#ana-conv-cap'), text(pg, '#ana-conv-cap'))
    slide(pg, '#ana-ic-call-in', 8)
    o = obs(pg)
    check(f'{tag} part 3: call 8: 73,011 in all, 6.1 times the last call, the verdict, the formula', '73,011' in o and '6.1' in o and Z(lang, 'was right', '是对的') in o
          and pg.evaluate("()=>document.querySelectorAll('#ana-rail [data-obs] .ana-fx math').length") == 1 and not any(nexts(pg)), o[:300])
    check(f'{tag} part 3: the test file first goes out with call 2, 7 times in all', Z(lang, 'from call 2 on, 7 times in all', '从第 2 次调用起每次都发，一共发了 7 遍') in o, o[-400:])
    # the owner (2026-10-14): resending between calls against reading the KV cache inside one call, and what the bill counts
    kv = pg.evaluate("()=>{const d=[...document.querySelectorAll('#ana-rail [data-obs] .def')].find(e=>/KV cache/.test(e.textContent)); return d?d.getAttribute('data-tipx'):''}")
    check(f'{tag} part 3: the resending happens between calls; inside one call the input is computed once and its KV cache (with its tooltip) reused, at no extra charge; each call is charged per input and output token: 73,011 and 242',
          Z(lang, 'The resending happens between calls', '重发发生在两次调用之间') in o and Z(lang, 'the input is computed only once', '输入只算一遍') in o
          and Z(lang, 'charges for the input tokens and the output tokens, each at its own price', '按输入 token 和输出 token 各自的单价收费') in o
          and Z(lang, '73,011 input tokens and 242 output tokens', '输入共 73,011 个，输出 242 个') in o and 'key' in kv and 'prompt caching' in kv, (o[-500:], kv))
    mt = pg.evaluate("()=>{const d=document.querySelector('#ana-app details.math'); return d?d.textContent.replace(/\\s+/g,' '):''}")
    check(f'{tag} part 3: Show the math adds the tokens written (242), the bill and why the input is about 300 times the output (short replies, growing inputs)', Z(lang, 'Tokens written, and the bill.', '写出的 token 和账单。') in mt and '242' in mt
          and Z(lang, 'about 300 times the output: the model writes a reply of only a few dozen tokens each time', '约是输出的 300 倍：模型每次只写几十个 token 的回复') in mt, mt[-600:])
    check(f'{tag} part 3: the chart: 8 bars, the total in its caption', pg.evaluate("()=>document.querySelectorAll('#ana-bars .ana-col:not(.todo)').length") == 8 and '73,011' in text(pg, '#ana-bars-cap'), text(pg, '#ana-bars-cap'))
    if phone:
        check(f'{tag} part 3: the phone’s slider sits next to the chart', vis(pg, '#ana-ph-call'))
        pg.evaluate("()=>{const t=document.getElementById('ana-bars'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+80)}"); pg.wait_for_timeout(700)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, status:document.getElementById('dkStatus').textContent, read:document.getElementById('dkRead').textContent}}")
        check(f'{tag} part 3: the mini result bar with a short status and the result', dk['shown'] and '11,882' in dk['read'] and len(dk['status']) <= 12, dk)
        shot(pg, f'{tag}_p3_dock')
        rules(pg, f'{tag} part 3 dock', lang)
    every(pg, f'{tag} part 3', lang, f'{tag}_p3')
    # ---- part 4: when a tool fails
    nav(pg, 'next')
    check(f'{tag} part 4: Next waits for the other two ways, with the reason', all(nexts(pg)) and set(why(pg)) == {Z(lang, 'Try the other two ways to continue.', '另外两种做法都试一下，才能继续。')}, why(pg))
    check(f'{tag} part 4: the failed test is marked is_error in red', pg.evaluate("()=>document.querySelectorAll('#ana-conv .ana-row.err').length") == 1)
    click(pg, '#ana-ic-fx [data-lever="stop"]')
    o = obs(pg)
    check(f'{tag} part 4: stop on error: the task ends after call 5, not done', Z(lang, 'the task ends after call 5', '任务在第 5 次调用后结束') in o and 'bad' in pg.evaluate("()=>document.getElementById('ana-status').className")
          and pg.evaluate("()=>!!document.querySelector('#ana-conv .ana-stoprow')") and all(nexts(pg)), o[:200])
    click(pg, '#ana-ic-fx [data-lever="terse"]')
    o = obs(pg)
    pressed = pg.evaluate("()=>[...document.querySelectorAll('#ana-ic-fx [data-lever]')].map(b=>b.getAttribute('aria-pressed'))")
    res = pg.evaluate("()=>[...document.querySelectorAll('#ana-ic-fx [data-lever] .lv-r')].map(e=>e.textContent)")
    check(f'{tag} part 4: single choice, all three results on their options', pressed == ['false', 'false', 'true'] and all(res), (pressed, res))
    check(f'{tag} part 4: only “exit code 1”: fixed at call 10 instead of 8, 18,109 more tokens, Next enabled', Z(lang, 'fixes it at call 10 instead of 8', '第 10 次调用才修好，而不是第 8 次') in o and '18,109' in o and not any(nexts(pg)), o[:300])
    every(pg, f'{tag} part 4', lang, f'{tag}_p4')
    # ---- part 5: where the time goes
    nav(pg, 'next')
    check(f'{tag} part 5: Next waits for three at once, with the reason', all(nexts(pg)) and set(why(pg)) == {Z(lang, 'Pick “Three tools in one call” to continue.', '选“一次要 3 个”才能继续。')}, why(pg))
    check(f'{tag} part 5: the time line and the assumed speeds are there', vis(pg, '#ana-time') and vis(pg, '#ana-ctl-dec') and '22.8' in text(pg, '#ana-time-cap'), text(pg, '#ana-time-cap'))
    click(pg, '#ana-ic-par [data-lever="par"]')
    o = obs(pg)
    check(f'{tag} part 5: three at once: 6 calls instead of 8, 20.2 s instead of 22.8 s, 58,290 tokens, Next enabled', '20.2' in o and '22.8' in o and '58,290' in o and not any(nexts(pg))
          and pg.evaluate("()=>document.querySelectorAll('#ana-time .t-tool.par').length") == 3, o[:300])
    pm = pg.evaluate("()=>({w:[...document.querySelectorAll('#ana-time .t-tool.par')].map(e=>e.getBoundingClientRect().width), note:(document.querySelector('#ana-time .ana-tnote')||{}).textContent||''})")
    check(f'{tag} part 5: the three tools run at once are visible (≥ 10 px) and labelled (R5.10)', len(pm['w']) == 3 and min(pm['w']) >= 9.5 and Z(lang, '3 tools at once', '3 个工具同时跑') in pm['note'], pm)
    check(f'{tag} part 5: the slow-test caveat (in this lab’s assumptions)', Z(lang, 'if one test run takes 10 s, the tools take more time than the model calls', '如果跑一遍测试要 10 秒，工具花的时间就会比模型调用还多') in o, o[-300:])
    click(pg, '#ana-controls .seg-ctl[data-key="tt"] button[data-v="10"]')
    check(f'{tag} part 5: a slower test run moves the time line', '34.2' in text(pg, '#ana-time-cap'), text(pg, '#ana-time-cap'))
    click(pg, '#ana-controls .seg-ctl[data-key="tt"] button[data-v="3"]')
    every(pg, f'{tag} part 5', lang, f'{tag}_p5')
    # ---- part 6: what to remember
    nav(pg, 'next')
    rt = text(pg, '#ana-rail')
    check(f'{tag} part 6: four points, three misconceptions, See the scenarios, the next-lab footer', pg.evaluate("()=>document.querySelectorAll('#ana-rail .ana-remlist li').length") == 4
          and rt.count('✗') == 3 and pg.evaluate("()=>!!document.querySelector('#ana-rail [data-mode-go=\"sandbox\"]')") and pg.evaluate("()=>!!document.querySelector('#ana-rail .nextlab, #ana-rail [class*=next]')"), rt[-300:])
    check(f'{tag} part 6 (G2, the owner 2026-10-08): the misconception states the pattern, not part 3’s calls', Z(lang, 'Whatever the model has seen has to be sent again in every later call.', '模型看过的内容，后面每次调用都要再发一遍。') in rt and not any(x in rt for x in ('In part', '第 2 部分里', '第 3 部分里', '第 4 部分里', '第 5 部分里')), rt[-500:])
    for k in ('memory', 'fail', 'time', 'who'):
        click(pg, f'#ana-rail [data-rem="{k}"]')
        st = state(pg)
        check(f'{tag} part 6: “show in the chart” {k}', st['remView'] == k and pg.evaluate(f"()=>document.querySelector('#ana-rail [data-rem=\"{k}\"]').getAttribute('aria-pressed')") == 'true', st)
    every(pg, f'{tag} part 6', lang, f'{tag}_p6')
    # skip on part 1 from a fresh page
    pg.close()
    pg = page(b, w, h, lang, dark)
    nav(pg, 'skip')
    check(f'{tag} part 1: Skip goes on without a prediction', state(pg)['step'] == 1 and state(pg)['pred'] is None)
    pg.close()


def run_watched(b, w, h, lang):
    """R3.20: part 2 shows a run already watched to the end in Scenarios finished at once (and its gate passes); a run not watched
    yet still plays on entering part 2."""
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    click(pg, '#ana-rail [data-pred="mid"]')
    mode(pg, 'sandbox'); click(pg, '#ana-rail [data-preset="0"]')   # the call limit: another run than part 2's
    # the owner, 2026-10-08 (G1, G2) scen: the Scenarios card leads in with what to pick and what to watch; no “one at a time” note is left
    _rt = pg.evaluate("s=>document.querySelector(s).innerText", '#ana-rail')
    _lead = Z(lang, 'Pick a scenario and see how many tokens each call sends.', '选一个场景，看看每次调用发出多少 token。')
    check(f'{tag} G1: the Scenarios card leads in “{_lead[:30]}…”', _lead in _rt, _rt[:300])
    check(f'{tag} G1: no “one at a time / 一次只能选 / 不会叠加” note in the Scenarios card', not any(x in _rt for x in ('一次只能选', '不会叠加', '一次选一个', 'do not stack', 'one at a time', 'One at a time')), _rt[:300])
    # end G1 G2 scen
    click(pg, '#ana-player .seg-ctl[data-key="spd"] button[data-v="fast"]', 50); click(pg, '#ana-player [data-act="play"]', 50)
    try: pg.wait_for_function("()=>!window.__ana.state().playing", timeout=12000)
    except Exception: pass
    mode(pg, 'guided'); nav(pg, 'next'); st = state(pg)
    check(f'{tag} R3.20 part 2: a run not watched yet still plays on entering (Scenarios played another)', st['step'] == 1 and st['playing'] and not st['replayDone'], st)
    mode(pg, 'sandbox'); st = state(pg)
    check(f'{tag} R3.20 Scenarios: the bug fix shown finished, nothing playing', st['run'] == 'base' and st['lim'] == 0 and not st['playing'], st)
    click(pg, '#ana-player [data-act="play"]', 50)   # ↺ Replay: the reader watches the same run to the end here
    try: pg.wait_for_function("()=>!window.__ana.state().playing", timeout=12000)
    except Exception: pass
    mode(pg, 'guided'); st = state(pg)
    check(f'{tag} R3.20 part 2: the run already watched to the end in Scenarios is shown finished, with no replay, and Next is open', st['step'] == 1 and not st['playing'] and st['replayDone'] and st['pos'] == 23 and not any(nexts(pg)), (st['step'], st['playing'], st['replayDone'], st['pos'], nexts(pg)))
    pg.wait_for_timeout(600)
    check(f'{tag} R3.20 part 2: still nothing playing a moment later', not state(pg)['playing'] and state(pg)['pos'] == 23)
    pg.close()


def run_modes(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    mode(pg, 'sandbox')
    check(f'{tag} scenarios: four', pg.evaluate("()=>document.querySelectorAll('#ana-rail [data-preset]').length") == 4)
    want = [('5', '8'), ('58,290', '73,011'), ('447,828', '18.3'), ('40,000', '187,635')]
    for i in range(4):
        click(pg, f'#ana-rail [data-preset="{i}"]')
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#ana-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        ins = text(pg, '#ana-insight')
        check(f'{tag} scenario {i}: pressed alone, with its result', pressed.count('true') == 1 and pressed[i] == 'true' and all(x in ins for x in want[i]), (pressed, ins[:160]))
        every(pg, f'{tag} scenario {i}', lang, f'{tag}_scenario_{i}' if i in (0, 2) else None)
    check(f'{tag} scenario full logs: the run ends, 8 calls', pg.evaluate("()=>window.__ana.state().run") == 'huge')
    click(pg, '#ana-rail [data-preset="0"]')
    check(f'{tag} scenario call limit: stopped, red status, the stop row', 'bad' in pg.evaluate("()=>document.getElementById('ana-status').className") and pg.evaluate("()=>!!document.querySelector('#ana-conv .ana-stoprow')"))
    sr = pg.evaluate("()=>{const e=document.querySelector('#ana-conv .ana-stoprow'); return e?e.textContent:''}")
    check(f'{tag} scenario call limit: the stop row in the page’s language (R8.3)', Z(lang, 'it has made 5 calls, the limit', '已经调用了 5 次，到了上限') in sr and (lang == 'en' or 'gave up' not in sr), sr)
    check(f'{tag} scenarios: no ▲▼ against the scenario clicked before (R6.6)', pg.evaluate("()=>[...document.querySelectorAll('#ana-metrics .d')].every(d=>!d.textContent.trim())"))
    hsh = pg.evaluate("()=>location.hash")
    check(f'{tag} scenarios: the URL carries the setup', hsh.startswith('#agent-anatomy~'), hsh)
    pg2 = page(b, w, h, lang, hash_=hsh[1:])
    st = state(pg2)
    check(f'{tag} hash round-trip: Scenarios, the call limit pressed', st['mode'] == 'sandbox' and (st['run'], st['lim']) == ('base', 5)
          and pg2.evaluate("()=>document.querySelector('#ana-rail [data-preset=\"0\"]').getAttribute('aria-pressed')") == 'true', st)
    pg2.close()
    click(pg, '#ana-controls .seg-ctl[data-key="lim"] button[data-v="10"]')
    pressed = pg.evaluate("()=>[...document.querySelectorAll('#ana-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: a change in Settings un-presses the scenario', pressed.count('true') == 0 and Z(lang, 'match none of these scenarios', '和这几个场景都不一样') in text(pg, '#ana-insight'), pressed)
    every(pg, f'{tag} sandbox settings', lang)
    bh = pg.evaluate("()=>[...document.querySelectorAll('#ana-ctl-run button')].filter(b=>b.getClientRects().length).map(b=>Math.round(b.getBoundingClientRect().height))")
    check(f'{tag} settings: the four task buttons keep one line each, no orphaned character (R7.8)', len(bh) in (0, 4) and (not bh or max(bh) - min(bh) <= 1), bh)
    # ---- the challenge
    mode(pg, 'challenge')
    st = state(pg)
    check(f'{tag} challenge: starts at limit 50, budget 300,000', (st['chO']['lim'], st['chO']['bud']) == (50, 300000), st)
    check(f'{tag} challenge: Settings locked', pg.evaluate("()=>document.getElementById('ana-controls').classList.contains('locked')") and vis(pg, '#ana-lock'))
    ct = pg.evaluate("()=>{const t=[...document.querySelectorAll('#ana-chart svg text')]; return {h:Math.min(...t.map(e=>e.getBoundingClientRect().height)), txt:t.map(e=>e.textContent)}}")
    check(f'{tag} challenge: the chart’s labels are readable (≥ 10 px), both axes named, 0 not 0K (R7.4, R5.3)', ct['h'] >= 10 and Z(lang, 'model call', '第几次调用') in ct['txt'] and Z(lang, 'tokens read so far', '累计读入的 token') in ct['txt']
          and not any(x.endswith('K') for x in ct['txt']) and '0' in ct['txt'], ct)
    check(f'{tag} challenge: plain statement and the assumptions', Z(lang, 'Set a call limit and a token budget', '请设一个调用次数上限和一个 token 预算') in text(pg, '#ana-rail') and pg.evaluate("()=>document.querySelectorAll('#ana-rail .ana-asm li').length") == 2)
    # the owner (2026-10-14): when the token budget wraps to a second row, something divides the rows. Since the 2026-10-22 refit an option
    # group is a tray of keys (R7.13): the keys sit 3px apart and the tray's colour, unlike the keys', shows between the rows
    sep = pg.evaluate("""()=>{const out={}; for(const k of ['chl','chb']){const g=document.querySelector('#ana-rail .seg-ctl[data-key="'+k+'"]'); const bs=[...g.querySelectorAll('button')], r=bs.map(b=>b.getBoundingClientRect());
      const rows=[...new Set(r.map(x=>Math.round(x.top)))].sort((a,b)=>a-b), bg=getComputedStyle(g).backgroundColor, key=getComputedStyle(bs.find(b=>b.getAttribute('aria-pressed')!=='true')||bs[0]).backgroundColor;
      out[k]={rows:rows.length, gaps:rows.slice(1).map((t,j)=>+(Math.min(...r.filter(x=>Math.round(x.top)===t).map(x=>x.top))-Math.max(...r.filter(x=>Math.round(x.top)===rows[j]).map(x=>x.bottom))).toFixed(2)), bg, key,
        tray:!/rgba\\(.*,\\s*0\\)$|transparent/.test(bg)&&bg!==key};} return out;}""")
    check(f'{tag} challenge: the rows of the call limit and the token budget stay apart when they wrap (keys on a tray)', all(v['tray'] and all(2 <= x <= 4.5 for x in v['gaps']) for v in sep.values()), sep)
    every(pg, f'{tag} challenge start', lang, f'{tag}_challenge_start')
    click(pg, '#ana-rail .seg-ctl[data-key="chb"] button[data-v="100000"]')
    click(pg, '#ana-rail [data-act="submit"]')
    r = text(pg, '#ana-ch-res')
    check(f'{tag} challenge tier 3 (budget 100,000): not yet, a hint, Show the answer', Z(lang, 'Not yet', '还没达到') in r and Z(lang, 'Hint', '提示') in r and vis(pg, '#ana-ch-res [data-act="reveal"]')
          and text(pg, '#ana-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), r[:200])
    click(pg, '#ana-ch-res [data-act="reveal"]')
    bf = brief(pg)
    check(f'{tag} challenge: Show the answer draws 7 rows, the 6 settings that let the normal run finish (the initial setup among them) and the pick, 9 left out', brief_ok(lang, bf, 7, 9), bf)
    click(pg, '#ana-rail .seg-ctl[data-key="chb"] button[data-v="300000"]')
    r = text(pg, '#ana-ch-res')
    check(f'{tag} challenge tier 2 (initial setup): goal met, but a tighter setting keeps the limit', Z(lang, 'Goal met', '达成目标') in r and Z(lang, 'a tighter setting also works', '还有更省的设置') in r
          and Z(lang, 'limit 50 calls, budget 150,000 tokens', '上限 50 次，预算 150,000 个 token') in r and '284,291' in r, r[:300])
    rows = pg.evaluate("()=>{const c=document.querySelector('#ana-ch-res .cmp'); if(!c) return null; const lab=[...c.querySelectorAll('*')].filter(e=>/^✓ /.test(e.textContent)&&e.children.length===0); return {ok:lab.length, all:c.innerText.split('\\n').filter(l=>/(budget|预算)/.test(l)).length}}")
    bf = brief(pg)
    check(f'{tag} challenge: 6 rows, the settings that let the normal run finish (the initial setup is the pick), no other setting, 10 left out', rows and rows['ok'] == 6 and rows['all'] == 6 and brief_ok(lang, bf, 6, 10), (rows, bf))
    check(f'{tag} challenge tier 2: the first line says that a tighter setting exists', Z(lang, 'But a tighter setting also works', '不过还有更省的设置') in text(pg, '#ana-ch-res > p:first-child'), text(pg, '#ana-ch-res > p:first-child'))
    click(pg, '#ana-rail .seg-ctl[data-key="chb"] button[data-v="150000"]')
    r = text(pg, '#ana-ch-res')
    check(f'{tag} challenge tier 1 (limit 50, budget 150,000): goal met, and no setting stops the stuck run sooner', Z(lang, 'no setting that lets the normal run finish stops the stuck one sooner', '这已经是最省的') in r and '138,675' in r, r[:300])
    tags = text(pg, '#ana-ch-res .cmp')
    check(f'{tag} challenge: the initial setup and the pick tagged', Z(lang, 'initial setup', '初始设置') in tags and Z(lang, 'your pick', '你的选择') in tags)
    bf = brief(pg)
    check(f'{tag} challenge tier 1: the chart still holds the 6 settings that let the normal run finish, 10 left out', brief_ok(lang, bf, 6, 10), bf)
    check(f'{tag} challenge: the capability card', vis(pg, '#ana-rail [data-exits]'))
    xt = pg.evaluate("""()=>{const g=document.querySelector('#ana-rail .ex-tabs'); if(!g) return null; const bs=[...g.querySelectorAll('button')];
      return {bg:getComputedStyle(g).backgroundColor, key:getComputedStyle(bs.find(b=>b.getAttribute('aria-pressed')!=='true')||bs[0]).backgroundColor, gap:getComputedStyle(g).rowGap, br:bs.map(x=>getComputedStyle(x).borderRightWidth)};}""")
    check(f'{tag} challenge: the capability card’s tabs keep their rows apart when they wrap (keys 2–4px apart on a tray of another colour, no lines drawn in the keys)',
          xt is not None and xt['bg'] != xt['key'] and 2 <= float(xt['gap'].rstrip('px') or 0) <= 4 and set(xt['br']) == {'0px'}, xt)
    every(pg, f'{tag} challenge solved', lang, f'{tag}_challenge_solved')
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} challenge: a language switch keeps the plan and the result', (st['chO']['lim'], st['chO']['bud']) == (50, 150000) and st['chSubmitted'] and vis(pg, '#ana-ch-res .cmp'), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    mode(pg, 'guided')
    check(f'{tag} back to Guided', state(pg)['mode'] == 'guided')
    every(pg, f'{tag} guided again', lang)
    pg.close()


# B4 (polish round 5, reviews/r5-fit-agents-eval.md): entering parts 1, 2, 3, 4 and 6 at 1440 × 900 the way the polish probe stage.py does
# (Next when it is enabled, else the part's heading), the stage needs no scrolling inside: Chinese with at least 30 px to spare,
# English with none (spare = the stage's max-height minus what its content and borders need). Its top is not under the top bar,
# and the main picture (the stage's largest svg or canvas, if it has one) is whole on the screen.
# The conversation's own box (#ana-conv, 176 px, kept scrolled to its newest row) scrolls by design and is not the stage.
# The top bar's bottom is where it is on the screen now: on a desktop lab page the bar scrolls away (refit step 2), on a phone it
# stays on top and this is its height.
B4_FIT = """(id)=>{const s=document.getElementById(id), r=s.getBoundingClientRect(), b=document.querySelector('.topbar'), tb=b?Math.max(0,Math.round(b.getBoundingClientRect().bottom)):0;
  const cur=document.querySelector('#'+id.replace(/-stage$/,'-rail')+' .steps>.step.current');
  const f=[...s.querySelectorAll('svg,canvas')].filter(e=>e.getClientRects().length&&!e.closest('button,.btn')).map(e=>e.getBoundingClientRect()).sort((a,c)=>c.width*c.height-a.width*a.height)[0];
  return {part:cur?[...cur.parentNode.children].indexOf(cur)+1:0, spare:Math.round(parseFloat(getComputedStyle(s).maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight)),
    scroll:s.scrollHeight-s.clientHeight, top:Math.round(r.top), bar:tb, fig:f?[Math.round(f.top),Math.round(f.bottom)]:null, figOn:!f||(f.top>=tb-2&&f.bottom<=innerHeight+2)};}"""


def b4_fit(b, lang, parts=(1, 2, 3, 4, 6)):
    pg = page(b, 1440, 900, lang)
    for k in range(1, max(parts) + 1):
        if k > 1:
            nb = pg.locator('.pbar [data-nav="next"]:visible:not([disabled])')
            if nb.count(): nb.first.click()
            else: pg.evaluate(f"()=>document.querySelector('#ana-rail .steps>.step:nth-child({k}) .step-h').click()")
            pg.wait_for_timeout(900)
        if k not in parts: continue
        y = None
        for _ in range(20):   # the page's scroll has settled (the rail's reveal, then the lab's keepStage)
            y2 = pg.evaluate('()=>scrollY')
            if y2 == y: break
            y = y2; pg.wait_for_timeout(150)
        f = pg.evaluate(B4_FIT, 'ana-stage')
        check(f'{lang}-1440 part {k}: entering it, the stage fits without scrolling inside{", 30 px to spare" if lang == "zh" else ""}; its top is below the top bar and its main picture whole on the screen (B4, R7.12)',
              f['part'] == k and f['scroll'] <= 0 and f['spare'] >= (30 if lang == 'zh' else 0) and f['top'] >= f['bar'] - 1 and f['figOn'], f)
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    if (1440, 900) in SIZES:
        for lang in LANGS: b4_fit(b, lang)
    for (w, h) in SIZES:
        for lang in LANGS:
            run_guided(b, w, h, lang)
            run_modes(b, w, h, lang)
            run_watched(b, w, h, lang)
    if FULL:
        run_guided(b, 1440, 900, 'en', dark=True)
    b.close()
check('no hover outside Show the math only expands a number to its exact value (“精确值是” / “Exactly”; the owner, 2026-10-07 23:37)', not [h for h in rule_hits if 'exact value' in h], [h for h in rule_hits if 'exact value' in h][:3])
for hit in rule_hits[:60]:
    check('writing rules: ' + hit, False)
for hit in fit_hits[:60]:
    check(hit, False)
# the owner, 2026-10-09 (H1, H2): every formula typeset as MathML (A.ui.math), never a plain-text string; Show the math in short blocks,
# a hairline (1px, the kit's line colour) between them, each chain's “=” in one column, nothing clipped (R6.33)
_H1_LAB, _H1_PFX = 'agent-anatomy', 'ana'
_H1_PLAIN = ['÷', '√']          # plain-text formula pieces that must not be left in the rendered text (outside <math>, <code>, <pre>, <svg>)
_H1_SRC_GONE = ['t<sub>0</sub>', 'N<sub>in</sub>', 'v<sub>prefill</sub>', 'input<sub>i</sub>', '输入<sub>i</sub>', '÷ 2 = ${fint(d.total)}']     # plain-text formulas converted on 2026-10-09: gone from the lab's source (comments aside)
_H1_SRC_NEW = ['ana-mb', 'ana-eqs', 'time of one call', '一次调用的时间']       # sentences or markup written on 2026-10-09
_H1_PROBE = """([pfx,plain])=>{const app=document.querySelector('#view-dyn:not([hidden])')||document; const mb=document.getElementById(pfx+'-mathbody');
  const out={nmath:0,words:[],plain:[],blocks:0,sep:true,clip:[],page:[],wide:[]};
  const TEXW=/^(frac|sqrt|left|right|text|hat|cdot|times|sum|quad|ldots|approx|mathrm|big|Big)$/;
  const scanText=(root,arr)=>{const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n; while((n=tw.nextNode())){const e=n.parentElement;
    if(!e||e.closest('math,code,pre,svg,script,style,#tip')||!e.getClientRects().length) continue; for(const p of plain) if(n.textContent.includes(p)) arr.push(p+' in “'+n.textContent.trim().slice(0,70)+'”');}};
  if(mb){const d=mb.closest('details'); if(d) d.open=true;
    out.nmath=mb.querySelectorAll('math').length;
    mb.querySelectorAll('mi,mtext').forEach(x=>{if(x.tagName.toLowerCase()==='mi'&&TEXW.test(x.textContent.trim())) out.words.push(x.textContent);});
    scanText(mb,out.plain);
    const bl=[...mb.querySelectorAll('.'+pfx+'-mb')]; out.blocks=bl.length;
    bl.slice(1).forEach(b=>{const cs=getComputedStyle(b); if(!(parseFloat(cs.borderTopWidth)>=1&&cs.borderTopStyle==='solid')) out.sep=false;});
    mb.querySelectorAll('.'+pfx+'-eqs').forEach(q=>{if(q.scrollHeight>q.clientHeight+1) out.clip.push(q.textContent.slice(0,40)); if(q.scrollWidth>q.clientWidth+1) out.wide.push(q.scrollWidth+' > '+q.clientWidth+': '+q.textContent.slice(0,30));
      const k=[...q.children]; for(let i=1;i<k.length;i+=3){const a=k[i].getBoundingClientRect(), b=k[i+3]&&k[i+3].getBoundingClientRect(); if(b&&Math.abs(a.left-b.left)>1) out.clip.push('“=” not in one column: '+q.textContent.slice(0,40));}});}
  scanText(app,out.page);
  app.querySelectorAll('mi').forEach(x=>{if(TEXW.test(x.textContent.trim())) out.words.push(x.textContent);});
  return out;}"""
# H5 (the owner, 2026-10-09): the stage's blocks, each with a 1px hairline (var(--line)) on top when another of them sits above it, none on the first or a lone one
_H5_SEP = ['#ana-loopbox', '#ana-convbox', '#ana-barsbox', '#ana-timebox', '#ana-chbox']
_H5_PROBE = """(sels)=>{const els=[...new Set(sels.flatMap(s=>[...document.querySelectorAll('#view-dyn '+s)]))].filter(e=>e.getClientRects().length&&e.offsetHeight>0);
  const out={bad:[],lines:0}; for(const e of els){const r=e.getBoundingClientRect(), above=els.some(o=>o!==e&&o.parentElement===e.parentElement&&o.getBoundingClientRect().bottom<=r.top+1);
    const cs=getComputedStyle(e), line=parseFloat(cs.borderTopWidth)>=1&&cs.borderTopStyle!=='none'; if(line) out.lines++;
    if(line!==above) out.bad.push((e.id||e.className)+(above?': no hairline under the block above':': a hairline with no block above'));} return out;}"""
def _h1_states(b, w, h, lang):
    pg = page(b, w, h, lang)
    rail = '#view-dyn aside.rail'
    nparts = pg.evaluate("r=>new Set([...document.querySelectorAll(r+' [data-go]')].map(e=>e.dataset.go)).size", rail)
    got = []
    for k in range(nparts):
        pg.evaluate("([r,k])=>{const e=document.querySelector(r+' [data-go=\"'+k+'\"]'); if(e) e.click();}", [rail, k]); pg.wait_for_timeout(2600)
        try: pg.evaluate("()=>{for(const b of document.querySelectorAll('#view-dyn [data-speed=\"fast\"], #view-dyn .seg-ctl[data-key=\"speed\"] button[data-v=\"fast\"]')) b.click();}")
        except Exception: pass
        pg.wait_for_timeout(1400)
        got.append((f'part {k + 1}', {**pg.evaluate(_H1_PROBE, [_H1_PFX, _H1_PLAIN]), 'h5': pg.evaluate(_H5_PROBE, _H5_SEP)}))
    for m in ('sandbox', 'challenge'):
        pg.evaluate("m=>{const b=[...document.querySelectorAll('#view-dyn .modes button[data-mode=\"'+m+'\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}", m); pg.wait_for_timeout(3500)
        got.append((m, {**pg.evaluate(_H1_PROBE, [_H1_PFX, _H1_PLAIN]), 'h5': pg.evaluate(_H5_PROBE, _H5_SEP)}))
    dr = pg.evaluate("p=>{const b=document.querySelector('#view-dyn [data-act=\"drill-reveal\"]'); if(!b) return null; b.click(); return true}", _H1_PFX); pg.wait_for_timeout(300)
    drill = pg.evaluate("p=>{const a=document.getElementById(p+'-drill-a'); if(!a||a.hidden) return null; return {n:a.querySelectorAll('math').length, t:[...a.querySelectorAll('p')].map(x=>{const c=x.cloneNode(true); c.querySelectorAll('math').forEach(m=>m.remove()); return c.textContent}).join(' ')}}", _H1_PFX) if dr else None
    pg.close()
    return got, drill
with sync_playwright() as _p:
    _b = _p.chromium.launch()
    for (_w, _h, _lang) in [(1440, 900, 'en'), (390, 844, 'zh')]:
        if str(_w) not in str(SIZES) or _lang not in LANGS: continue
        _got, _drill = _h1_states(_b, _w, _h, _lang)
        _with = [(s, o) for s, o in _got if o['nmath']]
        check(f'H1 {_w} {_lang}: Show the math has typeset formulas (MathML) in {len(_with)} of {len(_got)} states', len(_with) >= 2, [s for s, o in _got])
        for s, o in _got:
            for x in o.get('wide', []): print(f'INFO H2 {_w} {_lang} {s}: a chain wider than its box, it scrolls sideways: {x}')
            check(f'H1 {_w} {_lang} {s}: no TeX command shown as a word', not o['words'], o['words'][:5])
            check(f'H1 {_w} {_lang} {s}: no plain-text formula left in Show the math', not o['plain'], o['plain'][:4])
            check(f'H1 {_w} {_lang} {s}: no plain-text formula left on the page', not o['page'], o['page'][:4])
            check(f'H2 {_w} {_lang} {s}: a hairline between the blocks of Show the math ({o["blocks"]} blocks)', o['sep'] and (o['blocks'] >= 1 or not o['nmath']), o)
            check(f'H2/R6.33 {_w} {_lang} {s}: no formula box clipped, every “=” of a chain in one column', not o['clip'], o['clip'][:3])
            check(f'H5 {_w} {_lang} {s}: a hairline between the stage’s blocks, none above the first ({o["h5"]["lines"]} hairlines)', not o['h5']['bad'], o['h5']['bad'][:3])
        check(f'H5 {_w} {_lang}: the stage separates its blocks with a hairline in {sum(1 for s, o in _got if o["h5"]["lines"])} of {len(_got)} states', any(o['h5']['lines'] for s, o in _got), [(s, o['h5']) for s, o in _got])
        if _drill is not None:
            check(f'H1 {_w} {_lang}: the drill answer typesets its working', _drill['n'] >= 1 and not any(x in _drill['t'] for x in _H1_PLAIN), _drill)
    _b.close()
import re as _re
_h1src = _re.sub(r'/\*.*?\*/', '', (P / 'labs' / f'{_H1_LAB}.js').read_text() + (P / 'teasers' / f'{_H1_LAB}.js').read_text(), flags=_re.S)
for _x in _H1_SRC_GONE:
    check(f'H1 src: the plain-text formula “{_x}” is gone', _x not in _h1src)
check('H5 src: the stage hairline rule is in the lab’s CSS', 'H5 (the owner, 2026-10-09)' in (P / 'labs' / f'{_H1_LAB}.css').read_text() and '1px solid var(--line)' in (P / 'labs' / f'{_H1_LAB}.css').read_text())
for _x in _H1_SRC_NEW:
    check(f'H1/H2 src: present: {_x[:70]}', _x in _h1src)
# end H1 H2
check('no page errors', not errs, errs[:5])
# the owner, 2026-10-08 (G1, G2) src: every option group leads in with what to pick and what to watch (G1); no sentence leans on an earlier part's detail (G2)
_src = (P / 'labs' / 'agent-anatomy.js').read_text() + (P / 'teasers' / 'agent-anatomy.js').read_text()
for _x in ('一次只能选', '不会叠加', '一次选一个', 'do not stack', 'one at a time', 'One at a time'):
    check(f'G1: no “{_x}” left in the lab or its tile', _x not in _src)
for _x in ['<p>工具出错时，你的代码有三种做法。选一种，看看 bug 最后修没修好。</p>', '<p>第 1 到 3 次调用要的三样东西，互不依赖。选一种要工具的方式，看看整个任务要花多少秒。</p>', 'Pick one and see whether the bug gets fixed.', 'Pick a way to ask for them and see how long the whole task takes.', '模型发出的只是 tool_use，也就是工具名和参数。每个工具都在你的代码里执行，所以发什么给模型、循环什么时候停，都由你的代码决定。', '每次调用都把前面的内容再发一遍。所以调用越多，读进去的 token 总数涨得越快，远远超过最后那次对话的长度。', '报错告诉模型哪个测试没过、差了多少，这正是它改代码需要的。一出错就停下，等于在快要修好的时候放弃。', '每次模型调用都很慢，所以少一次调用，省下的时间最多。互不依赖的工具一次要齐，能省掉几次调用，工具也能同时跑。', '模型看过的内容，后面每次调用都要再发一遍。', '所以同一个测试结果，你的代码交回去的方式不同，任务的结局也就不同。', '最后那次对话只占总数的一小部分。', '模型那边什么都没执行：每一次工具调用，都是你的代码在两次模型调用之间执行的。', '在图里显示修 bug 的整次运行。', '在图里显示修 bug 的第 8 次调用，和每次调用的柱子。', '在图里显示有一个测试没通过的运行，并标出这个测试。', '在图里显示第 1 次调用就要 3 个工具时的时间线。']:
    check(f'G1/G2: the new sentence is there: {_x[:60]}', _x in _src)
for _x in ['在图里显示第 2 部分', '在图里显示第 3 部分', '在图里显示第 4 部分', '在图里显示第 5 部分', '第 2 部分里，${nt}', '第 3 部分里，最后一次', '第 4 部分里，模型读到', '第 5 部分里，模型调用占了', '第 3 部分里，测试文件', '第 4 部分里，同一个测试结果', '第 3 部分里，最后一次只占', '第 2 部分里，每一次工具调用', 'Shows part 2’s', 'Shows part 3 at', 'Shows part 4’s', 'Shows part 5’s', 'In part 2, all', 'In part 3, the last call', 'In part 4, the model read', 'In part 5, model calls', 'In part 3, call ${tf.first}', 'In part 4, the same test result', 'In part 3, the last call was only', 'in part 2 every tool call']:
    check(f'G2: the old sentence is gone: {_x[:60]}', _x not in _src)
# end G1 G2 src
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:60] if fails else 'none')
sys.exit(1 if fails else 0)
