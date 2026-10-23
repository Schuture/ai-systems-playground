"""Flow test for the post-capstone lab (CONTRACT §6).

Every guided part and every mode at 1440×900 and 390×844, in EN and ZH (plus one dark run when run in full):
- no page errors, no horizontal overflow, no text wider than its box (R7.8), a tooltip for every help cursor (R1.1), the writing rules
  on every visible text node and tooltip (R6.14, R6.7, Chinese spacing and punctuation, two colons in a sentence, no 亿/万, numbers
  over 6 digits), colours (R5.6);
- part 1: three task cards, each with its data, metric and target; the strip shows only the data (no method judged yet);
- part 2: the methods defined before they are offered (R2.10); SFT on the preference task can run on the chosen answers (green
  chip “prompt → chosen answer”), and the card separates running from reaching the target; DPO on the support task cannot run;
  Next waits for three task and method pairs, with its reason;
- part 3: the prediction gate with its reason; before the pick the answer (about 15 times, 1.15 and 0.078 GPU hours) appears
  nowhere, in text, tooltips or the header's popovers (R3.8); a wrong pick is told the right answer; the two cost bars;
- part 4: 8B full 150.6 GB (red, the part over 80 GB drawn), LoRA 22.8 GB, 70B LoRA 172.1 GB, QLoRA 70.5 GB; Next waits for QLoRA;
- part 5: 83.5% → 94.4% (below) → 97.1% (met, the bottom of the error bar 94.8% still below) → 97.8% (95.7%, clear);
- part 6: DPO 0.55, reward model + PPO 2.32 (4.2 times), GRPO with a reward model 4.04 GPU hours, over the budget (red);
- part 7: three points, three misconceptions, the button into the scenarios;
- R3.20 for a lab with no replay: picking the option that is already pressed changes nothing (no bump, same bar widths), and Back
  shows the part as it was left, with nothing moving;
- Scenarios: five, single choice, the first pressed on entry; each shows its own numbers; the URL carries the setup and a fresh
  page restores it; a change in Settings releases the scenario buttons; on a phone a one-line task with a link down to the card;
- the challenge (review codex-b9v3 item 8): Settings locked; the start (SFT, full) can run on the final answers but does not fit; SFT
  with LoRA passes data, memory and budget, misses the target in red (“the same as before training”) and is told why and which data
  would help; the four checks are separate rows; three tiers (not yet; met but not the cheapest; met and cheapest),
  the chart, the capability card; a language switch keeps the setup; the Connect drill opens;
- the home tile: one control (the data you have), the sentence changes with it, the rows match the lab.
- the owner's message of 2026-10-08: G1, no note on single choice anywhere in the lab (text, tooltips, labels) and a lead-in right above
  each group of options (parts 1, 4, 5, 6 and the scenarios); G2, the recap, two misconceptions, the Settings note on the metric rule,
  the Connect code card and the challenge's SFT feedback state the pattern instead of a detail of an earlier part; G8 (Codex), after
  Submit and “Show the answer” every one of the 60 picks is in the chart with “your pick”, a pick that cannot run (DPO, reward model +
  PPO) as a red row without a bar whose value says “cannot run: no preference pairs”, with its legend entry; the note counts the rest.
- the owner's message of 2026-10-09 (review b9v6): item 1, GPU hours defined in part 1 (a formula and an example), the task card and
  every budget line say whether they count all GPUs together; item 2, 70B QLoRA's memory blocks named by what they hold and the
  0.73 GPU hours shown with their inputs (tokens, work per token, the assumed 50%, the measured check); item 3 / H5, the stage in
  labelled blocks with a hairline in the line colour, in order, at 1440, 900 and 390; item 4 / H10, plain words in the Connect card,
  no 吃 / 喂 / 扛 / 吞 anywhere; item 5, every wrapped option group keeps a line between its rows (Settings, the rail, the tile);
  H1, no plain-text formula in the source or on the page, every formula MathML with no TeX word showing; H4, no live lab called
  planned; H6, every mark drawn has its legend entry; H7, nothing crosses the stage, the metric cards, the controls or the
  challenge chart at 1440, 900 and 390.
Screenshots: shots/post-capstone/.
Run: python3 tests/post-capstone_flow.py [en|zh] [1440|390|900] [guided|modes|tile]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/post-capstone.html', '--only', 'post-capstone'], check=True, cwd=P, stdout=subprocess.DEVNULL)
OUT = P / 'out' / 'post-capstone'; OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'wrapped-flow.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'post-capstone.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-flow.html')
SH = P / 'shots' / 'post-capstone'; SH.mkdir(parents=True, exist_ok=True)
ARGS = sys.argv[1:]
LANGS = [x for x in ('en', 'zh') if x in ARGS] or ['en', 'zh']
SIZES = [s for s in [(1440, 900), (390, 844)] if str(s[0]) in ARGS] or [(1440, 900), (390, 844)]
PARTS = [x for x in ('guided', 'modes', 'tile') if x in ARGS] or ['guided', 'modes', 'tile']
FULL = not ARGS
RUN900 = FULL or '900' in ARGS
if '900' in ARGS and not any(str(s[0]) in ARGS for s in [(1440, 900), (390, 844)]): SIZES = []
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
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour (R5.7)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, .fid-tag')||!e.getClientRects().length) continue; const t=n.textContent;
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
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #pcap-mathbody, .num-big, #pcap-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
# red only for failure (R5.6); pcap-norun is the challenge chart's row for a pick the data cannot run (rule G8, 2026-10-08)
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/pcap-mb w|pcap-mb q4|pcap-sw w|pcap-sw q4/,[amber]:/^$/,[red]:/status bad|badline|badc|pcap-bad|pcap-over|pcap-overb|pcap-overl|pcap-norun|warn|wrong|ex-res|red|over|bad/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#pcap-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())&&!(p==='stroke'&&s.strokeWidth==='0px')) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#pcap-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#pcap-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
rule_hits, fit_hits = [], []
# ---- the owner's message of 2026-10-09 (review b9v6)
LINE = "()=>{const d=document.createElement('i'); d.style.color='var(--line)'; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c}"
# H1: no plain-text formula on the page (outside MathML, code and the comments); every formula, also in a tooltip, is MathML with no TeX word
H1_PLAIN = r"""()=>{const out=[], re=/m∞|m₀|2\^\(|√\(|\^\(−|σ\(|Σ/; const tw=document.createTreeWalker(document.getElementById('pcap-app'),NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('math,pre,code')||!e.getClientRects().length) continue; if(re.test(n.textContent)) out.push(n.textContent.trim().slice(0,80));}
 document.querySelectorAll('#pcap-app [data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx').replace(/<math[\s\S]*?<\/math>/g,''); if(re.test(t)) out.push('tooltip: '+t.slice(0,80));});
 return out.slice(0,6)}"""
H1_MATH = r"""()=>{const tex=/\\|\bfrac\b|\bleft\b|\bright\b|\bsqrt\b|\binfty\b|\bmathrm\b|[{}]/, bad=[]; let k=0;
 const look=m=>{k++; const t=m.textContent; if(tex.test(t)||!m.querySelector('mi,mn,mo,mtext')) bad.push(t.slice(0,60));};
 document.querySelectorAll('#pcap-app math').forEach(m=>{if(m.getClientRects().length) look(m);});
 document.querySelectorAll('#pcap-app [data-tipx]').forEach(e=>{const s=e.getAttribute('data-tipx'); if(!s.includes('<math')) return; const d=document.createElement('div'); d.innerHTML=s; d.querySelectorAll('math').forEach(look);});
 return {n:k, bad:bad.slice(0,5)}}"""
# H5: the stage's blocks, each with a label (the tags excepted) and a 1 px hairline in the line colour, in order without overlap
H5_BLOCKS = """(line)=>{const g=document.getElementById('pcap-graph'); return [...g.querySelectorAll(':scope > .pcap-blk')].filter(e=>e.getClientRects().length).map(e=>{
 const cs=getComputedStyle(e), lab=e.querySelector(':scope > .pcap-bl'), r=e.getBoundingClientRect();
 return {id:e.id, label:lab?lab.textContent.trim():'', rule:cs.borderTopWidth==='1px'&&cs.borderTopStyle==='solid'&&cs.borderTopColor===line, top:r.top, bottom:r.bottom, left:r.left, right:r.right}})}"""
# H7: no element of the stage, the metric cards, the controls or the challenge chart crosses its panel (scrolling formulas aside)
H7_BOUNDS = """()=>{const out=[]; for(const sel of ['#pcap-graph','#pcap-metrics','#pcap-controls','#pcap-rail .pcap-ch-res','.tz-pcap']){const P=document.querySelector(sel);
 if(!P||!P.getClientRects().length) continue; const R=P.getBoundingClientRect();
 for(const e of P.querySelectorAll('*')){if(!e.getClientRects().length||e.closest('[data-endnotes],pre,.pcap-fx,.pcap-eq,math')) continue; const r=e.getBoundingClientRect(); if(r.width===0&&r.height===0) continue;
  if(r.left<R.left-1||r.right>R.right+1) out.push(sel+' '+(e.id||e.getAttribute('class')||e.tagName)+' '+Math.round(r.left)+'..'+Math.round(r.right)+' outside '+Math.round(R.left)+'..'+Math.round(R.right));}}
 return out.slice(0,6)}"""
# H6: every mark drawn on the stage has its legend entry
H6_LEGEND = """()=>{const miss=[], has=s=>!!document.querySelector(s), shown=s=>[...document.querySelectorAll(s)].some(e=>e.getClientRects().length);
 [...document.querySelectorAll('#pcap-mem .pcap-mb')].map(e=>[...e.classList].find(c=>c!=='pcap-mb')).forEach(k=>{if(k==='over'?!has('#pcap-mem-list .pcap-sw.over'):!has(`#pcap-mem-list li[data-k="${k}"] .pcap-sw.${k}`)) miss.push('memory '+k);});
 if(shown('#pcap-mem .pcap-cap80')&&!has('#pcap-mem-list .pcap-sw.line')) miss.push('80 GB line');
 [...document.querySelectorAll('#pcap-cost .pcap-cs')].forEach(e=>{const k=[...e.classList].find(c=>c!=='pcap-cs'); if(!has(`#pcap-cost-list .pcap-sw.c-${k}`)) miss.push('cost '+k);});
 if(shown('#pcap-cost .pcap-budget')&&!has('#pcap-cost-list .pcap-sw.dash')) miss.push('budget line');
 if(shown('#pcap-cost .pcap-overb')&&!has('#pcap-cost-list .pcap-sw.overb')) miss.push('over-budget ring');
 for(const [m,s] of [['.pcap-mfill','mf'],['.pcap-mband','mb'],['.pcap-m0','m0'],['.pcap-mt','dash'],['.pcap-zdot','zdot'],['.pcap-zbad','zbad']]) if(shown('#pcap-met '+m)&&!has('#pcap-met-list .pcap-sw.'+s)) miss.push('metric '+s);
 return miss}"""
# item 5: a wrapped option group keeps its rows and its buttons apart: since the 2026-10-22 refit the keys sit on a tray (R7.13) whose
# colour, unlike the keys', shows in the gaps
SEGLINES = """(line)=>[...document.querySelectorAll('#pcap-app .seg-ctl, .tz-pcap .seg-ctl')].filter(g=>g.getClientRects().length).map(g=>{
 const bs=[...g.querySelectorAll('button')].map(b=>b.getBoundingClientRect()), tops=[...new Set(bs.map(r=>Math.round(r.top)))].sort((a,b)=>a-b); let row=9, col=9;
 for(let i=1;i<tops.length;i++){const up=bs.filter(r=>Math.round(r.top)===tops[i-1]), lo=bs.filter(r=>Math.round(r.top)===tops[i]); row=Math.min(row,Math.min(...lo.map(r=>r.top))-Math.max(...up.map(r=>r.bottom)));}
 for(const t of tops){const rs=bs.filter(r=>Math.round(r.top)===t).sort((a,b)=>a.left-b.left); for(let i=1;i<rs.length;i++) col=Math.min(col,rs[i].left-rs[i-1].right);}
 const gbg=getComputedStyle(g).backgroundColor, kb=[...g.querySelectorAll('button')].find(b=>b.getAttribute('aria-pressed')!=='true');
 return {key:g.dataset.key||'tile', rows:tops.length, row:tops.length>1?row:null, col, bg:!/rgba\\(.*,\\s*0\\)$|transparent/.test(gbg)&&(!kb||getComputedStyle(kb).backgroundColor!==gbg)}})"""
SRC_JS = (P / 'labs' / 'post-capstone.js').read_text()


def b9v6(pg, where, labels=None):
    """H1, H5, H6 and H7 on whatever is on screen."""
    line = pg.evaluate(LINE)
    hp = pg.evaluate(H1_PLAIN); check(f'{where}: no plain-text formula on the page (H1)', not hp, hp)
    hm = pg.evaluate(H1_MATH); check(f'{where}: every formula is MathML with no TeX word showing (H1)', not hm['bad'], hm)
    bl = pg.evaluate(H5_BLOCKS, line)
    # in order, row by row: each block under the one before it, or (polish r4: two blocks share a row on a wide stage) beside it on its
    # right, tops level
    ordered = all(bl[i]['top'] >= bl[i - 1]['bottom'] - 0.5 or (abs(bl[i]['top'] - bl[i - 1]['top']) <= 1 and bl[i]['left'] >= bl[i - 1]['right'] - 0.5)
                  for i in range(1, len(bl)))
    check(f'{where}: the stage is in blocks, each with a hairline in the line colour and a label, in order (H5)',
          len(bl) >= 2 and all(x['rule'] for x in bl) and all(x['label'] for x in bl if x['id'] != 'pcap-tags') and ordered
          and (labels is None or [x['label'] for x in bl if x['id'] != 'pcap-tags'] == labels), bl)
    lg = pg.evaluate(H6_LEGEND); check(f'{where}: every mark on the stage has its legend entry (H6)', not lg, lg)
    bd = pg.evaluate(H7_BOUNDS); check(f'{where}: nothing crosses its panel (H7)', not bd, bd)


def seglines(pg, where, want_wrap=False):
    sl = pg.evaluate(SEGLINES, pg.evaluate(LINE))
    multi = [s for s in sl if s['rows'] > 1]
    check(f'{where}: every option group keeps its rows and its buttons apart, keys on a tray (item 5)', sl and all(s['bg'] and s['col'] >= 0.9 and (s['row'] is None or s['row'] >= 0.9) for s in sl)
          and (not want_wrap or multi), sl)


def page(b, w, h, lang, dark=False, hash_='post-capstone'):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light')
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    if hash_: pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__pcap.state()")
plan = lambda pg, c: pg.evaluate("c=>window.__pcap.plan(c)", c)
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#pcap-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#pcap-rail [data-next-why]')].filter(e=>!e.hidden&&e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden').map(e=>e.textContent)")
obs = lambda pg: pg.evaluate("()=>{const o=document.querySelector('#pcap-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}")
widths = lambda pg, sel: pg.evaluate("s=>[...document.querySelectorAll(s+' i')].map(e=>e.style.width+'@'+e.style.left)", sel)
status = lambda pg: pg.evaluate("()=>document.getElementById('pcap-status').className")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__pcap||!window.__pcap.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel, wait=250):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(wait); settle(pg)


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#pcap-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


# the owner's clarification (2026-10-07 23:37 UTC): no hover only expands a shown number to its exact value; Show the math may write one out
EXACT_HOVER = """r=>{const root=document.querySelector(r), out=[]; if(!root) return out;
  for(const e of root.querySelectorAll('[data-tipx],[data-tip]')){if(e.closest('.mathbody')) continue; let t=e.getAttribute('data-tipx')||''; const k=e.getAttribute('data-tip');
    if(k&&window.ASP) t+=' '+(window.ASP.t(k)||''); if(/精确值是|Exactly/.test(t)) out.push(t.slice(0,100));}
  for(const e of root.querySelectorAll('.num-big')) if(!e.closest('.mathbody')&&(e.hasAttribute('data-tipx')||e.closest('.def'))) out.push('hover on '+e.textContent);
  return out;}"""


def rules(pg, where, lang, sels=('#pcap-app', '#dock')):
    hx = pg.evaluate(EXACT_HOVER, '#pcap-app'); check(f'{where}: no exact-value hover outside Show the math', not hx, hx[:3])
    for sel in sels:
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        else:
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def fit(pg, where):
    for h in pg.evaluate(OVER): fit_hits.append(f'{where}: text wider than its box (R7.8): {h}')
    for h in pg.evaluate(HELP): fit_hits.append(f'{where}: help cursor without a tooltip (R1.1): {h}')


G1RE = '一次只能选|一次选一个|不会叠加|do not stack|one at a time'
LEADS = {'pcap-ic-lv1': ('Pick a task and see what data its team has', '选一个任务，看看这个团队手里有什么数据'),
         'pcap-ic-lv4': ('Pick a model and a way to train it, and see how much memory it needs', '选一种模型和训练方式，看看它要多少显存'),
         'pcap-ic-lv5': ('Pick a number of passes and see whether the format pass rate clears 95%', '选一个训练遍数，看看格式合格率能不能稳稳过 95%'),
         'pcap-ic-lv6': ('Pick a method and see how many GPU hours one pass takes', '选一种方法，看看训练一遍要多少 GPU 小时')}


def lead_in(pg, tag, box, lang):
    """G1 (the owner, 2026-10-08): the paragraph right above a group of options is a lead-in, what to pick and what to watch."""
    t = pg.evaluate("id=>{const g=document.getElementById(id), p=g&&g.previousElementSibling; return p&&p.matches('p.pcap-lead')?p.textContent.replace(/\\u00a0/g,' '):''}", box)
    check(f'{tag}: a lead-in right above the options says what to pick and what to watch (G1)', Z(lang, *LEADS[box]) in t, t)


def every(pg, where, lang, full_shot=None):
    check(f'{where}: no overflow', overflow(pg) <= 0, overflow(pg))
    g1 = pg.evaluate("re=>{const a=document.querySelector('#pcap-app'); const m=a?a.outerHTML.match(new RegExp(re,'i')):null; return m?m[0]:''}", G1RE)
    check(f'{where}: no note on single choice in the lab, in text, tooltips or labels (G1)', not g1, g1)
    rules(pg, where, lang); fit(pg, where)
    if full_shot: shot(pg, full_shot, full=pg.viewport_size['width'] <= 760)


def before_answer(pg):
    return pg.evaluate("""()=>{const t=[]; for(const s of ['#pcap-app .labhead','#pcap-rail','#pcap-stage','#pcap-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#pcap-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")


def nav(pg, what):
    click(pg, f'#pcap-rail [data-nav="{what}"]', 300)


def lever(pg, box, v, wait=300):
    click(pg, f'#{box} [data-lever="{v}"]', wait)


def bumped(pg):
    return pg.evaluate("()=>{const o=document.querySelector('#pcap-rail [data-obs]'); return !!o&&o.classList.contains('bump')}")


def same_pick_changes_nothing(pg, tag, box, v, bar):
    """R3.20: picking the option that is already pressed changes nothing: no bump on the result, the same bar widths."""
    pg.evaluate("()=>{const o=document.querySelector('#pcap-rail [data-obs]'); if(o) o.classList.remove('bump');}")
    w0, o0 = widths(pg, bar), obs(pg)
    lever(pg, box, v, 200)
    check(f'{tag}: picking the pressed option again changes nothing (R3.20)', not bumped(pg) and widths(pg, bar) == w0 and obs(pg) == o0, (bumped(pg), w0, widths(pg, bar)))


def run_guided(b, w, h, lang, dark=False):
    tag = f'{"dark_" if dark else ""}{w}_{lang}'
    pg = page(b, w, h, lang, dark)
    phone = w <= 760
    check(f'{tag} title', text(pg, '#pcap-app .labtitle') == Z(lang, 'SFT, DPO or GRPO: which one fits your task?', 'SFT、DPO 还是 GRPO：你的任务该选哪个？'), text(pg, '#pcap-app .labtitle'))
    check(f'{tag} no fidelity tag on top (owner, 2026-10-11 19:28 UTC)', not any(w in text(pg, '#pcap-app .labmeta') for w in ('Fidelity', '保真度')), text(pg, '#pcap-app .labmeta'))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#pcap-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        pg.evaluate(f"()=>document.querySelector('#pcap-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(200)
        check(f'{tag} popover {pid} opens', pg.evaluate(f"()=>{{const p=document.getElementById('{pid}'); return !!p&&p.getClientRects().length>0}}"))
        rules(pg, f'{tag} popover {pid}', lang)
        pg.evaluate(f"()=>document.querySelector('#pcap-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(150)
    # ---- part 1: the mission, three task cards
    st = state(pg)
    check(f'{tag} part 1: Next enabled, the strip shows only the data, status neutral', st['step'] == 0 and not any(nexts(pg)) and pg.evaluate("()=>document.querySelectorAll('#pcap-strip li').length") == 1 and 'neutral' in status(pg), (nexts(pg), status(pg)))
    o = obs(pg); card = text(pg, '#pcap-card')
    # polish r4: the picked task's goal, data and target are on the task card of the stage; the observation points to that card
    # instead of retelling it, so the task's numbers are checked on the card, and the observation is checked for the pointer
    check(f'{tag} part 1: the support card, 3,000 conversations, 95%; the observation points to the task card',
          '3,000' in card and '95%' in card and Z(lang, 'The task card in the chart', '图里的任务卡') in o, (o[:300], card[:300]))
    check(f'{tag} part 1: Show the math computes nothing that is not on the stage (SF-2)', 'GB' not in pg.evaluate("()=>document.getElementById('pcap-mathbody').innerText"))
    check(f'{tag} part 1: the full task card, not folded', not pg.evaluate("()=>!!document.querySelector('#pcap-card details')"))
    every(pg, f'{tag} part 1', lang, f'{tag}_p1')
    tip = pg.evaluate("()=>{const d=document.querySelector('#pcap-card .def'); return d?d.getAttribute('data-tipx'):''}")
    check(f'{tag} part 1: GPU hours defined with a formula and an example; the card says the budget is one GPU’s (item 1, H9)',
          Z(lang, '8 GPUs running for 1 hour use 8 GPU hours', '8 张卡一起跑 1 小时，就是 8 GPU 小时') in o and pg.evaluate("()=>!!document.querySelector('#pcap-rail [data-obs] .pcap-fx math')")
          and Z(lang, 'all GPUs together', '所有卡加起来') in o and Z(lang, '8 GPUs running for 1 hour use 8 GPU hours', '8 张卡一起跑 1 小时是 8 GPU 小时') in tip
          and Z(lang, '(one GPU, so', '（只有 1 张卡，就是') in text(pg, '#pcap-card'), (o[:300], tip, text(pg, '#pcap-card')[-120:]))
    b9v6(pg, f'{tag} part 1', Z(lang, ['From the data to the result'], ['从数据到结果']))
    lead_in(pg, f'{tag} part 1', 'pcap-ic-lv1', lang)
    lever(pg, 'pcap-ic-lv1', 'pref')
    badges = pg.evaluate("()=>[...document.querySelectorAll('#pcap-metrics .d')].map(e=>e.textContent).filter(Boolean)")
    check(f'{tag} part 1: another task card shows no change badge (its budget and target are not a change, SF-1)', not badges, badges)
    o = obs(pg); card = text(pg, '#pcap-card')
    check(f'{tag} part 1: the preference card, 20,000 pairs, 8 H100s, 60%', '20,000' in card and '60%' in card and '8' in card and '3.00' in card, (o[:200], card[:200]))
    check(f'{tag} part 1: the 8-GPU card says its budget counts all 8 GPUs together and how long they run (item 1)', Z(lang, 'all 8 GPUs together', '8 张卡加起来算') in card and Z(lang, 'about 23 minutes', '约 23 分钟') in card, card[-200:])
    check(f'{tag} part 1: no red on the stage before any method is chosen', not pg.evaluate("()=>!!document.querySelector('#pcap-stage .pcap-bad, #pcap-stage .status.bad')"))
    lever(pg, 'pcap-ic-lv1', 'math')
    card = text(pg, '#pcap-card')
    check(f'{tag} part 1: the math card, 7,473 problems, 39.7% → 70%', '7,473' in card and '70%' in card and '39.7%' in card, card[:200])
    # ---- part 2: start from the data
    nav(pg, 'next')
    rail = text(pg, '#pcap-rail .step.current') or text(pg, '#pcap-rail')
    check(f'{tag} part 2: the four methods defined before the choice (R2.10)', rail.index('GRPO') < rail.rindex(Z(lang, 'Pick a task and a method', '选一个任务')) and pg.evaluate("()=>document.querySelectorAll('#pcap-rail .pcap-defs li').length") == 4, rail[:200])
    check(f'{tag} part 2: Next waits for three pairs, with its reason', all(nexts(pg)) and any(Z(lang, 'Try three', '试三种') in x for x in why(pg)), why(pg))
    click(pg, '#pcap-ic-card button[data-v="pref"]'); click(pg, '#pcap-ic-meth button[data-v="sft"]')
    o = obs(pg)
    chip = pg.evaluate("()=>{const c=document.querySelectorAll('#pcap-strip .pcap-chip')[1]; return c?c.className+' | '+c.innerText:''}")
    check(f'{tag} part 2: SFT on the preference task can run on the chosen answers (green), and running is not reaching the target (item 8)',
          Z(lang, 'can run', '可以用') in o and Z(lang, 'chosen answer', '选中的回答') in o and Z(lang, 'reaches the target', '能不能达标') in o and pg.evaluate("()=>!!document.querySelector('#pcap-rail [data-obs] .okline')")
          and 'pcap-good' in chip and Z(lang, 'chosen answer', '选中的回答') in chip and 'bad' not in status(pg), (o[:240], chip))
    click(pg, '#pcap-ic-card button[data-v="fmt"]'); click(pg, '#pcap-ic-meth button[data-v="dpo"]')
    o = obs(pg)
    check(f'{tag} part 2: DPO on the support task cannot run, in red (its pairs cannot be formed)', Z(lang, 'cannot run', '用不了') in o and Z(lang, 'preference pairs', '偏好对') in o and 'bad' in status(pg), o[:200])
    click(pg, '#pcap-ic-card button[data-v="pref"]')
    click(pg, '#pcap-ic-meth button[data-v="dpo"]')
    o = obs(pg)
    check(f'{tag} part 2: DPO on the preference task can run', pg.evaluate("()=>!!document.querySelector('#pcap-rail [data-obs] .okline')") and 'DPO' in o, o[:200])
    click(pg, '#pcap-ic-card button[data-v="math"]'); click(pg, '#pcap-ic-meth button[data-v="grpo"]')
    st = state(pg)
    check(f'{tag} part 2: three pairs tried, Next enabled', len(st['tried2']) >= 3 and not any(nexts(pg)), st['tried2'])
    check(f'{tag} part 2: Show the math computes nothing that is not on the stage (SF-2)', 'GB' not in pg.evaluate("()=>document.getElementById('pcap-mathbody').innerText"))
    check(f'{tag} part 2: GRPO and RLVR are links to their labs', pg.evaluate("()=>!!document.querySelector('#pcap-rail .pcap-defs a[href=\"#grpo\"]')&&!!document.querySelector('#pcap-rail .pcap-defs a[href=\"#rlvr\"]')"))
    every(pg, f'{tag} part 2', lang, f'{tag}_p2')
    # ---- part 3: predict
    nav(pg, 'next')
    check(f'{tag} part 3: gate, Next disabled with its reason, Skip offered', all(nexts(pg)) and any(Z(lang, 'Pick an answer', '选一个答案') in x for x in why(pg)) and vis(pg, '#pcap-rail [data-nav="skip"]'), why(pg))
    ba = before_answer(pg)
    leak = [x for x in ('1.15', '0.078', '15 times', '15 倍', '14.8') if x in ba]
    check(f'{tag} part 3: the answer appears nowhere before the pick (R3.8)', not leak and not vis(pg, '#pcap-costbox'), leak)
    if phone: check(f'{tag} part 3: a link down to the answers above the stage', vis(pg, '#pcap-phone [data-act="to-q"]'))
    every(pg, f'{tag} part 3', lang, f'{tag}_p3')
    click(pg, '#pcap-rail [data-pred="1to3"]')
    o = obs(pg)
    check(f'{tag} part 3: a wrong pick is told the answer, about 15 times', pg.evaluate("()=>!!document.querySelector('#pcap-rail [data-obs] .badline')") and '15' in o and not any(nexts(pg)), o[:300])
    ct = text(pg, '#pcap-cost')
    check(f'{tag} part 3: the two cost bars, 0.078 and 1.15 GPU hours', vis(pg, '#pcap-costbox') and '0.078' in ct and '1.15' in ct, ct)
    check(f'{tag} part 3: the caption says the GPU hours count the 16 GPUs together (item 1)', Z(lang, 'GPU hours of the 16 GPUs together', 'GPU 小时按 16 张卡加起来算') in text(pg, '#pcap-cost-cap') and not vis(pg, '#pcap-cost-hw'), text(pg, '#pcap-cost-cap'))
    click(pg, '#pcap-rail [data-pred="gt10"]')
    check(f'{tag} part 3: the right pick', pg.evaluate("()=>!!document.querySelector('#pcap-rail [data-obs] .okline')") and state(pg)['pred'] == 'gt10')
    # ---- part 4: memory
    nav(pg, 'next')
    o = obs(pg)
    check(f'{tag} part 4: 8B full fine-tuning, 150.6 GB, out of memory, the part over 80 GB drawn', '150.6 GB' in o and 'bad' in status(pg) and pg.evaluate("()=>!!document.querySelector('#pcap-mem .pcap-mb.over')") and all(nexts(pg)), o[:200])
    check(f'{tag} part 4: Next waits for QLoRA, with its reason', any('QLoRA' in x for x in why(pg)), why(pg))
    lead_in(pg, f'{tag} part 4', 'pcap-ic-lv4', lang)
    same_pick_changes_nothing(pg, f'{tag} part 4', 'pcap-ic-lv4', '8full', '#pcap-mem')
    lever(pg, 'pcap-ic-lv4', '8lora'); o = obs(pg)
    check(f'{tag} part 4: LoRA, 22.8 GB, fits', '22.8 GB' in o and 'ok' in status(pg), o[:200])
    lever(pg, 'pcap-ic-lv4', '70lora'); o = obs(pg)
    check(f'{tag} part 4: 70B LoRA, 172.1 GB, does not fit', '172.1 GB' in o and 'bad' in status(pg) and all(nexts(pg)), o[:200])
    lever(pg, 'pcap-ic-lv4', '70qlora'); o = obs(pg)
    res = pg.evaluate("()=>[...document.querySelectorAll('#pcap-ic-lv4 [data-lever] .lv-r')].map(e=>e.textContent)")
    check(f'{tag} part 4: 70B QLoRA, 70.5 GB, fits; Next enabled', '70.5 GB' in o and not any(nexts(pg)), o[:200])
    check(f'{tag} part 4: results stay on the options tried', all(res) and len(res) == 4, res)
    ml, how, cc, stg = text(pg, '#pcap-mem-list'), text(pg, '#pcap-cost-hw'), text(pg, '#pcap-cost-cap'), text(pg, '#pcap-graph')
    check(f'{tag} part 4: 70B QLoRA, each memory block named by what it holds: 4.2 GB is the embedding table and LM head (BF16), 35.3 GB the frozen 4-bit linear layers (item 2)',
          Z(lang, 'Embedding table and LM head (BF16)', 'embedding 表和 LM head（BF16）') in ml and Z(lang, 'Frozen linear layers (4-bit)', '冻住的线性层（4 位）') in ml and '4.2 GB' in ml and '35.3 GB' in ml
          and Z(lang, 'LoRA matrices', 'LoRA 矩阵') in ml and Z(lang, 'BF16 weights', 'BF16 权重') not in stg, ml)
    check(f'{tag} part 4: the 0.73 GPU hours with their inputs: 4.61M tokens (3,000 × 512 × 3 passes), 282 GFLOP each, 50% of the peak, about 1,750 tokens/s; the measured check about 40%, 0.92 GPU hours (item 2)',
          vis(pg, '#pcap-cost-hw') and all(x in how for x in ('4.61M', '3,000', '512', '282 GFLOP', '50%', '1,750', '40%', '0.92', 'torchtune')) and Z(lang, 'How the GPU hours are counted', 'GPU 小时是怎么算出来的') in how
          and '0.73' in cc and Z(lang, 'One GPU, so it runs for about 44 minutes', '只有 1 张卡，也就是这张卡跑约 44 分钟') in cc and Z(lang, 'budget of 2.00 GPU hours', '预算 2.00 GPU 小时') in cc, (how, cc))
    check(f'{tag} part 4: the reading of 70B QLoRA points to where the time is counted (item 2)', Z(lang, 'The “GPU hours and budget” block of the chart shows how they add up', '图里“GPU 小时和预算”那一栏写了是怎么算出来的') in o, o[-300:])
    b9v6(pg, f'{tag} part 4', Z(lang, ['From the data to the result', 'Memory', 'GPU hours and budget'], ['从数据到结果', '显存', 'GPU 小时和预算']))
    if phone:
        check(f'{tag} part 4: on a phone the task card is one line, closed, above the picture (SF-6)', pg.evaluate("()=>{const d=document.querySelector('#pcap-card details.pcap-cardd'); return !!d&&!d.open}"))
    every(pg, f'{tag} part 4', lang, f'{tag}_p4')
    # ---- part 5: the metric
    nav(pg, 'next')
    o = obs(pg)
    check(f'{tag} part 5: one pass, 83.5%, below the target; Next waits', '83.5%' in o and all(nexts(pg)) and any('95%' in x for x in why(pg)), (o[:200], why(pg)))
    check(f'{tag} part 5: the formula on its own line', vis(pg, '#pcap-rail .pcap-fx'))
    lead_in(pg, f'{tag} part 5', 'pcap-ic-lv5', lang)
    same_pick_changes_nothing(pg, f'{tag} part 5', 'pcap-ic-lv5', '1', '#pcap-met')
    lever(pg, 'pcap-ic-lv5', '2'); o = obs(pg)
    check(f'{tag} part 5: two passes, 94.4%, still below', '94.4%' in o and all(nexts(pg)), o[:200])
    lever(pg, 'pcap-ic-lv5', '3'); o = obs(pg)
    check(f'{tag} part 5: three passes, 97.1%, met, but the bottom of the error bar 94.8% is below; Next enabled', '97.1%' in o and '94.8%' in o and not any(nexts(pg)) and pg.evaluate("()=>!!document.querySelector('#pcap-met .pcap-mfill.ok')"), o[:300])
    zw = pg.evaluate("()=>{const z=document.querySelector('#pcap-met .pcap-zbad'); return z?z.getBoundingClientRect().width:0}")
    check(f'{tag} part 5: three passes, the zoomed strip shows the part of the error bar below 95% in red, wide enough to see (SF-5)', zw >= 4 and '94.8%' in text(pg, '#pcap-met .pcap-zcap'), (zw, text(pg, '#pcap-met .pcap-zcap')))
    check(f'{tag} part 5: the dot in the zoomed strip has its legend entry (H6)', pg.evaluate("()=>!document.querySelector('#pcap-met .pcap-zdot')||!!document.querySelector('#pcap-met-list .pcap-sw.zdot')"))
    b9v6(pg, f'{tag} part 5', Z(lang, ['From the data to the result', 'GPU hours and budget', 'Result'], ['从数据到结果', 'GPU 小时和预算', '结果']))
    shot(pg, f'{tag}_p5_3passes')
    lever(pg, 'pcap-ic-lv5', '4'); o = obs(pg)
    check(f'{tag} part 5: four passes, 97.8%, the whole error bar above (95.7%)', '97.8%' in o and '95.7%' in o, o[:300])
    check(f'{tag} part 5: four passes, no red part in the zoomed strip', not pg.evaluate("()=>!!document.querySelector('#pcap-met .pcap-zbad')"))
    if phone:
        pg.evaluate("()=>{const t=document.getElementById('pcap-graph'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+80)}"); pg.wait_for_timeout(700)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, read:document.getElementById('dkRead').textContent}}")
        check(f'{tag} part 5: the mini result bar shows the run', dk['shown'] and '97.8%' in dk['read'] and 'GB' in dk['read'], dk)
        shot(pg, f'{tag}_p5_dock')
        rules(pg, f'{tag} part 5 dock', lang)
    every(pg, f'{tag} part 5', lang, f'{tag}_p5')
    # ---- part 6: when the obvious choice is wrong
    nav(pg, 'next')
    o = obs(pg)
    check(f'{tag} part 6: DPO 0.55 GPU hours, within the budget; Next waits', '0.55' in o and all(nexts(pg)), o[:200])
    lead_in(pg, f'{tag} part 6', 'pcap-ic-lv6', lang)
    same_pick_changes_nothing(pg, f'{tag} part 6', 'pcap-ic-lv6', 'dpo', '#pcap-cost')
    lever(pg, 'pcap-ic-lv6', 'ppo'); o = obs(pg)
    check(f'{tag} part 6: reward model + PPO 2.32 GPU hours, about 4.2 times DPO', '2.32' in o and '4.2' in o, o[:300])
    lever(pg, 'pcap-ic-lv6', 'grpo'); o = obs(pg)
    check(f'{tag} part 6: GRPO with a reward model 4.04 GPU hours, over the budget, red; Next enabled', '4.04' in o and pg.evaluate("()=>!!document.querySelector('#pcap-rail [data-obs] .badline')") and 'bad' in status(pg)
          and pg.evaluate("()=>!!document.querySelector('#pcap-cost .pcap-overb')") and not any(nexts(pg)), o[:300])
    ml, cc = text(pg, '#pcap-mem-list'), text(pg, '#pcap-cost-cap')
    check(f'{tag} part 6: the reference and reward models are their own memory block (BF16, 4.0 GB); the caption counts all 8 GPUs together (items 1, 2)',
          Z(lang, 'Reference and reward models (BF16)', '参考模型和奖励模型（BF16）') in ml and '4.0 GB' in ml and Z(lang, 'Both count all 8 GPUs together', '两个数都是 8 张卡加起来算的') in cc
          and Z(lang, 'budget of 3.00 GPU hours', '预算 3.00 GPU 小时') in cc and Z(lang, 'each 8 answers, 8 × 384 = 3,072 tokens', '每条 8 个回答，共 8 × 384 = 3,072 个') in text(pg, '#pcap-cost-hw'), (ml, cc))
    b9v6(pg, f'{tag} part 6', Z(lang, ['From the data to the result', 'Memory', 'GPU hours and budget'], ['从数据到结果', '显存', 'GPU 小时和预算']))
    every(pg, f'{tag} part 6', lang, f'{tag}_p6')
    mh = pg.evaluate("()=>{const d=document.getElementById('pcap-math'), was=d.open; d.open=true; const t=document.getElementById('pcap-mathbody').innerText; d.open=was; return t}")
    check(f'{tag} part 6: Show the math gives memory, GPU hours and the metric rule', '28.1 GB' in mh and '4.04' in mh and '65.0%' in mh, mh[:300])
    mb = pg.evaluate("()=>[...document.querySelectorAll('#pcap-mathbody .pcap-mblk')].map(b=>({lead:!!b.querySelector(':scope > .pcap-ml'), eq:b.querySelectorAll('.pcap-eq .pcap-eqr math').length, res:!!b.querySelector('.pcap-eqr b')}))")
    check(f'{tag} part 6: Show the math in short blocks, each a lead line, the formula and the numbers under it, and the result (H2)', len(mb) == 5 and all(x['lead'] and x['eq'] >= 2 and x['res'] for x in mb), mb)
    # ---- part 7: what to remember
    nav(pg, 'next')
    rt = text(pg, '#pcap-rail')
    check(f'{tag} part 7: three points, three misconceptions, See the scenarios', pg.evaluate("()=>document.querySelectorAll('#pcap-rail .pcap-ul > li').length") == 3 and rt.count('✗') == 3
          and pg.evaluate("()=>!!document.querySelector('#pcap-rail [data-mode-go=\"sandbox\"]')") and Z(lang, 'over ten times', '十几倍') in rt and 'QLoRA' in rt and Z(lang, 'Tülu 3 does exactly this', 'Tülu 3 就是这样做的') in rt, rt[-500:])
    rec = pg.evaluate("()=>[...document.querySelectorAll('#pcap-rail .pcap-ul > li, #pcap-rail .pcap-mis p')].map(e=>e.innerText).join('\\n')")
    old = [x for x in ('In part', 'in part', '第 2 部分', '第 3 部分', '第 4 部分', '第 5 部分', 'support team', 'math task', '客服团队', '数学任务', '多训了一遍', 'one more pass') if x in rec]
    new = Z(lang, ('decides which methods can run at all', 'several times the size of the weights', 'another sample of the same size may fail the check', 'To teach a format, example answers and SFT are enough', 'With a checker but no pairs'),
            ('决定了哪些方法跑得起来', '比权重本身大好几倍', '换一批同样多的样本再测', '只是教格式的话', '只有判分程序、没有偏好对时'))
    check(f'{tag} part 7: the recap and the misconceptions state each pattern and what follows, no detail of an earlier part (G2)', not old and all(x in rec for x in new), (old, [x for x in new if x not in rec], rec[:300]))
    every(pg, f'{tag} part 7', lang, f'{tag}_p7')
    # ---- the language switch keeps the place; Back shows part 5 as it was left, nothing moving (R3.20)
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    check(f'{tag} language switch keeps the part', state(pg)['step'] == 6)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    nav(pg, 'back'); nav(pg, 'back')
    st = state(pg)
    res = pg.evaluate("()=>[...document.querySelectorAll('#pcap-ic-lv5 [data-lever] .lv-r')].map(e=>e.textContent)")
    check(f'{tag} back in part 5: shown as it was left (4 passes, the results of the options tried), Next enabled, nothing moving (R3.20)', st['step'] == 4 and st['lv5'] == 4 and st['x'] == 4
          and [bool(x) for x in res] == [v in st['tried5'] for v in (0.5, 1, 2, 3, 4)] and not any(nexts(pg)) and not bumped(pg) and '97.8%' in obs(pg), (st, res))
    pg.close()


def run_modes(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    mode(pg, 'sandbox')
    pressed = lambda: pg.evaluate("()=>[...document.querySelectorAll('#pcap-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: five, the first pressed on entry', pressed() == ['true', 'false', 'false', 'false', 'false'], pressed())
    sl = pg.evaluate("()=>{const g=document.querySelector('#pcap-rail .presets'), p=g&&g.previousElementSibling; return [p?p.textContent:'', g?g.getAttribute('aria-label'):'']}")
    check(f'{tag} scenarios: a lead-in above the scenarios says what to pick and what to watch; the group is labelled “pick one” (G1)',
          Z(lang, 'Pick a scenario and see which method fits it', '选一个场景，看看它适合哪种方法') in sl[0] and sl[1] == Z(lang, 'Scenarios, pick one', '应用场景，选一个'), sl)
    presets = pg.evaluate("()=>window.__pcap.consts.PRESETS")
    for i in range(5):
        click(pg, f'#pcap-rail [data-preset="{i}"]', 400)
        r = plan(pg, {k: v for k, v in presets[i].items() if k != 'id'})
        ins = text(pg, '#pcap-insight'); mt = text(pg, '#pcap-metrics')
        mem = f"{r['mem'] / 1e9:.1f} GB"; met = f"{r['metric']:.1f}%"
        check(f'{tag} scenario {i + 1}: pressed alone, {mem}, {met} on the cards, one conclusion in the box (SF-18)', pressed() == ['true' if j == i else 'false' for j in range(5)] and mem in mt and met in mt and ins.strip() != '' and ' GB' not in ins, (pressed(), mt, ins[:160]))
        every(pg, f'{tag} scenario {i + 1}', lang, f'{tag}_scenario{i + 1}')
    # since Settings sit in the 400 px column (2026-10-11), the Chinese option names fit one row, so only English wraps at 1440; the
    # line between the buttons is checked in both, the line between the rows wherever a group wraps (English here, both at 900)
    seglines(pg, f'{tag} settings', want_wrap=w > 760 and lang == 'en')
    b9v6(pg, f'{tag} scenario 5', Z(lang, ['From the data to the result', 'Memory', 'GPU hours and budget', 'Result'], ['从数据到结果', '显存', 'GPU 小时和预算', '结果']))
    # the lab's own text (the kit's next-lab row, nav.next-lab, names a lab that is not built yet by a run-time check, which H4 allows)
    html = pg.evaluate("()=>[...document.querySelectorAll('#pcap-rail, #pcap-stage, #pcap-controls, #pcap-app .connect')].map(e=>{const c=e.cloneNode(true); c.querySelectorAll('.next-lab').forEach(x=>x.remove()); return c.outerHTML}).join('')")
    fig = [x for x in ('吃', '喂', '扛', '吞', '规划中', '还没上线', 'planned lab', 'a later lab', 'not built') if x in html] + re.findall(r'\b(?:eats|feeds on)\b', html)
    src_fig = [x for x in ('吃', '喂', '扛', '吞', '规划中', '还没上线', 'planned lab', 'a later lab', 'not built') if x in SRC_JS] + re.findall(r'\b(?:eats|feeds on)\b', SRC_JS)
    check(f'{tag}: plain words, no figurative verb, no live lab called planned, on the page and in the source (H10, H4)', not fig and not src_fig and len(html) > 5000, (fig, src_fig))
    cx = text(pg, '#pcap-app .connect')
    check(f'{tag} connect: the trainers in plain words (item 4)', Z(lang, 'which kind of data each trainer works with', '每个 trainer 适配哪种数据') in cx, cx[:300])
    if w <= 760:
        check(f'{tag} scenarios: a one-line task and a link down to the scenarios above the stage', vis(pg, '#pcap-phone [data-act="to-task"]') and Z(lang, 'scenario', '场景') in text(pg, '#pcap-phone'), text(pg, '#pcap-phone'))
    click(pg, '#pcap-rail [data-preset="2"]', 300)
    url = pg.url; st = state(pg)
    pg2 = b.new_page(viewport={'width': w, 'height': h}); open_drawers(pg2); pg2.goto(URL); pg2.wait_for_timeout(250)
    pg2.click(f'.langsw button[data-lang="{lang}"]'); pg2.wait_for_timeout(80)
    pg2.evaluate(f"()=>{{location.hash={url.split('#', 1)[1]!r}}}"); pg2.wait_for_timeout(500)
    st2 = pg2.evaluate("()=>window.__pcap.state()")
    check(f'{tag} scenarios: the URL carries the setup, a fresh page restores it', all(st2[k] == st[k] for k in ('card', 'meth', 'mode', 'x')) and st2['mode'] == 'sandbox' and '~' in url, (url, st2))
    pg2.close()
    click(pg, '#pcap-rail [data-preset="0"]', 300)
    click(pg, '#pcap-ctl-x button[data-v="2"]', 300)
    check(f'{tag} scenarios: after a change in Settings no scenario is pressed', pressed() == ['false'] * 5 and state(pg)['x'] == 2, (pressed(), state(pg)['x']))
    click(pg, '#pcap-ctl-card button[data-v="big"]', 300); click(pg, '#pcap-ctl-mode button[data-v="full"]', 300)
    check(f'{tag} settings: 70B full fine-tuning on one GPU is out of memory', 'bad' in status(pg) and pg.evaluate("()=>!!document.querySelector('#pcap-strip .pcap-bad')"))
    cn = pg.evaluate("()=>[...document.querySelectorAll('#pcap-controls .ctl-note')].map(e=>e.textContent).join(' ')")
    check(f'{tag} settings: the note on the metric rule states the rule itself, not the part that taught it (G2)',
          Z(lang, 'more slowly the closer it gets', '越接近上限涨得越慢') in cn and not any(x in cn for x in ('part 5', '第 5 部分')), cn[-300:])
    every(pg, f'{tag} sandbox changed', lang, f'{tag}_sandbox_changed')
    # ---- the challenge; no change badge carried over from the scenarios
    pg.evaluate("()=>document.querySelector('#pcap-rail [data-preset=\"2\"]').click()"); pg.wait_for_timeout(150)
    mode(pg, 'challenge')
    stale = pg.evaluate("()=>[...document.querySelectorAll('#pcap-metrics .d')].map(e=>e.textContent).filter(Boolean)")
    check(f'{tag} challenge: no change badge left from the scenarios', not stale, stale)
    if w <= 760:
        check(f'{tag} challenge: a one-line task and a link down to the task card above the stage', vis(pg, '#pcap-phone [data-act="to-task"]') and '65%' in text(pg, '#pcap-phone'), text(pg, '#pcap-phone'))
    st = state(pg)
    check(f'{tag} challenge: starts with SFT; Settings locked', (st['ch']['meth'], st['ch']['mode'], st['ch']['x']) == ('sft', 'full', 1) and not vis(pg, '#pcap-ctl-meth-wrap'), st)
    cls = lambda i: pg.evaluate(f"()=>document.getElementById('pcap-ch-g{i}').className")
    check(f'{tag} challenge: the start (SFT, full) can run on the final answers, but does not fit (item 8)', 'met' in cls(0) and Z(lang, 'final answer', '最后答案') in text(pg, '#pcap-ch-can') and 'pcap-over' in cls(1) and 'bad' in status(pg), (text(pg, '#pcap-ch-can'), cls(1)))
    every(pg, f'{tag} challenge start', lang, f'{tag}_challenge_start')
    click(pg, '#pcap-rail [data-act="submit"]', 400)
    r = text(pg, '#pcap-ch-res')
    check(f'{tag} challenge tier 1 (not yet): red, the two failing checks (memory, accuracy) each on its line, a hint, no chart yet',
          Z(lang, 'Not yet', '还没达到') in r and pg.evaluate("()=>!!document.querySelector('#pcap-ch-res .badline')") and not vis(pg, '#pcap-ch-res .cmp')
          and pg.evaluate("()=>document.querySelectorAll('#pcap-ch-res .pcap-badchk li').length") == 2 and '150.6 GB' in r and '39.7%' in r, r[:300])
    BRIEF = """()=>{const res=document.querySelector('#pcap-ch-res'), c=res.querySelector('.cmp'); const rs=c?[...c.querySelectorAll('.cmp-row')]:[], tg=r=>!!r.querySelector('.cmp-tag,.cmp-you');
      return {n:rs.length, ok:rs.filter(r=>r.classList.contains('ok')).length, me:rs.filter(r=>r.classList.contains('me')).length, stray:rs.filter(r=>!r.classList.contains('ok')&&!r.classList.contains('me')&&!tg(r)).length,
        labs:rs.map(r=>r.querySelector('.cmp-lab').textContent), title:c?c.querySelector('.cmp-title').textContent:'', note:(res.querySelector('.cmp-note')||{}).textContent||'', reveal:(res.querySelector('[data-act="reveal"]')||{}).textContent||''};}"""
    bf = pg.evaluate(BRIEF)
    check(f'{tag} challenge: the reveal button says “{Z(lang, "Show the answer", "直接看答案")}” (the owner, batch 8)', bf['reveal'].strip() == Z(lang, 'Show the answer', '直接看答案'), bf)
    click(pg, '#pcap-ch-res [data-act="reveal"]', 300)
    bf = pg.evaluate(BRIEF)
    check(f'{tag} challenge (the start, revealed): the 4 setups that pass and the initial setup, which is your pick; the note counts the other 55 of the 60',
          (bf['n'], bf['ok'], bf['me'], bf['stray']) == (5, 4, 1, 0) and Z(lang, 'initial setup', '初始设置') in bf['labs'][0] and Z(lang, 'The other 55 setups', '其余 55 种设置') in bf['note'], bf)
    check(f'{tag} challenge: the chart title no longer says “each number of passes”', bf['title'] and not any(x in bf['title'] for x in ('each', 'every', '不同遍数', '各要')), bf['title'])
    # item 8's acceptance: SFT on problem + final answer runs, fits with LoRA, stays in budget, and misses the target for a stated reason
    click(pg, '#pcap-ic-chmode button[data-v="lora"]', 250)
    r = text(pg, '#pcap-ch-res'); chips = pg.evaluate("()=>[...document.querySelectorAll('#pcap-strip .pcap-chip')].map(c=>c.className)")
    check(f'{tag} challenge: SFT with LoRA passes data, memory and budget; only the accuracy row is red', [('met' in cls(i), 'pcap-over' in cls(i)) for i in range(4)] == [(True, False), (True, False), (True, False), (False, True)], [cls(i) for i in range(4)])
    check(f'{tag} challenge: SFT with LoRA, the result names the one failing check and why, and which data would help (item 8)',
          pg.evaluate("()=>document.querySelectorAll('#pcap-ch-res .pcap-badchk li').length") == 1 and Z(lang, 'without the working', '不写过程') in r and Z(lang, 'worked solutions', '解题过程') in r
          and Z(lang, 'cannot run', '跑不起来') not in r, r[:400])
    check(f'{tag} challenge: the stage agrees: method chip green, result chip red, the metric caption says training left it unchanged',
          'pcap-good' in chips[1] and 'pcap-bad' in chips[-1] and Z(lang, 'the same as before training', '和训练前一样') in text(pg, '#pcap-met-cap') and 'bad' in status(pg), (chips, text(pg, '#pcap-met-cap')))
    bf = pg.evaluate(BRIEF)
    check(f'{tag} challenge (SFT with LoRA, revealed): the initial setup, the 4 that pass and your pick; the note counts the other 54', (bf['n'], bf['ok'], bf['me'], bf['stray']) == (6, 4, 1, 0) and Z(lang, 'The other 54 setups', '其余 54 种设置') in bf['note'], bf)
    flat = pg.evaluate("()=>[document.getElementById('pcap-status').dataset.tipx||'',...[...document.querySelectorAll('#pcap-strip [data-tipx]')].map(e=>e.dataset.tipx)].join(' ')") + ' ' + r
    check(f'{tag} challenge: SFT on final answers, the feedback and its tooltips name the missing data, not the math team’s card of the guided tour (G2)',
          Z(lang, 'worked solutions for it to imitate', '让它照着学') in r and Z(lang, 'Worked solutions in the data would teach it', '数据里有一步步的解题过程') in flat and not any(x in flat for x in ('math team', '数学团队')), flat[-400:])
    # G8 (Codex, forwarded by the owner on 2026-10-08): after Submit and Show the answer, every pick a reader can make is in the chart with
    # “your pick”. A pick the data cannot run (DPO, reward model + PPO: no preference pairs) has no bar, red words and a red ring, says why,
    # and adds its legend entry; the note counts the setups left out (60 minus the rows). Walked in one call: each click renders at once.
    WALK = """()=>{const out=[], btn=(k,v)=>document.querySelector(`#pcap-ic-ch${k} button[data-v=\"${v}\"]`);
      for(const meth of ['sft','dpo','ppo','grpo']) for(const mode of ['full','lora','qlora']) for(const x of ['0.5','1','2','3','4']){
        btn('meth',meth).click(); btn('mode',mode).click(); btn('x',x).click();
        const res=document.querySelector('#pcap-ch-res'), c=res.querySelector('.cmp'), rows=c?[...c.querySelectorAll('.cmp-row')]:[], tg=r=>!!r.querySelector('.cmp-tag,.cmp-you');
        const you=rows.filter(r=>r.querySelector('.cmp-you')), y=you[0], bl=res.querySelector('.badline');
        out.push({meth,mode,x,n:rows.length,ok:rows.filter(r=>r.classList.contains('ok')).length,stray:rows.filter(r=>!r.classList.contains('ok')&&!r.classList.contains('me')&&!tg(r)).length,you:you.length,
          lab:y?y.querySelector('.cmp-lab > span').textContent.replace(/^✓ /,''):'',val:y?y.querySelector('.cmp-val').textContent:'',norun:!!y&&y.classList.contains('pcap-norun'),
          bar:y?y.querySelector('.cmp-track i').getBoundingClientRect().width:-1,col:y?getComputedStyle(y.querySelector('.cmp-val')).color:'',ring:y?getComputedStyle(y.querySelector('.cmp-track')).boxShadow:'',
          inT:(()=>{if(!y) return false; const t=y.querySelector('.cmp-track').getBoundingClientRect(), v=y.querySelector('.cmp-val').getBoundingClientRect(); return v.left>=t.left-0.5&&v.right<=t.right+0.5&&v.top>=t.top-0.5&&v.bottom<=t.bottom+0.5;})(),
          red:bl?getComputedStyle(bl).color:'',leg:c?(c.querySelector('.cmp-leg')||{}).textContent||'':'',legRed:!!(c&&c.querySelector('.cmp-leg i.pcap-leg-norun')),note:(res.querySelector('.cmp-note')||{}).textContent||''});}
      return out;}"""
    can = {(e['meth'], e['mode'], e['x']): e for e in pg.evaluate("()=>window.__pcap.chAll()")}
    MN = {'sft': 'SFT', 'dpo': 'DPO', 'ppo': Z(lang, 'reward model + PPO', '奖励模型 + PPO'), 'grpo': 'GRPO'}
    MO = {'full': Z(lang, 'full fine-tuning', '全量微调'), 'lora': 'LoRA', 'qlora': 'QLoRA'}
    PW = lambda x: Z(lang, 'half a pass', '半遍') if x == 0.5 else Z(lang, f'{x:g} pass' + ('es' if x != 1 else ''), f'{x:g} 遍')
    walk, bad = pg.evaluate(WALK), []
    for e in walk:
        x = float(e['x']); k = (e['meth'], e['mode'], x if x != int(x) else int(x)); ref = can[k]
        want_n = 4 + 1 + (0 if ref['ok'] or k == ('sft', 'full', 1) else 1)
        name = Z(lang, ', ', '，').join([MN[k[0]], MO[k[1]], PW(x)])
        good = e['you'] == 1 and e['lab'] == name and e['n'] == want_n and e['ok'] == 4 and e['stray'] == 0 and Z(lang, f"The other {60 - want_n} setups", f"其余 {60 - want_n} 种设置") in e['note']
        if ref['can']:
            good = good and not e['norun'] and e['bar'] > 0 and Z(lang, ' h, ', '小时，') in e['val'] and not e['legRed'] and Z(lang, 'cannot run', '跑不起来') not in e['leg']
        else:
            good = (good and e['norun'] and e['bar'] == 0 and e['val'] == Z(lang, 'cannot run: no preference pairs', '跑不起来，没有偏好对') and e['red'] and e['col'] == e['red']
                    and e['red'] in e['ring'] and e['inT'] and e['legRed'] and Z(lang, 'cannot run', '跑不起来') in e['leg'])
        if not good: bad.append(e)
    check(f'{tag} challenge: all 60 picks after Show the answer: each in the chart with “your pick”; DPO and reward model + PPO as a red row without a bar, “{Z(lang, "cannot run: no preference pairs", "跑不起来，没有偏好对")}” inside its ring, with a legend entry; the note counts the rest (G8)',
          len(walk) == 60 and not bad, bad[:2])
    click(pg, '#pcap-ic-chmeth button[data-v="dpo"]', 120); click(pg, '#pcap-ic-chmode button[data-v="full"]', 120); click(pg, '#pcap-ic-chx button[data-v="1"]', 250); settle(pg)
    pg.evaluate("()=>document.querySelector('#pcap-ch-res .cmp').scrollIntoView({block:'center'})"); pg.wait_for_timeout(250)
    every(pg, f'{tag} challenge DPO, revealed (G8)', lang, f'{tag}_challenge_dpo')
    click(pg, '#pcap-ic-chmeth button[data-v="sft"]', 120); click(pg, '#pcap-ic-chmode button[data-v="lora"]', 120); click(pg, '#pcap-ic-chx button[data-v="1"]', 250); settle(pg)
    every(pg, f'{tag} challenge SFT with LoRA', lang, f'{tag}_challenge_sft_lora')
    click(pg, '#pcap-ic-chmode button[data-v="full"]', 250)
    click(pg, '#pcap-ic-chmeth button[data-v="grpo"]', 250)
    r = text(pg, '#pcap-ch-res')
    check(f'{tag} challenge: full GRPO does not fit (166.6 GB)', '166.6 GB' in r and 'pcap-over' in pg.evaluate("()=>document.getElementById('pcap-ch-g1').className"), r[:200])
    click(pg, '#pcap-ic-chmode button[data-v="lora"]', 250); click(pg, '#pcap-ic-chx button[data-v="3"]', 250)
    r = text(pg, '#pcap-ch-res')
    check(f'{tag} challenge tier 2 (met, not the cheapest): GRPO, LoRA, 3 passes', Z(lang, 'a cheaper setup exists', '还有更省的设置') in r and vis(pg, '#pcap-ch-res .cmp') and '1.88' in r, r[:300])
    click(pg, '#pcap-ic-chx button[data-v="2"]', 250)
    r = text(pg, '#pcap-ch-res')
    check(f'{tag} challenge tier 3 (goal met, the cheapest): 69.3%, 1.88 GPU hours', Z(lang, 'Goal met', '达成目标') in r and '69.3%' in r and '1.88' in r, r[:300])
    tags = text(pg, '#pcap-ch-res .cmp')
    bf = pg.evaluate(BRIEF)
    check(f'{tag} challenge: the chart holds the initial setup (SFT, full, 39.7%) and the 4 setups that pass, your pick among them; the note counts the other 55',
          Z(lang, 'your pick', '你的选择') in tags and (bf['n'], bf['ok'], bf['me'], bf['stray']) == (5, 4, 1, 0) and '39.7%' in tags and Z(lang, 'The other 55 setups', '其余 55 种设置') in bf['note'], (bf, tags[:300]))
    tips = pg.evaluate("()=>[...document.querySelectorAll('#pcap-ch-res [data-tipx]')].map(e=>e.dataset.tipx).join(' ')")
    check(f'{tag} challenge: the passes the chart leaves out are named in words, their numbers on hover (59.5%, 3.76 GPU hours)', '59.5%' in tips and '3.76' in tips, tips[-300:])
    check(f'{tag} challenge: the explanation says SFT runs on the final answers but teaches no working; no “cannot run” for SFT (item 8)',
          Z(lang, 'SFT does run on the final answers', 'SFT 能用最后答案来训练') in r and Z(lang, 'Worked solutions would change that', '有解题过程就不一样了') in r and Z(lang, 'SFT, cannot run', 'SFT 跑不起来') not in r, r[:600])
    check(f'{tag} challenge: the capability card', vis(pg, '#pcap-rail [data-exits]'))
    pg.evaluate("()=>document.getElementById('pcap-ch-res').scrollIntoView({block:'start'})"); pg.wait_for_timeout(300)
    every(pg, f'{tag} challenge solved', lang, f'{tag}_challenge_solved')
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} challenge: a language switch keeps the setup and the result', (st['ch']['meth'], st['ch']['mode'], st['ch']['x']) == ('grpo', 'lora', 2) and st['chSubmitted'] and vis(pg, '#pcap-ch-res .cmp'), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    pg.evaluate("()=>{const b=document.querySelector('[data-act=\"drill-reveal\"]'); if(b) b.click();}"); pg.wait_for_timeout(250)
    check(f'{tag} connect: the drill answer opens, its GPU-hours formula typeset (H1)', vis(pg, '#pcap-drill-a') and 'GB' in text(pg, '#pcap-drill-a') and pg.evaluate("()=>document.querySelectorAll('#pcap-drill-a .pcap-eq math').length>=4"))
    hm = pg.evaluate(H1_MATH); check(f'{tag} connect: every formula is MathML with no TeX word showing (H1)', not hm['bad'], hm)
    cc = text(pg, '#pcap-app .connect .c-card')
    check(f'{tag} connect: the code card says what the code computes, without naming a guided part (G2)', Z(lang, 'The planner in plain Python: the GPU hours', '用纯 Python 写的估算') in cc and not any(x in cc for x in ('part 3', '第 3 部分')), cc[:200])
    every(pg, f'{tag} connect', lang)
    mode(pg, 'guided')
    check(f'{tag} back to Guided', state(pg)['mode'] == 'guided')
    every(pg, f'{tag} guided again', lang)
    pg.close()


def run_tile(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang, hash_=None)
    pg.evaluate("()=>{const b=document.querySelector('[data-world=\"post\"]'); if(b) b.click();}"); pg.wait_for_timeout(500)
    has = pg.evaluate("()=>!!document.querySelector('.tz-pcap')")
    check(f'{tag} tile: the teaser renders on the home page', has)
    if not has: pg.close(); return
    pg.evaluate("()=>document.querySelector('.tz-pcap').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
    reads, rows = [], []
    for v in ('demo', 'pairs', 'checker'):
        pg.evaluate(f"()=>document.querySelector('.tz-pcap .seg-ctl button[data-v=\"{v}\"]').click()"); pg.wait_for_timeout(150)
        reads.append(text(pg, '.tz-pcap .tz-read')); rows.append(pg.evaluate("()=>[...document.querySelectorAll('.tz-pcap .tz-pcap-row')].map(r=>r.innerText.replace(/\\s+/g,' '))"))
        check(f'{tag} tile {v}: four rows, one control pressed', len(rows[-1]) == 4 and pg.evaluate(f"()=>document.querySelector('.tz-pcap .seg-ctl button[data-v=\"{v}\"]').getAttribute('aria-pressed')") == 'true', rows[-1])
        check(f'{tag} tile {v}: no overflow inside the tile', pg.evaluate("()=>{const t=document.querySelector('.tz-pcap'); return t.scrollWidth<=t.clientWidth+1}"))
        for h_ in pg.evaluate(RULES, '.tz-pcap'): rule_hits.append(f'{tag} tile {v}: {h_}')
        if lang == 'zh':
            for h_ in pg.evaluate(SPACING, '.tz-pcap'): rule_hits.append(f'{tag} tile {v}: Chinese spacing: {h_}')
            for h_ in pg.evaluate(PUNCT, '.tz-pcap'): rule_hits.append(f'{tag} tile {v}: punctuation: {h_}')
        for h_ in pg.evaluate(COLONS, '.tz-pcap'): rule_hits.append(f'{tag} tile {v}: two colons: {h_}')
    seglines(pg, f'{tag} tile')
    bd = pg.evaluate(H7_BOUNDS); check(f'{tag} tile: nothing crosses the tile (H7)', not bd, bd)
    check(f'{tag} tile: the sentence changes with every choice, at most 25 English words', len(set(reads)) == 3 and (lang == 'zh' or all(len(r.split()) <= 25 for r in reads)), reads)
    check(f'{tag} tile: SFT 0.042 with examples; DPO 0.55 with pairs; GRPO 1.15 with a checker', '0.042' in reads[0] and '0.55' in reads[1] and '1.15' in reads[2], reads)
    check(f'{tag} tile: methods the data does not allow are marked, red; SFT runs on the final answers (item 8)', sum('sw over' in x for x in pg.evaluate("()=>[...document.querySelectorAll('.tz-pcap .tz-pcap-no i')].map(i=>i.className)")) == 2 and '0.027' in rows[2][0], rows[2])
    shot(pg, f'{tag}_tile')
    pg.close()


def run_900(b, lang):
    """H5 and H7 at 900 px (the rail is narrow): the five scenarios, Settings, the challenge after Show the answer."""
    tag = f'900_{lang}'
    pg = page(b, 900, 900, lang)
    mode(pg, 'sandbox')
    for i in range(5):
        click(pg, f'#pcap-rail [data-preset="{i}"]', 350)
        b9v6(pg, f'{tag} scenario {i + 1}', Z(lang, ['From the data to the result', 'Memory', 'GPU hours and budget', 'Result'], ['从数据到结果', '显存', 'GPU 小时和预算', '结果']))
        check(f'{tag} scenario {i + 1}: no overflow', overflow(pg) <= 0, overflow(pg)); fit(pg, f'{tag} scenario {i + 1}')
    seglines(pg, f'{tag} settings', want_wrap=True)
    shot(pg, f'{tag}_scenario5')
    mode(pg, 'challenge')
    click(pg, '#pcap-rail [data-act="submit"]', 400)
    click(pg, '#pcap-ch-res [data-act="reveal"]', 400)
    check(f'{tag} challenge: the answer chart is shown', vis(pg, '#pcap-ch-res .cmp-row'))
    b9v6(pg, f'{tag} challenge')
    seglines(pg, f'{tag} challenge')
    check(f'{tag} challenge: no overflow', overflow(pg) <= 0, overflow(pg)); fit(pg, f'{tag} challenge')
    shot(pg, f'{tag}_challenge', full=True)
    pg.close()


for pat in ('2^(', '√(', 'm∞', 'm₀', '^(−', '1.96√', 'σ(', 'Σ'):
    check(f'source: no plain-text formula “{pat}” left in the lab (H1)', pat not in SRC_JS)
check('source: the converted formulas are typeset (A.ui.math)', SRC_JS.count('A.ui.math(') >= 2 and "EQ('m(x)'" in SRC_JS and '\\\\sqrt{\\\\frac{p(1 - p)}{n}}' in SRC_JS)

# B4 (polish r4, reviews/r4-fit-post-b.md): entering parts 4–6 at 1440 × 900 as the polish probe stage.py does (Next when it is
# enabled, else the part's heading), the stage needs no scrolling inside: Chinese with at least 30 px to spare, English with none.
# Part 7 shows all five blocks and cannot fit without hiding one; it scrolls no more than the layout of 2026-10-16 needs (Chinese
# 146 px, English 182 px; 501 px in Chinese before it).
FIT = """()=>{const s=document.getElementById('pcap-stage'); return {spare:Math.round(parseFloat(getComputedStyle(s).maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight)), scroll:s.scrollHeight-s.clientHeight}}"""


def fit_entry(b, lang, parts=(4, 5, 6, 7)):
    pg = page(b, 1440, 900, lang)
    for k in range(2, max(parts) + 1):
        nb = pg.locator('.pbar [data-nav="next"]:visible:not([disabled])')
        if nb.count(): nb.first.click()
        else: pg.evaluate(f"()=>document.querySelector('#pcap-rail .steps>.step:nth-child({k}) .step-h').click()")
        pg.wait_for_timeout(900)
        if k not in parts: continue
        f = pg.evaluate(FIT); f['step'] = state(pg)['step'] + 1
        if k == 7:
            cap = 160 if lang == 'zh' else 200
            check(f'{lang}-1440 part 7: entering it, the stage scrolls inside itself at most {cap} px (B4)', f['step'] == k and f['scroll'] <= cap, f)
        else:
            check(f'{lang}-1440 part {k}: entering it, the stage fits without scrolling inside{", 30 px to spare" if lang == "zh" else ""} (B4, R7.12)',
                  f['step'] == k and f['scroll'] <= 0 and f['spare'] >= (30 if lang == 'zh' else 0), f)
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    if (1440, 900) in SIZES and 'guided' in PARTS:
        for lang in LANGS: fit_entry(b, lang)
    if RUN900:
        for lang in LANGS: run_900(b, lang)
    for (w, h) in SIZES:
        for lang in LANGS:
            if 'guided' in PARTS: run_guided(b, w, h, lang)
            if 'modes' in PARTS: run_modes(b, w, h, lang)
            if 'tile' in PARTS: run_tile(b, w, h, lang)
    if FULL:
        run_guided(b, 1440, 900, 'zh', dark=True)
    b.close()
for hit in rule_hits[:60]:
    check('writing rules: ' + hit, False)
for hit in fit_hits[:60]:
    check(hit, False)
check('no page errors', not errs, errs[:5])
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:60] if fails else 'none')
sys.exit(1 if fails else 0)
