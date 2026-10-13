"""Kit fixes: five gaps in the shared UI kit (app.js + styles.css), checked on real labs and on a small injected fixture.

1. A.ui.cmpChart: a passing row carries a mark at any value (a green ring on its track; 0% too), in light and dark; the goal label
   stays inside the chart with the goal at 0 and at the maximum; value labels stay on one line (1440 and 390), inside the chart.
2. A.ui.levers: a long title is not squeezed by the result (title box at least about 60% of the option); a short title keeps its
   result on the title row (R3.6, R3.10). Train-debug parts 4 and 5 at 1440, and the fixture in a 300 px rail.
3. A.ui.steps({nextWhy}): the line says why Next is disabled, under the card's buttons (desktop) and in the phone's part bar;
   when Next becomes enabled in place, the line hides and keeps its height, so nothing below it moves.
4. The phone part bar (m2): 8 dots and Back/Next on one row at 360 and 390 in Chinese, tap targets at least 24 px
   (fixture with 8 parts; flash-attention and softmax-numerics with 7).
5. Hover terms: a tooltip trigger that is not a native control gets tabindex="0" and an accessible name; Tab reaches it, focus
   shows its tip, Escape hides it, blur hides it; mouse hover still shows it.
Screenshots: shots/kit/<tag>_*.png (tag "after" by default).
Run: python3 tests/kit_fixes.py                     (checks + screenshots)
     python3 tests/kit_fixes.py --shots before DIR  (screenshots only, into DIR; used on an untouched copy of the kit)"""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
ARGS = sys.argv[1:]
SHOTS_ONLY = '--shots' in ARGS
TAG = ARGS[ARGS.index('--shots') + 1] if SHOTS_ONLY else 'after'
SH = pathlib.Path(ARGS[ARGS.index('--shots') + 2]) if SHOTS_ONLY and len(ARGS) > ARGS.index('--shots') + 2 else P / 'shots' / 'kit'
SH.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/kit-fixes.html', '--only', 'sampling,softmax-numerics,train-debug,flash-attention'],
               check=True, cwd=P, stdout=subprocess.DEVNULL)

# a small lab registered through window.ASP: 8 guided parts (m2), levers with a long title, three comparison charts
# (goal at 0, goal at the maximum, 0% passing rows), a defined term, a legend item and a swatch with hover tips
FIXTURE = r"""
(function(){'use strict'; const A=window.ASP, tx=A.tx, esc=A.esc; let step=0, open=false, root=null;
const T=[['Mission','任务'],['Predict','预测'],['A healthy run','健康的运行'],['The learning rate','学习率'],['Three bugs','三个 bug'],['The checks','检查'],['Explain it','讲清楚'],['What to remember','要记住的']];
const rows0=()=>[{label:tx('Initial setup: FP16 scores, no max subtracted','初始设置：FP16 分数，不减最大值'),value:0,text:'0%',ok:true,tags:[{text:tx('initial setup','初始设置')}]},
  {label:tx('Subtract the max, FP16','减去最大值，FP16'),value:0,text:'0%',ok:true,me:true,tags:[{text:tx('your pick','你的选择'),me:true}]},
  {label:tx('BF16 scores','BF16 分数'),value:3.2,text:'3.2%'},{label:tx('FP8 scores','FP8 分数'),value:9.6,text:'9.6%'}];
const rowsM=()=>[{label:tx('One GPU','一张 GPU'),value:42,text:tx('4,200.5 tokens/s','4,200.5 tokens/s')},
  {label:tx('Two GPUs, tensor parallel','两张 GPU，张量并行'),value:97,text:tx('9,700.25 tokens/s','9,700.25 tokens/s')},
  {label:tx('Eight GPUs','八张 GPU'),value:100,text:tx('10,000.75 tokens/s','10,000.75 tokens/s'),ok:true,me:true,tags:[{text:tx('your pick','你的选择'),me:true}]}];
const rows1=()=>[1,0.95,0.9,0.85].map((p,i)=>({label:tx(`T 0.5, top-p ${p.toFixed(2)}`,`T 0.5，top-p ${p.toFixed(2)}`),value:[7.5,2.4,0,0][i],text:['7.5%','2.4%','0%','0%'][i],ok:i>=2}));
function rail(){
  const body=`<p>${tx('A card with a ','一张卡片，里面有')}${A.ui.def('KV cache',tx('The keys and values a model keeps for earlier tokens.','模型为之前的 token 存下的 key 和 value。'))}${tx(', a ','、一个')}<span class="kf-leg" data-tipx="${esc(tx('A legend item with a hover tip.','一个带悬停说明的图例项。'))}">${tx('legend item','图例项')}</span>${tx(' and a swatch ','和一个色块')}<i class="sw kv kf-sw" data-tipx="${esc(tx('Amber: the KV cache.','琥珀色：KV cache。'))}"></i>.</p>`+
    `<div class="kf-lv">`+A.ui.levers({aria:tx('Learning rate, pick one','学习率，一次选一个'),value:'b',options:[
      {v:'a',title:tx('Initial setup','初始设置'),desc:tx('The healthy learning rate.','健康的学习率。'),result:tx('0.135','0.135')},
      {v:'b',title:tx('Ten times the healthy learning rate, with warm-up turned off','学习率调到健康值的十倍，同时关掉 warm-up'),desc:tx('Only the learning rate changes.','只改学习率。'),result:tx('NaN at step 7','第 7 步变成 NaN')},
      {v:'c',title:tx('A hundredth of the healthy learning rate','学习率调到健康值的百分之一'),desc:tx('Only the learning rate changes.','只改学习率。'),result:tx('loss 0.523 after 300 steps','300 步后 loss 0.523')}]})+`</div>`+
    `<p><button type="button" class="btn ghost sm" id="kf-en">${tx('Enable Next','启用下一部分')}</button> <button type="button" class="btn ghost sm" id="kf-re">${tx('Re-render','重画')}</button></p>`;
  return A.ui.steps({steps:T.map(t=>({title:tx(t[0],t[1]),lead:esc(tx('One line about this part.','这一部分的一句话。'))})),m2:true,current:step,body,back:step>0,
    next:step<T.length-1,nextDisabled:!open,nextWhy:tx('Pick an option to continue, or Skip.','选一个选项才能继续，也可以跳过。'),skip:true});}
function stage(){return `<div class="stage-head"><h2 class="stage-title">${tx('Fixture charts','测试图表')}</h2></div>`+
  `<div id="kf-c0">`+A.ui.cmpChart({title:tx('Error of each fix (goal: 0%)','每种做法的误差（目标：0%）'),rows:rows0(),goal:0,goalLabel:tx('goal: 0%','目标：0%'),max:10})+`</div>`+
  `<div id="kf-cm">`+A.ui.cmpChart({title:tx('Throughput of each setup (goal: 10,000 tokens/s)','每种配置的吞吐（目标：10,000 tokens/s）'),rows:rowsM(),goal:100,goalLabel:tx('goal: 10,000 tokens/s','目标：10,000 tokens/s'),max:100})+`</div>`+
  `<div id="kf-c1">`+A.ui.cmpChart({title:tx('Chance of a wrong word (goal: below 1%)','抽中错词的概率（目标：低于 1%）'),rows:rows1(),goal:1,goalLabel:tx('goal: below 1%','目标：低于 1%'),max:14})+`</div>`;}
function draw(){root.innerHTML=`<div class="kf-app m2"><div class="wrap labhead"><h1 class="labtitle">${tx('Kit fixture','kit 测试页')}</h1></div><div class="wrap bench"><aside class="rail" id="kf-rail">${rail()}</aside>`+
  `<div class="panel stage" id="kf-stage">${stage()}</div><aside class="panel controls" id="kf-ctl"><div class="eyebrow">${tx('Settings','自由调参')}</div><p class="small">${tx('Nothing to set.','这里没有设置。')}</p></aside></div></div>`;}
A.registerLab({id:'_kitfx',mount(r){root=r; step=0; open=false; draw();
  r.addEventListener('click',e=>{const b=e.target.closest('button'); if(!b) return;
    if(b.id==='kf-en'){open=true; A.$$('[data-nav="next"]',r).forEach(x=>{x.disabled=false;}); return;}
    if(b.id==='kf-re'){draw(); return;}
    if(b.dataset.nav==='next'||b.dataset.nav==='skip'){step=Math.min(T.length-1,step+1); open=false; draw(); return;}
    if(b.dataset.nav==='back'){step=Math.max(0,step-1); open=false; draw(); return;}
    if(b.dataset.go!=null){step=+b.dataset.go; open=false; draw();}});},
  leave(){root=null;}, onLang(){if(root) draw();}});
})();
"""
WRAP = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>')
page_html = (P / 'out' / 'kit-fixes.html').read_text()
(P / 'out' / 'kit-fixes-wrapped.html').write_text(WRAP + page_html.replace('window.ASP.start();', FIXTURE + '\nwindow.ASP.start();') + '</body></html>')
URL = 'file://' + str(P / 'out' / 'kit-fixes-wrapped.html')
errs, fails, n = [], [], 0


def check(name, ok, info=''):
    global n
    if SHOTS_ONLY: return
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:500])


def open_page(b, w, h, lang, hash_, dark=False):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light')
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang} {hash_}: {e}'))
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(500)
    return pg


def click(pg, sel, ms=350):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(ms)


def mode(pg, m):
    click(pg, f'#view-dyn .modes button[data-mode="{m}"]', 500)


def shot(pg, name, sel=None):
    path = str(SH / f'{TAG}_{name}.png')
    if sel and pg.evaluate(f"()=>!!document.querySelector('{sel}')"):
        pg.evaluate("()=>{const s=document.createElement('style'); s.id='kf-shot'; s.textContent='.topbar,#dock{visibility:hidden!important}'; document.head.appendChild(s);}")
        pg.evaluate(f"()=>document.querySelector('{sel}').scrollIntoView({{block:'center'}})"); pg.wait_for_timeout(150)
        pg.locator(sel).first.screenshot(path=path)
        pg.evaluate("()=>{const s=document.getElementById('kf-shot'); if(s) s.remove();}")
    else:
        pg.screenshot(path=path)


# ---------- measurements (run in the page) ----------
CMP_GEO = """(sel)=>{const okc=(()=>{const d=document.createElement('i'); d.style.color=getComputedStyle(document.documentElement).getPropertyValue('--c-ok').trim(); document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;})();
  const out=[]; document.querySelectorAll(sel).forEach(c=>{const cr=c.getBoundingClientRect(); if(!cr.width) return;
    const ax=c.querySelector('.cmp-axis'), lab=ax&&(ax.querySelector('.cmp-gl')||ax.querySelector('span')), line=c.querySelector('.cmp-track .g'), tr=c.querySelector('.cmp-track');
    const rows=[...c.querySelectorAll('.cmp-row')].map(r=>{const t=r.querySelector('.cmp-track'), v=r.querySelector('.cmp-val'), vr=v.getBoundingClientRect(), lh=parseFloat(getComputedStyle(v).lineHeight)||parseFloat(getComputedStyle(v).fontSize)*1.25,
        bar=t.querySelector('i'), bs=getComputedStyle(t).boxShadow+' '+getComputedStyle(t).outlineColor+' '+getComputedStyle(t).outlineStyle, mark=r.querySelector('.cmp-ok');
      return {ok:r.classList.contains('ok'), w:bar.getBoundingClientRect().width, ring:bs.includes(okc), mark:!!mark&&mark.getClientRects().length>0,
        oneLine:vr.height<=lh*1.45+2&&v.getClientRects().length===1, inChart:vr.left>=cr.left-0.5&&vr.right<=cr.right+0.5, text:v.textContent, h:vr.height, lh};});
    let goal=null; if(lab&&line){const a=lab.getBoundingClientRect(), g=line.getBoundingClientRect(), t=tr.getBoundingClientRect();
      goal={inside:a.left>=t.left-0.5&&a.right<=t.right+0.5&&a.left>=cr.left-0.5&&a.right<=cr.right+0.5, covers:a.left<=g.left+1&&a.right>=g.left-1, fits:lab.scrollWidth<=lab.clientWidth+1, oneLine:a.height<24,
        a:[Math.round(a.left),Math.round(a.right)], t:[Math.round(t.left),Math.round(t.right)], g:Math.round(g.left)};}
    out.push({rows, goal});}); return out;}"""
LEVER_GEO = """(sel)=>[...document.querySelectorAll(sel)].filter(b=>b.getClientRects().length).map(b=>{const t=b.querySelector('.lv-t b'), r=b.querySelector('.lv-r'), br=b.getBoundingClientRect(), tr=t.getBoundingClientRect(), rr=r.getBoundingClientRect(),
  lh=parseFloat(getComputedStyle(t).lineHeight)||20;
  return {v:b.dataset.lever, ratio:tr.width/br.width, lines:Math.round(tr.height/lh), res:r.textContent, sameRow:!!r.textContent&&Math.abs(rr.top-tr.top)<8, under:!!r.textContent&&rr.top>=tr.bottom-2,
    resInside:!r.textContent||(rr.left>=br.left-0.5&&rr.right<=br.right+0.5), resOneLine:!r.textContent||rr.height<(parseFloat(getComputedStyle(r).lineHeight)||18)*1.5};})"""
PBAR_GEO = """(sel)=>{const p=document.querySelector(sel); if(!p||!p.getClientRects().length) return null; const row=p.querySelector('.pb-row'), dots=[...p.querySelectorAll('.pb-dot')], nav=[...p.querySelectorAll('.pb-nav button')];
  const R=e=>e.getBoundingClientRect(), mid=e=>{const r=R(e); return (r.top+r.bottom)/2;}, pr=R(p);
  const tops=dots.map(d=>Math.round(R(d).top)), mids=dots.concat(nav).map(mid);
  return {n:dots.length, oneRow:Math.max(...mids)-Math.min(...mids)<6, rowH:R(row).height, minDot:Math.min(...dots.map(d=>Math.min(R(d).width,R(d).height))),
    minNav:Math.min(...nav.map(b=>Math.min(R(b).width,R(b).height))), inside:dots.concat(nav).every(e=>R(e).left>=pr.left-0.5&&R(e).right<=pr.right+0.5),
    nav:nav.map(b=>b.textContent), overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth, tops};}"""


def cmp_checks(pg, tag, sel, want_goal=True, want_val=True):
    geo = pg.evaluate(CMP_GEO, sel)
    check(f'{tag}: chart found ({sel})', len(geo) > 0, geo)
    for i, c in enumerate(geo):
        oks = [r for r in c['rows'] if r['ok']]
        if oks:
            check(f'{tag} chart {i}: every passing row is marked at any value, 0% too (gap 1a)', all(r['ring'] or r['mark'] for r in oks), oks)
            check(f'{tag} chart {i}: a missing row carries no pass mark', not any(r['ring'] or r['mark'] for r in c['rows'] if not r['ok']), c['rows'])
        if want_goal and c['goal']:
            check(f'{tag} chart {i}: the goal label sits inside the chart and on the goal line (gap 1b)', c['goal']['inside'] and c['goal']['covers'] and c['goal']['fits'] and c['goal']['oneLine'], c['goal'])
        if want_val:
            check(f'{tag} chart {i}: value labels on one line, inside the chart (gap 1c)', all(r['oneLine'] and r['inChart'] for r in c['rows']), [(r['text'], r['h'], r['lh'], r['inChart']) for r in c['rows'] if not (r['oneLine'] and r['inChart'])])
    return geo


def fixture(b, w, h, lang, dark=False):
    pg = open_page(b, w, h, lang, '_kitfx', dark)
    tag = f'fixture {w} {lang}' + (' dark' if dark else '')
    phone = w <= 760
    check(f'{tag}: the fixture mounted', pg.evaluate("()=>!!document.querySelector('#kf-rail .steps')"))
    # gap 1: the three charts
    cmp_checks(pg, tag, '#kf-stage .cmp')
    shot(pg, f'cmp_fixture_{w}_{lang}' + ('_dark' if dark else ''), '#kf-stage')
    if dark:
        pg.close(); return
    # gap 2: levers in the rail (300 px wide at 1440)
    lv = pg.evaluate(LEVER_GEO, '#kf-rail .kf-lv .lever')
    check(f'{tag}: the long lever title keeps at least about 60% of the option (gap 2)', lv[1]['ratio'] >= 0.6, lv[1])
    check(f'{tag}: no lever title squeezed: on one line, or at least about 60% of the option (gap 2)', all(x['lines'] <= 1 or x['ratio'] >= 0.6 for x in lv), lv)
    check(f'{tag}: a short title keeps its result on the title row (R3.6)', lv[0]['sameRow'], lv[0])
    check(f'{tag}: every result is on the title row or right under the title, inside the option, on one line', all((x['sameRow'] or x['under']) and x['resInside'] and x['resOneLine'] for x in lv), lv)
    shot(pg, f'levers_fixture_{w}_{lang}', '#kf-rail .kf-lv')
    # gap 3: why Next is disabled
    why_sel = '#kf-rail .pbar [data-next-why]' if phone else '#kf-rail .step.current [data-next-why]'
    nx_sel = '#kf-rail .pbar [data-nav="next"]' if phone else '#kf-rail .step.current .stepnav [data-nav="next"]'
    below = '#kf-rail .pbar .pb-t' if phone else '#kf-rail .step.current .stepnav'
    g0 = pg.evaluate(f"""()=>{{const p=document.querySelector('{why_sel}'), nb=document.querySelector('{nx_sel}'); if(!p) return null; const r=p.getBoundingClientRect(), cs=getComputedStyle(p);
      return {{text:p.textContent, vis:cs.visibility==='visible'&&r.height>0&&p.getClientRects().length>0, h:r.height, dis:nb.disabled, all:document.querySelectorAll('#kf-rail [data-next-why]').length}}}}""")
    want = 'Pick an option to continue, or Skip.' if lang == 'en' else '选一个选项才能继续，也可以跳过。'
    check(f'{tag}: the why line shows for a disabled Next, in the card and the part bar (gap 3)', bool(g0) and g0['vis'] and g0['dis'] and g0['text'] == want and g0['all'] == 2, g0)
    shot(pg, f'nextwhy_fixture_{w}_{lang}_disabled', '#kf-rail .pbar' if phone else '#kf-rail .step.current')
    if g0:
        if not phone:
            vis_pb = pg.evaluate("()=>{const p=document.querySelector('#kf-rail .pbar'); return !!p&&p.getClientRects().length>0}")
            check(f'{tag}: the phone bar stays hidden on a desktop', not vis_pb)
        y_after = pg.evaluate("()=>{const c=document.querySelector('#kf-rail .step.current'); return c.getBoundingClientRect().bottom}")
        y_t = pg.evaluate(f"()=>document.querySelector('{below}').getBoundingClientRect().top")
        click(pg, '#kf-en')
        g1 = pg.evaluate(f"""()=>{{const p=document.querySelector('{why_sel}'), nb=document.querySelector('{nx_sel}'), r=p.getBoundingClientRect(), cs=getComputedStyle(p);
          return {{vis:cs.visibility, h:r.height, dis:nb.disabled}}}}""")
        y_after2 = pg.evaluate("()=>{const c=document.querySelector('#kf-rail .step.current'); return c.getBoundingClientRect().bottom}")
        y_t2 = pg.evaluate(f"()=>document.querySelector('{below}').getBoundingClientRect().top")
        check(f'{tag}: Next enabled in place: the line hides and keeps its height, nothing moves (gap 3)', not g1['dis'] and g1['vis'] == 'hidden' and abs(g1['h'] - g0['h']) < 0.5 and abs(y_after2 - y_after) < 0.5 and abs(y_t2 - y_t) < 0.5, (g0, g1, y_after, y_after2, y_t, y_t2))
        shot(pg, f'nextwhy_fixture_{w}_{lang}_enabled', '#kf-rail .pbar' if phone else '#kf-rail .step.current')
        # rendered while Next is enabled: no room taken; a lab that disables Next in place gets the line back (the kit's observer)
        click(pg, '#kf-re')
        g3 = pg.evaluate(f"()=>{{const p=document.querySelector('{why_sel}'); return {{hidden:p.hidden, h:p.getBoundingClientRect().height}}}}")
        pg.evaluate("()=>document.querySelectorAll('#kf-rail [data-nav=\"next\"]').forEach(b=>{b.disabled=true;})"); pg.wait_for_timeout(120)
        g4 = pg.evaluate(f"()=>{{const p=document.querySelector('{why_sel}'); return {{hidden:p.hidden, vis:getComputedStyle(p).visibility, h:p.getBoundingClientRect().height}}}}")
        check(f'{tag}: rendered with Next enabled the line takes no room; disabled in place, it shows (gap 3)', g3['hidden'] and g3['h'] == 0 and not g4['hidden'] and g4['vis'] == 'visible' and g4['h'] > 0, (g3, g4))
        pg.evaluate("()=>window.ASP.ui.setNext(document.getElementById('kf-rail'), false)"); pg.wait_for_timeout(120)
        g5 = pg.evaluate(f"()=>{{const p=document.querySelector('{why_sel}'); return {{vis:getComputedStyle(p).visibility, h:p.getBoundingClientRect().height}}}}")
        check(f'{tag}: A.ui.setNext enables every Next; the line keeps its height (gap 3)', g5['vis'] == 'hidden' and abs(g5['h'] - g4['h']) < 0.5 and pg.evaluate("()=>[...document.querySelectorAll('#kf-rail [data-nav=\"next\"]')].every(b=>!b.disabled)"), (g4, g5))
        click(pg, '#kf-rail .step.current [data-nav="next"]')
        g2 = pg.evaluate(f"()=>{{const p=document.querySelector('{why_sel}'); return !!p&&getComputedStyle(p).visibility==='visible'}}")
        check(f'{tag}: the next part renders its why line again while Next is disabled', g2)
    # gap 4: the part bar on phones (8 parts)
    if phone:
        pb = pg.evaluate(PBAR_GEO, '#kf-rail .pbar')
        check(f'{tag}: 8 dots, Back and Next on one row, tap targets ≥ 24 px, inside the bar (gap 4)', pb and pb['n'] == 8 and pb['oneRow'] and pb['minDot'] >= 24 and pb['minNav'] >= 24 and pb['inside'] and pb['overflow'] <= 0, pb)
        shot(pg, f'pbar_fixture_{w}_{lang}', '#kf-rail .pbar')
        if w == 360:   # every phone width from 360 to 760: one row for 8 parts, with Back
            bad = []
            for ww in (360, 370, 380, 389, 400, 420, 440, 460, 481, 520, 600, 700, 760):
                pg.set_viewport_size({'width': ww, 'height': h}); pg.wait_for_timeout(120)
                g = pg.evaluate(PBAR_GEO, '#kf-rail .pbar')
                if not (g and g['n'] == 8 and len(g['nav']) == 2 and g['oneRow'] and g['minDot'] >= 24 and g['minNav'] >= 24 and g['inside'] and g['overflow'] <= 0): bad.append((ww, g))
            check(f'{tag}: 8 parts with Back and Next on one row at every phone width 360–760 (gap 4)', not bad, bad[:2])
            pg.set_viewport_size({'width': w, 'height': h}); pg.wait_for_timeout(150)
    # gap 5: hover terms by keyboard
    pg.evaluate("()=>{const d=document.querySelector('#kf-rail .step.current .def'); d.scrollIntoView({block:'center'}); d.focus();}"); pg.wait_for_timeout(150)
    t0 = pg.evaluate("()=>({tip:!document.getElementById('tip').hidden, txt:document.getElementById('tip').textContent})")
    check(f'{tag}: a defined term shows its tip on focus', t0['tip'] and ('earlier tokens' in t0['txt'] or '之前的 token' in t0['txt']), t0)
    pg.keyboard.press('Tab'); pg.wait_for_timeout(150)
    t1 = pg.evaluate("""()=>{const a=document.activeElement, tip=document.getElementById('tip'); return {cls:a.className, ti:a.getAttribute('tabindex'), tip:!tip.hidden, txt:tip.textContent,
      name:(a.getAttribute('aria-label')||a.textContent||'').trim(), desc:a.getAttribute('aria-describedby')}}""")
    check(f'{tag}: Tab reaches the legend item, which gets tabindex 0 and a name (gap 5)', t1['cls'] == 'kf-leg' and t1['ti'] == '0' and bool(t1['name']), t1)
    check(f'{tag}: its tip shows on focus (gap 5)', t1['tip'] and ('legend item with a hover tip' in t1['txt'] or '带悬停说明的图例项' in t1['txt']), t1)
    shot(pg, f'tipfocus_fixture_{w}_{lang}')
    pg.keyboard.press('Escape'); pg.wait_for_timeout(120)
    t2 = pg.evaluate("()=>({tip:!document.getElementById('tip').hidden, cls:document.activeElement.className})")
    check(f'{tag}: Escape hides the tip and keeps the focus (gap 5)', not t2['tip'] and t2['cls'] == 'kf-leg', t2)
    pg.keyboard.press('Tab'); pg.wait_for_timeout(150)
    t3 = pg.evaluate("""()=>{const a=document.activeElement, tip=document.getElementById('tip'); return {cls:a.className, tip:!tip.hidden, label:a.getAttribute('aria-label')||'', txt:tip.textContent}}""")
    check(f'{tag}: the next Tab reaches the swatch, which has no text: its name is the tip (gap 5)', 'kf-sw' in t3['cls'] and t3['tip'] and t3['label'] == t3['txt'], t3)
    pg.keyboard.press('Shift+Tab'); pg.wait_for_timeout(120)
    t4 = pg.evaluate("()=>({tip:!document.getElementById('tip').hidden, cls:document.activeElement.className, txt:document.getElementById('tip').textContent})")
    check(f'{tag}: focus back on the legend item shows its tip again; the swatch’s tip went with its blur', t4['tip'] and t4['cls'] == 'kf-leg' and ('legend' in t4['txt'] or '图例' in t4['txt']), t4)
    pg.evaluate("()=>document.activeElement.blur()"); pg.wait_for_timeout(100)
    check(f'{tag}: blur hides the tip', pg.evaluate("()=>document.getElementById('tip').hidden"))
    if not phone:   # mouse hover is unchanged: hover shows, leaving hides
        pg.hover('#kf-rail .kf-leg'); pg.wait_for_timeout(150)
        h1 = pg.evaluate("()=>!document.getElementById('tip').hidden")
        pg.mouse.move(5, 5); pg.wait_for_timeout(150)
        h2 = pg.evaluate("()=>!document.getElementById('tip').hidden")
        check(f'{tag}: mouse hover still shows the tip and leaving hides it', h1 and not h2, (h1, h2))
        pg.hover('#kf-rail .kf-leg'); pg.wait_for_timeout(150); pg.keyboard.press('Escape'); pg.wait_for_timeout(100)
        e1 = pg.evaluate("()=>!document.getElementById('tip').hidden")
        pg.mouse.move(pg.evaluate("()=>{const r=document.querySelector('#kf-rail .kf-leg').getBoundingClientRect(); return r.left+3}"), pg.evaluate("()=>{const r=document.querySelector('#kf-rail .kf-leg').getBoundingClientRect(); return r.top+r.height/2}")); pg.wait_for_timeout(120)
        e2 = pg.evaluate("()=>!document.getElementById('tip').hidden")
        check(f'{tag}: Escape also hides a hover tip, which stays away while the pointer stays on the item', not e1 and not e2, (e1, e2))
    pg.close()


def run_sampling(b, w, lang, dark=False):
    pg = open_page(b, w, 900 if w > 760 else 844, lang, 'sampling', dark)
    tag = f'sampling {w} {lang}' + (' dark' if dark else '')
    mode(pg, 'challenge')
    click(pg, '#sp-rail .seg-ctl[data-key="chT"] button[data-v="0.5"]'); click(pg, '#sp-rail [data-act="submit"]'); click(pg, '#sp-ch-res [data-act="reveal"]')
    cmp_checks(pg, tag, '#sp-ch-res .cmp')
    shot(pg, f'cmp_sampling_{w}_{lang}' + ('_dark' if dark else ''), '#sp-ch-res .cmp')
    pg.close()


def run_softmax(b, w, lang):
    pg = open_page(b, w, 900 if w > 760 else 844, lang, 'softmax-numerics')
    tag = f'softmax {w} {lang}'
    if w <= 760:
        for ww in (390, 360):
            pg.set_viewport_size({'width': ww, 'height': 844}); pg.wait_for_timeout(350)
            pb = pg.evaluate(PBAR_GEO, '#sm-rail .pbar')
            check(f'{tag}: at {ww}, the {pb and pb["n"]} dots and Back/Next stay on one row, tap targets ≥ 24 px (gap 4)', pb and pb['oneRow'] and pb['minDot'] >= 24 and pb['minNav'] >= 24 and pb['inside'] and pb['overflow'] <= 0, pb)
            click(pg, '#sm-rail .pbar [data-go="3"]', 500)
            pb = pg.evaluate(PBAR_GEO, '#sm-rail .pbar')
            check(f'{tag}: at {ww}, part 4 with Back: one row (gap 4)', pb and pb['oneRow'] and pb['minNav'] >= 24 and pb['overflow'] <= 0, pb)
            shot(pg, f'pbar_softmax_{ww}_{lang}', '#sm-rail .pbar')
            click(pg, '#sm-rail .pbar [data-go="0"]', 500)
        pg.set_viewport_size({'width': 390, 'height': 844}); pg.wait_for_timeout(350)
    mode(pg, 'challenge')
    click(pg, '#sm-rail [data-lever="bf16"]'); click(pg, '#sm-rail [data-act="submit"]'); click(pg, '#sm-ch-res [data-act="reveal"]')
    cmp_checks(pg, tag, '#sm-ch-res .cmp', want_goal=False)
    shot(pg, f'cmp_softmax_{w}_{lang}', '#sm-ch-res .cmp')
    pg.close()


def run_fa(b, w, lang):
    pg = open_page(b, w, 900 if w > 760 else 844, lang, 'flash-attention')
    tag = f'flash-attention {w} {lang}'
    if w <= 760:
        for ww in (390, 360):
            pg.set_viewport_size({'width': ww, 'height': 844}); pg.wait_for_timeout(350)
            pb = pg.evaluate(PBAR_GEO, '#fa-rail .pbar')
            check(f'{tag}: at {ww}, the {pb and pb["n"]} dots and Next stay on one row, tap targets ≥ 24 px (gap 4)', pb and pb['oneRow'] and pb['minDot'] >= 24 and pb['minNav'] >= 24 and pb['inside'] and pb['overflow'] <= 0, pb)
            click(pg, '#fa-rail .pbar [data-go="3"]', 500)
            pb = pg.evaluate(PBAR_GEO, '#fa-rail .pbar')
            check(f'{tag}: at {ww}, part 4 with Back: one row (gap 4)', pb and pb['oneRow'] and pb['minNav'] >= 24 and pb['overflow'] <= 0, pb)
            shot(pg, f'pbar_fa_{ww}_{lang}', '#fa-rail .pbar')
            click(pg, '#fa-rail .pbar [data-go="0"]', 500)
        pg.set_viewport_size({'width': 390, 'height': 844}); pg.wait_for_timeout(350)
    mode(pg, 'challenge')
    click(pg, '#fa-rail [data-act="submit"]'); click(pg, '#fa-ch-res [data-act="reveal"]')
    cmp_checks(pg, tag, '#fa-ch-res .cmp', want_goal=False)
    shot(pg, f'cmp_fa_{w}_{lang}', '#fa-ch-res .cmp')
    pg.close()


def run_td(b, w, lang):
    pg = open_page(b, w, 900 if w > 760 else 844, lang, 'train-debug')
    tag = f'train-debug {w} {lang}'
    click(pg, '#td-rail [data-nav="next"]'); click(pg, '#td-rail [data-pred="high"]'); click(pg, '#td-rail [data-nav="next"]')
    click(pg, '#td-rail .seg-ctl[data-key="steps"] button[data-v="300"]', 600); click(pg, '#td-rail [data-nav="next"]', 500)
    click(pg, '#td-ic-lr [data-lever="10"]', 600)
    lv = pg.evaluate(LEVER_GEO, '#td-ic-lr .lever')
    check(f'{tag} part 4: no lever title squeezed by its result: on one line, or at least about 60% of the option (gap 2)', all(x['lines'] <= 1 or x['ratio'] >= 0.6 for x in lv), lv)
    check(f'{tag} part 4: the result on the title row or right under it, inside the option, on one line', all((x['sameRow'] or x['under']) and x['resInside'] and x['resOneLine'] for x in lv if x['res']), lv)
    shot(pg, f'levers_td_p4_{w}_{lang}', '#td-rail .step.current')
    click(pg, '#td-ic-lr [data-lever="0.003"]', 600); click(pg, '#td-rail [data-nav="next"]', 500)
    for v in ('nozero', 'twice', 'shuffle'): click(pg, f'#td-ic-bug [data-lever="{v}"]', 600)
    lv = pg.evaluate(LEVER_GEO, '#td-ic-bug .lever')
    check(f'{tag} part 5: no lever title squeezed by its result: on one line, or at least about 60% of the option (gap 2)', all(x['lines'] <= 1 or x['ratio'] >= 0.6 for x in lv), lv)
    check(f'{tag} part 5: every result on the title row or right under it, inside the option, on one line', all((x['sameRow'] or x['under']) and x['resInside'] and x['resOneLine'] for x in lv if x['res']), lv)
    shot(pg, f'levers_td_p5_{w}_{lang}', '#td-rail .step.current')
    # gap 5 on a real lab: every hover-tip trigger on the page is reachable by keyboard and has a name
    au = pg.evaluate("""()=>{const bad=[]; document.querySelectorAll('[data-tip],[data-tipx],[data-tipfn],.t,[data-jump]').forEach(e=>{if(!e.getClientRects().length||e.closest('[aria-hidden="true"],#mapSvg,[inert]')) return;
      const nat=e.matches('a[href],button,input,select,textarea,summary,iframe,[contenteditable="true"]'), ti=e.getAttribute('tabindex'), name=(e.getAttribute('aria-label')||e.getAttribute('aria-labelledby')||e.textContent||'').trim();
      if(e.closest('button,a[href],label,summary,[role="button"],[role="link"]')!==e&&e.parentElement&&e.parentElement.closest('button,a[href],summary,[role="button"],[role="link"]')) return;
      if(!(nat||ti==='0'||(ti!==null&&+ti>=0))||!name) bad.push(e.outerHTML.slice(0,140));}); return bad;}""")
    check(f'{tag}: every visible hover-tip trigger outside an aria-hidden box is focusable and named (gap 5)', not au, au[:6])
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ('zh', 'en'):
        for w, h in ((1440, 900), (390, 844), (360, 780)):
            if SHOTS_ONLY and lang == 'en': continue
            if w == 360 and not (lang == 'zh'): continue
            fixture(b, w, h, lang)
        if lang == 'zh' or not SHOTS_ONLY:
            fixture(b, 1440, 900, lang, dark=True)
    for lang in ('zh', 'en'):
        if SHOTS_ONLY and lang == 'en': continue
        for w in (1440, 390):
            run_sampling(b, w, lang)
            run_softmax(b, w, lang)
            run_fa(b, w, lang)
            run_td(b, w, lang)
        run_sampling(b, 1440, lang, dark=True)
    b.close()

if SHOTS_ONLY:
    print('shots in', SH, 'errors:', errs[:5])
else:
    print(f'kit_fixes: {n - len(fails)}/{n} checks passed; page errors: {len(errs)}')
    for e in errs[:10]: print('ERR', e)
    sys.exit(1 if fails or errs else 0)
