"""Kit: A.ui.cmpChart({brief:true}) after the owner's review of batch 8 (2026-10-07): the challenge's chart shows only the settings
that meet the goal, the reader's pick and the tagged rows (the initial setup), says how many it left out, and folds a long list of
passing rows behind a closed "See all settings that pass (N)" when the lab marks its best rows. Without brief, every row stays.
Checked in EN and ZH at 1440 and 390 on a fixture rendered into a sampling page.
Run: python3 tests/kit_brief_chart.py"""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/kit-brief.html', '--only', 'sampling'], check=True, cwd=P, stdout=subprocess.DEVNULL)
WRAP = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>')
(P / 'out' / 'kit-brief-wrapped.html').write_text(WRAP + (P / 'out' / 'kit-brief.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'kit-brief-wrapped.html')

# 16 rows: row 0 the initial setup (tagged, misses), rows 5, 9, 13 pass; the pick is row 2 (misses) or row 9 (passes).
# The fold case: 12 passing rows, two marked best, the pick misses.
RENDER = r"""(pick)=>{const A=window.ASP, tx=A.tx;
  const mk=(n,okSet,best,me)=>Array.from({length:n},(_,i)=>({label:tx(`Setting ${i}`,`设置 ${i}`),value:10+i*5,text:`${10+i*5}%`,
    ok:okSet.includes(i),best:best.includes(i),me:i===me,tags:(i===0?[{text:tx('initial setup','初始设置')}]:[]).concat(i===me?[{text:tx('your pick','你的选择'),me:true}]:[])}));
  const box=document.createElement('div'); box.id='kb'; box.className='wrap'; box.style.cssText='padding:16px;max-width:760px';
  box.innerHTML=`<div id="kb-a">${A.ui.cmpChart({title:'A',rows:mk(16,[5,9,13],[],pick),goal:50,goalLabel:'goal',max:100,brief:true})}</div>`+
    `<div id="kb-b">${A.ui.cmpChart({title:'B',rows:mk(16,[5,9,13],[],pick),goal:50,goalLabel:'goal',max:100})}</div>`+
    `<div id="kb-c">${A.ui.cmpChart({title:'C',rows:mk(16,[2,3,4,5,6,7,8,9,10,11,12,13],[7,8],1),goal:20,goalLabel:'goal',max:100,brief:true})}</div>`+
    `<div id="kb-d">${A.ui.cmpChart({title:'D',rows:mk(16,[5,9,13],[],pick),goal:50,goalLabel:'goal',max:100,brief:true,hiddenText:n=>tx(`${n} others do not fit.`,`另外 ${n} 种放不下。`)})}</div>`;
  const old=document.getElementById('kb'); if(old) old.remove(); document.body.prepend(box);
  const rows=id=>[...document.querySelectorAll(`#${id} > .cmp .cmp-row`)].map(r=>r.querySelector('.cmp-lab>span').textContent.replace(/\D+/g,''));
  const note=id=>{const e=document.querySelector(`#${id} > .cmp-note`); return e?e.textContent:'';};
  const det=document.querySelector('#kb-c details.cmp-more');
  return {a:rows('kb-a'),b:rows('kb-b').length,c:rows('kb-c'),na:note('kb-a'),nc:note('kb-c'),nd:note('kb-d'),
    det:!!det,open:det?det.open:null,sum:det?det.querySelector('summary').textContent:'',inDet:det?det.querySelectorAll('.cmp-row').length:0,
    over:document.documentElement.scrollWidth>innerWidth+1};}"""

n, fails, errs = 0, [], []


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(f'{name} {info}')
        print('FAIL', name, info)


with sync_playwright() as p:
    b = p.chromium.launch()
    for w, h in ((1440, 900), (390, 844)):
        for lang in ('en', 'zh'):
            pg = b.new_page(viewport={'width': w, 'height': h})
            pg.on('pageerror', lambda e: errs.append(str(e)))
            pg.goto(URL + '#sampling'); pg.wait_for_timeout(300)
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(100)
            t = f'{w} {lang}'
            r = pg.evaluate(RENDER, 2)
            check(f'{t}: brief keeps the initial setup, the pick and the 3 passing rows, in order', r['a'] == ['0', '2', '5', '9', '13'], r['a'])
            check(f'{t}: the note counts the 11 rows left out', ('11' in r['na']) and (('miss the goal' in r['na']) if lang == 'en' else ('没达到目标' in r['na'])), r['na'])
            check(f'{t}: without brief every row stays', r['b'] == 16, r['b'])
            check(f'{t}: fold shows the initial setup, the pick and the 2 best rows', r['c'] == ['0', '1', '7', '8'], r['c'])
            check(f'{t}: fold puts all 12 passing rows behind a closed disclosure', r['det'] and r['open'] is False and r['inDet'] == 12, (r['det'], r['open'], r['inDet']))
            check(f'{t}: the disclosure says how many pass', '12' in r['sum'] and (('See all settings that pass' in r['sum']) if lang == 'en' else ('看全部可行的设置' in r['sum'])), r['sum'])
            check(f'{t}: fold note counts the 2 rows that miss and are not shown (rows 14 and 15)', ('The other 2 options' in r['nc']) if lang == 'en' else ('其余 2 种' in r['nc']), r['nc'])
            check(f'{t}: a lab can word the note', ('11 others do not fit' in r['nd']) if lang == 'en' else ('另外 11 种放不下' in r['nd']), r['nd'])
            check(f'{t}: no horizontal overflow', not r['over'])
            r2 = pg.evaluate(RENDER, 9)
            check(f'{t}: a passing pick is not repeated: initial setup and the 3 passing rows', r2['a'] == ['0', '5', '9', '13'], r2['a'])
            pg.click('#kb-c details.cmp-more summary'); pg.wait_for_timeout(80)
            vis = pg.evaluate("()=>{const d=document.querySelector('#kb-c details.cmp-more'); return d.open&&[...d.querySelectorAll('.cmp-row')].every(r=>r.getBoundingClientRect().height>0)}")
            check(f'{t}: opening the disclosure shows its rows', vis)
            pg.screenshot(path=str(P / 'out' / f'kit-brief-{w}-{lang}.png'), full_page=False)
            pg.close()
    b.close()
print('page errors:', errs or 'none')
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails or 'none')
sys.exit(1 if fails or errs else 0)
