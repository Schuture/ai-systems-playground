"""The challenge result after Submit, in every lab given (the owner's review of batch 8, 2026-10-07): submit the initial setup, open
“Show the answer” if the lab offers it, then report what the result shows. A brief comparison chart (A.ui.cmpChart({brief:true}))
keeps only the passing rows, the pick and the tagged rows; a row that is none of these is a “stray” and fails. Also fails on page
errors and horizontal overflow; reports open disclosures and tables in the result. Screenshots of the result box: out/chal/.
Run: python3 tests/chal_result_check.py <lab> [<lab> …]            (builds out/chal-check.html with those labs)
     CHAL_PAGE=ai-systems-playground.html python3 tests/chal_result_check.py <lab> …   (uses a page already built)"""
import os, pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
IDS = [a for a in sys.argv[1:] if not a.startswith('-')]
if not IDS:
    sys.exit(__doc__)
SRC = os.environ.get('CHAL_PAGE')
if not SRC:
    subprocess.run([sys.executable, str(P / 'build.py'), f'out/chal-check-{os.getpid()}.html', '--only', ','.join(IDS)], check=True, cwd=P, stdout=subprocess.DEVNULL)
    SRC = f'out/chal-check-{os.getpid()}.html'
WRAP = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>')
WRAPPED = P / 'out' / f'chal-check-wrapped-{os.getpid()}.html'   # one page per run, so runs at the same time do not clash
WRAPPED.write_text(WRAP + (P / SRC).read_text() + '</body></html>')
URL = 'file://' + str(WRAPPED)
SH = P / 'out' / 'chal'; SH.mkdir(parents=True, exist_ok=True)

INFO = r"""()=>{const res=document.querySelector('[id$="-ch-res"],[id$="-chres"]'); if(!res) return {none:true};
  const cmp=res.querySelector(':scope .cmp:not(details .cmp)');
  const rows=cmp?[...cmp.querySelectorAll('.cmp-row')]:[];
  const tagged=r=>!!r.querySelector('.cmp-tag,.cmp-you');
  const brief=!!res.querySelector('.cmp-note,.cmp-more')||(cmp&&rows.every(r=>r.classList.contains('ok')||r.classList.contains('me')||tagged(r)));
  return {rows:rows.length, ok:rows.filter(r=>r.classList.contains('ok')).length, me:rows.filter(r=>r.classList.contains('me')).length,
    tag:rows.filter(tagged).length, stray:rows.filter(r=>!r.classList.contains('ok')&&!r.classList.contains('me')&&!tagged(r)).length,
    charts:res.querySelectorAll('.cmp').length, note:(res.querySelector('.cmp-note')||{}).textContent||'',
    more:[...res.querySelectorAll('details')].map(d=>(d.open?'OPEN ':'closed ')+(d.querySelector('summary')||{}).textContent),
    tables:[...res.querySelectorAll('table')].map(t=>t.querySelectorAll('tbody tr').length),
    reveal:(res.querySelector('[data-act="reveal"]')||{}).textContent||'',
    first:(res.firstElementChild||{}).textContent?.slice(0,90)||'', h:Math.round(res.getBoundingClientRect().height),
    over:document.documentElement.scrollWidth>innerWidth+1};}"""

bad = []
with sync_playwright() as p:
    b = p.chromium.launch()
    for lab in IDS:
        for w, h, lang in ((1440, 900, 'zh'), (390, 844, 'zh'), (1440, 900, 'en')):
            errs = []
            pg = b.new_page(viewport={'width': w, 'height': h})
            pg.on('pageerror', lambda e: errs.append(str(e)[:160]))
            pg.goto(URL); pg.wait_for_timeout(200)
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(60)
            pg.evaluate(f"()=>{{location.hash='{lab}'}}"); pg.wait_for_timeout(450)
            pg.evaluate("()=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"challenge\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}"); pg.wait_for_timeout(450)
            sub = pg.evaluate("()=>{const b=[...document.querySelectorAll('[data-act=\"submit\"]')].find(x=>x.offsetParent); if(!b) return 'no submit'; if(b.disabled) return 'submit disabled'; b.click(); return 'ok';}")
            if sub == 'submit disabled':   # Submit waits for a pick (backprop, train-debug, gqa-mla): the lab's flow test covers its result
                print(f'{lab} {w} {lang}: skipped, Submit waits for a pick'); pg.close(); continue
            # a lab may animate before it shows the result (grpo plays 30 steps): wait up to 25 s for a verdict or a reveal button
            if sub == 'ok':
                try:
                    pg.wait_for_function("()=>{const r=document.querySelector('[id$=\"-ch-res\"],[id$=\"-chres\"]'); return r&&(r.querySelector('.okline,.badline,[data-act=\"reveal\"],.cmp')||r.textContent.trim().length>40);}", timeout=25000)
                except Exception:
                    print(f'{lab} {w} {lang}: no result within 25 s after Submit')
            pg.wait_for_timeout(300)
            rv = pg.evaluate("()=>{const b=[...document.querySelectorAll('[data-act=\"reveal\"]')].find(x=>x.offsetParent); if(!b) return ''; const t=b.textContent; b.click(); return t;}")
            pg.wait_for_timeout(450)
            r = pg.evaluate(INFO)
            tag = f'{lab} {w} {lang}'
            if r.get('none'):
                if sub != 'ok':   # a quiz with no Submit (architecture), or a result box that appears only after a pick: the lab's flow test covers it
                    print(f'{tag}: skipped, no result box without a pick or a Submit (submit: {sub})'); pg.close(); continue
                print(f'{tag}: no result box; submit: {sub}'); bad.append(f'{tag} no result box'); pg.close(); continue
            line = (f"{tag}: submit {sub}; reveal [{rv}]; rows {r['rows']} (ok {r['ok']}, me {r['me']}, tagged {r['tag']}, stray {r['stray']}); "
                    f"charts {r['charts']}; note [{r['note']}]; details {r['more']}; tables {r['tables']}; height {r['h']}; first [{r['first']}]")
            print(line)
            if r['stray'] and (r['note'] or any('看全部可行' in m or 'See all settings that pass' in m for m in r['more'])):
                bad.append(f'{tag} stray rows in a brief chart')
            if r['over']: bad.append(f'{tag} horizontal overflow')
            if any(m.startswith('OPEN') for m in r['more']): bad.append(f'{tag} a list open by default: {r["more"]}')
            if errs: bad.append(f'{tag} page errors {errs}')
            try:
                if r['h'] > 0: pg.locator('[id$="-ch-res"],[id$="-chres"]').first.screenshot(path=str(SH / f'{lab}-{w}-{lang}.png'), timeout=8000)
            except Exception as e:
                print(f'{tag}: screenshot failed {str(e)[:80]}')
            pg.close()
    b.close()
print('PROBLEMS:', bad or 'none')
WRAPPED.unlink(missing_ok=True)
if not os.environ.get('CHAL_PAGE'): (P / SRC).unlink(missing_ok=True)
sys.exit(1 if bad else 0)
