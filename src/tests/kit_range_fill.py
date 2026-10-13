"""Kit: a slider's fill (--fill) follows its thumb on every input event, even when a lab's own handler forgets setFill (the owner's
review of batch 8, 2026-10-07: the scaling-law sliders' fill stayed behind the thumb). Checked on a fixture slider whose handler
does nothing, dragged with the mouse and moved with the keyboard, at 1440 and 390.
Run: python3 tests/kit_range_fill.py"""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/kit-range.html', '--only', 'sampling'], check=True, cwd=P, stdout=subprocess.DEVNULL)
WRAP = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>')
(P / 'out' / 'kit-range-wrapped.html').write_text(WRAP + (P / 'out' / 'kit-range.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'kit-range-wrapped.html')
ADD = r"""()=>{const d=document.createElement('div'); d.className='bench'; d.style.cssText='padding:40px;max-width:400px';
  d.innerHTML='<input type="range" id="kr" min="0" max="100" step="1" value="50" style="width:300px">';
  document.body.prepend(d); const r=d.querySelector('#kr'); window.ASP.ui.initRanges(d); r.addEventListener('input',()=>{}); window.scrollTo(0,0);
  const b=r.getBoundingClientRect(); return {x:b.x,y:b.y+b.height/2,w:b.width};}"""
FILL = "()=>{const r=document.getElementById('kr'); return [r.value, r.style.getPropertyValue('--fill')]}"
n, fails = 0, []


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(f'{name} {info}'); print('FAIL', name, info)


with sync_playwright() as p:
    b = p.chromium.launch()
    for w, h in ((1440, 900), (390, 844)):
        pg = b.new_page(viewport={'width': w, 'height': h}); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(URL + '#sampling'); pg.wait_for_timeout(300)
        g = pg.evaluate(ADD)
        v0, f0 = pg.evaluate(FILL)
        check(f'{w}: the fill starts at the value', f0 == '50%', (v0, f0))
        pg.mouse.move(g['x'] + g['w'] * 0.5, g['y']); pg.mouse.down(); pg.mouse.move(g['x'] + g['w'] * 0.85, g['y'], steps=6); pg.mouse.up(); pg.wait_for_timeout(50)
        v1, f1 = pg.evaluate(FILL)
        check(f'{w}: after a mouse drag the fill equals the value', int(v1) > 70 and f1 == f'{int(v1)}%', (v1, f1))
        pg.focus('#kr'); pg.keyboard.press('ArrowLeft'); pg.keyboard.press('ArrowLeft'); pg.wait_for_timeout(50)
        v2, f2 = pg.evaluate(FILL)
        check(f'{w}: after two arrow keys the fill equals the value', int(v2) == int(v1) - 2 and f2 == f'{int(v2)}%', (v2, f2))
        check(f'{w}: no page errors', not errs, errs)
        pg.close()
    b.close()
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails or 'none')
sys.exit(1 if fails else 0)
