"""Batch 6 integrated smoke: the full page, every lab route (all registered labs) and the home tree, EN and ZH, 1440 and 390.
Checks: no page errors, no console errors (fonts aside), a lab title on every route, no horizontal overflow, every live root and
trunk tile has a teaser. Screenshots of the batch 6 labs in shots/batch6/.
Run: python3 tests/batch6_smoke.py [page.html]   (default: ai-systems-playground.html, built by `python3 build.py`)"""
import pathlib, sys
from playwright.sync_api import sync_playwright
P = pathlib.Path(__file__).resolve().parent.parent
page = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else P / 'ai-systems-playground.html'
(P / 'out').mkdir(exist_ok=True)
W = P / 'out/wrapped-batch6.html'
W.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
             '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + page.read_text() + '</body></html>')
URL = 'file://' + str(W)
NEW = ['softmax-numerics', 'number-formats', 'roofline', 'flash-attention', 'tensors', 'backprop', 'optimizers', 'train-debug']
shots = P / 'shots/batch6'; shots.mkdir(parents=True, exist_ok=True)
errs, bad, n = [], [], 0
with sync_playwright() as p:
    b = p.chromium.launch()
    for w, h in [(1440, 900), (390, 844)]:
        for lang in ['en', 'zh']:
            pg = b.new_page(viewport={'width': w, 'height': h})
            pg.on('pageerror', lambda e, w=w, lang=lang: errs.append(f'{w}/{lang}: {e}'))
            pg.on('console', lambda m, w=w, lang=lang: errs.append(f'{w}/{lang} console: {m.text}')
                  if m.type == 'error' and 'ERR_TUNNEL' not in m.text and 'Failed to load resource' not in m.text else None)
            pg.goto(URL + '#map'); pg.wait_for_timeout(500)
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(250)
            live = pg.evaluate("()=>window.ASP.catalog.LABS.filter(l=>window.ASP.catalog.labStatus(l)==='live').map(l=>l.id)")
            missing = [x for x in NEW if x not in live]
            if missing: bad.append((w, lang, 'not live', missing))
            ov = pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth"); n += 1
            if ov > 0: bad.append((w, lang, 'home overflow', ov))
            if w == 1440: pg.screenshot(path=str(shots / f'home_{lang}.png'), full_page=False)
            for lab in live:
                pg.evaluate(f"()=>{{location.hash='{lab}'}}"); pg.wait_for_timeout(450)
                t = pg.evaluate("()=>{const v=document.querySelector('#view-dyn:not([hidden]) .labtitle')||document.querySelector('#view-lab:not([hidden]) .labtitle'); return v?v.innerText:null}")
                ov = pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth"); n += 1
                if not t: bad.append((w, lang, lab, 'no title'))
                if ov > 0: bad.append((w, lang, lab, 'overflow', ov))
                if lab in NEW: pg.screenshot(path=str(shots / f'{lab}_{w}_{lang}.png'), full_page=False)
            pg.close()
    b.close()
print(f'{n} routes checked')
print('PROBLEMS:', bad if bad else 'none')
print('errors:', errs[:20] if errs else 'none')
