"""Round 5 home screenshots: hero, map, strip, tiles at 1440/900/390 in EN and ZH, light and dark."""
import pathlib, sys
from playwright.sync_api import sync_playwright
P=pathlib.Path(__file__).resolve().parent.parent
src=(P/'ai-systems-playground.html').read_text()
(P/'out/wrapped-r5.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>'+src+'</body></html>')
URL='file://'+str(P/'out/wrapped-r5.html'); S=P/'shots/r5'; errs=[]
with sync_playwright() as p:
    b=p.chromium.launch()
    for w,h in [(1440,900),(900,1000),(390,844)]:
        for lang in ['en','zh']:
            pg=b.new_page(viewport={'width':w,'height':h})
            pg.on('pageerror',lambda e: errs.append(str(e)))
            pg.goto(URL); pg.wait_for_timeout(400); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
            pg.screenshot(path=str(S/f'{w}_{lang}_0_fold.png'))
            # hero at 128K (out of memory)
            pg.evaluate("()=>{const i=document.querySelector('#hmCtx'); i.value=6; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(200)
            pg.locator('.meter').screenshot(path=str(S/f'{w}_{lang}_1_hero_oom.png'))
            pg.evaluate("()=>document.querySelector('#map').scrollIntoView()"); pg.wait_for_timeout(300)
            pg.screenshot(path=str(S/f'{w}_{lang}_2_map.png'))
            # choose a path from the chips: the map must stay in view
            pg.click('.pathchip[data-path="serving"]'); pg.wait_for_timeout(400)
            pg.screenshot(path=str(S/f'{w}_{lang}_3_path.png'))
            pg.click('.pathchip[data-path=""]'); pg.wait_for_timeout(200)
            # scroll into the tiles: the strip sticks
            pg.evaluate("()=>{const t=document.querySelector('#tiles'); window.scrollTo(0,t.getBoundingClientRect().top+scrollY+300)}"); pg.wait_for_timeout(300)
            pg.screenshot(path=str(S/f'{w}_{lang}_4_tiles_strip.png'))
            pg.click('#domStrip .ds-btn[data-world="agents"]'); pg.wait_for_timeout(700)
            pg.screenshot(path=str(S/f'{w}_{lang}_5_strip_agents.png'))
            print(w,lang,'overflow',pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth"),'tipW',pg.evaluate("()=>document.querySelector('#tip').getBoundingClientRect().width"))
            pg.close()
    # dark mode at 1440 en
    pg=b.new_page(viewport={'width':1440,'height':900}, color_scheme='dark'); pg.goto(URL); pg.wait_for_timeout(400)
    pg.evaluate("()=>document.querySelector('#map').scrollIntoView()"); pg.wait_for_timeout(300); pg.screenshot(path=str(S/'1440_dark_map.png'))
    pg.evaluate("()=>{const t=document.querySelector('#tiles'); window.scrollTo(0,t.getBoundingClientRect().top+scrollY-120)}"); pg.wait_for_timeout(300); pg.screenshot(path=str(S/'1440_dark_tiles.png'))
    pg.close(); b.close()
print('errors', errs or 'none')
