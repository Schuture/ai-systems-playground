"""Integrated page: every lab route and every home tile, both languages, 1440 and 390, no errors, no overflow."""
import pathlib, json, sys
from playwright.sync_api import sync_playwright
P=pathlib.Path(__file__).resolve().parent.parent
src=(P/'ai-systems-playground.html').read_text()
(P/'out').mkdir(exist_ok=True)
(P/'out/wrapped-integrated.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>'+src+'</body></html>')
URL='file://'+str(P/'out/wrapped-integrated.html'); errs=[]; out=[]
shots=P/'shots/integrated'; shots.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b=p.chromium.launch()
    for w,h in [(1440,900),(390,844)]:
        for lang in ['en','zh']:
            pg=b.new_page(viewport={'width':w,'height':h})
            pg.on('pageerror',lambda e,w=w,lang=lang: errs.append(f'{w}/{lang}: {e}'))
            pg.on('console',lambda m,w=w,lang=lang: errs.append(f'{w}/{lang} console: {m.text}') if m.type=='error' and 'ERR_TUNNEL' not in m.text and 'Failed to load resource' not in m.text else None)
            pg.goto(URL); pg.wait_for_timeout(400); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
            # every domain: tiles and teasers
            for wid in ['inference','training','post','models','apps','agents','eval']:
                pg.evaluate(f"()=>document.querySelector('#mapSvg .world[data-world=\"{wid}\"]').dispatchEvent(new MouseEvent('click',{{bubbles:true}}))"); pg.wait_for_timeout(250)
                info=pg.evaluate("()=>({n:document.querySelectorAll('#tiles .tile').length, empty:[...document.querySelectorAll('#tiles .tile-tz')].filter(e=>e.children.length===0||e.querySelector('.tz-read.muted')&&e.innerText.includes('还在做')||e.innerText.includes('coming')).map(e=>e.dataset.teaser), live:[...document.querySelectorAll('#tiles .tile.is-live')].map(t=>t.dataset.lab), ov:document.documentElement.scrollWidth-innerWidth})")
                out.append((w,lang,wid,info))
                if lang=='zh' and w==1440: pg.locator('#tilesHead').scroll_into_view_if_needed(); pg.screenshot(path=str(shots/f'home_{wid}_{w}_{lang}.png'), full_page=False)
            for lab in ['kv-cache','paged-attention','continuous-batching','dpo']:
                pg.evaluate(f"()=>{{location.hash='{lab}'}}"); pg.wait_for_timeout(600)
                ttl=pg.evaluate("()=>{const v=document.querySelector('#view-dyn:not([hidden]) .labtitle')||document.querySelector('#view-lab:not([hidden]) .labtitle'); return v?v.innerText:null}")
                ov=pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
                out.append((w,lang,lab,{'title':ttl,'ov':ov}))
                pg.screenshot(path=str(shots/f'{lab}_{w}_{lang}.png'), full_page=False)
            pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(300)
            pg.close()
    b.close()
bad=[o for o in out if (isinstance(o[3],dict) and (o[3].get('ov',0)>0 or o[3].get('empty') or o[3].get('title',1) is None))]
for o in out: print(o)
print('\nPROBLEMS:', bad if bad else 'none')
print('errors:', errs if errs else 'none')
