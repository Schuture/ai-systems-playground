"""Dump every piece of text a learner meets in the PagedAttention lab and its home tile, in reading order.
python3 tests/pa_copy_deck.py [tag]  ->  out/pa-copy/<tag>-zh.md and <tag>-en.md
(without a tag: now-zh.md and now-en.md, and also zh.md and en.md, the files tests/numdense.py reads)"""
import pathlib, sys, subprocess, json
from playwright.sync_api import sync_playwright
P=pathlib.Path(__file__).resolve().parent.parent
tag=sys.argv[1] if len(sys.argv)>1 else 'now'
subprocess.run(['python3','build.py','out/pa-copy/page.html','--only','paged-attention'],cwd=P,check=True,capture_output=True)
src=(P/'out/pa-copy/page.html').read_text()
(P/'out/pa-copy/wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>'+src+'</body></html>')
URL='file://'+str(P/'out/pa-copy/wrapped.html')
TIPS="""(root)=>{const out=[];const seen=new Set();root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx');if(t&&!seen.has(t)&&e.offsetParent!==null){seen.add(t);out.push(((e.innerText||e.getAttribute('aria-label')||'').trim().slice(0,40))+' ⟶ '+t);}});return out;}"""
def txt(pg,sel):
    el=pg.query_selector(sel); return el.inner_text().strip() if el else ''
with sync_playwright() as p:
    b=p.chromium.launch()
    for lang in ['zh','en']:
        pg=b.new_page(viewport={'width':1440,'height':1000})
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        L=[]
        # home tile
        pg.evaluate("()=>document.querySelector('#tile-paged-attention').scrollIntoView()"); pg.wait_for_timeout(200)
        L.append('# Home tile'); L.append(txt(pg,'#tile-paged-attention'))
        L.append('## Assumptions & sources'); L.append(pg.evaluate("()=>[...document.querySelectorAll('#tile-paged-attention .tz-note')].map(n=>n.innerText.trim()).join('\\n\\n')"))
        for i,btn in enumerate(pg.query_selector_all('#tile-paged-attention .seg-ctl button')):
            if i==0: continue
            btn.click(); pg.wait_for_timeout(150); L.append(f'## Tile, option {i+1}'); L.append(txt(pg,'#tile-paged-attention .tz-read'))
        # lab
        pg.evaluate("()=>{location.hash='paged-attention'}"); pg.wait_for_timeout(700)
        L.append('# Lab header'); L.append(txt(pg,'.dyn-root .labhead'))
        for k in range(7):
            L.append(f'# Guided part {k+1}')
            L.append('## Card'); L.append(txt(pg,'#pa-rail .step.current'))
            L.append('## Stage'); L.append(txt(pg,'#pa-stage'))
            L.append('## Tooltips in view'); L.extend(pg.evaluate(TIPS, pg.query_selector('.dyn-root')))
            # hover one pool cell
            cell=pg.query_selector('#pa-pool [data-q]') or pg.query_selector('#pa-pool i') or pg.query_selector('#pa-pool span')
            if cell:
                try:
                    cell.hover(); pg.wait_for_timeout(150); L.append('## Pool cell tooltip'); L.append(txt(pg,'#tip'))
                except Exception as e: pass
            # advance
            # pick the last option (a wrong prediction), so part 3 shows the verdict with the right answer
            prs=pg.query_selector_all('#pa-rail .choices .choice'); pr=prs[-1] if prs else None
            if pr and not pg.query_selector('#pa-rail .choices .choice[aria-pressed="true"]'): pr.click(); pg.wait_for_timeout(200)
            nb=pg.query_selector('#pa-rail [data-nav="next"]')
            if nb and nb.is_disabled():
                sm=pg.query_selector('#pa-rail [data-nav="showme"]')
                if sm and sm.is_visible(): sm.click()
                for _ in range(120):
                    pg.wait_for_timeout(250); nb=pg.query_selector('#pa-rail [data-nav="next"]')
                    if nb and not nb.is_disabled(): break
                L.append('## Card after Show me / run'); L.append(txt(pg,'#pa-rail .step.current'))
            if nb: nb.click(); pg.wait_for_timeout(500)
        # free explore
        pg.click('.dyn-root .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(500)
        L.append('# Free explore'); L.append(txt(pg,'#pa-rail'))
        L.append('## Controls'); L.append(txt(pg,'#pa-controls'))
        # since 2026-10-11 the notes close the page (no Fidelity or Primer button)
        L.append('## Notes at the page end'); L.append(pg.evaluate("()=>{const p=document.querySelector('.dyn-root [data-endnotes]');return p?p.innerText:''}"))
        # scenarios: a click that changes the numbers, then one that changes none (with paging)
        pg.click('#pa-rail .seg-ctl[data-key="strat"] button[data-v="paged"]'); pg.wait_for_timeout(200)
        pg.click('#pa-rail [data-preset="0"]'); pg.wait_for_timeout(250)
        L.append('## Scenario click that changes the numbers'); L.append(txt(pg,'#pa-sbnote'))
        pg.click('#pa-rail [data-preset="3"]'); pg.wait_for_timeout(250)
        L.append('## Scenario click that changes no number'); L.append(txt(pg,'#pa-sbnote'))
        # challenge
        pg.click('.dyn-root .modes button[data-mode="challenge"]'); pg.wait_for_timeout(500)
        L.append('# Challenge'); L.append(txt(pg,'#pa-rail'))
        L.append('# Connect'); L.append(txt(pg,'.dyn-root .connect'))
        (P/f'out/pa-copy/{tag}-{lang}.md').write_text('\n\n'.join(x for x in L if x is not None))
        if len(sys.argv)<=1: (P/f'out/pa-copy/{lang}.md').write_text('\n\n'.join(x for x in L if x is not None))   # the name tests/numdense.py reads
        pg.close()
    b.close()
print('wrote', [str(P/f'out/pa-copy/{tag}-{l}.md') for l in ['zh','en']])
