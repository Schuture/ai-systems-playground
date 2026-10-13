"""Dump every piece of text a learner meets in the continuous batching lab and its home tile, in reading order.
Modeled on tests/pa_copy_deck.py. python3 tests/continuous-batching_copy_deck.py [tag]
  -> out/cb-copy/<tag>-zh.md and out/cb-copy/<tag>-en.md
  (without a tag: now-zh.md and now-en.md, and also zh.md and en.md, the files tests/numdense.py reads)
Order: home tile (both options, its notes), lab header and mode tooltips, each guided part (card before and after its
action, stage with legend, badge, readouts and metric cards, tooltips in view), Show the math, the mini result bar and
the run-finished line on a phone, the scenarios card (every preset), the settings panel with every note, the fidelity
and primer popovers, the challenge before and after submitting, and the connect section with the drill answer."""
import pathlib, sys, subprocess
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
tag = sys.argv[1] if len(sys.argv) > 1 else 'now'
OUT = P / 'out/cb-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/continuous-batching.html', '--only', 'continuous-batching'], cwd=P, check=True, capture_output=True)
(P / 'out/wrapped-continuous-batching.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>'
                                                      + (P / 'out/continuous-batching.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out/wrapped-continuous-batching.html')
# every visible tooltip once; of the 48 request bars only the first
TIPS = """(root)=>{const out=[], seen=new Set(); let bars=0;
  root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); if(!t||seen.has(t)||e.offsetParent===null) return;
    if(e.classList.contains('cb-req')){if(bars++) return;}
    seen.add(t); out.push(((e.innerText||e.getAttribute('aria-label')||'').replace(/\\s+/g,' ').trim().slice(0,60))+' ⟶ '+t);});
  return out;}"""

def txt(pg, sel):
    el = pg.query_selector(sel); return el.inner_text().strip() if el else ''
def ev(pg, js, arg=None): return pg.evaluate(js, arg)
def click(pg, sel, wait=250):
    el = pg.query_selector(sel)
    if el: el.scroll_into_view_if_needed(); el.click(); pg.wait_for_timeout(wait)
    return el is not None
def setr(pg, id_, v): ev(pg, "([i,v])=>{const e=document.getElementById(i); e.value=v; e.dispatchEvent(new Event('input',{bubbles:true}))}", [id_, v]); pg.wait_for_timeout(200)
def wait_next(pg, n=160):
    for _ in range(n):
        nb = pg.query_selector('#cb-rail [data-nav="next"]')
        if not nb or not nb.is_disabled(): return
        pg.wait_for_timeout(250)

with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ['zh', 'en']:
        pg = b.new_page(viewport={'width': 1440, 'height': 1000})
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        L = []
        # ---------- home tile ----------
        ev(pg, "()=>{location.hash='map'}"); pg.wait_for_timeout(400)
        ev(pg, "()=>{const t=document.querySelector('#tile-continuous-batching'); if(t) t.scrollIntoView();}"); pg.wait_for_timeout(200)
        L.append('# Home tile'); L.append(txt(pg, '#tile-continuous-batching'))
        L.append('## Assumptions & sources'); L.append(ev(pg, "()=>[...document.querySelectorAll('#tile-continuous-batching .tz-note')].map(n=>n.innerText.trim()).join('\\n\\n')"))
        L.append('## Tooltips in view'); L.extend(ev(pg, TIPS, pg.query_selector('#tile-continuous-batching')) if pg.query_selector('#tile-continuous-batching') else [])
        for i, btn in enumerate(pg.query_selector_all('#tile-continuous-batching .seg-ctl button')):
            if i == 0: continue
            btn.click(); pg.wait_for_timeout(150); L.append(f'## Tile, option {i + 1}'); L.append(txt(pg, '#tile-continuous-batching .tz-read'))
        # ---------- lab ----------
        ev(pg, "()=>{location.hash='continuous-batching'}"); pg.wait_for_timeout(700)
        L.append('# Lab header'); L.append(txt(pg, '#view-dyn .labhead'))
        for k in range(7):
            L.append(f'# Guided part {k + 1}')
            L.append('## Card'); L.append(txt(pg, '#cb-rail .step.current'))
            L.append('## Stage'); L.append(txt(pg, '#cb-stage'))
            L.append('## Tooltips in view'); L.extend(ev(pg, TIPS, pg.query_selector('#view-dyn')))
            if k == 2:
                # part 3: pause the static run part-way, to read what the stage says while a run is shown only in part
                click(pg, '#cb-rail [data-act="play"]', 2600); click(pg, '#cb-rail [data-act="play"]', 300)
                L.append('## Stage paused part-way through the run'); L.append(txt(pg, '#cb-stage'))
                L.append('## Tooltips in view part-way through the run'); L.extend(ev(pg, TIPS, pg.query_selector('#view-dyn')))
            # the part's action
            acted = False
            if pg.query_selector('#cb-rail .choices .choice') and not pg.query_selector('#cb-rail .choices .choice[aria-pressed="true"]'):
                pg.click('#cb-rail .choices .choice[data-pred="mid"]'); pg.wait_for_timeout(250); acted = True
            nb = pg.query_selector('#cb-rail [data-nav="next"]')
            if nb and nb.is_disabled() and not acted:
                sm = pg.query_selector('#cb-rail [data-nav="showme"]')
                if sm and sm.is_visible(): sm.click(); acted = True
                wait_next(pg)
            if acted:
                pg.wait_for_timeout(300)
                L.append('## Card after the action (Show me / answer)'); L.append(txt(pg, '#cb-rail .step.current'))
                L.append('## Stage after the action'); L.append(txt(pg, '#cb-stage'))
                L.append('## Tooltips in view after the action'); L.extend(ev(pg, TIPS, pg.query_selector('#view-dyn')))
            if k == 4:
                ev(pg, "()=>document.getElementById('cb-math').open=true"); pg.wait_for_timeout(150)
                L.append('# Show the math (open)'); L.append(txt(pg, '#cb-math'))
                ev(pg, "()=>document.getElementById('cb-math').open=false"); pg.wait_for_timeout(100)
            nb = pg.query_selector('#cb-rail [data-nav="next"]')
            if nb and not nb.is_disabled(): nb.click(); pg.wait_for_timeout(500)
        # ---------- phone: mini result bar and the run-finished line ----------
        ph = b.new_page(viewport={'width': 390, 'height': 844})
        ph.goto(URL); ph.wait_for_timeout(300); ph.click(f'.langsw button[data-lang="{lang}"]'); ph.wait_for_timeout(150)
        ev(ph, "()=>{location.hash='continuous-batching'}"); ph.wait_for_timeout(600)
        ev(ph, "()=>document.querySelector('#cb-rail [data-go=\"2\"]').click()"); ph.wait_for_timeout(300)
        L.append('# Phone (390 px)')
        click(ph, '#cb-rail [data-act="play"]', 3000)
        L.append('## Mini result bar during the run'); ev(ph, "()=>document.getElementById('cb-controls').scrollIntoView({block:'center'})"); ph.wait_for_timeout(400); L.append(txt(ph, '#dock'))
        ev(ph, "()=>document.getElementById('cb-graph').scrollIntoView({block:'start'})"); ph.wait_for_timeout(300)
        for _ in range(200):
            if ev(ph, "()=>{const o=document.querySelector('#cb-rail [data-obs]'); return o&&!o.hidden}"): break
            ph.wait_for_timeout(250)
        ph.wait_for_timeout(300)
        L.append('## Line under the chart when the run ends'); L.append(txt(ph, '#cb-done') if ev(ph, "()=>!document.getElementById('cb-done').hidden") else '(not shown)')
        ev(ph, "()=>document.getElementById('cb-controls').scrollIntoView({block:'center'})"); ph.wait_for_timeout(400)
        L.append('## Mini result bar after the run'); L.append(txt(ph, '#dock'))
        ph.close()
        # ---------- scenarios ----------
        ev(pg, "()=>window.scrollTo(0,0)"); pg.click('#view-dyn .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(500)
        L.append('# Scenarios'); L.append(txt(pg, '#cb-rail'))
        for i in range(len(pg.query_selector_all('#cb-rail [data-preset]'))):
            click(pg, f'#cb-rail [data-preset="{i}"]', 300)
            L.append(f'## Scenario {i + 1} picked'); L.append(txt(pg, '#cb-sb-res'))
        L.append('## Stage in this mode'); L.append(txt(pg, '#cb-stage'))
        L.append('## Tooltips in view'); L.extend(ev(pg, TIPS, pg.query_selector('#view-dyn')))
        # ---------- settings panel (every section open) ----------
        ev(pg, "()=>document.querySelectorAll('#cb-controls details').forEach(d=>d.open=true)"); pg.wait_for_timeout(150)
        L.append('# Settings panel'); L.append(txt(pg, '#cb-controls'))
        # ---------- popovers ----------
        # since 2026-10-11 the notes close the page (no Fidelity or Primer button)
        L.append('# Notes at the page end'); L.append(ev(pg, "()=>{const p=document.querySelector('#view-dyn [data-endnotes]'); return p?p.innerText:''}"))
        # ---------- challenge ----------
        ev(pg, "()=>window.scrollTo(0,0)"); pg.click('#view-dyn .modes button[data-mode="challenge"]'); pg.wait_for_timeout(500)
        L.append('# Challenge'); L.append(txt(pg, '#cb-rail'))
        L.append('## Stage in this mode'); L.append(txt(pg, '#cb-stage'))
        L.append('## Settings panel in this mode'); L.append(txt(pg, '#cb-controls'))
        click(pg, '#cb-rail [data-act="submit"]', 300)
        L.append('## After submitting the initial setup'); L.append(txt(pg, '#cb-rail'))
        click(pg, '#cb-rail .seg-ctl[data-key="pol"] button[data-v="cont"]', 300)
        L.append('## Continuous batching, 8 slots'); L.append(txt(pg, '#cb-ch-res'))
        setr(pg, 'cb-ic-B-in', 12)
        L.append('## Continuous batching, 12 slots'); L.append(txt(pg, '#cb-ch-res'))
        click(pg, '#cb-rail [data-act="reveal"]', 300)
        L.append('## After “Show the answer”'); L.append(txt(pg, '#cb-ch-res'))
        setr(pg, 'cb-ic-B-in', 11)
        L.append('## Continuous batching, 11 slots'); L.append(txt(pg, '#cb-ch-res'))
        setr(pg, 'cb-ic-B-in', 10)
        L.append('## Continuous batching, 10 slots'); L.append(txt(pg, '#cb-ch-res'))
        L.append('## Tooltips in view'); L.extend(ev(pg, TIPS, pg.query_selector('#view-dyn')))
        # ---------- connect ----------
        click(pg, '#view-dyn [data-act="drill-reveal"]', 250)
        L.append('# Connect'); L.append(txt(pg, '#view-dyn .connect'))
        (OUT / f'{tag}-{lang}.md').write_text('\n\n'.join(x for x in L if x is not None))
        if len(sys.argv) <= 1: (OUT / f'{lang}.md').write_text('\n\n'.join(x for x in L if x is not None))   # the name tests/numdense.py reads
        pg.close()
    b.close()
print('wrote', [str(OUT / f'{tag}-{l}.md') for l in ['zh', 'en']])
