"""Dump every piece of text a learner meets in the data-pipeline lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/data-pipeline_copy_deck.py  ->  out/data-pipeline/copy-en.md and out/data-pipeline/copy-zh.md

Order: home tile (the sentence at every position, the notes), lab header and its popovers, each guided part (card before and after its
action, the stage, the tooltips in view not printed before), Show the math, the scenarios, the settings panel, the challenge (every
feedback tier and the chart), the card "Can you use it on a new case?" (each tab), Connect with the drill answer, and the hover card
of one page of each kind (the canvas squares).
A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys, re
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'data-pipeline'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/data-pipeline.html', '--only', 'data-pipeline'], cwd=P, check=True, capture_output=True)
(OUT / 'wrapped-deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'data-pipeline.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-deck.html')
clean = lambda s: ' '.join(str(s).replace('‑', '-').replace(' ', ' ').split())
seen = set()


def txt(pg, sel):
    return pg.evaluate("s=>{const e=document.querySelector(s); return e&&e.getClientRects().length?e.innerText:''}", sel)


def lines(t):
    return [clean(x) for x in t.split('\n') if clean(x)]


def tips(pg, root):
    out = []
    for t in pg.evaluate("r=>[...document.querySelectorAll(r+' [data-tipx]')].filter(e=>e.getClientRects().length).map(e=>[(e.innerText||'').trim().slice(0,40),e.getAttribute('data-tipx')])", root):
        k = clean(t[1])
        if k and k not in seen:
            seen.add(k); out.append(f'- hover “{clean(t[0])}” → {k}')
    return out


def stage(pg):
    out = [clean(txt(pg, '#dl-st-title')), clean(txt(pg, '#dl-st-sub')), 'Status: ' + clean(txt(pg, '#dl-status'))]
    for sel in ('#dl-phone', '#dl-f-cap', '#dl-f-leg', '#dl-player', '#dl-xbox', '#dl-c-cap', '#dl-c-yt', '#dl-c-xt', '#dl-c-leg', '#dl-m-cap', '#dl-mix'):
        t = txt(pg, sel)
        if t: out.append(clean(t))
    cv = pg.evaluate("()=>{const c=document.querySelector('#dl-cv'); return c&&c.getClientRects().length?c.getAttribute('aria-label'):''}")
    if cv: out.append('Canvas: ' + clean(cv))
    sv = pg.evaluate("()=>{const c=document.querySelector('#dl-curve svg'); return c&&c.getClientRects().length?[c.getAttribute('aria-label'),...[...c.querySelectorAll('text')].map(t=>t.textContent)].join(' | '):''}")
    if sv: out.append('Curve: ' + clean(sv))
    out.append('Metrics: ' + ' || '.join(clean(m) for m in pg.evaluate("()=>[...document.querySelectorAll('#dl-metrics .metric')].map(m=>m.innerText)")))
    return [x for x in out if x]


def click(pg, sel, wait=300):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(wait)


def deck(b, lang):
    D = []
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
    seen.clear()
    D.append('# Home tile')
    for i in range(2, 13):
        D.append(f'- {clean(pg.evaluate("i=>window.__tzDl.read(i)", i)).replace("<b>", "").replace("</b>", "")}')
    pg.evaluate("()=>{const b=[...document.querySelectorAll('button[data-world=\"training\"]')].find(e=>e.offsetParent!==null); if(b) b.click();}"); pg.wait_for_timeout(300)
    tile = pg.evaluate("()=>{const t=document.querySelector('.tz-dl'); return t?[...t.querySelectorAll('.tz-lab span,.tz-leg,.tz-note')].map(e=>e.textContent).concat([...t.querySelectorAll('.tz-dl-v')].map(e=>'chart label: '+e.textContent),[t.querySelector('.tz-dl-vis').getAttribute('data-tipx')]):['(tile not found)']}")
    D += [f'- {clean(x)}' for x in tile if clean(x)]
    pg.evaluate("()=>{location.hash='data-pipeline'}"); pg.wait_for_function("()=>!!window.__dl&&!!document.querySelector('#view-dyn #dl-app #dl-rail')", timeout=30000); pg.wait_for_timeout(500)   # wait for the mounted lab, not only a fixed time
    D.append('\n# Lab header')
    D += lines(txt(pg, '#dl-app .labhead'))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#dl-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        D.append(f'\n## Popover {pid}')
        D += lines(pg.evaluate(f"()=>document.getElementById('{pid}').innerText||document.getElementById('{pid}').textContent"))
    D += tips(pg, '#dl-app .labhead')
    for i in range(6):
        sid = pg.evaluate("()=>window.__dl.state().step")
        D.append(f'\n# Guided part {sid + 1}: card')
        D += lines(txt(pg, '#dl-rail .step.current') or txt(pg, '#dl-rail'))
        D.append('## Stage'); D += stage(pg)
        D.append('## Settings'); D += lines(txt(pg, '#dl-controls'))
        D += tips(pg, '#dl-app')
        # the part's action
        if i == 0:
            click(pg, '#dl-rail [data-pred="tenth"]'); pg.wait_for_function('()=>window.__dl.state().g>=4', timeout=20000); pg.wait_for_timeout(300)
        elif i in (1, 2, 4): click(pg, '#dl-rail [data-nav="showme"]', 600); pg.evaluate("()=>{const i=document.querySelector('#dl-player input[data-key=scrub]'); if(i&&i.getClientRects().length){i.value=400; i.dispatchEvent(new Event('input',{bubbles:true}))}}"); pg.wait_for_timeout(200)
        elif i == 3:
            D.append('\n## Exact + MinHash (before the pick)'); D += lines(txt(pg, '#dl-rail [data-obs]'))
            D.append('## Hover of a signature value (A 1, B 4, B 5, C 5)'); D += [clean(re.sub(r'<[^>]+>', ' ', pg.evaluate(f"()=>window.__dl.xtip('{k}')").replace('</tr>', ';'))) for k in ('xA0', 'xB3', 'xB4', 'xC4')]
            D.append('## Hover of the similarity (B, C; the formula line is MathML, shown here as its text)'); D += [clean(re.sub(r'<[^>]+>', ' ', pg.evaluate(f"()=>window.__dl.xtip('{k}')"))) for k in ('xjB', 'xjC')]
            click(pg, '#dl-rail [data-lever="exact"]')
        if i < 5:
            D.append(f'\n# Guided part {sid + 1}: after the action')
            D += lines(txt(pg, '#dl-rail .step.current') or txt(pg, '#dl-rail'))
            D.append('## Stage'); D += stage(pg)
            pg.evaluate("()=>{document.getElementById('dl-math').open=true}"); pg.wait_for_timeout(100)
            D.append('## Show the math'); D += lines(txt(pg, '#dl-mathbody'))
            D += tips(pg, '#dl-app')
            click(pg, '#dl-rail [data-nav="next"]', 500)
    D.append('\n# Scenarios')
    click(pg, '#dl-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 500)
    D += lines(txt(pg, '#dl-rail'))
    for i in range(4):
        click(pg, f'#dl-rail [data-preset="{i}"]', 400)
        D.append(f'\n## Scenario {i + 1}'); D += lines(txt(pg, '#dl-insight')); D.append('### Stage'); D += stage(pg)
        if 'once the run ends' in txt(pg, '#dl-insight') or '动画播完后' in txt(pg, '#dl-insight'):   # a new picture: its result once the run has ended
            pg.evaluate("()=>{const i=document.querySelector('#dl-player input[data-key=scrub]'); i.value=400; i.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(300)
            D.append(f'### Scenario {i + 1} after the run'); D += lines(txt(pg, '#dl-insight')); D.append('### Stage'); D += stage(pg)
        D += tips(pg, '#dl-app')
    D.append('\n# Settings panel (Scenarios)'); D += lines(txt(pg, '#dl-controls'))
    D.append('\n# Challenge')
    click(pg, '#dl-app .modes button[data-mode="challenge"]', 500)
    D += lines(txt(pg, '#dl-rail .railcard'))
    D.append('## Stage'); D += stage(pg)
    click(pg, '#dl-rail [data-act="submit"]')
    D.append('\n## Tier 3: 14 × 8 (misses both goals)'); D += lines(txt(pg, '#dl-ch-res'))
    click(pg, '#dl-rail [data-act="reveal"]')
    D.append('\n## Show the cheapest setting'); D += lines(txt(pg, '#dl-ch-res'))
    D.append('### Stage'); D += stage(pg)
    pg.evaluate("()=>{const d=document.querySelector('#dl-ch-res details.dl-all'); if(d) d.open=true}"); pg.wait_for_timeout(150)
    D.append('### See all settings that pass (opened)'); D += lines(txt(pg, '#dl-ch-res details.dl-all'))
    D += tips(pg, '#dl-app')
    for b_, r_, name in ((20, 11, 'Tier 3: 20 × 11 (misses 0.8)'), (40, 8, 'Tier 3: 40 × 8 (misses 0.5)'), (40, 11, 'Tier 2: 40 × 11'), (50, 12, 'Tier 2: 50 × 12'), (34, 11, 'Tier 1: 34 × 11')):
        click(pg, f'#dl-rail .seg-ctl[data-key="chb"] button[data-v="{b_}"]'); click(pg, f'#dl-rail .seg-ctl[data-key="chr"] button[data-v="{r_}"]')
        D.append(f'\n## {name}'); D += [x for x in lines(txt(pg, '#dl-ch-res')) if not x.startswith('See all') and not x.startswith('看全部')]
    D += tips(pg, '#dl-app')
    D.append('\n# Can you use it on a new case?')
    tabs = pg.evaluate("()=>document.querySelectorAll('#dl-rail [data-exits] .ex-tabs button').length")
    D += lines(txt(pg, '#dl-rail [data-exits]'))
    for t in range(tabs):
        pg.evaluate(f"()=>document.querySelectorAll('#dl-rail [data-exits] .ex-tabs button')[{t}].click()"); pg.wait_for_timeout(200)
        D.append(f'## Tab {t + 1}'); D += lines(txt(pg, '#dl-rail [data-exits]'))
        ans = pg.evaluate("()=>window.__dl.exits()")
        if t == 1:   # the new cases: pick the right answer of each, so its explanation is in the deck too
            for q in range(len(ans['transferA'])):
                pg.evaluate(f"()=>{{const b=[...document.querySelectorAll('#dl-rail [data-exits] .ex-opts button')][{ans['transferA'][q]}]; b&&b.click()}}"); pg.wait_for_timeout(200)
                D.append(f'### New case {q + 1}, the right answer'); D += lines(txt(pg, '#dl-rail [data-exits]'))
                nx = pg.evaluate("()=>{const b=[...document.querySelectorAll('#dl-rail [data-exits] button')].find(b=>/Next question|下一题/.test(b.textContent)&&b.getClientRects().length); if(b){b.click(); return true} return false}")
                if not nx: break
                pg.wait_for_timeout(200)
            continue
        for q in range(3):
            nx = pg.evaluate("()=>{const b=[...document.querySelectorAll('#dl-rail [data-exits] button')].find(b=>/Next question|下一题/.test(b.textContent)&&b.getClientRects().length); if(b){b.click(); return true} return false}")
            if not nx: break
            pg.wait_for_timeout(200); D += lines(txt(pg, '#dl-rail [data-exits]'))
    D.append('\n# Hover cards of pages (one per kind, part 1’s run)')
    click(pg, '#dl-app .modes button[data-mode="guided"]', 500)
    for k, i in pg.evaluate("()=>{const d=window.__dl.docs('web'), seen={}; d.forEach((x,i)=>{if(!(x.kind in seen)) seen[x.kind]=i;}); return Object.entries(seen)}"):
        html = pg.evaluate(f"()=>window.__dl.tip({i})")
        D.append(f'- {k}: ' + clean(re.sub(r'<[^>]+>', ' ', html.replace('</tr>', ';'))))
    D.append('\n# Connect')
    click(pg, '#dl-app [data-act="drill-reveal"]')
    D += lines(txt(pg, '#dl-app .connect'))
    D += tips(pg, '#dl-app')
    pg.close()
    (OUT / f'copy-{lang}.md').write_text('\n'.join(D) + '\n')
    return len(D)


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ('en', 'zh'):
        print(lang, deck(b, lang), 'lines ->', OUT / f'copy-{lang}.md')
    b.close()
