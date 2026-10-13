"""Dump every piece of text a learner meets in the MoE lab and its home tile, in reading order, in English and Chinese
(CONTRACT §4.0: export the whole text in reading order and read it before reporting).

python3 tests/moe_copy_deck.py  ->  out/moe-copy/en.md and out/moe-copy/zh.md

Order: home tile (the control, the captions, the sentence and the three bars at every setting, the picture as read aloud, the legend,
the tooltips, the notes), the lab header with both popovers, each guided part (the card before and
after its action, the stage's captions, the legend with its tooltips, the metric cards with theirs, the mini result bar), Show the
math, the five scenarios, the settings panel, the challenge (every feedback tier and the chart), the card "Can you use it on a new
case?" (every question with its verdict for every option), and the Connect section with the drill answer. A tooltip is printed the
first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'moe-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/moe-deck.html', '--only', 'moe'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'moe-deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'moe-deck.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'moe-deck-wrapped.html')
clean = lambda s: (s or '').replace('‑', '-').replace(' ', ' ').replace(' ', ' ').strip()


def txt(pg, sel):
    el = pg.query_selector(sel)
    return clean(el.inner_text()) if el and el.is_visible() else ''


def deck(lang):
    out, seen = [], set()
    add = lambda s='': out.append(s)
    def tips(pg, root, label):
        items = pg.evaluate("""(r)=>[...document.querySelectorAll(r+' [data-tipx]')].filter(e=>e.getClientRects().length).map(e=>[e.innerText||e.getAttribute('aria-label')||'',e.getAttribute('data-tipx')])""", root)
        new = [(clean(a).replace('\n', ' / ')[:60], clean(t).replace('\n', ' ')) for a, t in items if clean(t) and clean(t) not in seen]
        if new:
            add(f'  _{label} tooltips:_')
            for a, t in new: seen.add(t); add(f'  - [{a}] {t}')
    def stage(pg):
        add('**Stage:** ' + ' / '.join(x for x in [txt(pg, '#moe-st-title'), txt(pg, '#moe-st-sub'), 'status: ' + txt(pg, '#moe-status')] if x))
        add('Legend: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#moe-legend > span')].map(s=>s.innerText)")))
        for sel in ('#moe-flow-cap', '#moe-flow-badge', '#moe-par-cap', '#moe-gpu-cap', '#moe-memleg', '#moe-read-cap', '#moe-rlab', '#moe-ep-cap'):
            t = txt(pg, sel)
            if t: add(f'- {sel}: {t}')
        if pg.is_visible('#moe-prows'): add('- parameter bars: ' + txt(pg, '#moe-prows').replace('\n', ' '))
        if pg.is_visible('#moe-eprows'): add('- GPUs: ' + txt(pg, '#moe-eprows').replace('\n', ' '))
        if pg.is_visible('#moe-cmp'): add('- comparison: ' + txt(pg, '#moe-cmp').replace('\n', ' | '))
        svg = pg.evaluate("()=>[...document.querySelectorAll('#moe-graph svg text')].filter(e=>e.getClientRects().length).map(e=>e.textContent)")
        if svg: add('- picture labels: ' + ' | '.join(clean(x) for x in svg))
        add('- scope: ' + txt(pg, '#moe-scope'))
        add('- cards: ' + ' || '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#moe-metrics .metric')].map(m=>m.innerText.replace(/\\n/g,' / '))")))
        tips(pg, '#moe-stage', 'stage')
    def card(pg, label):
        add(f'**{label}:**'); add(txt(pg, '#moe-rail .step.current .step-b') or txt(pg, '#moe-rail .railcard')); tips(pg, '#moe-rail', 'card')
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 1000})
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(300)
        # the home tile
        add('# Home tile'); tile = '.tile .tz-moe'
        pg.evaluate("()=>{location.hash='map~models'}"); pg.wait_for_timeout(700)
        if pg.query_selector(tile):
            add('Control: ' + clean(pg.evaluate("()=>document.querySelector('.tz-moe .ctl .lbl').innerText")) + ' | ' + ' | '.join(pg.evaluate("()=>[...document.querySelectorAll('.tz-moe .seg-ctl button')].map(b=>b.textContent)")))
            add('Captions: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('.tz-moe .tz-moe-cap')].map(e=>e.innerText)")))
            for v in ('1', '4', '16', '64'):
                pg.evaluate(f"()=>{{const b=document.querySelector('.tz-moe .seg-ctl button[data-v=\"{v}\"]'); if(b) b.click()}}"); pg.wait_for_timeout(150)
                add(f'- [{v}] ' + clean(pg.evaluate("()=>document.querySelector('.tz-moe .tz-read').innerText")))
                add('  bars: ' + ' | '.join(clean(n) + ' ' + clean(x) for n, x in pg.evaluate("()=>[...document.querySelectorAll('.tz-moe .tz-moe-n')].map((e,i)=>[e.textContent, document.querySelectorAll('.tz-moe .tz-moe-v')[i].textContent])")))
            add('- the picture, read aloud (at 64): ' + clean(pg.evaluate("()=>document.querySelector('.tz-moe .tz-moe-vis').getAttribute('aria-label')")))
            add('- legend: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('.tz-moe .tz-leg > span')].map(e=>e.innerText)")))
            tips(pg, '.tz-moe', 'tile')
            add(); add('## Assumptions & sources (the tile’s notes)')
            for i, x in enumerate(pg.evaluate("()=>[...document.querySelectorAll('.tz-moe .tz-note')].map(e=>e.textContent)")): add(f'{i + 1}. ' + clean(x))
        else: add('(the tile was not found on the map in this build)')
        pg.evaluate("()=>{location.hash='moe'}"); pg.wait_for_timeout(700)
        add(); add('# Header')
        add(txt(pg, '#moe-app .crumbs')); add(txt(pg, '#moe-app .labtitle')); add(txt(pg, '#moe-app .lab-outcome')); add(txt(pg, '#moe-app .labmeta')); add(txt(pg, '#moe-app [data-mode-desc]'))
        # since 2026-10-11 the notes close the page (no Fidelity or Primer button)
        add('**Notes at the page end:**'); add(txt(pg, '#moe-fid')); tips(pg, '#moe-fid', 'notes')
        acts = {1: [("#moe-rail [data-pred='lt40']", 'after a wrong prediction')], 2: [("#moe-rail [data-act='route']", 'after 1 token'), ("#moe-rail [data-act='route']", 'after 2 tokens'), ("#moe-rail [data-act='route']", 'after 3 tokens')],
                3: [("#moe-rail .seg-ctl[data-key='p'] button[data-v='fp8']", 'after FP8')],
                5: [("#moe-rail [data-lever='l70']", 'after “Llama 2 70B”'), ("#moe-rail [data-lever='d47']", 'after “Dense 46.7B”'), ("#moe-cmp .seg-ctl[data-key='bench'] button[data-v='1']", 'after the benchmark GSM8K')],
                6: [("#moe-rail [data-lever='q']", 'after “one popular expert”'), ("#moe-rail [data-lever='h']", 'after “every token picks it”')]}
        for i in range(8):
            pg.evaluate(f"()=>document.querySelector('#moe-rail [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(300)
            add(); add(f'# Part {i + 1}: ' + txt(pg, '#moe-rail .step.current .step-h'))
            add('Bar lead: ' + clean(pg.evaluate("()=>{const e=document.querySelector('#moe-rail .pbar-lead,.pb-lead'); return e?e.textContent:''}")))
            card(pg, 'Card'); stage(pg)
            for sel, label in acts.get(i, []):
                pg.evaluate(f"()=>document.querySelector(\"{sel}\").click()"); pg.wait_for_timeout(300)
                add(f'**Result {label}:**'); add(txt(pg, '#moe-rail [data-obs]')); tips(pg, '#moe-rail', 'result'); stage(pg)
            if i == 4:
                for B in (8, 64):
                    pg.evaluate(f"()=>{{const x=document.querySelector('#moe-ic-B-in'); x.value={B}; x.dispatchEvent(new Event('input',{{bubbles:true}}));}}"); pg.wait_for_timeout(300)
                    add(f'**Result at {B} requests:**'); add(txt(pg, '#moe-rail [data-obs]')); stage(pg)
        pg.evaluate("()=>document.querySelector('#moe-math').open=true"); pg.wait_for_timeout(150)
        add(); add('# Show the math (part 8)'); add(txt(pg, '#moe-mathbody')); tips(pg, '#moe-mathbody', 'math')
        pg.evaluate("()=>document.querySelector('#moe-rail [data-mode-go=\"sandbox\"]').click()"); pg.wait_for_timeout(400)
        add(); add('# Scenarios'); add(txt(pg, '#moe-rail .railcard'))
        for i in range(5):
            pg.evaluate(f"()=>document.querySelector('#moe-rail [data-preset=\"{i}\"]').click()"); pg.wait_for_timeout(300)
            add(f'**Scenario {i + 1}:**'); add(txt(pg, '#moe-insight')); stage(pg)
        pg.evaluate("()=>document.querySelector('#moe-rail [data-preset=\"2\"]').click()"); pg.wait_for_timeout(300)
        pg.evaluate("()=>document.querySelector('#moe-curve').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
        bb = pg.evaluate("()=>{const r=document.querySelector('#moe-curve svg').getBoundingClientRect(); return {x:r.left,y:r.top,w:r.width,h:r.height}}")
        pg.mouse.move(bb['x'] + bb['w'] * 0.3, bb['y'] + bb['h'] * 0.5); pg.wait_for_timeout(200)
        add('**Readout on the curve:** ' + clean(pg.evaluate("()=>{const t=document.getElementById('tip'); return t&&!t.hidden?t.innerText:''}")))
        pg.mouse.click(bb['x'] + bb['w'] * 0.3, bb['y'] + bb['h'] * 0.5); pg.wait_for_timeout(300)
        add('**After a click on the curve:**'); add(txt(pg, '#moe-insight')); add('- readout: ' + clean(pg.evaluate("()=>{const t=document.getElementById('tip'); return t&&!t.hidden?t.innerText:''}")))
        add(); add('# Settings'); add(txt(pg, '#moe-controls')); tips(pg, '#moe-controls', 'settings')
        pg.evaluate("()=>document.querySelector('#moe-rail [data-mode-go=\"challenge\"]').click()"); pg.wait_for_timeout(400)
        add(); add('# Challenge'); add(txt(pg, '#moe-rail .railcard')); stage(pg); add('Settings: ' + txt(pg, '#moe-controls'))
        for B, label in ((30, 'not met'), (10, 'met, more fit'), (25, 'met, the most')):
            pg.evaluate(f"()=>{{const x=document.querySelector('#moe-ic-chB-in'); x.value={B}; x.dispatchEvent(new Event('input',{{bubbles:true}}));}}"); pg.wait_for_timeout(250)
            if pg.query_selector('#moe-rail [data-act="submit"]'): pg.evaluate("()=>document.querySelector('#moe-rail [data-act=\"submit\"]').click()"); pg.wait_for_timeout(300)
            add(f'**Submitted, {B} users ({label}):**'); add(txt(pg, '#moe-ch-res'))
        add('Mini result bar text: ' + clean(pg.evaluate("()=>{const d=document.getElementById('dock'); return d?d.innerText:''}")))
        add(); add('# Can you use it on a new case?')
        for t in (0, 1, 2):
            pg.evaluate(f"()=>document.querySelector('[data-exits] [data-ex=\"tab\"][data-i=\"{t}\"]').click()"); pg.wait_for_timeout(200)
            if t == 2: add(txt(pg, '[data-exits] .ex-body')); continue
            nq = pg.evaluate("()=>document.querySelectorAll('[data-exits] .ex-qs [data-ex=\"q\"]').length") or 1   # the dots, not "Next question"
            for j in range(nq):
                pg.evaluate(f"()=>{{const b=document.querySelector('[data-exits] [data-ex=\"q\"][data-j=\"{j}\"]'); if(b) b.click()}}"); pg.wait_for_timeout(200)
                nopt = pg.evaluate("()=>document.querySelectorAll('[data-exits] .ex-opts .choice').length")
                add(f'**Tab {t + 1}, question {j + 1}:**'); add(txt(pg, '[data-exits] .ex-body > p'))
                for k in range(nopt):
                    pg.evaluate(f"()=>document.querySelectorAll('[data-exits] .ex-opts .choice')[{k}].click()"); pg.wait_for_timeout(200)
                    add(f'- pick {k + 1}: ' + clean(pg.evaluate(f"()=>document.querySelectorAll('[data-exits] .ex-opts .choice')[{k}].innerText")) + ' → ' + txt(pg, '[data-exits] .ex-res').replace('\n', ' '))
        add(); add('# Connect')
        pg.evaluate("()=>document.querySelector('[data-act=\"drill-reveal\"]').click()"); pg.wait_for_timeout(200)
        add('\n'.join(('    ' + l if l.lstrip().startswith('#') else l) for l in txt(pg, '#moe-app .connect').split('\n')))   # code comments are not headings
        b.close()
    return '\n'.join(out) + '\n'


for lang in ('en', 'zh'):
    (OUT / f'{lang}.md').write_text(deck(lang))
    print('wrote', OUT / f'{lang}.md', len((OUT / f'{lang}.md').read_text()), 'chars')
