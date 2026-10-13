"""Dump every piece of text a learner meets in the pass-k lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/pass-k_copy_deck.py  ->  out/pass-k/copy-en.md and out/pass-k/copy-zh.md

Order: home tile (each pick, the notes), lab header and both popovers, each guided part (card before and after its action, the stage,
the tooltips in view not shown before, Settings, the mini result bar), part 2 at both ends, part 3 with each way, part 5 with each
product, part 6 with each “Show in the chart”, Show the math, the scenarios, the settings panel, the challenge (every feedback tier and
the chart), the card “Can you use it on a new case?” (every question, a verdict for every option), and Connect with the drill answer.
A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'pass-k'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/pass-k.html', '--only', 'pass-k'], cwd=P, check=True, capture_output=True)
(OUT / 'deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'pass-k.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'deck.html')
clean = lambda s: s.replace('‑', '-').replace(' ', ' ').replace(' ', ' ').strip()


def txt(pg, sel):
    el = pg.query_selector(sel)
    return clean(el.inner_text()) if el and el.is_visible() else ''


def stage(pg):
    parts = [txt(pg, '#pak-st-title'), txt(pg, '#pak-st-sub'), 'Status: ' + txt(pg, '#pak-status'), 'Grid: ' + txt(pg, '#pak-gridbox .pak-cap'),
             'Grid groups: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#pak-grid .pak-gl')].map(t=>t.textContent)")),
             'Player: ' + ' '.join(txt(pg, '#pak-player').split()), 'Chart: ' + txt(pg, '#pak-chartbox .pak-cap'),
             'Chart labels: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#pak-chart text.pak-end, #pak-chart text.pak-vl, #pak-chart text.pak-taul, #pak-chart text.pak-wait, #pak-chart text.pak-q')].map(t=>t.textContent)")),
             'Chart note: ' + txt(pg, '#pak-chart-note')]
    mets = pg.evaluate("()=>[...document.querySelectorAll('#pak-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    return '\n'.join([p for p in parts if p] + ['Metrics:'] + [f'- {clean(m)}' for m in mets])


class Tips:
    def __init__(self): self.seen = set()

    def grab(self, pg, root):
        out = []
        for e in pg.query_selector_all(f'{root} [data-tipx], {root} .t, {root} [data-jump]'):
            try:
                if not e.is_visible(): continue
                label = ' '.join((e.evaluate("el=>el.innerText||el.textContent||''") or '').split())[:50]
                tip = e.get_attribute('data-tipx')
                if tip is None:
                    e.scroll_into_view_if_needed(timeout=1500); pg.wait_for_timeout(30)
                    e.focus(); pg.wait_for_timeout(80)
                    tip = ' '.join(pg.inner_text('#tip').split()) if pg.is_visible('#tip') else ''
                    pg.evaluate("()=>document.activeElement&&document.activeElement.blur()")
            except Exception:
                continue
            t = ' '.join((tip or '').split())
            if t and t not in self.seen:
                self.seen.add(t); out.append(f'- {clean(label)} ⟶ {clean(t)}')
        return out


def fast(pg):
    pg.evaluate("()=>{const b=document.querySelector('#pak-player .seg-ctl[data-key=\"spd\"] button[data-v=\"fast\"]'); if(b&&b.getAttribute('aria-pressed')!=='true') b.click()}")
    try: pg.wait_for_function("()=>{const s=window.__pak.state(); return s.shown>=s.n}", timeout=6000)
    except Exception: pass
    pg.wait_for_timeout(150)


def dock(pg):
    vp = pg.viewport_size
    pg.set_viewport_size({'width': 390, 'height': 844}); pg.wait_for_timeout(250)
    pg.evaluate("()=>{const t=document.getElementById('pak-chartbox'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+60)}"); pg.wait_for_timeout(500)
    t = clean(pg.inner_text('#dock')) if pg.is_visible('#dock') else '(mini result bar not shown)'
    pg.set_viewport_size(vp); pg.wait_for_timeout(200)
    return 'Mini result bar (390 px): ' + ' '.join(t.split())


def rng(pg, sel, v):
    pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}))}}"); pg.wait_for_timeout(220); fast(pg)


def go(pg, k):
    pg.evaluate(f"()=>document.querySelector('#pak-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(300); fast(pg)


def math(pg, title):
    pg.evaluate("()=>{document.getElementById('pak-math').open=true}"); pg.wait_for_timeout(150)
    out = [f'## Show the math ({title})', txt(pg, '#pak-mathbody')]
    pg.evaluate("()=>{document.getElementById('pak-math').open=false}")
    return out


def exits(pg):
    out = []
    card = "[...document.querySelectorAll('[data-exits]')].find(e=>e.offsetParent!==null)"
    for t in range(3):
        pg.evaluate(f"()=>{card}.querySelector('[data-ex=\"tab\"][data-i=\"{t}\"]').click()"); pg.wait_for_timeout(150)
        name = clean(pg.evaluate(f"()=>{card}.querySelector('[data-ex=\"tab\"][data-i=\"{t}\"]').textContent"))
        if t == 2:
            out += [f'## {name}', clean(pg.evaluate(f"()=>{card}.querySelector('.ex-body').innerText"))]; continue
        nq = pg.evaluate(f"()=>{card}.querySelectorAll('.ex-qs .pb-dot').length||1")
        for j in range(nq):
            if j: pg.evaluate(f"()=>{card}.querySelector('.ex-qs .pb-dot[data-j=\"{j}\"]').click()"); pg.wait_for_timeout(150)
            q = clean(pg.evaluate(f"()=>{{const e={card}.querySelector('.ex-body > p > b'); return e?e.innerText:''}}"))
            opts = pg.evaluate(f"()=>[...{card}.querySelectorAll('.ex-opts .choice')].map(b=>({{v:+b.dataset.v, t:b.innerText}}))")
            out += [f'## {name}, question {j + 1}', q] + [f'- ({o["v"]}) {clean(o["t"])}' for o in opts]
            for o in opts:
                pg.evaluate(f"()=>{card}.querySelector('.ex-opts .choice[data-v=\"{o['v']}\"]').click()"); pg.wait_for_timeout(200)
                r = pg.evaluate(f"()=>{{const e={card}.querySelector('.ex-res'); return e?{{ok:e.classList.contains('ok'),t:e.innerText}}:null}}")
                if not r: continue
                out += [f'### After picking ({o["v"]})', clean(r['t'])]
                if r['ok']: break
    return out


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ['en', 'zh']:
        pg = b.new_page(viewport={'width': 1440, 'height': 1000})
        errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        T = Tips(); Lines = []
        pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="eval"], #domStrip [data-world="eval"]')].find(e=>e.offsetParent); if(g) g.click();}"""); pg.wait_for_timeout(400)
        if pg.query_selector('#tile-pass-k'):
            pg.evaluate("()=>document.querySelector('#tile-pass-k').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
            Lines += ['# Home tile', txt(pg, '#tile-pass-k')]
            for v in ('even', 'mix', 'split'):
                pg.click(f'#tile-pass-k .seg-ctl button[data-v="{v}"]'); pg.wait_for_timeout(150)
                Lines += [f'## Tile sentence: {v}', txt(pg, '#tile-pass-k .tz-read')]
            Lines += ['## Tile notes', clean(pg.evaluate("()=>[...document.querySelectorAll('#tile-pass-k .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')"))]
            Lines += ['## Tile tooltips'] + T.grab(pg, '#tile-pass-k')
        else:
            Lines += ['# Home tile', '(tile not found)']
        pg.evaluate("()=>{location.hash='pass-k'}"); pg.wait_for_timeout(700)
        Lines += ['# Lab header', txt(pg, '.dyn-root .labhead')] + ['## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
            Lines += [f'## Popover: {clean(" ".join(sel.inner_text().split()))}', clean(pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid))]
            Lines += ['## Popover tooltips'] + T.grab(pg, f'#{pid}')
            sel.click(); pg.wait_for_timeout(150)
        ids = pg.evaluate("()=>window.__pak.consts.STEP_IDS")
        for k, sid in enumerate(ids):
            go(pg, k)
            Lines += [f'# Guided part {k + 1} ({sid})', '## Card', txt(pg, '#pak-rail .step.current'), '## Stage', stage(pg)]
            Lines += ['## Settings panel', txt(pg, '#pak-controls'), '## Tooltips in view (new)'] + T.grab(pg, '#pak-app .bench')
            if sid == 'mission':
                pg.click('#pak-rail [data-pred="high"]'); pg.wait_for_timeout(250)
                Lines += ['## Card after a prediction', txt(pg, '#pak-rail .step.current')]
                Lines += math(pg, 'part 1')
            if sid == 'uneven':
                for v in (0, 4, 2):
                    rng(pg, '#pak-ic-lv-in', v); Lines += [f'## Result at level {v}', txt(pg, '#pak-rail [data-obs]'), '## Stage', stage(pg)]
                Lines += math(pg, 'part 2, level 2') + ['## Math tooltips (new)'] + T.grab(pg, '#pak-mathbody')
            if sid == 'estimate':
                for lev in ('first', 'full'):
                    pg.click(f'#pak-rail [data-lever="{lev}"]'); pg.wait_for_timeout(250)
                    Lines += [f'## After picking “{lev}”', txt(pg, '#pak-rail .step.current')]
            if sid == 'every':
                for v in (1, 2, 3):
                    rng(pg, '#pak-ic-lv4-in', v); Lines += [f'## Result at level {v}', txt(pg, '#pak-rail [data-obs]')]
                Lines += ['## Stage', stage(pg)] + math(pg, 'part 4, level 3')
            if sid == 'decide':
                pg.click('#pak-rail [data-lever="support"]'); pg.wait_for_timeout(250); fast(pg)
                Lines += ['## After picking support', txt(pg, '#pak-rail .step.current'), '## Stage', stage(pg)]
            if sid == 'remember':
                for r in ('estimate', 'every', 'uneven'):
                    pg.click(f'#pak-rail [data-rem="{r}"]'); pg.wait_for_timeout(300); fast(pg)
                    Lines += [f'## Part 6 after “Show in the chart” ({r})', '## Stage', stage(pg), '## Tooltips in view (new)'] + T.grab(pg, '#pak-app .bench')
            Lines += [dock(pg)]
        pg.click('#pak-app .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(400)
        Lines += ['# Scenarios', txt(pg, '#pak-rail')]
        for i in range(4):
            pg.click(f'#pak-rail [data-preset="{i}"]'); pg.wait_for_timeout(300); fast(pg)
            name = clean(' '.join(pg.inner_text(f'#pak-rail [data-preset="{i}"]').split()))
            Lines += [f'## Scenario {i + 1}: {name}', txt(pg, '#pak-insight'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#pak-stage') + [dock(pg)]
        rng(pg, '#pak-ctl-k-in', 7)
        Lines += ['## After changing k under Settings (no scenario matches)', txt(pg, '#pak-insight')]
        Lines += ['# Settings panel', txt(pg, '#pak-controls'), '## Settings tooltips (new)'] + T.grab(pg, '#pak-controls')
        pg.click('#pak-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(400)
        Lines += ['# Challenge', '## Card before submitting', txt(pg, '#pak-rail .railcard'), '## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#pak-controls')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#pak-app .bench') + [dock(pg)]
        Lines += math(pg, 'challenge, B with 1 try')
        seg = lambda k, v: (pg.click(f'#pak-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]'), pg.wait_for_timeout(250))
        pg.click('#pak-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        Lines += ['## Submitted the initial setup (missed)', txt(pg, '#pak-ch-res')]
        seg('chAg', 'A'); seg('chK', '2'); Lines += ['## A with 2 tries (missed)', txt(pg, '#pak-ch-res')]
        pg.click('#pak-rail [data-act="reveal"]'); pg.wait_for_timeout(300)
        Lines += ['## After “Show the answer”', txt(pg, '#pak-ch-res')]
        seg('chK', '6'); Lines += ['## A with 6 tries (met, more calls than needed)', txt(pg, '#pak-ch-res')]
        seg('chK', '3'); Lines += ['## A with 3 tries (met, the fewest calls)', txt(pg, '#pak-ch-res'), 'Status: ' + txt(pg, '#pak-status'), '## Stage', stage(pg), dock(pg)]
        Lines += math(pg, 'challenge, A with 3 tries')
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#pak-rail')
        Lines += ['# Can you use it on a new case?'] + exits(pg)
        pg.click('#pak-app [data-act="drill-reveal"]'); pg.wait_for_timeout(200)
        Lines += ['# Connect', txt(pg, '#pak-app .connect'), '## Connect tooltips (new)'] + T.grab(pg, '#pak-app .connect')
        if errs: Lines += ['# Page errors'] + errs
        (OUT / f'copy-{lang}.md').write_text('\n\n'.join(x for x in Lines if x) + '\n')
        pg.close()
    b.close()
print('wrote', [str(OUT / f'copy-{l}.md') for l in ['en', 'zh']])
