"""Dump every piece of text a learner meets in the GQA and MLA lab and its home tile, in reading order.
Modeled on tests/kv_copy_deck.py. python3 tests/gqa-mla_copy_deck.py  ->  out/gqa-mla-copy/en.md and zh.md

Order: the home tile (every option, its notes and tooltips), the lab header (mode tooltips, primer and fidelity popovers), each of the
7 guided parts (the card before and after its action, every option of the part's control with its result, the stage with legend,
status, captions, chart, metrics, and every tooltip in view the first time it appears), Show the math with the tooltip of every term,
the mini result bar on a phone, the scenarios with every scenario picked, the settings panel with every note, the challenge before and
after each kind of answer, and the Connect section with one drill and its answer.
Tooltips are read by hovering each element and reading #tip, so jump lines ("Click to jump to …") are included.
Round 14 (review 8): the tour starts from MHA; part 2 shares keys and values, part 4 predicts, part 5 checks; the scenarios start with MHA;
the challenge shows what Settings says for a few designs before a submit.
No-break hyphens (U+2011) and no-break spaces (U+00A0), which keep names on one line on the page, are written as plain ones."""
import pathlib, sys, subprocess
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out/gqa-mla-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/gqa-mla-copy/page.html', '--only', 'gqa-mla'], cwd=P, check=True, capture_output=True)
(OUT / 'wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                                  '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (OUT / 'page.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped.html')
TIPSEL = '[data-tip],[data-tipx],[data-tipfn],.t,[data-jump]'
plain = lambda s: (s or '').replace('‑', '-').replace(' ', ' ')


def txt(pg, sel):
    el = pg.query_selector(sel); return plain(el.inner_text().strip()) if el else ''


def ev(pg, js, arg=None): return pg.evaluate(js, arg)


def click(pg, sel, wait=250):
    el = pg.query_selector(sel)
    if el:
        el.scroll_into_view_if_needed(); el.click(); pg.wait_for_timeout(wait)
    return el is not None


def tips(pg, root, seen):
    """Hover every visible tooltip target under root and return the tooltips not seen before (label ⟶ tooltip)."""
    out = []
    for e in pg.query_selector_all(', '.join(f'{root} {s}' for s in TIPSEL.split(','))):
        try:
            if not e.is_visible(): continue
            e.scroll_into_view_if_needed(timeout=1500)
            # the first line box: a term that wraps onto two lines has a bounding box whose middle can fall outside the term
            pt = e.evaluate("e=>{const r=e.getClientRects()[0]; return r?{x:r.left+Math.min(r.width/2,20), y:r.top+r.height/2}:null}")
            if not pt: continue
            pg.mouse.move(5, 5); pg.mouse.move(pt['x'], pt['y']); pg.mouse.move(pt['x'] + 1, pt['y'])
        except Exception:
            continue
        pg.wait_for_timeout(110)
        t = plain(ev(pg, "()=>{const t=document.getElementById('tip'); return t.hidden?'':t.innerText.trim()}"))
        if not t or t in seen: continue
        seen.add(t)
        label = plain((e.inner_text() or e.get_attribute('aria-label') or '').replace('\n', ' ').strip()[:60])
        out.append(f'- {label} ⟶ {t}'.replace('\n', ' | '))
    ev(pg, "()=>{const t=document.getElementById('tip'); t.hidden=true}")
    pg.mouse.move(5, 5)
    return out


def section(L, title, body):
    L.append(title); L.append(body if body else '(empty)')


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ['en', 'zh']:
        ctx = b.new_context(viewport={'width': 1440, 'height': 1000})
        pg = ctx.new_page(); pg.goto(URL); pg.wait_for_timeout(300)
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        L, seen = [], set()
        # ---------- home tile (Models domain) ----------
        ev(pg, "()=>document.querySelector('#mapSvg .world[data-world=\"models\"]').dispatchEvent(new MouseEvent('click',{bubbles:true}))"); pg.wait_for_timeout(300)
        ev(pg, "()=>document.querySelector('#tile-gqa-mla').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
        section(L, '# Home tile (Models domain)', txt(pg, '#tile-gqa-mla'))
        section(L, '## Assumptions & sources', plain(ev(pg, "()=>[...document.querySelectorAll('#tile-gqa-mla .tz-note')].map(n=>n.innerText.trim()).join('\\n\\n')")))
        L.append('## Tooltips in view'); L.extend(tips(pg, '#tile-gqa-mla', seen))
        for v in ['mha', 'gqa8', 'mqa', 'mla']:
            click(pg, f'#tile-gqa-mla .seg-ctl button[data-v="{v}"]', 150)
            section(L, f'## Tile, option {v}', txt(pg, '#tile-gqa-mla .tz-read') + '\n[legend: ' + txt(pg, '#tile-gqa-mla .tz-leg').replace('\n', ' | ') + ']')
        click(pg, '#tile-gqa-mla .seg-ctl button[data-v="gqa8"]', 100)
        # ---------- lab header ----------
        ev(pg, "()=>{location.hash='gqa-mla'}"); pg.wait_for_timeout(700)
        section(L, '# Lab header', txt(pg, '.dyn-root .labhead'))
        L.append('## Tooltips in view'); L.extend(tips(pg, '.dyn-root .labhead', seen))
        # since 2026-10-11 the notes close the page (no Fidelity or Primer button)
        section(L, '## Notes at the page end', txt(pg, '#gqa-mla-fid'))
        L.append('## Tooltips in view'); L.extend(tips(pg, '#gqa-mla-fid', seen))
        ev(pg, "()=>window.scrollTo(0,0)")
        # ---------- guided parts ----------
        for k in range(7):
            ev(pg, f"()=>document.querySelector('#gm-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(350)
            L.append(f'# Guided part {k + 1}')
            section(L, '## Card', txt(pg, '#gm-rail .rail-top') + '\n' + txt(pg, '#gm-rail .step.current'))
            section(L, '## Stage', txt(pg, '#gm-stage'))
            L.append('## Tooltips in view'); L.extend(tips(pg, '.dyn-root .bench', seen))
            if k == 3:
                click(pg, '#gm-rail [data-pred="many"]', 250)
                section(L, '## Card after a prediction (“64 or more”)', txt(pg, '#gm-rail .step.current'))
                section(L, '## Stage after the prediction', txt(pg, '#gm-stage .stage-head') + '\n' + txt(pg, '#gm-metrics'))
            if k in (1, 4):
                sel = '#gm-ic-kv' if k == 1 else '#gm-ic-kv5'
                for v in ['mha', 'gqa16', 'gqa4', 'gqa8', 'mqa']:
                    click(pg, f'{sel} button[data-v="{v}"]', 250)
                    section(L, f'## Picked {v}: result', txt(pg, '#gm-rail [data-obs]'))
                    if v in ('mha', 'mqa'):
                        section(L, f'## Picked {v}: stage', txt(pg, '#gm-stage'))
                        L.append('## Tooltips in view'); L.extend(tips(pg, '.dyn-root .bench', seen))
                ev(pg, "()=>{const i=document.getElementById('gm-model'); i.value='llama31-8b'; i.dispatchEvent(new Event('change',{bubbles:true}))}"); pg.wait_for_timeout(250)
                section(L, '## After changing the model under Settings by hand', txt(pg, '#gm-rail [data-obs]'))
                click(pg, f'{sel} button[data-v="mqa"]', 250)
            if k == 5:
                click(pg, '#gm-ic-mla button[data-v="mla"]', 300)
                section(L, '## Picked MLA: card', txt(pg, '#gm-rail .step.current'))
                section(L, '## Picked MLA: stage', txt(pg, '#gm-stage'))
                L.append('## Tooltips in view'); L.extend(tips(pg, '.dyn-root .bench', seen))
                click(pg, '#gm-ic-mla button[data-v="gqa8"]', 250)
                section(L, '## Picked GQA-8 again: result', txt(pg, '#gm-rail [data-obs]'))
                click(pg, '#gm-ic-mla button[data-v="mla"]', 250)
            if k == 4:
                ev(pg, "()=>document.getElementById('gm-math').open=true"); pg.wait_for_timeout(150)
                section(L, '# Show the math (open, part 5, MQA)', txt(pg, '#gm-math'))
                L.append('## Term tooltips'); L.extend(tips(pg, '#gm-mathbody', seen))
                ev(pg, "()=>document.getElementById('gm-math').open=false")
        # ---------- phone: the mini result bar ----------
        pc = b.new_context(viewport={'width': 390, 'height': 844})
        ph = pc.new_page(); ph.goto(URL); ph.wait_for_timeout(300); ph.click(f'.langsw button[data-lang="{lang}"]'); ph.wait_for_timeout(150)
        ev(ph, "()=>{location.hash='gqa-mla'}"); ph.wait_for_timeout(500)
        for k, what in [(1, 'part 2 (before the prediction is checked)'), (4, 'part 5')]:
            ev(ph, f"()=>document.querySelector('#gm-rail [data-go=\"{k}\"]').click()"); ph.wait_for_timeout(300)
            sel = '#gm-ic-kv' if k == 1 else '#gm-ic-kv5'
            ev(ph, f"()=>document.querySelector('{sel}').scrollIntoView({{block:'center'}})"); ph.wait_for_timeout(250)
            ph.click(f'{sel} button[data-v="mqa"]'); ph.wait_for_timeout(500)
            section(L, f'# Phone (390 px): mini result bar in {what}, after picking 1 KV head', txt(ph, '#dock'))
        pc.close()
        # ---------- scenarios ----------
        ev(pg, "()=>window.scrollTo(0,0)"); pg.click('.dyn-root .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(400)
        section(L, '# Scenarios', txt(pg, '.dyn-root [data-mode-desc]') + '\n\n' + txt(pg, '#gm-rail'))
        L.append('## Tooltips in view'); L.extend(tips(pg, '.dyn-root .bench', seen))
        for i in range(len(pg.query_selector_all('#gm-rail [data-preset]'))):
            click(pg, f'#gm-rail [data-preset="{i}"]', 300)
            section(L, f'## Scenario {i + 1} picked: result line', txt(pg, '#gm-sbnote'))
            section(L, f'## Scenario {i + 1} picked: stage', txt(pg, '#gm-stage'))
            L.append('## Tooltips in view'); L.extend(tips(pg, '#gm-stage', seen))
        ev(pg, "()=>{const i=document.getElementById('gm-ctl-ctx-in'); i.value=3; i.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(250)
        section(L, '## After a hand change (context 8K): result line', txt(pg, '#gm-sbnote'))
        # ---------- settings ----------
        click(pg, '#gm-rail [data-preset="0"]', 250)
        ev(pg, "()=>{document.getElementById('gm-fixed').open=true}"); pg.wait_for_timeout(150)
        section(L, '# Settings panel', txt(pg, '#gm-controls'))
        L.append('## Tooltips in view'); L.extend(tips(pg, '#gm-controls', seen))
        for v in ['mha', 'gqa16', 'mqa', 'mla']:
            click(pg, f'#gm-ctl-attn button[data-v="{v}"]', 200)
            section(L, f'## Attention design “{v}”: its note', txt(pg, '#gm-attn-note'))
        ev(pg, "()=>{const i=document.getElementById('gm-model'); i.value='deepseek-v3'; i.dispatchEvent(new Event('change',{bubbles:true}))}"); pg.wait_for_timeout(250)
        section(L, '## Model DeepSeek-V3: notes', txt(pg, '#gm-model-note') + '\n' + txt(pg, '#gm-attn-note'))
        section(L, '## Model DeepSeek-V3 on 8 × H100: stage status and users', txt(pg, '#gm-status') + '\n' + txt(pg, '#gm-metrics'))
        click(pg, '#gm-rail [data-preset="0"]', 250)
        # ---------- challenge ----------
        ev(pg, "()=>window.scrollTo(0,0)"); pg.click('.dyn-root .modes button[data-mode="challenge"]'); pg.wait_for_timeout(400)
        section(L, '# Challenge (start)', txt(pg, '.dyn-root [data-mode-desc]') + '\n\n' + txt(pg, '#gm-rail'))
        section(L, '## Stage', txt(pg, '#gm-stage'))
        section(L, '## Settings (locked)', txt(pg, '#gm-controls'))
        for v in ['mha', 'gqa8', 'mla']:
            click(pg, f'#gm-ctl-attn button[data-v="{v}"]', 200)
            section(L, f'## Challenge, “{v}” picked under Settings: its note, the stage and the result box (no answer before a submit)', txt(pg, '#gm-attn-note') + '\n' + txt(pg, '#gm-stage .stage-head') + '\n[result box: ' + (txt(pg, '#gm-chres') or 'empty') + ']')
        click(pg, '#gm-rail [data-act="reset"]', 200) if pg.query_selector('#gm-rail [data-act="reset"]') else click(pg, '#gm-controls [data-act="reset"]', 200)
        L.append('## Tooltips in view'); L.extend(tips(pg, '.dyn-root .bench', seen))
        for v, what in [('mha', 'not met (MHA)'), ('mla', 'not met (MLA)')]:
            click(pg, f'#gm-rail [data-lever="{v}"]', 200)
            if pg.query_selector('#gm-rail [data-act="submit"]'): click(pg, '#gm-rail [data-act="submit"]', 300)
            section(L, f'## Submitted: {what}', txt(pg, '#gm-chres'))
        click(pg, '#gm-rail [data-act="reveal"]', 300)
        section(L, '## “Show the answer”', txt(pg, '#gm-chres'))
        click(pg, '#gm-rail [data-lever="mqa"]', 300)
        section(L, '## Then picked MQA: met, not the best', txt(pg, '#gm-chres'))
        click(pg, '#gm-rail [data-lever="gqa8"]', 300)
        section(L, '## Then picked GQA-8: met and the best', txt(pg, '#gm-chres'))
        section(L, '## Dock text source (challenge)', txt(pg, '#gm-stage .stage-head'))
        # ---------- connect ----------
        section(L, '# Connect', txt(pg, '.dyn-root .connect'))
        click(pg, '.dyn-root [data-act="drill-show"]', 200)
        section(L, '## Drill answer', txt(pg, '#gm-drilla'))
        click(pg, '.dyn-root [data-act="drill-new"]', 150); click(pg, '.dyn-root [data-act="drill-new"]', 150); click(pg, '.dyn-root [data-act="drill-show"]', 200)
        section(L, '## Drill 3 (MLA) and its answer', txt(pg, '#gm-drillq') + '\n' + txt(pg, '#gm-drilla'))
        (OUT / f'{lang}.md').write_text('\n\n'.join(x for x in L if x is not None) + '\n')
        ctx.close()
    b.close()
print('wrote', [str(OUT / f'{l}.md') for l in ['en', 'zh']])
