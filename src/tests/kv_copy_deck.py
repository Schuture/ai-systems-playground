"""Dump every piece of text a learner meets in the KV cache lab and its home tile, in reading order.
Modeled on tests/pa_copy_deck.py. python3 tests/kv_copy_deck.py  ->  out/kv-copy/zh.md and en.md (the names tests/numdense.py reads);
python3 tests/kv_copy_deck.py <tag>  ->  out/kv-copy/<tag>-zh.md and <tag>-en.md (a named snapshot, e.g. before an edit)

Order: the home hero (for reference only), the KV tile in the Inference domain (every slider position, its notes, its
tooltips), the lab header with the mode tooltips, each of the 7 guided parts (card before and after its action, the stage
with legend, status, bar labels, note and metrics, and every tooltip in view the first time it appears), Show the math with
the tooltip of every term, the mini result bar on a phone, the scenarios card with every scenario picked, the settings
panel with every note, the fidelity and primer popovers, the challenge before and after a submit (wrong, show all, right
but not minimal, right and minimal), the card “Can you use it on a new case?” (every question, a verdict for every option,
wrong ones first), and the Connect section with one drill and its answer.
The KV lab lives in the core (app.js + body.html), so the page is built without any lab module (--only __none__).
Tooltips are read by hovering each element and reading #tip, so jump lines ("Click to jump to …") are included.
Math.random is seeded, so the drill question is the same in every run.
No-break hyphens (U+2011) and no-break spaces (U+00A0), which keep names on one line on the page, are written as plain ones."""
import pathlib, sys, subprocess
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
tag = sys.argv[1] if len(sys.argv) > 1 else ''
name = lambda lang: f'{tag}-{lang}.md' if tag else f'{lang}.md'
OUT = P / 'out/kv-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/kv-copy/page.html', '--only', '__none__'], cwd=P, check=True, capture_output=True)
(OUT / 'wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                                  '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (OUT / 'page.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped.html')
SEED = "(()=>{let s=20260924; Math.random=()=>{s=(s*1664525+1013904223)%4294967296; return s/4294967296;};})()"
TIPSEL = '[data-tip],[data-tipx],[data-tipfn],.t'

def txt(pg, sel):
    el = pg.query_selector(sel); return el.inner_text().strip() if el else ''
def ev(pg, js, arg=None): return pg.evaluate(js, arg)
def click(pg, sel, wait=250):
    el = pg.query_selector(sel)
    if el: el.scroll_into_view_if_needed(); el.click(); pg.wait_for_timeout(wait)
    return el is not None
def tips(pg, root, seen):
    """Hover every visible tooltip target under root and return the tooltips not seen before (label ⟶ tooltip)."""
    out = []
    for e in pg.query_selector_all(', '.join(f'{root} {s}' for s in TIPSEL.split(','))):
        try:
            if not e.is_visible(): continue
            e.hover(timeout=1500); pg.mouse.move(*[v + 1 for v in (lambda b: (b['x'] + b['width'] / 2, b['y'] + b['height'] / 2))(e.bounding_box())])
        except Exception:
            continue
        pg.wait_for_timeout(110)
        t = ev(pg, "()=>{const t=document.getElementById('tip'); return t.hidden?'':t.innerText.trim()}")
        if not t or t in seen: continue
        seen.add(t)
        label = (e.inner_text() or e.get_attribute('aria-label') or '').replace('\n', ' ').strip()[:60]
        out.append(f'{label} ⟶ {t}'.replace('\n', ' | '))
    ev(pg, "()=>{const t=document.getElementById('tip'); t.hidden=true}")
    return out
def section(L, title, body):
    L.append(title); L.append(body if body else '(empty)')
def exits(pg):
    """The card “Can you use it on a new case?”: every question, and the verdict for every option in turn (wrong ones first),
    as tests/prefill-decode_copy_deck.py reads it."""
    out = []
    card = "[...document.querySelectorAll('[data-exits]')].find(e=>e.offsetParent!==null)"
    for t in range(3):
        ev(pg, f"()=>{card}.querySelector('[data-ex=\"tab\"][data-i=\"{t}\"]').click()"); pg.wait_for_timeout(150)
        tab = ev(pg, f"()=>{card}.querySelector('[data-ex=\"tab\"][data-i=\"{t}\"]').textContent").strip()
        if t == 2:
            out += [f'## {tab}', ev(pg, f"()=>({card}.querySelector('.ex-body, .ex-links')||{card}).innerText").strip()]
            continue
        nq = ev(pg, f"()=>{card}.querySelectorAll('.ex-qs .pb-dot').length||1")
        for j in range(nq):
            if j:
                ev(pg, f"()=>{card}.querySelector('.ex-qs .pb-dot[data-j=\"{j}\"]').click()"); pg.wait_for_timeout(150)
            q = ev(pg, f"()=>{{const e={card}.querySelector('.ex-body > p > b'); return e?e.innerText:''}}").strip()
            opts = ev(pg, f"()=>[...{card}.querySelectorAll('.ex-opts .choice')].map(b=>({{v:+b.dataset.v, t:b.innerText}}))")
            out += [f'## {tab}, question {j + 1}', q] + [f'- ({o["v"]}) {o["t"].strip()}' for o in opts]
            for o in opts:
                ev(pg, f"()=>{card}.querySelector('.ex-opts .choice[data-v=\"{o['v']}\"]').click()"); pg.wait_for_timeout(200)
                r = ev(pg, f"()=>{{const e={card}.querySelector('.ex-res'); return e?{{ok:e.classList.contains('ok'),t:e.innerText}}:null}}")
                if not r: continue
                out += [f'### After picking ({o["v"]})', r['t'].strip()]
                if r['ok']: break
    return out

with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ['zh', 'en']:
        ctx = b.new_context(viewport={'width': 1440, 'height': 1000}); ctx.add_init_script(SEED)
        pg = ctx.new_page(); pg.goto(URL); pg.wait_for_timeout(300)
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        L, seen = [], set()
        # ---------- home: hero (reference) and the KV tile ----------
        section(L, '# Home hero (for reference; left alone unless a string plainly breaks a rule)', txt(pg, '#top'))
        ev(pg, "()=>document.querySelector('#mapSvg .world[data-world=\"inference\"]').dispatchEvent(new MouseEvent('click',{bubbles:true}))"); pg.wait_for_timeout(300)
        ev(pg, "()=>document.querySelector('#tile-kv-cache').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
        section(L, '# Home tile (Inference domain)', txt(pg, '#tile-kv-cache'))
        section(L, '## Assumptions & sources', ev(pg, "()=>[...document.querySelectorAll('#tile-kv-cache .tz-note')].map(n=>n.innerText.trim()).join('\\n\\n')"))
        L.append('## Tooltips in view'); L.extend(tips(pg, '#tile-kv-cache', seen))
        # the slider's own range (2K to 32K in 2K steps), every position in turn, then back to where it started (16K)
        lo, hi, st, v0 = ev(pg, "()=>{const e=document.querySelector('#tile-kv-cache input[type=range]'); return [+e.min,+e.max,+e.step,+e.value]}")
        setv = "v=>{const e=document.querySelector('#tile-kv-cache input[type=range]'); e.value=v; e.dispatchEvent(new Event('input',{bubbles:true}))}"
        for i, v in enumerate(range(lo, hi + 1, st)):
            ev(pg, setv, v); pg.wait_for_timeout(120)
            section(L, f'## Tile, slider position {i + 1}', txt(pg, '#tile-kv-cache .tz-lab') + '\n' + txt(pg, '#tile-kv-cache .tz-read') +
                    ('\n[legend shows: ' + txt(pg, '#tile-kv-cache .tz-leg').replace('\n', ' | ') + ']'))
        ev(pg, setv, v0)
        # ---------- lab header ----------
        ev(pg, "()=>{location.hash='kv-cache'}"); pg.wait_for_timeout(600)
        section(L, '# Lab header', txt(pg, '#view-lab .labhead'))
        L.append('## Tooltips in view'); L.extend(tips(pg, '#view-lab .labhead', seen))
        # ---------- guided parts ----------
        for k in range(7):
            L.append(f'# Guided part {k + 1}')
            section(L, '## Card', txt(pg, '#rail .rail-top') + '\n' + txt(pg, '#rail .step.current'))
            section(L, '## Stage', txt(pg, '#stage'))
            L.append('## Tooltips in view'); L.extend(tips(pg, '#view-lab', seen))
            acted = False
            if pg.query_selector('#rail [data-pred]') and not pg.query_selector('#rail [data-pred][aria-pressed="true"]'):
                click(pg, '#rail [data-pred="m16"]', 250); acted = True
            elif pg.query_selector('#rail [data-showme]') and pg.query_selector('#rail [data-showme]').is_visible():
                click(pg, '#rail [data-showme]', 350); acted = True
            if acted:
                section(L, '## Card after the action (answer / Show me)', txt(pg, '#rail .step.current'))
                section(L, '## Stage after the action', txt(pg, '#stage'))
                L.append('## Tooltips in view after the action'); L.extend(tips(pg, '#view-lab', seen))
            if pg.query_selector('#rail [data-lever]'):
                for i in range(1, len(pg.query_selector_all('#rail [data-lever]'))):
                    click(pg, f'#rail [data-lever="{i}"]', 300)
                    section(L, f'## Option {i + 1} picked: card result', txt(pg, '#rail [data-obs]'))
                    section(L, f'## Option {i + 1} picked: stage', txt(pg, '#stage'))
                    L.append('## Tooltips in view'); L.extend(tips(pg, '#view-lab', seen))
                section(L, '## Options with their results', txt(pg, '#rail .levers'))
                ev(pg, "()=>{const i=document.getElementById('ctx'); i.value=10; i.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(250)
                section(L, '## After changing context in Settings by hand (no option matches)', txt(pg, '#rail [data-obs]'))
                click(pg, '#rail [data-lever="0"]', 300)
                section(L, '## Initial setup picked again', txt(pg, '#rail [data-obs]'))
            if k == 3:
                ev(pg, "()=>document.getElementById('math').open=true"); pg.wait_for_timeout(150)
                section(L, '# Show the math (open)', txt(pg, '#math'))
                L.append('## Term tooltips'); L.extend(tips(pg, '#mathBody', seen))
            nb = pg.query_selector('#rail [data-nav="next"]')
            if nb and not nb.is_disabled(): nb.click(); pg.wait_for_timeout(450)
        # ---------- phone: the mini result bar ----------
        pc = b.new_context(viewport={'width': 390, 'height': 844}); pc.add_init_script(SEED)
        ph = pc.new_page(); ph.goto(URL); ph.wait_for_timeout(300); ph.click(f'.langsw button[data-lang="{lang}"]'); ph.wait_for_timeout(150)
        ev(ph, "()=>{location.hash='kv-cache'}"); ph.wait_for_timeout(500)
        ev(ph, "()=>document.querySelector('#rail [data-go=\"2\"]').click()"); ph.wait_for_timeout(300)
        ev(ph, "()=>document.getElementById('i-ctx').scrollIntoView({block:'center'})"); ph.wait_for_timeout(300)
        ph.focus('#i-ctx'); ph.keyboard.press('ArrowRight'); ph.wait_for_timeout(300)
        section(L, '# Phone (390 px): mini result bar while dragging the slider in part 3', txt(ph, '#dock'))
        ph.keyboard.press('ArrowRight'); ph.wait_for_timeout(300)
        section(L, '## One more notch', txt(ph, '#dock'))
        pc.close()
        # ---------- scenarios ----------
        ev(pg, "()=>window.scrollTo(0,0)"); pg.click('#view-lab .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(400)
        section(L, '# Scenarios', txt(pg, '#modeDesc') + '\n\n' + txt(pg, '#rail'))
        L.append('## Tooltips in view'); L.extend(tips(pg, '#view-lab', seen))
        for i in range(len(pg.query_selector_all('#rail [data-preset]'))):
            click(pg, f'#rail [data-preset="{i}"]', 250)
            section(L, f'## Scenario {i + 1} picked: stage', txt(pg, '#stage'))
            section(L, f'## Scenario {i + 1} picked: notes in Settings', ' | '.join(x for x in [txt(pg, '#modelNote'), txt(pg, '#attnNote'), txt(pg, '#ctxWarn') if pg.is_visible('#ctxWarn') else ''] if x))
            L.append('## Tooltips in view'); L.extend(tips(pg, '#view-lab #stage', seen))
        click(pg, '#rail [data-preset="0"]', 250); click(pg, '#pinBtn', 200); click(pg, '#rail [data-preset="1"]', 250)
        section(L, '## Chat service pinned as A, then Long-document QA picked: the bars', txt(pg, '#stageGraph'))
        L.append('## Tooltips in view'); L.extend(tips(pg, '#stageGraph', seen))
        click(pg, '#unpinBtn', 150)
        click(pg, '#rail [data-preset="0"]', 250)
        ev(pg, "()=>{const i=document.getElementById('ctx'); i.value=12; i.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(200)
        section(L, '## Context beyond the model’s window (256K): stage note and warning', txt(pg, '#stNote') + '\n' + txt(pg, '#ctxWarn'))
        section(L, '## Rail after a hand change (no scenario matches)', txt(pg, '#rail'))
        click(pg, '#rail [data-preset="0"]', 250)
        # ---------- settings panel ----------
        ev(pg, "()=>{document.getElementById('adv').open=true}"); pg.wait_for_timeout(150)
        section(L, '# Settings panel', txt(pg, '#controls'))
        pg.click('#view-lab .seg-ctl[data-key="attn"] button[data-v="mha"]'); pg.wait_for_timeout(150)
        section(L, '## Attention set to “No GQA”: its note', txt(pg, '#attnNote'))
        click(pg, '#rail [data-preset="0"]', 250)
        # ---------- popovers ----------
        # since 2026-10-11 the notes close the page (no Fidelity or Primer button)
        pg.locator('#fidPop').scroll_into_view_if_needed(); pg.wait_for_timeout(200)
        section(L, '# Notes at the page end', txt(pg, '#fidPop'))
        L.append('## Tooltips in view'); L.extend(tips(pg, '#fidPop', seen))
        ev(pg, "()=>window.scrollTo(0,0)"); pg.wait_for_timeout(100)
        # ---------- challenge ----------
        ev(pg, "()=>window.scrollTo(0,0)"); pg.click('#view-lab .modes button[data-mode="challenge"]'); pg.wait_for_timeout(400)
        section(L, '# Challenge', txt(pg, '#modeDesc') + '\n\n' + txt(pg, '#rail'))
        section(L, '## Stage in this mode', txt(pg, '#stage'))
        section(L, '## Settings panel in this mode', txt(pg, '#controls'))
        L.append('## Tooltips in view (fixed settings)'); L.extend(tips(pg, '#view-lab', seen))
        ev(pg, "()=>document.getElementById('math').open=true"); pg.wait_for_timeout(150)
        L.append('## Term tooltips in this mode'); L.extend(tips(pg, '#mathBody', seen))
        click(pg, '#chSubmit', 300)
        section(L, '## Submitted the initial setup (wrong)', txt(pg, '#chRes'))
        click(pg, '#rail [data-ch-reveal]', 300)
        section(L, '## After “Show the answer”', txt(pg, '#chRes'))
        click(pg, '#rail [data-bind-seg="wp"] button[data-v="fp8"]', 150); click(pg, '#rail [data-bind-seg="kp"] button[data-v="fp8"]', 300)
        section(L, '## Weights FP8 + KV FP8 (right, not minimal)', txt(pg, '#chRes'))
        click(pg, '#rail [data-bind-seg="wp"] button[data-v="bf16"]', 300)
        section(L, '## Weights BF16 + KV FP8 (right, minimal)', txt(pg, '#chRes'))
        L.append('## Tooltips in view'); L.extend(tips(pg, '#rail', seen))
        # ---------- can you use it on a new case? ----------
        L.append('# Can you use it on a new case?'); L.extend(exits(pg))
        # ---------- connect ----------
        click(pg, '#drillReveal', 250)
        section(L, '# Connect (one drill, answer shown)', txt(pg, '#view-lab .connect'))
        # the page keeps model names and "8 × H100" on one line with U+2011 and U+00A0; the deck shows plain ones, so it greps and diffs
        (OUT / name(lang)).write_text(('\n\n'.join(x for x in L if x is not None) + '\n').replace('\u2011', '-').replace('\u00a0', ' '))
        ctx.close()
    b.close()
print('wrote', [str(OUT / name(l)) for l in ['zh', 'en']])
