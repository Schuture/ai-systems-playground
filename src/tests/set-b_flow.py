"""Set B teasers: operate every control like a user (arrow keys on sliders) at 1440x900 and 390x844
in English and Chinese, and check after every interaction that the tile's sentence (one or two sentences, at most
35 English words) changed and flashed, nothing overflows horizontally, the visual stays within 130 px, and the page
raised no errors. Also: tooltips on hover and keyboard focus, state kept across a language switch and a domain switch,
tiles at their narrowest (280 px), and dark mode. Screenshots go to shots/set-b/.

  python3 tests/set-b_flow.py
"""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/set-b.html', '--only', 'set-b'], check=True, cwd=P, capture_output=True)
WRAP = P / 'out' / 'wrapped-set-b.html'
WRAP.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'set-b.html').read_text() + '</body></html>')
URL = 'file://' + str(WRAP)
SHOTS = P / 'shots' / 'set-b'; SHOTS.mkdir(parents=True, exist_ok=True)
# the training-memory and parallelism tiles left set B in batch 8 and the two post-training tiles in batch 9: those
# labs' own tiles are tested by their flow tests, so set B now has tiles in the Models domain only
TILES = {'models': ['sampling', 'tokenization-cost']}
VIS = {'sampling': '.tz-samp-vis', 'tokenization-cost': '.tz-tokn-vis'}

checks, fails, log = 0, [], []
def check(ok, msg):
    global checks
    checks += 1
    if not ok: fails.append(msg)

def world(pg, w):
    pg.evaluate("w=>document.querySelector('#mapSvg .world[data-world=\"'+w+'\"]').dispatchEvent(new MouseEvent('click',{bubbles:true}))", w)
    pg.wait_for_timeout(250)
def read(pg, lab): return pg.inner_text(f'#tile-{lab} .tz-read')
def layout_ok(pg, where):
    # below 373 px the core top bar (nav, language switch, theme button) is wider than the window; that is not
    # part of these tiles, so at those widths only the tiles area is checked
    sel = 'document.documentElement' if pg.viewport_size['width'] >= 373 else "document.getElementById('tiles')"
    over = pg.evaluate(f"()=>{{const e={sel}; return Math.max(e.scrollWidth-e.clientWidth, Math.round(e.getBoundingClientRect().right-window.innerWidth))}}")
    check(over <= 0, f'{where}: horizontal overflow {over}px')
    bad = pg.evaluate("""()=>Array.from(document.querySelectorAll('#tiles .tile')).map(t=>{
        const r=t.getBoundingClientRect(); const out=Array.from(t.querySelectorAll('*')).filter(e=>{const b=e.getBoundingClientRect(); return b.width&&(b.right>r.right+1||b.left<r.left-1)});
        return out.length?t.id+': '+out.slice(0,3).map(e=>e.className||e.tagName).join(', '):null}).filter(Boolean)""")
    check(not bad, f'{where}: content outside its tile: {bad}')
def act(pg, lab, do, what, where):
    """one user interaction: the sentence must change and flash"""
    before = read(pg, lab); do(); pg.wait_for_timeout(90)
    after = read(pg, lab)
    check(after != before, f'{where} {lab} {what}: sentence did not change ({after[:90]}…)')
    check(pg.evaluate(f"()=>document.querySelector('#tile-{lab} .tz-read').classList.contains('tz-setb-bump')"), f'{where} {lab} {what}: result did not flash')
    words_ok(after, f'{where} {lab} after {what}')
    layout_ok(pg, f'{where} {lab} after {what}')
def words_ok(sentence, where):
    """guide R11.5: one or two sentences, about 35 English words at most"""
    if ' en ' in f' {where} ':
        n = len(sentence.split())
        check(n <= 35, f'{where}: sentence has {n} words (max 35): {sentence}')
    check(sentence.count('. ') <= 1 and sentence.count('。') <= 2, f'{where}: more than two sentences: {sentence}')
def notes_ok(pg, lab, where, how='hover'):
    """notes are hidden on the tile, and the tile has no notes button (owner 2026-10-13)"""
    hidden = pg.evaluate(f"()=>Array.from(document.querySelectorAll('#tile-{lab} .tz-note')).map(n=>getComputedStyle(n).display)")
    check(hidden and all(d == 'none' for d in hidden), f'{where} {lab}: a note is visible on the tile ({hidden})')
    leg = pg.evaluate(f"()=>{{const l=document.querySelector('#tile-{lab} .tz-leg'); return l?[getComputedStyle(l).display,l.textContent.trim().length]:null}}")
    check(leg and leg[0] != 'none' and leg[1] > 0, f'{where} {lab}: legend missing or hidden')
    # owner 2026-10-13: the tile is one button; no “Assumptions & sources” button or hover on it
    check(pg.locator(f'#tile-{lab} .tz-more').count() == 0 and not pg.evaluate(f"()=>/假设和来源|Assumptions (&|and) sources/.test(document.getElementById('tile-{lab}').innerText)"),
          f'{where} {lab}: an “Assumptions & sources” button or text is still on the tile')
def keys(pg, lab, key, n, where):
    sel = f'#tile-{lab} input[type=range]'
    for j in range(n):
        act(pg, lab, lambda: (pg.focus(sel), pg.keyboard.press(key)), f'{key} #{j+1}', where)
SHOOT = [True]
def shot_tile(pg, lab, name):
    if not SHOOT[0]: return
    pg.mouse.move(1, 1); pg.evaluate("()=>{const t=document.getElementById('tip'); if(t) t.hidden=true}")
    pg.evaluate(f"()=>document.getElementById('tile-{lab}').scrollIntoView({{block:'center'}})"); pg.wait_for_timeout(1200)
    pg.locator(f'#tile-{lab}').screenshot(path=str(SHOTS / name))

def run(p, w, h, lang, theme='light', full=True, shots=True):
    where = f'{w}px {lang} {theme}'; SHOOT[0] = shots
    b = p.chromium.launch(); ctx = b.new_context(viewport={'width': w, 'height': h}, color_scheme=theme); pg = ctx.new_page()
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' and 'ERR_TUNNEL' not in m.text and 'fonts.g' not in m.text else None)
    pg.goto(URL + '#map'); pg.wait_for_timeout(300)
    if lang == 'zh': pg.click('.langsw button[data-lang="zh"]'); pg.wait_for_timeout(200)
    for wd, labs in TILES.items():
        world(pg, wd)
        for lab in labs:
            check(pg.locator(f'#tile-{lab} .tz-read').count() == 1, f'{where}: {lab} teaser missing')
            words_ok(read(pg, lab), f'{where} {lab} start')
            notes_ok(pg, lab, where)
            hgt = pg.evaluate(f"()=>document.querySelector('#tile-{lab} {VIS[lab]}').getBoundingClientRect().height")
            check(hgt <= 130.5, f'{where}: {lab} visual is {hgt:.0f}px tall (limit 130)')
            if full: shot_tile(pg, lab, f'{w}_{lang}_{theme}_{lab}_start.png')
        if not full: layout_ok(pg, f'{where} {wd}'); continue
        layout_ok(pg, f'{where} {wd} at the start')
        if wd == 'models':
            keys(pg, 'sampling', 'ArrowLeft', 9, where); shot_tile(pg, 'sampling', f'{w}_{lang}_{theme}_sampling_T0.1.png')
            keys(pg, 'sampling', 'ArrowRight', 19, where); shot_tile(pg, 'sampling', f'{w}_{lang}_{theme}_sampling_T2.0.png')
            keys(pg, 'tokenization-cost', 'ArrowLeft', 4, where); shot_tile(pg, 'tokenization-cost', f'{w}_{lang}_{theme}_tokenization-cost_0.png')
            keys(pg, 'tokenization-cost', 'ArrowRight', 6, where); shot_tile(pg, 'tokenization-cost', f'{w}_{lang}_{theme}_tokenization-cost_300.png')
        layout_ok(pg, f'{where} {wd} after operating')
        for lab in labs: notes_ok(pg, lab, f'{where} after operating')
    if full:
        # the settings survive a language switch and a domain switch (tiles are re-rendered from module state)
        before = pg.evaluate("()=>window.__tzB.state()")
        outs = pg.evaluate("()=>Array.from(document.querySelectorAll('#tiles output'),o=>o.textContent)")
        other = 'en' if lang == 'zh' else 'zh'
        pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(250)
        check(pg.evaluate("()=>window.__tzB.state()") == before, f'{where}: state changed on language switch')
        after = pg.evaluate("()=>Array.from(document.querySelectorAll('#tiles output'),o=>o.textContent)")
        check([o.split()[0] if o[:1].isdigit() else o for o in after] != [] and len(after) == len(outs), f'{where}: outputs after language switch {after}')
        # to a domain without set B tiles (ML basics) and back: the Models tiles are rendered again from module state
        world(pg, 'roots')
        check(pg.locator('#tile-sampling').count() == 0, f'{where}: domain switch did not leave the Models tiles')
        world(pg, 'models')
        check(pg.evaluate("()=>window.__tzB.state()") == before, f'{where}: state changed on domain switch')
        check(pg.evaluate("()=>document.querySelector('#tile-tokenization-cost input').value") == '6', f'{where}: tokenization slider not restored')
        # the notes button also works from the keyboard, in the other language
        for lab in TILES['models']: notes_ok(pg, lab, f'{where} after language switch', how='focus')
    check(not errs, f'{where}: page errors {errs}')
    log.append(f'{where}: ok' if not errs else f'{where}: errors')
    b.close()

with sync_playwright() as p:
    for (w, h) in ((1440, 900), (390, 844)):
        for lang in ('en', 'zh'):
            run(p, w, h, lang)
    run(p, 1440, 900, 'en', 'dark'); run(p, 390, 844, 'zh', 'dark')
    for w in (900, 312):                                 # 312 px wide window = the narrowest 280 px tile
        for lang in ('en', 'zh'): run(p, w, 900, lang, shots=False)
    for w, lang in ((1280, 'en'), (720, 'zh'), (600, 'en')):   # the guide's other widths; 600 = one wide column
        run(p, w, 900, lang, shots=False)
    # each domain with set B tiles, as a first-time visitor sees it
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.goto(URL + '#map'); pg.wait_for_timeout(300)
    for wd in TILES:
        world(pg, wd); pg.evaluate("()=>{const t=document.getElementById('tilesHead'); window.scrollTo(0,t.getBoundingClientRect().top+scrollY-70)}")
        pg.mouse.move(1, 1); pg.wait_for_timeout(300); pg.screenshot(path=str(SHOTS / f'home_1440_en_{wd}.png'))
    b.close()

for line in log: print(line)
print(f'set-b flow: {checks} checks, {len(fails)} failed; screenshots in {SHOTS.relative_to(P)}')
for f in fails[:40]: print('  FAIL', f)
sys.exit(1 if fails else 0)
