"""Set A teasers: operate every control the way a user does and check what the tile says back.

At 1440x900 and 390x844, in EN and ZH:
  - every slider position by keyboard (Home, then ArrowRight to the end) and one mouse click on each track;
    every segmented button by mouse; after each interaction the sentence (.tz-read) must change,
    flash (unless reduced motion), and the control must show its selected state;
  - no page errors, no console errors except the sandbox's font ERR_TUNNEL, no horizontal overflow,
    nothing inside a tile sticking out of it, visuals at most 130 px tall, timeline labels not overlapping;
  - each .tz-read is one or two sentences of at most 35 English words (guide R11.5) after every interaction; no .tz-note
    is visible on the tile; every tile has a .tz-leg legend; no "Assumptions & sources" button on the tile (owner 2026-10-13:
    the tile is one button); the hidden notes still follow the state (prompt length);
  - switching the language and reselecting the Inference domain keep every setting.
Also: reduced motion (no flash), a forced 280 px tile width, dark mode screenshots.
Screenshots go to shots/set-a/.   Run from the prototype folder:  python3 tests/set-a_flow.py"""
import pathlib, subprocess, re, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run(['python3', 'build.py', 'out/set-a.html', '--only', 'set-a'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'wrapped-set-a.html').write_text(
    '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>'
    + (P / 'out' / 'set-a.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-set-a.html')
SHOTS = P / 'shots' / 'set-a'; SHOTS.mkdir(parents=True, exist_ok=True)

IDS = ['prefill-decode', 'chunked-prefill', 'prefix-caching', 'quantization', 'spec-decoding']
SLIDERS = {'prefill-decode': (0, 6), 'quantization': (0, 600), 'spec-decoding': (50, 95)}
SEGS = {'chunked-prefill': ['8192', '2048', '512', '0'], 'prefix-caching': ['1', '10', '1000', '100']}
FAILS, CHECKS = [], [0]
def check(cond, what):
    CHECKS[0] += 1
    if not cond: FAILS.append(what)

READ = "(id)=>document.querySelector('#tile-'+id+' .tz-read').innerText"
FLASHED = "(id)=>document.querySelector('#tile-'+id+' .tz-read').classList.contains('tz-seta-flash')"
LAYOUT = """()=>{const res={docOverflow:document.documentElement.scrollWidth>window.innerWidth, tiles:{}};
  const ov=(a,b)=>{const r=a.getBoundingClientRect(), s=b.getBoundingClientRect(); return !(r.right<=s.left||s.right<=r.left||r.bottom<=s.top||s.bottom<=r.top);};
  for(const id of %s){const t=document.getElementById('tile-'+id), tz=t.querySelector('.tile-tz'), v=t.querySelector('.tz-vis'), tr=tz.getBoundingClientRect();
    const o={w:t.getBoundingClientRect().width, visH:v.getBoundingClientRect().height, overflow:tz.scrollWidth>tz.clientWidth+1,
      out:[...tz.querySelectorAll('*')].filter(e=>{const r=e.getBoundingClientRect(); return r.width>0&&(r.left<tr.left-1||r.right>tr.right+1);}).length};
    if(id==='chunked-prefill') o.overlap=ov(t.querySelector('.tz-chunk-arr'),t.querySelector('.tz-chunk-ft'));
    if(id==='prefill-decode') o.clipped=[...t.querySelectorAll('.tz-pfd-lab span')].some(e=>e.scrollWidth>e.parentElement.clientWidth+1);
    if(id==='quantization') o.clipped=[...t.querySelectorAll('.tz-quant-val')].some(e=>e.scrollWidth>e.clientWidth+1);
    res.tiles[id]=o;}
  return res;}""" % IDS
STATE = """()=>({pd:__tzA.pd.state(), cp:__tzA.cp.state(), pc:__tzA.pc.state(), qz:__tzA.qz.state(), sd:__tzA.sd.state(),
  inputs:[...document.querySelectorAll('#tiles [data-teaser] input[type=range]')].map(i=>i.closest('.tile').id+'='+i.value),
  pressed:[...document.querySelectorAll('#tiles [data-teaser] .seg-ctl button[aria-pressed="true"]')].map(b=>b.closest('.tile').id+'='+b.dataset.v)})"""
BOLD_NUMS = "(id)=>[...document.querySelectorAll('#tile-'+id+' .tz-read b')].map(b=>b.textContent.replace(/[^0-9.,≈]/g,''))"

def layout_ok(pg, tag, max_h=130):
    L = pg.evaluate(LAYOUT)
    check(not L['docOverflow'], f'{tag}: page scrolls horizontally')
    for id_, o in L['tiles'].items():
        check(not o['overflow'] and o['out'] == 0, f'{tag} {id_}: content wider than the tile ({o["out"]} elements stick out)')
        check(o['visH'] <= max_h, f'{tag} {id_}: visual {o["visH"]:.0f}px tall')
        check(not o.get('overlap'), f'{tag} {id_}: timeline labels overlap')
        check(not o.get('clipped'), f'{tag} {id_}: a label is clipped')
    return L

def interact(pg, tag, reduced=False):
    """Operate every control of the five tiles; the sentence must change each time."""
    for id_, (lo, hi) in SLIDERS.items():
        sel = f'#tile-{id_} input[type=range]'
        pg.focus(sel); pg.keyboard.press('Home'); prev = pg.evaluate(READ, id_)
        check(pg.eval_on_selector(sel, 'e=>+e.value') == lo, f'{tag} {id_}: Home -> min')
        for v in range(lo, hi, (5 if id_ == 'spec-decoding' else 1)):
            pg.keyboard.press('ArrowRight')
            cur = pg.evaluate(READ, id_)
            check(cur != prev, f'{tag} {id_}: sentence unchanged after ArrowRight from {v}')
            if tag.endswith('_en'): check(len(cur.split()) <= 35, f'{tag} {id_}: {len(cur.split())} words: {cur}')
            if v % 97 == 0 or id_ != 'quantization':
                check(pg.evaluate(FLASHED, id_) != reduced, f'{tag} {id_}: flash after a keyboard step should be {not reduced}')
            prev = cur
        check(pg.eval_on_selector(sel, 'e=>+e.value') == hi, f'{tag} {id_}: ArrowRight reaches max')
        # one pointer interaction: click near the left end of the track
        box = pg.locator(sel).bounding_box(); before = pg.evaluate(READ, id_)
        pg.mouse.click(box['x'] + 3, box['y'] + box['height'] / 2)
        check(pg.evaluate(READ, id_) != before, f'{tag} {id_}: sentence unchanged after a click on the track')
        check(pg.eval_on_selector(sel, 'e=>+e.value') == lo, f'{tag} {id_}: click at the left end -> min')
        check(pg.eval_on_selector(sel, 'e=>e.getAttribute("aria-valuetext")') not in (None, ''), f'{tag} {id_}: aria-valuetext')
    for id_, order in SEGS.items():
        prev = pg.evaluate(READ, id_)
        for v in order:
            b = f'#tile-{id_} .seg-ctl button[data-v="{v}"]'
            pg.click(b); cur = pg.evaluate(READ, id_)
            check(cur != prev, f'{tag} {id_}: sentence unchanged after clicking {v}')
            if tag.endswith('_en'): check(len(cur.split()) <= 35, f'{tag} {id_}: {len(cur.split())} words: {cur}')
            check(pg.evaluate(FLASHED, id_) != reduced, f'{tag} {id_}: flash after click should be {not reduced}')
            pressed = pg.eval_on_selector_all(f'#tile-{id_} .seg-ctl button', 'bs=>bs.filter(b=>b.getAttribute("aria-pressed")==="true").map(b=>b.dataset.v)')
            check(pressed == [v], f'{tag} {id_}: selected state {pressed} after clicking {v}')
            bg = pg.eval_on_selector(b, 'e=>getComputedStyle(e).backgroundColor'); bg2 = pg.eval_on_selector(f'#tile-{id_} .seg-ctl button[aria-pressed="false"]', 'e=>getComputedStyle(e).backgroundColor')
            check(bg != bg2, f'{tag} {id_}: the pressed button looks different from the others')
            prev = cur

NOTES = "(id)=>[...document.querySelectorAll('#tile-'+id+' .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).filter(Boolean)"
TIP = "()=>{const t=document.getElementById('tip'); return t.hidden?null:[...t.querySelectorAll('p')].map(p=>p.textContent.replace(/\\s+/g,' ').trim());}"

def notes_hidden(pg, tag):
    for id_ in IDS:
        vis = pg.eval_on_selector_all(f'#tile-{id_} .tz-note', 'ns=>ns.filter(n=>getComputedStyle(n).display!=="none"&&n.getBoundingClientRect().height>0).length')
        check(vis == 0, f'{tag} {id_}: {vis} .tz-note visible on the tile')
        check(pg.locator(f'#tile-{id_} .tz-note').count() >= 3, f'{tag} {id_}: notes exist')
        check(pg.locator(f'#tile-{id_} .tz-leg').count() == 1 and pg.eval_on_selector(f'#tile-{id_} .tz-leg', 'e=>e.getBoundingClientRect().height>0'), f'{tag} {id_}: legend shown')
        check(pg.locator(f'#tile-{id_} .tz-seta-info').count() == 0 and pg.locator(f'#tile-{id_} .tz-vis[data-tipx]').count() == 0, f'{tag} {id_}: no own tooltip links left')
        check(pg.locator(f'#tile-{id_} .tz-more').count() == 0, f'{tag} {id_}: no Assumptions & sources button on the tile (owner 2026-10-13: the tile is one button)')

def tooltips(pg, tag):
    # owner 2026-10-13: the tile is one button; no “Assumptions & sources” button or hover on it
    for id_ in IDS:
        check(pg.evaluate("(id)=>!/假设和来源|Assumptions (&|and) sources/.test(document.getElementById('tile-'+id).innerText)", id_), f'{tag} {id_}: no “Assumptions & sources” text on the tile')
    # the notes follow the state: change the prompt length (the notes name it)
    was = pg.eval_on_selector('#tile-prefill-decode input[type=range]', 'e=>+e.value')
    before = pg.evaluate(NOTES, 'prefill-decode')
    pg.focus('#tile-prefill-decode input[type=range]'); pg.keyboard.press('End' if was < 6 else 'Home')
    pg.evaluate('()=>document.activeElement.blur()'); pg.mouse.move(0, 0)
    check(pg.evaluate(NOTES, 'prefill-decode') != before, f'{tag}: the notes follow the prompt length')
    pg.evaluate("(v)=>{const i=document.querySelector('#tile-prefill-decode input[type=range]'); i.value=v; i.dispatchEvent(new Event('input',{bubbles:true}));}", was)
    # the chunked-prefill notes are the same four sentences at every chunk size (SITE-A-r7 item 12)
    before = pg.evaluate(NOTES, 'chunked-prefill')
    pg.click('#tile-chunked-prefill .seg-ctl button[data-v="512"]')
    check(pg.evaluate(NOTES, 'chunked-prefill') == before and len(before) == 4, f'{tag}: the chunked-prefill notes should not change with the chunk size')
    pg.click('#tile-chunked-prefill .seg-ctl button[data-v="0"]')

def keeps_state(pg, tag):
    """Non-default settings survive a language switch and reselecting the domain."""
    pg.evaluate("""()=>{const set=(id,v)=>{const i=document.querySelector('#tile-'+id+' input[type=range]'); i.value=v; i.dispatchEvent(new Event('input',{bubbles:true}));};
      set('prefill-decode',6); set('quantization',137); set('spec-decoding',65);
      document.querySelector('#tile-chunked-prefill button[data-v="2048"]').click(); document.querySelector('#tile-prefix-caching button[data-v="10"]').click();}""")
    s0 = pg.evaluate(STATE); nums0 = {i: pg.evaluate(BOLD_NUMS, i) for i in IDS}
    cur = pg.evaluate('()=>document.documentElement.lang')
    other = 'en' if cur.startswith('zh') else 'zh'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(150)
    s1 = pg.evaluate(STATE); nums1 = {i: pg.evaluate(BOLD_NUMS, i) for i in IDS}
    check(s0 == s1, f'{tag}: language switch lost state {s0} -> {s1}')
    for i in IDS:
        check(nums0[i] == nums1[i], f'{tag} {i}: numbers differ across languages {nums0[i]} vs {nums1[i]}')
    pg.click(f'.langsw button[data-lang="{"zh" if other == "en" else "en"}"]'); pg.wait_for_timeout(150)
    # choose another domain in the domain strip above the tiles (it is there at every width), then Inference again
    pg.click('#domStrip .ds-btn[data-world="training"]'); pg.wait_for_timeout(200)
    check(pg.locator('#tile-prefill-decode').count() == 0, f'{tag}: training domain shows other tiles')
    pg.click('#domStrip .ds-btn[data-world="inference"]'); pg.wait_for_timeout(250)
    s2 = pg.evaluate(STATE)
    check(s0 == s2, f'{tag}: reselecting the domain lost state {s0} -> {s2}')
    # back to the defaults for the screenshots
    pg.evaluate("""()=>{const set=(id,v)=>{const i=document.querySelector('#tile-'+id+' input[type=range]'); i.value=v; i.dispatchEvent(new Event('input',{bubbles:true}));};
      set('prefill-decode',3); set('quantization',270); set('spec-decoding',80);
      document.querySelector('#tile-chunked-prefill button[data-v="0"]').click(); document.querySelector('#tile-prefix-caching button[data-v="100"]').click();}""")

def shots(pg, tag):
    pg.evaluate("()=>document.getElementById('tilesHead').scrollIntoView({block:'start'})"); pg.wait_for_timeout(1300)
    pg.locator('#tiles').screenshot(path=str(SHOTS / f'tiles_{tag}.png'))
    for id_ in IDS:
        pg.locator(f'#tile-{id_}').screenshot(path=str(SHOTS / f'{id_}_{tag}.png'))

def console_filter(msgs):
    return [m for m in msgs if 'ERR_TUNNEL' not in m and 'fonts.googleapis' not in m and 'fonts.gstatic' not in m]

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    for (w, h) in [(1440, 900), (390, 844)]:
        for lang in ['en', 'zh']:
            tag = f'{w}_{lang}'
            ctx = browser.new_context(viewport={'width': w, 'height': h}, device_scale_factor=2 if w < 800 else 1)
            pg = ctx.new_page(); errors, cons = [], []
            pg.on('pageerror', lambda e: errors.append(str(e)))
            pg.on('console', lambda m: cons.append(m.text) if m.type == 'error' else None)
            pg.goto(URL + '#map'); pg.wait_for_timeout(300)
            if lang == 'zh': pg.click('.langsw button[data-lang="zh"]'); pg.wait_for_timeout(200)
            check(all(pg.locator(f'#tile-{i} .tz-read').count() == 1 for i in IDS), f'{tag}: all five teasers rendered')
            layout_ok(pg, tag + ' start')
            notes_hidden(pg, tag)
            shots(pg, tag)
            interact(pg, tag)
            layout_ok(pg, tag + ' after')
            tooltips(pg, tag)
            keeps_state(pg, tag)
            check(not errors, f'{tag}: page errors {errors}')
            check(not console_filter(cons), f'{tag}: console errors {console_filter(cons)[:3]}')
            ctx.close()
    # reduced motion: no flash, same content
    ctx = browser.new_context(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce'); pg = ctx.new_page()
    pg.goto(URL + '#map'); pg.wait_for_timeout(300)
    for id_, order in SEGS.items():
        pg.click(f'#tile-{id_} .seg-ctl button[data-v="{order[0]}"]')
        check(not pg.evaluate(FLASHED, id_), f'reduced motion: {id_} must not flash')
    pg.focus('#tile-quantization input[type=range]'); pg.keyboard.press('ArrowRight')
    check(not pg.evaluate(FLASHED, 'quantization'), 'reduced motion: quantization must not flash')
    ctx.close()
    # the narrowest tile the contract allows: 280 px, every position, both languages
    ctx = browser.new_context(viewport={'width': 1440, 'height': 900}); pg = ctx.new_page()
    pg.goto(URL + '#map'); pg.wait_for_timeout(300)
    pg.add_style_tag(content='#tiles{grid-template-columns:repeat(auto-fill,280px)!important}')
    for lang in ['en', 'zh']:
        if lang == 'zh': pg.click('.langsw button[data-lang="zh"]'); pg.wait_for_timeout(150)
        for k in range(7):
            pg.evaluate("""(k)=>{const set=(id,v)=>{const i=document.querySelector('#tile-'+id+' input[type=range]'); i.value=v; i.dispatchEvent(new Event('input',{bubbles:true}));};
              set('prefill-decode',k); set('quantization',[0,1,20,250,270,501,600][k]); set('spec-decoding',[50,55,65,75,80,90,95][k]);
              document.querySelector('#tile-chunked-prefill button[data-v="'+['0','8192','2048','512','0','8192','512'][k]+'"]').click();
              document.querySelector('#tile-prefix-caching button[data-v="'+['1','10','100','1000','1','10','100'][k]+'"]').click();}""", k)
            L = layout_ok(pg, f'280px {lang} #{k}'); notes_hidden(pg, f'280px {lang} #{k}')
            check(all(abs(o['w'] - 280) < 1 for o in L['tiles'].values()), '280px: tiles are 280 px wide')
        pg.evaluate("""()=>{const set=(id,v)=>{const i=document.querySelector('#tile-'+id+' input[type=range]'); i.value=v; i.dispatchEvent(new Event('input',{bubbles:true}));};
          set('prefill-decode',3); set('quantization',270); set('spec-decoding',80);
          document.querySelector('#tile-chunked-prefill button[data-v="0"]').click(); document.querySelector('#tile-prefix-caching button[data-v="100"]').click();}""")
        shots(pg, f'280_{lang}')
    ctx.close()
    # dark mode
    for (w, h, lang) in [(1440, 900, 'en'), (1440, 900, 'zh'), (390, 844, 'zh')]:
        ctx = browser.new_context(viewport={'width': w, 'height': h}, color_scheme='dark', device_scale_factor=2 if w < 800 else 1); pg = ctx.new_page()
        pg.goto(URL + '#map'); pg.wait_for_timeout(300)
        if lang == 'zh': pg.click('.langsw button[data-lang="zh"]'); pg.wait_for_timeout(150)
        check(pg.evaluate("()=>getComputedStyle(document.body).backgroundColor") == 'rgb(10, 15, 21)', 'dark mode is active')
        layout_ok(pg, f'dark {w} {lang}'); notes_hidden(pg, f'dark {w} {lang}')
        shots(pg, f'{w}_{lang}_dark')
        ctx.close()
    browser.close()

print(f'{CHECKS[0]} checks, {len(FAILS)} failed; screenshots in {SHOTS.relative_to(P)}')
for f in FAILS[:40]: print('FAIL', f)
sys.exit(1 if FAILS else 0)
