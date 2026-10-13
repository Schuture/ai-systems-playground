"""Set C teasers, interaction flow: 1440x900 and 390x844 in EN and ZH, light and dark, plus 1280/900/720 and a 280 px tile.
Operates every control position with real keyboard presses and clicks and checks, at every position:
  the sentence (.tz-read) changes, is one or two sentences of at most 35 English words (guide R11.5), and the Chinese
  one takes no more lines than the English one at the same width and setting,
  the visual is <= 130 px tall, the one-line legend (.tz-leg) is there, no .tz-note is visible on the tile,
  no horizontal overflow, no page errors.
Also: no "Assumptions & sources" button on the tile (owner 2026-10-13); tooltips on the visuals;
state kept across domain and language switches; reduced motion; the screen-reader announcement; dark-mode contrast.
Screenshots go to shots/set-c/.

    python3 tests/set-c_flow.py"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/set-c.html', '--only', 'set-c'], check=True, cwd=P, capture_output=True)
WR = P / 'out' / 'wrapped-set-c.html'
WR.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
              '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'set-c.html').read_text() + '</body></html>')
URL = 'file://' + str(WR)
SHOTS = P / 'shots' / 'set-c'; SHOTS.mkdir(parents=True, exist_ok=True)
for old in SHOTS.glob('*.png'): old.unlink()

TZ = {'rag-failure': ('apps', 'range', 4), 'agent-reliability': ('agents', 'range', 20), 'agent-anatomy': ('agents', 'range', 8),
      'prompt-injection': ('agents', 'seg', 'paste'), 'eval-ci': ('eval', 'seg', '500'), 'pass-k': ('eval', 'range', 8)}
TIPWORD = {'rag-failure': 'Top row', 'agent-reliability': 'one curve per', 'agent-anatomy': 'stacked by source',
           'prompt-injection': 'What the agent is given', 'eval-ci': 'One row per benchmark size', 'pass-k': 'Two curves'}
fails, n_ok, maxw = [], 0, {'en': 0, 'zh': 0}
EN_LINES = {}   # (width, lab, setting) -> lines the English sentence took, to compare the Chinese one with
def check(name, cond, info=''):
    global n_ok
    if cond: n_ok += 1
    else: fails.append(f'{name} [{info}]')

def new_page(b, w, h, lang, scheme='light', motion='no-preference'):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme=scheme, reduced_motion=motion)
    pg._errs = []
    pg.on('pageerror', lambda e: pg._errs.append(str(e)))
    pg.on('console', lambda m: pg._errs.append('console: ' + m.text) if m.type == 'error' and 'ERR_TUNNEL' not in m.text and 'fonts.g' not in m.text else None)
    pg.goto(URL + '#map'); pg.wait_for_timeout(350)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(120)
    return pg
def open_world(pg, world):   # the domain strip above the tiles exists at every width
    pg.click(f'#domStrip .ds-btn[data-world="{world}"]'); pg.wait_for_timeout(260)
def read(pg, lab): return pg.inner_text(f'#tile-{lab} .tz-read')
def sentence_ok(text, lang):
    """guide R11.5: one or two sentences, about 35 English words at most"""
    t = text.strip()
    if lang == 'zh':
        return t.endswith('。') and t.count('。') in (1, 2), len(t)
    words = t.split()
    few = len(re.split(r'(?<=[.!?])\s+', t)) <= 2 and t.endswith('.')
    return few and len(words) <= 35, len(words)
def lines(pg, lab):
    return pg.evaluate(f"()=>{{const r=document.querySelector('#tile-{lab} .tz-read'), cs=getComputedStyle(r); return Math.round(r.getBoundingClientRect().height/parseFloat(cs.lineHeight));}}")
def vis_h(pg, lab): return pg.evaluate(f"()=>document.querySelector('#tile-{lab} .tz-vis').getBoundingClientRect().height")
def tile_state(pg, lab):
    return pg.evaluate(f"""()=>{{const t=document.querySelector('#tile-{lab}'), leg=t.querySelector('.tz-leg');
      return {{notesShown:[...t.querySelectorAll('.tz-note')].filter(n=>n.getClientRects().length>0).length,
        notes:t.querySelectorAll('.tz-note').length, leg:!!leg&&leg.getClientRects().length>0&&[...leg.children].some(s=>!s.hidden),
        more:!!t.querySelector('.tz-more')&&!t.querySelector('.tz-more').hidden}};}}""")
def overflow(pg):
    return pg.evaluate("""()=>{const d=document.documentElement, bad=[];
      document.querySelectorAll('.tile').forEach(t=>{if(t.scrollWidth>t.clientWidth+1) bad.push(t.id);
        const tz=t.querySelector('.tile-tz'), r=tz.getBoundingClientRect();
        t.querySelectorAll('.tz > *, .tz-leg > span').forEach(c=>{const q=c.getBoundingClientRect(); if(q.width&&(q.right>r.right+1||q.left<r.left-1)) bad.push(t.id+' '+c.className);});});
      return {page:d.scrollWidth-d.clientWidth, bad};}""")
def page_ok(o, tag): return o['page'] <= 0 and not o['bad']
def shot_tiles(pg, name):
    pg.evaluate("()=>{const h=document.getElementById('tilesHead'); scrollTo(0, h.getBoundingClientRect().top+scrollY-120); if(document.activeElement) document.activeElement.blur();}")
    pg.mouse.move(pg.viewport_size['width'] - 2, pg.viewport_size['height'] - 2); pg.wait_for_timeout(1000)   # no stray tooltip, blink finished
    y0, y1 = pg.evaluate("()=>{const a=document.getElementById('tilesHead').getBoundingClientRect(), b=document.getElementById('tiles').getBoundingClientRect(); return [a.top+scrollY, b.bottom+scrollY]}")
    w = pg.viewport_size['width']
    pg.screenshot(path=str(SHOTS / f'{name}.png'), full_page=True, clip={'x': 0, 'y': max(0, y0 - 8), 'width': w, 'height': y1 - y0 + 16})

def after_step(pg, lab, tag, lang, what, prev):
    now = read(pg, lab)
    ok, n = sentence_ok(now, lang); maxw[lang] = max(maxw[lang], n)
    check(f'{tag} {lab} {what}: one or two short sentences', ok, (n, now))
    key = (pg.viewport_size['width'], lab, what, tag.split()[-1].rstrip(']') if tag.count(' ') > 1 else '')
    if lang == 'en': EN_LINES[key] = lines(pg, lab)
    elif key in EN_LINES: check(f'{tag} {lab} {what}: Chinese no longer than English', lines(pg, lab) <= EN_LINES[key], (lines(pg, lab), EN_LINES[key], now))
    check(f'{tag} {lab} {what}: sentence changes', now != prev, now[:60])
    st = tile_state(pg, lab)
    check(f'{tag} {lab} {what}: no note visible, legend present, no notes button (owner 2026-10-13: the tile is one button)', st['notesShown'] == 0 and st['notes'] >= 1 and st['leg'] and not st['more'], st)
    return now, vis_h(pg, lab)

def walk_range(pg, lab, tag, lang):
    sel = f'#tile-{lab} input[type=range]'
    lo, hi, v0 = pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); return [+i.min,+i.max,+i.value]}}")
    pg.focus(sel); pg.keyboard.press('Home'); pg.wait_for_timeout(40)
    prev, prev_out = read(pg, lab), pg.inner_text(f'#tile-{lab} output')
    check(f'{tag} {lab} Home -> min', pg.eval_on_selector(sel, 'e=>+e.value') == lo)
    worst = 0
    for v in range(lo + 1, hi + 1):
        pg.keyboard.press('ArrowRight'); pg.wait_for_timeout(15)
        check(f'{tag} {lab} value {v}', pg.eval_on_selector(sel, 'e=>+e.value') == v)
        out = pg.inner_text(f'#tile-{lab} output'); check(f'{tag} {lab} output changes at {v}', out != prev_out, out); prev_out = out
        prev, h = after_step(pg, lab, tag, lang, f'at {v}', prev); worst = max(worst, h)
        if v % 10 == 0 or v == hi:
            o = overflow(pg); check(f'{tag} {lab} no overflow at {v}', page_ok(o, tag), o)
    check(f'{tag} {lab} visual <= 130 px at every position', worst <= 130, worst)
    for _ in range(hi - v0): pg.keyboard.press('ArrowLeft')     # back to the default setting
    pg.wait_for_timeout(30)
    check(f'{tag} {lab} back at default', pg.eval_on_selector(sel, 'e=>+e.value') == v0)

def walk_seg(pg, lab, tag, lang):
    vals = pg.eval_on_selector_all(f'#tile-{lab} .seg-ctl button', 'bs=>bs.map(b=>b.dataset.v)')
    cur = pg.eval_on_selector(f'#tile-{lab} .seg-ctl button[aria-pressed="true"]', 'b=>b.dataset.v')
    prev, worst = read(pg, lab), 0
    for v in [x for x in vals if x != cur] + [cur]:
        pg.click(f'#tile-{lab} .seg-ctl button[data-v="{v}"]'); pg.wait_for_timeout(40)
        pressed = pg.eval_on_selector_all(f'#tile-{lab} .seg-ctl button[aria-pressed="true"]', 'bs=>bs.map(b=>b.dataset.v)')
        check(f'{tag} {lab} click {v}: only it is pressed', pressed == [v], pressed)
        prev, h = after_step(pg, lab, tag, lang, f'click {v}', prev); worst = max(worst, h)
        o = overflow(pg); check(f'{tag} {lab} no overflow after {v}', page_ok(o, tag), o)
    check(f'{tag} {lab} visual <= 130 px for every option', worst <= 130, worst)

def notes_button(pg, lab, tag):
    """Owner 2026-10-13: the tile is one button; no “Assumptions & sources” button or hover on it."""
    check(f'{tag} {lab} no notes button or text on the tile', pg.locator(f'#tile-{lab} .tz-more').count() == 0
          and not pg.evaluate(f"()=>/假设和来源|Assumptions (&|and) sources/.test(document.getElementById('tile-{lab}').innerText)"))

with sync_playwright() as p:
    b = p.chromium.launch()
    # ---- every control, every position: 1440x900 and 390x844, EN and ZH, light
    for (w, h) in [(1440, 900), (390, 844)]:
        for lang in ['en', 'zh']:
            tag = f'[{w} {lang}]'; pg = new_page(b, w, h, lang)
            for world in ['apps', 'agents', 'eval']:
                open_world(pg, world)
                for lab in [l for l, (wd, _, _) in TZ.items() if wd == world]:
                    check(f'{tag} {lab} tile rendered', pg.query_selector(f'#tile-{lab} .tz-c') is not None)
                    (walk_range if TZ[lab][1] == 'range' else walk_seg)(pg, lab, tag, lang)
                    notes_button(pg, lab, tag)
                shot_tiles(pg, f'{w}_{lang}_{world}')
            check(f'{tag} no page errors', not pg._errs, pg._errs); pg.close()

    # ---- state survives a domain switch and a language switch; the sentence follows the language
    pg = new_page(b, 1440, 900, 'en'); open_world(pg, 'apps')
    pg.focus('#tile-rag-failure input'); [pg.keyboard.press('ArrowRight') for _ in range(3)]
    en7 = read(pg, 'rag-failure'); check('rag moved to 7', pg.eval_on_selector('#tile-rag-failure input', 'e=>+e.value') == 7)
    check('rag k=7: one chunk from the end (TILES-B #9)', 'one chunk from the end' in en7, en7)
    open_world(pg, 'agents'); pg.click('#tile-prompt-injection .seg-ctl button[data-v="isolate"]')
    open_world(pg, 'apps')
    check('rag keeps k=7 after a domain switch', pg.eval_on_selector('#tile-rag-failure input', 'e=>+e.value') == 7 and read(pg, 'rag-failure') == en7)
    pg.click('.langsw button[data-lang="zh"]'); pg.wait_for_timeout(150)
    zh7 = read(pg, 'rag-failure')
    check('rag keeps k=7 after switching to ZH', pg.eval_on_selector('#tile-rag-failure input', 'e=>+e.value') == 7 and 'top-7' in zh7 and '离末尾只差一个' in zh7, zh7)
    open_world(pg, 'agents')
    check('injection keeps its setup after both switches', pg.eval_on_selector('#tile-prompt-injection .seg-ctl button[aria-pressed="true"]', 'b=>b.dataset.v') == 'isolate')
    pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(150); open_world(pg, 'apps')
    check('back in EN, same sentence as before', read(pg, 'rag-failure') == en7)
    # ---- screen-reader announcement after an interaction (the kit debounces it by 500 ms)
    pg.focus('#tile-rag-failure input'); pg.keyboard.press('ArrowLeft'); pg.wait_for_timeout(700)
    check('announces the new sentence', pg.inner_text('#live').strip() == read(pg, 'rag-failure').strip(), pg.inner_text('#live')[:80])
    # ---- tooltips on the visuals (hover and focus), legend items, the struck-out tool
    for lab, (world, _, _) in TZ.items():
        open_world(pg, world); pg.eval_on_selector(f'#tile-{lab} .tz-vis', 'e=>e.scrollIntoView({block:"center"})'); pg.wait_for_timeout(120)
        bb = pg.query_selector(f'#tile-{lab} .tz-vis').bounding_box()
        pg.hover(f'#tile-{lab} .tz-vis', position={'x': bb['width'] * 0.8, 'y': bb['height'] * 0.55}); pg.wait_for_timeout(150)
        tip = pg.inner_text('#tip') if pg.is_visible('#tip') else ''
        check(f'{lab} hover tooltip on the visual', TIPWORD[lab] in tip, tip[:80])
        pg.mouse.move(2, 2); pg.wait_for_timeout(60)
        pg.focus(f'#tile-{lab} .tz-vis'); pg.wait_for_timeout(150)
        tip = pg.inner_text('#tip') if pg.is_visible('#tip') else ''
        check(f'{lab} focus tooltip on the visual', TIPWORD[lab] in tip, tip[:80])
        check(f'{lab} visual has help cursor', pg.eval_on_selector(f'#tile-{lab} .tz-vis', 'e=>getComputedStyle(e).cursor') == 'help')
    open_world(pg, 'agents')
    pg.eval_on_selector('#tile-agent-anatomy .tz-leg', 'e=>e.scrollIntoView({block:"center"})'); pg.wait_for_timeout(150)
    pg.focus('#tile-agent-anatomy .tz-leg [data-tipx]'); pg.wait_for_timeout(150)
    check('anatomy legend item explains itself', 'Tool outputs' in (pg.inner_text('#tip') if pg.is_visible('#tip') else ''))
    pg.evaluate("()=>document.activeElement&&document.activeElement.blur()"); pg.mouse.move(2, 2)   # a focused item's tooltip survives scrolls (core)
    pg.eval_on_selector('#tile-prompt-injection .tz-inj-tool.off', 'e=>e.scrollIntoView({block:"center"})'); pg.wait_for_timeout(150)
    pg.hover('#tile-prompt-injection .tz-inj-tool.off'); pg.wait_for_timeout(150)
    check('struck-out tool explains itself', 'Not available' in (pg.inner_text('#tip') if pg.is_visible('#tip') else ''))
    # ---- a real mouse drag moves a slider
    open_world(pg, 'eval'); pg.eval_on_selector('#tile-pass-k input', 'e=>e.scrollIntoView({block:"center"})'); pg.wait_for_timeout(150)
    box = pg.query_selector('#tile-pass-k input').bounding_box()
    pg.mouse.move(box['x'] + box['width'] - 4, box['y'] + box['height'] / 2); pg.mouse.down()
    pg.mouse.move(box['x'] + 4, box['y'] + box['height'] / 2, steps=8); pg.mouse.up(); pg.wait_for_timeout(80)
    check('mouse drag moves pass-k to k=1', pg.eval_on_selector('#tile-pass-k input', 'e=>+e.value') == 1 and 'pass@1' in read(pg, 'pass-k'))
    # ---- the sentence blinks after a click (not under reduced motion)
    pg.click('#tile-eval-ci .seg-ctl button[data-v="5000"]')
    check('sentence blinks after a click', pg.eval_on_selector('#tile-eval-ci .tz-read', 'e=>e.classList.contains("tz-c-pulse")'))
    check('no page errors (state, tooltips, drag)', not pg._errs, pg._errs); pg.close()
    pg = new_page(b, 1440, 900, 'en', motion='reduce'); open_world(pg, 'eval')
    pg.click('#tile-eval-ci .seg-ctl button[data-v="1000"]')
    check('no blink under reduced motion', not pg.eval_on_selector('#tile-eval-ci .tz-read', 'e=>e.classList.contains("tz-c-pulse")'))
    pg.close()

    # ---- the guide's other widths (R7.1): 1280, 900, 720 in EN, every position
    for w in [1280, 900, 720]:
        pg = new_page(b, w, 900, 'en'); tag = f'[{w} en]'
        for world in ['apps', 'agents', 'eval']:
            open_world(pg, world)
            for lab in [l for l, (wd, _, _) in TZ.items() if wd == world]:
                (walk_range if TZ[lab][1] == 'range' else walk_seg)(pg, lab, tag, 'en')
            shot_tiles(pg, f'{w}_en_{world}')
        check(f'{tag} tiles 280-440 px wide', all(279 <= x <= 441 for x in pg.eval_on_selector_all('.tile', 'ts=>ts.map(t=>t.getBoundingClientRect().width)')))
        check(f'{tag} no page errors', not pg._errs, pg._errs); pg.close()

    # ---- the narrowest tile (280 px at a 312 px window), EN and ZH, light and dark: every position
    for lang, scheme in [('en', 'light'), ('zh', 'light'), ('en', 'dark'), ('zh', 'dark')]:
        pg = new_page(b, 312, 800, lang, scheme=scheme); tag = f'[312 {lang} {scheme}]'
        for world in ['apps', 'agents', 'eval']:
            open_world(pg, world)
            check(f'{tag} {world} tile is 280 px', all(abs(x - 280) <= 1 for x in pg.eval_on_selector_all('.tile', 'ts=>ts.map(t=>t.getBoundingClientRect().width)')))
            for lab in [l for l, (wd, _, _) in TZ.items() if wd == world]:
                (walk_range if TZ[lab][1] == 'range' else walk_seg)(pg, lab, tag, lang)
            shot_tiles(pg, f'280tile_{lang}_{scheme}_{world}')
        check(f'{tag} no page errors', not pg._errs, pg._errs); pg.close()

    # ---- dark mode at 1440 and 390, EN and ZH: screenshots, and text and marks keep contrast against the tile
    for (w, h) in [(1440, 900), (390, 844)]:
        for lang in ['en', 'zh']:
            pg = new_page(b, w, h, lang, scheme='dark')
            for world in ['apps', 'agents', 'eval']:
                open_world(pg, world); shot_tiles(pg, f'{w}_{lang}_dark_{world}')
                low = pg.evaluate("""()=>{const lum=c=>{const m=c.match(/[\\d.]+/g).map(Number).slice(0,3).map(v=>{v/=255;return v<=0.03928?v/12.92:Math.pow((v+0.055)/1.055,2.4)});return 0.2126*m[0]+0.7152*m[1]+0.0722*m[2]};
                  const ratio=(a,b)=>{const x=lum(a),y=lum(b);return (Math.max(x,y)+0.05)/(Math.min(x,y)+0.05)}; const out=[];
                  document.querySelectorAll('.tile').forEach(t=>{const bg=getComputedStyle(t).backgroundColor;
                    const ink=getComputedStyle(document.body).color;   // the answer cell is filled with --ink (35% when it was missed); its digit sits on that
                    const mix=(a,b,w)=>{const x=a.match(/[\\d.]+/g).map(Number), y=b.match(/[\\d.]+/g).map(Number); return 'rgb('+[0,1,2].map(i=>Math.round(x[i]*w+y[i]*(1-w))).join(',')+')';};
                    t.querySelectorAll('.tz-c-svg text, .tz-read, .tz-leg, .tz-inj-vis, .tz-more').forEach(e=>{const cs=getComputedStyle(e), c=e.tagName==='text'?cs.fill:cs.color;
                      const under=e.matches('.tz-rag-n.ans.miss')?mix(ink,bg,0.35):(e.matches('.tz-rag-n.ans')?ink:bg);
                      if(c.startsWith('rgb')&&ratio(c,under)<3) out.push(t.id+' '+(e.getAttribute('class')||e.tagName)+' '+ratio(c,under).toFixed(2));});});
                  return out;}""")
                check(f'[dark {w} {lang}] {world} text contrast >= 3:1', not low, low[:5])
            check(f'[dark {w} {lang}] no page errors', not pg._errs, pg._errs); pg.close()
    b.close()

print(f'{n_ok} checks passed, {len(fails)} failed; longest sentence: {maxw["en"]} words (EN), {maxw["zh"]} characters (ZH); {len(EN_LINES)} EN/ZH line comparisons; screenshots in {SHOTS.relative_to(P)}')
for f_ in fails[:40]: print('FAIL', f_)
sys.exit(1 if fails else 0)
