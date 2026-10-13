"""Dump every piece of text a learner meets in the sft lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/sft_copy_deck.py  ->  out/sft/copy-en.md and out/sft/copy-zh.md

Order: home tile (the sentence for each choice, the notes), lab header and its popovers, each guided part (card before and after its
action, the stage, the tooltips in view not printed before), Show the math, the scenarios (each insight), the settings panel, the
challenge (each feedback tier and the chart), the card "Can you use it on a new case?", Connect with the drill answer.
A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'sft'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/sft.html', '--only', 'sft'], cwd=P, check=True, capture_output=True)
URL = 'file://' + str(P / 'out' / 'sft.html')
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


def ttips(pg, root):
    """formula terms (.t): their tooltip is HTML from the lab's termTip; focus shows it"""
    out = []
    for i in range(pg.evaluate("r=>[...document.querySelectorAll(r+' .t[data-ctl]')].filter(e=>e.getClientRects().length).length", root)):
        t = pg.evaluate("([r,i])=>{const e=[...document.querySelectorAll(r+' .t[data-ctl]')].filter(e=>e.getClientRects().length)[i]; e.focus({preventScroll:true}); const tp=document.getElementById('tip'); const s=tp&&!tp.hidden?[e.innerText,tp.innerText]:null; e.blur(); return s}", [root, i])
        if t and clean(t[1]) not in seen:
            seen.add(clean(t[1])); out.append(f'- hover “{clean(t[0])}” → {clean(t[1])}')
    return out


def end(pg):
    pg.evaluate("()=>{const i=document.querySelector('#sft-player input[data-key=\"scrub\"]'); if(!i) return; i.value=99; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(250)


def click(pg, sel):
    pg.evaluate(f"()=>{{const e=[...document.querySelectorAll('{sel}')].find(x=>x.offsetParent!==null)||document.querySelector('{sel}'); if(e) e.click();}}"); pg.wait_for_timeout(300)


def stage(pg):
    return ['**Stage**'] + lines(txt(pg, '#sft-stage')) + tips(pg, '#sft-stage')


def rail(pg, title):
    return [f'**{title}**'] + lines(txt(pg, '#sft-rail')) + tips(pg, '#sft-rail') + ttips(pg, '#sft-rail')


def deck(b, lang):
    global seen
    seen = set()
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
    out = ['# sft copy deck (' + lang + ')', '', '## Home tile']
    tile = '[data-teaser="sft"]'
    pg.evaluate("()=>{const b=[...document.querySelectorAll('[data-world=\"post\"]')].find(e=>e.getClientRects().length); if(b) b.dispatchEvent(new MouseEvent('click',{bubbles:true}));}"); pg.wait_for_timeout(400)
    if pg.evaluate(f"()=>!!document.querySelector('{tile}')"):
        for v in ('all', 'resp', 'eot'):
            click(pg, f'{tile} button[data-v="{v}"]')
            out += lines(pg.evaluate(f"()=>document.querySelector('{tile} .tz-read').innerText"))
        out += [clean(x) for x in pg.evaluate(f"()=>[...document.querySelectorAll('{tile} .tz-note')].map(e=>e.textContent)")]
    else:
        out += ['(tile not on the home page in this build)']
    pg.evaluate("()=>{location.hash='sft'}"); pg.wait_for_timeout(600)
    out += ['', '## Header'] + lines(txt(pg, '.lab-head') or txt(pg, '#sft-app header')) + tips(pg, '#sft-app .lab-head')
    for sel in ('.fid-tag', '[data-tip="fid"]', '.lab-head [data-tipfn]'):
        pass
    pops = pg.evaluate("()=>[...document.querySelectorAll('#sft-app [data-pop], #sft-app .pop')].map(e=>e.innerText)")
    out += [clean(x) for x in pops if clean(x)]
    names = ['Your mission', 'A conversation becomes tokens', 'Predict first', 'Count the answer only', 'The end token must count', 'From the SFT loss to DPO', 'What to remember']
    for i, nm in enumerate(names):
        out += ['', f'## Part {i + 1}: {nm}']
        out += rail(pg, 'Card') + stage(pg)
        if i == 2:
            click(pg, '#sft-rail [data-pred="10to50"]'); out += rail(pg, 'Card after the prediction') + stage(pg)
        if i in (0, 1):
            end(pg); out += ['**Stage at the end of the run**'] + lines(txt(pg, '#sft-genbox')) + tips(pg, '#sft-stage')
        if i in (3, 4):
            end(pg); out += rail(pg, 'Card after the run') + ['**Stage at the end of the run**'] + lines(txt(pg, '#sft-genbox')) + tips(pg, '#sft-stage')
            click(pg, '#sft-rail [data-nav="showme"]'); end(pg)
            out += rail(pg, 'Card after Show me') + stage(pg)
        if i == 2:
            pg.evaluate("()=>{document.querySelector('#sft-math').open=true}"); pg.wait_for_timeout(150)
            out += ['**Show the math**'] + lines(txt(pg, '#sft-mathbody'))
        if i < len(names) - 1:
            click(pg, '#sft-rail [data-nav="next"]')
    pg.evaluate("()=>{document.querySelector('#sft-math').open=true}"); pg.wait_for_timeout(150)
    out += ['', '## Show the math (part 7)'] + lines(txt(pg, '#sft-mathbody'))
    pg.evaluate("()=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"sandbox\"]:not([data-flow=free])')].find(e=>e.offsetParent!==null); b.click();}"); pg.wait_for_timeout(400)
    out += ['', '## Scenarios'] + rail(pg, 'Card')
    for i in range(4):
        click(pg, f'#sft-rail [data-preset="{i}"]'); end(pg)
        out += [f'**Scenario {i + 1}**'] + lines(txt(pg, '#sft-insight')) + stage(pg)
    out += ['', '## Settings'] + lines(txt(pg, '#sft-controls')) + tips(pg, '#sft-controls')
    pg.evaluate("()=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"challenge\"]')].find(e=>e.offsetParent!==null); b.click();}"); pg.wait_for_timeout(400)
    out += ['', '## Challenge'] + rail(pg, 'Card') + stage(pg)
    click(pg, '#sft-rail [data-act="submit"]'); out += rail(pg, 'Submitted as given')
    click(pg, '#sft-rail .seg-ctl[data-key="count"] button[data-v="resp"]'); out += ['**One fix**'] + lines(txt(pg, '#sft-ch-res'))
    click(pg, '#sft-rail [data-act="reveal"]'); out += ['**Show the answer**'] + lines(txt(pg, '#sft-ch-res'))
    click(pg, '#sft-rail .seg-ctl[data-key="pad"] button[data-v="sep"]'); out += ['**Both fixes**'] + lines(txt(pg, '#sft-ch-res'))
    ex = '#sft-rail [data-exits]'
    out += ['', '## Can you use it on a new case?'] + lines(txt(pg, ex)) + tips(pg, ex)
    for t in pg.evaluate(f"()=>[...document.querySelectorAll('{ex} [data-ex=\"tab\"]')].map((b,i)=>i)"):
        pg.evaluate(f"(i)=>{{const b=[...document.querySelectorAll('{ex} [data-ex=\"tab\"]')][i]; b.click();}}", t); pg.wait_for_timeout(200)
        nq = pg.evaluate(f"()=>Math.max(1,document.querySelectorAll('{ex} .ex-qs [data-ex=\"q\"]').length)")
        for j in range(nq if t < 2 else 1):
            pg.evaluate(f"(j)=>{{const d=document.querySelector('{ex} .ex-qs [data-ex=\"q\"][data-j=\"'+j+'\"]'); if(d) d.click();}}", j); pg.wait_for_timeout(150)
            out += [f'**Tab {t + 1}, question {j + 1}**'] + lines(txt(pg, ex))
            if t == 2: break
            # answer the question shown with its first option (often wrong: the note and the why), then the right one
            pg.evaluate(f"()=>{{const o=document.querySelector('{ex} [data-ex=\"ans\"]'); if(o) o.click();}}"); pg.wait_for_timeout(150)
            out += ['(first option picked)'] + lines(txt(pg, '{ex} .ex-res'.format(ex=ex))) + tips(pg, ex)
    click(pg, '#sft-app [data-act="drill-reveal"]')
    out += ['', '## Connect'] + lines(txt(pg, '#sft-app .connect')) + tips(pg, '#sft-app .connect')
    pg.close()
    return out


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ('en', 'zh'):
        out = deck(b, lang)
        (OUT / f'copy-{lang}.md').write_text('\n'.join(out) + '\n')
        print('wrote', OUT / f'copy-{lang}.md', len(out), 'lines')
    b.close()
