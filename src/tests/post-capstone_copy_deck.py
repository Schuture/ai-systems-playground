"""Dump every piece of text a learner meets in the post-capstone lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/post-capstone_copy_deck.py  ->  out/post-capstone/copy-en.md and out/post-capstone/copy-zh.md

Order: home tile (the sentence for every choice, the notes, the legend), lab header and its popovers, each guided part (card before
and after its actions, every option's result, the stage, the tooltips in view not printed before, Show the math), the scenarios,
the settings panel, the challenge (every feedback tier and the chart), the card "Can you use it on a new case?" (each tab), and
Connect with the drill answer. A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'post-capstone'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/post-capstone.html', '--only', 'post-capstone'], cwd=P, check=True, capture_output=True)
(OUT / 'wrapped-deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'post-capstone.html').read_text() + '</body></html>')
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
    out = [clean(txt(pg, '#pcap-st-title')), clean(txt(pg, '#pcap-st-sub')), 'Status: ' + clean(txt(pg, '#pcap-status'))]
    labels = pg.evaluate("()=>[...document.querySelectorAll('#pcap-graph .pcap-bl')].filter(e=>e.getClientRects().length).map(e=>e.textContent.trim())")
    if labels: out.append('Blocks: ' + ' | '.join(labels))
    for sel in ('#pcap-phone', '#pcap-card', '#pcap-strip', '#pcap-mem-cap', '#pcap-mem-list', '#pcap-cost-cap', '#pcap-cost', '#pcap-cost-list', '#pcap-cost-hw', '#pcap-met-cap', '#pcap-met .pcap-zcap', '#pcap-met-list', '#pcap-tags'):
        t = txt(pg, sel)
        if t: out.append(clean(t))
    out.append('Metrics: ' + ' || '.join(clean(m) for m in pg.evaluate("()=>[...document.querySelectorAll('#pcap-metrics .metric')].map(m=>m.innerText)")))
    return [x for x in out if x]


def click(pg, sel, wait=300):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(wait)


def show_math(pg, D, head='## Show the math'):
    pg.evaluate("()=>{document.getElementById('pcap-math').open=true}"); pg.wait_for_timeout(100)
    D.append(head); D += lines(txt(pg, '#pcap-mathbody'))


ACTIONS = {0: [('pcap-ic-lv1', 'pref'), ('pcap-ic-lv1', 'math')],
           1: [('seg', 'pcap-ic-card|pref'), ('seg', 'pcap-ic-meth|sft'), ('seg', 'pcap-ic-meth|grpo'), ('seg', 'pcap-ic-card|math'), ('seg', 'pcap-ic-meth|sft')],
           2: [('pred', '1to3'), ('pred', 'gt10')], 3: [('pcap-ic-lv4', '8lora'), ('pcap-ic-lv4', '70lora'), ('pcap-ic-lv4', '70qlora')],
           4: [('pcap-ic-lv5', '0.5'), ('pcap-ic-lv5', '2'), ('pcap-ic-lv5', '3'), ('pcap-ic-lv5', '4')], 5: [('pcap-ic-lv6', 'ppo'), ('pcap-ic-lv6', 'grpo')]}


def deck(b, lang):
    D = []
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
    seen.clear()
    D.append('# Home tile')
    pg.evaluate("()=>{const b=document.querySelector('[data-world=\"post\"]'); if(b) b.click();}"); pg.wait_for_timeout(400)
    D += [f'- {clean(x)}' for x in pg.evaluate("()=>{const t=document.querySelector('.tz-pcap'); return t?[...t.querySelectorAll('.seg-ctl,.tz-leg,.tz-note')].map(e=>e.textContent):[]}") if clean(x)]
    for v in ('demo', 'pairs', 'checker'):
        pg.evaluate(f"()=>{{const b=document.querySelector('.tz-pcap .seg-ctl button[data-v=\"{v}\"]'); if(b) b.click();}}"); pg.wait_for_timeout(80)
        D.append(f'- [{v}] {clean(txt(pg, ".tz-pcap .tz-read"))}')
        D.append(f'  rows: {clean(txt(pg, ".tz-pcap .tz-pcap-rows"))}')
        for tz_tip in pg.evaluate("()=>[...document.querySelectorAll('.tz-pcap .tz-pcap-row')].map(r=>r.getAttribute('data-tipx'))"):
            D.append(f'  row tooltip: {clean(tz_tip)}')
    pg.evaluate("()=>{location.hash='post-capstone'}"); pg.wait_for_timeout(500)
    D.append('\n# Lab header')
    D += lines(txt(pg, '#pcap-app .labhead'))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#pcap-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        D.append(f'\n## Popover {pid}')
        D += lines(pg.evaluate(f"()=>document.getElementById('{pid}').innerText||document.getElementById('{pid}').textContent"))
    D += tips(pg, '#pcap-app .labhead')
    for i in range(7):
        sid = pg.evaluate("()=>window.__pcap.state().step")
        D.append(f'\n# Guided part {sid + 1}: card')
        D += lines(txt(pg, '#pcap-rail .step.current') or txt(pg, '#pcap-rail'))
        D.append('## Stage'); D += stage(pg)
        if i == 0:
            D.append('## Settings'); D += lines(txt(pg, '#pcap-controls'))
        D += tips(pg, '#pcap-app')
        for box, v in ACTIONS.get(i, []):
            if box == 'pred': click(pg, f'#pcap-rail [data-pred="{v}"]')
            elif box == 'seg':
                sid_, val = v.split('|'); click(pg, f'#{sid_} button[data-v="{val}"]')
            else: click(pg, f'#{box} [data-lever="{v}"]')
            D.append(f'\n## Part {sid + 1} after picking “{v}”')
            D += lines(txt(pg, '#pcap-rail [data-obs]'))
            D.append('### Stage'); D += stage(pg)
            D += tips(pg, '#pcap-app')
        if i in (0, 3, 4, 5):
            D.append('\n## Options with their results'); D += lines(txt(pg, '#pcap-rail .pcap-lv'))
        show_math(pg, D)
        D += tips(pg, '#pcap-app')
        if i < 6: click(pg, '#pcap-rail [data-nav="next"]', 500)
    D.append('\n# Scenarios')
    click(pg, '#pcap-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 500)
    D += lines(txt(pg, '#pcap-rail'))
    for i in range(5):
        click(pg, f'#pcap-rail [data-preset="{i}"]', 400)
        D.append(f'\n## Scenario {i + 1}'); D += lines(txt(pg, '#pcap-insight')); D.append('### Stage'); D += stage(pg)
        show_math(pg, D, '### Show the math')
        D += tips(pg, '#pcap-app')
    D.append('\n# Settings panel (Scenarios)'); D += lines(txt(pg, '#pcap-controls'))
    D += tips(pg, '#pcap-app')
    D.append('\n# Challenge')
    click(pg, '#pcap-app .modes button[data-mode="challenge"]', 500)
    D += lines(txt(pg, '#pcap-rail .railcard'))
    D.append('## Stage'); D += stage(pg)
    click(pg, '#pcap-rail [data-act="submit"]')
    D.append('\n## Tier 3: the start, SFT'); D += lines(txt(pg, '#pcap-ch-res'))
    D += tips(pg, '#pcap-app')

    def pick(meth, mode_, x):
        for k, v in (('meth', meth), ('mode', mode_), ('x', x)):
            click(pg, f'#pcap-ic-ch{k} button[data-v="{v}"]', 150)
    click(pg, '#pcap-ch-res [data-act="reveal"]')
    pick('dpo', 'full', '1')
    D.append('\n## Not yet: DPO, full, 1 pass, after “Show the answer” (it cannot run; its row stays in the chart, rule G8)'); D += lines(txt(pg, '#pcap-ch-res'))
    pick('grpo', 'full', '2')
    D.append('\n## Tier 3: GRPO, full, 2 passes'); D += lines(txt(pg, '#pcap-ch-res'))
    pick('grpo', 'lora', '3')
    D.append('\n## Tier 2: GRPO, LoRA, 3 passes'); D += lines(txt(pg, '#pcap-ch-res'))
    pick('grpo', 'qlora', '2')
    D.append('\n## Tier 1: GRPO, QLoRA, 2 passes'); D += lines(txt(pg, '#pcap-ch-res'))
    D.append('## Stage'); D += stage(pg)
    D += tips(pg, '#pcap-app')
    D.append('\n# Can you use it on a new case?')
    tabs = pg.evaluate("()=>document.querySelectorAll('#pcap-rail [data-exits] .ex-tabs button').length")
    D += lines(txt(pg, '#pcap-rail [data-exits]'))
    for t in range(tabs):
        pg.evaluate(f"()=>document.querySelectorAll('#pcap-rail [data-exits] .ex-tabs button')[{t}].click()"); pg.wait_for_timeout(200)
        D.append(f'## Tab {t + 1}'); D += lines(txt(pg, '#pcap-rail [data-exits]'))
        for q in range(3):
            nx = pg.evaluate("()=>{const b=[...document.querySelectorAll('#pcap-rail [data-exits] button')].find(b=>/Next question|下一题/.test(b.textContent)&&b.getClientRects().length); if(b){b.click(); return true} return false}")
            if not nx: break
            pg.wait_for_timeout(200); D += lines(txt(pg, '#pcap-rail [data-exits]'))
    D.append('\n# Connect')
    click(pg, '#pcap-app [data-act="drill-reveal"]')
    D += lines(txt(pg, '#pcap-app .connect'))
    D += tips(pg, '#pcap-app')
    pg.close()
    (OUT / f'copy-{lang}.md').write_text('\n'.join(D) + '\n')
    return len(D)


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ('en', 'zh'):
        print(lang, deck(b, lang), 'lines ->', OUT / f'copy-{lang}.md')
    b.close()
