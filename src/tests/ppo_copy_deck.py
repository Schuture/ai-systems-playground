"""Dump every piece of text a learner meets in the PPO lab and its home tile, in reading order, in Chinese and English.

python3 tests/ppo_copy_deck.py [tag]  ->  out/ppo/copy/<tag>-zh.md and out/ppo/copy/<tag>-en.md (default tag: now)

Order: home tile (both settings, legend, notes), lab header with the mode descriptions and both popovers, each guided part (card
before and after its action, the stage, the tooltips in view that were not shown before), "Show the math", the scenarios card for
every preset, the settings panel, the challenge before and after submitting (every feedback tier and the chart), the capability
layer, and the connect section. A tooltip is printed the first time it appears with a given text.
"""
import pathlib, sys, subprocess
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
tag = sys.argv[1] if len(sys.argv) > 1 else 'now'
OUT = P / 'out' / 'ppo' / 'copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/ppo.html', '--only', 'ppo'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'ppo' / 'wrapped-ppo.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'ppo.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'ppo' / 'wrapped-ppo.html')

def txt(pg, sel):
    el = pg.query_selector(sel)
    return el.inner_text().strip() if el and el.is_visible() else ''

def stage(pg):
    parts = [txt(pg, '#ppo-st-title'), txt(pg, '#ppo-status'), txt(pg, '#ppo-clip'), txt(pg, '#ppo-loop'), txt(pg, '#ppo-stage .ppo-legend'), txt(pg, '#ppo-bars'), txt(pg, '#ppo-kinds'),
             txt(pg, '#ppo-same'), txt(pg, '#ppo-run'), txt(pg, '#ppo-curve-title'), txt(pg, '#ppo-key'), txt(pg, '#ppo-mem')]
    mets = pg.evaluate("()=>[...document.querySelectorAll('#ppo-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    return '\n'.join([p for p in parts if p] + ['Metrics:'] + [f'- {m}' for m in mets])

class Tips:
    def __init__(self): self.seen = set()
    def grab(self, pg, root):
        out = []
        for e in pg.query_selector_all(f'{root} [data-tip], {root} [data-tipx], {root} .def'):
            try:
                if not e.is_visible(): continue
                label = ' '.join((e.inner_text() or e.get_attribute('aria-label') or '').split())[:50]
                t = ' '.join((e.get_attribute('data-tipx') or e.get_attribute('data-tip') or '').split())
            except Exception:
                continue
            if t and t not in self.seen:
                self.seen.add(t); out.append(f'- {label} ⟶ {t}')
        return out

def card(pg): return txt(pg, '#ppo-rail .step.current') or txt(pg, '#ppo-rail')

with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ('zh', 'en'):
        pg = b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce'); T = Tips(); L = []
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        if pg.query_selector('#tile-ppo'):
            pg.evaluate("()=>document.querySelector('#tile-ppo').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
            L += ['# Home tile', txt(pg, '#tile-ppo')]
            for v in ('0', '1', '2'):
                pg.evaluate(f"()=>{{const b=document.querySelector('#tile-ppo .seg-ctl[data-key=\"tzset\"] button[data-v=\"{v}\"]'); if(b) b.click()}}"); pg.wait_for_timeout(120)
                L += [f'## Tile sentence, setting {v}', txt(pg, '#tile-ppo .tz-read')]
            L += ['## Tile notes', pg.evaluate("()=>[...document.querySelectorAll('#tile-ppo .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')")]
        else:
            # the lab-only build has no home page: render the teaser through the read-only hook instead
            tz = pg.evaluate("""()=>{const el=document.createElement('div'); el.id='ppo-tz-copy'; document.body.appendChild(el); window.__ppo.teaser(el); const out={top:el.querySelector('.seg-ctl').innerText.replace(/\\s+/g,' ').trim(), leg:el.querySelector('.tz-leg').innerText.replace(/\\s+/g,' ').trim(), reads:[]};
              for(const v of ['0','1','2']){el.querySelector(`.seg-ctl[data-key="tzset"] button[data-v="${v}"]`).click(); out.reads.push(el.querySelector('.tz-read').textContent.trim());}
              out.notes=[...el.querySelectorAll('.tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()); el.remove(); return out;}""")
            L += ['# Home tile (rendered through the hook; the lab-only build has no home page)', tz['top'], tz['leg']]
            for v, r in enumerate(tz['reads']): L += [f'## Tile sentence, setting {v}', r]
            L += ['## Tile notes', '\n\n'.join(tz['notes'])]
        pg.evaluate("()=>{location.hash='ppo'}"); pg.wait_for_timeout(600)
        L += ['# Header', txt(pg, '#ppo-app .lab-head') or txt(pg, '#ppo-app header')]
        L += ['## Mode descriptions'] + [f'- {x}' for x in pg.evaluate("()=>[...document.querySelectorAll('#ppo-app .modes [data-mode]')].map(b=>b.innerText+': '+(b.dataset.tipx||''))")]
        L += ['## Popovers'] + [' '.join(x.split()) for x in pg.evaluate("()=>[...document.querySelectorAll('#ppo-app .pop, #ppo-app [data-pop], #ppo-app .popover')].map(e=>e.textContent)") if x.strip()]
        for i in range(8):
            pg.evaluate(f"()=>document.querySelector('#ppo-rail [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(300)
            L += [f'# Part {i + 1}', '## Card', card(pg)]
            acts = {1: ["#ppo-rail [data-nav='showme']"], 2: ["#ppo-rail [data-pred='best']"], 3: ["#ppo-rail [data-lever='1']"],
                    4: ["#ppo-rail [data-lever='0.05']", "#ppo-rail [data-lever='0.2']"], 5: ["#ppo-rail [data-lever='4-0']", "#ppo-rail [data-lever='4-1']"],
                    6: ["#ppo-rail [data-lever='grpo']", "#ppo-rail [data-lever='dpo']", "#ppo-rail [data-lever='ppo']"]}.get(i, [])
            for a in acts:
                pg.evaluate(f"()=>document.querySelector(\"{a}\").click()"); pg.wait_for_timeout(300)
                L += [f'## Card after {a.split("[")[-1].rstrip("]")}', card(pg)]
            L += ['## Stage', stage(pg), '## Tooltips (new)'] + T.grab(pg, '#ppo-app')
            if i == 5:   # part 6's picture plays step 1's batch frame by frame (the owner's message of 2026-10-08, item 3): every frame's text
                for lv in ('1-0', '4-0', '4-1'):
                    pg.evaluate(f"()=>document.querySelector(\"#ppo-rail [data-lever='{lv}']\").click()"); pg.wait_for_timeout(300)
                    L += [f'## Option {lv}: its result', txt(pg, f"#ppo-rail [data-lever='{lv}'] .lv-r")]
                    for v in [str(q) for q in range(1, (1 if lv == '1-0' else 4) + 1)] + ['0']:
                        pg.evaluate(f"()=>document.querySelector('#ppo-pass button[data-v=\"{v}\"]').click()"); pg.wait_for_timeout(200)
                        L += [f'### Picture, option {lv}, frame {"after" if v == "0" else "pass " + v}', txt(pg, '#ppo-clip')] + T.grab(pg, '#ppo-clip')
        pg.evaluate("()=>{document.querySelector('#ppo-math').open=true}"); pg.wait_for_timeout(200)
        L += ['# Show the math', txt(pg, '#ppo-mathbody')]
        pg.evaluate("()=>document.querySelector('#ppo-rail [data-mode-go=\"sandbox\"]').click()"); pg.wait_for_timeout(400)
        L += ['# Scenarios', txt(pg, '#ppo-rail')]
        for i in range(4):
            pg.evaluate(f"()=>document.querySelector('#ppo-rail [data-preset=\"{i}\"]').click()"); pg.wait_for_timeout(300)
            L += [f'## Scenario {i + 1}', txt(pg, '#ppo-insight'), '### Stage', stage(pg)]
        L += ['# Settings', txt(pg, '#ppo-controls')]
        pg.evaluate("()=>document.querySelector('#ppo-app .modes [data-mode=\"challenge\"]').click()"); pg.wait_for_timeout(400)
        L += ['# Challenge', txt(pg, '#ppo-rail'), '## Stage', stage(pg)]
        for sel, name in [('[data-act="submit"]', 'submitted at the start'), ('#ppo-ic-beta button[data-v="0.2"]', 'beta 0.2'), ('#ppo-ic-beta button[data-v="0.05"]', 'beta 0.05'),
                          ('[data-act="reveal"]', 'all options'), ('#ppo-ic-clip button[data-v="1"]', 'beta 0.05 + clipping'), ('#ppo-ic-K button[data-v="1"]', 'beta 0.05 + clipping + 1 pass')]:
            pg.evaluate(f"()=>{{const b=document.querySelector('#ppo-rail {sel}'); if(b) b.click()}}"); pg.wait_for_timeout(300)
            L += [f'## Challenge feedback: {name}', txt(pg, '#ppo-ch-res')]
        L += ['# Capability layer', txt(pg, '#ppo-rail [data-exits]')]
        L += ['# Connect', txt(pg, '#ppo-app .connect')]
        (OUT / f'{tag}-{lang}.md').write_text('\n\n'.join(x for x in L if x) + '\n')
        print('wrote', OUT / f'{tag}-{lang}.md')
        pg.close()
    b.close()
