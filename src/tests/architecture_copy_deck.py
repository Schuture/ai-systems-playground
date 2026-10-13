"""The Architecture overview v2's words, EN and 中文, for review (first version for the product owner).

python3 tests/architecture_copy_deck.py  ->  out/architecture-copy/en.md and out/architecture-copy/zh.md
Built from this lab's page (out/architecture-deck-build.html, --only architecture), read through window.__ar.texts(): the hero's stage
names and lines, the guided parts, the causal switch's result box, caption and mini result bar (mask on and off), every pop-up's steps,
every hover (definition, formula, use), the scenarios, the masks, the challenge and the capability layer under it (Explain it and
A new case: each why and each wrong-pick note, which tests/numdense.py scans)."""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/architecture-deck-build.html', '--only', 'architecture'], cwd=P, check=True, capture_output=True)
PAGE = P / 'out' / 'architecture-deck.html'
PAGE.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'architecture-deck-build.html').read_text() + '</body></html>')
OUT = P / 'out' / 'architecture-copy'; OUT.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto('file://' + str(PAGE)); pg.wait_for_timeout(500)
    for lang in ['en', 'zh']:
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(300)
        t = pg.evaluate("()=>window.__ar.texts()")
        L = ['# Architecture overview v2: the words (' + lang + ')', '', '## Home hero: stages', ''] + [f'- {x}' for x in t['hero']] + ['', '## Home hero: caption', '', t['heroNote']]
        L += ['', '## Modes', ''] + [f'- {k}: {v}' for k, v in t['descs'].items()]
        L += ['', '## Guided parts (title, then the phone bar’s one line)', ''] + [f'{i + 1}. {x}' + (f' — {t["leads"][i]}' if t['leads'][i] else '') for i, x in enumerate(t['parts'])]
        rail = pg.evaluate("()=>{location.hash='architecture'; return 1}"); pg.wait_for_timeout(700)
        for i in range(len(t['parts'])):
            pg.evaluate(f"()=>document.querySelector('#ar-rail .steps [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(150)
            L += ['', f'### Part {i + 1}', '', pg.inner_text('#ar-rail .step.current').strip()]
        # the causal switch (part 5): what appears only after the switch, for both states; one line per paragraph (tests/numdense.py)
        para = lambda h: [re.sub(r'<[^>]+>', '', x).strip() for x in re.split(r'</p>', h) if re.sub(r'<[^>]+>', '', x).strip()]
        for k, head in (('off', 'mask off'), ('on', 'mask on again')):
            L += ['', f'### Part 5 after the switch, {head}: the result box', ''] + para(t['mask'][k]['obs'])
        L += ['', '## Part 5 stage caption and mini result bar (mask on, mask off)', ''] + [f'- {t["mask"][k]["cap"]} / {t["mask"][k]["read"]}' for k in ('on', 'off')]
        L += ['', '## Part 5 hovers on the loss and the chances, and the reason Next waits', '', f'- Under the disabled Next: {t["mask"]["wait"]}', '', '_Part 5 result box, tooltips:_'] + [f'  - {x}' for x in t['mask']['tips']]
        # the part 5 picture's own words (re-review N4): its labels and hovers in both states, plus the legend above it; one grid cell of
        # each kind (hidden, seen, leaked) stands for the 36. Headings avoid numdense's skip words, so tests/numdense.py scans them too.
        pg.evaluate("()=>{location.hash='architecture'}"); pg.wait_for_timeout(500)
        pg.evaluate("()=>document.querySelector('#ar-rail .steps [data-go=\"4\"]').click()"); pg.wait_for_timeout(400)
        words = """()=>{const tx=e=>e.getAttribute('data-tipx'), lab=[...new Set([...document.querySelectorAll('#ar-mk svg text')].map(t=>t.textContent.trim()).filter(Boolean))];
          const tips=[...new Set([...document.querySelectorAll('#ar-legend [data-tipx], #ar-mk [data-tipx]:not(.ar-mkc)')].map(tx))], cell={};
          document.querySelectorAll('#ar-mk .ar-mkc').forEach(e=>{const k=e.classList.contains('hid')?'hid':(e.classList.contains('fut')?'fut':'vis'); if(!cell[k]) cell[k]=tx(e);});
          return {lab, tips:tips.concat(Object.values(cell))};}"""
        for v, head in (('on', 'mask on'), ('off', 'mask off')):
            pg.evaluate(f"()=>document.querySelector('#ar-rail [data-key=\"mkmask\"] [data-v=\"{v}\"]').click()"); pg.wait_for_timeout(300)
            wd = pg.evaluate(words)
            L += ['', f'## Part 5 picture, {head}: its labels', ''] + [f'- {x}' for x in wd['lab']]
            L += ['', f'## Part 5 picture, {head}: its hovers (with the key above it; one grid cell of each kind)', '', f'_Part 5 picture, {head}, tooltips:_'] + [f'  - {x}' for x in wd['tips']]
        pg.evaluate("()=>document.querySelector('#ar-rail [data-key=\"mkmask\"] [data-v=\"on\"]').click()"); pg.wait_for_timeout(200)
        L += ['', '## Pop-ups: one caption per step', '']
        for k, v in t['pops'].items():
            L += [f'### {v["t"]}', ''] + [f'{i + 1}. {s}' for i, s in enumerate(v['steps'])] + ['']
        L += ['## Hovers: definition, formula, use', ''] + [f'- **{k}**: {v}' for k, v in t['tips'].items()]
        L += ['', '## Masks', ''] + [f'- {x}' for x in t['masks']]
        L += ['', '## Scenarios', ''] + [f'- {x}' for x in t['presets']]
        L += ['', '## Challenge', '']
        for i, q in enumerate(t['chq']):
            L += [f'{i + 1}. {"(core) " if q["core"] else "(deeper) "}{q["q"]}'] + [f'   - {"(right) " if j == q["a"] else ""}{o}' for j, o in enumerate(q['o'])] + [f'   - Why: {q["why"]}', '']
        # the capability layer under the challenge: each question, its options, why the right one is right and the note on each wrong
        # pick (LAB_UX_GUIDE R6.20: tests/numdense.py scans these lines too)
        strip = lambda s: re.sub(r'<[^>]+>', '', s).replace(' ', ' ')
        for key, head in (('explain', 'Capability layer: Explain it'), ('transfer', 'Capability layer: A new case')):
            L += [f'## {head}', '']
            for i, q in enumerate(t['exits'][key]):
                L += [f'{i + 1}. {strip(q["q"])}'] + [f'   - {"(right) " if j == q["a"] else ""}{strip(o)}' for j, o in enumerate(q['o'])] + [f'   Why: {strip(q["why"])}']
                L += [f'   Not quite (option {k + 1}): {strip(n)}' for k, n in enumerate(q.get('n') or []) if n] + ['']
        L += ['## Interview questions (Connect)', ''] + [f'- {q["q"]} → {q["a"]}' for q in t['iq']]
        (OUT / f'{lang}.md').write_text('\n'.join(L) + '\n')
        pg.evaluate("()=>{location.hash=''}"); pg.wait_for_timeout(300)
    b.close()
print('wrote', [str(OUT / 'en.md'), str(OUT / 'zh.md')])
