"""Dump every piece of text a learner meets in the multi-gpu-serving lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/multi-gpu-serving_copy_deck.py  ->  out/multi-gpu-serving/copy-en.md and out/multi-gpu-serving/copy-zh.md (and out/multi-gpu-serving-copy/{en,zh}.md for tests/numdense.py)

Order: home tile (both settings, the note), lab header, each guided part (card before and after its action, the stage, the metrics),
Show the math, the scenarios (each insight), Settings, the challenge (every feedback tier and the chart), the capability card, and
Connect with the drill answer. A tooltip is printed the first time it appears with a given text. Reduced motion is on, so no layer plays."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'multi-gpu-serving'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/multi-gpu-serving.html', '--only', 'multi-gpu-serving'], cwd=P, check=True, capture_output=True)
(OUT / 'deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'multi-gpu-serving.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'deck-wrapped.html')
clean = lambda s: ' '.join(s.replace('‑', '-').replace(' ', ' ').split()) if s else ''


# Typeset formulas (MathML) are written on one line in a plain form, (a)/(b) for a fraction and √(x) for a root, and each row of an
# aligned chain stays on one line, so the deck reads as text instead of one token per line.
LIN_JS = r"""s=>{
  const lin=n=>{ if(n.nodeType===3) return n.textContent; if(n.nodeType!==1) return ''; const t=n.tagName.toLowerCase(), k=[...n.children];
    if(t==='annotation') return ''; if(t==='mspace') return ' ';
    if(t==='mfrac') return '('+lin(k[0])+')/('+lin(k[1])+')'; if(t==='msqrt') return '√('+k.map(lin).join('')+')';
    if(t==='msub') return lin(k[0])+'_'+lin(k[1]); if(t==='msup') return lin(k[0])+'^'+lin(k[1]);
    if(t==='msubsup') return lin(k[0])+'_'+lin(k[1])+'^'+lin(k[2]); if(t==='mover'||t==='munder') return lin(k[0]);
    return [...n.childNodes].map(lin).join(''); };
  const one=m=>lin(m).replace(/\s+/g,' ').trim();
  return [...document.querySelectorAll(s)].filter(e=>e.getClientRects().length).map(e=>{
    const undo=[];
    e.querySelectorAll('.mgs-eqs').forEach(g=>{ const c=[...g.children], d=document.createElement('div');
      for(let i=0;i<c.length;i+=3){ const r=document.createElement('div'); r.textContent=c.slice(i,i+3).map(x=>{const m=x.querySelector('math'); return m?one(m):x.textContent.trim();}).join(' ').trim(); d.appendChild(r); }
      g.replaceWith(d); undo.push([g,d]); });
    e.querySelectorAll('math').forEach(m=>{ const sp=document.createElement('span'); sp.textContent=one(m); m.replaceWith(sp); undo.push([m,sp]); });
    const t=e.innerText; undo.reverse().forEach(([a,b])=>b.replaceWith(a)); return t; }).join('\n'); }"""


def txt(pg, sel):
    return pg.evaluate(LIN_JS, sel)


class Deck:
    def __init__(self): self.lines, self.tips = [], set()

    def h(self, t): self.lines.append(f'\n## {t}\n')

    def add(self, label, t):
        t = (t or '').strip()
        if t: self.lines.append(f'#### {label}\n\n' + '\n'.join(clean(x) for x in t.split('\n') if clean(x)) + '\n')

    def tipsOf(self, pg, root):
        new = []
        for t in pg.evaluate("r=>[...document.querySelectorAll(r+' [data-tipx]')].filter(e=>e.getClientRects().length).map(e=>e.getAttribute('data-tipx'))", root):
            c = clean(t)
            if c and c not in self.tips: self.tips.add(c); new.append('  - ' + c)
        if new: self.lines.append('_New tooltips:_\n' + '\n'.join(new) + '\n')


def click(pg, sel, ms=200):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(ms)


def stage(pg, d, label):
    d.add(label + ' — stage', txt(pg, '#mgs-st-title, #mgs-st-sub, #mgs-status, #mgs-app .mgs-legend, #mgs-app .mgs-bh, #mgs-sum, #mgs-sweep, #mgs-sw-n, #mgs-app .mgs-lscale, #mgs-ep-host .mgs-ep-row, #mgs-ep-host .mgs-leg'))
    d.add(label + ' — chart labels', pg.evaluate("()=>[...document.querySelectorAll('#mgs-app .stage svg')].filter(e=>e.getClientRects().length).map(s=>[...s.querySelectorAll('text')].map(t=>t.textContent).join(' | ')).join('\\n')"))
    d.add(label + ' — metric cards', txt(pg, '#mgs-metrics'))


ACTIONS = {2: [('[data-lever="4"]', 'two TP 4 replicas'), ('[data-lever="2"]', 'four TP 2 replicas'), ('[data-lever="8"]', 'one TP 8 group')],
           3: [('[data-pred="2x"]', 'answer “twice as fast”')],
           4: [('[data-lever="16"]', 'TP 16'), ('[data-lever="8"]', 'TP 8')],
           5: [('[data-lever="tp16"]', 'TP 16'), ('[data-lever="pp"]', 'pipeline'), ('[data-lever="rep"]', 'replicas')],
           6: [('.seg-ctl[data-key="p7"] button[data-v="256"]', '256 tokens'), ('.seg-ctl[data-key="p7"] button[data-v="2048"]', '2,048 tokens')]}


def run(lang):
    d = Deck()
    with sync_playwright() as pw:
        b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce')
        pg.goto(URL); pg.wait_for_timeout(300); click(pg, f'.langsw button[data-lang="{lang}"]', 150)
        pg.evaluate("()=>{location.hash='map~inference'}"); pg.wait_for_timeout(700)
        if pg.query_selector('[data-teaser="multi-gpu-serving"]'):
            d.h('Home tile')
            for v in ('8', '4', '2'):
                click(pg, f'[data-teaser="multi-gpu-serving"] button[data-v="{v}"]', 120); d.add(f'Setting TP {v}', txt(pg, '[data-teaser="multi-gpu-serving"]'))
            d.add('Notes (hidden on the tile, shown by Assumptions & sources)', pg.evaluate("()=>[...document.querySelectorAll('[data-teaser=\"multi-gpu-serving\"] .tz-note')].map(n=>n.textContent).join('\\n')"))
        pg.evaluate("()=>{location.hash='multi-gpu-serving'}"); pg.wait_for_timeout(500)
        d.h('Header'); d.add('Header', txt(pg, '#mgs-app .lab-head, #mgs-app header, #mgs-app .modes')); d.tipsOf(pg, '#mgs-app .modes')
        pops = pg.evaluate("()=>[...document.querySelectorAll('#mgs-app [data-tipx]')].filter(e=>/fid|primer/i.test(e.className)).map(e=>e.getAttribute('data-tipx'))")
        d.add('Header pop-ups (fidelity, primer)', pg.evaluate("()=>[...document.querySelectorAll('#mgs-app details, #mgs-app .fid-pop, #mgs-app .primer-pop, #mgs-app [class*=pop]')].map(e=>e.textContent).join('\\n')"))
        for i in range(8):
            d.h(f'Guided part {i + 1}')
            d.add('Card', txt(pg, '#mgs-rail .step.current')); d.tipsOf(pg, '#mgs-rail')
            stage(pg, d, f'Part {i + 1}'); d.tipsOf(pg, '#mgs-app .stage')
            for sel, lab in ACTIONS.get(i, []):
                click(pg, '#mgs-rail ' + sel, 300)
                d.add(f'Card after {lab}', txt(pg, '#mgs-rail .step.current')); stage(pg, d, f'Part {i + 1} after {lab}'); d.tipsOf(pg, '#mgs-app')
            nb = pg.query_selector('#mgs-rail .step.current [data-nav="next"]')
            if nb and not nb.is_disabled(): click(pg, '#mgs-rail .step.current [data-nav="next"]', 300)
        d.add('Next-lab footer', txt(pg, '.next-lab'))
        d.h('Show the math')
        pg.evaluate("()=>{const m=document.querySelector('#mgs-math'); if(m) m.open=true}"); pg.wait_for_timeout(150); d.add('Show the math (TP 8)', txt(pg, '#mgs-math')); d.tipsOf(pg, '#mgs-math')
        d.h('Scenarios')
        click(pg, '#mgs-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 400)
        d.add('Rail', txt(pg, '#mgs-rail .railcard, #mgs-rail .presets'))
        for i in range(6):
            click(pg, f'#mgs-app [data-preset="{i}"]', 300); d.add(f'Scenario {i + 1} insight', txt(pg, '#mgs-insight')); stage(pg, d, f'Scenario {i + 1}')
        d.add('Show the math (405B BF16, TP 8 x 2 stages)', txt(pg, '#mgs-math'))
        d.h('Settings'); d.add('Settings', txt(pg, '#mgs-controls')); d.tipsOf(pg, '#mgs-controls')
        d.h('Challenge')
        click(pg, '#mgs-app .modes button[data-mode="challenge"]', 400)
        d.add('Card', txt(pg, '#mgs-rail .railcard')); d.tipsOf(pg, '#mgs-rail')
        click(pg, '#mgs-app [data-act="submit"]', 300); d.add('16 GPUs, TP 16 submitted (merges across nodes)', txt(pg, '#mgs-ch-res'))
        click(pg, '#mgs-app .seg-ctl[data-key="chtp"] button[data-v="2"]', 250); d.add('16 GPUs, TP 2 (does not fit)', txt(pg, '#mgs-ch-res'))
        click(pg, '#mgs-app .seg-ctl[data-key="chtp"] button[data-v="4"]', 250); d.add('16 GPUs, TP 4 (too many bytes per GPU)', txt(pg, '#mgs-ch-res'))
        click(pg, '#mgs-app .seg-ctl[data-key="chtp"] button[data-v="1"]', 250); d.add('16 GPUs, TP 1 (weights do not fit)', txt(pg, '#mgs-ch-res'))
        click(pg, '#mgs-ch-res [data-act="reveal"]', 300); d.add('Show the answer', txt(pg, '#mgs-ch-res'))
        click(pg, '#mgs-app .seg-ctl[data-key="chtp"] button[data-v="8"]', 250); click(pg, '#mgs-app .seg-ctl[data-key="chpp"] button[data-v="2"]', 250); d.add('16 GPUs, TP 8 x 2 stages (met, not the best)', txt(pg, '#mgs-ch-res'))
        click(pg, '#mgs-app .seg-ctl[data-key="chpp"] button[data-v="1"]', 250); d.add('16 GPUs, 2 replicas of TP 8 (met, not the best)', txt(pg, '#mgs-ch-res'))
        click(pg, '#mgs-app .seg-ctl[data-key="chg"] button[data-v="8"]', 250); d.add('8 GPUs, TP 8 (met, the best)', txt(pg, '#mgs-ch-res'))
        d.h('Capability card'); d.add('Card', txt(pg, '#mgs-rail [data-exits]')); d.tipsOf(pg, '#mgs-rail [data-exits]')
        verd = pg.evaluate("""async ()=>{const w=()=>new Promise(r=>setTimeout(r,30)); const $=s=>document.querySelector('#mgs-rail [data-exits] '+s); const out=[]; if(!$('[data-ex="tab"]')) return '';
          for(let t=0;t<3;t++){const tb=$('[data-ex="tab"][data-i="'+t+'"]'); if(!tb) continue; tb.click(); await w();
            if(t===2){out.push('Tab 3: '+$('.ex-body').innerText); continue;}
            const nq=document.querySelectorAll('#mgs-rail [data-exits] .ex-qs .pb-dot').length||1;
            for(let j=0;j<nq;j++){const dot=$('.ex-qs [data-ex="q"][data-j="'+j+'"]'); if(dot){dot.click(); await w();}
              const q=$('.ex-body p b'); out.push('Q: '+(q?q.innerText:''));
              const nopt=document.querySelectorAll('#mgs-rail [data-exits] [data-ex="ans"]').length;
              for(let v=0;v<nopt;v++){$('[data-ex="ans"][data-v="'+v+'"]').click(); await w(); const r=$('.ex-res'); out.push('  '+'ABC'[v]+'. '+$('[data-ex="ans"][data-v="'+v+'"]').innerText+' -> '+(r?r.innerText:''));}}}
          return out.join('\\n');}""")
        d.add('Every option and its verdict', verd)
        d.h('Connect'); ct = txt(pg, '#mgs-app section.connect, section.connect'); code = pg.evaluate("()=>document.querySelector('#mgs-code').innerText")
        d.add('Connect', ct.replace(code, '```\n' + code + '\n```') if code in ct else ct)
        if pg.query_selector('[data-act="drill-reveal"]'): click(pg, '[data-act="drill-reveal"]', 150)
        d.add('Drill with the answer', txt(pg, '#mgs-drill-q, #mgs-drill-a')); d.tipsOf(pg, '#mgs-app')
        b.close()
    deck = f'# multi-gpu-serving copy deck ({lang})\n' + '\n'.join(d.lines)
    (OUT / f'copy-{lang}.md').write_text(deck)
    nd = P / 'out' / 'multi-gpu-serving-copy'; nd.mkdir(parents=True, exist_ok=True); (nd / f'{lang}.md').write_text(deck)   # where tests/numdense.py reads it
    print(lang, len(d.lines), 'blocks')


for lang in ('zh', 'en'): run(lang)
