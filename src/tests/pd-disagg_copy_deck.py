"""Dump every piece of text a learner meets in the pd-disagg lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/pd-disagg_copy_deck.py  ->  out/pd-disagg/copy-en.md and out/pd-disagg/copy-zh.md (and out/pd-disagg-copy/{en,zh}.md for tests/numdense.py)

Order: home tile (both settings, the note), lab header, each guided part (card before and after its action, the stage, the metrics),
Show the math, the scenarios (each insight), Settings, the challenge (every feedback tier and the chart), the capability card, and
Connect with the drill answer. A tooltip is printed the first time it appears with a given text. Reduced motion is on, so no run plays."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'pd-disagg'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/pd-disagg.html', '--only', 'pd-disagg'], cwd=P, check=True, capture_output=True)
(OUT / 'deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'pd-disagg.html').read_text() + '</body></html>')
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
    e.querySelectorAll('.pdd-eqs').forEach(g=>{ const c=[...g.children], d=document.createElement('div');
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
    d.add(label + ' — stage', txt(pg, '#pdd-st-title, #pdd-st-sub, #pdd-status, #pdd-app .pdd-legend, #pdd-app .pdd-bh, #pdd-app .pdd-wcap, #pdd-app .pdd-scope'))
    d.add(label + ' — chart labels', pg.evaluate("()=>[...document.querySelectorAll('#pdd-app .stage svg')].filter(e=>e.getClientRects().length).map(s=>[...s.querySelectorAll('text')].map(t=>t.textContent).join(' | ')).join('\\n')"))
    d.add(label + ' — metric cards', txt(pg, '#pdd-metrics'))


ACTIONS = {2: [('[data-lever="2048"]', 'budget 2,048'), ('[data-lever="512"]', 'budget 512'), ('[data-lever="128"]', 'budget 128'), ('[data-lever="64"]', 'budget 64')],
           3: [('[data-pred="both"]', 'answer “both”')],
           4: [('[data-lever="2"]', '2 + 2'), ('[data-lever="3"]', '3 + 1')],
           5: [('.seg-ctl[data-key="p6l"] button[data-v="ib"]', 'across nodes')],
           6: [('.seg-ctl[data-key="p7"] button[data-v="0"]', 'colocated, budget 512'), ('.seg-ctl[data-key="p7"] button[data-v="7"]', '7:1'), ('.seg-ctl[data-key="p7"] button[data-v="6"]', '6:2')]}


def run(lang):
    d = Deck()
    with sync_playwright() as pw:
        b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce')
        pg.goto(URL); pg.wait_for_timeout(300); click(pg, f'.langsw button[data-lang="{lang}"]', 150)
        pg.evaluate("()=>{location.hash='map~inference'}"); pg.wait_for_timeout(700)
        if pg.query_selector('[data-teaser="pd-disagg"]'):
            d.h('Home tile')
            for v in ('co', 'split'):
                click(pg, f'[data-teaser="pd-disagg"] button[data-v="{v}"]', 120); d.add(f'Setting {v}', txt(pg, '[data-teaser="pd-disagg"]'))
            d.add('Notes (hidden on the tile, shown by Assumptions & sources)', pg.evaluate("()=>[...document.querySelectorAll('[data-teaser=\"pd-disagg\"] .tz-note')].map(n=>n.textContent).join('\\n')"))
        pg.evaluate("()=>{location.hash='pd-disagg'}"); pg.wait_for_timeout(500)
        d.h('Header'); d.add('Header', txt(pg, '#pdd-app .lab-head, #pdd-app header, #pdd-app .modes')); d.tipsOf(pg, '#pdd-app .modes')
        for i in range(8):
            if i == 6: pg.wait_for_function('()=>window.__pdd.state().gpQueue===0', timeout=90000); pg.wait_for_timeout(200)
            d.h(f'Guided part {i + 1}')
            d.add('Card', txt(pg, '#pdd-rail .step.current')); d.tipsOf(pg, '#pdd-rail')
            stage(pg, d, f'Part {i + 1}'); d.tipsOf(pg, '#pdd-app .stage')
            for sel, lab in ACTIONS.get(i, []):
                click(pg, '#pdd-rail ' + sel, 300)
                d.add(f'Card after {lab}', txt(pg, '#pdd-rail .step.current')); stage(pg, d, f'Part {i + 1} after {lab}'); d.tipsOf(pg, '#pdd-app')
            nb = pg.query_selector('#pdd-rail .step.current [data-nav="next"]')
            if nb and not nb.is_disabled(): click(pg, '#pdd-rail .step.current [data-nav="next"]', 300)
            if i == 4: d.add('Part 5 verdict on the answer “both”', txt(pg, '#pdd-verdict'))
        d.add('Next-lab footer', txt(pg, '.next-lab'))
        d.h('Show the math')
        pg.evaluate("()=>{const m=document.querySelector('#pdd-math'); if(m) m.open=true}"); pg.wait_for_timeout(150); d.add('Show the math', txt(pg, '#pdd-math')); d.tipsOf(pg, '#pdd-math')
        d.h('Scenarios')
        click(pg, '#pdd-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 400)
        d.add('Rail', txt(pg, '#pdd-rail .railcard, #pdd-rail .presets'))
        for i in range(5):
            click(pg, f'#pdd-app [data-preset="{i}"]', 300); d.add(f'Scenario {i + 1} insight', txt(pg, '#pdd-insight')); stage(pg, d, f'Scenario {i + 1}')
        d.h('Settings'); d.add('Settings', txt(pg, '#pdd-controls')); d.tipsOf(pg, '#pdd-controls')
        d.h('Challenge')
        click(pg, '#pdd-app .modes button[data-mode="challenge"]', 400)
        d.add('Card', txt(pg, '#pdd-rail .railcard')); d.tipsOf(pg, '#pdd-rail')
        click(pg, '#pdd-app .seg-ctl[data-key="chp"] button[data-v="7"]', 250)
        click(pg, '#pdd-app [data-act="submit"]', 300); d.add('7 + 1 submitted (decode short of room)', txt(pg, '#pdd-ch-res'))
        click(pg, '#pdd-app .seg-ctl[data-key="chp"] button[data-v="3"]', 250); d.add('3 + 5 (prefill queue)', txt(pg, '#pdd-ch-res'))
        click(pg, '#pdd-app .seg-ctl[data-key="chm"] button[data-v="co"]', 250); click(pg, '#pdd-app .seg-ctl[data-key="chb"] button[data-v="0"]', 250); d.add('colocated, no limit', txt(pg, '#pdd-ch-res'))
        click(pg, '#pdd-app .seg-ctl[data-key="chb"] button[data-v="128"]', 250); d.add('colocated, 128', txt(pg, '#pdd-ch-res'))
        click(pg, '#pdd-ch-res [data-act="reveal"]', 300); d.add('Show the answer', txt(pg, '#pdd-ch-res'))
        click(pg, '#pdd-app .seg-ctl[data-key="chm"] button[data-v="split"]', 250); click(pg, '#pdd-app .seg-ctl[data-key="chp"] button[data-v="5"]', 250)
        pg.wait_for_function('()=>window.__pdd.state().gpQueue===0', timeout=90000); pg.wait_for_timeout(300); d.add('5 + 3 (met, less room)', txt(pg, '#pdd-ch-res'))
        click(pg, '#pdd-app .seg-ctl[data-key="chm"] button[data-v="co"]', 250); click(pg, '#pdd-app .seg-ctl[data-key="chb"] button[data-v="2048"]', 250)
        pg.wait_for_timeout(300); d.add('colocated, 2,048 (met, most room)', txt(pg, '#pdd-ch-res'))
        d.h('Capability card'); d.add('Card', txt(pg, '#pdd-rail [data-exits]')); d.tipsOf(pg, '#pdd-rail [data-exits]')
        verd = pg.evaluate("""async ()=>{const w=()=>new Promise(r=>setTimeout(r,30)); const $=s=>document.querySelector('#pdd-rail [data-exits] '+s); const out=[]; if(!$('[data-ex="tab"]')) return '';
          for(let t=0;t<3;t++){const tb=$('[data-ex="tab"][data-i="'+t+'"]'); if(!tb) continue; tb.click(); await w();
            if(t===2){out.push('Tab 3: '+$('.ex-body').innerText); continue;}
            const nq=document.querySelectorAll('#pdd-rail [data-exits] .ex-qs .pb-dot').length||1;
            for(let j=0;j<nq;j++){const dot=$('.ex-qs [data-ex="q"][data-j="'+j+'"]'); if(dot){dot.click(); await w();}
              const q=$('.ex-body p b'); out.push('Q: '+(q?q.innerText:''));
              const nopt=document.querySelectorAll('#pdd-rail [data-exits] [data-ex="ans"]').length;
              for(let v=0;v<nopt;v++){$('[data-ex="ans"][data-v="'+v+'"]').click(); await w(); const r=$('.ex-res'); out.push('  '+'ABC'[v]+'. '+$('[data-ex="ans"][data-v="'+v+'"]').innerText+' -> '+(r?r.innerText:''));}}}
          return out.join('\\n');}""")
        d.add('Every option and its verdict', verd)
        d.h('Connect'); ct = txt(pg, '#pdd-app section.connect, section.connect'); code = pg.evaluate("()=>document.querySelector('#pdd-code').innerText")
        d.add('Connect', ct.replace(code, '```\n' + code + '\n```') if code in ct else ct)
        if pg.query_selector('[data-act="drill-reveal"]'): click(pg, '[data-act="drill-reveal"]', 150)
        d.add('Drill with the answer', txt(pg, '#pdd-drill-q, #pdd-drill-a')); d.tipsOf(pg, '#pdd-app')
        b.close()
    deck = f'# pd-disagg copy deck ({lang})\n' + '\n'.join(d.lines)
    (OUT / f'copy-{lang}.md').write_text(deck)
    nd = P / 'out' / 'pd-disagg-copy'; nd.mkdir(parents=True, exist_ok=True); (nd / f'{lang}.md').write_text(deck)   # where tests/numdense.py reads it
    print(lang, len(d.lines), 'blocks')


for lang in ('zh', 'en'): run(lang)
