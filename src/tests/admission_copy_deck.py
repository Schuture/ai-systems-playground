"""Dump every piece of text a learner meets in the admission lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/admission_copy_deck.py  ->  out/admission/copy-en.md and out/admission/copy-zh.md (and out/admission-copy/{en,zh}.md for tests/numdense.py)

Order: home tile (three settings, the note), lab header, each guided part (card before and after its action, the stage, the metrics),
Show the math, the scenarios (each insight), Settings, the challenge (every feedback tier and the chart), the capability card, and
Connect with the drill answer. A tooltip is printed the first time it appears with a given text. Reduced motion is on, so no run plays."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'admission'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/admission.html', '--only', 'admission'], cwd=P, check=True, capture_output=True)
(OUT / 'deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'admission.html').read_text() + '</body></html>')
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
    e.querySelectorAll('.adm-eqs').forEach(g=>{ const c=[...g.children], d=document.createElement('div');
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
    d.add(label + ' — stage', txt(pg, '#adm-st-title, #adm-st-sub, #adm-status, #adm-app .adm-legend, #adm-app .adm-bh, #adm-goals'))
    d.add(label + ' — chart labels', pg.evaluate("()=>[...document.querySelectorAll('#adm-app .stage svg')].filter(e=>e.getClientRects().length).map(s=>[...s.querySelectorAll('text')].map(t=>t.textContent).join(' | ')).join('\\n')"))
    d.add(label + ' — metric cards', txt(pg, '#adm-metrics'))


ACTIONS = {1: [('[data-lever="4"]', '4 GPUs'), ('[data-lever="6"]', '6 GPUs'), ('[data-lever="7"]', '7 GPUs')],
           2: [('[data-lever="0.7"]', '70%'), ('[data-lever="0.9"]', '90%'), ('[data-lever="1"]', '100%'), ('[data-lever="1.1"]', '110%')],
           3: [('[data-pred="slow20"]', 'answer “20% slower”'), ('[data-act="pred-show"]', 'Show the result')],
           4: [('[data-lever="none"]', 'no rules'), ('[data-lever="q"]', 'queue limit'), ('[data-lever="rl"]', 'rate limit'), ('[data-lever="pf"]', 'paid first')],
           5: [('[data-lever="fcfs"]', 'first come'), ('[data-lever="prio"]', 'paid first'), ('[data-lever="pre"]', 'preemption')],
           6: [('[data-lever="60"]', 'θ 60%'), ('[data-lever="80"]', 'θ 80%'), ('[data-lever="100"]', 'θ 100%')]}


def run(lang):
    d = Deck()
    with sync_playwright() as pw:
        b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce')
        pg.goto(URL); pg.wait_for_timeout(300); click(pg, f'.langsw button[data-lang="{lang}"]', 150)
        pg.evaluate("()=>{location.hash='map~inference'}"); pg.wait_for_timeout(700)
        if pg.query_selector('[data-teaser="admission"]'):
            d.h('Home tile')
            for v in ('4', '6', '4r'):
                click(pg, f'[data-teaser="admission"] button[data-v="{v}"]', 900); d.add(f'Setting {v}', txt(pg, '[data-teaser="admission"]'))
            d.add('Notes (hidden on the tile, shown by Assumptions & sources)', pg.evaluate("()=>[...document.querySelectorAll('[data-teaser=\"admission\"] .tz-note')].map(n=>n.textContent).join('\\n')"))
        pg.evaluate("()=>{location.hash='admission'}"); pg.wait_for_timeout(500)
        d.h('Header'); d.add('Header', txt(pg, '#adm-app .lab-head, #adm-app header, #adm-app .modes')); d.tipsOf(pg, '#adm-app .modes')
        for i in range(8):
            d.h(f'Guided part {i + 1}')
            d.add('Card', txt(pg, '#adm-rail .step.current')); d.tipsOf(pg, '#adm-rail')
            stage(pg, d, f'Part {i + 1}'); d.tipsOf(pg, '#adm-app .stage')
            for sel, lab in ACTIONS.get(i, []):
                click(pg, '#adm-rail ' + sel, 300)
                d.add(f'Card after {lab}', txt(pg, '#adm-rail .step.current')); stage(pg, d, f'Part {i + 1} after {lab}'); d.tipsOf(pg, '#adm-app')
            nb = pg.query_selector('#adm-rail .step.current [data-nav="next"]')
            if nb and not nb.is_disabled(): click(pg, '#adm-rail .step.current [data-nav="next"]', 300)
        d.add('Next-lab footer', txt(pg, '.next-lab'))
        d.h('Show the math')
        pg.evaluate("()=>{const m=document.querySelector('#adm-math'); if(m) m.open=true}"); pg.wait_for_timeout(150); d.add('Show the math', txt(pg, '#adm-math')); d.tipsOf(pg, '#adm-math')
        d.h('Scenarios')
        click(pg, '#adm-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 400)
        d.add('Rail', txt(pg, '#adm-rail .railcard, #adm-rail .presets'))
        for i in range(5):
            click(pg, f'#adm-app [data-preset="{i}"]', 300); d.add(f'Scenario {i + 1} insight', txt(pg, '#adm-insight')); stage(pg, d, f'Scenario {i + 1}')
        d.h('Settings'); d.add('Settings', txt(pg, '#adm-controls')); d.tipsOf(pg, '#adm-controls')
        d.h('Challenge')
        click(pg, '#adm-app .modes button[data-mode="challenge"]', 400)
        d.add('Card', txt(pg, '#adm-rail .railcard')); d.tipsOf(pg, '#adm-rail')
        click(pg, '#adm-app [data-act="submit"]', 300)
        pg.wait_for_function('()=>window.__adm.state().chReady', timeout=180000); pg.wait_for_timeout(300); d.add('Initial setup submitted (not met)', txt(pg, '#adm-ch-res'))
        click(pg, '#adm-ch-res [data-act="reveal"]', 300); d.add('Show the answer', txt(pg, '#adm-ch-res'))
        for k, v in (('chn', '5'), ('chr', '60'), ('chp', '1'), ('cht', '100')): click(pg, f'#adm-app .seg-ctl[data-key="{k}"] button[data-v="{v}"]', 250)
        d.add('5 GPUs, no free queue limit, rate limit, paid first, θ 100% (misses: above capacity at the peak)', txt(pg, '#adm-ch-res')); stage(pg, d, 'Challenge, 5 GPUs with no free queue limit')
        click(pg, '#adm-app .seg-ctl[data-key="chq"] button[data-v="200"]', 250)
        d.add('… with a free queue of 200 (met, fewest)', txt(pg, '#adm-ch-res'))
        click(pg, '#adm-app .seg-ctl[data-key="chn"] button[data-v="6"]', 250); d.add('the same on 6 GPUs (met, 5 enough)', txt(pg, '#adm-ch-res'))
        click(pg, '#adm-app .seg-ctl[data-key="chn"] button[data-v="5"]', 250); click(pg, '#adm-app .seg-ctl[data-key="chp"] button[data-v="0"]', 250); click(pg, '#adm-app .seg-ctl[data-key="chq"] button[data-v="50"]', 250)
        d.add('5 GPUs, queue 50, first come (misses at 3:00)', txt(pg, '#adm-ch-res'))
        click(pg, '#adm-app .seg-ctl[data-key="cht"] button[data-v="80"]', 250); d.add('… with θ 80% (misses the batch jobs)', txt(pg, '#adm-ch-res'))
        click(pg, '#adm-app .seg-ctl[data-key="chn"] button[data-v="4"]', 250); d.add('4 GPUs (misses free at 14:00)', txt(pg, '#adm-ch-res'))
        d.h('Capability card'); d.add('Card', txt(pg, '#adm-rail [data-exits]')); d.tipsOf(pg, '#adm-rail [data-exits]')
        verd = pg.evaluate("""async ()=>{const w=()=>new Promise(r=>setTimeout(r,30)); const $=s=>document.querySelector('#adm-rail [data-exits] '+s); const out=[]; if(!$('[data-ex="tab"]')) return '';
          for(let t=0;t<3;t++){const tb=$('[data-ex="tab"][data-i="'+t+'"]'); if(!tb) continue; tb.click(); await w();
            if(t===2){out.push('Tab 3: '+$('.ex-body').innerText); continue;}
            const nq=document.querySelectorAll('#adm-rail [data-exits] .ex-qs .pb-dot').length||1;
            for(let j=0;j<nq;j++){const dot=$('.ex-qs [data-ex="q"][data-j="'+j+'"]'); if(dot){dot.click(); await w();}
              const q=$('.ex-body p b'); out.push('Q: '+(q?q.innerText:''));
              const nopt=document.querySelectorAll('#adm-rail [data-exits] [data-ex="ans"]').length;
              for(let v=0;v<nopt;v++){$('[data-ex="ans"][data-v="'+v+'"]').click(); await w(); const r=$('.ex-res'); out.push('  '+'ABC'[v]+'. '+$('[data-ex="ans"][data-v="'+v+'"]').innerText+' -> '+(r?r.innerText:''));}}}
          return out.join('\\n');}""")
        d.add('Every option and its verdict', verd)
        d.h('Connect'); ct = txt(pg, '#adm-app section.connect, section.connect'); code = pg.evaluate("()=>document.querySelector('#adm-code').innerText")
        d.add('Connect', ct.replace(code, '```\n' + code + '\n```') if code in ct else ct)
        if pg.query_selector('[data-act="drill-reveal"]'): click(pg, '[data-act="drill-reveal"]', 150)
        d.add('Drill with the answer', txt(pg, '#adm-drill-q, #adm-drill-a')); d.tipsOf(pg, '#adm-app')
        b.close()
    deck = f'# admission copy deck ({lang})\n' + '\n'.join(d.lines)
    (OUT / f'copy-{lang}.md').write_text(deck)
    nd = P / 'out' / 'admission-copy'; nd.mkdir(parents=True, exist_ok=True); (nd / f'{lang}.md').write_text(deck)   # where tests/numdense.py reads it
    print(lang, len(d.lines), 'blocks')


for lang in ('zh', 'en'): run(lang)
