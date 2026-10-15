"""Dump every piece of text a learner meets in the chunk-rerank lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/chunk-rerank_copy_deck.py  ->  out/chunk-rerank/copy-en.md and out/chunk-rerank/copy-zh.md, and the same decks as
out/chunk-rerank-copy/en.md and zh.md, where tests/numdense.py reads them. Readouts (stage, metric cards, chart, settings) sit under
sub-headings numdense skips; tooltips under a "_New tooltips:_" line.

Order: home tile (each retriever, the note), lab header, each guided part (card before and after its action, the stage, the
metrics), Show the math, the scenarios (each insight), Settings, the challenge (every feedback tier and the chart), the capability
card, and Connect with the drill answer. A tooltip is printed the first time it appears with a given text, and only the first three
of each kind. Reduced motion is on, so every cut ends at once."""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'chunk-rerank'; OUT.mkdir(parents=True, exist_ok=True)
OUT2 = P / 'out' / 'chunk-rerank-copy'; OUT2.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/chunk-rerank.html', '--only', 'chunk-rerank'], cwd=P, check=True, capture_output=True)
(OUT / 'deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'chunk-rerank.html').read_text() + '</body></html>')
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
    e.querySelectorAll('.ckr-eqs').forEach(g=>{ const c=[...g.children], d=document.createElement('div');
      for(let i=0;i<c.length;i+=3){ const r=document.createElement('div'); r.textContent=c.slice(i,i+3).map(x=>{const m=x.querySelector('math'); return m?one(m):x.textContent.trim();}).join(' ').trim(); d.appendChild(r); }
      g.replaceWith(d); undo.push([g,d]); });
    e.querySelectorAll('math').forEach(m=>{ const sp=document.createElement('span'); sp.textContent=one(m); m.replaceWith(sp); undo.push([m,sp]); });
    const t=e.innerText; undo.reverse().forEach(([a,b])=>b.replaceWith(a)); return t; }).join('\n'); }"""


def txt(pg, sel):
    return pg.evaluate(LIN_JS, sel)


class Deck:
    def __init__(self): self.lines, self.tips, self.kinds = [], set(), {}

    def h(self, t): self.lines.append(f'\n## {t}\n')

    def add(self, label, t):
        t = (t or '').strip()
        if t: self.lines.append(f'#### {label}\n\n' + '\n'.join(clean(x) for x in t.split('\n') if clean(x)) + '\n')

    def tipsOf(self, pg, root):
        new = []
        for t in pg.evaluate("r=>[...document.querySelectorAll(r+' [data-tipx]')].filter(e=>e.getClientRects().length).map(e=>e.getAttribute('data-tipx'))", root):
            c = clean(t)
            k = re.sub(r'“[^”]*”|[0-9.,%−-]+', '#', c)[:24]
            if c and c not in self.tips:
                self.tips.add(c); self.kinds[k] = self.kinds.get(k, 0) + 1
                if self.kinds[k] <= 3: new.append('  - ' + c)
        if new: self.lines.append('_New tooltips:_\n' + '\n'.join(new) + '\n')


def click(pg, sel, ms=200):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(ms)


def stage(pg, d, label):
    d.add(label + ' — stage', txt(pg, '#ckr-stage .ckr-head, #ckr-stage .ckr-ql, #ckr-stage .ckr-pan:not([hidden]) .ckr-pt, #ckr-stage .ckr-pan:not([hidden]) .ckr-leg, #ckr-stage #ckr-pan-doc:not([hidden]) #ckr-text, #ckr-stage #ckr-pan-lists:not([hidden]) #ckr-ltools, #ckr-stage #ckr-pan-lists:not([hidden]) #ckr-cols, #ckr-stage #ckr-pan-lists:not([hidden]) #ckr-sum'))
    d.add(label + ' — pictures', pg.evaluate("()=>[...document.querySelectorAll('#ckr-stage svg')].filter(e=>e.getClientRects().length).map(s=>(s.getAttribute('aria-label')||'')+' | '+[...s.querySelectorAll('text')].map(t=>t.textContent).join(' | ')).join('\\n')"))
    d.add(label + ' — metric cards', txt(pg, '#ckr-metrics:not([hidden])'))
    d.add(label + ' — chart title', txt(pg, '#ckr-pan-chart:not([hidden]) .ckr-pt'))


def run(lang):
    d = Deck()
    with sync_playwright() as pw:
        b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce')
        pg.goto(URL); pg.wait_for_timeout(300); click(pg, f'.langsw button[data-lang="{lang}"]', 150)
        pg.evaluate("()=>{location.hash='map~apps'}"); pg.wait_for_timeout(700)
        if pg.query_selector('[data-teaser="chunk-rerank"]'):
            d.h('Home tile')
            for v in ('0', '1', '2'):
                if pg.query_selector(f'[data-teaser="chunk-rerank"] button[data-v="{v}"]'):
                    click(pg, f'[data-teaser="chunk-rerank"] button[data-v="{v}"]', 120)
                    d.add(f'Option {int(v) + 1}', txt(pg, '[data-teaser="chunk-rerank"] .tz-read'))
            d.add('Tile', txt(pg, '[data-teaser="chunk-rerank"]'))
            d.add('Tile picture', pg.evaluate("()=>{const s=document.querySelector('[data-teaser=\"chunk-rerank\"] svg'); return s?[s.getAttribute('aria-label'),...[...s.querySelectorAll('text')].map(t=>t.textContent)].join(' | '):''}"))
        pg.evaluate("()=>{location.hash='chunk-rerank'}"); pg.wait_for_timeout(500)
        d.h('Header'); d.add('Header', txt(pg, '#ckr-app .lab-head, #ckr-app header, #ckr-app .modes')); d.tipsOf(pg, '#ckr-app .modes')
        d.add('Fidelity note', pg.evaluate("()=>{const e=document.querySelector('#ckr-app [id$=\"-fid\"]'); return e?e.innerText:''}"))
        d.add('Primer note', pg.evaluate("()=>{const e=document.querySelector('#ckr-app [id$=\"-pri\"]'); return e?e.innerText:''}"))
        d.add('Transport', txt(pg, '#ckr-transport')); d.tipsOf(pg, '#ckr-transport')
        d.add('The 20 questions', pg.evaluate("()=>[...document.querySelectorAll('#ckr-qsel option')].map(o=>o.textContent).join('\\n')"))
        for i in range(7):
            d.h(f'Guided part {i + 1}')
            d.add('Card', txt(pg, '#ckr-rail .step.current')); d.tipsOf(pg, '#ckr-rail')
            stage(pg, d, f'Part {i + 1}'); d.tipsOf(pg, '#ckr-stage')
            acted = False
            st = pg.evaluate("()=>window.__ckr.state()")
            if st['step'] == 1:
                for k, v in (('size', '16'), ('size', '32'), ('size', 'h'), ('size', '64'), ('ov', '1'), ('ov', '2')):
                    click(pg, f'#ckr-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]', 200); d.add(f'After {k} = {v}', txt(pg, '#ckr-rail [data-obs]'))
                acted = True
            if pg.query_selector('#ckr-rail [data-pred="kw"]'): click(pg, '#ckr-rail [data-pred="vec"]'); acted = True
            if st['step'] == 3:
                for q in ('15', '8', '9', '0'):
                    click(pg, f'#ckr-rail .seg-ctl[data-key="q"] button[data-v="{q}"]', 200); d.add(f'After question {int(q) + 1}', txt(pg, '#ckr-rail [data-obs]')); d.add('Lists', txt(pg, '#ckr-cols'))
                click(pg, '#ckr-rail .seg-ctl[data-key="q"] button[data-v="8"]', 200)
            if st['step'] == 4:
                click(pg, '#ckr-rail .seg-ctl[data-key="rr"] button[data-v="1"]', 200); d.add('After rerank on', txt(pg, '#ckr-rail [data-obs]')); d.add('Lists', txt(pg, '#ckr-cols'))
                click(pg, '#ckr-rail .seg-ctl[data-key="q"] button[data-v="0"]', 200); d.add('After question 1 (E-201)', txt(pg, '#ckr-rail [data-obs]')); d.add('Lists', txt(pg, '#ckr-cols'))
                click(pg, '#ckr-rail .seg-ctl[data-key="q"] button[data-v="17"]', 200); acted = True
            if st['step'] == 5:
                for k, v in (('k', '3'), ('k', '5'), ('k', '1'), ('size', '128')):
                    click(pg, f'#ckr-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]', 200); d.add(f'After {k} = {v}', txt(pg, '#ckr-rail [data-obs]'))
                acted = True
            if acted: d.add('Card after the action', txt(pg, '#ckr-rail .step.current')); stage(pg, d, f'Part {i + 1} after'); d.tipsOf(pg, '#ckr-app')
            nb = pg.query_selector('#ckr-rail .step.current [data-nav="next"]')
            if nb and not nb.is_disabled(): click(pg, '#ckr-rail .step.current [data-nav="next"]', 300)
        d.add('Next-lab footer', txt(pg, '.next-lab'))
        d.h('Show the math')
        pg.evaluate("()=>{const m=document.querySelector('#ckr-math'); if(m) m.open=true}"); pg.wait_for_timeout(150); d.add('Show the math (part 7 state)', txt(pg, '#ckr-math')); d.tipsOf(pg, '#ckr-math')
        d.h('Scenarios')
        click(pg, '#ckr-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 400)
        d.add('Rail', txt(pg, '#ckr-rail .railcard'))
        for i in range(6):
            click(pg, f'#ckr-rail [data-preset="{i}"]', 300); d.add(f'Scenario {i + 1} insight', txt(pg, '#ckr-insight')); d.add(f'Scenario {i + 1} stage title', txt(pg, '#ckr-st-title, #ckr-status'))
        d.add('Chart', txt(pg, '#ckr-pan-chart'))
        d.h('Settings'); d.add('Settings', txt(pg, '#ckr-controls')); d.tipsOf(pg, '#ckr-controls')
        pg.evaluate("()=>{const m=document.querySelector('#ckr-controls details'); if(m) m.open=true}"); pg.wait_for_timeout(100); d.add('Lab assumptions', txt(pg, '#ckr-controls details'))
        d.h('Challenge')
        click(pg, '#ckr-app .modes button[data-mode="challenge"]', 400)
        d.add('Card', txt(pg, '#ckr-rail .railcard')); d.add('Settings note', txt(pg, '#ckr-lock'))
        d.add('Before Submit — goal lines and stage status', txt(pg, '#ckr-ch-a, #ckr-ch-c, #ckr-ch-l, #ckr-status'))
        d.add('Challenge stage', txt(pg, '#ckr-stage .ckr-head, #ckr-d-title, #ckr-d-cap, #ckr-d-cnt'))
        click(pg, '#ckr-rail [data-act="submit"]', 300); d.add('Initial setup submitted', txt(pg, '#ckr-ch-res, #ckr-status'))
        click(pg, '#ckr-rail .seg-ctl[data-key="k"] button[data-v="5"]', 200); click(pg, '#ckr-rail [data-act="submit"]', 300); d.add('Keep 5 submitted (over 200 tokens)', txt(pg, '#ckr-ch-res'))
        for k, v in (('k', '3'), ('ov', '2'), ('ret', 'hyb')): click(pg, f'#ckr-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]', 200)
        click(pg, '#ckr-rail [data-act="submit"]', 300); d.add('64, half overlap, hybrid, keep 3 submitted (met, more tokens)', txt(pg, '#ckr-ch-res'))
        for k, v in (('rr', '1'), ('k', '1')): click(pg, f'#ckr-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]', 200)
        click(pg, '#ckr-rail [data-act="submit"]', 300) if pg.query_selector('#ckr-rail [data-act="submit"]:not([hidden])') else None
        pg.evaluate("()=>{const e=document.querySelector('#ckr-ch-res details'); if(e) e.open=true}"); pg.wait_for_timeout(100)
        d.add('Result: 64, half overlap, hybrid + rerank, keep 1 (best; the comparison and the why)', txt(pg, '#ckr-ch-res'))
        click(pg, '#ckr-rail .seg-ctl[data-key="size"] button[data-v="128"]', 200); d.add('128 tokens with rerank (over 80 ms)', txt(pg, '#ckr-ch-res'))
        d.h('Capability card'); d.add('Card', txt(pg, '#ckr-rail [data-exits]')); d.tipsOf(pg, '#ckr-rail [data-exits]')
        verd = pg.evaluate("""async ()=>{const w=()=>new Promise(r=>setTimeout(r,30)); const $=s=>document.querySelector('#ckr-rail [data-exits] '+s); const out=[];
          for(let t=0;t<3;t++){const tb=$('[data-ex="tab"][data-i="'+t+'"]'); if(!tb) continue; tb.click(); await w();
            if(t===2){out.push('Tab 3: '+$('.ex-body').innerText); continue;}
            const nq=document.querySelectorAll('#ckr-rail [data-exits] .ex-qs .pb-dot').length||1;
            for(let j=0;j<nq;j++){const dot=$('.ex-qs [data-ex="q"][data-j="'+j+'"]'); if(dot){dot.click(); await w();}
              const q=$('.ex-body p b'); out.push('Q: '+(q?q.innerText:''));
              const nopt=document.querySelectorAll('#ckr-rail [data-exits] [data-ex="ans"]').length;
              for(let v=0;v<nopt;v++){$('[data-ex="ans"][data-v="'+v+'"]').click(); await w(); const r=$('.ex-res'); out.push('  ('+'abcd'[v]+') '+$('[data-ex="ans"][data-v="'+v+'"]').innerText+' -> '+(r?r.innerText:''));}}}
          return out.join('\\n');}""")
        d.add('Every option and its verdict', verd)
        d.h('Connect'); d.add('Connect', txt(pg, '#ckr-app section.connect'))
        if pg.query_selector('#ckr-app [data-act="drill-reveal"]'): click(pg, '#ckr-app [data-act="drill-reveal"]', 150)
        d.add('Drill with the answer', txt(pg, '#ckr-drill-q, #ckr-drill-a')); d.tipsOf(pg, '#ckr-app')
        b.close()
    body = f'# chunk-rerank copy deck ({lang})\n' + '\n'.join(d.lines)
    (OUT / f'copy-{lang}.md').write_text(body); (OUT2 / f'{lang}.md').write_text(body)
    print(lang, len(d.lines), 'blocks')


for lang in ('zh', 'en'): run(lang)
