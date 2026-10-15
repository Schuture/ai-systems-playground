"""Dump every piece of text a learner meets in the embeddings lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/embeddings_copy_deck.py  ->  out/embeddings/copy-en.md and out/embeddings/copy-zh.md (also out/embeddings-copy/en.md and
zh.md, where tests/numdense.py reads them)

Order: home tile (every step count, the note), lab header, each guided part (card before and after its action, the stage, the
metrics), Show the math, the scenarios (each insight), Settings, the challenge (every feedback tier and the chart), the capability
card, and Connect with the drill answer; then part 4 once more with motion on, while Train plays its slow steps and its fast part. A tooltip is printed the first time it appears with a given text, and
only the first three of each kind (map points, table cells). Reduced motion is on (except in that last capture), so
every run ends at once."""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'embeddings'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/embeddings.html', '--only', 'embeddings'], cwd=P, check=True, capture_output=True)
(OUT / 'deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'embeddings.html').read_text() + '</body></html>')
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
  const all=[...document.querySelectorAll(s)].filter(e=>e.getClientRects().length);
  return all.filter(e=>!all.some(o=>o!==e&&o.contains(e))).map(e=>{   // an element inside another match is already in that one's text
    const undo=[];
    e.querySelectorAll('.emb-eqs').forEach(g=>{ const c=[...g.children], d=document.createElement('div');
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
        if t: self.lines.append(f'**{label}**\n\n' + '\n'.join(clean(x) for x in t.split('\n') if clean(x)) + '\n')

    def tipsOf(self, pg, root):
        new = []
        for t in pg.evaluate("r=>[...document.querySelectorAll(r+' [data-tipx]')].filter(e=>e.getClientRects().length).map(e=>e.getAttribute('data-tipx'))", root):
            c = clean(t)
            k = re.sub(r'“[^”]*”|[0-9.,%−-]+', '#', c)[:24]
            if c and c not in self.tips:
                self.tips.add(c); self.kinds[k] = self.kinds.get(k, 0) + 1
                if self.kinds[k] <= 3: new.append('- ' + c)
        if new: self.lines.append('Tooltips (new):\n' + '\n'.join(new) + '\n')


def click(pg, sel, ms=200):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(ms)


def stage(pg, d, label):
    pan = '#emb-stage .emb-pan:not([hidden]) '
    d.add(label + ' — stage', txt(pg, '#emb-stage .emb-head, ' + ', '.join(pan + c for c in ('.emb-pt', '.emb-cviews', '.emb-pair', '.emb-note', '.emb-cap', '.emb-rsub', '.emb-rows', '.emb-rres', '.emb-trap', '.emb-leg', '.emb-side', '.legend-row'))))
    d.add(label + ' — chart labels', pg.evaluate("()=>[...document.querySelectorAll('#emb-stage svg')].filter(e=>e.getClientRects().length).map(s=>[...s.querySelectorAll('text')].map(t=>t.textContent).join(' | ')).join('\\n')"))
    d.add(label + ' — metric cards', txt(pg, '#emb-metrics'))


def anim_texts(b, lang, d):
    """Part 4 played at the default speed (motion on): the table and the rows during a slow step, then during the fast part."""
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.goto(URL); pg.wait_for_timeout(300); click(pg, f'.langsw button[data-lang="{lang}"]', 150)
    pg.evaluate("()=>{location.hash='embeddings'}"); pg.wait_for_timeout(500)
    click(pg, '#emb-rail [data-go="3"]', 500)
    sel = ', '.join('#emb-stage ' + c for c in ('#emb-pan-heat .emb-pt', '#emb-pan-heat .emb-cap', '#emb-pan-rows .emb-pt', '#emb-pan-rows .emb-rsub', '#emb-pan-rows .emb-rows', '#emb-pan-rows .emb-rres'))
    click(pg, '#emb-transport [data-act="play"]', 50)
    pg.wait_for_function("()=>{const a=window.__emb.animState(); return a&&a.slow&&a.k===5&&a.ph>0.1&&a.ph<0.35}", timeout=20000)
    d.add('Part 4 while Train plays: slow step 5, arrows on', txt(pg, sel))
    pg.wait_for_function("()=>{const a=window.__emb.animState(); return a&&a.slow&&a.k===5&&a.ph>0.75}", timeout=20000)
    d.add('Part 4 while Train plays: slow step 5, after the update', txt(pg, sel))
    pg.wait_for_function("()=>{const a=window.__emb.animState(); return a&&!a.slow&&a.frozen!==null}", timeout=30000)
    d.add('Part 4 while Train plays: the fast part', txt(pg, sel))
    pg.wait_for_function("()=>!window.__emb.animState()", timeout=40000)
    pg.close()


def avg_ready(pg):   # the 16-start averages are filled in the background; wait until the observation shows them
    pg.wait_for_function("()=>{const o=document.querySelector('#emb-rail [data-obs]'); return !o||!/Averaging 16|正在算 16/.test(o.innerText)}", timeout=60000); pg.wait_for_timeout(100)


def run(lang):
    d = Deck()
    with sync_playwright() as pw:
        b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce')
        pg.goto(URL); pg.wait_for_timeout(300); click(pg, f'.langsw button[data-lang="{lang}"]', 150)
        # home tile
        pg.evaluate("()=>{location.hash='map~apps'}"); pg.wait_for_timeout(700)
        if pg.query_selector('[data-teaser="embeddings"]'):
            d.h('Home tile')
            for i in range(6):
                click(pg, f'[data-teaser="embeddings"] button[data-v="{i}"]', 120)
                d.add(f'Step option {i + 1}', txt(pg, '[data-teaser="embeddings"] .tz-read'))
            d.add('Tile', txt(pg, '[data-teaser="embeddings"]'))
            d.add('Tile picture', pg.evaluate("()=>{const s=document.querySelector('[data-teaser=\"embeddings\"] svg'); return s?s.getAttribute('aria-label')+' | '+[...s.querySelectorAll('text')].map(t=>t.textContent).join(' | '):''}"))
        pg.evaluate("()=>{location.hash='embeddings'}"); pg.wait_for_timeout(500)
        d.h('Header'); d.add('Header', txt(pg, '#emb-app .lab-head, #emb-app header, #emb-app .modes')); d.tipsOf(pg, '#emb-app .modes')
        d.add('Fidelity note', pg.evaluate("()=>{const e=document.querySelector('#emb-app [id$=\"-fid\"]'); return e?e.innerText:''}"))
        d.add('Primer note', pg.evaluate("()=>{const e=document.querySelector('#emb-app [id$=\"-pri\"]'); return e?e.innerText:''}"))
        for i in range(8):
            d.h(f'Guided part {i + 1}')
            d.add('Card', txt(pg, '#emb-rail .step.current')); d.tipsOf(pg, '#emb-rail')
            stage(pg, d, f'Part {i + 1}'); d.tipsOf(pg, '#emb-stage')
            acted = False
            st = pg.evaluate("()=>window.__emb.state()")
            if st['step'] == 1:
                for q in ('9', '17', '22'):
                    click(pg, f'#emb-rail .seg-ctl[data-key="q"] button[data-v="{q}"]', 200); d.add(f'After question {int(q) + 1}', txt(pg, '#emb-rail [data-obs]')); d.add('Search list', txt(pg, '#emb-stage .emb-side'))
            if pg.query_selector('#emb-rail [data-pred="pos"]'): click(pg, '#emb-rail [data-pred="pos"]'); acted = True
            if st['step'] == 3: click(pg, '#emb-transport [data-act="play"]', 400); acted = True
            if st['step'] == 4:
                for B in ('2', '4', '16'):
                    click(pg, f'#emb-rail .seg-ctl[data-key="B"] button[data-v="{B}"]', 300); avg_ready(pg); d.add(f'After B = {B}', txt(pg, '#emb-rail [data-obs]'))
                acted = True
            if st['step'] == 5:
                for t in ('1', '0.2', '0.01'):
                    click(pg, f'#emb-rail .seg-ctl[data-key="tau"] button[data-v="{t}"]', 300); avg_ready(pg); d.add(f'After τ = {t}', txt(pg, '#emb-rail [data-obs]'))
                acted = True
            if st['step'] == 6:
                for k in ('order', 'neg'):
                    click(pg, f'#emb-rail [data-lever="{k}"]', 200); d.add(f'After the {k} pair', txt(pg, '#emb-rail .step.current')); stage(pg, d, f'Part {i + 1}, the {k} pair')
            if acted: d.add('Card after the action', txt(pg, '#emb-rail .step.current')); stage(pg, d, f'Part {i + 1} after'); d.tipsOf(pg, '#emb-app')
            nb = pg.query_selector('#emb-rail .step.current [data-nav="next"]')
            if nb and not nb.is_disabled(): click(pg, '#emb-rail .step.current [data-nav="next"]', 300)
        d.add('Next-lab footer', txt(pg, '.next-lab'))
        d.h('Part 4 while Train plays'); anim_texts(b, lang, d)
        d.h('Show the math')
        pg.evaluate("()=>{const m=document.querySelector('#emb-math'); if(m) m.open=true}"); pg.wait_for_timeout(150); d.add('Math', txt(pg, '#emb-math')); d.tipsOf(pg, '#emb-math')
        d.h('Scenarios')
        click(pg, '#emb-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 400)
        d.add('Rail', txt(pg, '#emb-rail .railcard, #emb-rail .presets'))
        for i in range(6):
            click(pg, f'#emb-rail [data-preset="{i}"]', 300); d.add(f'Scenario {i + 1} insight', txt(pg, '#emb-insight'))
        click(pg, '#emb-c-view button[data-v="loss"]', 200); d.add('Chart, loss view', txt(pg, '#emb-pan-chart'))
        d.h('Settings'); d.add('Settings', txt(pg, '#emb-controls')); d.tipsOf(pg, '#emb-controls')
        pg.evaluate("()=>{const m=document.querySelector('#emb-controls details'); if(m) m.open=true}"); pg.wait_for_timeout(100); d.add('Fixed in this lab', txt(pg, '#emb-controls details'))
        d.h('Challenge')
        click(pg, '#emb-app .modes button[data-mode="challenge"]', 400)
        d.add('Card', txt(pg, '#emb-rail .railcard'))
        done = "()=>!window.__emb.animState()"
        d.add('Before Submit — goal lines and stage status', txt(pg, '#emb-ch-a, #emb-ch-g, #emb-status'))
        click(pg, '#emb-rail [data-act="submit"]', 300); pg.wait_for_function(done, timeout=20000); pg.wait_for_timeout(200)
        d.add('B = 2 × 25 submitted', txt(pg, '#emb-ch-res, #emb-status'))
        click(pg, '#emb-rail .seg-ctl[data-key="B"] button[data-v="16"]', 300)
        click(pg, '#emb-rail [data-act="submit"]', 300); pg.wait_for_function(done, timeout=20000); pg.wait_for_timeout(200)
        d.add('B = 16 × 25 submitted', txt(pg, '#emb-ch-res'))
        click(pg, '#emb-rail .seg-ctl[data-key="chN"] button[data-v="10"]', 300); d.add('B = 16 × 10 (and the chart)', txt(pg, '#emb-ch-res'))
        d.h('Capability card'); d.add('Card', txt(pg, '#emb-rail [data-exits]')); d.tipsOf(pg, '#emb-rail [data-exits]')
        verd = pg.evaluate("""async ()=>{const w=()=>new Promise(r=>setTimeout(r,30)); const $=s=>document.querySelector('#emb-rail [data-exits] '+s); const out=[];
          for(let t=0;t<3;t++){const tb=$('[data-ex="tab"][data-i="'+t+'"]'); if(!tb) continue; tb.click(); await w();
            if(t===2){out.push('Tab 3: '+$('.ex-body').innerText); continue;}
            const nq=document.querySelectorAll('#emb-rail [data-exits] .ex-qs .pb-dot').length||1;
            for(let j=0;j<nq;j++){const dot=$('.ex-qs [data-ex="q"][data-j="'+j+'"]'); if(dot){dot.click(); await w();}
              const q=$('.ex-body p b'); out.push('Q: '+(q?q.innerText:''));
              const nopt=document.querySelectorAll('#emb-rail [data-exits] [data-ex="ans"]').length;
              for(let v=0;v<nopt;v++){$('[data-ex="ans"][data-v="'+v+'"]').click(); await w(); const r=$('.ex-res'); out.push('  ['+v+'] '+$('[data-ex="ans"][data-v="'+v+'"]').innerText+' -> '+(r?r.innerText:''));}}}
          return out.join('\\n');}""")
        d.add('Every option and its verdict', verd)
        d.h('Connect'); d.add('Connect', txt(pg, '#emb-app section.connect'))
        if pg.query_selector('#emb-app [data-act="drill-reveal"]'): click(pg, '#emb-app [data-act="drill-reveal"]', 150)
        d.add('Drill with the answer', txt(pg, '#emb-drill-q, #emb-drill-a')); d.tipsOf(pg, '#emb-app')
        b.close()
    (OUT / f'copy-{lang}.md').write_text(f'# embeddings copy deck ({lang})\n' + '\n'.join(d.lines))
    (P / 'out' / 'embeddings-copy').mkdir(parents=True, exist_ok=True)
    (P / 'out' / 'embeddings-copy' / f'{lang}.md').write_text(f'# embeddings copy deck ({lang})\n' + '\n'.join(d.lines))
    print(lang, len(d.lines), 'blocks')


for lang in ('zh', 'en'): run(lang)
