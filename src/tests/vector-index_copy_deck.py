"""Dump every piece of text a learner meets in the vector-index lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/vector-index_copy_deck.py  ->  out/vector-index/copy-en.md and out/vector-index/copy-zh.md (also out/vector-index-copy/en.md
and zh.md for tests/numdense.py; each block is a ### heading, so numdense skips the stage, chart, metric and mini-bar blocks)

Order: home tile (every nprobe, the note), lab header, each guided part (card before and after its action, the stage, the
metrics), Show the math, the scenarios (each insight), Settings, the challenge (every feedback tier and the chart), the capability
card, and Connect with the drill answer. A tooltip is printed the first time it appears with a given text, and only the first
three of each kind. Reduced motion is on, so every search in the picture ends at once."""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'vector-index'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/vector-index.html', '--only', 'vector-index'], cwd=P, check=True, capture_output=True)
(OUT / 'deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'vector-index.html').read_text() + '</body></html>')
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
    e.querySelectorAll('.vix-eqs').forEach(g=>{ const c=[...g.children], d=document.createElement('div');
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
        if t: self.lines.append(f'### {label}\n\n' + '\n'.join(clean(x) for x in t.split('\n') if clean(x)) + '\n')

    def tipsOf(self, pg, root):
        new = []
        for t in pg.evaluate("r=>[...document.querySelectorAll(r+' [data-tipx]')].filter(e=>e.getClientRects().length).map(e=>e.getAttribute('data-tipx'))", root):
            c = clean(t)
            k = re.sub(r'“[^”]*”|[0-9.,%−-]+', '#', c)[:24]
            if c and c not in self.tips:
                self.tips.add(c); self.kinds[k] = self.kinds.get(k, 0) + 1
                if self.kinds[k] <= 3: new.append('- ' + c)
        if new: self.lines.append('_New tooltips:_\n' + '\n'.join(new) + '\n')


def click(pg, sel, ms=200):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(ms)


def stage(pg, d, label):
    # each panel shown: its title, the picture's caption, legend and note, the scale bars and the k-d bars (review polish r1)
    d.add(label + ' — stage', txt(pg, '#vix-stage .vix-head, #vix-stage .vix-pan:not([hidden]) > .vix-pt, #vix-stage .vix-pan:not([hidden]) .vix-cap, #vix-stage .vix-pan:not([hidden]) .vix-leg, #vix-stage .vix-pan:not([hidden]) .vix-pnote, #vix-stage .vix-pan:not([hidden]) .vix-scale, #vix-stage .vix-pan:not([hidden]) .vix-kdbars'))
    d.add(label + ' — stage picture label', pg.evaluate("()=>{const s=document.querySelector('#vix-pic'); return s&&s.getClientRects().length?s.getAttribute('aria-label'):''}"))
    d.add(label + ' — chart labels', pg.evaluate("()=>[...document.querySelectorAll('#vix-stage svg')].filter(e=>e.getClientRects().length).map(s=>[...s.querySelectorAll('text')].map(t=>t.textContent).join(' | ')).join('\\n')"))
    d.add(label + ' — metric cards', txt(pg, '#vix-metrics'))


def res(pg, d, label):
    # the challenge result: its text, then its chart as a block of its own (numdense skips chart blocks: its rows read out values)
    d.add(label, txt(pg, '#vix-ch-res > :not(.cmp)')); d.add(label + ' — the chart', txt(pg, '#vix-ch-res .cmp'))


def dock_text(pg):
    # the kit renders the dock only while it shows, so scroll the stage away first (as a reader would), read it, and come back
    pg.evaluate("()=>{const a=document.querySelector('#vix-pan-pic:not([hidden])')||document.querySelector('#vix-pan-chart'); window.scrollTo(0,a.getBoundingClientRect().bottom+scrollY+400)}"); pg.wait_for_timeout(500)
    t = pg.evaluate("()=>{const d=document.getElementById('dock'); return d&&!d.hidden&&d.getBoundingClientRect().height>0?d.innerText:''}")
    pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(150)
    return t


def run(lang):
    d = Deck()
    with sync_playwright() as pw:
        b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce')
        pg.goto(URL); pg.wait_for_timeout(300); click(pg, f'.langsw button[data-lang="{lang}"]', 150)
        # home tile
        pg.evaluate("()=>{location.hash='map~apps'}"); pg.wait_for_timeout(700)
        if pg.query_selector('[data-teaser="vector-index"]'):
            d.h('Home tile')
            for v in ('1', '4', '16'):
                click(pg, f'[data-teaser="vector-index"] button[data-v="{v}"]', 120)
                d.add(f'nprobe {v}', txt(pg, '[data-teaser="vector-index"] .tz-read'))
            d.add('Tile', txt(pg, '[data-teaser="vector-index"]'))
            d.add('Tile picture', pg.evaluate("()=>{const s=document.querySelector('[data-teaser=\"vector-index\"] svg'); return s?s.getAttribute('aria-label')+' | '+[...s.querySelectorAll('text')].map(t=>t.textContent).join(' | '):''}"))
        pg.evaluate("()=>{location.hash='vector-index'}"); pg.wait_for_timeout(500)
        d.h('Header'); d.add('Header', txt(pg, '#vix-app .lab-head, #vix-app header, #vix-app .modes')); d.tipsOf(pg, '#vix-app .modes')
        d.add('Fidelity note', pg.evaluate("()=>{const e=document.querySelector('#vix-app [id$=\"-fid\"]'); return e?e.innerText:''}"))
        d.add('Primer note', pg.evaluate("()=>{const e=document.querySelector('#vix-app [id$=\"-pri\"]'); return e?e.innerText:''}"))
        for i in range(8):
            d.h(f'Guided part {i + 1}')
            d.add('Card', txt(pg, '#vix-rail .step.current')); d.tipsOf(pg, '#vix-rail')
            stage(pg, d, f'Part {i + 1}'); d.tipsOf(pg, '#vix-stage')
            d.add('Mini result bar (dock)', dock_text(pg))
            acted = False
            st = pg.evaluate("()=>window.__vix.state()")
            seg = {1: ('rn', ('1e8', '1e9')), 2: ('kd', ('8', '32')), 4: ('np', ('2', '4', '8', '16', '32')), 5: ('ef', ('16', '32', '64', '128', '256')), 6: ('m', ('4', '16'))}.get(st['step'])
            if seg:
                for v in seg[1]:
                    click(pg, f'#vix-rail .seg-ctl[data-key="{seg[0]}"] button[data-v="{v}"]', 250); d.add(f'After {seg[0]} = {v}', txt(pg, '#vix-rail [data-obs]'))
                acted = True
            if pg.query_selector('#vix-rail [data-pred="all"]'): click(pg, '#vix-rail [data-pred="all"]'); acted = True
            if acted: d.add('Card after the action', txt(pg, '#vix-rail .step.current')); stage(pg, d, f'Part {i + 1} after'); d.tipsOf(pg, '#vix-app')
            nb = pg.query_selector('#vix-rail .step.current [data-nav="next"]')
            if nb and not nb.is_disabled(): click(pg, '#vix-rail .step.current [data-nav="next"]', 300)
        d.add('Next-lab footer', txt(pg, '.next-lab'))
        d.h('Show the math')
        pg.evaluate("()=>{const m=document.querySelector('#vix-math'); if(m) m.open=true}"); pg.wait_for_timeout(150); d.add('Show the math', txt(pg, '#vix-math')); d.tipsOf(pg, '#vix-math')
        d.h('Scenarios')
        click(pg, '#vix-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 400)
        d.add('Rail', txt(pg, '#vix-rail .railcard, #vix-rail .presets'))
        for i in range(6):
            click(pg, f'#vix-rail [data-preset="{i}"]', 300); d.add(f'Scenario {i + 1} insight', txt(pg, '#vix-insight')); stage(pg, d, f'Scenario {i + 1}')
        d.h('Settings'); d.add('Settings', txt(pg, '#vix-controls')); d.tipsOf(pg, '#vix-controls')
        pg.evaluate("()=>{const m=document.querySelector('#vix-controls details'); if(m) m.open=true}"); pg.wait_for_timeout(100); d.add('Settings — fixed in this lab', txt(pg, '#vix-controls details'))
        d.h('Challenge')
        click(pg, '#vix-app .modes button[data-mode="challenge"]', 400)
        d.add('Card', txt(pg, '#vix-rail .railcard'))
        d.add('Before Submit — goal lines, stage title and status', txt(pg, '#vix-ch-a, #vix-ch-c, #vix-ch-g, #vix-st-title, #vix-status, #vix-c-title'))
        click(pg, '#vix-rail [data-act="submit"]', 300)
        d.add('Exact search submitted', txt(pg, '#vix-ch-res, #vix-status'))
        click(pg, '#vix-rail [data-act="reveal"]', 300); d.add('Exact search, answer shown', txt(pg, '#vix-ch-res > :not(.cmp)')); d.add('Exact search, answer shown — the chart', txt(pg, '#vix-ch-res .cmp'))
        click(pg, '#vix-rail .seg-ctl[data-key="ix"] button[data-v="pq"]', 300); click(pg, '#vix-rail [data-act="submit"]', 300)
        res(pg, d, 'IVF-PQ submitted')
        click(pg, '#vix-rail .seg-ctl[data-key="ix"] button[data-v="hnsw"]', 300); click(pg, '#vix-rail .seg-ctl[data-key="M"] button[data-v="32"]', 300); click(pg, '#vix-rail .seg-ctl[data-key="ef"] button[data-v="32"]', 300)
        click(pg, '#vix-rail [data-act="submit"]', 300); res(pg, d, 'HNSW M = 32, ef = 32 submitted')
        click(pg, '#vix-rail .seg-ctl[data-key="M"] button[data-v="16"]', 300); click(pg, '#vix-rail .seg-ctl[data-key="ef"] button[data-v="64"]', 300)
        click(pg, '#vix-rail [data-act="submit"]', 300); d.add('HNSW M = 16, ef = 64 submitted (and the chart)', txt(pg, '#vix-ch-res'))
        stage(pg, d, 'Challenge after Submit')
        d.h('Capability card'); d.add('Card', txt(pg, '#vix-rail [data-exits]')); d.tipsOf(pg, '#vix-rail [data-exits]')
        verd = pg.evaluate("""async ()=>{const w=()=>new Promise(r=>setTimeout(r,30)); const $=s=>document.querySelector('#vix-rail [data-exits] '+s); const out=[];
          for(let t=0;t<3;t++){const tb=$('[data-ex="tab"][data-i="'+t+'"]'); if(!tb) continue; tb.click(); await w();
            if(t===2){out.push('Tab 3: '+$('.ex-body').innerText); continue;}
            const nq=document.querySelectorAll('#vix-rail [data-exits] .ex-qs .pb-dot').length||1;
            for(let j=0;j<nq;j++){const dot=$('.ex-qs [data-ex="q"][data-j="'+j+'"]'); if(dot){dot.click(); await w();}
              const q=$('.ex-body p b'); out.push('Q: '+(q?q.innerText:''));
              const nopt=document.querySelectorAll('#vix-rail [data-exits] [data-ex="ans"]').length;
              for(let v=0;v<nopt;v++){$('[data-ex="ans"][data-v="'+v+'"]').click(); await w(); const r=$('.ex-res'); out.push('  ['+'abcd'[v]+'] '+$('[data-ex="ans"][data-v="'+v+'"]').innerText+'\\n    -> '+(r?r.innerText:''));}}}
          return out.join('\\n');}""")
        d.add('Every option and its verdict', verd)
        d.h('Connect'); d.add('Connect', txt(pg, '#vix-app section.connect'))
        if pg.query_selector('#vix-app [data-act="drill-reveal"]'): click(pg, '#vix-app [data-act="drill-reveal"]', 150)
        d.add('Drill with the answer', txt(pg, '#vix-drill-q, #vix-drill-a')); d.tipsOf(pg, '#vix-app')
        b.close()
    (OUT / f'copy-{lang}.md').write_text(f'# vector-index copy deck ({lang})\n' + '\n'.join(d.lines))
    (P / 'out' / 'vector-index-copy').mkdir(parents=True, exist_ok=True)   # the copy tests/numdense.py reads
    (P / 'out' / 'vector-index-copy' / f'{lang}.md').write_text(f'# vector-index copy deck ({lang})\n' + '\n'.join(d.lines))
    print(lang, len(d.lines), 'blocks')


for lang in ('zh', 'en'): run(lang)
