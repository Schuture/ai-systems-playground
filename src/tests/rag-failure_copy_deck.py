"""Dump every piece of text a learner meets in the rag-failure lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/rag-failure_copy_deck.py  ->  out/rag-failure-copy/en.md and out/rag-failure-copy/zh.md (read by tests/numdense.py)

Order: home tile (each stage, the notes), lab header, each guided part (card before and after its action, the stage, the
breakdown, the metric cards), Show the math, the scenarios (each insight), Settings, the challenge (every feedback tier and the
chart), the capability card, Connect with the drill answer, and what only a phone (390) shows: the dock's lines and the buttons under the
cards in Scenarios and Settings. A tooltip is printed the first time it appears with a given text, and only the first three of each kind.
Reduced motion is on at 1440, so every cut ends at once; it is off at 390, where a new run waits for ▶."""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'rag-failure-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/rag-failure.html', '--only', 'rag-failure'], cwd=P, check=True, capture_output=True)
(OUT / 'deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'rag-failure.html').read_text() + '</body></html>')
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
    e.querySelectorAll('.rgf-eqs').forEach(g=>{ const c=[...g.children], d=document.createElement('div');
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
    d.add(label + ' — stage', txt(pg, '#rgf-stage .rgf-head, #rgf-stage #rgf-pan-sum:not([hidden]) #rgf-two, #rgf-stage .rgf-pan:not([hidden]) .rgf-pt, #rgf-stage #rgf-pan-run:not([hidden]) #rgf-runtog, #rgf-stage #rgf-pan-run:not([hidden]) .rgf-ql, #rgf-stage #rgf-pan-run:not([hidden]) #rgf-rnote:not([hidden]), #rgf-stage #rgf-pan-run:not([hidden]) #rgf-flow, #rgf-stage #rgf-pan-run:not([hidden]) #rgf-r-leg, #rgf-stage #rgf-pan-bars:not([hidden]) #rgf-bars, #rgf-stage #rgf-pan-bars:not([hidden]) #rgf-b-leg'))
    d.add(label + ' — metric cards', txt(pg, '#rgf-metrics:not([hidden])'))


SEG = lambda k, v: f'#rgf-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]'
DOCK = "()=>{const d=document.getElementById('dock'); return d&&!d.hidden&&d.getBoundingClientRect().height>0?d.innerText:''}"


def phone(b, d, lang):
    """390 only (round-1 check: P18, N1, N6): the dock's lines and the buttons under the cards"""
    pg = b.new_page(viewport={'width': 390, 'height': 844})
    pg.goto(URL); pg.wait_for_timeout(300); click(pg, f'.langsw button[data-lang="{lang}"]', 150)
    pg.evaluate("()=>{location.hash='rag-failure'}"); pg.wait_for_timeout(500)

    def below():
        pg.evaluate("()=>{const a=document.querySelector('#rgf-stage'); window.scrollTo(0,a.getBoundingClientRect().bottom+scrollY-20)}")
        try: pg.wait_for_function("()=>{const d=document.getElementById('dock'); return d&&!d.hidden&&d.getBoundingClientRect().height>0}", timeout=3000)
        except Exception: pass
        pg.wait_for_timeout(400)
        return pg.evaluate(DOCK)
    d.h('Phone (390): the dock and the buttons under the cards')
    d.add('Dock while the first run waits for ▶', below())
    click(pg, '#rgf-rail [data-go="2"]', 300); click(pg, '#rgf-rail [data-tri="cut"]', 200); click(pg, '#rgf-rail [data-trigo="1"]', 200)
    pg.evaluate("()=>{const i=document.querySelector('#rgf-tr-p-in'); i.value=100; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(200)
    click(pg, '#rgf-rail [data-tri="stale"]', 200)
    d.add('Dock after the right pick on the outdated-page run', below())
    click(pg, '#rgf-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 400)
    d.add('Scenarios: the buttons under the cards', txt(pg, '#rgf-pk'))
    click(pg, '#rgf-pk [data-pick="2"]', 300)
    d.add('Dock with a scenario’s fixes on', below())
    click(pg, '#rgf-app .modes button[data-flow="free"]', 400)
    d.add('Settings: the buttons under the cards', txt(pg, '#rgf-pk'))
    pg.close()


def run(lang):
    d = Deck()
    with sync_playwright() as pw:
        b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce')
        pg.goto(URL); pg.wait_for_timeout(300); click(pg, f'.langsw button[data-lang="{lang}"]', 150)
        pg.evaluate("()=>{location.hash='map~apps'}"); pg.wait_for_timeout(700)
        if pg.query_selector('[data-teaser="rag-failure"]'):
            d.h('Home tile')
            for v in range(4):
                if pg.query_selector(f'[data-teaser="rag-failure"] button[data-v="{v}"]'):
                    click(pg, f'[data-teaser="rag-failure"] button[data-v="{v}"]', 120)
                    d.add(f'Option {v}', txt(pg, '[data-teaser="rag-failure"] .tz-read'))
            d.add('Tile', txt(pg, '[data-teaser="rag-failure"]'))
            d.add('Tile picture', pg.evaluate("()=>{const s=document.querySelector('[data-teaser=\"rag-failure\"] svg'); return s?s.getAttribute('aria-label'):''}"))
        pg.evaluate("()=>{location.hash='rag-failure'}"); pg.wait_for_timeout(500)
        d.h('Header'); d.add('Header', txt(pg, '#rgf-app .lab-head, #rgf-app header, #rgf-app .modes')); d.tipsOf(pg, '#rgf-app .modes')
        d.add('Fidelity note', pg.evaluate("()=>{const e=document.querySelector('#rgf-app [id$=\"-fid\"]'); return e?e.innerText:''}"))
        d.add('Primer note', pg.evaluate("()=>{const e=document.querySelector('#rgf-app [id$=\"-pri\"]'); return e?e.innerText:''}"))
        d.add('Transport', txt(pg, '#rgf-transport')); d.tipsOf(pg, '#rgf-transport')
        d.add('The play button over the record (phones, before a new run plays)', pg.evaluate("()=>{const e=document.querySelector('#rgf-go'); return e?e.textContent:''}"))
        d.add('The seven runs', pg.evaluate("()=>[...document.querySelectorAll('#rgf-rsel option')].map(o=>o.textContent).join('\\n')"))
        for i in range(7):
            d.h(f'Guided part {i + 1}')
            d.add('Card', txt(pg, '#rgf-rail .step.current')); d.tipsOf(pg, '#rgf-rail')
            stage(pg, d, f'Part {i + 1}'); d.tipsOf(pg, '#rgf-stage')
            acted = False
            st = pg.evaluate("()=>window.__rgf.state()")
            if st['step'] == 1:
                click(pg, SEG('log', '1'), 200); acted = True
            if st['step'] == 2:
                click(pg, '#rgf-rail [data-tri="unsup"]', 200); d.add('Run 1, a wrong pick (unsupported)', txt(pg, '#rgf-rail .rgf-verd')); stage(pg, d, 'Run 1 after the pick')
                for j, kind in ((1, 'stale'), (2, 'unsup')):
                    click(pg, f'#rgf-rail [data-trigo="{j}"]', 300); d.add(f'Run {j + 1} before the pick', txt(pg, '#rgf-rail .step.current')); stage(pg, d, f'Run {j + 1} before the pick')
                    click(pg, f'#rgf-rail [data-tri="{kind}"]', 200); d.add(f'Run {j + 1}, the right pick', txt(pg, '#rgf-rail .rgf-verd')); stage(pg, d, f'Run {j + 1} after the pick')
                acted = True
            if st['step'] == 3:
                for v in ('after', 'even'):
                    click(pg, f'#rgf-rail [data-pred="{v}"]', 150); d.add(f'Prediction {v}', txt(pg, '#rgf-rail .step.current'))
                click(pg, '#rgf-rail [data-pred="ret"]', 150); acted = True
            if st['step'] == 4:
                for v in ('ret', 'ctx', 'grd', 'cor'):
                    click(pg, SEG('met', v), 200); d.add(f'Check {v}', txt(pg, '#rgf-rail [data-obs]')); d.add(f'Breakdown with {v}', txt(pg, '#rgf-bars'))
                acted = True
            if st['step'] == 5:
                for f in ('hyb', 'ctx', 'ver', 'cite', 'dec'):
                    click(pg, f'#rgf-rail [data-lever="{f}"]', 200); d.add(f'Fix {f} on', txt(pg, '#rgf-rail [data-obs]'))
                    click(pg, f'#rgf-rail [data-lever="{f}"]', 200)
                click(pg, '#rgf-rail [data-lever="ver"]', 200); acted = True
            if acted: d.add('Card after the action', txt(pg, '#rgf-rail .step.current')); stage(pg, d, f'Part {i + 1} after'); d.tipsOf(pg, '#rgf-app')
            nb = pg.query_selector('#rgf-rail .step.current [data-nav="next"]')
            if nb and not nb.is_disabled(): click(pg, '#rgf-rail .step.current [data-nav="next"]', 300)
        d.add('Next-lab footer', txt(pg, '.next-lab'))
        d.h('Show the math')
        pg.evaluate("()=>{const m=document.querySelector('#rgf-math'); if(m) m.open=true}"); pg.wait_for_timeout(150); d.add('Math', txt(pg, '#rgf-math')); d.tipsOf(pg, '#rgf-math')
        d.h('Scenarios')
        click(pg, '#rgf-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 400)
        d.add('Rail', txt(pg, '#rgf-rail .railcard'))
        for i in range(5):
            click(pg, f'#rgf-rail [data-preset="{i}"]', 300); d.add(f'Scenario {i + 1} insight', txt(pg, '#rgf-insight')); d.add(f'Scenario {i + 1} status', txt(pg, '#rgf-st-title, #rgf-status'))
        stage(pg, d, 'Scenario 5')
        if pg.query_selector('#rgf-runtog'):
            click(pg, '#rgf-runtog', 300)
            d.add('Scenario 5 — the run record opened', txt(pg, '#rgf-stage #rgf-runtog, #rgf-stage #rgf-pan-run .rgf-ql, #rgf-stage #rgf-rnote:not([hidden]), #rgf-stage #rgf-flow, #rgf-stage #rgf-r-leg')); d.tipsOf(pg, '#rgf-stage')
        d.h('Settings'); d.add('Settings', txt(pg, '#rgf-controls')); d.tipsOf(pg, '#rgf-controls')
        pg.evaluate("()=>{const m=document.querySelector('#rgf-controls details'); if(m) m.open=true}"); pg.wait_for_timeout(100); d.add('Lab assumptions', txt(pg, '#rgf-controls details'))
        d.h('Challenge')
        click(pg, '#rgf-app .modes button[data-mode="challenge"]', 400)
        d.add('Card', txt(pg, '#rgf-rail .railcard')); d.add('Settings note', txt(pg, '#rgf-lock'))
        d.add('Before Submit — goal lines and stage status', txt(pg, '#rgf-ch-c, #rgf-ch-w, #rgf-ch-n, #rgf-status'))
        click(pg, '#rgf-rail [data-chfx="ver"]'); click(pg, '#rgf-rail [data-chfx="dec"]')
        click(pg, '#rgf-rail [data-act="submit"]', 300); d.add('Version filter + split the question submitted (over the wait)', txt(pg, '#rgf-ch-res, #rgf-status'))
        if pg.query_selector('#rgf-rail [data-act="reveal"]'):
            click(pg, '#rgf-rail [data-act="reveal"]', 200); d.add('Show the answer', txt(pg, '#rgf-ch-res'))
        click(pg, '#rgf-rail [data-chfx="dec"]'); click(pg, '#rgf-rail [data-chfx="ctx"]'); click(pg, '#rgf-rail [data-act="submit"]', 300)
        d.add('Version filter + context order submitted (met, not best)', txt(pg, '#rgf-ch-res'))
        click(pg, '#rgf-rail [data-chfx="ctx"]'); click(pg, '#rgf-rail [data-chfx="hyb"]'); click(pg, '#rgf-rail [data-act="submit"]', 300)
        d.add('Hybrid + version filter submitted (best, with the chart)', txt(pg, '#rgf-ch-res'))
        d.h('Capability card'); d.add('Card', txt(pg, '#rgf-rail [data-exits]')); d.tipsOf(pg, '#rgf-rail [data-exits]')
        verd = pg.evaluate("""async ()=>{const w=()=>new Promise(r=>setTimeout(r,30)); const $=s=>document.querySelector('#rgf-rail [data-exits] '+s); const out=[];
          for(let t=0;t<3;t++){const tb=$('[data-ex="tab"][data-i="'+t+'"]'); if(!tb) continue; tb.click(); await w();
            if(t===2){out.push('Tab 3: '+$('.ex-body').innerText); continue;}
            const nq=document.querySelectorAll('#rgf-rail [data-exits] .ex-qs .pb-dot').length||1;
            for(let j=0;j<nq;j++){const dot=$('.ex-qs [data-ex="q"][data-j="'+j+'"]'); if(dot){dot.click(); await w();}
              const q=$('.ex-body p b'); out.push('Q: '+(q?q.innerText:''));
              const nopt=document.querySelectorAll('#rgf-rail [data-exits] [data-ex="ans"]').length;
              for(let v=0;v<nopt;v++){$('[data-ex="ans"][data-v="'+v+'"]').click(); await w(); const r=$('.ex-res'); out.push('  ('+'abcdef'[v]+') '+$('[data-ex="ans"][data-v="'+v+'"]').innerText+' -> '+(r?r.innerText:''));}}}
          return out.join('\\n');}""")
        d.add('Every option and its verdict', verd)
        d.h('Connect'); d.add('Connect', txt(pg, '#rgf-app section.connect'))
        if pg.query_selector('#rgf-app [data-act="drill-reveal"]'): click(pg, '#rgf-app [data-act="drill-reveal"]', 150)
        d.add('Drill with the answer', txt(pg, '#rgf-drill-q, #rgf-drill-a')); d.tipsOf(pg, '#rgf-app')
        phone(b, d, lang)
        b.close()
    (OUT / f'{lang}.md').write_text(f'# rag-failure copy deck ({lang})\n' + '\n'.join(d.lines))
    print(lang, len(d.lines), 'blocks')


for lang in ('zh', 'en'): run(lang)
