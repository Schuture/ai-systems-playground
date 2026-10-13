"""Dump every piece of text a learner meets in the contamination lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/contamination_copy_deck.py  ->  out/contamination-copy/en.md and out/contamination-copy/zh.md (read by tests/numdense.py)

Order: home tile (each run length, the notes), lab header, each guided part (card before and after its action, the stage, the
metrics), Show the math, the scenarios (each insight), Settings, the challenge (every feedback tier and the chart), the capability
card, and Connect with the drill answer. A tooltip is printed the first time it appears with a given text, and only the first three
of each kind. The lab has no animation; reduced motion is on anyway."""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'contamination-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/contamination.html', '--only', 'contamination'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'wrapped-contamination.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'contamination.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-contamination.html')
clean = lambda s: ' '.join(s.replace('‑', '-').replace(' ', ' ').split()) if s else ''

# Typeset formulas (MathML) are written on one line in a plain form, (a)/(b) for a fraction and √(x) for a root.
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
            c = clean(re.sub(r'<[^>]+>', '', t))
            k = re.sub(r'“[^”]*”|[0-9.,%−-]+', '#', c)[:24]
            if c and c not in self.tips:
                self.tips.add(c); self.kinds[k] = self.kinds.get(k, 0) + 1
                if self.kinds[k] <= 3: new.append('- ' + c)
        if new: self.lines.append('_New tooltips:_\n' + '\n'.join('  ' + x for x in new) + '\n')


def click(pg, sel, ms=200):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(ms)


def seg(pg, key, v, root='#ctm-rail', ms=200):
    click(pg, f'{root} .seg-ctl[data-key="{key}"] button[data-v="{v}"]', ms)


def stage(pg, d, label):
    d.add(label + ' — stage', txt(pg, '#ctm-st-title, #ctm-status, #ctm-stage .ctm-pan .ctm-pt, #ctm-stage .ctm-pan .ctm-leg, #ctm-gtab, #ctm-strips, #ctm-pair'))
    d.add(label + ' — pictures', pg.evaluate("()=>[...document.querySelectorAll('#ctm-stage svg')].filter(e=>e.getClientRects().length).map(s=>(s.getAttribute('aria-label')||'')+' | '+[...s.querySelectorAll('text')].map(t=>t.textContent).join(' | ')).join('\\n')"))
    d.add(label + ' — metric cards', txt(pg, '#ctm-metrics:not([hidden])'))


def run(lang):
    d = Deck()
    with sync_playwright() as pw:
        b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce')
        pg.goto(URL); pg.wait_for_timeout(300); click(pg, f'.langsw button[data-lang="{lang}"]', 150)
        pg.evaluate("()=>{location.hash='map~eval'}"); pg.wait_for_timeout(700)
        if pg.query_selector('[data-teaser="contamination"]'):
            d.h('Home tile')
            for v in ('3', '8', '13'):
                click(pg, f'[data-teaser="contamination"] button[data-v="{v}"]', 120)
                d.add(f'n = {v}', txt(pg, '[data-teaser="contamination"] .tz-read'))
                d.add(f'n = {v} picture', pg.evaluate("()=>document.querySelector('[data-teaser=\"contamination\"] .tz-vis').getAttribute('aria-label')"))
            d.add('Tile', txt(pg, '[data-teaser="contamination"] .tz-lab, [data-teaser="contamination"] .seg-ctl, [data-teaser="contamination"] .tz-leg'))
            d.add('Notes', pg.evaluate("()=>[...document.querySelectorAll('[data-teaser=\"contamination\"] .tz-note')].map(e=>e.textContent).join('\\n')"))
        pg.evaluate("()=>{location.hash='contamination'}"); pg.wait_for_timeout(500)
        d.h('Header'); d.add('Header', txt(pg, '#ctm-app .lab-head, #ctm-app header, #ctm-app .mode-desc')); d.tipsOf(pg, '#ctm-app .modes')
        d.add('Fidelity note', pg.evaluate("()=>{const e=document.querySelector('#ctm-app [id$=\"-fid\"]'); return e?e.innerText:''}"))
        d.add('Primer note', pg.evaluate("()=>{const e=document.querySelector('#ctm-app [id$=\"-pri\"]'); return e?e.innerText:''}"))
        for i in range(7):
            d.h(f'Guided part {i + 1}')
            d.add('Card', txt(pg, '#ctm-rail .step.current')); d.tipsOf(pg, '#ctm-rail')
            stage(pg, d, f'Part {i + 1}'); d.tipsOf(pg, '#ctm-stage')
            st = pg.evaluate("()=>window.__ctm.state()"); acted = False
            if st['step'] == 1:
                for v in range(6):
                    seg(pg, 's', v); d.add(f'Sample {v + 1}', txt(pg, '#ctm-rail [data-obs]') + '\n' + txt(pg, '#ctm-pair'))
                acted = True
            if st['step'] == 2 and pg.query_selector('#ctm-rail [data-pred="more"]'):
                click(pg, '#ctm-rail [data-pred="long"]'); d.add('Picked "long" (wrong)', txt(pg, '#ctm-rail .step.current'))
                click(pg, '#ctm-rail [data-pred="more"]'); acted = True
            if st['step'] == 3:
                for v in ('5', '8', '13'):
                    seg(pg, 'n', v); d.add(f'After n = {v}', txt(pg, '#ctm-rail [data-obs]'))
                seg(pg, 't', '20'); d.add('After ≥ 20%', txt(pg, '#ctm-rail [data-obs]')); acted = True
            if st['step'] == 4:
                for v in ('3', '8', '13'):
                    seg(pg, 'n', v); d.add(f'After n = {v}', txt(pg, '#ctm-rail [data-obs]'))
                acted = True
            if st['step'] == 5:
                for v in ('all', 'clean'):
                    click(pg, f'#ctm-rail [data-lever="{v}"]'); d.add(f'After rule {v}', txt(pg, '#ctm-rail [data-obs]'))
                if pg.query_selector('#ctm-rail .seg-ctl[data-key="m"]'): seg(pg, 'm', '8'); d.add('After margin 8', txt(pg, '#ctm-rail [data-obs]'))
                acted = True
            if acted: d.add('Card after the action', txt(pg, '#ctm-rail .step.current')); stage(pg, d, f'Part {i + 1} after'); d.tipsOf(pg, '#ctm-app')
            nb = pg.query_selector('#ctm-rail .step.current [data-nav="next"]')
            if nb and not nb.is_disabled(): click(pg, '#ctm-rail .step.current [data-nav="next"]', 300)
        d.add('Next-lab footer', txt(pg, '.next-lab'))
        d.h('Show the math')
        pg.evaluate("()=>{const m=document.querySelector('#ctm-math'); if(m) m.open=true}"); pg.wait_for_timeout(150); d.add('Math', txt(pg, '#ctm-math')); d.tipsOf(pg, '#ctm-math')
        d.h('Scenarios')
        click(pg, '#ctm-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 400)
        d.add('Rail', txt(pg, '#ctm-rail .railcard'))
        for i in range(6):
            click(pg, f'#ctm-rail [data-preset="{i}"]', 300); d.add(f'Scenario {i + 1} insight', txt(pg, '#ctm-insight'))
            stage(pg, d, f'Scenario {i + 1}')
        d.tipsOf(pg, '#ctm-stage')
        d.h('Settings'); d.add('Settings', txt(pg, '#ctm-controls')); d.tipsOf(pg, '#ctm-controls')
        pg.evaluate("()=>{const m=document.querySelector('#ctm-controls details'); if(m) m.open=true}"); pg.wait_for_timeout(100); d.add('Lab assumptions', txt(pg, '#ctm-controls details'))
        d.h('Challenge')
        click(pg, '#ctm-app .modes button[data-mode="challenge"]', 400)
        d.add('Card', txt(pg, '#ctm-rail .railcard')); d.add('Settings note', txt(pg, '#ctm-lock'))
        stage(pg, d, 'Challenge before Submit')
        click(pg, '#ctm-rail [data-act="submit"]', 300); d.add('Initial setup submitted (not met)', txt(pg, '#ctm-ch-res')); stage(pg, d, 'Challenge after Submit')
        seg(pg, 'n', '3'); click(pg, '#ctm-rail [data-act="submit"]', 300); d.add('n = 3, any, margin 3 submitted', txt(pg, '#ctm-ch-res'))
        seg(pg, 't', '20'); seg(pg, 'm', '5'); click(pg, '#ctm-rail [data-act="submit"]', 300); d.add('n = 3, ≥ 20%, margin 5 submitted (met)', txt(pg, '#ctm-ch-res'))
        seg(pg, 'n', '8'); seg(pg, 't', '0'); click(pg, '#ctm-rail [data-act="submit"]', 300); d.add('n = 8, any, margin 5 submitted (best)', txt(pg, '#ctm-ch-res'))
        if pg.query_selector('#ctm-rail [data-act="reveal"]'): click(pg, '#ctm-rail [data-act="reveal"]', 300); d.add('Show the answer', txt(pg, '#ctm-ch-res'))
        d.tipsOf(pg, '#ctm-rail')
        d.h('Capability card'); d.add('Card', txt(pg, '#ctm-rail [data-exits]')); d.tipsOf(pg, '#ctm-rail [data-exits]')
        verd = pg.evaluate("""async ()=>{const w=()=>new Promise(r=>setTimeout(r,30)); const $=s=>document.querySelector('#ctm-rail [data-exits] '+s); const out=[];
          for(let t=0;t<3;t++){const tb=$('[data-ex="tab"][data-i="'+t+'"]'); if(!tb) continue; tb.click(); await w();
            if(t===2){out.push('Tab 3: '+$('.ex-body').innerText); continue;}
            const nq=document.querySelectorAll('#ctm-rail [data-exits] .ex-qs .pb-dot').length||1;
            for(let j=0;j<nq;j++){const dot=$('.ex-qs [data-ex="q"][data-j="'+j+'"]'); if(dot){dot.click(); await w();}
              const q=$('.ex-body p b'); out.push('Q: '+(q?q.innerText:''));
              const nopt=document.querySelectorAll('#ctm-rail [data-exits] [data-ex="ans"]').length;
              for(let v=0;v<nopt;v++){$('[data-ex="ans"][data-v="'+v+'"]').click(); await w(); const r=$('.ex-res'); out.push('  ['+v+'] '+$('[data-ex="ans"][data-v="'+v+'"]').innerText+' -> '+(r?r.innerText:''));}}}
          return out.join('\\n');}""")
        d.add('Every option and its verdict', verd)
        d.h('Connect'); d.add('Connect', txt(pg, '#ctm-app section.connect'))
        if pg.query_selector('#ctm-app [data-act="drill-reveal"]'): click(pg, '#ctm-app [data-act="drill-reveal"]', 150)
        d.add('Drill with the answer', txt(pg, '#ctm-drill-q, #ctm-drill-a')); d.tipsOf(pg, '#ctm-app')
        b.close()
    (OUT / f'{lang}.md').write_text(f'# contamination copy deck ({lang})\n' + '\n'.join(d.lines))
    print(lang, len(d.lines), 'blocks')


for lang in ('zh', 'en'): run(lang)
