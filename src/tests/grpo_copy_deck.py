"""Dump every piece of text a learner meets in the grpo lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/grpo_copy_deck.py  ->  out/grpo/copy-en.md and out/grpo/copy-zh.md

Order: home tile (every G, the note), lab header, each guided part (card before and after its action, the stage, the metrics), Show the
math, the scenarios (each insight), Settings, the challenge (every feedback tier and the chart), the capability card, and Connect with
the drill answer. A tooltip is printed the first time it appears with a given text. Reduced motion is on, so every replay ends at once."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'grpo'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/grpo.html', '--only', 'grpo'], cwd=P, check=True, capture_output=True)
(OUT / 'deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'grpo.html').read_text() + '</body></html>')
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
    e.querySelectorAll('.grpo-eqs').forEach(g=>{ const c=[...g.children], d=document.createElement('div');
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
        if t: self.lines.append(f'**{label}**\n\n' + '\n'.join(clean(x) for x in t.split('\n') if clean(x)) + '\n')

    def tipsOf(self, pg, root):
        new = []
        for t in pg.evaluate("r=>[...document.querySelectorAll(r+' [data-tipx]')].filter(e=>e.getClientRects().length).map(e=>e.getAttribute('data-tipx'))", root):
            c = clean(t)
            if c and c not in self.tips: self.tips.add(c); new.append('- ' + c)
        if new: self.lines.append('Tooltips (new):\n' + '\n'.join(new) + '\n')


def click(pg, sel, ms=200):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(ms)


def stage(pg, d, label):
    d.add(label + ' — stage', txt(pg, '#grpo-stage .grpo-head, #grpo-stage .grpo-pan:not([hidden]) .grpo-pt, #grpo-stage .grpo-pan:not([hidden]) .grpo-leg, #grpo-stage .grpo-gnote, #grpo-stage .grpo-ckey, #grpo-stage .grpo-mnote, #grpo-stage .grpo-mname, #grpo-stage .grpo-mfz'))
    d.add(label + ' — chart labels', pg.evaluate("()=>[...document.querySelectorAll('#grpo-stage svg')].filter(e=>e.getClientRects().length).map(s=>[...s.querySelectorAll('text')].map(t=>t.textContent).join(' | ')).join('\\n')"))
    d.add(label + ' — metric cards', txt(pg, '#grpo-metrics'))


def run(lang):
    d = Deck()
    with sync_playwright() as pw:
        b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce')
        pg.goto(URL); pg.wait_for_timeout(300); click(pg, f'.langsw button[data-lang="{lang}"]', 150)
        # home tile
        pg.evaluate("()=>{location.hash='map~post'}"); pg.wait_for_timeout(700)   # the post-training branch shows the grpo tile
        if pg.query_selector('[data-teaser="grpo"]'):
            d.h('Home tile')
            for g in ('2', '4', '8', '16', '32'):
                if pg.query_selector(f'[data-teaser="grpo"] button[data-v="{g}"]'):
                    click(pg, f'[data-teaser="grpo"] button[data-v="{g}"]', 120)
                d.add(f'G = {g}', txt(pg, '[data-teaser="grpo"] .tz-read'))
            d.add('Tile', txt(pg, '[data-teaser="grpo"]'))
        pg.evaluate("()=>{location.hash='grpo'}"); pg.wait_for_timeout(500)
        d.h('Header'); d.add('Header', txt(pg, '#grpo-app .lab-head, #grpo-app header, #grpo-app .modes')); d.tipsOf(pg, '#grpo-app .modes')
        n = pg.evaluate("()=>document.querySelectorAll('#grpo-rail .step').length") or 8
        for i in range(8):
            d.h(f'Guided part {i + 1}')
            d.add('Card', txt(pg, '#grpo-rail .step.current')); d.tipsOf(pg, '#grpo-rail')
            stage(pg, d, f'Part {i + 1}'); d.tipsOf(pg, '#grpo-stage')
            acted = False
            if pg.query_selector('#grpo-rail [data-act="gen"]'): click(pg, '#grpo-rail [data-act="gen"]', 300); acted = True
            if pg.query_selector('#grpo-rail [data-pred="same"]'): click(pg, '#grpo-rail [data-pred="same"]'); acted = True
            if pg.query_selector('#grpo-rail .seg-ctl[data-key="peq"]'):
                for g in ('2', '64'):
                    click(pg, f'#grpo-rail .seg-ctl[data-key="peq"] button[data-v="{g}"]'); d.add(f'After G = {g}', txt(pg, '#grpo-rail [data-obs]'))
            if pg.query_selector('#grpo-rail .seg-ctl[data-key="std"]'):
                click(pg, '#grpo-rail .seg-ctl[data-key="std"] button[data-v="off"]', 300); d.add('After “do not divide”', txt(pg, '#grpo-rail .step.current')); stage(pg, d, 'Without the division')
                click(pg, '#grpo-rail .seg-ctl[data-key="std"] button[data-v="on"]', 300); d.add('After “divide” again', txt(pg, '#grpo-rail [data-obs]'))
            if pg.query_selector('#grpo-transport [data-act="play"]') and pg.evaluate("()=>window.__grpo.state().step")==4:
                click(pg, '#grpo-transport [data-act="step1"]', 300); d.add('After +1 step', txt(pg, '#grpo-rail .step.current')); stage(pg, d, 'After +1 step')
                click(pg, '#grpo-transport [data-act="play"]', 400); acted = True
            if acted: d.add('Card after the action', txt(pg, '#grpo-rail .step.current')); stage(pg, d, f'Part {i + 1} after'); d.tipsOf(pg, '#grpo-app')
            if i == 4:
                for v in ('eq', 'kl', 'peq'):
                    if pg.query_selector(f'#grpo-view button[data-v="{v}"]'): click(pg, f'#grpo-view button[data-v="{v}"]'); d.add(f'Chart view {v}', txt(pg, '#grpo-pan-chart'))
                if pg.query_selector('#grpo-view button[data-v="acc"]'): click(pg, '#grpo-view button[data-v="acc"]')
            nb = pg.query_selector('#grpo-rail .step.current [data-nav="next"]')
            if nb and not nb.is_disabled(): click(pg, '#grpo-rail .step.current [data-nav="next"]', 300)
        d.add('Next-lab footer', txt(pg, '.next-lab'))
        d.h('Show the math')
        pg.evaluate("()=>{const m=document.querySelector('#grpo-math'); if(m) m.open=true}"); pg.wait_for_timeout(150); d.add('Math', txt(pg, '#grpo-math')); d.tipsOf(pg, '#grpo-math')
        d.h('Scenarios')
        click(pg, '#grpo-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 400)
        d.add('Rail', txt(pg, '#grpo-rail .railcard, #grpo-rail .presets'))
        for i in range(5):
            click(pg, f'#grpo-rail [data-preset="{i}"]', 300); d.add(f'Scenario {i + 1} insight', txt(pg, '#grpo-insight'))
        d.h('Settings'); d.add('Settings', txt(pg, '#grpo-controls')); d.tipsOf(pg, '#grpo-controls')
        d.h('Challenge')
        click(pg, '#grpo-app .modes button[data-mode="challenge"]', 400)
        d.add('Card', txt(pg, '#grpo-rail .railcard'))
        done = "()=>window.__grpo.state().steps>=30&&!window.__grpo.animState()"
        d.add('Before Submit — goal line and stage status', txt(pg, '#grpo-ch-a, #grpo-status'))
        click(pg, '#grpo-rail [data-act="submit"]', 300); pg.wait_for_function(done, timeout=20000); pg.wait_for_timeout(200)
        d.add('G = 2 submitted', txt(pg, '#grpo-ch-res, #grpo-status'))
        click(pg, '#grpo-rail .seg-ctl[data-key="G"] button[data-v="16"]', 300)
        click(pg, '#grpo-rail [data-act="submit"]', 300); pg.wait_for_function(done, timeout=20000); pg.wait_for_timeout(200)
        d.add('G = 16 submitted', txt(pg, '#grpo-ch-res'))
        click(pg, '#grpo-rail .seg-ctl[data-key="G"] button[data-v="8"]', 300); d.add('G = 8 (and the chart)', txt(pg, '#grpo-ch-res'))
        d.h('Capability card'); d.add('Card', txt(pg, '#grpo-rail [data-exits]')); d.tipsOf(pg, '#grpo-rail [data-exits]')
        verd = pg.evaluate("""async ()=>{const w=()=>new Promise(r=>setTimeout(r,30)); const $=s=>document.querySelector('#grpo-rail [data-exits] '+s); const out=[];
          for(let t=0;t<3;t++){$('[data-ex="tab"][data-i="'+t+'"]').click(); await w();
            if(t===2){out.push('Tab 3: '+$('.ex-body').innerText); continue;}
            const nq=document.querySelectorAll('#grpo-rail [data-exits] .ex-qs .pb-dot').length||1;
            for(let j=0;j<nq;j++){const dot=$('.ex-qs [data-ex="q"][data-j="'+j+'"]'); if(dot){dot.click(); await w();}
              const q=$('.ex-body p b'); out.push('Q: '+(q?q.innerText:''));
              const nopt=document.querySelectorAll('#grpo-rail [data-exits] [data-ex="ans"]').length;
              for(let v=0;v<nopt;v++){$('[data-ex="ans"][data-v="'+v+'"]').click(); await w(); const r=$('.ex-res'); out.push('  ['+v+'] '+$('[data-ex="ans"][data-v="'+v+'"]').innerText+' -> '+(r?r.innerText:''));}}}
          return out.join('\\n');}""")
        d.add('Every option and its verdict', verd)
        d.h('Connect'); d.add('Connect', txt(pg, '#grpo-app section.connect'))
        if pg.query_selector('#grpo-app [data-act="drill-reveal"]'): click(pg, '#grpo-app [data-act="drill-reveal"]', 150)
        d.add('Drill with the answer', txt(pg, '#grpo-drill-q, #grpo-drill-a')); d.tipsOf(pg, '#grpo-app')
        b.close()
    (OUT / f'copy-{lang}.md').write_text(f'# grpo copy deck ({lang})\n' + '\n'.join(d.lines))
    print(lang, len(d.lines), 'blocks')


for lang in ('zh', 'en'): run(lang)
