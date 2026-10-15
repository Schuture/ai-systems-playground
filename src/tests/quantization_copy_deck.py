"""Dump every piece of text a learner meets in the quantization lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/quantization_copy_deck.py  ->  out/quantization/copy-{en,zh}.md and out/quantization-copy/{en,zh}.md (the copy that
tests/numdense.py reads: each block under a `### <section> — <label>` heading, tooltips under `_new tooltips:_`)

Order: home tile (when built), lab header, each guided part (card before and after its action, the stage, the metrics), Show the
math, the scenarios (each insight), Settings, the challenge (every feedback tier and the chart), the capability card, and Connect with
the drill answer. A tooltip is printed the first time it appears with a given text. Reduced motion is on, so every replay ends at once."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'quantization'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/quantization.html', '--only', 'quantization'], cwd=P, check=True, capture_output=True)
(OUT / 'deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'quantization.html').read_text() + '</body></html>')
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
    e.querySelectorAll('.qz-eqs').forEach(g=>{ const c=[...g.children], d=document.createElement('div');
      for(let i=0;i<c.length;i+=3){ const r=document.createElement('div'); r.textContent=c.slice(i,i+3).map(x=>{const m=x.querySelector('math'); return m?one(m):x.textContent.trim();}).join(' ').trim(); d.appendChild(r); }
      g.replaceWith(d); undo.push([g,d]); });
    e.querySelectorAll('math').forEach(m=>{ const sp=document.createElement('span'); sp.textContent=one(m); m.replaceWith(sp); undo.push([m,sp]); });
    const t=e.innerText; undo.reverse().forEach(([a,b])=>b.replaceWith(a)); return t; }).join('\n'); }"""


def txt(pg, sel):
    return pg.evaluate(LIN_JS, sel)


class Deck:
    def __init__(self): self.lines, self.tips, self.sec = [], set(), ''

    def h(self, t): self.sec = t; self.lines.append(f'\n## {t}\n')

    def add(self, label, t):
        t = (t or '').strip()
        esc = lambda x: '\\' + x if x.startswith('#') else x
        if t: self.lines.append(f'### {self.sec} — {label}\n\n' + '\n'.join(esc(clean(x)) for x in t.split('\n') if clean(x)) + '\n')

    def tipsOf(self, pg, root):
        new = []
        # a hover may hold a typeset formula (R6.29): written as its plain text, as the reader sees it
        for t in pg.evaluate("r=>[...document.querySelectorAll(r+' [data-tipx]')].filter(e=>e.getClientRects().length).map(e=>{const d=document.createElement('div'); d.innerHTML=e.getAttribute('data-tipx'); d.querySelectorAll('math').forEach(m=>m.replaceWith(document.createTextNode(m.textContent))); return d.textContent;})", root):
            c = clean(t)
            if c and c not in self.tips: self.tips.add(c); new.append('- ' + c)
        if new: self.lines.append('_new tooltips:_\n' + '\n'.join('  ' + x for x in new) + '\n')


def click(pg, sel, ms=200):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(ms)


def stage(pg, d, label):
    d.add(label + ' — stage', txt(pg, '#qz-stage .stage-head, #qz-b-cap, #qz-tp, #qz-b-leg, #qz-b-note, #qz-chart, #qz-steps, #qz-mem'))
    d.add(label + ' — chart labels', pg.evaluate("()=>[...document.querySelectorAll('#qz-stage svg')].filter(e=>e.getClientRects().length).map(s=>[...s.querySelectorAll('text')].map(t=>t.textContent).join(' · ')).join(' / ')"))
    d.add(label + ' — metric cards', txt(pg, '#qz-metrics'))


def run(lang):
    d = Deck()
    with sync_playwright() as pw:
        b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce')
        pg.goto(URL); pg.wait_for_timeout(300); click(pg, f'.langsw button[data-lang="{lang}"]', 150)
        # the home tile lives in teasers/set-a.js and is built only with the full site; listed when present
        if pg.query_selector('[data-teaser="quantization"]'):
            d.h('Home tile'); d.add('Tile', txt(pg, '[data-teaser="quantization"]'))
        pg.evaluate("()=>{location.hash='quantization'}"); pg.wait_for_timeout(500)
        d.h('Header'); d.add('Header', txt(pg, '#qz-app .lab-head, #qz-app header, #qz-app .modes')); d.tipsOf(pg, '#qz-app .modes')
        for i in range(7):
            d.h(f'Guided part {i + 1}')
            d.add('Card', txt(pg, '#qz-rail .step.current')); d.tipsOf(pg, '#qz-rail')
            stage(pg, d, f'Part {i + 1}'); d.tipsOf(pg, '#qz-stage')
            acted = False
            if pg.query_selector('#qz-ic-p1s'):
                for v in ('int8', 'int4-128'):
                    if pg.query_selector(f'#qz-ic-p1s button[data-v="{v}"]'):
                        click(pg, f'#qz-ic-p1s button[data-v="{v}"]', 200); d.add(f'After {v}', txt(pg, '#qz-rail [data-obs]'))
                acted = True
            if pg.query_selector('#qz-rail [data-pred="zero"]') and not window_shown(pg):
                click(pg, '#qz-rail [data-pred="keep"]'); d.add('After picking an answer', txt(pg, '#qz-rail .step.current'))
                click(pg, '#qz-rail [data-pred="zero"]'); click(pg, '#qz-rail [data-act="pred-go"]', 300); acted = True
            for sel, vals in (('#qz-ic-p4', ('int4-row', 'int4-128', 'int4-32', 'int4-16', 'mx', 'nv')), ('#qz-ic-p5', ('w4', 'fp8', 'fp8kv'))):
                if pg.query_selector(sel):
                    for v in vals:
                        click(pg, f'{sel} [data-lever="{v}"]', 250); d.add(f'After {v}', txt(pg, '#qz-rail [data-obs]')); stage(pg, d, v)
                    acted = True
            if pg.query_selector('#qz-ic-p6s'):
                for v in ('70b', '405b'):
                    click(pg, f'#qz-ic-p6s button[data-v="{v}"]', 250); d.add(f'After {v}', txt(pg, '#qz-rail [data-obs]')); stage(pg, d, v)
                acted = True
            if acted or i == 1:
                d.add('Card after the action', txt(pg, '#qz-rail .step.current')); stage(pg, d, f'Part {i + 1} after'); d.tipsOf(pg, '#qz-app')
            nb = pg.query_selector('#qz-rail .step.current [data-nav="next"]')
            if nb and not nb.is_disabled(): click(pg, '#qz-rail .step.current [data-nav="next"]', 300)
        d.h('Show the math')
        pg.evaluate("()=>{const m=document.querySelector('#qz-math'); if(m) m.open=true}"); pg.wait_for_timeout(150); d.add('Math', txt(pg, '#qz-math')); d.tipsOf(pg, '#qz-math')
        d.h('Scenarios')
        click(pg, '#qz-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 400)
        d.add('Rail', txt(pg, '#qz-rail .railcard, #qz-rail .presets'))
        for i in range(6):
            click(pg, f'#qz-rail [data-preset="{i}"]', 300); d.add(f'Scenario {i + 1} insight', txt(pg, '#qz-insight')); stage(pg, d, f'Scenario {i + 1}')
        d.h('Settings'); d.add('Settings', txt(pg, '#qz-controls')); d.tipsOf(pg, '#qz-controls')
        d.h('Challenge')
        click(pg, '#qz-app .modes button[data-mode="challenge"]', 400)
        d.add('Card', txt(pg, '#qz-rail .railcard')); stage(pg, d, 'Challenge')
        d.add('Before Submit — goal line and stage status', txt(pg, '#qz-ch-g1, #qz-status'))
        click(pg, '#qz-rail [data-act="submit"]', 300); d.add('BF16 submitted (does not fit)', txt(pg, '#qz-ch-res, #qz-status'))
        click(pg, '#qz-ic-chw [data-lever="int4-128"]', 200); click(pg, '#qz-ic-chkv button[data-v="fp8"]', 200)
        click(pg, '#qz-rail [data-act="submit"]', 300); d.add('INT4 per 128 + FP8 KV submitted (fits, loses the pick)', txt(pg, '#qz-ch-res'))
        if pg.query_selector('#qz-rail [data-act="reveal"]'): click(pg, '#qz-rail [data-act="reveal"]', 300); d.add('After Show the answer', txt(pg, '#qz-ch-res'))
        click(pg, '#qz-ic-chw [data-lever="nv"]', 200); click(pg, '#qz-rail [data-act="submit"]', 300); d.add('NVFP4 + FP8 KV submitted (the answer, and the chart)', txt(pg, '#qz-ch-res, #qz-status')); d.tipsOf(pg, '#qz-rail')
        d.h('Capability card'); d.add('Card', txt(pg, '#qz-rail [data-exits]')); d.tipsOf(pg, '#qz-rail [data-exits]')
        verd = pg.evaluate("""async ()=>{const w=()=>new Promise(r=>setTimeout(r,30)); const $=s=>document.querySelector('#qz-rail [data-exits] '+s); const out=[];
          const lin=n=>{ if(n.nodeType===3) return n.textContent; if(n.nodeType!==1) return ''; const t=n.tagName.toLowerCase(), k=[...n.children];
            if(t==='annotation') return ''; if(t==='mspace') return ' '; if(t==='mfrac') return '('+lin(k[0])+')/('+lin(k[1])+')';
            if(t==='msub') return lin(k[0])+'_'+lin(k[1]); if(t==='msup') return lin(k[0])+'^'+lin(k[1]); if(t==='mover'||t==='munder') return lin(k[0]);
            return [...n.childNodes].map(lin).join(''); };
          const TX=el=>{if(!el) return ''; const undo=[]; el.querySelectorAll('math').forEach(m=>{const sp=document.createElement('span'); sp.textContent=lin(m).replace(/\\s+/g,' ').trim(); m.replaceWith(sp); undo.push([m,sp]);}); const t=el.innerText; undo.reverse().forEach(([a,b])=>b.replaceWith(a)); return t;};
          for(let t=0;t<3;t++){$('[data-ex="tab"][data-i="'+t+'"]').click(); await w();
            if(t===2){out.push('Tab 3: '+$('.ex-body').innerText); continue;}
            const nq=document.querySelectorAll('#qz-rail [data-exits] .ex-qs .pb-dot').length||1;
            for(let j=0;j<nq;j++){const dot=$('.ex-qs [data-ex="q"][data-j="'+j+'"]'); if(dot){dot.click(); await w();}
              const q=$('.ex-body p b'); out.push('Q: '+TX(q));
              const nopt=document.querySelectorAll('#qz-rail [data-exits] [data-ex="ans"]').length;
              for(let v=0;v<nopt;v++){$('[data-ex="ans"][data-v="'+v+'"]').click(); await w(); const r=$('.ex-res'); const L='ABCD'[v]; out.push('- ('+L+') '+TX($('[data-ex="ans"][data-v="'+v+'"]'))); out.push('('+L+') '+TX(r));}}}
          return out.join('\\n');}""")
        d.add('Every option and its verdict', verd)
        d.h('Connect'); d.add('Connect', txt(pg, '#qz-app section.connect'))
        if pg.query_selector('#qz-app [data-act="drill-reveal"]'): click(pg, '#qz-app [data-act="drill-reveal"]', 150)
        d.add('Drill with the answer', txt(pg, '#qz-drill-q, #qz-drill-a')); d.tipsOf(pg, '#qz-app')
        b.close()
    body = f'# quantization copy deck ({lang})\n' + '\n'.join(d.lines)
    (OUT / f'copy-{lang}.md').write_text(body)
    nd = P / 'out' / 'quantization-copy'; nd.mkdir(parents=True, exist_ok=True); (nd / f'{lang}.md').write_text(body)
    print(lang, len(d.lines), 'blocks')


window_shown = lambda pg: pg.evaluate("()=>window.__qz.state().predShown")
for lang in ('zh', 'en'): run(lang)
