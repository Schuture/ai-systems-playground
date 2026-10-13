"""Dump every piece of text a learner meets in the tool-scheduling lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/tool-scheduling_copy_deck.py  ->  out/tool-scheduling-copy/zh.md and en.md (tests/numdense.py reads them)

Order: home tile (both options, the note), lab header, each guided part (card before and after its actions, the stage, the metrics),
Show the math, the scenarios (each insight), Settings, the challenge (every feedback tier and the chart), the capability card, and
Connect with the drill answer. A tooltip is printed the first time it appears, and only the first three of each kind. Reduced motion
is on, so every run ends at once."""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
LAB = 'tool-scheduling'
OUT = P / 'out' / 'tool-scheduling-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), f'out/{LAB}.html', '--only', LAB], cwd=P, check=True, capture_output=True)
(OUT / 'deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / f'{LAB}.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'deck-wrapped.html')
clean = lambda s: ' '.join(s.replace('\u2011', '-').replace('\u00a0', ' ').split()) if s else ''


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
    e.querySelectorAll('.tls-eq').forEach(g=>{ const c=[...g.children], d=document.createElement('div');
      for(let i=0;i<c.length;i+=2){ const r=document.createElement('div'); r.textContent=c.slice(i,i+2).map(x=>{const m=x.querySelector('math'); return m?one(m):x.textContent.trim();}).join(' ').trim(); d.appendChild(r); }
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
    d.add(label + ' — stage', txt(pg, '#tls-stage .tls-head, #tls-stage .tls-pan:not([hidden]) .tls-pt, #tls-stage .tls-pan:not([hidden]) .tls-leg, #tls-g-ph:not([hidden])'))
    d.add(label + ' — pictures', pg.evaluate("()=>[...document.querySelectorAll('#tls-stage svg')].filter(e=>e.getClientRects().length).map(s=>(s.getAttribute('aria-label')||'')+' | '+[...s.querySelectorAll('text')].map(t=>t.textContent).join(' | ')).join('\\n')"))
    d.add(label + ' — metric cards', txt(pg, '#tls-metrics:not([hidden])'))


def obs(pg):
    return txt(pg, '#tls-rail [data-obs]')


def run(lang):
    d = Deck()
    with sync_playwright() as pw:
        b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce')
        pg.goto(URL); pg.wait_for_timeout(300); click(pg, f'.langsw button[data-lang="{lang}"]', 150)
        pg.evaluate("()=>{location.hash='map~agents'}"); pg.wait_for_timeout(700)
        T = f'[data-teaser="{LAB}"]'
        if pg.query_selector(T):
            d.h('Home tile')
            for v in ('0', '1'):
                click(pg, f'{T} button[data-v="{v}"]', 120); d.add(f'Option {v}', txt(pg, f'{T} .tz-read'))
            d.add('Tile', txt(pg, T)); d.add('Tile note', pg.evaluate("s=>[...document.querySelectorAll(s+' .tz-note')].map(e=>e.textContent).join('\\n')", T))
        pg.evaluate(f"()=>{{location.hash='{LAB}'}}"); pg.wait_for_timeout(500)
        d.h('Header'); d.add('Header', txt(pg, '#tls-app .lab-head, #tls-app header, #tls-app .modes')); d.tipsOf(pg, '#tls-app .modes')
        d.add('Fidelity note', pg.evaluate("()=>{const e=document.querySelector('#tls-app [id$=\"-fid\"]'); return e?e.innerText:''}"))
        d.add('Primer note', pg.evaluate("()=>{const e=document.querySelector('#tls-app [id$=\"-pri\"]'); return e?e.innerText:''}"))
        d.add('Transport', txt(pg, '#tls-transport')); d.tipsOf(pg, '#tls-transport')
        for i in range(7):
            d.h(f'Guided part {i + 1}')
            d.add('Card', txt(pg, '#tls-rail .step.current')); d.tipsOf(pg, '#tls-rail')
            stage(pg, d, f'Part {i + 1}'); d.tipsOf(pg, '#tls-stage')
            st = pg.evaluate("()=>window.__tls.state()")
            if st['step'] == 0:
                d.add('Observed', obs(pg)); click(pg, '#tls-rail [data-pred="ok"]')
            if st['step'] == 1:
                for v in ('bfl', 'bho', 'eml', 'wea'):
                    click(pg, f'#tls-ic-pick2 button[data-v="{v}"]'); d.add(f'After {v}', obs(pg))
            if st['step'] == 2:
                d.add('Observed on entry', obs(pg))
                for v in ('dep0', 'dep1'):
                    click(pg, f'#tls-rail button[data-lever="{v}"]'); d.add(f'After {v}', obs(pg))
            if st['step'] == 3:
                for v in ('3', '1', '5', '-1'):
                    click(pg, f'#tls-rail button[data-lever="{v}"]'); d.add(f'After half {v}', obs(pg))
            if st['step'] == 4:
                for v in ('1', '2', '3', '0'):
                    click(pg, f'#tls-ic-k button[data-v="{v}"]'); d.add(f'After k = {v}', obs(pg))
            if st['step'] == 5:
                for v in ('sho', 'bfl0', 'bfl1'):
                    click(pg, f'#tls-rail button[data-lever="{v}"]'); d.add(f'After {v}', obs(pg)); stage(pg, d, f'Part 6 {v}')
            if st['step'] in (0, 2, 5): stage(pg, d, f'Part {i + 1} after'); d.tipsOf(pg, '#tls-app')
            nb = pg.query_selector('#tls-rail .step.current [data-nav="next"]')
            if nb and not nb.is_disabled(): click(pg, '#tls-rail .step.current [data-nav="next"]', 300)
        d.add('Next-lab footer', txt(pg, '.next-lab'))
        d.h('Show the math')
        pg.evaluate("()=>{const m=document.querySelector('#tls-math'); if(m) m.open=true}"); pg.wait_for_timeout(150); d.add('Math', txt(pg, '#tls-math'))
        d.h('Scenarios')
        click(pg, '#tls-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 400)
        d.add('Rail', txt(pg, '#tls-rail .railcard'))
        for i in range(5):
            click(pg, f'#tls-rail [data-preset="{i}"]', 300); d.add(f'Scenario {i + 1} insight', txt(pg, '#tls-insight')); d.add(f'Scenario {i + 1} status', txt(pg, '#tls-st-title, #tls-status'))
        d.add('Math', txt(pg, '#tls-math'))
        d.h('Settings'); d.add('Settings', txt(pg, '#tls-controls')); d.tipsOf(pg, '#tls-controls')
        d.h('Challenge')
        click(pg, '#tls-app .modes button[data-mode="challenge"]', 400)
        d.add('Card', txt(pg, '#tls-rail .railcard')); d.add('Settings note', txt(pg, '#tls-lock')); stage(pg, d, 'Challenge before Submit')
        click(pg, '#tls-rail [data-act="submit"]', 300); d.add('Initial setup submitted', txt(pg, '#tls-ch-res, #tls-status'))
        for a, label in (([1, 1, 1, 2, 2, 2], 'tests in the edit batch'), ([1, 2, 2, 3, 3, 4], 'a 13.4-s arrangement'), ([1, 1, 1, 2, 2, 3], 'the best arrangement')):
            for j, v in enumerate(a): click(pg, f'#tls-ch-{j} button[data-v="{v}"]', 60)
            click(pg, '#tls-rail [data-act="submit"]', 300); d.add(label, txt(pg, '#tls-ch-res'))
        stage(pg, d, 'Challenge after Submit')
        d.h('Capability card'); d.add('Card', txt(pg, '#tls-rail [data-exits]')); d.tipsOf(pg, '#tls-rail [data-exits]')
        verd = pg.evaluate("""async ()=>{const w=()=>new Promise(r=>setTimeout(r,30)); const $=s=>document.querySelector('#tls-rail [data-exits] '+s); const out=[];
          for(let t=0;t<3;t++){const tb=$('[data-ex="tab"][data-i="'+t+'"]'); if(!tb) continue; tb.click(); await w();
            if(t===2){out.push('Tab 3: '+$('.ex-body').innerText); continue;}
            const nq=document.querySelectorAll('#tls-rail [data-exits] .ex-qs .pb-dot').length||1;
            for(let j=0;j<nq;j++){const dot=$('.ex-qs [data-ex="q"][data-j="'+j+'"]'); if(dot){dot.click(); await w();}
              const q=$('.ex-body p b'); out.push('Q: '+(q?q.innerText:''));
              const nopt=document.querySelectorAll('#tls-rail [data-exits] [data-ex="ans"]').length;
              for(let v=0;v<nopt;v++){$('[data-ex="ans"][data-v="'+v+'"]').click(); await w(); const r=$('.ex-res'); out.push('  ['+v+'] '+$('[data-ex="ans"][data-v="'+v+'"]').innerText+' -> '+(r?r.innerText:''));}}}
          return out.join('\\n');}""")
        d.add('Every option and its verdict', verd)
        d.h('Connect'); d.add('Connect', txt(pg, '#tls-app section.connect'))
        click(pg, '#tls-app [data-act="drill-reveal"]', 150)
        d.add('Drill with the answer', txt(pg, '#tls-drill-q, #tls-drill-a')); d.tipsOf(pg, '#tls-app')
        b.close()
    (OUT / f'{lang}.md').write_text(f'# {LAB} copy deck ({lang})\n' + '\n'.join(d.lines))
    print(lang, len(d.lines), 'blocks')


for lang in (sys.argv[1:] or ['zh', 'en']): run(lang)
