"""Text that crosses its picture or its panel (the owner's review of 2026-10-09: RLVR's “优势” label ran past the stage's right edge).
For every lab given, visits each guided part, Scenarios and the challenge, at the widths and languages given, and measures every visible
SVG <text> against its <svg> and against the panel that holds it (the nearest .panel / .stage / .card), and every HTML text element of
a stage against the stage. A label that ends more than 1 px outside fails. Measured twice per state (after 1.2 s and after 4 s), so a
label that moves with an animation is caught in its end state too.
Run: python3 tests/kit_text_bounds.py <lab> [<lab> …]             (builds out/bounds-<pid>.html with those labs; 1440 zh, 900 en, 390 zh, 390 en)
     BOUNDS_PAGE=ai-systems-playground.html python3 tests/kit_text_bounds.py <lab> …   (uses a page already built)
     BOUNDS_SIZES="1440:zh,390:en" python3 tests/kit_text_bounds.py <lab> …           (fewer widths and languages)"""
import os, pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
IDS = [a for a in sys.argv[1:] if not a.startswith('-')]
if not IDS:
    sys.exit(__doc__)
SRC = os.environ.get('BOUNDS_PAGE')
if not SRC:
    SRC = f'out/bounds-{os.getpid()}.html'
    subprocess.run([sys.executable, str(P / 'build.py'), SRC, '--only', ','.join(IDS)], check=True, cwd=P, stdout=subprocess.DEVNULL)
WRAP = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>')
WRAPPED = P / 'out' / f'bounds-wrapped-{os.getpid()}.html'
WRAPPED.write_text(WRAP + (P / SRC).read_text() + '</body></html>')
URL = 'file://' + str(WRAPPED)
SIZES = [(int(w), l) for w, l in (x.split(':') for x in os.environ.get('BOUNDS_SIZES', '1440:zh,900:en,390:zh,390:en').split(','))]

PROBE = r"""()=>{const root=document.querySelector('#view-dyn:not([hidden])')||document.querySelector('#view-lab:not([hidden])'); if(!root) return [];
  const vis=e=>{for(let x=e;x&&x!==document.body;x=x.parentElement){const cs=getComputedStyle(x); if(cs.display==='none'||cs.visibility==='hidden'||+cs.opacity===0) return false;} return true;};
  const panelOf=e=>e.closest('.panel,.stage,.card,.c-card');
  const name=e=>(e.textContent||'').trim().replace(/\s+/g,' ').slice(0,40);
  const out=[], T=1;
  for(const svg of root.querySelectorAll('svg')){const sr=svg.getBoundingClientRect(); if(sr.width<2||sr.height<2||!vis(svg)) continue;
    if(svg.closest('.tip,#tip')) continue;
    const pn=panelOf(svg), pr=pn?pn.getBoundingClientRect():null;
    for(const t of svg.querySelectorAll('text')){if(!name(t)||!vis(t)) continue; const r=t.getBoundingClientRect(); if(r.width<1) continue;
      const o={l:sr.left-r.left,r:r.right-sr.right,t:sr.top-r.top,b:r.bottom-sr.bottom};
      const k=Object.entries(o).filter(([,v])=>v>T).map(([s,v])=>s+Math.round(v));
      if(k.length) out.push({kind:'svg',text:name(t),by:k.join(' '),svg:(svg.getAttribute('aria-label')||svg.id||svg.className.baseVal||'').slice(0,40)});
      else if(pr){const q={l:pr.left-r.left,r:r.right-pr.right}; const k2=Object.entries(q).filter(([,v])=>v>T).map(([s,v])=>s+Math.round(v));
        if(k2.length) out.push({kind:'panel',text:name(t),by:k2.join(' '),svg:(svg.getAttribute('aria-label')||svg.id||'').slice(0,40)});}}}
  // HTML text: against every box between it and its panel (a bordered or filled box, or one that clips), e.g. a row button of the stage
  const isBox=x=>{const cs=getComputedStyle(x); const bw=['Top','Right','Bottom','Left'].some(sd=>parseFloat(cs['border'+sd+'Width'])>0&&!/rgba\(.*,\s*0\)$|transparent/.test(cs['border'+sd+'Color']));
    const bg=!/rgba\(.*,\s*0\)$|transparent/.test(cs.backgroundColor); const clip=/hidden|clip/.test(cs.overflowX); return bw||bg||clip;};
  for(const st of root.querySelectorAll('.stage,.panel,.card')){if(!vis(st)) continue;
    for(const e of st.querySelectorAll('span,b,strong,em,p,div,li,label,button,a,td,th,h3,h4,small')){if(e.children.length&&e.tagName!=='BUTTON') continue; if(!name(e)||!vis(e)) continue;
      if(e.closest('svg,.tip,pre,table,.map-scroll')) continue;
      // measure the text itself, not only its element: nowrap text can run past its own box (2026-10-10: a metric card's value
      // “+7.2 ± 2.8 分” ran 47 px past the card while the .v element stayed inside it)
      const r0=e.getBoundingClientRect(); if(r0.width<1) continue; const rg=document.createRange(); rg.selectNodeContents(e); const r1=rg.getBoundingClientRect();
      const ecs=getComputedStyle(e), own=/hidden|clip/.test(ecs.overflowX)||ecs.textOverflow==='ellipsis';   // text the element clips itself is not past it
      const r=r1.width>0&&!own?{left:Math.min(r0.left,r1.left),right:Math.max(r0.right,r1.right)}:r0;
      // text placed out of flow on purpose (an axis label beside a plot box) is checked against the panel only
      let abs=/absolute|fixed/.test(getComputedStyle(e).position);
      for(let x=e.parentElement;x&&x!==st.parentElement;x=x.parentElement){const cs=getComputedStyle(x); if(/auto|scroll/.test(cs.overflowX)) break;
        if(x!==st&&(abs||!isBox(x))){if(/absolute|fixed/.test(cs.position)) abs=true; continue;}
        const q=x.getBoundingClientRect(); const o={l:q.left-r.left,r:r.right-q.right}; const k=Object.entries(o).filter(([,v])=>v>T).map(([s,v])=>s+Math.round(v));
        if(k.length){out.push({kind:'html',text:name(e),by:k.join(' '),svg:'in '+x.tagName.toLowerCase()+(typeof x.className==='string'&&x.className?'.'+x.className.trim().split(/\s+/).slice(0,2).join('.'):'')}); break;}}}}
  return out;}"""

bad = {}
with sync_playwright() as p:
    b = p.chromium.launch()
    for lab in IDS:
        for w, lang in SIZES:
            pg = b.new_page(viewport={'width': w, 'height': 900 if w > 500 else 844})
            pg.goto(URL); pg.wait_for_timeout(200)
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(60)
            pg.evaluate(f"()=>{{location.hash='{lab}'}}"); pg.wait_for_timeout(500)
            dyn = lab != 'kv-cache'
            root, rail = ('#view-dyn', '#view-dyn aside.rail') if dyn else ('#view-lab', '#rail')
            nparts = pg.evaluate("r=>new Set([...document.querySelectorAll(r+' [data-go]')].map(e=>e.dataset.go)).size", rail)
            states = [('guided', k) for k in range(nparts)] + [('sandbox', None), ('challenge', None)]
            for mode, k in states:
                try:
                    if k is None or k == 0:
                        pg.evaluate("([r,m])=>{const b=[...document.querySelectorAll(r+' .modes button[data-mode=\"'+m+'\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}", [root, mode])
                        pg.wait_for_timeout(300)
                    if k is not None:
                        pg.evaluate("([r,k])=>{const e=document.querySelector(r+' [data-go=\"'+k+'\"]'); if(e) e.click();}", [rail, k])
                    seen = set()
                    for wait in (1200, 2800):
                        pg.wait_for_timeout(wait)
                        for f in pg.evaluate(PROBE):
                            key = (mode, k, f['kind'], f['text'])
                            if key in seen: continue
                            seen.add(key)
                            tag = f"{lab} {w} {lang} {mode}{'' if k is None else ' part ' + str(k + 1)}"
                            bad.setdefault(lab, []).append(f"{tag}: {f['kind']} text “{f['text']}” crosses by {f['by']} px ({f['svg']})")
                except Exception as e:
                    bad.setdefault(lab, []).append(f'{lab} {w} {lang} {mode} {k}: probe failed {str(e)[:100]}')
            pg.close()
        print(f"{lab}: {len(bad.get(lab, []))} problems", flush=True)
        for x in bad.get(lab, []): print('   ', x, flush=True)
    b.close()
n = sum(len(v) for v in bad.values())
print(f'TEXT BOUNDS: {n} problems in {len(bad)} of {len(IDS)} labs' if n else f'TEXT BOUNDS: none in {len(IDS)} labs')
sys.exit(1 if n else 0)
