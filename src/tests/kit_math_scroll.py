"""“Show the math” that has to be scrolled sideways (the owner's review of 2026-10-09: Show the math must read like a textbook page,
R6.37; a row the reader has to scroll to see is not read). For every lab given, visits each guided part, Scenarios and the challenge, at
the widths and languages given (in Scenarios, each scenario in turn), opens every “Show the math” (details.math), and reports each box inside it that scrolls sideways
(scrollWidth more than 1 px past its width) and any “Show the math” wider than its own box or the page.
Run: python3 tests/kit_math_scroll.py <lab> [<lab> …]               (builds out/mscroll-<pid>.html with those labs; 1440 zh, 900 en, 390 zh, 390 en)
     MSCROLL_PAGE=ai-systems-playground.html python3 tests/kit_math_scroll.py <lab> …   (uses a page already built)
     MSCROLL_SIZES="1440:zh,390:en" python3 tests/kit_math_scroll.py <lab> …           (fewer widths and languages)"""
import os, pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
IDS = [a for a in sys.argv[1:] if not a.startswith('-')]
if not IDS:
    sys.exit(__doc__)
SRC = os.environ.get('MSCROLL_PAGE')
if not SRC:
    SRC = f'out/mscroll-{os.getpid()}.html'
    subprocess.run([sys.executable, str(P / 'build.py'), SRC, '--only', ','.join(IDS)], check=True, cwd=P, stdout=subprocess.DEVNULL)
WRAP = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>')
WRAPPED = P / 'out' / f'mscroll-wrapped-{os.getpid()}.html'
WRAPPED.write_text(WRAP + (P / SRC).read_text() + '</body></html>')
URL = 'file://' + str(WRAPPED)
SIZES = [(int(w), l) for w, l in (x.split(':') for x in os.environ.get('MSCROLL_SIZES', '1440:zh,900:en,390:zh,390:en').split(','))]

OPEN = """r=>{const v=document.querySelector(r); if(!v) return 0; let n=0; for(const d of v.querySelectorAll('details.math')){if(!d.open){d.open=true;} n++;} return n;}"""
PROBE = r"""(r)=>{const v=document.querySelector(r); if(!v) return []; const out=[], T=1;
  const vis=e=>{for(let x=e;x&&x!==document.body;x=x.parentElement){const cs=getComputedStyle(x); if(cs.display==='none'||cs.visibility==='hidden') return false;} return true;};
  const name=e=>(e.textContent||'').trim().replace(/\s+/g,' ').slice(0,50);
  const cls=e=>e.tagName.toLowerCase()+(e.id?'#'+e.id:'')+(typeof e.className==='string'&&e.className.trim()?'.'+e.className.trim().split(/\s+/).slice(0,2).join('.'):'');
  for(const d of v.querySelectorAll('details.math')){if(!d.open||!vis(d)) continue; const dr=d.getBoundingClientRect(); if(dr.width<2) continue;
    for(const e of d.querySelectorAll('*')){const cs=getComputedStyle(e); if(!/auto|scroll/.test(cs.overflowX)) continue; if(!vis(e)) continue;
      const over=e.scrollWidth-e.clientWidth; if(over>T) out.push({box:cls(e),by:over,text:name(e)});}
    if(d.scrollWidth-d.clientWidth>T) out.push({box:'the whole “Show the math” ('+cls(d)+')',by:d.scrollWidth-d.clientWidth,text:''});
    const pr=d.closest('.panel,.stage,.card'); if(pr){const q=pr.getBoundingClientRect(); if(dr.right-q.right>T) out.push({box:'past its panel ('+cls(pr)+')',by:Math.round(dr.right-q.right),text:''});}}
  /* a typeset formula anywhere else in the lab (cards, results, captions) that is cut off or scrolls: its right edge past a box that
     clips or scrolls, or past its panel (the embeddings review found a part-2 formula cut off at 1440 and 900) */
  for(const m of v.querySelectorAll('math')){if(m.closest('details.math')||m.parentElement.closest('math')||!vis(m)) continue; const mr=m.getBoundingClientRect(); if(mr.width<1) continue;
    for(let x=m.parentElement;x&&x!==v;x=x.parentElement){const cs=getComputedStyle(x), q=x.getBoundingClientRect();
      if(/hidden|clip|auto|scroll/.test(cs.overflowX)&&(mr.right-q.right>T||x.scrollWidth-x.clientWidth>T)){out.push({box:'a formula cut off or scrolling in '+cls(x),by:Math.round(Math.max(mr.right-q.right,x.scrollWidth-x.clientWidth)),text:name(m)}); break;}
      if(x.matches('.panel,.stage,.card,.c-card')){if(mr.right-q.right>T) out.push({box:'a formula past its panel ('+cls(x)+')',by:Math.round(mr.right-q.right),text:name(m)}); break;}}}
  const pg=document.documentElement.scrollWidth-innerWidth; if(pg>T) out.push({box:'the page',by:pg,text:''});
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
            seen = set()
            for mode, k in states:
                try:
                    if k is None or k == 0:
                        pg.evaluate("([r,m])=>{const b=[...document.querySelectorAll(r+' .modes button[data-mode=\"'+m+'\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}", [root, mode])
                        pg.wait_for_timeout(300)
                    if k is not None:
                        pg.evaluate("([r,k])=>{const e=document.querySelector(r+' [data-go=\"'+k+'\"]'); if(e) e.click();}", [rail, k])
                    # in Scenarios, the first state and then each scenario in turn (the training math-wrap round: preset rows were missed)
                    npre = pg.evaluate("r=>[...document.querySelectorAll(r+' [data-preset]')].filter(b=>b.offsetParent).length", root) if mode == 'sandbox' else 0
                    for pi in range(-1, npre):
                        if pi >= 0:
                            pg.evaluate("([r,i])=>{const b=[...document.querySelectorAll(r+' [data-preset]')].filter(b=>b.offsetParent)[i]; if(b) b.click();}", [root, pi])
                        pg.wait_for_timeout(900)
                        pg.evaluate(OPEN, root); pg.wait_for_timeout(500)
                        for f in pg.evaluate(PROBE, root):
                            key = (f['box'], f['text'][:30])
                            if key in seen: continue
                            seen.add(key)
                            tag = f"{lab} {w} {lang} {mode}{'' if k is None else ' part ' + str(k + 1)}{'' if pi < 0 else ' scenario ' + str(pi + 1)}"
                            bad.setdefault(lab, []).append(f"{tag}: {f['box']} scrolls sideways by {f['by']} px  “{f['text']}”")
                except Exception as e:
                    bad.setdefault(lab, []).append(f'{lab} {w} {lang} {mode} {k}: probe failed {str(e)[:100]}')
            pg.close()
        print(f"{lab}: {len(bad.get(lab, []))} problems", flush=True)
        for x in bad.get(lab, []): print('   ', x, flush=True)
    b.close()
n = sum(len(v) for v in bad.values())
print(f'MATH SCROLL: {n} problems in {len(bad)} of {len(IDS)} labs' if n else f'MATH SCROLL: none in {len(IDS)} labs')
sys.exit(1 if n else 0)
