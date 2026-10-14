"""Option groups that wrap to a second row keep a line between the rows (R7.13: every option row lined; owner 2026-10-14: agent-anatomy's
token-budget rows lost their line because the kit's buttons draw only a right border). For every lab given, visits each guided part,
Settings and the challenge at the widths and languages given and reports a visible .seg-ctl whose buttons sit on two or more rows with no
line between two rows (no border between them, and no 1px gap over a line-coloured group background).
Run: python3 tests/kit_seg_rows.py <lab> [<lab> …]   (SEG_SIZES="1440:zh,390:en" for other widths; default 1440 zh, 1440 en, 1024 en, 390 zh)
     SEG_PAGE=ai-systems-playground.html uses a page already built"""
import os, pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
IDS = [a for a in sys.argv[1:] if not a.startswith('-')]
if not IDS:
    sys.exit(__doc__)
SRC = os.environ.get('SEG_PAGE')
if not SRC:
    SRC = f'out/segrows-{os.getpid()}.html'
    subprocess.run([sys.executable, str(P / 'build.py'), SRC, '--only', ','.join(IDS)], check=True, cwd=P, stdout=subprocess.DEVNULL)
WRAP = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>')
WRAPPED = P / 'out' / f'segrows-wrapped-{os.getpid()}.html'
WRAPPED.write_text(WRAP + (P / SRC).read_text() + '</body></html>')
URL = 'file://' + str(WRAPPED)
SIZES = [(int(w), l) for w, l in (x.split(':') for x in os.environ.get('SEG_SIZES', '1440:zh,1440:en,1024:en,390:zh').split(','))]

PROBE = r"""()=>{const root=document.querySelector('#view-dyn:not([hidden])')||document.querySelector('#view-lab:not([hidden])'); if(!root) return [];
  const vis=e=>{for(let x=e;x&&x!==document.body;x=x.parentElement){const cs=getComputedStyle(x); if(cs.display==='none'||cs.visibility==='hidden'||+cs.opacity===0) return false;} return true;};
  const clear=c=>/rgba\(.*,\s*0\)$|transparent/.test(c), out=[];
  for(const g of root.querySelectorAll('.seg-ctl')){if(!vis(g)) continue; const bs=[...g.children].filter(b=>b.tagName==='BUTTON'&&b.getClientRects().length); if(bs.length<2) continue;
    const rows=[]; for(const b of bs){const r=b.getBoundingClientRect(); let row=rows.find(x=>Math.abs(x.top-r.top)<4); if(!row){row={top:r.top,bottom:r.bottom,bs:[]}; rows.push(row);} row.bs.push(b); row.bottom=Math.max(row.bottom,r.bottom);}
    if(rows.length<2) continue; rows.sort((a,b)=>a.top-b.top);
    const gcs=getComputedStyle(g), gbg=gcs.backgroundColor;
    for(let i=1;i<rows.length;i++){const up=rows[i-1], dn=rows[i], gap=dn.top-up.bottom;
      const lined=dn.bs.some(b=>parseFloat(getComputedStyle(b).borderTopWidth)>0)||up.bs.some(b=>parseFloat(getComputedStyle(b).borderBottomWidth)>0)
        ||(gap>=0.5&&!clear(gbg)&&dn.bs.every(b=>getComputedStyle(b).backgroundColor!==gbg))||gap>=4;
      if(!lined){out.push({kind:'rows',text:(bs.map(b=>b.textContent.trim()).join(' | ')).slice(0,60),by:'row '+(i+1),svg:(g.dataset.key||g.getAttribute('aria-label')||g.className).slice(0,40)}); break;}}}
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
                    for wait in (400,):
                        pg.wait_for_timeout(wait)
                        for f in pg.evaluate(PROBE):
                            key = (mode, k, f['kind'], f['text'])
                            if key in seen: continue
                            seen.add(key)
                            tag = f"{lab} {w} {lang} {mode}{'' if k is None else ' part ' + str(k + 1)}"
                            bad.setdefault(lab, []).append(f"{tag}: {f['kind']} text “{f['text']}” has no line above {f['by']} ({f['svg']})")
                except Exception as e:
                    bad.setdefault(lab, []).append(f'{lab} {w} {lang} {mode} {k}: probe failed {str(e)[:100]}')
            pg.close()
        print(f"{lab}: {len(bad.get(lab, []))} problems", flush=True)
        for x in bad.get(lab, []): print('   ', x, flush=True)
    b.close()
n = sum(len(v) for v in bad.values())
print(f'SEG ROWS: {n} problems in {len(bad)} of {len(IDS)} labs' if n else f'SEG ROWS: none in {len(IDS)} labs')
sys.exit(1 if n else 0)
