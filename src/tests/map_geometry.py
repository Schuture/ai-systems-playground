"""Tree geometry (guide R11.6, R11.8, R11.10; batch 5: the knowledge tree replaced the two loops), in EN and ZH at 1440 wide, light and dark.
Fails when: a lab has no place on the tree or two places; two lab boxes, a box and a branch name, two branch names, or anything and the
bar chart overlap; a lab's name does not fit its box; anything leaves the drawing; the tree is not whole on screen at 1440 × 900;
with a path chosen, a number overlaps another number, a name or a box that is not its own; the dimmed labs are not exactly the ones
off the path; the bar chart's counts differ from the labs'. Also: no page errors, no horizontal overflow at 1440, 1280, 1024, 900.
Review tree-causal-A (round 1 of fixes) added:
- H1: at 1440 × 900, after every path chip and after "Explore the tree" (and the Tree link), the chips and the whole tree are on screen
  below the top bar;
- H2: no stem, limb, root or twig line crosses a branch's name (points sampled along each path with getPointAtLength);
- H3: a dimmed lab box keeps an opaque fill;
- H4: the sticky strip's labelled row fits, or it shows icons, and the chosen branch is fully visible (1440, 1280, 1024; EN and ZH);
- H5: the legend shows 4 colour swatches in the tree's colours, each with its words; the big BUILD / USE captions are not faint;
- H6: no visible text (or tooltip, label) in #map, the nav or the strip says map / domain / limb, 地图 / 领域;
- H7: every path chip has a tooltip that spells out its abbreviation; the strip's options say who each path is for;
- H8, H13 (390): the phone list's groups say what they are, a count never breaks inside itself, the chip without a count is not stretched
  (since 2026-10-13 phones show the whole tree in a drag box, owner; H8 and H13 now check that box, see below);
- H9, H10: a lab box's and a branch name's tooltips say what a click does.
Re-review round 1b added:
- N1: every box's tooltip promises a mini experiment only when its tile has one;
- N2: the H6 word check with each of the 9 branches and each path chosen (tile headers included);
- tablets: up to 1004 px the branch list replaced the tree; since 2026-10-13 (owner: “可以做成只显示其一部分，但是可以通过拖动来查看”)
  the whole tree is drawn at full size in a box the reader drags, like a map app; from 1005 px the whole tree fits its panel;
- O1: a fresh load and a reload at #map leave the chips and the whole tree on screen; O2: the chip row has its label;
- O3: no space after a Chinese colon in the first-live-stop line; O5: the "filled" status swatch does not look like evaluation's slate;
- O6, O7: "← Tree" and "See this path on the tree" have tooltips; O8: Back from a lab reached from a branch's tiles lands on its tile.
Round 3 (owner #1, #3): six path entries above the tree replace the route cards and the path chips. H1 now measures the line above the
tree (#mapBar); O2 checks that line's words; O5 became "no development stage (live, planned, x/y) on the tree"; H11, H12 became the
entries' and the path tiles' start links (first lab, data-path and data-route-start).
2026-10-13 (owner): on a desktop the tree is about 80% of the window wide (D1: 1440, 1512, 1728); up to 1004 px the tree box
starts on the trunk at the bottom and the line above says to drag (D2), a mouse drag pans it without opening a lab (D3); labs with a
知识梳理 carry a book icon on the tree and on their tile (D4).
Run: python3 tests/map_geometry.py [page.html]   (screenshots in shots/map/; builds out/tree-geo.html when no page is given)"""
import json, pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
P = pathlib.Path(__file__).resolve().parent.parent
W = P / 'shots' / 'map'; W.mkdir(parents=True, exist_ok=True)
page = sys.argv[1] if len(sys.argv) > 1 else None
if not page:
    subprocess.run([sys.executable, str(P / 'build.py'), 'out/tree-geo.html'], check=True, cwd=P, stdout=subprocess.DEVNULL)
    page = str(P / 'out' / 'tree-geo.html')
WRAP = P / 'out' / f'tree-geo-wrap-{pathlib.Path(page).stem}.html'
WRAP.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + pathlib.Path(page).read_text() + '</body></html>')
URL = 'file://' + str(WRAP)
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


GEO = """()=>{const svg=document.getElementById('mapSvg'), vb=svg.viewBox.baseVal;
  const bb=e=>{const r=e.getBBox(), m=e.getCTM(), s=svg.getCTM().inverse().multiply(m); const p=(x,y)=>{const q=svg.createSVGPoint(); q.x=x; q.y=y; return q.matrixTransform(s);};
    const a=p(r.x,r.y), b=p(r.x+r.width,r.y+r.height); return [Math.min(a.x,b.x),Math.min(a.y,b.y),Math.max(a.x,b.x),Math.max(a.y,b.y)];};
  const boxes=[...svg.querySelectorAll('.tnode')].map(g=>({id:g.dataset.lab,box:bb(g.querySelector('.tbx')),dim:g.classList.contains('dim'),
    text:(()=>{const t=g.querySelector('.tlb'); return {w:t.getComputedTextLength(),squeezed:t.hasAttribute('textLength'),s:t.textContent};})()}));
  const names=[...svg.querySelectorAll('.world')].map(g=>({id:g.dataset.world,box:bb(g.querySelector('.tw-bg')),dim:g.classList.contains('dim')}));
  const nums=[...svg.querySelectorAll('#routeLayer .tnum')].map(g=>({n:+g.textContent,box:bb(g.querySelector('circle'))}));
  const bars=svg.querySelector('.tr-bars'), caps=[...svg.querySelectorAll('.tr-cap, .cap-sub')].map(bb);
  const barRows=[bb(svg.querySelector('.tr-bars .tb-t'))].concat([...svg.querySelectorAll('.tr-bars g')].map(bb));
  const rows=[...svg.querySelectorAll('.tr-bars g')].map(g=>g.querySelector('.tb-c').textContent);
  const axis=[...svg.querySelectorAll('.tr-axis text')].map(bb);
  return {vb:[vb.x,vb.y,vb.width,vb.height],boxes,names,nums,bars:bb(bars),barRows,caps,rows,axis,scale:svg.getScreenCTM().a,svgRect:(()=>{const r=svg.getBoundingClientRect(); return [r.top,r.bottom,r.width,r.height]})()}}"""

def ov(a, b, pad=0):
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


# H2: points along every stem, limb, root and twig, against every branch name's box (its hover background, which pads the text)
CROSS = """()=>{const svg=document.getElementById('mapSvg'), inv=svg.getCTM().inverse();
  const root=(el,x,y)=>{const q=svg.createSVGPoint(); q.x=x; q.y=y; return q.matrixTransform(inv.multiply(el.getCTM()));};
  const names=[...svg.querySelectorAll('.world')].map(g=>{const r=g.querySelector('.tw-bg'), b=r.getBBox(), a=root(r,b.x,b.y), c=root(r,b.x+b.width,b.y+b.height); return {id:g.dataset.world,box:[a.x,a.y,c.x,c.y]};});
  const els=[...svg.querySelectorAll('.tr-limbs path:not(.lf), .tr-roots path, .tr-twigs line')], hits=[]; let pts=0;
  els.forEach(el=>{const h=(parseFloat(getComputedStyle(el).strokeWidth)||1)/2, P=[];
    if(el.tagName==='line'){const x1=el.x1.baseVal.value,y1=el.y1.baseVal.value,x2=el.x2.baseVal.value,y2=el.y2.baseVal.value,k=Math.max(2,Math.ceil(Math.hypot(x2-x1,y2-y1)/1.5));
      for(let i=0;i<=k;i++) P.push(root(el,x1+(x2-x1)*i/k,y1+(y2-y1)*i/k));}
    else {if(!el.getAttribute('d')) return; const L=el.getTotalLength(); for(let s=0;s<=L+0.01;s+=Math.max(0.5,Math.min(1.5,L/4))){const p=el.getPointAtLength(Math.min(s,L)); P.push(root(el,p.x,p.y));}}
    pts+=P.length;
    for(const p of P){const nm=names.find(m=>p.x+h>m.box[0]&&p.x-h<m.box[2]&&p.y+h>m.box[1]&&p.y-h<m.box[3]);
      if(nm){hits.push({el:el.getAttribute('class'),thread:el.dataset.thread||'',name:nm.id,at:[Math.round(p.x),Math.round(p.y)]}); break;}}});
  return {lines:els.length,pts,hits};}"""

# H1: the line above the tree (what the tree shows; it replaced the chip row in round 3) and the whole tree under the top bar
VIEW = """()=>{const bar=document.querySelector('.topbar').getBoundingClientRect().bottom, c=document.getElementById('mapBar').getBoundingClientRect(), s=document.getElementById('mapSvg').getBoundingClientRect();
  return {bar:Math.round(bar), chipsTop:Math.round(c.top), chipsBottom:Math.round(c.bottom), svgTop:Math.round(s.top), svgBottom:Math.round(s.bottom), vh:innerHeight, y:Math.round(scrollY)}}"""
SETTLE = "()=>new Promise(r=>{let last=-1,same=0,t=0;const f=()=>{const y=Math.round(scrollY); same=y===last?same+1:0; last=y; t+=50; if(same>=4||t>3000) r(y); else setTimeout(f,50)}; f();})"
HEADING_TOP = "()=>{const s=document.getElementById('mapH'); window.scrollTo(0, scrollY+s.getBoundingClientRect().top-70);}"   # where #map used to land: the heading under the top bar


def in_view(pg, tag, what):
    v = pg.evaluate(VIEW)
    ok = v['chipsTop'] >= v['bar'] and v['chipsBottom'] <= v['svgTop'] and v['svgTop'] >= v['bar'] and v['svgBottom'] <= v['vh']
    check(f'{tag}: after {what}, the line above the tree and the whole tree are on screen below the top bar (H1)', ok, v)


# H3: a dimmed box's fill is opaque; nothing above it is see-through
DIMS = """()=>[...document.querySelectorAll('#mapSvg .tnode.dim')].map(g=>{const r=g.querySelector('.tbx'), cs=getComputedStyle(r);
  let a=1; const f=cs.fill; const m=f.match(/\\/\\s*([\\d.]+%?)\\s*\\)$/) || f.match(/^rgba\\([^,]+,[^,]+,[^,]+,\\s*([\\d.]+)\\)$/); if(m) a=m[1].endsWith('%')?parseFloat(m[1])/100:parseFloat(m[1]);
  if(f==='none'||f==='transparent') a=0; return {id:g.dataset.lab, a, op:+getComputedStyle(g).opacity*+cs.opacity*+cs.fillOpacity, f}})"""
# H4: the strip
STRIP = """()=>{const s=document.getElementById('domStrip'), w=s.querySelector('.ds-worlds'), b=w.querySelector('.ds-btn[aria-pressed="true"]'), a=w.getBoundingClientRect(), r=b?b.getBoundingClientRect():null;
  return {sw:w.scrollWidth, cw:w.clientWidth, icons:s.classList.contains('ds-icons'), sel:b?b.dataset.world:null, vis:!!r&&r.left>=a.left-0.5&&r.right<=a.right+0.5&&r.width>0, label:b?getComputedStyle(b.querySelector('span')).display!=='none':null}}"""
# H6: words that must not be on the home's tree, nav and strip (visible text, tooltips, labels)
WORDS = r"""()=>{const re=/\b(maps?|domains?|limbs?)\b|地图|领域/i, out=[];
  const els=[document.getElementById('map'), document.querySelector('.topbar nav'), document.getElementById('domStrip')];
  els.forEach(e=>{if(!e) return; (e.innerText||'').split('\n').forEach(l=>{if(re.test(l)) out.push('text: '+l.trim().slice(0,90));});
    [e,...e.querySelectorAll('[data-tipx],[aria-label],[title]')].forEach(x=>['data-tipx','aria-label','title'].forEach(k=>{const v=x.getAttribute(k); if(v&&re.test(v)) out.push(k+': '+v.slice(0,90));}));});
  const svg=document.getElementById('mapSvg'); [...svg.querySelectorAll('text')].forEach(t=>{if(re.test(t.textContent)) out.push('svg: '+t.textContent);});
  return out;}"""


def hover_tip(pg, sel):
    bb = pg.evaluate(f"()=>{{const e=document.querySelector({json.dumps(sel)}); if(!e) return null; const r=e.getBoundingClientRect(); return [r.x+Math.min(r.width/2,40), r.y+r.height/2]}}")
    if not bb:
        return ''
    pg.mouse.move(2, 2); pg.wait_for_timeout(40); pg.mouse.move(bb[0] - 3, bb[1]); pg.mouse.move(bb[0], bb[1]); pg.wait_for_timeout(200)
    t = pg.evaluate("()=>{const t=document.getElementById('tip'); return t&&!t.hidden?t.innerText.trim():''}")
    pg.mouse.move(2, 2); pg.wait_for_timeout(40)
    return t


WORLDS9 = ['roots', 'models', 'eval', 'training', 'post', 'inference', 'apps', 'agents', 'product']
TREEVIEW = """()=>{const cv=document.querySelector('#map .map-canvas'), ms=document.querySelector('#map .map-scroll'), dg=document.getElementById('domGrid');
  return {tree:!!cv.offsetParent, list:getComputedStyle(dg).display!=='none', sw:ms.scrollWidth, cw:ms.clientWidth, svgw:Math.round(document.getElementById('mapSvg').getBoundingClientRect().width)}}"""
# D2: where the drag box starts (trunk centred, scrolled to the ground) and what the line above says
PANVIEW = """()=>{const ms=document.querySelector('#map .map-scroll'), svg=document.getElementById('mapSvg'), sr=svg.getBoundingClientRect(), mr=ms.getBoundingClientRect();
  const k=sr.width/1224, trunk=sr.left+624*k-mr.left, ground=sr.top+592*k-mr.top;
  return {trunk:Math.round(trunk), ground:Math.round(ground), cw:ms.clientWidth, ch:ms.clientHeight, sl:Math.round(ms.scrollLeft), now:(document.getElementById('mapNow')||{}).innerText||''}}"""
EXPAND = {'serving': 'infrastructure', 'apps': 'retrieval-augmented generation', 'research': 'research scientist'}   # four paths since 2026-10-11

with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ['zh', 'en']:
        for dark in (False, True):
            pg = b.new_page(viewport={'width': 1440, 'height': 900}, color_scheme='dark' if dark else 'light', reduced_motion='reduce')
            errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
            tag = f'{lang}{" dark" if dark else ""}'
            pg.goto(URL + '#map'); pg.wait_for_timeout(900)
            in_view(pg, tag + ' (before the language switch)', 'a fresh load at #map (O1)')
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
            lead = pg.evaluate("()=>{const e=document.getElementById('mapNow'); return e&&e.offsetParent?e.innerText.trim():''}")
            check(f'{tag}: the line above the tree says what the tree shows (O2, round 3)', len(lead) > 10 and (('路线' in lead) if lang == 'zh' else ('path' in lead)), lead)
            g = pg.evaluate(GEO)
            ids = [x['id'] for x in g['boxes']]
            check(f'{tag}: every lab has exactly one box on the tree', len(ids) == len(set(ids)) and len(ids) >= 60, (len(ids), len(set(ids))))
            bx = g['boxes']
            clash = [(a['id'], c['id']) for i, a in enumerate(bx) for c in bx[i + 1:] if ov(a['box'], c['box'], 3)]
            check(f'{tag}: no two lab boxes touch (3 px apart at least)', not clash, clash[:5])
            fit = [(x['id'], round(x['text']['w'], 1)) for x in bx if x['text']['w'] > (x['box'][2] - x['box'][0]) - 8 and not x['text']['squeezed']]
            check(f'{tag}: every lab name fits its box', not fit, fit[:5])
            sq = [x['text']['s'] for x in bx if x['text']['squeezed']]
            check(f'{tag}: at most 3 names squeezed to fit', len(sq) <= 3, sq)
            nm = g['names']
            nb = [(a['id'], c['id']) for a in nm for c in bx if ov(a['box'], c['box'], 2)]
            check(f'{tag}: no branch name touches a lab box', not nb, nb[:5])
            nn = [(a['id'], c['id']) for i, a in enumerate(nm) for c in nm[i + 1:] if ov(a['box'], c['box'], 2)]
            check(f'{tag}: no two branch names touch', not nn, nn)
            # each row of the bar chart (and its title) on its own: the rows' names are the branches' names, of different lengths
            barc = [(x['id'], k) for x in bx + nm for k, r in enumerate(g['barRows']) if ov(x['box'], r, 4)]
            check(f'{tag}: the bar chart touches no box and no name', not barc, barc)
            capc = [x['id'] for x in bx + nm for c in g['caps'] if ov(x['box'], c, 2)]
            check(f'{tag}: the limbs’ big names touch no box and no name', not capc, capc)
            X0, Y0, VW, VH = g['vb']
            out_ = [x['id'] for x in bx + nm if x['box'][0] < X0 or x['box'][1] < Y0 or x['box'][2] > X0 + VW or x['box'][3] > Y0 + VH] + (['bars'] if g['bars'][2] > X0 + VW or g['bars'][3] > Y0 + VH else []) \
                + [f'axis {k}' for k, a in enumerate(g['axis']) if a[0] < X0 or a[2] > X0 + VW]
            check(f'{tag}: nothing leaves the drawing (axis names included)', not out_, out_)
            # H2
            cr = pg.evaluate(CROSS)
            check(f'{tag}: no stem, limb, root or twig crosses a branch’s name (H2; {cr["lines"]} lines, {cr["pts"]} points)', not cr['hits'] and cr['lines'] >= 20 and cr['pts'] > 2000, cr['hits'][:4])
            # H5: the legend's colours and the limbs' big names
            lg = pg.evaluate("""()=>{const sw=[...document.querySelectorAll('#mapLegend .tsw.c')].map(i=>{const cs=getComputedStyle(i); return {bg:cs.backgroundColor, img:cs.backgroundImage, txt:i.parentElement.innerText.trim(), w:i.getBoundingClientRect().width}});
              const st=k=>{const e=document.querySelector('#mapSvg .tr-limbs .st.'+k); return e?getComputedStyle(e).stroke:null};
              const cap=[...document.querySelectorAll('#mapSvg .tr-cap')].map(e=>+getComputedStyle(e).opacity);
              return {sw, build:st('build'), use:st('use'), eval:st('eval'), cap}}""")
            sw = lg['sw']
            paints = [x['bg'] + '|' + x['img'] for x in sw]
            check(f'{tag}: the legend shows 4 colour swatches, each with its words (H5)', len(sw) == 4 and all(len(x['txt']) >= 2 and x['w'] >= 8 for x in sw) and len(set(paints)) == 4, sw)
            check(f'{tag}: the legend’s colours are the tree’s (building, using, evaluation; the trunk’s two woods)', len(sw) == 4 and sw[1]['bg'] == lg['build'] and sw[2]['bg'] == lg['use'] and sw[3]['bg'] == lg['eval'] and 'gradient' in sw[0]['img'], (sw, lg['build'], lg['use'], lg['eval']))
            check(f'{tag}: the big BUILD / USE captions are not faint (H5)', len(lg['cap']) == 2 and min(lg['cap']) >= (0.4 if dark else 0.3), lg['cap'])
            # round 3 (owner #3): no development stage on the tree, its legend or its bar chart (O5's status swatch went with it)
            dev = pg.evaluate(r"""()=>{const t=[...document.querySelectorAll('#pathList, #mapBar, #mapLegend, #tilesHead, #domStrip, #domGrid, #tiles .tile-meta')].map(e=>e.innerText).join(' ')+' '+[...document.querySelectorAll('#mapSvg text')].map(e=>e.textContent).join(' ');
              return (t.match(/已上线|上线|开发中|规划中|\blive\b|in progress|planned|next to build|\d+\s*\/\s*\d+/gi)||[]).slice(0,6)}""")
            check(f'{tag}: no development stage (live, planned, x/y counts) on the tree, its legend or its bar chart', not dev, dev)
            # H6
            bad = pg.evaluate(WORDS)
            check(f'{tag}: no “map”, “domain” or “limb” (地图, 领域) on the tree, the nav or the strip (H6)', not bad, bad[:6])
            # H7: chips' tooltips, the strip's options
            for pid in pg.evaluate("()=>[...document.querySelectorAll('.pentry .pathchip')].map(e=>e.dataset.path)"):
                # since 2026-10-12 the whole entry is the path's button (no tooltip): the entry itself says who the path is for
                tip = pg.evaluate(f"()=>{{const b=document.querySelector('.pentry .pathchip[data-path=\"{pid}\"]'), e=b.closest('.pentry'); return (b.getAttribute('aria-label')||'')+' | '+e.innerText}}")
                need = {'serving': {'en': 'AI infrastructure', 'zh': 'AI 基础设施'}, 'research': {'en': 'research scientists', 'zh': '研究科学家'}}.get(pid, {}).get(lang, '')   # the entry's own words for Infra and RS / RE
                check(f'{tag}: the path entry “{pid}” is named for screen readers and says who it is for{", spelling out its abbreviation" if need else ""} (H7)', len(tip) > 40 and need.lower() in tip.lower(), tip[:160])
            opts = pg.evaluate("()=>[...document.querySelectorAll('#pathSel option')].map(o=>({v:o.value,t:o.title||''}))")
            check(f'{tag}: every option of the strip’s path list says who the path is for (H7)', len(opts) >= 5 and all(len(o['t']) > (12 if lang == 'zh' else 20) for o in opts) and all(EXPAND[o['v']] in o['t'] for o in opts if o['v'] in EXPAND), opts)
            # H9, H10: a box's and a branch name's tooltips say what a click does
            pg.evaluate("()=>document.getElementById('mapSvg').scrollIntoView({block:'center'})"); pg.wait_for_timeout(150)
            tb = hover_tip(pg, '#mapSvg .tnode[data-lab="kv-cache"] .tbx'); tw = hover_tip(pg, '#mapSvg .world[data-world="inference"] .wname')
            click = 'Click' if lang == 'en' else '点击'
            check(f'{tag}: a lab box’s tooltip ends with what a click does (H9)', click in tb and len(tb) > 30, tb)
            check(f'{tag}: a branch name’s tooltip says what a click shows (H10)', click in tw and len(tw) > 20, tw)
            r = pg.evaluate("()=>{const r=document.getElementById('mapSvg').getBoundingClientRect(); return [r.top,r.bottom,r.width,r.height,innerHeight]}")
            check(f'{tag}: at 1440 × 900 the whole tree fits on screen, drawn at least 1,100 px wide', r[3] <= r[4] - 60 and g['scale'] * 1200 >= 1100, (r, round(g['scale'] * 1200)))
            # the bar chart counts what the tree shows
            want = pg.evaluate("""()=>{const L=[...document.querySelectorAll('#mapSvg .tnode')]; const W=['inference','agents','apps','product','eval','post','training','models','roots'];
              return W.map(w=>String(L.filter(g=>window.__treeW&&window.__treeW[g.dataset.lab]===w).length));}""") if pg.evaluate("()=>!!window.__treeW") else None
            if want is not None: check(f'{tag}: the bar chart counts each branch’s labs', g['rows'] == want, (g['rows'], want))
            if not dark: pg.query_selector('#mapSvg').screenshot(path=str(W / f'tree_{lang}.png'))
            # H1: "Explore the tree" (the address changes to #map, and again when it already is #map) and the Tree link
            pg.evaluate("()=>{history.replaceState(null,'','#top'); window.scrollTo(0,0)}"); pg.wait_for_timeout(100)
            pg.click('.topbar a[href="#map"]'); pg.evaluate(SETTLE); in_view(pg, tag, 'the top bar’s tree link (the hero’s “Explore the tree” was removed 2026-10-11)')
            pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(100)
            pg.click('.topbar a[href="#map"]'); pg.evaluate(SETTLE); in_view(pg, tag, 'the top bar’s tree link with the address already at #map')
            pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(100)
            pg.click('.topbar nav a[href="#map"]'); pg.evaluate(SETTLE); in_view(pg, tag, 'the Tree link in the top bar')
            # every path: in view after a real click (from where #map used to land), numbers clear of each other and of other labs, dimming exactly off the path
            paths = pg.evaluate("()=>[...document.querySelectorAll('.pentry .pathchip')].map(e=>e.dataset.path).filter(Boolean)")
            for pid in paths:
                pg.evaluate(HEADING_TOP); pg.wait_for_timeout(120)
                pg.click(f'.pathchip[data-path="{pid}"]'); pg.evaluate(SETTLE); pg.wait_for_timeout(100)
                in_view(pg, tag, f'the path entry for “{pid}”')
                if not dark:
                    bad = pg.evaluate(WORDS)
                    check(f'{tag} path {pid}: no “map”, “domain” or “limb” (地图, 领域) with this path chosen (N2)', not bad, bad[:4])
                g2 = pg.evaluate(GEO); on = pg.evaluate(f"()=>window.__treePath?window.__treePath('{pid}'):null")
                nums = g2['nums']
                check(f'{tag} path {pid}: numbered 1 to N', [x['n'] for x in nums] == list(range(1, len(nums) + 1)) and len(nums) >= 5, [x['n'] for x in nums][:10])
                nn2 = [(a['n'], c['n']) for i, a in enumerate(nums) for c in nums[i + 1:] if ov(a['box'], c['box'], 1)]
                check(f'{tag} path {pid}: no two numbers touch', not nn2, nn2[:5])
                if on:
                    offp = [x for x in g2['boxes'] if x['id'] not in on]
                    hit = [(k['n'], x['id']) for k in nums for x in offp if ov(k['box'], x['box'], 1)]
                    check(f'{tag} path {pid}: no number sits on a lab off the path', not hit, hit[:5])
                    hitn = [(k['n'], x['id']) for k in nums for x in g2['names'] if ov(k['box'], x['box'], 1)]
                    check(f'{tag} path {pid}: no number sits on a branch name', not hitn, hitn[:5])
                    dimw = sorted(x['id'] for x in g2['boxes'] if x['dim']); wantd = sorted(x['id'] for x in g2['boxes'] if x['id'] not in on)
                    check(f'{tag} path {pid}: exactly the labs off the path are dimmed', dimw == wantd, (len(dimw), len(wantd)))
                    dm = pg.evaluate(DIMS)
                    see = [(x['id'], x['f'], x['op']) for x in dm if x['a'] < 0.999 or x['op'] < 0.999]
                    check(f'{tag} path {pid}: every dimmed box keeps an opaque fill, so no line shows through its name (H3)', dm and not see, see[:4])
                if not dark and pid in ('serving', 'research', 'transformer'): pg.query_selector('#mapSvg').screenshot(path=str(W / f'tree_{lang}_{pid}.png'))
            pg.evaluate(HEADING_TOP); pg.wait_for_timeout(100)
            tip = hover_tip(pg, '.pathchip[data-path=""]')
            check(f'{tag}: “No path, by branch” says what it does on hover (H7)', len(tip) > 20, tip)
            pg.click('.pathchip[data-path=""]'); pg.evaluate(SETTLE); in_view(pg, tag, '“No path, by branch”')
            check(f'{tag}: with no path chosen, “No path, by branch” is not offered again', pg.evaluate("()=>!document.querySelector('.pathchip[data-path=\"\"]').offsetParent"))
            # round 3 (owner #1): every path entry's start opens its first lab and says it is a choice of this path (H11 and H12 retired with
            # the route cards and the not-yet-live stops)
            if not dark:
                st = pg.evaluate("""()=>[...document.querySelectorAll('.pentry')].map(e=>{const a=e.querySelector('a[data-route-start]'); return {pid:e.dataset.pid, h:a&&a.getAttribute('href'), p:a&&a.dataset.path, r:a&&a.dataset.routeStart, first:'#'+window.__treePath(e.dataset.pid)[0]}})""")
                check(f'{tag}: each of the four path entries starts at its first lab, as a choice of that path', len(st) == 4 and all(x['h'] == x['first'] and x['p'] == x['pid'] and x['r'] == x['pid'] for x in st), st)
                pg.evaluate(HEADING_TOP); pg.wait_for_timeout(80); pg.click('.pathchip[data-path="research"]'); pg.evaluate(SETTLE)
                st = pg.evaluate("()=>{const a=document.querySelector('#tilesHead .th-go a[data-route-start]'); return a?{h:a.getAttribute('href'), p:a.dataset.path, r:a.dataset.routeStart, t:a.innerText}:null}")
                check(f'{tag}: the research path’s tiles start at stop 1, as a choice of the path', bool(st) and st['h'] == '#tensors' and st['p'] == st['r'] == 'research' and st['t'].startswith('进入第一站' if lang == 'zh' else 'Stop 1'), st)
                pg.evaluate(HEADING_TOP); pg.wait_for_timeout(80); pg.click('.pathchip[data-path=""]'); pg.evaluate(SETTLE)
            # H4: the strip after choosing each branch on the tree (light only: the strip does not change with the theme)
            if not dark:
                bad4, badw, tz = [], [], {}
                for w_ in WORLDS9:
                    pg.evaluate(f"()=>document.querySelector('#mapSvg .world[data-world=\"{w_}\"]').dispatchEvent(new MouseEvent('click',{{bubbles:true}}))"); pg.evaluate(SETTLE)
                    s = pg.evaluate(STRIP)
                    if not ((s['sw'] <= s['cw'] or s['icons']) and s['sel'] == w_ and s['vis'] and s['label']): bad4.append((w_, s))
                    badw += [(w_, x) for x in pg.evaluate(WORDS)]
                    tz.update(pg.evaluate("()=>Object.fromEntries([...document.querySelectorAll('#tiles .tile')].map(t=>[t.dataset.lab, !t.querySelector('.tile-tz').classList.contains('tz-none')]))"))
                check(f'{tag}: the strip’s names fit or it shows icons, and the chosen branch is visible with its name (H4)', not bad4, bad4[:3])
                check(f'{tag}: no “map”, “domain” or “limb” (地图, 领域) with any of the 9 branches chosen, tile headers included (N2)', not badw, badw[:4])
                # N1: each box's tooltip, against its tile
                tips = pg.evaluate("""()=>[...document.querySelectorAll('#mapSvg .tnode')].map(g=>{g.focus({preventScroll:true}); const t=document.getElementById('tip');
                  const s=t&&!t.hidden?t.innerText.trim():''; g.blur(); return [g.dataset.lab, s];})""")
                yes, soon = (('mini experiment below', 'is coming') if lang == 'en' else ('看它的小实验', '还在做'))
                wrong = [(lab, txt[-70:]) for lab, txt in tips if lab not in tz or (tz[lab] and (yes not in txt or soon in txt)) or (not tz[lab] and soon not in txt)]
                check(f'{tag}: every box’s tooltip matches its tile: a mini experiment below, or one that is coming (N1; {sum(tz.values())} with, {len(tz) - sum(tz.values())} without)',
                      len(tips) >= 60 and len(tz) == len(tips) and not wrong, wrong[:4])
            check(f'{tag}: no page errors', not errs, errs[:3])
            pg.close()
    # H1 with smooth scrolling (no reduced motion): the scroll ends with the chips and the whole tree on screen
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.goto(URL); pg.wait_for_timeout(600)
    pg.evaluate(HEADING_TOP); pg.wait_for_timeout(200)
    pg.click('.pathchip[data-path="apps"]'); pg.wait_for_timeout(200); pg.evaluate(SETTLE)
    in_view(pg, 'en smooth', 'the path chip “apps”')
    # O1: a load and a reload at #map (normal motion), the reload after the reader had scrolled to the top
    pg.goto(URL + '#map'); pg.wait_for_timeout(900); in_view(pg, 'en smooth', 'a load at #map (O1)')
    pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(200); pg.reload(); pg.wait_for_timeout(1200); in_view(pg, 'en smooth', 'a reload at #map (O1)')
    pg.close()
    # O6, O7: the labs' "← Tree" and "See this path on the tree" say where they go; O8: Back after a branch crumb lands on the lab's tile
    for lang in ['en', 'zh']:
        pg = b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce'); errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(URL); pg.wait_for_timeout(500)
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(300)
        word = '知识树' if lang == 'zh' else 'tree'
        for lab, root in (('architecture', '#view-dyn'), ('kv-cache', '#view-lab')):
            pg.evaluate(f"()=>{{location.hash='{lab}'}}"); pg.wait_for_timeout(900); pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(200)
            tip = hover_tip(pg, f'{root} .crumbs a[href="#map"]')
            check(f'{lang} {lab}: “← Tree” says where it goes on hover (O6)', word in tip and len(tip) > 15, tip)
        # the last stop of a finished path shows the link (DPO's path goes on to the batch 7 evaluation labs, so it no longer ends there)
        # round 3: route links show only on a route the reader chose (owner #4), so choose the serving path first (its chip); its last
        # stop is no longer serving-capstone since batch 10 (the check failed from v20 on), so ask the path for it
        pg.evaluate("()=>{const c=document.querySelector('.pathchip[data-path=\"serving\"]'); if(c) c.click();}"); pg.wait_for_timeout(200)
        rl = pg.evaluate("()=>{const d=document.createElement('div'); d.innerHTML=ASP.shell.nextLab(ASP.catalog.PATHS.serving.labs.slice(-1)[0]); const a=d.querySelector('a[href^=\"#map~route:\"]'); return a?(a.dataset.tipx||''):null}")
        check(f'{lang}: “See this path on the tree” says where it goes on hover (O7)', bool(rl) and word in rl, rl)
        if lang == 'en':
            pg.evaluate("()=>{location.hash='architecture'}"); pg.wait_for_timeout(900); pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(200)
            pg.click('#view-dyn .crumbs a[href="#map~models"]'); pg.wait_for_timeout(900)
            hd = pg.evaluate("()=>Math.round(document.getElementById('tilesHead').getBoundingClientRect().top)")
            check('en: a lab’s branch crumb still lands on the branch’s tiles, from their header', 0 <= hd <= 900 - 150, hd)
            pg.click('#tile-sampling .tile-q a'); pg.wait_for_timeout(900)
            pg.evaluate("()=>history.back()"); pg.wait_for_timeout(1000)
            bk = pg.evaluate("()=>{const t=document.getElementById('tile-sampling'); if(!t) return null; const r=t.getBoundingClientRect(); return {hash:location.hash, mid:Math.round(r.top+r.height/2), vh:innerHeight, flash:t.classList.contains('flash')}}")
            check('en: Back from a lab opened from a branch’s tiles lands on that lab’s tile and flashes it (O8, R11.7)', bool(bk) and bk['hash'] == '#map~models' and 80 <= bk['mid'] <= bk['vh'] - 40 and bk['flash'], bk)
        check(f'{lang} labs: no page errors', not errs, errs[:3])
        pg.close()
    # D1 (2026-10-13, owner: “我希望要有8成宽”): on a laptop or desktop screen the tree is about 80% of the window wide
    for vw, vh in ((1440, 900), (1512, 860), (1728, 1000)):
        pg = b.new_page(viewport={'width': vw, 'height': vh}, reduced_motion='reduce')
        pg.goto(URL + '#map'); pg.wait_for_timeout(600)
        tw = pg.evaluate("()=>[Math.round(document.getElementById('mapSvg').getBoundingClientRect().width), document.documentElement.scrollWidth-innerWidth]")
        check(f'{vw}×{vh}: the tree is about 80% of the window wide, no sideways overflow (D1)', tw[0] >= 0.78 * vw and tw[1] <= 0, (tw, round(tw[0] / vw, 2)))
        if vw == 1440:
            # D4: every lab with a 知识梳理 has the book on its tree box and on its tile, and no other lab does
            want = sorted(re.search(r"\bID\s*=\s*'([\w-]+)'", f.read_text()).group(1) for f in (P / 'labs').glob('*.js') if 'registerPrimer(' in f.read_text())
            onTree = pg.evaluate("()=>[...document.querySelectorAll('#mapSvg .tnode')].filter(g=>g.querySelector('.tbk')).map(g=>g.dataset.lab).sort()")
            onTiles = set()
            for w_ in WORLDS9:
                pg.evaluate(f"()=>document.querySelector('#mapSvg .world[data-world=\"{w_}\"]').dispatchEvent(new MouseEvent('click',{{bubbles:true}}))"); pg.wait_for_timeout(120)
                onTiles |= set(pg.evaluate("()=>[...document.querySelectorAll('#tiles .tile')].filter(t=>t.querySelector('.tile-pr svg')).map(t=>t.id.replace(/^tile-/,''))"))
            check('1440: the labs with a 知识梳理, and only they, have a book on the tree and on their tile (D4)', len(want) >= 14 and onTree == want and sorted(onTiles) == want, (len(want), [x for x in want if x not in onTree], [x for x in want if x not in onTiles], sorted(set(onTiles) - set(want))))
            pg.evaluate("()=>document.querySelector('#mapSvg .world[data-world=models]').dispatchEvent(new MouseEvent('click',{bubbles:true}))"); pg.evaluate(SETTLE)
            pg.evaluate("()=>document.querySelector('#tiles .tile-pr').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
            tip = hover_tip(pg, '#tiles .tile-pr')
            check('1440: the tile’s book says on hover what it means (D4)', '知识梳理' in tip or 'primer' in tip, tip)
        pg.close()
    for w in (1280, 1024, 1005, 1004, 900, 720):
        for lang in (['en', 'zh'] if w in (1280, 1024, 720) else ['en']):
            pg = b.new_page(viewport={'width': w, 'height': 800}, reduced_motion='reduce'); errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
            pg.goto(URL + '#map'); pg.wait_for_timeout(600)
            if lang != 'en':
                pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(300)
            ovf = pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
            check(f'{w} {lang}: no horizontal overflow, no page errors', ovf <= 0 and not errs, (ovf, errs[:2]))
            # the whole tree in its panel, or the branch list in its place (R11.10; re-review: at 720 a third of the tree was cut off)
            tv = pg.evaluate(TREEVIEW)
            want_list = w <= 1004   # since 2026-10-13: the full-size tree in a drag box, no branch list (owner)
            check(f'{w} {lang}: {"the whole tree at full size in a box the reader drags, no branch list" if want_list else "the whole tree fits its panel, no sideways scrolling"}', (tv['tree'] and not tv['list'] and tv['svgw'] >= 1200 and tv['sw'] > tv['cw'] and tv['cw'] < w) if want_list else (tv['tree'] and not tv['list'] and tv['sw'] <= tv['cw']), tv)
            if w == 720 and lang == 'en':
                # loaded with the list, then widened: the tree, measured while hidden, is measured again (no stem crosses a name)
                pg.set_viewport_size({'width': 1440, 'height': 900}); pg.wait_for_timeout(500)
                cr = pg.evaluate(CROSS); nm0 = pg.evaluate("()=>[...document.querySelectorAll('#mapSvg .world .tw-bg')].filter(r=>+r.getAttribute('width')<20).length")
                check('720 → 1440 en: after widening, the tree is measured again: names have their size, no stem crosses one', not cr['hits'] and nm0 == 0, (nm0, cr['hits'][:3]))
                pg.set_viewport_size({'width': w, 'height': 800}); pg.wait_for_timeout(500)
            if want_list:
                pg.evaluate(HEADING_TOP); pg.wait_for_timeout(100)
                pg.click('.pathchip[data-path="serving"]'); pg.evaluate(SETTLE); pg.wait_for_timeout(100)
                pv = pg.evaluate(PANVIEW)
                check(f'{w} {lang}: the drag box starts on the trunk at the ground, the line above says to drag (D2)', 0.3 * pv['cw'] <= pv['trunk'] <= 0.7 * pv['cw'] and 0 <= pv['ground'] <= pv['ch'] and (('拖动' in pv['now']) if lang == 'zh' else ('Drag' in pv['now'])), pv)
                if w == 900:
                    # D3: a mouse drag pans the box and opens nothing; a plain click on a box still goes to its tile
                    pg.evaluate("()=>document.querySelector('#map .map-scroll').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
                    bx = pg.evaluate("()=>{const r=document.querySelector('#map .map-scroll').getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]}")
                    sl0 = pg.evaluate("()=>document.querySelector('#map .map-scroll').scrollLeft"); h0 = pg.evaluate("()=>location.hash")
                    pg.mouse.move(bx[0], bx[1]); pg.mouse.down(); pg.mouse.move(bx[0] - 80, bx[1], steps=4); pg.mouse.move(bx[0] - 150, bx[1], steps=4); pg.mouse.up(); pg.wait_for_timeout(250)
                    sl1 = pg.evaluate("()=>document.querySelector('#map .map-scroll').scrollLeft"); h1 = pg.evaluate("()=>location.hash")
                    check('900 en: a mouse drag pans the tree box and opens nothing (D3)', abs((sl1 - sl0) - 150) <= 3 and h1 == h0, (sl0, sl1, h0, h1))
                    pg.evaluate("()=>document.querySelector('#mapSvg .tnode[data-lab=sampling]').scrollIntoView({block:'center',inline:'center'})"); pg.wait_for_timeout(150)
                    nb = pg.evaluate("()=>{const r=document.querySelector('#mapSvg .tnode[data-lab=sampling]').getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]}")
                    pg.mouse.click(nb[0], nb[1]); pg.evaluate(SETTLE); pg.wait_for_timeout(200)
                    ft = pg.evaluate("()=>{const t=document.getElementById('tile-sampling'); if(!t) return null; const r=t.getBoundingClientRect(); return {top:Math.round(r.top), vh:innerHeight, flash:t.classList.contains('flash')}}")
                    check('900 en: after a drag, a plain click on a lab box still goes to its tile (D3, R11.7)', bool(ft) and 0 <= ft['top'] <= ft['vh'] - 60 and ft['flash'], ft)
                    pg.goto(URL + '#map'); pg.wait_for_timeout(600)
                pg.evaluate(HEADING_TOP); pg.wait_for_timeout(100)
                pg.click('.pathchip[data-path="serving"]'); pg.evaluate(SETTLE); pg.wait_for_timeout(100)
                th = pg.evaluate("()=>{const r=document.getElementById('tilesHead').getBoundingClientRect(); return [Math.round(r.top), innerHeight, document.querySelectorAll('#tiles .tile-num').length, document.querySelectorAll('#mapSvg .tnode.dim').length]}")
                check(f'{w} {lang}: with the drag box, a path chip scrolls to its numbered tiles and dims the labs off the path on the tree (R11.6)', 0 <= th[0] <= th[1] - 150 and th[2] >= 5 and th[3] >= 1, th)
                pg.close(); continue
            if w == 1280:
                pg.evaluate(HEADING_TOP); pg.wait_for_timeout(100)
                pg.click('.pathchip[data-path="research"]'); pg.evaluate(SETTLE)
                in_view(pg, f'{w} {lang}', 'the path entry for “research”')
            bad4 = []
            for w_ in ['roots', 'eval', 'inference', 'apps', 'agents', 'product']:
                pg.evaluate(f"()=>document.querySelector('#mapSvg .world[data-world=\"{w_}\"]').dispatchEvent(new MouseEvent('click',{{bubbles:true}}))"); pg.evaluate(SETTLE)
                s = pg.evaluate(STRIP)
                if not ((s['sw'] <= s['cw'] or s['icons']) and s['sel'] == w_ and s['vis'] and s['label']): bad4.append((w_, s))
            check(f'{w} {lang}: the strip’s names fit or it shows icons, and the chosen branch is visible with its name (H4)', not bad4, bad4[:3])
            pg.close()
    # phones (390): the whole tree in a drag box (since 2026-10-13; before, a branch list)
    for lang in ['en', 'zh']:
        pg = b.new_page(viewport={'width': 390, 'height': 844}, reduced_motion='reduce'); errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(URL + '#map'); pg.wait_for_timeout(600)
        if lang != 'en':
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(300)
        ph = pg.evaluate("""()=>{const box=t=>{const r=t.closest('.tnode').querySelector('rect'); return r?r.getBBox().width:0};
          /* each lab name fits its box at full size, measured in the drag box (the tree is measured while visible) */
          const tt=[...document.querySelectorAll('#mapSvg .tnode text')], broken=tt.filter(t=>{const w=t.getBBox().width; return !(w>4&&w<=box(t)-4);}).map(t=>t.textContent);
          /* since 2026-10-12 the whole entry is the button (owner): its hit area matches the entry, no more and no less */
          const all=Math.max(...[...document.querySelectorAll('.pentry')].map(en=>{const h=en.querySelector(':scope>.pe-hit'); if(!h) return 99;
            const r=h.getBoundingClientRect(), q=en.getBoundingClientRect(); return Math.max(Math.abs(r.top-q.top),Math.abs(r.bottom-q.bottom),Math.abs(r.left-q.left),Math.abs(r.right-q.right));}));
          return {broken, counts:tt.length, all, list:getComputedStyle(document.getElementById('domGrid')).display}}""")
        tv = pg.evaluate(TREEVIEW); pv = pg.evaluate(PANVIEW)
        check(f'390 {lang}: the whole tree at full size in a drag box that starts on the trunk at the ground, and the line above says to drag (H8, D2)', tv['tree'] and ph['list'] == 'none' and tv['svgw'] >= 1200 and tv['cw'] <= 390 and 0.3 * pv['cw'] <= pv['trunk'] <= 0.7 * pv['cw'] and 0 <= pv['ground'] <= pv['ch'] and (('拖动' in pv['now']) if lang == 'zh' else ('Drag' in pv['now'])), (tv, pv))
        check(f'390 {lang}: every lab name fits its box on the full-size tree (H13)', ph['counts'] >= 55 and not ph['broken'], (ph['counts'], ph['broken'][:5]))
        check(f'390 {lang}: each path entry’s button covers exactly that entry (H13)', 0 <= ph['all'] <= 1.5, ph['all'])
        bad = pg.evaluate(WORDS)
        check(f'390 {lang}: no “map”, “domain” or “limb” (地图, 领域) in the list, the chips or the strip (H6)', not bad, bad[:5])
        # a path on a phone goes to its numbered tiles, as before (R11.6)
        pg.click('.pathchip[data-path="serving"]'); pg.evaluate(SETTLE); pg.wait_for_timeout(100)
        th = pg.evaluate("()=>{const r=document.getElementById('tilesHead').getBoundingClientRect(); return [Math.round(r.top), innerHeight, document.querySelectorAll('#tiles .tile-num').length]}")
        check(f'390 {lang}: a path chip scrolls to its numbered tiles', 0 <= th[0] <= th[1] - 200 and th[2] >= 5, th)
        check(f'390 {lang}: no page errors', not errs, errs[:3])
        pg.close()
    b.close()
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails if fails else 'none')
sys.exit(1 if fails else 0)
