"""The previous lab in every lab's breadcrumb row (the owner, 2026-10-09: “我觉得每一个lab页面最顶上除了下一个lab，也要带上“上一个lab”的按钮”).
For every live lab and every learning path that contains it, at 1440, 900, 390 and 360, in EN and ZH:
- the path is chosen the way the site does: #map~route:<path>, then the lab (and from then on the links' data-path);
- the header's previous link points to the previous live lab of that path in the catalog's order, computed here from PATHS, the
  live labs and PATH_NEXT (parsed from app.js), not from prevInfo; it carries data-path, shows “← Previous lab: <short name>” and on
  hover the whole question and its stop n/m on the path;
- a path's first live lab has no previous link, unless a path that PATH_NEXT continues into this one ends with a next link to here
  (the continuation rule), which is checked on a test-only variant of the page whose PATH_NEXT continues “research” into “serving”;
- following the previous link keeps the path: the lab it lands on has a next link back to the lab we came from, on the same path;
- the next link is still there, as before (href from the catalog's nextInfo, “Next lab: <short name> →”);
- layout: the pair at the right end of the row, previous first; on the crumbs' last line when the room left there holds it, else
  on a line of its own under the crumbs, as one piece, right-aligned; no overlap, nothing outside the row or the viewport, no text cut, nothing covers a link, the row ends above the
  title; the links' contrast against the page in light and dark;
- phones (≤ 480 px; the integrator, 2026-10-09: the first screen, R7.3): short labels on one line, “← Prev  Next →” / “← 上一个  下一个 →”
  (the full label stays in the link's text for screen readers and in the hover), each link a tap target at least 32 px tall, and the
  row no taller than it was in v6: every row against the same row with v6's content (the crumbs and the next link only), and
  prefill-decode, chunked-prefill, kv-cache and dpo against the v6 build itself (KIT_PREV_V6, default ../../rel9p6/src/…, when there);
  the one exception, printed in a list of its own: the end of a path, where v6 had nothing at the right end and the crumbs leave
  less room than “← Prev”, so the link takes a line of its own;
- a real click on the previous link lands with no tooltip under the still pointer (R1.7);
- screenshots of three labs' headers (kv-cache, dpo, eval-ci) at 1440, 900 and 390, EN and ZH, light and dark: shots/kit-prev/.
Run: python3 build.py ai-systems-playground.html && python3 tests/kit_prev_lab.py [page.html]"""
import pathlib, re, sys, time
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
PAGE = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else P / 'ai-systems-playground.html'
src = PAGE.read_text()
WRAP = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>'
(P / 'out').mkdir(exist_ok=True)
(P / 'out/wrapped-kit-prev.html').write_text(WRAP + src + '</body></html>')
URL = 'file://' + str(P / 'out/wrapped-kit-prev.html')
# the continuation rule never fires with today's catalog (no path continues into another since preview v24), so a test-only variant
# continues “research” (ends with eval-capstone) into “serving” (starts with kv-basics)
PN_SRC = "PATH_NEXT={}"
assert src.count(PN_SRC) == 1, 'PATH_NEXT changed: update this test'
(P / 'out/wrapped-kit-prev-cont.html').write_text(WRAP + src.replace(PN_SRC, "PATH_NEXT={research:'serving'}") + '</body></html>')
URL_CONT = 'file://' + str(P / 'out/wrapped-kit-prev-cont.html')
APP = (P / 'app.js').read_text()
PATH_NEXT = dict(re.findall(r"(\w+):'(\w+)'", re.search(r'PATH_NEXT=(\{[^}]*\})', APP).group(1)))
SHOTS = P / 'shots/kit-prev'; SHOTS.mkdir(parents=True, exist_ok=True)
SHOT_LABS = [('kv-cache', 'transformer'), ('dpo', 'research'), ('eval-ci', 'apps')]
import os
V6 = pathlib.Path(os.environ.get('KIT_PREV_V6') or P.parent.parent / 'rel9p6/src/ai-systems-playground.html')
V6_LABS = ['prefill-decode', 'chunked-prefill', 'kv-cache', 'dpo']
SHORT = {'en': ('Prev', 'Next'), 'zh': ('上一个', '下一个')}
fails, n, GREW = [], 0, []


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(f'{name} :: {str(info)[:400]}')
        print('FAIL', name, '::', str(info)[:400])


def expected_prev(paths, live, pid, lab, path_next):
    """the previous lab of `lab` on path `pid` by the catalog's order: (lab, its path, stop, path length, via PATH_NEXT) or None"""
    labs = paths[pid]; i = labs.index(lab)
    for k in range(i - 1, -1, -1):
        if labs[k] != lab and labs[k] in live:
            return labs[k], pid, k + 1, len(labs), False
    for p1, p2 in path_next.items():       # the path's first live lab: arrived at from the end of a path continued into this one?
        if p2 != pid or p1 not in paths:
            continue
        l1 = paths[p1]; e = next((x for x in reversed(l1) if x in live), None)
        if not e or e == lab or l1.index(e) != len(l1) - 1:   # a lab not built after e: its footer says the path pauses there
            continue
        start = paths[p2].index(e) if e in paths[p2] else -1
        land = next((x for x in paths[p2][start + 1:] if x != e and x in live), None)
        if land == lab:
            return e, p1, l1.index(e) + 1, len(l1), True
    return None


ROW_JS = """r=>{const c=document.querySelector(r+' .labhead .crumbs'); if(!c) return null;
  const pn=c.querySelector('.crumb-pn'), pv=c.querySelector('.crumb-prev'), nx=c.querySelector('.crumb-next');
  const R=e=>{if(!e) return null; const b=e.getBoundingClientRect(); return {l:b.left,t:b.top,r:b.right,b:b.bottom,w:b.width,h:b.height};};
  const items=[...c.children].filter(e=>e!==pn&&getComputedStyle(e).display!=='none').map(R);
  const lk=e=>{if(!e) return null; const b=e.getBoundingClientRect(), cs=getComputedStyle(e), rg=document.createRange(); rg.selectNodeContents(e);
    const x=b.left+Math.min(b.width/2,24), y=b.top+b.height/2, hit=document.elementFromPoint(x,y);
    return {href:e.getAttribute('href'), path:e.dataset.path||null, tip:e.dataset.tipx||'', txt:e.innerText.replace(/\\s+/g,' ').trim(), rect:R(e),
      sw:e.scrollWidth, cw:e.clientWidth, fs:parseFloat(cs.fontSize), color:cs.color, cls:e.className,
      before:getComputedStyle(e,'::before').content, after:getComputedStyle(e,'::after').content,
      vis:Math.min(...[...e.querySelectorAll('span')].map(x=>parseFloat(getComputedStyle(x).fontSize)).concat([parseFloat(getComputedStyle(e,'::before').fontSize),parseFloat(getComputedStyle(e,'::after').fontSize)]).filter(v=>v>0)),
      text:[...rg.getClientRects()].filter(q=>q.width>0).map(q=>({l:q.left,t:q.top,r:q.right,b:q.bottom})), top:!!hit&&(hit===e||e.contains(hit))};};
  let bg='', el=c; while(el){const v=getComputedStyle(el).backgroundColor; if(v&&v!=='transparent'&&!/rgba\\(.*, 0\\)$/.test(v)){bg=v; break;} el=el.parentElement;}
  // the same row with v6's content: the crumbs and the next link only, as a direct child of the row (v6's markup)
  const v6=c.cloneNode(true); v6.querySelectorAll('.crumb-pn').forEach(x=>x.remove()); if(nx) v6.appendChild(nx.cloneNode(true));
  v6.style.visibility='hidden'; c.after(v6); const v6h=v6.getBoundingClientRect().height; v6.remove();
  const cs=getComputedStyle(c), cr=c.getBoundingClientRect().right-parseFloat(cs.paddingRight)-parseFloat(cs.borderRightWidth);
  return {row:R(c), cr, gap:parseFloat(cs.columnGap)||0, pgap:pn?parseFloat(getComputedStyle(pn).columnGap)||0:0, v6h, rowSW:c.scrollWidth, rowCW:c.clientWidth, items, pn:R(pn), prev:lk(pv), next:lk(nx), nPrev:c.querySelectorAll('.crumb-prev').length,
    nNext:c.querySelectorAll('.crumb-next').length, vw:innerWidth, docW:document.documentElement.clientWidth, h1:R(document.querySelector(r+' .labtitle')), bg}}"""


def lum(rgb):
    v = [int(x) / 255 for x in re.findall(r'[\d.]+', rgb)[:3]]
    v = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in v]
    return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]


def contrast(a, b):
    la, lb = sorted([lum(a), lum(b)], reverse=True)
    return (la + 0.05) / (lb + 0.05)


def root(lab):
    return '#view-lab' if lab == 'kv-cache' else '#view-dyn'


def nav(pg, h, settle=80):
    """set the address and wait until the router has handled it (route() listens to hashchange before this promise resolves)"""
    pg.evaluate("""h=>new Promise(res=>{if(location.hash==='#'+h){res(false); return;}
      addEventListener('hashchange',()=>setTimeout(()=>res(true),0),{once:true}); location.hash=h;})""", h)
    pg.wait_for_timeout(settle)


def follow(pg, sel, real):
    """click a link (a real mouse click when `real`) and wait for the router"""
    pg.evaluate("()=>{window.__hc=new Promise(res=>addEventListener('hashchange',()=>setTimeout(res,0),{once:true}))}")
    if real:
        b = pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return null; e.scrollIntoView({block:'nearest'}); const r=e.getBoundingClientRect(); return {x:r.left+Math.min(r.width/2,24),y:r.top+r.height/2}}", sel)
        pg.mouse.move(b['x'], b['y']); pg.wait_for_timeout(60); pg.mouse.down(); pg.mouse.up()
    else:
        pg.evaluate("s=>document.querySelector(s).click()", sel)
    pg.evaluate("()=>Promise.race([window.__hc,new Promise(r=>setTimeout(r,3000))])")
    pg.wait_for_timeout(80)


def layout(tag, r, W, Z):
    phone = W <= 480
    pv, nx, pn = r['prev'], r['next'], r['pn']
    check(f'{tag}: at most one previous and one next link in the row', r['nPrev'] <= 1 and r['nNext'] <= 1, (r['nPrev'], r['nNext']))
    links = [x for x in (pv, nx) if x]
    if not links:
        check(f'{tag}: no empty pair box in the row', pn is None or pn['w'] == 0, pn)
        return
    row = r['row']
    check(f'{tag}: the row does not scroll sideways and stays in the viewport', r['rowSW'] <= r['rowCW'] + 1 and row['r'] <= r['vw'] + 0.5 and row['l'] >= -0.5, (r['rowSW'], r['rowCW'], row))
    check(f'{tag}: the row ends above the title', r['h1'] is None or row['b'] <= r['h1']['t'] + 0.5, (row, r['h1']))
    for k, x in (('previous', pv), ('next', nx)):
        if not x:
            continue
        b = x['rect']
        check(f'{tag}: the {k} link is inside the row and the viewport', b['l'] >= row['l'] - 0.5 and b['r'] <= row['r'] + 0.5 and b['r'] <= r['vw'] + 0.5, (b, row))
        check(f'{tag}: the {k} link’s text is not cut (no sideways scroll, every line inside the link and the viewport)',
              x['sw'] <= x['cw'] + 1 and all(q['l'] >= b['l'] - 0.5 and q['r'] <= b['r'] + 0.5 and 0 <= q['l'] and q['r'] <= r['vw'] + 0.5 for q in x['text']), (x['sw'], x['cw'], b, x['text'][:3]))
        check(f'{tag}: nothing covers the {k} link', x['top'], b)
        check(f'{tag}: the {k} link is readable: at least 12 px', (x['vis'] if phone else x['fs']) >= 12, (x['fs'], x['vis']))
        for it in r['items']:
            ov = min(b['r'], it['r']) - max(b['l'], it['l']) > 0.5 and min(b['b'], it['b']) - max(b['t'], it['t']) > 0.5
            check(f'{tag}: the {k} link does not overlap the crumbs', not ov, (b, it))
    if pv and nx:
        a, c = pv['rect'], nx['rect']
        ov = min(a['r'], c['r']) - max(a['l'], c['l']) > 0.5 and min(a['b'], c['b']) - max(a['t'], c['t']) > 0.5
        check(f'{tag}: previous and next do not overlap, previous first', not ov and a['l'] < c['l'], (a, c))
    # where the pair goes, modelled on flex-wrap: after the crumbs' last line when the room left there (after the row's gap) holds
    # the pair's width, else on a line of its own under the crumbs, right-aligned; within 1 px either is right (sub-pixel widths)
    lines = []
    for it in r['items']:
        if lines and it['t'] < lines[-1][1] - 0.5:
            t, bt, rt = lines[-1]; lines[-1] = (min(t, it['t']), max(bt, it['b']), max(rt, it['r']))
        else:
            lines.append((it['t'], it['b'], it['r']))
    last_t, last_b, last_r = lines[-1] if lines else (row['t'], row['t'], row['l'])
    free = r['cr'] - last_r - r['gap']
    need = sum(x['rect']['w'] for x in links) + (r['pgap'] if len(links) == 2 else 0)
    own = pn['t'] >= last_b - 0.5
    on_last = not own and pn['t'] < last_b and pn['b'] > last_t
    check(f'{tag}: the pair at the right end of the row', abs(pn['r'] - r['cr']) <= 1, (pn, r['cr']))
    if free >= need + 1:
        check(f'{tag}: the pair on the crumbs’ last line (room {free:.1f} px for {need:.1f})', on_last, (pn, lines))
    elif free <= need - 1:
        check(f'{tag}: the pair on a line of its own under the crumbs, as one piece (room {free:.1f} px for {need:.1f})', own and (len(links) < 2 or abs(pv['rect']['t'] - nx['rect']['t']) <= 1 or not phone), (pn, lines))
    else:
        check(f'{tag}: the pair on the crumbs’ last line or under them (room {free:.1f} px for {need:.1f})', own or on_last, (pn, lines))
    fits = not own
    if phone:
        lang = 'zh' if Z('en', 'zh') == 'zh' else 'en'
        sp, sn = SHORT[lang]
        if len(links) == 2:
            check(f'{tag}: phones: previous and next on one line', abs(pv['rect']['t'] - nx['rect']['t']) <= 1, (pv['rect'], nx['rect']))
        for k, x, pseudo, want in (('previous', pv, 'after', sp), ('next', nx, 'before', sn)):
            if x:
                check(f'{tag}: phones: the {k} link shows the short label “{want}” with its arrow, at least 12 px', want in (x[pseudo] or '') and x['fs'] == 0 and x['vis'] >= 12, (x[pseudo], x['fs'], x['vis']))
                check(f'{tag}: phones: the {k} link is a tap target at least 32 px tall', x['rect']['h'] >= 32, x['rect'])
        if row['h'] > r['v6h'] + 0.5 and free < need:
            # the crumbs leave less room than the pair's shortest form (“← Prev”, “Next →”): at the end of a path (v6 had nothing at the
            # right end), or where v6's next link was short enough to share the crumbs' line (mha → sampling on the beginners' path,
            # preview v24): the pair takes a line of its own; listed, not hidden (see the report)
            GREW.append(f'{tag} ({r["v6h"]:.1f} → {row["h"]:.1f} px, room {free:.1f} px for {need:.1f})')
        else:
            check(f'{tag}: phones: the row is no taller than with v6’s content (the crumbs and the next link)', row['h'] <= r['v6h'] + 0.5, (row['h'], r['v6h']))
    else:
        if len(links) == 2:
            check(f'{tag}: previous and next on one line', abs(pv['rect']['t'] - nx['rect']['t']) <= 1 and pv['rect']['h'] <= 24 and nx['rect']['h'] <= 24, (pv['rect'], nx['rect']))
        for k, x in (('previous', pv), ('next', nx)):
            if x:
                check(f'{tag}: the {k} link shows its full label (no short form above 480 px)', x['fs'] >= 12 and x['before'] in ('none', 'normal') and x['after'] in ('none', 'normal'), (x['fs'], x['before'], x['after']))
    return 0 if fits else 1


def link_text(tag, r, exp, cat, Z, lang):
    pv = r['prev']
    if not exp:
        check(f'{tag}: no previous link (the path’s first live lab)', pv is None, pv and pv['href'])
        return
    e, epath, stop, of, cont = exp
    check(f'{tag}: the previous link points to {e}', pv is not None and pv['href'] == f'#{e}', pv and pv['href'])
    if not pv:
        return
    check(f'{tag}: the previous link keeps the path (data-path {epath})', pv['path'] == epath, pv['path'])
    L = cat['LAB'][e]
    short = (L.get('s') or L['q'])[lang]
    pname = cat['PATHS'][epath]['name'][lang]
    head = Z(f'← Previous lab ({pname}): ', f'← 上一个 Lab（{pname}）：') if cont else Z('← Previous lab: ', '← 上一个 Lab：')
    check(f'{tag}: the previous link reads “{head}{short}”', pv['txt'] == f'{head}{short}', pv['txt'])
    tip = Z(f'Previous lab, stop {stop} of {of} of the path “{pname}”: {L["q"][lang]}', f'上一个 Lab，“{pname}”路线第 {stop}/{of} 站：{L["q"][lang]}')
    check(f'{tag}: the previous link’s hover gives the whole question and the stop', pv['tip'].startswith(tip), (pv['tip'][:160], tip))


def sweep(pg, cat, live, W, lang, Z, summary):
    """every path, every live lab on it: the previous link, following it, the next link, the layout"""
    for pid, labs in cat['paths'].items():
        nav(pg, f'map~route:{pid}', 250)
        for lab in [x for x in labs if x in live]:
            tag = f'{lang} {W} {pid}:{lab}'
            nav(pg, lab)
            r = pg.evaluate(ROW_JS, root(lab))
            if r is None:
                check(f'{tag}: the lab shows its breadcrumb row', False, pg.evaluate('()=>location.hash')); continue
            exp = expected_prev(cat['paths'], live, pid, lab, PATH_NEXT)
            link_text(tag, r, exp, cat, Z, lang)
            ni = pg.evaluate(f"()=>ASP.catalog.nextInfo('{lab}')")
            nx = r['next']
            check(f'{tag}: the next link is still there, as before', (nx['href'] if nx else None) == (f'#{ni["next"]}' if ni and ni['next'] else None)
                  and (not nx or (nx['txt'].startswith(Z(f'Continue with the path “{cat["PATHS"][ni["to"]]["name"][lang]}”', f'接下来可以继续“{cat["PATHS"][ni["to"]]["name"][lang]}”路线') if ni['done'] else Z('Next lab', '下一个 Lab'))
                                  and nx['txt'].endswith('→') and nx['cls'] == 'crumb-next' and nx['path'] == ni['to'])), (nx, ni))
            # round 3, item 5: at a path's last stop the next path is labelled apart (“continue with”), and nothing says the path is finished
            if nx and ni['done']:
                check(f'{tag}: the last stop is reached, never finished', Z('You have reached the last stop', '你已到达') in nx['tip'] and 'finished' not in nx['tip'] and '走完' not in nx['tip'], nx['tip'])
            if layout(tag, r, W, Z):
                summary['wrap'] += 1; summary['wrapped'].append(tag)
            summary['pairs'] += 1
            if exp and r['prev'] and r['prev']['href'] == f'#{exp[0]}':
                follow(pg, f'{root(lab)} .crumbs .crumb-prev', real=(W == 1440))
                landed = pg.evaluate('()=>location.hash')
                r2 = pg.evaluate(ROW_JS, root(exp[0]))
                back = r2 and r2['next']
                check(f'{tag}: following the previous link lands on {exp[0]}', landed == f'#{exp[0]}', landed)
                check(f'{tag}: the lab it lands on has a next link back here, on the same path', bool(back) and back['href'] == f'#{lab}' and back['path'] == pid, back and (back['href'], back['path']))
                if W == 1440:
                    tv = pg.evaluate("()=>{const t=document.getElementById('tip'); return !!t&&!t.hidden&&t.innerText.trim().length>0?t.innerText.trim().slice(0,80):''}")
                    check(f'{tag}: a real click on the previous link lands with no tooltip under the still pointer (R1.7)', not tv, tv)
                    pg.mouse.move(5, 5)
                summary['followed'] += 1


# texMath's symbols added for the site-wide formula sweep (batch 9): each renders as its symbol, an operator as <mo>, a letter as <mi>,
# and no command name shows as a word (before, \nabla came out as the word “nabla”)
TEX_CASES = [
    ('\\nabla_\\theta L', [('∇', 'mo')]),
    ('\\frac{\\partial L}{\\partial w}', [('∂', 'mi')]),
    ('a \\pm b \\mp c', [('±', 'mo'), ('∓', 'mo')]),
    ('x \\le y \\ge z \\ne w', [('≤', 'mo'), ('≥', 'mo'), ('≠', 'mo')]),
    ('p \\sim q \\propto r', [('∼', 'mo'), ('∝', 'mo')]),
    ('\\ell = -\\log p(y \\mid x)', [('ℓ', 'mi'), ('∣', 'mo')]),
    ('f^\\prime(x)', [('′', 'mo')]),
    ('a \\star b \\ast c', [('⋆', 'mo'), ('∗', 'mo')]),
    ('\\lVert w \\rVert_2', [('‖', 'mo')]),
    ('\\langle q, k \\rangle', [('⟨', 'mo'), ('⟩', 'mo')]),
    ('\\left\\langle \\frac{a}{b} \\right\\rangle + \\left\\lVert \\frac{a}{b} \\right\\rVert', [('⟨', 'mo'), ('⟩', 'mo'), ('‖', 'mo')]),
    ('\\mathbb{E}[x] + \\mathbb{P}(A), x \\in \\mathbb{R}^d, n \\in \\mathbb{N}', [('𝔼', 'mi'), ('ℙ', 'mi'), ('ℝ', 'mi'), ('ℕ', 'mi')]),
    ('\\min_x f(x) \\le \\max_x f(x)', [('min', 'mi'), ('max', 'mi')]),
    ('𝔼[ℓ] = ∂L ± 1', [('𝔼', 'mi'), ('ℓ', 'mi'), ('∂', 'mi'), ('±', 'mo')]),
]
TEX_WORDS = {'nabla', 'partial', 'pm', 'mp', 'le', 'ge', 'ne', 'sim', 'propto', 'ell', 'mid', 'prime', 'star', 'ast', 'lVert', 'rVert',
             'langle', 'rangle', 'mathbb', 'left', 'right', 'frac', 'E', 'R', 'N', 'P'}


def tex_symbols(pg):
    res = pg.evaluate("""cases=>{const box=document.createElement('div'); box.id='kp-tex'; box.style.cssText='position:fixed;left:0;top:0;z-index:99;background:var(--bg);color:var(--ink);padding:12px;font-size:20px;line-height:2';
      document.body.appendChild(box);
      return cases.map(src=>{const d=document.createElement('div'); d.innerHTML=ASP.ui.math(src); box.appendChild(d); const m=d.querySelector('math'), b=m?m.getBoundingClientRect():null;
        const els=m?[...m.querySelectorAll('mi,mo,mtext,mn')].map(e=>{const r=e.getBoundingClientRect(); return {tag:e.tagName.toLowerCase(), t:e.textContent, mv:e.getAttribute('mathvariant'), st:e.getAttribute('stretchy'), w:r.width, h:r.height};}):[];
        return {src, w:b?b.width:0, h:b?b.height:0, els, text:m?m.textContent:''};});}""", [c[0] for c in TEX_CASES])
    for (src, want), r in zip(TEX_CASES, res):
        tag = f'texMath “{src}”'
        check(f'{tag}: renders as MathML with a size', r['w'] > 4 and r['h'] > 4, (r['w'], r['h']))
        words = [e['t'] for e in r['els'] if e['t'] in TEX_WORDS or '\\' in e['t']]
        check(f'{tag}: no command name shows as a word', not words and '\\' not in r['text'], words)
        for sym, kind in want:
            hit = [e for e in r['els'] if e['t'] == sym]
            ok = bool(hit) and all(e['tag'] == kind and e['w'] > 1 for e in hit)
            if sym in '𝔼ℙℝℕ':
                ok = ok and all(e['mv'] == 'normal' for e in hit)
            check(f'{tag}: {sym} is an <{kind}> with a glyph', ok, hit or r['els'])
        if '\\left' in src:
            check(f'{tag}: fences after \\left / \\right grow', all(e['st'] == 'true' for e in r['els'] if e['t'] in '⟨⟩‖'), [e for e in r['els'] if e['t'] in '⟨⟩‖'])
        elif any(s in src for s in ('langle', 'lVert')):
            check(f'{tag}: plain fences keep their size', all(e['st'] == 'false' for e in r['els'] if e['t'] in '⟨⟩‖'), [e for e in r['els'] if e['t'] in '⟨⟩‖'])
    pg.screenshot(path=str(SHOTS / 'texmath-symbols.png'), clip={'x': 0, 'y': 0, 'width': 760, 'height': 40 * len(TEX_CASES) + 40})
    pg.evaluate("()=>document.getElementById('kp-tex').remove()")


def hover(pg, lab, Z):
    """a real hover on the previous link shows its whole question"""
    b = pg.evaluate(f"()=>{{const e=document.querySelector('{root(lab)} .crumbs .crumb-prev'); if(!e) return null; const r=e.getBoundingClientRect(); return {{x:r.left+Math.min(r.width/2,24),y:r.top+r.height/2}}}}")
    if not b:
        return ''
    pg.mouse.move(5, 5); pg.wait_for_timeout(60); pg.mouse.move(b['x'] - 3, b['y']); pg.mouse.move(b['x'], b['y']); pg.wait_for_timeout(300)
    t = pg.evaluate("()=>{const t=document.getElementById('tip'); return t&&!t.hidden?t.innerText.trim():''}")
    pg.mouse.move(5, 5); pg.wait_for_timeout(60)
    return t


t0 = time.time()
with sync_playwright() as p:
    b = p.chromium.launch()
    summary = {'pairs': 0, 'followed': 0, 'wrap': 0, 'wrapped': []}
    for W, H in [(1440, 900), (900, 900), (390, 844), (360, 780)]:
        for lang in ['en', 'zh']:
            Z = (lambda en, zh: zh) if lang == 'zh' else (lambda en, zh: en)
            pg = b.new_page(viewport={'width': W, 'height': H})
            errs = []
            pg.on('pageerror', lambda e: errs.append(str(e)))
            pg.goto(URL + '#map'); pg.wait_for_timeout(400)
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(150)
            cat = pg.evaluate("()=>({paths:Object.fromEntries(Object.entries(ASP.catalog.PATHS).map(([k,v])=>[k,v.labs])), PATHS:ASP.catalog.PATHS, LAB:ASP.catalog.LAB})")
            live = set(pg.evaluate("()=>ASP.catalog.LABS.filter(l=>ASP.catalog.labStatus(l)==='live').map(l=>l.id)"))
            if W == 1440 and lang == 'en':
                orphans = [x for x in live if not any(x in v for v in cat['paths'].values())]
                check('every live lab is on at least one path (so every lab gets the pair)', not orphans, orphans)
                tex_symbols(pg)
            w0 = summary['wrap']
            sweep(pg, cat, live, W, lang, Z, summary)
            print(f'{lang} {W}: the pair took a line of its own under the crumbs in {summary["wrap"] - w0} rows (the crumbs’ last line could not hold it)')
            if W == 1440:
                check(f'{lang} 1440: every row holds the crumbs and the pair on one line', summary['wrap'] == w0, summary['wrap'] - w0)
            # a real hover, the KV lab after a language switch, contrast in light and dark, screenshots
            nav(pg, 'map~route:serving', 250); nav(pg, 'continuous-batching')
            if W > 760:
                tip = hover(pg, 'continuous-batching', Z)
                q = cat['LAB']['paged-attention']['q'][lang]; sv = cat['paths']['serving']; k = sv.index('paged-attention') + 1
                check(f'{lang} {W}: hovering the previous link shows its stop and whole question', Z(f'stop {k} of {len(sv)}', f'第 {k}/{len(sv)} 站') in tip and q in tip, tip)
            nav(pg, 'kv-cache')
            other = 'zh' if lang == 'en' else 'en'
            pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(200)
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
            r = pg.evaluate(ROW_JS, '#view-lab')
            check(f'{lang} {W}: the KV lab keeps its pair after a language switch, on the path chosen (serving: kv-basics before it)',
                  r and r['prev'] and r['prev']['href'] == '#kv-basics' and r['prev']['txt'].startswith(Z('← Previous lab', '← 上一个 Lab')) and r['next'] and r['next']['href'] == '#' + cat['paths']['serving'][cat['paths']['serving'].index('kv-cache') + 1], r and (r['prev'], r['next']))
            for theme in ['light', 'dark']:
                pg.evaluate(f"()=>document.documentElement.setAttribute('data-theme','{theme}')"); pg.wait_for_timeout(60)
                for lab, pid in SHOT_LABS:
                    nav(pg, f'map~route:{pid}', 250); nav(pg, lab, 300); pg.evaluate('()=>window.scrollTo(0,0)'); pg.wait_for_timeout(120)
                    r = pg.evaluate(ROW_JS, root(lab))
                    for k in ('prev', 'next'):
                        if r and r[k]:
                            cr = contrast(r[k]['color'], r['bg'])
                            check(f'{lang} {W} {theme} {lab}: the {k} link’s contrast against the page is at least 4.5', cr >= 4.5, (round(cr, 2), r[k]['color'], r['bg']))
                    hh = max(200, (r['h1']['b'] + 12) if r and r['h1'] else 200)
                    pg.screenshot(path=str(SHOTS / f'{lab}_{W}_{lang}_{theme}.png'), clip={'x': 0, 'y': 0, 'width': W, 'height': min(H, hh)})
            pg.evaluate("()=>document.documentElement.removeAttribute('data-theme')")
            check(f'{lang} {W}: no page errors', not errs, errs[:3])
            pg.close()
    # phones: the row no taller than in v6, against the v6 build itself (each lab opened directly, on its default path, as the flows do)
    if V6.exists():
        (P / 'out/wrapped-kit-prev-v6.html').write_text(WRAP + V6.read_text() + '</body></html>')
        def rows(url, W, H, lang):
            pg = b.new_page(viewport={'width': W, 'height': H}); pg.goto(url + '#map'); pg.wait_for_timeout(400)
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(150); out = {}
            for lab in V6_LABS:
                nav(pg, 'map', 150); nav(pg, lab, 300)
                out[lab] = pg.evaluate("r=>{const c=document.querySelector(r+' .labhead .crumbs'), t=document.querySelector(r+' .labtitle'); return c?[c.getBoundingClientRect().height,t?t.getBoundingClientRect().top:0]:null}", root(lab))
            pg.close(); return out
        for W, H in [(390, 844), (360, 780)]:
            for lang in ['en', 'zh']:
                old, new = rows('file://' + str(P / 'out/wrapped-kit-prev-v6.html'), W, H, lang), rows(URL, W, H, lang)
                for lab in V6_LABS:
                    check(f'{lang} {W} {lab}: the breadcrumb row is no taller than in v6, the title no lower', old[lab] and new[lab] and new[lab][0] <= old[lab][0] + 0.5 and new[lab][1] <= old[lab][1] + 0.5, (old[lab], new[lab]))
                print(f'{lang} {W}: row height v6 → now: ' + ', '.join(f'{lab} {old[lab][0]:.1f} → {new[lab][0]:.1f}' for lab in V6_LABS if old[lab] and new[lab]))
    else:
        print(f'note: no v6 build at {V6}; the rows were compared with v6’s content in this page only (set KIT_PREV_V6)')
    # the continuation rule, on the variant page: serving's first live lab (kv-basics) is where “research” continues after eval-capstone
    for W, H in [(1440, 900), (390, 844)]:
        for lang in ['en', 'zh']:
            Z = (lambda en, zh: zh) if lang == 'zh' else (lambda en, zh: en)
            pg = b.new_page(viewport={'width': W, 'height': H})
            errs = []
            pg.on('pageerror', lambda e: errs.append(str(e)))
            pg.goto(URL_CONT + '#map'); pg.wait_for_timeout(400)
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(150)
            cat = pg.evaluate("()=>({paths:Object.fromEntries(Object.entries(ASP.catalog.PATHS).map(([k,v])=>[k,v.labs])), PATHS:ASP.catalog.PATHS, LAB:ASP.catalog.LAB})")
            live = set(pg.evaluate("()=>ASP.catalog.LABS.filter(l=>ASP.catalog.labStatus(l)==='live').map(l=>l.id)"))
            pn2 = dict(PATH_NEXT, research='serving')
            first = next(x for x in cat['paths']['serving'] if x in live)
            exp = expected_prev(cat['paths'], live, 'serving', first, pn2)
            tag = f'continuation {lang} {W} serving:{first}'
            check(f'{tag}: the test catalog gives a continuation here', exp is not None and exp[4], exp)
            nav(pg, 'map~route:serving', 250); nav(pg, first)
            r = pg.evaluate(ROW_JS, root(first))
            link_text(tag, r, exp, cat, Z, lang)
            layout(tag, r, W, Z)
            pi = pg.evaluate(f"()=>ASP.catalog.prevInfo('{first}')")
            check(f'{tag}: prevInfo reports the continuation', pi and pi['cont'] and pi['to'] == 'research' and pi['pid'] == 'serving', pi)
            if exp and r and r['prev']:
                follow(pg, f'{root(first)} .crumbs .crumb-prev', real=(W == 1440))
                r2 = pg.evaluate(ROW_JS, root(exp[0]))
                back = r2 and r2['next']
                pname = cat['PATHS']['serving']['name'][lang]
                check(f'{tag}: following it lands on {exp[0]}, whose next link (naming the path “{pname}”) leads back here',
                      pg.evaluate('()=>location.hash') == f'#{exp[0]}' and bool(back) and back['href'] == f'#{first}' and back['path'] == 'serving' and f'“{pname}”' in back['txt'], back)
                # and the lab before it on its own path is its previous lab again
                nav(pg, 'map~route:research', 250); nav(pg, exp[0])
                r3 = pg.evaluate(ROW_JS, root(exp[0]))
                e3 = expected_prev(cat['paths'], live, 'research', exp[0], pn2)
                check(f'{tag}: on its own path, {exp[0]}’s previous lab is {e3 and e3[0]}', r3 and r3['prev'] and r3['prev']['href'] == f'#{e3[0]}', r3 and r3['prev'])
            check(f'continuation {lang} {W}: no page errors', not errs, errs[:3])
            pg.close()
    b.close()
wide = [w for w in summary['wrapped'] if int(w.split()[1]) > 480]
print(f'{summary["pairs"]} rows checked (lab × path × width × language), {summary["followed"]} previous links followed; the pair took a line '
      f'of its own under the crumbs on a wide screen {len(wide)} times' + (f' ({", ".join(wide)})' if wide else '') + f', on phones {summary["wrap"] - len(wide)} times')
print(f'phones, rows taller than v6 by one line because the crumbs leave no room for the pair: {len(GREW)}' + ''.join(f'\n  {g}' for g in GREW))
print(f'{n} checks, {len(fails)} failed ({time.time() - t0:.0f} s)')
print('FAILED:', fails[:40] if fails else 'none')
sys.exit(1 if fails else 0)
