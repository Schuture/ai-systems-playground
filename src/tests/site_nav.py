"""Site navigation and jump links (review 7, round 10). For every lab that can be opened, in EN and ZH at 1440 x 900:
- the breadcrumb's domain is a link to the map with that domain's labs in view;
- the header names the next lab (Codex review 1 #2), and the rail shows it below the card in Scenarios, in Challenge, and on the last
  guided part. Round 3 (the owner's items 4 and 5): a lab follows a path only when the learner chose it. The labs of ROUTE are opened
  with their path chosen on the map (#map~route:<path>): the next lab of the path with its stop number; at a path's last stop the footer
  says it is reached (never finished), names the next path as “continue with”, or links to the path on the map. Every other lab is
  opened from its branch (#map~<branch>): the next lab of the branch (lab n of m, no stop number, no path, no link that carries a
  path); at the branch's last lab a suggested next step with a button that joins its path, or a link back to the tree;
- every element with data-jump answers hover (a tooltip with text) and click (something on the page changes);
- terms marked .def show their definition on hover.
New labs are picked up from the catalog, so the Transformer series is covered once it is registered."""
import pathlib, sys, hashlib
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
import os
# NAV_PAGE=out/<id>.html lets a builder run this on a build that has only their lab: python3 build.py out/<id>.html --only <id>
PAGE = os.environ.get('NAV_PAGE', 'ai-systems-playground.html')
src = (P / PAGE).read_text()
TAG = pathlib.Path(PAGE).stem
(P / 'out').mkdir(exist_ok=True)
(P / f'out/wrapped-nav-{TAG}.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + src + '</body></html>')
URL = 'file://' + str(P / f'out/wrapped-nav-{TAG}.html')
ORDER = ['architecture', 'transformer', 'mha', 'kv-basics', 'gqa-mla', 'kv-cache', 'prefill-decode', 'paged-attention', 'continuous-batching', 'chunked-prefill', 'prefix-caching', 'dpo']
ONLY = sys.argv[1:]           # optional: lab ids to test
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(f'{name} :: {info}')
        print('FAIL', name, '::', str(info)[:300])


def roots(lab):
    if lab == 'kv-cache':
        return '#view-lab', '#rail'
    return '#view-dyn', '#view-dyn aside.rail'


# labs opened with a path chosen (its home path); every other lab is opened from its branch, with no path
ROUTE = {**{x: 'transformer' for x in ['architecture', 'transformer', 'mha', 'kv-basics', 'kv-cache']}, 'gqa-mla': 'research',
         **{x: 'serving' for x in ['prefill-decode', 'paged-attention', 'continuous-batching', 'chunked-prefill', 'prefix-caching']}, 'dpo': 'research'}
STOP_RE = __import__('re').compile(r'第 \d+/\d+ 站|stop \d+ of \d+')


def go(pg, lab):
    w = pg.evaluate(f"()=>ASP.catalog.LAB['{lab}'].w")
    pg.evaluate(f"()=>{{location.hash='{'map~route:' + ROUTE[lab] if lab in ROUTE else 'map~' + w}'}}"); pg.wait_for_timeout(250)
    pg.evaluate(f"()=>{{location.hash='{lab}'}}"); pg.wait_for_timeout(700)


def settle(pg, limit=2500):
    """wait until the page stops scrolling (the kit's jumps scroll smoothly)"""
    last, same, t = None, 0, 0
    while t < limit:
        y = pg.evaluate("()=>Math.round(scrollY)")
        same = same + 1 if y == last else 0
        if same >= 3:
            return
        last = y; pg.wait_for_timeout(60); t += 60


def state(pg, root):
    return pg.evaluate("""r=>{const el=document.querySelector(r), tip=document.getElementById('tip');
      return {y:Math.round(scrollY), fl:document.querySelectorAll('.flash,.bump,.linked').length, open:[...document.querySelectorAll('details')].filter(d=>d.open).length,
        html:el?el.innerHTML.length+':'+[...el.querySelectorAll('[aria-pressed="true"]')].map(b=>b.textContent).join('|'):'', tip:tip&&!tip.hidden?tip.innerText.trim():''}}""", root)


def place(pg, sel, i):
    """scroll element i of sel to the middle of the screen, wait for the page to settle, return where it is and whether it is the topmost element there"""
    pg.evaluate("""([s,i])=>{const e=[...document.querySelectorAll(s)].filter(e=>e.offsetParent&&!e.closest('details:not([open])'))[i]; if(e) e.scrollIntoView({block:'center'});}""", [sel, i])
    settle(pg)
    return pg.evaluate("""([s,i])=>{const e=[...document.querySelectorAll(s)].filter(e=>e.offsetParent&&!e.closest('details:not([open])'))[i]; if(!e) return null; const b=e.getClientRects()[0]||e.getBoundingClientRect(), x=b.x+Math.min(b.width/2,40), y=b.y+b.height/2;   // the first line box: a term may wrap
      const top=document.elementFromPoint(x,y); return {k:e.dataset.jump||'', x, y, t:e.innerText.trim().slice(0,40), tab:e.tabIndex, tip:e.dataset.tipx||'', key:e.dataset.tip||'', top:!!top&&(top===e||e.contains(top))}}""", [sel, i])


def hover_tip(pg, info):
    pg.mouse.move(5, 5); pg.wait_for_timeout(60)
    pg.mouse.move(info['x'] - 4, info['y'] - 2); pg.mouse.move(info['x'], info['y']); pg.wait_for_timeout(280)
    return pg.evaluate("()=>{const t=document.getElementById('tip'); return t&&!t.hidden?t.innerText.trim():''}")


def jump_links(pg, root, tag):
    """hover and click every visible [data-jump] under root (rail, stage, controls, connect)"""
    sel = root + ' [data-jump]'
    count = pg.evaluate("s=>[...document.querySelectorAll(s)].filter(e=>e.offsetParent).length", sel)
    for i in range(count):
        info = place(pg, sel, i)
        if not info:
            continue
        check(f'{tag}: jump link “{info["t"]}” ({info["k"]}) is not covered', info['top'], info)
        tip = hover_tip(pg, info)
        check(f'{tag}: hover on jump link “{info["t"]}” ({info["k"]}) shows a tooltip', len(tip) > 3, tip)
        pg.evaluate("()=>document.querySelectorAll('.flash,.bump').forEach(e=>e.classList.remove('flash','bump'))")
        before = state(pg, root)
        pg.mouse.click(info['x'], info['y']); pg.wait_for_timeout(350); settle(pg)
        after = state(pg, root)
        moved = (abs(after['y'] - before['y']) > 2 or after['fl'] > before['fl'] or after['open'] != before['open'] or after['html'] != before['html']
                 or (after['tip'] and after['tip'] != before['tip'])
                 # a link to a setting that is fixed in this state answers with its tip, which the hover already showed (since 2026-10-11 the notes
                 # at the page end, and their links, show in every step)
                 or (after['tip'] and ('Fixed' in after['tip'] or '固定' in after['tip'])))
        check(f'{tag}: click on jump link “{info["t"]}” ({info["k"]}) does something', moved, f'{before} -> {after}')
        pg.mouse.move(5, 5); pg.wait_for_timeout(60)


def defs(pg, root, tag):
    sel = root + ' .def'
    sel = sel  # (visible = rendered and not inside a closed <details>)
    k = pg.evaluate("s=>[...document.querySelectorAll(s)].filter(e=>e.offsetParent&&!e.closest('details:not([open])')).length", sel)
    for i in range(k):
        info = place(pg, sel, i)
        if not info:
            continue
        tip = hover_tip(pg, info)
        mn = 10 if tag.startswith('zh') else 20   # a real definition, not a stub; Chinese says it in about half the characters
        # a definition's text is inline (data-tipx) or a kit string key (data-tip, e.g. one holding a formula, batch 6 v6); either way the shown tip must be real
        check(f'{tag}: definition “{info["t"]}” shows on hover and is keyboard-reachable', len(tip) > mn and info['tab'] == 0 and (len(info['tip']) > mn or bool(info.get('key'))), tip[:80])
        pg.mouse.move(5, 5); pg.wait_for_timeout(60)


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ['en', 'zh']:
        Z = (lambda en, zh: zh) if lang == 'zh' else (lambda en, zh: en)
        other = 'zh' if lang == 'en' else 'en'
        for (W, H) in [(1440, 900), (720, 900), (390, 844)]:
            phone = W <= 760
            pg = b.new_page(viewport={'width': W, 'height': H})
            errs = []
            pg.on('pageerror', lambda e: errs.append(str(e)))
            pg.goto(URL); pg.wait_for_timeout(400)
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
            live = pg.evaluate("()=>ASP.catalog.LABS.filter(l=>ASP.catalog.labStatus(l)==='live').map(l=>l.id)")
            order = [x for x in ORDER if x in live] + [x for x in live if x not in ORDER]   # labs added after ORDER too, as the docstring says
            for i, lab in enumerate(order):
                if ONLY and lab not in ONLY:
                    continue
                tag = f'{lang} {W} {lab}'
                root, rail = roots(lab)
                go(pg, lab)
                if pg.evaluate("!!document.querySelector('#view-dyn .ls')"):   # a lesson page (2026-10-11): no breadcrumb, rail or modes; tests/lesson_architecture.py
                    print('skip', tag, '(lesson page)'); continue
                myworld = pg.evaluate(f"()=>ASP.catalog.LAB['{lab}'].w")
                ni = pg.evaluate(f"()=>ASP.catalog.nextInfo('{lab}')")
                nxt = ni['next'] if ni else None
                onroute = lab in ROUTE
                check(f'{tag}: the lab follows {"its chosen path" if onroute else "its branch"}', ni and ni['kind'] == ('route' if onroute else 'branch')
                      and (ni['pid'] == ROUTE[lab] if onroute else ni['branch'] == myworld), ni)
                # the next lab stays on the chosen path (and on the next path only at this one's last stop), or on the branch
                if nxt and onroute:
                    labs_to = pg.evaluate(f"()=>ASP.catalog.PATHS['{ni['to']}'].labs")
                    check(f'{tag}: the next lab is on the learner’s path', nxt in labs_to and (ni['to'] == ni['pid'] or ni['done']), ni)
                elif nxt:
                    labs_j = pg.evaluate(f"()=>ASP.catalog.PATHS['{ni['join']}'].labs") if ni['rec'] else []
                    check(f'{tag}: the next lab is on the branch, or at its end the next lab of the path it suggests', (not ni['end'] and pg.evaluate(f"()=>ASP.catalog.LAB['{nxt}'].w") == myworld)
                          or (ni['end'] and ni['rec'] and nxt in labs_j and ni['n'] is None), ni)
                if lab == 'continuous-batching':
                    check(f'{tag}: continuous batching does not jump to DPO (Codex review 1 #2)', nxt != 'dpo', ni)
                # the header: domain link and next lab
                crumbs = pg.evaluate("""r=>{const c=document.querySelector(r+' .crumbs'); if(!c) return null; const links=[...c.querySelectorAll('a')].map(a=>({h:a.getAttribute('href'),t:a.innerText.trim()}));
                  const nx=c.querySelector('.crumb-next'); return {links, next:nx?nx.getAttribute('href'):null, nextText:nx?nx.innerText:'', paths:[...c.querySelectorAll('[data-path]')].map(a=>a.dataset.path)}}""", root)
                if phone:   # Batch 1 (Codex review 1 #3): the part bar and the top of the picture on the first screen; Settings is a drawer
                    fs = pg.evaluate("""r=>{const st=document.querySelector(r+' .bench>.stage'), c=document.querySelector(r+' .controls'), h=c?c.querySelector(':scope>.eyebrow'):null;
                      const hidden=c?[...c.children].filter(x=>x!==h).every(x=>x.getBoundingClientRect().height===0):null;
                      return {stageTop:st?Math.round(st.getBoundingClientRect().top):null, drawerClosed:hidden, role:h?h.getAttribute('role'):null}}""", root)
                    check(f'{tag}: the picture starts on the first screen (R7.3)', fs['stageTop'] is not None and fs['stageTop'] < H - 200, fs)
                    if fs['drawerClosed'] is not None:
                        pg.evaluate("r=>document.querySelector(r+' .controls>.eyebrow').click()", root); pg.wait_for_timeout(150)
                        opened = pg.evaluate("r=>{const c=document.querySelector(r+' .controls'); return c.classList.contains('open')&&[...c.children].some(x=>!x.classList.contains('eyebrow')&&x.getBoundingClientRect().height>0)}", root)
                        check(f'{tag}: Settings is a drawer that opens from its heading', fs['drawerClosed'] and fs['role'] == 'button' and opened, fs)
                        pg.evaluate("r=>document.querySelector(r+' .controls>.eyebrow').click()", root); pg.wait_for_timeout(100)
                wl = [l for l in crumbs['links'] if l['h'].startswith('#map~')]
                check(f'{tag}: the breadcrumb domain is a link to the map', len(wl) == 1, crumbs)
                check(f'{tag}: the header names the next lab', crumbs['next'] == (f'#{nxt}' if nxt else None), crumbs)
                if nxt and onroute:
                    pn = pg.evaluate(f"()=>ASP.catalog.PATHS['{ni['to']}'].name[ASP.lang()]")
                    named = f'“{pn}”' in crumbs['nextText']
                    check(f'{tag}: a next lab on the next path names that path', named == bool(ni['done']), crumbs['nextText'])
                elif nxt:
                    check(f'{tag}: on its branch the header names no path and carries none', Z('path', '路线') not in crumbs['nextText'] and not crumbs['paths'], crumbs)
                    # a real click on the header's next lab: the new lab renders under a pointer that has not moved, so no tooltip pops up (R1.7, site review r12 #1)
                    bb = pg.evaluate("r=>{const a=document.querySelector(r+' .crumbs .crumb-next'); if(!a) return null; const b=a.getBoundingClientRect(); return {x:b.x+b.width/2,y:b.y+b.height/2}}", root)
                    if bb and 60 < bb['y'] < H:
                        pg.mouse.move(bb['x'], bb['y']); pg.wait_for_timeout(120)
                        pg.mouse.down(); pg.mouse.up(); pg.wait_for_timeout(1200)
                        tv = pg.evaluate("()=>{const t=document.getElementById('tip'); return !!t&&!t.hidden&&t.innerText.trim().length>0?t.innerText.trim().slice(0,80):''}")
                        landed = pg.evaluate("()=>location.hash")
                        check(f'{tag}: a real click on the header’s next lab lands there with no tooltip under the still pointer', landed.startswith(f'#{nxt}') and not tv, (landed, tv))
                        pg.mouse.move(5, 5); pg.wait_for_timeout(60)
                        go(pg, lab)

                def footer(mode_tag):
                    """the visible footer: in the rail on wide screens, after the bench on phones"""
                    sel = f'{root} .next-lab-end .next-lab' if phone else f'{rail} .next-lab'
                    info = pg.evaluate("""([s,r])=>{const e=document.querySelector(s), rl=document.querySelector(r+' .next-lab'); if(!e) return null; const a=e.querySelector('a');
                      return {vis:getComputedStyle(e.closest('.next-lab-end')||e).display!=='none'&&e.offsetParent!==null, href:a?a.getAttribute('href'):null, txt:e.innerText.replace(/\\s+/g,' '),
                        paths:[...e.querySelectorAll('[data-path]')].map(x=>x.dataset.path), join:e.querySelector('.nl-join')?e.querySelector('.nl-join').dataset.join:null,
                        railHidden:!rl||getComputedStyle(rl).display==='none'}}""", [sel, rail])
                    ok = info is not None and info['vis'] and (info['href'] == f'#{nxt}' if nxt else info['href'] == (f'#map~route:{ni["pid"]}' if onroute else '#map'))
                    if phone and info:
                        ok = ok and info['railHidden']
                    check(f'{tag} {mode_tag}: the next lab is shown ({"after the settings on phones" if phone else "below the rail card"})', ok, info)
                    if info:
                        check(f'{tag} {mode_tag}: the footer never says a path is finished', '走完' not in info['txt'] and 'finished' not in info['txt'] and '✓' not in info['txt'], info['txt'][:200])
                    if nxt and info and onroute:
                        q = pg.evaluate(f"()=>ASP.catalog.LAB['{nxt}'].q[ASP.lang()]")
                        check(f'{tag} {mode_tag}: the footer shows the next lab’s whole question and its stop on the path', q in info['txt'] and (Z('You can continue with the path', '接下来可以继续') in info['txt'] if ni['done'] else Z('Next lab', '下一个 Lab') in info['txt']), info['txt'][:200])
                        check(f'{tag} {mode_tag}: the stop on the path', Z(f'stop {ni["n"]} of {ni["of"]}', f'第 {ni["n"]}/{ni["of"]} 站') in info['txt'], info['txt'][:200])
                        if ni['done']:
                            check(f'{tag} {mode_tag}: at the path’s last stop the footer says it is reached', Z('You have reached the last stop', '你已到达') in info['txt'], info['txt'][:200])
                    elif nxt and info:
                        q = pg.evaluate(f"()=>ASP.catalog.LAB['{nxt}'].q[ASP.lang()]")
                        wn = pg.evaluate(f"()=>ASP.catalog.WORLDS['{myworld}'].name[ASP.lang()]")
                        if ni['end']:
                            check(f'{tag} {mode_tag}: at the branch’s last lab, the suggested next step and a button that joins its path', q in info['txt'] and Z('last lab of the branch', '这一枝') in info['txt']
                                  and Z('Suggested next lab', '推荐下一步') in info['txt'] and info['join'] == ni['join'], info)
                        else:
                            check(f'{tag} {mode_tag}: the footer shows the branch’s next lab', q in info['txt'] and Z(f'Next lab on this branch (“{wn}”, lab {ni["n"]} of {ni["of"]})', f'这一枝的下一个 Lab（“{wn}”第 {ni["n"]}/{ni["of"]} 个）') in info['txt'], info['txt'][:200])
                        check(f'{tag} {mode_tag}: on the branch no stop number and no link that carries a path', not STOP_RE.search(info['txt']) and not info['paths'], info)
                    elif info and onroute:
                        check(f'{tag} {mode_tag}: at the end of its path the footer says it pauses there or the last stop is reached', Z('pauses here', '暂时到此') in info['txt'] or Z('You have reached the last stop', '你已到达') in info['txt'], info['txt'][:200])
                    elif info:
                        check(f'{tag} {mode_tag}: at the branch’s last lab with nothing to suggest, a link back to the tree', Z('last lab of the branch', '这一枝的最后一个 Lab') in info['txt'] and not info['paths'], info['txt'][:200])

                nparts = pg.evaluate("r=>new Set([...document.querySelectorAll(r+' [data-go]')].map(e=>e.dataset.go)).size", rail)   # m2: the parts' bar repeats the dots
                for mode in ['guided', 'sandbox', 'challenge']:
                    pg.click(f'{root} .modes button[data-mode="{mode}"]'); pg.wait_for_timeout(350)
                    if mode == 'guided':
                        pg.evaluate(f"()=>document.querySelector('{rail} [data-go=\"0\"]').click()"); pg.wait_for_timeout(250)
                        endvis = pg.evaluate("r=>{const e=document.querySelector(r+' .next-lab-end'); return !!e&&getComputedStyle(e).display!=='none'&&e.offsetParent!==null}", root)
                        check(f'{tag} guided part 1: no next-lab footer yet (rail{" and after the bench" if phone else ""})', pg.query_selector(f'{rail} .next-lab') is None and not endvis, endvis)
                        pg.evaluate(f"()=>document.querySelector('{rail} [data-go=\"{nparts - 1}\"]').click()"); pg.wait_for_timeout(300)
                    footer(mode)
                    # a language switch keeps it (site review R10 #3)
                    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(300)
                    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(300)
                    footer(mode + ' after a language switch')
                # at the end of a path, its link opens the map on that path, with the path's tiles below it
                if ni and not nxt and onroute:
                    pg.click(f'{root} .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(300)
                    fsel = f'{root} .next-lab-end .next-lab a' if phone else f'{rail} .next-lab a'
                    pg.evaluate(f"()=>document.querySelector('{fsel}').click()"); pg.wait_for_timeout(800)
                    r = pg.evaluate("()=>({hash:location.hash, path:(document.getElementById('pathSel')||{}).value||null, home:!document.getElementById('view-home').hidden, tiles:[...document.querySelectorAll('#tiles .tile')].map(t=>t.dataset.lab)})")
                    check(f'{tag}: the end-of-path link opens the map on that path', r['hash'] == f'#map~route:{ni["pid"]}' and r['home'] and r['path'] == ni['pid'] and lab in r['tiles'], r)
                    go(pg, lab)
                # the mini result bar never covers the title, even with the Fidelity popover open (site review R10 #1, #2)
                pg.click(f'{root} .modes button[data-mode="guided"]'); pg.wait_for_timeout(300)
                pg.evaluate(f"()=>document.querySelector('{rail} [data-go=\"{max(0, nparts - 2)}\"]').click()"); pg.wait_for_timeout(300)
                pg.evaluate("r=>{const b=document.querySelector(r+' .bench button, '+r+' .bench input'); if(b) b.dispatchEvent(new MouseEvent('pointerdown',{bubbles:true}))}", root)
                fb = pg.query_selector(f'{root} .labmeta button[aria-controls$="fid"], {root} #fidBtn')
                if fb:
                    pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(200)
                    fb.click(); pg.wait_for_timeout(250)
                    covered = []
                    for y in range(0, 141, 20):
                        pg.evaluate(f"()=>window.scrollTo(0,{y})"); pg.wait_for_timeout(260)
                        r = pg.evaluate("""r=>{const t=document.querySelector(r+' .labtitle'), c=document.querySelector(r+' .crumbs a'), bar=document.querySelector('.topbar'), top=bar?bar.offsetHeight:58, out=[];
                          for(const e of [t,c]){if(!e) continue; const b=e.getBoundingClientRect(); if(b.bottom<=top+2||b.top>=innerHeight) continue; const x=b.x+Math.min(b.width/2,30), y=Math.max(b.top+4,top+4); if(y>=b.bottom-1) continue; const h=document.elementFromPoint(x,y);
                            if(h&&!(h===e||e.contains(h))) out.push((e.className||e.tagName)+' under '+(h.id||h.className||h.tagName));} return out}""", root)
                        if r:
                            covered.append((y, r))
                    check(f'{tag}: with the Fidelity popover open, nothing covers the breadcrumb or the title while scrolling', not covered, covered[:3])
                    pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(150)
                    if pg.evaluate("b=>b.getAttribute('aria-expanded')", fb) == 'true':
                        fb.click(); pg.wait_for_timeout(150)
                # every jump link and every definition, every mode, every guided part (wide screen only: the same elements)
                if W == 1440:
                    for mode in ['guided', 'sandbox', 'challenge']:
                        pg.click(f'{root} .modes button[data-mode="{mode}"]'); pg.wait_for_timeout(350)
                        parts = range(nparts) if mode == 'guided' else [None]
                        for k in parts:
                            if k is not None:
                                pg.evaluate(f"()=>document.querySelector('{rail} [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(300); settle(pg)   # the part change scrolls smoothly for about 0.7 s
                            t2 = f'{tag} {mode}' + (f' part {k + 1}' if k is not None else '')
                            jump_links(pg, root, t2)
                            defs(pg, root, t2)
                # the breadcrumb's domain opens the map with that domain's tiles in view
                href = wl[0]['h'] if wl else '#map~x'
                world = href.split('~')[1]
                pg.mouse.move(5, 5); pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(900); pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(400)
                hit = pg.evaluate("""s=>{const a=document.querySelector(s), r=a.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2, e=document.elementFromPoint(x,y);
                  return {x, y, ok:!!e&&(e===a||a.contains(e)), on:e?(e.id||e.className||e.tagName):null}}""", f'{root} .crumbs a[href="{href}"]')
                check(f'{tag}: nothing covers the breadcrumb at the top of the page', hit['ok'], hit)
                if not hit['ok']:
                    (P / 'shots/nav').mkdir(parents=True, exist_ok=True); pg.screenshot(path=str(P / f'shots/nav/covered_{lang}_{W}_{lab}.png'))
                pg.mouse.click(hit['x'], hit['y']); pg.wait_for_timeout(700)
                res = pg.evaluate("""w=>{const home=document.getElementById('view-home'), th=document.getElementById('tilesHead'), r=th?th.getBoundingClientRect():null;
                  const pressed=[...document.querySelectorAll('#domStrip .ds-btn[aria-pressed="true"], #domGrid .dg-btn[aria-pressed="true"]')].map(b=>b.dataset.world);
                  const tiles=[...document.querySelectorAll('#tiles .tile')].map(t=>t.dataset.lab);
                  return {hash:location.hash, home:!home.hidden, top:r?Math.round(r.top):null, vh:innerHeight, pressed, tiles}}""", world)
                check(f'{tag}: clicking the domain opens the map with its labs in view', res['hash'] == href and res['home'] and res['top'] is not None and 0 <= res['top'] <= res['vh'] - 150
                      and world in res['pressed'] and lab in res['tiles'], res)
            check(f'{lang} {W}: no page errors', not errs, errs[:3])
            pg.close()
    b.close()
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:40] if fails else 'none')
sys.exit(1 if fails else 0)
