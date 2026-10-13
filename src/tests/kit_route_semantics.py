"""Route semantics (round 3, the owner's items 4 and 5; Codex D02). A lab follows a learning path only when the learner chose it:
- entering tensors from the ML-basics branch's tile: the footer and the breadcrumb follow the branch (“这一枝的下一个 Lab（“ML 基本功”第
  2/7 个）”), with no stop number (第 x/y 站), no path named and no link that carries a path (data-path); the map still shows no path;
  the same for sampling from the branch “How LLMs work” (the owner's second example);
- a branch's last lab (roofline) shows “推荐下一步” with the next lab of its home path and no stop number, and a button that joins that
  path; after a click on it the footer and the breadcrumb show the path's stop numbers, the links carry the path and the map shows it;
- with the beginners' path chosen from its chip, the numbers show (mha: the next lab is the path's stop after mha's);
- a path's last stop says “你已到达…最后一站” and never “走完” / “finished”: eval-ci on the beginners' path (with a link to the three
  job paths on the map, preview v24), eval-capstone on the research path, and eval-capstone entered from the Evaluation branch at guided part 2,
  then Scenarios (the owner's item 5); the count of labs opened on this device shows once PROG has a record;
- with window.__aspResume=true a chosen path survives a reload (the footer's numbers and the map's path menu), and picking a branch
  clears it, also after a reload;
- no lab shows “复制配置链接” any more (the owner's decision of 2026-10-11 19:28 UTC): neither the kit's labs nor the KV lab.
Run: python3 build.py ai-systems-playground.html && python3 tests/kit_route_semantics.py [page.html]"""
import pathlib, re, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
PAGE = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else P / 'ai-systems-playground.html'
src = PAGE.read_text()
WRAP = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>'
(P / 'out').mkdir(exist_ok=True)
(P / 'out/wrapped-route-sem.html').write_text(WRAP + src + '</body></html>')
URL = 'file://' + str(P / 'out/wrapped-route-sem.html')
(P / 'out/wrapped-route-sem-frame.html').write_text('<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0">'
    '<iframe id="f" src="wrapped-route-sem.html" style="width:1440px;height:900px;border:0"></iframe></body></html>')
URL_FRAME = 'file://' + str(P / 'out/wrapped-route-sem-frame.html')
STOP = re.compile(r'第 \d+/\d+ 站|stop \d+ of \d+')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(f'{name} :: {str(info)[:300]}')
        print('FAIL', name, '::', str(info)[:300])


def nav(pg, h, settle=250):
    pg.evaluate("""h=>new Promise(res=>{if(location.hash==='#'+h){res(false); return;}
      addEventListener('hashchange',()=>setTimeout(()=>res(true),0),{once:true}); location.hash=h;})""", h)
    pg.wait_for_timeout(settle)


def root(lab):
    return ('#view-lab', '#rail') if lab == 'kv-cache' else ('#view-dyn', '#view-dyn aside.rail')


def mode(pg, lab, m):
    r, _ = root(lab)
    pg.click(f'{r} .modes button[data-mode="{m}"]'); pg.wait_for_timeout(300)


NAV_JS = """([r,rail])=>{const f=document.querySelector(rail+' .next-lab'), c=document.querySelector(r+' .labhead .crumbs');
  const pn=c&&c.querySelector('.crumb-pn'), nx=c&&c.querySelector('.crumb-next'), pv=c&&c.querySelector('.crumb-prev');
  const lk=a=>a?{href:a.getAttribute('href'),path:a.dataset.path||null,txt:a.innerText.replace(/\\s+/g,' ').trim(),tip:a.dataset.tipx||''}:null;
  return {foot:f?f.innerText.replace(/\\s+/g,' ').trim():null, footPaths:f?[...f.querySelectorAll('[data-path]')].map(a=>a.dataset.path):[],
    footLinks:f?[...f.querySelectorAll('a')].map(lk):[], join:f&&f.querySelector('.nl-join')?{pid:f.querySelector('.nl-join').dataset.join,txt:f.querySelector('.nl-join').innerText.trim()}:null,
    next:lk(nx), prev:lk(pv), crumbPaths:pn?[...pn.querySelectorAll('[data-path]')].map(a=>a.dataset.path):[], route:ASP.catalog.route()}}"""


def navinfo(pg, lab):
    return pg.evaluate(NAV_JS, list(root(lab)))


def home_path(pg):
    nav(pg, 'map', 400)
    return pg.evaluate("()=>{const s=document.getElementById('pathSel'); return s?s.value:null}")


def pick_branch(pg, w):
    """the learner picks a branch on the home page (the branch strip's button), then the page is at the tiles"""
    nav(pg, 'top', 300)
    pg.evaluate("w=>{const b=document.querySelector('.ds-btn[data-world=\"'+w+'\"]')||document.querySelector('.dg-btn[data-world=\"'+w+'\"]'); b.click();}", w)
    pg.wait_for_timeout(300)


def open_tile(pg, lab):
    """a click on the tile's link, as the learner does (the capture listener sees it)"""
    pg.evaluate("id=>document.querySelector('#tile-'+id+' a[href=\"#'+id+'\"]').click()", lab)
    pg.wait_for_timeout(700)


with sync_playwright() as p:
    b = p.chromium.launch()
    cat = None
    for lang in ['zh', 'en']:
        Z = (lambda en, zh: zh) if lang == 'zh' else (lambda en, zh: en)
        ctx = b.new_context(viewport={'width': 1440, 'height': 900})
        pg = ctx.new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(URL); pg.wait_for_timeout(500)
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        cat = pg.evaluate("()=>({paths:Object.fromEntries(Object.entries(ASP.catalog.PATHS).map(([k,v])=>[k,{labs:v.labs,name:v.name[ASP.lang()]}])),"
                          "w:Object.fromEntries(Object.entries(ASP.catalog.WORLDS).map(([k,v])=>[k,v.name[ASP.lang()]])),"
                          "q:Object.fromEntries(ASP.catalog.LABS.map(l=>[l.id,l.q[ASP.lang()]])), lw:Object.fromEntries(ASP.catalog.LABS.map(l=>[l.id,l.w]))})")
        Q, PN, WN = cat['q'], {k: v['name'] for k, v in cat['paths'].items()}, cat['w']

        # 1. tensors from the ML-basics branch's tile: the branch, no path
        pick_branch(pg, 'roots'); open_tile(pg, 'tensors')
        check(f'{lang}: the tile opens tensors', pg.evaluate('()=>location.hash') == '#tensors')
        mode(pg, 'tensors', 'sandbox'); r = navinfo(pg, 'tensors')
        k = Z(f'Next lab on this branch (“{WN["roots"]}”, lab 2 of 7)', f'这一枝的下一个 Lab（“{WN["roots"]}”第 2/7 个）')
        check(f'{lang} tensors from the branch: the footer follows the branch', r['foot'] and k in r['foot'] and Q['backprop'] in r['foot'], r['foot'])
        check(f'{lang} tensors from the branch: no stop number and no path named', r['foot'] and not STOP.search(r['foot']) and Z('path', '路线') not in r['foot'], r['foot'])
        check(f'{lang} tensors from the branch: no link carries a path', not r['footPaths'] and not r['crumbPaths'], (r['footPaths'], r['crumbPaths']))
        check(f'{lang} tensors from the branch: the breadcrumb’s next lab is backprop, its hover names the branch, not a path',
              r['next'] and r['next']['href'] == '#backprop' and not STOP.search(r['next']['tip']) and Z('path', '路线') not in r['next']['tip'] and WN['roots'] in r['next']['tip'], r['next'])
        check(f'{lang} tensors from the branch: no route chosen', r['route'] is None, r['route'])
        ni = pg.evaluate("()=>ASP.catalog.nextInfo('tensors')")
        check(f'{lang} tensors: nextInfo says branch', ni['kind'] == 'branch' and ni['branch'] == 'roots' and ni['next'] == 'backprop' and ni['pid'] is None, ni)
        # following it keeps the branch
        pg.evaluate("()=>document.querySelector('#view-dyn aside.rail .next-lab .nl-go').click()"); pg.wait_for_timeout(700)
        mode(pg, 'backprop', 'sandbox'); r = navinfo(pg, 'backprop')
        check(f'{lang} backprop after the branch link: still the branch, previous lab tensors', r['foot'] and not STOP.search(r['foot']) and r['prev'] and r['prev']['href'] == '#tensors' and r['route'] is None, (r['foot'], r['prev']))
        check(f'{lang} back on the map: no path chosen', home_path(pg) == '', None)

        # the owner's second example: sampling from the branch “How LLMs work”
        pick_branch(pg, 'models'); open_tile(pg, 'sampling'); mode(pg, 'sampling', 'sandbox'); r = navinfo(pg, 'sampling')
        check(f'{lang} sampling from its branch: lab 8 of 11 on the branch, no stop number', r['foot'] and Z(f'(“{WN["models"]}”, lab 8 of 11)', f'（“{WN["models"]}”第 8/11 个）') in r['foot']
              and not STOP.search(r['foot']) and not r['footPaths'], r['foot'])

        # 2. a branch's last lab: the suggested next step and the join button
        nav(pg, 'map~roots', 400); nav(pg, 'roofline', 700); mode(pg, 'roofline', 'sandbox'); r = navinfo(pg, 'roofline')
        rs = cat['paths']['serving']['labs']
        check(f'{lang} roofline: the branch’s last lab', r['foot'] and Z(f'This is the last lab of the branch “{WN["roots"]}”.', f'这是“{WN["roots"]}”这一枝的最后一个 Lab。') in r['foot'], r['foot'])
        check(f'{lang} roofline: the suggested next step, its question, no stop number', r['foot'] and Z('Suggested next lab', '推荐下一步') in r['foot'] and Q['prefill-decode'] in r['foot']
              and not STOP.search(r['foot']) and not r['footPaths'], r['foot'])
        check(f'{lang} roofline: a button joins the Infra path', r['join'] and r['join']['pid'] == 'serving' and PN['serving'] in r['join']['txt'], r['join'])
        check(f'{lang} roofline: the breadcrumb suggests the same lab with no path', r['next'] and r['next']['href'] == '#prefill-decode' and r['next']['path'] is None and not STOP.search(r['next']['tip']), r['next'])
        pg.click('#view-dyn aside.rail .next-lab .nl-join'); pg.wait_for_timeout(300); r = navinfo(pg, 'roofline')
        i = rs.index('roofline')
        check(f'{lang} roofline after joining: the stop numbers show', r['foot'] and Z(f'stop {i + 2} of {len(rs)} on the path “{PN["serving"]}”', f'“{PN["serving"]}”路线第 {i + 2}/{len(rs)} 站') in r['foot']
              and r['footPaths'] == ['serving'] and r['route'] == 'serving', r)
        check(f'{lang} roofline after joining: the breadcrumb pair carries the path', r['crumbPaths'] == ['serving', 'serving'] and r['prev'] and r['prev']['href'] == f'#{rs[i - 1]}', r)
        check(f'{lang} roofline after joining: focus on the new next link', pg.evaluate("()=>document.activeElement&&document.activeElement.matches('.next-lab .nl-go')"), None)
        check(f'{lang} after joining: the map shows the Infra path', home_path(pg) == 'serving', None)
        # picking a branch clears it
        pick_branch(pg, 'eval')
        check(f'{lang} picking a branch clears the path', pg.evaluate('()=>ASP.catalog.route()') is None, None)

        # 5. the owner's item 5: eval-capstone from the Evaluation branch, guided part 2, then Scenarios
        open_tile(pg, 'eval-capstone')
        pg.evaluate("()=>{const b=document.querySelector('#view-dyn aside.rail [data-go=\"1\"]'); if(b) b.click();}"); pg.wait_for_timeout(300)
        mode(pg, 'eval-capstone', 'sandbox'); r = navinfo(pg, 'eval-capstone')
        body = pg.evaluate("()=>document.body.innerText")
        check(f'{lang} eval-capstone from its branch: never “finished”', '走完' not in body and 'finished the path' not in body and r['foot'] and Z('last lab of the branch', '这一枝的最后一个 Lab') in r['foot'], r['foot'])

        # 3. the beginners' path chosen from its chip: the stop numbers
        nav(pg, 'top', 300); pg.evaluate("()=>document.querySelector('.pathchip[data-path=\"transformer\"]').click()"); pg.wait_for_timeout(400)
        check(f'{lang} the chip chooses the path', pg.evaluate('()=>ASP.catalog.route()') == 'transformer', None)
        nav(pg, 'mha', 700); mode(pg, 'mha', 'sandbox'); r = navinfo(pg, 'mha')
        tf = cat['paths']['transformer']['labs']; j = tf.index('mha')
        check(f'{lang} mha on the beginners’ path: the next stop', r['foot'] and Z(f'Next lab (stop {j + 2} of {len(tf)} on the path “{PN["transformer"]}”)', f'下一个 Lab（“{PN["transformer"]}”路线第 {j + 2}/{len(tf)} 站）') in r['foot']
              and r['footPaths'] == ['transformer'], r['foot'])
        check(f'{lang} mha on the beginners’ path: the breadcrumb pair', r['prev'] and r['prev']['href'] == f'#{tf[j - 1]}' and r['prev']['path'] == 'transformer'
              and Z(f'stop {j} of {len(tf)}', f'第 {j}/{len(tf)} 站') in r['prev']['tip'] and r['next'] and r['next']['href'] == f'#{tf[j + 1]}', (r['prev'], r['next']))
        # 4. the path's last stop: reached, never finished; the job paths offered on the map
        last = tf[-1]
        nav(pg, last, 700); mode(pg, last, 'sandbox'); pg.wait_for_timeout(500)
        nav(pg, 'map', 300); nav(pg, last, 700); r = navinfo(pg, last)
        check(f'{lang} {last}: the last stop of the beginners’ path is reached', r['foot'] and Z(f'You have reached the last stop of the path “{PN["transformer"]}”.', f'你已到达“{PN["transformer"]}”路线的最后一站。') in r['foot'], r['foot'])
        check(f'{lang} {last}: never “finished”', r['foot'] and '走完' not in r['foot'] and 'finished' not in r['foot'] and '✓' not in r['foot'], r['foot'])
        check(f'{lang} {last}: a link to the three job paths on the map, no next stop', r['foot'] and Z('Next, pick a job path', '接下来选一条岗位路线') in r['foot']
              and any(a['href'] == '#map' for a in r['footLinks']) and Z('Next lab (', '下一个 Lab（') not in r['foot'] and not r['next'], (r['foot'], r['next']))
        m = re.search(Z(rf'opened (\d+) of this path’s {len(tf)} labs', rf'这条路线的 {len(tf)} 个 Lab，你在这台设备上打开过 (\d+) 个'), r['foot'] or '')
        check(f'{lang} {last}: how many of the path’s labs were opened here (mha and {last} at least)', m and 2 <= int(m.group(1)) <= len(tf), r['foot'])
        nav(pg, 'map~route:research', 400); nav(pg, 'eval-capstone', 700); mode(pg, 'eval-capstone', 'challenge'); r = navinfo(pg, 'eval-capstone')
        body = pg.evaluate("()=>document.body.innerText")
        check(f'{lang} eval-capstone on the research path: the last stop is reached, never finished', r['foot'] and Z(f'last stop of the path “{PN["research"]}”', f'你已到达“{PN["research"]}”路线的最后一站') in r['foot']
              and '走完' not in body and 'finished the path' not in body, r['foot'])

        # 6. the setup link (“复制配置链接”) was removed by the owner's decision of 2026-10-11 19:28 UTC; a lab has no share button
        nav(pg, 'mha', 700)
        check(f'{lang} mha: no share button', not pg.evaluate("()=>!!document.querySelector('#view-dyn [data-share], #view-dyn [data-sharebox]')"), None)
        nav(pg, 'kv-cache', 700)
        check(f'{lang} KV: no share button', not pg.evaluate("()=>{const b=document.getElementById('shareBtn'); return !!(b&&b.getClientRects().length)}"), None)
        check(f'{lang}: no page errors', not errs, errs[:3])
        ctx.close()


    # 5. a chosen path survives a reload (window.__aspResume=true lets PROG keep it in an automated browser)
    ctx = b.new_context(viewport={'width': 1440, 'height': 900}); ctx.add_init_script('window.__aspResume=true')
    pg = ctx.new_page(); pg.goto(URL); pg.wait_for_timeout(500)
    pg.evaluate("()=>localStorage.clear()"); pg.reload(); pg.wait_for_timeout(600)
    pg.click('.langsw button[data-lang="zh"]'); pg.wait_for_timeout(200)
    pg.evaluate("()=>document.querySelector('.pathchip[data-path=\"transformer\"]').click()"); pg.wait_for_timeout(300)
    nav(pg, 'mha', 900); mode(pg, 'mha', 'sandbox'); pg.wait_for_timeout(500)
    pg.reload(); pg.wait_for_timeout(1200)
    mode(pg, 'mha', 'sandbox'); r = navinfo(pg, 'mha')
    tf = pg.evaluate("()=>ASP.catalog.PATHS.transformer.labs"); j = tf.index('mha')
    check('after a reload: the path is still chosen', r['route'] == 'transformer' and r['foot'] and f'路线第 {j + 2}/{len(tf)} 站' in r['foot'], r)
    check('after a reload: the map shows the path', home_path(pg) == 'transformer', None)
    pick_branch(pg, 'roots'); pg.reload(); pg.wait_for_timeout(1000)
    check('after picking a branch and a reload: no path', pg.evaluate('()=>ASP.catalog.route()') is None and home_path(pg) == '', None)
    ctx.close()
    b.close()

print(f'{n - len(fails)}/{n} checks passed')
if fails:
    print('\n'.join(fails[:30])); sys.exit(1)
