"""Home page, round 3 (owner #1, #2, #3, #6):
- no development stage anywhere on the home page: no 已上线 / live, in progress, planned, no live counts (x/y), in text, tooltips or labels;
- the four learning paths (the owner, 2026-10-11: by what the learner wants to do) sit above the tree, each with who it suits, what you can do afterwards, what to know first, its labs and minutes,
  and a start button that opens its first lab as a choice of that path (data-path, data-route-start);
- every lab tile names its point on the tree: its branch and the name on its box;
- with progress kept (window.__aspResume), after a lab was opened the hero offers to continue it (a[data-resume]) with where the learner
  was, its tile is marked, the paths that hold it count it; the strongest mark wins (challenge goal met > guided parts done > opened);
  clearing the record (after a confirmation) takes all of it away;
- no text runs past its box (path entries, the line above the tree, the hero's buttons, the tiles' first line) and no sideways scrolling
  at 1440 zh, 1024 zh, 900 en and 390 zh.
Run: python3 tests/home_r3.py [page.html]   (builds out/home-r3.html when no page is given; screenshots in shots/home-r3/)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
P = pathlib.Path(__file__).resolve().parent.parent
SH = P / 'shots' / 'home-r3'; SH.mkdir(parents=True, exist_ok=True)
page = sys.argv[1] if len(sys.argv) > 1 else None
if not page:
    subprocess.run([sys.executable, str(P / 'build.py'), 'out/home-r3.html'], check=True, cwd=P, stdout=subprocess.DEVNULL)
    page = str(P / 'out' / 'home-r3.html')
WRAP = P / 'out' / 'home-r3-wrapped.html'
WRAP.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + pathlib.Path(page).read_text() + '</body></html>')
URL = 'file://' + str(WRAP)
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


# development words in the home page's visible text, its tooltips, labels and titles, and the tree's SVG text
DEV = r"""()=>{const h=document.getElementById('view-home'), re=/已上线|还没上线|开发中|规划中|下一个开发|\d+\s+(of \d+\s+)?live\b|all live|: live\b|\bLive\b|[Ii]n progress|\b[Pp]lanned\b|[Nn]ext to build|being built/, out=[];
  (h.innerText||'').split('\n').forEach(l=>{if(re.test(l)) out.push('text: '+l.trim().slice(0,80));});
  [...h.querySelectorAll('[data-tipx],[aria-label],[title]')].forEach(x=>['data-tipx','aria-label','title'].forEach(k=>{const v=x.getAttribute(k); if(v&&re.test(v)) out.push(k+': '+v.slice(0,80));}));
  [...h.querySelectorAll('#mapSvg text')].forEach(t=>{if(re.test(t.textContent)||/^\d+\/\d+$/.test(t.textContent.trim())) out.push('svg: '+t.textContent);});
  [...h.querySelectorAll('#pathList, #mapBar, #mapLegend, #tilesHead, #domGrid, #domStrip, .tile-meta')].forEach(e=>{const m=(e.innerText||'').match(/\d+\s*\/\s*\d+/); if(m) out.push('count: '+m[0]);});
  return out;}"""
# the four entries: their fields, their start, their place above the tree
ENTRIES = r"""()=>{const tree=document.querySelector('#map .map-canvas'), grid=document.getElementById('domGrid'), below=(tree&&tree.offsetParent?tree:grid).getBoundingClientRect().top+scrollY;
  return [...document.querySelectorAll('#pathList .pentry')].map(e=>{const a=e.querySelector('a[data-route-start]'), f=k=>{const p=e.querySelector('.pe-'+k); return p?p.innerText.trim():''};
    return {pid:e.dataset.pid, name:f('name'), size:f('size'), who:f('who'), can:f('can'), pre:f('pre'), href:a&&a.getAttribute('href'), p:a&&a.dataset.path, r:a&&a.dataset.routeStart,
      vis:!!(a&&a.offsetParent), btn:a?a.innerText.trim():'', bottom:e.getBoundingClientRect().bottom+scrollY, below, first:'#'+window.__treePath(e.dataset.pid)[0], toggle:!!e.querySelector('.pathchip[data-path="'+e.dataset.pid+'"]')};});}"""
# each tile's first line against its box on the tree
TILES = r"""()=>[...document.querySelectorAll('#tiles .tile')].map(t=>{const a=t.querySelector('.tile-at'), b=document.querySelector('#mapSvg .tnode[data-lab="'+t.dataset.lab+'"] .tlb');
  return {id:t.dataset.lab, at:a?a.innerText.trim():'', box:b?b.textContent.trim():'', w:window.__treeW[t.dataset.lab], meta:t.querySelector('.tile-meta').innerText.replace(/\s+/g,' ')};})"""
# text that runs past its box: each text element against the entry, button row, bar or tile that holds it
BOUNDS = r"""()=>{const out=[], vis=e=>!!e.offsetParent&&e.getClientRects().length;
  const boxes=[...document.querySelectorAll('#pathList .pentry, #pathList .pl-lead, #pathList .pl-rec, #mapBar, #view-home .cta-row, #tiles .tile-meta, #tilesHead')];
  boxes.filter(vis).forEach(b=>{const r=b.getBoundingClientRect();
    [...b.querySelectorAll('*')].filter(e=>vis(e)&&e.getBoundingClientRect().width>1&&[...e.childNodes].some(c=>c.nodeType===3&&c.textContent.trim())).forEach(e=>{
      const q=document.createRange(); q.selectNodeContents(e); const rr=[...q.getClientRects()];
      const L=Math.min(...rr.map(x=>x.left)), R=Math.max(...rr.map(x=>x.right));
      if(L<r.left-1||R>r.right+1) out.push((b.id||b.className)+' > '+(e.className||e.tagName)+': '+e.textContent.trim().slice(0,40)+' ['+Math.round(L)+','+Math.round(R)+'] in ['+Math.round(r.left)+','+Math.round(r.right)+']');
      if(e.scrollWidth>e.clientWidth+1&&getComputedStyle(e).overflow!=='visible') out.push('clipped: '+e.textContent.trim().slice(0,40));});});
  [...document.querySelectorAll('#pathList .btn, #view-home .cta-row .btn')].filter(vis).forEach(e=>{if(e.scrollWidth>e.clientWidth+1) out.push('button: '+e.innerText.slice(0,40));});
  return out;}"""

with sync_playwright() as p:
    b = p.chromium.launch()
    # 1. a first visit: no development stage, the entries, the tiles' tree points, the bounds
    for w, h, lang in [(1440, 900, 'zh'), (1024, 800, 'zh'), (900, 900, 'en'), (390, 844, 'zh')]:
        pg = b.new_page(viewport={'width': w, 'height': h}, reduced_motion='reduce'); errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(URL); pg.wait_for_timeout(500)
        if lang == 'zh': pg.click('.langsw button[data-lang="zh"]'); pg.wait_for_timeout(300)
        tag = f'{w} {lang}'
        check(f'{tag}: no “Start with a path” section and no route cards', pg.evaluate("()=>!document.getElementById('routes')&&!document.querySelector('.route-card,#routeGrid,#routeMore')"))
        dev = pg.evaluate(DEV)
        check(f'{tag}: no development stage anywhere on the home page', not dev, dev[:6])
        ents = pg.evaluate(ENTRIES)
        check(f'{tag}: four path entries, by what the learner wants to do', len(ents) == 4 and [e['pid'] for e in ents] == ['transformer', 'serving', 'research', 'apps'], [e['pid'] for e in ents])
        # the owner, 2026-10-11: names, the start button names the first lab, "See the learning path", plain "Afterwards you can"
        NAMES = {'zh': ['AI 系统小白入门', 'Infra 岗', '研究岗（RS / RE）', 'Agent 应用岗'], 'en': ['AI systems for beginners', 'Infra engineer', 'Research (RS / RE)', 'Agent and app engineer']}[lang]
        check(f'{tag}: the four names', [e['name'] for e in ents] == NAMES, [e['name'] for e in ents])
        short = pg.evaluate("()=>Object.fromEntries(Object.keys(ASP.catalog.PATHS).map(k=>[k,ASP.catalog.LAB[ASP.catalog.PATHS[k].labs[0]].s]))")
        bad = [e for e in ents if ' '.join(e['btn'].split()) != ('进入第一站：' if lang == 'zh' else 'Stop 1: ') + short[e['pid']][lang] + ' →']
        check(f'{tag}: each start button reads “{"进入第一站：" if lang == "zh" else "Stop 1: "}<the first lab>”', not bad, [(e['pid'], e['btn']) for e in bad])
        # the owner, 2026-10-12: no “See the learning path” button; the whole entry is the button that shows the path on the tree
        tg = pg.evaluate("""()=>[...document.querySelectorAll('#pathList .pentry')].map(e=>{const b=e.querySelector(':scope>.pathchip.pe-hit'), r=e.getBoundingClientRect(), q=b?b.getBoundingClientRect():null;
          return {text:b?b.innerText.trim():null, label:b?b.getAttribute('aria-label'):'', cover:!!q&&Math.abs(q.width-r.width)<=2&&Math.abs(q.height-r.height)<=2, old:/查看学习路线|See the learning path/.test(e.innerText)}})""")
        check(f'{tag}: each entry is one button over the whole entry, named for screen readers, with no “{"查看学习路线" if lang == "zh" else "See the learning path"}”',
              len(tg) == 4 and all(x['text'] == '' and x['cover'] and len(x['label']) > 8 and not x['old'] for x in tg), tg)
        # second reader (2026-10-13, “nobody saw the four route boxes could be clicked”): without hovering, each entry looks pressable
        # (a shadow) and its start link is a solid button; the line above the tree names both clicks.
        # The owner, 2026-10-13 04:22: no extra line on the entry saying what a press does (“画蛇添足”); a card is itself the button
        cue = pg.evaluate("""()=>[...document.querySelectorAll('#pathList .pentry')].map(e=>{const a=e.querySelector('.pe-go .btn'), cs=getComputedStyle(e);
          return {pid:e.dataset.pid, shadow:cs.boxShadow!=='none', see:!!e.querySelector('.pe-see'), line:/在知识树上看这|按顺序列出这|已按顺序标出|on the tree ↓|in order ↓|are numbered below/.test(e.innerText),
            solid:!!a&&getComputedStyle(a).backgroundColor===getComputedStyle(document.body).color}})""")
        bad = [c for c in cue if not (c['shadow'] and not c['see'] and not c['line'] and c['solid'])]
        check(f'{tag}: each entry looks pressable without a hover (shadow), carries no extra “what a press does” line, and its start link is a solid button', len(cue) == 4 and not bad, bad[:2])
        now = pg.evaluate("()=>document.getElementById('mapNow').innerText")
        check(f'{tag}: the line above the tree says what a click on a branch name and on a box does', (('点枝的名字' in now and '点一个方块' in now) if lang == 'zh' else ('branch’s name' in now and 'box' in now)), now)
        # the owner, 2026-10-13 04:22: every lab tile works like a route card, the tile itself is the button;
        # no “Open the lab” button and no “Assumptions & sources” hover; the mini experiment keeps its own controls
        tg2 = pg.evaluate("""()=>{const t=[...document.querySelectorAll('#tiles .tile.is-live')].filter(e=>e.offsetParent), nx=document.querySelector('#tile-architecture .ar-tzbtn .ar-tz-next');
          return {n:t.length, go:document.querySelectorAll('#tiles .tile-go,#tiles .tile-foot').length, more:document.querySelectorAll('#tiles .tz-more').length,
            words:t.filter(e=>/打开 Lab|Open the lab|假设和来源|Assumptions (&|and) sources/.test(e.innerText)).map(e=>e.dataset.lab),
            look:t.every(e=>{const c=getComputedStyle(e); return c.cursor==='pointer'&&c.boxShadow!=='none';}), next:nx?nx.getBoundingClientRect().width>20:null}}""")
        check(f'{tag}: each tile is one button (pointer, shadow) with no “open” button and no assumptions-and-sources hover; the structure tile’s picture keeps its “next stage” button',
              tg2['n'] >= 3 and tg2['go'] == 0 and tg2['more'] == 0 and not tg2['words'] and tg2['look'] and tg2['next'] is not False, tg2)
        TERMS = ['config', 'MHA', 'GQA', 'MLA', 'DPO', 'SFT', 'GRPO', 'RLVR', 'PPO', 'KV', 'vLLM', 'RAG', 'agent', 'Agent', 'token', 'LLM', 'attention', 'cache', 'batch']
        tech = [(e['pid'], t_) for e in ents for t_ in TERMS if t_ in e['can']]
        check(f'{tag}: “Afterwards you can” uses no technical term', not tech, tech)
        hd = pg.evaluate("()=>document.getElementById('mapH').innerText.trim()")
        check(f'{tag}: the heading above the paths', hd == ('从最适合你的路线开始' if lang == 'zh' else 'Start with the path that suits you best'), hd)
        cta = pg.evaluate("()=>{const a=document.getElementById('heroCta'), p=document.getElementById('heroLab'); if(!a||!p) return null; const r=a.getBoundingClientRect(), q=p.getBoundingClientRect(); return {t:a.innerText.trim(), inCard:p.contains(a), right:Math.round(q.right-r.right), bottom:Math.round(q.bottom-r.bottom)};}")
        check(f'{tag}: the hero lab’s button sits at the bottom right of its card, “{"10 分钟学会 Transformer 内部结构" if lang == "zh" else "Learn what is inside a Transformer in 10 minutes"} →”',
              cta and cta['inCard'] and cta['t'].startswith('10 分钟学会 Transformer 内部结构' if lang == 'zh' else 'Learn what is inside a Transformer in 10 minutes') and 0 <= cta['right'] <= 24 and 0 <= cta['bottom'] <= 24, cta)
        bad = [e['pid'] for e in ents if not (e['bottom'] <= e['below'] and e['vis'] and e['href'] == e['first'] and e['p'] == e['r'] == e['pid'] and e['toggle'])]
        check(f'{tag}: each entry sits above the tree (or the branch list) with a start button to its first lab, as a choice of its path, and is itself the button that shows the path', not bad, [e for e in ents if e['pid'] in bad][:2])
        mins = pg.evaluate("()=>Object.fromEntries(Object.entries(ASP.catalog.PATHS).map(([k,p])=>[k,[p.labs.length,p.labs.reduce((a,id)=>a+ASP.catalog.LAB[id].min,0)]]))")
        words = ('个 Lab', '分钟') if lang == 'zh' else ('labs', 'min')
        bad = [e['pid'] for e in ents if not (len(e['who']) > 8 and len(e['can']) > 12 and len(e['pre']) > 8 and f"{mins[e['pid']][0]} {words[0]}" in e['size'] and f"{mins[e['pid']][1]} {words[1]}" in e['size'])]
        check(f'{tag}: each entry says who it suits, what you can do afterwards, what to know first, its labs and its minutes', not bad, [e for e in ents if e['pid'] in bad][:2])
        tl = pg.evaluate(TILES)
        pre = '知识树：' if lang == 'zh' else 'Tree: '
        wn = pg.evaluate("()=>Object.fromEntries(Object.entries(ASP.catalog.WORLDS).map(([k,w])=>[k,w.name[document.documentElement.lang.startsWith('zh')?'zh':'en']]))")
        bad = [t for t in tl if not (t['at'].startswith(pre) and wn[t['w']] in t['at'] and t['at'].endswith(t['box']) and t['box'])]
        check(f'{tag}: every tile names its point on the tree (branch › the name on its box)', len(tl) >= 5 and not bad, bad[:2])
        bd = pg.evaluate(BOUNDS)
        check(f'{tag}: no text runs past its box (entries, the line above the tree, hero buttons, tiles’ first lines)', not bd, bd[:5])
        ov = pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
        check(f'{tag}: no sideways scrolling', ov <= 0, ov)
        check(f'{tag}: no continue link and no marks on a first visit', pg.evaluate("()=>!document.querySelector('a[data-resume]:not([hidden])')&&!document.querySelector('#tiles .tm')&&document.getElementById('progRec').hidden"))
        pg.evaluate("()=>document.getElementById('mapH').scrollIntoView()"); pg.wait_for_timeout(150)
        pg.screenshot(path=str(SH / f'paths_{w}_{lang}.png'))
        pg.click('#pathList .pentry[data-pid="serving"] .pe-hit', position={'x': 30, 'y': 30}); pg.wait_for_timeout(400)
        sel = pg.evaluate("()=>document.querySelector('#pathList .pentry[data-pid=serving]').classList.contains('sel')")
        check(f'{tag}: after a press the entry shows it is the chosen path', sel, sel)
        # a tile is one button: a click on a mini experiment's own control stays on the home page, a click anywhere else opens the lab
        tc = pg.evaluate("""()=>{const t=[...document.querySelectorAll('#tiles .tile.is-live')].find(e=>e.offsetParent&&e.querySelector('.tile-tz .seg-ctl button:not([aria-pressed=true]):not(.on)'));
          if(!t) return null; t.scrollIntoView({block:'center'}); return t.dataset.lab;}""")
        if tc:
            pg.wait_for_timeout(150); h0 = pg.evaluate("()=>location.hash")
            pg.click(f'#tile-{tc} .tile-tz .seg-ctl button:not([aria-pressed=true]):not(.on)'); pg.wait_for_timeout(250)
            h1 = pg.evaluate("()=>location.hash")
            pg.click(f'#tile-{tc} .tile-meta', position={'x': 4, 'y': 4}); pg.wait_for_timeout(500)
            h2 = pg.evaluate("()=>location.hash")
            check(f'{tag}: a click on a tile’s own control stays on the home page; a click on the rest of the tile opens its lab', h1 == h0 and h2 == '#' + tc, (tc, h0, h1, h2))
        else:
            check(f'{tag}: a tile with its own control is on the page for the click check', False, 'none found')
        check(f'{tag}: no page errors', not errs, errs[:3])
        pg.close()

    # 2. progress kept: open a lab, come back; then the strongest mark; then clear
    for lang in ['zh', 'en']:
        ctx = b.new_context(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce')
        ctx.add_init_script('window.__aspResume=true')
        pg = ctx.new_page(); errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(URL); pg.wait_for_timeout(500)
        pg.evaluate("()=>{try{localStorage.removeItem('asp-progress-v1')}catch(e){} window.__prog.clear()}")
        if lang == 'zh': pg.click('.langsw button[data-lang="zh"]'); pg.wait_for_timeout(300)
        tag = f'1440 {lang} progress'
        pg.evaluate("()=>{location.hash='sampling'}"); pg.wait_for_timeout(900)
        pg.evaluate("()=>{const b=[...document.querySelectorAll('#view-dyn [data-nav=\"next\"]')].find(b=>!b.disabled&&b.offsetParent); if(b) b.click();}"); pg.wait_for_timeout(400)
        rec = pg.evaluate("()=>{window.__prog.save(); return window.__prog.get('sampling')}")
        pg.evaluate("()=>{location.hash='top'}"); pg.wait_for_timeout(700)
        pg.evaluate("()=>document.querySelector('#mapSvg .world[data-world=\"models\"]').dispatchEvent(new MouseEvent('click',{bubbles:true}))"); pg.wait_for_timeout(300)
        rs = pg.evaluate("()=>{const a=document.querySelector('a[data-resume]'); return a?{id:a.dataset.resume, vis:!!a.offsetParent, href:a.getAttribute('href'), t:a.innerText.replace(/\\s+/g,' '), inRow:!!a.closest('.hero .cta-row')}:null}")
        k = (rec or {}).get('step', 0) + 1; nS = (rec or {}).get('nSteps', 0)
        where = f'第 {k}/{nS} 部分' if lang == 'zh' else f'part {k} of {nS}'
        lead = '继续上次' if lang == 'zh' else 'Continue'
        check(f'{tag}: the hero offers to continue the lab, and says where the learner was', bool(rs) and rs['id'] == 'sampling' and rs['vis'] and rs['href'] == '#sampling' and rs['inRow'] and lead in rs['t'] and where in rs['t'] and nS > 1, (rs, rec))
        mk = pg.evaluate("()=>{const t=document.querySelector('#tile-sampling .tm'); return t?t.innerText.trim():null}")
        check(f'{tag}: the lab’s tile is marked “opened”', mk == ('打开过' if lang == 'zh' else 'Opened'), mk)
        pc = pg.evaluate("()=>Object.fromEntries([...document.querySelectorAll('[data-pprog]')].map(e=>[e.dataset.pprog, e.hidden?'':e.innerText.trim()]))")
        holds = pg.evaluate("()=>Object.entries(ASP.catalog.PATHS).filter(([k,p])=>p.labs.includes('sampling')).map(([k])=>k)")
        one = '你打开过其中 1 个' if lang == 'zh' else 'You opened 1 of them'
        check(f'{tag}: each path that holds the lab counts it, the others say nothing', all((pc[k] == one) == (k in holds) and (pc[k] in ('', one)) for k in pc) and holds, (pc, holds))
        check(f'{tag}: the clear button is offered', pg.evaluate("()=>!document.getElementById('progRec').hidden"))
        sh_dev = pg.evaluate(DEV)
        check(f'{tag}: still no development stage', not sh_dev, sh_dev[:3])
        bd = pg.evaluate(BOUNDS)
        check(f'{tag}: no text runs past its box with the continue link and the marks', not bd, bd[:4])
        if lang == 'zh':
            pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(150); pg.screenshot(path=str(SH / 'resume_1440_zh.png'))
        # the strongest mark: a record with the challenge met, one with the guided parts done (the record as the kit keeps it)
        pg.evaluate("""()=>{const d=JSON.parse(localStorage.getItem('asp-progress-v1')); const t=Date.now();
          d.labs['dpo']={mode:'challenge',step:0,nSteps:7,maxStep:6,guided:true,passed:true,t}; d.labs['lora']={mode:'guided',step:6,nSteps:7,maxStep:6,guided:true,passed:false,t:t-1};
          d.last='dpo'; localStorage.setItem('asp-progress-v1',JSON.stringify(d));}""")
        pg.reload(); pg.wait_for_timeout(700)
        pg.evaluate("()=>{const c=document.querySelector('.pathchip[data-path=\"research\"]'); c.click();}"); pg.wait_for_timeout(300)
        mk = pg.evaluate("()=>Object.fromEntries(['dpo','lora','sft'].map(id=>{const t=document.querySelector('#tile-'+id+' .tm'); return [id, t?t.innerText.trim():'']}))")
        want = {'dpo': '✓ 挑战已达标', 'lora': '引导看完了', 'sft': ''} if lang == 'zh' else {'dpo': '✓ Challenge goal met', 'lora': 'Guided parts done', 'sft': ''}
        check(f'{tag}: a tile shows one mark, the strongest (challenge goal met, guided parts done, opened)', mk == want, mk)
        rs = pg.evaluate("()=>{const a=document.querySelector('a[data-resume]'); return a?a.dataset.resume+' | '+a.innerText.replace(/\\s+/g,' '):null}")
        check(f'{tag}: the continue link follows the last lab, and says the challenge met its goal', bool(rs) and rs.startswith('dpo') and (('挑战题，已达标' in rs) if lang == 'zh' else ('Challenge, goal met' in rs)), rs)
        # clearing asks first; "Cancel" keeps everything, "OK" takes it all away
        pg.once('dialog', lambda d: d.dismiss())
        pg.click('#progClear'); pg.wait_for_timeout(200)
        check(f'{tag}: cancelling the confirmation keeps the record', pg.evaluate("()=>!!document.querySelector('a[data-resume]')&&!!document.querySelector('#tiles .tm')"))
        msgs = []
        pg.once('dialog', lambda d: (msgs.append(d.message), d.accept()))
        pg.click('#progClear'); pg.wait_for_timeout(300)
        gone = pg.evaluate("()=>({link:!!document.querySelector('a[data-resume]'), vis:!!document.querySelector('#heroResume:not([hidden])'), marks:document.querySelectorAll('#tiles .tm').length, counts:[...document.querySelectorAll('[data-pprog]')].filter(e=>!e.hidden).length, rec:!document.getElementById('progRec').hidden, last:window.__prog.last()})")
        check(f'{tag}: after confirming, the continue link, the marks and the counts are gone', not any([gone['link'], gone['vis'], gone['marks'], gone['counts'], gone['rec'], gone['last']]), gone)
        check(f'{tag}: the confirmation speaks of this browser on this device, not of storage', bool(msgs) and (('这台设备上的浏览器' in msgs[0]) if lang == 'zh' else ('this browser on this device' in msgs[0])) and not re.search(r'localStorage|storage|存储', msgs[0]), msgs)
        txt = pg.evaluate("()=>document.getElementById('view-home').innerText+' '+[...document.querySelectorAll('#view-home [data-tipx]')].map(e=>e.dataset.tipx).join(' ')")
        check(f'{tag}: the home page never says localStorage', 'localStorage' not in txt and 'local storage' not in txt.lower(), '')
        check(f'{tag}: no page errors', not errs, errs[:3])
        ctx.close()
    b.close()
print(f'home_r3: {n} checks, {len(fails)} failed')
print('FAILED:', fails if fails else 'none')
sys.exit(1 if fails else 0)
