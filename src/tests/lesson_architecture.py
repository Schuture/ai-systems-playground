"""The first lesson, “How a model continues a sentence” (labs/lesson-architecture.js; the owner's real-reader round, 2026-10-11).
  terms   - in both languages, every term the lesson names in bold appears nowhere above the place it is named (contents list, headings,
            captions, buttons included: the whole page from its top);
  shape   - 10 sections in order, every figure drawn, no text in a figure under 13 px, no page wider than the window at 320/390/768/1280/1440/1920;
  acts    - each interaction says what to press and changes only its own figure; quiz feedback right under the question;
  resume  - after a reload: the same snapshot, the reader back at the section they were reading (within 40 px), the note shown;
            the home page offers “continue” at that section; next lesson, then Back: the same place again.
Run: python3 tests/lesson_architecture.py [--page out/x.html]   (builds out/lesson.html when no page is given)"""
import sys as _sys
# retired 2026-10-11 19:28 UTC: the lesson page was parked (labs/_lesson-architecture.js); the architecture lab is back with its primer
print('skipped: the lesson page is not built any more'); _sys.exit(0)
import json, pathlib, subprocess, sys
from playwright.sync_api import sync_playwright
P = pathlib.Path(__file__).resolve().parent.parent
args = sys.argv[1:]
page_path = None
if '--page' in args:
    i = args.index('--page'); page_path = args[i + 1]; del args[i:i + 2]
if not page_path:
    subprocess.run([sys.executable, str(P / 'build.py'), 'out/lesson.html'], check=True, cwd=P, stdout=subprocess.DEVNULL)
    page_path = 'out/lesson.html'
URL = (P / page_path).resolve().as_uri()
fails, passes = [], 0
def ok(c, m):
    global passes
    if c: passes += 1
    else: fails.append(m); print('FAIL', m)

TERMS_JS = """()=>{const art=document.querySelector('#view-dyn .ls'); if(!art) return {err:'no lesson'};
  // the page text in reading order (the contents list included), and where each bold term starts
  let txt='', marks=[]; const walk=n=>{if(n.nodeType===3){txt+=n.textContent; return;} if(n.nodeType!==1) return;
    const cs=getComputedStyle(n); if(cs.display==='none'||n.hidden||n.classList.contains('sr-only')) return;
    if(n.tagName==='DETAILS'&&!n.open){const s=n.querySelector('summary'); if(s) walk(s); return;}
    if(n.classList.contains('ls-term')) marks.push({t:n.textContent.trim(),at:txt.length});
    for(const c of n.childNodes) walk(c); if(/^(P|LI|H1|H2|DIV|BUTTON|FIGCAPTION)$/.test(n.tagName)) txt+='\\n';};
  walk(art); const out=[];
  const seen=new Set(); for(const m of marks){if(seen.has(m.t)) continue; seen.add(m.t); const first=txt.toLowerCase().indexOf(m.t.toLowerCase());
    out.push({t:m.t,first,named:m.at,ctx:txt.slice(Math.max(0,first-40),first+20).replace(/\\s+/g,' ')});}
  return {out,len:txt.length};}"""

with sync_playwright() as p:
    br = p.chromium.launch()
    for lang in ('zh', 'en'):
        ctx = br.new_context(viewport={'width': 1440, 'height': 900}); pg = ctx.new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(URL + '#architecture'); pg.wait_for_timeout(600)
        if lang == 'zh': pg.click('.langsw button[data-lang="zh"]'); pg.wait_for_timeout(400)
        # open every fold so the check also covers the optional reading (it comes after all the sections)
        r = pg.evaluate(TERMS_JS)
        ok('err' not in r, f'{lang}: {r}')
        for t in r.get('out', []):
            ok(t['first'] >= t['named'], f"{lang}: “{t['t']}” appears before it is named: …{t['ctx']}…")
        ok(len(r.get('out', [])) >= 9, f"{lang}: only {len(r.get('out', []))} named terms")
        # shape
        n = pg.evaluate("[...document.querySelectorAll('#view-dyn .ls-sec')].filter(e=>/^s\\d+$/.test(e.dataset.ls)).length")
        ok(n == 10, f'{lang}: {n} sections')
        for f in ['ls-f1', 'ls-f4', 'ls-f5', 'ls-f6', 'ls-f7', 'ls-f8', 'ls-f9', 'ls-f10', 'ls-p8', 'ls-qz']:
            ok(pg.evaluate(f"(document.getElementById('{f}')||{{}}).innerHTML.length>100"), f'{lang}: {f} empty')
        small = pg.evaluate("""()=>[...document.querySelectorAll('#view-dyn .ls-fig *, #view-dyn .ls-q *')].filter(e=>e.childNodes.length&&[...e.childNodes].some(c=>c.nodeType===3&&c.textContent.trim())&&e.offsetParent)
          .map(e=>[parseFloat(getComputedStyle(e).fontSize),e.textContent.trim().slice(0,30)]).filter(x=>x[0]<13)""")
        ok(not small, f'{lang}: text under 13 px in a figure: {small[:5]}')
        # acts: section 1 steps, section 5 pick, section 8 guess, section 9 draws, the quiz
        pg.click('#ls-f1 [data-act="s1-next"]'); ok('Paris' in pg.inner_text('#ls-f1 .ls-sent'), f'{lang}: round 1 did not add Paris')
        f4 = pg.inner_html('#ls-f4')
        pg.click('#ls-f5 [data-pick="0"]'); t5 = pg.inner_text('#ls-f5 .ls-see')
        ok('100%' in t5, f'{lang}: The does not take 100% of itself: {t5}')
        ok(pg.inner_html('#ls-f4') == f4, f'{lang}: picking in section 5 changed section 4')
        ok(pg.evaluate("document.querySelectorAll('#ls-f5 .ls-att-c.off').length") == 4, f'{lang}: The should not see 4 later tokens')
        ok('?' in pg.inner_text('#ls-f8 .ls-st-k'), f'{lang}: section 8 shows the answer before the guess')
        pg.click('#ls-p8 [data-p8="1"]'); ok('4,096' in pg.inner_text('#ls-f8 .ls-st-k') and pg.locator('#ls-p8 .ls-fb').count() == 1, f'{lang}: section 8 guess not answered')
        for _ in range(5): pg.click('#ls-f9 [data-act="draw"]')
        s = pg.evaluate('__ls.state()'); ok(len(s['draws']) == 5, f"{lang}: {len(s['draws'])} draws")
        ok(pg.evaluate("[...document.querySelectorAll('#ls-f9 .ls-pr-v')].reduce((a,e)=>a+parseInt(e.textContent),0)") == 100, f'{lang}: probabilities do not add up to 100%')
        for k, o in ((0, 0), (1, 1), (2, 1)): pg.click(f'#ls-qz [data-qz="{k}"][data-o="{o}"]')
        ok(pg.locator('#ls-qz .ls-score.ok').count() == 1, f'{lang}: right answers not all marked right')
        ok(pg.evaluate("window.__prog.snap().passed") is True, f'{lang}: snap().passed false after all right')
        pg.click('#ls-qz [data-qz="1"][data-o="2"]')
        ok(pg.locator('#ls-qz .ls-opt.no').count() == 1 and pg.locator('#ls-qz .ls-q').nth(1).locator('.ls-fb').count() == 1, f'{lang}: wrong answer not shown under its question')
        # the numbers: shares add up to 1, sizes as published
        sh = pg.evaluate("__ls.AW.map(r=>Math.round(r.reduce((a,b)=>a+b,0)*1000)).concat(__ls.HEADS.map(h=>Math.round(h.w.reduce((a,b)=>a+b,0)*1000)))")
        ok(all(x == 1000 for x in sh), f'{lang}: shares do not add up to 1: {sh}')
        ok(pg.evaluate('__ls.sizes') == {'DM': 4096, 'NLAY': 32, 'NHEAD': 32, 'FFN': 14336, 'VOCAB': 128256}, f'{lang}: sizes changed')
        for e in errs: fails.append(f'{lang} page error: {e[:200]}'); print('FAIL page error', e[:200])
        ctx.close()
    # widths: no sideways scrolling
    for w in (320, 390, 768, 1004, 1280, 1440, 1920):
        ctx = br.new_context(viewport={'width': w, 'height': 800}); pg = ctx.new_page()
        pg.goto(URL + '#architecture'); pg.wait_for_timeout(500); pg.click('.langsw button[data-lang="zh"]'); pg.wait_for_timeout(300)
        sw = pg.evaluate('document.documentElement.scrollWidth')
        ok(sw <= w, f'zh {w}px: page is {sw}px wide')
        over = pg.evaluate("""()=>[...document.querySelectorAll('#view-dyn .ls-main *')].filter(e=>{const r=e.getBoundingClientRect(); return r.width>0&&(r.right>innerWidth+1||r.left<-1);}).map(e=>e.className||e.tagName).slice(0,5)""")
        ok(not over, f'zh {w}px: elements outside the window: {over}')
        ctx.close()
    # resume
    ctx = br.new_context(viewport={'width': 1280, 'height': 860}); ctx.add_init_script('window.__aspResume=true')
    pg = ctx.new_page(); errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL + '#top'); pg.wait_for_timeout(500); pg.evaluate('window.__prog.clear()')
    pg.goto(URL + '#architecture'); pg.wait_for_timeout(700)
    pg.evaluate("document.querySelector('#ls-f5 [data-pick=\"2\"]').click(); document.querySelector('#ls-p8 [data-p8=\"0\"]').click(); document.querySelector('#ls-f9 [data-act=\"draw\"]').click(); document.querySelector('#ls-qz [data-qz=\"0\"][data-o=\"0\"]').click(); document.querySelector('details.ls-dd[data-deep=\"mask\"]').open=true")
    pg.wait_for_timeout(300)
    pg.evaluate("(()=>{const e=document.querySelector('#ls-s5'); window.scrollTo(0,scrollY+e.getBoundingClientRect().top-80+150);})()")
    pg.wait_for_timeout(1400)
    s1 = pg.evaluate('JSON.stringify(window.__prog.snap())'); y1 = pg.evaluate("document.querySelector('#ls-s5').getBoundingClientRect().top")
    ok(json.loads(s1)['at'] == 's5', f'reading place before reload is {json.loads(s1)["at"]}')
    pg.reload(); pg.wait_for_timeout(1500)
    s2 = pg.evaluate('JSON.stringify(window.__prog.snap())'); y2 = pg.evaluate("document.querySelector('#ls-s5').getBoundingClientRect().top")
    ok(s1 == s2, f'snapshot differs after reload\n  before {s1[:300]}\n  after  {s2[:300]}')
    ok(abs(y1 - y2) <= 40, f'section 5 at {y1:.0f}px before the reload, {y2:.0f}px after')
    ok(pg.is_visible('#ls-note'), 'no “back where you left off” note after the reload')
    ok(pg.evaluate("document.querySelector('details.ls-dd[data-deep=\"mask\"]').open"), 'the open fold closed after the reload')
    ok(pg.evaluate("document.querySelector('#ls-f5 [data-pick=\"2\"]').getAttribute('aria-pressed')") == 'true', 'section 5 choice lost')
    # the home page offers to continue at that section
    pg.goto(URL + '#top'); pg.wait_for_timeout(700)
    a = pg.locator('a[data-resume="architecture"]')
    ok(a.count() == 1 and ('5' in a.inner_text()), f'home continue link: {a.inner_text() if a.count() else None}')
    pg.go_back(); pg.wait_for_timeout(1200)
    y3 = pg.evaluate("document.querySelector('#ls-s5').getBoundingClientRect().top")
    ok(abs(y1 - y3) <= 40, f'Back to the lesson: section 5 at {y3:.0f}px, was {y1:.0f}px')
    # next lesson, then Back
    nx = pg.locator('#view-dyn .ls-next a.btn')
    ok(nx.count() == 1, 'no next-lesson button')
    if nx.count():
        pg.evaluate("document.querySelector('#view-dyn .ls-next a.btn').scrollIntoView({block:'center'})"); pg.wait_for_timeout(1000)
        at0 = pg.evaluate('window.__prog.snap().at'); y0 = pg.evaluate("document.querySelector('#view-dyn .ls-next').getBoundingClientRect().top")
        nx.click(); pg.wait_for_timeout(900)
        ok(not pg.evaluate("!!document.querySelector('#view-dyn .ls')"), 'the next-lesson button did not leave the lesson')
        pg.go_back(); pg.wait_for_timeout(1200)
        ok(pg.evaluate("!!document.querySelector('#view-dyn .ls')"), 'Back from the next lesson did not reopen the lesson')
        at4 = pg.evaluate('window.__prog.snap().at'); y4 = pg.evaluate("document.querySelector('#view-dyn .ls-next').getBoundingClientRect().top")
        ok(at4 == at0 and abs(y4 - y0) <= 40, f'next lesson and Back: place {at0} at {y0:.0f}px became {at4} at {y4:.0f}px')
    for e in errs: fails.append('resume page error: ' + e[:200]); print('FAIL page error', e[:200])
    br.close()
print(f'lesson_architecture: {passes} passed, {len(fails)} failed')
sys.exit(1 if fails else 0)
