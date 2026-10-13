"""A reload keeps what the learner did (the owner, round 3 #6). For each lab given (default: every lab):
  guided   - go to part 3 (or the last part when there are fewer), answer the part's prediction if it has one; reload: the same part,
             the same answer, the same snapshot (window.__prog.snap()), the same card text;
  sandbox  - Scenarios, change the first setting that has another value; reload (the address keeps the settings): the same snapshot;
  challenge- Challenge, change one setting, press Submit if there is one; reload: still Challenge, the same snapshot and result text;
  home     - after the lab, the home page offers to continue it.
A lab without registerLab({snap, restore}) fails (except with --kit-only, which checks only the kit's record of mode and part).
Run: python3 tests/kit_resume.py [id ...] [--page out/x.html] [--kit-only]   (builds out/resume.html when no page is given)"""
import json, pathlib, subprocess, sys
from playwright.sync_api import sync_playwright
P = pathlib.Path(__file__).resolve().parent.parent
args = sys.argv[1:]
kit_only = '--kit-only' in args
if kit_only: args.remove('--kit-only')
page_path = None
if '--page' in args:
    i = args.index('--page'); page_path = args[i + 1]; del args[i:i + 2]
if not page_path:
    subprocess.run([sys.executable, str(P / 'build.py'), 'out/resume.html'], check=True, cwd=P, stdout=subprocess.DEVNULL)
    page_path = 'out/resume.html'
URL = (P / page_path).resolve().as_uri()
ids = args
fails, passes = [], 0

def ok(cond, msg):
    global passes
    if cond: passes += 1
    else: fails.append(msg); print('FAIL', msg)

def settle(pg, ms=500): pg.wait_for_timeout(ms)
def snap(pg): return pg.evaluate('JSON.stringify(window.__prog&&window.__prog.snap())')
def stable(pg, fn, ms=15000):
    """the value once it has stopped changing (an animation that plays after a click or a reload ends first)"""
    v, same, t = fn(), 0, 0
    while same < 5 and t < ms:
        pg.wait_for_timeout(300); t += 300; w = fn()
        same = same + 1 if w == v else 0; v = w
    return v
def card(pg, root):
    return pg.evaluate("""r=>{const v=document.querySelector(r); if(!v) return ''; const c=v.querySelector('.steps .step.current .step-b')||v.querySelector('.rail .railcard, #rail .railcard');
      return c?c.innerText.replace(/\\s+/g,' ').slice(0,400):'';}""", root)
def mode(pg, root): return pg.evaluate("r=>{const b=document.querySelector(r+' .modes button[aria-pressed=\"true\"]'); return b?b.dataset.mode:null;}", root)

with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(viewport={'width': 1440, 'height': 900})
    ctx.add_init_script('window.__aspResume=true')   # the kit keeps nothing in automated browsers unless a test opts in
    pg = ctx.new_page(); errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL + '#top'); settle(pg, 800)
    if not ids:
        ids = pg.evaluate("ASP.catalog.LABS.filter(l=>ASP.catalog.labStatus(l)==='live').map(l=>l.id)")   # every lab in the build
    for lid in ids:
        root = '#view-lab' if lid == 'kv-cache' else '#view-dyn'
        pg.evaluate("window.__prog&&window.__prog.clear()")
        pg.goto(URL + '#' + lid); pg.reload(); settle(pg, 900)
        if pg.evaluate("!!document.querySelector('#view-dyn .ls')"):   # a lesson page (no modes or parts): tests/lesson_architecture.py checks its resume
            print('skip', lid, '(lesson page)'); continue
        has = pg.evaluate("window.__prog.snap()!==null")
        if not has and not kit_only:
            ok(False, f'{lid}: no snap() (registerLab({{snap, restore}}) missing)'); continue
        # guided: part 3 (index 2) or the last
        n = pg.evaluate(f"document.querySelectorAll('{root} .steps>.step').length")
        k = min(2, max(0, n - 1))
        b = pg.locator(f'{root} .steps .step-h[data-go="{k}"]')
        if b.count(): b.first.click(); settle(pg)
        pr = pg.locator(f'{root} .steps .step.current [data-pred]:not([aria-pressed="true"])')
        if pr.count(): pr.first.click(); settle(pg)
        settle(pg, 500)   # snapshots are taken right before the reload: an animation that ends meanwhile changes what the lab records
        c1 = stable(pg, lambda: card(pg, root)); s1 = snap(pg)
        cur1 = pg.evaluate(f"[...document.querySelectorAll('{root} .steps>.step')].findIndex(x=>x.classList.contains('current'))")
        pg.reload(); settle(pg, 1000)
        cur2 = pg.evaluate(f"[...document.querySelectorAll('{root} .steps>.step')].findIndex(x=>x.classList.contains('current'))")
        ok(cur1 == cur2, f'{lid} guided: part {cur1} became {cur2} after reload')
        ok(mode(pg, root) == 'guided', f'{lid} guided: mode after reload is {mode(pg, root)}')
        if not kit_only:
            c2 = stable(pg, lambda: card(pg, root)); s2 = snap(pg)
            ok(s1 == s2, f'{lid} guided: snapshot differs after reload\n  before {str(s1)[:300]}\n  after  {str(s2)[:300]}')
            ok(c1 == c2, f'{lid} guided: card text differs after reload\n  before {c1[:160]}\n  after  {c2[:160]}')
        # sandbox: change the first setting with another value
        pg.locator(f'{root} .modes button[data-mode="sandbox"]:not([data-flow=free])').first.click(); settle(pg)
        seg = pg.locator(f'{root} .controls .seg-ctl button[data-v]:not([aria-pressed="true"]):not([disabled]):visible')
        if seg.count(): seg.first.click(); settle(pg)
        settle(pg, 500); s1, h1 = snap(pg), pg.evaluate('location.hash')
        pg.reload(); settle(pg, 1000)
        ok(mode(pg, root) == 'sandbox', f'{lid} sandbox: mode after reload is {mode(pg, root)}')
        if not kit_only:
            s2 = snap(pg); ok(s1 == s2, f'{lid} sandbox: snapshot differs after reload\n  before {str(s1)[:300]}\n  after  {str(s2)[:300]}')
        # challenge: one change, submit
        pg.locator(f'{root} .modes button[data-mode="challenge"]').first.click(); settle(pg)
        seg = pg.locator(f'{root} .rail .seg-ctl button[data-v]:not([aria-pressed="true"]):not([disabled]):visible, {root} #rail .seg-ctl button[data-v]:not([aria-pressed="true"]):not([disabled]):visible, {root} .controls .seg-ctl button[data-v]:not([aria-pressed="true"]):not([disabled]):visible')
        if seg.count(): seg.first.click(); settle(pg)
        if not seg.count():   # a challenge answered with choices instead of settings (not the practice questions' choices)
            ch = pg.locator(f'{root} .choice:not([aria-pressed="true"]):not([data-ex]):not([disabled]):visible')
            if ch.count(): ch.first.click(); settle(pg)
        sub = pg.locator(f'{root} [data-act="submit"]:visible:not([disabled]), {root} #chSubmit:visible:not([disabled])')
        if sub.count(): sub.first.click(); settle(pg, 700)
        settle(pg, 500); s1 = snap(pg)
        r1 = stable(pg, lambda: pg.evaluate(f"(()=>{{const v=document.querySelector('{root}'); const r=v&&(v.querySelector('.rail')||v.querySelector('#rail')); return r?r.innerText.replace(/\\s+/g,' ').slice(0,600):'';}})()"))
        pg.reload(); settle(pg, 1200)
        ok(mode(pg, root) == 'challenge', f'{lid} challenge: mode after reload is {mode(pg, root)}')
        if not kit_only:
            s2 = snap(pg)
            r2 = stable(pg, lambda: pg.evaluate(f"(()=>{{const v=document.querySelector('{root}'); const r=v&&(v.querySelector('.rail')||v.querySelector('#rail')); return r?r.innerText.replace(/\\s+/g,' ').slice(0,600):'';}})()"))
            ok(s1 == s2, f'{lid} challenge: snapshot differs after reload\n  before {str(s1)[:300]}\n  after  {str(s2)[:300]}')
            ok(r1 == r2, f'{lid} challenge: rail text differs after reload\n  before {r1[:200]}\n  after  {r2[:200]}')
        # home offers to continue this lab
        pg.goto(URL + '#top'); settle(pg, 700)
        ok(pg.evaluate(f"!!document.querySelector('a[data-resume=\"{lid}\"]')"), f'{lid}: the home page has no continue link (a[data-resume])')
    for e in errs: fails.append('page error: ' + e[:200]); print('FAIL page error', e[:200])
    br.close()
print(f'kit_resume: {passes} passed, {len(fails)} failed')
sys.exit(1 if fails else 0)
