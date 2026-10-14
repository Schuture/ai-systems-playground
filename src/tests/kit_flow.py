"""The lab's flow (the owner's decision of 2026-10-11 19:28 UTC): every lab shows a contents list of its steps on the left (知识梳理 when
the lab has a primer, 分步引导, 应用场景, 自由调参 when it has a settings panel, 挑战题) and the stage or the primer on the right; no
fidelity, primer or share buttons and no mode description on top; the fidelity notes without "what is modeled" close the page.
Each primer opens first; a lab without one shows “建议先读” pointing to the primer that starts its series (2026-10-12).
Usage: python3 tests/kit_flow.py [lab ...]   (default: every live lab, in Chinese and English)"""
import sys, json, pathlib
from playwright.sync_api import sync_playwright
PAGE = pathlib.Path(__file__).resolve().parent.parent / 'ai-systems-playground.html'
ok = 0; bad = []
def check(name, cond, info=''):
    global ok
    if cond: ok += 1
    else: bad.append(name); print('FAIL', name, '::', str(info)[:300])
STATE = """(()=>{const v=document.querySelector('#view-dyn:not([hidden])')||document.querySelector('#view-lab:not([hidden])');
  const nav=v.querySelector('[data-flow-nav]'), bench=nav&&nav.closest('.bench.flowed'), en=v.querySelector('[data-endnotes]'), txt=en?en.textContent:'';
  return {nav:!!bench, flow:bench&&bench.dataset.flow, en:!!en, last:!!en&&!en.nextElementSibling, modeled:/建模了什么|Modeled|模型里算了什么/.test(txt),
    old:!!v.querySelector('[data-pop$="-fid"],[data-pop$="-pri"],[data-share],#fidBtn,#priBtn,#shareBtn,.mode-desc'),
    ctl:!!(bench&&bench.querySelector(':scope>.controls')), free:!!(nav&&nav.querySelector('[data-flow=free]'))}})()"""
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900}); pg.add_init_script('window.__aspPrimer=true')   # the real flow, not the older tests' compat mode
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto('file://' + str(PAGE)); pg.wait_for_timeout(500)
    ids = sys.argv[1:] or pg.evaluate("ASP.catalog.LABS.filter(l=>ASP.catalog.labStatus(l)==='live').map(l=>l.id)")
    for lang in ['zh', 'en']:
        pg.click(f'.langsw button[data-lang={lang}]')
        for id in ids:
            n0 = len(errs); pg.evaluate(f"location.hash='{id}'"); pg.wait_for_timeout(250); r = pg.evaluate(STATE); t = f'{lang} {id}'
            check(f'{t}: the list of steps sits in the bench', r['nav'], r)
            check(f'{t}: the notes close the page, without “what is modeled”', r['en'] and r['last'] and not r['modeled'], r)
            check(f'{t}: no fidelity, primer or share button and no mode description', not r['old'], r)
            check(f'{t}: 自由调参 is listed exactly when the lab has a settings panel', r['ctl'] == r['free'], r)
            for f in ['scen', 'free', 'chal', 'guided']:
                loc = pg.locator(f'[data-flow-nav] button[data-flow={f}]:visible')
                if not loc.count(): continue
                n1 = len(errs); loc.first.click(timeout=4000); pg.wait_for_timeout(120)
                fl = pg.evaluate("[...document.querySelectorAll('.bench.flowed')].find(b=>b.offsetParent).dataset.flow")
                vis = pg.evaluate("""(()=>{const b=[...document.querySelectorAll('.bench.flowed')].find(b=>b.offsetParent), v=s=>{const e=b.querySelector(':scope>'+s); return !!(e&&e.getClientRects().length);};
                  return {rail:v('.rail'),ctl:v('.controls'),stage:v('.stage')}})()""")
                check(f'{t}: {f} shows its step', fl == f and not errs[n1:], (fl, errs[n1:][:1]))
                check(f'{t}: {f} shows the stage, and the settings panel only in 自由调参', vis['stage'] and vis['ctl'] == (f == 'free') and (vis['rail'] != (f == 'free')), vis)
            check(f'{t}: no page error', not errs[n0:], errs[n0:][:2])
    # the primers: opt in (automated browsers otherwise open a lab on its own mode)
    PRIMER_LABS = {'architecture': 10, 'backprop': 4, 'roofline': 4, 'prefill-decode': 4, 'eval-ci': 4,
                   'training-memory': 4, 'parallelism': 4, 'sft': 4, 'reward-model': 4, 'ppo': 3, 'embeddings': 4, 'agent-anatomy': 3,
                   'number-formats': 4, 'dpo': 4, 'spec-decoding': 4}   # lab → its sections
    pg2 = b.new_page(viewport={'width': 1440, 'height': 900}); pg2.add_init_script('window.__aspPrimer=true'); e2 = []; pg2.on('pageerror', lambda e: e2.append(str(e)))
    pg2.goto('file://' + str(PAGE)); pg2.wait_for_timeout(500); pg2.click('.langsw button[data-lang=zh]')
    for lab, n in PRIMER_LABS.items():
        if ids and lab not in ids and sys.argv[1:]: continue
        n0 = len(e2); pg2.evaluate(f"location.hash='{lab}'"); pg2.wait_for_timeout(900)
        r = pg2.evaluate("""(()=>{const b=document.querySelector('#view-dyn .bench.flowed'), s=[...b.querySelectorAll('.pr-sec')];
          return {flow:b.dataset.flow, n:s.length, k:s.map(x=>x.querySelector('.pr-k').textContent), toc:b.querySelectorAll('[data-flow-nav] .fl-sub a').length,
            figs:b.querySelectorAll('.pr-fig').length, inert:[...b.querySelectorAll('.pr-frames')].every(f=>f.inert), stage:!!b.querySelector(':scope>.stage').getClientRects().length,
            see:!!b.querySelector('[data-flow-nav] .fl-see')}})()""")
        check(f'{lab} opens on 知识梳理 with {n} sections numbered k/{n}', r['flow'] == 'primer' and r['n'] == n and r['k'][-1] == f'{n}/{n}', r)
        check(f'{lab}: the list under 知识梳理 names the {n} sections', r['toc'] == n, r)
        check(f'{lab}: the figures are a display (inert) and the stage is hidden', r['inert'] and not r['stage'], r)
        check(f'{lab}: a lab with its own primer points to no other primer', not r['see'], r)
        if lab == 'architecture':
            check('architecture: one figure or more per section', r['figs'] >= 10, r)
            pg2.evaluate("window.scrollTo(0, document.querySelector('.pr-sec[data-psec=\"6\"]').getBoundingClientRect().top+scrollY-80)"); pg2.wait_for_timeout(500)
            check('no mini result bar over the primer', pg2.evaluate("document.getElementById('dock').hidden"))
            check('the list marks the section being read', pg2.evaluate("(document.querySelector('[data-flow-nav] .fl-sub a.on')||{}).dataset.psec") == '6')
        pg2.locator('.pr-end [data-flow-go=guided]:visible').first.click(); pg2.wait_for_timeout(500)
        check(f'{lab}: “下一步：分步引导” opens the guided parts', pg2.evaluate("document.querySelector('#view-dyn .bench.flowed').dataset.flow") == 'guided')
        check(f'{lab}: the primer: no page error', not e2[n0:], e2[n0:][:2])
    # a lab without its own primer points to the one that starts its series, once that primer exists; the link opens it on the primer
    for lab, to in [('kv-basics', 'architecture'), ('optimizers', 'backprop'), ('flash-attention', 'roofline'), ('paged-attention', 'prefill-decode'), ('pass-k', 'eval-ci'),
                    ('checkpointing', 'training-memory'), ('model-parallel', 'parallelism'), ('lora', 'sft'), ('softmax-numerics', 'number-formats'), ('grpo', 'ppo'), ('vector-index', 'embeddings'), ('context-eng', 'agent-anatomy')]:
        pg2.evaluate(f"location.hash='{lab}'"); pg2.wait_for_timeout(400)
        a = pg2.evaluate("(()=>{const a=document.querySelector('#view-dyn:not([hidden]) [data-flow-nav] .fl-see a, #view-lab:not([hidden]) [data-flow-nav] .fl-see a'); return a&&{to:a.dataset.seePrimer,t:a.closest('.fl-see').textContent}})()")
        check(f'{lab}: “建议先读” points to the {to} primer', bool(a) and a['to'] == to and '知识梳理' in a['t'], a)
    pg2.evaluate("location.hash='architecture'"); pg2.wait_for_timeout(500); pg2.locator('[data-flow-nav] button[data-flow=guided]:visible').first.click(); pg2.wait_for_timeout(200)
    pg2.evaluate("location.hash='kv-basics'"); pg2.wait_for_timeout(500); pg2.locator('.fl-see a:visible').first.click(); pg2.wait_for_timeout(700)
    check('the “建议先读” link opens that lab on its primer, even after the reader left it', pg2.evaluate("location.hash") == '#architecture' and pg2.evaluate("document.querySelector('#view-dyn .bench.flowed').dataset.flow") == 'primer')
    check('the primers: no page error', not e2, e2[:2])
    b.close()
print(f'{ok} passed, {len(bad)} failed'); sys.exit(1 if bad else 0)
