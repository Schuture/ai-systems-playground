"""Dump every piece of text a learner meets in the serving capstone, in reading order, in English and Chinese (CONTRACT §0: export the
whole text in reading order and read it before reporting). tests/numdense.py reads the result (R6.20).

python3 tests/serving-capstone_copy_deck.py  ->  out/serving-capstone-copy/en.md and out/serving-capstone-copy/zh.md

Order: lab header with its tooltips and both popovers; each guided part (card before and after its action, the stage, the tooltips in view
that were not shown before, the result for every option, the prediction's verdict for a wrong and the right pick, a request's trace);
Scenarios for every scenario; the Settings panel; the challenge before and after a run (every feedback tier, the chart, the plan, the
practice links); the card "Can you use it on a new case?" (every question, a verdict for every option); Show the math; Connect with the
drill's answer; the mini result bar. A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'serving-capstone-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/serving-capstone-deck.html', '--only', 'serving-capstone'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'wrapped-serving-capstone-deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'serving-capstone-deck.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-serving-capstone-deck.html')
clean = lambda s: '\n'.join(' '.join(l.split()) for l in s.replace('\u2011', '-').replace('\u202f', ' ').replace('\xa0', ' ').split('\n') if l.strip()) if s else ''


def txt(pg, sel):
    el = pg.query_selector(sel)
    return clean(el.inner_text()) if el and el.is_visible() else ''


class Tips:
    def __init__(self): self.seen = set()

    def grab(self, pg, root):
        out = []
        for t in pg.evaluate(f"()=>[...document.querySelectorAll('{root} [data-tipx]')].filter(e=>e.getClientRects().length).map(e=>[(e.innerText||e.getAttribute('aria-label')||'').slice(0,60),e.getAttribute('data-tipx')])"):
            label, tip = clean(t[0]), clean(t[1])
            if tip and tip not in self.seen:
                self.seen.add(tip); out.append(f'- {label} ⟶ {tip}')
        return out


def stage(pg):
    parts = ['Head: ' + txt(pg, '#sc-st-title') + ' | ' + txt(pg, '#sc-st-sub') + ' | status: ' + txt(pg, '#sc-status') + ' | comparison (a metric readout): ' + txt(pg, '#sc-m2d'),
             'Settings pills: ' + txt(pg, '#sc-pills') + ' | kinds: ' + txt(pg, '#sc-kinds'),
             'Queue: ' + txt(pg, '#sc-q'), 'GPU step: ' + txt(pg, '#sc-gpu'), 'Tokens out: ' + txt(pg, '#sc-out'),
             'Clock: ' + txt(pg, '#sc-clock'), 'Timeline: ' + txt(pg, '#sc-tlcap') + ' | stall label: ' + txt(pg, '#sc-stall'),
             'Timeline legend: ' + txt(pg, '#sc-tlleg'), 'Memory: ' + txt(pg, '#sc-memt') + ' | ' + txt(pg, '#sc-memv') + ' | labels: ' + txt(pg, '#sc-mlab') + ' | axis: ' + txt(pg, '#sc-maxis'),
             'Memory legend: ' + txt(pg, '#sc-mleg'), 'Trace: ' + txt(pg, '#sc-trace') + ' | trace card: ' + txt(pg, '#sc-tcard'), 'Table: ' + txt(pg, '#sc-mtcap'),
             'Scope: ' + txt(pg, '#sc-scope')]
    if pg.is_visible('#sc-wc'):   # part 7: the scheduler's code with this step's values (a trace, like Show the math)
        parts.append('Code panel: ' + txt(pg, '#sc-wct') + ' | legend: ' + txt(pg, '#sc-wleg'))
        parts += [f'  {r}' for r in pg.evaluate("()=>[...document.querySelectorAll('#sc-wcb .sc-wl')].map(l=>l.dataset.i+' ['+(l.className.replace('sc-wl','').trim()||'plain')+'] '+l.querySelector('.sc-wx').textContent+(l.querySelector('.sc-wv').textContent?'  ⟶ '+l.querySelector('.sc-wv').textContent:''))")]
    rows = pg.evaluate("()=>[...document.querySelectorAll('#sc-mt tr')].map(t=>[...t.children].map(c=>c.innerText.trim()).join(' | '))")
    return '\n'.join(parts + [f'- {clean(r)}' for r in rows])


def calm(pg):
    pg.wait_for_timeout(250)


def end(pg):
    pg.evaluate("()=>{const b=document.querySelector('[data-act=toend]'); if(b){b.click(); return;} const i=document.querySelector('#sc-scrub'); i.value=1000; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(250)


def click(pg, sel): pg.evaluate("s=>document.querySelector(s).click()", sel); calm(pg)


def exits(pg):
    out = []
    card = "[...document.querySelectorAll('[data-exits]')].find(e=>e.offsetParent!==null)"
    for t in range(3):
        pg.evaluate(f"()=>{card}.querySelector('[data-ex=\"tab\"][data-i=\"{t}\"]').click()"); pg.wait_for_timeout(150)
        name = clean(pg.evaluate(f"()=>{card}.querySelector('[data-ex=\"tab\"][data-i=\"{t}\"]').textContent"))
        if t == 2:
            out += [f'## {name}', clean(pg.evaluate(f"()=>{card}.querySelector('.ex-body').innerText"))]; continue
        nq = pg.evaluate(f"()=>{card}.querySelectorAll('.ex-qs .pb-dot').length||1")
        for j in range(nq):
            if j: pg.evaluate(f"()=>{card}.querySelector('.ex-qs .pb-dot[data-j=\"{j}\"]').click()"); pg.wait_for_timeout(150)
            q = clean(pg.evaluate(f"()=>{{const e={card}.querySelector('.ex-body > p > b'); return e?e.innerText:''}}"))
            opts = pg.evaluate(f"()=>[...{card}.querySelectorAll('.ex-opts .choice')].map(b=>({{v:+b.dataset.v, t:b.innerText}}))")
            out += [f'## {name}, question {j + 1}', q] + [f'- ({o["v"]}) {clean(o["t"])}' for o in opts]
            for o in opts:
                pg.evaluate(f"()=>{card}.querySelector('.ex-opts .choice[data-v=\"{o['v']}\"]').click()"); pg.wait_for_timeout(150)
                r = pg.evaluate(f"()=>{{const e={card}.querySelector('.ex-res'); return e?{{ok:e.classList.contains('ok'),t:e.innerText}}:null}}")
                if r: out += [f'### After picking ({o["v"]})', clean(r['t'])]
    return out


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ('en', 'zh'):
        pg = b.new_page(viewport={'width': 1440, 'height': 1000})
        errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(150)
        pg.evaluate("()=>{location.hash='serving-capstone'}"); pg.wait_for_timeout(700)
        pg.add_style_tag(content='.sc-stage{max-height:none!important;overflow:visible!important;position:static!important}')
        T = Tips(); Lines = ['# Lab header', txt(pg, '.labhead'), '## Header tooltips'] + T.grab(pg, '.labhead')
        for sel in pg.query_selector_all('.labhead [data-pop]'):
            pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(150)
            Lines += [f'## Popover: {clean(sel.inner_text())}', clean(pg.evaluate("(id)=>document.getElementById(id).innerText", pid)), '## Popover tooltips'] + T.grab(pg, f'#{pid}')
            sel.click(); pg.wait_for_timeout(100)
        ids = pg.evaluate("()=>window.__sc.consts.STEP_IDS")
        for k, sid in enumerate(ids):
            click(pg, f'#sc-rail [data-go="{k}"]'); end(pg)
            Lines += [f'# Guided part {k + 1} ({sid})', '## Card', txt(pg, '#sc-rail .step.current'), '## Part bar (phones)', txt(pg, '#sc-rail .pbar') or clean(pg.evaluate("()=>document.querySelector('#sc-rail .pbar .pb-t').textContent")), '## Stage', stage(pg), '## Tooltips in view (new)'] + T.grab(pg, '#sc-app .bench')
            if sid == 'mission':
                pg.evaluate("()=>{const i=document.querySelector('#sc-scrub'); i.value=Math.round(3.3/12*1000); i.dispatchEvent(new Event('input',{bubbles:true}));}"); calm(pg)
                Lines += ['## Stage during the stall', stage(pg), '## Tooltips in view during the stall (new)'] + T.grab(pg, '#sc-app .bench')
                end(pg); Lines += ['## Card after the run', txt(pg, '#sc-rail .step.current')]
            if sid == 'predict':
                click(pg, '#sc-rail [data-pred="a"]'); Lines += ['## After a wrong pick', txt(pg, '#sc-rail [data-obs]')]
                click(pg, '#sc-rail [data-pred="b"]'); Lines += ['## After the right pick', txt(pg, '#sc-rail [data-obs]')]
                click(pg, '#sc-rail .step.current [data-nav=showme]'); Lines += ['## Stage after Show me', stage(pg), '## Tooltips in view after Show me (new)'] + T.grab(pg, '#sc-app .bench')
            opts = {'budget': ['8192', '2048', '512', '128'], 'cache': ['true'], 'cbudget': ['3072', '6144', 'null'], 'cap': ['8', '16', '64', '128']}.get(sid, [])
            for v in opts:
                click(pg, f'#sc-rail [data-lever="{v}"]'); Lines += [f'## Status line while option {v} plays (a metric readout)', txt(pg, '#sc-rail [data-obs]')]; end(pg)
                Lines += [f'## Result for option {v}', txt(pg, '#sc-rail [data-obs]')]
            if opts:
                Lines += ['## Card after the options', txt(pg, '#sc-rail .step.current'), '## Stage after the options', stage(pg), '## Tooltips in view after the options (new)'] + T.grab(pg, '#sc-app .bench')
            if sid == 'cap':
                click(pg, '#sc-ic-pol button[data-v="static"]'); end(pg); Lines += ['## Result for static batching', txt(pg, '#sc-rail [data-obs]')]
            if sid == 'code':
                for mid in pg.evaluate("()=>window.__sc.moments().map(m=>m.id)"):
                    click(pg, f'#sc-rail [data-moment="{mid}"]'); calm(pg)
                    Lines += [f'## Moment “{mid}”: the card', txt(pg, '#sc-rail [data-obs]'), f'## Moment “{mid}”: the stage', stage(pg), f'## Moment “{mid}”: tooltips in view (new)'] + T.grab(pg, '#sc-app .bench')
                    click(pg, '#sc-play [data-act="fwd"]')
                    Lines += [f'## Moment “{mid}” after ▸ (the step has ended)', stage(pg), f'## Moment “{mid}” after ▸: tooltips in view (new)'] + T.grab(pg, '#sc-app .bench')
                click(pg, '#sc-play [data-act="fwd"]'); Lines += ['## One more ▸ (the next step)', stage(pg)]
                Lines += ['## Card after the moments', txt(pg, '#sc-rail .step.current')]
            if sid == 'budget':
                click(pg, '#sc-rail [data-lever="2048"]'); end(pg)
                click(pg, '#sc-rail [data-obs] [data-sel]'); Lines += ['## A request traced (see why): the stage readout', txt(pg, '#sc-tcard') or txt(pg, '#sc-trace'), '## Show the math with it', clean(pg.evaluate("()=>document.querySelector('#sc-mathbody').innerText"))]
        # Scenarios
        click(pg, '#sc-app .modes button[data-mode="sandbox"]:not([data-flow=free])')
        Lines += ['# Scenarios', txt(pg, '#sc-rail .railcard')]
        for i in range(3):
            click(pg, f'#sc-rail [data-preset="{i}"]'); end(pg); Lines += [f'## Scenario {i + 1}', txt(pg, '#sc-insight')]
        pg.locator('#sc-ctl-budget button[data-v="2048"]').click(); calm(pg); end(pg); Lines += ['## Scenario 3 with budget 2,048', txt(pg, '#sc-insight')]
        Lines += ['# Settings panel', txt(pg, '#sc-controls'), '## Settings tooltips (new)'] + T.grab(pg, '#sc-controls')
        # the challenge
        click(pg, '#sc-app .modes button[data-mode="challenge"]')
        Lines += ['# Challenge', '## Before a run', txt(pg, '#sc-rail .sc-chcard'), '## Stage (initial setup)', stage(pg), '## Tooltips in view (new)'] + T.grab(pg, '#sc-app .bench')
        for plan in ({'cap': '16', 'budget': '2048', 'cache': 'true'}, {'cap': '128', 'budget': '128'}, {'cap': '64', 'budget': '8192'}, {'cap': '64', 'budget': '2048', 'cache': 'false'}, {'cap': '32', 'budget': '2048', 'cache': 'true'}, {'cap': '64'}):
            for k_, v_ in plan.items():
                pg.evaluate(f"()=>document.querySelector('#sc-rail .seg-ctl[data-key=\"ch-{k_}\"] button[data-v=\"{v_}\"]').click()"); pg.wait_for_timeout(100)
            click(pg, '#sc-rail [data-act=run]'); end(pg)
            pg.wait_for_function("()=>window.__sc.state.chTier!==null", timeout=20000); calm(pg)
            if pg.query_selector('#sc-ch-res [data-act=reveal]'): click(pg, '#sc-ch-res [data-act=reveal]')
            res = pg.evaluate("()=>{const r=document.querySelector('#sc-ch-res').cloneNode(true); const pl=r.querySelector('.sc-plan'); const pt=pl?pl.innerText:''; if(pl) pl.remove(); const jb=r.querySelector('.sc-jobs'); const jt=jb?jb.innerText:''; if(jb) jb.remove(); document.body.appendChild(r); const t=r.innerText; r.remove(); return [t,pt,jt]}")
            Lines += [f'## Result for {plan}', clean(res[0]), f'## The plan to copy for {plan} (a metric summary for notes, not an explanation)', clean(res[1])]
            if plan.get('cap') == '64' and len(plan) == 1: Lines += ['## Practice links', clean(res[2])]
        Lines += ['# Can you use it on a new case?'] + exits(pg)
        pg.evaluate("()=>{const d=document.querySelector('#sc-math'); d.open=true}"); calm(pg)
        Lines += ['# Show the math (challenge run)', clean(pg.evaluate("()=>document.querySelector('#sc-mathbody').innerText")), '## Math tooltips (new)'] + T.grab(pg, '#sc-math')
        click(pg, '#sc-app [data-act="drill-reveal"]')
        Lines += ['# Connect', txt(pg, '.connect'), '## Connect tooltips (new)'] + T.grab(pg, '.connect')
        pg.set_viewport_size({'width': 390, 'height': 844}); pg.wait_for_timeout(300)
        pg.evaluate("()=>{const r=document.querySelector('#sc-mem'); r.scrollIntoView({block:'start'}); window.scrollBy(0,500)}"); pg.wait_for_timeout(500)
        Lines += ['# Mini result bar (390 px)', txt(pg, '#dock') or '(not shown)']
        if errs: Lines += ['# PAGE ERRORS'] + errs
        (OUT / f'{lang}.md').write_text('\n'.join(x for x in Lines if x is not None) + '\n')
        print(lang, len(Lines), 'lines', 'errors:', errs)
        pg.close()
    b.close()
