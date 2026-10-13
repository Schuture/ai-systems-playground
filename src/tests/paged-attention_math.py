"""H1/H2/H4/H10 of the owner's decisions of 2026-10-09, for one serving lab (the lab id is this file's name before "_math.py").

Every formula is typeset (A.ui.math → MathML, no TeX command shown as a word); "Show the math" is a few short blocks, each with a name
and its typeset formula, the numbers under it and the result, with a hairline (1px, the kit's line colour) between the blocks; no
plain-text formula of the kinds converted in this round is left in the source; the H4 and H10 rewrites are in. Checked at 1440 and 390,
in English and Chinese, in the guided tour, Scenarios and the challenge.

Run: python3 tests/<lab>_math.py   (the same file serves each lab: kv-basics_math.py, prefill-decode_math.py, …)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
LAB = pathlib.Path(__file__).name[:-len('_math.py')]
CFG = {
    'kv-basics': dict(px='kvb', gone=['n(n + 1) ÷ 2', 'n × (n + 1) ÷ 2', 'per token 2 × layers', '每个 token 占 2 × 层数', 'arithmetic = ', 'the step takes the longer of the two:'],
                      have=["TERMFX={gauss:()=>A.ui.math(", "class=\"kvb-fx\">${A.ui.math('\\\\text{KV cache per token} ="],
                      extra=[('kv-basics~r_9_1_1_8192', 4)]),
    'prefill-decode': dict(px='pd', gone=["tx('1 ÷ TPOT','1 ÷ TPOT')", 'decode time ÷ (output tokens − 1)', 'decode 时间 ÷（输出 token 数 − 1）', '= FLOP ÷ ${tx', 'the batch size divided by the time per output token'],
                           have=["\\\\text{TPOT} = \\\\frac{\\\\text{decode time}}{\\\\text{output tokens} - 1}", "\\\\text{throughput} = \\\\frac{\\\\text{batch size}}{\\\\text{TPOT}}"], extra=[]),
    'chunked-prefill': dict(px='ck', gone=['两头都吃亏', 'Both ends lose', '`; steps = ${fint(r.N)} ÷'], have=['chunk 太大、太小都不好。'], extra=[]),
    'continuous-batching': dict(px='cb', gone=['T<sub>', 'H<sub>', '规划中的 Lab“一个超长 prompt', 'The planned lab “Why does one huge prompt'],
                                have=["A.shell.labLink('chunked-prefill')", "\\\\frac{\\\\mu}{\\\\mu \\\\times H_B}"], extra=[]),
    'paged-attention': dict(px='pa', gone=['⌈${tx(\'length\',\'长度\')} ÷'], have=[], extra=[]),
    'prefix-caching': dict(px='px', gone=['`${tx(\'changed block\',\'改动的 block\')} = ⌊('], have=[], extra=[]),
    'serving-capstone': dict(px='sc', gone=['(b(b + 1) − a(a + 1)) ${tx(\'for the new tokens', "<b>${tx('One step’s time','一步的耗时')}</b> = max("], have=[], extra=[]),
}[LAB]
SRC = (P / 'labs' / f'{LAB}.js').read_text()
results = []
def check(name, ok, detail=''):
    results.append((name, bool(ok)))
    print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'  :: {str(detail)[:400]}'))

# the source: no plain-text formula of the kinds converted, the rewrites in
for g in CFG['gone']:
    check(f'source: “{g[:50]}” is gone (H1/H4/H10)', g not in SRC, g)
for h in CFG['have']:
    check(f'source: “{h[:60]}” is in', h in SRC, h)
check('source: Show the math is built from typeset blocks (mathBlock with A.ui.math)', 'function mathBlock(' in SRC and 'A.ui.math' in SRC)

PAGE = f'out/{LAB}-math.html'
subprocess.run([sys.executable, str(P / 'build.py'), PAGE, '--only', LAB], check=True, cwd=P, stdout=subprocess.DEVNULL)
# The built page declares no charset (the kit's head.html has none; the published site adds one), so Chromium guesses the encoding of a
# file:// page, and once in a while guesses wrong (a page error like “Invalid regular expression: /[ä¸€-é¿¿]$/”). This copy says UTF-8.
_pg = P / PAGE; _h = _pg.read_text()
if 'charset' not in _h[:1024]: _pg.write_text('<meta charset="utf-8">\n' + _h)
PX = CFG['px']
PROBE = """px=>{const b=document.getElementById(px+'-mathbody'); if(!b) return null; const d=b.closest('details'); if(d) d.open=true;
  const ms=[...b.querySelectorAll('math')], txt=ms.map(m=>m.textContent);
  const leak=txt.filter(x=>/\\\\|frac|\\bleft\\b|\\bright\\b|text\\{|sqrt|ldots/.test(x));
  const blocks=[...b.querySelectorAll('.'+px+'-mb')];
  const line=(()=>{const i=document.createElement('i'); i.style.color='var(--line)'; b.appendChild(i); const c=getComputedStyle(i).color; i.remove(); return c;})();
  // a block is set off from the one before by a 1px rule in the kit's line colour, or by a group heading (<px>-mhead) placed between them
  const head=x=>{const p=x.previousElementSibling; return !!(p&&p.classList.contains(px+'-mhead'));};
  const rule=x=>{const cs=getComputedStyle(x); return cs.borderTopStyle==='solid'&&Math.round(parseFloat(cs.borderTopWidth))===1&&cs.borderTopColor===line;};
  const ruled=blocks.slice(1).filter(x=>head(x)||rule(x)).length, headed=blocks.slice(1).filter(head).length;
  const lineOk=blocks.slice(1).filter(x=>!head(x)).every(rule);
  const names=[...b.querySelectorAll('tr.'+px+'-mname th')].map(x=>x.textContent.trim());
  const sub=b.querySelectorAll('sub,sup').length;
  const plain=[...b.querySelectorAll('td,th,p')].filter(e=>!e.querySelector('math')).map(e=>e.textContent).filter(t=>/[A-Za-z]_\\{?[A-Za-z]|\\\\frac|\\\\text/.test(t));
  const clip=ms.filter(m=>{const r=m.getBoundingClientRect(), s=m.closest('[class*="fx"]'); if(!s||r.height<1) return false; const q=s.getBoundingClientRect(); return r.top<q.top-1||r.bottom>q.bottom+1;}).length;
  return {n:ms.length, leak, blocks:blocks.length, ruled, headed, lineOk, names, sub, plain, clip, open:!!(d&&d.open)};}"""

with sync_playwright() as p:
    br = p.chromium.launch()
    for w, lang in [(1440, 'en'), (1440, 'zh'), (390, 'zh'), (390, 'en')]:
        pg = br.new_page(viewport={'width': w, 'height': 900 if w > 500 else 844})
        errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto('file://' + str(P / PAGE)); pg.wait_for_timeout(250)
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(60)
        pg.evaluate(f"()=>{{location.hash='{LAB}'}}"); pg.wait_for_timeout(600)
        nparts = pg.evaluate("()=>new Set([...document.querySelectorAll('#view-dyn aside.rail [data-go]')].map(e=>e.dataset.go)).size")
        states = [('guided', k) for k in range(nparts)] + [('sandbox', None), ('challenge', None)]
        seen_blocks = 0
        for mode, k in states:
            if k is None or k == 0:
                pg.evaluate("m=>{const b=[...document.querySelectorAll('#view-dyn .modes button[data-mode=\"'+m+'\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}", mode); pg.wait_for_timeout(300)
            if k is not None:
                pg.evaluate("k=>{const e=document.querySelector('#view-dyn aside.rail [data-go=\"'+k+'\"]'); if(e) e.click();}", k); pg.wait_for_timeout(500)
            r = pg.evaluate(PROBE, PX)
            tag = f'{w} {lang} {mode}{"" if k is None else " part " + str(k + 1)}'
            if not r or not r['open']:
                check(f'{tag}: Show the math opens', False, r); continue
            if r['blocks'] == 0:
                continue   # hidden until a prediction, or before a run: nothing to typeset yet
            seen_blocks += 1
            check(f'{tag}: Show the math is typeset (MathML), no TeX command shown as a word, nothing clipped', r['n'] >= 2 and not r['leak'] and not r['plain'] and r['sub'] == 0 and r['clip'] == 0, r)
            check(f'{tag}: Show the math is {r["blocks"]} blocks, each set off from the one before by a 1px rule in the kit’s line colour (or a group heading), every formula named', r['ruled'] == r['blocks'] - 1 and r['lineOk'] and r['names'] and all(r['names']), r)
        check(f'{w} {lang}: Show the math was seen in blocks in some state', seen_blocks > 0, seen_blocks)
        for h, nb in CFG['extra']:
            pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(150); pg.evaluate(f"()=>{{location.hash='{h}'}}"); pg.wait_for_timeout(500)
            r = pg.evaluate(PROBE, PX)
            check(f'{w} {lang} {h}: {nb} blocks, typeset, ruled', r and r['blocks'] == nb and r['ruled'] == nb - 1 and not r['leak'] and r['n'] >= 10, r)
        check(f'{w} {lang}: no page errors', not errs, errs[:2])
        pg.close()
    br.close()

bad = [n for n, ok in results if not ok]
print(f'{len(results)} checks, {len(bad)} failed')
if bad:
    print('FAILED:', bad[:10]); sys.exit(1)
