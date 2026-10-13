"""Round 3, R07 and R08 (the owner's message of 2026-10-10, Codex review): the guided parts' questions and the optional practice.

R08, phones (390 x 844, m2 labs): a part with an open question puts "Answer below ↓" in the part bar, which brings the card's question
  and its options on screen; Next reads "Skip question" (ghost) while the question is open; a part passed without an answer keeps its
  number, marked skipped ("第 1 部分：跳过了"), and gets ✓ only once answered. The same in the KV lab, whose part 2 Next waits for the
  prediction. Desktop (1440 x 900): the step list marks the skipped part the same way and the card's Next reads "Skip question".
R07: the kit's practice card is headed "可选进阶" (since 2026-10-12, with no line telling the reader it may be skipped), and its next button is "这组下一题", never the
  same name as another button on screen. Architecture: after the three core questions are right, the challenge card says the core is done
  and offers the next lab as its main button; the deeper six are marked optional. KV: once the challenge is passed, the practice card
  opens with the same summary and the next lab.
Run: python3 tests/kit_r3_practice.py"""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/kit-r3-practice.html', '--only', 'architecture,transformer,prefill-decode'], check=True, cwd=P, stdout=subprocess.DEVNULL)   # prefill-decode: the KV lab's next lab
(P / 'out' / 'kit-r3-practice-w.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'kit-r3-practice.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'kit-r3-practice-w.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


def open_lab(b, w, h, lang, lab, errs):
    pg = b.new_page(viewport={'width': w, 'height': h})
    pg.set_default_timeout(8000)
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{lab}'}}"); pg.wait_for_timeout(800)
    return pg


def click(pg, sel, ms=450):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(ms)


BAR = """r=>{const b=document.querySelector(r+' .pbar'); if(!b) return null; const nx=b.querySelector('[data-nav="next"]'), go=b.querySelector('[data-pbgo]');
  return {dots:[...b.querySelectorAll('.pb-dot')].map(d=>({t:d.textContent, c:d.className, a:d.getAttribute('aria-label')})),
    next:nx?{t:nx.textContent.trim(), ghost:nx.classList.contains('ghost'), dis:nx.disabled}:null, go:go?{t:go.textContent.trim(), k:go.dataset.pbgo, ghost:go.classList.contains('ghost')}:null};}"""
LIST = """r=>[...document.querySelectorAll(r+' .steps>.step')].map(s=>({c:s.className, num:s.querySelector('.num').textContent, a:s.querySelector('.step-h').getAttribute('aria-label'),
  tag:(s.querySelector('.sk-tag')||{}).textContent||''}))"""
CARDNEXT = """r=>{const b=document.querySelector(r+' .steps .step.current .stepnav [data-nav="next"]'); return b?{t:b.textContent.trim(), ghost:b.classList.contains('ghost'), dis:b.disabled}:null;}"""


def Z(lang, en, zh): return en if lang == 'en' else zh


with sync_playwright() as p:
    b = p.chromium.launch()
    # ---------- R08 on a phone: architecture part 1 (a prediction), in both languages ----------
    for lang in ('zh', 'en'):
        tag = f'architecture 390 {lang}'; errs = []
        pg = open_lab(b, 390, 844, lang, 'architecture', errs); R = '#ar-rail'
        bar = pg.evaluate(BAR, R)
        check(f'{tag}: part 1 has a question, so the bar’s main action is “{Z(lang, "Answer below ↓", "去作答 ↓")}”',
              bar and bar['go'] and bar['go']['k'] == 'q' and bar['go']['t'] == Z(lang, 'Answer below ↓', '去作答 ↓') and not bar['go']['ghost'], bar and bar['go'])
        check(f'{tag}: while the question is open, Next reads “{Z(lang, "Skip question", "跳过本题")} ▶” (ghost)',
              bar and bar['next'] and bar['next']['t'] == Z(lang, 'Skip question ▶', '跳过本题 ▶') and bar['next']['ghost'] and not bar['next']['dis'], bar and bar['next'])
        row = pg.evaluate("()=>{const r=document.querySelector('.pbar .pb-row'), d=[...r.querySelectorAll('.pb-dot,.pb-nav .btn')].map(e=>e.getBoundingClientRect().top); return Math.max(...d)-Math.min(...d);}")
        check(f'{tag}: the dots, Back and Skip stay on one row', row < 8, row)
        # the task, then where to answer: the question and its options come on screen, below the top bar
        pg.click('.pbar [data-pbgo]'); pg.wait_for_timeout(1000)
        vis = pg.evaluate("""()=>{const g=document.querySelector('#ar-rail .step.current .ar-pred'), q=g.previousElementSibling, bar=document.querySelector('.topbar').getBoundingClientRect().bottom,
            dk=document.getElementById('dock'), safe=bar+(dk&&!dk.hidden?dk.getBoundingClientRect().height:0), r=g.getBoundingClientRect(), rq=q.getBoundingClientRect();
            return {q:Math.round(rq.top), top:Math.round(r.top), bottom:Math.round(r.bottom), safe:Math.round(safe), vh:innerHeight, focus:!!document.activeElement&&document.activeElement.hasAttribute('data-pred')};}""")
        check(f'{tag}: “Answer below” brings the question line and all its options on screen, the first option focused',
              vis['q'] >= vis['safe'] - 2 and vis['bottom'] <= vis['vh'] and vis['focus'], vis)
        # skip it: part 1 keeps its number, marked skipped
        pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(200)
        click(pg, '.pbar [data-nav="next"]', 700)
        bar = pg.evaluate(BAR, R)
        d0 = bar['dots'][0]
        check(f'{tag}: skipped part 1 has no ✓ in the bar: its number, dashed, labelled skipped',
              d0['t'] == '1' and 'skip' in d0['c'] and (d0['a'] or '').startswith(Z(lang, 'Part 1: skipped', '第 1 部分：跳过了')), d0)
        ls = pg.evaluate(LIST, R)
        check(f'{tag}: the step list marks part 1 skipped too', 'skipped' in ls[0]['c'] and ls[0]['num'] == '1', ls[0])
        check(f'{tag}: part 2 has no question: Next reads Next again, no “Answer below”',
              bar['next']['t'] == Z(lang, 'Next ▶', '下一部分 ▶') and not bar['next']['ghost'] and bar['go'] is None, bar)
        # back to part 1, answer: ✓
        click(pg, '.pbar [data-go="0"]')
        click(pg, '#ar-rail .step.current [data-pred="attn"]')
        bar = pg.evaluate(BAR, R)
        check(f'{tag}: once answered, Next reads Next and the bar has no “Answer below”', bar['next']['t'] == Z(lang, 'Next ▶', '下一部分 ▶') and not bar['next']['ghost'] and bar['go'] is None, bar)
        click(pg, '.pbar [data-nav="next"]', 700)
        d0 = pg.evaluate(BAR, R)['dots'][0]
        check(f'{tag}: answered part 1 shows ✓', d0['t'] == '✓' and 'done' in d0['c'] and 'skip' not in d0['c'], d0)
        ans = pg.evaluate("()=>(window.__prog.get('architecture')||{}).ans")
        check(f'{tag}: the kit keeps the answers per part (ans)', isinstance(ans, list) and ans[:2] == [True, True] and len(ans) == 6, ans)
        # the mask part: a control and no question: “Try it below”, the main action while Next waits
        click(pg, '.pbar [data-go="4"]', 700)
        bar = pg.evaluate(BAR, R)
        check(f'{tag}: the mask part (a switch, no question) offers “{Z(lang, "Try it below ↓", "去操作 ↓")}” while Next waits',
              bar['go'] and bar['go']['k'] == 'c' and not bar['go']['ghost'] and bar['next']['dis'] and bar['next']['t'] == Z(lang, 'Next ▶', '下一部分 ▶'), bar)
        check(f'{tag}: parts 3 and 4, jumped over, are marked skipped', [d['t'] for d in bar['dots'][2:4]] == ['3', '4'] and all('skip' in d['c'] for d in bar['dots'][2:4]), bar['dots'])
        if lang == 'zh':
            pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(200)
            pg.screenshot(path=str(P / 'shots' / 'kit' / 'r3-practice-architecture-390-skipped.png'))
        check(f'{tag}: no page errors', not errs, errs[:3])
        pg.close()

    # ---------- the marks survive a reload (PROG keeps ans; the kit keeps nothing in automated browsers unless the page opts in) ----------
    ctx = b.new_context(viewport={'width': 390, 'height': 844}); ctx.add_init_script('window.__aspResume=true')
    pg = ctx.new_page(); errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL + '#top'); pg.wait_for_timeout(400); pg.evaluate("window.__prog&&window.__prog.clear()")
    pg.goto(URL + '#architecture'); pg.reload(); pg.wait_for_timeout(900)
    click(pg, '.pbar [data-nav="next"]', 900)
    pg.reload(); pg.wait_for_timeout(1000)
    d = pg.evaluate(BAR, '#ar-rail')['dots']
    check('architecture 390 reload: back on part 2, part 1 still marked skipped', 'on' in d[1]['c'] and d[0]['t'] == '1' and 'skip' in d[0]['c'], d[:2])
    click(pg, '.pbar [data-go="0"]'); click(pg, '#ar-rail .step.current [data-pred="mlp"]'); click(pg, '.pbar [data-nav="next"]', 900)
    pg.reload(); pg.wait_for_timeout(1000)
    d = pg.evaluate(BAR, '#ar-rail')['dots']
    check('architecture 390 reload: answered (even wrongly), part 1 shows ✓', d[0]['t'] == '✓' and 'done' in d[0]['c'], d[:2])
    check('architecture 390 reload: no page errors', not errs, errs[:3])
    ctx.close()

    # ---------- R08 in the KV lab (its own rail): part 2's prediction ----------
    for w, h in ((390, 844), (1440, 900)):
        tag = f'kv-cache {w} zh'; errs = []
        pg = open_lab(b, w, h, 'zh', 'kv-cache', errs); R = '#view-lab'
        click(pg, '#rail .steps [data-go="1"]', 600)
        bar = pg.evaluate(BAR, R); cn = pg.evaluate(CARDNEXT, R)
        check(f'{tag}: part 2 (predict) puts “去作答 ↓” in the bar', bar['go'] and bar['go']['t'] == '去作答 ↓', bar)
        check(f'{tag}: the card’s Next waits for the prediction and keeps its name', cn and cn['dis'] and cn['t'] == '下一部分', cn)
        check(f'{tag}: so does the bar’s Next (never a greyed-out “Skip question”)', bar['next'] and bar['next']['dis'] and bar['next']['t'] == '下一部分 ▶', bar['next'])
        click(pg, '#rail .steps .step.current [data-nav="skip"]', 600)
        ls = pg.evaluate(LIST, R)
        check(f'{tag}: part 1 seen (no question) ✓; part 2 skipped: its number, “跳过了”, labelled',
              ls[0]['num'] == '✓' and 'skipped' in ls[1]['c'] and ls[1]['num'] == '2' and ls[1]['tag'] == '跳过了' and (ls[1]['a'] or '').startswith('第 2 部分：跳过了'), ls[:2])
        click(pg, '#rail .steps [data-go="1"]', 600)
        click(pg, '#rail .steps .step.current [data-pred]', 400)
        click(pg, '#rail .steps .step.current [data-nav="next"]', 600)
        ls = pg.evaluate(LIST, R)
        check(f'{tag}: answered, part 2 shows ✓', ls[1]['num'] == '✓' and 'done' in ls[1]['c'], ls[1])
        check(f'{tag}: no page errors', not errs, errs[:3])
        pg.close()

    # ---------- desktop: the card's Next and the step list ----------
    for lang in ('zh', 'en'):
        tag = f'architecture 1440 {lang}'; errs = []
        pg = open_lab(b, 1440, 900, lang, 'architecture', errs); R = '#ar-rail'
        cn = pg.evaluate(CARDNEXT, R)
        check(f'{tag}: the card’s Next reads “{Z(lang, "Skip question", "跳过本题")}” (ghost) while the question is open', cn and cn['t'] == Z(lang, 'Skip question', '跳过本题') and cn['ghost'], cn)
        click(pg, '#ar-rail .step.current [data-nav="next"]')
        ls = pg.evaluate(LIST, R)
        check(f'{tag}: skipped part 1 in the list: number, “{Z(lang, "skipped", "跳过了")}”, no ✓', ls[0]['num'] == '1' and ls[0]['tag'] == Z(lang, 'skipped', '跳过了') and 'skipped' in ls[0]['c'], ls[0])
        if lang == 'zh':
            pg.screenshot(path=str(P / 'shots' / 'kit' / 'r3-practice-architecture-1440-skipped.png'))
        # R07: the challenge
        click(pg, '#view-dyn .modes button[data-mode="challenge"]', 600)
        ex = pg.evaluate("""()=>{const c=document.querySelector('[data-exits]'); if(!c) return null; return {h:c.querySelector('.eyebrow').textContent, o:(c.querySelector('.ex-opt')||{}).textContent||'', done:!!c.querySelector('.ex-passed')};}""")
        check(f'{tag}: the practice card is headed “{Z(lang, "Going further (optional)", "可选进阶")}”, with no line saying it may be skipped (the owner, 2026-10-12)',
              ex and ex['h'] == Z(lang, 'Going further (optional): can you use it on a new case?', '可选进阶：能用到新情况上吗？') and ex['o'] == '', ex)
        check(f'{tag}: no completion summary before the core questions are answered', not pg.evaluate("()=>!!document.querySelector('#ar-rail .ex-passed')"))
        names = pg.evaluate("()=>[...document.querySelectorAll('#ar-rail button')].filter(b=>b.offsetParent!==null).map(b=>b.textContent.trim())")
        nexts = [x for x in names if 'Next question' in x or '下一题' in x or '下一道' in x]
        check(f'{tag}: no two buttons on screen share a next-question name', len(nexts) == len(set(nexts)), nexts)
        core = pg.evaluate("()=>window.__ar.CHQ_CORE.map(q=>q.a)")
        for i, a in enumerate(core):
            click(pg, f'#ar-rail [data-chi="{i}"]', 300)
            if i < len(core) - 1:
                click(pg, f'#ar-rail [data-cq="{i}"][data-v="{a}"]', 400); continue
            # the last one like a learner: the option in the middle of the screen, then a real click
            pg.evaluate(f"()=>document.querySelector('#ar-rail [data-cq=\"{i}\"][data-v=\"{a}\"]').scrollIntoView({{block:'center'}})"); pg.wait_for_timeout(200)
            pg.click(f'#ar-rail [data-cq="{i}"][data-v="{a}"]'); pg.wait_for_timeout(500)
        s = pg.evaluate("""()=>{const d=document.querySelector('#ar-rail .ar-ch .ex-passed'); if(!d) return null; const a=d.querySelector('a.btn'), f=[...document.querySelectorAll('#ar-rail .ar-ch .stepnav [data-chi]')].pop(),
            r=d.getBoundingClientRect(); return {t:d.querySelector('p').textContent, a:a?{t:a.textContent.trim(), h:a.getAttribute('href'), ghost:a.classList.contains('ghost')}:null,
            fw:f?{t:f.textContent.trim(), ghost:f.classList.contains('ghost')}:null, inEx:!!document.querySelector('[data-exits] .ex-passed'), on:r.top>=0&&r.bottom<=innerHeight};}""")
        check(f'{tag}: three core questions right: the challenge card says the core is done',
              s and s['t'] == Z(lang, '✓ You have finished this lab’s core: the challenge meets the goal. The practice below can wait for later.', '✓ 这个 Lab 的核心学习完成了：挑战已达标。下面的练习可以以后再做。'), s)
        check(f'{tag}: the next lab is the main button there', s and s['a'] and s['a']['h'].startswith('#') and len(s['a']['h']) > 1 and not s['a']['ghost']
              and s['a']['t'].startswith(Z(lang, ('Next lab on this branch: ', 'Suggested next lab: ', 'Next stop: ', 'Go on with the path'), ('这一枝的下一个 Lab：', '推荐下一步：', '下一站：', '接下来可以继续'))), s and s['a'])
        check(f'{tag}: the deeper questions are optional and secondary', s and s['fw'] and s['fw']['t'] == Z(lang, 'Deeper questions (optional)', '进阶题（可选）') and s['fw']['ghost'], s and s['fw'])
        check(f'{tag}: the summary is on screen, once (not again in the practice card)', s and s['on'] and not s['inEx'], s)
        check(f'{tag}: snap().passed after the core three', pg.evaluate("()=>window.__prog.snap().passed") is True)
        click(pg, '#ar-rail [data-chi="3"]', 300)
        k = pg.evaluate("()=>document.querySelector('#ar-rail .ar-chk').textContent")
        check(f'{tag}: a deeper question is labelled optional', k.startswith(Z(lang, 'Deeper (optional) 1 of 6', '进阶（可选）1 / 6')), k)
        check(f'{tag}: on a deeper question the summary stays at the top of the card', pg.evaluate("()=>{const c=document.querySelector('#ar-rail .ar-ch'); return !!c.querySelector(':scope>.ex-passed')&&c.querySelector(':scope>.ex-passed').compareDocumentPosition(c.querySelector('.ar-chq'))===4;}"))
        nb = pg.evaluate("()=>[...document.querySelectorAll('[data-exits] button')].map(b=>b.textContent.trim()).filter(t=>/下一题|Next question/.test(t))")
        check(f'{tag}: the practice card’s next button is “{Z(lang, "Next question in this set", "这组下一题")}”', nb == [Z(lang, 'Next question in this set', '这组下一题')], nb)
        if lang == 'zh':
            pg.evaluate("()=>document.querySelector('#ar-rail .ar-ch').scrollIntoView()"); pg.wait_for_timeout(200)
            pg.screenshot(path=str(P / 'shots' / 'kit' / 'r3-practice-architecture-1440-core-done.png'))
        check(f'{tag}: no page errors', not errs, errs[:3])
        pg.close()

    # ---------- R07 in a lab whose practice card shows the summary: the KV lab's challenge ----------
    for w, h in ((1440, 900), (390, 844)):
        tag = f'kv-cache challenge {w} zh'; errs = []
        pg = open_lab(b, w, h, 'zh', 'kv-cache', errs)
        click(pg, '#view-lab .modes button[data-mode="challenge"]', 600)
        check(f'{tag}: before Submit, the practice card has no summary', not pg.evaluate("()=>!!document.querySelector('[data-exits=\"kv-cache\"] .ex-passed')"))
        click(pg, '#rail [data-bind-seg="wp"] button[data-v="fp8"]', 300)
        click(pg, '#rail [data-bind-seg="kp"] button[data-v="fp8"]', 300)
        click(pg, '#chSubmit', 1200)
        ok = pg.evaluate("()=>window.__prog.snap().passed")
        s = pg.evaluate("""()=>{const c=document.querySelector('[data-exits="kv-cache"]'), d=c&&c.querySelector(':scope>.ex-passed'); if(!d) return null; const a=d.querySelector('a.btn');
            return {first:c.firstElementChild===d, t:d.querySelector('p').textContent, a:a?a.textContent.trim():null};}""")
        check(f'{tag}: FP8 weights and FP8 KV cache meet the goal', ok is True, ok)
        check(f'{tag}: once passed, the practice card opens with the summary and the next lab', s and s['first'] and s['t'].startswith('✓ 这个 Lab 的核心学习完成了') and (s['a'] or '').startswith(('这一枝的下一个 Lab：', '推荐下一步：', '下一站：', '接下来可以继续')), s)
        check(f'{tag}: no page errors', not errs, errs[:3])
        pg.close()
    b.close()

print(f'{n} checks, {len(fails)} failed')
sys.exit(1 if fails else 0)
