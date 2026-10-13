"""Numbers test for the prompt-injection lab (CONTRACT §6).

Re-derives every number the page shows, independently, in Python:
- the seeded stratified draws (mulberry32 + Fisher-Yates, seed 252) and the 100 tasks of each workload;
- the harness for one task, written here from the spec as gates (tools, detector, sending rule, the user's answer), not ported line by line;
- the engine against window.__pin.run on 2,000+ settings (every tool subset, defense and rule, and assumed values);
- the spec's verified numbers (page agent, inbox assistant, coding agent, research agent);
- the challenge: 32 combinations, 4 pass, the minimal one, and its feedback;
- the texts the learner reads: part 1, 3, 4 and 6 results, scenarios, the exits (Explain, New case), the drill, the math;
- the minimal implementation: runs the Python and compares its output with the printed one;
- the teaser: two states, no numbers in the sentence, notes plain text.
Run: python3 tests/prompt-injection_numbers.py"""
import pathlib, subprocess, sys, itertools, io, contextlib, re
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'prompt-injection'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/prompt-injection.html', '--only', 'prompt-injection'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(OUT / 'numbers.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'prompt-injection.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'numbers.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


# ---------------- the draws ----------------
M = 0xFFFFFFFF
def mulberry(a):
    a &= M
    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & M; t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & M
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & M)) & M
        return ((t ^ (t >> 14)) & M) / 4294967296
    return nxt


def perm(n_, seed):
    r = mulberry(seed); p = list(range(n_))
    for i in range(n_ - 1, 0, -1):
        j = int(r() * (i + 1)); p[i], p[j] = p[j], p[i]
    return p


SEED, WARN = 252, 0.25
# workload: (kind, tasks, poisoned, needs files, send: None | 'c' contact | 'o' outside), needed = a withheld poisoned item costs the task
WL = {'page': (True, [('S', 60, 12, 0, None), ('E', 25, 5, 0, 'c'), ('C', 15, 3, 1, None)]),
      'mail': (False, [('Q', 35, 7, 0, None), ('R', 38, 8, 0, 'c'), ('F', 12, 2, 1, 'c'), ('N', 15, 3, 1, 'o')]),
      'code': (True, [('X', 80, 8, 1, 'c'), ('T', 20, 2, 0, None)]),
      'res': (True, [('Y', 100, 20, 0, None)])}
BASE = dict(wl='page', inp=1, files=1, send=1, det=0, prm=0, scope=0, rule='free', f=0.5, d=0.8, fa=0.025)


def tasks(wl):
    T = []
    for kid, nn, k, nf, ns in WL[wl][1]:
        pos = {int((i + 0.5) * nn / k) for i in range(k)}
        for i in range(nn): T.append(dict(kind=kid, poi=i in pos, nf=nf, ns=ns))
    Pi = [j for j, t in enumerate(T) if t['poi']]; Ci = [j for j, t in enumerate(T) if not t['poi']]
    sg, tg, pc = perm(len(Pi), SEED * 7 + 1), perm(len(Pi), SEED * 7 + 2), perm(len(Ci), SEED * 7 + 3)
    for a, j in enumerate(Pi): T[j]['s'] = (sg[a] + .5) / len(Pi); T[j]['t'] = (tg[a] + .5) / len(Pi)
    for a, j in enumerate(Ci): T[j]['fp'] = (pc[a] + .5) / len(Ci)
    return T


TK = {w: tasks(w) for w in WL}


def gate_send(rule, dest):
    """the sending rule for one send: 'run' | 'block' | 'ask' (the user approves legitimate sends, rejects the injected one)"""
    if rule == 'free': return 'run'
    if rule == 'askAll': return 'ask'
    if rule == 'allow': return 'run' if dest == 'contact' else 'block'
    return 'run' if dest == 'contact' else 'ask'   # allowAsk


def one(t, c, needed):
    has_in, has_f, has_s = bool(c['inp']), bool(c['files']), bool(c['send'])
    if c['scope']: has_f, has_s = has_f and bool(t['nf']), has_s and t['ns'] is not None
    r = dict(done=0, att=0, harm=0, blk=0, ask=0, warn=0, held=0, fa=0, ign=0)
    if not has_in: return r
    reaches = True
    if c['det'] and t['poi'] and t['t'] < c['d']: r['held'] = 1; reaches = False
    if c['det'] and not t['poi'] and t['fp'] < c['fa']: r['fa'] = 1
    f = c['f'] / 2 if c['prm'] else c['f']
    if t['poi'] and reaches:
        if t['s'] < f:
            r['att'] = 1
            if not (has_f and has_s): r['blk'] = 1
            else:
                g = gate_send(c['rule'], 'outside')
                if g == 'run': r['harm'] = 1
                else:
                    r['blk'] = 1
                    if g == 'ask': r['ask'] += 1
        elif t['s'] < c['f'] + WARN: r['warn'] = 1
        else: r['ign'] = 1
    ok = (not t['nf'] or has_f) and (t['ns'] is None or has_s) and not r['fa'] and not (r['held'] and needed)
    if ok and t['ns'] is not None:
        g = gate_send(c['rule'], 'contact' if t['ns'] == 'c' else 'outside')
        if g == 'block': ok = False
        elif g == 'ask': r['ask'] += 1
    r['done'] = int(ok)
    return r


def run(**kw):
    c = dict(BASE, **kw); needed = WL[c['wl']][0]
    rs = [one(t, c, needed) for t in TK[c['wl']]]
    out = {k: sum(r[k] for r in rs) for k in ('done', 'att', 'harm', 'blk', 'ask', 'warn', 'held', 'fa', 'ign')}
    out['poi'] = sum(t['poi'] for t in TK[c['wl']]); out['rs'] = rs
    return out


# ---------------- the spec's verified numbers (pure Python) ----------------
b = run()
check('page base', (b['harm'], b['warn'], b['done'], b['att'], b['ign']) == (10, 5, 100, 10, 5), b)
for k, want in (('send', (0, 75)), ('files', (0, 85)), ('inp', (0, 0))):
    r = run(**{k: 0}); check(f'page {k} off', (r['harm'], r['done']) == want, (r['harm'], r['done']))
for kw, want in ((dict(det=1), (2, 82)), (dict(prm=1), (5, 100)), (dict(scope=1), (0, 100)), (dict(rule='askAll'), (0, 100))):
    r = run(**kw); check(f'page {kw}', (r['harm'], r['done']) == want, (r['harm'], r['done']))
check('page askAll asks 35', run(rule='askAll')['ask'] == 35)
check('page det: held 16, false alarms 2', (run(det=1)['held'], run(det=1)['fa']) == (16, 2))
check('page det+prm: 1 harmful', run(det=1, prm=1)['harm'] == 1)
m = run(wl='mail'); check('mail start 10 harmful', m['harm'] == 10, m)
combos = []
for fi, se, de, ru in itertools.product((1, 0), (1, 0), (0, 1), ('free', 'askAll', 'allow', 'allowAsk')):
    r = run(wl='mail', files=fi, send=se, det=de, rule=ru)
    combos.append(dict(files=fi, send=se, det=de, rule=ru, harm=r['harm'], done=r['done'], ask=r['ask'], ok=r['harm'] == 0 and r['done'] >= 90))
okc = sorted([c for c in combos if c['ok']], key=lambda c: (c['ask'], c['det']))
check('challenge: 4 of 32 pass', len(okc) == 4 and len(combos) == 32, len(okc))
check('challenge minimal: all tools, detector, contacts freely: 17 asks, 98 done',
      okc and (okc[0]['files'], okc[0]['send'], okc[0]['det'], okc[0]['rule'], okc[0]['ask'], okc[0]['done']) == (1, 1, 1, 'allowAsk', 17, 98), okc[:1])
check('challenge minimal is unique', len(okc) > 1 and okc[1]['ask'] > okc[0]['ask'], okc[:2])
cd = run(wl='code', rule='allow'); check('coding agent: allowlist: 0 harmful', cd['harm'] == 0 and cd['att'] > 0, cd)
# X1: least privilege only bounds the harm when no task needs both the files and a way out; the inbox has such tasks
m0, m1, m2 = run(wl='mail'), run(wl='mail', scope=1), run(wl='mail', scope=1, rule='askAll')
both = [k for k, nn, kk, nf, ns in WL['mail'][1] if nf and ns]
harm_by_kind = {}
for t, x in zip(TK['mail'], m1['rs']):
    if x['harm']: harm_by_kind[t['kind']] = harm_by_kind.get(t['kind'], 0) + 1
check('inbox under least privilege: 10 -> 4 harmful, all in the file-sending tasks', (m0['harm'], m1['harm']) == (10, 4) and set(harm_by_kind) <= set(both) and both == ['F', 'N'], (m0['harm'], m1['harm'], harm_by_kind))
check('inbox, least privilege, ask before every send: 0 harmful, 65 sends + 4 attempts = 69 asks', (m2['harm'], m2['ask']) == (0, 65 + 4), m2['ask'])
check('page agent: no kind needs both files and send', not [k for k, nn, kk, nf, ns in WL['page'][1] if nf and ns])
# S2: the counts equal the expectations at the default rates for the page agent and the inbox, not always for the coding agent
for wl_ in ('page', 'mail'):
    Pw = sum(t['poi'] for t in TK[wl_])
    for kw, d_, f_ in ((dict(), 0, .5), (dict(prm=1), 0, .25), (dict(det=1), .8, .5), (dict(det=1, prm=1), .8, .25)):
        check(f'{wl_} {kw}: count equals the expectation', abs(run(wl=wl_, **kw)['att'] - Pw * (1 - d_) * f_) < 1e-9)
check('coding agent with prompting: expected 2.5, counted 2', run(wl='code', prm=1)['att'] == 2 and abs(sum(t['poi'] for t in TK['code']) * .25 - 2.5) < 1e-9)
# check b7v2: a joint count only comes close to its average (fidelity popover); Show the math names how many attempts fall on the file-sending tasks
p_both = sum(1 for t in TK['mail'] if t['poi'] and t['kind'] in both)
on_both = sum(1 for t, x in zip(TK['mail'], m1['rs']) if x['att'] and t['kind'] in both)
check('inbox, least privilege: 5 poisoned emails in the file-sending tasks, average 2.5, counted 4 (attempts there: 4 of 10)', (p_both, p_both * .5, m1['harm'], on_both, m1['att']) == (5, 2.5, 4, 4, 10), (p_both, m1['harm'], on_both, m1['att']))
rs_ = run(wl='res', files=0, send=0); check('research agent read-only: 0 harmful', rs_['harm'] == 0, rs_)

# ---------------- the minimal implementation ----------------
with sync_playwright() as p:
    br = p.chromium.launch()
    pg = br.new_page(viewport={'width': 1440, 'height': 900})
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(250)
    C = pg.evaluate("()=>window.__pin.consts")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): exec(C['CODE'], {})
    check('minimal implementation prints what the page says', buf.getvalue().strip() == C['CODEOUT'].strip(), buf.getvalue())
    check('CODE is at most 40 lines', len(C['CODE'].strip().splitlines()) <= 40)

    # ---------------- the engine against the page ----------------
    js = []
    for wl in WL:
        for inp, fi, se in itertools.product((1, 0), (1, 0), (1, 0)):
            for det, prm, scope in itertools.product((0, 1), (0, 1), (0, 1)):
                for ru in ('free', 'askAll', 'allow', 'allowAsk'):
                    js.append(dict(wl=wl, inp=inp, files=fi, send=se, det=det, prm=prm, scope=scope, rule=ru))
    for f, d, fa in ((0.3, 0.8, 0.025), (0.75, 0.5, 0.05), (1.0, 1.0, 0.1), (0.0, 0.0, 0.0), (0.55, 0.65, 0.015)):
        for wl in WL:
            for det, prm, ru in itertools.product((0, 1), (0, 1), ('free', 'allowAsk')):
                js.append(dict(wl=wl, det=det, prm=prm, rule=ru, f=f, d=d, fa=fa))
    got = pg.evaluate("L=>L.map(s=>window.__pin.run(s))", js)
    bad = 0
    for s, g in zip(js, got):
        r = run(**s)
        mine = {k: r[k] for k in ('done', 'att', 'harm', 'blk', 'ask', 'warn', 'poi', 'held', 'fa', 'ign')}
        theirs = {k: g[k] for k in mine}
        cells = [[x['done'], x['harm'], x['ask'], x['att']] for x in r['rs']]
        if mine != theirs or cells != g['cells']:
            bad += 1
            if bad < 4: print('  mismatch', s, mine, theirs)
    check(f'engine matches on {len(js)} settings', bad == 0, bad)
    T = pg.evaluate("()=>['page','mail','code','res'].map(w=>window.__pin.tasks(w))")
    for w, tl in zip(WL, T):
        mine = [(t['kind'], t['poi'], round(t.get('s', -1), 9), round(t.get('t', -1), 9), round(t.get('fp', -1), 9)) for t in TK[w]]
        theirs = [(t['kind'], t['poi'], round(t['s'] if t.get('s') is not None else -1, 9), round(t['t'] if t.get('t') is not None else -1, 9), round(t['fp'] if t.get('fp') is not None else -1, 9)) for t in tl]
        check(f'draws of {w}', mine == theirs)
    # the replayed task: a poisoned task of the sample kind that the detector misses and that obeys even under prompting
    sm = pg.evaluate("()=>window.__pin.run({})")
    t = TK['page'][sm['sample']]
    check('replayed task: poisoned E task, s < 0.25, t >= 0.8', t['poi'] and t['kind'] == 'E' and t['s'] < .25 and t['t'] >= .8, t)
    check('replay of the initial setup ends in the harmful send', 'call:harm' in sm['sampleEv'] and sm['sampleEv'][:4] == ['sys', 'user', 'call:ok', 'content'], sm['sampleEv'])
    for kw in (dict(scope=1), dict(rule='askAll'), dict(send=0), dict(files=0), dict(rule='allow')):
        ev = pg.evaluate("s=>window.__pin.run(s).sampleEv", kw)
        check(f'replay with {kw} has no harmful send', 'call:harm' not in ev, ev)
    # challenge
    ch = pg.evaluate("()=>window.__pin.chAll()")
    mine = sorted((c['files'], c['send'], c['det'], c['rule'], c['harm'], c['done'], c['ask'], c['ok']) for c in combos)
    theirs = sorted((c['files'], c['send'], c['det'], c['rule'], c['harm'], c['done'], c['ask'], c['ok']) for c in ch)
    check('challenge: all 32 combinations', mine == theirs)
    best = pg.evaluate("()=>window.__pin.chBest()")
    check('challenge: minimal', (best['files'], best['send'], best['det'], best['rule'], best['ask']) == (1, 1, 1, 'allowAsk', 17), best)
    # drill
    for sd in range(1, 13):
        d = pg.evaluate("s=>window.__pin.drill(s)", sd)
        r = run(wl=d['wl'], f=d['f'], det=d['det'], rule=d['rule'])
        check(f'drill {sd}', (r['harm'], r['ask'], r['att']) == (d['harm'], d['ask'], d['att']), (d, r['harm'], r['ask'], r['att']))
        if d['rule'] == 'askAll':
            nS = sum(nn for k, nn, kk, nf, ns in WL[d['wl']][1] if ns)
            check(f'drill {sd}: no detector when the user is asked, asks = sending tasks + attempts', d['det'] == 0 and r['ask'] == nS + r['att'] and r['att'] == round(20 * d['f']), (d, nS, r['ask'], r['att']))

    # ---------------- texts the learner reads ----------------
    def txt(sel): return pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' '):''}", sel)
    # 2026-10-09 (H1, H2): the Show-the-math chain whose first left-hand side reads one of `names` (spaces dropped), as rows (left, relation, right)
    def chain(names): return next((rows for rows in pg.evaluate("""()=>[...document.querySelectorAll('#pin-mathbody .pin-eqs')].map(q=>{const k=[...q.children], rows=[]; for(let i=0;i+2<k.length;i+=3) rows.push([k[i],k[i+1],k[i+2]].map(c=>c.textContent.replace(/\\s+/g,''))); return rows;})""") if rows and rows[0][0] in names), None)
    def clk(sel, w=300): pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(w)
    for lang in ('en', 'zh'):
        pg.close(); pg = br.new_page(viewport={'width': 1440, 'height': 900}); pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(URL); pg.wait_for_timeout(250)
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
        pg.evaluate("()=>{location.hash='prompt-injection'}"); pg.wait_for_timeout(500)
        Z = lambda en, zh: zh if lang == 'zh' else en
        clk('#pin-rail [data-pred="c"]', 300)
        clk('#pin-player [data-act="end"]', 600)
        o = txt('#pin-rail [data-obs]')
        check(f'{lang} part 1 result', Z(f'acted on it {b["att"]} times', f'agent 照做了 {b["att"]} 次') in o and Z(f'In {b["warn"]} it warned', f'另有 {b["warn"]} 次它提醒了用户') in o, o)
        mt = pg.evaluate("()=>[...document.querySelectorAll('#pin-metrics .metric .v')].map(e=>e.textContent)")
        check(f'{lang} part 1 cards', mt == [str(b['done']), str(b['att']), str(b['harm']), str(b['blk']), str(b['ask'])], mt)
        pg.evaluate("()=>document.getElementById('pin-math').open=true"); pg.wait_for_timeout(100)
        mtx = txt('#pin-mathbody')
        # 2026-10-09 (H1, H2): the expectation and the asks are typeset chains; read them row by row (left, relation, right)
        hj, ak = chain(('hijackedNtried', '被劫持Ntried')), chain(('timesaskedNask', '问用户的次数Nask'))
        check(f'{lang} math: expectation, count, harm, asks', bool(hj) and [x[2] for x in hj[1:]] == ['20×(1−0)×50%', f'{20 * .5:g}'] and hj[2][1] == '='
              and Z(f'In this day’s 100 tasks it happened {b["att"]} times', f'这一天的 100 个任务里，实际是 {b["att"]} 次') in mtx
              and Z(f'Harmful actions here: {b["harm"]}.', f'这里的有害操作：{b["harm"]} 次') in mtx and bool(ak) and [x[2] for x in ak[1:]] == ['0+0', str(b['ask'])], (hj, ak, mtx[:600]))
        # part 3: email off, files off
        clk('#pin-rail [data-go="2"]', 400)
        clk('#pin-rail .seg-ctl[data-key="send"] button[data-v="0"]', 400)
        o = txt('#pin-rail [data-obs]'); r = run(send=0)
        check(f'{lang} part 3 email off', Z(f'from {b["harm"]} to {r["harm"]}', f'从 {b["harm"]} 次降到 {r["harm"]} 次') in o and Z(f'the {100 - r["done"]} tasks', f'代价是 {100 - r["done"]} 个') in o, o)
        clk('#pin-rail .seg-ctl[data-key="send"] button[data-v="1"]', 300)
        clk('#pin-rail .seg-ctl[data-key="files"] button[data-v="0"]', 400)
        o = txt('#pin-rail [data-obs]'); r = run(files=0)
        check(f'{lang} part 3 files off', Z(f'the {100 - r["done"]} tasks', f'代价是 {100 - r["done"]} 个') in o, o)
        # part 4: each lever
        clk('#pin-rail [data-go="3"]', 400)
        for v, kw in (('det', dict(det=1)), ('prm', dict(prm=1)), ('scope', dict(scope=1)), ('ask', dict(rule='askAll'))):
            clk(f'#pin-rail [data-lever="{v}"]', 400)
            o = txt('#pin-rail [data-obs]'); r = run(**kw)
            ok = Z(f'from {b["harm"]} to {r["harm"]}', f'从 {b["harm"]} 次降到 {r["harm"]} 次') in o
            if v == 'det':
                ok = ok and Z(f'the agent acted on {r["att"]} of them', f'agent 照做了其中 {r["att"]} 个') in o and Z(f'the tasks done drop to {r["done"]}', f'做成的任务降到了 {r["done"]} 个') in o
                pg.evaluate("()=>document.getElementById('pin-math').open=true"); pg.wait_for_timeout(100); mt4 = txt('#pin-mathbody')
                check(f'{lang} part 4 det: Show the math has the expectation and the counts', [x[2] for x in (chain(('hijackedNtried', '被劫持Ntried')) or [[0, 0, 0]])[1:]] == ['20×(1−80%)×50%', '2'] and all(x in mt4 for x in (Z(f'withheld {r["held"]} of the 20 poisoned items and let {20 - r["held"]} through', f'检测器扣下了 {r["held"]} 个藏了话的内容，漏过去 {20 - r["held"]} 个'), Z(f'flagged {r["fa"]} clean items', f'另有 {r["fa"]} 个正常的内容'))), mt4[:600])
                pg.evaluate("()=>document.getElementById('pin-math').open=false")
            if v == 'scope':
                ok = ok and all(x in o for x in (Z(f'only from {m0["harm"]} to {m1["harm"]}', f'只从 {m0["harm"]} 次降到 {m1["harm"]} 次'), Z(f'asked {m2["ask"]} times a day', f'一天要被问 {m2["ask"]} 次')))
                pg.evaluate("()=>document.getElementById('pin-math').open=true"); pg.wait_for_timeout(100); mt4 = txt('#pin-mathbody')
                check(f'{lang} part 4 scope: Show the math says at least one gate is 0 per task, harm 0', Z('at least one of', '至少有一个是 0') in mt4 and Z('Harmful actions here: 0.', '这里的有害操作：0 次') in mt4, mt4[:600])
                pg.evaluate("()=>document.getElementById('pin-math').open=false")
            if v == 'ask': ok = ok and Z(f'asked {r["ask"]} times a day', f'一天被问 {r["ask"]} 次') in o and Z(f'{r["ask"] - r["att"]} are ordinary', f'{r["ask"] - r["att"]} 次是发给同事') in o
            check(f'{lang} part 4 {v}', ok, o)
            lv = pg.evaluate(f"()=>document.querySelector('#pin-rail [data-lever=\"{v}\"] .lv-r').textContent")
            check(f'{lang} part 4 {v} lever result', Z(f'harmful: {r["harm"]}', f'有害 {r["harm"]} 次') in lv, lv)
        # part 5: the chart shows the inbox assistant with least privilege
        clk('#pin-rail [data-go="4"]', 400)
        o5 = txt('#pin-rail .step.current'); mt = pg.evaluate("()=>[...document.querySelectorAll('#pin-metrics .metric .v')].map(e=>e.textContent)")
        check(f'{lang} part 5: the inbox with least privilege still has {m1["harm"]} harmful', Z(f'still has {m1["harm"]} harmful actions', f'有害操作却还有 {m1["harm"]} 次') in o5 and mt == [str(m1['done']), str(m1['att']), str(m1['harm']), str(m1['blk']), str(m1['ask'])], (o5[:300], mt))
        mt5 = pg.evaluate("()=>{document.getElementById('pin-math').open=true; return document.getElementById('pin-mathbody').innerText}")
        check(f'{lang} part 5: Show the math names the tasks that get both tools', all(x in mt5 for x in (Z('“Send a file to a colleague”', '“把文件发给同事”'), Z('“Send a file to a new outside contact”', '“把文件发给新客户”'), Z(f'Harmful actions here: {m1["harm"]}.', f'这里的有害操作：{m1["harm"]} 次'))), mt5[:600])
        pg.evaluate("()=>document.getElementById('pin-math').open=false")
        # part 6
        clk('#pin-rail [data-go="5"]', 400)
        rm = txt('#pin-rail'); d1, p1, s1 = run(det=1), run(prm=1), run(scope=1)
        # the owner, 2026-10-08 (G2): part 6 states the pattern, not part 1's, 3's and 4's counts; the halving it states must hold
        check(f'{lang} part 6 (G2): the points state the pattern, no earlier part’s counts', all(x in rm for x in (Z('needs both the files and email', '既要读文件、又要发邮件'), Z('A leak needs all three in the same task.', '数据要漏出去，同一个任务里三样缺一不可。'), Z('only cuts the harmful actions by about half', '只能让有害操作少一半左右')))
              and not any(x in rm for x in (Z('In part', '第 1 部分里'), Z(f'still left {d1["harm"]} and', f'分别还剩 {d1["harm"]} 次'), Z(f'let {20 - d1["held"]} poisoned pages through', f'漏过了 {20 - d1["held"]} 个'))), rm[:800])
        check(f'{lang} part 6: “about half” holds for the warning in the prompt', 0.4 <= p1['harm'] / b['harm'] <= 0.6, (p1['harm'], b['harm']))
        # scenarios
        clk('#pin-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 500)
        for i, (kw, want) in enumerate(((dict(), 'page'), (dict(wl='mail'), 'mail'), (dict(wl='code', rule='allow'), 'code'), (dict(wl='res', files=0, send=0), 'res'))):
            clk(f'#pin-rail [data-preset="{i}"]', 400)
            o = txt('#pin-insight'); r = run(**kw)
            check(f'{lang} scenario {want}', Z(f'{r["harm"]} harmful', f'有害操作却是 {r["harm"]} 次' if want == 'code' else f'有害操作 {r["harm"]} 次') in o, o)
            mt = pg.evaluate("()=>[...document.querySelectorAll('#pin-metrics .metric .v')].map(e=>e.textContent)")
            check(f'{lang} scenario {want} cards', mt == [str(r['done']), str(r['att']), str(r['harm']), str(r['blk']), str(r['ask'])], mt)
        # challenge + exits
        clk('#pin-app .modes button[data-mode="challenge"]', 500)
        for k, v in (('chD', '1'), ('chR', 'allowAsk')):
            clk(f'#pin-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]', 300)
        g = [txt('#pin-ch-h'), txt('#pin-ch-d'), txt('#pin-ch-a')]
        check(f'{lang} challenge live goals', g == [Z('0 times', '0 次'), Z('98 tasks', '98 个任务'), Z('17 times', '17 次')], g)
        clk('#pin-rail [data-act="submit"]', 500)
        res = txt('#pin-ch-res')
        check(f'{lang} challenge minimal feedback names 17', '17' in res, res[:300])
        ex_t = lambda: txt('#pin-rail .exits')
        e0 = ex_t()
        check(f'{lang} explain 1 names the count of the 20 poisoned tasks, without a part number (the owner, 2026-10-08, G2)', Z(f'in {b["att"]} of the 20 poisoned tasks', f'20 个藏了话的任务里，agent 照着藏着的话做了 {b["att"]} 次') in e0 and not any(x in e0 for x in ('In part 1', '第 1 部分')), e0[:300])
        clk('#pin-rail .exits [data-ex="tab"][data-i="1"]', 250)
        clk('#pin-rail .exits [data-ex="ans"][data-t="1"][data-v="1"]', 250)
        e1 = ex_t(); cal = run(wl='res', files=0, send=0)
        check(f'{lang} new case 1: the read-only agent', Z(f'({cal["harm"]} harmful actions there)', f'那里有害操作 {cal["harm"]} 次') in e1, e1[:600])
        clk('#pin-rail .exits [data-ex="q"][data-t="1"][data-j="1"]', 250)
        clk('#pin-rail .exits [data-ex="ans"][data-t="1"][data-v="1"]', 250)
        e2 = ex_t()
        check(f'{lang} new case 2: twice the pages', all(x in e2 for x in (Z(f'with 20 it was {b["harm"]}', f'20 个时是 {b["harm"]} 次'), Z(f'About {2 * b["harm"]}', f'约 {2 * b["harm"]} 次'), Z(f'About {b["harm"]}', f'约 {b["harm"]} 次'))), e2[:600])
        # the drill
        clk('[data-act="drill-reveal"]', 300)
        dq, da = txt('#pin-drill-q'), txt('#pin-drill-a')
        d = pg.evaluate("()=>window.__pin.drill(1)")
        Pd = 20; ex = Pd * (0.2 if d['det'] else 1) * d['f']; exs = f'{round(ex, 2):g}'
        ok = Z(f'About {exs} attempts ({d["att"]} on this day)', f'照做约 {exs} 次，这一天是 {d["att"]} 次') in da and Z(f'{d["harm"]} harmful', f'有害 {d["harm"]} 次') in da
        if d['rule'] == 'askAll':
            nS = sum(nn for k, nn, kk, nf, ns in WL[d['wl']][1] if ns)
            ok = ok and d['det'] == 0 and Z(f'<b>{nS}</b>', f'<b>{nS}</b>') in pg.evaluate("()=>document.querySelector('#pin-drill-q').innerHTML") and f'{nS}+{d["att"]}={d["ask"]}' in ''.join(da.split()) and nS + d['att'] == d['ask']
        check(f'{lang} drill answer (seed 1)', ok, (d, dq, da))
    # teaser
    tz = pg.evaluate("()=>window.__tzPin?[window.__tzPin.read(1),window.__tzPin.read(0),window.__tzPin.notes()]:null")
    check('teaser hook', tz is not None)
    if tz:
        for i, s_ in enumerate(tz[:2]):
            plain = re.sub('<[^>]+>', '', s_)
            check(f'teaser sentence {i} has no number and at most 25 words', not re.search(r'\d', plain.replace('drop@example.com', '')) and len(plain.split()) <= 25, plain)
        check('teaser notes plain text, say illustrative', all('<' not in x for x in tz[2]) and ('Illustrative' in tz[2][0] or '示意' in tz[2][0]), tz[2])
    # the tile on the home page: the visual stays within 130 px in both states, at 1440 and 390, in both languages
    for w_, h_ in ((1440, 900), (390, 844)):
        for lang in ('en', 'zh'):
            tp = br.new_page(viewport={'width': w_, 'height': h_}); tp.on('pageerror', lambda e: errs.append(str(e)))
            tp.goto(URL); tp.wait_for_timeout(250); tp.click(f'.langsw button[data-lang="{lang}"]'); tp.wait_for_timeout(150)
            tp.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="agents"], #domStrip [data-world="agents"]')].find(e=>e.offsetParent); if(g) g.click();}"""); tp.wait_for_timeout(400)
            has = tp.evaluate("()=>!!document.querySelector('#tile-prompt-injection .tz-pin-vis')")
            check(f'teaser tile renders ({w_} {lang})', has)
            if has:
                for v in ('1', '0'):
                    tp.evaluate("v=>document.querySelector(`#tile-prompt-injection .seg-ctl button[data-v=\"${v}\"]`).click()", v); tp.wait_for_timeout(120)
                    hh = tp.evaluate("()=>{const v=document.querySelector('#tile-prompt-injection .tz-pin-vis'), t=document.querySelector('#tile-prompt-injection'); return [v.offsetHeight, t.scrollWidth-t.clientWidth, document.querySelector('#tile-prompt-injection .tz-read').textContent]}")
                    check(f'teaser visual at most 130 px, no overflow ({w_} {lang} send={v})', hh[0] <= 130 and hh[1] <= 0, hh)
            tp.close()
    check('no page errors', not errs, errs)
    br.close()

print(f'prompt-injection numbers: {n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
