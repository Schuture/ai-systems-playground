"""Golden numbers for the PagedAttention lab.

An independent Python simulation (written from the rules the page states, simulating slot by slot
with an explicit 12,288-entry owner array, unlike the page's interval and event bookkeeping) is
compared with the page's engine through the read-only hook window.__pa, and with the text the
page actually displays. Run: python3 tests/paged-attention_numbers.py
"""
import math, pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
POOL, N, PMIN, PMAX, SEED = 12288, 40, 100, 600, 7
M32 = 0xFFFFFFFF
fails, out = [], []


def check(name, cond, info=''):
    out.append(('OK   ' if cond else 'FAIL ') + name + (f'  [{info}]' if info else ''))
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------- reference model
def mulberry32(seed):
    st = [seed & M32]

    def nxt():
        st[0] = (st[0] + 0x6D2B79F5) & M32
        t = st[0]
        t = ((t ^ (t >> 15)) * (t | 1)) & M32
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296.0
    return nxt


def workload(mean, limit, seed=SEED):
    """prompt ~ uniform integer 100..600; output = round-half-up(-mean * ln(1-u)), clamped to [1, limit]"""
    rnd, reqs = mulberry32(seed), []
    for _ in range(N):
        p = PMIN + int(rnd() * (PMAX - PMIN + 1))
        o = int(math.floor(-mean * math.log(1.0 - rnd()) + 0.5))
        reqs.append((p, max(1, min(limit, o))))
    return reqs


def contiguous(reqs, limit, exact):
    owner, size = [-1] * POOL, [(p + o) if exact else (p + limit) for p, o in reqs]
    start, length, queue, steps = {}, {}, list(range(N)), []
    while queue or length:
        while queue:                                       # first fit, first come first served
            i, run, pos = queue[0], 0, None
            for s in range(POOL):
                run = run + 1 if owner[s] < 0 else 0
                if run == size[i]:
                    pos = s - size[i] + 1
                    break
            if pos is None:
                break
            owner[pos:pos + size[i]] = [i] * size[i]
            start[i], length[i] = pos, reqs[i][0]
            queue.pop(0)
        for i in length:
            length[i] += 1
        fin = {i: sum(reqs[i]) for i in length}
        free = owner.count(-1)
        steps.append(dict(tok=sum(length.values()), resv=sum(fin[i] - length[i] for i in length),
                          intf=sum(size[i] - fin[i] for i in length), ext=free if queue else 0,
                          free=0 if queue else free, run=len(length), wait=len(queue),
                          lens=dict(length), starts={i: start[i] for i in length}, size={i: size[i] for i in length}))
        for i in [i for i in length if length[i] >= fin[i]]:
            owner[start[i]:start[i] + size[i]] = [-1] * size[i]
            del length[i]
    return steps, 0


def paged(reqs, B):
    freeq, queue, running = list(range(POOL // B)), list(range(N)), []
    table, length, gen, steps, pre = {i: [] for i in range(N)}, {}, [0] * N, [], 0
    while queue or running:
        preempted, k = False, 0
        while k < len(running):                            # running requests grow first
            i = running[k]
            if length[i] + 1 > len(table[i]) * B:
                gone = False
                while not freeq:                           # pool empty: preempt the newest running request
                    v = running.pop()
                    freeq += table[v][::-1]
                    table[v] = []
                    del length[v]
                    queue.insert(0, v)
                    pre, preempted = pre + 1, True
                    if v == i:
                        gone = True
                        break
                if gone:
                    continue
                table[i].append(freeq.pop(0))
            k += 1
        if not preempted:                                   # admit only in steps without a preemption
            while queue:
                i = queue[0]
                have = reqs[i][0] + gen[i]
                need = -(-(have + 1) // B)
                if need > len(freeq):
                    break
                table[i] = [freeq.pop(0) for _ in range(need)]
                length[i] = have
                running.append(i)
                queue.pop(0)
        for i in running:
            length[i] += 1
            gen[i] += 1
        state = ['free'] * POOL                             # classify every slot explicitly
        for i in running:
            F = sum(reqs[i])
            for j in range(len(table[i]) * B):
                state[table[i][j // B] * B + j % B] = 'tok' if j < length[i] else ('resv' if j < F else 'intf')
        steps.append(dict(tok=state.count('tok'), resv=state.count('resv'), intf=state.count('intf'), ext=0,
                          free=state.count('free'), run=len(running), wait=len(queue)))
        for i in [i for i in running if gen[i] >= reqs[i][1]]:
            running.remove(i)
            freeq += table[i][::-1]
            del length[i]
    return steps, pre


def simulate(strat, B=16, mean=256, limit=1536):
    reqs = workload(mean, limit)
    steps, pre = paged(reqs, B) if strat == 'paged' else contiguous(reqs, limit, strat == 'exact')
    busy = [s for s in steps if s['wait'] > 0] or steps
    avg = lambda k: sum(s[k] for s in busy) / len(busy)
    fin = [p + o for p, o in reqs]
    if strat == 'paged':
        unused, table = [-(-F // B) * B - F for F in fin], sum(-(-F // B) for F in fin) / N
    else:
        unused, table = ([0] * N if strat == 'exact' else [limit - o for _, o in reqs]), None
    shares = {k: avg(k) / POOL for k in ('tok', 'resv', 'intf', 'ext', 'free')}
    # the typical moment the lab pauses on: the first waiting step whose mix of states is closest to the average mix
    busy_t = [t for t, st in enumerate(steps) if st['wait'] > 0] or list(range(len(steps)))
    dist = lambda t: sum(abs(steps[t][k] / POOL - shares[k]) for k in shares)
    showcase = min(busy_t, key=lambda t: (round(dist(t), 12), t))
    return dict(reqs=reqs, steps=steps, T=len(steps), busyN=len(busy), util=shares['tok'], shares=shares, showcase=showcase, limit=limit,
                running=avg('run'), unused=sum(unused) / N, table=table, pre=pre,
                tps=sum(o for _, o in reqs) / len(steps), meanF=sum(fin) / N, meanO=sum(o for _, o in reqs) / N,
                capped=sum(1 for _, o in reqs if o == limit), run0=steps[0]['run'])


pc = lambda x: f'{x * 100:.1f}'.rstrip('0').rstrip('.') + '%'          # 24.8%, 90%
pc0 = lambda x: f'{math.floor(x * 100 + 0.5):.0f}%'                    # 25%
room = lambda n: 'room for 1 token' if n == 1 else f'room for {n:,} tokens'
def n1(v):
    s = f'{v:.1f}'
    if s.endswith('.0'):
        s = s[:-2]
    whole, _, frac = s.partition('.')
    return f'{int(whole):,}' + ('.' + frac if frac else '')


# R6.20: the explanations say the trend in words, with at most one rounded key number. The same word rules as the page,
# applied to the independent reference numbers, so a wrong engine number shows up as a wrong word
NUMW = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}
def many_w(q):                                   # "over five times as many"
    f = math.floor(q + 1e-9)
    return ('over ' if q > f + 1e-9 else '') + ('twice' if f == 2 else f'{NUMW[f]} times') + ' as many'
def share_w(u, u16):                             # part 6: the stored-token share against 16-token blocks, from the shown values
    d = round(u * 1000) / 10 - round(u16 * 1000) / 10
    if abs(d) < 2:
        return True, 'barely changes'
    return False, (f'rises from about {pc0(u16)} to about {pc0(u)}' if d > 0 else f'drops from about {pc0(u16)} to about {pc0(u)}')
def table_w(t, t16):                             # part 6: the block table against 16-token blocks
    q = t / t16
    if q >= 1.1:
        return 'get a little longer' if q < 1.5 else ('get about twice as long' if q < 2.5 else f'get about {math.floor(q + 0.5):,} times as long')
    if q <= 0.9:
        s_ = 1 / q
        return 'get a little shorter' if s_ < 1.5 else ('get about half as long' if s_ < 2.5 else ('get about a third as long' if s_ < 3.5 else ('get about a quarter as long' if s_ < 4.5 else 'get much shorter')))
    return 'stay about as long'
def steps_vs_max(T, Tm):                         # scenarios: a strategy's decode steps against reserving the maximum's
    q = Tm / T
    if q < 0.87: return 'more decode steps than reserving the maximum'
    if q < 1.15: return 'about as many decode steps as reserving the maximum'
    if q < 1.8: return 'fewer decode steps than reserving the maximum'
    w = 'about half' if q < 2.5 else ('about a third' if q < 3.5 else ('about a quarter' if q < 4.5 else ('a small fraction' if q < 10 else 'under a tenth')))
    return f'{w} of the decode steps that reserving the maximum needs'


def fast_steps(strat, seed, B=16, mean=256, limit=1536):
    """step count only, with interval and count bookkeeping (checked below against the slot-level reference for seed 7)"""
    rnd, reqs = mulberry32(seed), []
    for _ in range(N):
        p = PMIN + int(rnd() * (PMAX - PMIN + 1))
        reqs.append((p, max(1, min(limit, int(math.floor(-mean * math.log(1.0 - rnd()) + 0.5))))))
    queue, T = list(range(N)), 0
    if strat == 'max':
        gaps, run = [[0, POOL]], {}
        while queue or run:
            while queue:
                need = reqs[queue[0]][0] + limit
                g = next((g for g in gaps if g[1] >= need), None)
                if g is None:
                    break
                i = queue.pop(0)
                run[i] = [g[0], need, reqs[i][0]]
                g[0] += need
                g[1] -= need
                gaps = [g for g in gaps if g[1] > 0]
            for i in run:
                run[i][2] += 1
            T += 1
            for i in [i for i in run if run[i][2] >= sum(reqs[i])]:
                st, ln, _ = run.pop(i)
                merged = []
                for g in sorted(gaps + [[st, ln]]):
                    if merged and merged[-1][0] + merged[-1][1] == g[0]:
                        merged[-1][1] += g[1]
                    else:
                        merged.append(g)
                gaps = merged
        return T
    free, running, nb, length, gen = POOL // B, [], {}, {}, [0] * N
    while queue or running:
        pre, k = False, 0
        while k < len(running):
            i = running[k]
            if length[i] + 1 > nb[i] * B:
                gone = False
                while free == 0:
                    v = running.pop()
                    free += nb[v]
                    nb[v] = 0
                    queue.insert(0, v)
                    pre = True
                    if v == i:
                        gone = True
                        break
                if gone:
                    continue
                nb[i] += 1
                free -= 1
            k += 1
        if not pre:
            while queue:
                i = queue[0]
                have = reqs[i][0] + gen[i]
                need = -(-(have + 1) // B)
                if need > free:
                    break
                free -= need
                nb[i], length[i] = need, have
                running.append(queue.pop(0))
        for i in running:
            length[i] += 1
            gen[i] += 1
        T += 1
        for i in [i for i in running if gen[i] >= reqs[i][1]]:
            running.remove(i)
            free += nb[i]
    return T


ratios = [fast_steps('max', sd) / fast_steps('paged', sd) for sd in range(1, 201)]
srt = sorted(ratios)
SW = dict(ratio=ratios[SEED - 1], median=(srt[99] + srt[100]) / 2, below=sum(1 for r_ in ratios if r_ < ratios[SEED - 1]))


# ---------------------------------------------------------------- page
subprocess.run([sys.executable, str(P / 'build.py'), 'out/paged-attention.html', '--only', 'paged-attention'], check=True, cwd=P, capture_output=True)
(P / 'out' / 'wrapped-paged-attention.html').write_text(
    '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'paged-attention.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-paged-attention.html')
errs = []

with sync_playwright() as pw:
    br = pw.chromium.launch()
    pg = br.new_page(viewport={'width': 1440, 'height': 900})
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL)
    pg.wait_for_timeout(300)
    pg.click('.langsw button[data-lang="en"]')
    pg.evaluate("()=>{location.hash='paged-attention'}")
    pg.wait_for_timeout(500)
    J = lambda expr: pg.evaluate(expr)

    # 1. the workload generator
    for mean, lim in [(256, 1536), (128, 1536), (1024, 2048), (128, 8192), (256, 512)]:
        page_w = J(f"()=>window.__pa.workload({mean},{lim},7).map(x=>[x.p,x.o])")
        check(f'workload mean={mean} limit={lim}', [list(x) for x in workload(mean, lim)] == page_w)

    # 2. run statistics for every configuration the lab names
    configs = [('max', 16, 256, 1536), ('exact', 16, 256, 1536)] + [('paged', B, 256, 1536) for B in (1, 8, 16, 32, 64, 128, 256)]
    configs += [(s, 16, m, l) for m, l in [(128, 1536), (1024, 2048), (128, 8192), (256, 512)] for s in ('max', 'paged')]
    REF = {}
    for strat, B, mean, lim in configs:
        ref = simulate(strat, B, mean, lim)
        REF[(strat, B, mean, lim)] = ref
        pgs = J(f"()=>window.__pa.stats({{strat:'{strat}',B:{B},mu:{mean},lim:{lim}}})")
        tag = f'{strat} B={B} mean={mean} limit={lim}'
        close = lambda a, b: abs(a - b) < 1e-9
        check(f'{tag}: steps', ref['T'] == pgs['T'], f"{ref['T']} vs {pgs['T']}")
        check(f'{tag}: stored-token share', close(ref['util'], pgs['util']), f"{ref['util']:.6f} vs {pgs['util']:.6f}")
        check(f'{tag}: waste shares', all(close(ref['shares'][k], pgs['shares'][k]) for k in ref['shares']))
        check(f'{tag}: running at once', close(ref['running'], pgs['running']))
        check(f'{tag}: unused per request', close(ref['unused'], pgs['unused']))
        check(f'{tag}: block table', (ref['table'] is None and pgs['table'] is None) or close(ref['table'], pgs['table']))
        check(f'{tag}: preemptions', ref['pre'] == pgs['pre'], f"{ref['pre']} vs {pgs['pre']}")
        check(f'{tag}: tokens per step', close(ref['tps'], pgs['tps']))

    # 3. the per-step series and the slot-by-slot reconstruction the pool draws
    for strat, B in [('max', 16), ('paged', 16), ('paged', 64)]:
        ref = REF[(strat, B, 256, 1536)]
        for key in ('tok', 'resv', 'intf', 'ext', 'free', 'run', 'wait'):
            ser = J(f"()=>window.__pa.series({{strat:'{strat}',B:{B}}},'{key}')")
            check(f'{strat} B={B}: per-step {key}', ser == [s[key] for s in ref['steps']])
        ser = J(f"()=>window.__pa.series({{strat:'{strat}',B:{B}}},'done')")
        check(f'{strat} B={B}: per-step finished = 40 - running - waiting', ser == [N - s['run'] - s['wait'] for s in ref['steps']])
        for tt in (0, ref['T'] // 3, ref['T'] // 2, ref['T'] - 1):
            sl = J(f"()=>window.__pa.slotsAt({{strat:'{strat}',B:{B}}},{tt})")
            s = ref['steps'][tt]
            check(f'{strat} B={B}: pool cells at step {tt}', all(sl[k] == s[k] for k in ('tok', 'resv', 'intf', 'ext', 'free')), str(sl))

    # 4. the paper's range and headline ratios (the simulation is calibrated to them)
    b, x, p16 = REF[('max', 16, 256, 1536)], REF[('exact', 16, 256, 1536)], REF[('paged', 16, 256, 1536)]
    check('reserve-the-maximum lands in the paper range 20.4%-38.2%', 0.204 <= b['util'] <= 0.382, pc(b['util']))
    check('paged 16 near the paper figure 96.3% (within 2 points)', abs(p16['util'] - 0.963) < 0.02, pc(p16['util']))
    check('paged 16 finishes in 2x-4x fewer steps', 2 <= b['T'] / p16['T'] <= 4, f"{b['T'] / p16['T']:.2f}")

    # 5. text the page displays in the guided parts (EN); differences are taken from the displayed (rounded) values
    shown = lambda x: round(x * 1000) / 10                     # a share as the page displays it, in percent
    fmt1 = lambda v: f'{v:.1f}'.rstrip('0').rstrip('.')
    kv_tok = 2 * 32 * 8 * 128 * 2                              # Llama-3.1-8B: 2 x 32 layers x 8 KV heads x 128 x 2 bytes
    est = lambda u: round(math.floor((24 * 2 ** 30 * u - 8030261248 * 2 - 2 * 2 ** 30) / kv_tok + 1e-9) / 1000) * 1000   # exact parameter count, as the kit uses

    def go(i):
        pg.evaluate(f"()=>document.querySelector('#view-dyn [data-go=\"{i}\"]').click()")
        pg.wait_for_timeout(200)

    def obs():
        return pg.inner_text('#view-dyn [data-obs]')

    def card():
        return pg.inner_text('#view-dyn .step.current .step-b')

    def lever_result(v):
        return pg.inner_text(f'#view-dyn [data-lever="{v}"] .lv-r').strip()

    def to_last_step():
        pg.evaluate("()=>{const i=document.getElementById('pa-ic-time-in'); i.value=i.max; i.dispatchEvent(new Event('input',{bubbles:true}))}")
        pg.wait_for_timeout(200)

    go(0)
    body = card()
    check('part 1: how many fit at first', f"That way the pool fits only {b['run0']} requests at first. This lab shows where the memory goes, and how PagedAttention gets it back." in body, body[-260:])
    check('part 1 does not give the prediction away (no average final length before part 2)', f"{round(b['meanF']):,} tokens" not in body)
    check('part 1: the model, the pool and the queue; the derived bytes per token are left to Fidelity and Settings (R6.20)', 'This lab runs Llama-3.1-8B: the memory set aside for the KV cache holds 12,288 tokens, and 40 requests are waiting.' in body and 'KiB' not in body)
    paras = J("()=>[...document.querySelectorAll('#view-dyn .step.current .step-b p')].map(p=>p.textContent)")
    chips = J("()=>document.querySelectorAll('#view-dyn .step.current .step-b .chips, #view-dyn .step.current .step-b .tag').length")
    check('part 1: three paragraphs, no chips and no RTX 4090 paragraph (Fidelity and Settings keep the estimate)', len(paras) == 3 and chips == 0 and 'RTX 4090' not in body, f'{len(paras)} paragraphs, {chips} chips')
    # part 2: four options that leave no gap; a wrong pick, so part 3 must name the right one
    go(1)
    opts2 = J("()=>[...document.querySelectorAll('#view-dyn [data-pred]')].map(b=>b.textContent)")
    check('part 2: four prediction options without gaps', opts2 == ['Under 30%', '30% to 60%', '60% to 80%', 'Over 80%'], str(opts2))
    pg.click('#view-dyn [data-pred="gt80"]')
    pg.wait_for_timeout(150)
    wl = pg.inner_text('#pa-wl')
    check('stage: the workload line, with the actual average and the longest answer', wl == f"40 requests: prompts of 100 to 600 tokens; answers are mostly short and a few long. These 40 answers actually average {round(b['meanO']):,} tokens; the longest is {max(o_ for _, o_ in b['reqs']):,} (the cap is 1,536).", wl)
    go(2)
    pg.click('#view-dyn [data-pa="step"]')
    pg.wait_for_timeout(150)
    check('part 3: no whole-run result before the last decode step', pg.evaluate("()=>document.querySelector('#view-dyn [data-obs]').hidden"))
    s1 = b['steps'][1]
    badge = pg.inner_text('#pa-status')
    check('part 3: while the run goes, the badge only says it is running', badge == 'Running', badge)
    live = pg.inner_text('#view-dyn .pa-live')
    head = f"Decode step 2 of {b['T']:,}: {s1['run']} requests running, {s1['wait']} waiting, {N - s1['run'] - s1['wait']} finished."
    check('part 3: the card\'s live line gives the share now', live == head + f" Right now {pc(s1['tok'] / POOL)} of the pool stores tokens.", live)
    check('part 3: the stage\'s live line leaves the share to the bar under it', pg.inner_text('#pa-st-time .pa-tread') == head and pg.inner_text('#pa-nowlab') == f"Now: {pc(s1['tok'] / POOL)} stores tokens", pg.inner_text('#pa-st-time .pa-tread'))
    to_last_step()
    o = obs()
    check('part 3: the result leads with the waiting-step share, rounded, and says few requests run at once (the cards hold the exact run)', f"When each request reserves the maximum, only about {pc0(b['util'])} of the pool stores tokens while requests wait, so only a few requests run at once." in o
          and b['running'] < 8 and f"{b['T']:,}" not in o, o[:260])
    bins = [(0, 30, 'Under 30%'), (30, 60, '30% to 60%'), (60, 80, '60% to 80%'), (80, 101, 'Over 80%')]
    right = next(lab_ for lo_, hi_, lab_ in bins if lo_ <= shown(b['util']) < hi_)
    check('part 3: a wrong prediction gets the verdict, and the right option holds the displayed share', right == 'Under 30%'
          and f"You predicted “Over 80%”; the right answer is “{right}”." in o and 'Why: most of each region is empty, either waiting for later tokens or never used.' in o, o[-260:])
    nl = b['steps'][-1]['run']
    live = pg.inner_text('#view-dyn .pa-live')
    check('part 3: the last decode step', live == f"Decode step {b['T']:,} (the last): {'the final request finishes' if nl == 1 else f'the final {nl} requests finish'}.", live)
    check('part 3: the badge shows the waiting-step average after the last decode step', pg.inner_text('#pa-status') == f"While requests wait: {pc(b['util'])} stores tokens on average", pg.inner_text('#pa-status'))
    check('stage: the average bar is named', pg.inner_text('#pa-avglab') == f"While requests wait: {pc(b['util'])} on average", pg.inner_text('#pa-avglab'))
    tipx = pg.get_attribute('#pa-avglab', 'data-tipx')
    check('stage: the whole-run average counts the decode steps with a request waiting', tipx == f"This average is taken over the {b['busyN']:,} decode steps in which a request was waiting.", tipx)
    sub = lambda k: pg.inner_text(f'#pa-metrics [data-k="{k}"] .sub')
    val = lambda k: pg.inner_text(f'#pa-metrics [data-k="{k}"] .v')
    lab = lambda k: pg.inner_text(f'#pa-metrics [data-k="{k}"] .k')
    check('metrics: stored share, running, decode steps', val('util') == pc(b['util']) and val('run') == n1(b['running']) and val('steps') == f"{b['T']:,}"
          and sub('steps') == f"decode steps, {n1(b['tps'])} tokens generated per step on average", sub('steps'))
    check('metrics: the two waiting-step averages are named', [lab('util'), sub('util'), lab('run'), sub('run')] == ['Stores tokens while requests wait', 'average share of the pool over the decode steps in which requests wait',
          'Running at once while requests wait', 'requests (average over the decode steps in which requests wait)'], str([lab('util'), lab('run')]))
    busy = [st_ for st_ in b['steps'] if st_['wait'] > 0]
    check('metrics: why they are named (the all-step averages differ)', abs(sum(st_['run'] for st_ in b['steps']) / b['T'] - b['tps']) < 1e-9 and b['running'] > b['tps'] + 0.5 and len(busy) == b['busyN'] < b['T'],
          f"running while waiting {b['running']:.2f}, over all steps {b['tps']:.2f}")
    check('metrics: never used per request', val('unused') == n1(b['unused']) and sub('unused') == 'tokens of room: on average each request reserved this much more than it used', val('unused'))
    go(3)
    t0 = b['showcase']
    check('part 4: the decode step it stops at, numbered from 1, and why', f"The pool stops at decode step {t0 + 1:,}: its mix of room is the closest to the average while requests wait." in card(), card()[:120])
    s0 = b['steps'][0]
    blocks0 = sorted((s0['starts'][i], s0['size'][i]) for i in s0['lens'])
    check('part 4: at decode step 1 the external fragmentation is the empty end of the pool', all(blocks0[k][0] == (blocks0[k - 1][0] + blocks0[k - 1][1] if k else 0) for k in range(len(blocks0)))
          and s0['ext'] == POOL - sum(sz for _, sz in blocks0) and s0['wait'] > 0, f"{s0['ext']} = {POOL} - {sum(sz for _, sz in blocks0)}")
    st0 = b['steps'][t0]
    ids = sorted(st0['lens'])
    reqs0 = b['reqs']
    iR = max(ids, key=lambda i: (sum(reqs0[i]) - st0['lens'][i], -i))           # most room still waiting for later tokens (first on ties)
    iI = max(ids, key=lambda i: (st0['size'][i] - sum(reqs0[i]), -i))          # most room never used
    pg.click('#view-dyn [data-lever="hl-resv"]')
    pg.wait_for_timeout(150)
    o = obs()
    p_, o_ = reqs0[iR]
    F_, len_ = p_ + o_, st0['lens'][iR]
    check('part 4: reserved for later tokens, its rounded share and why (the example request is in its tooltip, below)',
          f"Reserved for later tokens: about {pc0(b['shares']['resv'])} of the pool while requests wait. A request holds room for all its later tokens from the start, but fills it only one token per decode step" in o, o[:320])
    # the same request's cell tooltip (hover the middle of its region)
    slot = st0['starts'][iR] + st0['size'][iR] // 2
    geo = pg.evaluate("()=>{const c=document.getElementById('pa-cv'); c.scrollIntoView({block:'center'}); const r=c.getBoundingClientRect(); return {x:r.left,y:r.top,w:c.clientWidth}}")
    pg.wait_for_timeout(250)
    geo = pg.evaluate("()=>{const c=document.getElementById('pa-cv'), r=c.getBoundingClientRect(); return {x:r.left,y:r.top,w:c.clientWidth}}")
    cols = 48 if geo['w'] >= 440 else 32
    gap = 2 if geo['w'] >= 440 else 1
    cs = (geo['w'] - (cols - 1) * gap) / cols
    k, j = slot // 16, slot % 16
    pg.mouse.move(geo['x'] + (k % cols) * (cs + gap) + (j + 0.5) / 16 * cs, geo['y'] + (k // cols) * (cs + gap) + cs / 2)
    pg.wait_for_timeout(200)
    tv = pg.evaluate("()=>document.getElementById('tip').hidden?'':document.getElementById('tip').innerText").replace('\n', ' ')
    R_ = st0['size'][iR]
    check('pool tooltip: the request, its prompt and answer, generated so far, and its room',
          f"Request {iR + 1}: a {p_:,}-token prompt; its answer will be {o_:,} tokens, {len_ - p_:,} generated so far." in tv
          and f"It holds room for {R_:,} tokens (prompt plus 1,536): {F_ - len_:,} wait for its later tokens and {R_ - F_:,} will never be used." in tv, tv[:260])
    pg.mouse.move(2, 2)
    pg.click('#view-dyn [data-lever="hl-intf"]')
    pg.wait_for_timeout(150)
    o = obs()
    p_, o_ = reqs0[iI]
    F_, R_ = p_ + o_, st0['size'][iI]
    check('part 4: internal fragmentation, its rounded share, and “most answers are far shorter than the cap” holds',
          f"Internal fragmentation: about {pc0(b['shares']['intf'])} of the pool while requests wait, the biggest waste. Most answers are far shorter than the answer cap" in o
          and max(o2 for _, o2 in b['reqs']) < 1536 and b['meanO'] < 1536 / 4, o[:360])
    # the example request with the most room never used: its numbers, from its own tooltip on the pool
    slot = st0['starts'][iI] + st0['size'][iI] - 1
    pg.evaluate("()=>document.getElementById('pa-cv').scrollIntoView({block:'center'})")
    pg.wait_for_timeout(250)
    geo = pg.evaluate("()=>{const c=document.getElementById('pa-cv'), r=c.getBoundingClientRect(); return {x:r.left,y:r.top,w:c.clientWidth}}")
    k, j = slot // 16, slot % 16
    pg.mouse.move(geo['x'] + (k % cols) * (cs + gap) + (j + 0.5) / 16 * cs, geo['y'] + (k // cols) * (cs + gap) + cs / 2)
    pg.wait_for_timeout(200)
    tv = pg.evaluate("()=>document.getElementById('tip').hidden?'':document.getElementById('tip').innerText").replace('\n', ' ')
    len_ = st0['lens'][iI]
    check('pool tooltip: the request with the most room never used, its prompt, answer and room',
          f"Request {iI + 1}: a {p_:,}-token prompt; its answer will be {o_:,} tokens, {len_ - p_:,} generated so far." in tv
          and f"It holds room for {R_:,} tokens (prompt plus 1,536): {F_ - len_:,} wait for its later tokens and {R_ - F_:,} will never be used." in tv, tv[:260])
    pg.mouse.move(2, 2)
    check('part 4: internal fragmentation is the biggest waste', b['shares']['intf'] > max(b['shares']['resv'], b['shares']['ext']))
    pg.click('#view-dyn [data-lever="hl-ext"]')
    pg.wait_for_timeout(150)
    o = obs()
    check('part 4: external fragmentation, including the end of the pool', f"External fragmentation: about {pc0(b['shares']['ext'])} of the pool while requests wait. The room nobody holds (gaps between regions and the end of the pool) adds up" in o, o[:200])
    check('part 4: tried options keep their share of the pool', lever_result('hl-intf') == f"{pc(b['shares']['intf'])} of the pool while requests wait" and lever_result('hl-all') == f"{pc(b['shares']['tok'])} stores tokens while requests wait", lever_result('hl-intf'))
    pg.click('#view-dyn [data-lever="hl-all"]')
    pg.wait_for_timeout(150)
    check('part 4: all states', f"All states: while requests wait, about {pc0(b['shares']['tok'])} stores tokens; the rest is the three kinds of waste." in obs(), obs()[:120])
    go(4)
    pg.click('#view-dyn [data-lever="st-paged"]')
    pg.wait_for_timeout(300)
    o = obs()
    ratio = float(n1(p16['running']).replace(',', '')) / float(n1(b['running']).replace(',', ''))     # from the shown values
    fold = math.floor(ratio + 1e-9)
    frac = {2: 'half', 3: 'a third', 4: 'a quarter', 5: 'a fifth', 6: 'a sixth', 7: 'a seventh', 8: 'an eighth', 9: 'a ninth'}[fold]
    check('part 5: paging against the initial setup (one rounded pair), then in words why the decode steps do not drop as much as the running requests rise',
          f"Paging: while requests wait, about {pc0(p16['util'])} of the pool stores tokens (initial setup: about {pc0(b['util'])})." in o
          and f"So {many_w(ratio)} requests run at once as in the initial setup. The decode steps do not fall to {frac}, though: once every waiting request has started, fewer and fewer are still running." in o, o[:520])
    lastw = max(t_ for t_, st_ in enumerate(p16['steps']) if st_['wait'] > 0)
    tail = [st_['run'] for st_ in p16['steps'][lastw + 1:]]
    check('part 5: that reason holds (after the last wait the running count only falls; steps drop less than running rises)',
          all(a_ >= b_ for a_, b_ in zip(tail, tail[1:])) and b['T'] / p16['T'] < fold and fold == 5 and ratio > 5, f"{b['T'] / p16['T']:.2f} vs {ratio:.2f}")
    check('part 5: why paging wastes so little', 'Why: a request wastes at most the unfilled part of its last block, room for fewer than 16 tokens, and a free block works anywhere, so no gap is too small.' in o)
    check('part 5: at most one part-filled block per request', all(0 <= -(-sum(r_) // 16) * 16 - sum(r_) < 16 for r_ in p16['reqs']))
    pg.click('#view-dyn [data-lever="st-exact"]')
    pg.wait_for_timeout(200)
    o = obs()
    check('part 5: exact length against the initial setup', f"Exact length: while requests wait, about {pc0(x['util'])} of the pool stores tokens (initial setup: about {pc0(b['util'])})." in o, o[:160])
    pg.click('#view-dyn [data-lever="st-start"]')
    pg.wait_for_timeout(200)
    o = obs()
    check('part 5: the initial setup, its rounded share, and what follows from it in words', o.strip().endswith(f"Initial setup: while requests wait, only about {pc0(b['util'])} of the pool stores tokens, so few requests run at once and the run takes many decode steps."), o[-200:])
    check('part 5: tried options keep their result', [lever_result(v) for v in ('st-start', 'st-exact', 'st-paged')] == [f"{pc(r_['util'])} stores tokens while requests wait" for r_ in (b, x, p16)],
          str([lever_result(v) for v in ('st-start', 'st-exact', 'st-paged')]))
    go(5)
    p1, p8, p64, p256 = (REF[('paged', B_, 256, 1536)] for B_ in (1, 8, 64, 256))

    for B_, r_ in ((64, p64), (256, p256), (1, p1), (8, p8)):
        pg.click(f'#view-dyn #pa-ic-B button[data-v="{B_}"]')
        pg.wait_for_timeout(200)
        o = obs()
        small, sw = share_w(r_['util'], p16['util'])
        lead = f"{B_}-token blocks: against the default size, the share of the pool storing tokens {sw}, {'but' if B_ < 16 or not small else 'and'} block tables {table_w(r_['table'], p16['table'])}."
        if B_ == 1:
            why = 'Why: no block is half written, but every token needs its own entry.'
            check('part 6: 1-token blocks leave no room unfilled, and the metric says so', r_['unused'] == 0 and pg.inner_text('#pa-metrics [data-k="unused"] .v') == '0'
                  and pg.inner_text('#pa-metrics [data-k="unused"] .sub') == 'tokens of room: a block holds 1 token, so none is left unfilled', pg.inner_text('#pa-metrics [data-k="unused"] .sub'))
        elif B_ < 16:
            why = 'Why: smaller blocks leave less of the last one empty, but more are needed.'
        else:
            why = 'Why: bigger blocks leave more of the last one empty, but fewer are needed.'
        # the exact values the words come from are on the metric cards (and their ▲▼ against the previous size)
        cards = [pg.inner_text(f'#pa-metrics [data-k="{k_}"] .v') for k_ in ('util', 'table')]
        check(f'part 6: {B_}-token blocks against 16, in words from the reference numbers; the cards hold the exact values', lead + ' ' + why in o and cards == [pc(r_['util']), n1(r_['table'])], o[:320] + ' | ' + str(cards))
        check(f'part 6: {B_}-token blocks, closing line', 'vLLM defaults to 16: the paper found it large enough for the GPU to read efficiently and small enough to avoid much internal fragmentation in most workloads (§7.2).' in o)
    tried = pg.inner_text('#view-dyn .pa-btried')
    check('part 6: tried sizes keep their results', all(v in tried for v in (pc(p16['util']), pc(p64['util']), pc(p256['util']), n1(p64['table']), n1(p1['table']))), tried.replace('\n', ' '))
    go(6)
    body = card()
    check('part 7: what to remember', f"only about {pc0(b['util'])} of the pool stores tokens on average while requests wait" in body
          and f"Each request wastes at most the unfilled part of one block, so about {pc0(p16['util'])} of the pool stores tokens while requests wait." in body
          and f"Then still only about {pc0(x['util'])} of the pool stores tokens while requests wait." in body
          and f"The pool stayed the same size, yet with paging {many_w(ratio)} requests run at once while requests wait." in body, body[:200])

    # 6. the challenge: tiers follow from the independent numbers
    goal = lambda r: r['util'] >= 0.9 and (r['table'] is None or r['table'] <= 64)
    opts = [('max', 16), ('exact', 16)] + [('paged', B) for B in (1, 8, 16, 32, 64, 128, 256)]
    met = [o_ for o_ in opts if goal(REF[(o_[0], o_[1], 256, 1536)])]
    page_ch = J("()=>window.__pa.challenge().map(c=>[c.o.strat,c.o.B||16,c.met])")
    check('challenge: which options meet both targets', [[s_, B, goal(REF[(s_, B, 256, 1536)])] for s_, B in opts] == page_ch, str(met))
    best = sorted(met, key=lambda o_: ((o_[0] != 'max') + (o_[1] != 16), abs(math.log2(o_[1] / 16))))[0]
    check('challenge: minimal answer is paged with 16-token blocks', best == ('paged', 16) and J("()=>window.__pa.best()") == {'strat': 'paged', 'B': 16})
    # the page's own feedback: the badge at the start, and the explanation after "Show the answer", from the reference numbers
    pg.click('#view-dyn .modes button[data-mode="challenge"]')
    pg.wait_for_timeout(300)
    check('challenge: the badge names the failing goal', pg.inner_text('#pa-status') == f"Not yet: while requests wait, only {pc(b['util'])} of the pool stores tokens, under 90%", pg.inner_text('#pa-status'))
    pg.click('#view-dyn [data-act="submit"]')
    pg.wait_for_timeout(200)
    pg.click('#pa-chres [data-act="reveal"]')
    pg.wait_for_timeout(200)
    chres = pg.inner_text('#pa-chres')
    pagedR = {B_: REF[('paged', B_, 256, 1536)] for B_ in (1, 8, 16, 32, 64, 128, 256)}
    okB = [B_ for B_ in pagedR if goal(pagedR[B_])]
    lo, hi = okB[0], okB[-1]
    lst = str(okB[0]) if len(okB) < 2 else ', '.join(map(str, okB[:-1])) + ' or ' + str(okB[-1])
    small_fail = all(pagedR[B_]['table'] > 64 for B_ in pagedR if B_ < lo)
    large_fail = all(pagedR[B_]['util'] < 0.9 for B_ in pagedR if B_ > hi)
    check('challenge: the sizes that pass form one range, failing on the table below it and on the share above it', okB == [B_ for B_ in pagedR if lo <= B_ <= hi] and small_fail and large_fail, str(okB))
    check('challenge: after showing all options, baseline → which sizes pass → why, in words (the chart holds every value)',
          f"Initial setup, reserving the maximum: only about {pc0(b['util'])} of the pool stores tokens while requests wait. Reserving the exact length still falls {'well ' if 0.9 - x['util'] > 0.15 else ''}short: every region holds room for tokens not generated yet, and the gaps between regions go unused." in chres
          and x['util'] > b['util']
          and f"With paging, blocks of {lo} to {hi} tokens meet both goals" + (': smaller blocks need too many block-table entries, larger ones leave too much of their last block empty' if small_fail and large_fail else '') + '.' in chres
          and 'the paper’s two workloads had different best sizes, but 16 did well on both (§7.2).' in chres, chres[-520:])
    tags = pg.evaluate("()=>[...document.querySelectorAll('#pa-chres .cmp-tag')].map(x=>x.textContent)")
    # the owner's review of batch 8: a brief chart, the sizes that pass and the initial setup (= the pick); the failing sizes are left out and counted
    note = pg.inner_text('#pa-chres .cmp-note') if pg.query_selector('#pa-chres .cmp-note') else ''
    nrows = pg.evaluate("()=>document.querySelectorAll('#pa-chres .cmp-row').length")
    check('challenge: the chart tags the initial setup, holds only the sizes that pass and the initial setup, and counts the other choices',
          'initial setup' in tags and sum(1 for t_ in tags if 'over 64' in t_) == 0 and nrows == len(okB) + 1
          and note == f'The other {9 - len(okB) - 1} choices miss at least one goal and are not shown.', (tags, nrows, note))
    pg.click('#view-dyn .modes button[data-mode="guided"]')
    pg.wait_for_timeout(300)

    # 7. the home-tile teaser: the first 8 requests at their final lengths, under the lab's own rule (prompt + 1,536, 12,288-token pool)
    W8 = workload(256, 1536)[:8]
    F8 = [p_ + o_ for p_, o_ in W8]
    R8 = [p_ + 1536 for p_, _ in W8]
    held = fit = 0
    while fit < 8 and held + R8[fit] <= POOL:
        held, fit = held + R8[fit], fit + 1
    alloc = sum(-(-f // 16) * 16 for f in F8)
    tz = J("()=>window.__pa.teaser()")
    check('teaser: lengths', tz['F'] == F8, str(F8))
    check('teaser: how many fit when reserving prompt + 1,536', tz['fit'] == fit and tz['held'] == held and tz['left'] == POOL - held and tz['next'] == R8[fit], f'{fit} fit, {held} held')
    check('teaser: tokens at final length', tz['tokFit'] == sum(F8[:fit]) and tz['tokAll'] == sum(F8), f'{sum(F8[:fit])} / {sum(F8)}')
    check('teaser: 16-token blocks', tz['allocSum'] == alloc and tz['pFit'] == 8, str(alloc))

    # 7b. the seed sweep behind the Fidelity card's comparison
    check('sweep: the fast step counter matches the reference for seed 7', fast_steps('max', SEED) == b['T'] and fast_steps('paged', SEED) == p16['T'])
    sw = J("()=>window.__pa.sweep()")
    check('sweep: seeds 1-200, ratio, rank and median', sw['n'] == 200 and abs(sw['ratio'] - SW['ratio']) < 1e-12 and sw['below'] == SW['below'] and abs(sw['median'] - SW['median']) < 1e-12, f"{SW['ratio']:.3f}, {SW['below']} below, median {SW['median']:.3f}")

    # 8. the interview drill answers
    for k in range(6):
        d = J(f"()=>window.__pa.drill({k})")
        if d['kind'] == 0:
            n = -(-d['L'] // d['B'])
            check(f'drill {k}: blocks and unused', d['n'] == n and d['w'] == n * d['B'] - d['L'])
        else:
            per = -(-d['len'] // 16) * 16
            check(f'drill {k}: contiguous and paged capacity', d['c'] == POOL // (d['p'] + d['lim']) and d['g'] == POOL // per)

    # 9. pool size in bytes, the RTX 4090 estimate (KV lab's model: 24 GiB x 0.9 - 8,030,261,248 x 2 B - 2 GiB) and at vLLM's 0.92 default
    check('Llama-3.1-8B KV per token is 128 KiB', kv_tok == 128 * 1024)
    check('pool is 1.5 GiB', POOL * kv_tok == 1.5 * 2 ** 30)
    mu_note, lim_note = pg.inner_text('#pa-mu-note'), pg.inner_text('#pa-lim-note')
    cap_txt = 'None of these 40 answers reaches the cap.' if b['capped'] == 0 else f"{b['capped']} of these 40 {'reaches' if b['capped'] == 1 else 'reach'} the cap."
    check('controls: the answer-length and cap notes (none reaches the cap here)', f"These 40 actually average {round(b['meanO']):,} tokens." in mu_note and cap_txt in lim_note and b['capped'] == 0, mu_note[-60:] + ' | ' + lim_note[-60:])
    # the pool and the preemption count were in the Fidelity card's “Modeled” list, deleted with it (owner, 2026-10-11 19:28 UTC)
    fid = pg.inner_text('#paged-attention-fid')
    check('fidelity: the workload', 'prompts of 100 to 600 tokens, spread evenly; answer lengths drawn from an exponential distribution with mean 256, cut at the cap; random seed 7.' in fid)
    check('fidelity: RTX 4090 estimates at 0.9 and 0.92', f"a real RTX 4090 holds about {est(0.9):,} tokens after the weights (estimated with a GPU memory utilization of 0.9 and 2 GiB of runtime overhead; about {est(0.92):,} with vLLM’s default of 0.92)" in fid, f'{est(0.9)} {est(0.92)}')
    check('fidelity: the seed sweep', f"For these requests the initial setup needs {SW['ratio']:.2f} times as many decode steps as paging, more than for {SW['below']} of random seeds 1 to 200 (median {SW['median']:.2f})." in fid, f"{SW['ratio']:.2f} {SW['below']} {SW['median']:.2f}")
    check('fidelity: agreement with the paper, not a validation', f"this simulation’s {pc(p16['util'])} agrees with the paper’s 96.3% for vLLM, but that is not a validation" in fid)
    check('fidelity: the paper link', pg.evaluate("()=>!!document.querySelector('#paged-attention-fid a[href=\"https://arxiv.org/abs/2309.06180\"]')"))
    # timing: each decode step reads the weights once plus the running requests' KV cache (tokens stored); compare step counts with bytes read
    dim, layers, heads, kv_heads, ffn, vocab = 4096, 32, 32, 8, 14336, 128256         # Llama 3.1 8B (Meta llama-models, sku_list.py)
    hd = dim // heads
    per_layer = dim * heads * hd * 2 + dim * kv_heads * hd * 2 + 3 * dim * ffn + 2 * dim
    params = layers * per_layer + 2 * vocab * dim + dim                              # untied embedding and output layer, final norm
    check('Llama-3.1-8B parameters from its shape match the 8,030,261,248 the KV cache memory lab uses', params == 8030261248 and round(params / 1e7) / 100 == 8.03, f'{params:,}')
    Wall, emb = 8030261248 * 2, vocab * dim * 2   # the kit's exact count (CONTRACT §9)
    Wb = Wall - emb                                                                   # a decode step only looks up the embedding table
    cost = lambda r: len(r['steps']) * Wb + kv_tok * sum(st['tok'] for st in r['steps'])
    top = max(max(st['tok'] for st in r_['steps']) for r_ in (b, p16))
    bytes_ratio, steps_ratio = cost(b) / cost(p16), b['T'] / p16['T']
    tm = J("()=>window.__pa.timing()")
    check('timing: bytes-weighted ratio', abs(tm['bytes'] - bytes_ratio) < 1e-9 and abs(tm['steps'] - steps_ratio) < 1e-12, f'{bytes_ratio:.4f} vs {steps_ratio:.4f}')
    check('timing: fidelity states the weights and the most KV cache read per decode step',
          f"Each step reads about {round(Wb / 2 ** 30)} GiB of weights and at most {POOL * kv_tok / 2 ** 30:g} GiB of KV cache" in fid and top <= POOL, f'{Wb / 2 ** 30:.2f} GiB')
    check('timing: the KV cache read is small next to the weights (under 11%)', POOL * kv_tok / Wb < 0.11, f'{POOL * kv_tok / Wb * 100:.1f}%')
    check('timing: fidelity states both ratios', f"counting bytes read, the initial setup takes {bytes_ratio:.2f} times as long as paging, against {steps_ratio:.2f} times as many decode steps" in fid, f'{bytes_ratio:.2f} {steps_ratio:.2f}')
    check('playback speeds: 25, 100 and 400 decode steps per second', J("()=>window.__pa.SPEEDS") == {'slow': 25, 'med': 100, 'fast': 400})

    # 10. the minimal implementation runs and prints what its comment says
    code = pg.inner_text('#pa-code')
    check('minimal implementation is at most 40 lines', len(code.strip().splitlines()) <= 40, str(len(code.strip().splitlines())))
    res = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    check('minimal implementation prints its comment', res.stdout.strip() == code.strip().splitlines()[-1].split('# ')[-1].strip(), res.stdout.strip() + res.stderr[-200:])
    check('minimal implementation: 768 blocks of 16 tokens are room for 12,288 tokens', 768 * 16 == POOL and 'room for 12,288 tokens' in code)

    # 10a. scenarios: each result line adds the same requests' decode steps with the initial strategy; a click that changes no number says so
    pg.click('#view-dyn .modes button[data-mode="sandbox"]:not([data-flow=free])')
    pg.wait_for_timeout(300)
    if J("()=>window.__pa.state().S.strat") != 'paged':
        pg.click('#pa-ctl-strat button[data-v="paged"]')
        pg.wait_for_timeout(200)
    SCN = ['Customer support', 'Chat service', 'Long-form writing', 'API with a large default cap', 'Summaries']
    for i, (mu, lim) in enumerate([(128, 1536), (256, 1536), (1024, 2048), (128, 8192), (256, 512)]):
        pg.click(f'#view-dyn [data-preset="{i}"]')
        pg.wait_for_timeout(200)
        pr, mx = simulate('paged', 16, mu, lim), simulate('max', 16, mu, lim)
        line = pg.inner_text('#pa-sbnote')
        steps_card = pg.inner_text('#pa-metrics [data-k="steps"] .v')
        check(f'scenario {mu}/{lim}: one rounded share, and the decode steps against reserving the maximum in words (the steps card has the exact count)',
              line == f"Selected: {SCN[i]}. With paging, about {pc0(pr['util'])} of the pool stores tokens while requests wait, so the requests finish in {steps_vs_max(pr['T'], mx['T'])}."
              and steps_card == f"{pr['T']:,}", line[-240:] + ' | ' + steps_card)
    m128, m8k = simulate('max', 16, 128, 1536), simulate('max', 16, 128, 8192)
    tail = lambda why, up: f"{why}, and no answer here reaches even the lower cap, so {'raising' if up else 'lowering'} the cap changes no number."
    def max_t(up, q):
        diff = 'many more' if q >= 2 else ('more' if q >= 1.1 else ('far fewer' if q <= 0.5 else ('fewer' if q <= 0.9 else 'about as many')))
        return f"Reserving the maximum is different: each request then reserves the {'higher' if up else 'lower'} cap, so the run takes {diff} decode steps."
    whyP, whyX = 'paging hands out the next block only when the last one is full', 'reserving the exact length holds only each request’s final length'
    samek = lambda a_, b_: all(a_[k] == b_[k] for k in ('T', 'util', 'running', 'unused', 'table')) and max(o_ for _, o_ in a_['reqs']) < 1536
    pg.click('#view-dyn [data-preset="0"]')
    pg.wait_for_timeout(200)
    pg.click('#view-dyn [data-preset="3"]')
    pg.wait_for_timeout(200)
    line = pg.inner_text('#pa-sbnote')
    check('scenario (paging): Customer support to the large default cap changes nothing; the card leads with that, says why, and where reserving the maximum differs',
          samek(simulate('paged', 16, 128, 1536), simulate('paged', 16, 128, 8192)) and m8k['T'] != m128['T']
          and line == f"Selected: API with a large default cap. No change from “Customer support”: {tail(whyP, True)} {max_t(True, m8k['T'] / m128['T'])}"
          and max(o_ for _, o_ in m128['reqs']) < 1536, line)
    x128, x8k = simulate('exact', 16, 128, 1536), simulate('exact', 16, 128, 8192)
    pg.click('#view-dyn #pa-ic-strat button[data-v="exact"]')
    pg.wait_for_timeout(200)
    line = pg.inner_text('#pa-sbnote')
    check('scenario: reserving the exact length leads with its decode steps against reserving the maximum',
          line == f"Selected: API with a large default cap. When each request reserves its exact length, about {pc0(x8k['util'])} of the pool stores tokens while requests wait, and the requests finish in {steps_vs_max(x8k['T'], m8k['T'])}.", line)
    pg.click('#view-dyn [data-preset="0"]')
    pg.wait_for_timeout(200)
    line = pg.inner_text('#pa-sbnote')
    check('scenario (exact length): back to Customer support changes nothing, lowering the cap',
          samek(x128, x8k) and line == f"Selected: Customer support. No change from “API with a large default cap”: {tail(whyX, False)} {max_t(False, m128['T'] / m8k['T'])}", line)
    pg.click('#view-dyn [data-preset="3"]')
    pg.wait_for_timeout(200)
    line = pg.inner_text('#pa-sbnote')
    check('scenario (exact length): to the large default cap changes nothing, raising the cap',
          line == f"Selected: API with a large default cap. No change from “Customer support”: {tail(whyX, True)} {max_t(True, m8k['T'] / m128['T'])}", line)
    pg.click('#view-dyn #pa-ic-strat button[data-v="max"]')
    pg.wait_for_timeout(200)
    line = pg.inner_text('#pa-sbnote')
    check('scenario: reserving the maximum gives its rounded share and why; its steps are on the card', line == f"Selected: API with a large default cap. When each request reserves the maximum, only about {pc0(m8k['util'])} of the pool stores tokens while requests wait: each request holds room for the whole answer cap from the start."
          and pg.inner_text('#pa-metrics [data-k="steps"] .v') == f"{m8k['T']:,}", line)
    # a cap change under Settings that changes no number
    pg.click('#view-dyn #pa-ic-strat button[data-v="paged"]')
    pg.click('#view-dyn [data-preset="0"]')
    pg.wait_for_timeout(200)
    pg.click('#pa-ctl-lim button[data-v="2048"]')
    pg.wait_for_timeout(200)
    m2k, c2k = simulate('max', 16, 128, 2048), simulate('paged', 16, 128, 2048)
    same = all(c2k[k] == simulate('paged', 16, 128, 1536)[k] for k in ('T', 'util', 'running', 'unused', 'table'))
    line = pg.inner_text('#pa-sbnote')
    check('scenario: a cap change under Settings that changes no number says so, and why',
          same and line == f"No scenario is selected: answer lengths with a mean of 128 tokens and a cap of 2,048 tokens match none of them. Changing the cap changed nothing: {tail(whyP, True)} {max_t(True, m2k['T'] / m128['T'])}", line)
    pg.click('#view-dyn .modes button[data-mode="guided"]')
    pg.wait_for_timeout(300)

    # 10b. the home tile: the sentence under each option, from the reference numbers (spec example: 1 - 2,898 / 11,210 and 3,882 / 3,920)
    pg.evaluate("()=>{location.hash='map'}")
    pg.wait_for_timeout(500)
    pg.evaluate("()=>document.getElementById('tile-paged-attention').scrollIntoView({block:'center'})")
    pg.wait_for_timeout(200)
    r1 = pg.inner_text('#tile-paged-attention .tz-read')
    check('tile: the spec example values', (fit, held, sum(F8[:fit]), alloc, sum(F8)) == (6, 11210, 2898, 3920, 3882), str((fit, held, sum(F8[:fit]), alloc, sum(F8))))
    waits = pg.evaluate("()=>document.querySelectorAll('#tile-paged-attention .tz-pa-wait').length")
    check('tile: reserve the maximum (one number; the bars that do not fit are drawn waiting)', r1 == f"Each request reserves room for its prompt plus the longest answer allowed: not all 8 requests fit, and {pc0(1 - sum(F8[:fit]) / held)} of the reserved room is never used."
          and fit < 8 and waits == 8 - fit, r1 + f' | {waits} waiting bars')
    pg.click('#tile-paged-attention .seg-ctl button[data-v="paged"]')
    pg.wait_for_timeout(150)
    r2 = pg.inner_text('#tile-paged-attention .tz-read')
    check('tile: 16-token blocks', r2 == f"With 16-token blocks, handed out one at a time as they fill: every request fits, and {pc0(sum(F8) / alloc)} of the room they hold stores tokens.", r2)
    notes = pg.evaluate("()=>[...document.querySelectorAll('#tile-paged-attention .tz-note')].map(x=>x.textContent)")
    check('tile: the notes state the pool and the paper range', len(notes) == 3 and '(128 KiB per token for Llama-3.1-8B, 1.5 GiB in total)' in notes[0] and 'prompts of 100 to 600 tokens, answers of at most 1,536 tokens' in notes[1] and '20.4% to 38.2%' in notes[2], str(notes)[:200])
    br.close()

# 11. "A bigger pool with proportionally more requests gives similar percentages" (fidelity card and Controls):
#     a fast interval/count version of the same rules at 3x and 8x the pool and the request count
def scaled_util(strat, k, B=16, mean=256, limit=1536):
    pool, reqs = POOL * k, []
    rnd = mulberry32(SEED)
    for _ in range(N * k):
        p = PMIN + int(rnd() * (PMAX - PMIN + 1))
        o = max(1, min(limit, int(math.floor(-mean * math.log(1.0 - rnd()) + 0.5))))
        reqs.append((p, o))
    queue, busy_tok, busy_n = list(range(len(reqs))), 0, 0
    if strat == 'max':
        gaps, run = [[0, pool]], {}
        while queue or run:
            while queue:
                need = reqs[queue[0]][0] + limit
                g = next((g for g in gaps if g[1] >= need), None)
                if g is None:
                    break
                i = queue.pop(0)
                run[i] = [g[0], need, reqs[i][0]]
                g[0] += need
                g[1] -= need
                gaps = [g for g in gaps if g[1] > 0]
            for i in run:
                run[i][2] += 1
            if queue:
                busy_tok += sum(v[2] for v in run.values())
                busy_n += 1
            for i in [i for i in run if run[i][2] >= sum(reqs[i])]:
                s, l, _ = run.pop(i)
                gaps = sorted(gaps + [[s, l]])
                merged = []
                for g in gaps:
                    if merged and merged[-1][0] + merged[-1][1] == g[0]:
                        merged[-1][1] += g[1]
                    else:
                        merged.append(g)
                gaps = merged
    else:
        free, running, nb, length, gen = pool // B, [], {}, {}, [0] * len(reqs)
        while queue or running:
            pre, k2 = False, 0
            while k2 < len(running):
                i = running[k2]
                if length[i] + 1 > nb[i] * B:
                    gone = False
                    while free == 0:
                        v = running.pop()
                        free += nb[v]
                        nb[v] = 0
                        queue.insert(0, v)
                        pre = True
                        if v == i:
                            gone = True
                            break
                    if gone:
                        continue
                    nb[i] += 1
                    free -= 1
                k2 += 1
            if not pre:
                while queue:
                    i = queue[0]
                    have = reqs[i][0] + gen[i]
                    need = -(-(have + 1) // B)
                    if need > free:
                        break
                    free -= need
                    nb[i], length[i] = need, have
                    running.append(queue.pop(0))
            for i in running:
                length[i] += 1
                gen[i] += 1
            if queue:
                busy_tok += sum(length[i] for i in running)
                busy_n += 1
            for i in [i for i in running if gen[i] >= reqs[i][1]]:
                running.remove(i)
                free += nb[i]
    return busy_tok / busy_n / pool


for strat in ('max', 'paged'):
    base = scaled_util(strat, 1)
    check(f'fast variant reproduces the 1x {strat} share', abs(base - REF[(strat, 16, 256, 1536)]['util']) < 1e-12)
    for k in (3, 8):
        u = scaled_util(strat, k)
        check(f'{strat}: {k}x pool and requests gives a similar share (within 5 points)', abs(u - base) <= 0.05, f'{pc(base)} -> {pc(u)}')

for line in out:
    print(line)
print('page errors:', errs)
print(f'{len(out) - len(fails)} passed, {len(fails)} failed')
sys.exit(1 if fails or errs else 0)
