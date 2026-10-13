"""Golden numbers for the spec-decoding lab: every number the page shows, recomputed independently in Python and compared with the page's
engine (window.__spd) through Playwright, plus the text the page shows for key states.

Independent derivations:
- parameter counts from the llama-models sku_list.py shapes (ffn_dim_multiplier, multiple_of), not from the lab's table;
- the step model of brief S2 (FLOPs and bytes per step, 50% / 60% of the H100 peaks);
- tokens per target pass by brute-force summation of the acceptance probabilities and by enumerating the distribution of accepted counts;
- the simulators (mulberry32, as capstone/engine.py) reimplemented here with the same seeds;
- the acceptance rule: the output distribution by exact enumeration, and the seeded sampling twin.
Run: python3 tests/spec-decoding_numbers.py"""
import math, pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/spec-decoding.html', '--only', 'spec-decoding'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'spd').mkdir(parents=True, exist_ok=True)
(P / 'out' / 'spd' / 'numbers.html').write_text('<!doctype html><html><head><meta charset="utf-8"></head><body>' + (P / 'out' / 'spec-decoding.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'spd' / 'numbers.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', info)


def close(a, b, rel=1e-9, ab=1e-12):
    return abs(a - b) <= max(ab, rel * max(abs(a), abs(b)))


# ---------- models, from the sku_list arch_args ----------
V = 128256


def ffn(dim, mult, multiple_of):
    h = int(2 * (4 * dim) / 3)
    h = int(mult * h)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


def model(dim, layers, heads, kv, mult, mof, tied):
    hd = dim // heads
    F = ffn(dim, mult, mof)
    per = dim * dim * 2 + 2 * dim * kv * hd + 3 * dim * F + 2 * dim
    PL = layers * per + dim
    PH = V * dim
    P_ = PL + PH + (0 if tied else PH)
    return dict(L=layers, D=dim, PL=PL, PH=PH, P=P_, W=2 * (PL + PH), KV=2 * layers * kv * hd * 2, F=F, hd=hd)


M = {'t8': model(4096, 32, 32, 8, 1.3, 1024, False), 't70': model(8192, 80, 64, 8, 1.3, 4096, False),
     'd1': model(2048, 16, 32, 8, 1.5, 256, True), 'd3': model(3072, 28, 24, 8, 1.0, 256, True)}
check('8B count', M['t8']['P'] == 8030261248, M['t8']['P'])
check('70B count', M['t70']['P'] == 70553706496, M['t70']['P'])
check('3B count (brief S1)', M['d3']['P'] == 3212749824, M['d3']['P'])
check('1B count', M['d1']['P'] == 1235814400, M['d1']['P'])
check('FFN sizes', (M['t8']['F'], M['t70']['F'], M['d1']['F'], M['d3']['F']) == (14336, 28672, 8192, 8192))
GPUS = {'t8': 1, 't70': 4}
FS = 989.5e12 * 0.5
BS = 3.35e12 * 0.6
CTX = 1024
ALPHA = {'t8': {'ngram': 0.3, 'd1': 0.7, 'd3': 0.8}, 't70': {'ngram': 0.3, 'd1': 0.65, 'd3': 0.75}}


def attn_flops(m, a, b):
    # causal attention: query i sees i+1 keys; QK^T and AV each 2·d per key per layer
    return sum(4 * m['L'] * m['D'] * (i + 1) for i in range(a, b))


def step(m, B, ntok, ctx, n):
    fl = 2 * m['PL'] * B * ntok + 2 * m['PH'] * B * ntok + B * attn_flops(m, ctx, ctx + ntok)
    by = m['W'] + m['KV'] * B * (ctx + ntok)
    return max(fl / n / FS, by / n / BS), fl / n / FS, by / n / BS, fl, by


def E_brute(a, k):
    # enumerate: P(exactly j accepted) = a^j (1-a) for j<k, a^k for j=k; tokens = j+1
    return sum((j + 1) * (a ** j) * ((1 - a) if j < k else 1) for j in range(k + 1))


def compute(s):
    T = M[s['tgt']]; n = GPUS[s['tgt']]
    tp = step(T, s['B'], 1, CTX, n)[0]
    if s['dr'] == 'none':
        return dict(E=1, tPlain=tp, tRound=tp, speed=1.0, tps=1 / tp, td=0.0, tv=tp)
    td = 0.0
    if s['dr'] != 'ngram':
        for j in range(s['k']):
            td += step(M[s['dr']], s['B'], 1, CTX + j, n)[0]
    tv = step(T, s['B'], s['k'] + 1, CTX, n)[0]
    E = E_brute(s['a'], s['k'])
    tr = td + tv
    return dict(E=E, tPlain=tp, tRound=tr, speed=E * tp / tr, tps=E / tr, td=td, tv=tv)


def mulberry(seed):
    a = seed & 0xFFFFFFFF

    def imul(x, y):
        return (x * y) & 0xFFFFFFFF

    def rnd():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = imul(t ^ (t >> 15), t | 1)
        t = (t ^ ((t + imul(t ^ (t >> 7), t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rnd


def sim_mean(a, k, nr=2000):
    rnd = mulberry(1000 + round(a * 100) * 10 + k); tot = 0
    for _ in range(nr):
        j = 0
        while j < k and rnd() < a:
            j += 1
        tot += j + 1
    return tot / nr


PHI = (math.sqrt(5) - 1) / 2


def timeline(s, OUT=24):
    r = compute(s)
    if s['dr'] == 'none':
        return OUT * r['tPlain'], OUT * r['tPlain'], OUT, []
    made, t, i, acc = 0, 0.0, 0, []
    while made < OUT:
        i += 1; u = (i * PHI) % 1; j = 0
        while j < s['k'] and s['a'] ** (j + 1) > u:
            j += 1
        acc.append(j); t += r['tRound']; made += j + 1
    return OUT * r['tPlain'], t, made, acc


p_ = [0.40, 0.25, 0.15, 0.12, 0.08]; q_ = [0.55, 0.15, 0.20, 0.05, 0.05]


def acc_exact():
    out = [0.0] * 5
    rest = [max(0.0, p_[i] - q_[i]) for i in range(5)]; rs = sum(rest)
    for x in range(5):            # the draft proposes x with q(x); accepted with min(1, p/q), else redraw from the rest
        a = min(1.0, p_[x] / q_[x])
        out[x] += q_[x] * a
        for y in range(5):
            out[y] += q_[x] * (1 - a) * rest[y] / rs
    return out


def acc_sample(mode, N=10000):
    rnd = mulberry(11); cnt = [0] * 5
    rest = [max(0.0, p_[i] - q_[i]) for i in range(5)]; rs = sum(rest); res = [x / rs for x in rest]

    def pick(w):
        u = rnd(); c = 0.0
        for i, x in enumerate(w):
            c += x
            if u < c:
                return i
        return len(w) - 1
    for _ in range(N):
        x = pick(q_)
        if mode == 'nocheck':
            cnt[x] += 1; continue
        if mode == 'greedy':
            cnt[0] += 1; continue
        if rnd() < min(1.0, p_[x] / q_[x]):
            cnt[x] += 1
        else:
            cnt[pick(res)] += 1
    return [c / N for c in cnt]


with sync_playwright() as pw:
    b = pw.chromium.launch(); pg = b.new_page()
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    J = lambda expr, *a: pg.evaluate(expr, *a)
    mods = J('()=>window.__spd.models()')
    for k_, m in M.items():
        for f in ('P', 'W', 'KV', 'PL', 'PH'):
            check(f'model {k_} {f}', mods[k_][f] == m[f], (mods[k_][f], m[f]))
    # the facts the copy states
    check('8B weights read 15.0 GB', f"{M['t8']['W']/1e9:.1f}" == '15.0')
    check('8B KV 128 KiB per token', M['t8']['KV'] == 128 * 1024)
    check('70B KV 320 KiB per token', M['t70']['KV'] == 320 * 1024)
    # every setting
    for tgt in ('t8', 't70'):
        for dr in ('none', 'ngram', 'd1', 'd3'):
            for k in (1, 2, 3, 4, 5, 6, 7, 8):
                for B in (1, 4, 8, 16, 32, 64, 128):
                    a = ALPHA[tgt].get(dr, 0.7)
                    s = dict(tgt=tgt, dr=dr, k=k, B=B, a=a)
                    js = J('s=>window.__spd.compute(s)', s); py = compute(s)
                    for f in ('E', 'tPlain', 'tRound', 'speed', 'tps'):
                        if not close(js[f], py[f], 1e-9):
                            check(f'compute {s} {f}', False, (js[f], py[f])); break
                    else:
                        check('compute', True)
                    mf = J('s=>window.__spd.memFit(s)', s)
                    check(f'fits in memory {s}', mf['ok'], mf)
    # the step-cost rows of part 2
    for ntok, (B, t_) in {1: (1, 1), 5: (1, 5), 80: (16, 5), 128: (128, 1), 640: (128, 5)}.items():
        js = J('a=>window.__spd.step("t8",a[0],a[1])', [B, t_]); py = step(M['t8'], B, t_, CTX, 1)
        check(f'cost row {ntok}', close(js['t'], py[0]) and close(js['tc'], py[1]) and close(js['tm'], py[2]), (js, py))
    r1 = step(M['t8'], 1, 1, CTX, 1); r5 = step(M['t8'], 1, 5, CTX, 1); r80 = step(M['t8'], 16, 5, CTX, 1); r128 = step(M['t8'], 128, 1, CTX, 1); r640 = step(M['t8'], 128, 5, CTX, 1)
    check('1 token 7.53 ms', f'{r1[0]*1e3:.2f}' == '7.53', r1[0])
    check('5 tokens 7.53 ms (same as 1)', f'{r5[0]*1e3:.2f}' == '7.53', r5[0])
    check('80 tokens (16 x 5) still memory-bound, 8.54 ms', r80[1] < r80[2] and f'{r80[0]*1e3:.2f}' == '8.54', r80)
    check('plain decode of 128 requests: memory-bound, 16.0 ms (the baseline of 640)', r128[1] < r128[2] and f'{r128[0]*1e3:.1f}' == '16.0', r128)
    check('640 tokens (128 x 5) compute-bound, 20.1 ms', r640[1] > r640[2] and f'{r640[0]*1e3:.1f}' == '20.1', r640)
    # E: formula = brute force = page
    for a in (0.3, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9):
        for k in range(1, 9):
            f_ = (1 - a ** (k + 1)) / (1 - a)
            check(f'E formula {a} {k}', close(f_, E_brute(a, k)) and close(J('a=>window.__spd.expTok(a[0],a[1])', [a, k]), f_))
    check('E(0.8,4) = 3.36', f'{E_brute(0.8, 4):.2f}' == '3.36')
    check('E(0.9,4) = 4.10', f'{E_brute(0.9, 4):.2f}' == '4.10')
    check('prediction distractor 0.8×4+1 = 4.2', f'{0.8*4+1:.1f}' == '4.2')
    check('0.8^4 about 41%', f'{0.8**4*100:.0f}' == '41'); check('0.8^8 about 17%', f'{0.8**8*100:.0f}' == '17')
    # simulated dots
    for a in (0.6, 0.7, 0.8, 0.9):
        for k in range(1, 9):
            js = J('a=>window.__spd.simMean(a[0],a[1])', [a, k]); py = sim_mean(a, k)
            check(f'sim {a} {k}', close(js, py), (js, py))
            check(f'sim near formula {a} {k}', abs(py - E_brute(a, k)) < 0.15, (py, E_brute(a, k)))
    # timeline
    for s in ({'dr': 'd1', 'k': 4, 'B': 1, 'a': 0.7}, {'dr': 'd3', 'k': 4, 'B': 64, 'a': 0.8}, {'dr': 'ngram', 'k': 4, 'B': 8, 'a': 0.6}, {'dr': 'd1', 'k': 3, 'B': 64, 'a': 0.7}, {'dr': 'd1', 'k': 4, 'B': 128, 'a': 0.7}):
        s2 = dict(tgt='t8', **s)
        js = J('s=>window.__spd.timeline(s)', s2); pl, sp, made, acc = timeline(s2)
        check(f'timeline {s}', close(js['plain'], pl) and close(js['spec'], sp) and js['made'] == made and js['acc'] == acc, (js, pl, sp, made, acc))
        r = compute(s2)
        # the evenly spread run agrees with the speed card in direction, and within 12%
        note = J('s=>window.__spd.tlNote(s)', s2)
        agree = (sp < pl) == (r['speed'] > 1.0) and abs(r['speed'] - 1) >= 0.005
        check(f'timeline agrees with the speed card, or the page says why {s}', bool(note) != agree, (pl, sp, r['speed'], note))
        check(f'timeline near the average {s}', abs(pl / sp * made / 24 - r['speed']) / r['speed'] < 0.12, (pl / sp * made / 24, r['speed']))
    # acceptance rule
    ex = acc_exact()
    check('rule reproduces p exactly', all(close(ex[i], p_[i]) for i in range(5)), ex)
    acc = J('()=>window.__spd.accRule()')
    check('alpha of the position 0.80', close(acc['al'], sum(min(p_[i], q_[i]) for i in range(5))) and f"{acc['al']:.2f}" == '0.80')
    for i in range(5):
        check(f'rule output {i}', close(acc['out'][i], p_[i]))
    for mode in ('greedy', 'sample', 'nocheck'):
        js = J('m=>window.__spd.accFreq(m)', mode); py = acc_sample(mode)
        check(f'acc sample {mode}', all(close(js[i], py[i]) for i in range(5)), (js, py))
    smp = acc_sample('sample')
    check('sampled within 1.5 points of p', max(abs(smp[i] - p_[i]) for i in range(5)) < 0.015, smp)
    nc = acc_sample('nocheck')
    check('no check is off by 15 points on token 1', abs(max(abs(q_[i] - p_[i]) for i in range(5)) - 0.15) < 1e-12 and abs(nc[0] - p_[0]) > 0.1)
    # challenge: the 70B on 4 H100s at 128 requests, which the guided parts never price (review b10r1 item 19)
    A70 = ALPHA['t70']
    rows = J('()=>window.__spd.chAll()')
    py_rows = [('none', 0, 1.0)] + [(d, k, compute(dict(tgt='t70', B=128, dr=d, k=k, a=A70[d]))['speed']) for d in ('ngram', 'd1', 'd3') for k in range(1, 9)]
    check('challenge rows', len(rows) == len(py_rows) and all(r['dr'] == p[0] and r['k'] == p[1] and close(r['speed'], p[2]) for r, p in zip(rows, py_rows)))
    best = max(py_rows, key=lambda x: x[2])
    jb = J('()=>window.__spd.chBest()')
    check('challenge best 1B k=2', best[0] == 'd1' and best[1] == 2 and jb['dr'] == 'd1' and jb['k'] == 2, (best, jb))
    okset = [(p[0], p[1]) for p in py_rows if p[2] >= 1.5]
    check('challenge passing set: 1B and 3B with k = 1 or 2', okset == [('d1', 1), ('d1', 2), ('d3', 1), ('d3', 2)], okset)
    init = compute(dict(tgt='t70', B=128, dr='d3', k=4, a=A70['d3']))
    check('challenge initial setup (3B, k = 4) misses the goal: 1.22 times', f"{init['speed']:.2f}" == '1.22', init['speed'])
    k4 = {d: compute(dict(tgt='t70', B=128, dr=d, k=4, a=A70[d]))['speed'] for d in ('ngram', 'd1', 'd3')}
    check('the k = 4 batch chart of the 70B does not answer the challenge (no guesser reaches 1.5 at 128)', max(k4.values()) < 1.5, k4)
    pl = step(M['t70'], 128, 1, CTX, 4); c1 = step(M['t70'], 128, 2, CTX, 4); c2 = step(M['t70'], 128, 3, CTX, 4)
    check('why: 70B plain decode at 128 is memory-bound', pl[1] < pl[2], pl)
    check('why: a check of 2 tokens per request is still memory-bound, of 3 tokens compute-bound', c1[1] < c1[2] and c2[1] > c2[2], (c1, c2))
    s1 = step(M['d1'], 128, 1, CTX, 4); s3 = step(M['d3'], 128, 1, CTX, 4)
    kvb = lambda m: m['KV'] * 128 * (CTX + 1)
    check('why: draft steps are mostly reading their own KV cache, and cheap next to a 70B step', kvb(M['d1']) > M['d1']['W'] and kvb(M['d3']) > M['d3']['W'] and s3[0] < 0.2 * pl[0], (s1, s3, pl))
    check('why: the 3B step costs about three times the 1B step', 2.5 < s3[0] / s1[0] < 3.5, s3[0] / s1[0])
    ngb = max(p[2] for p in py_rows if p[0] == 'ngram')
    check('why: n-gram lookup stays under 1.5', ngb < 1.5, ngb)
    # part 7 and the challenge of batch 64 (review b10r1 items 1-3): the check stays memory-bound, the drafts get dearer
    for B in (32, 64):
        for k in (3, 4):
            v = step(M['t8'], B, k + 1, CTX, 1)
            check(f'8B check at batch {B}, k = {k}: memory-bound', v[1] < v[2], v)
    cb = J('()=>window.__spd.bCB()')
    pyb = next(B for B in range(1, 513) if (lambda v: v[1] > v[2])(step(M['t8'], B, 5, CTX, 1)))
    check('the batch from which the 8B check of 5 tokens is compute-bound (part 7 card)', cb == pyb and 64 < cb <= 128, (cb, pyb))
    b32 = compute(dict(tgt='t8', dr='d3', k=4, B=32, a=0.8))
    check('batch 32: the 3B takes more than two 8B steps', b32['td'] > 2 * b32['tPlain'], (b32['td'], b32['tPlain']))
    check('3B KV cache per token is almost the 8B one (112 vs 128 KiB)', M['d3']['KV'] == 112 * 1024 and M['t8']['KV'] == 128 * 1024)
    b128f = compute(dict(tgt='t8', dr='d3', k=4, B=128, a=0.8))
    free = b128f['E'] * b128f['tPlain'] / (b128f['td'] + b128f['tPlain'])
    check('batch 128: the 3B is below 1 even with a free check', free < 1, free)
    # numbers near 1 are not rounded to 1.00 (review item 12)
    check('a speed of 0.9985 shows three decimals', J('v=>window.__spd.xT(v)', 0.9985).startswith('0.998') or J('v=>window.__spd.xT(v)', 0.9985).startswith('0.999'))
    # the parts' claims
    b1 = compute(dict(tgt='t8', dr='d1', k=4, B=1, a=0.7)); b128 = compute(dict(tgt='t8', dr='d3', k=4, B=128, a=0.8)); b128_1 = compute(dict(tgt='t8', dr='d1', k=4, B=128, a=0.7))
    check('1 user, 1B: 1.67 times', f"{b1['speed']:.2f}" == '1.67', b1['speed'])
    check('1B draft 4 steps about 66% of an 8B step', f"{b1['td']/b1['tPlain']*100:.0f}" == '66', b1['td'] / b1['tPlain'])
    check('batch 128, 3B slower than plain decode', b128['speed'] < 1, b128['speed'])
    check('batch 128, 1B still faster', b128_1['speed'] > 1, b128_1['speed'])
    b3_1 = compute(dict(tgt='t8', dr='d3', k=4, B=1, a=0.8))
    check('3B 4 steps longer than one 8B step', b3_1['td'] > b3_1['tPlain'], (b3_1['td'], b3_1['tPlain']))
    c70 = compute(dict(tgt='t70', dr='d3', k=4, B=1, a=0.75)); c70_1 = compute(dict(tgt='t70', dr='d1', k=4, B=1, a=0.65))
    check('70B: 3B beats 1B at batch 1', c70['speed'] > c70_1['speed'], (c70['speed'], c70_1['speed']))
    check('new case: 70B + 1B faster than 8B + 1B', c70_1['speed'] > b1['speed'], (c70_1['speed'], b1['speed']))
    # the speedup on 4 GPUs equals the one-GPU value (communication left out)
    one = dict(M=M)
    t70_1 = step(M['t70'], 1, 1, CTX, 1)[0]; t70_4 = step(M['t70'], 1, 1, CTX, 4)[0]
    check('70B: 4 GPUs divide every step by 4', close(t70_1 / 4, t70_4))
    # drill answers
    for sd in range(6):
        d = J('x=>window.__spd.drill(x)', sd); r = compute(dict(tgt='t8', **{k: d['s'][k] for k in ('dr', 'k', 'B', 'a')}))
        check(f'drill {sd}', close(d['E'], r['E']) and close(d['speed'], r['speed']) and close(d['td'] + d['tv'], r['tRound']))
    check('no page errors', not errs, errs)
    b.close()
print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
