"""rlvr numbers: an independent Python version of the lab's toy GRPO run, compared with the page's engine (window.__rlvr).

The toy (specs/rlvr.md): a policy = softmax over 4 ways of writing x Bernoulli("writes ####"), 5 logits. Each step: 8 problems x 8
answers drawn with mulberry32(11) in the order kind, then per answer (way, format, correctness draw); reward 1/0 (+0.1 format reward);
advantage (r - mean) / (std_{n-1} + 1e-4) as TRL; loss = mean over the batch of -A + beta * k3 (ratio 1, one update per batch);
plain gradient step with learning rate 2 on the logits. Dynamic sampling (verl DAPO filter_groups) drops groups with equal rewards.
Here the gradient of log pi is written out from the softmax / sigmoid derivatives, the curves from expectations over problem kinds.

Run: python3 tests/rlvr_numbers.py   (builds out/rlvr.html with --only rlvr first)
"""
import io, json, math, pathlib, re, subprocess, sys, contextlib, itertools
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'out' / 'rlvr'; OUT.mkdir(parents=True, exist_ok=True)

# ---------------- the lab's assumptions, restated ----------------
LEN = [25, 140, 240, 60]
Q_STRICT = [[0.40, 0.10, 0.00, 0.0], [0.80, 0.52, 0.22, 0.0], [0.95, 0.80, 0.50, 0.0], [0.30, 0.08, 0.02, 0.0]]   # [way][kind]
Q_LOOSE = Q_STRICT[:3] + [[0.98, 0.95, 0.85, 0.0]]
SETS = {'main': [0.4, 0.4, 0.1, 0.1], 'hard': [0.0, 0.1, 0.5, 0.4], 'ch': [0.05, 0.25, 0.4, 0.3]}
P0 = [0.68, 0.25, 0.05, 0.02]
FMT_LOGIT = {'sft': 50.0, 'base': math.log(0.1 / 0.9)}
G = B = 8; BETA = 0.05; LR = 2.0; SEED = 11; T = 60; TCH = 20; MAXTRY = 64; FMT = 0.1
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


class Rng:
    """mulberry32, written from its definition with explicit 32-bit wrapping"""
    def __init__(self, seed): self.a = seed & 0xFFFFFFFF
    def __call__(self):
        self.a = (self.a + 0x6D2B79F5) & 0xFFFFFFFF
        a = self.a
        t = ((a ^ (a >> 15)) * (a | 1)) & 0xFFFFFFFF
        t = (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 2**32


def categorical(u, probs):
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if u < c: return i
    return len(probs) - 1


def softmax(z):
    m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e)
    return [v / s for v in e]


def sigmoid(x): return 1 / (1 + math.exp(-x))


def expectations(pi, pf, w, chk, fmt):
    right = [pf * sum(pi[s] * Q_STRICT[s][k] for s in range(4)) for k in range(4)]       # pass@1 per kind (strict)
    p1 = sum(w[k] * right[k] for k in range(4))
    ok_any = sum(w[k] * pi[s] * Q_STRICT[s][k] for s in range(4) for k in range(4))
    ok_loose = sum(w[k] * pi[s] * Q_LOOSE[s][k] for s in range(4) for k in range(4))
    if chk == 'loose': paid = ok_loose + (FMT * pf * (1 - ok_loose) if fmt else 0)
    else: paid = pf * ok_any + (FMT * pf * (1 - ok_any) if fmt else 0)
    return right, p1, paid, sum(p * l for p, l in zip(pi, LEN))


def kl_exact(pi, pf, pr, pfr):
    k = sum(p * math.log(p / q) for p, q in zip(pi, pr) if p > 0)
    for a, b in ((pf, pfr), (1 - pf, 1 - pfr)):
        if a > 0: k += a * math.log(a / b)
    return k


def train(chk='strict', fmt=0, dyn=0, start='sft', set='main', T=T):
    w = SETS[set]; rnd = Rng(SEED)
    theta = [math.log(p) for p in P0]; phi = FMT_LOGIT[start]
    ref_pi, ref_pf = softmax(theta), sigmoid(phi)
    steps = []
    for t in range(T + 1):
        pi, pf = softmax(theta), sigmoid(phi)
        right, p1, paid, avg_len = expectations(pi, pf, w, chk, fmt)
        groups, kept, tries = [], 0, 0
        while kept < B and tries < MAXTRY:
            tries += 1
            kind = categorical(rnd(), w)
            answers = []
            for _ in range(G):
                way = categorical(rnd(), pi); wrote = 1 if rnd() < pf else 0; u = rnd()
                ok_s, ok_l = u < Q_STRICT[way][kind], u < Q_LOOSE[way][kind]
                if chk == 'loose': r = 1.0 if ok_l else (FMT if fmt and wrote else 0.0)
                else: r = 1.0 if (wrote and ok_s) else (FMT if fmt and wrote else 0.0)
                answers.append([way, wrote, int(ok_s), int(ok_l), r])
            rewards = [a[4] for a in answers]
            mean = sum(rewards) / G
            std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / (G - 1))
            for a in answers: a.append((a[4] - mean) / (std + 1e-4))
            same = len(set_(rewards)) == 1
            drop = bool(dyn) and same
            groups.append({'kind': kind, 'resp': answers, 'm': mean, 'sd': std, 'flat': same, 'drop': drop})
            if not drop: kept += 1
            if not dyn and kept >= B: break
        steps.append({'pi': pi, 'pf': pf, 'p1': p1, 'pk': right, 'rew': paid, 'len': avg_len, 'kl': kl_exact(pi, pf, ref_pi, ref_pf),
                      'groups': groups, 'tries': tries})
        used = [g for g in groups if not g['drop']]
        if t == T or not used: continue
        n = len(used) * G
        # d loss / d theta_k = mean over answers of coef * d log pi(answer) / d theta_k, coef = -A + beta * (1 - ref/pi) (k3 gradient)
        d_theta = [0.0] * 4; d_phi = 0.0
        for g in used:
            for way, wrote, _, _, _, adv in g['resp']:
                p_ans = pi[way] * (pf if wrote else 1 - pf)
                q_ans = ref_pi[way] * (ref_pf if wrote else 1 - ref_pf)
                coef = -adv + BETA * (1 - q_ans / p_ans)
                for k in range(4): d_theta[k] += coef * ((1.0 if k == way else 0.0) - pi[k]) / n
                d_phi += coef * (wrote - pf) / n
        theta = [th - LR * d for th, d in zip(theta, d_theta)]
        phi -= LR * d_phi
    return steps


def set_(xs): return {round(x, 12) for x in xs}


def pass_k(right, w): return [sum(w[i] * (1 - (1 - right[i]) ** k) for i in range(4)) for k in KS]


# ---------------- compare with the page ----------------
def main():
    subprocess.run([sys.executable, str(ROOT / 'build.py'), 'out/rlvr.html', '--only', 'rlvr'], cwd=ROOT, check=True, capture_output=True)
    html = (ROOT / 'out/rlvr.html').read_text()
    wrapped = OUT / 'numbers-wrapped.html'
    wrapped.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + html + '</body></html>')
    ok = fail = 0
    def check(name, cond, info=''):
        nonlocal ok, fail
        if cond: ok += 1
        else: fail += 1; print('FAIL', name, info)
    close = lambda a, b, tol=1e-9: abs(a - b) <= tol
    with sync_playwright() as p:
        br = p.chromium.launch(); pg = br.new_page()
        errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(wrapped.as_uri() + '#rlvr'); pg.wait_for_function('()=>!!window.__rlvr')
        C = pg.evaluate('()=>window.__rlvr.consts')
        check('consts', C['LEN'] == LEN and C['QS'] == Q_STRICT and C['QL'] == Q_LOOSE and C['SETS'] == SETS and C['P0'] == P0 and
              C['G'] == G and C['B'] == B and C['BETA'] == BETA and C['LR'] == LR and C['SEED'] == SEED and C['T'] == T and C['TCH'] == TCH
              and C['MAXTRY'] == MAXTRY and C['FMT'] == FMT and C['KS'] == KS and C['CH_GOAL'] == 0.40 and C['CH_G'] == 0.05, C)
        # every configuration the Settings can reach (2^5 = 32), and the challenge's 8 options
        cfgs = [dict(chk=a, fmt=b, dyn=c, start=d, set=e) for a, b, c, d, e in itertools.product(['strict', 'loose'], [0, 1], [0, 1], ['sft', 'base'], ['main', 'hard'])]
        cfgs += [dict(chk=a, fmt=b, dyn=c, start='sft', set='ch', T=TCH) for a, b, c in itertools.product(['strict', 'loose'], [0, 1], [0, 1])]
        worst = 0.0
        for c in cfgs:
            js = pg.evaluate('c=>window.__rlvr.run(c)', c)
            py = train(**c)
            check(f'steps {c}', len(js) == len(py))
            for t, (a, b) in enumerate(zip(js, py)):
                for key in ('p1', 'rew', 'len', 'kl', 'pf'):
                    worst = max(worst, abs(a[key] - b[key])); check(f'{key} {c} t={t}', close(a[key], b[key], 1e-9), (a[key], b[key]))
                for i in range(4):
                    check(f'pi {c} t={t}', close(a['pi'][i], b['pi'][i], 1e-9)); check(f'pk {c} t={t}', close(a['pk'][i], b['pk'][i], 1e-9))
                check(f'tries {c} t={t}', a['tries'] == b['tries'], (a['tries'], b['tries']))
                if t in (0, 1, len(py) // 2, len(py) - 1):
                    same = len(a['groups']) == len(b['groups']) and all(
                        ga['kind'] == gb['kind'] and ga['drop'] == gb['drop'] and ga['flat'] == gb['flat'] and close(ga['m'], gb['m']) and close(ga['sd'], gb['sd']) and
                        all(ra[:4] == rb[:4] and close(ra[4], rb[4]) and close(ra[5], rb[5], 1e-7) for ra, rb in zip(ga['resp'], gb['resp']))
                        for ga, gb in zip(a['groups'], b['groups']))
                    check(f'groups {c} t={t}', same)
        # the numbers the copy quotes
        main_ = train(); loose = train(chk='loose'); base = train(start='base'); basef = train(start='base', fmt=1); hard = train(set='hard')
        w = SETS['main']
        way_avg = [sum(w[k] * Q_STRICT[s][k] for k in range(4)) for s in range(4)]
        check('prediction averages 20/55/75%', [round(x * 100) for x in way_avg[:3]] == [20, 55, 75], way_avg)
        r = lambda x: round(x * 100)
        facts = {'main p1 31% -> 73%': (r(main_[0]['p1']), r(main_[-1]['p1'])) == (31, 73),
                 'main check 5% -> 93%': (r(main_[0]['pi'][2]), r(main_[-1]['pi'][2])) == (5, 93),
                 'main length 65 -> 230 tokens': (round(main_[0]['len']), round(main_[-1]['len'])) == (65, 230),
                 'loose candidates 2% -> 93%, p1 falls to 19%': (r(loose[-1]['pi'][3]), r(loose[-1]['p1'])) == (93, 19),
                 'loose p1 peaks above its start then falls': max(s['p1'] for s in loose) > loose[0]['p1'] > loose[-1]['p1'],
                 'base start 3%; step 10: 16% without, 45% with format reward': (r(base[0]['p1']), r(base[10]['p1']), r(basef[10]['p1'])) == (3, 16, 45),
                 'base end about 73% either way': abs(base[-1]['p1'] - basef[-1]['p1']) < 0.02,
                 'too-hard set starts at 6%': r(hard[0]['p1']) == 6,
                 # added in the b9v1 fix pass (part 5 reason, part 6 chart marks, Show the math, challenge feedback)
                 'part 5: on this set the loose checker pays candidates 86%, solve-then-check 75%': (r(sum(w[k] * Q_LOOSE[3][k] for k in range(4))), r(sum(w[k] * Q_LOOSE[2][k] for k in range(4)))) == (86, 75),
                 'part 6: step 10 “####” share 41% without, 91% with the format reward': (r(base[10]['pf']), r(basef[10]['pf'])) == (41, 91),
                 'part 6: step 60 right in one try rounds to 73% with and without': r(base[-1]['p1']) == r(basef[-1]['p1']) == 73,
                 'Show the math: exact KL at step 60 is 2.55 nats': round(main_[-1]['kl'], 2) == 2.55,
                 'challenge feedback: from the initial setup the candidates reach 10% (limit 5%)': r(train(chk='loose', start='sft', set='ch', T=TCH)[-1]['pi'][3]) == 10}
        for k, v in facts.items(): check(k, v)
        print('quoted:', {k: v for k, v in facts.items()})
        # pass@k
        for c in (dict(), dict(chk='loose'), dict(set='hard')):
            js = pg.evaluate('c=>window.__rlvr.passK(c)', c); py = train(**c); ww = SETS[c.get('set', 'main')]
            b0, b1 = pass_k(py[0]['pk'], ww), pass_k(py[-1]['pk'], ww)
            check(f'passK {c}', all(close(x, y) for x, y in zip(js['before'], b0)) and all(close(x, y) for x, y in zip(js['after'], b1)))
        pk0, pk1 = pass_k(main_[0]['pk'], w), pass_k(main_[-1]['pk'], w)
        check('pass@256 ceiling 90% before and after', r(pk0[-1]) == 90 and r(pk1[-1]) == 90, (pk0[-1], pk1[-1]))
        print('pass@1 / pass@256 main: before %.3f / %.3f, after %.3f / %.3f' % (pk0[0], pk0[-1], pk1[0], pk1[-1]))
        # the challenge: 8 options, minimal answer strict + skip
        res = {}
        for chk_, dyn_, fmt_ in itertools.product(['loose', 'strict'], [0, 1], [0, 1]):
            e = pg.evaluate('o=>window.__rlvr.chEval(o)', dict(chk=chk_, dyn=dyn_, fmt=fmt_))
            py = train(chk=chk_, fmt=fmt_, dyn=dyn_, start='sft', set='ch', T=TCH)[-1]
            okp = py['p1'] >= 0.40; okg = py['pi'][3] <= 0.05
            check(f'chEval {chk_} dyn={dyn_} fmt={fmt_}', close(e['p1'], py['p1']) and close(e['g'], py['pi'][3]) and e['ok'] == (okp and okg), (e, py['p1']))
            res[(chk_, dyn_, fmt_)] = (round(py['p1'], 3), round(py['pi'][3], 3), okp and okg)
        print('challenge:', res)
        best = pg.evaluate('()=>window.__rlvr.chBest()')
        check('challenge minimal answer is strict + skip', [(b['chk'], b['dyn'], b['fmt']) for b in best] == [('strict', 1, 0)], best)
        check('only strict+skip options pass', sorted(k for k, v in res.items() if v[2]) == [('strict', 1, 0), ('strict', 1, 1)], res)
        # the drill
        for s in range(1, 9):
            d = pg.evaluate('s=>window.__rlvr.drill(s)', s)
            rnd = Rng(s * 7919 + 5); k = 1 + int(rnd() * 6); rs = [1.0] * k + [0.0] * (8 - k); m = k / 8
            sd = math.sqrt(sum((x - m) ** 2 for x in rs) / 7)
            check(f'drill {s}', d['k'] == k and close(d['ap'], (1 - m) / (sd + 1e-4)) and close(d['an'], -m / (sd + 1e-4)), d)
        # the teaser's values and its fallback
        for chk_ in ('strict', 'loose'):
            py = train(chk=chk_)
            td = pg.evaluate('c=>window.__tzRlvr?window.__tzRlvr.data(c):null', chk_)
            fb = pg.evaluate('c=>window.__tzRlvr?window.__tzRlvr.fallback[c]:null', chk_)
            check(f'teaser data {chk_}', td is not None and all(close(td['pi'][i], py[-1]['pi'][i]) for i in range(4)) and close(td['p1'], py[-1]['p1']) and close(td['p10'], py[0]['p1']))
            check(f'teaser fallback {chk_}', fb is not None and all(abs(fb['pi'][i] - py[-1]['pi'][i]) < 6e-5 for i in range(4)) and abs(fb['p1'] - py[-1]['p1']) < 6e-5 and abs(fb['p10'] - py[0]['p1']) < 6e-5, fb)
        # the code block in Connect runs and prints what the page says it prints
        code = pg.evaluate("()=>{const el=document.querySelector('#rlvr-code'); return el?el.textContent:''}")
        check('code block present', 'def advantages' in code)
        printed_claim = [l[2:] for l in code.splitlines() if l.startswith('# strict') or l.startswith('# loose')]
        body = '\n'.join(l for l in code.splitlines() if not (l.startswith('# strict') or l.startswith('# loose')))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): exec(body, {})
        check('code block output', buf.getvalue().strip().splitlines() == printed_claim, (buf.getvalue(), printed_claim))
        # part 2 (codex-b9v3 #1): the picture reads the three example answers with both checkers. Recheck, in both languages, on the
        # answers' own texts: what a strict checker (the number on the final “####” line) and a loose checker (every number) read and
        # pay, written here from their definitions; the Connect code's strict() and loose() must agree; only a wrong answer that the
        # loose checker pays is marked as gamed.
        def strict_py(lines, ref):
            m = re.fullmatch(r'#### (\d+)', lines[-1])
            return ([int(m.group(1))] if m else []), (1 if m and int(m.group(1)) == ref else 0)
        def loose_py(lines, ref):
            nums = [int(x) for l in lines for x in re.findall(r'\d+', l)]
            return nums, (1 if ref in nums else 0)
        connect = {}
        with contextlib.redirect_stdout(io.StringIO()): exec(body, connect)
        WANT = {1: ([20], [3, 6, 18, 18, 2, 20, 20], 20), 0: ([18], [18], 18), 3: ([18], [24, 20, 18, 18], 18)}   # strict reads, loose reads, final answer
        for lg in ('en', 'zh'):
            if lg == 'zh': pg.click('.langsw button[data-lang="zh"]'); pg.wait_for_timeout(300)
            ex = pg.evaluate('()=>window.__rlvr.ex2()')
            check(f'part 2 ({lg}): the three answers in the order of the options', [e['v'] for e in ex] == [1, 0, 3], [e['v'] for e in ex])
            for e in ex:
                v, ref = e['v'], e['ref']; sr, sp = strict_py(e['lines'], ref); lr, lp = loose_py(e['lines'], ref); text_ = '\n'.join(e['lines'])
                check(f'part 2 ({lg}) answer {v}: reference answer 20 and the final answer {WANT[v][2]}', ref == 20 and e['fin'] == WANT[v][2], e)
                check(f'part 2 ({lg}) answer {v}: the strict checker reads {WANT[v][0]} and pays {sp}', sr == WANT[v][0] and e['strictRead'] == sr and e['strictPay'] == sp == e['flags']['strict'], e)
                check(f'part 2 ({lg}) answer {v}: the loose checker reads {WANT[v][1]} and pays {lp}', lr == WANT[v][1] and e['looseRead'] == lr and e['loosePay'] == lp == e['flags']['loose'], e)
                check(f'part 2 ({lg}) answer {v}: the Connect code’s checkers give the same scores', connect['strict'](text_, str(ref)) == sp and connect['loose'](text_, str(ref)) == lp, text_)
                check(f'part 2 ({lg}) answer {v}: red (a wrong answer paid) only for the candidate numbers', e['gamed'] == (v == 3) == (lp == 1 and e['fin'] != ref), e)
            print(f'part 2 ({lg}):', [(e['v'], e['strictPay'], e['loosePay'], e['gamed']) for e in ex])
        check('no page errors', not errs, errs)
        br.close()
    print(f'worst |page - python| = {worst:.2e}')
    print(f'numbers: {ok} passed, {fail} failed')
    sys.exit(1 if fail else 0)


if __name__ == '__main__':
    main()
