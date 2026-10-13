"""Numbers test for the backprop lab (CONTRACT §6): an independent Python derivation of every number the page computes.

- the engine: forward values, the backward order (reverse topological, depth-first from the loss, inputs in written order), every
  backward step's contributions and running gradients, for both networks with every bug switch (+= or =, tanh′ as 1 − z², h clipped
  in place in a small engine, h clipped in place in PyTorch);
- the gradient check: central differences for every parameter and h, relative errors, pass below 10⁻⁷, suspect above 10⁻⁴;
- the challenge: its forward values, the four bugs' failing sets, which bug a seed hides (mulberry32 ported), which bugs fit a set of
  checks, the fewest checks that single each bug out;
- the capability layer's new cases (a sum of three contributions; θ³'s central difference at h = 0.1 and 0.01);
- the seeded interview drill for 40 seeds; the Connect code runs as Python and prints the engine's numbers;
- the teaser's numbers (dy/dw = (1 − y²)·x) at every slider position;
- the formula box beside the graph: every line of numbers it shows works out to the shown 4 decimals, at every step of every run;
- the copy: no model size in part 4 (review round 3), and the numbers the copy states.
Run: python3 tests/backprop_numbers.py"""
import math, pathlib, re, subprocess, sys
from itertools import combinations
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/backprop-num.html', '--only', 'backprop'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'backprop-num-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'backprop-num.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'backprop-num-wrapped.html')
SRC = (P / 'labs' / 'backprop.js').read_text(encoding='utf-8')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


close = lambda a, b, tol=1e-9: a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(a), abs(b))

# ---------------------------------------------------------------- the engine, written independently
NETS = {
    'two': dict(x1=1.5, x2=-0.6, t=0.5, P=dict(w1=0.6, w2=0.5, b=0.1, v=1.5), params=['w1', 'w2', 'b', 'v']),
    'tied': dict(x1=1.5, x2=-0.6, t=0.5, P=dict(w=0.6, b=0.1, v=1.5), params=['w', 'b', 'v']),
    'ch': dict(x1=2.0, x2=-0.5, t=1.0, P=dict(w=0.5, b=-0.2, v=1.5, c=0.3), params=['w', 'b', 'v', 'c']),
}
CLIP, HS, PASS, SUSPECT = 0.5, [0.5, 0.1, 0.001, 0.00001], 1e-7, 1e-4


def graph(net):
    """op id -> (kind, [input ids]); leaves are the parameters"""
    wa, wb = ('w1', 'w2') if net == 'two' else ('w', 'w')
    g = {'m1': ('mulc', [wa]), 'm2': ('mulc', [wb]), 'z': ('add', ['m1', 'm2', 'b'])}
    if net == 'ch':
        g.update({'s': ('sig', ['z']), 'y': ('aff', ['s', 'v', 'c'])})
    else:
        g.update({'h': ('tanh', ['z']), 'y': ('mul', ['h', 'v'])})
    g['L'] = ('sq', ['y'])
    return g


def forward(net, P=None):
    X = NETS[net]; p = dict(X['P']); p.update(P or {})
    w1 = p['w1'] if net == 'two' else p['w']; w2 = p['w2'] if net == 'two' else p['w']
    v = dict(p); v['m1'] = w1 * X['x1']; v['m2'] = w2 * X['x2']; v['z'] = v['m1'] + v['m2'] + p['b']
    if net == 'ch':
        v['s'] = 1 / (1 + math.exp(-v['z'])); v['y'] = p['v'] * v['s'] + p['c']
    else:
        v['h'] = math.tanh(v['z']); v['y'] = v['h'] * p['v']
    v['L'] = (v['y'] - X['t']) ** 2
    return v


def order(net):
    g, seen, out = graph(net), set(), []
    def visit(i):
        if i in seen: return
        seen.add(i)
        for k in (g[i][1] if i in g else []): visit(k)
        out.append(i)
    visit('L')
    return [i for i in reversed(out) if i in g]


def backward(net, over=False, tz=False, inpl='none', sg=False, sw=False, half=False):
    X, g = NETS[net], graph(net)
    vals = forward(net)
    changed = None
    if inpl in ('eng', 'pt') and vals['h'] > CLIP:
        changed = (vals['h'], CLIP)
        if inpl == 'eng': vals['h'] = CLIP          # the backward reads the changed h; the loss was computed before
    grads, got, steps = {'L': 1.0}, {'L'}, []
    for i in order(net):
        kind, ins = g[i]
        if inpl == 'pt' and changed and 'h' in ins:
            steps.append(dict(id=i, err=True)); return dict(vals=vals, grads={}, steps=steps, error=len(steps), changed=changed)
        up = grads[i]
        if kind == 'sq': loc = [(1 if half else 2) * (vals['y'] - X['t'])]
        elif kind == 'mul': loc = [vals['v'], vals['h']] if not sw else [vals['h'], vals['v']]
        elif kind == 'aff': loc = [vals['v'], vals['s'], 1.0] if not sw else [vals['s'], vals['v'], 1.0]
        elif kind == 'tanh': loc = [1 - (vals['z'] if tz else vals['h']) ** 2]
        elif kind == 'sig': loc = [1 - vals['s'] ** 2] if sg else [vals['s'] * (1 - vals['s'])]
        elif kind == 'add': loc = [1.0, 1.0, 1.0]
        elif kind == 'mulc': loc = [X['x1'] if i == 'm1' else X['x2']]
        cs = []
        for k, l in zip(ins, loc):
            c = l * up
            grads[k] = c if (over or k not in grads) else grads[k] + c
            got.add(k); cs.append(dict(to=k, local=l, up=up, c=c))
        steps.append(dict(id=i, contribs=cs, grads=dict(grads), got=set(got)))
    return dict(vals=vals, grads={p: grads.get(p) for p in X['params']}, steps=steps, error=None, changed=changed)


def loss_at(net, P=None): return forward(net, P)['L']
def fdiff(net, p, h):
    th = NETS[net]['P'][p]
    return (loss_at(net, {p: th + h}) - loss_at(net, {p: th - h})) / (2 * h)
def rel(a, b): return abs(a - b) / max(abs(a) + abs(b), 1e-12)


# ---------------------------------------------------------------- mulberry32, as in the page
def mulberry(a):
    st = [a & 0xFFFFFFFF]
    def f():
        a = (st[0] + 0x6D2B79F5) & 0xFFFFFFFF; st[0] = a
        t = ((a ^ (a >> 15)) * (1 | a)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return f


CHB = [dict(over=True), dict(sg=True), dict(sw=True), dict(half=True)]
def ch_fails(i):
    r = backward('ch', **CHB[i])
    return [p for p in NETS['ch']['params'] if not rel(r['grads'][p], fdiff('ch', p, 1e-5)) < PASS]
def consistent(hid, S): return [j for j in range(4) if all((p in ch_fails(j)) == (p in ch_fails(hid)) for p in S)]
def min_checks(hid):
    ps = NETS['ch']['params']
    for k in range(0, 5):
        ok = [list(S) for S in combinations(ps, k) if consistent(hid, S) == [hid]]
        if ok: return k, ok
    return None, []


with sync_playwright() as pw:
    b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1280, 'height': 900})
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(200); pg.evaluate("()=>{location.hash='backprop'}"); pg.wait_for_timeout(400)
    J = lambda js, *a: pg.evaluate(js, *a)
    C = J("()=>window.__bp.consts")
    check('constants: h choices, thresholds, clip', C['HS'] == HS and C['PASS'] == PASS and C['SUSPECT'] == SUSPECT and C['CLIP'] == CLIP, C)
    check('constants: the three networks', all(C['NETS'][k]['P'] == NETS[k]['P'] and C['NETS'][k]['params'] == NETS[k]['params'] and C['NETS'][k]['x1'] == NETS[k]['x1'] and C['NETS'][k]['x2'] == NETS[k]['x2'] and C['NETS'][k]['t'] == NETS[k]['t'] for k in NETS), C['NETS'])
    check('constants: seven parts, prediction “one pass”', C['STEP_IDS'] == ['mission', 'predict', 'walk', 'check', 'fan', 'inplace', 'remember'] and C['PRED_RIGHT'] == 'one', C['STEP_IDS'])

    # every network and switch: values, order, contributions, gradients, checks
    cases = []
    for net in ('two', 'tied'):
        for over in (False, True):
            for tz in (False, True):
                for inpl in ('none', 'eng', 'pt'):
                    cases.append(dict(net=net, over=over, tz=tz, inpl=inpl))
    for s in cases:
        tag = f"{s['net']} over={s['over']} tz={s['tz']} inpl={s['inpl']}"
        js = J("s=>window.__bp.run(s)", s)
        py = backward(s['net'], over=s['over'], tz=s['tz'], inpl=s['inpl'])
        fv = forward(s['net'])
        check(f'{tag}: loss', close(js['L'], fv['L']), (js['L'], fv['L']))
        check(f'{tag}: forward values (before any in-place change)', all(close(js['vals'][k], fv[k]) for k in fv), js['vals'])
        ids = [x['id'] for x in js['steps'][1:]]
        check(f'{tag}: backward order', ids == [x['id'] for x in py['steps']], (ids, [x['id'] for x in py['steps']]))
        if s['inpl'] == 'pt':
            ck = [J("([s,p])=>window.__bp.check(s,p,1e-5)", [s, p]) for p in NETS[s['net']]['params']]
            check(f'{tag}: PyTorch stops at y’s step (2) with no parameter gradient', js['error'] and js['error']['step'] == 2 and js['steps'][2]['err'] and js['n'] == 2
                  and not any(p in js['steps'][-1]['got'] for p in NETS[s['net']]['params']) and all(c['ga'] is None for c in ck), (js['error'], js['n'], js['steps'][-1]['got'], ck[:1]))
            continue
        ok = True
        for k, (a, q) in enumerate(zip(js['steps'][1:], py['steps']), 1):
            for ca, cq in zip(a['contribs'], q['contribs']):
                ok &= ca['to'] == cq['to'] and close(ca['local'], cq['local']) and close(ca['up'], cq['up']) and close(ca['c'], cq['c'])
            ok &= set(a['got']) >= {x for x in q['got'] if x in NETS[s['net']]['params']}
        check(f'{tag}: every step’s contributions', ok)
        check(f'{tag}: final gradients', all(close(js['grads'][p], py['grads'][p]) for p in NETS[s['net']]['params']), (js['grads'], py['grads']))
        if s['inpl'] == 'eng':
            check(f'{tag}: h clipped to 0.5 after the forward', js['changed'] and close(js['changed']['to'], 0.5) and close(js['changed']['from'], fv['h']), js['changed'])
        for p in NETS[s['net']]['params']:
            for h in HS:
                c = J("([s,p,h])=>window.__bp.check(s,p,h)", [s, p, h])
                gn = fdiff(s['net'], p, h); er = rel(py['grads'][p], gn)
                good = close(c['ga'], py['grads'][p]) and close(c['gn'], gn, 1e-7) and abs(c['err'] - er) <= 1e-6 * max(er, 1e-9) + 1e-15 and c['pass'] == (er < PASS) and c['bad'] == (er > SUSPECT)
                if not good: check(f'{tag}: check of {p} at h = {h}', False, (c, py['grads'][p], gn, er))
        n_ok = n

    # the formula box (review round 3): every line of numbers it shows works out, at every step of every run and of the four
    # hidden bugs. A reader who multiplies or adds the shown numbers and rounds to 4 decimals gets the shown result.
    def tex_val(e):
        e = e.replace('({', '(').replace('})', ')').replace('−', '-').replace('\\times', '*').replace('^2', '**2').replace('σ', 'sig')
        return eval(e, {'__builtins__': {}}, {'tanh': math.tanh, 'sig': lambda x: 1 / (1 + math.exp(-x))})
    num = lambda e: float(e.replace('({', '').replace('})', '').replace('−', '-'))
    bad, lines = [], 0
    for s in cases + [{'net': 'ch', **f} for f in CHB]:
        nsteps = J("s=>window.__bp.run(s)", s)['n']
        for k in range(nsteps + 1):
            for g in J("([s,k])=>window.__bp.fx(s,k)", [s, k]):
                for it, nx in zip(g['items'], g['items'][1:] + [None]):
                    if it.get('n'):
                        parts = it['n'][2:].split(' = ')
                        if len(parts) == 2:
                            lines += 1
                            if round(tex_val(parts[0]), 4) != num(parts[1]): bad.append((s, k, it['n']))
                    m = re.fullmatch(r'\\frac\{∂L\}\{∂\w+\} = (.+)', it.get('m') or '')
                    if m and it.get('n'):
                        lines += 1
                        if round(tex_val(m.group(1)), 4) != num(it['n'][2:]): bad.append((s, k, it['m'], it['n']))
    check(f'formula box: all {lines} lines of numbers work out to the shown 4 decimals', lines > 300 and not bad, bad[:3])
    check('formula box: two weights, step 6: ∂L/∂w₁ += 0.774163 × 1.5 = 1.1612',
          J("()=>window.__bp.fx({net:'two',over:false,tz:false,inpl:'none'},6)")[1]['items'][0]['n'] == '= 0.774163 \\times 1.5 = 1.1612')

    # the numbers the copy states
    two, tied = backward('two'), backward('tied')
    check('two weights: L 0.1653, gradients 1.1612, −0.4645, 0.7742, 0.4914', round(forward('two')['L'], 4) == 0.1653 and [round(two['grads'][p], 4) for p in ('w1', 'w2', 'b', 'v')] == [1.1612, -0.4645, 0.7742, 0.4914])
    st = [c for x in tied['steps'] for c in x['contribs'] if c['to'] == 'w']
    check('one weight used twice: −0.4257 (via x₂, first) + 1.0643 (via x₁) = 0.6386; = keeps 1.0643', [round(c['c'], 4) for c in st] == [-0.4257, 1.0643] and round(tied['grads']['w'], 4) == 0.6386 and round(backward('tied', over=True)['grads']['w'], 4) == 1.0643)
    check('six backward steps; the overwrite fails only w', len(two['steps']) == 6 and [p for p in ('w', 'b', 'v') if rel(backward('tied', over=True)['grads'][p], fdiff('tied', p, 1e-5)) >= PASS] == ['w'])
    eng = backward('tied', inpl='eng')
    check('in-place clip, small engine: every check fails (w, b, v)', all(rel(eng['grads'][p], fdiff('tied', p, 1e-5)) > SUSPECT for p in ('w', 'b', 'v')), eng['grads'])
    e5 = rel(two['grads']['w1'], fdiff('two', 'w1', 0.5)); e0 = rel(two['grads']['w1'], fdiff('two', 'w1', 1e-5))
    check('part 4: h = 0.5 fails, h = 0.00001 passes for w₁', e5 > SUSPECT and e0 < PASS, (e5, e0))
    errs_h = [rel(two['grads']['w1'], fdiff('two', 'w1', h)) for h in HS[:3]]
    check('part 4: the error shrinks about 100 times per 10 times smaller h (0.1 → 0.001)', 5000 < errs_h[1] / errs_h[2] < 20000, errs_h)

    # the challenge
    cv = forward('ch')
    check('challenge forward: z 0.55, s 0.6341, y 1.2512, L 0.0631', [round(cv['z'], 4), round(cv['s'], 4), round(cv['y'], 4), round(cv['L'], 4)] == [0.55, 0.6341, 1.2512, 0.0631], cv)
    want = [['w'], ['w', 'b'], ['w', 'b', 'v'], ['w', 'b', 'v', 'c']]
    for i in range(4):
        check(f'challenge bug {"ABCD"[i]}: failing set', J("i=>window.__bp.fails(i)", i) == ch_fails(i) == want[i], (J("i=>window.__bp.fails(i)", i), ch_fails(i)))
        k, sets = min_checks(i); m = J("i=>window.__bp.minChecks(i)", i)
        check(f'challenge bug {"ABCD"[i]}: fewest checks ({k})', len(m) == k and sorted(m) in [sorted(x) for x in sets], (m, sets))
        r = backward('ch', **CHB[i])
        worst = max(rel(r['grads'][p], fdiff('ch', p, 1e-5)) for p in ch_fails(i))
        least = min(rel(r['grads'][p], fdiff('ch', p, 1e-5)) for p in ch_fails(i))
        check(f'challenge bug {"ABCD"[i]}: failing checks are wrong by 14% or more, passing ones below 10⁻⁷', least > 0.14 and all(rel(r['grads'][p], fdiff('ch', p, 1e-5)) < PASS for p in NETS['ch']['params'] if p not in ch_fails(i)), (least, worst))
    for S in (['w'], ['c'], ['v'], ['b', 'v'], ['w', 'b', 'v', 'c']):
        for hid in range(4):
            if J("([h,S])=>window.__bp.consistent(h,S)", [hid, S]) != consistent(hid, S): check(f'consistent({hid},{S})', False)
    hid_js = [J("s=>window.__bp.hidden(s)", s) for s in range(1, 41)]
    def hidden(seed):
        blk = (seed - 1) // 4; r = mulberry(blk * 7919 + 17); o = [0, 1, 2, 3]
        for i in (3, 2, 1):
            j = math.floor(r() * (i + 1)); o[i], o[j] = o[j], o[i]
        return o[(seed - 1) % 4]
    hid_py = [hidden(s) for s in range(1, 41)]
    check('challenge: the hidden bug for seeds 1–40 (mulberry32 ported, a seeded order per block of four)', hid_js == hid_py, (hid_js[:10], hid_py[:10]))
    check('challenge: every block of four seeds hides all four bugs', all(set(hid_py[k:k + 4]) == {0, 1, 2, 3} for k in range(0, 40, 4)), hid_py[:12])

    # capability layer
    ex = J("()=>window.__bp.exits()")
    cube = lambda h: ((1 + h) ** 3 - (1 - h) ** 3) / (2 * h)
    check('new case 1: 0.2 − 0.5 + 0.1 = −0.2 is option 0', ex['transferA'][0] == 0 and abs((0.2 - 0.5 + 0.1) + 0.2) < 1e-12)
    check('new case 2: θ³ central difference 3.01 at h = 0.1, 3.0001 at h = 0.01 (option 2)', ex['transferA'][1] == 2 and round(cube(0.1), 2) == 3.01 and round(cube(0.01), 4) == 3.0001)

    # the drill
    picks = lambda seed: (lambda r: [[0.5, 1.5, 2.0, -0.5, -1.5][math.floor(r() * 5)], [0.3, 0.4, 0.5, -0.3, -0.5][math.floor(r() * 5)], [0.1, 0.2, -0.1][math.floor(r() * 3)], [0.0, 0.5, 1.0][math.floor(r() * 3)]])(mulberry(seed * 104729 + 7))
    bad = []
    for seed in range(1, 41):
        d = J("s=>window.__bp.drill(s)", seed); x, w, bb, t = picks(seed)
        z = w * x + bb; y = math.tanh(z); gb = 2 * (y - t) * (1 - y * y); gw = gb * x
        eps = 1e-6; fd = ((math.tanh((w + eps) * x + bb) - t) ** 2 - (math.tanh((w - eps) * x + bb) - t) ** 2) / (2 * eps)
        if not ([d['x'], d['w'], d['b'], d['t']] == [x, w, bb, t] and close(d['gw'], gw) and close(d['gb'], gb) and abs(gw - fd) < 1e-6): bad.append((seed, d, x, w, bb, t))
    check('drill: 40 seeds, inputs and both gradients (and a finite-difference check of ∂L/∂w)', not bad, bad[:2])

    # the teaser
    T = J("()=>window.__tzBp?{range:window.__tzBp.range,W:window.__tzBp.W,m:Array.from({length:41},(_,k)=>window.__tzBp.model(k-20)),r:window.__tzBp.read(15)}:null")
    check('teaser: registered, w = 0.8, x from −2 to 2', T and T['W'] == 0.8 and T['range'] == [-20, 20], T and T['range'])
    tb = [m for m in T['m'] if not (close(m['y'], math.tanh(0.8 * m['x'])) and close(m['dw'], (1 - math.tanh(0.8 * m['x']) ** 2) * m['x']))] if T else ['none']
    check('teaser: y = tanh(0.8x) and dy/dw = (1 − y²)·x at every slider position', not tb, tb[:2])
    check('teaser: at x = 1.5 the sentence says y 0.8337, 1 × 0.3050 × 1.5, dy/dw 0.4575 (at most 25 words)',
          T and all(s in T['r'] for s in ('0.8337', '0.3050', '0.4575')) and len(re.sub('<[^>]+>', '', T['r']).split()) <= 25, T and T['r'])
    check('page errors', not errs, errs)
    b.close()

# the Connect code runs and prints the engine's numbers
code = re.search(r"const CODE=`(.*?)`;", SRC, re.S).group(1)
out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True).stdout.split('\n')
exp = [l[2:] for l in code.split('\n') if l.startswith('# ')]
check('Connect code: runs and prints what its comments say', out[:2] == exp, (out, exp))
check('Connect code: its numbers are the lab’s (L and the four gradients, the check of w₁)', exp == [f"{round(forward('two')['L'], 4)} {[round(two['grads'][p], 4) for p in ('w1', 'w2', 'b', 'v')]}", f"{round(fdiff('two', 'w1', 1e-5), 4)}"], exp)
check('Connect code: at most 40 lines', len(code.split('\n')) <= 40, len(code.split('\n')))
# the shared numbers in the copy
check('part 4 names no model and no forward-pass count (review round 3: nobody cares about the exact number)', not any(x in SRC for x in ('Llama', '8030261248', '16.1B')))
check('no company named on the page (Meta, NVIDIA, Google, OpenAI, Anthropic)', not re.search(r'\b(Meta|NVIDIA|Nvidia|Google|OpenAI|Anthropic|DeepMind)\b', re.sub(r"https?://[^'\"` ]+", '', SRC)))
print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
