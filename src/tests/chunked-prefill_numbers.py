"""Golden numbers for the chunked-prefill lab (guide R9.2, CONTRACT §6).

The model is derived here independently of the JS: Llama-3.1-8B's parameters are counted component by component from its config
(q, k, v, o, gate, up, down and two RMSNorms per layer, the final RMSNorm, an untied embedding table and LM head); causal attention FLOPs
are counted query by query on small prompts and checked against the closed form; a step is priced as the slower of its arithmetic (at
50% of the H100 SXM's dense BF16 peak, 1,979 TFLOPS "with sparsity" ÷ 2) and its memory traffic (at 60% of 3.35 TB/s). The chunked run
is simulated here chunk by chunk, from the prompt's token counts, not by porting the JS.
Checks:
- the spec's verified numbers (usual step 11.74 ms; unchunked 1,494.0 ms; 8K 480.6 / 1,495.7; 2K 127.2 / 1,502.4; 1K 64.5 / 1,511.2;
  512 32.6 / 1,529.0; 256 16.6 / 1,714.0; the first chunk at 384 tokens 11.77 ms with arithmetic 11.47 < memory 11.77; at 512 15.14 ms,
  set by arithmetic);
- the small chunks the product owner asked for (29 Sep): 128 13.9 / 3,280.3, 64 13.9 / 6,559.6, 32 13.9 / 13,118.1: every step set by
  memory, so the users' longest wait stops shrinking (it is the last step's reads) while the first token doubles with each halving;
  the options (none, 2K, 1K, 512, 256, 128, 64, 32); the time axis, which covers the runs from none to 256 and leaves smaller chunks'
  first tokens past its right edge;
- the TTFT decomposition the page uses (the prompt's arithmetic + steps × the users' arithmetic + what memory adds) is exact;
- the page's engine (window.__ck) against this derivation for every workload the settings allow, and against the home tile
  (window.__tzA from teasers/set-a.js) for its four chunk sizes;
- the users that fit, against the KV cache memory lab's engine (window.__kv.compute) and a formula here;
- the challenge (32 users at 2K, a 16K document: 1K at 47.2 ms and 621 ms; 2K over 50 ms; 512 within it but 638 ms), the free-chunk crossings, the capability
  layer's cases, the scenarios, the interview drill (every step arithmetic-bound, closed form), the vLLM default budget line;
- the Connect card's code, run here: at most 40 lines of at most 74 characters, printing what its comments say;
- what the page shows: the metric cards, the timeline labels, Show the math (its TTFT line adds up from the parts it shows).
Run: python3 tests/chunked-prefill_numbers.py"""
import math, pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/chunked-prefill-num.html', '--only', 'chunked-prefill,set-a'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-chunked-prefill-num.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'chunked-prefill-num.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-chunked-prefill-num.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


def close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


# ---------- the model, derived from the config (hidden 4,096, 32 layers, 32 heads, 8 KV heads, head_dim 128, FFN 14,336, vocab 128,256)
D, LAYERS, HEADS, KVH, HD, FFN, VOCAB = 4096, 32, 32, 8, 128, 14336, 128256
comp = {'q': D * HEADS * HD, 'k': D * KVH * HD, 'v': D * KVH * HD, 'o': HEADS * HD * D, 'gate': D * FFN, 'up': D * FFN, 'down': FFN * D, 'norms': 2 * D}
layers = LAYERS * sum(comp.values()) + D              # the 32 layers and the final RMSNorm: multiplied (or scaled) by every token
emb = head = VOCAB * D                                # untied: the embedding table is a lookup, the LM head a matmul per new token
total = layers + emb + head
check('parameters: 8,030,261,248 (the Hugging Face total)', total == 8_030_261_248, total)
W_READ, W_MEM, KV_TOK = 2 * (layers + head), 2 * total, 2 * LAYERS * KVH * HD * 2
check('KV cache per token: 131,072 bytes = 128 KiB', KV_TOK == 131072)
PEAK, BW, MFU, MBU = 1979e12 / 2, 3.35e12, 0.5, 0.6
FS, BS = PEAK * MFU, BW * MBU


def pairs_loop(a, b):
    """query–key pairs for query positions a..b-1 under the causal mask: query i meets keys 0..i"""
    return sum(i + 1 for i in range(a, b))


def attn(a, b):   # 2 FLOPs for q·k and 2 for adding the value, per dimension of the hidden size, in every layer
    return 4 * LAYERS * D * ((b * (b + 1) - a * (a + 1)) // 2)


for a, b in [(0, 1), (0, 7), (0, 64), (5, 6), (100, 101), (10, 50), (4096, 4097)]:
    check(f'attention pairs {a}..{b}: closed form equals the count', (b * (b + 1) - a * (a + 1)) // 2 == pairs_loop(a, b))


def step(tok, logits, at, kv):
    fl = 2 * layers * tok + 2 * head * logits + at
    by = W_READ + KV_TOK * kv
    return {'t': max(fl / FS, by / BS), 'tc': fl / FS, 'tm': by / BS, 'fl': fl, 'by': by}


def run(U, c, N, C):
    """U users decode at context c; an N-token prompt is prefilled C tokens per step (C = 0 or ≥ N: in one step)"""
    chunk = C if 0 < C < N else N
    users = U * attn(c, c + 1)
    steps, t, a = [], 0.0, 0
    while a < N:
        b = min(N, a + chunk)
        s = step(U + (b - a), U + (1 if b == N else 0), users + attn(a, b), U * c + b)
        s.update(a=a, b=b, start=t); t += s['t']; steps.append(s); a = b
    normal = step(U, U, users, U * c)
    after = step(U + 1, U + 1, users + attn(N, N + 1), U * c + N)
    return {'steps': steps, 'n': len(steps), 'ttft': t, 'longest': max(s['t'] for s in steps), 'normal': normal, 'after': after,
            'tcD': normal['tc'], 'tcP': (2 * layers * N + 2 * head + attn(0, N)) / FS, 'extra': sum(max(0.0, s['tm'] - s['tc']) for s in steps),
            'nMem': sum(1 for s in steps if s['tm'] > s['tc'])}


ms = lambda s: s * 1e3
r0 = run(16, 4096, 32768, 0)
check('spec: a usual step, 16 users at 4K → 11.74 ms, set by memory', round(ms(r0['normal']['t']), 2) == 11.74 and r0['normal']['tm'] > r0['normal']['tc'], ms(r0['normal']['t']))
check('spec: unchunked 32K prefill → 1,494.0 ms, one step, set by arithmetic', round(ms(r0['ttft']), 1) == 1494.0 and r0['n'] == 1 and r0['steps'][0]['tc'] > r0['steps'][0]['tm'], ms(r0['ttft']))
SPEC = {8192: (480.6, 1495.7), 4096: (249.5, 1497.9), 2048: (127.2, 1502.4), 1024: (64.5, 1511.2), 512: (32.6, 1529.0), 256: (16.6, 1714.0),
        128: (13.9, 3280.3), 64: (13.9, 6559.6), 32: (13.9, 13118.1)}
for C, (w, tt) in SPEC.items():
    r = run(16, 4096, 32768, C)
    check(f'spec: {C}-token chunks → longest step {w} ms, TTFT {tt} ms', round(ms(r['longest']), 1) == w and round(ms(r['ttft']), 1) == tt, (ms(r['longest']), ms(r['ttft'])))
    check(f'{C}-token chunks: the last chunk’s step is the longest', r['steps'][-1]['t'] == r['longest'])
f384 = step(16 + 384, 16, 16 * attn(4096, 4097) + attn(0, 384), 16 * 4096 + 384)
check('spec: a first chunk of 384 tokens → 11.77 ms (arithmetic 11.47 < memory 11.77)', round(ms(f384['t']), 2) == 11.77 and round(ms(f384['tc']), 2) == 11.47 and round(ms(f384['tm']), 2) == 11.77, (ms(f384['tc']), ms(f384['tm'])))
f512 = step(16 + 512, 16, 16 * attn(4096, 4097) + attn(0, 512), 16 * 4096 + 512)
check('spec: a first chunk of 512 tokens → 15.14 ms, set by arithmetic', round(ms(f512['t']), 2) == 15.14 and f512['tc'] > f512['tm'], ms(f512['t']))
# the exact decomposition the page states: TTFT = the prompt's arithmetic + steps × the users' arithmetic + what memory adds
for C in [0, 8192, 2048, 512, 256]:
    r = run(16, 4096, 32768, C)
    check(f'TTFT decomposition at {C or "none"}: tcP + n × tcD + extra = TTFT', close(r['tcP'] + r['n'] * r['tcD'] + r['extra'], r['ttft']), (r['tcP'] + r['n'] * r['tcD'] + r['extra'], r['ttft']))
check('256-token chunks: 75 of 128 steps are set by memory, adding 149.5 ms', run(16, 4096, 32768, 256)['nMem'] == 75 and round(ms(run(16, 4096, 32768, 256)['extra']), 1) == 149.5)
# too small to help: every step set by memory; the longest wait is the last step's reads (weights + the users' KV + the whole prompt's KV)
last_reads = (W_READ + KV_TOK * (16 * 4096 + 32768)) / BS
for C in (128, 64, 32):
    r = run(16, 4096, 32768, C)
    check(f'{C}-token chunks: every step is set by memory, and the longest wait is the last step’s reads ({ms(last_reads):.2f} ms)', r['nMem'] == r['n'] and close(r['longest'], last_reads), (r['nMem'], r['n'], ms(r['longest'])))
check('256-token chunks are not “too small”: some steps are set by arithmetic', run(16, 4096, 32768, 256)['nMem'] < run(16, 4096, 32768, 256)['n'])
check('each halving below 256 about doubles the first token (3.3 s, 6.6 s, 13 s; 8.8 times the unchunked 1.5 s at 32)',
      all(1.9 < run(16, 4096, 32768, C // 2)['ttft'] / run(16, 4096, 32768, C)['ttft'] < 2.1 for C in (128, 64)) and 8.5 < run(16, 4096, 32768, 32)['ttft'] / r0['ttft'] < 9.0)
OPTS_ALL = [0, 2048, 1024, 512, 256, 128, 64, 32]
def span(U, c, N):
    m = max(run(U, c, N, C)['ttft'] for C in OPTS_ALL if C == 0 or C >= 256)
    return -max(0.06 * m, 4 * run(U, c, N, 0)['normal']['t']), m * 1.06
x0m, x1m = span(16, 4096, 32768)
check('the mission’s time axis ends at 1.06 × the 256-token run (1,817 ms); 128-, 64- and 32-token chunks put the first token past it',
      round(ms(x1m)) == 1817 and all(run(16, 4096, 32768, C)['ttft'] > x1m for C in (128, 64, 32)) and all(run(16, 4096, 32768, C)['ttft'] <= x1m for C in (0, 2048, 1024, 512, 256)), ms(x1m))
# the free chunk: the smallest chunk whose step is set by arithmetic, for the prompt's first chunk and for its last
def cross(U, c, N, last=False):
    for C in range(1, N + 1):
        a = N - C if last else 0
        s = step(U + C, U + (1 if a + C == N else 0), U * attn(c, c + 1) + attn(a, a + C), U * c + a + C)
        if s['tc'] >= s['tm']: return C
    return N + 1
X1, XL = cross(16, 4096, 32768), cross(16, 4096, 32768, True)
check('the first chunk rides free up to 394 tokens (395 is set by arithmetic): “about 390”', X1 == 395, X1)
check('the last chunk of a 32K prompt rides free only below 213 tokens (its step also reads the prompt’s 32K-token KV cache)', XL == 213, XL)

# ---------- memory: the KV cache memory lab's model (80 GB read as 80 GiB, 90% usable, 2 GiB overhead)
GiB = 2 ** 30
TOK_FIT = math.floor((80 * GiB * 0.9 - W_MEM - 2 * GiB) / KV_TOK + 1e-9)
check('tokens of KV cache that fit: 450,907', TOK_FIT == 450907, TOK_FIT)
max_users = lambda c, N: (TOK_FIT - N) // c
check('users that fit next to a 32K prompt at 4K: 102; next to 120K: 80', max_users(4096, 32768) == 102 and max_users(4096, 122880) == 80)

# ---------- the challenge (product owner's review of the Batch 3 preview, R3.8: a setting the guided parts never show):
# 32 users at 2K, a 16K-token document, every user within 50 ms; the earliest first token
OPTS = OPTS_ALL
pts = {C: run(32, 2048, 16384, C) for C in OPTS}
ok = [C for C in OPTS if pts[C]['longest'] <= 0.050]
best = min(ok, key=lambda C: pts[C]['ttft'])
check('challenge: 1K and every smaller chunk keep every user within 50 ms', sorted(ok) == [32, 64, 128, 256, 512, 1024], ok)
check('challenge: 1K is the answer (47.2 ms, 621 ms); 2K breaks at 92.2 ms; 512 meets it with the first token at 638 ms',
      best == 1024 and round(ms(pts[1024]['longest']), 1) == 47.2 and round(ms(pts[1024]['ttft'])) == 621 and round(ms(pts[2048]['longest']), 1) == 92.2 and round(ms(pts[512]['ttft'])) == 638, best)
gpts = {C: run(16, 4096, 32768, C) for C in OPTS}   # the guided parts' workload
check('the guided parts’ best under 50 ms (512) is not the challenge’s answer', min([C for C in OPTS if run(16, 4096, 32768, C)['longest'] <= 0.050], key=lambda C: run(16, 4096, 32768, C)['ttft']) == 512 and best != 512, best)

# ---------- the capability layer's cases
t1 = run(64, 4096, 32768, 0)
check('new case 1: 64 users, 32K unchunked → 1,496 ms, in “1 to 2 s”', 1.0 <= t1['longest'] < 2.0 and round(ms(t1['longest'])) == 1496, ms(t1['longest']))
t2 = run(16, 4096, 65536, 2048)
check('new case 2: 16 users, 64K prompt in 2K chunks → 198.4 ms, in “150 to 250 ms”', 0.150 <= t2['longest'] < 0.250 and round(ms(t2['longest']), 1) == 198.4, ms(t2['longest']))
f256 = run(16, 4096, 32768, 256)['steps'][0]
check('explain 2: the usual step 11.74 ms, with the first 256-token chunk 11.76 ms', round(ms(f256['t']), 2) == 11.76 and round(ms(r0['normal']['t']), 2) == 11.74)
# ---------- the scenarios
SC = {'doc': (16, 4096, 32768), 'upload': (32, 4096, 122880), 'chat': (64, 2048, 2048), 'code': (8, 8192, 8192)}
for k, (U, c, N) in SC.items():
    check(f'scenario {k}: fits in memory', U * c + N <= TOK_FIT)
check('scenario upload: unchunked 11.47 s; 8K chunks 1,263 ms; 2K 323.4 ms', round(run(32, 4096, 122880, 0)['ttft'], 2) == 11.47 and round(ms(run(32, 4096, 122880, 8192)['longest'])) == 1263 and round(ms(run(32, 4096, 122880, 2048)['longest']), 1) == 323.4)
check('scenario chat: a 2K prompt stalls 64 users 62.09 ms (usual 16.01 ms); 256 chunks double its first token (128.7 ms)',
      round(ms(run(64, 2048, 2048, 0)['longest']), 2) == 62.09 and round(ms(run(64, 2048, 2048, 0)['normal']['t']), 2) == 16.01 and round(ms(run(64, 2048, 2048, 256)['ttft']), 1) == 128.7)
check('scenario code: 8 users at 8K, 8K prompt: 267.0 ms unchunked, 384.5 ms first token with 256 chunks', round(ms(run(8, 8192, 8192, 0)['ttft']), 1) == 267.0 and round(ms(run(8, 8192, 8192, 256)['ttft']), 1) == 384.5)
# vLLM's default budget on an H100 (8,192 tokens) with 16 users decoding: chunks of 8,176
v8 = run(16, 4096, 32768, 8192 - 16)

with sync_playwright() as pw:
    b = pw.chromium.launch()
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(80)
    pg.evaluate("()=>{location.hash='chunked-prefill'}"); pg.wait_for_timeout(600)
    J = lambda expr: pg.evaluate(expr)
    c = J('()=>window.__ck.consts')
    check('page: the chunk sizes are none, 2K, 1K, 512, 256, 128, 64 and 32', c['K_OPTS'] == OPTS_ALL, c['K_OPTS'])
    for (U_, c_, N_) in [(16, 4096, 32768), (32, 2048, 16384), (32, 4096, 122880), (64, 2048, 2048)]:
        sp = J(f'()=>window.__ck.span({{U:{U_},ctx:{c_},N:{N_}}})'); ex0, ex1 = span(U_, c_, N_)
        check(f'page: the time axis for {U_} users at {c_}, a {N_}-token prompt, equals the derivation', close(sp['x0'], ex0) and close(sp['x1'], ex1), (sp, ex0, ex1))
    for k, v in [('P_TOTAL', total), ('P_LAYERS', layers), ('P_HEAD', head), ('P_EMB', emb), ('KV_TOK', KV_TOK), ('W_READ', W_READ), ('W_MEM', W_MEM), ('TOK_FIT', TOK_FIT)]:
        check(f'page constant {k} = {v}', c[k] == v, c[k])
    # ---------- the lab and the home tile: the same step model, the same numbers for the tile's chunk sizes
    tz = J('()=>window.__tzA&&window.__tzA.consts')
    check('the home tile (set-a) is in the page', tz is not None)
    if tz:
        for k in ['P_TOTAL', 'P_LAYERS', 'P_EMB', 'P_HEAD', 'KV_TOK', 'W_READ']:
            check(f'tile and lab share {k}', tz[k] == c[k], (tz[k], c[k]))
        setup = J('()=>window.__tzA.cp.setup')
        check('the tile’s setup is the mission: 16 users at 4K, a 32K prompt', (setup['users'], setup['userCtx'], setup['prompt']) == (16, 4096, 32768), setup)
        for C in J('()=>window.__tzA.cp.options'):
            t = J(f'()=>window.__tzA.cp.compute({C})'); lr = J(f'()=>window.__ck.run({{U:16,ctx:4096,N:32768,C:{C}}})')
            check(f'tile and lab agree at chunk {C or "none"}: usual step, longest wait, TTFT, steps',
                  t['normal'] == lr['normal']['t'] and t['longest'] == lr['longest'] and t['ttft'] == lr['ttft'] and t['n'] == lr['n'] and t['after'] == lr['after']['t'], (t['longest'], lr['longest'], t['ttft'], lr['ttft']))
    # ---------- the page's engine against the derivation, for every workload the settings allow
    grid = J("""()=>{const c=window.__ck.consts, out=[]; for(const U of c.U_STOPS) for(const x of c.X_STOPS) for(const N of c.N_STOPS){ if(U>window.__ck.maxUsers(x,N)) continue;
       for(const C of c.K_OPTS){const r=window.__ck.run({U,ctx:x,N,C}); out.push([U,x,N,C,r.ttft,r.longest,r.n,r.nMem,r.normal.t,r.after.t]);}} return out;}""")
    bad = []
    for U, x, N, C, ttft, longest, nn, nm, nt, at in grid:
        r = run(U, x, N, C)
        if not (close(ttft, r['ttft']) and close(longest, r['longest']) and nn == r['n'] and nm == r['nMem'] and close(nt, r['normal']['t']) and close(at, r['after']['t'])):
            bad.append((U, x, N, C, ttft, r['ttft']))
    check(f'the page’s engine equals the derivation for all {len(grid)} workloads and chunk sizes the settings allow', not bad and len(grid) > 800, bad[:3])
    fitgrid = J("()=>{const c=window.__ck.consts; return c.X_STOPS.flatMap(x=>c.N_STOPS.map(N=>[x,N,window.__ck.maxUsers(x,N)]))}")
    check('users that fit: the page equals the formula at every context and prompt', all(m == max_users(x, N) for x, N, m in fitgrid), [g for g in fitgrid if g[2] != max_users(g[0], g[1])][:3])
    for x in (1024, 4096, 16384):
        kv = J(f"()=>window.__kv.compute({{model:'llama31-8b',gpu:'h100',n:1,ctx:{x},users:1,wp:'bf16',kp:'bf16',attn:'gqa',util:0.9,overhead:2}}).maxUsers")
        check(f'the KV cache memory lab’s engine agrees: users that fit at {x} tokens each (no prompt) = {TOK_FIT // x}', kv == TOK_FIT // x, (kv, TOK_FIT // x))
    cb = J('()=>window.__ck.chBest()')
    check('page: the challenge’s best is 1K', cb['C'] == 1024 and close(cb['wait'], pts[1024]['longest']), cb['C'])
    for C in OPTS:
        p = J(f'()=>window.__ck.chPoint({C})')
        check(f'page: challenge point {C or "none"}', close(p['wait'], pts[C]['longest']) and p['ok'] == (pts[C]['longest'] <= 0.050))
    cr = J('()=>window.__ck.cross({U:16,ctx:4096,N:32768})')
    check('page: the free-chunk crossings equal the derivation', cr['first'] == X1 and cr['last'] == XL, (cr, X1, XL))
    for s in range(1, 21):
        d = J(f'()=>window.__ck.drill({s})')
        r = run(d['U'], d['ctx'], d['N'], d['C'])
        closed = (2 * layers * d['N'] + 2 * head + attn(0, d['N']) + r['n'] * (2 * layers * d['U'] + 2 * head * d['U'] + d['U'] * attn(d['ctx'], d['ctx'] + 1))) / FS
        check(f'drill {s}: every step set by arithmetic, the closed form equals the run, within memory, never the mission or a new case',
              r['nMem'] == 0 and close(closed, r['ttft']) and close(d['r']['ttft'], r['ttft']) and d['U'] * d['ctx'] + d['N'] <= TOK_FIT
              and (d['U'], d['ctx'], d['N']) not in [(16, 4096, 32768), (64, 4096, 32768), (16, 4096, 65536)], d)
    # ---------- the Connect card's code, run here: its comments state its own output
    code = J("()=>document.getElementById('ck-code').textContent")
    lines = code.splitlines()
    check('Connect code: at most 40 lines, none longer than 74 characters', len(lines) <= 40 and max(len(l) for l in lines) <= 74, (len(lines), max(len(l) for l in lines)))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True).stdout.strip().splitlines()
    promised = [l[2:] for l in lines if l.startswith('# budget')]
    check('Connect code: runs and prints what its comments say', out == promised and len(out) == 3, (out, promised))
    # ---------- what the page shows
    def cards():
        return J("()=>Object.fromEntries([...document.querySelectorAll('#ck-metrics .metric')].map(m=>[m.dataset.k,m.querySelector('.v').textContent.replace(/\\u202f/g,' ')]))")
    got = cards()
    check('mission cards: 1,494 ms, 1,494 ms, 1,494 ms, 1 step', got == {'wait': '1,494 ms', 'tpot': '1,494 ms', 'ttft': '1,494 ms', 'steps': '1'}, got)
    J("()=>document.querySelector('#ck-app .modes button[data-mode=\"sandbox\"]:not([data-flow=free])').click()"); pg.wait_for_timeout(300)
    def chunk(C):
        J(f"()=>document.querySelector('#ck-controls .seg-ctl[data-key=\"C\"] button[data-v=\"{C}\"]').click()"); pg.wait_for_timeout(200)
    chunk(2048); got = cards()
    check('2K cards: 127.2 ms, 93.90 ms (1,502.4 ÷ 16), 1,502 ms, 16 steps', got == {'wait': '127.2 ms', 'tpot': '93.90 ms', 'ttft': '1,502 ms', 'steps': '16'} and f"{ms(gpts[2048]['ttft'])/16:.2f}" == '93.90', got)
    tl = J("()=>document.getElementById('ck-tl').innerText.replace(/\\u202f/g,' ')")
    check('2K timeline: “longest wait: 127.2 ms” and “first token at 1,502 ms”', 'longest wait: 127.2 ms' in tl and 'first token at 1,502 ms' in tl, tl[:300])
    J("()=>{document.getElementById('ck-math').open=true}"); pg.wait_for_timeout(100)
    mt = J("()=>document.getElementById('ck-mathbody').innerText.replace(/\\u202f/g,' ')")
    mt = ' '.join(mt.split())   # H2 (2026-10-09): Show the math is small tables (name and formula, the numbers under it, the result)
    m = re.search(r'= ([\d,.]+) ms \+ (\d+) × ([\d.]+) ms \+ ([\d.]+) ms = ([\d,.]+) ms', mt)
    check('Show the math: the TTFT line adds up from the parts it shows, and rounds to the card', bool(m) and abs(float(m.group(1).replace(',', '')) + int(m.group(2)) * float(m.group(3)) + float(m.group(4)) - float(m.group(5).replace(',', ''))) < 0.051
          and round(float(m.group(5).replace(',', ''))) == round(ms(gpts[2048]['ttft'])), m.groups() if m else mt[:400])
    check('Show the math: the longest step’s arithmetic and memory lines at 2K chunks (127.235 ms, 13.878 ms)', f"{ms(gpts[2048]['steps'][-1]['tc']):.3f}" in mt and f"{ms(gpts[2048]['steps'][-1]['tm']):.3f}" in mt, mt[:600])
    check('Show the math: the tokens that fit, 450,907', '450,907' in mt)
    chunk(512); got = cards()
    check('512 cards: 32.64 ms, 1,529 ms, 64 steps', got['wait'] == '32.64 ms' and got['ttft'] == '1,529 ms' and got['steps'] == '64', got)
    chunk(256); got = cards()
    check('256 cards: 16.63 ms, 1,714 ms, 128 steps', got['wait'] == '16.63 ms' and got['ttft'] == '1,714 ms' and got['steps'] == '128', got)
    for C, want in [(128, ('13.88 ms', '3,280 ms', '256')), (64, ('13.88 ms', '6,560 ms', '512')), (32, ('13.88 ms', '13.12 s', '1,024'))]:
        chunk(C); got = cards()
        check(f'{C} cards: {want[0]}, {want[1]}, {want[2]} steps', (got['wait'], got['ttft'], got['steps']) == want and f"{ms(gpts[C]['longest']):.2f} ms" == want[0], got)
    ftl = J("()=>document.querySelector('#ck-arr .ck-arr-f').textContent.replace(/\u202f/g,' ')")
    check('32-token chunks: the first token past the axis, said at its right edge', ftl == 'first token at 13.12 s, off the chart ▸', ftl)
    # the Connect card's vLLM line: 8,176-token chunks in the mission
    prod = J("()=>document.querySelector('#ck-app .ck-prod').innerText.replace(/\\u202f/g,' ')")
    want_v8 = f"{ms(v8['longest']):.1f} ms"
    check(f'Connect: vLLM’s 8,192-token default leaves 8,176-token chunks, users wait up to {want_v8}', '8,176' in prod and want_v8 in prod, prod[:500])
    # the fidelity note's ridge range and intensities
    fid = J("()=>document.getElementById('chunked-prefill-fid').textContent").replace(' ', ' ').replace(' ', ' ')
    lo, hi = PEAK * 0.2 / BW, PEAK / (BW * 0.2)
    aiU = r0['normal']['fl'] / r0['normal']['by']; aiP = r0['steps'][0]['fl'] / r0['steps'][0]['by']
    want_f = f"between {round(lo)} and {round(hi):,} FLOPs per byte, while a usual step of the mission does {aiU:.1f} and the unchunked prefill {round(aiP):,}"
    check(f'fidelity note: {want_f}', want_f in fid and aiU < lo and hi < aiP, fid[:400])
    check('no page errors', not errs, errs[:3])
    b.close()
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails if fails else 'none')
sys.exit(1 if fails else 0)
