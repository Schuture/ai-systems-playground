"""Golden numbers for the prefix-caching lab (guide R9.2, CONTRACT §6).

Derived here independently of the JS:
- Llama-3.1-8B's parameters are counted component by component from its config (q, k, v, o, gate, up, down, two RMSNorms per layer,
  the final RMSNorm, an untied embedding table and LM head); a step is priced as the slower of its arithmetic (50% of the H100 SXM's
  dense BF16 peak, 1,979 TFLOPS "with sparsity" ÷ 2) and its memory traffic (60% of 3.35 TB/s); attention pairs are counted one by one
  on small prompts and checked against the closed form.
- The cache is a different algorithm from the page's: exact token tuples in a trie of 16-token blocks (a node is its parent node plus
  its 16 tokens, compared by value, no hashing), LRU order kept per node, a request's nodes freed tail first (vLLM's order), the hit
  capped at the prompt minus one token, and the blocks of prompt + answer − 1 cached after each request. The page uses a 53-bit hash
  chain over made-up token IDs; the two must agree on every hit.
Checks: the spec's verified numbers; the page's engine (window.__px) against this derivation; the home tile (window.__tzA from
teasers/set-a.js) against the lab (same constants, the same tokens to prefill for 1 to 8 requests); the mission, every lever of part 4,
part 5's experiment (one changed token at the front, in the middle, at the end: hit, first missing block, tokens recomputed, first token,
new blocks and their memory, the block keys, the counters during the sweep; since review r1 the counters count the system prompt only:
its tokens recomputed, its blocks stored twice and their memory, and which of the three predictions give the right count at each
position; the stops of “block by block”; the counters and the title-row results as shown),
the eviction demo of part 6 at each memory, the four scenarios, the challenge (every one of the 720 orders: its average, the best
average, the fewest moved parts that meet the goal), the exits' new cases, the interview drill, the Connect card's code (run here),
and the numbers the page shows.
Run: python3 tests/prefix-caching_numbers.py"""
import collections, itertools, math, pathlib, random, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/prefix-caching-num.html', '--only', 'prefix-caching,set-a'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-prefix-caching-num.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'prefix-caching-num.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-prefix-caching-num.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


def close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


# ---------- the model (hidden 4,096, 32 layers, 32 heads, 8 KV heads, head_dim 128, FFN 14,336, vocab 128,256)
D, LAYERS, HEADS, KVH, HD, FFN, VOCAB = 4096, 32, 32, 8, 128, 14336, 128256
comp = {'q': D * HEADS * HD, 'k': D * KVH * HD, 'v': D * KVH * HD, 'o': HEADS * HD * D, 'gate': D * FFN, 'up': D * FFN, 'down': FFN * D, 'norms': 2 * D}
layers = LAYERS * sum(comp.values()) + D
emb = head = VOCAB * D
total = layers + emb + head
check('parameters: 8,030,261,248 (the Hugging Face total)', total == 8_030_261_248, total)
W_READ, W_MEM, KV_TOK = 2 * (layers + head), 2 * total, 2 * LAYERS * KVH * HD * 2
check('KV cache per token: 131,072 bytes = 128 KiB', KV_TOK == 131072)
PEAK, BW, MFU, MBU = 1979e12 / 2, 3.35e12, 0.5, 0.6
GiB = 2 ** 30


def pairs(a, b):
    return sum(i + 1 for i in range(a, b))   # query i (0-based) meets i + 1 keys under the causal mask


for a, b in [(0, 1), (0, 9), (5, 40), (16, 48)]:
    check(f'attention pairs {a}..{b}: closed form equals the count', (b * (b + 1) - a * (a + 1)) // 2 == pairs(a, b))


def ttft(N, H):
    fl = 2 * layers * (N - H) + 2 * head + 4 * LAYERS * D * ((N * (N + 1) - H * (H + 1)) // 2)
    by = W_READ + KV_TOK * N
    tc, tm = fl / (PEAK * MFU), by / (BW * MBU)
    return {'t': max(tc, tm), 'tc': tc, 'tm': tm, 'fl': fl, 'by': by}


ms = lambda s: s * 1e3
# ---------- the spec's verified numbers
for (Pp, O, want0, want1) in [(10000, 200, 342.9, 8.13), (2000, 200, 64.6, 7.61), (100000, 200, 8146.9, 26.86)]:
    a, b = ttft(Pp + O, 0), ttft(Pp + O, Pp)
    check(f'spec: P={Pp:,}, O={O}: {want0} ms → {want1} ms', round(ms(a['t']), 1) == want0 and round(ms(b['t']), 2) == want1, (ms(a['t']), ms(b['t'])))
m = ttft(10200, 10000)
check('spec: the hit is memory-bound, arithmetic 7.79 ms', round(ms(m['tc']), 2) == 7.79 and m['tm'] > m['tc'])
check('spec: 42 times sooner (342.9 ÷ 8.13 as shown)', round(342.9 / 8.13) == 42)
check('spec: 10,000 × 128 KiB = 1.22 GiB (1.31 GB)', round(10000 * KV_TOK / GiB, 2) == 1.22 and round(10000 * KV_TOK / 1e9, 2) == 1.31)
pool_b = 80 * GiB * 0.9 - W_MEM - 2 * GiB
check('the KV cache pool: 55.0 GiB, room for 450,907 tokens', round(pool_b / GiB, 1) == 55.0 and int(pool_b // KV_TOK) == 450907, (pool_b / GiB, pool_b // KV_TOK))

# ---------- the cache, derived differently: a trie of 16-token blocks with exact token tuples
BS, ANS, DATE_T, NAME_T = 16, 300, 12, 8


def toks(segs):
    out = []
    for s in segs:
        e = s.get('edit')
        out += [(s['cid'], i, i == e) for i in range(s['n'])]
    return out


class Trie:
    def __init__(self, cap_blocks):
        self.cap, self.kids, self.lru, self.content = cap_blocks, {}, collections.OrderedDict(), collections.Counter()
        self.next = 1

    def node(self, parent, blk, create):
        k = (parent, blk)
        if k not in self.kids:
            if not create: return None
            self.kids[k] = self.next; self.next += 1
        return self.kids[k]

    def run(self, prompt, ans):
        t = toks(prompt); N = len(t); nb = N // BS; cap = (N - 1) // BS
        blocks = [tuple(t[b * BS:(b + 1) * BS]) for b in range(nb)]
        hit, par = 0, 0
        while hit < min(nb, cap):
            nd = self.node(par, blocks[hit], False)
            if nd is None or nd not in self.lru: break
            par = nd; hit += 1
        same = sum(1 for b in range(hit, nb) if self.content[blocks[b]] > 0)
        full = (t + toks([ans]))[:N + ans['n'] - 1]
        fb = [tuple(full[b * BS:(b + 1) * BS]) for b in range(len(full) // BS)]
        ids, par = [], 0
        for blk in fb:
            par = self.node(par, blk, True); ids.append((par, blk))
        for nd, blk in reversed(ids):       # the tail first: it is evicted first
            if nd in self.lru: self.lru.pop(nd)
            else: self.content[blk] += 1
            self.lru[nd] = blk
        while len(self.lru) > self.cap:
            nd, blk = self.lru.popitem(last=False); self.content[blk] -= 1
        return {'N': N, 'hit': hit * BS, 'same': same, 'cached': len(self.lru)}


def stream(Pp=10000, O=200, n=2, kind='same', change='none', apps=1, mem='pool', at=None, trie=False):
    reqs = []
    if kind == 'chat':
        hist = []
        for i in range(n):
            msg, ans = {'cid': f'm{i}', 'n': O}, {'cid': f'a{i}', 'n': ANS}
            reqs.append(([{'cid': 'preA', 'n': Pp}] + hist + [msg], ans, 'A'))
            hist = hist + [msg, ans]
    else:
        for i in range(n):
            app = 'AABACABA'[i] if apps == 3 else 'A'
            pre = {'cid': 'pre' + app, 'n': Pp, 'edit': ((Pp // 2 - 1) if at is None else at) if (change == 'edit' and i > 0) else None}
            own, date, name = {'cid': f'own{i}', 'n': O}, {'cid': 'day' + ('1' if i == 0 else '2'), 'n': DATE_T}, {'cid': f'name{i}', 'n': NAME_T}
            segs = {'dateStart': [date, pre, own], 'dateAfter': [pre, date, own], 'nameStart': [name, pre, own]}.get(change, [pre, own])
            reqs.append((segs, {'cid': f'ans{i}', 'n': ANS}, app))
    cap_t = int(pool_b // KV_TOK) if mem == 'pool' else mem * (Pp + O + ANS)
    tr = Trie(cap_t // BS)
    out = []
    for s, a, app in reqs:
        # before each request: how many of its full prompt blocks the cache does not hold yet (walked down the trie, block by block)
        t = toks(s); fb = [tuple(t[b * BS:(b + 1) * BS]) for b in range(len(t) // BS)]; par, new, new_sys = 0, 0, 0
        for bi, blk in enumerate(fb):
            nd = tr.node(par, blk, False) if par is not None else None
            if nd is None or nd not in tr.lru:
                new += 1; par = None
                if bi < Pp // BS: new_sys += 1       # a block of the system prompt that the cache will now hold a second time
            else: par = nd
        out.append(dict(tr.run(s, a), app=app, new=new, newSys=new_sys))
    return (out, tr) if trie else out


mis = stream()
check('mission: request 2 hits 10,000 of 10,200 tokens', [r['hit'] for r in mis] == [0, 10000] and mis[1]['N'] == 10200, mis)
LEV = {'none': (10000, 8.13), 'dateStart': (0, 343.4), 'dateAfter': (10000, 8.25), 'edit': (4992, 188.9), 'nameStart': (0, 343.2)}
for ch, (h, t) in LEV.items():
    r = stream(change=ch)[1]; tt = ttft(r['N'], r['hit'])['t']
    shown = round(ms(tt), 2) if ms(tt) < 9.995 else round(ms(tt), 1)
    check(f'part 4, {ch}: request 2 hits {h:,} tokens, first token {t} ms', r['hit'] == h and shown == t, (r, ms(tt)))
check('part 4, date first: the 10,000 system-prompt tokens are cached after another prefix (hatched blocks: 624 of 639)', stream(change='dateStart')[1]['same'] == 624, stream(change='dateStart')[1])
check('part 4, one word: 312 blocks after the edit hold cached tokens', stream(change='edit')[1]['same'] == 312, stream(change='edit')[1])
EVICT = {'pool': [0, 10000, 0, 10000, 0, 10000, 10000, 10000], 3: [0, 10000, 0, 10000, 0, 10000, 10000, 10000], 2: [0, 10000, 0, 10000, 0, 10000, 0, 10000], 1: [0, 10000, 0, 0, 0, 0, 0, 0]}
for mem, want in EVICT.items():
    r = stream(n=8, apps=3, mem=mem)
    check(f'part 5, memory {mem}: hits per request {want}', [x['hit'] for x in r] == want, [x['hit'] for x in r])
check('part 5: room for 2 requests = 21,000 tokens = 2.56 GiB; 1 request = 1.28 GiB', round(21000 * KV_TOK / GiB, 2) == 2.56 and round(10500 * KV_TOK / GiB, 2) == 1.28)
SCEN = {'support': dict(Pp=10000, O=200, n=5), 'chat': dict(Pp=2000, O=100, n=5, kind='chat'), 'doc': dict(Pp=100000, O=200, n=4), 'fewshot': dict(Pp=2000, O=200, n=5)}
chat = stream(**SCEN['chat'])
check('multi-turn chat: each turn hits the last one up to its answer’s last full block (0, 2,384, 2,784, 3,184, 3,584)', [x['hit'] for x in chat] == [0, 2384, 2784, 3184, 3584], [x['hit'] for x in chat])

# ---------- part 5, the experiment (Batch 4): request 2 changes one token of the system prompt, at the front (token 10, in block 1), in the
# middle (token 5,000, part 4's word) or at the end (token 9,990, in the last full block). Derived here with the trie (exact token tuples,
# no hashing) and the step model above: the hit, the first missing block, the tokens recomputed, request 2's time to first token, the full
# prompt blocks the cache did not hold (they are stored next to request 1's, under different keys) and their memory
fmt_t = lambda sec: (f'{sec * 1e3:,.2f} ms' if sec * 1e3 < 9.995 else (f'{sec * 1e3:,.1f} ms' if sec * 1e3 < 999.95 else f'{sec:,.2f} s'))
gib2 = lambda b: f'{b / GiB:,.2f} GiB'
OT = {'front': 10, 'middle': 5000, 'end': 9990}
OTX = {}
for v, tok in OT.items():
    (r1, r2), tr = stream(change='edit', at=tok - 1, trie=True)
    N, blk = r2['N'], (tok - 1) // BS + 1                      # the changed token's block, counted from 1
    hb = r2['hit'] // BS
    OTX[v] = dict(tok=tok, N=N, hb=hb, hit=r2['hit'], first=hb + 1, blk=blk, rec=N - r2['hit'], t=ttft(N, r2['hit'])['t'], same=r2['same'], new=r2['new'],
                  bytes=r2['new'] * BS * KV_TOK, cached1=r1['cached'], cached2=r2['cached'], ans=(r2['cached'] - r1['cached']) - r2['new'],
                  recSys=sum(1 for i in range(10000) if i >= r2['hit']), dup=r2['newSys'], dupBytes=r2['newSys'] * BS * KV_TOK)
    check(f'part 5 {v}: token {tok:,} sits in block {blk}; every block before it hits and the first miss is that block', r2['hit'] == (blk - 1) * BS and hb + 1 == blk, (r2, blk))
    check(f'part 5 {v}: request 1 (the empty cache) hits nothing', r1['hit'] == 0, r1)
    check(f'part 5 {v}: the new prompt blocks are the full blocks from the changed one on', r2['new'] == N // BS - hb, (r2['new'], N // BS, hb))
    check(f'part 5 {v}: the answer adds the same blocks at every position ((10,200 + 300 − 1) ÷ 16 − 637 = 19)', OTX[v]['ans'] == (N + ANS - 1) // BS - N // BS == 19, OTX[v]['ans'])
GOLD = {'front': (0, 1, 10200, '342.9 ms', 624, 637, '1.24 GiB'), 'middle': (4992, 313, 5208, '188.9 ms', 312, 325, '0.63 GiB'), 'end': (9984, 625, 216, '8.41 ms', 0, 13, '0.03 GiB')}
for v, (hit, first, rec, t, same, new, g) in GOLD.items():
    x = OTX[v]
    check(f'part 5 {v}: {hit:,} tokens cached, first miss at block {first}, {rec:,} recomputed, first token {t}, {same} hatched blocks, {new} new blocks = {g}',
          (x['hit'], x['first'], x['rec'], fmt_t(x['t']), x['same'], x['new'], gib2(x['bytes'])) == (hit, first, rec, t, same, new, g), x)
# review r1 #3: the counters and the choices count the system prompt only (the question's 200 tokens are new at every position)
gibc = lambda b: '0 GiB' if b <= 0 else ('< 0.01 GiB' if b < 0.005 * GiB else gib2(b))
GOLD_SYS = {'front': (10000, 625, '1.22 GiB'), 'middle': (5008, 313, '0.61 GiB'), 'end': (16, 1, '< 0.01 GiB')}
for v, (rs, dup, g) in GOLD_SYS.items():
    x = OTX[v]
    check(f'part 5 {v}: {rs:,} system-prompt tokens recomputed, {dup} system-prompt blocks stored twice = {g}', (x['recSys'], x['dup'], gibc(x['dupBytes'])) == (rs, dup, g), x)
    check(f'part 5 {v}: the new blocks are the system prompt’s from the changed one on plus the question’s 12 full blocks', x['new'] == x['dup'] + 12 and x['dup'] == 10000 // BS - x['hb'], (x['new'], x['dup']))
# what each prediction says about the system prompt's blocks recomputed: A the changed block alone, B that block and every block after it
# in the system prompt, C all of them; a prediction is right where it gives the count the trie gives
SAYS = lambda x: {'block': 1, 'before': 10000 // BS - (x['blk'] - 1), 'none': 10000 // BS}
RIGHT = {v: [SAYS(x)[q] == x['dup'] for q in ['block', 'before', 'none']] for v, x in OTX.items()}
check('part 5: “every block before it” is right at every position; at the front “none of it” gives the same count, at the end “that block alone” does',
      RIGHT == {'front': [False, True, True], 'middle': [False, True, False], 'end': [True, True, False]}, RIGHT)
# block by block (review r1 #6): all the hits at once, the changed block, the block after it, the rest of the system prompt, the question
def stops(x):
    hb, pb, T = x['hb'], 10000 // BS, -(-x['N'] // BS); out = ([hb] if hb else []) + [hb + 1]
    if hb + 2 <= pb: out.append(hb + 2)
    if pb > out[-1]: out.append(pb)
    if T > out[-1]: out.append(T)
    return out
STOPS = {v: stops(x) for v, x in OTX.items()}
check('part 5 block by block: the stops (front 1, 2, 625, 638; middle 312, 313, 314, 625, 638; end 624, 625, 638)',
      STOPS == {'front': [1, 2, 625, 638], 'middle': [312, 313, 314, 625, 638], 'end': [624, 625, 638]}, STOPS)
check('part 5: the later the change, the more is saved (time and memory fall from the front to the end)', OTX['front']['t'] > OTX['middle']['t'] > OTX['end']['t'] and OTX['front']['new'] > OTX['middle']['new'] > OTX['end']['new'])
check('part 5: at the front the hit is nothing at all, the same as without the cache (“the blocks before it” and “nothing” agree)', OTX['front']['hit'] == 0 and close(OTX['front']['t'], ttft(10200, 0)['t']))
check('part 5: at the end the first token is almost as with no change (8.41 ms against 8.13 ms)', fmt_t(OTX['end']['t']) == '8.41 ms' and fmt_t(ttft(10200, 10000)['t']) == '8.13 ms')


# ---------- the challenge: every order
CHP = ['date', 'name', 'sys', 'shots', 'hist', 'msg']
LEN = {'sys': 3000, 'shots': 1000, 'date': DATE_T, 'name': NAME_T, 'msg': 150}
PRE = {'Alice': 1500, 'Bob': 800, 'Carol': 2500}
STREAM = [(1, 'Alice'), (1, 'Bob'), (1, 'Carol'), (2, 'Alice'), (2, 'Carol'), (2, 'Bob'), (3, 'Bob'), (3, 'Alice')]


def ch_run(order):
    tr = Trie(int(pool_b // KV_TOK) // BS); hist = {u: [{'cid': 'h0' + u, 'n': PRE[u]}] for u in PRE}; out = []
    for i, (day, u) in enumerate(STREAM):
        part = {'sys': [{'cid': 'sys', 'n': 3000}], 'shots': [{'cid': 'shots', 'n': 1000}], 'date': [{'cid': f'd{day}', 'n': DATE_T}],
                'name': [{'cid': 'n' + u, 'n': NAME_T}], 'hist': hist[u], 'msg': [{'cid': f'msg{i}', 'n': 150}]}
        segs = [s for p in order for s in part[p]]; ans = {'cid': f'ans{i}', 'n': ANS}
        r = tr.run(segs, ans); out.append(r)
        hist[u] = hist[u] + [{'cid': f'msg{i}', 'n': 150}, ans]
    return out


def moves(order):
    pos = [CHP.index(p) for p in order]; best = [1] * len(pos)
    for i in range(len(pos)):
        for j in range(i):
            if pos[j] < pos[i]: best[i] = max(best[i], best[j] + 1)
    return len(order) - max(best)


ALL = {}
for o in itertools.permutations(CHP):
    r = ch_run(o); ALL[o] = (sum(x['hit'] for x in r) / 8, moves(o), r)
best = max(v[0] for v in ALL.values())
check('challenge: the best average is 4,594 cached tokens per request', best == 4594, best)
check('challenge: the initial order caches nothing', ALL[tuple(CHP)][0] == 0, ALL[tuple(CHP)][0])
ok = [o for o, v in ALL.items() if v[0] >= 4400]
check('challenge: the fewest moved parts that meet 4,400 is 2', min(ALL[o][1] for o in ok) == 2)
check('challenge: moving one part never meets the goal', all(v[0] < 4400 for o, v in ALL.items() if v[1] <= 1))
check('challenge: between the orders that meet the goal and the others there is a gap (4,496 and up vs 4,342 and down)', min(ALL[o][0] for o in ok) >= 4496 and max(v[0] for v in ALL.values() if v[0] < 4400) <= 4342)
SHOW = [(('date', 'name', 'sys', 'shots', 'hist', 'msg'), 0), (('name', 'sys', 'shots', 'hist', 'date', 'msg'), 3500), (('date', 'sys', 'shots', 'name', 'hist', 'msg'), 2500),
        (('sys', 'shots', 'name', 'hist', 'date', 'msg'), 4500), (('sys', 'shots', 'name', 'hist', 'msg', 'date'), 4594)]
for o, want in SHOW:
    check(f'challenge chart: {",".join(o)} averages {want:,}', ALL[o][0] == want, ALL[o][0])
avgN = sum(x['N'] for x in ALL[tuple(CHP)][2]) / 8
check('challenge: prompts average about 6,050 tokens', round(avgN) == 6051, avgN)


# ---------- the reveal's side-by-side (GPT-6 review of v9, #2): Alice's day 1 (prompt, then answer: what the cache holds) against her
# day 2 prompt, part by part, for A the best order, B the date before the new message, C the name before the new message
def ch_segs(order, i, save_date=False, stream_=STREAM):
    # request i's segments as ch_run builds them (the history holds messages and answers only, unless save_date)
    hist = {u: [('h0' + u, PRE[u])] for u in PRE}
    for j, (day, u) in enumerate(stream_):
        part = {'sys': [('sys', 3000)], 'shots': [('shots', 1000)], 'date': [(f'd{day}', DATE_T)], 'name': [('n' + u, NAME_T)], 'hist': hist[u], 'msg': [(f'msg{j}', 150)]}
        segs = [(c, n_, p) for p in order for (c, n_) in part[p]]
        if j == i: return segs, (f'ans{j}', ANS)
        turn = [x for p in order if p in (('date', 'msg') if save_date else ('msg',)) for x in part[p]]
        hist[u] = hist[u] + turn + [(f'ans{j}', ANS)]


def side_by_side(order):
    s1, ans = ch_segs(order, 0); s2, _ = ch_segs(order, 3)
    a = [(c, n_) for c, n_, _p in s1] + [ans]; b = [(c, n_) for c, n_, _p in s2]
    m = 0
    while m < min(len(a), len(b)) and a[m][0] == b[m][0]: m += 1
    ca = {c for c, _ in a}
    return dict(m=m, match=sum(n_ for _, n_ in a[:m]), a=[(n_, 'hit' if i < m else 'held') for i, (_, n_) in enumerate(a)],
                b=[(n_, 'hit' if i < m else ('same' if c in ca else 'new')) for i, (c, n_) in enumerate(b)], hit=ALL[tuple(order)][2][3]['hit'], N=ALL[tuple(order)][2][3]['N'])


AL = [('sys', 'shots', 'name', 'hist', 'msg', 'date'), ('sys', 'shots', 'name', 'hist', 'date', 'msg'), ('sys', 'shots', 'hist', 'name', 'msg', 'date')]
SBS = {o: side_by_side(o) for o in AL}
GOLD_AL = {AL[0]: (5, 5658, 5648, 472, 4594), AL[1]: (4, 5508, 5504, 616, 4500), AL[2]: (3, 5500, 5488, 632, 4496)}
for o, (m, match, hit, comp, avg) in GOLD_AL.items():
    x = SBS[o]
    check(f'side by side {",".join(o)}: {m} parts match ({match:,} tokens), the hit is that rounded down to blocks ({hit:,}), day 2 computes {comp}, average {avg:,}',
          (x['m'], x['match'], x['hit'], x['N'] - x['hit'], ALL[o][0]) == (m, match, hit, comp, avg) and x['hit'] == x['match'] // BS * BS, x)
check('side by side: A keeps message 1 cached, B and C lose it at the end of the old history (the first differing part: date 1, date 1, the name)',
      [ch_segs(o, 0)[0][SBS[o]['m']][0] if SBS[o]['m'] < len(ch_segs(o, 0)[0]) else None for o in AL] == ['d1', 'd1', 'nAlice'])
check('side by side: after the cut day 2 recomputes answer 1 in A, message 1 and answer 1 in B and C (hatched), the rest is new',
      [[st for _, st in SBS[o]['b'][SBS[o]['m']:]] for o in AL] == [['same', 'new', 'new'], ['same', 'same', 'new', 'new'], ['same', 'same', 'same', 'new', 'new']], [SBS[o]['b'] for o in AL])
check('side by side: A has the best average of the three, B and C tie within 4 tokens', ALL[AL[0]][0] == best and abs(ALL[AL[1]][0] - ALL[AL[2]][0]) <= 4)
check('the name before the history or after the message: both keep message 1 (4,594 and 4,590); between them it breaks the match (4,496)',
      ALL[('sys', 'shots', 'hist', 'msg', 'name', 'date')][0] == 4590 and ALL[AL[2]][0] == 4496)


# the premises the note names: with the date saved in the history (next to each message), B matches A; with every return on the same
# day, the date before the name and the history beats A, since even the answer hits
def ch_run2(order, stream_=STREAM, save_date=False):
    tr = Trie(int(pool_b // KV_TOK) // BS); out = []
    for i in range(len(stream_)):
        segs, (ac, an) = ch_segs(order, i, save_date, stream_)
        out.append(tr.run([{'cid': c, 'n': n_} for c, n_, _p in segs], {'cid': ac, 'n': an}))
    return sum(x['hit'] for x in out) / len(out)


check('premises: the rebuilt stream gives the same averages as ch_run (A 4,594, B 4,500, C 4,496)', [ch_run2(o) for o in AL] == [4594, 4500, 4496])
sd = {o: ch_run2(o, save_date=True) for o in AL}
check('premise 1: with the date saved in the history next to each message, B does as well as A (4,792 each)', sd[AL[0]] == sd[AL[1]] == 4792, sd)
SAME = [(1, u) for _, u in STREAM]
early = ch_run2(('sys', 'shots', 'date', 'name', 'hist', 'msg'), stream_=SAME)
check('premise 2: with every return on the same day, the date before the name and the history (4,790) beats A (4,594), and no order does better',
      early == 4790 and ch_run2(AL[0], stream_=SAME) == 4594 and max(ch_run2(o, stream_=SAME) for o in itertools.permutations(CHP)) == 4790, early)

# ---------- the exits' new cases and the drill
nc = ttft(50500, 50000)
check('exits: 50,000 cached + 500 new tokens: 40.73 ms, arithmetic sets it, in “30 to 100 ms”', round(ms(nc['t']), 2) == 40.73 and nc['tc'] > nc['tm'] and 30 <= ms(nc['t']) < 100, ms(nc['t']))
kv70 = 2 * 80 * 8 * 128 * 2
check('exits: Llama-3.1-70B’s 10,000 cached tokens hold 3.05 GiB, in “2 to 5 GiB”', kv70 == 327680 and round(10000 * kv70 / GiB, 2) == 3.05)

with sync_playwright() as pw:
    b = pw.chromium.launch()
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(80)
    pg.evaluate("()=>{location.hash='prefix-caching'}"); pg.wait_for_timeout(600)
    J = lambda expr: pg.evaluate(expr)
    c = J('()=>window.__px.consts')
    for k, v in [('P_TOTAL', total), ('P_LAYERS', layers), ('P_HEAD', head), ('P_EMB', emb), ('KV_TOK', KV_TOK), ('W_READ', W_READ), ('W_MEM', W_MEM)]:
        check(f'page constant {k} = {v:,}', c[k] == v, c[k])
    check('page pool: the KV cache memory lab’s engine gives the same 450,907 tokens', c['POOL_T'] == int(pool_b // KV_TOK) and close(c['POOL_B'], pool_b), (c['POOL_T'], c['POOL_B']))
    # the home tile and the lab: the same constants, the same tokens to prefill
    tz = J('()=>window.__tzA&&window.__tzA.consts')
    check('the home tile (set-a) is in the page', tz is not None)
    if tz:
        for k in ['P_TOTAL', 'P_LAYERS', 'P_EMB', 'P_HEAD', 'KV_TOK', 'W_READ']:
            check(f'tile and lab share {k}', tz[k] == c[k], (tz[k], c[k]))
        check('tile and lab share the peaks and efficiencies', tz['peak'] == c['H100']['peak'] and tz['bw'] == c['H100']['bw'] and tz['mfu'] == c['EFF']['mfu'] and tz['mbu'] == c['EFF']['mbu'])
        setup = J('()=>window.__tzA.pc.setup')
        check('the tile’s setup is the lab’s mission: 10,000 shared tokens and 200 of each request’s own', setup['prefix'] == 10000 and setup['own'] == 200)
        for nn in range(1, 9):
            t = J(f'()=>window.__tzA.pc.compute({nn})'); r = J(f"()=>window.__px.run({{P:10000,O:200,n:{nn}}})")
            check(f'{nn} requests: the tile’s tokens to prefill (with {t["with"]:,}, without {t["without"]:,}) equal the lab’s stream', r['tot']['pre'] == t['with'] and r['tot']['all'] == t['without'], (r['tot'], t))
        check('the tile’s KV cache of the prefix equals the lab’s 1.22 GiB', J('()=>window.__tzA.pc.compute(1).kvBytes') == 10000 * KV_TOK)
    # the page's engine against the derivation
    for N, H in [(10200, 10000), (10200, 0), (2200, 2000), (100200, 100000), (10212, 0), (10200, 4992), (50500, 50000), (2500, 2384)]:
        e = J(f'()=>window.__px.ttft({N},{H})'); d = ttft(N, H)
        check(f'ttft({N:,}, {H:,}): FLOPs, bytes and time', e['fl'] == d['fl'] and e['by'] == d['by'] and close(e['t'], d['t']), (e, d))
    def page_hits(s):
        return J(f"()=>window.__px.run({s}).res.map(x=>x.hit)")
    for ch, (h, t) in LEV.items():
        check(f'page, part 4 {ch}: hits', page_hits(f"{{change:'{ch}'}}") == [x['hit'] for x in stream(change=ch)])
        check(f'page, part 4 {ch}: hatched blocks', J(f"()=>window.__px.run({{change:'{ch}'}}).res[1].same") == stream(change=ch)[1]['same'])
    # part 5: the page's engine against the trie and the step model, position by position
    check('page, part 5 comes right after “What breaks a hit”, 7 parts in all', J('()=>window.__px.consts.STEP_IDS') == ['mission', 'predict', 'send', 'break', 'onetoken', 'cost', 'remember'], J('()=>window.__px.consts.STEP_IDS'))
    check('page, part 5: the positions are tokens 10, 5,000 and 9,990', [p['t'] for p in J('()=>window.__px.consts.OT_POS')] == [10, 5000, 9990])
    for v, x in OTX.items():
        e = J(f"()=>{{const o=window.__px.ot('{v}'); delete o.k1; delete o.k2; delete o.state; delete o.state1; return o}}")
        check(f'page, part 5 {v}: hit blocks, first missing block, tokens recomputed, first token, hatched and new blocks, their bytes, the cache after request 2',
              (e['hitBlocks'], e['firstMiss'], e['blk'], e['rec'], e['same'], e['nw'], e['bytes'], e['cached1'], e['cached2'], e['ansB']) == (x['hb'], x['first'], x['blk'], x['rec'], x['same'], x['new'], x['bytes'], x['cached1'], x['cached2'], x['ans'])
              and close(e['t'], x['t']) and close(e['wo'], ttft(x['N'], 0)['t']), (e, x))
        check(f'page, part 5 {v}: the system prompt’s tokens recomputed, its blocks stored twice and their bytes (review r1 #3)',
              (e['recSys'], e['dup'], e['dupBytes'], e['preB']) == (x['recSys'], x['dup'], x['dupBytes'], 10000 // BS), (e['recSys'], e['dup'], e['dupBytes']))
        check(f'page, part 5 {v}: the predictions that give the right count ({RIGHT[v]}), and the stops of block by block ({STOPS[v]})', e['right'] == RIGHT[v] and e['stops'] == STOPS[v], (e['right'], e['stops']))
        k1, k2, st = J(f"()=>window.__px.ot('{v}').k1"), J(f"()=>window.__px.ot('{v}').k2"), J(f"()=>window.__px.ot('{v}').state")
        nbF = x['N'] // BS
        check(f'page, part 5 {v}: the block keys agree before the changed block and differ from it to the last full block (“same tokens, different key”)',
              len(k1) == len(k2) == nbF and all(k1[b] == k2[b] for b in range(x['hb'])) and all(k1[b] != k2[b] for b in range(x['hb'], nbF)))
        check(f'page, part 5 {v}: request 2’s blocks: cached up to the change, then computed to the end, hatched where request 1 has the same tokens',
              st[:x['hb']] == [0] * x['hb'] and all(s in (1, 2) for s in st[x['hb']:]) and st.count(2) == x['same'] and (x['same'] == 0 or st[x['hb']] == 1))
        bad = []
        for k in sorted({0, 1, x['hb'], x['hb'] + 1, (x['hb'] + 638) // 2, 637, 638}):
            a = J(f"()=>window.__px.otAt('{v}',{k})"); Nk, Hk = min(x['N'], k * BS), min(k, x['hb']) * BS
            ok = a['recAll'] == Nk - Hk and a['nwAll'] == max(0, min(k, nbF) - x['hb']) and (a['t'] is None if k == 0 else close(a['t'], ttft(Nk, Hk)['t']))
            # the counters as shown (review r1 #3): the system prompt's tokens recomputed so far and its blocks stored twice so far, with their bytes
            ok = ok and a['rec'] == max(0, min(k * BS, 10000) - Hk) and a['nw'] == max(0, min(k, 10000 // BS) - x['hb']) and a['bytes'] == a['nw'] * BS * KV_TOK
            if not ok: bad.append((k, a, Nk, Hk))
        check(f'page, part 5 {v}: the three counters after k blocks (tokens recomputed, the step model’s time for the blocks so far, new blocks; the whole prompt and the system prompt) at 0, 1, the change, halfway and the end', not bad, bad[:2])
    for mem, want in EVICT.items():
        mm = f"'{mem}'" if mem == 'pool' else mem
        check(f'page, part 5 memory {mem}: hits', page_hits(f"{{n:8,apps:3,mem:{mm}}}") == want, page_hits(f"{{n:8,apps:3,mem:{mm}}}"))
        cached_py = stream(n=8, apps=3, mem=mem)[-1]['cached']
        check(f'page, part 5 memory {mem}: blocks cached after the last request ({cached_py:,})', J(f"()=>window.__px.run({{n:8,apps:3,mem:{mm}}}).res[7].cached") == cached_py)
    for name, s in SCEN.items():
        js = '{' + ','.join(f"{'P' if k == 'Pp' else k}:{v!r}" if not isinstance(v, str) else f"{k}:'{v}'" for k, v in s.items()) + '}'
        ph = page_hits(js); py = [x['hit'] for x in stream(**s)]
        check(f'page, scenario {name}: hits {py}', ph == py, (ph, py))
        tot = J(f"()=>window.__px.run({js}).tot"); sr = stream(**s)
        check(f'page, scenario {name}: prefill totals', tot['pre'] == sum(x['N'] - x['hit'] for x in sr) and tot['all'] == sum(x['N'] for x in sr) and close(tot['w'], sum(ttft(x['N'], x['hit'])['t'] for x in sr)))
    # every challenge order: the page and the trie agree
    bad = []
    for o, v in ALL.items():
        e = J(f"()=>window.__px.chRun({list(o)!r})")
        if e['hits'] != [x['hit'] for x in v[2]] or e['moves'] != v[1] or e['ok'] != (v[0] >= 4400): bad.append((o, e['hits'], [x['hit'] for x in v[2]]))
    check('page, challenge: all 720 orders give the same hits, moved parts and verdict as the trie', not bad, bad[:2])
    check('page, challenge: best 4,594, fewest moves 2', J('()=>window.__px.chBestAvg()') == 4594 and J('()=>window.__px.chMinMoves()') == 2)
    # the reveal's side-by-side: the page and the derivation agree on every part's tokens and state, the cut and the hit
    check('page, side by side: the same three orders (A best, B date before the message, C name before the message)', [tuple(o) for o in J('()=>window.__px.AL_ORDERS')] == AL)
    for o in AL:
        e, x = J(f"()=>window.__px.al({list(o)!r})"), SBS[o]
        check(f'page, side by side {",".join(o)}: parts, states, cut, hit and average as derived',
              [(p['n'], p['st']) for p in e['a']] == x['a'] and [(p['n'], p['st']) for p in e['b']] == x['b'] and (e['m'], e['match'], e['hit'], e['N'], e['avg']) == (x['m'], x['match'], x['hit'], x['N'], ALL[o][0]), (e, x))
        check(f'page, side by side {",".join(o)}: day 2’s history is the old history, message 1 and answer 1',
              [p['cid'] for p in e['b'] if p['part'] == 'hist'] == ['hist0-Alice', 'ch-msg-0', 'ch-ans-0'], e['b'])
    # the drill
    for s in range(1, 21):
        d = J(f'()=>window.__px.drill({s})')
        check(f'drill {s}: the times from the derivation, never the mission’s 10,000 + 200 or the new case’s 50,000 + 500',
              close(d['w']['t'], ttft(d['N'], d['H'])['t']) and close(d['wo']['t'], ttft(d['N'], 0)['t']) and d['H'] == (d['P'] // 16) * 16 and (d['P'], d['O']) not in [(10000, 200), (50000, 500)], d)
    # the Connect card's code, run here: its comments state its own output
    code = J("()=>document.getElementById('px-code').textContent")
    lines = code.splitlines()
    check('Connect code: at most 40 lines, none longer than 74 characters', len(lines) <= 40 and max(len(l) for l in lines) <= 74, (len(lines), max(len(l) for l in lines)))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True).stdout.strip().splitlines()
    promised = [l[2:] for l in lines if re.match(r'^# \d+ \d+$', l)]
    check('Connect code: runs and prints what its comments say (0 0, 1 10000, 2 10000)', out == promised == ['0 0', '1 10000', '2 10000'], (out, promised))
    # what the page shows: the mission's request 2 in Scenarios (the link opens it), and the challenge's initial order
    def cards():
        return J("()=>Object.fromEntries([...document.querySelectorAll('#px-metrics .metric')].map(m=>[m.dataset.k,m.querySelector('.v').textContent.replace(/\\u00a0/g,' ')]))")
    J("()=>{location.hash='map'}"); pg.wait_for_timeout(200)
    J("()=>{location.hash='prefix-caching~10000_200_2_same_none_1_pool'}"); pg.wait_for_timeout(600)
    got = cards()
    check('request 2 cards: 8.13 ms, 342.9 ms, 200 prefilled, 10,000 cached, 1.22 GiB', got == {'w': '8.13 ms', 'wo': '342.9 ms', 'pre': '200', 'hit': '10,000', 'mem': '1.22 GiB'}, got)
    tl = J("()=>document.getElementById('px-tl').innerText.replace(/\\u00a0/g,' ')")
    check('timeline: first token at 342.9 ms without, 8.13 ms with', 'first token at 342.9 ms' in tl and 'first token at 8.13 ms' in tl, tl)
    J("()=>{location.hash='map'}"); pg.wait_for_timeout(200)
    J("()=>{location.hash='prefix-caching~100000_200_4_same_none_1_pool'}"); pg.wait_for_timeout(600)
    got = cards()
    check('the long document, question 4: 26.86 ms shown as 26.9 ms, 8.15 s without, 12.21 GiB cached', got['w'] == '26.9 ms' and got['wo'] == '8.15 s' and got['mem'] == '12.21 GiB' and got['hit'] == '100,000', got)
    J("()=>document.querySelector('#px-app .modes button[data-mode=\"challenge\"]').click()"); pg.wait_for_timeout(400)
    got = cards()
    check('challenge, initial order: 0 cached tokens per request, 190.4 ms on average with and without the cache', got.get('avg') == '0' and got.get('avgw') == '190.4 ms' == got.get('avgwo'), got)
    # part 5 as shown: the counters at the end state and each position's result on its title row (reduced motion: the end state at once)
    pr = b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce')
    pr.on('pageerror', lambda e: errs.append(str(e)))
    pr.goto(URL); pr.wait_for_timeout(300); pr.click('.langsw button[data-lang="en"]'); pr.wait_for_timeout(80)
    pr.evaluate("()=>{location.hash='prefix-caching'}"); pr.wait_for_timeout(600)
    pr.evaluate("()=>document.querySelector('#px-rail [data-go=\"4\"]').click()"); pr.wait_for_timeout(300)
    pr.evaluate("()=>document.querySelector('#px-stage [data-otpos=\"middle\"]').click()"); pr.wait_for_timeout(200)
    pr.evaluate("()=>document.querySelector('#px-rail [data-otpred=\"before\"]').click()"); pr.wait_for_timeout(300)
    ctr = lambda: pr.evaluate("()=>Object.fromEntries([...document.querySelectorAll('#px-ctrs .px-ctr')].map(c=>[c.dataset.k,[c.querySelector('.v').textContent,c.querySelector('.sub').textContent].map(s=>s.replace(/\\u00a0/g,' '))]))")
    for v in ['middle', 'front', 'end']:
        pr.evaluate(f"()=>document.querySelector('#px-stage [data-otpos=\"{v}\"]').click()"); pr.wait_for_timeout(250)
        c, x = ctr(), OTX[v]
        check(f'part 5 {v} as shown: {x["recSys"]:,} system-prompt tokens recomputed, first token {fmt_t(x["t"])}, {x["dup"]} system-prompt blocks stored twice ({gibc(x["dupBytes"])})',
              c['rec'][0] == f'{x["recSys"]:,}' and c['t'][0] == fmt_t(x['t']) and c['nw'][0] == f'{x["dup"]:,}' and gibc(x['dupBytes']) in c['nw'][1] and fmt_t(ttft(x['N'], 0)['t']) in c['t'][1], c)
    lv = pr.evaluate("()=>Object.fromEntries([...document.querySelectorAll('#px-rail [data-lever]')].map(b=>[b.dataset.lever,b.querySelector('.lv-r').textContent.replace(/\\u00a0/g,' ')]))")
    fracw = lambda q: 'a small fraction' if q < 0.15 else 'about a quarter' if q < 0.3 else 'about a third' if q < 0.42 else 'about half' if q < 0.6 else 'most' if q < 0.85 else 'almost all'
    want = {v: ('none of it cached' if x['hit'] == 0 else f'{fracw(x["hit"] / 10000)} cached') for v, x in OTX.items()}
    check('part 5: each position keeps its conclusion on its title row, how much of the system prompt request 2 finds in the cache (R3.6, R3.17: none of it, about half, almost all)',
          lv == want == {'front': 'none of it cached', 'middle': 'about half cached', 'end': 'almost all cached'}, (lv, want))
    pr.close()
    check('no page errors', not errs, errs[:3])
    b.close()
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails if fails else 'none')
sys.exit(1 if fails else 0)
