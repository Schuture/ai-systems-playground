"""Serving capstone: reference event/time model (Python prototype; the lab's JS engine is a port of this file).

One H100 SXM serves Llama-3.1-8B in BF16. Time advances step by step; every step is priced by the shared step model of the
prefill/decode, chunked-prefill and prefix-caching labs: t = max(FLOPs / (989.5 TFLOP/s x 50%), bytes / (3.35 TB/s x 60%)).
  FLOPs = 2 x P_LAYERS x tokens + 2 x P_HEAD x logits + attention; bytes = W_READ + 128 KiB x (KV tokens the step reads)
  a decode token at context c (c tokens already cached): attention attn(c, c+1), reads c tokens of KV, 1 logit
  a prefill chunk a..b-1: attention attn(a, b), reads b tokens of KV, 1 logit if it is the prompt's last chunk

Scheduling, continuous batching (simplified vLLM v1):
  at a step boundary t:
    1. every running request that has its first token and still has answer tokens left decodes 1 token
    2. running requests still in prefill continue, in admission order, each taking min(prompt left, budget left)
    3. waiting requests that have arrived (arrival <= t) are admitted first come, first served, while
         running < cap, the step's token budget has room, and KV memory can hold the request's prompt beyond its cache hit and its
         whole answer (a conservative admission: room is set aside for the whole answer, so nothing is ever preempted)
       a request that cannot be admitted blocks the ones behind it (first come, first served); they wait for the same reason
       an admitted request looks up its shared prefix blocks (prefix caching on) and prefills from the first uncached token
  the budget counts every token of the step (decode tokens too), as vLLM's max_num_batched_tokens; budget 0 = no limit
  the step's time comes from its work; every event of the step (a chunk done, a first token, a decoded token, a finish) happens at its
  end; requests that arrive during a step wait for its end. A request finishes with its whole answer and releases its blocks.
  With nothing to run, time jumps to the next arrival.
Static batching: a batch of up to `cap` waiting requests starts only when the whole previous batch is done; its prompts are prefilled
  together in one step, then it decodes until its longest answer is done.

KV memory, in 16-token blocks of 128 KiB per token:
  a running request holds its prompt blocks as they are computed and its answer blocks (a new block every 16 tokens);
  with prefix caching on, the blocks of an app's shared system prompt are shared: one physical copy, a reference count per request;
  two requests that compute the same shared block in the same step keep one copy (the computation is still done twice);
  when no running request uses them they stay cached (idle, least recently used evicted first) within the cache budget, and a new
  allocation evicts idle blocks when the pool runs out. Blocks in use are never evicted. Only whole shared-prefix blocks are cached.
"""
import math

# ---------------- the shared step model (same constants as the other inference labs)
L, D, H, KVH, HD, FFN, VOCAB = 32, 4096, 32, 8, 128, 14336, 128256
P_LAYER = 2 * D * D + 2 * D * KVH * HD + 3 * D * FFN + 2 * D
P_LAYERS = L * P_LAYER + D
P_HEAD = VOCAB * D
P_TOTAL = P_LAYERS + 2 * P_HEAD
KV_TOK = 2 * L * KVH * HD * 2                       # 131,072 bytes
W_READ = 2 * (P_LAYERS + P_HEAD)                   # bytes read per step
W_MEM = 2 * P_TOTAL
FS = 1979e12 / 2 * 0.5                              # FLOP/s reached
BS = 3.35e12 * 0.6                                  # bytes/s reached
GiB = 2 ** 30
BLOCK = 16
POOL_T = math.floor((80 * GiB * 0.9 - W_MEM - 2 * GiB) / KV_TOK)   # 450,907 tokens of KV cache
POOL_B = POOL_T // BLOCK                            # 28,181 blocks
EPS = 1e-9


def attn(a, b):   # QK^T and AV FLOPs for query positions a..b-1, causal, all layers
    return 2 * L * D * (b * (b + 1) - a * (a + 1))


def blocks(n):
    return (n + BLOCK - 1) // BLOCK


# ---------------- a seeded random generator (mulberry32, the same as the site's)
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


# ---------------- the simulation
class Req:
    __slots__ = ('id', 'cls', 'app', 'arr', 'pre', 'own', 'out', 'P', 'admit', 'hit', 'done_p', 'gen', 'toks', 'first', 'fin',
                 'blk_own', 'resv', 'waits', 'chunks', 'dsteps', 'shared_new')

    def __init__(self, i, spec):
        self.id = i; self.cls = spec['cls']; self.app = spec.get('app'); self.arr = spec['t']
        self.pre = spec.get('pre', 0); self.own = spec['own']; self.out = spec['out']; self.P = self.pre + self.own
        self.admit = None; self.hit = 0; self.done_p = 0; self.gen = 0; self.toks = []; self.first = None; self.fin = None
        self.blk_own = 0          # blocks this request holds that are not shared prefix blocks
        self.resv = 0             # blocks set aside for it that it has not allocated yet
        self.waits = {}           # reason -> seconds spent waiting for it
        self.chunks = []          # (step index, a, b)
        self.dsteps = []          # step indices of its decode tokens
        self.shared_new = 0       # shared prefix blocks it computed itself (cache misses)


def simulate(reqs_spec, cfg, trace=False):
    """cfg: policy ('continuous'|'static'), cap, budget (0 = no limit: a whole prompt in one step), cache (bool), cache_blocks
    (max idle cached prefix blocks; None = no limit but the pool). Returns the run: requests, steps, events, memory samples."""
    R = [Req(i, s) for i, s in enumerate(reqs_spec)]
    order = sorted(range(len(R)), key=lambda i: (R[i].arr, i))
    cap, budget0, cache_on, policy = cfg['cap'], cfg['budget'] or 0, cfg['cache'], cfg['policy']
    cache_cap = cfg.get('cache_blocks')
    pblk = {}                     # prefix blocks: (app, i) -> {'ref': n, 'last': t}
    st = {'free': POOL_B, 'evictions': 0, 't': 0.0}
    steps, events = [], []
    waiting, running = [], []
    batch = []                    # static batching: the requests of the current batch
    nxt = 0
    peak = 0; prefill_tok = 0; hit_tok = 0; mem = []

    def idle_count():
        return sum(1 for v in pblk.values() if v['ref'] == 0)

    def reserved():
        return sum(r.resv for r in running) + sum(r.resv for r in batch if r not in running)

    def evict(n):
        idle = sorted((k for k, v in pblk.items() if v['ref'] == 0), key=lambda k: (pblk[k]['last'], k[0], -k[1]))
        got = 0
        for k in idle[:n]:
            del pblk[k]; st['free'] += 1; got += 1; st['evictions'] += 1
            if trace: events.append((st['t'], 'evict', k))
        return got

    def can_hold(n):
        return st['free'] + idle_count() - reserved() >= n

    def alloc(r, n):
        if n <= 0:
            return
        if st['free'] < n:
            evict(n - st['free'])
        assert st['free'] >= n, 'KV pool overflow (the admission check must prevent it)'
        st['free'] -= n
        r.resv = max(0, r.resv - n)

    def lookup(r):
        """leading shared-prefix blocks of r already cached (in use or idle)"""
        if not cache_on or not r.app or not r.pre:
            return 0
        nb = r.pre // BLOCK; h = 0
        while h < nb and (r.app, h) in pblk:
            h += 1
        return h

    def need_blocks(r, hitb):
        # blocks r may allocate: its prompt beyond the hit and its whole answer (conservative, no preemption)
        return blocks(r.P + r.out) - hitb

    def admit(r, now):
        nonlocal hit_tok
        hb = lookup(r)
        for i in range(hb):
            v = pblk[(r.app, i)]; v['ref'] += 1; v['last'] = now
        r.hit = hb * BLOCK; r.done_p = r.hit; r.admit = now; hit_tok += r.hit
        r.resv = need_blocks(r, hb)
        running.append(r)
        if trace: events.append((now, 'admit', r.id, r.hit))

    def trim_cache():
        if cache_cap is None:
            return
        n = idle_count()
        if n > cache_cap:
            evict(n - cache_cap)

    held = {}                     # request id -> list of shared block keys it holds a reference to

    def hold(r, key):
        held.setdefault(r.id, []).append(key)

    def release_all(r):
        st['free'] += r.blk_own; r.blk_own = 0; r.resv = 0
        for k in held.pop(r.id, []):
            v = pblk[k]; v['ref'] -= 1; v['last'] = st['t']
            assert v['ref'] >= 0
        trim_cache()

    def used():
        return POOL_B - st['free']

    guard = 0
    while True:
        guard += 1; assert guard < 400000
        t = st['t']
        while nxt < len(order) and R[order[nxt]].arr <= t + EPS:
            waiting.append(R[order[nxt]]); nxt += 1
        if not waiting and not running:
            if nxt >= len(order):
                break
            st['t'] = R[order[nxt]].arr; continue
        budget = budget0 if budget0 else 10 ** 12
        work = []                 # (r, kind, a, b)
        reasons = {}
        if policy == 'continuous':
            for r in running:                                   # 1. decodes
                if r.first is not None and r.gen < r.out:
                    work.append((r, 'd', 0, 0)); budget -= 1
            for r in running:                                   # 2. prefills in progress
                if r.first is None and r.done_p < r.P and budget > 0:
                    n = min(r.P - r.done_p, budget); work.append((r, 'p', r.done_p, r.done_p + n)); budget -= n
            blocked = None
            for r in list(waiting):                             # 3. admissions, first come first served
                if blocked is None:
                    if len(running) >= cap:
                        blocked = 'cap'
                    elif budget <= 0:
                        blocked = 'budget'
                    elif not can_hold(need_blocks(r, lookup(r))):
                        blocked = 'memory'
                if blocked is not None:
                    reasons[r.id] = blocked; continue
                waiting.remove(r); admit(r, t)
                for i in range(r.hit // BLOCK):
                    hold(r, (r.app, i))
                n = min(r.P - r.done_p, budget); work.append((r, 'p', r.done_p, r.done_p + n)); budget -= n
        else:   # static batching
            if not batch:
                while waiting and len(batch) < cap:
                    r = waiting[0]
                    if not can_hold(need_blocks(r, lookup(r))):
                        break
                    waiting.pop(0); admit(r, t); batch.append(r)
                    for i in range(r.hit // BLOCK):
                        hold(r, (r.app, i))
            for r in waiting:
                reasons[r.id] = 'batch'
            for r in batch:
                if r.fin is not None:
                    continue
                if r.first is None:
                    work.append((r, 'p', r.done_p, r.P))
                elif r.gen < r.out:
                    work.append((r, 'd', 0, 0))
        if not work:
            raise RuntimeError('deadlock: requests wait but nothing can run (a request larger than the KV pool?)')
        # allocate the blocks this step writes
        for (r, k, a, b) in work:
            if k == 'p':
                shared_end = (r.pre // BLOCK) * BLOCK if (cache_on and r.app) else 0
                if a < shared_end:
                    for i in range(a // BLOCK, min(b, shared_end) // BLOCK):
                        key = (r.app, i)
                        if key in pblk:              # computed in this same step by an earlier request: keep one copy
                            pblk[key]['ref'] += 1; pblk[key]['last'] = t
                        else:
                            alloc(r, 1); pblk[key] = {'ref': 1, 'last': t}; r.shared_new += 1
                        hold(r, key)
                own_a = max(a, shared_end)
                if b > own_a:
                    nb = blocks(b - shared_end) - blocks(own_a - shared_end)
                    alloc(r, nb); r.blk_own += nb
            else:
                c = r.P + r.gen - 1        # tokens already cached; this step writes position c
                if c % BLOCK == 0:
                    alloc(r, 1); r.blk_own += 1
        # price the step
        T = 0; logits = 0; fl_att = 0; kv = 0
        for (r, k, a, b) in work:
            if k == 'd':
                c = r.P + r.gen - 1
                T += 1; logits += 1; fl_att += attn(c, c + 1); kv += c
            else:
                T += b - a; logits += 1 if b == r.P else 0; fl_att += attn(a, b); kv += b
        fl = 2 * P_LAYERS * T + 2 * P_HEAD * logits + fl_att
        by = W_READ + KV_TOK * kv
        tc, tm = fl / FS, by / BS
        dt = max(tc, tm)
        si = len(steps)
        t1 = t + dt
        steps.append({'t0': t, 't1': t1, 'tc': tc, 'tm': tm, 'T': T, 'fl': fl, 'by': by,
                      'work': [(r.id, k, a, b) for (r, k, a, b) in work],
                      'nrun': len(running) if policy == 'continuous' else len(batch)})
        for r in waiting:
            why = reasons.get(r.id, 'batch')
            r.waits[why] = r.waits.get(why, 0) + dt
        for j in range(nxt, len(order)):                        # arrivals during the step wait for its end
            q = R[order[j]]
            if q.arr < t1 - EPS:
                q.waits['step'] = q.waits.get('step', 0) + (t1 - q.arr)
            else:
                break
        st['t'] = t1
        for (r, k, a, b) in work:
            if k == 'p':
                r.chunks.append((si, a, b)); prefill_tok += b - a; r.done_p = b
                if b == r.P:
                    r.first = t1; r.toks.append(t1); r.gen = 1
            else:
                r.gen += 1; r.toks.append(t1); r.dsteps.append(si)
        peak = max(peak, used())
        fin_now = []
        for r in list(running):
            if r.first is not None and r.gen >= r.out:
                r.fin = t1; running.remove(r); fin_now.append(r)
        for r in fin_now:
            release_all(r)
            if trace: events.append((t1, 'finish', r.id))
        if policy == 'static' and batch and all(r.fin is not None for r in batch):
            batch = []
        mem.append((t1, used(), idle_count(), reserved()))
    return {'R': R, 'steps': steps, 'events': events, 'peak': peak, 'evictions': st['evictions'], 'prefill_tok': prefill_tok,
            'hit_tok': hit_tok, 'mem': mem, 'cfg': cfg}


# ---------------- metrics
def pct(xs, p):
    """nearest-rank percentile"""
    s = sorted(xs)
    if not s:
        return None
    k = max(1, math.ceil(p / 100 * len(s) - 1e-9))
    return s[k - 1]


def gap_of(r):
    return max((b - a) for a, b in zip(r.toks, r.toks[1:])) if len(r.toks) > 1 else 0.0


def metrics(run, goals):
    R = run['R']
    out = {}
    for cls in ('chat', 'doc'):
        rs = [r for r in R if r.cls == cls]
        if not rs:
            continue
        ttft = [r.first - r.arr for r in rs]
        gaps = [gap_of(r) for r in rs]
        g = goals.get(cls, {})
        ok = [((r.first - r.arr) <= g.get('ttft', 1e9) + EPS) and (gp <= g.get('gap', 1e9) + EPS) for r, gp in zip(rs, gaps)]
        out[cls] = {'n': len(rs), 'ttft_p50': pct(ttft, 50), 'ttft_p95': pct(ttft, 95), 'ttft_max': max(ttft),
                    'gap_max': max(gaps), 'gap_p95': pct(gaps, 95), 'ok': sum(ok), 'ok_frac': sum(ok) / len(rs)}
    toks = sum(r.out for r in R)
    t0 = min(r.arr for r in R); t1 = max(r.fin for r in R)
    out['all'] = {'n': len(R), 'ok': sum(out[c]['ok'] for c in out), 'thr': toks / (t1 - t0), 'makespan': t1 - t0,
                  'peak_gib': run['peak'] * BLOCK * KV_TOK / GiB, 'prefill_tok': run['prefill_tok'], 'hit_tok': run['hit_tok'],
                  'evictions': run['evictions'], 'steps': len(run['steps'])}
    out['all']['ok_frac'] = out['all']['ok'] / len(R)
    return out


def check(run):
    """invariants: no work before arrival, time moves forward, every request finishes once with its whole answer, memory adds up"""
    R = run['R']; st = run['steps']
    for a, b in zip(st, st[1:]):
        assert b['t0'] >= a['t1'] - EPS, 'time went backwards'
    seen = set()
    for s in st:
        ids = [w[0] for w in s['work']]
        assert len(ids) == len(set(ids)), 'a request twice in one step'
        for (i, k, a, b) in s['work']:
            assert R[i].arr <= s['t0'] + EPS, 'work before arrival'
            if k == 'd':
                assert R[i].first is not None and R[i].first <= s['t0'] + EPS, 'decode before first token'
    for r in R:
        assert r.fin is not None and len(r.toks) == r.out, (r.id, r.fin, len(r.toks), r.out)
        assert r.admit >= r.arr - EPS and r.first > r.admit and r.fin >= r.first
        assert sum(b - a for (_, a, b) in r.chunks) + r.hit == r.P, 'prompt tokens do not add up'
        assert all(x < y for x, y in zip(r.toks, r.toks[1:])), 'tokens out of order'
    for (t, u, idle, resv) in run['mem']:
        assert 0 <= u <= POOL_B and u + resv <= POOL_B + 1, 'memory over the pool'
    return True


# ---------------- the fixed loads (seeded)
def exp_gap(rnd, rate):
    return -math.log(1 - rnd()) / rate


def ms(x):                      # rounded to the millisecond, as the JS port does it
    return math.floor(x * 1000 + 0.5) / 1000


def uni(rnd, lo, hi):          # integer in [lo, hi]
    return lo + int(rnd() * (hi - lo + 1))


def load_chats_doc(seed=11):
    """L1: short chats; a 32K-token document arrives at 2.5 s"""
    rnd = mulberry(seed); out = []; t = 0.0
    for i in range(36):
        t += exp_gap(rnd, 4.0)
        out.append({'t': ms(t), 'cls': 'chat', 'own': uni(rnd, 400, 1600), 'out': uni(rnd, 120, 320)})
    out.append({'t': 2.5, 'cls': 'doc', 'own': 32768, 'out': 200})
    out.sort(key=lambda s: s['t'])
    return out


def load_shared(seed=12, n=48, rate=6.0, pre=6144, app='support'):
    """L2: one app; every request starts with the same 6,144-token system prompt"""
    rnd = mulberry(seed); out = []; t = 0.0
    for i in range(n):
        t += exp_gap(rnd, rate)
        out.append({'t': ms(t), 'cls': 'chat', 'app': app, 'pre': pre, 'own': uni(rnd, 200, 800), 'out': uni(rnd, 100, 300)})
    return out


def load_apps(seed=22, n=48, rate=1.5, pre=24576, mix=(('manual', .5), ('policy', .25), ('codebase', .15), ('paper', .1))):
    """L3: questions about four long documents (24,576 tokens each, the shared prefix), sharing a limited cache"""
    rnd = mulberry(seed); out = []; t = 0.0
    for i in range(n):
        t += exp_gap(rnd, rate)
        u = rnd(); acc = 0; app = mix[-1][0]
        for a, p in mix:
            acc += p
            if u < acc:
                app = a; break
        out.append({'t': ms(t), 'cls': 'chat', 'app': app, 'pre': pre, 'own': uni(rnd, 100, 400), 'out': uni(rnd, 40, 120)})
    return out


if __name__ == '__main__':
    import sys
    G = {'chat': {'ttft': 1.0, 'gap': 0.15}, 'doc': {'ttft': 5.0, 'gap': 0.15}}
    for name, ld in (('L1', load_chats_doc()), ('L2', load_shared()), ('L3', load_apps())):
        print('==', name, len(ld), 'requests, last arrival', ld[-1]['t'])
        for cfg in ({'policy': 'continuous', 'cap': 64, 'budget': 0, 'cache': False},
                    {'policy': 'continuous', 'cap': 64, 'budget': 2048, 'cache': False},
                    {'policy': 'continuous', 'cap': 64, 'budget': 2048, 'cache': True},
                    {'policy': 'static', 'cap': 16, 'budget': 0, 'cache': False}):
            run = simulate(ld, cfg); check(run); m = metrics(run, G)
            s = ' '.join(f"{c}: ttft p50 {m[c]['ttft_p50']:.3f} p95 {m[c]['ttft_p95']:.3f} max {m[c]['ttft_max']:.3f} gap {m[c]['gap_max']*1e3:.0f}ms ok {m[c]['ok']}/{m[c]['n']}"
                         for c in ('chat', 'doc') if c in m)
            a = m['all']
            print(f"  {cfg['policy'][:4]} cap {cfg['cap']} b {cfg['budget']} cache {cfg['cache']}: {s} | thr {a['thr']:.0f} tok/s peak {a['peak_gib']:.2f} GiB prefill {a['prefill_tok']} hit {a['hit_tok']} ev {a['evictions']} steps {a['steps']}")


def load_challenge(seed=21, n=64, rate=6.0):
    """C: three apps with their own system prompts, plain chats and two long documents (none of it shown in the guided parts)"""
    apps = (('support', 4096, .2), ('code', 8192, .15), ('contracts', 12288, .1))
    rnd = mulberry(seed); out = []; t = 0.0
    for i in range(n):
        t += exp_gap(rnd, rate)
        u = rnd(); acc = 0; s = None
        for a, pre, p in apps:
            acc += p
            if u < acc:
                s = {'cls': 'chat', 'app': a, 'pre': pre, 'own': uni(rnd, 200, 800)}; break
        if s is None:
            s = {'cls': 'chat', 'own': uni(rnd, 300, 1500)}
        s['t'] = ms(t); s['out'] = uni(rnd, 80, 240)
        out.append(s)
    out.append({'t': 3.0, 'cls': 'doc', 'own': 24576, 'out': 200})
    out.append({'t': 8.0, 'cls': 'doc', 'own': 16384, 'out': 200})
    out.sort(key=lambda s: s['t'])
    return out


LOADS = {'stall': load_chats_doc, 'shared': load_shared, 'apps': load_apps, 'challenge': load_challenge}
GOALS = {'chat': {'ttft': 1.0, 'gap': 0.15}, 'doc': {'ttft': 5.0}}
TARGET = 0.9
BASELINE = {'policy': 'continuous', 'cap': 32, 'budget': 0, 'cache': False, 'cache_blocks': None}
