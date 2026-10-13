"""Numbers test for the vector index lab: an independent Python twin rebuilds every index from the spec and compares with the page.

The twin (numpy for k-means and distances, plain Python for the graph) follows specs/vector-index.md:
- data: mulberry32(seed); ncl directions with coordinates 2u - 1; a point = direction + noise * (u1 + u2 + u3 + u4 - 2) per coordinate,
  scaled to length 1 (32-dimensional sets) and stored as floor(x * 4096 + 0.5); squared distances are exact integers; ties by smaller id;
- exact search, IVF (k-means: seeded partial shuffle for the start, rounds of assign + rounded mean), HNSW (hnswlib's insertion,
  heuristic and search; levels by coin flips with odds 1/M), IVF-PQ (256 centres per piece, a table of distances, codes in IVF lists),
  a k-d tree (median of (coordinate, id), axis = depth mod d, branch and bound);
- every recall@10 and vectors-compared figure of every setting, the k-d tree shares, the 2-D picture (the question, its cells, its
  graph, its PQ grid), bytes per vector and the real-scale memory and time, the challenge's answer, the drill, the home tile's numbers;
- the page text: observations, scenario insights and the challenge result quote the engine's numbers;
- ef outside the menu (review b10r1, B1): the page scans ef 10 to 400 for M = 4 and 8, and the twin rebuilds the edges of the ranges
  the challenge's answer quotes (EF_FREE); IVF and IVF-PQ never meet both bars at any nprobe from 1 to 50; at the same number of
  comparisons, M = 32 is no better than M = 16 (the FAISS-defaults scenario says so).
Run: python3 tests/vector-index_numbers.py"""
import heapq, math, pathlib, re, subprocess, sys
import numpy as np
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
(P / 'out' / 'vector-index').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/vector-index.html', '--only', 'vector-index'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'vector-index' / 'wrapped-num.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'vector-index.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'vector-index' / 'wrapped-num.html')
fails, n = [], 0


def pc1(x):
    t = f'{x*100:.1f}'
    return (t[:-2] if t.endswith('.0') else t) + '%'


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


U32 = 0xFFFFFFFF


class Mulberry:
    def __init__(self, seed): self.a = seed & U32
    def __call__(self):
        self.a = (self.a + 0x6D2B79F5) & U32
        t = ((self.a ^ (self.a >> 15)) * (1 | self.a)) & U32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & U32)) & U32) ^ t
        return ((t ^ (t >> 14)) & U32) / 4294967296


SEED, D, N, NQ, NCL, NOISE, SC, K = 20261010, 32, 5000, 100, 200, 1.2, 4096, 10
NLIST, NTRAIN, KM_IT, PQ_K, PQ_IT, PQ_TRAIN, EFC = 50, 2000, 8, 256, 6, 2500, 40
NPROBES, EFS, MS, PQMS, KDD = [1, 2, 4, 8, 16, 32], [10, 16, 32, 64, 128, 256], [4, 8, 16, 32], [4, 8, 16], [2, 8, 32]


def gen(seed, d, n, ncl, noise, unit):
    r = Mulberry(seed); C = [[r() * 2 - 1 for _ in range(d)] for _ in range(ncl)]; X = np.zeros((n, d), dtype=np.int64)
    for i in range(n):
        c = math.floor(r() * ncl); v = []
        for k in range(d):
            e = r() + r() + r() + r() - 2; v.append(C[c][k] + noise * e)
        s = 1.0
        if unit:
            s = 0.0
            for x in v: s += x * x
            s = math.sqrt(s)
        X[i] = [math.floor(x / s * SC + 0.5) for x in v]
    return X


def sqd(A, b): return ((A - b) ** 2).sum(-1)


def topk(dist, ids, k): return [int(ids[j]) for j in np.lexsort((ids, dist))[:k]]


def kmeans(X, rows, k, it, seed):
    r = Mulberry(seed); p = list(rows)
    for i in range(k):
        j = i + math.floor(r() * (len(p) - i)); p[i], p[j] = p[j], p[i]
    C = X[p[:k]].copy(); S = X[list(rows)]
    for _ in range(it):
        a = np.argmin(((S[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        for c in range(k):
            m = a == c
            if m.any(): C[c] = np.floor(S[m].sum(0) / m.sum() + 0.5).astype(np.int64)
    return C


def nearest(X, C): return np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)


def ivf_build(X, nlist, ntrain, it, seed):
    C = kmeans(X, range(ntrain), nlist, it, seed); a = nearest(X, C)
    return C, [list(np.where(a == c)[0]) for c in range(nlist)]


def probe_order(C, q):
    d = sqd(C, q); return list(np.lexsort((np.arange(len(C)), d)))


def ivf_search(X, C, lists, q, nprobe, k):
    o = probe_order(C, q); ids = np.array([j for c in o[:nprobe] for j in lists[c]], dtype=np.int64)
    return topk(sqd(X[ids], q), ids, k), len(C) + len(ids), o[:nprobe]


class HNSW:
    """hnswlib's algorithm (hnswalg.h) with the lab's rules: heaps by (distance, id), levels by coin flips."""
    def __init__(self, X, M, efc, seed):
        self.X, self.M = X, M; n = len(X); r = Mulberry(seed)
        self.Dm = np.stack([sqd(X, X[i]) for i in range(n)]) if n <= 6000 else None
        self.links, self.level, self.entry, self.top = [], [0] * n, 0, -1
        for i in range(n):
            l = 0
            while r() < 1 / M: l += 1
            self.level[i] = l
            while len(self.links) <= l: self.links.append({})
            for L in range(l + 1): self.links[L][i] = []
            if self.top < 0: self.entry, self.top = i, l; continue
            row = self.Dm[i]; dq = lambda e: int(row[e])
            cur = self.entry; dc = dq(cur)
            for L in range(self.top, l, -1):
                ch = True
                while ch:
                    ch = False
                    for e in self.links[L][cur]:
                        de = dq(e)
                        if de < dc or (de == dc and e < cur): dc, cur, ch = de, e, True
            for L in range(min(l, self.top), -1, -1):
                W = self.search_layer(dq, cur, dc, efc, L)
                sel = self.heur(W, M); self.links[L][i] = [x[1] for x in sel]; mmax = 2 * M if L == 0 else M
                for (dx, x) in sel:
                    nb = self.links[L][x]
                    if len(nb) < mmax: nb.append(i)
                    else:
                        cs = [(self.dist(x, e), e) for e in nb] + [(dx, i)]
                        self.links[L][x] = [y[1] for y in self.heur(cs, mmax)]
                cur, dc = sel[0][1], sel[0][0]
            if l > self.top: self.top, self.entry = l, i

    def dist(self, a, b): return int(self.Dm[a, b])

    def heur(self, cands, M):
        s = sorted(cands)
        if len(s) <= M: return s
        keep = []
        for c in s:
            if len(keep) >= M: break
            if all(self.dist(k[1], c[1]) >= c[0] for k in keep): keep.append(c)
        return keep

    def search_layer(self, dq, ep, dep, ef, L, trace=None):
        C = [(dep, ep)]; W = [(-dep, -ep)]; vis = {ep}
        while C:
            c = heapq.heappop(C)
            if len(W) >= ef and c > (-W[0][0], -W[0][1]): break
            for e in self.links[L][c[1]]:
                if e in vis: continue
                vis.add(e); x = (dq(e), e)
                if len(W) < ef or x < (-W[0][0], -W[0][1]):
                    heapq.heappush(C, x); heapq.heappush(W, (-x[0], -x[1]))
                    if len(W) > ef: heapq.heappop(W)
        return [(-a, -b) for (a, b) in W]

    def search(self, q, ef, k):
        cnt = [0]; dq_row = sqd(self.X, q)
        def dq(e): cnt[0] += 1; return int(dq_row[e])
        cur = self.entry; dc = dq(cur)
        for L in range(self.top, 0, -1):
            ch = True
            while ch:
                ch = False
                for e in self.links[L][cur]:
                    de = dq(e)
                    if de < dc or (de == dc and e < cur): dc, cur, ch = de, e, True
        W = self.search_layer(dq, cur, dc, max(ef, k), 0)
        return [x[1] for x in sorted(W)[:k]], cnt[0]


def pq_build(X, m, ntrain, seed):
    w = D // m; CB = [kmeans(X[:, p * w:(p + 1) * w], range(ntrain), PQ_K, PQ_IT, seed + p) for p in range(m)]
    codes = np.stack([nearest(X[:, p * w:(p + 1) * w], CB[p]) for p in range(m)], axis=1)
    return w, CB, codes


def ivfpq_search(C, lists, w, CB, codes, q, nprobe, k):
    o = probe_order(C, q); m = len(CB)
    T = np.stack([sqd(CB[p], q[p * w:(p + 1) * w]) for p in range(m)])
    ids = np.array([j for c in o[:nprobe] for j in lists[c]], dtype=np.int64)
    s = T[np.arange(m)[None, :], codes[ids]].sum(1) if len(ids) else np.zeros(0, dtype=np.int64)
    return topk(s, ids, k), len(C) + PQ_K + len(ids)


def kd_build(X):
    d = X.shape[1]
    def rec(arr, dep):
        if not arr: return None
        ax = dep % d; arr = sorted(arr, key=lambda a: (int(X[a, ax]), a)); m = len(arr) // 2
        return (arr[m], ax, int(X[arr[m], ax]), rec(arr[:m], dep + 1), rec(arr[m + 1:], dep + 1))
    sys.setrecursionlimit(10000); return rec(list(range(len(X))), 0)


def kd_search(X, T, q):
    best, bi, seen = [math.inf], [-1], [0]; dq = sqd(X, q)
    def go(nd):
        if nd is None: return
        i, ax, v, l, r = nd; dd = int(dq[i]); seen[0] += 1
        if dd < best[0] or (dd == best[0] and i < bi[0]): best[0], bi[0] = dd, i
        diff = int(q[ax]) - v; near, far = (l, r) if diff < 0 else (r, l)
        go(near)
        if diff * diff <= best[0]: go(far)
    go(T); return bi[0], seen[0]


# ---------------- the twin ----------------
XALL = gen(SEED, D, N + NQ, NCL, NOISE, True); X, Q = XALL[:N], XALL[N:]
GT = [topk(sqd(X, Q[q]), np.arange(N), K) for q in range(NQ)]
rec = lambda ids, q: len(set(ids) & set(GT[q])) / K
CIV, LISTS = ivf_build(X, NLIST, NTRAIN, KM_IT, SEED + 1)
TW = {}
for np_ in NPROBES:
    r = [ivf_search(X, CIV, LISTS, Q[q], np_, K) for q in range(NQ)]
    TW[('ivf', np_, 0)] = (sum(rec(x[0], q) for q, x in enumerate(r)) / NQ, sum(x[1] for x in r) / NQ)
HN = {}
for M_ in MS:
    G = HNSW(X, M_, EFC, SEED + 2 + M_); HN[M_] = G
    for ef in EFS:
        r = [G.search(Q[q], ef, K) for q in range(NQ)]
        TW[('hnsw', M_, ef)] = (sum(rec(x[0], q) for q, x in enumerate(r)) / NQ, sum(x[1] for x in r) / NQ)
    print('built HNSW M =', M_, flush=True)
# B1: the edges of the ef ranges outside the menu that the challenge's answer quotes (the page's EF_FREE)
EF_FREE = {8: (70, 120), 4: (210, 244)}
for M_, (lo, hi) in EF_FREE.items():
    for ef in (lo - 1, lo, hi, hi + 1):
        r = [HN[M_].search(Q[q], ef, K) for q in range(NQ)]
        TW[('hnswx', M_, ef)] = (sum(rec(x[0], q) for q, x in enumerate(r)) / NQ, sum(x[1] for x in r) / NQ)
PQB = {}
for m in PQMS:
    w, CB, codes = pq_build(X, m, PQ_TRAIN, SEED + 100 + m); PQB[m] = codes
    for np_ in NPROBES:
        r = [ivfpq_search(CIV, LISTS, w, CB, codes, Q[q], np_, K) for q in range(NQ)]
        TW[('pq', m, np_)] = (sum(rec(x[0], q) for q, x in enumerate(r)) / NQ, sum(x[1] for x in r) / NQ)
KDS = {}
for d in KDD:
    if d == D: XB, QB = X, Q
    else: A2 = gen(SEED + 7 + d, d, N + NQ, NCL, NOISE, True); XB, QB = A2[:N], A2[N:]
    T = kd_build(XB); KDS[d] = sum(kd_search(XB, T, QB[q])[1] for q in range(NQ)) / NQ / N
# the 2-D picture
A2 = gen(SEED + 20, 2, 340, 8, 0.35, False); PX, PQs = A2[:300], A2[300:]
PC, PL = ivf_build(PX, 16, 300, 10, SEED + 21)
PQI = next(i for i in range(40) if 3 <= len(set(ivf_search(PX, PC, PL, PQs[i], 1, K)[0]) & set(topk(sqd(PX, PQs[i]), np.arange(300), K))) <= 7)
PGT = topk(sqd(PX, PQs[PQI]), np.arange(300), K)
PG = HNSW(PX, 4, 20, SEED + 22)


def pic_pq(c):
    cx = kmeans(PX[:, 0:1], range(300), c, 10, SEED + 30)[:, 0]; cy = kmeans(PX[:, 1:2], range(300), c, 10, SEED + 31)[:, 0]
    sx = cx[nearest(PX[:, 0:1], cx[:, None])]; sy = cy[nearest(PX[:, 1:2], cy[:, None])]
    q = PQs[PQI]; d = (sx - q[0]) ** 2 + (sy - q[1]) ** 2
    return topk(d, np.arange(300), K)


def bytes_per(s, d):
    return 4 * d if s['ix'] == 'flat' else 4 * d + 8 if s['ix'] == 'ivf' else 4 * d + 8 * s['M'] if s['ix'] == 'hnsw' else s['m'] + 8


def tw_stat(s):
    if s['ix'] == 'flat': return (1.0, float(N))
    if s['ix'] == 'ivf': return TW[('ivf', s['np'], 0)]
    if s['ix'] == 'hnsw': return TW[('hnsw', s['M'], s['ef'])]
    return TW[('pq', s['m'], s['np'])]


ALL = [{'ix': 'flat'}] + [{'ix': 'ivf', 'np': x} for x in NPROBES] + [{'ix': 'hnsw', 'M': a, 'ef': b} for a in MS for b in EFS] + [{'ix': 'pq', 'm': a, 'np': b} for a in PQMS for b in NPROBES]
OK = [s for s in ALL if tw_stat(s)[0] >= 0.95 - 1e-9 and tw_stat(s)[1] <= 1000]
MINB = min(bytes_per(s, D) for s in OK); BEST = [s for s in OK if bytes_per(s, D) == MINB]

# ---------------- compare with the page ----------------
with sync_playwright() as pw:
    b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(250); pg.evaluate("()=>{location.hash='vector-index'}"); pg.wait_for_timeout(600)
    C = pg.evaluate("()=>window.__vix.consts")
    check('constants', [C['SEED'], C['D'], C['N'], C['NQ'], C['NCL'], C['NLIST'], C['EFC'], C['PQ_K']] == [SEED, D, N, NQ, NCL, NLIST, EFC, PQ_K], C)
    check('data: first and last vectors', pg.evaluate("()=>[window.__vix.data(0),window.__vix.data(4999)]") == [list(map(int, X[0])), list(map(int, X[4999]))])
    check('exact search: the true 10 nearest of all 100 questions', pg.evaluate("()=>window.__vix.gt()") == GT)
    check('IVF lists', pg.evaluate("()=>window.__vix.ivfLists()") == [[int(j) for j in l] for l in LISTS])
    for M_ in MS:
        check(f'HNSW M={M_}: levels', pg.evaluate(f"()=>window.__vix.hnswLevels({M_})") == HN[M_].level)
        lk = pg.evaluate(f"()=>window.__vix.hnswLinks({M_})")
        check(f'HNSW M={M_}: bottom-layer links', lk == [HN[M_].links[0][i] for i in range(N)])
    for m in PQMS: check(f'PQ m={m}: the first 200 code bytes', pg.evaluate(f"()=>window.__vix.pqCodes({m})") == [int(c) for c in PQB[m].reshape(-1)[:200]])
    for key, (r, c) in TW.items():
        if key[0] == 'hnswx': continue   # the B1 edges, checked with the challenge below
        s = pg.evaluate(f"()=>window.__vix.stat('{key[0]}',{key[1]},{key[2]})")
        check(f'stat {key}', abs(s['rec'] - r) < 1e-12 and abs(s['comps'] - c) < 1e-9, (s, r, c))
    kd = pg.evaluate("()=>window.__vix.kd()")
    for d in KDD: check(f'k-d tree share, {d} dims', abs(kd[str(d)] - KDS[d]) < 1e-12, (kd, KDS))
    pic = pg.evaluate("()=>window.__vix.pic()")
    check('picture: the question and its 10 nearest', pic['q'] == PQI and pic['gt'] == PGT, (pic['q'], PQI))
    check('picture: cells', pic['lists'] == [[int(j) for j in l] for l in PL])
    check('picture: graph levels', pic['level'] == PG.level and pic['entry'] == PG.entry)
    for c in (4, 8, 16): check(f'picture: PQ grid with {c} centres', pg.evaluate(f"()=>window.__vix.picPQ({c}).ids") == pic_pq(c))
    pv = pg.evaluate("()=>window.__vix.picEv('ivf',1,16)")
    check('picture: IVF probing 1 cell finds 3 to 7 of the 10', 3 <= len(set(pv['found']) & set(PGT)) <= 7)
    # bytes per vector and real scale
    bp = lambda s, d: pg.evaluate(f"()=>window.__vix.bytesPer({s},{d})".replace('True', 'true'))
    for s in ({'ix': 'flat'}, {'ix': 'ivf', 'np': 1}, {'ix': 'hnsw', 'M': 16, 'ef': 10}, {'ix': 'pq', 'm': 8, 'np': 1}):
        import json
        js = json.dumps(s)
        check(f'bytes per vector {s}', pg.evaluate(f"()=>window.__vix.bytesPer({js},32)") == bytes_per(s, 32) and pg.evaluate(f"()=>window.__vix.bytesPer({js},768)") == bytes_per(s, 768))
    check('real scale: 1B × 768 float32 = 3,072 GB, 30.7 s at 100 GB/s', 1e9 * 768 * 4 / 1e9 == 3072 and round(1e9 * 768 * 4 / 100e9, 1) == 30.7)
    check('real scale: IVF-PQ 64-byte codes + 8-byte ids = 72 GB; HNSW M=32 links 256 GB', 1e9 * (64 + 8) / 1e9 == 72 and 1e9 * 64 * 4 / 1e9 == 256)
    # the challenge
    ch = pg.evaluate("()=>window.__vix.chAll()")
    okp = [x['s'] for x in ch if x['ok']]
    check('challenge: the settings that pass', sorted(map(str, okp)) == sorted(map(str, [{k: v for k, v in s.items()} for s in OK])), (okp, OK))
    check('challenge: the answer is HNSW, M = 16, ef = 64, 256 bytes', BEST == [{'ix': 'hnsw', 'M': 16, 'ef': 64}] and MINB == 256, (BEST, MINB))
    # B1: ef is any whole number in hnswlib and FAISS, so the answer is only the fewest bytes among the menu's ef; check the ranges it quotes
    passes = lambda rc: rc[0] >= 0.95 - 1e-9 and rc[1] <= 1000
    check('B1: the page quotes the same ef ranges as this test', {int(k): tuple(v) for k, v in C['EF_FREE'].items()} == EF_FREE, C['EF_FREE'])
    for M_, (lo, hi) in EF_FREE.items():
        tw = [passes(TW[('hnswx', M_, ef)]) for ef in (lo - 1, lo, hi, hi + 1)]
        check(f'B1 twin: M = {M_} passes at ef {lo} and {hi}, not at {lo - 1} or {hi + 1}', tw == [False, True, True, False], tw)
        for ef in (lo - 1, lo, hi, hi + 1):
            sx = pg.evaluate(f"()=>window.__vix.stat('hnsw',{M_},{ef})"); r_, c_ = TW[('hnswx', M_, ef)]
            check(f'B1 page = twin: M = {M_}, ef = {ef}', abs(sx['rec'] - r_) < 1e-12 and abs(sx['comps'] - c_) < 1e-9, (sx, r_, c_))
    SC = pg.evaluate("""()=>{const V=window.__vix, o={};
      for(const M of [4,8]){const a=[]; for(let ef=10;ef<=400;ef++){const s=V.stat('hnsw',M,ef); if(s.rec>=0.95-1e-9&&s.comps<=1000) a.push(ef);} o['p'+M]=a;}
      const c16=[]; for(let ef=10;ef<=256;ef++){const s=V.stat('hnsw',16,ef); c16.push([s.comps,s.rec]);} o.c16=c16;
      o.c32=[10,16,32,64].map(ef=>{const s=V.stat('hnsw',32,ef); return [s.comps,s.rec];});
      o.ivf=[]; for(let np=1;np<=50;np++){const s=V.stat('ivf',np,0); if(s.rec>=0.95-1e-9&&s.comps<=1000) o.ivf.push(np);}
      o.pq=[]; for(const m of [4,8,16]) for(let np=1;np<=50;np++){const s=V.stat('pq',m,np); if(s.rec>=0.95-1e-9&&s.comps<=1000) o.pq.push([m,np]);}
      return o;}""")
    for M_, (lo, hi) in EF_FREE.items():
        check(f'B1 page scan: with M = {M_}, exactly ef {lo} to {hi} of 10 to 400 meet both bars', SC[f'p{M_}'] == list(range(lo, hi + 1)), SC[f'p{M_}'][:5])
    check('B1 page scan: IVF meets both bars at no nprobe from 1 to 50', SC['ivf'] == [], SC['ivf'])
    check('B1 page scan: IVF-PQ meets both bars at no code size and nprobe from 1 to 50', SC['pq'] == [], SC['pq'])
    c16 = sorted(SC['c16'])
    def at16(c):
        for (c0, r0), (c1, r1) in zip(c16, c16[1:]):
            if c0 <= c <= c1: return r0 + (math.log(c) - math.log(c0)) / (math.log(c1) - math.log(c0)) * (r1 - r0) if c1 > c0 else r0
    dif = [r32 - at16(c32) for c32, r32 in SC['c32']]
    check('S5: at the same comparisons, M = 32 is within 2 points of M = 16 and never clearly above it', all(-0.02 < x < 0.005 for x in dif), dif)
    # the drill
    for sd in (1, 2, 3):
        dr = pg.evaluate(f"()=>window.__vix.drill({sd})")
        check(f'drill {sd}', abs(dr['comps'] - (dr['nl'] + dr['np'] * dr['n'] / dr['nl'])) < 1e-6 and dr['mem'] == dr['n'] * (4 * dr['d'] + 8))
    # page text: the guided observations quote the engine
    def go_step(i):
        pg.evaluate(f"()=>document.querySelector('#vix-rail [data-go=\"{i}\"]')?document.querySelector('#vix-rail [data-go=\"{i}\"]').click():null"); pg.wait_for_timeout(300)
    tx = lambda sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' '):''}", sel)
    pg.evaluate("()=>{const b=[...document.querySelectorAll('#vix-rail [data-nav=\"next\"]')].find(x=>x.offsetParent); b&&b.click();}"); pg.wait_for_timeout(300)
    pg.evaluate("()=>document.querySelector('#vix-rail .seg-ctl[data-key=\"rn\"] button[data-v=\"1e9\"]').click()"); pg.wait_for_timeout(300)
    o = tx('#vix-rail [data-obs]'); check('part 2: 1B vectors take 3,072 GB and 30.7 s', '3,072 GB' in o and '30.7 s' in o, o)
    pg.evaluate("()=>{const b=[...document.querySelectorAll('#vix-rail [data-nav=\"next\"]')].find(x=>x.offsetParent); b&&b.click();}"); pg.wait_for_timeout(300)
    pg.evaluate("()=>document.querySelector('#vix-rail .seg-ctl[data-key=\"kd\"] button[data-v=\"32\"]').click()"); pg.wait_for_timeout(300)
    o = tx('#vix-rail [data-obs]'); check('part 3: at 32 dims the share is over 99.9% but not 100%, and the page says so', 0.999 < KDS[32] < 1 and 'over 99.9%' in o and 'almost every vector' in o, (KDS[32], o))
    for i in range(2):
        pg.evaluate("()=>{const p=document.querySelector('#vix-rail [data-pred=\"miss\"]'); p&&p.click(); const b=[...document.querySelectorAll('#vix-rail [data-nav=\"next\"]')].find(x=>x.offsetParent); b&&!b.disabled&&b.click();}"); pg.wait_for_timeout(300)
    pg.evaluate("()=>document.querySelector('#vix-rail .seg-ctl[data-key=\"np\"] button[data-v=\"16\"]').click()"); pg.wait_for_timeout(300)
    o = tx('#vix-rail [data-obs]'); r16, c16 = TW[('ivf', 16, 0)]
    check('part 5: nprobe 16 numbers', pc1(r16) in o and f'{round(c16):,}' in o, o)
    big = 65536 + 16 * 1e9 / 65536
    check('part 5: at 1B in 65,536 cells, probing 16 takes about 0.03% as many comparisons as vectors', f'{big / 1e9 * 100:.2f}%' == '0.03%' and '65,536 cells' in o and '0.03%' in o, (big, o))
    pg.evaluate("()=>{const b=[...document.querySelectorAll('#vix-rail [data-nav=\"next\"]')].find(x=>x.offsetParent); b&&b.click();}"); pg.wait_for_timeout(600)
    pg.evaluate("()=>document.querySelector('#vix-rail .seg-ctl[data-key=\"ef\"] button[data-v=\"64\"]').click()"); pg.wait_for_timeout(300)
    o = tx('#vix-rail [data-obs]'); r, c = TW[('hnsw', 16, 64)]
    check('part 6: ef 64 numbers', pc1(r) in o and f'{round(c):,}' in o, o)
    pg.evaluate("()=>{const b=[...document.querySelectorAll('#vix-rail [data-nav=\"next\"]')].find(x=>x.offsetParent); b&&b.click();}"); pg.wait_for_timeout(600)
    pg.evaluate("()=>document.querySelector('#vix-rail .seg-ctl[data-key=\"m\"] button[data-v=\"16\"]').click()"); pg.wait_for_timeout(300)
    o = tx('#vix-rail [data-obs]'); r, c = TW[('pq', 16, 16)]
    check('part 7: 16-byte codes numbers', pc1(r) in o and '72 GB' in o, o)
    # challenge: pick the answer and submit
    pg.evaluate("()=>document.querySelector('#vix-app .modes button[data-mode=\"challenge\"]').click()"); pg.wait_for_timeout(500)
    for sel in ('ix" ] button[data-v="hnsw', ):
        pass
    pg.evaluate("()=>document.querySelector('#vix-rail .seg-ctl[data-key=\"ix\"] button[data-v=\"hnsw\"]').click()"); pg.wait_for_timeout(300)
    pg.evaluate("()=>document.querySelector('#vix-rail .seg-ctl[data-key=\"ef\"] button[data-v=\"64\"]').click()"); pg.wait_for_timeout(300)
    pg.evaluate("()=>document.querySelector('#vix-rail [data-act=\"submit\"]').click()"); pg.wait_for_timeout(500)
    o = tx('#vix-ch-res'); check('challenge: HNSW M = 16, ef = 64 is met and the fewest bytes among the offered ef', 'Among the ef values this task offers, no setting meets both bars with fewer bytes' in o and '256 bytes' in o, o[:300])
    h8a, h8, h4 = TW[('hnsw', 8, 64)], TW[('hnsw', 8, 128)], TW[('hnsw', 4, 256)]
    check('challenge why: M = 8 at ef 64 and 128, M = 4 at ef 256', f'ef = 64 reaches only {pc1(h8a[0])}' in o and f'ef = 128 reaches {pc1(h8[0])} but compares {round(h8[1]):,}' in o and f'ef = 256 compares {round(h4[1]):,}' in o, o)
    check('challenge why: the ef ranges outside the menu and their bytes', 'any ef from 70 to 120 passes' in o and 'ef from 210 to 244 also passes' in o and f"only {bytes_per({'ix': 'hnsw', 'M': 8}, D)} bytes" in o and f"{bytes_per({'ix': 'hnsw', 'M': 4}, D)} bytes per vector, the fewest" in o, o)
    check('challenge why: IVF-PQ probing 32 cells with 16-byte codes', pc1(TW[('pq', 16, 32)][0]) in o and max(TW[('pq', m, 32)][0] for m in PQMS) == TW[('pq', 16, 32)][0], o)
    check('no page errors', not errs, errs[:3])
    # the home tile's numbers (teasers/vector-index.js) equal the engine's
    tz = (P / 'teasers' / 'vector-index.js').read_text()
    m = re.search(r'const TZ=(\{[^;]*\});', tz)
    if m:
        T = pg.evaluate(f"()=>({m.group(1)})")
        for k, v in T.items():
            r, c = TW[('ivf', int(k), 0)]
            check(f'tile: nprobe {k}', abs(v[0] - round(r * 100, 1)) < 1e-9 and v[1] == round(c), (v, r, c))
    else: check('tile constants found', False)
    b.close()

print(f'{n - len(fails)} / {n} passed')
sys.exit(1 if fails else 0)
