"""Numbers test for admission: an independent Python twin of the lab's fleet simulator (same seeds, same requests), compared with the
page's engine (window.ADM_ENGINE in the built page). Checks: constants (KV bytes per token, the pool of 450,907 tokens), the seeded
generator, the goodput per GPU (27) and the batch rate, every guided run the page quotes (parts 2-7: one GPU at 70-110%, the 15-minute
overload, the 14:00 peak with each set of rules, the surge with first come / priority / preemption, the nights at 3:00), the day model
(capacity, hours over capacity, batch jobs per threshold), the five scenarios, the challenge (the initial setup misses; a setup above
capacity at 14:00 with no free queue limit does not pass, and a 10-minute peak shows why; the cheapest passing setups use 5 GPUs, 120 GPU
hours, with a free queue limit of 200 or 50; the Python twin agrees on every 4- and 5-GPU option with the rate limit, and on the initial
setup), the trend words the cards use (almost twice, several times, half again, a quarter, about 8 a second), the Show-the-math
substitutions (the measured serve rate, the day's batch sum), the transfer question's numbers, and the output of the minimal implementation.
Run: python3 tests/admission_numbers.py"""
import math, pathlib, subprocess, sys, json
from collections import deque
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent

L, D, KVH, HD, FFN, VOCAB = 32, 4096, 8, 128, 14336, 128256
P_LAYER = 2*D*D + 2*D*KVH*HD + 3*D*FFN + 2*D
P_LAYERS = L*P_LAYER + D
P_HEAD = VOCAB*D
P_TOTAL = P_LAYERS + 2*P_HEAD
KV_TOK = 2*L*KVH*HD*2
W_READ = 2*(P_LAYERS + P_HEAD)
W_MEM = 2*P_TOTAL
FS = 1979e12/2*0.5
BS = 3.35e12*0.6
GiB = 2**30
BLOCK = 16
POOL_T = math.floor((80*GiB*0.9 - W_MEM - 2*GiB)/KV_TOK)
POOL_B = POOL_T // BLOCK
EPS = 1e-9
BUDGET = 2048
CAP = 1024
GOALS = {'paid': (1.0, 0.15), 'free': (5.0, 0.15)}
TARGET = {'paid': 0.9, 'free': 0.8}
LEN = {'paid': (400, 1600, 120, 320), 'free': (400, 1600, 120, 320), 'batch': (8000, 16000, 300, 800)}


def attn(a, b): return 2*L*D*(b*(b+1) - a*(a+1))
def blocks(n): return (n + BLOCK - 1)//BLOCK


def mulberry(seed):
    h = [seed & 0xffffffff]
    def imul(a, b): return ((a & 0xffffffff)*(b & 0xffffffff)) & 0xffffffff
    def f():
        h[0] = (h[0] + 0x6D2B79F5) & 0xffffffff
        x = imul(h[0] ^ (h[0] >> 15), 1 | h[0])
        x = ((x + imul(x ^ (x >> 7), 61 | x)) & 0xffffffff) ^ x
        return ((x ^ (x >> 14)) & 0xffffffff)/4294967296
    return f


def ms(x): return math.floor(x*1000 + 0.5)/1000
def uni(rnd, lo, hi): return lo + math.floor(rnd()*(hi - lo + 1))
def exp_gap(rnd, rate): return -math.log(1 - rnd())/rate


def load(phases, seed, backlog=0):
    """phases: [(duration_s, {'paid': r, 'free': r, 'script': r, 'batch': r}), ...] one after another; backlog: batch jobs waiting at t=0"""
    out = []
    for k, cls in enumerate(('paid', 'free', 'script', 'batch')):
        rnd = mulberry(seed*16 + k + 1)
        lens = LEN['free' if cls == 'script' else cls]
        t0 = 0.0
        for dur, rates in phases:
            r = rates.get(cls, 0)
            if r > 0:
                t = t0
                while True:
                    t += exp_gap(rnd, r)
                    if t >= t0 + dur: break
                    out.append({'t': ms(t), 'cls': 'free' if cls == 'script' else cls, 'key': 'script' if cls == 'script' else '',
                                'P': uni(rnd, lens[0], lens[1]), 'out': uni(rnd, lens[2], lens[3])})
            t0 += dur
    if backlog:
        rnd = mulberry(seed*16 + 9)
        lens = LEN['batch']
        for i in range(backlog):
            out.append({'t': 0.0, 'cls': 'batch', 'key': '', 'P': uni(rnd, lens[0], lens[1]), 'out': uni(rnd, lens[2], lens[3])})
    idx = sorted(range(len(out)), key=lambda i: (out[i]['t'], i))
    return [out[i] for i in idx]


def step_time(work):
    T = lg = fa = kv = 0
    for (r, kind, a, b) in work:
        if kind == 'd':
            c = r['P'] + r['gen'] - 1
            T += 1; lg += 1; fa += attn(c, c + 1); kv += c
        else:
            T += b - a; lg += 1 if b == r['Pr'] else 0; fa += attn(a, b); kv += b
    return max((2*P_LAYERS*T + 2*P_HEAD*lg + fa)/FS, (W_READ + KV_TOK*kv)/BS)


def simulate(spec, cfg):
    n = cfg['n']; qlim = cfg.get('qlim'); rl = cfg.get('rl'); prio = bool(cfg.get('prio')); pre_on = prio and cfg.get('preempt', True)
    dt_s = cfg.get('sample', 0.5); win = cfg.get('window', 0)
    R = [dict(id=i, cls=s['cls'], key=s['key'], arr=s['t'], P=s['P'], out=s['out'], Pr=s['P'], doneP=0, gen=0, toks=[], first=None,
              fin=None, rej=None, adm=None, pre=0) for i, s in enumerate(spec)]
    G = [dict(i=i, busy=False, t1=0.0, work=None, run=[], used=0) for i in range(n)]
    Q = {'paid': deque(), 'free': deque(), 'batch': deque()}
    tok = [float(rl) if rl else 0.0]; tlast = [0.0]
    stats = {'preempt': 0, 'redo': 0, 'samples': []}
    nxs = [0.0]

    def pick():
        if prio:
            for c in ('paid', 'free', 'batch'):
                if Q[c]: return Q[c][0]
            return None
        best = None
        for c in ('paid', 'free', 'batch'):
            if Q[c] and (best is None or (Q[c][0]['arr'], Q[c][0]['id']) < (best['arr'], best['id'])): best = Q[c][0]
        return best

    def start(g, now):
        budget = BUDGET
        work = []
        for r in g['run']:
            if r['first'] is not None and r['doneP'] >= r['Pr'] and r['gen'] < r['out']:
                work.append([r, 'd', 0, 0]); budget -= 1
        for r in g['run']:
            if r['doneP'] < r['Pr'] and budget > 0:
                m = min(r['Pr'] - r['doneP'], budget); work.append([r, 'p', r['doneP'], r['doneP'] + m]); budget -= m
        while budget > 0 and len(g['run']) < CAP:
            c = pick()
            if c is None: break
            need = blocks(c['P'] + c['out'])
            while g['used'] + need > POOL_B and pre_on and c['cls'] != 'batch':
                vs = [r for r in g['run'] if r['cls'] == 'batch']
                if not vs: break
                v = max(vs, key=lambda r: (r['adm'], r['id']))
                g['run'].remove(v); g['used'] -= blocks(v['P'] + v['out'])
                for w in work:
                    if w[0] is v: budget += 1 if w[1] == 'd' else w[3] - w[2]
                work = [w for w in work if w[0] is not v]
                stats['redo'] += v['doneP'] if v['doneP'] < v['Pr'] else v['P'] + v['gen'] - 1
                v['Pr'] = v['P'] + v['gen']; v['doneP'] = 0; v['pre'] += 1; stats['preempt'] += 1
                Q['batch'].appendleft(v)
            if g['used'] + need > POOL_B: break
            Q[c['cls']].popleft()
            g['run'].append(c); g['used'] += need; c['adm'] = now
            m = min(c['Pr'] - c['doneP'], budget); work.append([c, 'p', c['doneP'], c['doneP'] + m]); budget -= m
        if not work:
            g['busy'] = False; g['work'] = None; return
        g['busy'] = True; g['work'] = work; g['t1'] = now + step_time(work)

    def finish(g):
        t = g['t1']
        for (r, kind, a, b) in g['work']:
            if kind == 'p':
                r['doneP'] = b
                if b == r['Pr']:
                    if r['first'] is None: r['first'] = t; r['gen'] = 1
                    else: r['gen'] += 1
                    r['toks'].append(t)
            else:
                r['gen'] += 1; r['toks'].append(t)
        for r in list(g['run']):
            if r['first'] is not None and r['doneP'] >= r['Pr'] and r['gen'] >= r['out']:
                r['fin'] = t; g['run'].remove(r); g['used'] -= blocks(r['P'] + r['out'])
        g['work'] = None; g['busy'] = False

    def sample(upto):
        while nxs[0] <= upto and nxs[0] <= win + EPS:
            stats['samples'].append((nxs[0], len(Q['paid']), len(Q['free']), len(Q['batch'])))
            nxs[0] = round(nxs[0] + dt_s, 6)

    nxt = 0
    while True:
        ta = R[nxt]['arr'] if nxt < len(R) else math.inf
        tg = math.inf; gb = None
        for g in G:
            if g['busy'] and g['t1'] < tg: tg = g['t1']; gb = g
        if ta == math.inf and tg == math.inf: break
        if ta <= tg:
            r = R[nxt]; nxt += 1; t = r['arr']; sample(t - EPS)
            ok = True
            if r['key'] == 'script' and rl:
                tok[0] = min(float(rl), tok[0] + (t - tlast[0])*rl/60.0); tlast[0] = t
                if tok[0] >= 1 - 1e-12: tok[0] -= 1
                else: r['rej'] = 'rate'; ok = False
            if ok and r['cls'] == 'free' and qlim is not None and len(Q['free']) >= qlim:
                r['rej'] = 'queue'; ok = False
            if ok:
                Q[r['cls']].append(r)
                for g in G:
                    if not g['busy']: start(g, t)
        else:
            t = tg; sample(t - EPS); finish(gb); start(gb, t)
            for h in G:
                if not h['busy']: start(h, t)
    sample(win)
    return {'R': R, **stats}


def gap(r): return max((r['toks'][i] - r['toks'][i-1] for i in range(1, len(r['toks']))), default=0.0)


def pct(xs, p):
    s = sorted(xs)
    if not s: return None
    k = max(1, math.ceil(p/100*len(s) - 1e-9)); return s[k-1]


def metrics(run):
    out = {}
    for cls in ('paid', 'free'):
        rs = [r for r in run['R'] if r['cls'] == cls]
        if not rs: out[cls] = None; continue
        lim = [r for r in rs if r['rej'] != 'rate']
        served = [r for r in lim if r['rej'] is None]
        g = GOALS[cls]
        ok = sum(1 for r in served if r['first'] - r['arr'] <= g[0] + EPS and gap(r) <= g[1] + EPS)
        out[cls] = dict(n=len(rs), nlim=len(lim), ok=ok, frac=ok/len(lim), p90=pct([r['first'] - r['arr'] for r in served], 90) if served else None,
                        rejq=sum(1 for r in rs if r['rej'] == 'queue'), rejr=len(rs) - len(lim),
                        gapmax=max((gap(r) for r in served), default=0.0))
    b = [r for r in run['R'] if r['cls'] == 'batch']
    out['batch'] = dict(n=len(b), pre=run['preempt'], redo=run['redo'])
    return out


# ---------------- the day model, the checks (as the lab) ----------------
PROF = [0.30, 0.22, 0.18, 0.15, 0.15, 0.18, 0.25, 0.40, 0.60, 0.78, 0.90, 0.97, 0.95, 0.97, 1.00, 0.98, 0.93, 0.85, 0.75, 0.70, 0.68, 0.62, 0.50, 0.40]
PEAK_H, WIN, NWIN = 14, 60, 120
BASE_T, CH_T = {'paid': 40, 'free': 85, 'script': 25}, {'paid': 50, 'free': 100, 'script': 30}
BASE_JOBS, CH_JOBS = 120000, 150000
RUNS = {}


def run_of(key, mk):
    if key not in RUNS:
        run = mk(); run['m'] = metrics(run); RUNS[key] = run
    return RUNS[key]


def goodput():
    def ok(r): return run_of(('gp', r), lambda: simulate(load([(WIN, {'paid': r})], 1), {'n': 1, 'window': WIN}))['m']['paid']['frac'] >= TARGET['paid'] - 1e-9
    lo, hi = 1, 40
    while hi - lo > 1:
        mid = (lo + hi)//2
        if ok(mid): lo = mid
        else: hi = mid
    return lo, hi


def batch_rate():
    run = run_of('bj', lambda: simulate(load([(1, {})], 2, 600), {'n': 1}))
    return 600/max(r['fin'] for r in run['R'])


def chat_rate(T, h, rl): return PROF[h]*(T['paid'] + T['free']) + (min(T['script'], rl/60) if rl else T['script'])


def day(T, s):
    g = GP[0]; b = BJ; cap = s['n']*g; hours = []; jobs = 0; over = 0
    for h in range(24):
        c = chat_rate(T, h, s['rl']); u = c/cap; sh = max(0.0, s['th']/100 - u); j = sh*s['n']*b*3600
        jobs += j; over += 1 if u > 1 + 1e-12 else 0; hours.append(dict(h=h, c=c, u=u, sh=sh, j=j))
    return dict(hours=hours, jobs=jobs, over=over, cap=cap)


def quiet_h(): return min(range(24), key=lambda h: (PROF[h], h))


def peak(T, s, seed=7):
    return run_of(('pk', T['paid'], T['free'], T['script'], s['n'], s['q'], s['rl'], s['prio'], seed),
                  lambda: simulate(load([(WIN, dict(T))], seed), {'n': s['n'], 'qlim': s['q'] or None, 'rl': s['rl'], 'prio': bool(s['prio']), 'window': WIN}))


def night(T, s, seed=8):
    h = quiet_h(); d = day(T, s); x = d['hours'][h]
    return run_of(('nt', T['paid'], T['free'], T['script'], s['n'], s['q'], s['rl'], s['prio'], s['th'], seed),
                  lambda: simulate(load([(NWIN, {'paid': T['paid']*PROF[h], 'free': T['free']*PROF[h], 'script': T['script'], 'batch': x['sh']*s['n']*BJ})], seed),
                                   {'n': s['n'], 'qlim': s['q'] or None, 'rl': s['rl'], 'prio': bool(s['prio']), 'window': NWIN, 'sample': 1}))


def passes(m): return m['paid']['frac'] >= TARGET['paid'] - 1e-9 and m['free']['frac'] >= TARGET['free'] - 1e-9


def over_no_limit(T, s): return not s['q'] and chat_rate(T, PEAK_H, s['rl']) > s['n']*GP[0] + 1e-9


def verdict(T, s, jobs):
    pk = peak(T, s)['m']; d = day(T, s); batch_ok = d['jobs'] >= jobs - 1e-6; chat_ok = passes(pk) and not over_no_limit(T, s)
    nt = night(T, s)['m'] if chat_ok and batch_ok else None
    return dict(ok=chat_ok and batch_ok and nt is not None and passes(nt), chat=chat_ok, batch=batch_ok, night=None if nt is None else passes(nt), gpuh=s['n']*24)


# ---------------- the page's engine ----------------
subprocess.run([sys.executable, str(P / 'build.py'), 'out/admission.html', '--only', 'admission'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'admission').mkdir(parents=True, exist_ok=True)
W = P / 'out' / 'admission' / 'numbers-wrapped.html'
W.write_text('<!doctype html><html><head><meta charset="utf-8"></head><body>' + (P / 'out' / 'admission.html').read_text() + '</body></html>')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok: fails.append(name); print('FAIL', name, '::', str(info)[:300])


def close(a, b, tol=1e-9): return a is None and b is None or (a is not None and b is not None and abs(a - b) <= tol*max(1.0, abs(b)))


JS = """()=>{const E=window.ADM_ENGINE, H=window.__adm, out={};
 const mm=m=>({paid:m.paid&&{n:m.paid.n,ok:m.paid.ok,frac:m.paid.frac,p90:m.paid.p90,rejq:m.paid.rejq,rejr:m.paid.rejr,gapmax:m.paid.gapmax},
   free:m.free&&{n:m.free.n,ok:m.free.ok,frac:m.free.frac,p90:m.free.p90,rejq:m.free.rejq,rejr:m.free.rejr,gapmax:m.free.gapmax},batch:{n:m.batch.n,pre:m.batch.pre,redo:m.batch.redo}});
 out.C={KV_TOK:E.C.KV_TOK,POOL_T:E.C.POOL_T,W_MEM:E.C.W_MEM}; const r=E.mulberry(17); out.rnd=[r(),r(),r()];
 out.gp=E.goodput(); out.bj=E.batchRate().per; out.quiet=E.quietH();
 const cf=s=>({n:s.n,q:s.q,rl:s.rl,prio:!!s.prio,th:s.th/100});
 const one=f=>{const x=Math.round(out.gp.g*f*10)/10; return E.runOf('one|'+x,()=>E.simulate(E.load([[E.WIN,{paid:x}]],1),{n:1,window:E.WIN}));};
 out.one={}; for(const f of H.LOADS) out.one[f]=mm(one(f).m);
 const hi=Math.round(out.gp.g*1.2*10)/10, lo=Math.round(out.gp.g*0.7*10)/10;
 const L=E.runOf('long|'+hi,()=>E.simulate(E.load([[600,{paid:hi}],[300,{paid:lo}]],3),{n:1,window:900,sample:5}));
 out.long={m:mm(L.m),q10:(L.samples.find(x=>Math.abs(x[0]-600)<1e-6)||[0,0])[1],q15:(L.samples.find(x=>Math.abs(x[0]-900)<1e-6)||[0,0])[1],hi,lo};
 out.peak={}; const B=H.BASE_T, setups={n4:{n:4,q:0,rl:0,prio:0,th:60},n6:{n:6,q:0,rl:0,prio:0,th:60},n7:{n:7,q:0,rl:0,prio:0,th:60},q:H.P5.q,rl:H.P5.rl,pf:H.P5.pf,lean:{n:4,q:200,rl:60,prio:1,th:100},noprio:{n:4,q:200,rl:60,prio:0,th:100},spare:{n:7,q:0,rl:60,prio:1,th:80}};
 for(const k in setups) out.peak[k]=mm(E.peakRun(B,cf(setups[k])).m);
 out.day={}; for(const k in setups){const d=E.day(B,cf(setups[k])); out.day[k]={jobs:d.jobs,over:d.over,cap:d.cap,u3:d.hours[out.quiet].u};}
 for(const t of [60,80]){const d=E.day(B,cf({n:4,q:200,rl:60,prio:1,th:t})); out.day['t'+t]={jobs:d.jobs,over:d.over,cap:d.cap};}
 out.night={}; for(const t of [60,80,100]) out.night['t'+t]=mm(E.nightRun(B,cf({n:4,q:200,rl:60,prio:1,th:t})).m); out.night.np=mm(E.nightRun(B,cf({n:4,q:200,rl:60,prio:0,th:100})).m);
 const S=H.SURGE; out.surge={};
 for(const k of ['fcfs','prio','pre']) out.surge[k]=mm(E.runOf('surge|'+k,()=>E.simulate(E.load([[S.quiet,{}],[E.WIN-S.quiet,{paid:B.paid,free:B.free,script:B.script}]],11,S.backlog),{n:S.n,qlim:S.q,rl:S.rl,prio:k!=='fcfs',preempt:k==='pre',window:E.WIN})).m);
 out.ch=H.chOpts(); out.exits=H.exits(); out.code=H.code(); out.presets=H.PRESETS; out.chStart=H.CH_START;
 out.serve=E.serveRate(L,600); out.avg=(()=>{let a=0; for(let h=0;h<24;h++) a+=E.chatRate(B,h,0); return a/24;})();
 const C=H.CH_T; out.over={}; for(const o of out.ch) out.over[[o.n,o.q,o.rl,o.prio,o.th].join('_')]=E.overNoLimit(C,cf(o));
 out.long10={}; for(const q of [0,200]){const r=E.simulate(E.load([[600,{paid:C.paid,free:C.free,script:C.script}]],7),{n:5,qlim:q||null,rl:60,prio:true,window:600}); out.long10[q]=mm(E.metrics(r));}
 out.scen={}; for(const p of H.PRESETS){const v=E.verdict(B,cf(p.s),H.BASE_JOBS); out.scen[p.id]={ok:v.ok,over:v.over};} return out;}"""

with sync_playwright() as pw:
    b = pw.chromium.launch(); pg = b.new_page(); pg.goto('file://' + str(W)); pg.wait_for_timeout(300)
    J = pg.evaluate(JS); b.close()

GP = goodput(); BJ = batch_rate()
check('constants: 131,072 bytes of KV per token, a pool of 450,907 tokens', J['C']['KV_TOK'] == KV_TOK == 131072 and J['C']['POOL_T'] == POOL_T == 450907, (J['C'], KV_TOK, POOL_T))
r = mulberry(17); check('the seeded generator matches', all(close(a, r()) for a in J['rnd']), J['rnd'])
check('goodput per GPU: 27 requests per second (28 misses)', (J['gp']['g'], J['gp']['next']) == GP == (27, 28), (J['gp'], GP))
check('batch jobs per GPU-second match', close(J['bj'], BJ, 1e-9), (J['bj'], BJ))
check('the quietest hour is 3:00', J['quiet'] == quiet_h() == 3, J['quiet'])


def same(name, jm, pm):
    for cls in ('paid', 'free'):
        a, p_ = jm[cls], pm[cls]
        if a is None or p_ is None:
            check(f'{name}: {cls} present on both sides', a is None and p_ is None, (a, p_)); continue
        check(f'{name}: {cls} counts', (a['n'], a['ok'], a['rejq'], a['rejr']) == (p_['n'], p_['ok'], p_['rejq'], p_['rejr']), (a, p_))
        check(f'{name}: {cls} p90 and longest gap', close(a['p90'], p_['p90'], 1e-7) and close(a['gapmax'], p_['gapmax'], 1e-7), (a['p90'], p_['p90'], a['gapmax'], p_['gapmax']))
    check(f'{name}: batch jobs, preemptions, recomputed tokens', (jm['batch']['n'], jm['batch']['pre'], jm['batch']['redo']) == (pm['batch']['n'], pm['batch']['pre'], pm['batch']['redo']), (jm['batch'], pm['batch']))


for f, jm in J['one'].items():
    x = round(GP[0]*float(f)*10)/10
    same(f'part 3: one GPU at {f}', jm, run_of(('one', x), lambda: simulate(load([(WIN, {'paid': x})], 1), {'n': 1, 'window': WIN}))['m'])
o = J['one']
check('part 3: 70% and 90% meet the goals, 110% misses', o['0.7']['paid']['frac'] >= 0.9 and o['0.9']['paid']['frac'] >= 0.9 and o['1.1']['paid']['frac'] < 0.9, {k: v['paid']['frac'] for k, v in o.items()})
lr = J['long']
LR = run_of('long', lambda: simulate(load([(600, {'paid': lr['hi']}), (300, {'paid': lr['lo']})], 3), {'n': 1, 'window': 900, 'sample': 5}))
same('part 4: 20% over for 10 minutes', lr['m'], LR['m'])
q10 = next(s[1] for s in LR['samples'] if abs(s[0] - 600) < 1e-6)
check('part 4: the queue at 10 minutes matches and is in the hundreds; it is not empty at 15 minutes', q10 == lr['q10'] and lr['q10'] > 300 and lr['q15'] > 0, (q10, lr['q10'], lr['q15']))
check('part 4: almost every request misses (under 20% meet the goals)', lr['m']['paid']['frac'] < 0.2, lr['m']['paid']['frac'])
SET = {'n4': (4, 0, 0, 0, 60), 'n6': (6, 0, 0, 0, 60), 'n7': (7, 0, 0, 0, 60), 'q': (4, 200, 0, 0, 60), 'rl': (4, 200, 60, 0, 60), 'pf': (4, 200, 60, 1, 60), 'lean': (4, 200, 60, 1, 100), 'noprio': (4, 200, 60, 0, 100), 'spare': (7, 0, 60, 1, 80)}
S_ = {k: dict(zip(('n', 'q', 'rl', 'prio', 'th'), v)) for k, v in SET.items()}
for k, s in S_.items():
    same(f'peak at 14:00, {k}', J['peak'][k], peak(BASE_T, s)['m'])
    d = day(BASE_T, s)
    check(f'day model, {k}: capacity, hours over capacity, batch jobs', close(J['day'][k]['cap'], d['cap']) and J['day'][k]['over'] == d['over'] and close(J['day'][k]['jobs'], d['jobs'], 1e-9), (J['day'][k], d['jobs'], d['over']))
pk_ = J['peak']
check('part 2: 4 GPUs miss at the peak, 6 and 7 meet both targets', not (pk_['n4']['paid']['frac'] >= 0.9) and all(pk_[k]['paid']['frac'] >= 0.9 and pk_[k]['free']['frac'] >= 0.8 for k in ('n6', 'n7')), {k: (v['paid']['frac'], v['free']['frac']) for k, v in pk_.items()})
check('part 2: 4 GPUs leave some hours over capacity, 6 GPUs none', J['day']['n4']['over'] > 0 and J['day']['n6']['over'] == 0, (J['day']['n4']['over'], J['day']['n6']['over']))
check('part 5: the queue limit helps free requests, the rate limit more; paid still miss without priority', pk_['q']['free']['frac'] > pk_['n4']['free']['frac'] and pk_['rl']['free']['frac'] >= pk_['q']['free']['frac'] and pk_['rl']['paid']['frac'] < 0.9 and pk_['rl']['free']['rejr'] > 0, {k: (pk_[k]['paid']['frac'], pk_[k]['free']['frac']) for k in ('n4', 'q', 'rl')})
check('part 7 / lean: with every rule, 4 GPUs meet both targets at 14:00', pk_['lean']['paid']['frac'] >= 0.9 and pk_['lean']['free']['frac'] >= 0.8, pk_['lean'])
for t in (60, 80):
    d = day(BASE_T, dict(n=4, q=200, rl=60, prio=1, th=t))
    check(f'part 7: θ = {t}% jobs match', close(J['day'][f't{t}']['jobs'], d['jobs'], 1e-9), (J['day'][f't{t}'], d['jobs']))
check('part 7: θ = 60% and 80% fall short of 120,000 jobs, 100% passes', J['day']['t60']['jobs'] < BASE_JOBS and J['day']['t80']['jobs'] < BASE_JOBS and J['day']['lean']['jobs'] >= BASE_JOBS, (J['day']['t60']['jobs'], J['day']['t80']['jobs'], J['day']['lean']['jobs']))
for t in (60, 80, 100):
    same(f'part 7: the night at 3:00, θ = {t}%', J['night'][f't{t}'], night(BASE_T, dict(n=4, q=200, rl=60, prio=1, th=t))['m'])
same('part 7: the night at 3:00, θ = 100%, first come', J['night']['np'], night(BASE_T, dict(n=4, q=200, rl=60, prio=0, th=100))['m'])
nt = J['night']
check('part 7: at θ = 100% batch jobs are preempted at night and paid chats still meet the goal; without priority they miss', nt['t100']['batch']['pre'] > 0 and nt['t100']['paid']['frac'] >= 0.9 and nt['np']['paid']['frac'] < 0.9, (nt['t100'], nt['np']['paid']))
for k in ('fcfs', 'prio', 'pre'):
    sp = {'n': 4, 'q': 200, 'rl': 60}
    same(f'part 6: the surge, {k}', J['surge'][k], run_of(('surge', k), lambda: simulate(load([(20, {}), (WIN - 20, dict(BASE_T))], 11, 600), {'n': 4, 'qlim': 200, 'rl': 60, 'prio': k != 'fcfs', 'preempt': k == 'pre', 'window': WIN}))['m'])
su = J['surge']
check('part 6: first come misses both; priority saves paid only; preemption saves both', su['fcfs']['paid']['frac'] < 0.9 and su['prio']['paid']['frac'] >= 0.9 and su['prio']['free']['frac'] < 0.8 and su['pre']['paid']['frac'] >= 0.9 and su['pre']['free']['frac'] >= 0.8 and su['pre']['batch']['pre'] > 0, {k: (v['paid']['frac'], v['free']['frac'], v['batch']['pre']) for k, v in su.items()})
# the challenge
ch = J['ch']; feas = [o_ for o_ in ch if o_['ok']]; best = min(o_['gpuh'] for o_ in feas)
cs = J['chStart']; c0 = next(o_ for o_ in ch if all(o_[k] == cs[k] for k in ('n', 'q', 'rl', 'prio', 'th')))
check('challenge: 144 options; the initial setup misses', len(ch) == 144 and not c0['ok'], (len(ch), c0))
check('challenge: the cheapest passing setups use 5 GPUs (120 GPU hours): rate limit, paid first, θ = 100%, a free queue limit of 50 or 200', best == 120 and sorted((o_['q'], o_['rl'], o_['prio'], o_['th']) for o_ in feas if o_['gpuh'] == 120) == [(50, 60, 1, 100), (200, 60, 1, 100)], [o_ for o_ in feas if o_['gpuh'] == 120])
ov = sorted(k for k, x in J['over'].items() if x)
check('challenge: the over-capacity rule flags exactly the setups with no free queue limit whose 14:00 traffic is above n x 27', ov == sorted(f"{o_['n']}_{o_['q']}_{o_['rl']}_{o_['prio']}_{o_['th']}" for o_ in ch if over_no_limit(CH_T, o_)) and all(not o_['ok'] for o_ in ch if J['over'][f"{o_['n']}_{o_['q']}_{o_['rl']}_{o_['prio']}_{o_['th']}"]), ov[:6])
l10 = J['long10']
check('challenge, a 10-minute peak (B1): 5 GPUs with the rate limit and paid first but no free queue limit drop below 80% of free requests; with a limit of 200 they stay above', l10['0']['free']['frac'] < 0.8 and l10['200']['free']['frac'] >= 0.8 and l10['0']['paid']['frac'] >= 0.9, (l10['0']['free']['frac'], l10['200']['free']['frac']))
r10 = simulate(load([(600, dict(CH_T))], 7), {'n': 5, 'qlim': None, 'rl': 60, 'prio': True, 'window': 600})
check('challenge, a 10-minute peak: the twin agrees on the run with no queue limit', metrics(r10)['free']['ok'] == l10['0']['free']['ok'] and metrics(r10)['free']['n'] == l10['0']['free']['n'], (metrics(r10)['free'], l10['0']['free']))
check('scenarios: every scenario but the two sized-for-average or no-priority ones passes; none is flagged over capacity with no limit except the average one', {k: v['ok'] for k, v in J['scen'].items()} == {'avg': False, 'peak': False, 'lean': True, 'noprio': False, 'spare': True} and [k for k, v in J['scen'].items() if v['over']] == ['avg'], J['scen'])
sub = [o_ for o_ in ch if (o_['n'] in (4, 5) and o_['rl'] == 60) or o_ is c0]
for o_ in sub:
    v = verdict(CH_T, o_, CH_JOBS)
    check(f'challenge twin: {o_["n"]}_{o_["q"]}_{o_["rl"]}_{o_["prio"]}_{o_["th"]}', v['ok'] == o_['ok'], (v, o_))
# the transfer question: 200 / 27
ex = J['exits']; check('exits: the right answers are written first', ex['explainA'] == [0, 0] and ex['transferA'] == [0, 0], ex)
check('transfer: 200 requests per second need ceil(200 / 27) = 8 GPUs', math.ceil(200/GP[0]) == 8, GP)
av2 = J['avg']*200/150
check('transfer, option B: sized by the average the peak of 200 asks for 5 GPUs (about 133 / 27 = 4.9)', round(av2) == 133 and math.ceil(av2/GP[0]) == 5 and f'{round(av2)/GP[0]:.1f}' == '4.9', av2)
# trend words on the cards
o9, o7, o10 = J['one']['0.9']['paid']['p90'], J['one']['0.7']['paid']['p90'], J['one']['1']['paid']['p90']
check('part 3: at 90% the first token takes almost twice as long as at 70%; at 100% several times as long', 1.6 <= o9/o7 <= 2.1 and o10/o7 >= 3, (o9/o7, o10/o7))
check('part 4: almost no request meets its goal (under 5%), the queue is about 2,800 at 10 minutes', lr['m']['paid']['frac'] < 0.05 and round(lr['q10']/100)*100 == 2800, (lr['m']['paid']['frac'], lr['q10']))
check('part 4: after traffic falls back the queue drains by about 8 a second', round((lr['q10'] - lr['q15'])/300) == 8, (lr['q10'] - lr['q15'])/300)
check('part 5: with the rate limit, about a quarter as many free requests are turned away for a full queue; most turned away are the script', 0.2 <= pk_['rl']['free']['rejq']/pk_['q']['free']['rejq'] < 0.35 and pk_['rl']['free']['rejr'] > pk_['rl']['free']['rejq'], (pk_['rl']['free'], pk_['q']['free']['rejq']))
check('part 5: paid first meets the paid goal, and free requests stay above 80% while some get 429', pk_['pf']['paid']['frac'] >= 1 and pk_['pf']['free']['frac'] >= 0.8 and pk_['pf']['free']['rejq'] > 0, pk_['pf'])
check('part 8: the peak is half as high again as the day average', 1.45 <= 150/J['avg'] <= 1.55, J['avg'])
check('part 6: first come, first served: not one chat meets its goal', su['fcfs']['paid']['ok'] + su['fcfs']['free']['ok'] == 0, su['fcfs'])
check('part 7: θ = 60% does less than half of the jobs, θ = 80% more than half', J['day']['t60']['jobs'] < BASE_JOBS/2 <= J['day']['t80']['jobs'], (J['day']['t60']['jobs'], J['day']['t80']['jobs']))
# Show the math: the substituted numbers compute to the shown results
served = sum(1 for r in LR['R'] if r['first'] is not None and r['first'] <= 600 + EPS)/600
mu1 = round(served*10)/10
check('math: the measured serve rate matches (about 27.7 a second) and (32.4 - 27.7) x 600 is close to the queue on the stage', close(J['serve'], served, 1e-12) and mu1 == 27.7 and abs(round((lr['hi'] - mu1)*600) - lr['q10']) <= 0.01*lr['q10'], (J['serve'], served, lr['q10']))
dl = day(BASE_T, dict(n=4, q=200, rl=60, prio=1, th=100)); sh = round(sum(x['sh'] for x in dl['hours'])*1000)/1000
check('math: 7.919 x 4 x 1.2668 x 3600 rounds to the 144,000 on the card', f'{sh:.3f}' == '7.919' and round(sh*4*round(BJ*1e4)/1e4*3600/1000) == round(dl['jobs']/1000) == 144, (sh, dl['jobs']))
# the minimal implementation prints what the page says
code = J['code']; ns = {}
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf): exec(compile(code, 'gateway', 'exec'), ns)
want = ['queued', 'queued', '429 queue full, Retry-After: 30', '429 queue full, Retry-After: 30', 'queued', '9']
check('minimal implementation: its output matches the comment on the page', buf.getvalue().split('\n')[:-1] == want, buf.getvalue())
check('minimal implementation: no line longer than 74 characters', max(len(x) for x in code.split('\n')) <= 74, max(len(x) for x in code.split('\n')))
print(f'{n - len(fails)}/{n} checks passed')
print('FAILED:', fails if fails else 'none')
sys.exit(1 if fails else 0)
