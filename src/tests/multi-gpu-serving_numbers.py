"""Numbers test for multi-gpu-serving: an independent Python twin of the lab's formulas, compared with the page (window.__mgs and the
visible text). Checks: the model shapes (parameters, weight bytes, KV bytes per token), the all-reduce formula inside and across nodes,
every setup the settings allow (4 models x 2 GPU counts x TP x pipeline stages x users x context), the MoE routing counts, every number
the guided parts quote (en and zh), the metric cards, the scenarios, every challenge option and its best answer, the exits' numbers,
the minimal implementation's output (run here) and the interview drill. Run: python3 tests/multi-gpu-serving_numbers.py"""
import io, contextlib, itertools, math, pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
(P / 'out' / 'multi-gpu-serving').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/multi-gpu-serving.html', '--only', 'multi-gpu-serving'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'multi-gpu-serving' / 'num-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'multi-gpu-serving.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'multi-gpu-serving' / 'num-wrapped.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


# ---------------- the twin ----------------
GiB = 2**30
HBM = 80*GiB
USABLE = HBM*0.9 - 2*GiB          # vLLM's 90% minus about 2 GiB for the runtime
BW = 3.35e12*0.6                   # bytes/s a GPU reads (60% of peak)
NVLINK, IBAND = 450e9, 50e9        # one way, per GPU
A_IN, A_X = 18e-6, 60e-6           # fixed cost of one all-reduce: in a node, across nodes
LLAMA = {'70b': (80, 8192, 28672, 2, 989.5e12), '70b8': (80, 8192, 28672, 1, 1979e12),
         '405b8': (126, 16384, 53248, 1, 1979e12), '405b': (126, 16384, 53248, 2, 989.5e12)}
KVH, HDIM, VOCAB = 8, 128, 128256


def params(m):
    L, d, ffn, wb, peak = LLAMA[m]
    per_layer = (d*d + d*d) + 2*(d*KVH*HDIM) + 3*d*ffn + 2*d   # q,o + k,v + gate/up/down + 2 norms
    body = L*per_layer + d                                       # + final norm
    head = VOCAB*d                                               # embedding table = LM head size
    return {'L': L, 'd': d, 'wb': wb, 'peak': peak, 'attn': 2*d*d + 2*d*KVH*HDIM, 'mlp': 3*d*ffn, 'body': body, 'head': head,
            'total': body + 2*head, 'kv': 2*L*KVH*HDIM*2}


def allreduce(nbytes, tp):
    if tp == 1: return 0.0, False
    if tp <= 8: return A_IN + (2*(tp - 1)/tp)*nbytes/NVLINK, False
    k = tp//8
    return A_X + (2*7/8)*nbytes/NVLINK + (2*(k - 1)/k)*(nbytes/8)/IBAND, True


def twin(m, g, tp, pp, u, ctx):
    p = params(m); L, d = p['L'], p['d']
    gpr = tp*pp; R = max(1, g//gpr); users = -(-u//R)
    w = p['wb']*p['total']/gpr
    kv_gpu_tok = p['kv']/(min(tp, KVH)*pp)
    room = USABLE - w; need = users*ctx*kv_gpu_tok
    fits = room > 0 and need <= room
    mb = min(pp, users); b = -(-users//mb)
    flops = (2*p['body']*b + 2*p['head']*b + b*4*L*d*(ctx + 1))/(pp*tp)
    rd = p['wb']*(p['body'] + p['head'])/gpr + b*ctx*kv_gpu_tok
    t_rc = max(flops/(p['peak']*0.5), rd/BW)
    n_ar = 2*L//pp if tp > 1 else 0
    S = b*d*2; t_ar, cross = allreduce(S, tp)
    stage = t_rc + n_ar*t_ar
    sends = []
    if pp > 1:
        for i in range(pp):
            a, z = (i*tp)//8, (((i + 1) % pp)*tp)//8
            nb = S if i < pp - 1 else b*4
            sends.append((a != z, nb, (A_X + nb/IBAND) if a != z else (A_IN + nb/NVLINK)))
    t_send = sum(x[2] for x in sends)
    tpot = pp*stage + t_send
    comm = pp*n_ar*t_ar + t_send
    # first token of a new prompt alone on one replica
    fp = (2*p['body']*ctx + 2*p['head'] + 2*L*d*ctx*(ctx + 1))/tp
    bp = p['wb']*(p['body'] + p['head'])/tp + ctx*p['kv']/min(tp, KVH)
    sp = ctx*d*2; t_arp = allreduce(sp, tp)[0]
    ttft = max(fp/(p['peak']*0.5), bp/BW) + (2*L*t_arp if tp > 1 else 0)
    for x in sends[:-1]: ttft += (A_X + sp/IBAND) if x[0] else (A_IN + sp/NVLINK)
    lt = t_rc/(L/pp)
    ab = p['wb']*p['attn']/tp + b*ctx*(p['kv']/L)/min(tp, KVH); mbb = p['wb']*p['mlp']/tp
    return {'R': R, 'u': users, 'w': w, 'room': room, 'need': need, 'fits': fits, 'b': b, 'tcm': t_rc, 'S': S, 'ar': t_ar, 'cross': cross,
            'nAR': n_ar, 'tSend': t_send, 'tpot': tpot, 'tComm': comm, 'share': comm/tpot, 'tps': (u/tpot/g) if fits else None, 'ttft': ttft,
            'attn': lt*ab/(ab + mbb), 'mlp': lt*mbb/(ab + mbb)}


def mulberry(seed):
    h = [seed & 0xffffffff]
    im = lambda a, b: ((a & 0xffffffff)*(b & 0xffffffff)) & 0xffffffff
    def f():
        h[0] = (h[0] + 0x6D2B79F5) & 0xffffffff; x = h[0]
        x = im(x ^ (x >> 15), 1 | x)
        x = ((x + im(x ^ (x >> 7), 61 | x)) & 0xffffffff) ^ x
        return ((x ^ (x >> 14)) & 0xffffffff)/4294967296
    return f


def route(ntok, seed=11):
    rnd = mulberry(seed); per = [0]*8
    for _ in range(ntok):
        ch = []
        while len(ch) < 8:
            e = math.floor(rnd()*256)
            if e not in ch: ch.append(e)
        for e in ch: per[e//32] += 1
    return per


# ---------------- formats, as the page writes them ----------------
def trim0(s): return s.rstrip('0').rstrip('.') if '.' in s else s
def num(v, d): return trim0(f'{v:.{d}f}')
def fx(v, d): return f'{v:,.{d}f}'          # milliseconds and GiB keep one decimal, 23.0 beside 14.5 (R6.6)
def ms(t, z=False): return f'{fx(t*1e3, 1)} 毫秒' if z else f'{fx(t*1e3, 1)} ms'
def us(t, z=False): return f'{round(t*1e6)} 微秒' if z else f'{round(t*1e6)} µs'
def gi(b): return f'{fx(b/GiB, 1)} GiB'
def pc(x): return f'{round(x*100)}%'
def it(x): return f'{round(x):,}'
def kib(b): return f'{num(b/2**20, 0 if b >= 10*2**20 else 1)} MiB' if b >= 2**20 else f'{round(b/1024):,} KiB'
def close(a, b, rel=1e-9): return a is None and b is None or (a is not None and b is not None and abs(a - b) <= rel*max(1e-12, abs(a), abs(b)))


# ---------------- 1. shapes and the all-reduce formula, by hand ----------------
p70 = params('70b')
check('70B parameters', p70['total'] == 70553706496, p70['total'])
check('70B weights 131.4 GiB in BF16', gi(2*p70['total']) == '131.4 GiB', gi(2*p70['total']))
check('70B KV per token 320 KiB', p70['kv'] == 327680, p70['kv'])
check('405B weights 756 GiB in BF16', abs(2*params('405b')['total']/GiB - 756.0) < 0.5, 2*params('405b')['total']/GiB)
check('all-reduce of 512 KiB on TP 8 = 18 + 2.04 µs', abs(allreduce(524288, 8)[0] - (18e-6 + 1.75*524288/450e9)) < 1e-15)
check('TP 8 / TP 16 TPOT = 14.5 / 17.1 ms', (ms(twin('70b', 8, 8, 1, 32, 4096)['tpot']), ms(twin('70b', 16, 16, 1, 32, 4096)['tpot'])) == ('14.5 ms', '17.1 ms'))

with sync_playwright() as pw:
    b = pw.chromium.launch()
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(250)

    def lang(l):
        pg.click(f'.langsw button[data-lang="{l}"]'); pg.wait_for_timeout(120)
    lang('en')
    pg.evaluate("()=>{location.hash='multi-gpu-serving'}"); pg.wait_for_timeout(500)

    # ---------------- 2. every setup ----------------
    grid = [dict(m=m, g=g, tp=tp, pp=pp, u=u, ctx=c) for m, g, tp, pp, u, c in itertools.product(LLAMA, (8, 16), (1, 2, 4, 8, 16), (1, 2, 4), (8, 32, 128, 256), (1024, 4096, 16384)) if tp*pp <= g]
    got = pg.evaluate("""(G)=>G.map(c=>{const r=window.__mgs.engine.evalSetup(c); return {R:r.R,u:r.u,w:r.w,room:r.room,need:r.need,fits:r.fits,b:r.b,tcm:r.tcm,S:r.S,ar:r.ar.t,cross:r.ar.cross,nAR:r.nAR,tSend:r.tSend,
      tpot:r.tpot,tComm:r.tComm,share:r.share,tps:r.tps,ttft:r.ttft,attn:r.layer.attn,mlp:r.layer.mlp};})""", grid)
    bad = []
    for c, r in zip(grid, got):
        t = twin(**c)
        for k, v in t.items():
            if isinstance(v, bool) or isinstance(v, int) and not isinstance(r[k], float):
                if v != r[k]: bad.append((c, k, v, r[k]))
            elif not close(v, r[k]): bad.append((c, k, v, r[k]))
    check(f'every setup ({len(grid)}) matches the twin', not bad, bad[:3])

    # ---------------- 3. MoE routing ----------------
    for ntok in (32, 256, 2048):
        check(f'route({ntok}) matches', pg.evaluate(f"()=>window.__mgs.engine.route({ntok})") == route(ntok), route(ntok))
        pk = pg.evaluate(f"()=>{{const per=new Array(8).fill(0); window.__mgs.engine.picks({ntok}).forEach(ch=>ch.forEach(e=>per[Math.floor(e/32)]++)); return per;}}")
        check(f'picks({ntok}), the draws the stage animates, give the same counts as route({ntok})', pk == route(ntok), pk)
    r32 = route(32)
    check('32 tokens: busiest GPU 1.28 x average', num(max(r32)/(32*8/8), 2) == '1.28', r32)

    # ---------------- 4. the guided parts ----------------
    T = lambda: pg.evaluate("()=>document.querySelector('#mgs-rail .step.current').innerText.replace(/\\u00a0/g,' ')")
    click = lambda s, w=250: (pg.evaluate("s=>document.querySelector(s).click()", s), pg.wait_for_timeout(w))
    nxt = lambda: click('#mgs-rail .step.current [data-nav="next"]', 350)

    def finish():
        pg.evaluate("()=>{const r=document.querySelector('#mgs-prog'); r.value='1000'; r.dispatchEvent(new Event('input',{bubbles:true})); r.dispatchEvent(new Event('change',{bubbles:true}));}")
        pg.wait_for_timeout(150)

    def has(where, txt, needles):
        miss = [x for x in needles if x not in txt]
        check(where, not miss, f'missing {miss} in: {txt[:240]}')

    r1, r8, r16 = twin('70b', 8, 1, 1, 32, 4096), twin('70b', 8, 8, 1, 32, 4096), twin('70b', 16, 16, 1, 32, 4096)
    r4, r2 = twin('70b', 8, 4, 1, 32, 4096), twin('70b', 8, 2, 1, 32, 4096)
    rpp, rr2 = twin('70b', 16, 8, 2, 32, 4096), twin('70b', 16, 8, 1, 32, 4096)
    one = 2*(p70['body'] + p70['head'])/BW                       # one GPU reading the weights alone (part 1)
    # S1: the same work as TP 8 for 32 users (weights read plus the users' KV cache), on one GPU if it fit, computed here
    one_all = (2*(p70['body'] + p70['head']) + 32*4096*p70['kv'])/BW
    check('S1: one GPU doing TP 8\'s work takes 90.5 ms; TP 8 reads and computes exactly an eighth of it', ms(one_all) == '90.5 ms' and close(r8['tcm'], one_all/8), (one_all, r8['tcm']))
    check('S1: TP 8 is about 6.2 times faster than that, not 4.8', num(one_all/r8['tpot'], 1) == '6.2', one_all/r8['tpot'])
    check('S2: 139 GB read leaves out the embedding table; 131.4 GiB stored includes both tables', round(2*(p70['body'] + p70['head'])/1e9) == 139 and gi(2*p70['total']) == '131.4 GiB')
    for L_ in ('en', 'zh'):
        z = L_ == 'zh'
        lang(L_); pg.evaluate("()=>{location.hash='';}"); pg.wait_for_timeout(200)
        pg.reload(); pg.wait_for_timeout(300); lang(L_)
        pg.evaluate("()=>{location.hash='multi-gpu-serving'}"); pg.wait_for_timeout(500)
        has(f'{L_} part 1 numbers', T(), ['131.4 GiB', '70 GiB', '139 GB', ms(one, z)])
        tipm = pg.evaluate("()=>[...document.querySelectorAll('#mgs-rail [data-tipx]')].map(e=>e.getAttribute('data-tipx')).join(' | ')")
        has(f'{L_} part 1: the 70 GiB tooltip adds up and pins the vLLM default (S3)', tipm, ['80 GiB × 90% − 2 GiB = 70 GiB', 'v0.19', 'v0.20.0', '92%'])
        check(f'{L_} part 1: 80 x 0.9 - 2 = 70', 80*0.9 - 2 == 70)
        nxt(); finish()
        has(f'{L_} part 2 numbers', T(), ['8,192', '16 KiB', kib(r8['S']), ms(r8['tpot'], z), ms(one, z), ms(one_all, z), num(one_all/r8['tpot'], 1)])
        check(f'{L_} part 2: 512 KiB is 32 users x 16 KiB, written next to it', kib(r8['S']) == '512 KiB' and r8['S'] == 32*16*1024 and ('(32 users × 16 KiB)' in T() if L_ == 'en' else '（32 个用户 × 16 KiB）' in T()), T()[-300:])
        check(f'{L_} part 2: no 4.8 times any more', '4.8' not in T(), T()[-300:])
        nxt()
        click('#mgs-rail [data-lever="4"]')
        has(f'{L_} part 3, TP 4', T(), [ms(r4['tpot'], z), ms(r8['tpot'], z), pc(r4['share']), pc(r8['share'])])
        check(f'{L_} part 3: TP 4 shows 23.0 with its decimal', ms(r4['tpot'], z) in ('23.0 ms', '23.0 毫秒'), ms(r4['tpot'], z))
        click('#mgs-rail [data-lever="2"]')
        has(f'{L_} part 3, TP 2 does not fit', T(), [gi(r2['w']), gi(r2['room']), gi(r2['need'])])
        check(f'{L_} part 3: TP 2 does not fit (twin)', not r2['fits'])
        click('#mgs-rail [data-lever="8"]')
        has(f'{L_} part 3, TP 8', T(), [ms(r8['tpot'], z), ms(r4['tpot'], z)])
        check(f'{L_} part 3: the group of eight also makes more tokens per GPU', r8['tps'] > r4['tps'])
        nxt()
        has(f'{L_} part 4 numbers', T(), [ms(r8['tcm'], z), ms(r8['tComm'], z)])
        check(f'{L_} part 4: the two parts shown add up to the 14.5 ms shown', round(float(fx(r8['tcm']*1e3, 1)) + float(fx(r8['tComm']*1e3, 1)), 1) == float(fx(r8['tpot']*1e3, 1)))
        click('#mgs-rail [data-pred="slow"]'); nxt()
        click('#mgs-rail [data-lever="16"]')
        has(f'{L_} part 5 numbers', T(), [ms(r16['tpot'], z), ms(r8['tpot'], z), ms(r8['tcm'], z), ms(r16['tcm'], z)])
        check(f'{L_} part 5: the merges take more than half, and 7.0 is more than half of 11.3 (not halved)', r16['share'] > 0.5 and r16['tcm'] > r8['tcm']/2 and ms(r16['tcm']) == '7.0 ms')
        check(f'{L_} part 5: TP 16 slower, sooner first token', r16['tpot'] > r8['tpot'] and r16['ttft'] < r8['ttft'])
        nxt()
        click('#mgs-rail [data-lever="pp"]'); click('#mgs-rail [data-lever="rep"]')
        has(f'{L_} part 6 numbers', T(), [ms(rpp['tpot'], z), ms(rr2['tpot'], z), ms(r8['tpot'], z), gi(2*params('405b')['total'])])
        check(f'{L_} part 6: pipeline and replicas both beat TP 16, and one node of TP 8 (S12)', max(rpp['tpot'], rr2['tpot']) < min(r16['tpot'], r8['tpot']))
        check(f'{L_} part 6: the KV cache read per GPU halves on 16 GPUs', close(rr2['need'], r8['need']/2))
        nxt()
        for ntok in (256, 2048, 32):
            click(f'#mgs-rail .seg-ctl[data-key="p7"] button[data-v="{ntok}"]'); finish()   # the layer plays; its end shows every count
            per = route(ntok); avg = ntok
            has(f'{L_} part 7, {ntok} tokens (card)', T(), [num(max(per)/avg, 2)])
            st = pg.evaluate("()=>document.querySelector('#mgs-stage').innerText")
            has(f'{L_} part 7, {ntok} tokens (the stage draws the MoE layer, S13)', st, ['DeepSeek', it(avg)] + [it(v) for v in per])
        hid = pg.evaluate("()=>['#mgs-gb','#mgs-lb','#mgs-metrics','#mgs-math'].map(s=>document.querySelector(s).hidden)")
        check(f'{L_} part 7: the 70B pictures, metric cards and math are hidden', hid == [True]*4, hid)
        st7 = pg.evaluate("()=>document.querySelector('#mgs-st-title').textContent")
        check(f'{L_} part 7: no 671B / two-node sentence in the card; the stage title names one MoE layer on 8 GPUs, 32 routed experts each (reader 2026-10-16)',
              '671B' not in T() and 'two H100 nodes' not in T() and '两台 H100' not in T() and all(x in st7 for x in ((['One MoE layer', 'on 8 GPUs', '32 routed experts each'] if L_ == 'en' else ['的一个 MoE 层', '8 张卡各放 32 个路由专家']))), (T()[:300], st7))
        check('256 routed experts on 8 GPUs: 32 each', 256 % 8 == 0 and 256 // 8 == 32)
        check('671B at one byte each is about 625 GiB: more than 8 x 70 GiB, within 16 x 70 GiB (two nodes)', 8*70 < 671e9/GiB <= 16*70 and round(671e9/GiB) == 625)
        nxt()
        has(f'{L_} part 8 numbers', T(), [ms(r16['tpot'], z), ms(r8['tpot'], z), '131.4 GiB', num(max(route(32))/32, 1)])
        # metric cards of the TP 8 view
        cards = pg.evaluate("()=>[...document.querySelectorAll('#mgs-metrics .metric .v')].map(e=>e.textContent)")
        check(f'{L_} metric cards (TP 8)', cards == [ms(r8['tpot'], z), it(r8['tps']), gi(r8['w']), pc(r8['share']), ms(r8['ttft'], z)], cards)

    # ---------------- 5. scenarios ----------------
    lang('en')
    click('#mgs-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 400)
    presets = pg.evaluate("()=>window.__mgs.PRESETS")
    for i, p_ in enumerate(presets):
        click(f'#mgs-rail [data-preset="{i}"]', 300)
        s = p_['s']; t = twin(**s)
        txt = pg.evaluate("()=>document.querySelector('#mgs-insight').innerText")
        need = [ms(t['tpot'])]
        if p_['id'] in ('tp8', 'tp16', 'pp', 'fp8'): need.append(pc(t['share']))
        if p_['id'] == 'fp8': need += [ms(r8['tpot']), ms(r8['tcm']), ms(t['tcm']), pc(r8['share'])]
        if p_['id'] == '405f': need += [gi(t['w']), gi(t['need'])]
        if p_['id'] == '405b': need += [ms(twin('405b', 16, 16, 1, 32, 4096)['tpot']), ms(twin('405b', 16, 8, 2, 128, 4096)['tpot']), '128']
        has(f'scenario {p_["id"]}', txt, need)
        check(f'scenario {p_["id"]} fits', t['fits'])
        cards = pg.evaluate("()=>document.querySelector('#mgs-metrics .metric .v').textContent")
        check(f'scenario {p_["id"]} card', cards == ms(t['tpot']), cards)
    check('405B BF16: TP 16 faster than TP 8 x 2 here (the copy says so)', twin('405b', 16, 16, 1, 32, 4096)['tpot'] < twin('405b', 16, 8, 2, 32, 4096)['tpot'])
    check('FP8 70B: merges take a larger share than BF16', twin('70b8', 8, 8, 1, 32, 4096)['share'] > r8['share'])
    f8 = twin('70b8', 8, 8, 1, 32, 4096)
    check('S17: FP8 halves only the weight bytes: reading falls 11.3 -> 7.0 ms, not to half', ms(r8['tcm']) == '11.3 ms' and ms(f8['tcm']) == '7.0 ms' and f8['tcm'] > r8['tcm']/2)
    check('S18: the official 405B FP8 release (56.7 GiB) plus 32 users (7.9 GiB) fits in 70 GiB', 56.7 + float(fx(twin('405b8', 8, 8, 1, 32, 4096)['need']/GiB, 1)) < 70)
    b128 = (twin('405b', 16, 16, 1, 128, 4096), twin('405b', 16, 8, 2, 128, 4096))
    check('B2: with 128 users TP 16 does not fit and the pipeline does (65.4 ms)', not b128[0]['fits'] and b128[1]['fits'] and ms(b128[1]['tpot']) == '65.4 ms', b128[1]['tpot'])

    # ---------------- 5b. Show the math: every substitution computes, here, from the values it shows (R6.6, S2, S20) ----------------
    def mathexp(m, g, tp, pp, u, ctx):
        p = params(m); t = twin(m, g, tp, pp, u, ctx); n_ = tp*pp; L, d = p['L'], p['d']
        g2 = lambda x: f'{x/1e9:,.2f}'
        wRead = p['wb']*(p['body'] + p['head'])/n_; kvRead = t['b']*ctx*p['kv']/(min(tp, KVH)*pp)
        tm = (float(g2(wRead).replace(',', '')) + float(g2(kvRead)))*1e9/BW
        F = (2*p['body']*t['b'] + 2*p['head']*t['b'] + t['b']*4*L*d*(ctx + 1))/(pp*tp); Fs = round(F/1e12, 3 if F < 1e12 else 2)
        tc = Fs*1e12/(p['peak']*0.5); tmS, tcS = round(tm*1e3, 2), round(tc*1e3, 2); tcmS = max(tmS, tcS)
        out = [f"{g2(p['wb']*p['total'])}GB", f"={g2(t['w'])}GB={fx(t['w']/GiB, 1)}GiB", f"={g2(t['need'])}GB={fx(t['need']/GiB, 1)}GiB",
               f'{g2(wRead)}GB+{g2(kvRead)}GB', f'={tmS:.2f}ms', f'{Fs}TFLOP', f'={tcS:.2f}ms']
        arS = 0.0
        if tp > 1:
            S_ = t['b']*d*2
            if tp <= 8: bi, bx, t0 = round(2*(tp - 1)/tp*S_/NVLINK*1e6, 2), 0.0, 18
            else: k = tp//8; bi, bx, t0 = round(2*7/8*S_/NVLINK*1e6, 2), round(2*(k - 1)/k*(S_/8)/IBAND*1e6, 2), 60
            arS = round(t0 + bi + bx, 2)
            out.append(f'{t0}+{bi:.2f}' + (f'+{bx:.2f}' if tp > 8 else '') + f'={arS:.2f}µs')
        sendS = round(t['tSend']*1e3, 2); tpS = pp*(tcmS + t['nAR']*arS/1e3) + sendS
        if pp > 1: out.append(f"{pp}×({tcmS:.2f}ms+{t['nAR']}×{arS:.2f}µs)+{sendS:.2f}ms={tpS:.2f}ms")
        elif tp > 1: out.append(f"{tcmS:.2f}ms+{t['nAR']}×{arS:.2f}µs={tpS:.2f}ms")
        if t['fits']: out.append(f"={round(u/(round(tpS, 2)/1e3)/g):,}")
        return out, t, tpS
    pg.evaluate("()=>{location.hash=''}"); pg.wait_for_timeout(300)
    for tok in ('70b_8_8_1_32_4096', '70b_8_4_1_32_4096', '70b_16_16_1_32_4096', '70b_16_8_2_32_4096', '70b8_8_8_1_32_4096', '405b8_8_8_1_32_4096', '405b_16_8_2_32_4096', '405b_16_16_1_32_4096', '70b_8_8_1_128_4096'):
        m_, g_, tp_, pp_, u_, c_ = tok.split('_')
        pg.evaluate(f"()=>{{location.hash='multi-gpu-serving~{tok}'}}"); pg.wait_for_timeout(450)
        pg.evaluate("()=>{const d=document.querySelector('#mgs-math'); d.open=true}"); pg.wait_for_timeout(120)
        mt = ''.join(pg.evaluate("()=>document.querySelector('#mgs-mathbody').innerText").split())
        exp, t_, tpS = mathexp(m_, int(g_), int(tp_), int(pp_), int(u_), int(c_))
        miss = [x for x in exp if x not in mt]
        check(f'math {tok}: every substitution computes to its shown result', not miss, (miss, mt[:400]))
        check(f'math {tok}: its TPOT, from rounded inputs, is within 0.01 ms of the engine', abs(tpS - t_['tpot']*1e3) < 0.01, (tpS, t_['tpot']))
        sm = pg.evaluate("()=>document.querySelector('#mgs-sum').innerText")
        nums = [float(x.replace(',', '')) for x in __import__('re').findall(r'about ([\d,.]+) ms|\(([\d,.]+) ms\)', sm) for x in x if x]
        tot = __import__('re').search(r'Each output token: ([\d,.]+) ms', sm)
        check(f'math {tok}: the summary line adds up', tot and abs(sum(nums) - float(tot.group(1))) < 1e-9, sm)
        pg.evaluate("()=>{location.hash=''}"); pg.wait_for_timeout(250)
    pg.evaluate("()=>{location.hash='multi-gpu-serving'}"); pg.wait_for_timeout(500)

    # ---------------- 6. the challenge (B1: the reader also picks 8 or 16 GPUs; 128 users) ----------------
    CH = dict(m='70b', u=128, ctx=4096)
    opts = pg.evaluate("()=>window.__mgs.chOpts()")
    mine = [(g, tp, pp) for g in (8, 16) for tp in (1, 2, 4, 8, 16) for pp in (1, 2, 4, 8, 16) if tp <= g and tp*pp <= g]
    check('challenge options', [(o['g'], o['tp'], o['pp']) for o in opts] == mine, opts[:3])
    badc = []
    for o in opts:
        t = twin(g=o['g'], tp=o['tp'], pp=o['pp'], **CH)
        ok = t['fits'] and t['tpot'] <= 0.025 + 1e-12
        if ok != o['ok'] or not close(t['tpot'], o['tpot']) or not close(t['tps'], o['tps']): badc.append((o, t['tpot'], ok))
    check('challenge options match the twin', not badc, badc[:3])
    tw = {k: twin(g=k[0], tp=k[1], pp=k[2], **CH) for k in mine}
    feas = [k for k in mine if tw[k]['fits'] and tw[k]['tpot'] <= 0.025]
    best = max(feas, key=lambda k: tw[k]['tps'])
    check('challenge: passing setups are one node of TP 8, two replicas of TP 8, TP 8 x 2 stages', sorted(feas) == [(8, 8, 1), (16, 8, 1), (16, 8, 2)], feas)
    check('challenge: best is one node of TP 8 (23.5 ms, 680)', best == (8, 8, 1) and ms(tw[best]['tpot']) == '23.5 ms' and it(tw[best]['tps']) == '680', (best, tw[best]['tpot'], tw[best]['tps']))
    check('challenge: the page finds the same best', pg.evaluate("()=>window.__mgs.chBest()") == {'g': 8, 'tp': 8, 'pp': 1})
    check('challenge: initial TP 16 on 16 GPUs misses (26.8 ms)', ms(tw[(16, 16, 1)]['tpot']) == '26.8 ms' and tw[(16, 16, 1)]['tpot'] > 0.025)
    check('challenge: two nodes are faster per token but make fewer tokens per GPU', tw[(16, 8, 1)]['tpot'] < tw[best]['tpot'] and tw[(16, 8, 1)]['tps'] < tw[best]['tps'])
    # B1: no guided part, scenario or exit shows 128 users of 70B in BF16, so no part prints this answer
    check('challenge: no scenario serves 128 users of 70B in BF16', all(not (p_['s']['m'] == '70b' and p_['s']['u'] == 128) for p_ in presets))
    click('#mgs-app .modes button[data-mode="challenge"]', 400)
    click('#mgs-rail .seg-ctl[data-key="chg"] button[data-v="8"]')
    st = pg.evaluate("()=>window.__mgs.state()")
    check('challenge: 8 GPUs bring TP 16 back to TP 8', (st['g'], st['tp'], st['pp']) == (8, 8, 1), st)
    click('#mgs-rail [data-act="submit"]', 350)
    res = pg.evaluate("()=>document.querySelector('#mgs-ch-res').innerText")
    tb = tw[best]
    has('challenge: one node of TP 8 is told it is the best', res, ['Met', ms(tb['tpot']), it(tb['tps']), 'most tokens per second', it(tw[(16, 8, 1)]['tps'])])
    # exits
    click('#mgs-rail [data-exits] [data-ex="tab"][data-i="1"]')
    ex = pg.evaluate("()=>document.querySelector('#mgs-rail [data-exits]').textContent")
    q1 = pg.query_selector('#mgs-rail [data-exits] [data-ex="q"][data-j="1"]')
    if q1:
        click('#mgs-rail [data-exits] [data-ex="q"][data-j="1"]'); click('#mgs-rail [data-exits] [data-ex="ans"][data-v="0"]')
        ex += pg.evaluate("()=>document.querySelector('#mgs-rail [data-exits]').textContent")
    t32 = 64*(A_IN + 1*(16*4096*2)/NVLINK)
    # S26 (b): 64 users, a scale no part or scenario shows; the premise gives TP 16's own reading time (the KV cache does not halve)
    t16 = twin('405b', 16, 16, 1, 64, 4096); tpp = twin('405b', 16, 8, 2, 64, 4096)
    has('exits numbers', ex, [ms(t32), ms(r8['tComm']), ms(2*tpp['tcm']), ms(t16['tcm']), str(t16['nAR']), us(t16['ar']), ms(t16['tpot']), ms(tpp['tpot'])])
    check('exits: 405B, 64 users: TP 16 52.0 ms beats the pipeline 60.1 ms; TP 16 reads more than half of the pipeline', (ms(t16['tpot']), ms(tpp['tpot'])) == ('52.0 ms', '60.1 ms') and t16['tcm'] > tpp['tcm'], (t16['tpot'], tpp['tpot']))
    check('exits: no scenario uses 64 users', all(p_['s']['u'] != 64 for p_ in presets))

    # ---------------- 7. the minimal implementation ----------------
    code = pg.evaluate("()=>document.querySelector('#mgs-code').textContent")
    out = io.StringIO()
    with contextlib.redirect_stdout(out): exec(compile(code, 'mgs-code', 'exec'), {})
    printed = out.getvalue().strip().splitlines()
    shown = [l[2:] for l in code.splitlines() if l.startswith('# TP ')]
    check('code: printed output equals the comment lines', printed == shown, (printed, shown))
    check('code: equals the engine', printed == [f'TP 8: {r8["tpot"]*1e3:.1f} ms per token', f'TP 16: {r16["tpot"]*1e3:.1f} ms per token'], printed)
    check('code: W_READ', 'W_READ = 139.006e9' in code and abs(2*(p70['body'] + p70['head']) - 139.006e9) < 1e6)

    # ---------------- 8. the interview drill ----------------
    for k in range(4):
        seed = k + 1
        rnd = mulberry(seed*7919 + 5); tp = [4, 8][math.floor(rnd()*2)]; u = [8, 16, 32, 64][math.floor(rnd()*4)]
        check(f'drill {seed}: TP {tp} with {u} users fits at 4,096 tokens (O7)', twin('70b', tp, tp, 1, u, 4096)['fits'])
        q = pg.evaluate("()=>document.querySelector('#mgs-drill-q').innerText")
        check(f'drill {seed}: question', f'TP {tp} ' in q and f' {u} users' in q, q[:160])
        click('[data-act="drill-reveal"]')
        a = pg.evaluate("()=>document.querySelector('#mgs-drill-a').innerText")
        S = u*8192*2; arS = f'{allreduce(S, tp)[0]*1e6:.2f}'; tot = f'{160*float(arS)/1e3:.2f}'
        has(f'drill {seed}: answer (S27: two decimals, the total from the shown value)', a, [kib(S), f'{S:,}', arS, f'{tot} ms'])
        click('[data-act="drill-new"]')
    check('no page errors', not errs, errs[:3])
    b.close()

print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
