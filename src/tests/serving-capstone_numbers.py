"""The serving capstone's numbers, recomputed in Python from capstone/engine.py (the reference) and compared with what the page shows.
Independent of the lab's view code: Python re-derives what the page derives (goal checks, the worst stall and its step, the KV pool's
split into own / shared-in-use / idle blocks, waits by reason, the challenge's tiers by brute force, the new cases' custom loads).
- window.__sc.run / .base equal the Python metrics for every state visited;
- the metrics table (initial setup | yours) shows the Python values in the page's formats, in EN and ZH;
- each part's option results: only "x / n meet the goal" (review r3, R3.17), from Python;
- the longest-stall label and, after "Show me", the GPU step in the strip (its number, its duration, the document's 32,768-token block);
- the KV pool at every 37th step of four runs (with evictions): own + shared blocks = Python's used; idle and set-aside blocks equal;
- waits by reason for every request of three runs; the per-request goal (met or missed) matches the metrics' counts;
- the challenge: goal met for several plans; the tier (fewest changes) against a Python search over every setting; the plan text;
- "A new case": the 64K document's worst stall and the 12,288-token prompt's prefill tokens from Python runs of those loads;
- the interview drill for three seeds; the diagnosis of each plausible-but-failing plan against Python's dominant wait reason;
- the home tile's three options (no document; the document under the initial setup; the document under the setup the guided parts end
  on): every row's segments, met or missed, the box, the axis, the sentence, the picture's description, the legend and the notes from
  Python runs; the setting is the lab's own and does not meet the challenge; nothing names the cause or a setting (R3.8); the height
  kept across the options; 390 px, a 280 px tile, dark mode.
Run: python3 tests/serving-capstone_numbers.py"""
import pathlib, subprocess, sys, math, json
from playwright.sync_api import sync_playwright
P = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / 'capstone'))
import engine as E

subprocess.run([sys.executable, str(P / 'build.py'), 'out/serving-capstone-num.html', '--only', 'serving-capstone'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-serving-capstone-num.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'serving-capstone-num.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-serving-capstone-num.html')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


NB = lambda s: s.replace(' ', ' ').replace(' ', ' ').replace('‑', '-')
BASE = {'policy': 'continuous', 'cap': 32, 'budget': 0, 'cache': False, 'cache_blocks': None}
EPS = 1e-9
_C = {}


def run(load, **kw):
    cfg = {**BASE, **kw}
    key = (load, json.dumps(cfg, sort_keys=True))
    if key not in _C:
        r = E.simulate(E.LOADS[load](), cfg, trace=True); E.check(r); _C[key] = (r, E.metrics(r, E.GOALS))
    return _C[key]


# ---------- the page's formats, written again
def fint(v): return f'{math.floor(v + 0.5):,}'
def fS(sec):
    v = sec * 1e3
    if v < 9.995: return f'{v:.2f} ms'
    if v < 99.95: return f'{v:.1f} ms'
    if v < 999.5: return f'{fint(v)} ms'
    if sec < 9.995: return f'{sec:.2f} s'
    return f'{sec:.1f} s'
def fG(g): return f'{g:.1f} GiB'
def pct(x): return f'{fint(x * 100)}%'


def gap(r): return max((b - a for a, b in zip(r.toks, r.toks[1:])), default=0.0)
# GPU time used: the step times added up, from the Python run's own steps (review r2, N1: it replaced "GPU busy", 100% whenever requests run)
def gput(runr): return sum(s['t1'] - s['t0'] for s in runr['steps'])
# the step during which the engine's peak is reached: memory after it + what its finishing requests release + the idle prompts trimmed
# after those releases = the peak (the engine takes its peak before the releases)
def own_blocks(r, k, cache):
    se = (r.pre // 16) * 16 if cache and r.app else 0; ob = 0
    for (si, a, b) in r.chunks:
        if si > k: break
        oa = max(a, se)
        if b > oa: ob += E.blocks(b - se) - E.blocks(oa - se)
    nd = sum(1 for d in r.dsteps if d <= k)
    return ob + sum(1 for j in range(1, nd + 1) if (r.P + j - 1) % 16 == 0)
def peak_step(runr, cache):
    ev, st = runr['events'], runr['steps']; t1i = {s['t1']: i for i, s in enumerate(st)}; trims = {}
    for q0, e in enumerate(ev):
        if e[1] != 'evict': continue
        q = q0 + 1
        while q < len(ev) and ev[q][1] == 'evict': q += 1
        if q < len(ev) and ev[q][1] == 'finish' and ev[q][0] == e[0]: trims[t1i[e[0]]] = trims.get(t1i[e[0]], 0) + 1
    for k, s in enumerate(st):
        rel = sum(own_blocks(x, k, cache) for x in runr['R'] if abs(x.fin - s['t1']) < 1e-12)
        if runr['mem'][k][1] + rel + trims.get(k, 0) == runr['peak']: return k
    return None
def ok_of(r):
    g = E.GOALS[r.cls]
    return (r.first - r.arr) <= g['ttft'] + EPS and gap(r) <= g.get('gap', 1e9) + EPS
def worst(runr):
    best = None
    for r in runr['R']:
        if r.cls != 'chat': continue
        for i in range(1, len(r.toks)):
            gp = r.toks[i] - r.toks[i - 1]
            if best is None or gp > best[0] + 1e-12: best = (gp, r.id, r.dsteps[i - 1])
    return best


# ---------- the KV pool after step k, derived from the Python run (independent of the page's code)
def pool_after(runr, cfg, k):
    st = runr['steps']; s = st[k]; cache = cfg['cache']
    own = 0; inuse = {}
    for r in runr['R']:
        if not (r.admit <= s['t0'] + EPS and r.fin > s['t1'] + EPS): continue
        se = (r.pre // 16) * 16 if cache and r.app else 0
        done = r.hit; ob = 0
        for (si, a, b) in r.chunks:
            if si > k: break
            done = b; oa = max(a, se)
            if b > oa: ob += E.blocks(b - se) - E.blocks(oa - se)
        nd = sum(1 for d in r.dsteps if d <= k)
        for j in range(1, nd + 1):
            if (r.P + j - 1) % 16 == 0: ob += 1
        own += ob
        if se:
            inuse[r.app] = max(inuse.get(r.app, 0), min(done, se) // 16)
    return own, inuse


# ---------- part 7: the scheduler's code, step by step. An instrumented copy of the Python engine records what the page can only derive:
# the reason it recorded for every request it did not admit (in order, per step), the blocks each request frees, and where each eviction
# happens (while a step allocates, or after a release over the cache budget)
_src = (P / 'capstone' / 'engine.py').read_text()
_patches = [("        st['free'] += r.blk_own; r.blk_own = 0; r.resv = 0\n", "        OWN[r.id] = r.blk_own; st['free'] += r.blk_own; r.blk_own = 0; r.resv = 0\n"),
            ("        trim_cache()\n\n", "        PH[0] = 'trim'; trim_cache(); PH[0] = 'alloc'\n\n"),
            ("            if trace: events.append((st['t'], 'evict', k))\n", "            if trace: events.append((st['t'], 'evict', k))\n            EVS.append((len(steps) - (1 if PH[0] == 'trim' else 0), PH[0]))\n"),
            ("                    reasons[r.id] = blocked; continue\n", "                    reasons[r.id] = blocked; BLK.setdefault(len(steps), []).append((r.id, blocked)); continue\n")]
for a_, b_ in _patches:
    assert _src.count(a_) == 1, a_
    _src = _src.replace(a_, b_)
EI = {'__name__': 'engine_instr', 'OWN': {}, 'PH': ['alloc'], 'EVS': [], 'BLK': {}}
exec(compile(_src, 'engine_instr', 'exec'), EI)


def irun(cfg):
    for k in ('OWN', 'EVS', 'BLK'): EI[k].clear() if hasattr(EI[k], 'clear') else None
    r = EI['simulate'](EI['LOADS'][cfg['load']](), {'policy': cfg['policy'], 'cap': cfg['cap'], 'budget': cfg['budget'], 'cache': cfg['cache'], 'cache_blocks': cfg['cb'] if cfg['cache'] else None}, trace=True)
    return r, dict(EI['OWN']), list(EI['EVS']), {k: list(v) for k, v in EI['BLK'].items()}


def step_facts(r, own, evs, blk, s):
    st = r['steps'][s]; R = r['R']; b0 = r_budget = None
    adm = sorted([q for q in R if q.admit == st['t0']], key=lambda q: (q.arr, q.id)); aid = {q.id for q in adm}
    dec = [w for w in st['work'] if w[1] == 'd']; pre = [w for w in st['work'] if w[1] == 'p']
    cont = [w for w in pre if w[0] not in aid]; apre = {w[0]: w for w in pre if w[0] in aid}
    return {'dec': len(dec), 'cont': [(w[0], w[3] - w[2], R[w[0]].P - w[2]) for w in cont], 'adm': [(q.id, q.hit, apre[q.id][3] - apre[q.id][2] if q.id in apre else 0) for q in adm],
            'wait': [i for i, _ in blk.get(s, [])], 'why': blk[s][0][1] if blk.get(s) else None, 'T': st['T'], 'nrun': st['nrun'],
            'fin': [(q.id, own[q.id]) for q in sorted([q for q in R if q.fin == st['t1']], key=lambda q: (q.arr, q.id))],
            'evT': sum(1 for i, ph in evs if i == s and ph == 'trim'), 'evA': sum(1 for i, ph in evs if i == s and ph == 'alloc')}


def find_moment(mid, r, own, evs, blk):
    for s in range(len(r['steps'])):
        f = step_facts(r, own, evs, blk, s)
        if mid == 'usual' and f['dec'] >= 3 and f['adm'] and not f['wait'] and not f['cont']: return s
        if mid == 'budget' and f['wait'] and f['why'] == 'budget': return s
        if mid == 'cache' and any(h > 0 for _, h, _ in f['adm']): return s
        if mid == 'evict' and f['evT'] > 0: return s
        if mid == 'cap' and f['wait'] and f['why'] == 'cap': return s
    return -1


with sync_playwright() as p:
    b = p.chromium.launch()
    # ---------- part 7 against the instrumented engine
    pg = b.new_page(viewport={'width': 1440, 'height': 1000}); pg.goto(URL); pg.wait_for_timeout(250)
    pg.evaluate("()=>{location.hash='serving-capstone'}"); pg.wait_for_timeout(600)
    moms = pg.evaluate("()=>window.__sc.moments()")
    for m in moms:
        r, own, evs, blk = irun(m['cfg'])
        s = find_moment(m['id'], r, own, evs, blk)
        check(f'part 7 “{m["id"]}”: the page finds the same step as the instrumented engine ({s + 1})', m['s'] == s and s >= 0, (m['s'], s))
        f = step_facts(r, own, evs, blk, s)
        ci = pg.evaluate(f"()=>window.__sc.codeInfo('{m['cfg']['load']}',{json.dumps(m['cfg'])},{s})")
        B0 = m['cfg']['budget'] or None
        got = {'dec': len(ci['dec']), 'cont': [(c['id'], c['n'], c['left']) for c in ci['cont']], 'adm': [(a['id'], a['hit'], a['n']) for a in ci['adm']], 'wait': ci['wait'], 'why': ci['why'],
               'T': ci['st']['T'], 'nrun': ci['st']['nrun'], 'fin': [(x['id'], x['own']) for x in ci['fin']], 'evT': ci['evT'], 'evA': ci['evA']}
        check(f'part 7 “{m["id"]}”: decodes, prefills that go on, admissions with their cache hit, the waiting and why, the finished with their blocks, evictions', got == f, (got, f))
        if B0:
            bd = B0 - f['dec']; bp = bd - sum(n for _, n, _ in f['cont']); be = bp - sum(n for _, _, n in f['adm'])
            check(f'part 7 “{m["id"]}”: the budget left after each loop ({bd:,}, {bp:,}, {be:,})', (ci['bDec'], ci['bPre'], ci['bEnd']) == (bd, bp, be), (ci['bDec'], ci['bPre'], ci['bEnd']))
        st_ = r['steps'][s]
        L_ = pg.evaluate(f"()=>window.__sc.codeLines('{m['cfg']['load']}',{json.dumps(m['cfg'])},{s},'end')")
        exp = {'tm': '+' + fS(st_['t1'] - st_['t0']), 'dl': (f"{f['dec']} decode" + ('s' if f['dec'] > 1 else '')) if f['dec'] else 'none'}
        if f['cont']: exp['pn'] = f"n = {f['cont'][0][1]:,}"
        if f['adm']:
            exp['an'] = f"n = {f['adm'][0][2]:,}"; h_ = f['adm'][0][1]
            # the cache hit is a number of blocks, as the code uses it (review A, finding 5); r.done = hit × BLOCK tokens
            exp['rd'] = f"{h_:,} tokens"
            if m['cfg']['cache']:
                exp['lk'] = f"{h_ // 16:,} blocks" if h_ else 'miss: 0'
                if h_: exp['rf'] = f"+1 on {h_ // 16:,}"
        if f['fin']:
            exp['fr'] = f"frees {sum(o for _, o in f['fin']):,}"
            sh_ = [R_ for R_ in (r['R'][i] for i, _ in f['fin']) if m['cfg']['cache'] and R_.app and R_.pre // 16 > 0]
            if sh_: exp['un'] = f"−1 on {sh_[0].pre // 16:,}"
        if f['evT']: exp['ev'] = f"evicts {f['evT']:,}"
        if f['why'] == 'budget': exp['ac'] = '0 left: stop'
        if f['why'] == 'cap': exp['ac'] = f"cap {m['cfg']['cap']}: stop"
        bad = {k: (NB(L_[k]['v']), v) for k, v in exp.items() if NB(L_[k]['v']) != v}
        check(f'part 7 “{m["id"]}”: the panel’s values in English ({", ".join(f"{k}={v}" for k, v in exp.items())})', not bad, bad)
        long_ = {k: v['v'] for k, v in L_.items() if len(NB(v['v'])) > 12}
        check(f'part 7 “{m["id"]}”: every English value fits the 12-character column', not long_, long_)
        if m['id'] == 'cache':   # the cached prefix was computed by the app's last request that computed shared blocks itself
            a_ = r['R'][f['adm'][0][0]]
            src = max((q for q in r['R'] if q.app == a_.app and q.shared_new > 0 and q.admit < st_['t0']), key=lambda q: q.admit)
            ob_ = pg.evaluate(f"()=>window.__sc.codeObs('cache')")
            check(f'part 7 “cache”: the card names the request that computed the cached prompt ({a_.app} #{src.id + 1})', f"{a_.app} #{src.id + 1}" in NB(ob_), ob_[:200])
        if m['id'] == 'budget':  # “the chunk takes the rest of the budget” only when it does
            check('part 7 “budget”: the running prefill takes all of the budget the decodes leave', f['cont'][0][1] == m['cfg']['budget'] - f['dec'], (f['cont'], f['dec']))
    # ---------- review B: every step of the moments' four runs, both phases (values, sums, wording by state)
    runs = {}
    for m in moms: runs.setdefault(m['cfg']['load'] + json.dumps(m['cfg'], sort_keys=True), m)
    for m in runs.values():
        res = pg.evaluate("""(m)=>{const out={n:0,long:[],sum:[],multi:0,even:[],nocb:[],un:[]}; const V=window.__sc.view(m.cfg.load,m.cfg), N=V.nSteps;
          for(let s=0;s<N;s++){const ci=window.__sc.codeInfo(m.cfg.load,m.cfg,s);
            for(const ph of ['run','end']){const L=window.__sc.codeLines(m.cfg.load,m.cfg,s,ph); out.n++;
              for(const k in L){const v=L[k].v.replace(/\u00a0/g,' '); if(v.length>12) out.long.push([s,k,v]);}
              if(ph!=='end') continue;
              if(ci.adm.length>1){out.multi++; const v=L.an.v.replace(/\u00a0/g,' '), ns=ci.adm.map(a=>a.n), tot=ns.reduce((a,b)=>a+b,0);
                const ok=v.includes('Σ')?+v.replace(/[^0-9]/g,'')===tot:v.split(' + ').map(x=>+x.replace(/[^0-9]/g,'')).join()===ns.join();
                const add=ci.bPre==null||ci.bPre-tot===ci.bEnd; if(!ok||!add) out.sum.push([s,v,ns,ci.bPre,ci.bEnd]);}
              if((ci.why==='cap'||ci.why==='budget')&&/even if/.test(L.ab.tip)) out.even.push([s,L.ab.tip]);
              if(ci.cacheOn&&ci.cb==null&&!ci.evT&&!/no cache budget/.test(L.ev.tip)) out.nocb.push([s,L.ev.tip]);
              if(ci.cacheOn&&ci.fin.length>1) out.un.push([s,L.un.v.replace(/\u00a0/g,' '),ci.fin.map(x=>x.id)]);}}
          return out;}""", m)
        tag_ = f'part 7, the {m["id"]} moment’s run ({res["n"]:,} step views)'
        check(f'{tag_}: every English value fits the 12-character column (review B, B5)', not res['long'], res['long'][:5])
        check(f'{tag_}: when a step admits several requests, line 21 lists every n and the budget adds up (review B, B3; {res["multi"]} steps)', not res['sum'], res['sum'][:3])
        check(f'{tag_}: after a cap or budget stop, nobody is said to fit (review B, B7)', not res['even'], res['even'][:2])
        check(f'{tag_}: with no cache budget, line 27 says so (review B, B8)', not res['nocb'], res['nocb'][:2])
        r_, own_, evs_, blk_ = irun(m['cfg']); bad_un = []
        for s_, v_, ids in res['un']:
            sh = [i for i in ids if r_['R'][i].app and r_['R'][i].pre // 16 > 0]          # the page's order: arrival, then id
            if not sh: want = 'none'
            else:
                a0 = r_['R'][sh[0]].app
                want = f"−{sum(1 for i in sh if r_['R'][i].app == a0)} on {r_['R'][sh[0]].pre // 16:,}"
            if v_ != want: bad_un.append((s_, v_, want))
        check(f'{tag_}: when several requests of one app finish together, line 26 counts every reference given up (review B, B9)', not bad_un, bad_un[:3])
    ob_ = NB(pg.evaluate("()=>window.__sc.codeObs('usual')"))
    check('part 7 “usual”: the card names the deciding line (review B, B15)', 'n = min(r.prompt_left, budget)' in ob_, ob_[:200])
    pg.close()
    for lang in ('en', 'zh'):
        pg = b.new_page(viewport={'width': 1440, 'height': 1000})
        errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(URL); pg.wait_for_timeout(250)
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
        pg.evaluate("()=>{location.hash='serving-capstone'}"); pg.wait_for_timeout(600)
        J = lambda s: pg.evaluate(s)
        def end():
            J("()=>{const b=document.querySelector('[data-act=toend]'); if(b){b.click(); return;} const i=document.querySelector('#sc-scrub'); i.value=1000; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(250)
        def click(sel): J(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(250)
        def table():
            return J("()=>Object.fromEntries([...document.querySelectorAll('#sc-mt tr[data-k]')].map(t=>[t.dataset.k,[...t.querySelectorAll('td')].map(x=>x.textContent)]))")
        def cmp_metrics(tag, load, kw, ref=None):
            # the table's first column: in a guided part the part's own start (review r1, S2); __sc.base stays the initial setup
            r, m = run(load, **kw); rb, mb = run(load, **(ref or {})); _, m0 = run(load)
            page = J("()=>window.__sc.run"); base = J("()=>window.__sc.base")
            for c in m:
                for k, v in m[c].items():
                    pv = page[c][k]
                    check(f'{lang} {tag}: __sc.run {c}.{k}', abs(pv - v) <= 1e-9 * max(1, abs(v)), (pv, v))
                    bv = base[c][k]; wv = m0[c][k]
                    check(f'{lang} {tag}: __sc.base {c}.{k}', abs(bv - wv) <= 1e-9 * max(1, abs(wv)), (bv, wv))
            t = table()
            exp = {'ok': (f"{mb['all']['ok']} ({pct(mb['all']['ok_frac'])})", f"{m['all']['ok']} ({pct(m['all']['ok_frac'])})"),
                   'p50': (fS(mb['chat']['ttft_p50']), fS(m['chat']['ttft_p50'])), 'p95': (fS(mb['chat']['ttft_p95']), fS(m['chat']['ttft_p95'])),
                   'gap': (fS(mb['chat']['gap_max']), fS(m['chat']['gap_max'])), 'thr': (fint(mb['all']['thr']), fint(m['all']['thr'])),
                   'peak': (fG(mb['all']['peak_gib']), fG(m['all']['peak_gib'])), 'pre': (fint(mb['all']['prefill_tok']), fint(m['all']['prefill_tok'])),
                   'hit': (fint(mb['all']['hit_tok']), fint(m['all']['hit_tok'])), 'gput': (fS(gput(rb)), fS(gput(r)))}
            if 'doc' in m: exp['doc'] = (fS(mb['doc']['ttft_max']), fS(m['doc']['ttft_max']))
            for k, (bx, yx) in exp.items():
                cells = [NB(x) for x in t.get(k, ['', '', ''])]
                check(f'{lang} {tag}: table {k} initial setup', cells[0] == bx, (cells, bx))
                check(f'{lang} {tag}: table {k} yours', cells[1] == yx, (cells, yx))
            # every request's goal (met or missed) adds up to the metrics
            oks = J(f"()=>window.__sc.view('{load}',{json.dumps({'policy':kw.get('policy','continuous'),'cap':kw.get('cap',32),'budget':kw.get('budget',0),'cache':kw.get('cache',False),'cb':kw.get('cache_blocks')})}).ok")
            check(f'{lang} {tag}: per-request goals add up to the metrics', sum(oks) == m['all']['ok'] and oks == [ok_of(x) for x in r['R']], (sum(oks), m['all']['ok']))
        # ---------- part 1: the mission, played to the end
        click('#sc-play [data-act=play]'); end()
        cmp_metrics('part 1', 'stall', {})
        w = worst(run('stall')[0])
        lab = NB(J("()=>document.querySelector('#sc-stall').textContent"))
        check(f'{lang} part 1: the stall label shows the worst gap', fS(w[0]) in lab, (lab, fS(w[0])))
        # ---------- part 2: predict, then Show me: the strip shows the step behind the stall
        click('#sc-rail .step.current [data-nav=next]'); click('#sc-rail [data-pred=b]'); click('#sc-rail .step.current [data-nav=showme]')
        st = run('stall')[0]['steps'][w[2]]
        gh = NB(J("()=>document.querySelector('#sc-gh').textContent")); dt = NB(J("()=>document.querySelector('.sc-dt').textContent"))
        blk = NB(J("()=>[...document.querySelectorAll('#sc-gbar .sc-pb em')].map(e=>e.textContent).join('|')"))
        check(f'{lang} part 2: the strip shows step {w[2] + 1}', f'{w[2] + 1:,}' in gh, gh)
        check(f'{lang} part 2: the step lasts the worst gap', dt == fS(st['t1'] - st['t0']), (dt, fS(st['t1'] - st['t0'])))
        big = max((x for x in st['work'] if x[1] == 'p'), key=lambda x: x[3] - x[2])
        check(f'{lang} part 2: the document block carries its size', f'{big[3] - big[2]:,}' in blk, (blk, big))
        # ---------- part 3: budgets
        click('#sc-rail [data-go="2"]'); end()
        for v in (8192, 2048, 512, 128):
            click(f'#sc-rail [data-lever="{v}"]'); end()
            cmp_metrics(f'part 3 budget {v}', 'stall', {'budget': v})
            r, m = run('stall', budget=v)
            res = NB(J(f"()=>document.querySelector('#sc-rail [data-lever=\"{v}\"] .lv-r').textContent"))
            check(f'{lang} part 3 lever {v}: the result is exactly the goal count (R3.17)', res == (f"{m['all']['ok']} / {m['all']['n']} meet the goal" if lang == 'en' else f"{m['all']['ok']} / {m['all']['n']} 达标"), (res,))
            docc = len([x for x in r['R'] if x.cls == 'doc'][0].chunks)
            desc = NB(J(f"()=>document.querySelector('#sc-rail [data-lever=\"{v}\"] .lv-d').textContent"))
            check(f'{lang} part 3 lever {v}: the document in {docc} steps', str(docc) in desc, desc)
        # the two chats that still miss with 2,048 wait for the budget (the document's chunks go first)
        r, m = run('stall', budget=2048)
        miss = [x for x in r['R'] if not ok_of(x)]
        check(f'{lang} part 3: with 2,048 two chats miss, both waiting for the budget', len(miss) == 2 and all(x.cls == 'chat' and x.waits.get('budget', 0) > 0 for x in miss), [(x.id, x.waits) for x in miss])
        click('#sc-rail [data-lever="2048"]'); end()
        sel = J("()=>{const b=document.querySelector('#sc-rail [data-obs] [data-sel]'); return b?+b.dataset.sel:null}")
        check(f'{lang} part 3: "see why" selects one of them', sel in [x.id for x in miss], sel)
        # ---------- part 4: caching
        click('#sc-rail [data-go="3"]'); end()
        cmp_metrics('part 4 caching off', 'shared', {'budget': 2048}, {'budget': 2048})
        click('#sc-rail [data-lever="true"]'); end()
        cmp_metrics('part 4 caching on', 'shared', {'budget': 2048, 'cache': True}, {'budget': 2048})
        for v, kw in (('false', {}), ('true', {'cache': True})):
            _, m = run('shared', budget=2048, **kw)
            res = NB(J(f"()=>document.querySelector('#sc-rail [data-lever=\"{v}\"] .lv-r').textContent"))
            check(f'{lang} part 4 lever {v}: the result is exactly the goal count (R3.17)', res == (f"{m['all']['ok']} / {m['all']['n']} meet the goal" if lang == 'en' else f"{m['all']['ok']} / {m['all']['n']} 达标"), (res,))
        # the shared prompt's ×N badge: N = running support requests holding its blocks at the peak
        r, m = run('shared', budget=2048, cache=True)
        kpk = peak_step(r, True)
        users = sum(1 for x in r['R'] if x.admit <= r['steps'][kpk]['t0'] + EPS and x.fin > r['steps'][kpk]['t0'] + EPS and min(x.hit + sum(b_ - a_ for (si, a_, b_) in x.chunks if si <= kpk), (x.pre // 16) * 16) >= 16)
        memv = NB(J("()=>document.querySelector('#sc-memv').textContent"))
        check(f'{lang} part 4: the memory bar shows the peak of the table, {fG(m["all"]["peak_gib"])}, during the step ending at {r["steps"][kpk]["t1"]:.2f} s', fG(m['all']['peak_gib']) in memv and f"{r['steps'][kpk]['t1']:.2f} s" in memv, memv)
        badge = NB(J("()=>[...document.querySelectorAll('#sc-mlab .sc-mb')].map(e=>e.textContent).join('|')"))
        check(f'{lang} part 4: the shared prompt badge counts its users at the peak (×{users})', f'×{users}' in badge, badge)
        # ---------- part 5: cache budgets
        click('#sc-rail [data-go="4"]'); end()
        for v in ('3072', '6144', 'null'):
            click(f'#sc-rail [data-lever="{v}"]'); end()
        oks = []
        for v, cb in (('1536', 1536), ('3072', 3072), ('6144', 6144), ('null', None)):
            r, m = run('apps', budget=2048, cache=True, cache_blocks=cb); oks.append(m['all']['ok'])
            res = NB(J(f"()=>document.querySelector('#sc-rail [data-lever=\"{v}\"] .lv-r').textContent"))
            check(f'{lang} part 5 lever {v}: the result is exactly the goal count (R3.17)', res == (f"{m['all']['ok']} / {m['all']['n']} meet the goal" if lang == 'en' else f"{m['all']['ok']} / {m['all']['n']} 达标"), (res,))
        # R3.18: the smallest cache fails, the four-document cache passes, more budget gains nothing
        check(f'part 5 calibration (R3.18): 3 GiB fails, 12 GiB passes, no limit adds nothing {oks}', oks[0] < 0.9 * 48 and oks[2] >= 0.9 * 48 and oks[3] == oks[2], oks)
        cmp_metrics('part 5 no limit', 'apps', {'budget': 2048, 'cache': True}, {'budget': 2048, 'cache': True, 'cache_blocks': 1536})
        r, m = run('apps', budget=2048, cache=True)
        check('part 5: with every document cached, the requests that still miss are each document’s first question', all(x.hit == 0 for x in r['R'] if not ok_of(x)) and sum(1 for x in r['R'] if not ok_of(x)) <= 4, [(x.id, x.hit) for x in r['R'] if not ok_of(x)])
        # ---------- part 6: the cap
        click('#sc-rail [data-go="5"]'); end()
        for v in (8, 16, 64):
            click(f'#sc-rail [data-lever="{v}"]'); end()
            cmp_metrics(f'part 6 cap {v}', 'shared', {'budget': 2048, 'cache': True, 'cap': v}, {'budget': 2048, 'cache': True})
            _, m = run('shared', budget=2048, cache=True, cap=v)
            res = NB(J(f"()=>document.querySelector('#sc-rail [data-lever=\"{v}\"] .lv-r').textContent"))
            check(f'{lang} part 6 lever {v}: the result is exactly the goal count (R3.17)', res == (f"{m['all']['ok']} / {m['all']['n']} meet the goal" if lang == 'en' else f"{m['all']['ok']} / {m['all']['n']} 达标"), (res,))
        click('#sc-ic-pol button[data-v=static]'); end()
        cmp_metrics('part 6 static, cap 64 (the last cap picked)', 'shared', {'budget': 2048, 'cache': True, 'cap': 64, 'policy': 'static'}, {'budget': 2048, 'cache': True})
        pr = NB(J("()=>document.querySelector('#sc-polres').textContent")); _, ms_ = run('shared', budget=2048, cache=True, cap=64, policy='static'); _, mc_ = run('shared', budget=2048, cache=True, cap=64)
        check(f'{lang} part 6: static batching has its own result line (static {ms_["all"]["ok"]}, continuous {mc_["all"]["ok"]})', f"{ms_['all']['ok']} / 48" in pr and f"{mc_['all']['ok']} / 48" in pr, pr)
        res64 = NB(J("()=>document.querySelector('#sc-rail [data-lever=\"64\"] .lv-r').textContent"))
        check(f'{lang} part 6: with static on, the cap 64 option keeps its continuous result', f"{mc_['all']['ok']} / 48" in res64, res64)
        # ---------- part 8: the misconception's number
        click('#sc-rail [data-go="7"]')
        _, m = run('stall', budget=128)
        txt = NB(J("()=>document.querySelector('#sc-rail .step.current').innerText"))
        check(f'{lang} part 8: budget 128 meets the goal for {m["all"]["ok"]} of {m["all"]["n"]}', (f"{m['all']['ok']} of {m['all']['n']}" if lang == 'en' else f"{m['all']['n']} 个请求里只有 {m['all']['ok']} 个") in txt, txt[:300])
        check(f'{lang} no page errors', not errs, errs)
        pg.close()

    # ---------- the view model through the hook: memory, waits, goals (EN page)
    pg = b.new_page(viewport={'width': 1440, 'height': 1000}); errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(250); pg.evaluate("()=>{location.hash='serving-capstone'}"); pg.wait_for_timeout(500)
    J = lambda s: pg.evaluate(s)
    # one peak everywhere: the memory bar's peak view holds the engine's peak (Python), during the same step; GPU busy from Python's steps
    for load in ('stall', 'shared', 'apps', 'challenge'):
        for kw in ({}, {'budget': 2048}, {'budget': 2048, 'cache': True}, {'budget': 512, 'cache': True, 'cache_blocks': 512}, {'cap': 64, 'budget': 2048, 'cache': True}, {'policy': 'static', 'cache': True}):
            r, m = run(load, **kw); s = {'policy': kw.get('policy', 'continuous'), 'cap': kw.get('cap', 32), 'budget': kw.get('budget', 0), 'cache': kw.get('cache', False), 'cb': kw.get('cache_blocks')}
            pp = J(f"()=>window.__sc.peakPre('{load}',{json.dumps(s)})"); ks = peak_step(r, kw.get('cache', False))
            check(f'peak {load} {kw}: the peak view holds the engine peak ({r["peak"]} blocks) during step {ks}', pp['used'] == r['peak'] and pp['k'] == ks, (pp, r['peak'], ks))
            bz = J(f"()=>window.__sc.gpuTime('{load}',{json.dumps(s)})")
            check(f'GPU time {load} {kw}: {gput(r):.4f} s', abs(bz['sum'] - gput(r)) < 1e-9, (bz, gput(r)))
    # GPU time used shows saved work: in the challenge some settings differ from the initial setup, as shown (review r2, N1)
    g0 = fS(gput(run('challenge')[0])); gs = {fS(gput(run('challenge', **kw)[0])) for kw in ({'cap': 32, 'budget': 2048, 'cache': True}, {'cap': 16, 'budget': 2048, 'cache': True}, {'cap': 64, 'budget': 8192, 'cache': True})}
    check(f'challenge: GPU time used moves with the settings (initial {g0}; {sorted(gs)})', any(x != g0 for x in gs), (g0, gs))
    # big counts in the capability card: 3 significant digits with a power of ten, never "2040887B" (review r2, N2)
    for lang in ('en', 'zh'):
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        et = NB(J("()=>window.__sc.exitsText()"))
        import re as _re
        check(f'{lang} capability card: no 6-digit-plus count with “B”', not _re.search(r'\d{6,}B', et), _re.findall(r'\d{6,}B', et)[:3])
        check(f'{lang} capability card: the FLOP totals as powers of ten', '× 10¹⁵' in et, et[:120])
        check(f'{lang} capability card: no number of 7 digits or more, tooltips included (R3.19)', not _re.search(r'\d{1,3}(,\d{3}){2,}|\d{7,}', et), _re.findall(r'\d{1,3}(?:,\d{3}){2,}|\d{7,}', et)[:3])
    pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(200)
    # memory waits happen only where the pool note says: the four-document load with caching off and a budget of 8,192 or none
    memw = 0; where = set()
    for load in ('stall', 'shared', 'apps', 'challenge'):
        for pol in ('continuous', 'static'):
            for cap in (8, 16, 32, 64, 128):
                for bud in ((0,) if pol == 'static' else (0, 8192, 2048, 512, 128)):
                    for cache, cb in ((False, None), (True, 512), (True, 1536), (True, 3072), (True, 6144), (True, None)):
                        r, _ = run(load, policy=pol, cap=cap, budget=bud, cache=cache, cache_blocks=cb)
                        k = sum(1 for x in r['R'] if x.waits.get('memory', 0) > 0); memw += k
                        if k: where.add((load, cache, bud))
    check(f'memory waits only in the four-document load with caching off and a budget of 8,192 or none (720 settings; {sorted(where)})', where and all(l == 'apps' and not c and b in (0, 8192) for l, c, b in where), sorted(where))
    for load, kw in (('apps', {'budget': 2048, 'cache': True, 'cache_blocks': 512}), ('apps', {'budget': 2048, 'cache': True, 'cache_blocks': 3072}),
                     ('challenge', {'cap': 32, 'budget': 512, 'cache': True, 'cache_blocks': 512}), ('shared', {'budget': 2048, 'cache': True})):
        r, m = run(load, **kw)
        s = {'policy': 'continuous', 'cap': kw.get('cap', 32), 'budget': kw.get('budget', 0), 'cache': kw.get('cache', False), 'cb': kw.get('cache_blocks')}
        bad = 0
        for k in range(0, len(r['steps']), 37):
            pm = J(f"()=>window.__sc.memAt('{load}',{json.dumps(s)},{k})")
            own, inuse = pool_after(r, {'cache': kw.get('cache', False)}, k)
            t, used, idle, resv = r['mem'][k]
            pres = sum(pm['pres'].values()) if pm['pres'] else 0
            pown = sum(x['b'] for x in pm['own'])
            if not (pown == own and pown + pres == used and pres - sum(pm['inuse'].values()) == idle and pm['mem']['resv'] == resv and pm['inuse'] == inuse):
                bad += 1
                if bad < 3: print('  pool', load, kw, k, (pown, own, pres, used, idle, resv, pm['inuse'], inuse))
        check(f'KV pool split of {load} {kw} matches Python at every 37th step', bad == 0, bad)
        if r['evictions']:
            ev = J(f"()=>window.__sc.view('{load}',{json.dumps(s)}).evAt")
            check(f'{load} {kw}: every eviction placed at a step ({r["evictions"]})', sum(sum(d.values()) for _, d in ev) == r['evictions'], (ev[:3], r['evictions']))
    for load, kw in (('stall', {'budget': 2048}), ('shared', {'budget': 2048, 'cache': True, 'cap': 8}), ('challenge', {})):
        r, m = run(load, **kw)
        s = {'policy': 'continuous', 'cap': kw.get('cap', 32), 'budget': kw.get('budget', 0), 'cache': kw.get('cache', False), 'cb': None}
        bad = 0
        for x in r['R']:
            segs = J(f"()=>window.__sc.waits('{load}',{json.dumps(s)},{x.id})")
            tot = {}
            for a_, b_, why in segs: tot[why] = tot.get(why, 0) + b_ - a_
            keys = set(tot) | set(x.waits)
            if any(abs(tot.get(k, 0) - x.waits.get(k, 0)) > 1e-6 for k in keys): bad += 1
        check(f'waits by reason of every request of {load} {kw} match Python', bad == 0, bad)
    # ---------- the challenge: plans, tiers against a brute-force search, the plan text
    GRID = []
    for pol in ('continuous', 'static'):
        for cap in (8, 16, 32, 64, 128):
            for bud in ((0,) if pol == 'static' else (0, 8192, 2048, 512, 128)):
                for cache, cb in ((False, None), (True, 512), (True, 1536), (True, 3072), (True, 6144), (True, None)):
                    GRID.append({'policy': pol, 'cap': cap, 'budget': bud, 'cache': cache, 'cache_blocks': cb})
    chg = lambda g: (g['policy'] != 'continuous') + (g['cap'] != 32) + (g['budget'] != 0) + (g['cache'] != False) + (1 if g['cache'] and g['cache_blocks'] is not None else 0)
    passing = [g for g in GRID if run('challenge', **{k: v for k, v in g.items()})[1]['all']['ok_frac'] >= E.TARGET - EPS]
    fewest = min(chg(g) for g in passing)
    check('the challenge has a passing and a plausible-but-failing plan', passing and run('challenge', cap=16, budget=2048, cache=True)[1]['all']['ok_frac'] < 0.9)
    J("()=>document.querySelector('#sc-app .modes button[data-mode=challenge]').click()"); pg.wait_for_timeout(500)
    for plan, tier in (({'cap': '32', 'budget': '2048', 'cache': 'true'}, None), ({'cap': '64', 'budget': '2048', 'cache': 'true'}, None), ({'cap': '16', 'budget': '2048', 'cache': 'true'}, 3), ({'budget': '128', 'cap': '128', 'cache': 'true'}, 3)):
        J("()=>{for(const [k,v] of [['cap','32'],['budget','0'],['cache','false'],['policy','continuous']]){const b=document.querySelector(`#sc-rail .seg-ctl[data-key=\"ch-${k}\"] button[data-v=\"${v}\"]`); if(b) b.click();}}"); pg.wait_for_timeout(150)
        for k, v in plan.items():
            J(f"()=>document.querySelector('#sc-rail .seg-ctl[data-key=\"ch-{k}\"] button[data-v=\"{v}\"]').click()"); pg.wait_for_timeout(120)
        J("()=>document.querySelector('#sc-rail [data-act=run]').click()"); pg.wait_for_timeout(300)
        J("()=>{const b=document.querySelector('[data-act=toend]'); if(b) b.click();}")
        pg.wait_for_function("()=>window.__sc.state.chTier!==null", timeout=20000)
        g = {'policy': 'continuous', 'cap': int(plan.get('cap', 32)), 'budget': int(plan.get('budget', 0)), 'cache': plan.get('cache') == 'true', 'cache_blocks': None}
        r, m = run('challenge', **g)
        st = J("()=>window.__sc.state")
        ok = m['all']['ok_frac'] >= E.TARGET - EPS
        want = 3 if not ok else (1 if chg(g) <= fewest else 2)
        check(f'challenge {plan}: tier {want}', st['chTier']['tier'] == want, (st['chTier'], chg(g), fewest))
        res = NB(J("()=>document.querySelector('#sc-ch-res').innerText"))
        check(f'challenge {plan}: goal met {m["all"]["ok"]} of {m["all"]["n"]}', f"{m['all']['ok']} of {m['all']['n']}" in res, res[:160])
        if want == 2:
            alt = st['chTier']['alt']; ag = {'policy': alt['policy'], 'cap': alt['cap'], 'budget': alt['budget'], 'cache': alt['cache'], 'cache_blocks': alt['cb']}
            check(f'challenge {plan}: the smaller change passes with {chg(ag)} changes', run('challenge', **ag)[1]['all']['ok_frac'] >= 0.9 - EPS and chg(ag) < chg(g), ag)
        plan_txt = NB(J("()=>window.__sc.plan()") or '')
        _, mb = run('challenge')
        for want_s in (f"{mb['all']['ok']} / {mb['all']['n']}", f"{m['all']['ok']} / {m['all']['n']}", fS(m['chat']['ttft_p50']), fS(m['chat']['ttft_p95']), fS(m['chat']['gap_max']), fG(m['all']['peak_gib'])):
            check(f'challenge {plan}: the plan says {want_s}', want_s in plan_txt, plan_txt[:300])
    # the diagnosis of each plausible-but-failing plan against Python's dominant wait reason of the requests that miss
    for kw, want in (({'cap': 16, 'budget': 2048, 'cache': True}, 'cap'), ({'cap': 64, 'budget': 8192, 'cache': True}, 'long'), ({'cap': 128, 'budget': 128, 'cache': True}, 'crawl'),
                     ({'cap': 64, 'budget': 2048}, 'nocache'), ({'cap': 64, 'cache': True}, 'long'), ({'policy': 'static', 'cache': True}, 'batch')):
        r, m = run('challenge', **kw)
        miss = [x for x in r['R'] if not ok_of(x)]
        gapm = sum(1 for x in miss if gap(x) > E.GOALS[x.cls].get('gap', 1e9) + EPS)
        w = {}
        for x in miss:
            for k, v in x.waits.items(): w[k] = w.get(k, 0) + v
        dom = max(w, key=w.get) if w else None
        py = 'long' if gapm * 2 >= len(miss) else {'budget': ('nocache' if not kw.get('cache') else 'crawl' if kw.get('budget') == 128 else 'ahead')}.get(dom, dom)
        s = {'policy': kw.get('policy', 'continuous'), 'cap': kw.get('cap', 32), 'budget': kw.get('budget', 0), 'cache': kw.get('cache', False), 'cb': None}
        pd = J(f"()=>window.__sc.view('challenge',{json.dumps(s)}).diag")
        check(f'challenge {kw}: the page names the limit “{want}”', pd == want and py == want, (pd, py))
    # ---------- "A new case": custom loads
    ld = E.LOADS['stall']()
    for x in ld:
        if x['cls'] == 'doc': x['own'] = 65536
    r64 = E.simulate(ld, BASE); m64 = E.metrics(r64, E.GOALS)
    x64 = J("()=>window.__sc.xrun('doc64')")
    check('new case 1: the 64K document’s worst stall', abs(x64['worst']['g'] - m64['chat']['gap_max']) < 1e-9, (x64['worst'], m64['chat']['gap_max']))
    check('new case 1: the answer is the range 3.5 to 5 s', 3.5 <= m64['chat']['gap_max'] < 5, m64['chat']['gap_max'])
    l12 = E.load_shared(12, 48, 6.0, 12288)
    r12 = E.simulate(l12, {**BASE, 'budget': 2048, 'cache': True}); m12 = E.metrics(r12, E.GOALS)
    x12 = J("()=>window.__sc.xrun('sys12')")
    check('new case 2: prefill tokens with a 12,288-token prompt', x12['m']['all']['prefill_tok'] == m12['all']['prefill_tok'] == 12288 + sum(x['own'] for x in l12), (x12['m']['all']['prefill_tok'], m12['all']['prefill_tok']))
    check('new case 2: the answer is under 50K', m12['all']['prefill_tok'] < 50000, m12['all']['prefill_tok'])
    # R3.19: each new case gives a reference the page shows and asks for an estimate; the answer matches the engine (review r3, findings 9 and 10)
    def fR(sec):
        v = sec * 1e3
        if sec >= 9.95: return f'{fint(sec)} s'
        if v >= 995: return f'{sec:.1f} s'
        if v >= 99.5: return f'{fint(round(v / 10) * 10)} ms'
        return f'{fint(v)} ms'
    et = NB(J("()=>window.__sc.exitsText()"))
    g32 = run('stall')[1]['chat']['gap_max']; g64 = m64['chat']['gap_max']; mult = round(g64 / g32, 1)
    check(f'new case 1: the reference stall is stated in full, not as part 1’s (G2), as the page rounds it ({fR(g32)})', f'every running chat stops for about {fR(g32)}' in et and 'In part one' not in et, et[:200])
    check(f'new case 1: the right option is the engine’s multiple ({mult}×, {fR(g64)})', f'About {mult} times as long, about {fR(g64)}' in et and 2 < mult < 3.5, (mult, fR(g64)))
    p6 = run('shared', budget=2048, cache=True)[1]['all']['prefill_tok']; p12 = m12['all']['prefill_tok']
    check(f'new case 2: the support app and its prefill total stated in full, not as part 4’s (G2), as the page rounds it (about {round(p6 / 1000)}K)', f'the GPU prefills about {round(p6 / 1000)}K prompt tokens' in et and 'In part four' not in et and 'the table there' not in et, et[-900:])
    rng = _re.findall(r'(\d+)K–(\d+)K, about 6K more', et)
    check(f'new case 2: the right option is a range that holds the engine’s total ({p12:,})', len(rng) == 1 and int(rng[0][0]) * 1000 <= p12 < int(rng[0][1]) * 1000, rng)
    # ---------- the interview drill: its own small load, run in Python too
    for seed in (1, 2, 3):
        d = J(f"()=>window.__sc.drill({seed})")
        sp = [{'t': 0, 'cls': 'chat', 'own': d['c'], 'out': 400} for _ in range(d['U'])] + [{'t': 3, 'cls': 'doc', 'own': d['N'], 'out': 50}]
        a_ = E.metrics(E.simulate(sp, BASE), E.GOALS)['chat']['gap_max']; b_ = E.metrics(E.simulate(sp, {**BASE, 'budget': d['B']}), E.GOALS)['chat']['gap_max']
        check(f'drill seed {seed}: the worst waits without and with a budget', abs(d['a'] - a_) < 1e-9 and abs(d['b'] - b_) < 1e-9, (d, a_, b_))
    check('no page errors (hook page)', not errs, errs)
    # ---------- the home tile (CONTRACT §5, R11.1, R11.5; review home-v4 A #2, #7, B-18): part 1's load without the 32K document, with it
    # under the initial setup, and with it under the setup the guided parts end on (one scheduler setting changed). Every row of the three
    # pictures is redrawn here from Python runs (its segments: thin from arrival to the first token and across any wait over the chat gap
    # goal, thick while tokens come; met or missed; its place in arrival order); the sentence, the picture's description, the legend and
    # the notes are checked against the same runs. The third option's setting is the lab's own and does not meet the challenge; nothing on
    # the tile says why the chats stop or names a setting (R3.8: part 2 asks why); the tile keeps its height across the options;
    # readable at a 280 px tile, at 390 px and in dark mode.
    import re as _re
    GG = E.GOALS['chat']['gap']; TT = E.GOALS['chat']['ttft']; FULL = E.LOADS['stall'](); DI = next(i for i, x in enumerate(FULL) if x['cls'] == 'doc'); DOC = FULL[DI]
    assert DOC['own'] % 1024 == 0
    K_ = f"{DOC['own'] // 1024}K"; AT = f"{DOC['t']:.1f} s"; OPTS = ['none', 'sent', 'fix']
    T = '#tile-serving-capstone'; tz_errs = []
    def tz_page(lang, w=1440, h=900, dark=False):
        ctx = b.new_context(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light'); pg_ = ctx.new_page()
        pg_.on('pageerror', lambda e: tz_errs.append(str(e)))
        pg_.goto(URL); pg_.wait_for_timeout(300)
        if lang == 'zh': pg_.click('.langsw button[data-lang="zh"]'); pg_.wait_for_timeout(200)
        pg_.locator('#domStrip .ds-btn[data-world="inference"]').first.click(); pg_.wait_for_timeout(400)
        return pg_
    def tz_set(pg_, opt):
        pg_.click(f'{T} .seg-ctl[data-key="sc-tz"] button[data-v="{opt}"]'); pg_.wait_for_timeout(120)
    # the third option's setting, read from the lab: the setup of its last guided part, on part 1's load, one setting away from the initial setup
    pg = tz_page('en')
    FIX = pg.evaluate("()=>window.__scTz.fix()"); LAB = pg.evaluate("()=>({pc:window.__sc.consts.PART_CFG, base:window.__sc.consts.BASE, ids:window.__sc.consts.STEP_IDS})")
    DIFF = {k: v for k, v in FIX.items() if k != 'load' and v != LAB['base'].get(k)}
    check(f'tile: the third option is the setup the guided parts end on (part “{LAB["ids"][-1]}”), on part 1’s load, one scheduler setting changed ({DIFF})',
          FIX == LAB['pc'][LAB['ids'][-1]] and FIX['load'] == 'stall' and len(DIFF) == 1, (FIX, LAB['pc'].get(LAB['ids'][-1])))
    PYFIX = {**BASE, **{('cache_blocks' if k == 'cb' else k): v for k, v in DIFF.items()}}
    def tz_py(opt):
        sp = FULL if opt != 'none' else [x for x in FULL if x['cls'] != 'doc']
        r = E.simulate(sp, PYFIX if opt == 'fix' else BASE); E.check(r); m = E.metrics(r, E.GOALS); rows = []
        for q in r['R']:
            seg = [(q.arr, q.first, 'w')]; s0 = q.toks[0]
            for a, b_ in zip(q.toks, q.toks[1:]):
                if b_ - a > GG + EPS: seg += [(s0, a, 't'), (a, b_, 'w')]; s0 = b_
            seg.append((s0, q.toks[-1], 't'))
            rows.append({'id': q.id, 'row': q.id if opt != 'none' else q.id + (q.id >= DI), 'cls': q.cls, 'ok': ok_of(q), 'late': q.first - q.arr > E.GOALS[q.cls]['ttft'] + EPS,
                         'seg': seg, 'arr': q.arr, 'first': q.first, 'fin': q.fin, 'gap': gap(q)})
        d = next((x for x in rows if x['cls'] == 'doc'), None)
        answering = [x['id'] for x in rows if d and x['cls'] == 'chat' and x['first'] < d['arr'] < x['fin']]
        stalled = [x['id'] for x in rows if d and x['cls'] == 'chat' and x['first'] < d['arr'] < x['fin'] and x['gap'] > GG + EPS]
        return {'m': m, 'rows': rows, 'n': len(sp), 'nChat': sum(1 for x in rows if x['cls'] == 'chat'), 'ok': sum(1 for x in rows if x['ok']),
                'answering': answering, 'stalled': stalled, 'end': max(x['fin'] for x in rows), 'doc': d, 'miss': [x for x in rows if not x['ok']]}
    TP = {o: tz_py(o) for o in OPTS}
    SPAN = math.ceil(max(TP[o]['end'] for o in OPTS))
    X = lambda v: 0.4 + v / SPAN * 99.2
    TICKS = [round(k * DOC['t'], 3) for k in range(int(SPAN / DOC['t'] + 1e-9) + 1)]
    py = TP['sent']
    check(f'tile: Python’s run with the document: {py["ok"]} of {py["n"]} meet their goal, under the service’s {pct(E.TARGET)}; the {len(py["stalled"])} chats answering at {AT} all stall for {fR(py["m"]["chat"]["gap_max"])}',
          py['ok'] == py['m']['all']['ok'] and py['n'] == 37 and py['ok'] / py['n'] < E.TARGET and py['stalled'] == py['answering'] and len(py['stalled']) > 0
          and all(fR(x['gap']) == fR(py['m']['chat']['gap_max']) for x in py['rows'] if x['id'] in py['stalled']), (py['ok'], py['stalled'], py['answering']))
    check(f'tile: Python’s run without it: all {TP["none"]["n"]} meet their goal, no chat waits over {fS(GG)}', TP['none']['ok'] == TP['none']['n'] == 36 and TP['none']['m']['chat']['gap_max'] <= GG + EPS, TP['none']['m']['chat'])
    pf = TP['fix']
    check(f'tile: Python’s run with the document and the guided parts’ setting: {pf["ok"]} of {pf["n"]} meet their goal, at least the service’s {pct(E.TARGET)}; no chat waits over {fS(GG)}; '
          f'the {len(pf["miss"])} that miss are chats with a first token later than {fS(TT)}',
          pf['ok'] == pf['m']['all']['ok'] and pf['n'] == 37 and pf['ok'] / pf['n'] >= E.TARGET - EPS and pf['ok'] > py['ok'] and pf['m']['chat']['gap_max'] <= GG + EPS
          and 0 < len(pf['miss']) and all(x['cls'] == 'chat' and x['late'] for x in pf['miss']), (pf['ok'], pf['m']['chat']['gap_max'], [(x['id'], x['cls'], x['late']) for x in pf['miss']]))
    lab_ok = pg.evaluate(f"()=>window.__sc.view('stall',{json.dumps(DIFF)}).ok.filter(Boolean).length")
    check(f'tile: the lab’s own run of that setting (its guided part) has the same {pf["ok"]} requests meeting their goal', lab_ok == pf['ok'], lab_ok)
    rc_ = E.simulate(E.LOADS['challenge'](), PYFIX); mc_ = E.metrics(rc_, E.GOALS)
    check(f'tile: that one change does not meet the challenge ({mc_["all"]["ok"]} of {mc_["all"]["n"]}), so the tile gives none of its answer away', mc_['all']['ok_frac'] < E.TARGET - EPS, mc_['all'])
    pg.context.close()
    TZ_DOM = """()=>{const t=document.querySelector('#tile-serving-capstone'), s=t.querySelector('svg.sc-tz-svg'), R=s.getBoundingClientRect(), px=v=>parseFloat(v);
      const bb=e=>{const r=e.getBoundingClientRect(); return [r.left-R.left,r.top-R.top,r.right-R.left,r.bottom-R.top]};
      const seg=t.querySelector('.seg-ctl[data-key="sc-tz"]');
      return {pressed:[...seg.querySelectorAll('button[data-v]')].filter(b=>b.getAttribute('aria-pressed')==='true').map(b=>b.dataset.v),
        btn:[...seg.querySelectorAll('button[data-v]')].map(b=>[b.dataset.v,b.textContent]), lbl:t.querySelector('.ctl .lbl')?t.querySelector('.ctl .lbl').textContent:null,
        btnH:[...seg.querySelectorAll('button[data-v]')].map(b=>b.getBoundingClientRect().height),
        role:s.getAttribute('role'), aria:s.getAttribute('aria-label'), svgH:R.height, svgW:R.width,
        rows:[...s.querySelectorAll('g[data-row]')].map(g=>({row:+g.dataset.row, cls:g.dataset.cls, ok:g.dataset.ok, k:g.getAttribute('class'),
          seg:[...g.querySelectorAll('rect.sc-tz-w,rect.sc-tz-t')].map(r=>[r.getAttribute('class').slice(6),px(r.getAttribute('x')),px(r.getAttribute('width')),px(r.getAttribute('y'))]),
          box:[...g.querySelectorAll('rect.sc-tz-box')].map(r=>[px(r.getAttribute('x')),px(r.getAttribute('width'))])})),
        ticks:[...s.querySelectorAll('.sc-tz-ax text')].map(e=>[e.textContent,px(e.getAttribute('x'))]),
        lab:[...s.querySelectorAll('text.sc-tz-lab')].map(e=>e.textContent), cap:[...s.querySelectorAll('text.sc-tz-cap')].map(e=>e.textContent),
        texts:[...s.querySelectorAll('text')].map(e=>({c:e.getAttribute('class')||'ax',b:bb(e)})), marks:[...s.querySelectorAll('g[data-row] rect')].map(e=>({row:e.parentNode.dataset.row,b:bb(e)})),
        read:t.querySelector('.tz-read').textContent, readH:t.querySelector('.tz-read').innerHTML, twins:[...t.querySelectorAll('.sc-tz-twin')].map(e=>e.textContent),
        twinHid:[...t.querySelectorAll('.sc-tz-twin')].every(e=>e.getAttribute('aria-hidden')==='true'&&getComputedStyle(e).visibility==='hidden'),
        leg:[...t.querySelectorAll('.tz-leg>span')].map(e=>({t:e.textContent,tip:e.getAttribute('data-tipx')||'',tab:e.getAttribute('tabindex'),vis:e.getBoundingClientRect().height>0,top:Math.round(e.getBoundingClientRect().top)})),
        notes:[...t.querySelectorAll('.tz-note')].map(e=>e.textContent), noteKids:t.querySelectorAll('.tz-note *').length, more:!!t.querySelector('.tz-more'),
        chip:t.querySelector('.chip.tz-kind').textContent, tileH:t.getBoundingClientRect().height, tileW:t.getBoundingClientRect().width,
        over:Math.max(document.documentElement.scrollWidth-innerWidth,t.scrollWidth-t.clientWidth), vis:t.querySelector('.tile-tz').innerText,
        tips:[...t.querySelectorAll('.tile-tz [data-tipx]')].map(e=>e.getAttribute('data-tipx'))};}"""
    def hit(a, c): return a[0] < c[2] - 0.01 and c[0] < a[2] - 0.01 and a[1] < c[3] - 0.01 and c[1] < a[3] - 0.01
    def texts_clear(o, tag):
        W_ = o['svgW']; inside = [x for x in o['texts'] if x['b'][0] < -0.5 or x['b'][2] > W_ + 0.5]
        clash = sorted({(x['c'], m_['row']) for x in o['texts'] if x['c'] != 'ax' for m_ in o['marks'] if hit(x['b'], m_['b'])})
        check(f'tile {tag}: the caption, the document’s label and the axis labels stay inside the picture; the caption and the label cross no row', not inside and not clash, (inside, clash[:6]))
    # R3.8: no cause, no setting's name, no setting's value (the third option's value included), no batching policy
    vals = '|'.join(_re.escape(f'{v:,}') + '|' + _re.escape(str(v)) for v in DIFF.values() if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 100)
    BAD = {'en': r'budget|chunk|prefix|cach|\bcap\b|prefill|one step|whole prompt|prompt|split|piece|static|continuous' + ('|' + vals if vals else ''),
           'zh': r'预算|chunk|前缀|缓存|上限|prefill|一步里|一步算|整个 prompt|prompt|分块|切|static|continuous' + ('|' + vals if vals else '')}
    def res_en(p_):
        ok, n_ = p_['ok'], p_['n']; low = ok / n_ < E.TARGET - EPS
        return (f'all {n_} requests meet their goal', f'<b>{n_}</b>') if ok == n_ else (f'{"only " if low else ""}{ok} of {n_} requests meet their goal', f'<b>{ok} of {n_}</b>')
    def res_zh(p_):
        ok, n_ = p_['ok'], p_['n']; low = ok / n_ < E.TARGET - EPS
        return (f'{n_} 个请求全部达标', '<b>全部达标</b>') if ok == n_ else (f'{n_} 个请求里{"只有" if low else ""} {ok} 个达标', f'<b>{ok}</b>')
    def want_read(opt, lang):
        p_ = TP[opt]; (re_, be), (rz, bz) = res_en(p_), res_zh(p_)
        en = {'sent': f'When one H100 SXM serves Llama-3.1-8B to {p_["nChat"]} short chats and a {K_}-token document at {AT}, {re_}.',
              'none': f'When one H100 SXM serves Llama-3.1-8B to the same {p_["nChat"]} short chats and no document, {re_}.',
              'fix': f'When one H100 SXM serves Llama-3.1-8B to the same {p_["nChat"]} chats and document with one scheduler setting changed, {re_}.'}[opt]
        zh = {'sent': f'一张 H100 SXM 用 Llama-3.1-8B 服务 {p_["nChat"]} 个短对话，{AT} 时又来了一份 {K_} 个 token 的文档：{rz}。',
              'none': f'一张 H100 SXM 用 Llama-3.1-8B 服务同样的 {p_["nChat"]} 个短对话，没有那份文档：{rz}。',
              'fix': f'一张 H100 SXM 用 Llama-3.1-8B 服务同样的 {p_["nChat"]} 个短对话和那份文档，调度器改了一项设置：{rz}。'}[opt]
        return (en, be) if lang == 'en' else (zh, bz)
    def want_aria(opt, lang):
        p_ = TP[opt]; n_, ok, gm = p_['n'], p_['ok'], p_['m']['chat']['gap_max']; calm = gm <= GG + EPS
        if opt == 'none':
            return (f'Timeline of the {n_} short chats, one row each in arrival order, from 0 to {SPAN} s, without the document. ' + (f'No chat waits more than {fS(GG)} between two tokens' if calm else f'The longest wait between two tokens is {fS(gm)}') + f', and {f"all {n_}" if ok == n_ else f"{ok} of the {n_}"} meet their goal.'
                    if lang == 'en' else f'{n_} 个短对话的时间线，每个请求一行，按到达顺序排列，横轴从 0 到 {SPAN} s，没有那份文档。' + (f'没有一个对话在两个 token 之间等超过 {fS(GG)}' if calm else f'两个 token 之间最长等了 {fS(gm)}') + f'，{f"{n_} 个全部达标" if ok == n_ else f"{n_} 个里 {ok} 个达标"}。')
        if opt == 'sent':
            return (f'Timeline of the {n_} requests, one row each in arrival order, from 0 to {SPAN} s. When the {K_}-token document arrives at {AT}, the {len(p_["stalled"])} chats that are answering get no token for about {fR(gm)}; {n_ - ok} of the {n_} requests miss their goal.'
                    if lang == 'en' else f'{n_} 个请求的时间线，每个请求一行，按到达顺序排列，横轴从 0 到 {SPAN} s。{AT} 时那份 {K_} 个 token 的文档到达，正在回答的 {len(p_["stalled"])} 个对话约 {fR(gm)} 拿不到 token；{n_} 个请求里 {n_ - ok} 个没达标。')
        mi = p_['miss']; lc = bool(mi) and all(x['cls'] == 'chat' and x['late'] for x in mi)
        men = f'all {n_} requests meet their goal' if ok == n_ else f'{n_ - ok} of the {n_} requests miss their goal' + (f', {"a chat" if len(mi) == 1 else "both chats" if len(mi) == 2 else "all chats"} whose first token comes more than {fS(TT)} after {"it arrives" if len(mi) == 1 else "they arrive"}' if lc else '')
        mzh = f'{n_} 个请求全部达标' if ok == n_ else f'{n_} 个请求里 {n_ - ok} 个没达标' + (f'，都是到达后超过 {fS(TT)} 才出首 token 的对话' if lc else '')
        return (f'Timeline of the {n_} requests, one row each in arrival order, from 0 to {SPAN} s, with one scheduler setting changed. When the {K_}-token document arrives at {AT}, ' + (f'no chat waits more than {fS(GG)} between two tokens' if calm else f'the longest wait between two tokens is {fS(gm)}') + f'; {men}.'
                if lang == 'en' else f'{n_} 个请求的时间线，每个请求一行，按到达顺序排列，横轴从 0 到 {SPAN} s，调度器改了一项设置。{AT} 时那份 {K_} 个 token 的文档到达，' + (f'没有一个对话在两个 token 之间等超过 {fS(GG)}' if calm else f'两个 token 之间最长等了 {fS(gm)}') + f'；{mzh}。')
    seen = {}
    for lang in ('en', 'zh'):
        Z = lambda en, zh: en if lang == 'en' else zh
        pg = tz_page(lang)
        o = pg.evaluate(TZ_DOM)
        check(f'tile {lang}: opens on the document sent under the initial setup (the mission’s run); chip “{Z("Toy run", "玩具实跑")}”, no “Assumptions & sources” button (owner 2026-10-13)',
              pg.evaluate("()=>window.__scTz.state.opt") == 'sent' and o['pressed'] == ['sent'] and o['chip'] == Z('Toy run', '玩具实跑') and not o['more'], (o['pressed'], o['chip'], o['more']))
        check(f'tile {lang}: one control with three options, labelled from the load ({K_}, {AT}), none naming a setting', o['lbl'] == Z(f'The {K_} document', f'那份 {K_} 的文档')
              and [(v, NB(t_)) for v, t_ in o['btn']] == [('none', Z('not sent', '没来')), ('sent', Z(f'sent at {AT}', f'{AT} 时来了')), ('fix', Z('sent, retuned', '来了，调过调度器'))]
              and pg.locator(f'{T} .seg-ctl').count() == 1, (o['lbl'], o['btn']))
        for opt in OPTS:
            tz_set(pg, opt); o = pg.evaluate(TZ_DOM); py = TP[opt]; st = {'none': 'not sent', 'sent': 'sent', 'fix': 'retuned'}[opt]; seen[(lang, opt)] = o; doc = opt != 'none'
            check(f'tile {lang} {st}: the pressed button and the hook follow the choice', o['pressed'] == [opt] and pg.evaluate("()=>window.__scTz.state.opt") == opt, o['pressed'])
            if lang == 'en': check(f'tile {lang} {st}: the three buttons each on one line at 1440 px', max(o['btnH']) < 34, o['btnH'])
            # the rows, redrawn from Python
            pr = {x['row']: x for x in py['rows']}
            check(f'tile {lang} {st}: one row per request ({py["n"]}), in arrival order, each at its place in the whole load', len(o['rows']) == py['n'] and sorted(x['row'] for x in o['rows']) == sorted(pr)
                  and all(o['rows'][i]['seg'][0][3] < o['rows'][i + 1]['seg'][0][3] for i in range(len(o['rows']) - 1)) and [x['row'] for x in o['rows']] == sorted(pr), [x['row'] for x in o['rows']][:14])
            bad_ok = [x['row'] for x in o['rows'] if x['row'] not in pr or x['cls'] != pr[x['row']]['cls'] or x['ok'] != ('1' if pr[x['row']]['ok'] else '0') or x['k'] != ('sc-tz-met' if pr[x['row']]['ok'] else 'sc-tz-miss')]
            nmiss = len(py['miss'])
            check(f'tile {lang} {st}: met grey and missed red as Python’s per-request goal check ({nmiss} red)', not bad_ok and sum(1 for x in o['rows'] if x['k'] == 'sc-tz-miss') == nmiss, bad_ok)
            bad_seg = []
            for x in o['rows']:
                q = pr.get(x['row'])
                if not q or len(x['seg']) != len(q['seg']): bad_seg.append((x['row'], 'n', len(x['seg']), q and len(q['seg']))); continue
                for (k, xp, wp, _), (a, c, kk) in zip(x['seg'], q['seg']):
                    if k != kk or abs(xp - X(a)) > 6e-4 or abs(wp - max(0.35 if kk == 't' else 0, X(c) - X(a))) > 6e-4: bad_seg.append((x['row'], k, kk, xp, round(X(a), 4), wp, round(X(c) - X(a), 4)))
            check(f'tile {lang} {st}: every thin and thick segment of every row at Python’s times ({sum(len(x["seg"]) for x in py["rows"])} segments, axis 0–{SPAN} s)', not bad_seg, bad_seg[:4])
            boxes = [(x['row'], x['box']) for x in o['rows'] if x['box']]
            d = py['doc']
            check(f'tile {lang} {st}: ' + (f'the box frames the document’s row only, from its arrival ({d["arr"]:.1f} s) to its last token ({d["fin"]:.2f} s)' if doc else 'no box without the document'),
                  (len(boxes) == 1 and boxes[0][0] == DI and abs(boxes[0][1][0][0] - (X(d['arr']) - 0.7)) < 6e-4 and abs(boxes[0][1][0][1] - (X(d['fin']) - X(d['arr']) + 1.4)) < 6e-4) if doc else not boxes, boxes)
            check(f'tile {lang} {st}: the label names the document', o['lab'] == ([Z(f'◂ {K_} document', f'◂ {K_} 的文档')] if doc else []), o['lab'])
            check(f'tile {lang} {st}: the rows’ caption', o['cap'] == Z(['one row per request,', 'in arrival order'], ['每行一个请求，', '按到达顺序排列']), o['cap'])
            lbl = [f'{v:.1f}' if v % 1 else str(int(v)) for v in TICKS]; lbl[-1] += ' s'
            check(f'tile {lang} {st}: the time axis, the same for every option, ticks every {DOC["t"]} s (the document’s arrival) up to {SPAN} s: {lbl}', [NB(t_) for t_, _ in o['ticks']] == lbl and all(abs(xp - X(v)) < 6e-4 for (_, xp), v in zip(o['ticks'], TICKS))
                  and pg.evaluate("()=>window.__scTz.span()") == SPAN and pg.evaluate("()=>window.__scTz.ticks()") == TICKS, o['ticks'])
            check(f'tile {lang} {st}: the picture is an image with a label, at most 130 px tall', o['role'] == 'img' and bool(o['aria']) and o['svgH'] <= 130, (o['role'], o['svgH']))
            want = want_aria(opt, lang)
            check(f'tile {lang} {st}: the picture’s description, from Python', NB(o['aria']) == want, (NB(o['aria']), want))
            # the sentence (R11.5, CONTRACT §5: one sentence, at most 25 words): GPU, model, load and result, from Python
            want, bold = want_read(opt, lang)
            check(f'tile {lang} {st}: the sentence, from Python ({py["ok"]} of {py["n"]}), the result in bold', NB(o['read']) == want and bold in o['readH'], (NB(o['read']), want))
            check(f'tile {lang} {st}: one sentence' + (', at most 25 words' if lang == 'en' else ', one “：”'), (len(o['read'].split()) <= 25 and o['read'].count('. ') == 0) if lang == 'en' else (o['read'].count('。') == 1 and o['read'].count('：') <= 1), len(o['read'].split()))
            check(f'tile {lang} {st}: the GPU and the model never break across lines', 'H100 SXM' in o['read'] and 'Llama‑3.1‑8B' in o['read'] and 'Llama-3' not in o['read'], o['read'])
            others = [want_read(k, lang)[0] for k in OPTS if k != opt]
            check(f'tile {lang} {st}: the other two options’ sentences hold the height, hidden from sight and from screen readers', o['twinHid'] and [NB(x) for x in o['twins']] == others, [x[:40] for x in o['twins']])
            # the legend: every colour and line style, each with its sentence
            lg = [(NB(x['t']), x['tab'] == '0' and len(x['tip']) > len(x['t']) and x['vis']) for x in o['leg']]
            check(f'tile {lang} {st}: the legend names both line styles, both colours and the box, each with a tooltip', [t_ for t_, _ in lg] == Z(['thin: waiting for a token', 'thick: getting tokens', 'red: goal missed', 'grey: goal met', 'box: document'],
                  ['细线：在等 token', '粗线：在出 token', '红色：没达标', '灰色：达标', '方框：那份文档']) and all(v for _, v in lg), lg)
            check(f'tile {lang} {st}: the legend on two lines at 1440 px', len({x['top'] for x in o['leg']}) == 2, sorted({x['top'] for x in o['leg']}))
            tips = [NB(x['tip']) for x in o['leg']]
            check(f'tile {lang} {st}: the legend’s tooltips state the goals the colours stand for ({fS(GG)}, {fS(TT)}) and the document’s arrival ({AT})',
                  fS(GG) in tips[0] and fS(GG) in tips[1] and fS(GG) in tips[2] and fS(TT) in tips[2] and K_ in tips[4] and AT in tips[4], tips)
            # the notes (R11.5: 2-4 short sentences, plain text; review home-v4 A #7: no sentence that introduces the lab; B-18: 989.5 is the dense half of 1,979)
            nt = [NB(x) for x in o['notes']]; nsent = sum(len(_re.findall(r'[.。](?=\s|$)', x)) for x in nt); words = sum(len(x.split()) for x in nt)
            check(f'tile {lang} {st}: 4 plain-text notes, one sentence each' + (f', {words} words in all (was 141)' if lang == 'en' else ''), len(nt) == 4 and nsent == 4 and o['noteKids'] == 0 and (lang == 'zh' or words <= 110), (len(nt), nsent, words))
            check(f'tile {lang} {st}: note 1 states the goals from the engine ({fS(TT)}, {fS(GG)}, {fS(E.GOALS["doc"]["ttft"])}, {pct(E.TARGET)})',
                  all(v in nt[0] for v in (fS(TT), fS(GG), fS(E.GOALS['doc']['ttft']), pct(E.TARGET))), nt[0])
            check(f'tile {lang} {st}: note 2 says it is the lab’s seeded toy run, with the engine’s step model ({pct(E.FS / 989.5e12)} of 989.5 TFLOP/s, {pct(E.BS / 3.35e12)} of 3.35 TB/s)',
                  all(v in nt[1] for v in (Z('seeded toy run', '固定随机种子的玩具实跑'), 'H100 SXM', 'Llama-3.1-8B', 'BF16',
                                            Z(f'({pct(E.FS / 989.5e12)} of the dense BF16 peak)', f'稠密 BF16 峰值的 {pct(E.FS / 989.5e12)}'), Z(f'({pct(E.BS / 3.35e12)} of 3.35 TB/s)', f'3.35 TB/s 的 {pct(E.BS / 3.35e12)}'))), nt[1])
            check(f'tile {lang} {st}: note 3 says what the options run, without naming the setting; no note introduces the lab',
                  nt[2] == Z('The first two options use the lab’s initial setup; the third changes one scheduler setting, as the lab’s guided parts end up doing.', '前两个选项用 Lab 的初始设置；第三个只改了调度器的一项设置，和 Lab 引导部分最后的设置一样。')
                  and not any(Z('In the lab you', '在 Lab 里，你') in x for x in nt), nt[2])
            check(f'tile {lang} {st}: note 4 gives the sources, 989.5 TFLOP/s as the dense half of the page’s 1,979 with sparsity (B-18)', 1979 / 2 == 989.5 and abs(E.FS - 989.5e12 * 0.5) < 1e3
                  and all(v in nt[3] for v in ('1,979 TFLOP/s', Z('with sparsity', '含稀疏'), '989.5', '3.35 TB/s', 'config.json', 'H100')), nt[3])
            # R3.8 and the writing rules, over everything the tile says (visible text, tooltips, the picture's description, the notes, the hidden sentences)
            allt = '\n'.join([o['vis'], o['aria']] + o['twins'] + o['notes'] + o['tips'])
            check(f'tile {lang} {st}: nothing says why the chats stop or names a setting or its value (R3.8; part 2’s answer, the token budget, prefix caching, the cap, the batching)', not _re.search(BAD[lang], allt, _re.I), _re.findall(BAD[lang], allt, _re.I)[:5])
            check(f'tile {lang} {st}: no “·” between clauses' + ('' if lang == 'en' else ', a space between Chinese and Latin, at most one “：” per sentence'),
                  '·' not in allt and (lang == 'en' or (not _re.search(r'[一-鿿][A-Za-z0-9]|[A-Za-z0-9][一-鿿]', allt) and all(s_.count('：') <= 1 for s_ in _re.split(r'[。\n]', allt)))),
                  [s_ for s_ in _re.split(r'[。\n]', allt) if s_.count('：') > 1][:2] + _re.findall(r'[一-鿿][A-Za-z0-9]|[A-Za-z0-9][一-鿿]', allt)[:4])
            check(f'tile {lang} {st}: “toy” only as the fidelity’s name', lang == 'zh' or not _re.search(r'\btoy\b(?! run)', allt, _re.I), _re.findall(r'.{20}\btoy\b.{10}', allt, _re.I))
            js = pg.evaluate(f"()=>window.__scTz.run('{opt}')")
            check(f'tile {lang} {st}: the hook’s run equals Python’s (met, longest gap, the stalled chats, the end)', js['m']['all']['ok'] == py['m']['all']['ok'] and js['n'] == py['n'] and abs(js['m']['chat']['gap_max'] - py['m']['chat']['gap_max']) < 1e-9
                  and sorted(js['stalled']) == sorted(py['stalled']) and abs(js['end'] - py['end']) < 1e-9, (js['m']['all']['ok'], js['stalled']))
            check(f'tile {lang} {st}: no horizontal overflow at 1440 px', o['over'] <= 0, o['over'])
            texts_clear(o, f'{lang} {st} 1440')
        hs = [seen[(lang, k)]['tileH'] for k in OPTS]
        check(f'tile {lang}: the tile keeps its height across the three options', max(hs) - min(hs) < 0.5, hs)
        # a language switch re-renders the tile and keeps the choice
        pg.click(f'.langsw button[data-lang="{"en" if lang == "zh" else "zh"}"]'); pg.wait_for_timeout(250)
        o2 = pg.evaluate(TZ_DOM)
        check(f'tile {lang}: after a language switch the tile keeps “{Z("sent, retuned", "来了，调过调度器")}”', o2['pressed'] == ['fix'] and len(o2['rows']) == TP['fix']['n'] and sum(1 for x in o2['rows'] if x['k'] == 'sc-tz-miss') == len(TP['fix']['miss']), o2['pressed'])
        pg.context.close()
    # 390 px, a 280 px tile, dark mode: no overflow, the texts clear of the rows, the height kept, the marks visible against the tile (WCAG 1.4.11: 3:1)
    def lum(c):
        v = [int(x) / 255 for x in _re.findall(r'[\d.]+', c)[:3]]; v = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in v]
        return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
    def cr(a, c): la, lc = sorted((lum(a), lum(c)), reverse=True); return (la + 0.05) / (lc + 0.05)
    COL = """()=>{const t=document.querySelector('#tile-serving-capstone'); let e=t, bg='';
      while(e){const c=getComputedStyle(e).backgroundColor; if(c&&c!=='rgba(0, 0, 0, 0)'&&c!=='transparent'){bg=c;break;} e=e.parentElement;}
      const f=sel=>{const r=t.querySelector(sel); return r?getComputedStyle(r).fill:null};
      return {bg, met:f('g.sc-tz-met rect.sc-tz-t'), miss:f('g.sc-tz-miss rect.sc-tz-t'), box:t.querySelector('rect.sc-tz-box')?getComputedStyle(t.querySelector('rect.sc-tz-box')).stroke:null}}"""
    for lang, w, dark, force in (('en', 390, False, None), ('zh', 390, False, None), ('en', 1000, False, 280), ('zh', 1000, False, 280), ('en', 1440, True, None), ('zh', 1440, True, None), ('en', 390, True, None)):
        pg = tz_page(lang, w, 844 if w == 390 else 900, dark)
        if force: pg.add_style_tag(content=f'#tiles{{grid-template-columns:repeat(3,{force}px)!important}}'); pg.wait_for_timeout(150)
        tag = f'{lang} {force or w} px' + (' tile' if force else '') + (' dark' if dark else ''); hs = []
        for opt in OPTS:
            tz_set(pg, opt); o = pg.evaluate(TZ_DOM); hs.append(o['tileH'])
            if force: check(f'tile {tag}: the tile is {force} px wide', abs(o['tileW'] - force) < 0.5, o['tileW'])
            check(f'tile {tag} {opt}: no horizontal overflow, the legend visible', o['over'] <= 0 and all(x['vis'] for x in o['leg']), o['over'])
            texts_clear(o, f'{tag} {opt}')
            if opt == 'sent':
                c = pg.evaluate(COL)
                check(f'tile {tag}: the red and grey marks and the box stand out from the tile (3:1)', all(v and cr(v, c['bg']) >= 3 for v in (c['met'], c['miss'], c['box'])) and c['met'] != c['miss'],
                      {k: (v, round(cr(v, c['bg']), 2) if v and c['bg'] else None) for k, v in c.items()})
        check(f'tile {tag}: the tile keeps its height across the three options', max(hs) - min(hs) < 0.5, hs)
        pg.context.close()
    check('tile: no page errors', not tz_errs, tz_errs)
    b.close()
print(f'{n} checks, {len(fails)} failed')
sys.exit(1 if fails else 0)
