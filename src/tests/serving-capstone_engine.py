"""The serving capstone's engine, as the page runs it (tests/serving-capstone_engine.py <lab file>, default labs/serving-capstone.js).
- The ENGINE block (from `/* ENGINE BEGIN` to `/* ENGINE END */`) is extracted from the lab file and must be byte-identical to the one in
  capstone/engine.js (the spec: pasted unchanged).
- It is run in node on all four loads × a grid of settings (continuous and static, caps 8–128, budgets none/8192/2048/512/128, caching off
  and on with cache budgets 512/1024/2048 blocks and none) and compared with the Python reference capstone/engine.py: every request's
  admission, cache hit, first token, finish, token and chunk counts, waits by reason; the steps, evictions, peak; every metric.
- Invariants: SC_ENGINE.check(run) in node and check(run) in Python on every run; determinism: two runs of the same load and settings give
  identical output (JSON-equal), and the loads regenerate identically.
Adapted from capstone/cmp_js_py.py (kept as it is)."""
import json, subprocess, sys, os, tempfile, pathlib
P = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / 'capstone'))
import engine as E

SRC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else P / 'labs' / 'serving-capstone.js'
src = SRC.read_text()
a = src.index('/* ENGINE BEGIN'); b = src.index('/* ENGINE END */')
block = src[a:b]
ref = (P / 'capstone' / 'engine.js').read_text()
ra = ref.index('/* ENGINE BEGIN'); rb = ref.index('/* ENGINE END */')
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:300])


check('the ENGINE block is pasted unchanged from capstone/engine.js', block == ref[ra:rb], (len(block), len(ref[ra:rb])))
check('the ENGINE block sits at the top of the lab file', src.index('/* ENGINE BEGIN') == 0, src.index('/* ENGINE BEGIN'))

GRID = []
for pol in ('continuous', 'static'):
    for cap in (8, 16, 32, 64, 128):
        for bud in (0, 8192, 2048, 512, 128):
            for cache, cb in ((False, None), (True, 512), (True, 1024), (True, 2048), (True, None)):
                if pol == 'static' and bud not in (0,):
                    continue
                GRID.append({'policy': pol, 'cap': cap, 'budget': bud, 'cache': cache, 'cache_blocks': cb})
js = block + r'''
const E=SC_ENGINE, grid=JSON.parse(process.argv[2]), out={}, det=[];
for(const name of Object.keys(E.LOADS)){const ld=E.LOADS[name](); out[name]={load:ld,runs:[],same:JSON.stringify(ld)===JSON.stringify(E.LOADS[name]())};
  for(const g of grid){const cfg={policy:g.policy,cap:g.cap,budget:g.budget,cache:g.cache,cacheBlocks:g.cache_blocks};
    const run=E.simulate(ld,cfg); let inv=true; try{E.check(run);}catch(e){inv=String(e);} const m=E.metrics(run);
    out[name].runs.push({inv,m,steps:run.steps.length,t_end:run.steps[run.steps.length-1].t1,
      R:run.R.map(r=>[r.admit,r.hit,r.first,r.fin,r.toks.length,r.chunks.length,Object.keys(r.waits).sort().map(k=>[k,r.waits[k]])]),
      ev:run.evictions,peak:run.peak});}}
/* determinism: the same load and settings twice give the same run */
for(const [name,cfg] of [['stall',E.BASELINE],['challenge',{policy:'continuous',cap:64,budget:2048,cache:true,cacheBlocks:512}],['apps',{policy:'continuous',cap:32,budget:2048,cache:true,cacheBlocks:1024}]]){
  const x=E.simulate(E.LOADS[name](),cfg), y=E.simulate(E.LOADS[name](),cfg);
  const pick=r=>JSON.stringify({R:r.R,steps:r.steps,events:r.events,mem:r.mem,peak:r.peak,evictions:r.evictions,prefillTok:r.prefillTok,hitTok:r.hitTok});
  det.push([name,pick(x)===pick(y)]);}
process.stdout.write(JSON.stringify({out,det}));
'''
with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
    f.write(js); fn = f.name
res = json.loads(subprocess.run(['node', fn, json.dumps(GRID)], capture_output=True, text=True, check=True, timeout=500).stdout)
os.unlink(fn)
for name, ok in res['det']:
    check(f'determinism: two runs of {name} are identical', ok)


def close(x, y):
    if x is None or y is None:
        return x is None and y is None
    if isinstance(x, str):
        return x == y
    return abs(x - y) <= 1e-9 * max(1, abs(x), abs(y))


runs = 0; bad = 0
for name, f in E.LOADS.items():
    ld = f()
    check(f'{name}: the load regenerates identically in JS', res['out'][name]['same'])
    check(f'{name}: the load regenerates identically in Python', ld == f())
    jl = res['out'][name]['load']
    check(f'{name}: same number of requests', len(jl) == len(ld), (len(jl), len(ld)))
    for s, j in zip(ld, jl):
        for k in ('t', 'cls', 'own', 'out', 'pre', 'app'):
            if s.get(k) != j.get(k) and not (k in ('pre', 'app') and not s.get(k) and not j.get(k)):
                check(f'{name}: load field {k}', False, (s, j)); break
    for g, jr in zip(GRID, res['out'][name]['runs']):
        run = E.simulate(ld, g); runs += 1
        try:
            E.check(run); pinv = True
        except AssertionError as e:
            pinv = str(e)
        if jr['inv'] is not True or pinv is not True:
            check(f'{name} {g}: invariants hold (JS {jr["inv"]}, Python {pinv})', False); continue
        m = E.metrics(run, E.GOALS)
        py_R = [[r.admit, r.hit, r.first, r.fin, len(r.toks), len(r.chunks), sorted(r.waits.items())] for r in run['R']]
        ok = len(run['steps']) == jr['steps'] and run['evictions'] == jr['ev'] and run['peak'] == jr['peak'] and close(run['steps'][-1]['t1'], jr['t_end'])
        for pr, jrr in zip(py_R, jr['R']):
            for x, y in zip(pr[:6], jrr[:6]):
                if not close(x, y): ok = False
            if [k for k, _ in pr[6]] != [k for k, _ in jrr[6]] or any(not close(v, w) for (_, v), (_, w) in zip(pr[6], jrr[6])): ok = False
        for c in m:
            for k, v in m[c].items():
                if not close(v, jr['m'][c][k]): ok = False
        if not ok:
            bad += 1
            if bad < 6: print('MISMATCH', name, g)
check(f'JS and Python agree on every run ({runs} runs)', bad == 0, bad)
check('the grid covers 4 loads × 150 settings', runs == 600, runs)
print(f'{n} checks, {len(fails)} failed ({runs} runs compared, {bad} mismatches)')
sys.exit(1 if fails else 0)
