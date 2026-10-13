"""Compare the JS engine (engine.js, or the ENGINE block of a lab file) with the Python reference on every load x a settings grid."""
import json, subprocess, sys, os, itertools, math, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'engine.js')
src = open(SRC).read()
a = src.index('/* ENGINE BEGIN'); b = src.index('/* ENGINE END */')
block = src[a:b]
GRID = []
for pol in ('continuous', 'static'):
    for cap in (8, 16, 32, 64, 128):
        for bud in (0, 8192, 2048, 512, 128):
            for cache, cb in ((False, None), (True, 512), (True, 1024), (True, 2048), (True, None)):
                if pol == 'static' and bud not in (0,):
                    continue
                GRID.append({'policy': pol, 'cap': cap, 'budget': bud, 'cache': cache, 'cache_blocks': cb})
js = block + r'''
const E=SC_ENGINE, grid=JSON.parse(process.argv[2]), out={};
for(const name of Object.keys(E.LOADS)){const ld=E.LOADS[name](); out[name]={load:ld,runs:[]};
  for(const g of grid){const cfg={policy:g.policy,cap:g.cap,budget:g.budget,cache:g.cache,cacheBlocks:g.cache_blocks};
    const run=E.simulate(ld,cfg); E.check(run); const m=E.metrics(run);
    out[name].runs.push({m,steps:run.steps.length,t_end:run.steps[run.steps.length-1].t1,
      R:run.R.map(r=>[r.admit,r.hit,r.first,r.fin,r.toks.length,r.chunks.length,Object.keys(r.waits).sort().map(k=>[k,r.waits[k]])]),
      ev:run.evictions,peak:run.peak});}}
process.stdout.write(JSON.stringify(out));
'''
with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
    f.write(js); fn = f.name
res = json.loads(subprocess.run(['node', fn, json.dumps(GRID)], capture_output=True, text=True, check=True).stdout)
bad = 0; n = 0
def close(x, y):
    if x is None or y is None:
        return x is None and y is None
    if isinstance(x, str):
        return x == y
    return abs(x - y) <= 1e-9 * max(1, abs(x), abs(y))
for name, f in E.LOADS.items():
    ld = f()
    jl = res[name]['load']
    assert len(jl) == len(ld), (name, 'load length')
    for s, j in zip(ld, jl):
        for k in ('t', 'cls', 'own', 'out', 'pre', 'app'):
            if s.get(k) != j.get(k) and not (k in ('pre', 'app') and not s.get(k) and not j.get(k)):
                print('LOAD MISMATCH', name, k, s, j); bad += 1; break
    for g, jr in zip(GRID, res[name]['runs']):
        run = E.simulate(ld, g); E.check(run); m = E.metrics(run, E.GOALS)
        n += 1
        py_R = [[r.admit, r.hit, r.first, r.fin, len(r.toks), len(r.chunks), sorted(r.waits.items())] for r in run['R']]
        ok = len(run['steps']) == jr['steps'] and run['evictions'] == jr['ev'] and run['peak'] == jr['peak'] and close(run['steps'][-1]['t1'], jr['t_end'])
        for pr, jrr in zip(py_R, jr['R']):
            for x, y in zip(pr[:6], jrr[:6]):
                if not close(x, y): ok = False
            if [k for k, _ in pr[6]] != [k for k, _ in jrr[6]] or any(not close(v, w) for (_, v), (_, w) in zip(pr[6], jrr[6])): ok = False
        for c in m:
            for k, v in m[c].items():
                if not close(v, jr['m'][c][k]): ok = False; print('  metric', name, g, c, k, v, jr['m'][c][k])
        if not ok:
            bad += 1
            if bad < 6: print('MISMATCH', name, g)
print(f'{n} runs compared, {bad} mismatches')
sys.exit(1 if bad else 0)
