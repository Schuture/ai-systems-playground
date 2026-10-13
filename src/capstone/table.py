from engine import *
def row(name, cfg):
    ld = LOADS[name](); run = simulate(ld, cfg); check(run); m = metrics(run, GOALS); a = m['all']; c = m['chat']; d = m.get('doc')
    s = f"{name:9} {cfg['policy'][:4]} cap {cfg['cap']:>3} budget {cfg['budget'] or 'none':>5} cache {'on ' if cfg['cache'] else 'off'} {('%dGiB'%(cfg['cache_blocks']//512)) if cfg.get('cache_blocks') else ('   ∞' if cfg['cache'] else '    ')} | ok {a['ok']:>2}/{a['n']} = {a['ok_frac']*100:5.1f}% | chat TTFT p50 {c['ttft_p50']*1e3:6.0f} ms p95 {c['ttft_p95']*1e3:6.0f} ms (n={c['n']}) max {c['ttft_max']*1e3:5.0f} | worst gap {c['gap_max']*1e3:5.0f} ms"
    if d: s += f" | doc TTFT {d['ttft_max']:.2f} s"
    s += f" | thr {a['thr']:4.0f} tok/s | peak KV {a['peak_gib']:4.1f} GiB | prefill {a['prefill_tok']:>6} | hit {a['hit_tok']:>6} | evict {a['evictions']}"
    print(s)
B = dict(BASELINE)
for name in ('stall', 'shared', 'apps', 'challenge'):
    row(name, B)
print()
for b in (0, 8192, 2048, 512, 128): row('stall', {**B, 'budget': b})
row('stall', {**B, 'policy': 'static'})
print()
for c, cb in ((False, None), (True, None)): row('shared', {**B, 'budget': 2048, 'cache': c, 'cache_blocks': cb})
for cap in (8, 16, 32, 64, 128): row('shared', {**B, 'budget': 2048, 'cache': True, 'cap': cap})
row('shared', {**B, 'budget': 0, 'cache': True, 'policy': 'static'})
print()
for c, cb in ((False, None), (True, 512), (True, 1536), (True, 3072), (True, 6144), (True, None)): row('apps', {**B, 'budget': 2048, 'cache': c, 'cache_blocks': cb})
print()
for cfg in ({**B}, {**B, 'cap': 64, 'budget': 2048, 'cache': True}, {**B, 'cap': 32, 'budget': 512, 'cache': True, 'cache_blocks': 512},
            {**B, 'cap': 16, 'budget': 2048, 'cache': True}, {**B, 'cap': 64, 'budget': 8192, 'cache': True}, {**B, 'cap': 128, 'budget': 128, 'cache': True},
            {**B, 'cap': 64, 'budget': 2048, 'cache': False}, {**B, 'cap': 64, 'budget': 0, 'cache': True}, {**B, 'policy': 'static', 'cap': 32, 'cache': True}):
    row('challenge', cfg)
